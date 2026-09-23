import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from peewee import DoesNotExist
from starlette.testclient import TestClient, WebSocketDenialResponse

from app.algorithm import BayesianDecisionProcess
from app.db import AssignmentTable, db
from app.main import create_app
from app.ratelimit import limiter, subscriptions
from app.session import Session
from app.settings import ApiKey, settings

CSV = """Project Title,Note
Alpha,csv-secret-cell
Beta,second
Gamma,third
"""


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        settings,
        "API_KEYS",
        [
            ApiKey(
                key_hash=_digest("admin-secret"),
                user_id="admin-1",
                role="admin",
                admin_capacity=10,
                admin_refill_per_second=1,
            ),
            ApiKey(
                key_hash=_digest("judge-a-secret"),
                user_id="judge-a",
                role="judge",
                pair_capacity=1,
                pair_refill_per_second=0,
                submit_capacity=1,
                submit_refill_per_second=0,
            ),
            ApiKey(
                key_hash=_digest("judge-b-secret"),
                user_id="judge-b",
                role="judge",
                pair_capacity=2,
                pair_refill_per_second=0,
                submit_capacity=2,
                submit_refill_per_second=0,
            ),
        ],
    )
    limiter.reset()
    subscriptions.reset()
    with _open_client(tmp_path) as test_client:
        yield test_client


@contextmanager
def _open_client(tmp_path: Path) -> Iterator[TestClient]:
    if not db.is_closed():
        db.close()
    tmp_path.mkdir(parents=True, exist_ok=True)
    db.init(str(tmp_path / "mdredd.db"))
    with TestClient(create_app()) as test_client:
        yield test_client
    if not db.is_closed():
        db.close()


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _post(
    client: TestClient,
    query: str,
    key: str | None = None,
    variables: dict[str, object] | None = None,
) -> Any:
    headers = {} if key is None else _headers(key)
    return client.post("/", json={"query": query, "variables": variables or {}}, headers=headers)


def _data(payload: dict[str, Any]) -> dict[str, Any]:
    assert "errors" not in payload
    data = payload["data"]
    assert isinstance(data, dict)
    return data


def _extensions(payload: dict[str, Any]) -> dict[str, Any]:
    errors = payload["errors"]
    assert isinstance(errors, list) and errors
    error = errors[0]
    assert isinstance(error, dict)
    extensions = error["extensions"]
    assert isinstance(extensions, dict)
    return extensions


def _start(client: TestClient) -> None:
    operations = {
        "query": "mutation ($file: Upload!) { startJudging(entitiesCsv: $file) { isStarted } }",
        "variables": {"file": None},
    }
    response = client.post(
        "/",
        headers=_headers("admin-secret"),
        files={
            "operations": (None, json.dumps(operations), "application/json"),
            "map": (None, json.dumps({"0": ["variables.file"]}), "application/json"),
            "0": ("entities.csv", CSV.encode(), "text/csv"),
        },
    )
    assert response.status_code == 200
    payload: object = response.json()
    assert isinstance(payload, dict)
    assert _data(payload)["startJudging"] == {"isStarted": True}


def _pair(client: TestClient, key: str, force: bool) -> list[int]:
    response = _post(
        client,
        "query ($force: Boolean!) { pair(force: $force) { id } }",
        key,
        {"force": force},
    )
    assert response.status_code == 200
    payload: object = response.json()
    assert isinstance(payload, dict)
    pair = _data(payload)["pair"]
    assert isinstance(pair, list)
    ids: list[int] = []
    for entity in pair:
        assert isinstance(entity, dict)
        entity_id = entity["id"]
        assert isinstance(entity_id, int)
        ids.append(entity_id)
    return ids


def _submit(client: TestClient, key: str, entity_ids: list[int], winner_id: int) -> dict[str, Any]:
    response = _post(
        client,
        """
        mutation ($ids: [Int!]!, $winner: Int!) {
          submitComparison(entityIds: $ids, winnerId: $winner)
        }
        """,
        key,
        {"ids": entity_ids, "winner": winner_id},
    )
    assert response.status_code == 200
    payload: object = response.json()
    assert isinstance(payload, dict)
    return payload


def _session(client: TestClient) -> Session:
    session = client.app_state["session"]
    assert isinstance(session, Session)
    return session


def _model(client: TestClient) -> BayesianDecisionProcess:
    model = _session(client).worker.bdp
    assert model is not None
    return model


def _assignment(user_id: str) -> tuple[int, int] | None:
    try:
        row = AssignmentTable.get(AssignmentTable.judge_id == user_id)
    except DoesNotExist:
        return None
    return (int(row.entity_id_1), int(row.entity_id_2))


def test_missing_or_unknown_key_is_rejected(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    missing = _post(client, "{ session { isStarted } }")
    assert missing.status_code == 401
    missing_body: object = missing.json()
    assert isinstance(missing_body, dict)
    assert missing_body["detail"] == {"code": "UNAUTHENTICATED", "reason": "missing_key"}

    unknown_key = "not-a-real-key"
    unknown = _post(client, "{ session { isStarted } }", unknown_key)
    assert unknown.status_code == 401
    unknown_body: object = unknown.json()
    assert isinstance(unknown_body, dict)
    assert unknown_body["detail"] == {"code": "UNAUTHENTICATED", "reason": "unknown_key"}
    assert unknown_key not in unknown.text
    assert unknown_key not in caplog.text
    assert _session(client).get_enabled() is False


def test_websocket_without_a_key_is_rejected(client: TestClient) -> None:
    with pytest.raises(WebSocketDenialResponse):
        with client.websocket_connect("/", subprotocols=["graphql-transport-ws"]):
            pass


def test_judge_cannot_start_or_stop(client: TestClient) -> None:
    started = _post(client, "mutation { startJudging { isStarted } }", "judge-a-secret")
    assert started.status_code == 200
    started_body: object = started.json()
    assert isinstance(started_body, dict)
    assert _extensions(started_body)["code"] == "FORBIDDEN"
    assert _session(client).get_enabled() is False

    stopped = _post(client, "mutation { stopJudging { isStarted } }", "judge-a-secret")
    assert stopped.status_code == 200
    stopped_body: object = stopped.json()
    assert isinstance(stopped_body, dict)
    assert _extensions(stopped_body)["code"] == "FORBIDDEN"
    assert _session(client).get_enabled() is False


def test_pair_and_submit_limits_do_not_affect_another_user(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _start(client)
    assert "csv-secret-cell" not in caplog.text
    _pair(client, "judge-a-secret", force=False)
    held = _assignment("judge-a")
    assert held is not None
    model = _model(client)
    alpha = np.array(model.alpha_t)
    frequency = np.array(model.frequency)

    limited = _post(
        client,
        "query { pair(force: true) { id } }",
        "judge-a-secret",
    )
    assert limited.status_code == 200
    limited_body: object = limited.json()
    assert isinstance(limited_body, dict)
    extensions = _extensions(limited_body)
    assert extensions["code"] == "RATE_LIMITED"
    retry_after = extensions["retryAfterMs"]
    assert isinstance(retry_after, int)
    assert _assignment("judge-a") == held
    assert np.array_equal(np.array(model.alpha_t), alpha)
    assert np.array_equal(np.array(model.frequency), frequency)

    _pair(client, "judge-b-secret", force=False)
    assert _assignment("judge-a") == held
    assert _assignment("judge-b") is not None

    ids = list(held)
    missing = ({0, 1, 2} - set(ids)).pop()
    rejected = _submit(client, "judge-a-secret", ids, missing)
    assert _extensions(rejected)["code"] == "INCORRECT_PAIR_FORMAT"
    assert _assignment("judge-a") == held
    assert np.array_equal(np.array(model.alpha_t), alpha)

    blocked = _submit(client, "judge-a-secret", ids, ids[0])
    blocked_extensions = _extensions(blocked)
    assert blocked_extensions["code"] == "RATE_LIMITED"
    assert isinstance(blocked_extensions["retryAfterMs"], int)
    assert _assignment("judge-a") == held
    assert np.array_equal(np.array(model.alpha_t), alpha)

    other_ids = _pair(client, "judge-b-secret", force=False)
    applied = _submit(client, "judge-b-secret", other_ids, other_ids[0])
    assert _data(applied)["submitComparison"] is True
    assert _assignment("judge-a") == held
    still_limited = _submit(client, "judge-a-secret", ids, ids[0])
    assert _extensions(still_limited)["code"] == "RATE_LIMITED"
