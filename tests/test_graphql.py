import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from app.algorithm import BayesianDecisionProcess
from app.db import db
from app.main import create_app
from app.session import Session
from app.settings import settings

CSV = """Project Title,Submission Url,Table Number,Highest Step Completed,M Hacks Main Track
Alpha,http://alpha.example,1,Submit,General
Beta,http://beta.example,2,Submit,General
Gamma,http://gamma.example,3,Submit,General
"""


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
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


def _graphql(
    client: TestClient,
    query: str,
    variables: dict[str, object] | None = None,
) -> dict[str, Any]:
    response = client.post("/", json={"query": query, "variables": variables or {}})
    assert response.status_code == 200
    payload: object = response.json()
    assert isinstance(payload, dict)
    return payload


def _error_code(payload: dict[str, Any]) -> str:
    errors = payload["errors"]
    assert isinstance(errors, list) and errors
    error = errors[0]
    assert isinstance(error, dict)
    extensions = error["extensions"]
    assert isinstance(extensions, dict)
    code = extensions["code"]
    assert isinstance(code, str)
    return code


def _data(payload: dict[str, Any]) -> dict[str, Any]:
    assert "errors" not in payload
    data = payload["data"]
    assert isinstance(data, dict)
    return data


def _start(client: TestClient) -> None:
    operations = {
        "query": """
            mutation Start($file: Upload!) {
              startJudging(entitiesCsv: $file) { isStarted }
            }
        """,
        "variables": {"file": None},
    }
    response = client.post(
        "/",
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


def _pair(client: TestClient, judge_id: str, force: bool = False) -> list[dict[str, Any]]:
    payload = _graphql(
        client,
        """
        query Pair($judgeId: String!, $force: Boolean!) {
          pair(judgeId: $judgeId, force: $force) { id projectName }
        }
        """,
        {"judgeId": judge_id, "force": force},
    )
    pair = _data(payload)["pair"]
    assert isinstance(pair, list) and len(pair) == 2
    entities: list[dict[str, Any]] = []
    for entity in pair:
        assert isinstance(entity, dict)
        entities.append(entity)
    return entities


def _ids(pair: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for entity in pair:
        entity_id = entity["id"]
        assert isinstance(entity_id, int)
        ids.append(entity_id)
    return ids


def _submit(
    client: TestClient,
    judge_id: str,
    entity_ids: list[int],
    winner_id: int,
) -> dict[str, Any]:
    return _graphql(
        client,
        """
        mutation Submit($judgeId: String!, $entityIds: [Int!]!, $winnerId: Int!) {
          submitComparison(
            judgeId: $judgeId
            entityIds: $entityIds
            winnerId: $winnerId
          )
        }
        """,
        {"judgeId": judge_id, "entityIds": entity_ids, "winnerId": winner_id},
    )


def _ranking_names(client: TestClient) -> list[str]:
    payload = _graphql(client, "query { rankings { projectName } }")
    rankings = _data(payload)["rankings"]
    assert isinstance(rankings, list)
    names: list[str] = []
    for entity in rankings:
        assert isinstance(entity, dict)
        name = entity["projectName"]
        assert isinstance(name, str)
        names.append(name)
    return names


def _session(client: TestClient) -> Session:
    session = client.app_state["session"]
    assert isinstance(session, Session)
    return session


def _mutation_names(client: TestClient) -> set[str]:
    payload = _graphql(
        client,
        """
        {
          __type(name: "Mutation") {
            fields { name }
          }
        }
        """,
    )
    mutation = _data(payload)["__type"]
    assert isinstance(mutation, dict)
    fields = mutation["fields"]
    assert isinstance(fields, list)
    names: set[str] = set()
    for field in fields:
        assert isinstance(field, dict)
        name = field["name"]
        assert isinstance(name, str)
        names.add(name)
    return names


def test_start_pair_submit_changes_rankings(client: TestClient) -> None:
    assert _data(_graphql(client, "query { session { isStarted } }"))["session"] == {
        "isStarted": False
    }
    _start(client)
    assert _data(_graphql(client, "query { session { isStarted } }"))["session"] == {
        "isStarted": True
    }

    pair = _pair(client, "judge-a")
    before = _ranking_names(client)
    winner = max(pair, key=lambda entity: int(entity["id"]))
    loser = min(pair, key=lambda entity: int(entity["id"]))
    winner_id = winner["id"]
    assert isinstance(winner_id, int)
    submitted = _submit(client, "judge-a", _ids(pair), winner_id)
    assert _data(submitted)["submitComparison"] is True

    deadline = time.monotonic() + 5
    after = before
    while time.monotonic() < deadline:
        after = _ranking_names(client)
        if after != before:
            break
        time.sleep(0.02)

    assert after != before
    assert after.index(str(winner["projectName"])) < after.index(str(loser["projectName"]))


def test_second_judge_does_not_receive_a_checked_out_pair(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start(client)
    samples = iter(((0, 1), (1, 2)))

    def next_pair(self: BayesianDecisionProcess, temp: float = 1.0) -> tuple[int, int]:
        return next(samples)

    monkeypatch.setattr(BayesianDecisionProcess, "get_next_pair", next_pair)
    checked_out = _ids(_pair(client, "judge-a"))
    assert checked_out == [0, 1]

    other = _ids(_pair(client, "judge-b"))
    assert other == [1, 2]
    assert other != checked_out
    assert _ids(_pair(client, "judge-a")) == checked_out
    assert _ids(_pair(client, "judge-b")) == other


def test_judging_failures_use_graphql_error_codes(client: TestClient) -> None:
    assert _error_code(_graphql(client, 'query { pair(judgeId: "judge-a") { id } }')) == (
        "JUDGING_NOT_STARTED"
    )
    assert _error_code(_graphql(client, "query { rankings { projectName } }")) == (
        "JUDGING_NOT_STARTED"
    )
    assert (
        _error_code(
            _submit(client, "judge-a", [0, 1], 0),
        )
        == "JUDGING_NOT_STARTED"
    )
    assert _error_code(_graphql(client, "mutation { stopJudging { isStarted } }")) == (
        "JUDGING_NOT_STARTED"
    )
    assert _error_code(_graphql(client, "mutation { resumeJudging { isStarted } }")) == (
        "JUDGING_NEVER_STARTED"
    )

    _start(client)
    assert _error_code(_graphql(client, "mutation { startJudging { isStarted } }")) == (
        "JUDGING_ALREADY_STARTED"
    )

    pair = _pair(client, "judge-a")
    ids = _ids(pair)
    missing = ({0, 1, 2} - set(ids)).pop()
    assert _error_code(_submit(client, "judge-a", ids, missing)) == "INCORRECT_PAIR_FORMAT"
    assert _error_code(_submit(client, "judge-a", [missing, ids[0]], ids[0])) == (
        "JUDGE_DOES_NOT_OWN_PAIR"
    )

    assert _data(_graphql(client, "mutation { stopJudging { isStarted } }"))["stopJudging"] == {
        "isStarted": False
    }
    assert _data(_graphql(client, "mutation { resumeJudging { isStarted } }"))[
        "resumeJudging"
    ] == {"isStarted": True}


def test_rankings_updated_is_pushed_when_the_worker_publishes(client: TestClient) -> None:
    _start(client)
    pair = _pair(client, "judge-a")
    received: list[dict[str, Any]] = []
    subscribed = threading.Event()

    def listen() -> None:
        with client.websocket_connect(
            "/", subprotocols=["graphql-transport-ws"]
        ) as websocket:
            websocket.send_json({"type": "connection_init"})
            ack = websocket.receive_json()
            assert ack["type"] == "connection_ack"
            websocket.send_json(
                {
                    "id": "1",
                    "type": "subscribe",
                    "payload": {
                        "query": "subscription { rankingsUpdated { projectName } }",
                    },
                }
            )
            subscribed.set()
            message: object = websocket.receive_json()
            assert isinstance(message, dict)
            received.append(message)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    assert subscribed.wait(timeout=5)

    session = _session(client)
    deadline = time.monotonic() + 5
    while session.worker.ranking_subscriber_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.worker.ranking_subscriber_count() == 1

    winner = max(pair, key=lambda entity: int(entity["id"]))
    winner_id = winner["id"]
    assert isinstance(winner_id, int)
    assert _data(_submit(client, "judge-a", _ids(pair), winner_id))["submitComparison"] is True
    listener.join(timeout=5)
    assert not listener.is_alive()
    assert received[0]["type"] == "next"
    payload = received[0]["payload"]
    assert isinstance(payload, dict)
    data = payload["data"]
    assert isinstance(data, dict)
    rankings = data["rankingsUpdated"]
    assert isinstance(rankings, list) and rankings


def test_crash_mutation_is_registered_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "ENABLE_CRASH_ROUTE", False)
    with _open_client(tmp_path / "off") as disabled:
        assert "crash" not in _mutation_names(disabled)

    monkeypatch.setattr(settings, "ENABLE_CRASH_ROUTE", True)
    with _open_client(tmp_path / "on") as enabled:
        assert "crash" in _mutation_names(enabled)


def test_crash_mutation_exits_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "ENABLE_CRASH_ROUTE", True)
    exited: dict[str, int] = {}

    def _exit(code: int) -> None:
        exited["code"] = code

    monkeypatch.setattr(os, "_exit", _exit)
    with _open_client(tmp_path) as client:
        _graphql(client, "mutation { crash }")
    assert exited["code"] == 1
