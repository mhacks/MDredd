from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.db import db
from app.main import app
from app.ratelimit import limiter
from app.settings import settings

CSV = b"Team Name,1st Place\nNorth,yes\nSouth,no\n"
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db.init(str(tmp_path / "judge.db"), pragmas={"journal_mode": "wal"})
    limiter.reset()
    with TestClient(app) as test_client:
        yield test_client


def upload(client: TestClient, payload: bytes = CSV) -> Response:
    return client.post(
        "/datasets",
        headers=AUTH,
        files={"entities_csv": ("entities.csv", payload, "text/csv")},
    )


def test_health_does_not_require_a_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_or_wrong_token_is_rejected(client: TestClient) -> None:
    missing = client.get("/judging")
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "missing_key"

    wrong = client.get("/judging", headers={"Authorization": "Bearer " + "n" * 32})
    assert wrong.status_code == 401
    assert wrong.json()["detail"]["code"] == "unknown_key"


def test_openapi_lists_rest_routes(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/datasets",
        "/judging",
        "/judging/start",
        "/judging/stop",
        "/judging/resume",
        "/columns",
        "/rows/{row_id}",
        "/rankings",
        "/pairs",
        "/comparisons",
    } <= paths
    assert "/dev/crash" not in paths


def test_dataset_rows_keep_csv_headers(client: TestClient) -> None:
    created = upload(client)
    assert created.status_code == 200
    assert created.json() == {
        "is_started": True,
        "headers": ["Team Name", "1st Place"],
    }

    judging = client.get("/judging", headers=AUTH)
    assert judging.json() == {"is_started": True}
    columns = client.get("/columns", headers=AUTH)
    assert columns.json() == {"headers": ["Team Name", "1st Place"]}

    row = client.get("/rows/0", headers=AUTH)
    assert row.status_code == 200
    assert row.json() == {
        "id": 0,
        "attributes": {"Team Name": "North", "1st Place": "yes"},
    }
    missing = client.get("/rows/9", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "UNKNOWN_ROW"


def test_judges_keep_separate_pairs(client: TestClient) -> None:
    assert upload(client).status_code == 200

    first = client.post("/pairs", headers=AUTH, json={"judge_id": "ada"})
    assert first.status_code == 200
    again = client.post("/pairs", headers=AUTH, json={"judge_id": "ada"})
    assert again.json()["pair"] == first.json()["pair"]

    other = client.post("/pairs", headers=AUTH, json={"judge_id": "grace"})
    assert other.status_code == 200

    pair = first.json()["pair"]
    ids = [pair[0]["id"], pair[1]["id"]]
    submitted = client.post(
        "/comparisons",
        headers=AUTH,
        json={"judge_id": "ada", "entity_ids": ids, "winner_id": ids[0]},
    )
    assert submitted.status_code == 200
    assert submitted.json() == {"ok": True}

    other_pair = other.json()["pair"]
    other_ids = [other_pair[0]["id"], other_pair[1]["id"]]
    other_submit = client.post(
        "/comparisons",
        headers=AUTH,
        json={
            "judge_id": "grace",
            "entity_ids": other_ids,
            "winner_id": other_ids[0],
        },
    )
    assert other_submit.status_code == 200

    rankings = client.get("/rankings", headers=AUTH)
    assert rankings.status_code == 200
    assert sorted(row["id"] for row in rankings.json()) == [0, 1]


def test_repeated_comparison_is_accepted(client: TestClient) -> None:
    assert upload(client).status_code == 200
    drawn = client.post("/pairs", headers=AUTH, json={"judge_id": "ada"}).json()["pair"]
    body = {
        "judge_id": "ada",
        "entity_ids": [drawn[0]["id"], drawn[1]["id"]],
        "winner_id": drawn[0]["id"],
    }
    assert client.post("/comparisons", headers=AUTH, json=body).status_code == 200
    assert client.post("/comparisons", headers=AUTH, json=body).status_code == 200


def test_submit_must_match_the_open_pair(client: TestClient) -> None:
    assert upload(client).status_code == 200
    drawn = client.post("/pairs", headers=AUTH, json={"judge_id": "ada"}).json()["pair"]
    ids = [drawn[0]["id"], drawn[1]["id"]]

    rejected = client.post(
        "/comparisons",
        headers=AUTH,
        json={"judge_id": "ada", "entity_ids": ids, "winner_id": 99},
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "INCORRECT_PAIR_FORMAT"

    foreign = client.post(
        "/comparisons",
        headers=AUTH,
        json={"judge_id": "grace", "entity_ids": ids, "winner_id": ids[0]},
    )
    assert foreign.status_code == 409
    assert foreign.json()["detail"]["code"] == "JUDGE_DOES_NOT_OWN_PAIR"


def test_stop_and_resume(client: TestClient) -> None:
    assert upload(client).status_code == 200
    stopped = client.post("/judging/stop", headers=AUTH)
    assert stopped.json() == {"is_started": False}

    blocked = client.post("/pairs", headers=AUTH, json={"judge_id": "ada"})
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "JUDGING_NOT_STARTED"

    resumed = client.post("/judging/resume", headers=AUTH)
    assert resumed.json() == {"is_started": True}
    assert client.post("/pairs", headers=AUTH, json={"judge_id": "ada"}).status_code == 200


def test_upload_replaces_a_stopped_dataset(client: TestClient) -> None:
    assert upload(client).status_code == 200
    conflict = upload(client)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "JUDGING_ALREADY_STARTED"

    assert client.post("/judging/stop", headers=AUTH).status_code == 200
    replaced = upload(client, b"Project,Track\nOne,A\nTwo,B\n")
    assert replaced.status_code == 200
    assert replaced.json()["headers"] == ["Project", "Track"]


def test_bad_csv_is_rejected(client: TestClient) -> None:
    short = upload(client, b"Name\nOnly\n")
    assert short.status_code == 422
    assert short.json()["detail"]["code"] == "TOO_FEW_ENTITIES"

    duplicate = upload(client, b"Name,Name\nA,B\nC,D\n")
    assert duplicate.status_code == 422
    assert duplicate.json()["detail"]["code"] == "INVALID_COLUMNS"

    rankings = client.get("/rankings", headers=AUTH)
    assert rankings.status_code == 409
    assert rankings.json()["detail"]["code"] == "JUDGING_NEVER_STARTED"


def test_pair_rate_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PAIR_CAPACITY", 1)
    monkeypatch.setattr(settings, "PAIR_REFILL_PER_SECOND", 0)
    assert upload(client).status_code == 200

    assert client.post("/pairs", headers=AUTH, json={"judge_id": "ada"}).status_code == 200
    limited = client.post("/pairs", headers=AUTH, json={"judge_id": "grace"})
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "RATE_LIMITED"
    assert limited.headers["retry-after"]
