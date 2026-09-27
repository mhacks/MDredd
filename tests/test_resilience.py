import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import db
from app.exceptions import WorkerUnavailableException
from app.main import app
from app.session import Session
from app.settings import settings
from app.worker import JudgeWorker


class WorkerKilled(BaseException):
    """Escapes the worker's error handling and ends its thread."""


@pytest.fixture(autouse=True)
def database(tmp_path: Path) -> None:
    db.init(str(tmp_path / "judge.db"), pragmas={"journal_mode": "wal"})


@pytest.fixture
def worker() -> Iterator[JudgeWorker]:
    worker = JudgeWorker(timeout=0.2, stuck_after=0.3)
    worker.start()
    yield worker
    worker.shutdown()


def kill(worker: JudgeWorker) -> None:
    def die() -> None:
        raise WorkerKilled()

    with pytest.raises(WorkerUnavailableException):
        worker._call(die)
    worker._thread.join(timeout=1)


def test_slow_job_times_out(worker: JudgeWorker) -> None:
    started = time.monotonic()
    with pytest.raises(WorkerUnavailableException):
        worker._call(lambda: time.sleep(0.5))
    assert time.monotonic() - started < 0.45


def test_worker_recovers_after_slow_job(worker: JudgeWorker) -> None:
    with pytest.raises(WorkerUnavailableException):
        worker._call(lambda: time.sleep(0.25))
    time.sleep(0.1)

    assert worker.healthy()
    assert worker.get_enabled() is False


def test_abandoned_job_is_skipped(worker: JudgeWorker) -> None:
    ran = threading.Event()
    blocker = threading.Thread(
        target=lambda: pytest.raises(
            WorkerUnavailableException, worker._call, lambda: time.sleep(0.4)
        )
    )
    blocker.start()
    time.sleep(0.05)
    with pytest.raises(WorkerUnavailableException):
        worker._call(ran.set)
    blocker.join()
    time.sleep(0.1)

    assert not ran.is_set()


def test_stuck_job_marks_worker_unhealthy(worker: JudgeWorker) -> None:
    with pytest.raises(WorkerUnavailableException):
        worker._call(lambda: time.sleep(0.6))
    time.sleep(0.2)

    assert not worker.healthy()


def test_dead_worker_fails_fast(worker: JudgeWorker) -> None:
    kill(worker)

    started = time.monotonic()
    with pytest.raises(WorkerUnavailableException):
        worker.get_enabled()
    assert time.monotonic() - started < 0.05
    assert not worker.healthy()


def test_watchdog_reports_dead_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "WATCHDOG_INTERVAL_SECONDS", 0.05)
    unhealthy = threading.Event()
    session = Session(on_unhealthy=unhealthy.set)
    try:
        kill(session.worker)
        assert unhealthy.wait(timeout=1)
    finally:
        session.close()


def test_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "WATCHDOG_INTERVAL_SECONDS", 60)
    sessions: list[Session] = []

    class RecordingSession(Session):
        def __init__(self) -> None:
            super().__init__()
            sessions.append(self)

    monkeypatch.setattr("app.main.Session", RecordingSession)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        kill(sessions[0].worker)

        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
