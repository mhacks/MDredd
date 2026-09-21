import io
import os

import numpy as np
import pytest
from fastapi import UploadFile

from app.db import db
from app.main import create_app
from app.models import ComparisonInputModel, PairRequestModel
from app.session import Session
from app.settings import settings

CSV = """Project Title,Submission Url,Table Number,Highest Step Completed,M Hacks Main Track
Alpha,http://alpha.example,1,Submit,General
Beta,http://beta.example,2,Submit,General
Gamma,http://gamma.example,3,Submit,General
"""


@pytest.fixture
def judging_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SNAPSHOT_INTERVAL", 1000)
    if not db.is_closed():
        db.close()
    db.init(str(tmp_path / "mdredd.db"))
    yield
    if not db.is_closed():
        db.close()


def _csv_upload() -> UploadFile:
    return UploadFile(file=io.BytesIO(CSV.encode()), filename="entities.csv")


def _compare_once(session: Session, judge: str, choose_left: bool) -> None:
    left, right = session.get_pair(PairRequestModel(uuid=judge, force=False))
    winner = left.id if choose_left else right.id
    session.submit_pair(
        ComparisonInputModel(
            uuid=judge,
            entity_ids=(left.id, right.id),
            winner_id=winner,
        )
    )
    session.worker.flush()


def test_reload_matches_alphas_after_later_comparisons(judging_db):
    session = Session()
    try:
        session.start(_csv_upload())
        _compare_once(session, "judge-a", choose_left=True)
        session.worker.snapshot()

        _compare_once(session, "judge-a", choose_left=False)
        _compare_once(session, "judge-b", choose_left=True)
        expected = session.worker.alphas()
    finally:
        session.close()

    reloaded = Session()
    try:
        actual = reloaded.worker.alphas()
    finally:
        reloaded.close()

    assert np.array_equal(expected, actual)


def test_crash_route_is_registered_only_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CRASH_ROUTE", False)
    assert "/dev/crash" not in _paths(create_app())

    monkeypatch.setattr(settings, "ENABLE_CRASH_ROUTE", True)
    assert "/dev/crash" in _paths(create_app())


def test_crash_route_exits_the_process(monkeypatch):
    exited = {}

    def _exit(code):
        exited["code"] = code

    monkeypatch.setattr(os, "_exit", _exit)
    from app.api.dev_router import crash

    crash()
    assert exited["code"] == 1


def _paths(application) -> set[str]:
    paths: set[str] = set()
    for route in application.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)
    return paths
