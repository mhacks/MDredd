from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from app.db import db
from app.entity import Entity
from app.exceptions import JudgeDoesNotOwnPairException
from app.models import ComparisonInputModel, PairRequestModel
from app.worker import JudgeWorker

JUDGE = "api"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "judge.db"
    db.init(str(path), pragmas={"journal_mode": "wal"})
    return path


@pytest.fixture
def worker(db_path: Path) -> Iterator[JudgeWorker]:
    worker = start_worker()
    worker.replace_entities(
        [Entity(attributes={"name": name}) for name in "ABCDE"], ["name"]
    )
    yield worker
    worker.shutdown()


def start_worker() -> JudgeWorker:
    worker = JudgeWorker()
    worker.start()
    return worker


def draw(worker: JudgeWorker) -> tuple[int, int]:
    left, right = worker.request_pair(PairRequestModel(uuid=JUDGE))
    return left.id, right.id


def submit(worker: JudgeWorker, pair: tuple[int, int], winner: int) -> None:
    worker.submit(
        ComparisonInputModel(uuid=JUDGE, entity_ids=pair, winner_id=winner)
    )


def alphas(worker: JudgeWorker) -> np.ndarray:
    assert worker.bdp is not None
    return worker.bdp.get_alphas().copy()


def test_retry_after_success_is_not_counted_twice(worker: JudgeWorker) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])
    after_first = alphas(worker)

    submit(worker, pair, pair[0])

    np.testing.assert_array_equal(alphas(worker), after_first)


def test_retry_with_ids_reversed_is_not_counted_twice(worker: JudgeWorker) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])
    after_first = alphas(worker)

    submit(worker, (pair[1], pair[0]), pair[0])

    np.testing.assert_array_equal(alphas(worker), after_first)


def test_retry_after_next_pair_is_not_counted(worker: JudgeWorker) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])
    draw(worker)
    after_first = alphas(worker)

    submit(worker, pair, pair[0])

    np.testing.assert_array_equal(alphas(worker), after_first)


def test_retry_with_different_winner_is_rejected(worker: JudgeWorker) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])

    with pytest.raises(JudgeDoesNotOwnPairException):
        submit(worker, pair, pair[1])


def test_retry_is_recognized_after_restart(
    worker: JudgeWorker, db_path: Path
) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])
    after_first = alphas(worker)
    worker.shutdown()

    restarted = start_worker()
    try:
        submit(restarted, pair, pair[0])
        np.testing.assert_array_equal(alphas(restarted), after_first)
    finally:
        restarted.shutdown()


def test_duplicate_entity_ids_are_rejected(worker: JudgeWorker) -> None:
    pair = draw(worker)
    before = alphas(worker)

    with pytest.raises(JudgeDoesNotOwnPairException):
        submit(worker, (pair[0], pair[0]), pair[0])

    np.testing.assert_array_equal(alphas(worker), before)


def test_new_entities_clear_completed_comparisons(worker: JudgeWorker) -> None:
    pair = draw(worker)
    submit(worker, pair, pair[0])
    worker.stop()
    worker.replace_entities(
        [Entity(attributes={"name": name}) for name in "ABCDE"], ["name"]
    )

    with pytest.raises(JudgeDoesNotOwnPairException):
        submit(worker, pair, pair[0])
