from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from app.algorithm import BayesianDecisionProcess
from app.db import JudgeRecord, close_db, db
from app.entity import Entity
from app.models import ComparisonInputModel, PairRequestModel
from app.worker import JudgeWorker


class ArgsortCounter:
    def __init__(self) -> None:
        self.calls = 0
        self._original = np.argsort

    def __call__(
        self, array: np.ndarray, *, kind: Literal["stable"] = "stable"
    ) -> np.ndarray:
        self.calls += 1
        return self._original(array, kind=kind)


def _entities(count: int) -> list[Entity]:
    return [Entity(attributes={"name": f"entity-{index}"}) for index in range(count)]


def _worker_with_alphas(alphas: list[float]) -> JudgeWorker:
    worker = JudgeWorker()
    entities = _entities(len(alphas))
    worker._install(
        JudgeRecord(
            enabled=True,
            headers=["name"],
            entities=entities,
            assignments={},
            bdp=BayesianDecisionProcess(
                K=len(alphas),
                alpha_t=jnp.asarray(alphas, dtype=jnp.float32),
                frequency=jnp.zeros(len(alphas), dtype=jnp.int32),
                key=jr.PRNGKey(0),
            ),
        )
    )
    return worker


@pytest.fixture
def isolated_database(tmp_path: Path) -> Iterator[None]:
    close_db()
    db.init(str(tmp_path / "mdredd.db"), pragmas={"journal_mode": "wal"})
    try:
        yield
    finally:
        close_db()


def test_rankings_are_descending_and_stable_for_ties() -> None:
    worker = _worker_with_alphas([1.0, 3.0, 3.0, 2.0])

    ranked = worker._rankings_snapshot()

    assert [entity.id for entity in ranked] == [1, 2, 3, 0]


def test_repeated_rankings_reuse_cached_order(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker_with_alphas([2.0, 4.0, 1.0, 3.0])
    counter = ArgsortCounter()
    monkeypatch.setattr(np, "argsort", counter)

    first = worker._rankings_snapshot()
    second = worker._rankings_snapshot()

    assert [entity.id for entity in first] == [entity.id for entity in second]
    assert counter.calls == 1


def test_pair_draw_keeps_cache_but_comparison_invalidates_it(
    isolated_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = JudgeWorker()
    counter = ArgsortCounter()
    monkeypatch.setattr(np, "argsort", counter)

    try:
        worker.start()
        worker.replace_entities(_entities(5), ["name"])
        worker.rankings()

        pair = worker.request_pair(PairRequestModel(uuid="judge"))
        worker.rankings()
        assert counter.calls == 1

        pair_ids = (pair[0].id, pair[1].id)
        worker.submit(
            ComparisonInputModel(
                uuid="judge", entity_ids=pair_ids, winner_id=pair_ids[0]
            )
        )
        worker.rankings()
        assert counter.calls == 2
    finally:
        worker.shutdown()


def test_replacing_entities_invalidates_cached_order(
    isolated_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = JudgeWorker()
    counter = ArgsortCounter()
    monkeypatch.setattr(np, "argsort", counter)

    try:
        worker.start()
        worker.replace_entities(_entities(3), ["name"])
        worker.rankings()
        worker.stop()
        worker.replace_entities(_entities(4), ["name"])
        worker.rankings()
        assert counter.calls == 2
    finally:
        worker.shutdown()
