from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from app.algorithm import BayesianDecisionProcess
from app.db import (
    Assignment,
    EntityRow,
    EntityState,
    Judge,
    JudgeState,
    close_db,
    db,
    load_state,
    open_db,
    replace_state,
    save_assignment,
    save_comparison,
)
from app.entity import Entity
from app.models import ComparisonInputModel, PairRequestModel
from app.worker import JudgeWorker


@pytest.fixture(autouse=True)
def isolated_database(tmp_path: Path) -> Iterator[None]:
    close_db()
    db.init(str(tmp_path / "mdredd.db"), pragmas={"journal_mode": "wal"})
    open_db()
    try:
        yield
    finally:
        close_db()


def _entities(count: int) -> list[Entity]:
    return [Entity(attributes={"name": f"entity-{index}"}) for index in range(count)]


def _model(frequencies: list[int]) -> BayesianDecisionProcess:
    count = len(frequencies)
    return BayesianDecisionProcess(
        K=count,
        alpha_t=jnp.linspace(1.0, 2.0, count, dtype=jnp.float32),
        frequency=jnp.asarray(frequencies, dtype=jnp.int32),
        key=jr.PRNGKey(42),
    )


def _assert_model_equal(
    actual: BayesianDecisionProcess | None, expected: BayesianDecisionProcess
) -> None:
    assert actual is not None
    assert actual.K == expected.K
    np.testing.assert_array_equal(actual.alpha_t, expected.alpha_t)
    np.testing.assert_array_equal(actual.frequency, expected.frequency)
    np.testing.assert_array_equal(actual.key, expected.key)


def test_replace_state_round_trips_binary_model_state() -> None:
    model = _model([3, 1, 4, 2])

    replace_state(["name"], _entities(model.K), model)
    record = load_state()

    _assert_model_equal(record.bdp, model)
    assert record.enabled
    assert record.headers == ["name"]
    assert Judge.get_by_id(1).bdp is None
    assert len(JudgeState.get_by_id(1).alpha) == model.K * 4
    assert len(JudgeState.get_by_id(1).key) == 8


def test_pair_save_updates_only_frequency_key_and_assignment() -> None:
    model = _model([0, 0, 0, 0])
    replace_state(["name"], _entities(model.K), model)
    alpha_before = bytes(JudgeState.get_by_id(1).alpha)

    pair = model.get_next_pair()
    save_assignment(model, "judge", pair)
    record = load_state()

    _assert_model_equal(record.bdp, model)
    assert bytes(JudgeState.get_by_id(1).alpha) == alpha_before
    assert Judge.get_by_id(1).bdp is None
    assert record.assignments == {"judge": pair}


def test_comparison_save_updates_alpha_and_removes_assignment() -> None:
    model = _model([0, 0, 0, 0])
    replace_state(["name"], _entities(model.K), model)
    pair = model.get_next_pair()
    save_assignment(model, "judge", pair)
    frequency_before = np.asarray(model.frequency)

    model.submit_comparison(pair[0], pair[1], pair[0])
    save_comparison(model, "judge")
    record = load_state()

    _assert_model_equal(record.bdp, model)
    assert record.bdp is not None
    np.testing.assert_array_equal(record.bdp.frequency, frequency_before)
    assert record.assignments == {}


def test_pair_save_rolls_back_when_entity_state_is_incomplete() -> None:
    model = _model([0, 0, 0, 0])
    replace_state(["name"], _entities(model.K), model)
    pair = model.get_next_pair()
    EntityState.delete().where(EntityState.entity_id == pair[1]).execute()
    key_before = bytes(JudgeState.get_by_id(1).key)
    frequency_before = EntityState.get_by_id(pair[0]).frequency

    with pytest.raises(RuntimeError, match="Pair state is missing"):
        save_assignment(model, "judge", pair)

    assert bytes(JudgeState.get_by_id(1).key) == key_before
    assert EntityState.get_by_id(pair[0]).frequency == frequency_before
    assert Assignment.get_or_none(Assignment.judge_id == "judge") is None


def test_open_db_migrates_legacy_json_state() -> None:
    model = _model([5, 3, 1])
    EntityRow.insert_many(
        [
            {"id": index, "attributes": entity.attributes}
            for index, entity in enumerate(_entities(model.K))
        ]
    ).execute()
    Judge.create(id=1, enabled=True, headers=["name"], bdp=model.snapshot())

    close_db()
    open_db()
    record = load_state()

    _assert_model_equal(record.bdp, model)
    assert Judge.get_by_id(1).bdp is None
    assert EntityState.select().count() == model.K


def test_legacy_json_state_supersedes_stale_binary_state() -> None:
    stale = _model([0, 0])
    replace_state(["name"], _entities(stale.K), stale)

    current = _model([8, 5, 3])
    EntityRow.delete().execute()
    EntityRow.insert_many(
        [
            {"id": index, "attributes": entity.attributes}
            for index, entity in enumerate(_entities(current.K))
        ]
    ).execute()
    Judge.update(bdp=current.snapshot()).where(Judge.id == 1).execute()

    close_db()
    open_db()

    _assert_model_equal(load_state().bdp, current)
    assert Judge.get_by_id(1).bdp is None
    assert EntityState.select().count() == current.K


def test_worker_recovers_assignment_and_comparison() -> None:
    close_db()
    first = JudgeWorker()
    try:
        first.start()
        first.replace_entities(_entities(4), ["name"])
        pair = first.request_pair(PairRequestModel(uuid="judge"))
        pair_ids = (pair[0].id, pair[1].id)
    finally:
        first.shutdown()

    recovered = JudgeWorker()
    try:
        recovered.start()
        assigned = recovered.request_pair(PairRequestModel(uuid="judge"))
        assert (assigned[0].id, assigned[1].id) == pair_ids
        recovered.submit(
            ComparisonInputModel(
                uuid="judge", entity_ids=pair_ids, winner_id=pair_ids[0]
            )
        )
    finally:
        recovered.shutdown()

    open_db()
    record = load_state()
    assert record.bdp is not None
    assert "judge" not in record.assignments
    assert not np.allclose(record.bdp.alpha_t, np.ones(record.bdp.K))
