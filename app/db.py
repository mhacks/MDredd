from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from peewee import (
    BlobField,
    BooleanField,
    Case,
    IntegerField,
    JSONField,
    Model,
    SqliteDatabase,
    TextField,
)

from app.algorithm import BayesianDecisionProcess
from app.entity import Entity
from app.settings import settings

db = SqliteDatabase(settings.DB_FILE, pragmas={"journal_mode": "wal"})


class Judge(Model):
    id = IntegerField(primary_key=True)
    enabled = BooleanField()
    headers = JSONField()
    bdp = JSONField(null=True)

    class Meta:
        database = db
        table_name = "judge"


class EntityRow(Model):
    id = IntegerField(primary_key=True)
    attributes = JSONField()

    class Meta:
        database = db
        table_name = "entities"


class Assignment(Model):
    judge_id = TextField(primary_key=True)
    entity_id_1 = IntegerField()
    entity_id_2 = IntegerField()

    class Meta:
        database = db
        table_name = "assignments"


class JudgeState(Model):
    id = IntegerField(primary_key=True)
    alpha = BlobField()
    key = BlobField()

    class Meta:
        database = db
        table_name = "judge_state"


class EntityState(Model):
    entity_id = IntegerField(primary_key=True)
    frequency = IntegerField()

    class Meta:
        database = db
        table_name = "entity_state"


@dataclass
class JudgeRecord:
    enabled: bool
    headers: list[str]
    entities: list[Entity]
    assignments: dict[str, tuple[int, int]]
    bdp: BayesianDecisionProcess | None


def open_db() -> None:
    if db.is_closed():
        db.connect()
    db.create_tables([Judge, EntityRow, Assignment, JudgeState, EntityState])
    _migrate_legacy_state()


def close_db() -> None:
    if not db.is_closed():
        db.close()


def load_state() -> JudgeRecord:
    entities = [
        Entity(attributes=row.attributes)
        for row in EntityRow.select().order_by(EntityRow.id)
    ]
    assignments = {
        row.judge_id: (row.entity_id_1, row.entity_id_2) for row in Assignment.select()
    }
    judge = Judge.get_or_none(Judge.id == 1)
    if judge is None:
        return JudgeRecord(
            enabled=False,
            headers=[],
            entities=entities,
            assignments=assignments,
            bdp=None,
        )
    model = _load_model(len(entities))
    return JudgeRecord(
        enabled=bool(judge.enabled) and model is not None,
        headers=list(judge.headers),
        entities=entities,
        assignments=assignments,
        bdp=model,
    )


def replace_state(
    headers: list[str], entities: list[Entity], bdp: BayesianDecisionProcess
) -> JudgeRecord:
    record = JudgeRecord(
        enabled=True,
        headers=list(headers),
        entities=list(entities),
        assignments={},
        bdp=bdp,
    )
    with db.atomic():
        EntityRow.delete().execute()
        EntityState.delete().execute()
        Assignment.delete().execute()
        if record.entities:
            EntityRow.insert_many(
                [
                    {"id": index, "attributes": entity.attributes}
                    for index, entity in enumerate(record.entities)
                ]
            ).execute()
            frequency = np.asarray(bdp.frequency)
            EntityState.insert_many(
                [
                    {"entity_id": index, "frequency": int(frequency[index])}
                    for index in range(len(record.entities))
                ]
            ).execute()
        JudgeState.replace(
            id=1,
            alpha=_array_bytes(bdp.alpha_t, np.dtype("<f4")),
            key=_array_bytes(bdp.key, np.dtype("<u4")),
        ).execute()
        Judge.replace(
            id=1,
            enabled=record.enabled,
            headers=record.headers,
            bdp=None,
        ).execute()
    return record


def save_assignment(
    bdp: BayesianDecisionProcess, judge_id: str, pair: tuple[int, int]
) -> None:
    left, right = pair
    frequencies = (
        int(bdp.frequency[left]),
        int(bdp.frequency[right]),
    )
    with db.atomic():
        updated = (
            EntityState.update(
                frequency=Case(
                    EntityState.entity_id,
                    ((left, frequencies[0]), (right, frequencies[1])),
                )
            )
            .where(EntityState.entity_id.in_(pair))
            .execute()
        )
        if updated != 2:
            raise RuntimeError("Pair state is missing")
        updated = (
            JudgeState.update(key=_array_bytes(bdp.key, np.dtype("<u4")))
            .where(JudgeState.id == 1)
            .execute()
        )
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        Assignment.replace(
            judge_id=judge_id,
            entity_id_1=left,
            entity_id_2=right,
        ).execute()


def save_comparison(bdp: BayesianDecisionProcess, judge_id: str) -> None:
    with db.atomic():
        updated = (
            JudgeState.update(alpha=_array_bytes(bdp.alpha_t, np.dtype("<f4")))
            .where(JudgeState.id == 1)
            .execute()
        )
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        Assignment.delete().where(Assignment.judge_id == judge_id).execute()


def save_enabled(enabled: bool) -> None:
    updated = Judge.update(enabled=enabled).where(Judge.id == 1).execute()
    if updated != 1:
        raise RuntimeError("Judge row is missing")


def _load_model(entity_count: int) -> BayesianDecisionProcess | None:
    state = JudgeState.get_or_none(JudgeState.id == 1)
    if state is None:
        return None

    frequency_rows = cast(
        list[tuple[int, int]],
        list(
            EntityState.select(EntityState.entity_id, EntityState.frequency)
            .order_by(EntityState.entity_id)
            .tuples()
        ),
    )
    if len(frequency_rows) != entity_count or any(
        entity_id != index
        for index, (entity_id, _frequency) in enumerate(frequency_rows)
    ):
        raise RuntimeError("Entity state does not match the entity table")
    alpha = _array_from_bytes(state.alpha, np.dtype("<f4"), entity_count, "alpha")
    key = _array_from_bytes(state.key, np.dtype("<u4"), 2, "key")
    return BayesianDecisionProcess.model_validate(
        {
            "K": entity_count,
            "alpha_t": alpha,
            "frequency": [frequency for _entity_id, frequency in frequency_rows],
            "key": key,
        }
    )


def _migrate_legacy_state() -> None:
    judge = Judge.get_or_none(Judge.id == 1)
    if judge is None or judge.bdp is None:
        return
    model = BayesianDecisionProcess.model_validate(judge.bdp)
    frequencies = np.asarray(model.frequency, dtype=np.int32)
    with db.atomic():
        EntityState.delete().execute()
        JudgeState.replace(
            id=1,
            alpha=_array_bytes(model.alpha_t, np.dtype("<f4")),
            key=_array_bytes(model.key, np.dtype("<u4")),
        ).execute()
        if frequencies.size:
            EntityState.insert_many(
                [
                    {"entity_id": index, "frequency": int(frequency)}
                    for index, frequency in enumerate(frequencies)
                ]
            ).execute()
        Judge.update(bdp=None).where(Judge.id == 1).execute()


def _array_bytes(array: object, dtype: np.dtype[Any]) -> bytes:
    return np.asarray(array, dtype=dtype).tobytes()


def _array_from_bytes(
    payload: bytes, dtype: np.dtype[Any], size: int, name: str
) -> np.ndarray:
    array = np.frombuffer(payload, dtype=dtype)
    if array.size != size:
        raise RuntimeError(f"Stored {name} state has an invalid size")
    return array.copy()
