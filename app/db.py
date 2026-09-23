from dataclasses import dataclass

from peewee import (
    BooleanField,
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
    db.create_tables([Judge, EntityRow, Assignment])


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
    model = (
        None
        if judge.bdp is None
        else BayesianDecisionProcess.model_validate(judge.bdp)
    )
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
        Assignment.delete().execute()
        if record.entities:
            EntityRow.insert_many(
                [
                    {"id": index, "attributes": entity.attributes}
                    for index, entity in enumerate(record.entities)
                ]
            ).execute()
        Judge.replace(
            id=1,
            enabled=record.enabled,
            headers=record.headers,
            bdp=bdp.snapshot(),
        ).execute()
    return record


def _store_model(bdp: BayesianDecisionProcess) -> None:
    Judge.update(bdp=bdp.snapshot()).where(Judge.id == 1).execute()


def save_assignment(
    bdp: BayesianDecisionProcess, judge_id: str, pair: tuple[int, int]
) -> None:
    with db.atomic():
        _store_model(bdp)
        Assignment.replace(
            judge_id=judge_id,
            entity_id_1=pair[0],
            entity_id_2=pair[1],
        ).execute()


def save_comparison(bdp: BayesianDecisionProcess, judge_id: str) -> None:
    with db.atomic():
        _store_model(bdp)
        Assignment.delete().where(Assignment.judge_id == judge_id).execute()


def save_enabled(enabled: bool) -> None:
    updated = Judge.update(enabled=enabled).where(Judge.id == 1).execute()
    if updated != 1:
        raise RuntimeError("Judge row is missing")
