import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from peewee import (
    BlobField,
    BooleanField,
    FloatField,
    Case,
    IntegerField,
    JSONField,
    Model,
    SqliteDatabase,
    TextField,
)

from app.algorithm import BayesianDecisionProcess
from app.entity import Entity
from app.exceptions import UnknownArchiveException
from app.logging import attach_log_file, detach_log_file, log_path
from app.settings import settings

logger = logging.getLogger(__name__)

# FULL fsyncs each commit. A lock waits one second, then the worker fails the command.
db = SqliteDatabase(
    settings.DB_FILE,
    pragmas={
        "journal_mode": "wal",
        "synchronous": "FULL",
        "busy_timeout": 1000,
    },
)


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
    # Unix time the pair was handed out, for the judge's timer.
    assigned_at = FloatField(null=True)

    class Meta:
        database = db
        table_name = "assignments"


class CompletedComparison(Model):
    judge_id = TextField(primary_key=True)
    entity_id_1 = IntegerField()
    entity_id_2 = IntegerField()
    winner_id = IntegerField()

    class Meta:
        database = db
        table_name = "completed_comparisons"


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
    strikes = IntegerField(default=0)

    class Meta:
        database = db
        table_name = "entity_state"


class LastSkip(Model):
    judge_id = TextField(primary_key=True)
    absent_ids = JSONField()
    entity_id_1 = IntegerField(null=True)
    entity_id_2 = IntegerField(null=True)

    class Meta:
        database = db
        table_name = "last_skips"


class ProjectTable(Model):
    project_url = TextField(primary_key=True)
    table_number = IntegerField()

    class Meta:
        database = db
        table_name = "project_tables"


# Absent project ids, and the pair produced by that report. A missing pair
# means the report committed and the next draw did not.
AbsentSkip = tuple[tuple[int, ...], tuple[int, int] | None]


@dataclass
class JudgeRecord:
    enabled: bool
    headers: list[str]
    entities: list[Entity]
    assignments: dict[str, tuple[int, int]]
    # Judge id to the Unix time their open pair was handed out.
    assigned_at: dict[str, float]
    completed: dict[str, tuple[int, int, int]]
    strikes: list[int]
    last_skips: dict[str, AbsentSkip]
    bdp: BayesianDecisionProcess | None
    # Normalized project URL to table number, sent by the dashboard. It is kept
    # across uploads, since it is keyed by URL and not by row.
    tables: dict[str, int]


def open_db() -> None:
    if db.is_closed():
        db.connect()
    db.create_tables(
        [
            Judge,
            EntityRow,
            Assignment,
            CompletedComparison,
            JudgeState,
            EntityState,
            LastSkip,
            ProjectTable,
        ]
    )
    _migrate_strikes()
    _migrate_assignment_time()
    _migrate_legacy_state()


def close_db() -> None:
    if not db.is_closed():
        db.close()


def archive_db() -> Path | None:
    # Move the file aside. create_tables will not alter a table that already
    # exists, so a schema change has to start from an empty file.
    close_db()
    try:
        return _archive_closed_db()
    finally:
        # A failed archive must leave logging and SQLite usable.
        attach_log_file()
        if db.is_closed():
            try:
                open_db()
            except Exception:
                logger.exception(
                    "Could not reopen the SQLite database after archive failed"
                )


def _archive_closed_db() -> Path | None:
    if settings.DB_FILE == ":memory:":
        return None
    db_path = Path(settings.DB_FILE)
    candidates = [
        db_path.with_name(db_path.name + suffix)
        for suffix in ("", "-wal", "-shm", "-journal")
    ]
    candidates.append(log_path())
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    destination = _create_archive_directory(archive_root())
    logger.info("Archiving database to %s", destination)
    detach_log_file()
    try:
        _move_together(existing, destination)
    except Exception:
        if not any(destination.iterdir()):
            destination.rmdir()
        raise
    open_db()
    attach_log_file()
    logger.info("Archived database to %s", destination)
    return destination


def _move_together(sources: list[Path], destination: Path) -> None:
    # A rename is atomic for one file. If a later file fails, put the earlier
    # ones back so the database and its WAL are never split.
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            target = destination / source.name
            source.rename(target)
            moved.append((target, source))
    except Exception:
        for target, source in reversed(moved):
            try:
                target.rename(source)
            except Exception:
                logger.exception("Could not move %s back to %s", target, source)
        raise


def archive_root() -> Path:
    return Path(settings.DB_FILE).parent / "archive"


def list_archives() -> list[str]:
    root = archive_root()
    if not root.is_dir():
        return []
    names = [path.name for path in root.iterdir() if path.is_dir()]
    names.sort(reverse=True)
    return names


def list_archive_files(name: str) -> list[Path]:
    root = archive_root()
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_dir():
        raise UnknownArchiveException()
    return sorted(file for file in path.iterdir() if file.is_file())


def _create_archive_directory(root: Path) -> Path:
    # mkdir is the existence check. A name chosen earlier can appear before
    # create, and that must not leave SQLite closed or the log detached.
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = 1
    while True:
        name = stamp if suffix == 1 else f"{stamp}-{suffix}"
        try:
            (root / name).mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return root / name


def load_state() -> JudgeRecord:
    # One read transaction so the tables are a single snapshot.
    with db.atomic():
        entities = [
            Entity(attributes=row.attributes)
            for row in EntityRow.select().order_by(EntityRow.id)
        ]
        assignment_rows = list(Assignment.select())
        assignments = {
            row.judge_id: (row.entity_id_1, row.entity_id_2)
            for row in assignment_rows
        }
        loaded_at = time.time()
        assigned_at = {
            row.judge_id: float(row.assigned_at)
            if row.assigned_at is not None
            else loaded_at
            for row in assignment_rows
        }
        completed = {
            row.judge_id: (row.entity_id_1, row.entity_id_2, row.winner_id)
            for row in CompletedComparison.select()
        }
        frequencies, strikes = _load_entity_state(len(entities))
        last_skips = _load_last_skips()
        tables = _load_tables()
        judge = Judge.get_or_none(Judge.id == 1)
        if judge is None:
            return JudgeRecord(
                enabled=False,
                headers=[],
                entities=entities,
                assignments=assignments,
            assigned_at=assigned_at,
                completed=completed,
                strikes=strikes,
                last_skips=last_skips,
                bdp=None,
                tables=tables,
            )
        model = _load_model(len(entities), frequencies)
        return JudgeRecord(
            enabled=bool(judge.enabled) and model is not None,
            headers=list(judge.headers),
            entities=entities,
            assignments=assignments,
            assigned_at=assigned_at,
            completed=completed,
            strikes=strikes,
            last_skips=last_skips,
            bdp=model,
            tables=tables,
        )


def replace_state(
    headers: list[str], entities: list[Entity], bdp: BayesianDecisionProcess
) -> JudgeRecord:
    record = JudgeRecord(
        enabled=True,
        headers=list(headers),
        entities=list(entities),
        assignments={},
        assigned_at={},
        completed={},
        strikes=[0] * len(entities),
        last_skips={},
        bdp=bdp,
        tables={},
    )
    with db.atomic():
        record.tables = _load_tables()
        EntityRow.delete().execute()
        EntityState.delete().execute()
        Assignment.delete().execute()
        CompletedComparison.delete().execute()
        LastSkip.delete().execute()
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
                    {
                        "entity_id": index,
                        "frequency": int(frequency[index]),
                        "strikes": 0,
                    }
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
    frequency: Any,
    key: Any,
    judge_id: str,
    pair: tuple[int, int],
    assigned_at: float,
) -> None:
    left, right = pair
    stored_frequency = np.asarray(frequency)
    frequencies = (int(stored_frequency[left]), int(stored_frequency[right]))
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
            JudgeState.update(key=_array_bytes(key, np.dtype("<u4")))
            .where(JudgeState.id == 1)
            .execute()
        )
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        Assignment.replace(
            judge_id=judge_id,
            entity_id_1=left,
            entity_id_2=right,
            assigned_at=assigned_at,
        ).execute()


def save_comparison(
    alpha: Any,
    judge_id: str,
    completed: tuple[int, int, int],
    strikes: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    entity_id_1, entity_id_2, winner_id = completed
    with db.atomic():
        updated = (
            JudgeState.update(alpha=_array_bytes(alpha, np.dtype("<f4")))
            .where(JudgeState.id == 1)
            .execute()
        )
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        for entity_id, strike_count in strikes:
            updated = (
                EntityState.update(strikes=strike_count)
                .where(EntityState.entity_id == entity_id)
                .execute()
            )
            if updated != 1:
                raise RuntimeError("Entity state is missing")
        Assignment.delete().where(Assignment.judge_id == judge_id).execute()
        CompletedComparison.replace(
            judge_id=judge_id,
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            winner_id=winner_id,
        ).execute()


def save_strikes(entity_id: int, strikes: int) -> None:
    updated = (
        EntityState.update(strikes=strikes)
        .where(EntityState.entity_id == entity_id)
        .execute()
    )
    if updated != 1:
        raise RuntimeError("Entity state is missing")


def save_absence(
    frequency: Any,
    strikes: list[int],
    key: Any,
    alpha: Any | None,
    assignments: dict[str, tuple[int, int] | None],
    completed: tuple[str, tuple[int, int, int]] | None,
    judge_id: str,
    absent_ids: tuple[int, ...],
    pair: tuple[int, int] | None,
    assigned_at: float,
) -> None:
    stored_frequency = np.asarray(frequency)
    if stored_frequency.shape != (len(strikes),):
        raise RuntimeError("Frequency and strikes do not match")
    with db.atomic():
        for index, strike_count in enumerate(strikes):
            updated = (
                EntityState.update(
                    frequency=int(stored_frequency[index]),
                    strikes=int(strike_count),
                )
                .where(EntityState.entity_id == index)
                .execute()
            )
            if updated != 1:
                raise RuntimeError("Entity state is missing")
        state: dict[str, bytes] = {"key": _array_bytes(key, np.dtype("<u4"))}
        if alpha is not None:
            state["alpha"] = _array_bytes(alpha, np.dtype("<f4"))
        updated = JudgeState.update(state).where(JudgeState.id == 1).execute()
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        for assigned_judge, assigned_pair in assignments.items():
            if assigned_pair is None:
                Assignment.delete().where(
                    Assignment.judge_id == assigned_judge
                ).execute()
                continue
            left, right = assigned_pair
            Assignment.replace(
                judge_id=assigned_judge,
                entity_id_1=left,
                entity_id_2=right,
                assigned_at=assigned_at,
            ).execute()
        if completed is not None:
            completed_judge, (entity_id_1, entity_id_2, winner_id) = completed
            CompletedComparison.replace(
                judge_id=completed_judge,
                entity_id_1=entity_id_1,
                entity_id_2=entity_id_2,
                winner_id=winner_id,
            ).execute()
        LastSkip.replace(
            judge_id=judge_id,
            absent_ids=list(absent_ids),
            entity_id_1=None if pair is None else pair[0],
            entity_id_2=None if pair is None else pair[1],
        ).execute()


def save_tables(tables: dict[str, int]) -> None:
    with db.atomic():
        ProjectTable.delete().execute()
        if tables:
            ProjectTable.insert_many(
                [
                    {"project_url": url, "table_number": number}
                    for url, number in tables.items()
                ]
            ).execute()


def save_skip(
    frequency: Any,
    key: Any,
    judge_id: str,
    pair: tuple[int, int] | None,
    assigned_at: float,
) -> None:
    """Store a skipped pair's refund and the judge's next pair, if any."""
    stored_frequency = np.asarray(frequency)
    with db.atomic():
        for index, count in enumerate(stored_frequency):
            updated = (
                EntityState.update(frequency=int(count))
                .where(EntityState.entity_id == index)
                .execute()
            )
            if updated != 1:
                raise RuntimeError("Entity state is missing")
        updated = (
            JudgeState.update(key=_array_bytes(key, np.dtype("<u4")))
            .where(JudgeState.id == 1)
            .execute()
        )
        if updated != 1:
            raise RuntimeError("Judge state is missing")
        if pair is None:
            Assignment.delete().where(Assignment.judge_id == judge_id).execute()
            return
        Assignment.replace(
            judge_id=judge_id,
            entity_id_1=pair[0],
            entity_id_2=pair[1],
            assigned_at=assigned_at,
        ).execute()


def save_enabled(enabled: bool) -> None:
    updated = Judge.update(enabled=enabled).where(Judge.id == 1).execute()
    if updated != 1:
        raise RuntimeError("Judge row is missing")


def _load_model(
    entity_count: int, frequencies: list[int]
) -> BayesianDecisionProcess | None:
    state = JudgeState.get_or_none(JudgeState.id == 1)
    if state is None:
        return None
    alpha = _array_from_bytes(state.alpha, np.dtype("<f4"), entity_count, "alpha")
    key = _array_from_bytes(state.key, np.dtype("<u4"), 2, "key")
    return BayesianDecisionProcess.model_validate(
        {
            "K": entity_count,
            "alpha_t": alpha,
            "frequency": frequencies,
            "key": key,
        }
    )


def _load_entity_state(entity_count: int) -> tuple[list[int], list[int]]:
    rows = cast(
        list[tuple[int, int, int]],
        list(
            EntityState.select(
                EntityState.entity_id,
                EntityState.frequency,
                EntityState.strikes,
            )
            .order_by(EntityState.entity_id)
            .tuples()
        ),
    )
    if len(rows) != entity_count or any(
        entity_id != index
        for index, (entity_id, _frequency, _strikes) in enumerate(rows)
    ):
        raise RuntimeError("Entity state does not match the entity table")
    return (
        [frequency for _entity_id, frequency, _strikes in rows],
        [strike_count for _entity_id, _frequency, strike_count in rows],
    )


def _load_tables() -> dict[str, int]:
    return {
        str(row.project_url): int(row.table_number) for row in ProjectTable.select()
    }


def _load_last_skips() -> dict[str, AbsentSkip]:
    loaded: dict[str, AbsentSkip] = {}
    for row in LastSkip.select():
        absent = tuple(sorted(int(entity_id) for entity_id in row.absent_ids))
        pair = None
        if row.entity_id_1 is not None and row.entity_id_2 is not None:
            pair = (int(row.entity_id_1), int(row.entity_id_2))
        loaded[str(row.judge_id)] = (absent, pair)
    return loaded


def _migrate_strikes() -> None:
    # create_tables does not add columns to a table that already exists.
    if "entity_state" not in db.get_tables():
        return
    names = {column.name for column in db.get_columns("entity_state")}
    if "strikes" in names:
        return
    db.execute_sql(
        "ALTER TABLE entity_state ADD COLUMN strikes INTEGER NOT NULL DEFAULT 0"
    )


def _migrate_assignment_time() -> None:
    # Pairs handed out before this column existed load as handed out at startup.
    if "assignments" not in db.get_tables():
        return
    names = {column.name for column in db.get_columns("assignments")}
    if "assigned_at" in names:
        return
    db.execute_sql("ALTER TABLE assignments ADD COLUMN assigned_at REAL")


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
                    {
                        "entity_id": index,
                        "frequency": int(frequency),
                        "strikes": 0,
                    }
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
