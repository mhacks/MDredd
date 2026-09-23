import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.algorithm import BayesianDecisionProcess
from app.entity import Entity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL,
    headers TEXT NOT NULL,
    bdp TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    attributes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
    judge_id TEXT PRIMARY KEY,
    entity_id_1 INTEGER NOT NULL,
    entity_id_2 INTEGER NOT NULL
);
"""


@dataclass
class JudgeRecord:
    enabled: bool
    headers: list[str]
    entities: list[Entity]
    assignments: dict[str, tuple[int, int]]
    bdp: BayesianDecisionProcess | None


class JudgeDB:
    """SQLite file owned by the judge thread.

    Entities are written when a CSV is uploaded. A pair draw or comparison
    commits the model and that one assignment, and leaves the entities alone.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        _ = self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def load(self) -> JudgeRecord:
        row = self._conn.execute("SELECT enabled, headers, bdp FROM judge WHERE id = 1").fetchone()
        if row is None:
            return JudgeRecord(enabled=False, headers=[], entities=[], assignments={}, bdp=None)
        entities = [
            Entity(attributes=_string_dict(attributes))
            for (attributes,) in self._conn.execute(
                "SELECT attributes FROM entities ORDER BY id"
            )
        ]
        assignments = {
            str(judge_id): (int(left), int(right))
            for judge_id, left, right in self._conn.execute(
                "SELECT judge_id, entity_id_1, entity_id_2 FROM assignments"
            )
        }
        return JudgeRecord(
            enabled=bool(row[0]),
            headers=_string_list(str(row[1])),
            entities=entities,
            assignments=assignments,
            bdp=_bdp(None if row[2] is None else str(row[2])),
        )

    def replace(self, record: JudgeRecord) -> None:
        rows = [
            (index, json.dumps(entity.attributes)) for index, entity in enumerate(record.entities)
        ]
        with self._conn:
            self._conn.execute("DELETE FROM entities")
            self._conn.execute("DELETE FROM assignments")
            self._conn.executemany("INSERT INTO entities (id, attributes) VALUES (?, ?)", rows)
            self._write_judge(record.enabled, record.headers, record.bdp)

    def assign(self, bdp: BayesianDecisionProcess, judge_id: str, pair: tuple[int, int]) -> None:
        with self._conn:
            self._write_bdp(bdp)
            self._conn.execute(
                """
                INSERT INTO assignments (judge_id, entity_id_1, entity_id_2)
                VALUES (?, ?, ?)
                ON CONFLICT(judge_id) DO UPDATE SET
                    entity_id_1 = excluded.entity_id_1,
                    entity_id_2 = excluded.entity_id_2
                """,
                (judge_id, pair[0], pair[1]),
            )

    def finish_comparison(self, bdp: BayesianDecisionProcess, judge_id: str) -> None:
        with self._conn:
            self._write_bdp(bdp)
            self._conn.execute("DELETE FROM assignments WHERE judge_id = ?", (judge_id,))

    def set_enabled(self, enabled: bool) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE judge SET enabled = ? WHERE id = 1",
                (int(enabled),),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Judge row is missing")

    def _write_judge(
        self,
        enabled: bool,
        headers: list[str],
        bdp: BayesianDecisionProcess | None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO judge (id, enabled, headers, bdp)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled = excluded.enabled,
                headers = excluded.headers,
                bdp = excluded.bdp
            """,
            (int(enabled), json.dumps(headers), _bdp_json(bdp)),
        )

    def _write_bdp(self, bdp: BayesianDecisionProcess) -> None:
        self._conn.execute("UPDATE judge SET bdp = ? WHERE id = 1", (_bdp_json(bdp),))


def _bdp_json(bdp: BayesianDecisionProcess | None) -> str | None:
    if bdp is None:
        return None
    return json.dumps(
        {
            "K": bdp.K,
            "alpha_t": bdp.alpha_t.tolist(),
            "frequency": bdp.frequency.tolist(),
            "key": bdp.key.tolist(),
        }
    )


def _bdp(raw: str | None) -> BayesianDecisionProcess | None:
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Judge model is not an object")
    return BayesianDecisionProcess(**cast(dict[str, Any], value))


def _string_list(raw: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("Expected a list of strings")
    return value


def _string_dict(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise TypeError("Expected a JSON object")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    attributes: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("Expected string attributes")
        attributes[key] = item
    return attributes
