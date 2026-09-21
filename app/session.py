from typing import Tuple

from fastapi import UploadFile

from app.adapters import (
    AssignmentAdapter,
    EntityAdapter,
    SnapshotAdapter,
    WriteAheadAdapter,
)
from app.exceptions import (
    JudgingAlreadyStartedException,
    JudgingNeverStartedException,
    JudgingNotStartedException,
)
from app.models import ComparisonInputModel, EntityWithId
from app.worker import JudgeWorker


class Session:
    def __init__(self):
        self.enabled = False
        self.entities = EntityAdapter()
        self.snapshots = SnapshotAdapter()
        self.assignments = AssignmentAdapter()
        self.wal = WriteAheadAdapter()
        self.worker = JudgeWorker(
            entities=self.entities,
            snapshots=self.snapshots,
            assignments=self.assignments,
            wal=self.wal,
        )
        self.worker.start()
        if self.worker.recovered:
            self.enabled = True

    def close(self) -> None:
        self.worker.shutdown()

    def get_enabled(self) -> bool:
        return self.enabled

    def start(self, entity_csv: UploadFile | None = None):
        if self.enabled:
            raise JudgingAlreadyStartedException()

        if entity_csv is not None:
            self.worker.flush()
            self.entities.clear()
            self.snapshots.clear()
            self.assignments.clear()
            self.wal.clear()

            self.entities.load(entity_csv)
            self.worker.reset(len(self.entities))

        self.enabled = True

    def resume(self):
        if self.enabled:
            raise JudgingAlreadyStartedException()

        if not self.worker.has_bdp():
            raise JudgingNeverStartedException()

        self.enabled = True

    def stop(self):
        if not self.enabled:
            raise JudgingNotStartedException()
        self.enabled = False

    def get_pair(self, judge, force: bool = False) -> Tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()

        return self.worker.request_pair(judge, force)

    def submit_pair(
        self, judge: str, entity_id_1: int, entity_id_2: int, winner_id: int
    ):
        if not self.enabled:
            raise JudgingNotStartedException()

        comparison = ComparisonInputModel(
            uuid=judge,
            entity_ids=(entity_id_1, entity_id_2),
            winner_id=winner_id,
        )
        self.worker.submit(comparison)

    def get_rankings(self):
        if not self.enabled:
            raise JudgingNotStartedException()
        return self.worker.rankings()
