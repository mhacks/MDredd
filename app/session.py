
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
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.worker import JudgeWorker


class Session:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.entities: EntityAdapter = EntityAdapter()
        self.snapshots: SnapshotAdapter = SnapshotAdapter()
        self.assignments: AssignmentAdapter = AssignmentAdapter()
        self.wal: WriteAheadAdapter = WriteAheadAdapter()
        self.worker: JudgeWorker = JudgeWorker(
            entities=self.entities,
            snapshots=self.snapshots,
            assignments=self.assignments,
            wal=self.wal,
        )
        self.worker.start()
        if self.worker.has_bdp():
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

    def get_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()

        return self.worker.request_pair(pair_request)

    def submit_pair(self, comparison: ComparisonInputModel):
        if not self.enabled:
            raise JudgingNotStartedException()

        self.worker.submit(comparison)

    def get_rankings(self):
        if not self.enabled:
            raise JudgingNotStartedException()
        return self.worker.rankings()
