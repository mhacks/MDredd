import logging
from typing import Tuple
from fastapi import UploadFile
import numpy as np

from app.exceptions import (
    IncorrectPairFormatException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingNeverStartedException,
    JudgingNotStartedException,
)
from app.adapters import (
    SnapshotAdapter,
    EntityAdapter,
    WriteAheadAdapter,
    AssignmentAdapter,
)
from app.models import (
    EntityWithId,
)
from app.algorithm import BayesianDecisionProcess

logger = logging.getLogger(__name__)


class Session:
    def __init__(self):
        self.enabled = False
        self.entities = EntityAdapter()
        self.snapshots = SnapshotAdapter()
        self.assignments = AssignmentAdapter()
        self.wal = WriteAheadAdapter()

        snapshot = self.snapshots.load()
        if snapshot is not None:
            timestamp, bdp_instance = snapshot
            self.bdp = bdp_instance
            self.wal.replay(timestamp, self.bdp)
            self.enabled = True

    def get_enabled(self) -> bool:
        return self.enabled

    def start(self, entity_csv: UploadFile | None = None):
        if self.enabled:
            raise JudgingAlreadyStartedException()

        if entity_csv is not None:
            self.entities.clear()
            self.snapshots.clear()
            self.assignments.clear()
            self.wal.clear()

            self.entities.load(entity_csv)
            self.bdp = BayesianDecisionProcess(K=len(self.entities))

        self.enabled = True

    def resume(self):
        if self.enabled:
            raise JudgingAlreadyStartedException()

        if not hasattr(self, "BDP"):
            raise JudgingNeverStartedException()

        self.enabled = True

    def stop(self):
        if not self.enabled:
            raise JudgingNotStartedException()
        self.enabled = False

    def get_pair(self, judge, force: bool = False) -> Tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()

        if not force and judge in self.assignments:
            i, j = self.assignments[judge]
        else:
            i, j = self.bdp.get_next_pair()
            self.assignments[judge] = (i, j)

        response_entity_i = EntityWithId(**self.entities[i].dict(), id=i)
        response_entity_j = EntityWithId(**self.entities[j].dict(), id=j)

        return (response_entity_i, response_entity_j)

    def submit_pair(
        self, judge: str, entity_id_1: int, entity_id_2: int, winner_id: int
    ):
        if not self.enabled:
            raise JudgingNotStartedException()

        if not self.assignments.verify(judge, entity_id_1, entity_id_2):
            logger.info(self.assignments[judge])
            logger.info((entity_id_1, entity_id_2))
            raise JudgeDoesNotOwnPairException()

        if winner_id not in (entity_id_1, entity_id_2):
            raise IncorrectPairFormatException()

        self.bdp.submit_comparison(entity_id_1, entity_id_2, winner_id)
        del self.assignments[judge]

    def get_rankings(self):
        if self.enabled:
            sorted_indices = np.flip(np.argsort(self.bdp.get_alphas()))
            entities = self.entities.to_list()
            return [entities[i] for i in sorted_indices]
        else:
            raise JudgingNotStartedException()
