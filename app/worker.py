from dataclasses import dataclass, field
import logging
import queue
import threading

import numpy as np

from app.adapters import AssignmentAdapter, EntityAdapter, SnapshotAdapter, WriteAheadAdapter
from app.algorithm import BayesianDecisionProcess
from app.db import db
from app.exceptions import IncorrectPairFormatException, JudgeDoesNotOwnPairException
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Command:
    name: str
    payload: object = None
    reply: queue.Queue = field(default_factory=queue.Queue)
    answered: bool = False


class JudgeWorker:
    """Owns the Bayesian judge on one thread.

    Request handlers send commands over ``channel``. Submit is acknowledged
    after the write-ahead row is stored; ``submit_comparison`` and
    ``get_next_pair`` run only on this thread. Rankings are published for
    readers that must not call JAX.
    """

    def __init__(
        self,
        entities: EntityAdapter,
        snapshots: SnapshotAdapter,
        assignments: AssignmentAdapter,
        wal: WriteAheadAdapter,
    ):
        self.entities = entities
        self.snapshots = snapshots
        self.assignments = assignments
        self.wal = wal
        self.channel: queue.Queue[Command] = queue.Queue()
        self.bdp: BayesianDecisionProcess | None = None
        self.recovered = False
        self._updates = 0
        self._rankings: list = []
        self._has_bdp = False
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._bootstrap_error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name="judge-worker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()
        if self._bootstrap_error is not None:
            raise self._bootstrap_error

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        self._call("stop")
        self._thread.join(timeout=5)

    def has_bdp(self) -> bool:
        with self._lock:
            return self._has_bdp

    def rankings(self) -> list:
        with self._lock:
            if not self._has_bdp:
                raise AttributeError("bdp")
            return list(self._rankings)

    def reset(self, entity_count: int) -> None:
        self._call("reset", entity_count)

    def request_pair(self, judge: str, force: bool):
        return self._call("get_pair", (judge, force))

    def submit(self, comparison: ComparisonInputModel) -> None:
        self._call("submit", comparison)

    def flush(self) -> None:
        self._call("flush")

    def snapshot(self) -> None:
        self._call("snapshot")

    def alphas(self) -> np.ndarray:
        return self._call("alphas")

    def _call(self, name: str, payload: object = None):
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.channel.put(Command(name=name, payload=payload, reply=reply))
        result = reply.get()
        if isinstance(result, Exception):
            raise result
        return result

    def _run(self) -> None:
        try:
            try:
                self._bootstrap()
            except Exception as exc:
                self._bootstrap_error = exc
                logger.exception("Judge worker failed to recover")
            finally:
                self._ready.set()

            if self._bootstrap_error is not None:
                return

            while True:
                command = self.channel.get()
                try:
                    if self._handle(command):
                        return
                except Exception as exc:
                    logger.exception("Judge worker command %s failed", command.name)
                    self._reply(command, exc)
        finally:
            if not db.is_closed():
                db.close()

    def _bootstrap(self) -> None:
        snapshot = self.snapshots.load()
        if snapshot is None:
            return

        timestamp, bdp_instance = snapshot
        self.wal.replay(timestamp, bdp_instance)
        self.bdp = bdp_instance
        self._publish()
        self.recovered = True
        logger.info("Replayed write-ahead log from snapshot %s", timestamp)

    def _handle(self, command: Command) -> bool:
        if command.name == "stop":
            self._reply(command, None)
            return True
        if command.name == "flush":
            self._reply(command, None)
            return False
        if command.name == "snapshot":
            self._snapshot()
            self._reply(command, None)
            return False
        if command.name == "alphas":
            self._reply(command, self.bdp.get_alphas().copy())
            return False
        if command.name == "reset":
            self._reset(command.payload)
            self._reply(command, None)
            return False
        if command.name == "get_pair":
            self._get_pair(command)
            return False
        if command.name == "submit":
            self._submit(command)
            return False
        raise RuntimeError(f"Unknown judge command: {command.name}")

    def _reset(self, entity_count: int) -> None:
        self.bdp = BayesianDecisionProcess(K=entity_count)
        self._updates = 0
        self._publish()

    def _get_pair(self, command: Command):
        judge, force = command.payload
        self.wal.log(PairRequestModel(uuid=judge, force=force))
        if not force and judge in self.assignments:
            i, j = self.assignments[judge]
        else:
            i, j = self.bdp.get_next_pair()
            self.assignments[judge] = (i, j)
            self._note_update()

        pair = (
            EntityWithId(**self.entities[i].model_dump(), id=i),
            EntityWithId(**self.entities[j].model_dump(), id=j),
        )
        self._reply(command, pair)

    def _submit(self, command: Command) -> None:
        comparison: ComparisonInputModel = command.payload
        judge = comparison.uuid
        entity_id_1, entity_id_2 = comparison.entity_ids
        winner_id = comparison.winner_id

        if not self.assignments.verify(judge, entity_id_1, entity_id_2):
            logger.info(
                "Rejected comparison from %s for %s",
                judge,
                (entity_id_1, entity_id_2),
            )
            raise JudgeDoesNotOwnPairException()

        if winner_id not in (entity_id_1, entity_id_2):
            raise IncorrectPairFormatException()

        self.wal.log(comparison)
        self._reply(command, None)

        self.bdp.submit_comparison(entity_id_1, entity_id_2, winner_id)
        del self.assignments[judge]
        self._publish()
        self._note_update()
        logger.info(
            "Applied comparison from %s: %s beat %s",
            judge,
            winner_id,
            entity_id_2 if winner_id == entity_id_1 else entity_id_1,
        )

    def _publish(self) -> None:
        sorted_indices = np.flip(np.argsort(self.bdp.get_alphas()))
        entities = self.entities.to_list()
        rankings = [entities[int(i)] for i in sorted_indices]
        with self._lock:
            self._has_bdp = True
            self._rankings = rankings

    def _note_update(self) -> None:
        self._updates += 1
        if self._updates >= settings.SNAPSHOT_INTERVAL:
            self._updates = 0
            self._snapshot()

    def _snapshot(self) -> None:
        logger.info("Taking snapshot")
        self.snapshots.record(self.bdp)

    def _reply(self, command: Command, value) -> None:
        if command.answered:
            return
        command.answered = True
        command.reply.put(value)
