import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
from peewee import DoesNotExist

from app.adapters import (
    AssignmentAdapter,
    EntityAdapter,
    SnapshotAdapter,
    WriteAheadAdapter,
)
from app.algorithm import BayesianDecisionProcess
from app.db import db
from app.entity import Entity
from app.exceptions import IncorrectPairFormatException, JudgeDoesNotOwnPairException
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.settings import settings

logger = logging.getLogger(__name__)

CommandPayload = int | PairRequestModel | ComparisonInputModel | None


@dataclass
class Command:
    name: str
    reply: queue.Queue
    payload: CommandPayload = None


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
        self._updates = 0
        self._entities: list[Entity] | None = None
        self._rankings: list[Entity] | None = None
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
            return self._rankings is not None

    def rankings(self) -> list[Entity]:
        with self._lock:
            if self._rankings is None:
                raise AttributeError("bdp")
            return list(self._rankings)

    def reset(self, entity_count: int) -> None:
        self._call("reset", entity_count)

    def request_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        return self._call("get_pair", pair_request)

    def submit(self, comparison: ComparisonInputModel) -> None:
        self._call("submit", comparison)

    def flush(self) -> None:
        self._call("flush")

    def snapshot(self) -> None:
        self._call("snapshot")

    def alphas(self) -> np.ndarray:
        return self._call("alphas")

    def _call(self, name: str, payload: CommandPayload = None):
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
                    command.reply.put(exc)
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
        self._entities = self.entities.to_list()
        self._publish()
        logger.info("Replayed write-ahead log from snapshot %s", timestamp)

    def _handle(self, command: Command) -> bool:
        if command.name == "stop":
            command.reply.put(None)
            return True
        if command.name == "flush":
            command.reply.put(None)
            return False
        if command.name == "snapshot":
            self._snapshot()
            command.reply.put(None)
            return False
        if command.name == "alphas":
            command.reply.put(self._require_bdp().get_alphas())
            return False
        if command.name == "reset":
            self._reset(command.payload)
            command.reply.put(None)
            return False
        if command.name == "get_pair":
            self._get_pair(command)
            return False
        if command.name == "submit":
            self._submit(command)
            return False
        raise RuntimeError(f"Unknown judge command: {command.name}")

    def _reset(self, entity_count: CommandPayload) -> None:
        if not isinstance(entity_count, int):
            raise TypeError("reset command is missing an entity count")
        self.bdp = BayesianDecisionProcess.create(entity_count)
        self._updates = 0
        self._entities = self.entities.to_list()
        self._publish()

    def _get_pair(self, command: Command):
        pair_request = command.payload
        if not isinstance(pair_request, PairRequestModel):
            raise TypeError("get_pair command is missing a pair request")
        self.wal.log(pair_request)
        assigned = None if pair_request.force else self._current_assignment(pair_request.uuid)
        if assigned is None:
            i, j = self._require_bdp().get_next_pair()
            self.assignments[pair_request.uuid] = (i, j)
        else:
            i, j = assigned

        command.reply.put(self._pair(i, j))
        if assigned is None:
            self._after_ack(command, self._note_update)

    def _submit(self, command: Command) -> None:
        comparison = command.payload
        if not isinstance(comparison, ComparisonInputModel):
            raise TypeError("submit command is missing a comparison")
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
        command.reply.put(None)

        def apply() -> None:
            self._require_bdp().submit_comparison(entity_id_1, entity_id_2, winner_id)
            del self.assignments[judge]
            self._publish()
            self._note_update()
            logger.info(
                "Applied comparison from %s: %s beat %s",
                judge,
                winner_id,
                entity_id_2 if winner_id == entity_id_1 else entity_id_1,
            )

        self._after_ack(command, apply)

    def _after_ack(self, command: Command, apply) -> None:
        try:
            apply()
        except Exception:
            logger.exception("Judge worker command %s failed after ack", command.name)

    def _current_assignment(self, judge: str) -> tuple[int, int] | None:
        try:
            return self.assignments[judge]
        except DoesNotExist:
            return None

    def _require_bdp(self) -> BayesianDecisionProcess:
        if self.bdp is None:
            raise RuntimeError("Judge model is not initialized")
        return self.bdp

    def _require_entities(self) -> list[Entity]:
        if self._entities is None:
            self._entities = self.entities.to_list()
        return self._entities

    def _pair(self, i: int, j: int) -> tuple[EntityWithId, EntityWithId]:
        entities = self._require_entities()
        return (
            EntityWithId(**entities[i].model_dump(), id=i),
            EntityWithId(**entities[j].model_dump(), id=j),
        )

    def _publish(self) -> None:
        entities = self._require_entities()
        order = np.flip(np.argsort(self._require_bdp().get_alphas()))
        rankings = [entities[int(i)] for i in order]
        with self._lock:
            self._rankings = rankings

    def _note_update(self) -> None:
        self._updates += 1
        if self._updates >= settings.SNAPSHOT_INTERVAL:
            self._updates = 0
            self._snapshot()

    def _snapshot(self) -> None:
        logger.info("Taking snapshot")
        self.snapshots.record(self._require_bdp())
