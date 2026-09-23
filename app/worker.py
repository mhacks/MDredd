import asyncio
import logging
import queue
import threading
from collections.abc import Callable
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
Reply = tuple[EntityWithId, EntityWithId] | None


@dataclass
class Command:
    name: str
    reply: queue.Queue[Reply | Exception]
    payload: CommandPayload = None


class JudgeWorker:
    """Owns the Bayesian judge on one thread.

    Request handlers send commands over ``channel``. Submit is acknowledged
    after the write-ahead row is stored; ``submit_comparison`` and
    ``get_next_pair`` run only on this thread. Rankings are published for
    readers that must not call JAX, and pushed to rankings subscribers.
    """

    def __init__(
        self,
        entities: EntityAdapter,
        snapshots: SnapshotAdapter,
        assignments: AssignmentAdapter,
        wal: WriteAheadAdapter,
    ):
        self.entities: EntityAdapter = entities
        self.snapshots: SnapshotAdapter = snapshots
        self.assignments: AssignmentAdapter = assignments
        self.wal: WriteAheadAdapter = wal
        self.channel: queue.Queue[Command] = queue.Queue()
        self.bdp: BayesianDecisionProcess | None = None
        self._updates: int = 0
        self._entities: list[Entity] | None = None
        self._rankings: list[EntityWithId] | None = None
        self._lock: threading.Lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue[list[EntityWithId]]] = []
        self._subscriber_lock: threading.Lock = threading.Lock()
        self._ready: threading.Event = threading.Event()
        self._bootstrap_error: Exception | None = None
        self._thread: threading.Thread = threading.Thread(
            target=self._run, name="judge-worker", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        _ = self._ready.wait()
        if self._bootstrap_error is not None:
            raise self._bootstrap_error

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        _ = self._call("stop")
        self._thread.join(timeout=5)

    def has_bdp(self) -> bool:
        with self._lock:
            return self._rankings is not None

    def rankings(self) -> list[EntityWithId]:
        with self._lock:
            if self._rankings is None:
                raise AttributeError("bdp")
            return list(self._rankings)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe_rankings(self) -> asyncio.Queue[list[EntityWithId]]:
        subscriber: asyncio.Queue[list[EntityWithId]] = asyncio.Queue()
        with self._subscriber_lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe_rankings(self, subscriber: asyncio.Queue[list[EntityWithId]]) -> None:
        with self._subscriber_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def ranking_subscriber_count(self) -> int:
        with self._subscriber_lock:
            return len(self._subscribers)

    def reset(self, entity_count: int) -> None:
        _ = self._call("reset", entity_count)

    def request_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        result = self._call("get_pair", pair_request)
        if isinstance(result, tuple):
            return result
        raise RuntimeError("Pair request did not return a pair")

    def submit(self, comparison: ComparisonInputModel) -> None:
        _ = self._call("submit", comparison)

    def flush(self) -> None:
        _ = self._call("flush")

    def _call(self, name: str, payload: CommandPayload = None) -> Reply:
        reply: queue.Queue[Reply | Exception] = queue.Queue(maxsize=1)
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
                _ = db.close()

    def _bootstrap(self) -> None:
        snapshot = self.snapshots.load()
        if snapshot is None:
            return

        timestamp, bdp_instance = snapshot
        replayed = self.wal.replay(timestamp, bdp_instance)
        self.bdp = bdp_instance
        self._entities = self.entities.to_list()
        self._publish()
        logger.info(
            "Replayed write-ahead log",
            extra={"event": "replay", "replayed": replayed},
        )

    def _handle(self, command: Command) -> bool:
        if command.name == "stop":
            command.reply.put(None)
            return True
        if command.name == "flush":
            command.reply.put(None)
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
                "Rejected comparison",
                extra={
                    "event": "comparison_rejected",
                    "user_id": judge,
                    "entity_ids": [entity_id_1, entity_id_2],
                },
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
                "Applied comparison",
                extra={
                    "event": "comparison",
                    "user_id": judge,
                    "entity_ids": [entity_id_1, entity_id_2],
                    "winner_id": winner_id,
                },
            )

        self._after_ack(command, apply)

    def _after_ack(self, command: Command, apply: Callable[[], None]) -> None:
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
        return (self._with_id(entities[i], i), self._with_id(entities[j], j))

    def _with_id(self, entity: Entity, index: int) -> EntityWithId:
        return EntityWithId(attributes=dict(entity.attributes), id=index)

    def _publish(self) -> None:
        entities = self._require_entities()
        alphas = np.asarray(self._require_bdp().get_alphas(), dtype=np.float64)
        ranked_ids = sorted(
            range(len(entities)),
            key=lambda index: alphas[index],
            reverse=True,
        )
        rankings = [self._with_id(entities[index], index) for index in ranked_ids]
        with self._lock:
            self._rankings = rankings
        self._broadcast(rankings)

    def _broadcast(self, rankings: list[EntityWithId]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        payload = list(rankings)
        with self._subscriber_lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                loop.call_soon_threadsafe(subscriber.put_nowait, payload)
            except RuntimeError:
                return

    def _note_update(self) -> None:
        self._updates += 1
        if self._updates >= settings.SNAPSHOT_INTERVAL:
            self._updates = 0
            self._snapshot()

    def _snapshot(self) -> None:
        logger.info("Wrote snapshot", extra={"event": "snapshot"})
        self.snapshots.record(self._require_bdp())
