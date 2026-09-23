import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

from app.algorithm import BayesianDecisionProcess
from app.entity import Entity
from app.exceptions import (
    IncorrectPairFormatException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingFailure,
    JudgingNeverStartedException,
    JudgingNotStartedException,
    UnknownRowException,
)
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.settings import settings
from app.store import JudgeDB, JudgeRecord

logger = logging.getLogger(__name__)

Job = Callable[[], object] | None
Reply = queue.Queue[tuple[bool, object]]


class JudgeWorker:
    """Owns the Bayesian judge on one thread.

    Pair draws and comparisons update the in-memory model, then commit that
    model and the one assignment in SQLite. Entities stay as they were written
    at upload. The caller is answered only after the commit succeeds.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path if path is not None else settings.DB_FILE)
        self._db: JudgeDB | None = None
        self.channel: queue.Queue[tuple[Job, Reply]] = queue.Queue()
        self.bdp: BayesianDecisionProcess | None = None
        self.enabled = False
        self.headers: list[str] = []
        self._entities: list[Entity] = []
        self._assignments: dict[str, tuple[int, int]] = {}
        self._rankings: list[EntityWithId] | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue[list[EntityWithId] | None]] = []
        self._subscriber_lock = threading.Lock()
        self._ready = threading.Event()
        self._bootstrap_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="judge-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()
        _ = self._ready.wait()
        if self._bootstrap_error is not None:
            raise self._bootstrap_error

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        reply: Reply = queue.Queue(maxsize=1)
        self.channel.put((None, reply))
        _ = reply.get()
        self._thread.join(timeout=5)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe_rankings(self) -> asyncio.Queue[list[EntityWithId] | None]:
        subscriber: asyncio.Queue[list[EntityWithId] | None] = asyncio.Queue()
        with self._subscriber_lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe_rankings(self, subscriber: asyncio.Queue[list[EntityWithId] | None]) -> None:
        with self._subscriber_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def get_enabled(self) -> bool:
        return self.enabled

    def get_headers(self) -> list[str]:
        return list(self.headers)

    def get_row(self, row_id: int) -> EntityWithId:
        entities = self._entities
        if row_id < 0 or row_id >= len(entities):
            raise UnknownRowException()
        return self._with_id(entities[row_id], row_id)

    def replace_entities(self, entities: list[Entity], headers: list[str]) -> None:
        self._call(lambda: self._replace_entities(entities, headers))

    def resume(self) -> None:
        self._call(lambda: self._set_enabled(True))

    def stop(self) -> None:
        self._call(lambda: self._set_enabled(False))

    def request_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        return self._call(lambda: self._get_pair(pair_request))

    def submit(self, comparison: ComparisonInputModel) -> None:
        self._call(lambda: self._submit(comparison))

    def rankings(self) -> list[EntityWithId]:
        return self._call(self._rankings_snapshot)

    def _call[T](self, fn: Callable[[], T]) -> T:
        reply: Reply = queue.Queue(maxsize=1)
        self.channel.put((fn, reply))
        ok, result = reply.get()
        if not ok:
            if isinstance(result, Exception):
                raise result
            raise RuntimeError("Judge worker failed")
        return cast(T, result)

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
                job, reply = self.channel.get()
                if job is None:
                    reply.put((True, None))
                    return
                try:
                    reply.put((True, job()))
                except Exception as exc:
                    if not isinstance(exc, JudgingFailure):
                        logger.exception("Judge worker command failed")
                    reply.put((False, exc))
        finally:
            if self._db is not None:
                self._db.close()
                self._db = None

    def _bootstrap(self) -> None:
        self._db = JudgeDB(self._path)
        self._install(self._db.load())
        if self.bdp is not None:
            self._publish()

    def _replace_entities(self, entities: list[Entity], headers: list[str]) -> None:
        if self.enabled:
            raise JudgingAlreadyStartedException()
        record = JudgeRecord(
            enabled=True,
            headers=list(headers),
            entities=list(entities),
            assignments={},
            bdp=BayesianDecisionProcess.create(len(entities)),
        )
        self._require_db().replace(record)
        self._install(record)
        self._close_subscribers()
        self._publish()

    def _set_enabled(self, enabled: bool) -> None:
        if enabled:
            if self.enabled:
                raise JudgingAlreadyStartedException()
            if self.bdp is None:
                raise JudgingNeverStartedException()
        elif not self.enabled:
            raise JudgingNotStartedException()
        self.enabled = enabled
        self._commit()

    def _get_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()
        assigned = None if pair_request.force else self._assignments.get(pair_request.uuid)
        if assigned is not None:
            return self._pair(*assigned)

        try:
            i, j = self._require_bdp().get_next_pair()
            self._assignments[pair_request.uuid] = (i, j)
            self._require_db().assign(self._require_bdp(), pair_request.uuid, (i, j))
        except Exception:
            self._reload()
            raise
        return self._pair(i, j)

    def _submit(self, comparison: ComparisonInputModel) -> None:
        if not self.enabled:
            raise JudgingNotStartedException()
        judge = comparison.uuid
        entity_id_1, entity_id_2 = comparison.entity_ids
        winner_id = comparison.winner_id
        pair = self._assignments.get(judge)
        if pair is None or entity_id_1 not in pair or entity_id_2 not in pair:
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

        try:
            self._require_bdp().submit_comparison(entity_id_1, entity_id_2, winner_id)
            del self._assignments[judge]
            self._require_db().finish_comparison(self._require_bdp(), judge)
        except Exception:
            self._reload()
            raise
        self._publish()
        logger.info(
            "Applied comparison",
            extra={
                "event": "comparison",
                "user_id": judge,
                "entity_ids": [entity_id_1, entity_id_2],
                "winner_id": winner_id,
            },
        )

    def _commit(self) -> None:
        try:
            self._require_db().set_enabled(self.enabled)
        except Exception:
            logger.exception("Failed to commit judge state")
            self._reload()
            raise

    def _install(self, record: JudgeRecord) -> None:
        self.enabled = record.enabled
        self.headers = list(record.headers)
        self._entities = list(record.entities)
        self._assignments = dict(record.assignments)
        self.bdp = record.bdp

    def _require_db(self) -> JudgeDB:
        if self._db is None:
            raise RuntimeError("Judge database is not open")
        return self._db

    def _reload(self) -> None:
        self._install(self._require_db().load())
        if self.bdp is not None:
            self._publish()
        else:
            with self._lock:
                self._rankings = None

    def _require_bdp(self) -> BayesianDecisionProcess:
        if self.bdp is None:
            raise RuntimeError("Judge model is not initialized")
        return self.bdp

    def _pair(self, i: int, j: int) -> tuple[EntityWithId, EntityWithId]:
        return (self._with_id(self._entities[i], i), self._with_id(self._entities[j], j))

    def _with_id(self, entity: Entity, index: int) -> EntityWithId:
        return EntityWithId(attributes=dict(entity.attributes), id=index)

    def _rankings_snapshot(self) -> list[EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()
        with self._lock:
            if self._rankings is None:
                raise RuntimeError("Judge model is not initialized")
            return list(self._rankings)

    def _publish(self) -> None:
        entities = self._entities
        alphas = np.asarray(self._require_bdp().get_alphas(), dtype=np.float64)
        ranked_ids = sorted(range(len(entities)), key=lambda index: alphas[index], reverse=True)
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

    def _close_subscribers(self) -> None:
        loop = self._loop
        with self._subscriber_lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        if loop is None or loop.is_closed():
            return
        for subscriber in subscribers:
            try:
                loop.call_soon_threadsafe(subscriber.put_nowait, None)
            except RuntimeError:
                return
