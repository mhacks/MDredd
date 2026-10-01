import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import cast

import numpy as np

from app.algorithm import BayesianDecisionProcess
from app.db import (
    JudgeRecord,
    close_db,
    load_state,
    open_db,
    replace_state,
    save_assignment,
    save_comparison,
    save_enabled,
)
from app.entity import Entity
from app.exceptions import (
    IncorrectPairFormatException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingFailure,
    JudgingNeverStartedException,
    JudgingNotStartedException,
    TooFewEntitiesException,
    UnknownRowException,
    WorkerUnavailableException,
)
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.settings import settings

logger = logging.getLogger(__name__)

Job = Callable[[], object] | None
Reply = queue.Queue[object]


class JudgeWorker:
    """Owns the Bayesian judge on one thread.

    A command commits its next state, then publishes it, and the caller is
    answered after that commit. Repeating a command that already landed reports
    the stored state. Callers wait at most `timeout` seconds. A full queue, a
    dead thread, or a command running longer than `stuck_after` seconds is
    refused.
    """

    def __init__(
        self,
        timeout: float = settings.WORKER_TIMEOUT_SECONDS,
        stuck_after: float = settings.WORKER_STUCK_SECONDS,
        queue_size: int = settings.WORKER_QUEUE_SIZE,
    ) -> None:
        self.channel: queue.Queue[tuple[Job, Reply]] = queue.Queue(maxsize=queue_size)
        self._timeout = timeout
        self._stuck_after = stuck_after
        self._busy_since: float | None = None
        self.bdp: BayesianDecisionProcess | None = None
        self.enabled = False
        self.headers: list[str] = []
        self._entities: list[Entity] = []
        self._assignments: dict[str, tuple[int, int]] = {}
        self._completed: dict[str, tuple[int, int, int]] = {}
        self._ready = threading.Event()
        self._bootstrap_error: Exception | None = None
        self._thread = threading.Thread(
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
        reply: Reply = queue.Queue(maxsize=1)
        try:
            self.channel.put((None, reply), timeout=self._timeout)
            _ = reply.get(timeout=self._timeout)
        except (queue.Full, queue.Empty):
            logger.error("Judge worker did not stop in time")
            return
        self._thread.join(timeout=5)

    def healthy(self) -> bool:
        if not self._thread.is_alive():
            return False
        busy_since = self._busy_since
        return busy_since is None or time.monotonic() - busy_since < self._stuck_after

    def get_enabled(self) -> bool:
        return self._call(lambda: self.enabled)

    def get_headers(self) -> list[str]:
        return self._call(lambda: list(self.headers))

    def get_row(self, row_id: int) -> EntityWithId:
        return self._call(lambda: self._row(row_id))

    def replace_entities(
        self, entities: list[Entity], headers: list[str]
    ) -> list[str]:
        return self._call(lambda: self._replace_entities(entities, headers))

    def resume(self) -> bool:
        return self._call(lambda: self._set_enabled(True))

    def stop(self) -> bool:
        return self._call(lambda: self._set_enabled(False))

    def request_pair(
        self, pair_request: PairRequestModel
    ) -> tuple[EntityWithId, EntityWithId]:
        return self._call(lambda: self._get_pair(pair_request))

    def submit(self, comparison: ComparisonInputModel) -> None:
        self._call(lambda: self._submit(comparison))

    def rankings(self) -> list[EntityWithId]:
        return self._call(self._rankings_snapshot)

    def _call[T](self, fn: Callable[[], T]) -> T:
        if not self.healthy():
            raise WorkerUnavailableException()
        reply: Reply = queue.Queue(maxsize=1)
        try:
            self.channel.put_nowait((fn, reply))
        except queue.Full:
            logger.error("Judge worker queue is full")
            raise WorkerUnavailableException() from None
        try:
            result = reply.get(timeout=self._timeout)
        except queue.Empty:
            logger.error("Judge worker did not answer in time")
            raise WorkerUnavailableException() from None
        if isinstance(result, Exception):
            raise result
        return cast(T, result)

    def _run(self) -> None:
        try:
            self._bootstrap()
        except Exception as exc:
            self._bootstrap_error = exc
            logger.exception("Judge worker failed to recover")
            self._ready.set()
            # Nothing committed is lost. The restart loads the last snapshot.
            os._exit(1)
        self._ready.set()
        try:
            self._serve()
        finally:
            self._fail_pending()
            close_db()

    def _serve(self) -> None:
        while True:
            job, reply = self.channel.get()
            if job is None:
                reply.put(None)
                return
            self._busy_since = time.monotonic()
            try:
                reply.put(job())
            except Exception as exc:
                if not isinstance(exc, JudgingFailure):
                    logger.exception("Judge worker command failed")
                reply.put(exc)
            except BaseException:
                reply.put(WorkerUnavailableException())
                raise
            finally:
                self._busy_since = None

    def _fail_pending(self) -> None:
        while True:
            try:
                job, reply = self.channel.get_nowait()
            except queue.Empty:
                return
            reply.put(None if job is None else WorkerUnavailableException())

    def _bootstrap(self) -> None:
        open_db()
        self._install(load_state())

    def _replace_entities(
        self, entities: list[Entity], headers: list[str]
    ) -> list[str]:
        if len(entities) < 2:
            raise TooFewEntitiesException()
        if self.enabled and entities == self._entities and headers == self.headers:
            return list(self.headers)
        if self.enabled:
            raise JudgingAlreadyStartedException()
        bdp = BayesianDecisionProcess.create(len(entities))
        self._install(replace_state(headers, entities, bdp))
        return list(self.headers)

    def _set_enabled(self, enabled: bool) -> bool:
        if enabled and self.bdp is None:
            raise JudgingNeverStartedException()
        if self.enabled == enabled:
            return enabled
        save_enabled(enabled)
        self.enabled = enabled
        return enabled

    def _get_pair(
        self, pair_request: PairRequestModel
    ) -> tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()
        assigned = (
            None if pair_request.force else self._assignments.get(pair_request.judge_id)
        )
        if assigned is not None:
            return self._pair(*assigned)

        bdp = self._require_bdp()
        left, right, frequency, key = bdp.propose_pair()
        save_assignment(frequency, key, pair_request.judge_id, (left, right))
        bdp.frequency = frequency
        bdp.key = key
        self._assignments[pair_request.judge_id] = (left, right)
        return self._pair(left, right)

    def _submit(self, comparison: ComparisonInputModel) -> None:
        if not self.enabled:
            raise JudgingNotStartedException()
        judge = comparison.judge_id
        entity_id_1, entity_id_2 = comparison.entity_ids
        winner_id = comparison.winner_id
        submitted = (min(entity_id_1, entity_id_2), max(entity_id_1, entity_id_2))
        pair = self._assignments.get(judge)
        if pair != submitted:
            # A retry of a comparison that was already applied (for example
            # after a lost response) succeeds without being counted again.
            if self._completed.get(judge) == (*submitted, winner_id):
                logger.info(
                    "Ignored repeated comparison",
                    extra={
                        "event": "comparison_repeated",
                        "user_id": judge,
                        "entity_ids": [entity_id_1, entity_id_2],
                        "winner_id": winner_id,
                    },
                )
                return
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

        bdp = self._require_bdp()
        alpha = bdp.propose_comparison(entity_id_1, entity_id_2, winner_id)
        completed = (*submitted, winner_id)
        save_comparison(alpha, judge, completed)
        bdp.alpha_t = alpha
        del self._assignments[judge]
        self._completed[judge] = completed
        logger.info(
            "Applied comparison",
            extra={
                "event": "comparison",
                "user_id": judge,
                "entity_ids": [entity_id_1, entity_id_2],
                "winner_id": winner_id,
            },
        )

    def _install(self, record: JudgeRecord) -> None:
        self.enabled = record.enabled
        self.headers = list(record.headers)
        self._entities = list(record.entities)
        self._assignments = dict(record.assignments)
        self._completed = dict(record.completed)
        self.bdp = record.bdp

    def _require_bdp(self) -> BayesianDecisionProcess:
        if self.bdp is None:
            raise RuntimeError("Judge model is not initialized")
        return self.bdp

    def _pair(self, i: int, j: int) -> tuple[EntityWithId, EntityWithId]:
        return (
            self._with_id(self._entities[i], i),
            self._with_id(self._entities[j], j),
        )

    def _row(self, row_id: int) -> EntityWithId:
        entities = self._entities
        if row_id < 0 or row_id >= len(entities):
            raise UnknownRowException()
        return self._with_id(entities[row_id], row_id)

    def _with_id(self, entity: Entity, index: int) -> EntityWithId:
        return EntityWithId(attributes=dict(entity.attributes), id=index)

    def _rankings_snapshot(self) -> list[EntityWithId]:
        # Rankings stay readable after judging stops; only a missing model blocks them.
        if self.bdp is None:
            raise JudgingNeverStartedException()
        entities = self._entities
        alphas = self.bdp.get_alphas()[: len(entities)]
        ranked_ids = np.argsort(-alphas, kind="stable").tolist()
        return [self._with_id(entities[index], index) for index in ranked_ids]
