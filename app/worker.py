import logging
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

    Pair draws and comparisons update the in-memory model, then commit that
    model and the one assignment. Entities stay as they were written at upload.
    The caller is answered only after the commit succeeds.

    A caller waits at most `timeout` seconds for an answer. A job whose caller
    has already given up is skipped, and a job still running after
    `stuck_after` seconds marks the worker unhealthy.
    """

    def __init__(
        self,
        timeout: float = settings.WORKER_TIMEOUT_SECONDS,
        stuck_after: float = settings.WORKER_STUCK_SECONDS,
    ) -> None:
        self.channel: queue.Queue[tuple[Job, Reply, threading.Event]] = queue.Queue()
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
        self.channel.put((None, reply, threading.Event()))
        try:
            _ = reply.get(timeout=self._timeout)
        except queue.Empty:
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

    def replace_entities(self, entities: list[Entity], headers: list[str]) -> None:
        self._call(lambda: self._replace_entities(entities, headers))

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
        if not self._thread.is_alive():
            raise WorkerUnavailableException()
        reply: Reply = queue.Queue(maxsize=1)
        abandoned = threading.Event()
        self.channel.put((fn, reply, abandoned))
        try:
            result = reply.get(timeout=self._timeout)
        except queue.Empty:
            abandoned.set()
            logger.error("Judge worker did not answer in time")
            raise WorkerUnavailableException() from None
        if isinstance(result, Exception):
            raise result
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
                job, reply, abandoned = self.channel.get()
                if job is None:
                    reply.put(None)
                    return
                if abandoned.is_set():
                    continue
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
        finally:
            self._fail_pending()
            close_db()

    def _fail_pending(self) -> None:
        while True:
            try:
                job, reply, _abandoned = self.channel.get_nowait()
            except queue.Empty:
                return
            reply.put(None if job is None else WorkerUnavailableException())

    def _bootstrap(self) -> None:
        open_db()
        self._install(load_state())

    def _replace_entities(self, entities: list[Entity], headers: list[str]) -> None:
        if self.enabled:
            raise JudgingAlreadyStartedException()
        if len(entities) < 2:
            raise TooFewEntitiesException()
        bdp = BayesianDecisionProcess.create(len(entities))
        self._install(replace_state(headers, entities, bdp))

    def _set_enabled(self, enabled: bool) -> bool:
        if enabled:
            if self.enabled:
                raise JudgingAlreadyStartedException()
            if self.bdp is None:
                raise JudgingNeverStartedException()
        elif not self.enabled:
            raise JudgingNotStartedException()
        self.enabled = enabled
        self._commit()
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

        def draw() -> tuple[int, int]:
            bdp = self._require_bdp()
            i, j = bdp.get_next_pair()
            self._assignments[pair_request.judge_id] = (i, j)
            save_assignment(bdp, pair_request.judge_id, (i, j))
            return i, j

        return self._pair(*self._persist(draw))

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

        def apply() -> None:
            self._require_bdp().submit_comparison(entity_id_1, entity_id_2, winner_id)
            del self._assignments[judge]
            self._completed[judge] = (*submitted, winner_id)
            save_comparison(self._require_bdp(), judge, self._completed[judge])

        self._persist(apply)
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
        self._persist(lambda: save_enabled(self.enabled))

    def _persist[T](self, write: Callable[[], T]) -> T:
        try:
            return write()
        except Exception:
            try:
                self._reload()
            except Exception:
                # Keeping unsaved state would let it drift from the database.
                logger.exception("Judge worker failed to reload after a write")
                self._install(
                    JudgeRecord(
                        enabled=False,
                        headers=[],
                        entities=[],
                        assignments={},
                        completed={},
                        bdp=None,
                    )
                )
            raise

    def _install(self, record: JudgeRecord) -> None:
        self.enabled = record.enabled
        self.headers = list(record.headers)
        self._entities = list(record.entities)
        self._assignments = dict(record.assignments)
        self._completed = dict(record.completed)
        self.bdp = record.bdp

    def _reload(self) -> None:
        self._install(load_state())

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
