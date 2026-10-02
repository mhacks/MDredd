import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import numpy as np
from peewee import DatabaseError

from app.algorithm import BayesianDecisionProcess
from app.db import (
    AbsentSkip,
    JudgeRecord,
    close_db,
    load_state,
    open_db,
    replace_state,
    reset_db,
    save_absence,
    save_assignment,
    save_comparison,
    save_enabled,
    save_strikes,
)
from app.entity import Entity
from app.exceptions import (
    AbsentNotInPairException,
    DatabaseUnreadableException,
    IncorrectPairFormatException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingFailure,
    JudgingNeverStartedException,
    JudgingNotStartedException,
    PoolExhaustedException,
    TooFewEntitiesException,
    UnknownRowException,
    WorkerUnavailableException,
)
from app.models import (
    ComparisonInputModel,
    EntityWithId,
    PairRequestModel,
    PoolEntryModel,
)
from app.settings import settings

logger = logging.getLogger(__name__)


def _active_mask(strikes: list[int], exclude: tuple[int, ...] = ()) -> np.ndarray:
    limit = settings.STRIKE_LIMIT
    mask = np.array([count < limit for count in strikes], dtype=bool)
    for entity_id in exclude:
        mask[entity_id] = False
    return mask


def _streak(count: int, appeared: bool) -> int:
    if count >= settings.STRIKE_LIMIT:
        return count
    if appeared:
        return 0
    return count + 1


def _refund(frequency: jnp.ndarray, entity_id: int) -> jnp.ndarray:
    refunded = jnp.maximum(frequency[entity_id] - jnp.int32(1), jnp.int32(0))
    return frequency.at[entity_id].set(refunded.astype(frequency.dtype))


Job = Callable[[], object] | None
Reply = queue.Queue[object]
Command = tuple[Job, Reply, bool]


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
        self.channel: queue.Queue[Command] = queue.Queue(maxsize=queue_size)
        self._timeout = timeout
        self._stuck_after = stuck_after
        self._busy_since: float | None = None
        self.bdp: BayesianDecisionProcess | None = None
        self.enabled = False
        self.headers: list[str] = []
        self._entities: list[Entity] = []
        self._assignments: dict[str, tuple[int, int]] = {}
        self._completed: dict[str, tuple[int, int, int]] = {}
        self._strikes: list[int] = []
        self._last_skips: dict[str, AbsentSkip] = {}
        self._ready = threading.Event()
        self._unreadable = False
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
            self.channel.put((None, reply, True), timeout=self._timeout)
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

    def pool(self) -> list[PoolEntryModel]:
        return self._call(self._pool)

    def restore(self, entity_id: int) -> PoolEntryModel:
        return self._call(lambda: self._restore(entity_id))

    def reset(self) -> None:
        self._call(self._reset, when_unreadable=True)

    def _call[T](self, fn: Callable[[], T], *, when_unreadable: bool = False) -> T:
        if not self.healthy():
            raise WorkerUnavailableException()
        reply: Reply = queue.Queue(maxsize=1)
        try:
            self.channel.put_nowait((fn, reply, when_unreadable))
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
            job, reply, when_unreadable = self.channel.get()
            if job is None:
                reply.put(None)
                return
            if self._unreadable and not when_unreadable:
                reply.put(DatabaseUnreadableException())
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

    def _fail_pending(self) -> None:
        while True:
            try:
                job, reply, _when_unreadable = self.channel.get_nowait()
            except queue.Empty:
                return
            reply.put(None if job is None else WorkerUnavailableException())

    def _bootstrap(self) -> None:
        try:
            open_db()
            self._install(load_state())
        except DatabaseError:
            # Leave the file in place. The process keeps serving so an
            # organizer can call DELETE /database.
            close_db()
            self._unreadable = True
            logger.critical(
                "SQLite database could not be loaded. "
                "DELETE /database to remove it and start empty."
            )

    def _reset(self) -> None:
        reset_db()
        self._install(load_state())
        self._unreadable = False

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
        if pair_request.absent:
            return self._report_absence(pair_request.judge_id, pair_request.absent)
        assigned = self._assignments.get(pair_request.judge_id)
        if assigned is not None:
            return self._pair(*assigned)
        return self._draw_for(pair_request.judge_id, ())

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
        strike_counts = [
            (entity_id, _streak(self._strikes[entity_id], True))
            for entity_id in submitted
        ]
        save_comparison(alpha, judge, completed, (strike_counts[0], strike_counts[1]))
        bdp.alpha_t = alpha
        for entity_id, strike_count in strike_counts:
            self._strikes[entity_id] = strike_count
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

    def _draw_for(
        self,
        judge_id: str,
        exclude: tuple[int, ...],
        absent_key: tuple[int, ...] | None = None,
    ) -> tuple[EntityWithId, EntityWithId]:
        mask = _active_mask(self._strikes, exclude)
        if int(mask.sum()) < 2:
            raise PoolExhaustedException()
        bdp = self._require_bdp()
        left, right, frequency, key = bdp.propose_pair(mask)
        pair = (left, right)
        if absent_key is None:
            save_assignment(frequency, key, judge_id, pair)
        else:
            save_absence(
                frequency,
                self._strikes,
                key,
                None,
                {judge_id: pair},
                None,
                judge_id,
                absent_key,
                pair,
            )
        bdp.frequency = frequency
        bdp.key = key
        self._assignments[judge_id] = pair
        if absent_key is not None:
            self._last_skips[judge_id] = (absent_key, pair)
        return self._pair(*pair)

    def _report_absence(
        self, judge_id: str, absent: list[int]
    ) -> tuple[EntityWithId, EntityWithId]:
        absent_key = tuple(sorted(set(absent)))
        current = self._assignments.get(judge_id)
        last = self._last_skips.get(judge_id)
        if last is not None and last[0] == absent_key:
            stored = last[1]
            if stored is not None and current == stored:
                logger.info(
                    "Ignored repeated strike",
                    extra={
                        "event": "strike_repeated",
                        "user_id": judge_id,
                        "entity_ids": list(absent_key),
                    },
                )
                return self._pair(*stored)
            if stored is None and current is None:
                exclude = absent_key if len(absent_key) == 2 else ()
                return self._draw_for(judge_id, exclude, absent_key)
        if (
            not absent_key
            or current is None
            or not set(absent_key).issubset(current)
        ):
            logger.info(
                "Rejected strike",
                extra={
                    "event": "strike_rejected",
                    "user_id": judge_id,
                    "entity_ids": list(absent),
                },
            )
            raise AbsentNotInPairException()

        bdp = self._require_bdp()
        frequency = bdp.frequency
        key = bdp.key
        strikes = list(self._strikes)
        struck: list[int] = []
        for entity_id in absent_key:
            count = _streak(strikes[entity_id], False)
            if count != strikes[entity_id]:
                struck.append(entity_id)
            strikes[entity_id] = count
        alpha = None
        winner_id = None
        ordered = None
        if len(absent_key) == 1:
            missing = absent_key[0]
            present = current[0] if current[1] == missing else current[1]
            alpha = bdp.propose_comparison(current[0], current[1], present)
            winner_id = present
            ordered = (min(current), max(current))
            strikes[present] = _streak(strikes[present], True)
        else:
            for entity_id in absent_key:
                frequency = _refund(frequency, entity_id)

        limit = settings.STRIKE_LIMIT
        removed = [
            entity_id
            for entity_id in absent_key
            if strikes[entity_id] >= limit and self._strikes[entity_id] < limit
        ]
        assignment_updates: dict[str, tuple[int, int] | None] = {}
        exclude = absent_key if len(absent_key) == 2 else ()
        mask = _active_mask(strikes, exclude)
        new_pair = None
        if int(mask.sum()) < 2:
            assignment_updates[judge_id] = None
        else:
            left, right, frequency, key = bdp.propose_pair(
                mask, frequency=frequency, key=key
            )
            new_pair = (left, right)
            assignment_updates[judge_id] = new_pair

        completed = None
        if ordered is not None and winner_id is not None:
            completed = (judge_id, (*ordered, winner_id))
        save_absence(
            frequency,
            strikes,
            key,
            alpha,
            assignment_updates,
            completed,
            judge_id,
            absent_key,
            new_pair,
        )
        bdp.frequency = frequency
        bdp.key = key
        if alpha is not None:
            bdp.alpha_t = alpha
        self._strikes = strikes
        for other, pair in assignment_updates.items():
            if pair is None:
                self._assignments.pop(other, None)
            else:
                self._assignments[other] = pair
        if completed is not None:
            self._completed[judge_id] = completed[1]
        self._last_skips[judge_id] = (absent_key, new_pair)
        if ordered is not None and winner_id is not None:
            logger.info(
                "Applied comparison",
                extra={
                    "event": "comparison",
                    "user_id": judge_id,
                    "entity_ids": [ordered[0], ordered[1]],
                    "winner_id": winner_id,
                },
            )
        for entity_id in struck:
            logger.info(
                "Applied strike",
                extra={
                    "event": "strike",
                    "user_id": judge_id,
                    "entity_ids": [entity_id],
                },
            )
        for entity_id in removed:
            logger.info(
                "Removed project",
                extra={
                    "event": "project_removed",
                    "user_id": judge_id,
                    "entity_ids": [entity_id],
                },
            )
        if new_pair is None:
            raise PoolExhaustedException()
        return self._pair(*new_pair)

    def _pool(self) -> list[PoolEntryModel]:
        return [self._pool_entry(index) for index in range(len(self._entities))]

    def _restore(self, entity_id: int) -> PoolEntryModel:
        entry = self._pool_entry(entity_id)
        if not entry.removed:
            logger.info(
                "Ignored repeated restore",
                extra={"event": "restore_repeated", "entity_ids": [entity_id]},
            )
            return entry
        save_strikes(entity_id, 0)
        self._strikes[entity_id] = 0
        logger.info(
            "Applied restore",
            extra={"event": "restore", "entity_ids": [entity_id]},
        )
        return self._pool_entry(entity_id)

    def _pool_entry(self, entity_id: int) -> PoolEntryModel:
        row = self._row(entity_id)
        count = self._strikes[row.id]
        return PoolEntryModel(
            id=row.id,
            attributes=dict(row.attributes),
            strikes=count,
            removed=count >= settings.STRIKE_LIMIT,
        )

    def _install(self, record: JudgeRecord) -> None:
        self.enabled = record.enabled
        self.headers = list(record.headers)
        self._entities = list(record.entities)
        self._assignments = dict(record.assignments)
        self._completed = dict(record.completed)
        self._strikes = list(record.strikes)
        self._last_skips = dict(record.last_skips)
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
