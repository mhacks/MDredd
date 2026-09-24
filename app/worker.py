import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

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
    UnknownRowException,
)
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel

logger = logging.getLogger(__name__)


class JudgeWorker:
    """Owns the Bayesian judge on one thread.

    Pair draws and comparisons update the in-memory model, then commit that
    model and the one assignment. Entities stay as they were written at upload.
    The caller is answered only after the commit succeeds.
    """

    def __init__(self) -> None:
        self.bdp: BayesianDecisionProcess | None = None
        self.enabled = False
        self.headers: list[str] = []
        self._entities: list[Entity] = []
        self._assignments: dict[str, tuple[int, int]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="judge-worker"
        )

    async def start(self) -> None:
        try:
            await asyncio.wrap_future(self._executor.submit(self._bootstrap))
        except Exception:
            logger.exception("Judge worker failed to recover")
            await self.shutdown()
            raise

    async def shutdown(self, timeout: float = 5) -> None:
        try:
            closed = asyncio.wrap_future(self._executor.submit(close_db))
        except RuntimeError:
            return
        self._executor.shutdown(wait=False)
        _, pending = await asyncio.wait({closed}, timeout=timeout)
        if pending:
            logger.warning("Judge worker did not stop in time")

    async def get_enabled_async(self) -> bool:
        return await self._call_async(lambda: self.enabled)

    async def get_headers_async(self) -> list[str]:
        return await self._call_async(lambda: list(self.headers))

    async def get_row_async(self, row_id: int) -> EntityWithId:
        return await self._call_async(lambda: self._row(row_id))

    async def replace_entities_async(
        self, entities: list[Entity], headers: list[str]
    ) -> None:
        await self._call_async(lambda: self._replace_entities(entities, headers))

    async def resume_async(self) -> None:
        await self._call_async(lambda: self._set_enabled(True))

    async def stop_async(self) -> None:
        await self._call_async(lambda: self._set_enabled(False))

    async def request_pair_async(
        self, pair_request: PairRequestModel
    ) -> tuple[EntityWithId, EntityWithId]:
        return await self._call_async(lambda: self._get_pair(pair_request))

    async def submit_async(self, comparison: ComparisonInputModel) -> None:
        await self._call_async(lambda: self._submit(comparison))

    async def rankings_async(self) -> list[EntityWithId]:
        return await self._call_async(self._rankings_snapshot)

    async def _call_async[T](self, fn: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run, fn)

    def _run[T](self, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except Exception as exc:
            if not isinstance(exc, JudgingFailure):
                logger.exception("Judge worker command failed")
            raise

    def _bootstrap(self) -> None:
        open_db()
        self._install(load_state())

    def _replace_entities(self, entities: list[Entity], headers: list[str]) -> None:
        if self.enabled:
            raise JudgingAlreadyStartedException()
        bdp = BayesianDecisionProcess.create(len(entities))
        self._install(replace_state(headers, entities, bdp))

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

    def _get_pair(
        self, pair_request: PairRequestModel
    ) -> tuple[EntityWithId, EntityWithId]:
        if not self.enabled:
            raise JudgingNotStartedException()
        assigned = (
            None if pair_request.force else self._assignments.get(pair_request.uuid)
        )
        if assigned is not None:
            return self._pair(*assigned)

        def draw() -> tuple[int, int]:
            i, j = self._require_bdp().get_next_pair()
            self._assignments[pair_request.uuid] = (i, j)
            save_assignment(self._require_bdp(), pair_request.uuid, (i, j))
            return i, j

        return self._pair(*self._persist(draw))

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

        def apply() -> None:
            self._require_bdp().submit_comparison(entity_id_1, entity_id_2, winner_id)
            del self._assignments[judge]
            save_comparison(self._require_bdp(), judge)

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
            self._reload()
            raise

    def _install(self, record: JudgeRecord) -> None:
        self.enabled = record.enabled
        self.headers = list(record.headers)
        self._entities = list(record.entities)
        self._assignments = dict(record.assignments)
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
        if not self.enabled:
            raise JudgingNotStartedException()
        entities = self._entities
        alphas = np.asarray(self._require_bdp().get_alphas(), dtype=np.float64)
        ranked_ids = sorted(
            range(len(entities)), key=lambda index: alphas[index], reverse=True
        )
        return [self._with_id(entities[index], index) for index in ranked_ids]
