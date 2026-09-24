import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from graphql import GraphQLError

from app.api.types import run_judging
from app.db import close_db, db
from app.entity import Entity
from app.exceptions import JudgingNotStartedException
from app.models import ComparisonInputModel, PairRequestModel
from app.worker import JudgeWorker


@pytest.fixture
def isolated_database(tmp_path: Path) -> Iterator[None]:
    close_db()
    db.init(str(tmp_path / "mdredd.db"), pragmas={"journal_mode": "wal"})
    try:
        yield
    finally:
        close_db()


def test_async_api_runs_judging_workflow(isolated_database: None) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        worker.start()
        try:
            assert await worker.get_enabled_async() is False
            await worker.replace_entities_async(
                [
                    Entity(attributes={"name": "Ada"}),
                    Entity(attributes={"name": "Grace"}),
                    Entity(attributes={"name": "Katherine"}),
                ],
                ["name"],
            )

            assert await worker.get_headers_async() == ["name"]
            assert (await worker.get_row_async(1)).attributes == {"name": "Grace"}

            pair = await worker.request_pair_async(PairRequestModel(uuid="judge"))
            entity_ids = (pair[0].id, pair[1].id)
            await worker.submit_async(
                ComparisonInputModel(
                    uuid="judge",
                    entity_ids=entity_ids,
                    winner_id=entity_ids[0],
                )
            )

            rankings = await worker.rankings_async()
            assert sorted(entity.id for entity in rankings) == [0, 1, 2]

            await worker.stop_async()
            with pytest.raises(JudgingNotStartedException):
                await worker.request_pair_async(PairRequestModel(uuid="judge"))
            await worker.resume_async()
            assert await worker.get_enabled_async() is True
        finally:
            worker.shutdown()

    asyncio.run(exercise())


def test_cancelled_wait_skips_queued_job_and_worker_survives(
    isolated_database: None,
) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        worker.start()
        first_started = threading.Event()
        release_first = threading.Event()
        cancelled_job_ran = threading.Event()

        def blocking_job() -> str:
            first_started.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            return "finished"

        def queued_job() -> None:
            cancelled_job_ran.set()

        try:
            first = asyncio.create_task(worker._call_async(blocking_job))
            while not first_started.is_set():
                await asyncio.sleep(0.001)

            cancelled = asyncio.create_task(worker._call_async(queued_job))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled

            release_first.set()
            assert await first == "finished"
            assert await worker.get_headers_async() == []
            assert not cancelled_job_ran.is_set()
        finally:
            release_first.set()
            worker.shutdown()

    asyncio.run(exercise())


def test_sync_api_remains_available(isolated_database: None) -> None:
    worker = JudgeWorker()
    worker.start()
    try:
        assert worker.get_enabled() is False
        assert worker.get_headers() == []
    finally:
        worker.shutdown()


def test_run_judging_maps_async_failures_to_graphql_errors() -> None:
    async def fail() -> None:
        raise JudgingNotStartedException()

    async def exercise() -> None:
        with pytest.raises(GraphQLError) as caught:
            await run_judging(fail)
        assert caught.value.extensions == {"code": "JUDGING_NOT_STARTED"}

    asyncio.run(exercise())
