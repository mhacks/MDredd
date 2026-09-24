import asyncio
import io
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import UploadFile
from graphql import GraphQLError

import app.api.schema as schema_module
from app.api import graphql_router
from app.api.types import GraphQLContext, run_judging
from app.db import close_db, db
from app.entity import Entity
from app.exceptions import JudgingNotStartedException
from app.models import ComparisonInputModel, PairRequestModel
from app.session import Session
from app.worker import JudgeWorker


@pytest.fixture
def isolated_database(tmp_path: Path) -> Iterator[None]:
    close_db()
    original = db.database
    db.init(str(tmp_path / "mdredd.db"), pragmas={"journal_mode": "wal"})
    try:
        yield
    finally:
        close_db()
        db.init(original)


async def wait_for(event: threading.Event) -> None:
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_async_api_runs_judging_workflow(isolated_database: None) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        await worker.start()
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
            await worker.shutdown()

    asyncio.run(exercise())


def test_cancelled_wait_skips_queued_job_and_worker_survives(
    isolated_database: None,
) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        await worker.start()
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
            await wait_for(first_started)

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
            await worker.shutdown()

    asyncio.run(exercise())


def test_cancelled_wait_on_running_job_still_completes(
    isolated_database: None,
) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        await worker.start()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_job() -> None:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            finished.set()

        try:
            running = asyncio.create_task(worker._call_async(blocking_job))
            await wait_for(started)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running

            release.set()
            assert await worker.get_headers_async() == []
            assert finished.is_set()
        finally:
            release.set()
            await worker.shutdown()

    asyncio.run(exercise())


def test_calls_after_shutdown_fail_fast(isolated_database: None) -> None:
    async def exercise() -> None:
        worker = JudgeWorker()
        await worker.start()
        await worker.shutdown()
        await worker.shutdown()
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError):
                await worker.get_headers_async()

    asyncio.run(exercise())


def test_start_async_loads_csv_and_maps_invalid_columns(
    isolated_database: None,
) -> None:
    async def exercise() -> None:
        session = Session()
        await session.open()
        try:
            columns = await session.start_async(b"name,score\nAda,1\nGrace,2\n")
            assert [column.header for column in columns] == ["name", "score"]
            assert await session.worker.get_headers_async() == ["name", "score"]
            assert await session.worker.get_enabled_async() is True

            await session.worker.stop_async()
            with pytest.raises(GraphQLError) as caught:
                await run_judging(lambda: session.start_async(b"id\n1\n"))
            assert caught.value.extensions == {
                "code": "INVALID_COLUMNS",
                "names": ["id"],
            }
            assert await session.worker.get_headers_async() == ["name", "score"]
        finally:
            await session.close()

    asyncio.run(exercise())


def test_cancelled_upload_still_rebinds_schema(
    isolated_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_module, "_router", None)
    router = graphql_router()
    mutation = "mutation($csv: Upload) { startJudging(entitiesCsv: $csv) { isStarted } }"

    async def exercise() -> None:
        session = Session()
        await session.open()
        replace_started = threading.Event()
        release_replace = threading.Event()
        replace = session.worker._replace_entities

        def slow_replace(entities: list[Entity], headers: list[str]) -> None:
            replace_started.set()
            if not release_replace.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            replace(entities, headers)

        monkeypatch.setattr(session.worker, "_replace_entities", slow_replace)
        try:
            upload = asyncio.create_task(
                router.schema.execute(
                    mutation,
                    variable_values={
                        "csv": UploadFile(io.BytesIO(b"name\nAda\nGrace\n"))
                    },
                    context_value=GraphQLContext(session),
                )
            )
            await wait_for(replace_started)
            upload.cancel()
            with pytest.raises(asyncio.CancelledError):
                await upload

            release_replace.set()
            async with asyncio.timeout(5):
                while "name: String!" not in router.schema.as_str():
                    await asyncio.sleep(0.001)
            assert await session.worker.get_headers_async() == ["name"]
        finally:
            release_replace.set()
            await session.close()

    asyncio.run(exercise())


def test_run_judging_maps_async_failures_to_graphql_errors() -> None:
    async def fail() -> None:
        raise JudgingNotStartedException()

    async def exercise() -> None:
        with pytest.raises(GraphQLError) as caught:
            await run_judging(fail)
        assert caught.value.extensions == {"code": "JUDGING_NOT_STARTED"}

    asyncio.run(exercise())
