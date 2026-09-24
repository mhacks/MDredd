import argparse
import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter

from starlette.concurrency import run_in_threadpool

from app.db import close_db, db
from app.worker import JudgeWorker


async def measure(operation: Callable[[], Awaitable[object]]) -> float:
    started = perf_counter()
    await operation()
    return perf_counter() - started


async def compare(requests: int) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as directory:
        close_db()
        db.init(str(Path(directory) / "mdredd.db"))
        worker = JudgeWorker()
        worker.start()
        try:
            await worker.get_headers_async()
            await run_in_threadpool(worker.get_headers)

            async def legacy() -> object:
                return await asyncio.gather(
                    *(run_in_threadpool(worker.get_headers) for _ in range(requests))
                )

            async def future_bridge() -> object:
                return await asyncio.gather(
                    *(worker.get_headers_async() for _ in range(requests))
                )

            legacy_seconds = await measure(legacy)
            future_seconds = await measure(future_bridge)
            return legacy_seconds, future_seconds
        finally:
            worker.shutdown()
            close_db()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5_000)
    args = parser.parse_args()

    legacy, future = asyncio.run(compare(args.requests))
    speedup = legacy / future
    print(f"requests:       {args.requests:,}")
    print(f"thread pool:    {legacy:.4f}s")
    print(f"future bridge:  {future:.4f}s")
    print(f"speedup:        {speedup:.2f}x")


if __name__ == "__main__":
    main()
