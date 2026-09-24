import argparse
import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter

from starlette.concurrency import run_in_threadpool

from app.db import close_db, db
from app.entity import Entity
from app.models import PairRequestModel
from app.worker import JudgeWorker

Job = Callable[[], object]
Bridge = Callable[[JudgeWorker, Job], Awaitable[object]]


async def thread_pool(worker: JudgeWorker, job: Job) -> object:
    # The previous bridge: a pool thread blocks until the worker runs the job.
    return await run_in_threadpool(lambda: worker._executor.submit(job).result())


async def future_bridge(worker: JudgeWorker, job: Job) -> object:
    return await worker._call_async(job)


BRIDGES: dict[str, Bridge] = {
    "thread pool": thread_pool,
    "future bridge": future_bridge,
}


async def measure(
    bridge: Bridge, worker: JudgeWorker, job: Job, requests: int
) -> float:
    started = perf_counter()
    await asyncio.gather(*(bridge(worker, job) for _ in range(requests)))
    return perf_counter() - started


async def best_times(
    worker: JudgeWorker, job: Job, requests: int, rounds: int
) -> dict[str, float]:
    for bridge in BRIDGES.values():
        await bridge(worker, job)
    best = dict.fromkeys(BRIDGES, float("inf"))
    for round_index in range(rounds):
        # Alternate which bridge goes first so warm-up and GC do not favour one.
        names = list(BRIDGES)[:: -1 if round_index % 2 else 1]
        for name in names:
            seconds = await measure(BRIDGES[name], worker, job, requests)
            best[name] = min(best[name], seconds)
    return best


async def compare(
    requests: int, pairs: int, rounds: int
) -> dict[str, tuple[int, dict[str, float]]]:
    with tempfile.TemporaryDirectory() as directory:
        close_db()
        db.init(str(Path(directory) / "mdredd.db"))
        worker = JudgeWorker()
        await worker.start()
        try:
            await worker.replace_entities_async(
                [Entity(attributes={"name": str(index)}) for index in range(50)],
                ["name"],
            )
            draw = PairRequestModel(uuid="benchmark", force=True)
            workloads: dict[str, tuple[Job, int]] = {
                "no-op": (lambda: list(worker.headers), requests),
                "pair draw": (lambda: worker._get_pair(draw), pairs),
            }
            return {
                name: (count, await best_times(worker, job, count, rounds))
                for name, (job, count) in workloads.items()
            }
        finally:
            await worker.shutdown()
            close_db()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5_000)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    results = asyncio.run(compare(args.requests, args.pairs, args.rounds))
    for workload, (count, best) in results.items():
        legacy, future = best["thread pool"], best["future bridge"]
        print(f"{workload} ({count:,} requests, best of {args.rounds})")
        print(f"  thread pool:    {legacy:.4f}s")
        print(f"  future bridge:  {future:.4f}s")
        print(f"  speedup:        {legacy / future:.2f}x")


if __name__ == "__main__":
    main()
