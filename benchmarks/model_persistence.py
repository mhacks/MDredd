"""Compare full JSON snapshots with incremental pair persistence.

Run with:
    uv run python -m benchmarks.model_persistence
"""

import argparse
import time
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import jax.random as jr

from app.algorithm import BayesianDecisionProcess
from app.db import (
    Assignment,
    Judge,
    close_db,
    db,
    open_db,
    replace_state,
    save_assignment,
)
from app.entity import Entity


def _advance(model: BayesianDecisionProcess) -> None:
    model.frequency = model.frequency.at[0].add(1).at[1].add(1)
    model.key, _ = jr.split(model.key)


def _save_legacy(model: BayesianDecisionProcess, judge_id: str) -> None:
    with db.atomic():
        Judge.update(bdp=model.snapshot()).where(Judge.id == 1).execute()
        Assignment.replace(
            judge_id=judge_id, entity_id_1=0, entity_id_2=1
        ).execute()


def _measure(entity_count: int, iterations: int) -> tuple[float, float]:
    entities = [Entity(attributes={"name": str(index)}) for index in range(entity_count)]
    model = BayesianDecisionProcess(
        K=entity_count,
        alpha_t=jnp.linspace(1.0, 2.0, entity_count, dtype=jnp.float32),
        frequency=jnp.zeros(entity_count, dtype=jnp.int32),
        key=jr.PRNGKey(0),
    )
    replace_state(["name"], entities, model)

    _advance(model)
    _save_legacy(model, "legacy-warmup")
    _advance(model)
    save_assignment(model, "optimized-warmup", (0, 1))

    legacy_started = time.perf_counter()
    for iteration in range(iterations):
        _advance(model)
        _save_legacy(model, f"legacy-{iteration}")
    legacy_ms = (time.perf_counter() - legacy_started) * 1_000 / iterations

    optimized_started = time.perf_counter()
    for iteration in range(iterations):
        _advance(model)
        save_assignment(model, f"optimized-{iteration}", (0, 1))
    optimized_ms = (time.perf_counter() - optimized_started) * 1_000 / iterations
    return legacy_ms, optimized_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1_000, 10_000])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    print("entities  legacy JSON  incremental  speedup")
    with TemporaryDirectory(prefix="mdredd-benchmark-") as directory:
        try:
            for entity_count in args.sizes:
                close_db()
                db.init(
                    f"{directory}/{entity_count}.db", pragmas={"journal_mode": "wal"}
                )
                open_db()
                legacy_ms, optimized_ms = _measure(entity_count, args.iterations)
                print(
                    f"{entity_count:>8}  {legacy_ms:>9.3f} ms"
                    f"  {optimized_ms:>9.3f} ms"
                    f"  {legacy_ms / optimized_ms:>6.1f}x"
                )
        finally:
            close_db()


if __name__ == "__main__":
    main()
