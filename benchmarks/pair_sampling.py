"""Compare pair sampling implementations.

Run with:
    uv run python -m benchmarks.pair_sampling
"""

import argparse
import time
from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from jax import jit

from app.algorithm.bayesian_decision_process import _draw_next_pair

Draw = Callable[
    [jnp.ndarray, jnp.ndarray, float],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]


@jit
def _legacy_softmax(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    scaled = logits / temperature
    weights = jnp.exp(scaled - jnp.max(scaled))
    return weights / jnp.sum(weights)


def _legacy_draw(
    frequency: jnp.ndarray, key: jnp.ndarray, temperature: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    entity_count = frequency.shape[0]
    left, right = jnp.triu_indices(entity_count, k=1)
    pair_frequency = frequency[left] + frequency[right]
    distribution = _legacy_softmax(-pair_frequency, temperature)
    next_key, subkey = jr.split(key)
    pair_index = jr.choice(subkey, left.shape[0], p=distribution)
    selected_left = left[pair_index]
    selected_right = right[pair_index]
    frequency = frequency.at[selected_left].add(1).at[selected_right].add(1)
    return frequency, next_key, selected_left, selected_right


def _measure(draw: Draw, entity_count: int, iterations: int) -> tuple[float, float]:
    frequency = jnp.zeros(entity_count, dtype=jnp.int32)
    key = jr.PRNGKey(0)

    started = time.perf_counter()
    frequency, key, left, _ = draw(frequency, key, 1.0)
    frequency.block_until_ready()
    int(left)
    cold_ms = (time.perf_counter() - started) * 1_000

    started = time.perf_counter()
    for _ in range(iterations):
        frequency, key, left, _ = draw(frequency, key, 1.0)
        frequency.block_until_ready()
        int(left)
    warm_ms = (time.perf_counter() - started) * 1_000 / iterations
    return cold_ms, warm_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[100, 500, 1_000, 2_000]
    )
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    print("entities  legacy cold  optimized cold  legacy warm  optimized warm")
    for entity_count in args.sizes:
        legacy_cold, legacy_warm = _measure(
            _legacy_draw, entity_count, args.iterations
        )
        optimized_cold, optimized_warm = _measure(
            _draw_next_pair, entity_count, args.iterations
        )
        print(
            f"{entity_count:>8}  {legacy_cold:>10.2f} ms"
            f"  {optimized_cold:>12.2f} ms"
            f"  {legacy_warm:>10.3f} ms"
            f"  {optimized_warm:>13.3f} ms"
        )


if __name__ == "__main__":
    main()
