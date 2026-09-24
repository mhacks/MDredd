"""Compare ranking-order computation strategies.

Run with:
    uv run python -m benchmarks.rankings
"""

import argparse
import time
from collections.abc import Callable

import numpy as np

Ranking = Callable[[np.ndarray], list[int]]


def _legacy_rank(alphas: np.ndarray) -> list[int]:
    return sorted(
        range(len(alphas)), key=lambda index: alphas[index], reverse=True
    )


def _numpy_rank(alphas: np.ndarray) -> list[int]:
    return np.argsort(-alphas, kind="stable").tolist()


def _measure(rank: Ranking, alphas: np.ndarray, iterations: int) -> float:
    rank(alphas)
    started = time.perf_counter()
    for _ in range(iterations):
        rank(alphas)
    return (time.perf_counter() - started) * 1_000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[100, 1_000, 10_000, 100_000]
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    generator = np.random.default_rng(0)

    print("entities  Python sort  NumPy stable  speedup")
    for entity_count in args.sizes:
        alphas = generator.random(entity_count, dtype=np.float32)
        legacy_ms = _measure(_legacy_rank, alphas, args.iterations)
        numpy_ms = _measure(_numpy_rank, alphas, args.iterations)
        print(
            f"{entity_count:>8}  {legacy_ms:>9.3f} ms"
            f"  {numpy_ms:>10.3f} ms"
            f"  {legacy_ms / numpy_ms:>6.1f}x"
        )


if __name__ == "__main__":
    main()
