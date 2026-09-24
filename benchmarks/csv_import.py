"""Compare CSV-to-entity conversion strategies.

Run with:
    uv run python -m benchmarks.csv_import
"""

import argparse
import io
import time
from collections.abc import Callable

import pandas as pd

from app.entity import Entity

Parser = Callable[[bytes], tuple[list[str], list[Entity]]]


def _legacy_parse(raw_csv: bytes) -> tuple[list[str], list[Entity]]:
    frame = pd.read_csv(io.BytesIO(raw_csv), dtype=str, keep_default_na=False)
    columns = [str(column) for column in frame.columns]
    entities = [
        Entity(attributes={column: str(row[column]) for column in columns})
        for _, row in frame.iterrows()
    ]
    return columns, entities


def _csv_bytes(row_count: int, column_count: int = 10) -> bytes:
    headers = ",".join(f"column-{index}" for index in range(column_count))
    row = ",".join(f"value-{index}" for index in range(column_count))
    return f"{headers}\n".encode() + (f"{row}\n" * row_count).encode()


def _measure(parser: Parser, raw_csv: bytes, iterations: int) -> float:
    parser(raw_csv)
    started = time.perf_counter()
    for _ in range(iterations):
        parser(raw_csv)
    return (time.perf_counter() - started) * 1_000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1_000, 10_000, 50_000]
    )
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    print("rows      iterrows    itertuples  speedup")
    for row_count in args.rows:
        raw_csv = _csv_bytes(row_count)
        legacy_ms = _measure(_legacy_parse, raw_csv, args.iterations)
        optimized_ms = _measure(Entity.list_from_csv, raw_csv, args.iterations)
        print(
            f"{row_count:>8}  {legacy_ms:>9.2f} ms"
            f"  {optimized_ms:>9.2f} ms"
            f"  {legacy_ms / optimized_ms:>6.1f}x"
        )


if __name__ == "__main__":
    main()
