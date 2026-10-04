import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

from app.settings import settings

Bucket = tuple[float, float]
Operation = Literal["admin", "pair", "submit", "tables"]


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    retry_after_ms: int
    remaining: int


class TokenBucket:
    def __init__(self) -> None:
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def try_consume(self, operation: Operation, key: str = "") -> LimitResult:
        """Take a token from the bucket for this operation and key.

        Judge operations pass the judge id, so each judge has their own
        bucket. Admin operations share one.
        """
        now = time.monotonic()
        capacity, refill = _limits(operation)
        bucket = f"{operation}:{key}"
        with self._lock:
            tokens, updated = self._buckets.get(bucket, (float(capacity), now))
            if refill > 0:
                tokens = min(float(capacity), tokens + (now - updated) * refill)
            tokens = min(tokens, float(capacity))
            if tokens < 1:
                self._buckets[bucket] = (tokens, now)
                retry = (
                    60_000
                    if refill <= 0
                    else max(1, math.ceil((1 - tokens) / refill * 1000))
                )
                return LimitResult(
                    allowed=False, retry_after_ms=retry, remaining=math.floor(tokens)
                )
            left = tokens - 1
            self._buckets[bucket] = (left, now)
            return LimitResult(
                allowed=True, retry_after_ms=0, remaining=math.floor(left)
            )


def _limits(operation: Operation) -> tuple[int, float]:
    match operation:
        case "pair":
            return settings.PAIR_CAPACITY, settings.PAIR_REFILL_PER_SECOND
        case "submit":
            return settings.SUBMIT_CAPACITY, settings.SUBMIT_REFILL_PER_SECOND
        case "admin":
            return settings.ADMIN_CAPACITY, settings.ADMIN_REFILL_PER_SECOND
        case "tables":
            return settings.TABLES_CAPACITY, settings.TABLES_REFILL_PER_SECOND


limiter = TokenBucket()
