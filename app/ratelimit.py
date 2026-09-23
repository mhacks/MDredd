import math
import threading
import time
from dataclasses import dataclass

from app.settings import settings

Bucket = tuple[float, float]


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

    def try_consume(self, operation: str) -> LimitResult:
        now = time.monotonic()
        capacity, refill = _limits(operation)
        with self._lock:
            tokens, updated = self._buckets.get(operation, (float(capacity), now))
            if refill > 0:
                tokens = min(float(capacity), tokens + (now - updated) * refill)
            tokens = min(tokens, float(capacity))
            if tokens < 1:
                self._buckets[operation] = (tokens, now)
                retry = (
                    60_000
                    if refill <= 0
                    else max(1, math.ceil((1 - tokens) / refill * 1000))
                )
                return LimitResult(
                    allowed=False, retry_after_ms=retry, remaining=math.floor(tokens)
                )
            left = tokens - 1
            self._buckets[operation] = (left, now)
            return LimitResult(
                allowed=True, retry_after_ms=0, remaining=math.floor(left)
            )


def _limits(operation: str) -> tuple[int, float]:
    if operation == "pair":
        return settings.PAIR_CAPACITY, settings.PAIR_REFILL_PER_SECOND
    if operation == "submit":
        return settings.SUBMIT_CAPACITY, settings.SUBMIT_REFILL_PER_SECOND
    return settings.ADMIN_CAPACITY, settings.ADMIN_REFILL_PER_SECOND


limiter = TokenBucket()
