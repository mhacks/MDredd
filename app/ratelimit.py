import math
import threading
import time
from dataclasses import dataclass

from app.auth import key_for_user
from app.settings import settings

Bucket = tuple[float, float]


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    retry_after_ms: int
    remaining: int


class TokenBucket:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], Bucket] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def try_consume(self, user_id: str, operation: str) -> LimitResult:
        return self.try_consume_all(user_id, [operation])

    def try_consume_all(self, user_id: str, operations: list[str]) -> LimitResult:
        if not operations:
            return LimitResult(allowed=True, retry_after_ms=0, remaining=0)
        now = time.monotonic()
        with self._lock:
            states = {
                operation: self._refill(user_id, operation, now) for operation in operations
            }
            blocked = next(
                (operation for operation, tokens in states.items() if tokens < 1),
                None,
            )
            if blocked is not None:
                for operation, tokens in states.items():
                    self._buckets[(user_id, operation)] = (tokens, now)
                return LimitResult(
                    allowed=False,
                    retry_after_ms=self._retry_after(user_id, blocked, states[blocked]),
                    remaining=math.floor(states[blocked]),
                )
            remaining: int | None = None
            for operation, tokens in states.items():
                left = tokens - 1
                self._buckets[(user_id, operation)] = (left, now)
                left_count = math.floor(left)
                remaining = left_count if remaining is None else min(remaining, left_count)
            return LimitResult(
                allowed=True,
                retry_after_ms=0,
                remaining=0 if remaining is None else remaining,
            )

    def _refill(self, user_id: str, operation: str, now: float) -> float:
        capacity, refill = _limits(user_id, operation)
        tokens, updated = self._buckets.get((user_id, operation), (float(capacity), now))
        if refill > 0:
            tokens = min(float(capacity), tokens + (now - updated) * refill)
        return min(tokens, float(capacity))

    def _retry_after(self, user_id: str, operation: str, tokens: float) -> int:
        _, refill = _limits(user_id, operation)
        if refill <= 0:
            return 60_000
        return max(1, math.ceil((1 - tokens) / refill * 1000))


class SubscriptionRegistry:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()

    def try_acquire(self, user_id: str) -> bool:
        with self._lock:
            if self._counts.get(user_id, 0) >= 1:
                return False
            self._counts[user_id] = 1
            return True

    def release(self, user_id: str) -> None:
        with self._lock:
            count = self._counts.get(user_id, 0) - 1
            if count <= 0:
                self._counts.pop(user_id, None)
            else:
                self._counts[user_id] = count


def _limits(user_id: str, operation: str) -> tuple[int, float]:
    record = key_for_user(user_id)
    if operation == "pair":
        return (
            _value(record.pair_capacity, settings.PAIR_CAPACITY),
            _value(record.pair_refill_per_second, settings.PAIR_REFILL_PER_SECOND),
        )
    if operation == "submit":
        return (
            _value(record.submit_capacity, settings.SUBMIT_CAPACITY),
            _value(record.submit_refill_per_second, settings.SUBMIT_REFILL_PER_SECOND),
        )
    return (
        _value(record.admin_capacity, settings.ADMIN_CAPACITY),
        _value(record.admin_refill_per_second, settings.ADMIN_REFILL_PER_SECOND),
    )


def _value[T](override: T | None, default: T) -> T:
    return default if override is None else override


limiter = TokenBucket()
subscriptions = SubscriptionRegistry()
