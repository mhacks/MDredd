import logging
import time
from collections.abc import Iterator
from typing import ClassVar, Literal

from graphql import GraphQLError
from strawberry.extensions import SchemaExtension
from strawberry.permission import BasePermission
from strawberry.types import Info

from app.api.types import GraphQLContext, graphql_code
from app.ratelimit import limiter

logger = logging.getLogger(__name__)


def authorize(
    info: Info[GraphQLContext],
    role: Literal["admin", "judge"],
    operation: str | None,
) -> bool:
    context = info.context
    if not isinstance(context, GraphQLContext):
        raise RuntimeError("GraphQL context is missing a principal")
    if context.principal.role != role:
        raise graphql_code("FORBIDDEN")
    if operation is None:
        return True
    decision = limiter.try_consume(context.principal.user_id, operation)
    context.rate_limit_remaining = decision.remaining
    if not decision.allowed:
        raise graphql_code("RATE_LIMITED", retryAfterMs=decision.retry_after_ms)
    return True


class JudgePair(BasePermission):
    role: ClassVar[Literal["admin", "judge"]] = "judge"
    operation: ClassVar[str | None] = "pair"

    def has_permission(self, source: object, info: Info[GraphQLContext], **kwargs: object) -> bool:
        return authorize(info, self.role, self.operation)


class JudgeSubmit(JudgePair):
    operation: ClassVar[str | None] = "submit"


class AdminRead(JudgePair):
    role: ClassVar[Literal["admin", "judge"]] = "admin"
    operation: ClassVar[str | None] = None


class AdminWrite(AdminRead):
    operation: ClassVar[str | None] = "admin"


class AccessLog(SchemaExtension):
    def on_execute(self) -> Iterator[None]:
        started = time.perf_counter()
        status = "ok"
        try:
            yield
            code = _error_code(self.execution_context.result)
            if code is not None:
                status = code
            elif _has_errors(self.execution_context.result):
                status = "error"
        except GraphQLError as exc:
            code = None if exc.extensions is None else exc.extensions.get("code")
            status = code if isinstance(code, str) else "error"
            raise
        finally:
            self._log(started, status)

    def _log(self, started: float, status: str) -> None:
        context = self.execution_context.context
        extra: dict[str, object] = {
            "operation": self.execution_context.operation_name or "",
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if isinstance(context, GraphQLContext):
            extra["user_id"] = context.principal.user_id
            extra["role"] = context.principal.role
            if context.rate_limit_remaining is not None:
                extra["rate_limit_remaining"] = context.rate_limit_remaining
        logger.info("graphql request", extra=extra)


def _has_errors(result: object) -> bool:
    errors = getattr(result, "errors", None)
    return isinstance(errors, list) and len(errors) > 0


def _error_code(result: object) -> str | None:
    errors = getattr(result, "errors", None)
    if not isinstance(errors, list) or not errors:
        return None
    extensions = getattr(errors[0], "extensions", None)
    if not isinstance(extensions, dict):
        return None
    code = extensions.get("code")
    if isinstance(code, str):
        return code
    return None
