import logging
import time
from collections.abc import Iterator
from typing import NoReturn

from graphql import GraphQLError
from graphql.language import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    OperationDefinitionNode,
    SelectionNode,
)
from strawberry.extensions import SchemaExtension

from app.api.types import GraphQLContext
from app.ratelimit import limiter

logger = logging.getLogger(__name__)

ADMIN_FIELDS = frozenset({"startJudging", "stopJudging", "resumeJudging", "crash"})
JUDGE_FIELDS = frozenset(
    {
        "pair",
        "submitComparison",
        "rankings",
        "session",
        "rankingsUpdated",
        "columns",
        "row",
    }
)


class RequestGuard(SchemaExtension):
    def __init__(self) -> None:
        self.remaining: int | None = None
        self.fields: list[str] = []

    def on_execute(self) -> Iterator[None]:
        started = time.perf_counter()
        status = "ok"
        try:
            self._authorize()
            yield
            result = self.execution_context.result
            if result is not None and getattr(result, "errors", None):
                status = "error"
        except GraphQLError as exc:
            code = None if exc.extensions is None else exc.extensions.get("code")
            status = code if isinstance(code, str) else "error"
            raise
        finally:
            self._log_access(started, status)

    def _authorize(self) -> None:
        context = self.execution_context.context
        if not isinstance(context, GraphQLContext):
            self._reject("unauthenticated", "Unauthorized", "UNAUTHENTICATED")
        self.fields = _root_fields(
            self.execution_context.graphql_document,
            self.execution_context.operation_name,
        )
        allowed = ADMIN_FIELDS if context.principal.role == "admin" else JUDGE_FIELDS
        if any(not field.startswith("__") and field not in allowed for field in self.fields):
            self._reject("forbidden", "Forbidden", "FORBIDDEN", context)
        operations = _limited_operations(self.fields)
        if not operations:
            return
        decision = limiter.try_consume_all(context.principal.user_id, operations)
        self.remaining = decision.remaining
        if not decision.allowed:
            self._reject(
                "rate_limited",
                "Rate limit exceeded",
                "RATE_LIMITED",
                context,
                decision.retry_after_ms,
            )

    def _reject(
        self,
        reason: str,
        message: str,
        code: str,
        context: GraphQLContext | None = None,
        retry_after_ms: int | None = None,
    ) -> NoReturn:
        extra: dict[str, object] = {"reason": reason, "status": code}
        if context is not None:
            extra["user_id"] = context.principal.user_id
            extra["role"] = context.principal.role
        logger.warning("Rejected request", extra=extra)
        extensions: dict[str, object] = {"code": code}
        if retry_after_ms is not None:
            extensions["retryAfterMs"] = retry_after_ms
        raise GraphQLError(message, extensions=extensions)

    def _log_access(self, started: float, status: str) -> None:
        context = self.execution_context.context
        extra: dict[str, object] = {
            "operation": ",".join(self.fields) or self.execution_context.operation_name or "",
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if isinstance(context, GraphQLContext):
            extra["user_id"] = context.principal.user_id
            extra["role"] = context.principal.role
        if self.remaining is not None:
            extra["rate_limit_remaining"] = self.remaining
        logger.info("graphql request", extra=extra)


def _limited_operations(fields: list[str]) -> list[str]:
    operations: list[str] = []
    if "pair" in fields:
        operations.append("pair")
    if "submitComparison" in fields:
        operations.append("submit")
    if ADMIN_FIELDS.intersection(fields):
        operations.append("admin")
    return operations


def _root_fields(document: DocumentNode | None, operation_name: str | None) -> list[str]:
    if document is None:
        return []
    definitions = document.definitions
    fragments: dict[str, FragmentDefinitionNode] = {}
    operations: list[OperationDefinitionNode] = []
    for definition in definitions:
        if isinstance(definition, FragmentDefinitionNode):
            fragments[definition.name.value] = definition
        elif isinstance(definition, OperationDefinitionNode):
            operations.append(definition)
    if operation_name is not None:
        operations = [
            operation
            for operation in operations
            if operation.name is not None and operation.name.value == operation_name
        ]
    names: list[str] = []
    for operation in operations:
        names.extend(_selection_fields(operation.selection_set.selections, fragments))
    return names


def _selection_fields(
    selections: tuple[SelectionNode, ...] | list[SelectionNode],
    fragments: dict[str, FragmentDefinitionNode],
) -> list[str]:
    names: list[str] = []
    for selection in selections:
        if isinstance(selection, FieldNode):
            names.append(selection.name.value)
        elif isinstance(selection, FragmentSpreadNode):
            fragment = fragments.get(selection.name.value)
            if fragment is not None:
                names.extend(_selection_fields(fragment.selection_set.selections, fragments))
        elif isinstance(selection, InlineFragmentNode):
            names.extend(_selection_fields(selection.selection_set.selections, fragments))
    return names
