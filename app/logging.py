import json
import logging
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_FIELDS = (
    "operation",
    "user_id",
    "role",
    "status",
    "latency_ms",
    "rate_limit_remaining",
    "reason",
    "event",
    "replayed",
    "entity_ids",
    "winner_id",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        current_request = request_id.get()
        if current_request:
            payload["request_id"] = current_request
        for field in _FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _JsonHandler(logging.StreamHandler[Any]):
    pass


def configure_logging() -> None:
    root = logging.getLogger()
    if not any(isinstance(handler, _JsonHandler) for handler in root.handlers):
        handler = _JsonHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


class RequestIdMiddleware:
    def __init__(self, app: Callable[[MutableMapping[str, Any], Any, Any], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        header_name = b"x-request-id"
        found: str | None = None
        for key, value in scope.get("headers", []):
            if key.lower() == header_name:
                found = value.decode("latin-1")
                break
        rid = found or uuid.uuid4().hex
        token = request_id.set(rid)

        async def send_with_id(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id.reset(token)
