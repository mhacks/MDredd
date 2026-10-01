import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger(__name__)

_HANDLER_NAME = "mdredd-json"

_FIELDS = (
    "operation",
    "user_id",
    "status",
    "latency_ms",
    "rate_limit_remaining",
    "reason",
    "event",
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


def configure_logging() -> None:
    root = logging.getLogger()
    if not any(handler.name == _HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        found: str | None = None
        for key, value in scope.get("headers", []):
            if key.lower() == b"x-request-id":
                found = value.decode("latin-1")
                break
        rid = found or uuid.uuid4().hex
        token = request_id.set(rid)
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if scope.get("path") != "/health":
                _log_access(scope, started, status_code)
            request_id.reset(token)


def _log_access(scope: Scope, started: float, status_code: int) -> None:
    state = scope.get("state")
    remaining = (
        getattr(state, "rate_limit_remaining", None) if state is not None else None
    )
    extra: dict[str, object] = {
        "operation": f"{scope.get('method', '')} {scope.get('path', '')}",
        "status": status_code,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if isinstance(remaining, int):
        extra["rate_limit_remaining"] = remaining
    logger.info("http request", extra=extra)
