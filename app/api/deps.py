import logging
import math
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import AuthError, authenticate, bearer
from app.ratelimit import limiter
from app.session import Session

logger = logging.getLogger(__name__)


async def require_session(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Session:
    try:
        authenticate(credentials)
    except AuthError as exc:
        logger.warning("Rejected request", extra={"reason": exc.reason})
        raise HTTPException(status_code=401, detail={"code": exc.reason}) from exc
    session = getattr(request.state, "session", None)
    if not isinstance(session, Session):
        raise TypeError("Judging session is missing")
    return session


SessionDep = Annotated[Session, Depends(require_session)]


def limited(operation: str) -> Depends:
    def consume(request: Request, _session: SessionDep) -> None:
        decision = limiter.try_consume(operation)
        request.state.rate_limit_remaining = decision.remaining
        if decision.allowed:
            return
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMITED",
                "retry_after_ms": decision.retry_after_ms,
            },
            headers={
                "Retry-After": str(max(1, math.ceil(decision.retry_after_ms / 1000)))
            },
        )

    return Depends(consume)
