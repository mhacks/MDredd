import logging
from collections.abc import Callable
from typing import TypeVar

import strawberry
from fastapi import HTTPException
from graphql import GraphQLError
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from strawberry.fastapi import BaseContext

from app.auth import AuthError, Principal, authenticate, bearer
from app.exceptions import InvalidColumnsException, JudgingFailure
from app.session import Session

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GraphQLContext(BaseContext):
    def __init__(self, session: Session, principal: Principal) -> None:
        super().__init__()
        self.session = session
        self.principal = principal
        self.rate_limit_remaining: int | None = None


@strawberry.type
class JudgingSession:
    is_started: bool


def graphql_code(code: str, **extra: object) -> GraphQLError:
    return GraphQLError(code, extensions={"code": code, **extra})


async def run_judging(func: Callable[[], T]) -> T:
    try:
        return await run_in_threadpool(func)
    except InvalidColumnsException as exc:
        raise graphql_code(exc.code, names=exc.names) from exc
    except JudgingFailure as exc:
        raise graphql_code(exc.code) from exc


async def get_context(connection: HTTPConnection) -> GraphQLContext:
    try:
        principal = authenticate(await bearer(connection))
    except AuthError as exc:
        logger.warning("Rejected request", extra={"reason": exc.reason})
        raise HTTPException(status_code=401, detail={"code": exc.reason}) from exc
    session = getattr(connection.state, "session", None)
    if not isinstance(session, Session):
        raise TypeError("Judging session is missing")
    return GraphQLContext(session, principal)
