from collections.abc import Callable
from typing import TypeVar

import strawberry
from graphql import GraphQLError
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from strawberry.fastapi import BaseContext

from app.exceptions import JudgingFailure, UnknownAttributeException
from app.models import EntityWithId
from app.session import Session

T = TypeVar("T")


class GraphQLContext(BaseContext):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session


@strawberry.type
class Attribute:
    name: str
    value: str


@strawberry.type
class Row:
    id: int
    values: strawberry.Private[dict[str, str]]

    @strawberry.field
    def attributes(self, names: list[str]) -> list[Attribute]:
        unknown = [name for name in names if name not in self.values]
        if unknown:
            exc = UnknownAttributeException(unknown)
            raise GraphQLError(str(exc), extensions={"code": exc.code}) from exc
        return [Attribute(name=name, value=self.values[name]) for name in names]


@strawberry.type
class JudgingSession:
    is_started: bool


def to_row(entity: EntityWithId) -> Row:
    return Row(id=entity.id, values=dict(entity.attributes))


async def run_judging(func: Callable[[], T]) -> T:
    try:
        return await run_in_threadpool(func)
    except JudgingFailure as exc:
        raise GraphQLError(str(exc), extensions={"code": exc.code}) from exc


async def get_context(connection: HTTPConnection) -> GraphQLContext:
    session = getattr(connection.state, "session", None)
    if not isinstance(session, Session):
        raise TypeError("Judging session is missing")
    return GraphQLContext(session)
