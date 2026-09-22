from collections.abc import Callable
from typing import TypeVar

import strawberry
from graphql import GraphQLError
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from strawberry.fastapi import BaseContext

from app.entity import Entity as DomainEntity
from app.exceptions import JudgingFailure
from app.models import EntityWithId
from app.session import Session

T = TypeVar("T")


class GraphQLContext(BaseContext):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session


@strawberry.type
class Entity:
    project_name: str
    devpost_link: str
    table_num: str
    tracks: str


@strawberry.type
class AssignedEntity:
    id: int
    project_name: str
    devpost_link: str
    table_num: str
    tracks: str


@strawberry.type
class JudgingSession:
    is_started: bool


def to_entity(entity: DomainEntity) -> Entity:
    return Entity(
        project_name=entity.project_name,
        devpost_link=entity.devpost_link,
        table_num=entity.table_num,
        tracks=entity.tracks,
    )


def to_assigned(entity: EntityWithId) -> AssignedEntity:
    return AssignedEntity(
        id=entity.id,
        project_name=entity.project_name,
        devpost_link=entity.devpost_link,
        table_num=entity.table_num,
        tracks=entity.tracks,
    )


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
