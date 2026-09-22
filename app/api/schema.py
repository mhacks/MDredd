import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import TypeVar

import strawberry
from fastapi import UploadFile
from graphql import GraphQLError
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from strawberry.fastapi import BaseContext
from strawberry.file_uploads import UploadDefinition

from app.entity import Entity as DomainEntity
from app.exceptions import IncorrectPairFormatException, JudgingFailure
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.session import Session
from app.settings import settings

logger = logging.getLogger(__name__)

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


def _entity(entity: DomainEntity) -> Entity:
    return Entity(
        project_name=entity.project_name,
        devpost_link=entity.devpost_link,
        table_num=entity.table_num,
        tracks=entity.tracks,
    )


def _assigned(entity: EntityWithId) -> AssignedEntity:
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


@strawberry.type
class Query:
    @strawberry.field
    def session(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        return JudgingSession(is_started=info.context.session.get_enabled())

    @strawberry.field
    async def rankings(self, info: strawberry.Info[GraphQLContext]) -> list[Entity]:
        session = info.context.session
        rankings = await run_judging(session.get_rankings)
        return [_entity(entity) for entity in rankings]

    @strawberry.field
    async def pair(
        self,
        info: strawberry.Info[GraphQLContext],
        judge_id: str,
        force: bool = False,
    ) -> list[AssignedEntity]:
        logger.info("Got request for pair by %s (force=%s).", judge_id, force)
        session = info.context.session
        request = PairRequestModel(uuid=judge_id, force=force)
        left, right = await run_judging(lambda: session.get_pair(request))
        return [_assigned(left), _assigned(right)]


@strawberry.type
class _JudgingMutation:
    @strawberry.mutation
    async def start_judging(
        self,
        info: strawberry.Info[GraphQLContext],
        entities_csv: UploadFile | None = None,
    ) -> JudgingSession:
        logger.info("Got request to start judging.")
        session = info.context.session
        await run_judging(lambda: session.start(entities_csv))
        return JudgingSession(is_started=session.get_enabled())

    @strawberry.mutation
    async def stop_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        logger.info("Got request to stop judging.")
        session = info.context.session
        await run_judging(session.stop)
        return JudgingSession(is_started=session.get_enabled())

    @strawberry.mutation
    async def resume_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        logger.info("Got request to resume judging.")
        session = info.context.session
        await run_judging(session.resume)
        return JudgingSession(is_started=session.get_enabled())

    @strawberry.mutation
    async def submit_comparison(
        self,
        info: strawberry.Info[GraphQLContext],
        judge_id: str,
        entity_ids: list[int],
        winner_id: int,
    ) -> bool:
        session = info.context.session

        def submit() -> None:
            if len(entity_ids) != 2:
                raise IncorrectPairFormatException()
            session.submit_pair(
                ComparisonInputModel(
                    uuid=judge_id,
                    entity_ids=(entity_ids[0], entity_ids[1]),
                    winner_id=winner_id,
                )
            )

        await run_judging(submit)
        return True


@strawberry.type(name="Mutation")
class Mutation(_JudgingMutation):
    pass


@strawberry.type(name="Mutation")
class CrashMutation(_JudgingMutation):
    @strawberry.mutation
    def crash(self) -> bool:
        logger.warning("Dev crash mutation invoked")
        os._exit(1)


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def rankings_updated(
        self, info: strawberry.Info[GraphQLContext]
    ) -> AsyncGenerator[list[Entity]]:
        session = info.context.session
        subscriber = session.worker.subscribe_rankings()
        try:
            while True:
                rankings = await subscriber.get()
                yield [_entity(entity) for entity in rankings]
        finally:
            session.worker.unsubscribe_rankings(subscriber)


async def get_context(connection: HTTPConnection) -> GraphQLContext:
    session = getattr(connection.state, "session", None)
    if not isinstance(session, Session):
        raise TypeError("Judging session is missing")
    return GraphQLContext(session)


def build_schema() -> strawberry.Schema:
    mutation = CrashMutation if settings.ENABLE_CRASH_ROUTE else Mutation
    return strawberry.Schema(
        query=Query,
        mutation=mutation,
        subscription=Subscription,
        scalar_overrides={UploadFile: UploadDefinition},
    )
