from collections.abc import AsyncGenerator

import strawberry
from fastapi import UploadFile

from app.api.types import GraphQLContext, graphql_code, run_judging
from app.columns import graphql_columns
from app.models import EntityWithId
from app.ratelimit import subscriptions


def session_started(info: strawberry.Info[GraphQLContext]) -> bool:
    return info.context.session.get_enabled()


def column_pairs(info: strawberry.Info[GraphQLContext]) -> list[tuple[str, str]]:
    columns = graphql_columns(info.context.session.headers())
    return [(column.field, column.header) for column in columns]


def load_row(info: strawberry.Info[GraphQLContext], row_id: int) -> EntityWithId:
    return info.context.session.get_row(row_id)


async def load_rankings(info: strawberry.Info[GraphQLContext]) -> list[EntityWithId]:
    session = info.context.session
    return await run_judging(session.get_rankings)


async def start_judging(
    info: strawberry.Info[GraphQLContext],
    entities_csv: UploadFile | None,
) -> bool:
    session = info.context.session
    return await run_judging(lambda: session.start(entities_csv))


async def stop_judging(info: strawberry.Info[GraphQLContext]) -> bool:
    session = info.context.session
    await run_judging(session.stop)
    return session.get_enabled()


async def resume_judging(info: strawberry.Info[GraphQLContext]) -> bool:
    session = info.context.session
    await run_judging(session.resume)
    return session.get_enabled()


async def iter_rankings(
    info: strawberry.Info[GraphQLContext],
) -> AsyncGenerator[list[EntityWithId] | None]:
    session = info.context.session
    user_id = info.context.principal.user_id
    if not subscriptions.try_acquire(user_id):
        raise graphql_code("SUBSCRIPTION_LIMIT")
    try:
        subscriber = session.worker.subscribe_rankings()
        try:
            while True:
                yield await subscriber.get()
        finally:
            session.worker.unsubscribe_rankings(subscriber)
    finally:
        subscriptions.release(user_id)
