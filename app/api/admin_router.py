from collections.abc import AsyncGenerator

import strawberry
from fastapi import UploadFile
from graphql import GraphQLError

from app.api.types import GraphQLContext, JudgingSession, Row, run_judging, to_row
from app.ratelimit import limiter, subscriptions


@strawberry.type
class AdminQuery:
    @strawberry.field
    def session(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        return JudgingSession(is_started=info.context.session.get_enabled())

    @strawberry.field
    def columns(self, info: strawberry.Info[GraphQLContext]) -> list[str]:
        return info.context.session.columns()

    @strawberry.field
    async def row(self, info: strawberry.Info[GraphQLContext], id: int) -> Row:
        session = info.context.session
        entity = await run_judging(lambda: session.get_row(id))
        return to_row(entity)

    @strawberry.field
    async def rankings(self, info: strawberry.Info[GraphQLContext]) -> list[Row]:
        session = info.context.session
        rankings = await run_judging(session.get_rankings)
        return [to_row(entity) for entity in rankings]


@strawberry.type
class AdminMutation:
    @strawberry.mutation
    async def start_judging(
        self,
        info: strawberry.Info[GraphQLContext],
        entities_csv: UploadFile | None = None,
    ) -> JudgingSession:
        session = info.context.session
        await run_judging(lambda: session.start(entities_csv))
        return JudgingSession(is_started=session.get_enabled())

    @strawberry.mutation
    async def stop_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        session = info.context.session
        await run_judging(session.stop)
        return JudgingSession(is_started=session.get_enabled())

    @strawberry.mutation
    async def resume_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
        session = info.context.session
        await run_judging(session.resume)
        return JudgingSession(is_started=session.get_enabled())


@strawberry.type
class AdminSubscription:
    @strawberry.subscription
    async def rankings_updated(
        self, info: strawberry.Info[GraphQLContext]
    ) -> AsyncGenerator[list[Row]]:
        session = info.context.session
        user_id = info.context.principal.user_id
        if not subscriptions.try_acquire(user_id):
            decision = limiter.try_consume(user_id, "pair")
            if not decision.allowed:
                raise GraphQLError(
                    "Rate limit exceeded",
                    extensions={"code": "RATE_LIMITED", "retryAfterMs": decision.retry_after_ms},
                )
            raise GraphQLError(
                "One rankings subscription is already active",
                extensions={"code": "SUBSCRIPTION_LIMIT"},
            )
        try:
            subscriber = session.worker.subscribe_rankings()
            try:
                while True:
                    rankings = await subscriber.get()
                    yield [to_row(entity) for entity in rankings]
            finally:
                session.worker.unsubscribe_rankings(subscriber)
        finally:
            subscriptions.release(user_id)
