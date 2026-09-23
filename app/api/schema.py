import types
from collections.abc import AsyncGenerator
from typing import Any

import strawberry
from fastapi import UploadFile
from strawberry.fastapi import GraphQLRouter
from strawberry.file_uploads import UploadDefinition
from strawberry.tools import merge_types

from app.api.admin_router import (
    column_pairs,
    iter_rankings,
    load_rankings,
    load_row,
    resume_judging,
    session_started,
    start_judging,
    stop_judging,
)
from app.api.dev_router import DevMutation
from app.api.guard import AccessLog, AdminRead, AdminWrite, JudgePair, JudgeSubmit
from app.api.judge_router import load_pair, submit_comparison
from app.api.types import GraphQLContext, JudgingSession
from app.columns import Column, graphql_columns
from app.models import EntityWithId
from app.settings import settings

_router: GraphQLRouter[GraphQLContext, None] | None = None


@strawberry.type(name="Column")
class ColumnRecord:
    field: str
    header: str


def make_row_type(columns: list[Column]) -> type[Any]:
    annotations: dict[str, Any] = {"id": int}
    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "__module__": __name__,
        "id": strawberry.field(name="id"),
    }
    for column in columns:
        annotations[column.attr] = str
        namespace[column.attr] = strawberry.field(
            name=column.field,
            description=column.header,
        )
    row = type("Row", (), namespace)
    return strawberry.type(row)


def row_list(row_type: type[Any]) -> Any:
    return types.GenericAlias(list, (row_type,))


def materialize(columns: list[Column], row_type: type[Any], entity: EntityWithId) -> Any:
    payload: dict[str, object] = {"id": entity.id}
    for column in columns:
        payload[column.attr] = entity.attributes.get(column.header, "")
    return row_type(**payload)


def schema_for_headers(headers: list[str]) -> strawberry.Schema:
    columns = graphql_columns(headers)
    return _assemble(make_row_type(columns), columns)


def bind_router(router: GraphQLRouter[GraphQLContext, None]) -> None:
    global _router
    _router = router


def rebind_schema(headers: list[str]) -> None:
    if _router is None:
        raise RuntimeError("GraphQL router is not ready")
    _router.schema = schema_for_headers(headers)


def _assemble(row_type: type[Any], columns: list[Column]) -> strawberry.Schema:
    @strawberry.type
    class Query:
        @strawberry.field(permission_classes=[AdminRead])
        def session(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
            return JudgingSession(is_started=session_started(info))

        @strawberry.field(permission_classes=[AdminRead])
        def columns(self, info: strawberry.Info[GraphQLContext]) -> list[ColumnRecord]:
            return [
                ColumnRecord(field=field, header=header) for field, header in column_pairs(info)
            ]

        @strawberry.field(permission_classes=[AdminRead], graphql_type=row_type)
        async def row(self, info: strawberry.Info[GraphQLContext], id: int) -> Any:
            return materialize(columns, row_type, load_row(info, id))

        @strawberry.field(permission_classes=[AdminRead], graphql_type=row_list(row_type))
        async def rankings(self, info: strawberry.Info[GraphQLContext]) -> Any:
            rankings = await load_rankings(info)
            return [materialize(columns, row_type, entity) for entity in rankings]

        @strawberry.field(permission_classes=[JudgePair], graphql_type=row_list(row_type))
        async def pair(
            self,
            info: strawberry.Info[GraphQLContext],
            force: bool = False,
        ) -> Any:
            pair = await load_pair(info, force)
            return [materialize(columns, row_type, entity) for entity in pair]

    @strawberry.type
    class Mutation:
        @strawberry.mutation(permission_classes=[AdminWrite])
        async def start_judging(
            self,
            info: strawberry.Info[GraphQLContext],
            entities_csv: UploadFile | None = None,
        ) -> JudgingSession:
            changed = await start_judging(info, entities_csv)
            if changed:
                rebind_schema(info.context.session.headers())
            return JudgingSession(is_started=session_started(info))

        @strawberry.mutation(permission_classes=[AdminWrite])
        async def stop_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
            return JudgingSession(is_started=await stop_judging(info))

        @strawberry.mutation(permission_classes=[AdminWrite])
        async def resume_judging(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
            return JudgingSession(is_started=await resume_judging(info))

        @strawberry.mutation(permission_classes=[JudgeSubmit])
        async def submit_comparison(
            self,
            info: strawberry.Info[GraphQLContext],
            entity_ids: list[int],
            winner_id: int,
        ) -> bool:
            return await submit_comparison(info, entity_ids, winner_id)

    @strawberry.type
    class Subscription:
        @strawberry.subscription(
            permission_classes=[AdminRead],
            graphql_type=row_list(row_type),
        )
        async def rankings_updated(
            self,
            info: strawberry.Info[GraphQLContext],
        ) -> AsyncGenerator[Any]:
            async for rankings in iter_rankings(info):
                if rankings is None:
                    return
                yield [materialize(columns, row_type, entity) for entity in rankings]

    mutation = Mutation
    if settings.ENABLE_CRASH_ROUTE:
        mutation = merge_types("Mutation", (Mutation, DevMutation))
    return strawberry.Schema(
        query=Query,
        mutation=mutation,
        subscription=Subscription,
        scalar_overrides={UploadFile: UploadDefinition},
        extensions=[AccessLog],
    )
