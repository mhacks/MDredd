from typing import Any

import strawberry
from fastapi import UploadFile
from strawberry.fastapi import GraphQLRouter
from strawberry.file_uploads import UploadDefinition
from strawberry.tools import merge_types

from app.api.admin_router import build_admin
from app.api.dev_router import DevMutation
from app.api.guard import AccessLog
from app.api.judge_router import build_judge
from app.api.types import GraphQLContext
from app.columns import Column, graphql_columns
from app.settings import settings

_router: GraphQLRouter[GraphQLContext, None] | None = None


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
            name=column.field, description=column.header
        )
    return strawberry.type(type("Row", (), namespace))


def schema_for_columns(columns: list[Column]) -> strawberry.Schema:
    row_type = make_row_type(columns)
    admin_query, admin_mutation = build_admin(row_type, columns, rebind_columns)
    judge_query, judge_mutation = build_judge(row_type, columns)
    mutations: tuple[type[Any], ...] = (admin_mutation, judge_mutation)
    if settings.ENABLE_CRASH_ROUTE:
        mutations = (*mutations, DevMutation)
    return strawberry.Schema(
        query=merge_types("Query", (admin_query, judge_query)),
        mutation=merge_types("Mutation", mutations),
        scalar_overrides={UploadFile: UploadDefinition},
        extensions=[AccessLog],
    )


def schema_for_headers(headers: list[str]) -> strawberry.Schema:
    return schema_for_columns(graphql_columns(headers))


def bind_router(router: GraphQLRouter[GraphQLContext, None]) -> None:
    global _router
    _router = router


def rebind_columns(columns: list[Column]) -> None:
    if _router is None:
        raise RuntimeError("GraphQL router is not ready")
    _router.schema = schema_for_columns(columns)


def rebind_schema(headers: list[str]) -> None:
    rebind_columns(graphql_columns(headers))
