import strawberry
from fastapi import UploadFile
from strawberry.fastapi import GraphQLRouter
from strawberry.file_uploads import UploadDefinition
from strawberry.tools import merge_types

from app.settings import settings

from .admin_router import AdminMutation, AdminQuery, AdminSubscription
from .dev_router import DevMutation
from .guard import RequestGuard
from .judge_router import JudgeMutation, JudgeQuery
from .types import GraphQLContext, get_context


def build_schema() -> strawberry.Schema:
    mutations = [AdminMutation, JudgeMutation]
    if settings.ENABLE_CRASH_ROUTE:
        mutations.append(DevMutation)
    return strawberry.Schema(
        query=merge_types("Query", (AdminQuery, JudgeQuery)),
        mutation=merge_types("Mutation", tuple(mutations)),
        subscription=merge_types("Subscription", (AdminSubscription,)),
        scalar_overrides={UploadFile: UploadDefinition},
        extensions=[RequestGuard],
    )


def graphql_router() -> GraphQLRouter[GraphQLContext, None]:
    return GraphQLRouter(
        build_schema(),
        path="/",
        context_getter=get_context,
        multipart_uploads_enabled=True,
    )


__all__ = ["build_schema", "get_context", "graphql_router"]
