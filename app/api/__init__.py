from starlette.routing import WebSocketRoute
from strawberry.fastapi import GraphQLRouter

from app.api.schema import bind_router, rebind_schema, schema_for_headers
from app.api.types import GraphQLContext, get_context


def graphql_router() -> GraphQLRouter[GraphQLContext, None]:
    router = GraphQLRouter(
        schema_for_headers([]),
        path="/",
        context_getter=get_context,
        multipart_uploads_enabled=True,
    )
    router.routes[:] = [
        route for route in router.routes if not isinstance(route, WebSocketRoute)
    ]
    bind_router(router)
    return router


__all__ = ["get_context", "graphql_router", "rebind_schema", "schema_for_headers"]
