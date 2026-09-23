from typing import Any

import strawberry
from fastapi import UploadFile

from app.api.guard import admin_limit
from app.api.types import GraphQLContext, JudgingSession, run_judging
from app.columns import Column, graphql_columns
from app.models import EntityWithId


@strawberry.type(name="Column")
class ColumnRecord:
    field: str
    header: str


def build_admin(
    row_type: type[Any],
    columns: list[Column],
) -> tuple[type[Any], type[Any]]:
    from app.api.schema import materialize, row_list

    listed = row_list(row_type)

    def as_rows(entities: list[EntityWithId]) -> list[Any]:
        return [materialize(columns, row_type, entity) for entity in entities]

    @strawberry.type
    class AdminQuery:
        @strawberry.field
        def session(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
            return JudgingSession(is_started=info.context.session.worker.get_enabled())

        @strawberry.field
        def columns(self, info: strawberry.Info[GraphQLContext]) -> list[ColumnRecord]:
            return [
                ColumnRecord(field=column.field, header=column.header)
                for column in graphql_columns(info.context.session.worker.get_headers())
            ]

        @strawberry.field(graphql_type=row_type)
        def row(self, info: strawberry.Info[GraphQLContext], id: int) -> Any:
            return materialize(
                columns, row_type, info.context.session.worker.get_row(id)
            )

        @strawberry.field(graphql_type=listed)
        async def rankings(self, info: strawberry.Info[GraphQLContext]) -> Any:
            worker = info.context.session.worker
            return as_rows(await run_judging(worker.rankings))

    @strawberry.type
    class AdminMutation:
        @strawberry.mutation(permission_classes=[admin_limit])
        async def start_judging(
            self,
            info: strawberry.Info[GraphQLContext],
            entities_csv: UploadFile | None = None,
        ) -> JudgingSession:
            from app.api.schema import rebind_schema

            session = info.context.session
            csv_bytes = None if entities_csv is None else await entities_csv.read()
            changed = await run_judging(lambda: session.start(csv_bytes))
            if changed:
                rebind_schema(session.worker.get_headers())
            return JudgingSession(is_started=session.worker.get_enabled())

        @strawberry.mutation(permission_classes=[admin_limit])
        async def stop_judging(
            self, info: strawberry.Info[GraphQLContext]
        ) -> JudgingSession:
            worker = info.context.session.worker
            await run_judging(worker.stop)
            return JudgingSession(is_started=worker.get_enabled())

        @strawberry.mutation(permission_classes=[admin_limit])
        async def resume_judging(
            self, info: strawberry.Info[GraphQLContext]
        ) -> JudgingSession:
            worker = info.context.session.worker
            await run_judging(worker.resume)
            return JudgingSession(is_started=worker.get_enabled())

    return AdminQuery, AdminMutation
