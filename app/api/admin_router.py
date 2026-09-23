from collections.abc import Callable
from typing import Any

import strawberry
from fastapi import UploadFile

from app.api.guard import admin_limit
from app.api.types import GraphQLContext, JudgingSession, run_judging
from app.columns import Column, materialize, materialize_all


@strawberry.type(name="Column")
class ColumnRecord:
    field: str
    header: str


def build_admin(
    row_type: type[Any],
    columns: list[Column],
    rebind: Callable[[list[Column]], None],
) -> tuple[type[Any], type[Any]]:
    @strawberry.type
    class AdminQuery:
        @strawberry.field
        async def session(self, info: strawberry.Info[GraphQLContext]) -> JudgingSession:
            worker = info.context.session.worker
            return JudgingSession(is_started=await run_judging(worker.get_enabled))

        @strawberry.field
        def columns(self) -> list[ColumnRecord]:
            return [
                ColumnRecord(field=column.field, header=column.header)
                for column in columns
            ]

        @strawberry.field(graphql_type=row_type)
        async def row(self, info: strawberry.Info[GraphQLContext], id: int) -> Any:
            worker = info.context.session.worker
            return materialize(
                columns, row_type, await run_judging(lambda: worker.get_row(id))
            )

        @strawberry.field(graphql_type=list[row_type])
        async def rankings(self, info: strawberry.Info[GraphQLContext]) -> Any:
            worker = info.context.session.worker
            return materialize_all(
                columns, row_type, await run_judging(worker.rankings)
            )

    @strawberry.type
    class AdminMutation:
        @strawberry.mutation(permission_classes=[admin_limit])
        async def start_judging(
            self,
            info: strawberry.Info[GraphQLContext],
            entities_csv: UploadFile | None = None,
        ) -> JudgingSession:
            session = info.context.session
            if entities_csv is None:
                await run_judging(session.worker.resume)
            else:
                csv_bytes = await entities_csv.read()
                rebind(await run_judging(lambda: session.start(csv_bytes)))
            return JudgingSession(
                is_started=await run_judging(session.worker.get_enabled)
            )

        @strawberry.mutation(permission_classes=[admin_limit])
        async def stop_judging(
            self, info: strawberry.Info[GraphQLContext]
        ) -> JudgingSession:
            worker = info.context.session.worker
            await run_judging(worker.stop)
            return JudgingSession(is_started=await run_judging(worker.get_enabled))

        @strawberry.mutation(permission_classes=[admin_limit])
        async def resume_judging(
            self, info: strawberry.Info[GraphQLContext]
        ) -> JudgingSession:
            worker = info.context.session.worker
            await run_judging(worker.resume)
            return JudgingSession(is_started=await run_judging(worker.get_enabled))

    return AdminQuery, AdminMutation
