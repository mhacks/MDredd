from typing import Any

import strawberry

from app.api.guard import pair_limit, submit_limit
from app.api.types import GraphQLContext, run_judging
from app.columns import Column, materialize_all
from app.exceptions import IncorrectPairFormatException
from app.models import ComparisonInputModel, PairRequestModel

JUDGE_ID = "api"


def build_judge(
    row_type: type[Any],
    columns: list[Column],
) -> tuple[type[Any], type[Any]]:
    @strawberry.type
    class JudgeQuery:
        @strawberry.field(permission_classes=[pair_limit], graphql_type=list[row_type])
        async def pair(
            self,
            info: strawberry.Info[GraphQLContext],
            force: bool = False,
        ) -> Any:
            worker = info.context.session.worker
            request = PairRequestModel(uuid=JUDGE_ID, force=force)
            left, right = await run_judging(lambda: worker.request_pair(request))
            return materialize_all(columns, row_type, [left, right])

    @strawberry.type
    class JudgeMutation:
        @strawberry.mutation(permission_classes=[submit_limit])
        async def submit_comparison(
            self,
            info: strawberry.Info[GraphQLContext],
            entity_ids: list[int],
            winner_id: int,
        ) -> bool:
            worker = info.context.session.worker

            def submit() -> None:
                if len(entity_ids) != 2:
                    raise IncorrectPairFormatException()
                worker.submit(
                    ComparisonInputModel(
                        uuid=JUDGE_ID,
                        entity_ids=(entity_ids[0], entity_ids[1]),
                        winner_id=winner_id,
                    )
                )

            await run_judging(submit)
            return True

    return JudgeQuery, JudgeMutation
