import logging

import strawberry

from app.api.types import GraphQLContext, Row, run_judging, to_row
from app.exceptions import IncorrectPairFormatException
from app.models import ComparisonInputModel, PairRequestModel

logger = logging.getLogger(__name__)


@strawberry.type
class JudgeQuery:
    @strawberry.field
    async def pair(
        self,
        info: strawberry.Info[GraphQLContext],
        judge_id: str,
        force: bool = False,
    ) -> list[Row]:
        logger.info("Got request for pair by %s (force=%s).", judge_id, force)
        session = info.context.session
        request = PairRequestModel(uuid=judge_id, force=force)
        left, right = await run_judging(lambda: session.get_pair(request))
        return [to_row(left), to_row(right)]


@strawberry.type
class JudgeMutation:
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
