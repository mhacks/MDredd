import strawberry

from app.api.types import GraphQLContext, run_judging
from app.exceptions import IncorrectPairFormatException
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel


async def load_pair(info: strawberry.Info[GraphQLContext], force: bool) -> list[EntityWithId]:
    session = info.context.session
    request = PairRequestModel(uuid=info.context.principal.user_id, force=force)
    left, right = await run_judging(lambda: session.get_pair(request))
    return [left, right]


async def submit_comparison(
    info: strawberry.Info[GraphQLContext],
    entity_ids: list[int],
    winner_id: int,
) -> bool:
    session = info.context.session
    user_id = info.context.principal.user_id

    def submit() -> None:
        if len(entity_ids) != 2:
            raise IncorrectPairFormatException()
        session.submit_pair(
            ComparisonInputModel(
                uuid=user_id,
                entity_ids=(entity_ids[0], entity_ids[1]),
                winner_id=winner_id,
            )
        )

    await run_judging(submit)
    return True
