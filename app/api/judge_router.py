from fastapi import APIRouter, Depends, status

from app.api.deps import SessionDep, limited, require_session
from app.api.errors import error_responses
from app.models import (
    ComparisonInputModel,
    ComparisonResultModel,
    PairModel,
    PairRequestModel,
    RowModel,
)

router = APIRouter(
    tags=["judge"],
    dependencies=[Depends(require_session)],
    responses=error_responses(status.HTTP_503_SERVICE_UNAVAILABLE),
)


@router.post(
    "/pairs",
    response_model=PairModel,
    description=(
        "Draw this judge's open pair, or return the pair they already hold. "
        "One absent project from that pair forfeits, and the other project wins. "
        "Both absent strikes each project and draws a new pair."
    ),
    response_description="The two rows to compare.",
    dependencies=[limited("pair")],
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def create_pair(body: PairRequestModel, session: SessionDep) -> PairModel:
    left, right = session.worker.request_pair(body)
    return PairModel(
        pair=(RowModel.from_entity(left), RowModel.from_entity(right))
    )


@router.post(
    "/comparisons",
    response_model=ComparisonResultModel,
    description="Record the winner of this judge's open pair.",
    response_description="The comparison was recorded.",
    dependencies=[limited("submit")],
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def submit_comparison(
    body: ComparisonInputModel, session: SessionDep
) -> ComparisonResultModel:
    session.worker.submit(body)
    return ComparisonResultModel(ok=True)
