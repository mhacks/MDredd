from fastapi import APIRouter

from app.api.deps import SessionDep, limited
from app.models import (
    ComparisonInputModel,
    ComparisonResultModel,
    PairModel,
    PairRequestModel,
    RowModel,
)

router = APIRouter(tags=["judge"])


@router.post("/pairs", response_model=PairModel, dependencies=[limited("pair")])
def create_pair(body: PairRequestModel, session: SessionDep) -> PairModel:
    left, right = session.worker.request_pair(body)
    return PairModel(
        pair=(RowModel.from_entity(left), RowModel.from_entity(right))
    )


@router.post(
    "/comparisons",
    response_model=ComparisonResultModel,
    dependencies=[limited("submit")],
)
def submit_comparison(
    body: ComparisonInputModel, session: SessionDep
) -> ComparisonResultModel:
    session.worker.submit(body)
    return ComparisonResultModel(ok=True)
