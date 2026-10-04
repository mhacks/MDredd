import time

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import SessionDep, enforce_limit, require_session
from app.api.errors import error_responses
from app.models import (
    ComparisonInputModel,
    ComparisonResultModel,
    PairModel,
    PairRequestModel,
    ProjectModel,
    RowModel,
)

router = APIRouter(
    tags=["judge"],
    dependencies=[Depends(require_session)],
    responses=error_responses(status.HTTP_503_SERVICE_UNAVAILABLE),
)


@router.get(
    "/projects",
    response_model=list[RowModel],
    tags=["admin"],
    description="List every project in upload order. Organizers and judges share this route.",
    response_description="Projects in upload order.",
)
def get_projects(session: SessionDep) -> list[RowModel]:
    return [RowModel.from_entity(entity) for entity in session.worker.projects()]


@router.post(
    "/pairs",
    response_model=PairModel,
    description=(
        "Draw this judge's open pair, or return the pair they already hold. "
        "One absent project from that pair forfeits, and the other project wins. "
        "Both absent strikes each project and draws a new pair. "
        "skip gives up the open pair unjudged and draws a new one."
    ),
    response_description="The two projects to compare: id, Devpost URL, name, and tracks.",
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def create_pair(
    body: PairRequestModel, request: Request, session: SessionDep
) -> PairModel:
    # Each judge has their own bucket, since every judge may come through one client.
    enforce_limit(request, "pair", body.judge_id)
    left, right, assigned_at = session.worker.request_pair(body)
    return PairModel(
        pair=(ProjectModel.from_entity(left), ProjectModel.from_entity(right)),
        assigned_at=assigned_at,
        server_time=time.time(),
    )


@router.post(
    "/comparisons",
    response_model=ComparisonResultModel,
    description="Record the winner of this judge's open pair.",
    response_description="The comparison was recorded.",
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def submit_comparison(
    body: ComparisonInputModel, request: Request, session: SessionDep
) -> ComparisonResultModel:
    enforce_limit(request, "submit", body.judge_id)
    session.worker.submit(body)
    return ComparisonResultModel(ok=True)
