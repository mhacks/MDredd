from logging import getLogger
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.exceptions import JudgingNotStartedException
from app.models import (
    ComparisonInputModel,
    GenericResponseModel,
    PairRequestModel,
    PairResponseModel,
)
from app.session import get_session

logger = getLogger(__name__)
judge_router = APIRouter(prefix="/judge", tags=["judge"])


@judge_router.get("/pair", response_model=PairResponseModel)
def get_pair(request: Request, pair_request: Annotated[PairRequestModel, Depends()]):
    session = get_session(request)

    logger.info(
        "Got request for pair by %s (force=%s).",
        pair_request.uuid,
        pair_request.force,
    )
    try:
        pair = session.get_pair(pair_request)
        return {
            "is_started": session.get_enabled(),
            "pair": pair,
            "message": "Successfully got pair!",
            "status_code": 200,
        }
    except JudgingNotStartedException:
        return {
            "is_started": session.get_enabled(),
            "message": "Judging has not started!",
            "status_code": 409,
        }


@judge_router.post("/submit", response_model=GenericResponseModel)
def submit_comparison(request: Request, comparison_request: ComparisonInputModel):
    session = get_session(request)
    try:
        session.submit_pair(comparison_request)
        return {"message": "Successfully submitted pair!", "status_code": 200}
    except JudgingNotStartedException:
        return {"message": "Judging has not started!", "status_code": 409}
