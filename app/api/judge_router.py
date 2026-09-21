from fastapi.responses import JSONResponse
from fastapi import Depends, APIRouter, Request

from app.exceptions import JudgingNotStartedException
from app.models import (
    ComparisonInputModel,
    GenericResponseModel,
    PairResponseModel,
    PairRequestModel,
)
from logging import getLogger

logger = getLogger(__name__)
judge_router = APIRouter(prefix="/judge", tags=["judge"])


@judge_router.get("/pair", response_model=PairResponseModel)
def get_pair(request: Request, pair_request: PairRequestModel = Depends()):
    session = request.state.session

    uuid = pair_request.uuid
    force = pair_request.force

    logger.info(f"Got request for pair by {uuid} (force={force}).")
    try:
        pair = session.get_pair(uuid, force)
        session.wal.log(pair_request)
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
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to get pair. Please check logs."},
        )


@judge_router.post("/submit", response_model=GenericResponseModel)
def submit_comparison(request: Request, comparison_request: ComparisonInputModel):
    session = request.state.session
    try:
        session.submit_pair(
            comparison_request.uuid,
            comparison_request.entity_ids[0],
            comparison_request.entity_ids[1],
            comparison_request.winner_id,
        )
        session.wal.log(comparison_request)
        return {"message": "Successfully submitted pair!", "status_code": 200}
    except JudgingNotStartedException:
        return {"message": "Judging has not started!", "status_code": 409}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to submit comparison. Please check logs."},
        )
