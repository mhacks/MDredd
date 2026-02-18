from fastapi.responses import JSONResponse
from fastapi import UploadFile, APIRouter, Request

from app.exceptions import (
    JudgingAlreadyStartedException,
    JudgingNeverStartedException,
    JudgingNotStartedException,
)
from app.models import (
    GenericResponseModel,
    RankingsResponseModel,
)

from logging import getLogger

logger = getLogger(__name__)
admin_router = APIRouter(prefix="/admin", tags=["admin"])


@admin_router.post("/start", response_model=GenericResponseModel)
def start_judging(request: Request, entities_csv: UploadFile | None = None):
    logger.info("Got request to start judging.")
    session = request.state.session
    try:
        session.start(entities_csv)
        return {"status_code": 200, "message": "Successfully started!"}
    except JudgingAlreadyStartedException:
        return {"status_code": 200, "message": "Judging has already started!"}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to start API. Please check logs."},
        )


@admin_router.post("/stop", response_model=GenericResponseModel)
def stop_judging(request: Request):
    logger.info("Got request to stop judging.")
    session = request.state.session
    try:
        session.stop()
        return {"message": "Successfully stopped!", "status_code": 200}
    except JudgingNotStartedException:
        return {"message": "Judging has not started!", "status_code": 200}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to stop API. Please check logs."},
        )


@admin_router.post("/resume", response_model=GenericResponseModel)
def resume_judging(request: Request):
    logger.info("Got request to resume judging.")
    session = request.state.session
    try:
        session.resume()
        return {"message": "Successfully resumed!", "status_code": 200}
    except JudgingAlreadyStartedException:
        return {"message": "Judging has already started", "status_code": 200}
    except JudgingNeverStartedException:
        return {"message": "Judging never started", "status_code": 200}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to resume API. Please check logs."},
        )


@admin_router.get("/rankings", response_model=RankingsResponseModel)
def get_rankings(request: Request):
    session = request.state.session
    try:
        rankings = session.get_rankings()
        return {
            "message": "Successfully got rankings",
            "status_code": 200,
            "is_started": True,
            "rankings": rankings,
        }
    except JudgingNotStartedException:
        return {
            "message": "Judging has never been started!",
            "status_code": 409,
            "is_started": False,
            "rankings": [],
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to get rankings. Please check logs."},
        )
