import logging
from typing import Tuple

from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from app.exceptions import IncorrectPairFormatException, JudgeDoesNotOwnPairException, JudgingAlreadyStartedException, JudgingNotStartedException
from app.adapters import JudgeManager, ProjectAdapter
from app.models import ComparisonInputModel, GenericResponseModel, PairResponseModel, Project, RankingsResponseModel
from app import constants

from dredd.bdp import BDPVectorized

logger = logging.getLogger("uvicorn")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JudgingAPI:
    def __init__(self):
        self.enabled = False

    def get_enabled(self) -> bool:
        return self.enabled

    def start(self, projects_csv: UploadFile | None = None):
        if self.enabled:
            raise JudgingAlreadyStartedException()
        if projects_csv is not None:
            self.project_adapter = ProjectAdapter(projects_csv)
            self.judge_manager = JudgeManager()

        self.BDP = BDPVectorized(len(self.project_adapter))
        self.enabled = True

    def resume(self):
        if self.enabled:
            raise JudgingAlreadyStartedException()
        self.enabled = True

    def stop(self):
        if not self.enabled:
            raise JudgingNotStartedException()
        self.enabled = False

    def get_pair(self, judge, force: bool = False) -> Tuple[Project, Project]:
        if not self.enabled:
            raise JudgingNotStartedException()

        if not force and judge in self.judge_manager.judge_map:
            i, j = self.judge_manager.judge_map[judge]
            return (self.project_adapter.get_project_from_id(i), self.project_adapter.get_project_from_id(j))

        i, j = self.BDP.get_next_pair()
        project_i = self.project_adapter.get_project_from_id(i)
        project_j = self.project_adapter.get_project_from_id(j)
        pair = (project_i, project_j)

        self.judge_manager.judge_map[judge] = (i, j)

        return pair

    def submit_pair(self, judge: str, left_project_id: int, right_project_id: int, winner_id: int):
        if not self.enabled:
            raise JudgingNotStartedException()
        
        if not self.judge_manager.verify_judge_assignment(judge, left_project_id, right_project_id):
            logger.info(self.judge_manager.judge_map[judge])
            logger.info((left_project_id, right_project_id))
            raise JudgeDoesNotOwnPairException()

        if winner_id not in (left_project_id, right_project_id):
            raise IncorrectPairFormatException()

        self.BDP.submit_comparison(
            left_project_id,
            right_project_id,
            winner_id
        )
        del self.judge_manager.judge_map[judge]

    def get_rankings(self):
        if hasattr(self, "BDP") and self.project_adapter and self.judge_manager: # fix
            sorted_indices = np.flip(np.argsort(self.BDP.get_alphas()))
            return [self.project_adapter.projects[i] for i in sorted_indices]
        else:
            raise JudgingNotStartedException()


api = JudgingAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/start", response_model=GenericResponseModel)
def start_judging(projects_csv: UploadFile):
    logger.info("Got request to start judging.")
    try:
        api.start(projects_csv)
        return {"status_code": 200, "message": "Successfully started!"}
    except JudgingAlreadyStartedException:
        return {"status_code": 200, "message": "Judging has already started!"}
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to start API. Please check logs."})


@app.post("/stop", response_model=GenericResponseModel)
def stop_judging():
    logger.info("Got request to stop judging.")
    try:
        api.stop()
        return {
            "message": "Successfully stopped!",
            "status_code": 200
        }
    except JudgingNotStartedException:
        return {
            "message": "Judging has not started!",
            "status_code": 200
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to stop API. Please check logs."})


@app.post("/resume", response_model=GenericResponseModel)
def resume_judging():
    logger.info("Got request to resume judging.")
    try:
        api.resume()
        return {
            "message": "Successfully resumed!",
            "status_code": 200
        }
    except JudgingAlreadyStartedException:
        return {
            "message": "Judging has already started",
            "status_code": 200
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to stop API. Please check logs."})


@app.get("/pair", response_model=PairResponseModel)
def get_pair(uuid: str, force: bool = False):
    logger.info(f"Got request for pair by {uuid} (force={force}).")
    try:
        return {
            "is_started": api.get_enabled(),
            "pair": api.get_pair(uuid, force),
            "message": "Successfully got pair!",
            "status_code": 200
        }
    except JudgingNotStartedException:
        return {
            "is_started": api.get_enabled(),
            "message": "Judging has not started!",
            "status_code": 409
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to get pair. Please check logs."})


@app.post("/submit", response_model=GenericResponseModel)
def submit_comparison(comparison_request: ComparisonInputModel):
    try:
        api.submit_pair(
            comparison_request.uuid,
            comparison_request.project_ids[0],
            comparison_request.project_ids[1],
            comparison_request.winner_id
        )
        return {
            "message": "Successfully submitted pair!",
            "status_code": 200
        }
    except JudgingNotStartedException:
        return {
            "message": "Judging has not started!",
            "status_code": 409
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to submit comparison. Please check logs."})


@app.get("/rankings", response_model=RankingsResponseModel)
def get_rankings():
    try:
        projects, convergence_history = api.get_rankings()
        return {
            "message": "Successfully got rankings",
            "status_code": 200,
            "is_started": True,
            "projects": projects,
            "convergence": convergence_history
        }
    except JudgingNotStartedException:
        return {
            "message": "Judging has never been started!",
            "status_code": 409,
            "is_started": False,
            "projects": [],
            "convergence": [],
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"message": "Unable to get rankings. Please check logs."})


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
