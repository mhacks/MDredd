import logging
from typing import Tuple

from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import threading

from app.exceptions import (
    IncorrectPairFormatException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingNotStartedException,
)
from app.adapters import SnapshotAdapter, ProjectAdapter, LogAdapter, AssignmentAdapter
from app.models import (
    ComparisonInputModel,
    GenericResponseModel,
    PairResponseModel,
    Project,
    RankingsResponseModel,
    PairRequestModel,
)
from app import constants
from dredd.bdp import BDPVectorized

logger = logging.getLogger("uvicorn")
app = FastAPI()

snapshot_lock = threading.Lock()
snapshot_counter = 0

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
        self.projects = ProjectAdapter()
        self.snapshots = SnapshotAdapter()
        self.assignments = AssignmentAdapter()
        self.logs = LogAdapter()

        snapshot = self.snapshots.load()
        if snapshot is not None:
            timestamp, bdp_instance = snapshot
            self.BDP = bdp_instance

            self.logs.replay(timestamp, self.BDP)
            self.enabled = True

    def get_enabled(self) -> bool:
        return self.enabled

    def start(self, projects_csv: UploadFile | None = None):
        if self.enabled:
            raise JudgingAlreadyStartedException()

        self.projects.load(projects_csv)
        if projects_csv is not None:
            self.projects.clear()
            self.snapshots.clear()
            self.logs.clear()
            self.BDP = BDPVectorized(K=len(self.projects))

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

        if not force and judge in self.snapshots.judge_map:
            i, j = self.snapshots.judge_map[judge]
            return (
                self.projects[i],
                self.projects[j],
            )

        i, j = self.BDP.get_next_pair()
        project_i = self.projects[i]
        project_j = self.projects[j]
        pair = (project_i, project_j)

        self.assignments[judge] = (i, j)

        return pair

    def submit_pair(
        self, judge: str, left_project_id: int, right_project_id: int, winner_id: int
    ):
        if not self.enabled:
            raise JudgingNotStartedException()

        if not self.assignments.verify(judge, left_project_id, right_project_id):
            logger.info(self.snapshots.judge_map[judge])
            logger.info((left_project_id, right_project_id))
            raise JudgeDoesNotOwnPairException()

        if winner_id not in (left_project_id, right_project_id):
            raise IncorrectPairFormatException()

        self.BDP.submit_comparison(left_project_id, right_project_id, winner_id)

        del self.assignments[judge]

    def get_rankings(self):
        if self.enabled:
            sorted_indices = np.flip(np.argsort(self.BDP.get_alphas()))
            projects = self.projects.to_list()
            return [projects[i] for i in sorted_indices]
        else:
            raise JudgingNotStartedException()


api = JudgingAPI()


@app.middleware("http")
def snapshot(request, call_next):
    if request.url.path in ["/"]:
        return call_next(request)

    if api.get_enabled():
        with snapshot_lock:
            global snapshot_counter
            snapshot_counter += 1

            if snapshot_counter >= constants.SNAPSHOT_INTERVAL:
                snapshot_counter = 0
                print("Taking snapshot")
                api.snapshots.record(api.BDP)

    return call_next(request)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/start", response_model=GenericResponseModel)
def start_judging(projects_csv: UploadFile | None = None):
    logger.info("Got request to start judging.")
    try:
        api.start(projects_csv)
        return {"status_code": 200, "message": "Successfully started!"}
    except JudgingAlreadyStartedException:
        return {"status_code": 200, "message": "Judging has already started!"}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to start API. Please check logs."},
        )


@app.post("/stop", response_model=GenericResponseModel)
def stop_judging():
    logger.info("Got request to stop judging.")
    try:
        api.stop()
        return {"message": "Successfully stopped!", "status_code": 200}
    except JudgingNotStartedException:
        return {"message": "Judging has not started!", "status_code": 200}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to stop API. Please check logs."},
        )


@app.post("/resume", response_model=GenericResponseModel)
def resume_judging():
    logger.info("Got request to resume judging.")
    try:
        api.resume()
        return {"message": "Successfully resumed!", "status_code": 200}
    except JudgingAlreadyStartedException:
        return {"message": "Judging has already started", "status_code": 200}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to stop API. Please check logs."},
        )


@app.get("/pair", response_model=PairResponseModel)
def get_pair(pair_request: PairRequestModel = Depends()):
    uuid = pair_request.uuid
    force = pair_request.force

    logger.info(f"Got request for pair by {uuid} (force={force}).")
    try:
        pair = api.get_pair(uuid, force)
        api.logs.log(pair_request)
        return {
            "is_started": api.get_enabled(),
            "pair": pair,
            "message": "Successfully got pair!",
            "status_code": 200,
        }
    except JudgingNotStartedException:
        return {
            "is_started": api.get_enabled(),
            "message": "Judging has not started!",
            "status_code": 409,
        }
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to get pair. Please check logs."},
        )


@app.post("/submit", response_model=GenericResponseModel)
def submit_comparison(comparison_request: ComparisonInputModel):
    try:
        api.submit_pair(
            comparison_request.uuid,
            comparison_request.project_ids[0],
            comparison_request.project_ids[1],
            comparison_request.winner_id,
        )
        api.logs.log(comparison_request)
        return {"message": "Successfully submitted pair!", "status_code": 200}
    except JudgingNotStartedException:
        return {"message": "Judging has not started!", "status_code": 409}
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Unable to submit comparison. Please check logs."},
        )


@app.get("/rankings", response_model=RankingsResponseModel)
def get_rankings():
    try:
        rankings = api.get_rankings()
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


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
