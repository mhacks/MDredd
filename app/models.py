from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class Project(BaseModel):
    project_name: str
    link_to_devpost: str = Field(alias="devpost_link")
    table_num: str
    project_id: int
    tracks: str

    # class.model_dump() => api response format

class GenericResponseModel(BaseModel):
    status_code: int
    message: str


class PairResponseModel(GenericResponseModel):
    is_started: bool
    pair: Optional[Tuple[Project, Project]] = None


class RankingsResponseModel(GenericResponseModel):
    is_started: bool
    projects: List[Project]
    convergence: List[List[float]]


class ComparisonInputModel(BaseModel):
    uuid: str
    project_ids: Tuple[int, int]
    winner_id: int
