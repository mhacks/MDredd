from typing import List, Optional, Tuple
from pydantic import BaseModel


class Entity(BaseModel):
    project_name: str
    devpost_link: str
    table_num: str
    project_id: int
    tracks: str


class GenericResponseModel(BaseModel):
    status_code: int
    message: str


class PairResponseModel(GenericResponseModel):
    is_started: bool
    pair: Optional[Tuple[Entity, Entity]] = None


class RankingsResponseModel(GenericResponseModel):
    is_started: bool
    rankings: List[Entity]


class ComparisonInputModel(BaseModel):
    uuid: str
    entity_ids: Tuple[int, int]
    winner_id: int


class PairRequestModel(BaseModel):
    uuid: str
    force: bool = False
