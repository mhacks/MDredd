from typing import List, Optional, Tuple
from pydantic import BaseModel
from app.entity import Entity


class EntityWithId(Entity):
    id: int


class GenericResponseModel(BaseModel):
    status_code: int
    message: str


class PairResponseModel(GenericResponseModel):
    is_started: bool
    pair: Optional[Tuple[EntityWithId, EntityWithId]] = None


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
