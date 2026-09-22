
from pydantic import BaseModel

from app.entity import Entity


class EntityWithId(Entity):
    id: int


class GenericResponseModel(BaseModel):
    status_code: int
    message: str


class PairResponseModel(GenericResponseModel):
    is_started: bool
    pair: tuple[EntityWithId, EntityWithId] | None = None


class RankingsResponseModel(GenericResponseModel):
    is_started: bool
    rankings: list[Entity]


class ComparisonInputModel(BaseModel):
    uuid: str
    entity_ids: tuple[int, int]
    winner_id: int


class PairRequestModel(BaseModel):
    uuid: str
    force: bool = False
