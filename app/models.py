
from pydantic import BaseModel

from app.entity import Entity


class EntityWithId(Entity):
    id: int


class ComparisonInputModel(BaseModel):
    uuid: str
    entity_ids: tuple[int, int]
    winner_id: int


class PairRequestModel(BaseModel):
    uuid: str
    force: bool = False
