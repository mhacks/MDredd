from pydantic import BaseModel, Field

from app.entity import Entity


class EntityWithId(Entity):
    id: int


class ComparisonInputModel(BaseModel):
    judge_id: str = Field(min_length=1)
    entity_ids: tuple[int, int]
    winner_id: int


class PairRequestModel(BaseModel):
    judge_id: str = Field(min_length=1)
    force: bool = False


class RowModel(BaseModel):
    id: int
    attributes: dict[str, str]

    @classmethod
    def from_entity(cls, entity: EntityWithId) -> RowModel:
        return cls(id=entity.id, attributes=dict(entity.attributes))


class JudgingModel(BaseModel):
    is_started: bool


class DatasetModel(BaseModel):
    is_started: bool
    headers: list[str]


class ColumnsModel(BaseModel):
    headers: list[str]


class PairModel(BaseModel):
    pair: tuple[RowModel, RowModel]


class ComparisonResultModel(BaseModel):
    ok: bool
