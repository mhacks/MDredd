from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app import project
from app.entity import Entity


class EntityWithId(Entity):
    id: int


class ComparisonInputModel(BaseModel):
    judge_id: str = Field(min_length=1)
    entity_ids: tuple[int, int]
    winner_id: int


class PairRequestModel(BaseModel):
    judge_id: str = Field(min_length=1)
    absent: list[int] = Field(default_factory=list, max_length=2)
    # The open pair to give up unjudged, e.g. when the judge's timer runs out.
    skip: tuple[int, int] | None = None

    @model_validator(mode="after")
    def one_action(self) -> PairRequestModel:
        if self.skip is not None and self.absent:
            raise ValueError("Send either absent or skip, not both")
        return self


class PoolEntryModel(BaseModel):
    id: int
    attributes: dict[str, str]
    strikes: int
    removed: bool


class RowModel(BaseModel):
    id: int
    attributes: dict[str, str]

    @classmethod
    def from_entity(cls, entity: EntityWithId) -> RowModel:
        return cls(id=entity.id, attributes=dict(entity.attributes))


class TablesInputModel(BaseModel):
    tables: dict[Annotated[str, Field(min_length=1)], Annotated[int, Field(gt=0)]] = Field(
        max_length=10_000
    )


class TablesModel(BaseModel):
    stored: int
    unknown_urls: list[str]


class JudgingModel(BaseModel):
    is_started: bool


class DatasetModel(BaseModel):
    is_started: bool
    headers: list[str]


class ArchiveModel(BaseModel):
    path: str | None


class ArchiveListModel(BaseModel):
    archives: list[str]


class ColumnsModel(BaseModel):
    headers: list[str]


class ProjectModel(BaseModel):
    id: int
    url: str
    name: str
    tracks: list[str]

    @classmethod
    def from_entity(cls, entity: EntityWithId) -> ProjectModel:
        return cls(
            id=entity.id,
            url=entity.attributes.get(project.PROJECT_URL, ""),
            name=entity.attributes.get(project.TITLE, ""),
            tracks=project.tracks(entity.attributes),
        )


class PairModel(BaseModel):
    pair: tuple[ProjectModel, ProjectModel]
    # Unix time the pair was handed out, and MDredd's clock when it answered,
    # so a client can time the pair without trusting its own clock.
    assigned_at: float
    server_time: float


class ComparisonResultModel(BaseModel):
    ok: bool
