from fastapi import APIRouter, UploadFile

from app.api.deps import SessionDep, limited
from app.models import ColumnsModel, DatasetModel, JudgingModel, RowModel

router = APIRouter(tags=["admin"])


@router.post("/datasets", response_model=DatasetModel, dependencies=[limited("admin")])
def create_dataset(session: SessionDep, entities_csv: UploadFile) -> DatasetModel:
    headers = session.start(entities_csv.file.read())
    return DatasetModel(is_started=True, headers=headers)


@router.get("/judging", response_model=JudgingModel)
def get_judging(session: SessionDep) -> JudgingModel:
    return JudgingModel(is_started=session.worker.get_enabled())


@router.post(
    "/judging/start", response_model=JudgingModel, dependencies=[limited("admin")]
)
def start_judging(session: SessionDep) -> JudgingModel:
    session.worker.resume()
    return JudgingModel(is_started=session.worker.get_enabled())


@router.post(
    "/judging/stop", response_model=JudgingModel, dependencies=[limited("admin")]
)
def stop_judging(session: SessionDep) -> JudgingModel:
    session.worker.stop()
    return JudgingModel(is_started=session.worker.get_enabled())


@router.post(
    "/judging/resume", response_model=JudgingModel, dependencies=[limited("admin")]
)
def resume_judging(session: SessionDep) -> JudgingModel:
    session.worker.resume()
    return JudgingModel(is_started=session.worker.get_enabled())


@router.get("/columns", response_model=ColumnsModel)
def get_columns(session: SessionDep) -> ColumnsModel:
    return ColumnsModel(headers=session.worker.get_headers())


@router.get("/rows/{row_id}", response_model=RowModel)
def get_row(row_id: int, session: SessionDep) -> RowModel:
    return RowModel.from_entity(session.worker.get_row(row_id))


@router.get("/rankings", response_model=list[RowModel])
def get_rankings(session: SessionDep) -> list[RowModel]:
    return [RowModel.from_entity(entity) for entity in session.worker.rankings()]
