import io
import zipfile

from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import Response

from app.api.deps import SessionDep, limited, require_session
from app.api.errors import error_responses
from app.db import list_archive_files, list_archives
from app.models import (
    ArchiveListModel,
    ArchiveModel,
    ColumnsModel,
    DatasetModel,
    JudgingModel,
    PoolEntryModel,
    RowModel,
)

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(require_session)],
    responses=error_responses(status.HTTP_503_SERVICE_UNAVAILABLE),
)


@router.post(
    "/datasets",
    response_model=DatasetModel,
    status_code=status.HTTP_201_CREATED,
    description=(
        "Replace the stored dataset from a CSV upload and start judging. "
        "Repeating that dataset is accepted."
    ),
    response_description="The stored headers and that judging is on.",
    dependencies=[limited("admin")],
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def create_dataset(session: SessionDep, entities_csv: UploadFile) -> DatasetModel:
    headers = session.start(entities_csv.file.read())
    return DatasetModel(is_started=True, headers=headers)


@router.get(
    "/judging",
    response_model=JudgingModel,
    description="Report whether judging is accepting pairs.",
    response_description="Whether judging is on.",
)
def get_judging(session: SessionDep) -> JudgingModel:
    return JudgingModel(is_started=session.worker.get_enabled())


@router.post(
    "/judging/start",
    response_model=JudgingModel,
    description=(
        "Turn judging on for the stored dataset. "
        "A request while judging is on succeeds."
    ),
    response_description="Judging is on.",
    dependencies=[limited("admin")],
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
@router.post(
    "/judging/resume",
    response_model=JudgingModel,
    description=(
        "Turn judging on for the stored dataset. "
        "A request while judging is on succeeds."
    ),
    response_description="Judging is on.",
    dependencies=[limited("admin")],
    responses=error_responses(status.HTTP_409_CONFLICT, limited=True),
)
def resume_judging(session: SessionDep) -> JudgingModel:
    return JudgingModel(is_started=session.worker.resume())


@router.post(
    "/judging/stop",
    response_model=JudgingModel,
    description=(
        "Stop issuing pairs and accepting comparisons. "
        "A request while judging is off succeeds."
    ),
    response_description="Judging is off.",
    dependencies=[limited("admin")],
    responses=error_responses(limited=True),
)
def stop_judging(session: SessionDep) -> JudgingModel:
    return JudgingModel(is_started=session.worker.stop())


@router.post(
    "/archive",
    response_model=ArchiveModel,
    description=(
        "Move the SQLite database and the log file into a new archive folder "
        "and start empty. Earlier archives are kept. Judging is off afterward. "
        "If startup cannot read the file, it logs that and keeps serving "
        "until this route is called."
    ),
    response_description="The folder that holds the archived database and log.",
    dependencies=[limited("admin")],
    responses=error_responses(limited=True),
)
def archive_database(session: SessionDep) -> ArchiveModel:
    return ArchiveModel(path=session.worker.archive())


@router.get(
    "/archives",
    response_model=ArchiveListModel,
    description="List archived databases, newest first.",
    response_description="Archive folder names, newest first.",
)
def get_archives() -> ArchiveListModel:
    return ArchiveListModel(archives=list_archives())


@router.get(
    "/archives/{archive_id}",
    description="Download every file in one archive as a zip.",
    response_description="A zip of the database, its sidecars, and the log.",
    responses=error_responses(status.HTTP_404_NOT_FOUND),
)
def get_archive(archive_id: str) -> Response:
    files = list_archive_files(archive_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{archive_id}.zip"'},
    )


@router.get(
    "/columns",
    response_model=ColumnsModel,
    description="List the CSV headers in upload order.",
    response_description="The CSV headers in upload order.",
)
def get_columns(session: SessionDep) -> ColumnsModel:
    return ColumnsModel(headers=session.worker.get_headers())


@router.get(
    "/rows/{row_id}",
    response_model=RowModel,
    description="Return one uploaded row.",
    response_description="The row id and its attributes.",
    responses=error_responses(status.HTTP_404_NOT_FOUND),
)
def get_row(row_id: int, session: SessionDep) -> RowModel:
    return RowModel.from_entity(session.worker.get_row(row_id))


@router.get(
    "/rankings",
    response_model=list[RowModel],
    description="Return every row, highest strength first.",
    response_description="Rows ordered by strength.",
    responses=error_responses(status.HTTP_409_CONFLICT),
)
def get_rankings(session: SessionDep) -> list[RowModel]:
    return [RowModel.from_entity(entity) for entity in session.worker.rankings()]


@router.get(
    "/pool",
    response_model=list[PoolEntryModel],
    description="List every project with its strike count and whether it is removed.",
    response_description="Projects in upload order.",
)
def get_pool(session: SessionDep) -> list[PoolEntryModel]:
    return session.worker.pool()


@router.post(
    "/pool/{entity_id}/restore",
    response_model=PoolEntryModel,
    description=(
        "Return a removed project to the pool by clearing its strikes. "
        "A project that is not removed is left unchanged."
    ),
    response_description="The project and its strike count.",
    dependencies=[limited("admin")],
    responses=error_responses(status.HTTP_404_NOT_FOUND, limited=True),
)
def restore_pool_entity(entity_id: int, session: SessionDep) -> PoolEntryModel:
    return session.worker.restore(entity_id)
