from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.exceptions import (
    IncorrectPairFormatException,
    InvalidColumnsException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingFailure,
    JudgingNeverStartedException,
    JudgingNotStartedException,
    TooFewEntitiesException,
    UnknownRowException,
    WorkerUnavailableException,
)

_STATUS: dict[type[JudgingFailure], int] = {
    JudgingNotStartedException: status.HTTP_409_CONFLICT,
    JudgingAlreadyStartedException: status.HTTP_409_CONFLICT,
    JudgingNeverStartedException: status.HTTP_409_CONFLICT,
    JudgeDoesNotOwnPairException: status.HTTP_409_CONFLICT,
    IncorrectPairFormatException: status.HTTP_422_UNPROCESSABLE_CONTENT,
    TooFewEntitiesException: status.HTTP_422_UNPROCESSABLE_CONTENT,
    InvalidColumnsException: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UnknownRowException: status.HTTP_404_NOT_FOUND,
    WorkerUnavailableException: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def judging_failure_handler(_request: Request, exc: JudgingFailure) -> JSONResponse:
    detail: dict[str, object] = {"code": exc.code}
    if isinstance(exc, InvalidColumnsException):
        detail["names"] = exc.names
    return JSONResponse(
        status_code=_STATUS.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR),
        content={"detail": detail},
    )
