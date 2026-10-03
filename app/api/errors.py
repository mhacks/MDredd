from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.exceptions import (
    AbsentNotInPairException,
    DatabaseUnreadableException,
    DevpostUnresolvedException,
    IncorrectPairFormatException,
    InvalidColumnsException,
    JudgeDoesNotOwnPairException,
    JudgingAlreadyStartedException,
    JudgingFailure,
    JudgingNeverStartedException,
    JudgingNotStartedException,
    PoolExhaustedException,
    TooFewEntitiesException,
    UnknownArchiveException,
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
    UnknownArchiveException: status.HTTP_404_NOT_FOUND,
    AbsentNotInPairException: status.HTTP_409_CONFLICT,
    PoolExhaustedException: status.HTTP_409_CONFLICT,
    DatabaseUnreadableException: status.HTTP_503_SERVICE_UNAVAILABLE,
    WorkerUnavailableException: status.HTTP_503_SERVICE_UNAVAILABLE,
    DevpostUnresolvedException: status.HTTP_502_BAD_GATEWAY,
}


class CodeDetail(BaseModel):
    code: str


class CodeError(BaseModel):
    detail: CodeDetail


class RateDetail(BaseModel):
    code: str
    retry_after_ms: int


class RateError(BaseModel):
    detail: RateDetail


_DOCUMENTED: dict[int, dict[str, object]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": CodeError,
        "description": "The API token is missing or unknown.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": CodeError,
        "description": "The row does not exist.",
    },
    status.HTTP_409_CONFLICT: {
        "model": CodeError,
        "description": "The judging state rejected the request.",
    },
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": RateError,
        "description": "The rate limit is exhausted.",
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": CodeError,
        "description": "Some submission URLs did not resolve on Devpost.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": CodeError,
        "description": "The judge worker is unavailable.",
    },
}


def error_responses(*codes: int, limited: bool = False) -> dict[int, dict[str, object]]:
    documented = [status.HTTP_401_UNAUTHORIZED, *codes]
    if limited:
        documented.append(status.HTTP_429_TOO_MANY_REQUESTS)
    return {code: _DOCUMENTED[code] for code in documented}


def judging_failure_handler(_request: Request, exc: JudgingFailure) -> JSONResponse:
    detail: dict[str, object] = {"code": exc.code}
    if isinstance(exc, InvalidColumnsException):
        detail["names"] = exc.names
    if isinstance(exc, DevpostUnresolvedException):
        detail["failures"] = exc.failures
    return JSONResponse(
        status_code=_STATUS.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR),
        content={"detail": detail},
    )
