from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions import InvalidColumnsException, JudgingFailure

_STATUS = {
    "JUDGING_NOT_STARTED": 409,
    "JUDGING_ALREADY_STARTED": 409,
    "JUDGING_NEVER_STARTED": 409,
    "JUDGE_DOES_NOT_OWN_PAIR": 409,
    "INCORRECT_PAIR_FORMAT": 422,
    "TOO_FEW_ENTITIES": 422,
    "UNKNOWN_ROW": 404,
    "INVALID_COLUMNS": 422,
    "WORKER_UNAVAILABLE": 503,
}


def judging_failure_handler(_request: Request, exc: JudgingFailure) -> JSONResponse:
    detail: dict[str, object] = {"code": exc.code}
    if isinstance(exc, InvalidColumnsException):
        detail["names"] = exc.names
    return JSONResponse(
        status_code=_STATUS.get(exc.code, 500),
        content={"detail": detail},
    )
