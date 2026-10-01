import logging
import os

from fastapi import APIRouter, Depends

from app.api.deps import limited, require_session
from app.api.errors import error_responses

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dev",
    tags=["dev"],
    dependencies=[Depends(require_session)],
    responses=error_responses(),
)


@router.post(
    "/crash",
    description="Exit the process so the container can restart.",
    dependencies=[limited("admin")],
    responses=error_responses(limited=True),
)
def crash() -> None:
    logger.warning("Dev crash route invoked")
    os._exit(1)
