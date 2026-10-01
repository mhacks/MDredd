import logging
import os

from fastapi import APIRouter, Depends

from app.api.deps import limited, require_session

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dev",
    tags=["dev"],
    dependencies=[Depends(require_session)],
)


@router.post("/crash", dependencies=[limited("admin")])
def crash() -> None:
    logger.warning("Dev crash route invoked")
    os._exit(1)
