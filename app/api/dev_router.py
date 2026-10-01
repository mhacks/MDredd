import logging
import os

from fastapi import APIRouter

from app.api.deps import SessionDep, limited

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dev", tags=["dev"])


@router.post("/crash", dependencies=[limited("admin")])
def crash(_session: SessionDep) -> bool:
    logger.warning("Dev crash route invoked")
    os._exit(1)
