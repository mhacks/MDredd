import logging
import os

from fastapi import APIRouter

logger = logging.getLogger(__name__)
dev_router = APIRouter(prefix="/dev", tags=["dev"])


@dev_router.post("/crash")
def crash():
    logger.warning("Dev crash route invoked")
    os._exit(1)
