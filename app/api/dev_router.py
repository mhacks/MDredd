import logging
import os

import strawberry

logger = logging.getLogger(__name__)


@strawberry.type
class DevMutation:
    @strawberry.mutation
    def crash(self) -> bool:
        logger.warning("Dev crash mutation invoked")
        os._exit(1)
