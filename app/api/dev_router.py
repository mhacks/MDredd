import logging
import os

import strawberry

from app.api.guard import AdminWrite

logger = logging.getLogger(__name__)


@strawberry.type
class DevMutation:
    @strawberry.mutation(permission_classes=[AdminWrite])
    def crash(self) -> bool:
        logger.warning("Dev crash mutation invoked")
        os._exit(1)
