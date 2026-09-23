import logging
import os

import strawberry

from app.api.guard import admin_limit

logger = logging.getLogger(__name__)


@strawberry.type
class DevMutation:
    @strawberry.mutation(permission_classes=[admin_limit])
    def crash(self) -> bool:
        logger.warning("Dev crash mutation invoked")
        os._exit(1)
