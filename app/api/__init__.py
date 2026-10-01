from app.api.admin_router import router as admin_router
from app.api.dev_router import router as dev_router
from app.api.judge_router import router as judge_router

__all__ = ["admin_router", "dev_router", "judge_router"]
