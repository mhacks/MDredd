from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
import logging

from app.api import admin_router, judge_router
from app.api.dev_router import dev_router
from app.session import Session
from app.settings import settings

logger = logging.getLogger(__name__)


class State(TypedDict):
    session: Session


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    session = Session()
    try:
        yield {"session": session}
    finally:
        session.close()


def create_app() -> FastAPI:
    application = FastAPI(lifespan=lifespan)
    application.include_router(admin_router)
    application.include_router(judge_router)
    if settings.ENABLE_CRASH_ROUTE:
        application.include_router(dev_router)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return application


app = create_app()


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
