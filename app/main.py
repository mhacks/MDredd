import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypedDict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import graphql_router, rebind_schema
from app.logging import RequestIdMiddleware, configure_logging
from app.session import Session

logger = logging.getLogger(__name__)


class State(TypedDict):
    session: Session


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[State]:
    session = Session()
    session.worker.bind_loop(asyncio.get_running_loop())
    rebind_schema(session.headers())
    try:
        yield {"session": session}
    finally:
        session.close()


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(lifespan=lifespan)
    application.include_router(graphql_router())
    application.add_middleware(RequestIdMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return application


app = create_app()


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
