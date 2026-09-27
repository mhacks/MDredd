import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypedDict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api import graphql_router, rebind_schema
from app.logging import RequestIdMiddleware, configure_logging
from app.session import Session
from app.settings import settings

logger = logging.getLogger(__name__)


class State(TypedDict):
    session: Session


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[State]:
    session = Session()
    rebind_schema(await run_in_threadpool(session.worker.get_headers))
    try:
        yield {"session": session}
    finally:
        session.close()


async def health(request: Request) -> JSONResponse:
    session: Session = request.state.session
    if session.worker.healthy():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(lifespan=lifespan)
    application.add_api_route("/health", health, methods=["GET"])
    application.include_router(graphql_router())
    application.add_middleware(RequestIdMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return application


app = create_app()


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
