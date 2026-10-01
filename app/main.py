import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypedDict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import admin_router, dev_router, judge_router
from app.api.errors import judging_failure_handler
from app.exceptions import JudgingFailure
from app.logging import AccessLogMiddleware, RequestIdMiddleware, configure_logging
from app.session import Session
from app.settings import settings

logger = logging.getLogger(__name__)


class State(TypedDict):
    session: Session


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[State]:
    session = Session()
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
    application.add_exception_handler(JudgingFailure, judging_failure_handler)
    application.add_api_route("/health", health, methods=["GET"])
    application.include_router(admin_router)
    application.include_router(judge_router)
    if settings.ENABLE_CRASH_ROUTE:
        application.include_router(dev_router)
    application.add_middleware(AccessLogMiddleware)
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
