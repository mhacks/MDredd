from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
import logging

from app.api import admin_router, judge_router
from app.session import Session

logger = logging.getLogger(__name__)


class State(TypedDict):
    session: Session


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    session = Session()
    yield {"session": session}


app = FastAPI(lifespan=lifespan)
app.include_router(admin_router)
app.include_router(judge_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    logger.info("Starting API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
