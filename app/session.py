import logging
import os
import threading
from collections.abc import Callable

from app.columns import Column, graphql_columns
from app.entity import Entity
from app.settings import settings
from app.worker import JudgeWorker

logger = logging.getLogger(__name__)


def _exit_process() -> None:
    # Every applied change is already committed, so exiting loses nothing and
    # lets the container restart policy bring up a fresh worker.
    os._exit(1)


class Session:
    def __init__(self, on_unhealthy: Callable[[], None] = _exit_process) -> None:
        self.worker = JudgeWorker()
        self.worker.start()
        self._on_unhealthy = on_unhealthy
        self._closed = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watch, name="judge-watchdog", daemon=True
        )
        self._watchdog.start()

    def close(self) -> None:
        self._closed.set()
        self.worker.shutdown()

    def start(self, entity_csv: bytes) -> list[Column]:
        headers, entities = Entity.list_from_csv(entity_csv)
        columns = graphql_columns(headers)
        self.worker.replace_entities(entities, headers)
        return columns

    def _watch(self) -> None:
        while not self._closed.wait(settings.WATCHDOG_INTERVAL_SECONDS):
            if not self.worker.healthy():
                logger.critical("Judge worker is unresponsive; exiting for a restart")
                self._on_unhealthy()
                return
