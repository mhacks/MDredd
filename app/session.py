import logging
import os
import threading
from collections.abc import Callable

from app import project
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

    def start(self, entity_csv: bytes) -> list[str]:
        headers, entities = Entity.list_from_csv(entity_csv)
        entities = project.submitted(headers, entities)
        # A Project Url column, filled in ahead of time, saves those rows a
        # Devpost request. It is moved to the end like a resolved one.
        known = [
            entity.attributes.get(project.PROJECT_URL, "").strip()
            for entity in entities
        ]
        headers = [name for name in headers if name != project.PROJECT_URL]
        entities = [project.without_project_url(entity) for entity in entities]
        # Answer a repeat or a conflict before spending a Devpost request per row.
        current = self.worker.check_replaceable(entities, headers)
        if current is not None:
            return current
        # Resolve here, off the worker thread, which would time out on Devpost.
        entities = project.with_project_urls(
            entities, known, settings.DEVPOST_COOKIE, settings.DEVPOST_CONCURRENCY
        )
        return self.worker.replace_entities(entities, [*headers, project.PROJECT_URL])

    def _watch(self) -> None:
        while not self._closed.wait(settings.WATCHDOG_INTERVAL_SECONDS):
            if not self.worker.healthy():
                logger.critical("Judge worker is unresponsive; exiting for a restart")
                self._on_unhealthy()
                return
