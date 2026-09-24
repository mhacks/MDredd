from starlette.concurrency import run_in_threadpool

from app.columns import Column, graphql_columns
from app.entity import Entity
from app.worker import JudgeWorker


def _prepare_entities(entity_csv: bytes) -> tuple[list[str], list[Entity], list[Column]]:
    headers, entities = Entity.list_from_csv(entity_csv)
    return headers, entities, graphql_columns(headers)


class Session:
    def __init__(self) -> None:
        self.worker = JudgeWorker()
        self.worker.start()

    def close(self) -> None:
        self.worker.shutdown()

    def start(self, entity_csv: bytes) -> list[Column]:
        headers, entities, columns = _prepare_entities(entity_csv)
        self.worker.replace_entities(entities, headers)
        return columns

    async def start_async(self, entity_csv: bytes) -> list[Column]:
        headers, entities, columns = await run_in_threadpool(
            _prepare_entities, entity_csv
        )
        await self.worker.replace_entities_async(entities, headers)
        return columns
