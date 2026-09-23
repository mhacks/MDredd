from app.columns import Column, graphql_columns
from app.entity import Entity
from app.worker import JudgeWorker


class Session:
    def __init__(self) -> None:
        self.worker = JudgeWorker()
        self.worker.start()

    def close(self) -> None:
        self.worker.shutdown()

    def start(self, entity_csv: bytes) -> list[Column]:
        headers, entities = Entity.list_from_csv(entity_csv)
        columns = graphql_columns(headers)
        self.worker.replace_entities(entities, headers)
        return columns
