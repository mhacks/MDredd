from app.columns import graphql_columns
from app.entity import Entity
from app.exceptions import JudgingAlreadyStartedException
from app.worker import JudgeWorker


class Session:
    def __init__(self) -> None:
        self.worker = JudgeWorker()
        self.worker.start()

    def close(self) -> None:
        self.worker.shutdown()

    def start(self, entity_csv: bytes | None = None) -> bool:
        if entity_csv is None:
            self.worker.resume()
            return False
        if self.worker.get_enabled():
            raise JudgingAlreadyStartedException()
        headers, entities = Entity.list_from_csv(entity_csv)
        _ = graphql_columns(headers)
        self.worker.replace_entities(entities, headers)
        return True
