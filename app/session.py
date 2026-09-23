from fastapi import UploadFile

from app.columns import graphql_columns
from app.entity import Entity
from app.exceptions import JudgingAlreadyStartedException
from app.models import ComparisonInputModel, EntityWithId, PairRequestModel
from app.worker import JudgeWorker


class Session:
    def __init__(self) -> None:
        self.worker = JudgeWorker()
        self.worker.start()

    def close(self) -> None:
        self.worker.shutdown()

    def get_enabled(self) -> bool:
        return self.worker.get_enabled()

    def headers(self) -> list[str]:
        return self.worker.get_headers()

    def start(self, entity_csv: UploadFile | None = None) -> bool:
        if entity_csv is None:
            self.worker.resume()
            return False
        if self.worker.get_enabled():
            raise JudgingAlreadyStartedException()
        headers, entities = Entity.list_from_csv(entity_csv)
        _ = graphql_columns(headers)
        self.worker.replace_entities(entities, headers)
        return True

    def resume(self) -> None:
        self.worker.resume()

    def stop(self) -> None:
        self.worker.stop()

    def get_pair(self, pair_request: PairRequestModel) -> tuple[EntityWithId, EntityWithId]:
        return self.worker.request_pair(pair_request)

    def submit_pair(self, comparison: ComparisonInputModel) -> None:
        self.worker.submit(comparison)

    def get_rankings(self) -> list[EntityWithId]:
        return self.worker.rankings()

    def get_row(self, row_id: int) -> EntityWithId:
        return self.worker.get_row(row_id)
