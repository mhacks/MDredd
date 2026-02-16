from fastapi import UploadFile
from typing import Tuple, List
import json
import logging
import time

from dredd import bdp

from app.entity import Entity
from app.models import ComparisonInputModel, PairRequestModel
from app.db import db, EntityTable, WriteAheadTable, SnapshotTable, AssignmentTable
from app.constants import MAX_SNAPSHOTS

logger = logging.getLogger("uvicorn")


class EntityAdapter:
    def __init__(self):
        db.create_tables([EntityTable], safe=True)

    def __len__(self):
        return EntityTable.select().count()

    def __getitem__(self, id: int) -> Entity:
        record = EntityTable.get(EntityTable.id == (id + 1))  # SQLite IDs start at 1
        return Entity(**json.loads(record.data))

    def to_list(self) -> List[Entity]:
        records = EntityTable.select().order_by(EntityTable.id)
        entities = [Entity(**json.loads(record.data)) for record in records]
        return entities

    def clear(self):
        db.drop_tables([EntityTable], safe=True)
        db.create_tables([EntityTable], safe=True)

    def load(self, raw_csv: UploadFile = None):
        if raw_csv is not None:
            self.clear()
            entities = Entity.list_from_csv(raw_csv)
            rows = [{"data": e.model_dump_json()} for e in entities]

            with db.atomic():
                EntityTable.insert_many(rows).execute()


class SnapshotAdapter:
    def __init__(self):
        db.create_tables([SnapshotTable], safe=True)

    def clear(self):
        db.drop_tables([SnapshotTable], safe=True)
        db.create_tables([SnapshotTable], safe=True)

    def record(self, bdp_instance: bdp.BDPVectorized):
        with db.atomic():
            SnapshotTable.create(
                bdp=bdp_instance.model_dump_json(), timestamp=time.time()
            )

            subquery = (
                SnapshotTable.select(SnapshotTable.id)
                .order_by(SnapshotTable.id.asc())
                .offset(MAX_SNAPSHOTS)
            )

            SnapshotTable.delete().where(SnapshotTable.id.in_(subquery)).execute()

    def load(self) -> Tuple[int, bdp.BDPVectorized] | None:
        record = SnapshotTable.select().order_by(SnapshotTable.timestamp.desc()).first()

        if record is not None:
            timestamp = record.timestamp
            algo = bdp.BDPVectorized(**json.loads(record.bdp))
            return (timestamp, algo)
        else:
            return None


class AssignmentAdapter:
    def __init__(self):
        db.create_tables([AssignmentTable], safe=True)

    def __getitem__(self, uuid: str):
        judge_row = AssignmentTable.get(AssignmentTable.judge_id == uuid)
        return (judge_row.entity_id_1, judge_row.entity_id_2)

    def __setitem__(self, uuid: str, entities):
        AssignmentTable.create(
            judge_id=uuid,
            entity_id_1=entities[0],
            entity_id_2=entities[1],
            timestamp=time.time(),
        )

    def __delitem__(self, uuid: str):
        AssignmentTable.delete().where(AssignmentTable.judge_id == uuid).execute()

    def __contains__(self, uuid: str):
        return AssignmentTable.select().where(AssignmentTable.judge_id == uuid).exists()

    def clear(self):
        db.drop_tables([AssignmentTable], safe=True)
        db.create_tables([AssignmentTable], safe=True)

    def verify(self, uuid: str, entity_id_1: int, entity_id_2: int):
        pair = self[uuid]
        return entity_id_1 in pair and entity_id_2 in pair


class WriteAheadAdapter:
    def __init__(self):
        db.create_tables([WriteAheadTable], safe=True)

    def clear(self):
        db.drop_tables([WriteAheadTable], safe=True)
        db.create_tables([WriteAheadTable], safe=True)

    def log(self, log_data: ComparisonInputModel | PairRequestModel):
        match log_data:
            case ComparisonInputModel():
                event_type = "submit_pair"
            case PairRequestModel():
                event_type = "get_pair"
            case _:
                raise

        WriteAheadTable.create(
            event=event_type, timestamp=time.time(), params=log_data.model_dump_json()
        )

    def replay(self, snapshot_time: int, bdp_instance: bdp.BDPVectorized) -> None:
        records = (
            WriteAheadTable.select()
            .where(WriteAheadTable.timestamp > snapshot_time)
            .order_by(WriteAheadTable.timestamp.asc())
        )
        for record in records:
            logger.info(f"Replaying log with timestamp: {record.timestamp}")
            params = json.loads(record.params)

            match record.event:
                case "get_pair":
                    bdp_instance.get_next_pair()
                case "submit_pair":
                    submit_params = ComparisonInputModel(**params)
                    bdp_instance.submit_comparison(
                        submit_params.entity_ids[0],
                        submit_params.entity_ids[1],
                        submit_params.winner_id,
                    )
                case _:
                    raise
