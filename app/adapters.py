from fastapi import UploadFile
import pandas as pd
from typing import Tuple, List
import json
import logging

from dredd import bdp

from app.models import ComparisonInputModel, PairRequestModel, Entity
from app.db import db, Entities, Logs, Snapshots, Assignments
from app.constants import MAX_SNAPSHOTS

logger = logging.getLogger("uvicorn")


class EntityAdapter:
    def __init__(self):
        db.create_tables([Entities], safe=True)

    def __len__(self):
        return Entities.select().count()

    def __getitem__(self, id: int) -> Entity:
        record = Entities.get(Entities.id == (id - 1))  # SQLite IDs start at 1
        return Entity(**json.loads(record.data))

    def to_list(self) -> List[Entity]:
        records = Entity.select().order_by(Entity.id)
        entities = [Entity(**json.loads(record.data)) for record in records]
        return entities

    def clear(self):
        db.drop_tables([Entities], safe=True)

    def load(self, raw_csv: UploadFile = None):
        if raw_csv is not None:
            self.clear()

            # Read entities from csv
            df = pd.read_csv(raw_csv.file)
            df["Table Number"] = df["Table Number"].fillna("").astype(str)
            entities = []
            filtered_df = df[df["Highest Step Completed"] == "Submit"]

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                track_value = row.get("M Hacks Main Track", None)
                entities.append(
                    Entity(
                        project_name=row["Project Title"],
                        devpost_link=row["Submission Url"],
                        table_num=row["Table Number"],
                        tracks=track_value
                        if track_value is not None and not pd.isna(track_value)
                        else "No Track",
                    )
                )
            rows = [{"data": e.model_dump_json()} for e in entities]

            with db.atomic():
                Entities.insert_many(rows).execute()


class SnapshotAdapter:
    def __init__(self):
        db.create_tables([Snapshots], safe=True)

    def clear(self):
        db.drop_tables([Snapshots], safe=True)

    def record(self, bdp_instance: bdp.BDPVectorized):
        with db.atomic():
            Snapshots.create(bdp=bdp_instance.model_dump_json())

            subquery = (
                Snapshots.select(Snapshots.id)
                .order_by(Snapshots.created_at.asc())
                .offset(MAX_SNAPSHOTS)
            )

            Snapshots.delete().where(Snapshots.id.in_(subquery)).execute()

    def load(self) -> Tuple[int, bdp.BDPVectorized] | None:
        record = Snapshots.select().order_by(Snapshots.timestamp.desc()).first()

        if record is not None:
            timestamp = record.timestamp
            algo = bdp.BDPVectorized(**json.loads(record.data))
            return (timestamp, algo)
        else:
            return None


class AssignmentAdapter:
    def __init__(self):
        db.create_tables([Snapshots], safe=True)

    def __setitem__(self, uuid: str, entities):
        Assignments.create(
            judge_id=uuid,
            entity_id_1=entities[0],
            entity_id_2=entities[1],
        )

    def __delitem__(self, uuid: str):
        Assignments.delete().where(Assignments.judge_id == uuid).execute()

    def clear(self):
        db.drop_tables([Snapshots], safe=True)

    def verify(self, uuid: str, entity_id_1: int, entity_id_2: int):
        judge_row = Assignments.get(Assignments.judge_id == uuid)
        pair = (judge_row.entity_id_1, judge_row.entity_id_2)
        return entity_id_1 in pair and entity_id_2 in pair


class LogAdapter:
    def __init__(self):
        db.create_tables([Logs], safe=True)

    def clear(self):
        db.drop_tables([Logs], safe=True)

    def log(self, log_data: ComparisonInputModel | PairRequestModel):
        match log_data:
            case ComparisonInputModel():
                log_type = "submit_pair"
            case PairRequestModel():
                log_type = "get_pair"
            case _:
                raise

        Logs.create(type=log_type, params=log_data.model_dump_json())

    def replay(self, snapshot_time: int, bdp_instance: bdp.BDPVectorized) -> None:
        records = (
            Logs.select()
            .where(Logs.timestamp > snapshot_time)
            .order_by(Logs.timestamp.asc())
        )
        for record in records:
            logger.info(f"Replaying log with timestamp: {record.timestamp}")
            log_type = record.type
            params = json.loads(record.params)

            if log_type == "get_pair":
                bdp_instance.get_next_pair()
            else:
                submit_params = ComparisonInputModel(**params)
                bdp_instance.submit_comparison(
                    submit_params.entity_ids[0],
                    submit_params.entity_ids[1],
                    submit_params.winner_id,
                )
