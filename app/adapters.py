from fastapi import UploadFile
import pandas as pd
from typing import Tuple, List
import json
import logging

from dredd import bdp

from app.models import ComparisonInputModel, PairRequestModel, Project
from app.db import db, EntityTable, LogTable, SnapshotTable, AssignmentTable
from app.constants import MAX_SNAPSHOTS

logger = logging.getLogger("uvicorn")


class ProjectAdapter:
    def __init__(self):
        db.create_tables([EntityTable], safe=True)

    def __len__(self):
        return EntityTable.select().count()

    def __getitem__(self, id: int) -> Project:
        record = EntityTable.get(EntityTable.id == id)
        return Project(**json.loads(record.data))

    def to_list(self) -> List[Project]:
        records = Project.select().order_by(Project.project_id)
        projects = [Project(**json.loads(record.data)) for record in records]
        return projects

    def clear(self):
        db.drop_tables([EntityTable], safe=True)

    def load(self, raw_csv: UploadFile = None):
        if raw_csv is not None:
            self.clear()

            # Read projects from csv
            df = pd.read_csv(raw_csv.file)
            df["Table Number"] = df["Table Number"].fillna("").astype(str)
            projects = []
            filtered_df = df[df["Highest Step Completed"] == "Submit"]

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                track_value = row.get("M Hacks Main Track", None)
                projects.append(
                    Project(
                        project_name=row["Project Title"],
                        devpost_link=row["Submission Url"],
                        table_num=row["Table Number"],
                        tracks=track_value
                        if track_value is not None and not pd.isna(track_value)
                        else "No Track",
                    )
                )
            rows = [{"data": p.model_dump_json()} for p in projects]

            with db.atomic():
                EntityTable.insert_many(rows).execute()


class SnapshotAdapter:
    def __init__(self):
        db.create_tables([SnapshotTable], safe=True)

    def clear(self):
        db.drop_tables([SnapshotTable], safe=True)

    def record(self, bdp_instance: bdp.BDPVectorized):
        with db.atomic():
            SnapshotTable.create(bdp=bdp_instance.model_dump_json())

            subquery = (
                SnapshotTable.select(SnapshotTable.id)
                .order_by(SnapshotTable.created_at.asc())
                .offset(MAX_SNAPSHOTS)
            )

            SnapshotTable.delete().where(SnapshotTable.id.in_(subquery)).execute()

    def load(self) -> Tuple[int, bdp.BDPVectorized] | None:
        record = SnapshotTable.select().order_by(SnapshotTable.time.desc()).first()

        if record is not None:
            timestamp = record.time
            algo = bdp.BDPVectorized(**json.loads(record.data))
            return (timestamp, algo)
        else:
            return None


class AssignmentAdapter:
    def __init__(self):
        db.create_tables([SnapshotTable], safe=True)

    def __setitem__(self, uuid: str, projects):
        AssignmentTable.create(
            judge_id=uuid,
            project_id_1=projects[0],
            project_id_2=projects[1],
        )

    def __delitem__(self, uuid: str):
        AssignmentTable.delete().where(AssignmentTable.judge_id == uuid).execute()

    def clear(self):
        db.drop_tables([SnapshotTable], safe=True)

    def verify(self, uuid: str, project_id_1: int, project_id_2: int):
        judge_row = AssignmentTable.get(AssignmentTable.judge_id == uuid)
        pair = (judge_row.project_id_1, judge_row.project_id_2)
        return project_id_1 in pair and project_id_2 in pair


class LogAdapter:
    def __init__(self):
        db.create_tables([LogTable], safe=True)

    def clear(self):
        db.drop_tables([LogTable], safe=True)

    def log(self, log_data: ComparisonInputModel | PairRequestModel):
        log_type = ""
        match log_data:
            case ComparisonInputModel():
                log_type = "submit_pair"
            case PairRequestModel():
                log_type = "get_pair"
            case _:
                raise

        LogTable.create(type=log_type, params=log_data.model_dump_json())

    def replay(self, snapshot_time: int, bdp_instance: bdp.BDPVectorized) -> None:
        records = (
            LogTable.select()
            .where(LogTable.timestamp > snapshot_time)
            .order_by(LogTable.timestamp.asc())
        )
        for record in records:
            logger.info(f"Replaying log with timestamp: {record.time}")
            log_type = record.type
            params = json.loads(record.params)

            if log_type == "get_pair":
                bdp_instance.get_next_pair()
            else:
                submit_params = ComparisonInputModel(**params)
                bdp_instance.submit_comparison(
                    submit_params.project_ids[0],
                    submit_params.project_ids[1],
                    submit_params.winner_id,
                )
