from collections import defaultdict
from fastapi import UploadFile
import pandas as pd
from app import constants
from app.models import ComparisonInputModel, PairRequestModel, Project
from typing import Tuple, List, Dict
import sqlite3
from dredd import bdp
import time
import json
import logging

logger = logging.getLogger("uvicorn")


class ProjectAdapter:
    def __init__(self):
        self.conn = sqlite3.connect(constants.DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.projects = []

        # Setup projects table
        create_projects_table = """
            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY,
                project_name TEXT NOT NULL,
                devpost_link TEXT NOT NULL,
                table_num TEXT NOT NULL,
                tracks TEXT NOT NULL
            );
        """
        self.cursor.execute(create_projects_table)

    def __del__(self) -> None:
        self.conn.close()

    def clear(self):
        self.cursor.execute("DELETE FROM projects")

    def load_projects(self, raw_csv: UploadFile = None):
        if raw_csv is not None:
            # Clear old projects
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
                        project_id=i,
                        tracks=track_value
                        if track_value is not None and not pd.isna(track_value)
                        else "No Track",
                    )
                )

            # Load new projects
            projects_insert_stmt = """
                INSERT INTO projects (
                    project_id,
                    project_name,
                    devpost_link,
                    table_num,
                    tracks
                ) VALUES (?, ?, ?, ?, ?)
            """

            rows = [
                (
                    p.project_id,
                    p.project_name,
                    p.devpost_link,
                    p.table_num,
                    p.tracks,
                )
                for p in projects
            ]

            self.cursor.executemany(projects_insert_stmt, rows)
            self.conn.commit()

        self.projects = self.get_projects()

    def get_projects(self) -> List[Project]:
        self.cursor.execute("SELECT * FROM projects ORDER BY project_id")
        rows = self.cursor.fetchall()
        logger.info(f"Fetched {len(rows)} projects")

        projects = []
        for row in rows:
            project = Project(
                project_id=row[0],
                project_name=row[1],
                devpost_link=row[2],
                table_num=row[3],
                tracks=row[4] if row[4] is not None else "No Track",
            )
            projects.append(project)

        return projects

    def get_project_from_id(self, id: int) -> Project:
        return self.projects[id]


class SnapshotAdapter:
    def __init__(self):
        self.conn = sqlite3.connect(constants.DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Setup snapshot table
        create_snapshot_table = """
            CREATE TABLE IF NOT EXISTS snapshots (
                timestamp INTEGER PRIMARY KEY,
                judge_map TEXT NOT NULL,
                bdp TEXT NOT NULL
            );
        """
        self.cursor.execute(create_snapshot_table)
        self.conn.commit()
        self.judge_map = defaultdict(Tuple[int, int])

    def __del__(self) -> None:
        self.conn.close()

    def clear(self):
        self.cursor.execute("DELETE FROM snapshots")
        self.conn.commit()

    def snapshot(self, bdp_instance: bdp.BDPVectorized):
        snapshot_data = {
            "timestamp": int(time.time()),
            "judge_map": json.dumps(dict(self.judge_map)),
            "bdp": bdp_instance.model_dump_json(),
        }

        self.cursor.execute(
            "INSERT INTO snapshots (timestamp, judge_map, bdp) VALUES (?, ?, ?)",
            (
                snapshot_data["timestamp"],
                snapshot_data["judge_map"],
                snapshot_data["bdp"],
            ),
        )
        self.conn.commit()

    def load_snapshot(self) -> Tuple[int, bdp.BDPVectorized | None]:
        snapshot_record = self.cursor.execute(
            "SELECT * FROM snapshots ORDER BY timestamp DESC"
        ).fetchone()
        if not snapshot_record:
            return (0, None)

        timestamp = snapshot_record[0]
        logger.info(f"Loaded snapshot from timestamp: {timestamp}")
        judge_map = json.loads(snapshot_record[1])
        bdp_data = json.loads(snapshot_record[2])

        self.judge_map = judge_map
        return (timestamp, bdp.BDPVectorized(**bdp_data))

    def remove_judge_assignment(self, uuid) -> bool:
        try:
            del self.judge_map[uuid]
            return True
        except Exception:
            return False

    def verify_judge_assignment(
        self, uuid: str, left_project_id: int, right_project_id: int
    ):
        return (
            left_project_id in self.judge_map[uuid]
            and right_project_id in self.judge_map[uuid]
        )


class LogAdapter:
    def __init__(self):
        self.conn = sqlite3.connect(constants.DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Setup logs table
        create_logs_table = """
            CREATE TABLE IF NOT EXISTS logs (
                timestamp INTEGER PRIMARY KEY,
                type TEXT NOT NULL CHECK (type IN ('get_pair', 'submit_pair')),
                params TEXT NOT NULL
            );
        """

        self.cursor.execute(create_logs_table)
        self.conn.commit()

    def __del__(self) -> None:
        self.conn.close()

    def clear(self):
        self.cursor.execute("DELETE FROM logs")
        self.conn.commit()

    def log(self, log_data: ComparisonInputModel | PairRequestModel):
        log_type = (
            "submit_pair" if isinstance(log_data, ComparisonInputModel) else "get_pair"
        )

        self.cursor.execute(
            "INSERT INTO logs (timestamp, type, params) VALUES (?, ?, ?)",
            (int(time.time()), log_type, log_data.model_dump_json()),
        )
        self.conn.commit()

    def replay(
        self,
        snapshot_time: int,
        bdp_instance: bdp.BDPVectorized,
        judge_map: Dict[str, Tuple[int, int]],
    ) -> None:
        records = self.cursor.execute(
            "SELECT * FROM logs WHERE timestamp > (?) ORDER BY timestamp ASC",
            (snapshot_time,),
        )
        for record in records:
            logger.info(f"Replaying log with timestamp: {record[0]}")
            log_type = record[1]
            params = json.loads(record[2])

            if log_type == "get_pair":
                pair_params = PairRequestModel(**params)
                i, j = bdp_instance.get_next_pair()
                judge_map[pair_params.uuid] = (i, j)
            else:
                submit_params = ComparisonInputModel(**params)
                bdp_instance.submit_comparison(
                    submit_params.project_ids[0],
                    submit_params.project_ids[1],
                    submit_params.winner_id,
                )
                del judge_map[submit_params.uuid]
