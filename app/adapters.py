from collections import defaultdict
from fastapi import UploadFile
import pandas as pd
from app import constants
from app.models import Project
from typing import Tuple
import sqlite3

class ProjectAdapter:
    def __init__(self, raw_csv: UploadFile):
        df = pd.read_csv(raw_csv.file)
        df["Table Number"] = df["Table Number"].fillna("").astype(str)
        self.projects = []
        filtered_df = df[df["Highest Step Completed"] == "Submit"]

        for i, (_, row) in enumerate(filtered_df.iterrows()):
            self.projects.append(Project(
                project_name=row["Project Title"],
                devpost_link=row["Submission Url"],
                table_num=row["Table Number"],
                project_id=i,
                tracks=row["M Hacks Main Track"] if not pd.isna(row["M Hacks Main Track"]) else "No Track"
            ))

    def __len__(self) -> int:
        return len(self.projects)

    def get_project_from_id(self, id: int) -> Project:
        return self.projects[id]

class JudgeManager:
    def __init__(self):
        self.judge_map = defaultdict(Tuple[int, int])

    def add_judge_assignment(self, uuid: str, pair: Tuple[int, int], force: bool = False):
        self.judge_map[uuid] = pair

    def remove_judge_assignment(self, uuid) -> bool:
        try:
            del self.judge_map[uuid]
            return True
        except Exception:
            return False
        
    def verify_judge_assignment(self, uuid: str, left_project_id: int, right_project_id: int):
        return left_project_id in self.judge_map[uuid] and right_project_id in self.judge_map[uuid]
