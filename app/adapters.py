from collections import defaultdict
import pathlib
import pickle as pkl
from fastapi import UploadFile
import pandas as pd
from app import constants
from app.models import Project
from typing import List, Tuple
import numpy as np


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

        self.save()

    def __len__(self) -> int:
        return len(self.projects)

    def get_project_from_id(self, id: int) -> Project:
        return self.projects[id]

    def save(self):
        with open(constants.PROJECT_ADAPTER_SAVE_FILE, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls):
        with open(constants.PROJECT_ADAPTER_SAVE_FILE, "rb") as f:
            return pkl.load(f)


class StateManager:
    def __init__(self, n_state: int = 0):
        self.convergence_history = []
        self.judge_map = defaultdict(tuple)

        self.num_save_state = n_state

    def add_alpha_to_history(self, alpha: List[float]):
        self.convergence_history.append(alpha)

    def get_most_recent_alpha(self) -> np.ndarray | None:
        if len(self.convergence_history) == 0:
            return None
        return np.array(self.convergence_history[-1])

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

    def save(self):
        with open(constants.STATE_MANAGER_SAVE_FILE_DIR / constants.STATE_MANAGER_SAVE_FILE_TEMPLATE.format(num_saves=self.num_save_state), "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, filename: pathlib.Path):
        with open(filename, "rb") as f:
            return pkl.load(f)
