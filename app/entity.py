import math

import pandas as pd
from fastapi import UploadFile
from pydantic import BaseModel


def _text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


class Entity(BaseModel):
    project_name: str
    devpost_link: str
    table_num: str
    tracks: str

    @staticmethod
    def list_from_csv(raw_csv: UploadFile) -> list[Entity]:
        df = pd.read_csv(raw_csv.file)
        df["Table Number"] = df["Table Number"].fillna("").astype(str)
        entities: list[Entity] = []
        filtered_df = df[df["Highest Step Completed"] == "Submit"]

        for _, row in filtered_df.iterrows():
            entities.append(
                Entity(
                    project_name=_text(row["Project Title"]),
                    devpost_link=_text(row["Submission Url"]),
                    table_num=_text(row["Table Number"]),
                    tracks=_text(row.get("M Hacks Main Track")) or "No Track",
                )
            )

        return entities
