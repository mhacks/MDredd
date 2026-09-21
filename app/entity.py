
import pandas as pd
from fastapi import UploadFile
from pydantic import BaseModel


class Entity(BaseModel):
    project_name: str
    devpost_link: str
    table_num: str
    tracks: str

    @staticmethod
    def list_from_csv(raw_csv: UploadFile) -> list[Entity]:
        df = pd.read_csv(raw_csv.file)
        df["Table Number"] = df["Table Number"].fillna("").astype(str)
        entities = []
        filtered_df = df[df["Highest Step Completed"] == "Submit"]

        for i, (_, row) in enumerate(filtered_df.iterrows()):
            track_value = row.get("M Hacks Main Track", None)
            tracks = (
                str(track_value)
                if track_value is not None and not pd.isna(track_value)
                else "No Track"
            )
            entities.append(
                Entity(
                    project_name=str(row["Project Title"]),
                    devpost_link=str(row["Submission Url"]),
                    table_num=str(row["Table Number"]),
                    tracks=tracks,
                )
            )

        return entities
