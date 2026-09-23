import pandas as pd
from fastapi import UploadFile
from pydantic import BaseModel


class Entity(BaseModel):
    attributes: dict[str, str]

    @staticmethod
    def list_from_csv(raw_csv: UploadFile) -> tuple[list[str], list[Entity]]:
        frame = pd.read_csv(raw_csv.file, dtype=str, keep_default_na=False)
        columns = [str(column) for column in frame.columns]
        entities = [
            Entity(attributes={column: str(row[column]) for column in columns})
            for _, row in frame.iterrows()
        ]
        return columns, entities
