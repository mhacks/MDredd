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
    attributes: dict[str, str]

    @staticmethod
    def list_from_csv(raw_csv: UploadFile) -> list[Entity]:
        frame = pd.read_csv(raw_csv.file, dtype=str, keep_default_na=False)
        columns = [str(column) for column in frame.columns]
        entities: list[Entity] = []
        for _, row in frame.iterrows():
            entities.append(Entity(attributes={column: _text(row[column]) for column in columns}))
        return entities
