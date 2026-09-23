import io
import math

import pandas as pd
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
    def list_from_csv(raw_csv: bytes) -> list[Entity]:
        frame = pd.read_csv(io.BytesIO(raw_csv), dtype=str, keep_default_na=False)
        columns = [str(column) for column in frame.columns]
        entities: list[Entity] = []
        for _, row in frame.iterrows():
            entities.append(Entity(attributes={column: _text(row[column]) for column in columns}))
        return entities
