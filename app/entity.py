import csv
import io

import pandas as pd
from pydantic import BaseModel

from app.exceptions import InvalidColumnsException, TooFewEntitiesException


class Entity(BaseModel):
    attributes: dict[str, str]

    @staticmethod
    def list_from_csv(raw_csv: bytes) -> tuple[list[str], list[Entity]]:
        try:
            text = raw_csv.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidColumnsException([]) from exc
        headers = _header_row(text)
        _require_headers(headers)
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError as exc:
            raise TooFewEntitiesException() from exc
        except pd.errors.ParserError as exc:
            raise InvalidColumnsException([]) from exc
        if [str(column) for column in frame.columns] != headers:
            raise InvalidColumnsException(headers)
        entities = [
            Entity(attributes={header: str(row[header]) for header in headers})
            for _, row in frame.iterrows()
        ]
        return headers, entities


def _header_row(text: str) -> list[str]:
    try:
        return next(csv.reader(io.StringIO(text)))
    except StopIteration as exc:
        raise TooFewEntitiesException() from exc
    except csv.Error as exc:
        raise InvalidColumnsException([]) from exc


def _require_headers(headers: list[str]) -> None:
    seen: set[str] = set()
    invalid: list[str] = []
    for header in headers:
        if header == "" or header in seen:
            invalid.append(header)
            continue
        seen.add(header)
    if invalid:
        raise InvalidColumnsException(invalid)
