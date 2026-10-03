import csv
import io

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
        header_row = _header_row(text)
        # A spreadsheet that re-saves the export pads every row, header
        # included, with unnamed columns. They have no name to store a value
        # under, so they are dropped rather than rejected.
        named = [index for index, name in enumerate(header_row) if name.strip()]
        headers = [header_row[index] for index in named]
        _require_headers(headers)
        try:
            rows = list(csv.reader(io.StringIO(text)))[1:]
        except csv.Error as exc:
            raise InvalidColumnsException([]) from exc
        # Devpost headers only the first team member, so larger teams' rows run
        # past the header. Blank lines are skipped, short rows padded, and
        # cells past the last header dropped.
        entities = [
            Entity(
                attributes={
                    headers[position]: row[index] if index < len(row) else ""
                    for position, index in enumerate(named)
                }
            )
            for row in rows
            if any(cell.strip() for cell in row)
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
    if not headers:
        raise InvalidColumnsException([])
    seen: set[str] = set()
    duplicates: list[str] = []
    for header in headers:
        if header in seen:
            duplicates.append(header)
            continue
        seen.add(header)
    if duplicates:
        raise InvalidColumnsException(duplicates)
