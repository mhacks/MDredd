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
        headers = _header_row(text)
        _require_headers(headers)
        try:
            rows = list(csv.reader(io.StringIO(text)))[1:]
        except csv.Error as exc:
            raise InvalidColumnsException([]) from exc
        # Devpost headers only the first team member, so larger teams' rows run
        # past the header. Blank lines are skipped, short rows padded, and
        # cells past the last header dropped.
        width = len(headers)
        entities = [
            Entity(attributes=dict(zip(headers, [*row, *[""] * (width - len(row))])))
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
    seen: set[str] = set()
    invalid: list[str] = []
    for header in headers:
        if header == "" or header in seen:
            invalid.append(header)
            continue
        seen.add(header)
    if invalid:
        raise InvalidColumnsException(invalid)
