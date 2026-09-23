import keyword
import re
from dataclasses import dataclass

from app.exceptions import InvalidColumnsException

_GRAPHQL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Column:
    field: str
    attr: str
    header: str


def graphql_columns(headers: list[str]) -> list[Column]:
    columns: list[Column] = []
    invalid: list[str] = []
    fields: set[str] = set()
    attrs: set[str] = set()
    for header in headers:
        field = _field_name(header)
        if field is None:
            invalid.append(header)
            continue
        attr = f"{field}_" if keyword.iskeyword(field) else field
        if field in fields or attr in attrs:
            invalid.append(header)
            continue
        fields.add(field)
        attrs.add(attr)
        columns.append(Column(field=field, attr=attr, header=header))
    if invalid:
        raise InvalidColumnsException(invalid)
    return columns


def _field_name(header: str) -> str | None:
    if _valid(header) and header != "id":
        return header
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", header)
    if sanitized[:1].isdigit():
        sanitized = f"_{sanitized}"
    if _valid(sanitized) and sanitized != "id":
        return sanitized
    return None


def _valid(name: str) -> bool:
    return _GRAPHQL_NAME.fullmatch(name) is not None and not name.startswith("__")
