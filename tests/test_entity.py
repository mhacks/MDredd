import pytest
from pandas.errors import EmptyDataError

from app.entity import Entity


def test_csv_preserves_headers_and_string_values() -> None:
    raw = (
        'name,notes,code,missing\n'
        'Alice,"contains, comma",001,\n'
        'Bob,"line one\nline two",NA,value\n'
    ).encode()

    headers, entities = Entity.list_from_csv(raw)

    assert headers == ["name", "notes", "code", "missing"]
    assert [entity.attributes for entity in entities] == [
        {
            "name": "Alice",
            "notes": "contains, comma",
            "code": "001",
            "missing": "",
        },
        {
            "name": "Bob",
            "notes": "line one\nline two",
            "code": "NA",
            "missing": "value",
        },
    ]


def test_csv_with_headers_only_has_no_entities() -> None:
    headers, entities = Entity.list_from_csv(b"name,description\n")

    assert headers == ["name", "description"]
    assert entities == []


def test_csv_preserves_unicode_and_whitespace() -> None:
    headers, entities = Entity.list_from_csv(
        "label,value\n\u6771\u4eac, spaced value \nemoji,\U0001f680\n".encode()
    )

    assert headers == ["label", "value"]
    assert [entity.attributes for entity in entities] == [
        {"label": "\u6771\u4eac", "value": " spaced value "},
        {"label": "emoji", "value": "\U0001f680"},
    ]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\n",
    ],
)
def test_csv_without_headers_is_rejected(raw: bytes) -> None:
    with pytest.raises(EmptyDataError):
        Entity.list_from_csv(raw)
