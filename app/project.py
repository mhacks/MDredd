from urllib.parse import urlsplit

from app import devpost
from app.entity import Entity
from app.exceptions import (
    DevpostUnresolvedException,
    InvalidColumnsException,
    TooFewEntitiesException,
)

# Columns of the Devpost projects export.
TITLE = "Project Title"
SUBMISSION_URL = "Submission Url"
MAIN_TRACK = "M Hacks Main Track"
SPONSOR_PRIZES = "Sponsor Opt In Prizes"
REQUIRED = (TITLE, SUBMISSION_URL, MAIN_TRACK, SPONSOR_PRIZES)

# Added at upload: the public page each submission URL redirects to.
PROJECT_URL = "Project Url"


def submitted(headers: list[str], entities: list[Entity]) -> list[Entity]:
    """Check the export's columns and drop drafts, which have no submission URL."""
    missing = [name for name in REQUIRED if name not in headers]
    if PROJECT_URL in headers:
        missing.append(PROJECT_URL)
    if missing:
        raise InvalidColumnsException(missing)
    kept = [
        entity for entity in entities if entity.attributes[SUBMISSION_URL].strip()
    ]
    if len(kept) < 2:
        raise TooFewEntitiesException()
    return kept


def with_project_urls(entities: list[Entity]) -> list[Entity]:
    """Add each project's resolved Devpost URL, or fail listing every row that did not resolve."""
    results = devpost.resolve_all(
        [entity.attributes[SUBMISSION_URL].strip() for entity in entities]
    )
    failures = [
        {
            "title": entity.attributes[TITLE],
            "submission_url": entity.attributes[SUBMISSION_URL],
            "code": result.code,
        }
        for entity, result in zip(entities, results, strict=True)
        if isinstance(result, devpost.DevpostError)
    ]
    if failures:
        raise DevpostUnresolvedException(failures)
    return [
        Entity(attributes={**entity.attributes, PROJECT_URL: str(result)})
        for entity, result in zip(entities, results, strict=True)
    ]


def normalize_url(url: str) -> str:
    """Compare Devpost links by host and path, ignoring case, www, a trailing slash, the query, and the scheme."""
    try:
        parts = urlsplit(url.strip().lower())
    except ValueError:
        return url.strip().lower()
    host = (parts.hostname or "").removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


def without_project_url(entity: Entity) -> Entity:
    attributes = dict(entity.attributes)
    attributes.pop(PROJECT_URL, None)
    return Entity(attributes=attributes)


def tracks(attributes: dict[str, str]) -> list[str]:
    # Devpost writes the opted-in prizes as a list: "A, B, and C".
    prizes = attributes.get(SPONSOR_PRIZES, "").split(", ")
    if len(prizes) > 1:
        prizes[-1] = prizes[-1].removeprefix("and ")
    named = [attributes.get(MAIN_TRACK, ""), *prizes]
    return [track.strip() for track in named if track.strip()]
