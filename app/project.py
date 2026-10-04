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
    if missing:
        raise InvalidColumnsException(missing)
    kept = [entity for entity in entities if entity.attributes[SUBMISSION_URL].strip()]
    if len(kept) < 2:
        raise TooFewEntitiesException()
    return kept


def project_urls(
    entities: list[Entity], known: list[str], cookie: str, concurrency: int
) -> list[str | devpost.DevpostError]:
    """Each row's Devpost URL, in order. A row that did not resolve is its DevpostError.

    `known` is each row's Project Url from the upload, looked up ahead of time.
    Only rows where it is blank are resolved here.
    """
    pending = [index for index, url in enumerate(known) if not url]
    resolved = devpost.resolve_all(
        [entities[index].attributes[SUBMISSION_URL].strip() for index in pending],
        cookie,
        concurrency,
    )
    invalid = devpost.DevpostError("INVALID_DEVPOST_URL")
    results: list[str | devpost.DevpostError] = [
        url if devpost.is_devpost_url(url) else invalid for url in known
    ]
    for index, result in zip(pending, resolved, strict=True):
        results[index] = result
    return results


def failure(entity: Entity, error: devpost.DevpostError) -> dict[str, str]:
    return {
        "title": entity.attributes[TITLE],
        "submission_url": entity.attributes[SUBMISSION_URL],
        "code": error.code,
    }


def with_project_urls(
    entities: list[Entity], known: list[str], cookie: str, concurrency: int
) -> list[Entity]:
    """Add each project's Devpost URL, or fail listing every row that did not resolve."""
    results = project_urls(entities, known, cookie, concurrency)
    failures = [
        failure(entity, result)
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
    # Devpost writes the opted-in prizes as a list: "A", "A and B", or
    # "A, B, and C".
    listed = attributes.get(SPONSOR_PRIZES, "")
    prizes = listed.split(", ")
    if len(prizes) > 1:
        prizes[-1] = prizes[-1].removeprefix("and ")
    else:
        prizes = listed.split(" and ")
    named = [attributes.get(MAIN_TRACK, ""), *prizes]
    return [track.strip() for track in named if track.strip()]
