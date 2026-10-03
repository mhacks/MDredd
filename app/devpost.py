from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit

import httpx

from app.settings import settings

MAX_REDIRECTS = 10
TIMEOUT_SECONDS = 10
# Devpost sits behind Cloudflare, which rejects bare or library user agents.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


class DevpostError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def is_devpost_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (
        host == "devpost.com" or host.endswith(".devpost.com")
    )


def _is_login_page(url: str) -> bool:
    parts = urlsplit(url)
    return parts.hostname == "secure.devpost.com" and parts.path.startswith(
        "/users/login"
    )


def resolve(client: httpx.Client, url: str) -> str:
    """Follow a Devpost link's redirects and return the URL it lands on.

    Every hop must stay on https devpost.com, so the request (and the session
    cookie, when configured) never leaves Devpost.
    """
    if not is_devpost_url(url):
        raise DevpostError("INVALID_DEVPOST_URL")
    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if _is_login_page(current):
                raise DevpostError("DEVPOST_LOGIN_REQUIRED")
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    current = urljoin(current, response.headers.get("location", ""))
                    if not is_devpost_url(current):
                        raise DevpostError("DEVPOST_REDIRECTED_OFFSITE")
                    continue
                if response.status_code == httpx.codes.NOT_FOUND:
                    raise DevpostError("DEVPOST_NOT_FOUND")
                if not response.is_success:
                    raise DevpostError("DEVPOST_UNAVAILABLE")
                return current
    except httpx.HTTPError as exc:
        raise DevpostError("DEVPOST_UNAVAILABLE") from exc
    raise DevpostError("DEVPOST_TOO_MANY_REDIRECTS")


def resolve_all(urls: list[str]) -> list[str | DevpostError]:
    """Resolve each link, in order. A failed link is its DevpostError."""
    headers = {"User-Agent": USER_AGENT}
    if settings.DEVPOST_COOKIE:
        headers["Cookie"] = settings.DEVPOST_COOKIE

    with httpx.Client(
        headers=headers, timeout=TIMEOUT_SECONDS, follow_redirects=False
    ) as client:

        def attempt(url: str) -> str | DevpostError:
            try:
                return resolve(client, url)
            except DevpostError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=settings.DEVPOST_CONCURRENCY) as pool:
            return list(pool.map(attempt, urls))
