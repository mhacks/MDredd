"""Resolve a Devpost projects export's submission URLs ahead of upload.

    uv run --group preprocess python -m app.preprocess export.csv -o resolved.csv

Each submission URL is opened in a real Chromium, all rows at once by default,
and the public project page it lands on is written as the Project Url column,
so POST /datasets stores the output without contacting Devpost. While
submissions are private, pass --login once to sign in to Devpost in a browser
window; the session is kept in --profile for later runs. Rows that do not
resolve are listed and left blank; run the script on its own output to retry
only those rows. Needs no MDREDD_API_TOKEN.
"""

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import (
    BrowserContext,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from app import devpost, project
from app.entity import Entity
from app.exceptions import JudgingFailure

LOGIN_URL = "https://secure.devpost.com/users/login"
TIMEOUT_MS = 30_000
# Time Cloudflare gets to clear its challenge page before a row fails.
CHALLENGE_TIMEOUT_MS = 20_000
# Only the redirects matter, so skip what the page would render.
SKIPPED_RESOURCES = {"image", "media", "font", "stylesheet"}


async def _launch(
    playwright: Playwright, profile: Path, headed: bool, cookie: str
) -> BrowserContext:
    context = await playwright.chromium.launch_persistent_context(
        profile, headless=not headed, user_agent=devpost.USER_AGENT
    )
    if cookie:
        await context.add_cookies(
            [
                {"name": name, "value": value, "domain": ".devpost.com", "path": "/"}
                for name, _, value in (
                    part.strip().partition("=") for part in cookie.split(";")
                )
                if name
            ]
        )
    return context


async def _login(playwright: Playwright, profile: Path, cookie: str) -> None:
    context = await _launch(playwright, profile, headed=True, cookie=cookie)
    page = context.pages[0] if context.pages else await context.new_page()
    await page.goto(LOGIN_URL)
    await asyncio.to_thread(
        input, "Log in to Devpost in the browser window, then press Enter here. "
    )
    await context.close()


async def _resolve(context: BrowserContext, url: str) -> str:
    """Open a submission URL and return the project page it lands on."""
    if not devpost.is_devpost_url(url):
        raise devpost.DevpostError("INVALID_DEVPOST_URL")
    page = await context.new_page()
    try:
        await page.route(
            "**/*",
            lambda route: (
                route.abort()
                if route.request.resource_type in SKIPPED_RESOURCES
                else route.continue_()
            ),
        )
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=TIMEOUT_MS
            )
            if response is not None and response.status in (403, 429, 503):
                # Cloudflare's challenge reloads the page once it passes.
                async with page.expect_navigation(
                    wait_until="domcontentloaded", timeout=CHALLENGE_TIMEOUT_MS
                ) as navigation:
                    pass
                response = await navigation.value
        except PlaywrightTimeoutError as exc:
            raise devpost.DevpostError("DEVPOST_UNAVAILABLE") from exc
        except PlaywrightError as exc:
            raise devpost.DevpostError("DEVPOST_UNAVAILABLE") from exc
        landed = page.url
    finally:
        await page.close()
    if devpost.is_login_page(landed):
        raise devpost.DevpostError("DEVPOST_LOGIN_REQUIRED")
    if not devpost.is_devpost_url(landed):
        raise devpost.DevpostError("DEVPOST_REDIRECTED_OFFSITE")
    if response is not None and response.status == 404:
        raise devpost.DevpostError("DEVPOST_NOT_FOUND")
    if response is None or not response.ok:
        raise devpost.DevpostError("DEVPOST_UNAVAILABLE")
    if not urlsplit(landed).path.startswith("/software/"):
        raise devpost.DevpostError("DEVPOST_NOT_PROJECT_PAGE")
    return landed


async def _resolve_all(
    entities: list[Entity],
    known: list[str],
    profile: Path,
    headed: bool,
    cookie: str,
    concurrency: int,
) -> list[str | devpost.DevpostError]:
    """Each row's Devpost URL, in order. Rows with a known URL are not opened."""
    invalid = devpost.DevpostError("INVALID_DEVPOST_URL")
    results: list[str | devpost.DevpostError] = [
        url if devpost.is_devpost_url(url) else invalid for url in known
    ]
    pending = [index for index, url in enumerate(known) if not url]
    print(
        f"{len(entities)} submitted projects, {len(pending)} to resolve",
        file=sys.stderr,
    )
    if not pending:
        return results

    limit = asyncio.Semaphore(concurrency)
    done = 0
    async with async_playwright() as playwright:
        context = await _launch(playwright, profile, headed, cookie)

        async def attempt(index: int) -> None:
            nonlocal done
            attributes = entities[index].attributes
            async with limit:
                try:
                    results[index] = await _resolve(
                        context, attributes[project.SUBMISSION_URL].strip()
                    )
                except devpost.DevpostError as exc:
                    results[index] = exc
            done += 1
            outcome = results[index]
            shown = (
                outcome.code if isinstance(outcome, devpost.DevpostError) else outcome
            )
            print(
                f"[{done}/{len(pending)}] {attributes[project.TITLE]}: {shown}",
                file=sys.stderr,
            )

        try:
            await asyncio.gather(*(attempt(index) for index in pending))
        finally:
            await context.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.preprocess",
        description="Fill in each row's Project Url so the upload skips Devpost.",
    )
    parser.add_argument("export", type=Path, help="Devpost projects export (CSV)")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="where to write the CSV"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="open a browser window to sign in to Devpost before resolving",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(".devpost-profile"),
        help="browser profile that keeps the Devpost session (default: .devpost-profile)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="show the browser while resolving, e.g. to pass a Cloudflare check",
    )
    parser.add_argument(
        "--cookie",
        default=os.environ.get("MDREDD_DEVPOST_COOKIE", ""),
        help=(
            "Cookie header of a logged-in Devpost session, an alternative to "
            "--login (default: $MDREDD_DEVPOST_COOKIE)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="pages open at once (default: every row at once)",
    )
    args = parser.parse_args()
    if args.concurrency < 0:
        parser.error("--concurrency must not be negative")

    try:
        headers, entities = Entity.list_from_csv(args.export.read_bytes())
        entities = project.submitted(headers, entities)
    except JudgingFailure as exc:
        names = getattr(exc, "names", [])
        print(f"{exc.code}: {', '.join(names)}" if names else exc.code, file=sys.stderr)
        return 2

    known = [
        entity.attributes.get(project.PROJECT_URL, "").strip() for entity in entities
    ]
    headers = [name for name in headers if name != project.PROJECT_URL]

    async def run() -> list[str | devpost.DevpostError]:
        if args.login:
            async with async_playwright() as playwright:
                await _login(playwright, args.profile, args.cookie)
        return await _resolve_all(
            entities,
            known,
            args.profile,
            args.headed,
            args.cookie,
            args.concurrency or len(entities),
        )

    results = asyncio.run(run())

    failures = [
        project.failure(entity, result)
        for entity, result in zip(entities, results, strict=True)
        if isinstance(result, devpost.DevpostError)
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*headers, project.PROJECT_URL])
        for entity, result in zip(entities, results, strict=True):
            url = "" if isinstance(result, devpost.DevpostError) else result
            writer.writerow([*(entity.attributes[name] for name in headers), url])

    for row in failures:
        print(
            f"{row['code']}: {row['title']} ({row['submission_url']})",
            file=sys.stderr,
        )
    if failures:
        print(
            f"{len(failures)} of {len(entities)} left blank in {args.output}. "
            "Run again on that file to retry them; add --login if "
            "DEVPOST_LOGIN_REQUIRED, or --headed if DEVPOST_UNAVAILABLE.",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
