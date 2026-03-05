import logging
import time
from typing import Generator

import contentful
import contentful.errors

from config import CONTENTFUL_ACCESS_TOKEN, CONTENTFUL_SPACE_ID

log = logging.getLogger(__name__)

client = contentful.Client(
    access_token=CONTENTFUL_ACCESS_TOKEN,
    space_id=CONTENTFUL_SPACE_ID,
    api_url="cdn.eu.contentful.com",
    timeout_s=30,
)

PAGE_SIZE = 200
MAX_RETRIES = 5


def _fetch_page(content_type: str, skip: int, page_size: int, include: int = 2):
    for attempt in range(MAX_RETRIES):
        try:
            return client.entries({
                "content_type": content_type,
                "limit": page_size,
                "skip": skip,
                "locale": "*",
                "include": include,
            })
        except contentful.errors.RateLimitExceededError as e:
            wait = float(getattr(e, "reset_time", 2 ** attempt))
            log.warning("Rate limited (skip=%d). Retrying in %.0fs...", skip, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch {content_type} entries after {MAX_RETRIES} retries (skip={skip})"
    )


def get_all_entries(content_type: str, page_size: int = PAGE_SIZE, include: int = 2) -> Generator:
    """Paginate through all entries for a content type, yielding one entry at a time.

    Uses locale='*' so all localized field values are available on each entry.
    Automatically reduces page_size if Contentful returns a 'Response size too big' error.
    """
    skip = 0
    total = None

    while total is None or skip < total:
        while True:
            try:
                response = _fetch_page(content_type, skip, page_size, include)
                break
            except contentful.errors.BadRequestError as e:
                if "Response size too big" in str(e) and page_size > 1:
                    page_size = max(1, page_size // 2)
                    log.warning("Response too large. Reducing page size to %d and retrying...", page_size)
                else:
                    raise

        if total is None:
            total = response.total
            pages = (total + page_size - 1) // page_size
            log.info(
                "Found %d %s entries (%d pages of %d)",
                total, content_type, pages, page_size,
            )

        entries = list(response)
        for entry in entries:
            yield entry

        skip += len(entries)

        if skip < total:
            time.sleep(0.3)


def get_banner_url(entry) -> str | None:
    """Extract banner image URL from a Contentful entry's bannerImage asset field.

    With locale="*" fetched entries, asset file fields are locale-keyed
    (e.g. {"en": {"url": "//..."}}), so asset.url() doesn't work — we read
    the raw asset fields directly instead.
    Requires include >= 1 at fetch time so the asset is resolved in includes.
    Returns an https:// URL, or None if the field is absent or unresolvable.
    """
    try:
        banner = entry.fields("en").get("banner_image")
        if not banner or not hasattr(banner, "raw"):
            # Absent or unresolved Link (entry fetched with include=0)
            return None
        # With locale="*", file is {"en": {"url": "//...", ...}}
        file_data = banner.raw.get("fields", {}).get("file", {})
        en_file = file_data.get("en", file_data)
        url = en_file.get("url", "") if isinstance(en_file, dict) else ""
        if url:
            return f"https:{url}" if url.startswith("//") else url
    except Exception:
        pass
    return None


def getEntry(entry_id: str):
    for include in [2, 1, 0]:
        for attempt in range(MAX_RETRIES):
            try:
                return client.entry(entry_id, {"include": include, "locale": "*"})
            except contentful.errors.NotFoundError:
                raise  # Entry doesn't exist — no point retrying
            except contentful.errors.BadRequestError as e:
                if "Response size too big" in str(e):
                    log.warning(
                        "Response too large for entry %s (include=%d). Retrying with include=%d...",
                        entry_id, include, include - 1,
                    )
                    break  # try next include level
                raise
            except contentful.errors.RateLimitExceededError as e:
                wait = float(getattr(e, "reset_time", 2 ** attempt))
                log.warning("Rate limited (entry=%s). Retrying in %.0fs...", entry_id, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch entry {entry_id} after {MAX_RETRIES} retries")

    raise RuntimeError(f"Failed to fetch entry {entry_id} — response too large even with include=0")
