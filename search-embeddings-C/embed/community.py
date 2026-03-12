"""
Embed Contentful communityPage entries into Pinecone — one vector per chunk (dense only).

Usage:
    python embed/community.py
"""

import logging
import time

from modules.contentful import get_all_entries
from modules.embedding import chunk_text, embed_dense_passages
from modules.pinecone_utils import upsert_batch
from modules.stream import get_feed_text
from modules.summarise import summarise_text

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "community"
FEED_GROUP = "community"
CONTENTFUL_CONTENT_TYPE = "communityPage"


def embed_community():
    entries = get_all_entries(CONTENTFUL_CONTENT_TYPE)
    vectors = []
    total_upserted = 0

    for entry in entries:
        entry_id = entry.id
        fields = entry.raw.get("fields", {})
        created_at = entry.sys["created_at"]

        title_by_locale = fields.get("title", {})
        slug_by_locale = fields.get("slug", {})
        privacy = fields.get("privacy", {}).get("en", "")

        published_date_raw = fields.get("publishedDate", {}).get("en", "")
        published_date = _parse_date(published_date_raw) or int(created_at.timestamp())

        title = title_by_locale.get("en", "")
        slug = slug_by_locale.get("en", "")

        if not slug:
            log.warning("Skipping %s: no slug", entry_id)
            continue

        log.info(
            "Community %s: title=%r slug=%r privacy=%r published=%d",
            entry_id,
            title,
            slug,
            privacy,
            published_date,
        )

        feed_text = get_feed_text(FEED_GROUP, slug)
        log.info("  Feed text length: %d chars", len(feed_text))
        time.sleep(1)

        summary = summarise_text(feed_text, title=title)
        if not summary:
            log.warning("  Skipping %s: no summary generated", entry_id)
            continue

        log.info("  Summary length: %d chars", len(summary))

        full_text = f"{title}. {title}. {summary}"
        chunks = chunk_text(full_text)
        dense_vecs = embed_dense_passages(chunks)

        for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
            vectors.append(
                {
                    "id": f"{entry_id}::en::{i}",
                    "values": dense_vec,
                    "metadata": {
                        "content_id": entry_id,
                        "content_type": CONTENT_TYPE,
                        "slug": slug,
                        "title": title,
                        "snippet": chunk[:300],
                        "published_date": published_date,
                        "access_privacy": privacy,
                    },
                }
            )

            if len(vectors) >= BATCH_SIZE:
                upsert_batch(vectors)
                total_upserted += len(vectors)
                vectors = []

    if vectors:
        upsert_batch(vectors)
        total_upserted += len(vectors)

    log.info("Done. Upserted %d vectors for %s.", total_upserted, CONTENT_TYPE)


def _get_image_url(entry) -> str | None:
    """Extract image URL from a communityPage entry's image asset field."""
    try:
        image = entry.fields("en").get("image")
        if not image or not hasattr(image, "raw"):
            return None
        file_data = image.raw.get("fields", {}).get("file", {})
        en_file = file_data.get("en", file_data)
        url = en_file.get("url", "") if isinstance(en_file, dict) else ""
        if url:
            return f"https:{url}" if url.startswith("//") else url
    except Exception:
        pass
    return None


def _parse_date(date_str: str) -> int | None:
    """Convert 'YYYY-MM-DD' string to Unix timestamp int, or None on failure."""
    try:
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    from modules.logger import setup_logging

    setup_logging(log_file="community-embedding.log")
    embed_community()
