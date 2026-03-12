"""
Embed Contentful solutionArticle entries into Pinecone — one vector per locale per chunk (dense only).

Usage:
    python embed/article.py
"""

import logging
import time

from config import SUPPORTED_LOCALES
from modules.contentful import get_all_entries, get_banner_url
from modules.embedding import chunk_text, embed_dense_passages
from modules.pinecone_utils import upsert_batch

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "solutionArticle"


def embed_article():
    entries = get_all_entries(CONTENT_TYPE)
    vectors = []
    total_upserted = 0

    for entry in entries:
        entry_id = entry.id
        fields = entry.raw.get("fields", {})

        body_by_locale = fields.get("body", {})
        title_by_locale = fields.get("title", {})
        slug_by_locale = fields.get("slug", {})

        published_date_raw = fields.get("publishedDate", {}).get("en", "")
        published_date = _parse_date(published_date_raw) or int(
            entry.sys["created_at"].timestamp()
        )

        available_locales = list(body_by_locale.keys())
        banner_url = _get_image_url(entry)

        for locale, body in body_by_locale.items():
            if not body or locale not in SUPPORTED_LOCALES:
                continue

            title = title_by_locale.get(locale, title_by_locale.get("en", ""))
            slug = slug_by_locale.get(locale, slug_by_locale.get("en", ""))

            full_text = f"{title}. {title}. {body}"
            chunks = chunk_text(full_text)

            log.debug("Processing %s [%s]: %d chunks", entry_id, locale, len(chunks))

            dense_vecs = embed_dense_passages(chunks)

            for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
                vectors.append(
                    {
                        "id": f"{entry_id}::{locale}::{i}",
                        "values": dense_vec,
                        "metadata": {
                            "content_id": entry_id,
                            "content_type": CONTENT_TYPE,
                            "locale": locale,
                            "title": title,
                            "slug": slug,
                            "snippet": chunk[:300],
                            "published_date": published_date,
                            "available_locales": available_locales,
                            **({"banner_url": banner_url} if banner_url else {}),
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
    """Return image URL for an article: coverImage.url first, bannerImage asset as fallback."""
    try:
        cover = entry.fields("en").get("cover_image")
        if cover and hasattr(cover, "raw"):
            url = cover.raw.get("fields", {}).get("url", {}).get("en", "")
            if url:
                return url
    except Exception:
        pass
    return get_banner_url(entry)


def _parse_date(date_str: str) -> int | None:
    """Convert 'YYYY-MM-DD' string to Unix timestamp int, or None on failure."""
    try:
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    from modules.logger import setup_logging

    setup_logging(log_file="article-embedding.log")
    embed_article()
