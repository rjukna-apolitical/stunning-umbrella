"""
Embed Contentful event entries into Pinecone — one vector per locale per chunk.

Usage:
    python embed/event.py
"""
import logging

from config import BM25_STATS_PATH
from modules.bm25 import SUPPORTED_LOCALES, load_bm25
from modules.contentful import get_all_entries
from modules.embedding import chunk_text, embed_dense_passages
from modules.pinecone_utils import upsert_batch

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "event"


def embed_event():
    bm25 = load_bm25(BM25_STATS_PATH)

    entries = get_all_entries(CONTENT_TYPE)
    vectors = []
    total_upserted = 0

    for entry in entries:
        entry_id = entry.id
        fields = entry.raw.get("fields", {})
        created_at = entry.sys["created_at"]

        description_by_locale = fields.get("description", {})
        title_by_locale = fields.get("title", {})
        slug_by_locale = fields.get("slug", {})

        available_locales = list(description_by_locale.keys())

        for locale, description in description_by_locale.items():
            if not description or locale not in SUPPORTED_LOCALES:
                continue

            title = title_by_locale.get(locale, title_by_locale.get("en", ""))
            slug = slug_by_locale.get(locale, slug_by_locale.get("en", ""))

            full_text = f"{title}. {title}. {description}"
            chunks = chunk_text(full_text)

            log.debug("Processing %s [%s]: %d chunks", entry_id, locale, len(chunks))

            dense_vecs = embed_dense_passages(chunks)

            for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
                sparse = bm25.encode_document(chunk, locale)
                if not sparse["indices"]:
                    log.warning("Skipping %s::%s::%d: empty sparse vector", entry_id, locale, i)
                    continue

                vectors.append({
                    "id": f"{entry_id}::{locale}::{i}",
                    "values": dense_vec,
                    "sparse_values": sparse,
                    "metadata": {
                        "content_id": entry_id,
                        "content_type": CONTENT_TYPE,
                        "locale": locale,
                        "title": title,
                        "slug": slug,
                        "snippet": chunk[:300],
                        "published_date": int(created_at.timestamp()),
                        "available_locales": available_locales,
                    },
                })

                if len(vectors) >= BATCH_SIZE:
                    upsert_batch(vectors)
                    total_upserted += len(vectors)
                    vectors = []

    if vectors:
        upsert_batch(vectors)
        total_upserted += len(vectors)

    log.info("Done. Upserted %d vectors for %s.", total_upserted, CONTENT_TYPE)


if __name__ == "__main__":
    from modules.logger import setup_logging
    setup_logging(log_file="event-embedding.log")
    embed_event()
