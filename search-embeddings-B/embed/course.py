"""
Embed Contentful course journey entries into Pinecone — one vector per locale per chunk.

Courses contain linked journey entries. For each journey, the overviewLeftColumn
and overviewRightColumn rich text fields are extracted (per locale) and combined
for embedding.

Usage:
    python embed/course.py
"""
import logging

from config import BM25_STATS_PATH
from modules.bm25 import SUPPORTED_LOCALES, load_bm25
from modules.contentful import get_all_entries, getEntry
from modules.embedding import chunk_text, embed_dense_passages
from modules.pinecone_utils import upsert_batch
from modules.richtext import extract_values

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "course"


def embed_course():
    bm25 = load_bm25(BM25_STATS_PATH)

    # include=0: only need journey sys IDs from the course list
    courses = get_all_entries(CONTENT_TYPE, include=0)
    vectors = []
    total_upserted = 0

    for course_entry in courses:
        course_id = course_entry.id
        course_fields = course_entry.raw.get("fields", {})
        course_slug_by_locale = course_fields.get("slug", {})
        journeys = course_fields.get("journeys", {}).get("en", [])

        for journey_ref in journeys:
            journey_id = journey_ref["sys"]["id"]

            # Fetch journey with all locales
            journey = getEntry(journey_id)
            fields = journey.raw.get("fields", {})

            journey_title_by_locale = fields.get("title", {})
            journey_slug_by_locale = fields.get("slug", {})
            access_privacy = fields.get("accessPrivacy", {}).get("en", "")

            # Collect locales that have overview content
            left_by_locale = fields.get("overviewLeftColumn", {})
            right_by_locale = fields.get("overviewRightColumn", {})

            all_locales = set(left_by_locale.keys()) | set(right_by_locale.keys())

            for locale in all_locales:
                if locale not in SUPPORTED_LOCALES:
                    continue

                left_content = left_by_locale.get(locale, {}).get("content", [])
                right_content = right_by_locale.get(locale, {}).get("content", [])
                combined_text = " ".join(
                    extract_values(left_content) + extract_values(right_content)
                ).strip()

                if not combined_text:
                    log.warning(
                        "Skipping journey %s [%s]: no overview content", journey_id, locale
                    )
                    continue

                title = journey_title_by_locale.get(locale, journey_title_by_locale.get("en", ""))
                journey_slug = journey_slug_by_locale.get(locale, journey_slug_by_locale.get("en", ""))
                course_slug = course_slug_by_locale.get(locale, course_slug_by_locale.get("en", ""))

                log.debug("Processing journey %s [%s]", journey_id, locale)

                chunks = chunk_text(combined_text)
                dense_vecs = embed_dense_passages(chunks)

                for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vecs)):
                    sparse = bm25.encode_document(chunk, locale)
                    if not sparse["indices"]:
                        log.warning(
                            "Skipping %s::%s::%d: empty sparse vector", journey_id, locale, i
                        )
                        continue

                    vectors.append({
                        "id": f"{journey_id}::{locale}::{i}",
                        "values": dense_vec,
                        "sparse_values": sparse,
                        "metadata": {
                            "content_id": journey_id,
                            "content_type": CONTENT_TYPE,
                            "locale": locale,
                            "title": title,
                            "slug": journey_slug,
                            "snippet": chunk[:300],
                            "journey_id": journey_id,
                            "course_id": course_id,
                            "course_slug": course_slug,
                            "access_privacy": access_privacy,
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
    setup_logging(log_file="course-embedding.log")
    embed_course()
