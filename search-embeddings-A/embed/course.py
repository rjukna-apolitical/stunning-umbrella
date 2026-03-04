"""
Embed Contentful course journey entries into Pinecone.

Courses contain linked journey entries. The overviewLeftColumn and
overviewRightColumn rich text fields of each journey are extracted and combined
for embedding.

Usage:
    python embed/course.py
"""
import logging

from modules.contentful import get_all_entries, getEntry
from modules.embedding import chunk_markdown_by_characters, embed_documents
from modules.pinecone_utils import get_sparse_vectors, upsert_batch
from modules.richtext import extract_values

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "course"


def embed_course():
    # include=0: we only need journey sys IDs from the list — full content is fetched per-journey via getEntry
    courses = get_all_entries(CONTENT_TYPE, include=0)
    vectors = []
    total_upserted = 0

    for course_entry in courses:
        course_id = course_entry.id
        course_slug = course_entry.raw["fields"].get("slug", {}).get("en", "")
        journeys = course_entry.raw["fields"].get("journeys", {}).get("en", [])

        for journey_ref in journeys:
            journey_id = journey_ref["sys"]["id"]
            journey = getEntry(journey_id)
            fields = journey.raw.get("fields", {})
            journey_slug = fields.get("slug", "")
            journey_title = fields.get("title", "")
            access_privacy = fields.get("accessPrivacy", "")

            left = extract_values(fields.get("overviewLeftColumn", {}).get("content", []))
            right = extract_values(fields.get("overviewRightColumn", {}).get("content", []))
            combined_text = " ".join(left + right).strip()

            if not combined_text:
                log.warning("Skipping journey %s: no overview content found", journey_id)
                continue

            log.debug("Processing journey %s (%s)", journey_id, journey_title)
            metadata = {
                "journey_title": journey_title,
                "course_id": course_id,
                "course_slug": course_slug,
                "journey_id": journey_id,
                "journey_slug": journey_slug,
                "access_privacy": access_privacy,
                "type": CONTENT_TYPE,
            }

            col = max(len(k) for k in metadata)
            log.info("  Metadata:")
            for k, v in metadata.items():
                log.info("    %-*s  %s", col, k, v)

            chunks = chunk_markdown_by_characters(journey_id, combined_text)
            texts = [chunk.page_content for chunk in chunks]

            dense_vectors = embed_documents(texts)
            sparse_vectors = get_sparse_vectors(texts)

            for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
                if not sparse_vec.sparse_indices:
                    log.warning("Skipping %s-%d: empty sparse vector", journey_id, i)
                    continue

                vectors.append({
                    "id": f"{journey_id}-{i}",
                    "values": dense_vec.tolist(),
                    "sparse_values": {
                        "indices": sparse_vec.sparse_indices,
                        "values": sparse_vec.sparse_values,
                    },
                    "metadata": {**metadata, "body": chunk.page_content},
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
