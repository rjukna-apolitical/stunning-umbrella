"""
Embed Contentful solutionArticle entries into Pinecone.

Usage:
    python embed/article.py
"""

import logging

from modules.contentful import get_all_entries
from modules.embedding import chunk_markdown_by_characters, embed_documents
from modules.metadata import build_locale_metadata
from modules.pinecone_utils import get_sparse_vectors, upsert_batch

log = logging.getLogger(__name__)

BATCH_SIZE = 64
CONTENT_TYPE = "solutionArticle"


def embed_article():
    entries = get_all_entries(CONTENT_TYPE)
    vectors = []
    total_upserted = 0

    for entry in entries:
        entry_id = entry.id

        body = entry.raw["fields"].get("body", {}).get("en")
        if not body:
            log.warning("Skipping %s: no body found", entry_id)
            continue

        log.debug("Processing %s", entry_id)
        created_at = entry.sys["created_at"]
        metadata = build_locale_metadata(
            entry.raw, fields_to_include={"title", "subtitle", "slug"}
        )
        chunks = chunk_markdown_by_characters(entry_id, body)
        texts = [chunk.page_content for chunk in chunks]

        dense_vectors = embed_documents(texts)
        sparse_vectors = get_sparse_vectors(texts)

        for i, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            if not sparse_vec.sparse_indices:
                log.warning("Skipping %s-%d: empty sparse vector", entry_id, i)
                continue

            vectors.append(
                {
                    "id": f"{entry_id}-{i}",
                    "values": dense_vec.tolist(),
                    "sparse_values": {
                        "indices": sparse_vec.sparse_indices,
                        "values": sparse_vec.sparse_values,
                    },
                    "metadata": {
                        "entry_id": entry_id,
                        "type": CONTENT_TYPE,
                        "publishedDate": entry.raw["fields"]
                        .get("publishedDate", {})
                        .get("en")
                        or str(created_at),
                        "authorIds": entry.raw["fields"].get("authorIds", {}).get("en")
                        or [],
                        "body": chunk.page_content,
                        **metadata,
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


if __name__ == "__main__":
    from modules.logger import setup_logging

    setup_logging(log_file="article-embedding.log")
    embed_article()
