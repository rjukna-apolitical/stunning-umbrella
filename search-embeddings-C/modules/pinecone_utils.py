import logging

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

log = logging.getLogger(__name__)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def upsert_batch(vectors: list):
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    log.info("Upserted batch of %d vectors", len(vectors))


def delete_by_type(content_type: str):
    """Delete all vectors for a given content type using metadata filter."""
    index.delete(
        filter={"content_type": {"$eq": content_type}},
        namespace=PINECONE_NAMESPACE,
    )
    log.info(
        "Deleted all vectors with content_type='%s' from namespace '%s'",
        content_type, PINECONE_NAMESPACE,
    )
