"""
Pinecone index management for platform-v3.

Run this once before first embedding:
    python setup_index.py
"""

from pinecone import NotFoundException, Pinecone, ServerlessSpec

from config import (
    DENSE_DIM,
    METRIC,
    MODALITY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    VECTOR_TYPE,
)

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"ℹ️ Index '{PINECONE_INDEX_NAME}' already exists")
        return

    print(f"Creating index '{PINECONE_INDEX_NAME}' with:")
    print(f"  - Modality: {MODALITY}")
    print(f"  - Vector type: {VECTOR_TYPE}")
    print(f"  - Dimensions: {DENSE_DIM}")
    print(f"  - Metric: {METRIC}")
    print("  - Dense model: multilingual-e5-large")

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=DENSE_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud="gcp", region="europe-west4"),
    )
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created")


def clear_namespace():
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"✅ Namespace '{PINECONE_NAMESPACE}' cleared")
    except NotFoundException:
        print(
            f"ℹ️ Namespace '{PINECONE_NAMESPACE}' doesn't exist yet — nothing to delete"
        )


if __name__ == "__main__":
    create_index()
    clear_namespace()
