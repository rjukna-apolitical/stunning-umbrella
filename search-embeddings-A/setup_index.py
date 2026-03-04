"""
Pinecone index management.

Run this once to create the index and clear the namespace before a full re-embed:
    python setup_index.py
"""
from pinecone import Pinecone, ServerlessSpec, NotFoundException
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE, EMBEDDING_DIMENSION

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    if PINECONE_INDEX_NAME in [i.name for i in pc.list_indexes()]:
        pc.delete_index(PINECONE_INDEX_NAME)

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,  # matches multilingual-e5-large
        metric="dotproduct",            # required for hybrid search
        spec=ServerlessSpec(cloud="gcp", region="europe-west4")
    )
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created")


def clear_namespace():
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"✅ Namespace '{PINECONE_NAMESPACE}' cleared")
    except NotFoundException:
        print(f"ℹ️ Namespace '{PINECONE_NAMESPACE}' doesn't exist yet — nothing to delete")


if __name__ == "__main__":
    create_index()
    clear_namespace()
