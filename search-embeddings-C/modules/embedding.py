import logging

from pinecone import Pinecone

from config import DENSE_MODEL, PINECONE_API_KEY

log = logging.getLogger(__name__)

pc = Pinecone(api_key=PINECONE_API_KEY)


def embed_dense_passages(texts: list[str]) -> list[list[float]]:
    """Embed a batch of passages using Pinecone-hosted multilingual-e5-large."""
    embeddings = pc.inference.embed(
        model=DENSE_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [e.values for e in embeddings]


def chunk_text(text: str, max_words: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + max_words]))
        start += max_words - overlap
    return chunks
