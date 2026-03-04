import logging
from typing import List

import torch
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import HF_TOKEN, EMBEDDING_MODEL

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Loading embedding model %s on %s", EMBEDDING_MODEL, device)

model = SentenceTransformer(EMBEDDING_MODEL, device=device, token=HF_TOKEN)
log.info("Embedding model loaded")


def chunk_markdown_by_characters(
    document_id: str,
    text_body: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    markdown_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "header_1"), ("##", "header_2"), ("###", "header_3")]
    )
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits_by_headers = markdown_header_splitter.split_text(text_body)
    splits_by_characters = character_splitter.split_documents(splits_by_headers)

    chunks = []
    for idx, split in enumerate(splits_by_characters):
        split.metadata = {
            **split.metadata,
            "chunk_id": f"{document_id}#chunk-{idx:05}|size{chunk_size}-overlap{chunk_overlap}",
            "document_id": document_id,
        }
        chunks.append(split)
    return chunks


def embed_documents(texts: List[str], batch_size: int = 32):
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
