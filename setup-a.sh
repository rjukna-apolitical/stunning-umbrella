#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Colab → Zed IDE Project Setup Script
# Project: Contentful + Pinecone Embedding Pipeline
# Uses: uv + Python 3.12
# ============================================================

PROJECT_NAME="contentful-pinecone-embeddings-A"
PYTHON_VERSION="3.12"
KERNEL_DISPLAY_NAME="Python ($PROJECT_NAME)"

echo "============================================================"
echo "  🚀 Setting up Zed IDE project: $PROJECT_NAME"
echo "============================================================"

# ----------------------------------------------------------
# 0. Ensure uv is installed
# ----------------------------------------------------------
echo ""
echo "🔍 Step 0: Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "   → uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env so uv is available in this session
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    echo "   → uv installed: $(uv --version)"
else
    echo "   → uv found: $(uv --version)"
fi

# ----------------------------------------------------------
# 1. Create project with uv
# ----------------------------------------------------------
echo ""
echo "📁 Step 1: Initializing uv project..."
uv init "$PROJECT_NAME" --python "$PYTHON_VERSION"
cd "$PROJECT_NAME"
echo "   → Project created at ./$PROJECT_NAME"

# ----------------------------------------------------------
# 2. Pin Python 3.12 explicitly
# ----------------------------------------------------------
echo ""
echo "🐍 Step 2: Pinning Python $PYTHON_VERSION..."
uv python pin "$PYTHON_VERSION"
echo "   → Python $PYTHON_VERSION pinned"

# ----------------------------------------------------------
# 3. Add project dependencies
# ----------------------------------------------------------
echo ""
echo "📦 Step 3: Adding dependencies..."
uv add \
  ipykernel \
  scipy \
  numpy \
  contentful \
  pinecone-client \
  sentence-transformers \
  langchain \
  langchain-text-splitters \
  langchain-core \
  langchain-pinecone \
  langchain-community \
  torch \
  python-dotenv

echo "   → All dependencies added to pyproject.toml and installed"

# ----------------------------------------------------------
# 4. Register Jupyter kernel for Zed REPL
# ----------------------------------------------------------
echo ""
echo "🔧 Step 4: Registering Jupyter kernel..."
uv run python -m ipykernel install \
  --user \
  --name "$PROJECT_NAME" \
  --display-name "$KERNEL_DISPLAY_NAME"
echo "   → Kernel '$KERNEL_DISPLAY_NAME' registered"

# ----------------------------------------------------------
# 5. Create .env file for secrets
# ----------------------------------------------------------
echo ""
echo "🔐 Step 5: Creating .env file for secrets..."
cat > .env << 'ENVEOF'
# ============================================================
# Environment Variables (replace with your actual values)
# ============================================================
PINECONE_API_KEY=your_pinecone_api_key_here
CONTENTFUL_ACCESS_TOKEN=your_contentful_access_token_here
CONTENTFUL_SPACE_ID=your_contentful_space_id_here
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
ENVEOF
echo "   → .env file created (edit with your real keys!)"

# ----------------------------------------------------------
# 6. Update .gitignore
# ----------------------------------------------------------
echo ""
echo "📝 Step 6: Updating .gitignore..."
cat >> .gitignore << 'GIEOF'

# Secrets
.env

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
dist/
build/
GIEOF
echo "   → .gitignore updated"

# ----------------------------------------------------------
# 7. Create Zed project settings
# ----------------------------------------------------------
echo ""
echo "⚙️  Step 7: Creating Zed project settings..."
mkdir -p .zed
cat > .zed/settings.json << ZEDEOF
{
  "jupyter": {
    "kernel_selections": {
      "python": "$PROJECT_NAME"
    }
  }
}
ZEDEOF
echo "   → .zed/settings.json created"

# ----------------------------------------------------------
# 8. Generate main.py (Zed REPL cell format)
# ----------------------------------------------------------
echo ""
echo "📄 Step 8: Generating main.py..."

# Remove the default hello.py that uv init creates
rm -f hello.py

cat > 'main.py' << 'PYEOF'
# %% Cell 0 - Environment Setup & Secrets
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CONTENTFUL_ACCESS_TOKEN = os.getenv("CONTENTFUL_ACCESS_TOKEN")
CONTENTFUL_SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

assert PINECONE_API_KEY, "❌ PINECONE_API_KEY not set in .env"
assert CONTENTFUL_ACCESS_TOKEN, "❌ CONTENTFUL_ACCESS_TOKEN not set in .env"
assert CONTENTFUL_SPACE_ID, "❌ CONTENTFUL_SPACE_ID not set in .env"
assert HF_TOKEN, "❌ HF_TOKEN not set in .env"
print("✅ All environment variables loaded")

# %% Cell 1 - Create Pinecone Index
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "platform"

# Delete and recreate with correct dimensions
if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    pc.delete_index(INDEX_NAME)

pc.create_index(
    name=INDEX_NAME,
    dimension=1024,  # matches multilingual-e5-large
    metric="dotproduct",  # required for hybrid search
    spec=ServerlessSpec(cloud="gcp", region="europe-west4")
)

index = pc.Index(INDEX_NAME)
print(f"✅ Pinecone index '{INDEX_NAME}' created")

# %% Cell 1.1 - Clear search namespace (optional)
from pinecone import Pinecone, NotFoundException

INDEX_NAME = "platform"
NAMESPACE_NAME = "search"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

try:
    index.delete(delete_all=True, namespace=NAMESPACE_NAME)
    print(f"✅ Namespace {NAMESPACE_NAME} cleared")
except NotFoundException:
    print(f"ℹ️ Namespace {NAMESPACE_NAME} doesn't exist yet — nothing to delete")

# %% Cell 2 - Import libraries & load embedding model
import os
import re
import contentful
import torch
import pinecone
from pinecone import Pinecone
from pprint import pprint
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device=device,
    token=HF_TOKEN
)
print("✅ Embedding model loaded")

# %% Cell 3 - Define Contentful client and helpers
client = contentful.Client(
    access_token=CONTENTFUL_ACCESS_TOKEN,
    space_id=CONTENTFUL_SPACE_ID,
    api_url='cdn.eu.contentful.com'
)


def getEntries(content_type, limit=15):
    entries = client.entries({
        "content_type": content_type,
        "limit": limit,
        "locale": "*",
        "include": 2
    })
    return entries


def getEntry(entry_id):
    entry = client.entry(entry_id, {"include": 2})
    return entry


print("✅ Contentful client initialized")

# %% Cell 4 - Metadata helpers
def normalize_locale(locale: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", locale.lower()).strip("_")


def build_locale_metadata(entry_raw: dict, fields_to_include=None) -> dict:
    out = {}
    fields = entry_raw.get("fields", {})
    for field_name, localized_values in fields.items():
        if fields_to_include and field_name not in fields_to_include:
            continue
        if not isinstance(localized_values, dict):
            continue
        for locale, value in localized_values.items():
            if value is None:
                continue
            loc = normalize_locale(locale)
            key = f"{field_name}_{loc}"
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            elif isinstance(value, list):
                cleaned = [v for v in value if isinstance(v, (str, int, float, bool))]
                if cleaned:
                    out[key] = cleaned
            else:
                out[key] = str(value)
    return out


print("✅ Metadata helpers defined")

# %% Cell 5 - Chunking & embedding helpers
def chunk_markdown_by_characters(
    document_id: str, text_body: str, chunk_size: int = 500, chunk_overlap: int = 50
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
        chunk_id = f"{document_id}#chunk-{idx:05}|size{chunk_size}-overlap{chunk_overlap}"
        split.metadata = {
            **split.metadata,
            "chunk_id": chunk_id,
            "document_id": document_id,
        }
        chunks.append(split)
    return chunks


def embed_documents(texts, batch_size=32):
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )


print("✅ Chunking & embedding helpers defined")

# %% Cell 6 - Solution Articles embedding
entries = getEntries("solutionArticle")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "platform"
index = pc.Index(INDEX_NAME)
NAMESPACE_NAME = "search"

batch_size = 64
vectors = []

for entry in entries:
    entry_id = entry.id

    body_field = entry.raw["fields"].get("body", {})
    body = body_field.get("en")
    if not body:
        print(f"Skipping entry {entry_id}: no body found")
        continue

    article_metadata = build_locale_metadata(
        entry.raw,
        fields_to_include={"title", "subtitle", "slug"}
    )

    chunks = chunk_markdown_by_characters(entry_id, body)
    texts = [chunk.page_content for chunk in chunks]

    dense_vectors = embed_documents(texts)

    sparse_vectors = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=texts,
        parameters={"input_type": "passage", "return_tokens": False}
    )

    for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        # Skip vectors with empty sparse values (fixes Pinecone 400 error)
        if not sparse_vec.sparse_indices or len(sparse_vec.sparse_indices) == 0:
            print(f"⚠️ Skipping {entry_id}-{i}: empty sparse vector")
            continue

        vectors.append({
            "id": f"{entry_id}-{i}",
            "values": dense_vec.tolist(),
            "sparse_values": {
                "indices": sparse_vec.sparse_indices,
                "values": sparse_vec.sparse_values
            },
            "metadata": {
                "entry_id": entry_id,
                "publishedDate": entry.raw["fields"].get("publishedDate", {}).get("en"),
                "authorIds": entry.raw["fields"].get("authorIds", {}).get("en") or [],
                "body": chunk.page_content,
                "type": "solutionArticle",
                **article_metadata
            }
        })

        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=NAMESPACE_NAME)
            vectors = []

if vectors:
    index.upsert(vectors=vectors, namespace=NAMESPACE_NAME)

print("✅ Done. Upserted all article chunks.")

# %% Cell 7 - Events embedding
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "platform"
index = pc.Index(INDEX_NAME)
NAMESPACE_NAME = "search"

batch_size = 64
vectors = []

entries = getEntries("event", limit=1)

for entry in entries:
    entry_id = entry.id
    created_at = entry.sys['created_at']
    print(f"events entry created_at:{created_at}")

    description_field = entry.raw["fields"].get("description", {})
    description = description_field.get("en")
    if not description:
        print(f"Skipping entry {entry_id}: no description found")
        continue

    metadata = build_locale_metadata(
        entry.raw,
        fields_to_include={"title", "subtitle", "slug"}
    )

    chunks = chunk_markdown_by_characters(entry_id, description)
    texts = [chunk.page_content for chunk in chunks]

    dense_vectors = embed_documents(texts)

    sparse_vectors = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=texts,
        parameters={"input_type": "passage", "return_tokens": False}
    )

    for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        if not sparse_vec.sparse_indices or len(sparse_vec.sparse_indices) == 0:
            print(f"⚠️ Skipping {entry_id}-{i}: empty sparse vector")
            continue

        vectors.append({
            "id": f"{entry_id}-{i}",
            "values": dense_vec.tolist(),
            "sparse_values": {
                "indices": sparse_vec.sparse_indices,
                "values": sparse_vec.sparse_values
            },
            "metadata": {
                "entry_id": entry_id,
