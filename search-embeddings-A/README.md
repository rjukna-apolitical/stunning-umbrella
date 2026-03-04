# Content Embeddings — POC A (English Hybrid Search)

This pipeline pulls content from Contentful CMS, converts it into vector embeddings, and stores them in Pinecone for hybrid semantic search. It targets English content with high-quality structure-preserving chunking and a zero-configuration sparse model.

**POC A vs POC B:** This POC is English-first and operationally simpler — no corpus fitting step, no BM25 statistics to manage. POC B extends the approach to 17 locales using a custom BM25 implementation, at the cost of a more complex setup. Choose A when English coverage is the priority and you want the least operational overhead.

---

## The Approach

### Why Run the Dense Model Locally?

POC A downloads and runs `intfloat/multilingual-e5-large` (560 MB) inside the Docker container. This might look like unnecessary complexity compared to calling a hosted API, but it has a specific advantage: once the model is loaded into memory, every batch of embeddings runs at full hardware speed with no network round-trips and no API rate limits.

For a corpus of thousands of articles, events, and courses, the dense embedding step is the computational bottleneck. A local model that processes 32 chunks per second on CPU is often faster end-to-end than an API that processes them individually with 20–50ms latency per call, especially when a hosted inference tier has concurrency limits.

The model is cached in a Docker volume (`hf-cache`), so the 560 MB download happens only once. Subsequent container runs skip the download and go straight to embedding.

The model name says "multilingual" — despite this POC targeting English, the same model is used in POC B for cross-lingual recall, so the dense representation is already linguistically robust. Any improvements from A carry directly into B.

### Why Pinecone's Sparse Model Instead of Custom BM25?

For English content, Pinecone provides `pinecone-sparse-english-v0` — a learned sparse model trained on a large web corpus. This is not BM25. The weights are learned rather than computed from term frequency, which means the model captures subword and morphological patterns automatically. "Government" and "govt" will share signal without writing a single stemming rule.

The practical advantage: **zero corpus setup.** You do not need to collect documents, compute IDF statistics, or manage a stats file. The sparse model is called as an API at ingestion time, with no pre-fitting required. This makes the pipeline single-phase and simpler to reason about.

The trade-off is that the model's weights were learned from general web data, not from Apolitical's government/policy corpus specifically. Terms like "strategy" and "framework" that appear in nearly every document may get higher sparse weight than warranted. If this becomes a quality issue, POC B's corpus-fitted BM25 is the answer.

### Why Markdown-Aware Chunking?

Article and event content from Contentful is stored as Markdown. Rather than splitting blindly on character count, POC A uses a two-stage splitter:

1. **`MarkdownHeaderTextSplitter`** — splits on `#`, `##`, `###` headers first. Each section becomes its own initial chunk, preserving the semantic boundary that the author drew.
2. **`RecursiveCharacterTextSplitter`** — then splits each section by character count (500 chars, 50 overlap) if it is still too large.

The result is that a chunk about "Section 3: Implementation" never contains text from "Section 4: Evaluation". This matters because an embedding that spans two unrelated sections is semantically incoherent — it describes nothing precisely. Staying within section boundaries makes each chunk a coherent unit, which produces cleaner embeddings and higher search precision.

### Why `locale: "*"` Instead of Per-Locale Fetching?

Each Contentful API call fetches entries with `locale: "*"`, which returns all localized field values in a single response. The embedding pipeline then reads only the English (`en`) value for each field.

The benefit: **one paginated fetch to cover all entries**, regardless of how many locales exist. Per-locale fetching (as in POC B) multiplies API calls by the number of locales. For a corpus of 5,000 articles across 17 locales, that is 85,000 pagination calls versus ~25. This makes re-embedding significantly faster when you only need the English index.

Storing all localized field values in the vector metadata (via `build_locale_metadata`) means the full localized title and slug are available in search results without any extra lookups — even though the vectors themselves are English-only.

### Why No BM25 Fitting Step?

BM25 requires seeing the entire corpus before encoding any single document, because IDF depends on how often a term appears across all `N` documents. This creates a mandatory two-phase pipeline: collect everything, then encode.

By delegating sparse encoding to Pinecone's hosted model, POC A eliminates this phase entirely. You run one command per content type and it is done. There is no stats file to generate, version, or synchronize between environments.

---

## Architecture

```
Contentful (locale="*", paginated, adaptive page size)
    │
    ▼
For each entry:
    ├── Read English body / description / rich text
    ├── Markdown-aware chunk:
    │     MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter
    │
    ├── Dense embed — multilingual-e5-large (local SentenceTransformer, batched)
    └── Sparse embed — pinecone-sparse-english-v0 (Pinecone API)
         │
         ▼
    Pinecone upsert (batches of 64)
```

The pipeline is single-phase and stateless. There is nothing to pre-compute or load before running — configure `.env` and embed.

### Adaptive Page Size

Contentful enforces a 7 MB response size limit. For content types with large bodies (articles) or deeply linked entries (courses), a full page of 200 entries can exceed this. The Contentful module automatically halves the page size and retries whenever it hits this limit, so embedding runs are self-healing without manual intervention.

### Course Content: Linked Entry Resolution

Courses are the most complex content type. A course entry contains references to multiple journey entries, and each journey holds the actual embeddable text in its `overviewLeftColumn` and `overviewRightColumn` rich text fields. The pipeline:

1. Fetches all courses with `include=0` (no linked content inlined) to avoid the 7 MB limit
2. Extracts journey sys IDs from each course
3. Fetches each journey individually via `getEntry(journey_id)`
4. Extracts plain text from the Contentful rich text AST via `extract_values`
5. Combines left + right column text and chunks it

Each journey produces its own set of vectors, tagged with both `journey_id` and `course_id` in metadata.

### Per-Type Delete

Vectors carry a `type` metadata field (`solutionArticle`, `event`, `course`). A single Pinecone metadata filter delete clears all vectors for one content type without touching the others. This means you can re-embed articles without affecting events or courses, which is useful when only one content type has been updated.

---

## Project Structure

```
├── config.py          — env vars and constants (index name, model, dimensions)
├── setup_index.py     — create the Pinecone index (run once)
├── entrypoint.py      — Docker entrypoint (embed | clear)
│
├── modules/
│   ├── contentful.py  — paginated client with rate limit retry and adaptive page size
│   ├── embedding.py   — SentenceTransformer model loader + markdown chunker
│   ├── pinecone_utils.py — get_sparse_vectors, upsert_batch, delete_by_type
│   ├── metadata.py    — build_locale_metadata: flattens all localized fields to key-value pairs
│   ├── richtext.py    — recursive Contentful rich text → plain text extractor
│   └── logger.py      — stdout + per-run file logging
│
└── embed/
    ├── article.py     — solutionArticle: markdown body chunking
    ├── event.py       — event: markdown description chunking
    └── course.py      — course: journey rich text extraction and chunking
```

### Vector ID and Metadata Schema

```
id:       "{entry_id}-{chunk_index}"
          e.g. "abc123-0", "abc123-1"

metadata: {
  entry_id:       Contentful entry ID
  type:           "solutionArticle" | "event" | "course"
  publishedDate:  ISO date string (falls back to entry created_at)
  body:           full chunk text (for result previews without a second Contentful API call)
  title_en:       English title
  slug_en:        English slug
  title_fr:       French title (stored even though the vector is English-only)
  # ... all other localized fields as {field}_{locale} flat key-value pairs
}
```

Locale metadata is stored flat via `build_locale_metadata`. This means the full localized title is available in search results without any extra lookups.

---

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — `brew install uv`
- Docker (for containerised runs)
- A `.env` file with:

```env
PINECONE_API_KEY=...
CONTENTFUL_SPACE_ID=...
CONTENTFUL_ACCESS_TOKEN=...
HF_TOKEN=...
```

`HF_TOKEN` is required to download `intfloat/multilingual-e5-large` from HuggingFace on first run.

---

## Running Locally

### First-Time Setup

```bash
# Install dependencies (includes torch + sentence-transformers, ~2 GB)
make install

# Create the Pinecone index (platform, dotproduct, 1024 dims, GCP europe-west4)
make setup

# Embed each content type
make create-embedding article
make create-embedding event
make create-embedding course
```

### Re-Embedding a Single Content Type

```bash
make clear-embedding article && make create-embedding article
```

### Clear Without Re-Embedding

```bash
make clear-embedding course
```

---

## Running in Docker

### First-Time Setup

```bash
# Build the image (installs torch — takes a few minutes)
make docker-build

# Embed (mounts hf-cache volume so the model is downloaded only once)
make docker-embed article
make docker-embed event
make docker-embed course
```

The `hf-cache` Docker volume stores the downloaded model. The first `docker-embed` run downloads ~560 MB. All subsequent runs skip this and go straight to embedding.

### Re-Embedding in Docker

```bash
make docker-clear-embedding article && make docker-embed article
```

---

## Dependency Notes

`torch` and `sentence-transformers` are large dependencies (~2 GB installed). If you are on a machine with an NVIDIA GPU, torch will use CUDA automatically — the dense embedding step is roughly 10× faster on GPU than CPU.

The Dockerfile uses `python:3.12-slim` with `gcc` and `g++` added to satisfy build requirements for torch extensions. If you hit compilation errors during `docker build`, ensure Docker has at least 4 GB RAM allocated.
