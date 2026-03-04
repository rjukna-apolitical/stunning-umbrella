# Multilingual Hybrid Search Embeddings — POC B

This pipeline pulls content from Contentful CMS, converts it into vector embeddings, and stores them in Pinecone for hybrid semantic search. It supports 17 languages simultaneously.

**POC A vs POC B:** POC A embeds English-only content using a local SentenceTransformer model and Pinecone's English sparse model. This POC (B) extends that to 17 locales using Pinecone's hosted dense model and a custom multilingual BM25 implementation.

---

## The Core Problem

Our content exists in 17 languages. When a French-speaking user searches for "transformation numérique gouvernement", we want results in French. When the platform falls back to English, we want the dense vector component to bridge the language gap semantically.

This requires two things that English-only search cannot provide:

1. **Locale-scoped sparse matching** — the term "gift" means poison in German. An English sparse model has no concept of this, so it would incorrectly boost German documents about poison when searching for English gift ideas.
2. **Corpus-aware term weighting** — on Apolitical's government/policy corpus, the word "government" appears in nearly every document. A general-purpose sparse model trained on web data doesn't know this, so it assigns "government" a high weight. We need IDF statistics computed from *our* corpus to correctly downweight domain-ubiquitous jargon.

---

## Architecture

### Why Hybrid Search (Dense + Sparse)?

Dense vectors (from `multilingual-e5-large`) capture semantic meaning. They can map "how governments use AI ethically" to conceptually similar documents, bridge synonyms, and work cross-lingually. But they struggle with exact entity matching — a query for "OECD PISA report" relies on the model having seen those terms together.

Sparse vectors (BM25) capture exact term frequency. They are precise for named entities, acronyms, and rare technical terms. But they cannot handle paraphrasing or cross-lingual retrieval at all.

Hybrid search combines both using a weighted sum (alpha parameter). The result handles both "what does 'participatory budgeting' mean?" (dense-led) and "OECD PISA 2023 results" (sparse-led) well in the same index.

### Why Per-Locale Vectors?

Rather than storing one vector per document and relying entirely on the dense model for cross-lingual recall, we create a separate set of vectors for each locale a document has been translated into. For example, an article translated into English, French, and German produces three independent groups of vectors in Pinecone.

Each group carries a `locale` metadata field. At search time, a French user queries with `filter: {locale: "fr"}`, getting BM25-boosted results in French. If too few results are found, a fallback query runs against English vectors with `alpha=1.0` (dense-only), so the multilingual dense model provides cross-lingual bridging.

This design gives us:
- **Exact BM25 matching in the user's language** — no cross-language sparse noise
- **Locale filtering** via cheap Pinecone metadata filter (no post-processing)
- **Graceful fallback to English** when a locale has sparse content

### Why BM25 Instead of Pinecone's Sparse Model?

Pinecone provides `pinecone-sparse-english-v0`, which is convenient for English. We cannot use it for two reasons:

1. **It only supports English.** We need 17 locales.
2. **Its IDF weights come from a generic corpus (likely MS MARCO).** Terms like "strategy", "framework", and "initiative" are common in government/policy content but might be unusual in MS MARCO web data — so the model would assign them inflated weights. Our BM25 is fitted on Apolitical's own corpus, so those terms automatically get near-zero IDF weight because `df ≈ N`.

### Why Locale-Prefixed Tokens in BM25?

Every token in our BM25 is prefixed with its locale code before hashing into the sparse vector space. `en:transform` and `de:transform` become different sparse dimensions, even though they look similar. This prevents two problems:

- A German document matching an English query on the word "gift" (German: poison)
- Regional variants sharing weight incorrectly (French Canadian `fr-CA` uses the same stemmer as `fr`, but their documents should not cross-boost each other's sparse scores)

### Why Pinecone Inference for Dense Embeddings?

POC A downloads and runs `intfloat/multilingual-e5-large` locally inside Docker. This means a ~560MB model download on every fresh container, a GPU-like RAM requirement, and a cached HuggingFace volume to manage.

POC B calls Pinecone's hosted inference endpoint instead. The model is identical, but the embedding workload runs in Pinecone's infrastructure. The Docker image is dramatically smaller and simpler — no torch, no transformers, no GPU memory pressure.

### Two-Phase Pipeline

BM25 needs to see the entire corpus before encoding any individual document — the IDF denominator is `N / df`, so you need `N` (total document count) before you can score a term in any single document.

This creates a natural two-phase design:

```
Phase 1 — Fit BM25
  ├── Fetch all content (all locales, all types) from Contentful
  ├── Tokenize using per-locale strategy
  ├── Compute IDF statistics (doc_freq, avg_doc_len)
  └── Save to data/bm25_corpus_stats.json

Phase 2 — Embed (per content type)
  ├── Load BM25 stats from file
  ├── Fetch all entries (locale="*" in one paginated pass)
  ├── For each entry, for each locale with content:
  │     ├── Chunk text (word-count, 400 words, 50 overlap)
  │     ├── Dense embed via Pinecone Inference (batched)
  │     └── BM25 encode_document (instant, in-process)
  └── Upsert to Pinecone in batches of 64
```

The stats file is persisted in a Docker volume (`bm25-data-b`), so fitting only needs to re-run when the corpus changes significantly (new locales, significant content growth, or new domain stopwords).

---

## Language Support

| Locale | Tier | Tokenisation |
|--------|------|-------------|
| en, ar, fr, fr-CA, de, id, it, pt, pt-BR, es, es-419 | 1 | Snowball stemmer + NLTK stopwords |
| sr-Cyrl | 2 | Snowball Serbian + custom stopwords |
| ja | 3 | fugashi/MeCab morphological analysis |
| ko | 3 | kiwipiepy/Kiwi morphological analysis |
| vi | 3 | pyvi CRF word segmenter |
| pl, uk | 4 | Whitespace tokenisation + custom stopwords |

Tier 3 tokenisers are optional. Without them, Japanese, Korean, and Vietnamese fall back to whitespace tokenisation — BM25 IDF still works, you just lose compound word splitting.

---

## Project Structure

```
├── config.py          — env vars and constants (index name, BM25 path, etc.)
├── setup_index.py     — create the Pinecone index (run once)
├── setup_bm25.py      — fit BM25 on corpus and save stats (run once, re-run if corpus changes significantly)
├── entrypoint.py      — Docker entrypoint (fit | embed | clear)
│
├── modules/
│   ├── bm25.py        — MultilingualBM25 class, LOCALE_CONFIG, tokenisation tiers
│   ├── contentful.py  — paginated Contentful client (locale="*", rate limit retry, adaptive page size)
│   ├── embedding.py   — Pinecone Inference dense embed + word-count chunker
│   ├── pinecone_utils.py — upsert_batch and delete_by_type
│   ├── richtext.py    — Contentful rich text → plain text extractor
│   └── logger.py      — stdout + per-run file logging
│
└── embed/
    ├── article.py     — solutionArticle: embeds body per locale
    ├── event.py       — event: embeds description per locale
    └── course.py      — course: fetches journey overview columns per locale
```

### Vector ID and Metadata Schema

Every vector stored in Pinecone has:

```
id:       "{entry_id}::{locale}::{chunk_index}"
          e.g. "abc123::fr::0"

metadata: {
  content_id:        Contentful entry ID
  content_type:      "solutionArticle" | "event" | "course"
  locale:            "en" | "fr" | "de" | ...
  title:             localized title
  slug:              localized slug
  snippet:           first 300 chars of the chunk (for result previews)
  published_date:    Unix timestamp int (0 if not available)
  available_locales: list of locales this entry has content in
}
```

The `content_type` field enables deleting and re-embedding a single type without touching others. The `locale` field enables per-locale search filtering.

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
```

---

## Running Locally

### First-Time Setup

```bash
# Install dependencies
make install

# Create the Pinecone index (platform-v2, dotproduct, 1024 dims, GCP europe-west4)
make setup

# Fit BM25 on the full corpus across all locales
# This fetches all content from Contentful and computes IDF statistics.
# Takes several minutes depending on corpus size. Saves data/bm25_corpus_stats.json.
make fit-bm25

# Embed each content type
make create-embedding article
make create-embedding event
make create-embedding course
```

### Re-Embedding a Single Content Type

```bash
# Delete existing vectors for that type, then re-embed
make clear-embedding article && make create-embedding article
```

### When to Re-Run fit-bm25

Re-fit BM25 when:
- The corpus has grown substantially (>20% more documents)
- A new locale is added to Contentful
- You add domain stopwords to `DOMAIN_STOPWORDS_EN` or `CUSTOM_STOPWORDS`

You do **not** need to re-fit just because individual articles were updated. BM25 IDF statistics are stable across small content changes.

---

## Running in Docker

### First-Time Setup

```bash
# Build the image
make docker-build

# Fit BM25 — persists stats to the 'bm25-data-b' Docker volume
make docker-fit-bm25

# Embed each content type (reads BM25 stats from the same volume)
make docker-embed article
make docker-embed event
make docker-embed course
```

The `bm25-data-b` volume is shared between the `fit` and `embed` runs. If you run `docker-fit-bm25` on one machine and `docker-embed` on another, you need to copy the `data/bm25_corpus_stats.json` file manually, or re-run fit on the target machine.

### Re-Embedding in Docker

```bash
make docker-clear-embedding article && make docker-embed article
```

---

## Optional: Tier 3 Tokeniser Support (Japanese, Korean, Vietnamese)

Tier 3 tokenisers are not installed by default because they pull in large dictionaries (~50–110 MB). Without them, `ja`, `ko`, and `vi` fall back to whitespace tokenisation, which is functional but less accurate.

To install them locally:

```bash
uv sync --extra cjk
```

To include them in Docker, add `--extra cjk` to the `uv sync` line in the `Dockerfile`.

---

## Tuning and Diagnostics

**BM25 `k1` and `b`** are in `modules/bm25.py` (`BM25_K1 = 1.2`, `BM25_B = 0.75`). These are standard Robertson BM25 defaults and work well without tuning. Increase `k1` if you want term frequency to matter more; decrease `b` if documents vary widely in length.

**`HASH_SPACE`** (`2**18 = 262,144`) controls the sparse vector dimension. Larger values reduce hash collisions between tokens but don't affect Pinecone storage (only non-zero values are stored).

**Domain stopwords** in `DOMAIN_STOPWORDS_EN` remove corpus-ubiquitous English terms before BM25 encoding. After running fit-bm25, review terms with very low IDF (appearing in >80% of documents) and add them here to clean up the sparse vectors.

**`alpha` at search time** controls the hybrid blend: `0.0` = sparse-only, `1.0` = dense-only, `0.7` = recommended starting point (dense-led). The embedding pipeline doesn't set alpha — this is a search-time parameter in the application consuming the index.
