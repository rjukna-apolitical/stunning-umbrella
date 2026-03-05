# Search API

A multilingual hybrid search service built with NestJS, backed by Pinecone vector database. It powers content discovery across Apolitical's 17-language content corpus by combining two complementary retrieval techniques: semantic understanding (dense vectors) and exact-term matching (sparse BM25 vectors).

---

## Table of Contents

1. [Why Hybrid Search?](#why-hybrid-search)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Setup](#setup)
5. [API Reference](#api-reference)
6. [Query Examples](#query-examples)
7. [Language Support](#language-support)
8. [Key Design Decisions](#key-design-decisions)
9. [Known Limitations](#known-limitations)

---

## Why Hybrid Search?

Before diving into the code, it helps to understand why two retrieval techniques are used together.

### Dense vectors (semantic search)

A dense vector is a list of ~1000 numbers that encodes the _meaning_ of a piece of text. The model used here — `multilingual-e5-large` — produces a 1024-dimensional vector for any text in any of the 17 supported languages. Two texts that mean the same thing in different languages will produce vectors that are close together in this 1024-dimensional space.

**What it does well:** Synonyms, paraphrases, cross-lingual queries, conceptual similarity.
**What it struggles with:** Exact term matching. A query for "GDPR" and a document containing "GDPR" are not guaranteed to score well if the model hasn't seen that acronym frequently during training.

### Sparse vectors (BM25 — exact term matching)

BM25 is the algorithm behind most traditional search engines (Elasticsearch, Solr, and Google's early days). It represents a document or query as a sparse vector — a list of (word_index → score) pairs where the score reflects how important that word is, taking into account:

- **Term frequency (TF):** How often does the word appear in this document?
- **Inverse document frequency (IDF):** How rare is this word across the whole corpus? Rare words signal relevance more strongly.
- **Document length normalisation:** A word appearing once in a short document is more significant than the same word appearing once in a 10,000-word article.

**What it does well:** Exact term matches, acronyms, proper nouns, technical terms.
**What it struggles with:** Synonyms, multilingual queries (a German query cannot match English BM25 terms).

### Why combine them?

Neither technique alone is sufficient for a multilingual policy content platform:

| Scenario                            | Dense alone        | BM25 alone         | Hybrid                |
| ----------------------------------- | ------------------ | ------------------ | --------------------- |
| "digital transformation" (English)  | ✓ Semantic match   | ✓ Exact match      | ✓                     |
| "transformation numérique" (French) | ✓ Cross-lingual    | ✗ French BM25 only | ✓ French BM25 + dense |
| "GDPR compliance framework"         | ~ May miss acronym | ✓ Exact GDPR match | ✓                     |
| Vague concept query                 | ✓                  | ✗                  | ✓                     |

The hybrid score for each result is: `alpha × dense_score + (1 - alpha) × bm25_score`, where `alpha = 0.7` by default (70% semantic, 30% exact-term).

---

## How It Works

Every search request goes through five steps:

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 1 — Encode query                               │
│                                                     │
│  Dense:  Pinecone Inference API                     │
│          "digital transformation" ──► [0.12, -0.34, …] (1024 floats)
│                                                     │
│  Sparse: BM25Encoder (in-process, <1ms)             │
│          "digital transformation" ──► {4821: 3.2, 17043: 2.1, …}
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 2 — Primary hybrid query (user's locale)       │
│                                                     │
│  Pinecone dotproduct index, filtered to locale=fr   │
│  Scaled: dense × 0.7,  sparse × 0.3                │
│  Returns top-K matches                              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 3 — English fallback (only if needed)          │
│                                                     │
│  Fires when the current page would be under-full    │
│  (e.g. French corpus has 3 results but page asks    │
│   for 10). Fetches English results to backfill,     │
│  deduplicating by content_id.                       │
│  Uses alpha=1.0 (pure dense) — BM25 sparse terms   │
│  from a French query cannot match English documents.│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 4 — Enrollment boost                           │
│                                                     │
│  If the caller passes enrolledIds, content the user │
│  is already enrolled in gets a 1.3× score boost.   │
│  All results are re-sorted by boosted score.        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 5 — Paginate and return                        │
└─────────────────────────────────────────────────────┘
```

### What is corpus-aware BM25?

The BM25 IDF (word rarity) scores are not generic — they were computed from Apolitical's own content corpus by the `search-embeddings-B` Python project. This matters because a word like "government" appears in almost every article on a policy platform and should be treated as low-signal, whereas it would be treated as a rare, high-signal term by a generic web-trained model.

The corpus statistics (word frequencies, document count, average document length) are stored in `bm25_corpus_stats.json` and loaded once at service startup.

---

## Architecture

```
search-api/                         search-embeddings-B/
├── src/                            ├── setup_bm25.py         ← fits BM25 corpus
│   ├── main.ts                     ├── embed/                ← upserts vectors
│   ├── app.module.ts               └── data/
│   └── search/                         └── bm25_corpus_stats.json ◄── shared file
│       ├── search.module.ts
│       ├── search.controller.ts
│       ├── search.service.ts       ← orchestrates search pipeline
│       └── bm25-encoder.ts         ← TypeScript port of Python BM25 encoder
│
└── .env                            Pinecone (cloud)
    PINECONE_API_KEY                ├── Index: platform-v2
    BM25_STATS_PATH ────────────►  │   Namespace: search
                                    │   Metric: dotproduct (required for hybrid)
                                    │   Dimension: 1024 (multilingual-e5-large)
                                    └── Inference: multilingual-e5-large (hosted)
```

### Component responsibilities

| Component                | Responsibility                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `search.controller.ts`   | HTTP routing — `GET /search` (quick curl testing) and `POST /search` (full options)             |
| `search.service.ts`      | Search pipeline orchestration, fallback logic, enrollment boost                                 |
| `bm25-encoder.ts`        | Tokenises query text and computes BM25 sparse vector using pre-fitted corpus stats              |
| `bm25_corpus_stats.json` | ~50MB JSON of word frequencies across all 17 locales, produced by `search-embeddings-B`         |
| Pinecone index           | Stores 1024-dim dense + sparse vectors for every content chunk, queryable with metadata filters |

---

## Setup

### Prerequisites

- Node.js 20+
- pnpm (or npm)
- A Pinecone account with the `platform-v2` index already created and populated (run `search-embeddings-B` first)
- The fitted BM25 stats file (`bm25_corpus_stats.json`) from `search-embeddings-B`

### 1. Install dependencies

```bash
cd search-api
pnpm install   # or npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Your Pinecone API key — find it at https://app.pinecone.io
PINECONE_API_KEY=pcsk_...

# Path to the corpus stats file produced by search-embeddings-B.
# If you're running from the sandbox root, this default works as-is:
BM25_STATS_PATH=../search-embeddings-B/data/bm25_corpus_stats.json

# Port (optional, defaults to 8300)
PORT=8300
```

### 3. Ensure BM25 stats exist

The `bm25_corpus_stats.json` file is produced by the embedding pipeline:

```bash
cd ../search-embeddings-B
make docker-fit-bm25   # fits BM25 on the Contentful corpus
```

Wait for it to finish — you'll see `BM25 fitting complete. Stats saved to data/bm25_corpus_stats.json` in the logs.

### 4. Start the server

```bash
# Development (watch mode — restarts on file changes)
pnpm start:dev

# Production
pnpm build && pnpm start:prod
```

On startup you should see:

```
[SearchService] Pinecone index 'platform-v2' connected
[SearchService] BM25 loaded: 42301 docs, 187432 terms, avgLen=143
Search API listening on http://localhost:8300
```

If the BM25 stats file is missing you'll get a `ENOENT` error — check `BM25_STATS_PATH` in your `.env`.

---

## API Reference

### `GET /search`

Quick testing endpoint — all options as query parameters.

| Parameter      | Type    | Default  | Description                                              |
| -------------- | ------- | -------- | -------------------------------------------------------- |
| `q`            | string  | required | Search query                                             |
| `locale`       | string  | `en`     | BCP-47 locale code (e.g. `fr`, `de`, `pt-BR`)            |
| `contentType`  | string  | —        | Filter to `solutionArticle`, `event`, or `course`        |
| `page`         | number  | `1`      | Page number (1-indexed)                                  |
| `pageSize`     | number  | `10`     | Results per page                                         |
| `crossLingual` | boolean | `false`  | Search across all locales (pure dense, no locale filter) |

### `POST /search`

Full control endpoint — send a JSON body.

```typescript
{
  query: string;           // required
  locale: string;          // required — e.g. "en", "fr", "de", "ja"
  contentType?: string;    // optional filter: "solutionArticle" | "event" | "course"
  page?: number;           // default 1
  pageSize?: number;       // default 10
  enrolledIds?: string[];  // content IDs the user is already enrolled in (gets 1.3× boost)
  crossLingual?: boolean;  // default false
}
```

### Response shape

```typescript
{
  matches: [
    {
      id: string;                        // Pinecone vector ID: "{contentId}::{locale}::{chunkIndex}"
      score: number;                     // boosted score (after enrollment boost)
      originalScore: number;             // raw Pinecone score before boost
      isEnrolled: boolean;               // true if content_id was in enrolledIds
      isFallback: boolean;               // true if result came from English fallback
      metadata: {
        content_id: string;              // Contentful entry ID
        content_type: string;            // "solutionArticle" | "event" | "course"
        locale: string;                  // locale of this vector
        title: string;
        slug: string;
        snippet: string;                 // first 300 chars of the chunk
        published_date: number;          // Unix timestamp
        available_locales: string[];     // all locales this content exists in
        banner_url?: string;             // image URL if available
        is_fallback?: true;              // only present on fallback results
      }
    }
  ],
  totalPrimary: number;    // how many results came from the primary locale query
  totalFallback: number;   // how many results came from the English fallback
  page: number;
  pageSize: number;
  timing: {
    embedMs: number;       // Pinecone inference API latency
    bm25Ms: number;        // BM25 encoding time (always <1ms, in-process)
    pineconeMs: number;    // Pinecone query latency
    totalMs: number;       // total wall time
  }
}
```

---

## Query Examples

These examples are designed to test different aspects of the hybrid retrieval. Run them after the index has been populated.

### Basic English search

```bash
curl "http://localhost:8300/search?q=digital+transformation&locale=en"
```

```bash
curl -X POST http://localhost:8300/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "digital transformation", "locale": "en", "pageSize": 5}'
```

### Multilingual — same concept, different languages

These four queries express the same concept. Each should return locale-specific content with similar semantic relevance:

```bash
# French
curl "http://localhost:8300/search?q=transformation+num%C3%A9rique&locale=fr"

# German
curl "http://localhost:8300/search?q=digitale+Transformation&locale=de"

# Spanish
curl "http://localhost:8300/search?q=transformaci%C3%B3n+digital&locale=es"

# Portuguese (Brazil)
curl "http://localhost:8300/search?q=transforma%C3%A7%C3%A3o+digital&locale=pt-BR"
```

**What to check:** All four should return semantically similar content (even if different articles). The `locale` field on each result should match the query locale.

### Testing the English fallback

Search in a locale with limited content — French-Canadian or Indonesian often have fewer entries:

```bash
curl "http://localhost:8300/search?q=open+data+transparency&locale=fr-CA&pageSize=10"
```

**What to check:** `totalFallback > 0` in the response indicates the fallback fired. Results with `isFallback: true` are English content shown because the French-Canadian corpus didn't have enough relevant results. These results should still be semantically relevant to the query.

### Cross-lingual search

Search across all 17 language corpuses simultaneously using only semantic similarity (no BM25):

```bash
curl "http://localhost:8300/search?q=open+government+data&locale=en&crossLingual=true&pageSize=10"
```

**What to check:** Results will include vectors from multiple locales — you should see a mix of `locale` values in the response. This is useful for finding conceptually related content that was written in a language different from the query.

### Filtering by content type

```bash
# Only events
curl "http://localhost:8300/search?q=leadership+workshop&locale=en&contentType=event"

# Only articles
curl "http://localhost:8300/search?q=climate+adaptation+policy&locale=en&contentType=solutionArticle"

# Only courses
curl "http://localhost:8300/search?q=data+analysis+skills&locale=en&contentType=course"
```

### Enrollment boost

Content IDs in `enrolledIds` receive a 1.3× score boost and bubble up in the ranking:

```bash
curl -X POST http://localhost:8300/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "procurement reform",
    "locale": "en",
    "enrolledIds": ["abc123EntryId", "def456EntryId"],
    "pageSize": 10
  }'
```

**What to check:** Results where `isEnrolled: true` should appear near the top even if their `originalScore` is not the highest. Compare `score` vs `originalScore` to see the 1.3× boost applied.

### Japanese search

```bash
curl -X POST http://localhost:8300/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "デジタル変革", "locale": "ja", "pageSize": 5}'
```

### Korean search

```bash
curl -X POST http://localhost:8300/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "디지털 전환", "locale": "ko", "pageSize": 5}'
```

### Arabic search

```bash
curl -X POST http://localhost:8300/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "الحوكمة الرقمية", "locale": "ar", "pageSize": 5}'
```

### Testing timing

Check the `timing` object to validate latency profile:

```bash
curl -s "http://localhost:8300/search?q=public+sector+innovation&locale=en" | \
  node -e "const d=require('fs').readFileSync('/dev/stdin','utf8'); console.log(JSON.parse(d).timing)"
```

Expected latency profile:

- `bm25Ms`: `< 1` — BM25 runs in-process, no network
- `embedMs`: `15–40` — Pinecone inference API (co-located with index in `europe-west4`)
- `pineconeMs`: `15–40` — Pinecone query
- `totalMs`: `< 100` at p95

---

## Language Support

| Locale         | Language           | Stemmer                        | Stopwords               |
| -------------- | ------------------ | ------------------------------ | ----------------------- |
| `en`           | English            | Porter (Snowball-compatible)   | NLTK + domain stopwords |
| `fr`, `fr-CA`  | French             | Porter Fr                      | NLTK                    |
| `es`, `es-419` | Spanish            | Porter Es                      | NLTK                    |
| `pt`, `pt-BR`  | Portuguese         | Porter Pt                      | NLTK                    |
| `it`           | Italian            | Porter It                      | NLTK                    |
| `de`           | German             | _(none — see limitations)_     | NLTK                    |
| `ar`           | Arabic             | _(none)_                       | NLTK                    |
| `id`           | Indonesian         | _(none)_                       | NLTK                    |
| `pl`           | Polish             | _(none)_                       | NLTK                    |
| `sr-Cyrl`      | Serbian (Cyrillic) | _(none)_                       | Custom                  |
| `uk`           | Ukrainian          | _(none)_                       | Custom                  |
| `ja`           | Japanese           | _(none — whitespace fallback)_ | Custom                  |
| `ko`           | Korean             | _(none — whitespace fallback)_ | Custom                  |
| `vi`           | Vietnamese         | _(none — whitespace fallback)_ | Custom                  |

Where stemming is absent, the sparse BM25 component has lower recall (e.g. "Regierung" and "regiert" won't share the same sparse dimension). The dense vector from `multilingual-e5-large` fully compensates for semantic relevance — the impact is mainly on exact-term precision for inflected forms.

---

## Key Design Decisions

### Why NestJS?

NestJS provides dependency injection, module scoping, and a clear lifecycle (`onModuleInit`) that maps cleanly to this service's startup needs: connect to Pinecone, load a large JSON file into memory, and keep both available for the lifetime of the server. A plain Express app would work too but would require manual wiring.

### Why load BM25 stats into memory?

The corpus statistics file (~50MB) is loaded once at startup into a `Map<string, number>`. This allows BM25 query encoding to complete in under 1ms without any I/O. The alternative — reading from disk or a database on every query — would add 5–50ms per request, undermining the hybrid approach's latency target.

### Why implement MurmurHash3 inline?

The sparse vector indices produced by this TypeScript encoder must match the indices stored in Pinecone (which were produced by the Python embedding pipeline). Python's `mmh3` library hashes the UTF-8 byte representation of each token string. Several JavaScript MurmurHash3 packages use UTF-16 code units instead, which produces different values for any non-ASCII token (accented letters, CJK characters, Cyrillic, Arabic). Rather than risk a silent mismatch, MurmurHash3 x86 32-bit is implemented directly using `Buffer.from(key, 'utf8')` to guarantee byte-for-byte compatibility.

### Why `alpha = 0.7` (70% dense)?

Pinecone's hybrid scoring multiplies the dense and sparse vector values before dotproducing. Alpha controls the trade-off:

- `alpha = 1.0` — pure semantic search. Good for vague, conceptual queries.
- `alpha = 0.0` — pure BM25. Good for exact term lookup.
- `alpha = 0.7` — empirically found to work well for policy/government content, where users often mix conceptual intent ("what is being done about climate") with specific terms ("Paris Agreement", "NDC").

Cross-lingual mode forces `alpha = 1.0` because sparse BM25 vectors are language-specific — a French query produces French-prefixed token indices (`fr:transform`) that cannot match English token indices (`en:transform`) in the vector space.

### Why dotproduct metric instead of cosine similarity?

Pinecone's hybrid search requires the `dotproduct` metric. Cosine similarity normalises vectors before computing similarity, which removes the ability to scale them by `alpha`. With `dotproduct`, multiplying the dense vector by 0.7 and the sparse vector by 0.3 directly controls their relative contribution to the final score.

### Why over-fetch from Pinecone?

Pinecone's `topK` parameter limits results before any re-ranking. If we fetched exactly `pageSize` results and then applied the enrollment boost, enrolled content that Pinecone ranked 11th (just outside page 1) could never surface to the top. Fetching `page × pageSize × 3` (capped at 200) gives the re-ranker enough candidates to work with while keeping the Pinecone payload reasonable.

### Why is the English fallback pure dense (alpha=1.0)?

When a user queries in French and gets fewer results than needed, we fall back to English content. At this point, the sparse (BM25) component is counterproductive: the French query tokens produce French-prefixed sparse indices that simply won't match English documents. Sending a non-zero sparse vector would effectively add noise. Pure dense (`alpha=1.0`) gives the best English results for a cross-language query.

---

## Known Limitations

**Stemming gaps for de/ar/id**
The `natural` npm package does not include German, Arabic, or Indonesian Snowball stemmers. Queries in these languages will not stem, meaning "Digitalisierung" and "digitalisier" won't share a BM25 sparse dimension. Dense vectors compensate but exact-term sparse recall is reduced for inflected forms.

**Pagination is stateless**
Each request independently fetches up to 200 results from Pinecone. There is no server-side cursor or cache. For pages beyond ~7 (at the default page size of 10), results may shift between requests if the index is being updated. This is acceptable for a POC; production would use cursor-based pagination or a Redis result cache.

**BM25 stats must be re-fitted after significant content updates**
The IDF scores are computed once from the entire corpus. If large amounts of content are added or removed, the word rarity scores become stale. Re-running `make docker-fit-bm25` and re-deploying the stats file corrects this. In practice, for a corpus of this size, re-fitting weekly is sufficient.
