# Search API (Approach A)

A multilingual hybrid search service built with NestJS, backed by Pinecone vector database. It powers content discovery across Apolitical's content corpus by combining two complementary retrieval techniques: semantic understanding (dense vectors) and exact-term matching (sparse vectors) — both hosted on Pinecone's inference API.

This is **Approach A**, which uses Pinecone-hosted `pinecone-sparse-english-v0` for sparse encoding. See [`search-api-b`](../search-api-b) for Approach B, which uses a corpus-fitted BM25 encoder instead. Both APIs expose an identical HTTP interface to enable objective latency and quality comparison.

---

## Table of Contents

1. [Why Hybrid Search?](#why-hybrid-search)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Setup](#setup)
5. [API Reference](#api-reference)
6. [Query Examples](#query-examples)
7. [Key Design Decisions](#key-design-decisions)
8. [Approach A vs Approach B](#approach-a-vs-approach-b)
9. [Known Limitations](#known-limitations)

---

## Why Hybrid Search?

Before diving into the code, it helps to understand why two retrieval techniques are used together.

### Dense vectors (semantic search)

A dense vector is a list of ~1000 numbers that encodes the _meaning_ of a piece of text. The model used here — `multilingual-e5-large` — produces a 1024-dimensional vector for any text in any supported language. Two texts that mean the same thing in different languages will produce vectors that are close together in this 1024-dimensional space.

**What it does well:** Synonyms, paraphrases, cross-lingual queries, conceptual similarity.
**What it struggles with:** Exact term matching. A query for "GDPR" and a document containing "GDPR" are not guaranteed to score well if the model hasn't seen that acronym frequently during training.

### Sparse vectors (Pinecone-hosted sparse model)

`pinecone-sparse-english-v0` is a learned sparse retrieval model hosted on Pinecone's inference API. Like BM25, it produces a sparse vector of (term_index → weight) pairs, but it uses a neural approach to compute those weights rather than pure term-frequency statistics.

**What it does well:** Exact term matches, acronyms, proper nouns, technical terms.
**What it struggles with:** Non-English text — the model is English-focused, so sparse weights for other languages are less reliable.

### Why combine them?

Neither technique alone is sufficient for a content discovery platform:

| Scenario                            | Dense alone        | Sparse alone       | Hybrid             |
| ----------------------------------- | ------------------ | ------------------ | ------------------ |
| "digital transformation" (English)  | ✓ Semantic match   | ✓ Exact match      | ✓                  |
| "GDPR compliance framework"         | ~ May miss acronym | ✓ Exact GDPR match | ✓                  |
| Vague concept query                 | ✓                  | ✗                  | ✓                  |
| Non-English query                   | ✓ Cross-lingual    | ~ English-only     | ✓ (dense dominant) |

The hybrid score for each result is: `alpha × dense_score + (1 - alpha) × sparse_score`, where `alpha = 0.7` for English queries (70% semantic, 30% exact-term) and `alpha = 1.0` for non-English queries (pure dense, since the sparse model is English-focused).

---

## How It Works

Every search request goes through four steps:

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 1 — Encode query (dense + sparse in parallel)  │
│                                                     │
│  Dense:  Pinecone Inference API                     │
│          "digital transformation" ──► [0.12, -0.34, …] (1024 floats)
│                                                     │
│  Sparse: Pinecone Inference API                     │
│          "digital transformation" ──► {4821: 3.2, 17043: 2.1, …}
│          (pinecone-sparse-english-v0, ~25ms)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 2 — Hybrid query (no locale filter)            │
│                                                     │
│  Pinecone dotproduct index `platform`               │
│  Scaled: dense × alpha,  sparse × (1 - alpha)      │
│  All locales searched simultaneously (per-locale   │
│  vectors were not used in search-embeddings-A)      │
│  Optional filter: content type                      │
│  Returns top-K matches                              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 3 — Enrollment boost                           │
│                                                     │
│  If the caller passes enrolledIds, content the user │
│  is already enrolled in gets a 1.3× score boost.   │
│  All results are re-sorted by boosted score.        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Step 4 — Paginate and return                        │
└─────────────────────────────────────────────────────┘
```

> **No English fallback:** Approach A stores all locales in the same vector set (no per-locale filtering), so every query already searches across the full corpus. Fallback logic is not needed.

---

## Architecture

```
search-api-a/                       search-embeddings-A/
├── src/                            ├── entrypoint.py     ← upserts vectors
│   ├── main.ts                     └── modules/
│   ├── app.module.ts                   ├── embedding.py  ← multilingual-e5-large (local)
│   └── search/                         └── pinecone_utils.py ← sparse via Pinecone API
│       ├── search.module.ts
│       ├── search.controller.ts
│       └── search.service.ts       ← orchestrates search pipeline
│
└── .env                            Pinecone (cloud)
    PINECONE_API_KEY                ├── Index: platform
                                    │   Namespace: search
                                    │   Metric: dotproduct (required for hybrid)
                                    │   Dimension: 1024 (multilingual-e5-large)
                                    └── Inference:
                                        ├── multilingual-e5-large (dense, hosted)
                                        └── pinecone-sparse-english-v0 (sparse, hosted)
```

### Component responsibilities

| Component              | Responsibility                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `search.controller.ts` | HTTP routing — `GET /search` (quick curl testing) and `POST /search` (full options)                      |
| `search.service.ts`    | Search pipeline orchestration: parallel encoding, hybrid query, enrollment boost                         |
| Pinecone index         | Stores 1024-dim dense + sparse vectors for every content chunk across all locales                        |
| Pinecone Inference API | Hosts both `multilingual-e5-large` and `pinecone-sparse-english-v0` — no local model files required     |

---

## Setup

### Prerequisites

- Node.js 20+
- npm (or pnpm)
- A Pinecone account with the `platform` index already created and populated (run `search-embeddings-A` first)

### 1. Install dependencies

```bash
cd search-api-a
npm install
```

### 2. Configure environment

Edit `.env`:

```env
# Your Pinecone API key — find it at https://app.pinecone.io
PINECONE_API_KEY=pcsk_...

# Port (optional, defaults to 3001)
PORT=8301
```

### 3. Start the server

```bash
# Development (watch mode — restarts on file changes)
npm run start:dev

# Production
npm run build && npm run start:prod
```

On startup you should see:

```
[SearchService] Pinecone index 'platform' connected
Search API (approach A) listening on http://localhost:8301
```

---

## API Reference

### `GET /search`

Quick testing endpoint — all options as query parameters.

| Parameter      | Type    | Default  | Description                                              |
| -------------- | ------- | -------- | -------------------------------------------------------- |
| `q`            | string  | required | Search query                                             |
| `locale`       | string  | `en`     | BCP-47 locale code — affects sparse alpha only           |
| `contentType`  | string  | —        | Filter to `solutionArticle`, `event`, or `course`        |
| `page`         | number  | `1`      | Page number (1-indexed)                                  |
| `pageSize`     | number  | `10`     | Results per page                                         |
| `crossLingual` | boolean | `false`  | Force pure dense (alpha=1.0), no sparse component        |

### `POST /search`

Full control endpoint — send a JSON body.

```typescript
{
  query: string;           // required
  locale: string;          // required — affects alpha (en → 0.7, others → 1.0)
  contentType?: string;    // optional filter: "solutionArticle" | "event" | "course"
  page?: number;           // default 1
  pageSize?: number;       // default 10
  enrolledIds?: string[];  // entry IDs the user is enrolled in (gets 1.3× boost)
  crossLingual?: boolean;  // default false — forces alpha=1.0
}
```

### Response shape

```typescript
{
  matches: [
    {
      id: string;                       // Pinecone vector ID: "{entryId}-{chunkIndex}"
      score: number;                    // boosted score (after enrollment boost)
      originalScore: number;            // raw Pinecone score before boost
      isEnrolled: boolean;              // true if entry_id was in enrolledIds
      isFallback: boolean;              // always false in approach A
      metadata: {
        entry_id: string;               // Contentful entry ID
        type: string;                   // "solutionArticle" | "event" | "course"
        publishedDate: string;          // ISO date string
        authorIds: string[];            // array of author IDs
        body: string;                   // chunk text
        title_en: string;               // title in English (and other locales as title_{locale})
        slug_en: string;                // slug in English (and other locales as slug_{locale})
        subtitle_en?: string;           // subtitle if available
      }
    }
  ],
  totalPrimary: number;    // total results returned from Pinecone
  totalFallback: number;   // always 0 — no fallback in approach A
  page: number;
  pageSize: number;
  timing: {
    embedMs: number;       // dense embedding latency (Pinecone inference API)
    sparseMs: number;      // sparse embedding latency (Pinecone inference API)
    pineconeMs: number;    // Pinecone query latency
    totalMs: number;       // total wall time
  }
}
```

---

## Query Examples

### Basic English search

```bash
curl "http://localhost:8301/search?q=digital+transformation&locale=en"
```

```bash
curl -X POST http://localhost:8301/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "digital transformation", "locale": "en", "pageSize": 5}'
```

### Non-English query (pure dense)

For non-English locales, the sparse component is disabled (`alpha=1.0`) because `pinecone-sparse-english-v0` is English-focused:

```bash
# French
curl "http://localhost:8301/search?q=transformation+num%C3%A9rique&locale=fr"

# German
curl "http://localhost:8301/search?q=digitale+Transformation&locale=de"

# Spanish
curl "http://localhost:8301/search?q=transformaci%C3%B3n+digital&locale=es"
```

**What to check:** Since all locales are in the same index, results will include content from all languages. Compare with `search-api-b` where results are filtered to the requested locale.

### Cross-lingual search

Force pure dense regardless of locale:

```bash
curl "http://localhost:8301/search?q=open+government+data&locale=en&crossLingual=true&pageSize=10"
```

### Filtering by content type

```bash
# Only events
curl "http://localhost:8301/search?q=leadership+workshop&locale=en&contentType=event"

# Only articles
curl "http://localhost:8301/search?q=climate+adaptation+policy&locale=en&contentType=solutionArticle"

# Only courses
curl "http://localhost:8301/search?q=data+analysis+skills&locale=en&contentType=course"
```

### Enrollment boost

Content IDs in `enrolledIds` receive a 1.3× score boost:

```bash
curl -X POST http://localhost:8301/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "procurement reform",
    "locale": "en",
    "enrolledIds": ["abc123EntryId", "def456EntryId"],
    "pageSize": 10
  }'
```

**What to check:** Results where `isEnrolled: true` should appear near the top. Compare `score` vs `originalScore` to see the 1.3× boost applied.

### Testing timing

```bash
curl -s "http://localhost:8301/search?q=public+sector+innovation&locale=en" | \
  node -e "const d=require('fs').readFileSync('/dev/stdin','utf8'); console.log(JSON.parse(d).timing)"
```

Expected latency profile for Approach A:

- `embedMs`: `15–40` — dense embedding via Pinecone inference
- `sparseMs`: `15–40` — sparse embedding via Pinecone inference (runs in parallel with dense)
- `pineconeMs`: `15–50` — Pinecone query
- `totalMs`: `50–130` at p95 (two remote encode calls vs one in Approach B)

---

## Key Design Decisions

### Why NestJS?

NestJS provides dependency injection, module scoping, and a clear lifecycle (`onModuleInit`) that maps cleanly to this service's startup needs: connect to Pinecone and keep the client available for the lifetime of the server. The identical structure between `search-api-a` and `search-api-b` also makes the comparison straightforward.

### Why `alpha = 0.7` for English, `1.0` for other locales?

`pinecone-sparse-english-v0` was trained on English text. For English queries, mixing 70% dense with 30% sparse improves exact-term recall (acronyms, proper nouns). For non-English queries, the sparse weights are unreliable, so the sparse component is dropped entirely and pure dense (`alpha=1.0`) is used. This is different from Approach B, where the BM25 encoder supports all 17 locales with locale-aware tokenisation.

### Why dotproduct metric instead of cosine similarity?

Pinecone's hybrid search requires the `dotproduct` metric. Cosine similarity normalises vectors before computing similarity, which removes the ability to scale them by `alpha`. With `dotproduct`, multiplying the dense vector by 0.7 and the sparse vector by 0.3 directly controls their relative contribution to the final score.

### Why fetch dense and sparse in parallel?

Unlike Approach B (where BM25 runs in-process in under 1ms), Approach A requires a Pinecone API call for sparse encoding. Running both encode calls with `Promise.all` keeps the overhead close to the latency of a single call rather than doubling it. In practice, both calls route to the same Pinecone region (`europe-west4`), so they compete for the same network round-trip.

### Why over-fetch from Pinecone?

Pinecone's `topK` parameter limits results before any re-ranking. If we fetched exactly `pageSize` results and then applied the enrollment boost, enrolled content ranked just outside the requested page could never surface. Fetching `page × pageSize × 3` (capped at 200) gives the re-ranker enough candidates while keeping the Pinecone payload reasonable.

### Why no locale filter?

`search-embeddings-A` stores one vector per content chunk regardless of locale — localized titles and slugs are embedded as metadata fields (`title_en`, `slug_fr`, etc.) rather than separate vectors. There is no `locale` metadata field to filter on. Approach B takes the opposite stance: one vector per (entry, locale, chunk), enabling precise locale filtering at query time.

---

## Approach A vs Approach B

| Aspect                  | Approach A (`search-api-a`)                             | Approach B (`search-api-b`)                          |
| ----------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| **Pinecone index**      | `platform`                                              | `platform-v2`                                        |
| **Sparse model**        | `pinecone-sparse-english-v0` (Pinecone-hosted)          | BM25 corpus-fitted (TypeScript, in-process)          |
| **Encode calls**        | 2 Pinecone API calls (dense + sparse, parallel)         | 1 Pinecone API call + local BM25 (<1ms)              |
| **Locale vectors**      | Single vector per chunk (all locales merged)            | One vector per (entry, locale, chunk)                |
| **Locale filter**       | Not applied                                             | `locale: { $eq: userLocale }`                        |
| **English fallback**    | Not needed                                              | Triggered when primary page under-full               |
| **Non-English sparse**  | Disabled (alpha=1.0 for non-en locales)                 | Enabled (locale-aware BM25 tokenisation)             |
| **Content ID field**    | `entry_id`                                              | `content_id`                                         |
| **Content type field**  | `type`                                                  | `content_type`                                       |
| **sparseMs / bm25Ms**   | ~15–40ms (network call)                                 | <1ms (in-process)                                    |
| **totalMs (expected)**  | ~50–130ms p95                                           | ~30–100ms p95                                        |
| **Setup complexity**    | None — no corpus stats file required                    | Requires `bm25_corpus_stats.json` from embedding run |

---

## Known Limitations

**Sparse model is English-only**
`pinecone-sparse-english-v0` produces meaningful sparse weights only for English text. Non-English queries automatically fall back to pure dense (`alpha=1.0`). Approach B's per-locale BM25 encoder supports all 17 languages with proper stemming and stopword removal.

**No locale-scoped results**
Because all locales share the same vector space in the `platform` index, a French query may surface English, German, or Spanish content alongside French content. Approach B guarantees locale-specific results via per-locale vectors and metadata filtering.

**Pagination is stateless**
Each request independently fetches up to 200 results from Pinecone. There is no server-side cursor or cache. For pages beyond ~7 (at the default page size of 10), results may shift between requests if the index is being updated. This is acceptable for comparison purposes; production would use cursor-based pagination or a Redis result cache.

**Two remote encode calls per request**
Approach A requires two Pinecone inference API calls per query (dense + sparse) versus one in Approach B. Even with parallel execution, this adds network overhead and increases the chance of a cold-start latency spike from either call.
