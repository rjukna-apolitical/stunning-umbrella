# RFC: Search Embeddings Architecture

**Status:** Draft
**Date:** 2026-03-11
**Authors:** Engineering Team

---

## Motivation

Our current search infrastructure suffers from two critical problems: **high latency** (400–700ms end-to-end) and **inconsistent relevance** across our 20 supported locales. As the platform expands its content library and user base across language regions, search quality directly impacts engagement and retention for public servants relying on the platform to find policy resources.

This RFC evaluates three embedding-based approaches to replace the existing search pipeline, with the goal of achieving:

- Sub-100ms query latency
- Accurate multilingual keyword and semantic search across 20 locales
- Scalable ingestion pipeline with manageable operational overhead
- Extensibility to new content types (e.g. community discussions)

---

## Considered Approaches

Three proof-of-concept implementations were built and evaluated:

- **Approach A** — Pinecone hybrid search with dense multilingual embeddings + Pinecone's learned sparse model (`pinecone-sparse-english-v0`), embedding English content only
- **Approach B** — Pinecone hybrid search with dense multilingual embeddings + custom corpus-fitted BM25 sparse vectors, embedding all 17 content locales
- **Approach C** — Pinecone dense-only search with Pinecone-hosted multilingual embeddings, extended with LLM summarisation of GetStream community feeds

### Approach Comparison

| **Approach** | Advantages | Disadvantages | Migration Effort |
| --- | --- | --- | --- |
| **A** — Pinecone dense (`multilingual-e5-large`, local) + sparse (`pinecone-sparse-english-v0`) on English content only | • Zero corpus setup — no IDF fitting required<br>• Single-phase pipeline (configure → embed)<br>• Multilingual query support out of the box (search in Spanish, retrieve English)<br>• Markdown-aware chunking preserves semantic boundaries<br>• Lean operational footprint | • Sparse model trained on general web data, not government/policy domain<br>• English-only indexed content loses native language term precision<br>• Large Docker image (torch + sentence-transformers, ~3 GB)<br>• Sparse API calls are English-only; cross-lingual keyword gaps remain<br>• High latency (400–700ms) inherited from current infrastructure | **M** |
| **B** — Pinecone dense (`multilingual-e5-large`, Pinecone-hosted) + custom multilingual BM25 on all 17 locales | • True native-language keyword matching across all 20 locales<br>• Domain-fitted IDF: terms like "government" and "policy" auto-downweighted<br>• Locale-prefixed token space prevents cross-language lexical collisions (e.g. German "gift" vs English "gift")<br>• Tier-based tokenisation supports Snowball, MeCab (Japanese), Kiwi (Korean), and whitespace fallback<br>• Lean Docker image (no torch/CUDA) via Pinecone-hosted dense inference | • Two-phase pipeline: corpus fitting must run before embedding<br>• BM25 statistics file (`bm25_corpus_stats.json`) must be versioned and synchronised across environments<br>• IDF must be recomputed on significant corpus changes<br>• Per-locale vectors multiply storage and upsert volume by ~17×<br>• High latency (400–700ms) inherited from current infrastructure | **L** |
| **C** — Pinecone dense-only (`multilingual-e5-large`, Pinecone-hosted) + GPT-4o summarisation of GetStream community feeds | • Introduces community discussion content as a searchable type<br>• LLM summarisation condenses noisy feed data into embedding-optimised text<br>• Map-reduce handles large feeds exceeding context window limits<br>• Lean Docker image (no torch/CUDA)<br>• Single-phase pipeline<br>• Privacy-aware metadata (`access_privacy`) | • No sparse/keyword component — exact term and acronym matching degrades<br>• GPT-4o cost per community re-index (~$0.125 per community at 100k chars)<br>• Summarisation introduces latency and non-determinism during ingestion<br>• Cosine metric (vs dotproduct used in A/B) — not directly comparable without reindexing<br>• Community feeds are English-only; multilingual discussion search not supported | **M** |

---

## Recommended Approach

**Recommended: Approach C as the primary architecture, incorporating the multilingual BM25 layer from Approach B as a follow-on milestone.**

The immediate recommendation is to adopt **Approach C** — Pinecone dense-only with community feed support — and pair it with **Meilisearch for keyword search**, replacing the sparse vector component entirely. This hybrid of Pinecone (semantic) + Meilisearch (keyword) addresses the two biggest production blockers from Approaches A and B: latency and operational complexity.

For multilingual keyword precision, the BM25 tokenisation work from **Approach B** should be contributed directly to the Meilisearch ingestion pipeline as a follow-on, leveraging Meilisearch's native language analyser hooks.

### Why This Is Superior to the Alternatives

| Concern | A | B | C + Meilisearch |
|---|---|---|---|
| Query latency | 400–700ms | 400–700ms | ~40ms (dense) + ~70ms (keyword) |
| Keyword accuracy (EN) | Good | Good | Excellent (Meilisearch BM25) |
| Keyword accuracy (multilingual) | Poor | Excellent | Good → Excellent (with B's tokenisers) |
| Semantic accuracy | Good | Good | Good |
| Corpus fitting required | No | Yes | No |
| Community search | No | No | Yes |
| Docker image size | Large (~3 GB) | Lean | Lean |
| Operational systems | 1 (Pinecone) | 1 (Pinecone) | 2 (Pinecone + Meilisearch) |
| Re-index cost | Low | Medium | Medium (GPT-4o per community) |
| Cross-lingual query support | Yes | Yes | Yes (dense via Pinecone) |

The primary latency problem is solved by decoupling semantic and keyword search into dedicated systems rather than forcing both into Pinecone's hybrid vector space. Pinecone query latency in hybrid mode is fundamentally constrained by sparse vector operations at scale; Meilisearch's inverted-index keyword search operates at an order of magnitude lower latency (~70ms) with a mature multilingual tokenisation story.

---

## Technical Design

### Architecture

```
                          ┌─────────────────────────────────────────────────┐
                          │                Search Query                      │
                          └──────────────────┬──────────────────────────────┘
                                             │
                          ┌──────────────────▼──────────────────────────────┐
                          │            Search API (NestJS)                  │
                          │  • Fanout to both backends in parallel           │
                          │  • RRF (Reciprocal Rank Fusion) merge            │
                          └──────────┬──────────────────────┬───────────────┘
                                     │                      │
              ┌──────────────────────▼─────┐  ┌────────────▼───────────────┐
              │    Pinecone (semantic)      │  │   Meilisearch (keyword)    │
              │  multilingual-e5-large      │  │  BM25 inverted index       │
              │  dense-only, cosine         │  │  per-locale analysers      │
              │  ~40ms p50                  │  │  ~70ms p50                 │
              └─────────────────────────────┘  └────────────────────────────┘
```

### Ingestion Pipeline

```
Contentful CMS ──── Locale-aware fetch ────┐
                                           ▼
GetStream feeds ─── Feed aggregation ──► GPT-4o ─── Summarise
                                           │
                               ┌───────────▼───────────┐
                               │  Chunker (400w / 50w) │
                               └───────────┬───────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                                     ▼
            Pinecone Inference API                  Meilisearch index
            (multilingual-e5-large)              (BM25 + locale analyser)
                        │                                     │
                  Dense vectors                      Tokenised documents
                  upserted to Pinecone              upserted to Meilisearch
```

### Main Components

**Dense Embedding (Pinecone-hosted)**
- Model: `multilingual-e5-large` (1024 dimensions)
- Inference via Pinecone API — no local GPU/torch dependency
- Input prefix: `passage: {text}` at ingestion; `query: {text}` at search time
- Metric: `cosine`
- Batch size: 64 vectors per upsert

**Keyword Search (Meilisearch)**
- Inverted-index BM25 at native Meilisearch speed (~70ms)
- Per-locale documents stored with locale tag for filter-based scoping
- Multilingual analyser configured per locale (Meilisearch supports custom tokenisation)
- The tokeniser tiers from Approach B (Snowball, MeCab, Kiwi) to be integrated via Meilisearch's custom analyser API as a follow-on

**Community Feed Summarisation**
- GetStream feeds fetched via cursor-based pagination (25 activities/page)
- GPT-4o summarisation with domain-specific system prompt (public servant context)
- Map-reduce for feeds > 100k characters (80k-char chunks, 2k overlap)
- Resulting summary chunked and embedded identically to article/event/course content
- `access_privacy` metadata propagated to both Pinecone and Meilisearch for ACL filtering

**Result Merging (Reciprocal Rank Fusion)**
- Both backends queried in parallel
- Results merged using RRF: `score(d) = Σ 1 / (k + rank(d))` where k=60
- Final top-N returned to client
- Latency dominated by slower of the two backends (~70ms)

**Metadata Schema (unified across all content types)**

```json
{
  "content_id": "string",
  "content_type": "solutionArticle | event | course | community",
  "locale": "en | fr | de | ...",
  "title": "string",
  "slug": "string",
  "snippet": "first 300 chars of chunk",
  "published_date": 1709251200,
  "available_locales": ["en", "fr"],
  "access_privacy": "public | private",
  "banner_url": "string (optional)"
}
```

### Notable Technology Decisions

| Decision | Rationale |
|---|---|
| Pinecone-hosted dense inference over local SentenceTransformer | Eliminates ~3 GB Docker layer (torch + CUDA); consistent latency across environments; no GPU provisioning required |
| Meilisearch over Pinecone sparse vectors for keyword | ~10× lower keyword latency; native multilingual tokeniser support; Meilisearch UI for query debugging; avoids per-locale sparse vector multiplication |
| Dense-only in Pinecone (no hybrid) | Hybrid mode reintroduces sparse latency bottleneck; keyword accuracy delegated to Meilisearch where it is handled more efficiently |
| Cosine metric over dotproduct | Appropriate for normalised dense-only vectors; dotproduct only required for hybrid search (where vector magnitude carries signal) |
| GPT-4o for community summarisation | Community feeds are conversational and noisy; raw text embedding produces poor recall; domain-tuned summaries significantly improve semantic retrieval |
| Word-count chunking (400w / 50w) over markdown-aware | Consistent chunk sizes improve embedding quality for shorter snippets; markdown-aware (Approach A) offers marginal gains not worth the added complexity for multilingual content |

---

## Open Questions & Risks

| Risk / Question | Severity | Mitigation |
|---|---|---|
| **Meilisearch multilingual analyser coverage** — Meilisearch supports custom tokenisers but CJK language support (Japanese MeCab, Korean Kiwi) requires validation | High | Prototype B's Tier 3 tokenisers as Meilisearch custom analysers before committing to production; fallback to whitespace tokenisation for CJK locales initially |
| **GPT-4o re-index cost at scale** — If community count grows to 1,000+, full re-index costs ~$125+ | Medium | Cache summarisation output; only re-summarise communities with new feed activity since last index run (activity-delta detection via GetStream timestamps) |
| **RRF tuning** — k=60 is standard but may need tuning for domain-specific result distribution | Medium | A/B test RRF k values against held-out query relevance judgements; consider boosting Pinecone semantic score for navigational queries |
| **IDF from Approach B discarded** — Moving to Meilisearch loses the domain-fitted BM25 statistics (government/policy term downweighting) | Low–Medium | Meilisearch BM25 is corpus-fitted automatically from indexed documents; domain downweighting will emerge naturally as the corpus grows |
| **Pinecone latency regression** — Current 400–700ms is partially attributed to query-time sparse encoding; dense-only query latency needs validation at production query volume | Medium | Load test Pinecone dense-only namespace with 10k queries at expected peak QPS before launch |
| **GetStream API availability** — Embedding pipeline takes a hard dependency on GetStream uptime | Low | Implement fallback: skip community summarisation if GetStream is unreachable; alert and re-queue for async retry |
| **Cosine vs dotproduct for cross-lingual queries** — Multilingual queries where query language differs from document language have not been benchmarked under cosine metric | Medium | Run cross-lingual retrieval benchmarks on held-out test set before launch |

---

## Execution Plan

### Phase 1 — Foundation (Weeks 1–3)

- [ ] Provision Meilisearch instance with production-grade configuration (persistence, auth, CORS)
- [ ] Implement Meilisearch ingestion for articles, events, and courses using per-locale document schema
- [ ] Configure Meilisearch language analysers for Tier 1 locales (EN, FR, DE, ES, PT, IT, AR, ID)
- [ ] Implement RRF merge layer in Search API
- [ ] Migrate Approach C's Pinecone dense ingestion pipeline (articles, events, courses, communities) to production index (`platform-v3`)
- [ ] Benchmark p50/p99 query latency under load

### Phase 2 — Community Search & Validation (Weeks 4–6)

- [ ] Deploy community feed summarisation pipeline (Approach C's `embed/community.py` + `modules/summarise.py`)
- [ ] Implement activity-delta detection to avoid full re-summarisation on each run
- [ ] Run cross-lingual relevance evaluation against held-out query set (target: MRR@10 ≥ current baseline)
- [ ] Validate `access_privacy` ACL filtering in both Pinecone and Meilisearch

### Phase 3 — Multilingual Keyword Depth (Weeks 7–10)

- [ ] Port Approach B's Tier 2/3 tokenisers (Serbian, Japanese, Korean, Vietnamese) as Meilisearch custom analysers
- [ ] Validate CJK keyword recall against baseline
- [ ] Implement Tier 4 whitespace fallback for Polish and Ukrainian
- [ ] Production cutover with traffic shadowing (old index → new index comparison at 5% traffic)
- [ ] Decommission legacy Pinecone hybrid index

### Key Milestones

| Milestone | Target |
|---|---|
| EN/FR/DE keyword search via Meilisearch live | End of Week 3 |
| Community search live | End of Week 6 |
| All 20 locales covered in Meilisearch | End of Week 9 |
| Legacy index decommissioned | End of Week 10 |

### Required Collaboration

- **Platform Engineering** — Meilisearch infrastructure provisioning and monitoring
- **Product** — Define relevance evaluation criteria and query test sets
- **Content/Editorial** — Validate community summarisation quality across sample feeds
- **Security** — Review GetStream API credential handling and `access_privacy` ACL enforcement

---

## Relevant Links

- [Search revamp project page](https://www.notion.so/Search-Revamp-2fd68a6e3b92800283cdf2eb4ac8dea7?pvs=21)
- [Research page for keyword search custom logic](https://aclanthology.org/2021.mrl-1.12.pdf)
- `search-embeddings-A/` — English hybrid POC (dense local + Pinecone sparse)
- `search-embeddings-B/` — Multilingual hybrid POC (dense Pinecone + custom BM25)
- `search-embeddings-C/` — Dense-only + community summarisation POC
- `search-api-a/` — Search API implementation for Approach A
- `search-api-b/` — Search API implementation for Approach B
- `search-comparison/` — Side-by-side comparison UI for evaluating search quality
