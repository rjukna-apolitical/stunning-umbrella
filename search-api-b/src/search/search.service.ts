import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { Pinecone } from '@pinecone-database/pinecone';
import { BM25Encoder, SparseVector } from './bm25-encoder';

// ── Types ────────────────────────────────────────────────────────────────────

export interface SearchRequest {
  query: string;
  locale: string;
  contentType?: string;
  page?: number;       // 1-indexed, default 1
  pageSize?: number;   // default 10
  enrolledIds?: string[];
  crossLingual?: boolean;
}

export interface SearchMatch {
  id: string;
  score: number;
  originalScore: number;
  isEnrolled: boolean;
  isFallback: boolean;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  matches: SearchMatch[];
  totalPrimary: number;
  totalFallback: number;
  page: number;
  pageSize: number;
  timing: {
    embedMs: number;
    bm25Ms: number;
    pineconeMs: number;
    totalMs: number;
  };
}

// ── Config ───────────────────────────────────────────────────────────────────

const CONFIG = {
  indexName: 'platform-v2',
  namespace: 'search',
  denseModel: 'multilingual-e5-large',

  defaultAlpha: 0.7,       // 70% dense, 30% sparse
  crossLingualAlpha: 1.0,  // pure dense — sparse can't match across languages
  fallbackAlpha: 1.0,      // pure dense for English fallback

  enrollmentBoost: 1.3,

  // Per-request Pinecone fetch limit.
  // We over-fetch so that enrollment re-ranking + pagination stay accurate.
  // For page N we need at least N*pageSize results; cap at 200.
  maxFetch: 200,
} as const;

// ── Service ──────────────────────────────────────────────────────────────────

@Injectable()
export class SearchService implements OnModuleInit {
  private readonly logger = new Logger(SearchService.name);
  private pinecone!: Pinecone;
  private index!: ReturnType<Pinecone['Index']>;
  private bm25!: BM25Encoder;

  async onModuleInit(): Promise<void> {
    this.pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    this.index = this.pinecone.Index(CONFIG.indexName);
    this.logger.log(`Pinecone index '${CONFIG.indexName}' connected`);

    this.bm25 = new BM25Encoder();
    const statsPath =
      process.env.BM25_STATS_PATH ?? '../search-embeddings-B/data/bm25_corpus_stats.json';
    await this.bm25.loadStats(statsPath);
    const stats = this.bm25.getStats();
    this.logger.log(
      `BM25 loaded: ${stats.docCount} docs, ${stats.vocabularySize} terms, avgLen=${stats.avgDocLen.toFixed(0)}`,
    );
  }

  /**
   * Hybrid search with locale-aware fallback and enrollment boost.
   *
   * Flow:
   *   1. Encode query — dense (Pinecone inference) + sparse (BM25, <1 ms)
   *   2. Primary query filtered to user's locale
   *   3. English fallback if primary page is under-full and locale ≠ en
   *   4. Enrollment boost + re-sort
   *   5. Paginate
   */
  async search(req: SearchRequest): Promise<SearchResponse> {
    const totalStart = performance.now();
    const page = req.page ?? 1;
    const pageSize = req.pageSize ?? 10;  // BUG FIX: was req.page ?? 10 in original

    // How many results to fetch from Pinecone per query.
    // We need enough to fill `page` pages after dedup + re-rank.
    // Fetching page × pageSize × 3 gives a generous buffer; cap at maxFetch.
    const topK = Math.min(page * pageSize * 3, CONFIG.maxFetch);

    const alpha = req.crossLingual ? CONFIG.crossLingualAlpha : CONFIG.defaultAlpha;

    // ── Step 1: Encode query ─────────────────────────────────────────────────
    const embedStart = performance.now();
    const denseVec = await this.embedQuery(req.query);
    const embedMs = performance.now() - embedStart;

    const bm25Start = performance.now();
    const sparseVec = this.bm25.encodeQuery(req.query, req.locale);
    const bm25Ms = performance.now() - bm25Start;

    // ── Step 2: Primary search ───────────────────────────────────────────────
    const pineconeStart = performance.now();
    const primaryResults = req.crossLingual
      // Cross-lingual: omit locale filter so all languages are searched
      ? await this.hybridQuery(denseVec, sparseVec, alpha, undefined, req.contentType, topK)
      : await this.hybridQuery(denseVec, sparseVec, alpha, req.locale, req.contentType, topK);
    const pineconeMs = performance.now() - pineconeStart;

    // ── Step 3: Deduplicate primary results by content_id ────────────────────
    // Pinecone returns one match per chunk; keep only the highest-scoring chunk
    // per content_id so each article/event/course appears at most once.
    const primarySeen = new Map<string, typeof primaryResults[number]>();
    for (const match of primaryResults) {
      const contentId = match.metadata?.['content_id'] as string | undefined;
      const key = contentId ?? match.id;
      const existing = primarySeen.get(key);
      if (!existing || (match.score ?? 0) > (existing.score ?? 0)) {
        primarySeen.set(key, match);
      }
    }
    const dedupedPrimary = [...primarySeen.values()];

    // ── Step 4: English fallback ──────────────────────────────────────────────
    // Trigger when the requested page would be under-full from primary results alone.
    const seenContentIds = new Set<string>(
      dedupedPrimary.map((m) => m.metadata?.['content_id'] as string).filter(Boolean),
    );

    let fallbackResults: typeof primaryResults = [];
    const primaryCoversPage = dedupedPrimary.length >= page * pageSize;

    if (!primaryCoversPage && req.locale !== 'en' && !req.crossLingual) {
      const fallbackK = Math.min((page * pageSize - dedupedPrimary.length) * 2, CONFIG.maxFetch);
      const fallbackRaw = await this.hybridQuery(
        denseVec,
        sparseVec,
        CONFIG.fallbackAlpha,
        'en',
        req.contentType,
        fallbackK,
      );
      for (const match of fallbackRaw) {
        const contentId = match.metadata?.['content_id'] as string;
        if (contentId && !seenContentIds.has(contentId)) {
          seenContentIds.add(contentId);
          fallbackResults.push(match);
        }
      }
    }

    // ── Step 5: Enrollment boost + re-sort ───────────────────────────────────
    const enrolledSet = new Set(req.enrolledIds ?? []);
    const allMatches: SearchMatch[] = [];

    for (const match of dedupedPrimary) {
      const contentId = match.metadata?.['content_id'] as string;
      const isEnrolled = enrolledSet.has(contentId);
      allMatches.push({
        id: match.id,
        score: (match.score ?? 0) * (isEnrolled ? CONFIG.enrollmentBoost : 1.0),
        originalScore: match.score ?? 0,
        isEnrolled,
        isFallback: false,
        metadata: match.metadata ?? {},
      });
    }

    for (const match of fallbackResults) {
      const contentId = match.metadata?.['content_id'] as string;
      const isEnrolled = enrolledSet.has(contentId);
      allMatches.push({
        id: match.id,
        score: (match.score ?? 0) * (isEnrolled ? CONFIG.enrollmentBoost : 1.0),
        originalScore: match.score ?? 0,
        isEnrolled,
        isFallback: true,
        metadata: { ...match.metadata, is_fallback: true },
      });
    }

    allMatches.sort((a, b) => b.score - a.score);

    // ── Step 6: Paginate ─────────────────────────────────────────────────────
    const startIdx = (page - 1) * pageSize;
    const pageMatches = allMatches.slice(startIdx, startIdx + pageSize);

    return {
      matches: pageMatches,
      totalPrimary: dedupedPrimary.length,
      totalFallback: fallbackResults.length,
      page,
      pageSize,
      timing: {
        embedMs: round1(embedMs),
        bm25Ms: round1(bm25Ms),
        pineconeMs: round1(pineconeMs),
        totalMs: round1(performance.now() - totalStart),
      },
    };
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private async embedQuery(text: string): Promise<number[]> {
    const response = await this.pinecone.inference.embed(
      CONFIG.denseModel,
      [text],
      { inputType: 'query', truncate: 'END' },
    );
    return (response[0] as { values: number[] }).values;
  }

  private async hybridQuery(
    denseVec: number[],
    sparseVec: SparseVector,
    alpha: number,
    locale: string | undefined,
    contentType: string | undefined,
    topK: number,
  ) {
    const scaledDense = denseVec.map((v) => v * alpha);

    // Only pass sparse vector when alpha < 1 and there are sparse terms
    const scaledSparse =
      alpha < 1.0 && sparseVec.indices.length > 0
        ? {
            indices: sparseVec.indices,
            values: sparseVec.values.map((v) => v * (1 - alpha)),
          }
        : undefined;

    // Build metadata filter
    const filterClauses: Record<string, unknown>[] = [];
    if (locale) filterClauses.push({ locale: { $eq: locale } });
    if (contentType) filterClauses.push({ content_type: { $eq: contentType } });

    const filter =
      filterClauses.length === 0
        ? undefined
        : filterClauses.length === 1
          ? filterClauses[0]
          : { $and: filterClauses };

    const results = await this.index.namespace(CONFIG.namespace).query({
      vector: scaledDense,
      ...(scaledSparse ? { sparseVector: scaledSparse } : {}),
      ...(filter ? { filter } : {}),
      topK,
      includeMetadata: true,
    });

    return results.matches ?? [];
  }
}

function round1(ms: number): number {
  return Math.round(ms * 10) / 10;
}
