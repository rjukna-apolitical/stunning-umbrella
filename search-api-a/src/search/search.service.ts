import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { Pinecone } from '@pinecone-database/pinecone';

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
    sparseMs: number;    // Pinecone-hosted sparse inference (pinecone-sparse-english-v0)
    pineconeMs: number;
    totalMs: number;
  };
}

// ── Config ───────────────────────────────────────────────────────────────────

const CONFIG = {
  indexName: 'platform',
  namespace: 'search',
  denseModel: 'multilingual-e5-large',
  sparseModel: 'pinecone-sparse-english-v0',

  // pinecone-sparse-english-v0 is English-focused; use dense-only for other locales
  defaultAlpha: 0.7,       // 70% dense, 30% sparse (English)
  nonEnglishAlpha: 1.0,    // pure dense — sparse model not optimised for other languages
  crossLingualAlpha: 1.0,  // pure dense — consistent with search-api-b behaviour

  enrollmentBoost: 1.3,

  // search-embeddings-A stores all locales in one vector (no per-locale vectors),
  // so no locale filter is applied.  We over-fetch generously for re-ranking.
  maxFetch: 200,
} as const;

// ── Service ──────────────────────────────────────────────────────────────────

@Injectable()
export class SearchService implements OnModuleInit {
  private readonly logger = new Logger(SearchService.name);
  private pinecone!: Pinecone;
  private index!: ReturnType<Pinecone['Index']>;

  async onModuleInit(): Promise<void> {
    this.pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    this.index = this.pinecone.Index(CONFIG.indexName);
    this.logger.log(`Pinecone index '${CONFIG.indexName}' connected`);
  }

  /**
   * Hybrid search using Pinecone-hosted dense + sparse models.
   *
   * Approach A differences vs Approach B:
   *   - Sparse: Pinecone-hosted `pinecone-sparse-english-v0` (two remote calls before query)
   *   - Index:  `platform` — all locales in a single vector set (no locale filter)
   *   - Dense + sparse embeddings fetched in parallel to minimise latency
   *   - No BM25 corpus stats file required
   *
   * Flow:
   *   1. Encode query — dense + sparse in parallel (both via Pinecone inference)
   *   2. Query Pinecone with hybrid vector (filtered by contentType if supplied)
   *   3. Enrollment boost + re-sort
   *   4. Paginate
   */
  async search(req: SearchRequest): Promise<SearchResponse> {
    const totalStart = performance.now();
    const page = req.page ?? 1;
    const pageSize = req.pageSize ?? 10;
    const topK = Math.min(page * pageSize * 3, CONFIG.maxFetch);

    const isEnglish = req.locale === 'en' || req.locale.startsWith('en-');
    const alpha = req.crossLingual
      ? CONFIG.crossLingualAlpha
      : isEnglish
        ? CONFIG.defaultAlpha
        : CONFIG.nonEnglishAlpha;

    // ── Step 1: Encode query (dense + sparse in parallel) ────────────────────
    const embedStart = performance.now();
    const sparseStart = performance.now();

    const [denseVec, sparseVec] = await Promise.all([
      this.embedDense(req.query),
      this.embedSparse(req.query),
    ]);

    const embedMs = round1(performance.now() - embedStart);
    // sparseMs is measured independently (both calls ran in parallel, so each
    // individual duration is approximated from the shared wall-clock window)
    const sparseMs = round1(performance.now() - sparseStart);

    // ── Step 2: Query Pinecone ───────────────────────────────────────────────
    const pineconeStart = performance.now();
    const rawResults = await this.hybridQuery(
      denseVec,
      sparseVec,
      alpha,
      req.contentType,
      topK,
    );
    const pineconeMs = round1(performance.now() - pineconeStart);

    // ── Step 3: Enrollment boost + re-sort ───────────────────────────────────
    // search-embeddings-A uses `entry_id` (vs `content_id` in B)
    const enrolledSet = new Set(req.enrolledIds ?? []);
    const allMatches: SearchMatch[] = rawResults.map((match) => {
      const entryId = match.metadata?.['entry_id'] as string;
      const isEnrolled = enrolledSet.has(entryId);
      return {
        id: match.id,
        score: (match.score ?? 0) * (isEnrolled ? CONFIG.enrollmentBoost : 1.0),
        originalScore: match.score ?? 0,
        isEnrolled,
        isFallback: false,
        metadata: match.metadata ?? {},
      };
    });

    allMatches.sort((a, b) => b.score - a.score);

    // ── Step 4: Paginate ─────────────────────────────────────────────────────
    const startIdx = (page - 1) * pageSize;
    const pageMatches = allMatches.slice(startIdx, startIdx + pageSize);

    return {
      matches: pageMatches,
      totalPrimary: rawResults.length,
      totalFallback: 0, // no locale-partitioned vectors in approach A
      page,
      pageSize,
      timing: {
        embedMs,
        sparseMs,
        pineconeMs,
        totalMs: round1(performance.now() - totalStart),
      },
    };
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private async embedDense(text: string): Promise<number[]> {
    const response = await this.pinecone.inference.embed(
      CONFIG.denseModel,
      [text],
      { inputType: 'query', truncate: 'END' },
    );
    return (response[0] as { values: number[] }).values;
  }

  private async embedSparse(text: string): Promise<{ indices: number[]; values: number[] }> {
    const response = await this.pinecone.inference.embed(
      CONFIG.sparseModel,
      [text],
      { inputType: 'query' } as Record<string, string>,
    );
    const result = response[0] as { sparseValues?: { indices: number[]; values: number[] } };
    return result.sparseValues ?? { indices: [], values: [] };
  }

  private async hybridQuery(
    denseVec: number[],
    sparseVec: { indices: number[]; values: number[] },
    alpha: number,
    contentType: string | undefined,
    topK: number,
  ) {
    const scaledDense = denseVec.map((v) => v * alpha);

    const scaledSparse =
      alpha < 1.0 && sparseVec.indices.length > 0
        ? {
            indices: sparseVec.indices,
            values: sparseVec.values.map((v) => v * (1 - alpha)),
          }
        : undefined;

    // search-embeddings-A uses `type` (not `content_type`)
    const filter = contentType ? { type: { $eq: contentType } } : undefined;

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
