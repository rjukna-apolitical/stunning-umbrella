"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var SearchService_1;
Object.defineProperty(exports, "__esModule", { value: true });
exports.SearchService = void 0;
const common_1 = require("@nestjs/common");
const pinecone_1 = require("@pinecone-database/pinecone");
const bm25_encoder_1 = require("./bm25-encoder");
const CONFIG = {
    indexName: 'platform-v2',
    namespace: 'search',
    denseModel: 'multilingual-e5-large',
    defaultAlpha: 0.7,
    crossLingualAlpha: 1.0,
    fallbackAlpha: 1.0,
    enrollmentBoost: 1.3,
    maxFetch: 200,
};
let SearchService = SearchService_1 = class SearchService {
    constructor() {
        this.logger = new common_1.Logger(SearchService_1.name);
    }
    async onModuleInit() {
        this.pinecone = new pinecone_1.Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        this.index = this.pinecone.Index(CONFIG.indexName);
        this.logger.log(`Pinecone index '${CONFIG.indexName}' connected`);
        this.bm25 = new bm25_encoder_1.BM25Encoder();
        const statsPath = process.env.BM25_STATS_PATH ?? '../search-embeddings-B/data/bm25_corpus_stats.json';
        await this.bm25.loadStats(statsPath);
        const stats = this.bm25.getStats();
        this.logger.log(`BM25 loaded: ${stats.docCount} docs, ${stats.vocabularySize} terms, avgLen=${stats.avgDocLen.toFixed(0)}`);
    }
    async search(req) {
        const totalStart = performance.now();
        const page = req.page ?? 1;
        const pageSize = req.pageSize ?? 10;
        const topK = Math.min(page * pageSize * 3, CONFIG.maxFetch);
        const alpha = req.crossLingual ? CONFIG.crossLingualAlpha : CONFIG.defaultAlpha;
        const embedStart = performance.now();
        const denseVec = await this.embedQuery(req.query);
        const embedMs = performance.now() - embedStart;
        const bm25Start = performance.now();
        const sparseVec = this.bm25.encodeQuery(req.query, req.locale);
        const bm25Ms = performance.now() - bm25Start;
        const pineconeStart = performance.now();
        const primaryResults = req.crossLingual
            ? await this.hybridQuery(denseVec, sparseVec, alpha, undefined, req.contentType, topK)
            : await this.hybridQuery(denseVec, sparseVec, alpha, req.locale, req.contentType, topK);
        const pineconeMs = performance.now() - pineconeStart;
        const primarySeen = new Map();
        for (const match of primaryResults) {
            const contentId = match.metadata?.['content_id'];
            const key = contentId ?? match.id;
            const existing = primarySeen.get(key);
            if (!existing || (match.score ?? 0) > (existing.score ?? 0)) {
                primarySeen.set(key, match);
            }
        }
        const dedupedPrimary = [...primarySeen.values()];
        const seenContentIds = new Set(dedupedPrimary.map((m) => m.metadata?.['content_id']).filter(Boolean));
        let fallbackResults = [];
        const primaryCoversPage = dedupedPrimary.length >= page * pageSize;
        if (!primaryCoversPage && req.locale !== 'en' && !req.crossLingual) {
            const fallbackK = Math.min((page * pageSize - dedupedPrimary.length) * 2, CONFIG.maxFetch);
            const fallbackRaw = await this.hybridQuery(denseVec, sparseVec, CONFIG.fallbackAlpha, 'en', req.contentType, fallbackK);
            for (const match of fallbackRaw) {
                const contentId = match.metadata?.['content_id'];
                if (contentId && !seenContentIds.has(contentId)) {
                    seenContentIds.add(contentId);
                    fallbackResults.push(match);
                }
            }
        }
        const enrolledSet = new Set(req.enrolledIds ?? []);
        const allMatches = [];
        for (const match of dedupedPrimary) {
            const contentId = match.metadata?.['content_id'];
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
            const contentId = match.metadata?.['content_id'];
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
    async embedQuery(text) {
        const response = await this.pinecone.inference.embed(CONFIG.denseModel, [text], { inputType: 'query', truncate: 'END' });
        return response[0].values;
    }
    async hybridQuery(denseVec, sparseVec, alpha, locale, contentType, topK) {
        const scaledDense = denseVec.map((v) => v * alpha);
        const scaledSparse = alpha < 1.0 && sparseVec.indices.length > 0
            ? {
                indices: sparseVec.indices,
                values: sparseVec.values.map((v) => v * (1 - alpha)),
            }
            : undefined;
        const filterClauses = [];
        if (locale)
            filterClauses.push({ locale: { $eq: locale } });
        if (contentType)
            filterClauses.push({ content_type: { $eq: contentType } });
        const filter = filterClauses.length === 0
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
};
exports.SearchService = SearchService;
exports.SearchService = SearchService = SearchService_1 = __decorate([
    (0, common_1.Injectable)()
], SearchService);
function round1(ms) {
    return Math.round(ms * 10) / 10;
}
//# sourceMappingURL=search.service.js.map