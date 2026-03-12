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
const CONFIG = {
    indexName: 'platform',
    namespace: 'search',
    denseModel: 'multilingual-e5-large',
    sparseModel: 'pinecone-sparse-english-v0',
    defaultAlpha: 0.7,
    nonEnglishAlpha: 1.0,
    crossLingualAlpha: 1.0,
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
    }
    async search(req) {
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
        const embedStart = performance.now();
        const sparseStart = performance.now();
        const [denseVec, sparseVec] = await Promise.all([
            this.embedDense(req.query),
            this.embedSparse(req.query),
        ]);
        const embedMs = round1(performance.now() - embedStart);
        const sparseMs = round1(performance.now() - sparseStart);
        const pineconeStart = performance.now();
        const rawResults = await this.hybridQuery(denseVec, sparseVec, alpha, req.contentType, topK);
        const pineconeMs = round1(performance.now() - pineconeStart);
        const seen = new Map();
        for (const match of rawResults) {
            const entityId = (match.metadata?.['course_id'] ??
                match.metadata?.['entry_id'] ??
                match.id);
            const existing = seen.get(entityId);
            if (!existing || (match.score ?? 0) > (existing.score ?? 0)) {
                seen.set(entityId, match);
            }
        }
        const deduped = [...seen.values()];
        const enrolledSet = new Set(req.enrolledIds ?? []);
        const allMatches = deduped.map((match) => {
            const entryId = (match.metadata?.['journey_id'] ??
                match.metadata?.['entry_id'] ??
                '');
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
        const startIdx = (page - 1) * pageSize;
        const pageMatches = allMatches.slice(startIdx, startIdx + pageSize);
        return {
            matches: pageMatches,
            totalPrimary: deduped.length,
            totalFallback: 0,
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
    async embedDense(text) {
        const response = await this.pinecone.inference.embed(CONFIG.denseModel, [text], { inputType: 'query', truncate: 'END' });
        return response[0].values;
    }
    async embedSparse(text) {
        const response = await this.pinecone.inference.embed(CONFIG.sparseModel, [text], { inputType: 'query' });
        const result = response[0];
        return result.sparseValues ?? { indices: [], values: [] };
    }
    async hybridQuery(denseVec, sparseVec, alpha, contentType, topK) {
        const scaledDense = denseVec.map((v) => v * alpha);
        const scaledSparse = alpha < 1.0 && sparseVec.indices.length > 0
            ? {
                indices: sparseVec.indices,
                values: sparseVec.values.map((v) => v * (1 - alpha)),
            }
            : undefined;
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
};
exports.SearchService = SearchService;
exports.SearchService = SearchService = SearchService_1 = __decorate([
    (0, common_1.Injectable)()
], SearchService);
function round1(ms) {
    return Math.round(ms * 10) / 10;
}
//# sourceMappingURL=search.service.js.map