import { OnModuleInit } from '@nestjs/common';
export interface SearchRequest {
    query: string;
    locale: string;
    contentType?: string;
    page?: number;
    pageSize?: number;
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
        sparseMs: number;
        pineconeMs: number;
        totalMs: number;
    };
}
export declare class SearchService implements OnModuleInit {
    private readonly logger;
    private pinecone;
    private index;
    onModuleInit(): Promise<void>;
    search(req: SearchRequest): Promise<SearchResponse>;
    private embedDense;
    private embedSparse;
    private hybridQuery;
}
