export interface SparseVector {
    indices: number[];
    values: number[];
}
export interface BM25Stats {
    docCount: number;
    avgDocLen: number;
    vocabularySize: number;
}
export declare class BM25Encoder {
    private docCount;
    private avgDocLen;
    private docFreq;
    private readonly stopwords;
    constructor();
    loadStats(statsPath: string): Promise<void>;
    getStats(): BM25Stats;
    encodeQuery(text: string, locale: string): SparseVector;
    private tokenize;
    private hashToken;
}
