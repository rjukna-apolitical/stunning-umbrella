"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BM25Encoder = void 0;
const fs_1 = require("fs");
const natural = require("natural");
const stopword_1 = require("stopword");
const HASH_SPACE = 2 ** 18;
const STEMMERS = {
    en: natural.PorterStemmer,
    fr: natural.PorterStemmerFr,
    'fr-CA': natural.PorterStemmerFr,
    es: natural.PorterStemmerEs,
    'es-419': natural.PorterStemmerEs,
    pt: natural.PorterStemmerPt,
    'pt-BR': natural.PorterStemmerPt,
    it: natural.PorterStemmerIt,
};
const BASE_STOPWORDS = {
    en: stopword_1.eng,
    ar: stopword_1.arb,
    fr: stopword_1.fra,
    'fr-CA': stopword_1.fra,
    de: stopword_1.deu,
    id: stopword_1.ind,
    it: stopword_1.ita,
    pt: stopword_1.por,
    'pt-BR': stopword_1.por,
    es: stopword_1.spa,
    'es-419': stopword_1.spa,
    pl: stopword_1.pol,
};
const CUSTOM_STOPWORDS = {
    'sr-Cyrl': ['и', 'у', 'је', 'да', 'на', 'се', 'за', 'од', 'са', 'су', 'то', 'не', 'ће', 'као', 'из', 'или', 'али', 'овај', 'тај', 'који', 'све', 'бити', 'може'],
    ja: ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また', 'もの', 'という', 'あり'],
    ko: ['이', '그', '저', '것', '수', '등', '들', '및', '에', '는', '을', '를', '의', '가', '으로', '에서', '와', '과', '도', '로', '한', '하다', '있다', '되다', '없다'],
    vi: ['của', 'và', 'là', 'có', 'được', 'cho', 'không', 'trong', 'để', 'với', 'này', 'các', 'từ', 'một', 'những', 'đã', 'theo', 'về', 'sẽ', 'hay', 'như', 'cũng', 'khi', 'tại', 'bị', 'do', 'đến', 'nên', 'vì', 'rất'],
    uk: ['і', 'в', 'на', 'з', 'що', 'не', 'у', 'як', 'до', 'це', 'та', 'за', 'від', 'але', 'є', 'по', 'для', 'він', 'яка', 'який', 'все', 'або', 'вже', 'ще', 'так', 'бути', 'може'],
};
const DOMAIN_STOPWORDS_EN = [
    'government', 'policy', 'public', 'sector', 'service',
    'strategy', 'framework', 'approach', 'initiative', 'programme',
];
function murmur3x86_32(key, seed = 0) {
    const bytes = Buffer.from(key, 'utf8');
    const len = bytes.length;
    const c1 = 0xcc9e2d51;
    const c2 = 0x1b873593;
    let h = seed >>> 0;
    const nblocks = len >> 2;
    for (let i = 0; i < nblocks; i++) {
        let k = bytes.readUInt32LE(i * 4);
        k = Math.imul(k, c1) >>> 0;
        k = ((k << 15) | (k >>> 17)) >>> 0;
        k = Math.imul(k, c2) >>> 0;
        h ^= k;
        h = ((h << 13) | (h >>> 19)) >>> 0;
        h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
    }
    let k = 0;
    const tail = nblocks * 4;
    switch (len & 3) {
        case 3: k ^= bytes[tail + 2] << 16;
        case 2: k ^= bytes[tail + 1] << 8;
        case 1:
            k ^= bytes[tail];
            k = Math.imul(k, c1) >>> 0;
            k = ((k << 15) | (k >>> 17)) >>> 0;
            k = Math.imul(k, c2) >>> 0;
            h ^= k;
    }
    h = (h ^ len) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x85ebca6b) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 0xc2b2ae35) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h;
}
class BM25Encoder {
    constructor() {
        this.docCount = 0;
        this.avgDocLen = 0;
        this.docFreq = new Map();
        this.stopwords = new Map();
        const allLocales = new Set([
            ...Object.keys(BASE_STOPWORDS),
            ...Object.keys(CUSTOM_STOPWORDS),
        ]);
        for (const locale of allLocales) {
            const stops = new Set([
                ...(BASE_STOPWORDS[locale] ?? []),
                ...(CUSTOM_STOPWORDS[locale] ?? []),
            ]);
            const parent = locale.split('-')[0];
            if (parent !== locale && CUSTOM_STOPWORDS[parent]) {
                for (const w of CUSTOM_STOPWORDS[parent])
                    stops.add(w);
            }
            if (locale === 'en') {
                for (const w of DOMAIN_STOPWORDS_EN)
                    stops.add(w);
            }
            this.stopwords.set(locale, stops);
        }
    }
    async loadStats(statsPath) {
        const raw = await fs_1.promises.readFile(statsPath, 'utf-8');
        const stats = JSON.parse(raw);
        this.docCount = stats.doc_count;
        this.avgDocLen = stats.avg_doc_len;
        this.docFreq = new Map(Object.entries(stats.doc_freq));
    }
    getStats() {
        return {
            docCount: this.docCount,
            avgDocLen: this.avgDocLen,
            vocabularySize: this.docFreq.size,
        };
    }
    encodeQuery(text, locale) {
        const tokens = this.tokenize(text, locale);
        if (tokens.length === 0)
            return { indices: [], values: [] };
        const tfCounts = new Map();
        for (const t of tokens)
            tfCounts.set(t, (tfCounts.get(t) ?? 0) + 1);
        const pairs = [];
        for (const [term, tf] of tfCounts) {
            const df = this.docFreq.get(term) ?? 0;
            const idf = df === 0
                ? Math.log(this.docCount + 1)
                : Math.log((this.docCount - df + 0.5) / (df + 0.5) + 1);
            if (idf > 0) {
                pairs.push([this.hashToken(term), idf * Math.min(tf, 2)]);
            }
        }
        const merged = new Map();
        for (const [idx, score] of pairs) {
            merged.set(idx, (merged.get(idx) ?? 0) + score);
        }
        const sorted = [...merged.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 1000);
        return {
            indices: sorted.map(([idx]) => idx),
            values: sorted.map(([, v]) => Math.round(v * 10000) / 10000),
        };
    }
    tokenize(text, locale) {
        const tokens = (text.toLowerCase().match(/[\p{L}0-9]+/gu) ?? []).filter((t) => t.length > 1);
        const stops = this.stopwords.get(locale) ?? new Set();
        const filtered = tokens.filter((t) => !stops.has(t));
        const stemmer = STEMMERS[locale];
        const stemmed = stemmer ? filtered.map((t) => stemmer.stem(t)) : filtered;
        return stemmed.map((t) => `${locale}:${t}`);
    }
    hashToken(token) {
        return murmur3x86_32(token) % HASH_SPACE;
    }
}
exports.BM25Encoder = BM25Encoder;
//# sourceMappingURL=bm25-encoder.js.map