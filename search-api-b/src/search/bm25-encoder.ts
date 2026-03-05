/**
 * Multilingual BM25 query encoder — TypeScript port of modules/bm25.py
 *
 * Loads pre-fitted corpus statistics (doc_count, avg_doc_len, doc_freq) from
 * the JSON file produced by search-embeddings-B's `make fit-bm25`.
 *
 * Tokenisation fidelity vs Python:
 *   Tier 1 (en, fr, fr-CA, es, es-419, pt, pt-BR, it) — Snowball-style stemming
 *     via `natural` (PorterStemmer per language). Stems may differ slightly from
 *     Python's snowballstemmer for edge cases; quality is high for common words.
 *   Tier 1 (ar, de, id) — no stemmer available in `natural`; raw tokens used.
 *     Sparse recall is reduced but dense (multilingual-e5-large) compensates.
 *   Tier 2 (sr-Cyrl), Tier 4 (pl, uk) — whitespace + custom stopwords, no stemming.
 *   Tier 3 (ja, ko, vi) — whitespace fallback (same as Python when CJK libs absent).
 *
 * Hash compatibility: MurmurHash3 x86 32-bit implemented inline, hashing UTF-8
 * bytes — identical to Python `mmh3.hash(token, signed=False)` with seed=0.
 */

import { promises as fs } from 'fs';
import * as natural from 'natural';
import { eng, fra, deu, spa, por, ita, ind, arb, pol } from 'stopword';

// ── Constants (must match Python BM25) ──────────────────────────────────────
const HASH_SPACE = 2 ** 18; // 262,144

// ── Stemmers from `natural` ──────────────────────────────────────────────────
// Each exposes a `.stem(word: string): string` method.
const STEMMERS: Partial<Record<string, { stem(w: string): string }>> = {
  en: natural.PorterStemmer,
  fr: natural.PorterStemmerFr,
  'fr-CA': natural.PorterStemmerFr,
  es: natural.PorterStemmerEs,
  'es-419': natural.PorterStemmerEs,
  pt: natural.PorterStemmerPt,
  'pt-BR': natural.PorterStemmerPt,
  it: natural.PorterStemmerIt,
};

// ── Stopword lists ───────────────────────────────────────────────────────────
// Base lists from `stopword` package (NLTK-compatible).
const BASE_STOPWORDS: Partial<Record<string, readonly string[]>> = {
  en: eng,
  ar: arb,
  fr: fra,
  'fr-CA': fra,
  de: deu,
  id: ind,
  it: ita,
  pt: por,
  'pt-BR': por,
  es: spa,
  'es-419': spa,
  pl: pol,
};

// Custom stopwords for languages not covered by the stopword package.
// Mirrors Python CUSTOM_STOPWORDS in modules/bm25.py.
const CUSTOM_STOPWORDS: Record<string, string[]> = {
  'sr-Cyrl': ['и','у','је','да','на','се','за','од','са','су','то','не','ће','као','из','или','али','овај','тај','који','све','бити','може'],
  ja: ['の','に','は','を','た','が','で','て','と','し','れ','さ','ある','いる','も','する','から','な','こと','として','い','や','れる','など','なっ','ない','この','ため','その','あっ','よう','また','もの','という','あり'],
  ko: ['이','그','저','것','수','등','들','및','에','는','을','를','의','가','으로','에서','와','과','도','로','한','하다','있다','되다','없다'],
  vi: ['của','và','là','có','được','cho','không','trong','để','với','này','các','từ','một','những','đã','theo','về','sẽ','hay','như','cũng','khi','tại','bị','do','đến','nên','vì','rất'],
  uk: ['і','в','на','з','що','не','у','як','до','це','та','за','від','але','є','по','для','він','яка','який','все','або','вже','ще','так','бути','може'],
};

// Domain-specific stopwords for English. Mirrors Python DOMAIN_STOPWORDS_EN.
const DOMAIN_STOPWORDS_EN = [
  'government','policy','public','sector','service',
  'strategy','framework','approach','initiative','programme',
];

// ── MurmurHash3 x86 32-bit ───────────────────────────────────────────────────
// Hashes the UTF-8 byte representation of `key` with `seed` (default 0).
// Produces the same value as Python `mmh3.hash(key, signed=False)` with seed=0.
function murmur3x86_32(key: string, seed = 0): number {
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
    // eslint-disable-next-line no-fallthrough
    case 3: k ^= bytes[tail + 2] << 16;
    // eslint-disable-next-line no-fallthrough
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

// ── Types ────────────────────────────────────────────────────────────────────
export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface BM25Stats {
  docCount: number;
  avgDocLen: number;
  vocabularySize: number;
}

// ── Encoder ──────────────────────────────────────────────────────────────────
export class BM25Encoder {
  private docCount = 0;
  private avgDocLen = 0;
  private docFreq = new Map<string, number>();
  private readonly stopwords = new Map<string, Set<string>>();

  constructor() {
    // Build per-locale stopword sets, merging base + custom + domain lists.
    const allLocales = new Set([
      ...Object.keys(BASE_STOPWORDS),
      ...Object.keys(CUSTOM_STOPWORDS),
    ]);

    for (const locale of allLocales) {
      const stops = new Set<string>([
        ...(BASE_STOPWORDS[locale] ?? []),
        ...(CUSTOM_STOPWORDS[locale] ?? []),
      ]);
      // Share fr/es/pt variant stopwords with parent locale
      const parent = locale.split('-')[0];
      if (parent !== locale && CUSTOM_STOPWORDS[parent]) {
        for (const w of CUSTOM_STOPWORDS[parent]) stops.add(w);
      }
      if (locale === 'en') {
        for (const w of DOMAIN_STOPWORDS_EN) stops.add(w);
      }
      this.stopwords.set(locale, stops);
    }
  }

  async loadStats(statsPath: string): Promise<void> {
    const raw = await fs.readFile(statsPath, 'utf-8');
    const stats = JSON.parse(raw) as {
      doc_count: number;
      avg_doc_len: number;
      doc_freq: Record<string, number>;
    };
    this.docCount = stats.doc_count;
    this.avgDocLen = stats.avg_doc_len;
    this.docFreq = new Map(Object.entries(stats.doc_freq));
  }

  getStats(): BM25Stats {
    return {
      docCount: this.docCount,
      avgDocLen: this.avgDocLen,
      vocabularySize: this.docFreq.size,
    };
  }

  /**
   * Encode a search query into a BM25 sparse vector.
   * Mirrors Python MultilingualBM25.encode_query().
   */
  encodeQuery(text: string, locale: string): SparseVector {
    const tokens = this.tokenize(text, locale);
    if (tokens.length === 0) return { indices: [], values: [] };

    const tfCounts = new Map<string, number>();
    for (const t of tokens) tfCounts.set(t, (tfCounts.get(t) ?? 0) + 1);

    const pairs: [number, number][] = [];
    for (const [term, tf] of tfCounts) {
      const df = this.docFreq.get(term) ?? 0;
      const idf =
        df === 0
          ? Math.log(this.docCount + 1)
          : Math.log((this.docCount - df + 0.5) / (df + 0.5) + 1);
      if (idf > 0) {
        pairs.push([this.hashToken(term), idf * Math.min(tf, 2)]);
      }
    }

    // Merge hash collisions by summing scores, keep top-1000
    const merged = new Map<number, number>();
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

  // ── Private ────────────────────────────────────────────────────────────────

  private tokenize(text: string, locale: string): string[] {
    // Unicode word split — matches Python re.findall(r'\b\w+\b', text.lower())
    // /[\p{L}0-9]+/gu captures Unicode letters + digits, filtering punctuation.
    const tokens = (text.toLowerCase().match(/[\p{L}0-9]+/gu) ?? []).filter(
      (t) => t.length > 1,
    );

    const stops = this.stopwords.get(locale) ?? new Set<string>();
    const filtered = tokens.filter((t) => !stops.has(t));

    const stemmer = STEMMERS[locale];
    const stemmed = stemmer ? filtered.map((t) => stemmer.stem(t)) : filtered;

    // Locale-prefix — critical for cross-language hash isolation
    return stemmed.map((t) => `${locale}:${t}`);
  }

  private hashToken(token: string): number {
    return murmur3x86_32(token) % HASH_SPACE;
  }
}
