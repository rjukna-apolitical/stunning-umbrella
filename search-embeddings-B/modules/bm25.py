"""
Multilingual BM25 encoder producing Pinecone-compatible sparse vectors.
Supports 17 Contentful locales across 4 tiers of language complexity.

Locale tiers:
  Tier 1 — Snowball stemmer + NLTK stopwords:   en, ar, fr, fr-CA, de, id, it, pt, pt-BR, es, es-419
  Tier 2 — Snowball stemmer, custom stopwords:  sr-Cyrl
  Tier 3 — Specialised tokenisers:              ja, ko, vi
  Tier 4 — Whitespace + custom stopwords:       pl, uk
"""

import json
import logging
import math
import os
import re
import warnings
from collections import Counter, defaultdict

import mmh3
import nltk
import snowballstemmer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

log = logging.getLogger(__name__)

# ── Index and BM25 hyper-parameters ──
BM25_K1 = 1.2
BM25_B = 0.75
HASH_SPACE = 2**18  # 262,144 sparse dimensions

# ── Locale → tokenisation strategy ──
LOCALE_CONFIG = {
    # Tier 1: Snowball stemmer + NLTK stopwords
    "en":      {"stemmer": "snowball:english",     "nltk_stopwords": "english"},
    "ar":      {"stemmer": "snowball:arabic",      "nltk_stopwords": "arabic"},
    "fr":      {"stemmer": "snowball:french",      "nltk_stopwords": "french"},
    "fr-CA":   {"stemmer": "snowball:french",      "nltk_stopwords": "french"},
    "de":      {"stemmer": "snowball:german",      "nltk_stopwords": "german"},
    "id":      {"stemmer": "snowball:indonesian",  "nltk_stopwords": "indonesian"},
    "it":      {"stemmer": "snowball:italian",     "nltk_stopwords": "italian"},
    "pt":      {"stemmer": "snowball:portuguese",  "nltk_stopwords": "portuguese"},
    "pt-BR":   {"stemmer": "snowball:portuguese",  "nltk_stopwords": "portuguese"},
    "es":      {"stemmer": "snowball:spanish",     "nltk_stopwords": "spanish"},
    "es-419":  {"stemmer": "snowball:spanish",     "nltk_stopwords": "spanish"},
    # Tier 2: Snowball stemmer, custom stopwords
    "sr-Cyrl": {"stemmer": "snowball:serbian",     "nltk_stopwords": None},
    # Tier 3: Specialised tokenisers
    "ja":      {"stemmer": "fugashi",              "nltk_stopwords": None},
    "ko":      {"stemmer": "kiwipiepy",            "nltk_stopwords": None},
    "vi":      {"stemmer": "pyvi",                 "nltk_stopwords": None},
    # Tier 4: Whitespace + custom stopwords
    "pl":      {"stemmer": None,                   "nltk_stopwords": None},
    "uk":      {"stemmer": None,                   "nltk_stopwords": None},
}

SUPPORTED_LOCALES = list(LOCALE_CONFIG.keys())

CUSTOM_STOPWORDS = {
    "pl": {"i", "w", "z", "na", "do", "nie", "że", "to", "się", "jest",
           "o", "jak", "ale", "po", "co", "tak", "za", "od", "jej",
           "go", "by", "są", "już", "ten", "tym", "czy", "dla"},
    "sr-Cyrl": {"и", "у", "је", "да", "на", "се", "за", "од", "са",
                "су", "то", "не", "ће", "као", "из", "или", "али",
                "овај", "тај", "који", "све", "бити", "може"},
    "ja": {"の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
           "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
           "として", "い", "や", "れる", "など", "なっ", "ない", "この",
           "ため", "その", "あっ", "よう", "また", "もの", "という", "あり"},
    "ko": {"이", "그", "저", "것", "수", "등", "들", "및", "에",
           "는", "을", "를", "의", "가", "으로", "에서", "와", "과",
           "도", "로", "한", "하다", "있다", "되다", "없다"},
    "vi": {"của", "và", "là", "có", "được", "cho", "không", "trong",
           "để", "với", "này", "các", "từ", "một", "những", "đã",
           "theo", "về", "sẽ", "hay", "như", "cũng", "khi", "tại",
           "bị", "do", "đến", "nên", "vì", "rất"},
    "uk": {"і", "в", "на", "з", "що", "не", "у", "як", "до", "це",
           "та", "за", "від", "але", "є", "по", "для", "він", "яка",
           "який", "все", "або", "вже", "ще", "так", "бути", "може"},
}

# Domain-specific stopwords for English (high-frequency policy corpus terms)
DOMAIN_STOPWORDS_EN = {
    "government", "policy", "public", "sector", "service",
    "strategy", "framework", "approach", "initiative", "programme",
}

# ── Load Tier 3 specialised tokenisers (optional) ──
_fugashi_tokenizer = None
_kiwi_tokenizer = None
_pyvi_tokenizer = None

try:
    import fugashi
    _fugashi_tokenizer = fugashi.Tagger()
    log.info("Japanese tokeniser (fugashi/MeCab) loaded")
except ImportError:
    log.warning("fugashi not installed — ja will use whitespace fallback")

try:
    from kiwipiepy import Kiwi
    _kiwi_tokenizer = Kiwi()
    log.info("Korean tokeniser (kiwipiepy/Kiwi) loaded")
except ImportError:
    log.warning("kiwipiepy not installed — ko will use whitespace fallback")

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyvi")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyvi")
        from pyvi import ViTokenizer
    _pyvi_tokenizer = ViTokenizer
    log.info("Vietnamese tokeniser (pyvi) loaded")
except ImportError:
    log.warning("pyvi not installed — vi will use whitespace fallback")


class MultilingualBM25:
    """
    Corpus-aware BM25 encoder producing Pinecone-compatible sparse vectors.

    Tokens are prefixed with locale code to prevent cross-language hash
    collisions (e.g. German "gift" vs English "gift").
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B, hash_space: int = HASH_SPACE):
        self.k1 = k1
        self.b = b
        self.hash_space = hash_space

        # Initialise Snowball stemmers for Tier 1 + 2 locales
        self.stemmers = {}
        for locale, config in LOCALE_CONFIG.items():
            spec = config["stemmer"]
            if spec and spec.startswith("snowball:"):
                lang = spec.split(":")[1]
                try:
                    self.stemmers[locale] = snowballstemmer.stemmer(lang)
                except KeyError:
                    log.warning("Snowball stemmer '%s' not available for '%s'", lang, locale)

        # Build per-locale stopword sets
        self.stopwords: dict[str, set[str]] = {}
        for locale, config in LOCALE_CONFIG.items():
            stops: set[str] = set()
            if config["nltk_stopwords"]:
                try:
                    stops.update(nltk_stopwords.words(config["nltk_stopwords"]))
                except OSError:
                    pass
            if locale in CUSTOM_STOPWORDS:
                stops.update(CUSTOM_STOPWORDS[locale])
            if locale == "en":
                stops.update(DOMAIN_STOPWORDS_EN)
            parent = locale.split("-")[0]
            if parent != locale and parent in CUSTOM_STOPWORDS:
                stops.update(CUSTOM_STOPWORDS[parent])
            self.stopwords[locale] = stops

        # Corpus statistics — populated by fit()
        self.doc_count = 0
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0
        self._total_tokens = 0

    def _tokenize_japanese(self, text: str) -> list[str]:
        if _fugashi_tokenizer is None:
            return re.findall(r"\w+", text.lower())
        tokens = []
        for word in _fugashi_tokenizer(text):
            surface = word.surface.strip()
            if len(surface) > 1 and surface not in self.stopwords.get("ja", set()):
                tokens.append(surface)
        return tokens

    def _tokenize_korean(self, text: str) -> list[str]:
        if _kiwi_tokenizer is None:
            return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
        content_tags = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "MAG", "MAJ", "XR", "SL"}
        tokens = []
        for token in _kiwi_tokenizer.tokenize(text):
            if token.tag in content_tags and len(token.form) > 1:
                if token.form not in self.stopwords.get("ko", set()):
                    tokens.append(token.form)
        return tokens

    def _tokenize_vietnamese(self, text: str) -> list[str]:
        if _pyvi_tokenizer is None:
            return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
        segmented = _pyvi_tokenizer.tokenize(text)
        tokens = segmented.lower().split()
        stops = self.stopwords.get("vi", set())
        return [t for t in tokens if t not in stops and len(t) > 1]

    def tokenize(self, text: str, locale: str) -> list[str]:
        """Tokenize text using the locale-appropriate strategy.

        All tokens are prefixed with the locale code to prevent cross-language
        hash collisions in the shared sparse vector space.
        """
        config = LOCALE_CONFIG.get(locale, {})
        spec = config.get("stemmer")

        if spec == "fugashi":
            return [f"{locale}:{t}" for t in self._tokenize_japanese(text)]
        if spec == "kiwipiepy":
            return [f"{locale}:{t}" for t in self._tokenize_korean(text)]
        if spec == "pyvi":
            return [f"{locale}:{t}" for t in self._tokenize_vietnamese(text)]

        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        stops = self.stopwords.get(locale, set())
        tokens = [t for t in tokens if t not in stops and len(t) > 1]

        stemmer = self.stemmers.get(locale)
        if stemmer:
            tokens = stemmer.stemWords(tokens)

        return [f"{locale}:{t}" for t in tokens]

    def hash_token(self, token: str) -> int:
        return mmh3.hash(token, signed=False) % self.hash_space

    def fit(self, documents: list[dict]):
        """Compute corpus IDF statistics from [{"text": str, "locale": str}]."""
        self.doc_count = len(documents)
        self.doc_freq = defaultdict(int)
        self._total_tokens = 0
        for doc in documents:
            tokens = self.tokenize(doc["text"], doc["locale"])
            self._total_tokens += len(tokens)
            for term in set(tokens):
                self.doc_freq[term] += 1
        self.avg_doc_len = self._total_tokens / max(self.doc_count, 1)
        log.info(
            "BM25 fitted: %d docs, %d unique terms, avg_len=%.0f",
            self.doc_count, len(self.doc_freq), self.avg_doc_len,
        )

    def _build_sparse_vector(self, scored_pairs: list[tuple[int, float]]) -> dict:
        """Merge hash collisions by summing scores, then return top-1000 sorted pairs."""
        merged: dict[int, float] = {}
        for idx, score in scored_pairs:
            merged[idx] = merged.get(idx, 0.0) + score
        top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:1000]
        return (
            {"indices": [p[0] for p in top], "values": [round(p[1], 4) for p in top]}
            if top
            else {"indices": [], "values": []}
        )

    def encode_document(self, text: str, locale: str) -> dict:
        """BM25 sparse vector for a document (ingestion time)."""
        tokens = self.tokenize(text, locale)
        if not tokens:
            return {"indices": [], "values": []}
        doc_len = len(tokens)
        tf_counts = Counter(tokens)
        pairs = []
        for term, tf in tf_counts.items():
            df = self.doc_freq.get(term, 0)
            idf = (
                math.log(self.doc_count + 1)
                if df == 0
                else math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            )
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            score = idf * (numerator / denominator)
            if score > 0:
                pairs.append((self.hash_token(term), score))
        return self._build_sparse_vector(pairs)

    def encode_query(self, text: str, locale: str) -> dict:
        """BM25 sparse vector for a query (search time)."""
        tokens = self.tokenize(text, locale)
        if not tokens:
            return {"indices": [], "values": []}
        tf_counts = Counter(tokens)
        pairs = []
        for term, tf in tf_counts.items():
            df = self.doc_freq.get(term, 0)
            idf = (
                math.log(self.doc_count + 1)
                if df == 0
                else math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            )
            if idf > 0:
                pairs.append((self.hash_token(term), idf * min(tf, 2)))
        return self._build_sparse_vector(pairs)

    def save_stats(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stats = {
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_freq": dict(self.doc_freq),
        }
        with open(path, "w") as f:
            json.dump(stats, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        log.info("Saved BM25 stats: %s (%.1fMB, %d terms)", path, size_mb, len(self.doc_freq))

    def load_stats(self, path: str):
        with open(path) as f:
            stats = json.load(f)
        self.doc_count = stats["doc_count"]
        self.avg_doc_len = stats["avg_doc_len"]
        self.doc_freq = defaultdict(int, stats["doc_freq"])
        log.info("Loaded BM25 stats: %d docs, %d terms", self.doc_count, len(self.doc_freq))


def load_bm25(stats_path: str) -> MultilingualBM25:
    """Load a pre-fitted BM25 instance from saved stats."""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"BM25 stats not found at '{stats_path}'. "
            "Run 'make fit-bm25' (or 'make docker-fit-bm25') first."
        )
    bm25 = MultilingualBM25()
    bm25.load_stats(stats_path)
    return bm25
