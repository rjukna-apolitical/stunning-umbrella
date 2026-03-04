# =============================================================================
# POC: Multilingual Hybrid Search — mE5-large + BM25 on Pinecone
# =============================================================================
# Designed for Zed IDE's REPL (# %% cell markers, Cmd+Shift+Enter to run).
#
# LOCALE SUPPORT AUDIT (17 Contentful locales):
# ─────────────────────────────────────────────
# Tier 1 — Full Snowball stemmer + NLTK stopwords:
#   en, ar, fr, fr-CA, de, id, it, pt, pt-BR, es, es-419
#
# Tier 2 — Snowball stemmer available, custom stopword lists:
#   sr-Cyrl (Snowball Serbian added Oct 2019)
#
# Tier 3 — Needs specialised word segmentation (no stemming):
#   ja  → fugashi (MeCab wrapper) — Japanese compound words need morphological splitting
#   ko  → kiwipiepy (Kiwi morphological analyser) — agglutinative language
#   vi  → pyvi (CRF-based Vietnamese word segmenter) — multi-syllable words not whitespace-separated
#
# Tier 4 — Whitespace tokenisation + custom stopwords:
#   pl  → Snowball Polish stemmer was added Oct 2025 but snowballstemmer 3.0.1
#         (May 2025) predates it. Move to Tier 2 when snowballstemmer >=3.1.0 ships.
#   uk  → No Snowball stemmer exists for Ukrainian.
#
# Regional variants share their parent's stemmer because morphology is
# identical — differences are vocabulary, which BM25 IDF handles:
#   fr-CA  → french   | pt-BR  → portuguese   | es-419 → spanish
# =============================================================================


# %% Cell 0 — Environment setup
import os
from pathlib import Path
from dotenv import load_dotenv

for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent if '__file__' in dir() else Path.cwd()]:
    env_path = candidate / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        break

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip()
CONTENTFUL_SPACE_ID = os.environ.get("CONTENTFUL_SPACE_ID", "").strip()
CONTENTFUL_ACCESS_TOKEN = os.environ.get("CONTENTFUL_ACCESS_TOKEN", "").strip()

missing = [k for k, v in {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "CONTENTFUL_SPACE_ID": CONTENTFUL_SPACE_ID,
    "CONTENTFUL_ACCESS_TOKEN": CONTENTFUL_ACCESS_TOKEN,
}.items() if not v or v.startswith("your-")]

if missing:
    print(f"⚠️  Missing keys in .env: {', '.join(missing)}")
else:
    print(f"✅ Environment loaded")
    print(f"   Pinecone key: ...{PINECONE_API_KEY[-8:]}")
    print(f"   Contentful space: {CONTENTFUL_SPACE_ID}")


# %% Cell 1 — Configuration: all 17 Contentful locales
INDEX_NAME = "platform-v2"
NAMESPACE = "search"
DENSE_DIM = 1024
METRIC = "dotproduct"
DENSE_MODEL = "multilingual-e5-large"

BM25_K1 = 1.2
BM25_B = 0.75
HASH_SPACE = 2**18  # 262,144 sparse dimensions

# ── Locale → tokenisation strategy mapping ──
#
# Each Contentful locale maps to a dict with:
#   stemmer:        which tokenisation approach to use
#   nltk_stopwords: NLTK stopword language name, or None if we provide custom lists
#
# Stemmer values:
#   "snowball:<lang>"  → Snowball stemmer from snowballstemmer package
#   "fugashi"          → Japanese MeCab morphological analysis
#   "kiwipiepy"        → Korean Kiwi morphological analyser
#   "pyvi"             → Vietnamese CRF word segmenter
#   None               → whitespace tokenisation only

LOCALE_CONFIG = {
    # ── Tier 1: Snowball stemmer + NLTK stopwords ──
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

    # ── Tier 2: Snowball stemmer, custom stopwords (NLTK doesn't have these) ──
    # NOTE: Polish (pl) was added to the Snowball project in Oct 2025, but
    # snowballstemmer 3.0.1 on PyPI was released May 2025 — before Polish was
    # included. Once snowballstemmer >=3.1.0 ships, move pl back to Tier 2 by
    # changing its stemmer to "snowball:polish". Until then it uses whitespace
    # tokenisation + custom stopwords (same as Ukrainian).
    "sr-Cyrl": {"stemmer": "snowball:serbian",     "nltk_stopwords": None},

    # ── Tier 3: Specialised tokenisers (CJK + Vietnamese) ──
    "ja":      {"stemmer": "fugashi",              "nltk_stopwords": None},
    "ko":      {"stemmer": "kiwipiepy",            "nltk_stopwords": None},
    "vi":      {"stemmer": "pyvi",                 "nltk_stopwords": None},

    # ── Tier 4: Whitespace tokenisation + custom stopwords ──
    # These languages either have no Snowball stemmer, or the stemmer hasn't
    # been released to PyPI yet. BM25 IDF still provides correct term weighting
    # — you just miss morphological conflation (e.g., pl: "transformacja" ≠
    # "transformacji", uk: "пошук" ≠ "пошуку" won't be treated as the same term).
    "pl":      {"stemmer": None,                   "nltk_stopwords": None},
    "uk":      {"stemmer": None,                   "nltk_stopwords": None},
}

SUPPORTED_LOCALES = list(LOCALE_CONFIG.keys())
print(f"✅ Configured {len(SUPPORTED_LOCALES)} locales: {', '.join(SUPPORTED_LOCALES)}")

for tier, locales in [
    ("Tier 1 (Snowball + NLTK stops)", [k for k, v in LOCALE_CONFIG.items() if v["stemmer"] and v["stemmer"].startswith("snowball:") and v["nltk_stopwords"]]),
    ("Tier 2 (Snowball, custom stops)", [k for k, v in LOCALE_CONFIG.items() if v["stemmer"] and v["stemmer"].startswith("snowball:") and not v["nltk_stopwords"]]),
    ("Tier 3 (Specialised tokeniser)", [k for k, v in LOCALE_CONFIG.items() if v["stemmer"] and not v["stemmer"].startswith("snowball:")]),
    ("Tier 4 (Whitespace + stopwords)", [k for k, v in LOCALE_CONFIG.items() if v["stemmer"] is None]),
]:
    print(f"   {tier}: {', '.join(locales)}")


# %% Cell 2 — BM25 Multilingual Tokenizer (17 locales)
import re
import math
import json
import mmh3
# IMPORTANT: using snowballstemmer in pure Python mode. We deliberately do NOT
# install PyStemmer — it causes an API mismatch on newer Python where
# snowballstemmer calls Stemmer.language() but PyStemmer exposes
# Stemmer.algorithms(). Pure Python performance is more than sufficient:
# stemming a 5-word query takes <0.1ms while Pinecone calls take 15-30ms.
import snowballstemmer
import nltk
from collections import Counter, defaultdict

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

# ── Load Tier 3 specialised tokenisers ──
# These are optional. If not installed, the affected locales fall back to
# whitespace tokenisation. BM25 IDF still works — you just miss compound
# word splitting and morphological analysis.

_fugashi_tokenizer = None
_kiwi_tokenizer = None
_pyvi_tokenizer = None

try:
    import fugashi
    _fugashi_tokenizer = fugashi.Tagger()
    print("✅ Japanese tokeniser (fugashi/MeCab) loaded")
except ImportError:
    print("⚠️  fugashi not installed — ja will use whitespace fallback")
    print("   Install: pip install fugashi unidic-lite")

try:
    from kiwipiepy import Kiwi
    _kiwi_tokenizer = Kiwi()
    print("✅ Korean tokeniser (kiwipiepy/Kiwi) loaded")
except ImportError:
    print("⚠️  kiwipiepy not installed — ko will use whitespace fallback")
    print("   Install: pip install kiwipiepy")

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyvi")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyvi")
        from pyvi import ViTokenizer
    _pyvi_tokenizer = ViTokenizer
    print("✅ Vietnamese tokeniser (pyvi) loaded")
except ImportError:
    print("⚠️  pyvi not installed — vi will use whitespace fallback")
    print("   Install: pip install pyvi")


# ── Custom stopword lists ──
# For languages where NLTK doesn't ship stopwords. These are curated lists
# of high-frequency function words. Extend based on the IDF analysis in Cell 13
# which reveals which terms to add per locale.

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

# Domain-specific stopwords: terms that appear in nearly every document on
# Apolitical's government/policy corpus. BM25 IDF would already downweight
# these, but removing them entirely keeps the sparse vectors cleaner and
# reduces hash collisions. English only for now — add per-locale equivalents
# after reviewing the IDF output in Cell 13.
DOMAIN_STOPWORDS_EN = {
    "government", "policy", "public", "sector", "service",
    "strategy", "framework", "approach", "initiative", "programme",
}


class MultilingualBM25:
    """
    Corpus-aware BM25 encoder producing Pinecone-compatible sparse vectors.
    Supports 17 Contentful locales across 4 tiers of language complexity.

    Why BM25 over BGE-M3's learned sparse?
    On Apolitical's government/policy corpus, terms like "strategy" and
    "framework" appear in nearly every document. BM25's IDF formula
    auto-assigns these near-zero weight because df ≈ N. BGE-M3's sparse
    weights were learned from MS MARCO where these terms might be rare —
    it can't know they're noise in YOUR corpus.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B, hash_space: int = HASH_SPACE):
        self.k1 = k1
        self.b = b
        self.hash_space = hash_space

        # Initialise Snowball stemmers for all Tier 1 + 2 locales
        self.stemmers = {}
        for locale, config in LOCALE_CONFIG.items():
            stemmer_spec = config["stemmer"]
            if stemmer_spec and stemmer_spec.startswith("snowball:"):
                algo_name = stemmer_spec.split(":")[1]
                try:
                    self.stemmers[locale] = snowballstemmer.stemmer(algo_name)
                except KeyError:
                    print(f"⚠️  Snowball stemmer '{algo_name}' not available for '{locale}'")

        # Build per-locale stopword sets
        self.stopwords: dict[str, set[str]] = {}
        for locale, config in LOCALE_CONFIG.items():
            stops = set()
            # NLTK stopwords (Tier 1)
            if config["nltk_stopwords"]:
                try:
                    stops.update(nltk_stopwords.words(config["nltk_stopwords"]))
                except OSError:
                    pass
            # Custom stopwords (Tier 2/3/4)
            if locale in CUSTOM_STOPWORDS:
                stops.update(CUSTOM_STOPWORDS[locale])
            # Domain-specific (English only for now)
            if locale == "en":
                stops.update(DOMAIN_STOPWORDS_EN)
            # Regional variants inherit parent stopwords
            parent = locale.split("-")[0]
            if parent != locale and parent in CUSTOM_STOPWORDS:
                stops.update(CUSTOM_STOPWORDS[parent])
            self.stopwords[locale] = stops

        # Corpus statistics — populated by fit()
        self.doc_count = 0
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0
        self._total_tokens = 0

    # ── Tier 3 tokenisers ──

    def _tokenize_japanese(self, text: str) -> list[str]:
        """MeCab morphological analysis via fugashi. Splits compound words
        and returns surface forms. We don't need lemmatisation because BM25
        IDF handles term frequency naturally — "政策" appearing in many docs
        gets low weight regardless of whether we lemmatise it."""
        if _fugashi_tokenizer is None:
            return re.findall(r'\w+', text.lower())
        tokens = []
        for word in _fugashi_tokenizer(text):
            surface = word.surface.strip()
            if len(surface) > 1 and surface not in self.stopwords.get("ja", set()):
                tokens.append(surface)
        return tokens

    def _tokenize_korean(self, text: str) -> list[str]:
        """Kiwi morphological analyser splits agglutinative Korean into
        morphemes. We keep only content-word POS tags (nouns, verbs,
        adjectives) and discard particles, suffixes, and punctuation."""
        if _kiwi_tokenizer is None:
            return [t for t in re.findall(r'\w+', text.lower()) if len(t) > 1]
        content_tags = {"NNG", "NNP", "NNB", "NR", "NP",  # Nouns
                        "VV", "VA", "VX",                    # Verbs, adjectives
                        "MAG", "MAJ",                        # Adverbs
                        "XR", "SL"}                          # Roots, foreign words
        tokens = []
        for token in _kiwi_tokenizer.tokenize(text):
            if token.tag in content_tags and len(token.form) > 1:
                if token.form not in self.stopwords.get("ko", set()):
                    tokens.append(token.form)
        return tokens

    def _tokenize_vietnamese(self, text: str) -> list[str]:
        """pyvi CRF-based word segmenter. Vietnamese is a non-segmented
        language where multi-syllable words are written as separate tokens.
        For example "chính phủ" (government) is two syllables that form one
        word. pyvi joins them: "chính_phủ". Without this, "chính" and "phủ"
        would be indexed separately, matching unrelated content."""
        if _pyvi_tokenizer is None:
            return [t for t in re.findall(r'\w+', text.lower()) if len(t) > 1]
        segmented = _pyvi_tokenizer.tokenize(text)
        tokens = segmented.lower().split()
        stops = self.stopwords.get("vi", set())
        return [t for t in tokens if t not in stops and len(t) > 1]

    # ── Main tokenize method ──

    def tokenize(self, text: str, locale: str) -> list[str]:
        """Tokenize text using locale-appropriate strategy.

        All tokens are prefixed with the locale code to prevent cross-language
        hash collisions. This is critical: German "gift" (poison) and English
        "gift" must map to different sparse dimensions, otherwise a search
        for "gift ideas" in English could get a BM25 boost from German
        documents about poison.
        """
        config = LOCALE_CONFIG.get(locale, {})
        stemmer_spec = config.get("stemmer")

        # Tier 3: specialised tokenisers
        if stemmer_spec == "fugashi":
            return [f"{locale}:{t}" for t in self._tokenize_japanese(text)]
        if stemmer_spec == "kiwipiepy":
            return [f"{locale}:{t}" for t in self._tokenize_korean(text)]
        if stemmer_spec == "pyvi":
            return [f"{locale}:{t}" for t in self._tokenize_vietnamese(text)]

        # Tier 1, 2, 4: regex word split → stopword removal → optional stemming
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        stops = self.stopwords.get(locale, set())
        tokens = [t for t in tokens if t not in stops and len(t) > 1]

        stemmer = self.stemmers.get(locale)
        if stemmer:
            tokens = stemmer.stemWords(tokens)

        return [f"{locale}:{t}" for t in tokens]

    def hash_token(self, token: str) -> int:
        """MurmurHash3 → integer index. Same algo + seed as TypeScript port."""
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
        print(f"✅ BM25 fitted: {self.doc_count} docs, {len(self.doc_freq)} unique terms, avg_len={self.avg_doc_len:.0f}")

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
            idf = math.log(self.doc_count + 1) if df == 0 else math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            score = idf * (numerator / denominator)
            if score > 0:
                pairs.append((self.hash_token(term), round(score, 4)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:1000]
        return {"indices": [p[0] for p in pairs], "values": [p[1] for p in pairs]} if pairs else {"indices": [], "values": []}

    def encode_query(self, text: str, locale: str) -> dict:
        """BM25 sparse vector for a query (search time, <1ms)."""
        tokens = self.tokenize(text, locale)
        if not tokens:
            return {"indices": [], "values": []}
        tf_counts = Counter(tokens)
        pairs = []
        for term, tf in tf_counts.items():
            df = self.doc_freq.get(term, 0)
            idf = math.log(self.doc_count + 1) if df == 0 else math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            if idf > 0:
                pairs.append((self.hash_token(term), round(idf * min(tf, 2), 4)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:1000]
        return {"indices": [p[0] for p in pairs], "values": [p[1] for p in pairs]} if pairs else {"indices": [], "values": []}

    def save_stats(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stats = {"doc_count": self.doc_count, "avg_doc_len": self.avg_doc_len, "doc_freq": dict(self.doc_freq)}
        with open(path, "w") as f:
            json.dump(stats, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Saved BM25 stats: {path} ({size_mb:.1f}MB, {len(self.doc_freq)} terms)")

    def load_stats(self, path: str):
        with open(path, "r") as f:
            stats = json.load(f)
        self.doc_count = stats["doc_count"]
        self.avg_doc_len = stats["avg_doc_len"]
        self.doc_freq = defaultdict(int, stats["doc_freq"])
        print(f"✅ Loaded BM25 stats: {self.doc_count} docs, {len(self.doc_freq)} terms")


bm25 = MultilingualBM25()

# Report loaded stemmers
snowball_ok = [loc for loc in SUPPORTED_LOCALES if loc in bm25.stemmers]
print(f"\n✅ Snowball stemmers loaded: {', '.join(snowball_ok)}")
for loc, label in [("ja", "fugashi/MeCab"), ("ko", "kiwipiepy/Kiwi"), ("vi", "pyvi")]:
    loaded = {"ja": _fugashi_tokenizer, "ko": _kiwi_tokenizer, "vi": _pyvi_tokenizer}[loc]
    print(f"   {loc}: {'✅ ' + label if loaded else '⚠️  whitespace fallback'}")
print(f"   uk: whitespace only (no stemmer exists)")


# %% Cell 3 — Pinecone index + dense embedding
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME in existing:
    pc.delete_index(INDEX_NAME)
    print(f"🗑️ Deleted existing '{INDEX_NAME}'")

pc.create_index(
    name=INDEX_NAME,
    dimension=DENSE_DIM,
    metric=METRIC,
    spec=ServerlessSpec(cloud="gcp", region="europe-west4"),
)
index = pc.Index(INDEX_NAME)
print(f"✅ Index '{INDEX_NAME}' created (dim={DENSE_DIM}, metric={METRIC})")


def embed_dense_passages(texts: list[str]) -> list[list[float]]:
    embeddings = pc.inference.embed(
        model=DENSE_MODEL, inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [e.values for e in embeddings]


def embed_dense_query(text: str) -> list[float]:
    embeddings = pc.inference.embed(
        model=DENSE_MODEL, inputs=[text],
        parameters={"input_type": "query", "truncate": "END"},
    )
    return embeddings[0].values


test_vec = embed_dense_query("Hello world")
print(f"✅ Pinecone Inference: dim={len(test_vec)}")


# %% Cell 4 — Contentful client
import contentful

cf_client = contentful.Client(
    space_id=CONTENTFUL_SPACE_ID,
    access_token=CONTENTFUL_ACCESS_TOKEN,
)


def rich_text_to_plain(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        text = node.get("value", "")
        for child in node.get("content", []):
            text += rich_text_to_plain(child)
        return text
    return ""


def fetch_entries(content_type: str, locale: str, limit: int = 100) -> list[dict]:
    entries, skip = [], 0
    while True:
        try:
            response = cf_client.entries({
                "content_type": content_type, "locale": locale,
                "limit": limit, "skip": skip,
            })
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                return []
            raise
        for entry in response:
            fields = entry.fields()
            title = fields.get("title", "")
            if not title:
                continue
            body_raw = fields.get("body", "")
            body = rich_text_to_plain(body_raw) if isinstance(body_raw, dict) else str(body_raw or "")
            entries.append({
                "id": entry.sys["id"], "title": title, "body": body[:3000],
                "slug": fields.get("slug", ""), "content_type": content_type,
                "locale": locale, "published_date": fields.get("publishedDate", ""),
                "available_locales": fields.get("availableLocales", [locale]),
            })
        if len(response) < limit:
            break
        skip += limit
    return entries


test_entries = fetch_entries("solutionArticle", "en", limit=5)
print(f"✅ Contentful: {len(test_entries)} test entries")
for e in test_entries[:3]:
    print(f"   - {e['title'][:60]}...")


# %% Cell 5 — Fit BM25 on the full multilingual corpus
from tqdm import tqdm

print("Collecting corpus across all 17 locales...")
corpus_docs = []
locale_counts = {}

for content_type in ["solutionArticle"]:  # Extend: "course", "event", "community"
    for locale in SUPPORTED_LOCALES:
        entries = fetch_entries(content_type, locale)
        for entry in entries:
            corpus_docs.append({"text": f"{entry['title']}. {entry['body']}", "locale": locale})
        if entries:
            locale_counts[locale] = locale_counts.get(locale, 0) + len(entries)

print(f"\n📊 Corpus by locale:")
for locale in SUPPORTED_LOCALES:
    count = locale_counts.get(locale, 0)
    config = LOCALE_CONFIG[locale]
    tier = "T1" if config["nltk_stopwords"] else "T2" if config["stemmer"] and config["stemmer"].startswith("snowball:") else "T3" if config["stemmer"] else "T4"
    bar = "█" * (count // 5) if count else ""
    print(f"  {locale:8s} [{tier}]: {count:4d} entries  {bar}")

bm25.fit(corpus_docs)

# Show IDF effect across multiple languages
print("\n📊 Sample IDF values across locales:")
test_terms = [
    ("en", "strategy"),        # Expect LOW (jargon)
    ("en", "blockchain"),      # Expect HIGH (distinctive)
    ("en", "participatory"),   # Expect HIGH
    ("es", "gobierno"),        # Spanish: government
    ("de", "verwaltung"),      # German: administration
    ("fr", "numérique"),       # French: digital
    ("ar", "حكومة"),           # Arabic: government
    ("pl", "rząd"),            # Polish: government
    ("uk", "уряд"),            # Ukrainian: government
]
for locale, term in test_terms:
    tokens = bm25.tokenize(term, locale)
    if tokens:
        df = bm25.doc_freq.get(tokens[0], 0)
        idf = math.log((bm25.doc_count - df + 0.5) / (df + 0.5) + 1) if df > 0 else math.log(bm25.doc_count + 1)
        pct = (df / max(bm25.doc_count, 1)) * 100
        print(f"  {locale:8s} {term:20s} → df={df:4d} ({pct:5.1f}%)  idf={idf:.3f}")

bm25.save_stats("data/bm25_corpus_stats.json")


# %% Cell 6 — Ingestion pipeline
import time

def chunk_text(text: str, max_words: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += max_words - overlap
    return chunks


def ingest_content(content_type: str, locale: str, batch_size: int = 32) -> int:
    entries = fetch_entries(content_type, locale)
    if not entries:
        return 0
    vectors, total = [], 0
    for entry in entries:
        full_text = f"{entry['title']}. {entry['title']}. {entry['body']}"
        chunks = chunk_text(full_text)
        dense_vectors = embed_dense_passages(chunks)
        for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vectors)):
            sparse = bm25.encode_document(chunk, locale)
            vector = {
                "id": f"{entry['id']}::{locale}::{i}",
                "values": dense_vec,
                "metadata": {
                    "content_id": entry["id"], "content_type": content_type,
                    "locale": locale, "chunk_index": i,
                    "title": entry["title"], "slug": entry["slug"],
                    "snippet": chunk[:300],
                    "published_date": int(time.mktime(time.strptime(entry["published_date"], "%Y-%m-%d"))) if entry.get("published_date") else 0,
                    "available_locales": entry.get("available_locales", [locale]),
                },
            }
            if sparse["indices"]:
                vector["sparse_values"] = {"indices": sparse["indices"], "values": sparse["values"]}
            vectors.append(vector)
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors, namespace=NAMESPACE)
                total += len(vectors)
                vectors = []
    if vectors:
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        total += len(vectors)
    return total


for ct in ["solutionArticle"]:
    for locale in SUPPORTED_LOCALES:
        count = ingest_content(ct, locale)
        if count > 0:
            print(f"✅ {ct}/{locale}: {count} vectors")

time.sleep(5)
stats = index.describe_index_stats()
print(f"\n📊 Index total: {stats.total_vector_count} vectors")


# %% Cell 7 — Search: monolingual hybrid
import time as timer_module

def search_monolingual(
    query: str, locale: str, content_type: str | None = None,
    top_k: int = 10, alpha: float = 0.7,
) -> dict:
    start = timer_module.time()
    t0 = timer_module.time()
    dense_vec = embed_dense_query(query)
    embed_ms = (timer_module.time() - t0) * 1000

    t1 = timer_module.time()
    sparse = bm25.encode_query(query, locale)
    bm25_ms = (timer_module.time() - t1) * 1000

    filters = {"locale": {"$eq": locale}}
    if content_type:
        filters = {"$and": [{"locale": {"$eq": locale}}, {"content_type": {"$eq": content_type}}]}

    query_kwargs = {
        "vector": [v * alpha for v in dense_vec],
        "filter": filters, "top_k": top_k,
        "include_metadata": True, "namespace": NAMESPACE,
    }
    if sparse["indices"] and alpha < 1.0:
        query_kwargs["sparse_vector"] = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

    t2 = timer_module.time()
    results = index.query(**query_kwargs)
    query_ms = (timer_module.time() - t2) * 1000

    return {
        "matches": results.matches,
        "timing": {
            "embed_ms": round(embed_ms, 1), "bm25_ms": round(bm25_ms, 1),
            "query_ms": round(query_ms, 1), "total_ms": round((timer_module.time() - start) * 1000, 1),
        },
    }


print("=" * 60)
print("TEST 1: Monolingual hybrid across locales")
print("=" * 60)
for query, locale in [
    ("digital government transformation", "en"),
    ("gobierno digital transformación", "es"),
    ("transformation numérique gouvernement", "fr"),
    ("digitale Verwaltung", "de"),
    ("governo digitale", "it"),
    ("pemerintahan digital", "id"),
    ("cyfrowa transformacja", "pl"),
    ("цифрова трансформація", "uk"),
]:
    r = search_monolingual(query, locale=locale)
    n = len(r["matches"])
    top = r["matches"][0].score if r["matches"] else 0
    print(f"  [{locale:8s}] \"{query[:40]}\" → {n} results, top={top:.3f}, {r['timing']['total_ms']}ms")


# %% Cell 8 — Search: locale fallback (primary → English backfill)
def search_with_fallback(
    query: str, user_locale: str, content_type: str | None = None,
    top_k: int = 10, alpha: float = 0.7,
) -> dict:
    start = timer_module.time()
    primary = search_monolingual(query, user_locale, content_type, top_k, alpha)
    seen = {m.metadata["content_id"] for m in primary["matches"]}
    fallback = []

    if len(primary["matches"]) < top_k and user_locale != "en":
        fb = search_monolingual(query, "en", content_type, (top_k - len(primary["matches"])) * 2, alpha=1.0)
        for m in fb["matches"]:
            if m.metadata["content_id"] not in seen:
                m.metadata["is_fallback"] = True
                fallback.append(m)
                seen.add(m.metadata["content_id"])
                if len(fallback) >= top_k - len(primary["matches"]):
                    break

    return {
        "matches": primary["matches"] + fallback,
        "primary_count": len(primary["matches"]),
        "fallback_count": len(fallback),
        "timing": {"primary": primary["timing"], "total_ms": round((timer_module.time() - start) * 1000, 1)},
    }


print("=" * 60)
print("TEST 2: Locale fallback")
print("=" * 60)
for query, locale in [
    ("cambio climático gobierno digital", "es-419"),
    ("transformation numérique", "fr-CA"),
    ("цифрова трансформація", "uk"),
    ("transformacja cyfrowa", "pl"),
    ("digitalna transformacija", "sr-Cyrl"),
]:
    r = search_with_fallback(query, user_locale=locale)
    print(f"  [{locale:8s}] {r['primary_count']} primary + {r['fallback_count']} EN fallback ({r['timing']['total_ms']}ms)")


# %% Cell 9 — Enrollment-aware reranking
def search_with_enrollment(
    query: str, user_locale: str, enrolled_ids: list[str],
    content_type: str | None = None, top_k: int = 10,
    boost: float = 1.3, alpha: float = 0.7,
) -> dict:
    r = search_with_fallback(query, user_locale, content_type, min(top_k * 3, 50), alpha)
    enrolled_set = set(enrolled_ids)
    boosted = [{
        "id": m.id, "metadata": m.metadata,
        "score": m.score * (boost if m.metadata["content_id"] in enrolled_set else 1.0),
        "original_score": m.score,
        "is_enrolled": m.metadata["content_id"] in enrolled_set,
    } for m in r["matches"]]
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return {"matches": boosted[:top_k], "timing": r["timing"]}


print("=" * 60)
print("TEST 3: Enrollment-aware search")
print("=" * 60)
sample_enrolled = [m.metadata["content_id"] for m in search_monolingual("climate", "en", top_k=3)["matches"]]
r = search_with_enrollment("climate adaptation policy", "en", enrolled_ids=sample_enrolled)
for m in r["matches"][:5]:
    tag = " ⭐" if m["is_enrolled"] else ""
    print(f"  {m['score']:.3f} (raw {m['original_score']:.3f}) | {m['metadata']['title'][:50]}{tag}")


# %% Cell 10 — Hybrid vs dense-only A/B test
def compare_hybrid_vs_dense():
    queries = [
        ("OECD PISA report", "en", "Entity"),
        ("participatory budgeting", "en", "Domain term"),
        ("Benachrichtigungen konfigurieren", "de", "German exact"),
        ("how governments use AI ethically", "en", "Conceptual"),
        ("cambio climático adaptación", "es", "Spanish"),
        ("pemerintahan digital Indonesia", "id", "Indonesian"),
    ]
    print("=" * 60)
    print("HYBRID (alpha=0.7) vs DENSE-ONLY (alpha=1.0)")
    print("=" * 60)
    for query, locale, label in queries:
        d = search_monolingual(query, locale, top_k=3, alpha=1.0)
        h = search_monolingual(query, locale, top_k=3, alpha=0.7)
        d_top = d["matches"][0].score if d["matches"] else 0
        h_top = h["matches"][0].score if h["matches"] else 0
        winner = "🟢 HYBRID" if h_top > d_top else "🔵 DENSE" if d_top > h_top else "⚪ TIE"
        print(f"  [{label:15s}] dense={d_top:.3f} hybrid={h_top:.3f} → {winner}")


compare_hybrid_vs_dense()


# %% Cell 11 — Cross-lingual similarity validation
import numpy as np

def test_cross_lingual():
    """Test mE5-large's ability to map equivalent concepts across all
    platform languages to nearby vectors. This validates that the dense
    component of hybrid search can power cross-lingual fallback."""
    pairs = [
        ("Climate Change", "Cambio Climático"),               # Spanish
        ("Digital Government", "Gouvernement Numérique"),      # French
        ("Artificial Intelligence", "Künstliche Intelligenz"), # German
        ("Open Data", "Dados Abertos"),                        # Portuguese
        ("Public Procurement", "المشتريات العامة"),            # Arabic
        ("Digital Transformation", "Trasformazione digitale"), # Italian
        ("Government Policy", "Kebijakan Pemerintah"),         # Indonesian
        ("Innovation", "Innowacja"),                           # Polish
        ("Digital Economy", "Цифрова економіка"),              # Ukrainian
        ("Public Administration", "Јавна управа"),             # Serbian
    ]
    print("=" * 60)
    print("CROSS-LINGUAL SIMILARITY (mE5-large dense cosine)")
    print("=" * 60)
    for en, other in pairs:
        vecs = embed_dense_passages([f"passage: {en}", f"passage: {other}"])
        sim = np.dot(vecs[0], vecs[1])
        q = "✅" if sim > 0.85 else "⚠️" if sim > 0.75 else "❌"
        print(f"  {q} {sim:.3f} | '{en}' ↔ '{other}'")


test_cross_lingual()


# %% Cell 12 — Latency benchmark
import statistics

def benchmark(n: int = 20):
    queries = [
        ("digital government", "en"), ("cambio climático", "es"),
        ("intelligence artificielle", "fr"), ("Digitale Verwaltung", "de"),
        ("governo digitale", "it"), ("pemerintahan digital", "id"),
        ("transformacja cyfrowa", "pl"), ("цифрова трансформація", "uk"),
        ("climate adaptation cities", "en"), ("gobierno abierto", "es-419"),
    ]
    lat = {"embed": [], "bm25": [], "pinecone": [], "total": []}
    for i in range(n):
        q, loc = queries[i % len(queries)]
        r = search_monolingual(q, loc, top_k=10)
        for k in lat:
            lat[k].append(r["timing"][f"{k}_ms"] if k != "pinecone" else r["timing"]["query_ms"])

    print("=" * 60)
    print(f"LATENCY BENCHMARK ({n} queries across 10 locales)")
    print("=" * 60)
    for k, v in lat.items():
        p50 = statistics.median(v)
        p95 = sorted(v)[int(len(v) * 0.95)]
        mean = statistics.mean(v)
        print(f"  {k:10s}: p50={p50:5.0f}ms  p95={p95:5.0f}ms  mean={mean:5.0f}ms")
    p95_total = sorted(lat["total"])[int(len(lat["total"]) * 0.95)]
    status = "✅ PASS" if p95_total <= 100 else "⚠️ OVER BUDGET"
    print(f"\n  {status}: p95={p95_total:.0f}ms vs 100ms budget → {100-p95_total:.0f}ms headroom")


benchmark(20)


# %% Cell 13 — IDF distribution analysis (per-locale)
def analyze_idf():
    """Show the most common and rarest terms per locale. This is the key
    diagnostic for BM25 tuning: common terms should have LOW IDF (noise)
    and distinctive terms should have HIGH IDF (signal). Use this output
    to decide which terms to add to per-locale domain stopword lists."""
    print("=" * 60)
    print("IDF DISTRIBUTION BY LOCALE")
    print("=" * 60)
    for locale in ["en", "es", "fr", "de", "ar", "id", "it", "pl", "uk", "pt"]:
        prefix = f"{locale}:"
        terms = {k: v for k, v in bm25.doc_freq.items() if k.startswith(prefix)}
        if not terms:
            print(f"\n  {locale}: no terms in corpus")
            continue
        sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  {locale}: {len(terms)} unique terms")
        print(f"    Top 5 most common → LOW weight (consider adding to stopwords):")
        for term, df in sorted_terms[:5]:
            idf = math.log((bm25.doc_count - df + 0.5) / (df + 0.5) + 1)
            print(f"      {term:35s} df={df:4d}  idf={idf:.2f}")
        rare = [t for t in sorted_terms if 1 <= t[1] <= 3][:5]
        if rare:
            print(f"    5 rarest → HIGH weight (distinctive):")
            for term, df in rare:
                idf = math.log((bm25.doc_count - df + 0.5) / (df + 0.5) + 1)
                print(f"      {term:35s} df={df:4d}  idf={idf:.2f}")


analyze_idf()


# %% Cell 14 — Export for NestJS
bm25.save_stats("exports/bm25_corpus_stats.json")

print("\n📦 NestJS deployment notes:")
print("   1. Copy exports/bm25_corpus_stats.json to platform-v2")
print("   2. npm install snowball-stemmers murmurhash3js")
print("")
print("   Tier 3 language support in NestJS:")
print("   ja → kuromoji.js (MeCab-compatible, pure JS, ~5MB dictionary)")
print("   ko → whitespace (Korean has natural spacing between words)")
print("   vi → whitespace (likely sufficient for query-time only)")
print("")
print("   Tier 4 (whitespace + custom stopwords):")
print("   pl → move to snowball:polish when snowball-stemmers npm updates")
print("   uk → no stemmer exists, whitespace is the long-term approach")
print("")
print("   snowball-stemmers npm package covers: en, ar, fr, de, id, it,")
print("   pt, es, sr — all Tier 1 + 2 locales.")


# %% Cell 15 — Cleanup (uncomment to delete index)
# pc.delete_index(INDEX_NAME)
# print(f"🗑️ Deleted '{INDEX_NAME}'")
