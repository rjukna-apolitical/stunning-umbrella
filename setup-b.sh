#!/bin/bash
# =============================================================================
# POC Environment Setup for Zed IDE — 17 Contentful Locales
# =============================================================================
# Multilingual Hybrid Search — mE5-large + BM25 on Pinecone
#
# Uses `uv` for package management AND Python version management.
# uv installs a standalone Python 3.12 — no need to modify your system Python.
#
# Language support tiers:
#   Tier 1 (Snowball + NLTK stops): en, ar, fr, fr-CA, de, id, it, pt, pt-BR, es, es-419
#   Tier 2 (Snowball, custom stops): pl, sr-Cyrl
#   Tier 3 (Specialised tokeniser): ja (fugashi), ko (kiwipiepy), vi (pyvi)
#   Tier 4 (Whitespace only):       uk
#
# Prerequisites:
#   - uv (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Zed (latest stable)
# =============================================================================

set -euo pipefail

PROJECT_NAME="contentful-pinecone-embeddings-B"
PROJECT_DIR="$HOME/Workspace/apolitical/sandbox/$PROJECT_NAME"

echo "══════════════════════════════════════════════════════════════"
echo "  Setting up Search POC for Zed IDE (17 locales)"
echo "══════════════════════════════════════════════════════════════"

# ── Check uv is installed ──
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "📦 uv $(uv --version)"

# ── Install Python 3.12 via uv ──
# uv manages its own Python installations in ~/.local/share/uv/python/.
# This doesn't touch your system Python (3.14) or brew at all.
# Python 3.12 is the sweet spot: universal wheel coverage for C extension
# packages (PyStemmer, fugashi, kiwipiepy), active support until Oct 2028.
echo ""
echo "🐍 Ensuring Python 3.12 is available..."
uv python install 3.12

# ── Initialise project ──
mkdir -p "$PROJECT_DIR"/{data,exports}
cd "$PROJECT_DIR"

# Create pyproject.toml with Python 3.12 pin. The requires-python constraint
# tells uv to use 3.12 for the venv even though your system `python3` is 3.14.
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    cat > "$PROJECT_DIR/pyproject.toml" << 'TOMLEOF'
[project]
name = "search-poc-b"
version = "0.1.0"
description = "Multilingual hybrid search POC — mE5-large + BM25 on Pinecone"
requires-python = ">=3.12,<3.13"

dependencies = [
    # Jupyter kernel for Zed's REPL
    "ipykernel>=6.29.5",

    # Pinecone vector DB + hosted mE5-large inference
    "pinecone-client>=5.1.0",

    # Contentful CMS client
    "contentful>=2.2.0",

    # BM25 tokenisation pipeline
    # snowballstemmer 3.0.1 in pure Python mode. Covers all 13 Snowball
    # languages including Polish (Oct 2025) and Serbian (Oct 2019).
    # We deliberately do NOT install PyStemmer — it causes an API mismatch
    # on newer Python versions where snowballstemmer calls Stemmer.language()
    # but PyStemmer only exposes Stemmer.algorithms(). The pure Python
    # performance is more than sufficient: stemming a query takes <0.1ms,
    # while Pinecone API calls take 15-30ms.
    "snowballstemmer>=3.0.1",
    "nltk>=3.9.1",
    "mmh3>=5.1.0",

    # Data processing
    "numpy>=2.2.0",
    "pandas>=2.2.0",

    # Utilities
    "python-dotenv>=1.1.0",
    "requests>=2.32.0",
    "tqdm>=4.67.0",
]

[project.optional-dependencies]
# Tier 3 language tokenisers — install with: uv sync --extra cjk
# These pull in larger dependencies (MeCab dictionaries ~50MB, Kiwi
# models ~60MB). The POC falls back to whitespace tokenisation for
# ja/ko/vi if these aren't installed.
cjk = [
    "fugashi>=1.4.0",       # Japanese: MeCab morphological analyser
    "unidic-lite>=1.0.8",   # Japanese: MeCab dictionary
    "kiwipiepy>=0.20.0",    # Korean: Kiwi morphological analyser
    "pyvi>=0.1.1",          # Vietnamese: CRF word segmenter
]
TOMLEOF
    echo "✅ Created pyproject.toml (pinned to Python 3.12)"
fi

# ── Pin Python version for this project ──
# This creates a .python-version file so uv always uses 3.12 in this
# directory, regardless of what's on PATH.
echo "3.12" > "$PROJECT_DIR/.python-version"
echo "✅ Pinned project to Python 3.12 via .python-version"

# ── Install dependencies ──
# uv sync creates the .venv with Python 3.12, resolves all dependencies,
# and installs everything. --extra cjk pulls in the Tier 3 tokenisers.
echo ""
echo "📦 Installing dependencies..."

if uv sync --extra cjk 2>/dev/null; then
    echo "✅ All dependencies installed (including Tier 3 tokenisers)"
else
    echo "⚠️  Tier 3 tokenisers failed — installing core only..."
    uv sync
    echo "   ja/ko/vi will use whitespace fallback"
    echo "   To retry: uv sync --extra cjk"
fi

# ── Verify Python version in venv ──
echo ""
VENV_PYTHON=$("$PROJECT_DIR/.venv/bin/python" --version)
echo "✅ Venv Python: $VENV_PYTHON"

# ── Download NLTK data ──
uv run python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('✅ NLTK data ready')
"

# ── Register Jupyter kernel ──
# Points at the .venv/bin/python so Zed's REPL uses Python 3.12 + all deps.
uv run python -m ipykernel install --user --name search-poc --display-name "Search POC (3.12)"
echo "✅ Kernel 'Search POC (3.12)' registered"

# ── .env file ──
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# Search POC — API Keys (never commit this file)
PINECONE_API_KEY=your-pinecone-api-key
CONTENTFUL_SPACE_ID=your-contentful-space-id
CONTENTFUL_ACCESS_TOKEN=your-contentful-access-token
ENVEOF
    echo "⚠️  Created .env — fill in: $ENV_FILE"
fi

# ── .gitignore ──
cat > "$PROJECT_DIR/.gitignore" << 'EOF'
.venv/
.env
data/
exports/
__pycache__/
*.pyc
.DS_Store
EOF

# ── Zed project settings ──
mkdir -p "$PROJECT_DIR/.zed"
cat > "$PROJECT_DIR/.zed/settings.json" << 'ZEDEOF'
{
  "languages": {
    "Python": {
      "language_servers": ["pyright"],
      "formatter": {
        "external": {
          "command": ".venv/bin/ruff",
          "arguments": ["format", "--stdin-filename", "{buffer_path}", "-"]
        }
      }
    }
  },
  "lsp": {
    "pyright": {
      "settings": {
        "python": {
          "pythonPath": ".venv/bin/python"
        }
      }
    }
  },
  "jupyter": {
    "kernel": {
      "default": "search-poc"
    }
  }
}
ZEDEOF

# ── Verify tokeniser support ──
echo ""
echo "🔍 Verifying tokeniser support..."
uv run python -c "
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import snowballstemmer
required = {
    'english': 'en', 'arabic': 'ar', 'french': 'fr/fr-CA',
    'german': 'de', 'indonesian': 'id', 'italian': 'it',
    'portuguese': 'pt/pt-BR', 'spanish': 'es/es-419',
    'serbian': 'sr-Cyrl',
}
available = snowballstemmer.algorithms()
for algo, locales in required.items():
    status = '✅' if algo in available else '❌'
    print(f'  {status} snowball:{algo} → {locales}')
try:
    import fugashi; print('  ✅ fugashi → ja')
except ImportError: print('  ⚠️  fugashi → ja (whitespace fallback)')
try:
    from kiwipiepy import Kiwi; print('  ✅ kiwipiepy → ko')
except ImportError: print('  ⚠️  kiwipiepy → ko (whitespace fallback)')
try:
    from pyvi import ViTokenizer; print('  ✅ pyvi → vi')
except ImportError: print('  ⚠️  pyvi → vi (whitespace fallback)')
print('  ℹ️  pl: whitespace + stopwords (snowball:polish not yet on PyPI)')
print('  ℹ️  uk: whitespace + stopwords (no stemmer exists)')
"

# ── Summary ──
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ Ready!"
echo ""
echo "     zed $PROJECT_DIR"
echo ""
echo "  1. Fill in .env with your API keys"
echo "  2. Open poc_search.py"
echo "  3. Cmd+Shift+Enter to run cells"
echo "  4. First run: pick kernel 'Search POC (3.12)'"
echo ""
echo "  Useful commands:"
echo "     uv sync                  Reinstall all deps"
echo "     uv sync --extra cjk     Add ja/ko/vi tokenisers"
echo "     uv add <package>        Add a new dependency"
echo "     uv run python script.py Run in the venv (3.12)"
echo "══════════════════════════════════════════════════════════════"
