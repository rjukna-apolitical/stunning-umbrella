FROM python:3.12-slim

# System deps required by some tokeniser packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies (own layer — only rebuilds when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra cjk

# Copy source
COPY config.py setup_index.py setup_bm25.py entrypoint.py ./
COPY modules/ modules/
COPY embed/ embed/

ENV PYTHONPATH=/app

# BM25 stats are stored in /app/data — mount a volume to persist between runs:
#   docker run -v bm25-data-b:/app/data ...
# Run 'make docker-fit-bm25' once to generate the stats before embedding.

ENTRYPOINT ["uv", "run", "python", "entrypoint.py"]
