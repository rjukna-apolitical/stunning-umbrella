FROM python:3.12-slim

# System deps required by torch / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies (own layer — only rebuilds when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source
COPY config.py setup_index.py entrypoint.py ./
COPY modules/ modules/
COPY embed/ embed/

ENV PYTHONPATH=/app
# Cache HuggingFace models outside the container — mount a volume in production:
#   docker run -v hf-cache:/cache/huggingface ...
ENV HF_HOME=/cache/huggingface

ENTRYPOINT ["uv", "run", "python", "entrypoint.py"]
