FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY config.py setup_index.py entrypoint.py ./
COPY modules/ modules/
COPY embed/ embed/

ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "python", "entrypoint.py"]
