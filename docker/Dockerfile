# ===================================================================
# Sol Scalper — Dockerfile
# Works on Raspberry Pi 4 (linux/arm64) and x86_64.
# Uses an in-project .venv so the runtime stage is clean and minimal.
# ===================================================================

# ─────────────────────────── Builder ───────────────────────────────
FROM python:3.13-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry via pip (faster + no curl dependency)
ENV POETRY_VERSION=1.8.4 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy lockfile + manifest first for layer caching
COPY pyproject.toml poetry.lock* ./

# Install production deps only into /app/.venv
RUN poetry install --no-root --only main

# ─────────────────────────── Runtime ───────────────────────────────
FROM python:3.13-slim-bookworm

WORKDIR /app

# Grab only the virtual environment from the builder — nothing else
COPY --from=builder /app/.venv /app/.venv

# Put the venv on PATH (no need for `poetry run` at runtime)
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy source code
COPY src/ ./src/
COPY docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
