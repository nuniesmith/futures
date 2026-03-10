# Dockerfile.data — Data Service (FastAPI)
# Thin API layer: REST + SSE endpoints, bar cache management, Redis reads.
# Delegates to lib.services.data.main via entrypoints/data/main.py.

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS deps

WORKDIR /app

# Copy only dependency metadata first (cache-friendly layer).
# Hatchling needs the package dir to exist for the build to succeed.
COPY pyproject.toml .
COPY README.md .
COPY src/lib/__init__.py src/lib/__init__.py

RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# libpq5   — psycopg binary needs the Postgres client lib at runtime
# libgomp1 — needed by numpy/scipy compiled extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages + console scripts from the deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree (services + shared modules)
COPY src/ ./src/

# Copy static assets (trading dashboard HTML, etc.)
COPY static/ ./static/

# Copy the entrypoint shell script
COPY docker/data/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create data directory for SQLite journal / any local cache
RUN mkdir -p /app/data

# PYTHONPATH: bare imports resolve from /app/src
ENV PYTHONPATH="/app/src"

# Environment defaults (override via compose or docker run -e)
ENV DATA_SERVICE_HOST=0.0.0.0
ENV DATA_SERVICE_PORT=8000
ENV LOG_LEVEL=info
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${DATA_SERVICE_PORT:-8000}/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
