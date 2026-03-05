# Dockerfile.engine — Combined Backend Service
# Runs both the Data Service (FastAPI REST + SSE) and the Engine worker
# (DashboardEngine: Ruby modules, optimization, backtesting, scheduling).
#
# Both processes share the same src/ tree, Redis, and Postgres connections.
# The entrypoint starts uvicorn (data API) and the engine worker in parallel;
# if either exits, the container exits so Docker can restart it.
#
# CPU-only image — no CUDA/PyTorch. CNN inference uses the cpu-fallback path.
# Pull the latest trained model with:  bash scripts/sync_models.sh

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS deps

WORKDIR /app

# Copy only dependency metadata first (cache-friendly layer).
# Hatchling needs the package dir to exist for the build to succeed.
COPY pyproject.toml .
COPY src/lib/__init__.py src/lib/__init__.py

RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# libpq5 — psycopg binary needs the Postgres client lib at runtime
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

# Copy the full src/ tree (both services + shared modules)
COPY src/ ./src/

# Copy the entrypoint that fans out both processes
COPY docker/engine/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy scripts/ tree (sync_models, monitoring helpers, etc.)
COPY scripts/ ./scripts/

# Persistent data directory (journal, model cache)
RUN mkdir -p /app/data

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Defaults — override via compose environment
ENV DATA_SERVICE_HOST=0.0.0.0
ENV DATA_SERVICE_PORT=8000
ENV LOG_LEVEL=info

EXPOSE 8000

# Health check targets the data API — if the API is up, the container is healthy.
# The engine worker writes /tmp/engine_health.json; compose can add a second
# check there if needed, but a single HTTP probe is enough for depends_on.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
