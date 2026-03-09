# Dockerfile.engine — Engine Worker
# Runs the DashboardEngine: Ruby modules, optimization, backtesting,
# daily focus computation, session-aware scheduling, and live risk publishing.
#
# The Data Service (FastAPI REST + SSE) now runs in its own container built
# from docker/data/Dockerfile. The engine publishes computed state to Redis
# for the data service to serve.
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

# libpq5   — psycopg binary needs the Postgres client lib at runtime
# libgomp1 — needed by numpy/scipy compiled extensions
# bash     — entrypoint.sh uses #!/bin/bash
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages + console scripts from the deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree (engine + shared modules)
COPY src/ ./src/

# Copy the entrypoint
COPY docker/engine/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy scripts/ tree (sync_models, monitoring helpers, etc.)
COPY scripts/ ./scripts/

# Persistent data directory (journal, model cache)
RUN mkdir -p /app/data

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=info

# Health check — the engine worker writes a heartbeat file every cycle.
# No HTTP port is exposed; the data service owns port 8000.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD test -f /tmp/engine_health.json || exit 1

ENTRYPOINT ["/entrypoint.sh"]
