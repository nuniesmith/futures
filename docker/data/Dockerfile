# Dockerfile.data — Data Service (FastAPI + HTMX Dashboard)
# Thin API layer: serves REST endpoints, reads computed results from Redis.

# Stage 1: Install dependencies
FROM python:3.13-slim AS deps

WORKDIR /app

# Copy only dependency metadata first (cache-friendly layer)
# Hatchling needs the package dir to exist for the build to succeed
COPY pyproject.toml .
COPY src/lib/__init__.py src/lib/__init__.py
RUN pip install --no-cache-dir .

# Stage 2: Copy app code and run
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree (services + shared modules)
COPY src/ ./src/

# Create data directory for SQLite journal
RUN mkdir -p /app/data

# PYTHONPATH: bare imports resolve from /app/src
ENV PYTHONPATH="/app/src"

# Environment defaults
ENV DATA_SERVICE_HOST=0.0.0.0
ENV DATA_SERVICE_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "lib.services.data.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
