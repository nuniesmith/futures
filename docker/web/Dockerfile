# Dockerfile.web — Web Dashboard Frontend Service
# Thin HTMX dashboard that reverse-proxies API/SSE to the data service.
# No Redis, no Postgres, no heavy computation — just FastAPI + httpx.

# Stage 1: Install dependencies
FROM python:3.13-slim AS deps

WORKDIR /app

# Copy dependency metadata + minimal package stub so hatchling can resolve
COPY pyproject.toml .
RUN mkdir -p src/lib && touch src/lib/__init__.py

# Install only the base deps (no gpu extras needed for the web frontend).
# The web service only uses: fastapi, uvicorn, httpx, structlog
RUN pip install --no-cache-dir .

# Stage 2: Copy app code and run
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree (web service + shared modules it may reference)
COPY src/ ./src/

# Copy static assets (trading dashboard HTML, etc.)
COPY static/ ./static/

# PYTHONPATH: bare imports resolve from /app/src
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Environment defaults
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8080
ENV DATA_SERVICE_URL=http://data:8000
ENV LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "lib.services.web.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
