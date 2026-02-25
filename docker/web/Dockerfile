# Dockerfile.streamlit — Lightweight Streamlit UI container
# Only serves the dashboard; all heavy computation happens in data-service.

# Stage 1: Install dependencies
FROM python:3.13-slim AS deps

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy app code and run
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree — the web app needs shared modules
# (models, cache, grok_helper, scorer, etc.) for direct imports
COPY src/ ./src/

# Create data directory for SQLite journal reads
RUN mkdir -p /app/data

# PYTHONPATH setup:
#   1. /app/src  — so bare imports like `from cache import ...`, `from models import ...`
#      resolve correctly (these are shared modules used by the Streamlit app)
#   2. /app      — so `src.services.web.app` style imports work if needed
ENV PYTHONPATH="/app/src:/app"

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/services/web/app.py"]
