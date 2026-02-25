# Dockerfile.data — Data Service (FastAPI + Background Engine)
# All heavy computation runs here: Massive WS, optimization, backtesting,
# wave analysis, volatility, signal quality, regime, CVD, ICT.

# Stage 1: Install dependencies
FROM python:3.13-slim AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# PYTHONPATH setup:
#   1. /app/src                     — so bare imports like `from cache import ...`,
#      `from engine import ...`, etc. resolve correctly (single source of truth)
#   2. /app/src/services/data       — so `from api.X import ...` in main.py resolves
#   3. /app                         — so `src.services.data.main` style imports work
ENV PYTHONPATH="/app/src:/app/src/services/data:/app"

# Environment defaults
ENV DATA_SERVICE_HOST=0.0.0.0
ENV DATA_SERVICE_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.services.data.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
