# Dockerfile.engine — Background Engine Service
# Runs DashboardEngine: FKS modules, optimization, backtesting,
# daily focus computation, session-aware scheduling.
# Reads market data from Redis (written by data-service), writes results back.

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

# Create data directory
RUN mkdir -p /app/data

# PYTHONPATH: bare imports resolve from /app/src
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, json; f=open('/tmp/engine_health.json'); d=json.load(f); assert d.get('healthy')" || exit 1

CMD ["python", "-m", "futures_lib.services.engine.main"]
