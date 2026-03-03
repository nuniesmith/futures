# Dockerfile.engine — Background Engine Service (CPU-only)
# Runs DashboardEngine: Ruby modules, optimization, backtesting,
# daily focus computation, session-aware scheduling.
# Reads market data from Redis (written by data-service), writes results back.
#
# CPU-only image — no CUDA runtime. PyTorch is NOT installed here.
# CNN inference in the engine uses the cpu-fallback path in breakout_cnn.py.
# GPU training is handled by the separate `trainer` service defined in
# docker-compose.train.yml, intended to run on a dedicated GPU machine.
# Trained models are copied back via scripts/sync_models.sh.
#
# Python 3.13 via deadsnakes PPA on Ubuntu 24.04.

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

# Install deadsnakes PPA for Python 3.13 + build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
        build-essential \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.13 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Bootstrap pip for 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

WORKDIR /app

# Copy dependency metadata + minimal package stub so hatchling can resolve
COPY pyproject.toml .
RUN mkdir -p src/lib && touch src/lib/__init__.py

# Install base deps only — no [gpu] extras, no CUDA torch wheels.
# torch/torchvision are intentionally excluded; the engine uses the
# cpu-safe fallback in breakout_cnn.py for any inference it needs to do.
RUN pip install --no-cache-dir "."

# ---------------------------------------------------------------------------
# Stage 2: Copy app code and run
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.13 runtime only (no dev/build packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        libpq5 \
        libgomp1 \
    && apt-get purge -y --auto-remove software-properties-common gpg-agent \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

WORKDIR /app

# Copy installed packages + pip/scripts from deps stage
COPY --from=deps /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=deps /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=deps /usr/lib/python3.13 /usr/lib/python3.13
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the full src/ tree (services + shared modules)
COPY src/ ./src/

# Copy scripts/ tree (retrain_overnight.py, train_gpu.py, etc.)
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p /app/data

# PYTHONPATH: bare imports resolve from /app/src
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, json; f=open('/tmp/engine_health.json'); d=json.load(f); assert d.get('healthy')" || exit 1

CMD ["python", "-m", "lib.services.engine.main"]
