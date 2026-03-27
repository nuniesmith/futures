# ===================================================================
# Sol Scalper — Dockerfile
# Works on Raspberry Pi 4 (linux/arm64) and x86_64.
#
# CHANGES from v1:
#   - Switched Poetry → pip + requirements.txt.
#     Poetry needs pyproject.toml + poetry.lock; pip only needs
#     requirements.txt. Much faster to build on ARM64 Pi.
#   - Fixed entrypoint COPY path: file lives at repo root, not docker/.
#   - Python 3.12 (slim-bookworm) — rock-solid numpy/pandas on ARM64.
# ===================================================================

# ─────────────────────────── Builder ───────────────────────────────
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps into an isolated venv for a clean runtime image
COPY requirements.txt ./
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# ─────────────────────────── Runtime ───────────────────────────────
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy only the virtual environment from builder — nothing else
COPY --from=builder /app/.venv /app/.venv

# Put the venv on PATH — no `python -m venv activate` needed
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy source and entrypoint
COPY src/           ./src/
# FIX: entrypoint lives at repo root, not in docker/ subfolder
COPY entrypoint.sh  /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
