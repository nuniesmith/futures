# ===================================================================
# Futures Dashboard — Web Service Dockerfile
# FastAPI + HTMX dashboard (port 8080)
# Works on Raspberry Pi 4 (linux/arm64) and x86_64.
# ===================================================================

# ─────────────────────────── Builder ───────────────────────────────
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all deps (including web extras) into an isolated venv
COPY requirements.txt ./
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /app/.venv/bin/pip install --no-cache-dir \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "jinja2>=3.1" \
        "python-multipart>=0.0.9" \
        "bcrypt>=4.1" \
        "aiofiles>=23.2"

# ─────────────────────────── Runtime ───────────────────────────────
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Tell uvicorn / the app where templates + static files live
    WEB_HOST="0.0.0.0" \
    WEB_PORT="8080"

# Copy only what the web service needs
COPY src/ ./src/

# The web service reads config to resolve asset names — mount or bake it in
COPY config/ ./config/

EXPOSE 8080

# Run with uvicorn; --workers 1 keeps Redis connection singleton clean
CMD ["uvicorn", "src.web.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
