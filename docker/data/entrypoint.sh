#!/bin/sh
set -e

echo "[startup] Starting Data Service (FastAPI + Background Engine) on port 8000..."
echo "[startup] PYTHONPATH=${PYTHONPATH}"

# Start the data service — FastAPI with background engine
exec uvicorn src.services.data.main:app \
    --host "${DATA_SERVICE_HOST:-0.0.0.0}" \
    --port "${DATA_SERVICE_PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}"
