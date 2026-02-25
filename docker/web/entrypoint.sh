#!/bin/sh
set -e

echo "[startup] Starting Streamlit UI (thin client) on port 8501..."
echo "[startup] DATA_SERVICE_URL=${DATA_SERVICE_URL:-http://localhost:8000}"
echo "[startup] PYTHONPATH=${PYTHONPATH}"

# Start Streamlit dashboard — pure UI, all heavy computation in data-service
exec streamlit run src/services/web/app.py \
    --server.port "${STREAMLIT_SERVER_PORT:-8501}" \
    --server.address "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.headless true \
    --browser.gatherUsageStats false
