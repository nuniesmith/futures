#!/bin/bash
# entrypoint.sh — start uvicorn (data API) + engine worker in parallel.
# Monitors both PIDs; if either exits the container exits so Docker restarts it.

set -eo pipefail

echo "[startup] PYTHONPATH=${PYTHONPATH}"
echo "[startup] Starting Data Service (FastAPI) on port ${DATA_SERVICE_PORT:-8000}..."

uvicorn lib.services.data.main:app \
    --host "${DATA_SERVICE_HOST:-0.0.0.0}" \
    --port "${DATA_SERVICE_PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}" &
DATA_PID=$!
echo "[startup] Data service started (PID ${DATA_PID})"

echo "[startup] Starting Engine worker..."
python -m lib.services.engine.main &
ENGINE_PID=$!
echo "[startup] Engine worker started (PID ${ENGINE_PID})"

# ---------------------------------------------------------------------------
# Graceful shutdown — kill both children on SIGTERM / SIGINT
# ---------------------------------------------------------------------------
_shutdown() {
    echo "[shutdown] Signal received — stopping children..."
    kill "${DATA_PID}" "${ENGINE_PID}" 2>/dev/null || true
    wait "${DATA_PID}" "${ENGINE_PID}" 2>/dev/null || true
    echo "[shutdown] Done"
    exit 0
}
trap _shutdown TERM INT

# ---------------------------------------------------------------------------
# Monitor loop — poll both PIDs every second.
# If either process exits, bring down the container so Docker can restart it.
# Using a polling loop instead of `wait -n` for POSIX portability
# (dash / sh does not support `wait -n`).
# ---------------------------------------------------------------------------
while true; do
    # Check data service
    if ! kill -0 "${DATA_PID}" 2>/dev/null; then
        wait "${DATA_PID}" 2>/dev/null
        EXIT_CODE=$?
        echo "[monitor] Data service (PID ${DATA_PID}) exited with code ${EXIT_CODE} — shutting down"
        kill "${ENGINE_PID}" 2>/dev/null || true
        exit "${EXIT_CODE}"
    fi

    # Check engine worker
    if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
        wait "${ENGINE_PID}" 2>/dev/null
        EXIT_CODE=$?
        echo "[monitor] Engine worker (PID ${ENGINE_PID}) exited with code ${EXIT_CODE} — shutting down"
        kill "${DATA_PID}" 2>/dev/null || true
        exit "${EXIT_CODE}"
    fi

    sleep 1
done
