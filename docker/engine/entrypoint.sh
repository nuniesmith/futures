#!/bin/sh
set -e

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
echo "[startup] Engine started (PID ${ENGINE_PID})"

# If either process dies, bring down the container so Docker can restart it
wait_and_exit() {
    wait -n
    EXIT_CODE=$?
    echo "[shutdown] A process exited with code ${EXIT_CODE} — shutting down container"
    kill "${DATA_PID}" "${ENGINE_PID}" 2>/dev/null || true
    exit "${EXIT_CODE}"
}

# Trap SIGTERM/SIGINT for clean shutdown
trap 'echo "[shutdown] Signal received"; kill "${DATA_PID}" "${ENGINE_PID}" 2>/dev/null || true; wait; exit 0' TERM INT

wait_and_exit
