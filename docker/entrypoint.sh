#!/bin/sh
set -e

# Start the FastAPI trade/positions API on port 8000 in the background
echo "[startup] Starting Trade API on port 8000..."
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --log-level info &
API_PID=$!

# Give the API a moment to bind
sleep 1

# Start Streamlit dashboard on port 8501 in the foreground
echo "[startup] Starting Streamlit dashboard on port 8501..."
exec streamlit run src/app.py
