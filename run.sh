#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: ./run.sh [--docker | --down | --help]"
    echo ""
    echo "  (no args)   Run locally with a Python virtual environment"
    echo "  --docker    Build and start with Docker Compose (Redis + app)"
    echo "  --down      Stop Docker Compose services"
    echo "  --help      Show this help message"
}

# ---------------------------------------------------------------------------
# Docker mode
# ---------------------------------------------------------------------------
run_docker() {
    # Generate .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "No .env file found. Generating with placeholders..."
        cat > .env <<'EOF'
# xAI Grok API key for the AI Analyst tab
# Get yours at https://console.x.ai
XAI_API_KEY=your_xai_api_key_here

# Redis URL (set automatically by docker-compose, only override for custom setups)
# REDIS_URL=redis://redis:6379/0

# SQLite journal path (inside the container)
# DB_PATH=/app/data/futures_journal.db
EOF
        echo ".env file created. Edit it to add your XAI_API_KEY, then re-run this script."
        exit 0
    fi

    # Check if XAI_API_KEY is still the placeholder
    if grep -q "your_xai_api_key_here" .env; then
        echo "WARNING: XAI_API_KEY is still set to placeholder in .env"
        echo "  The Grok AI Analyst tab will not work until you update it."
        echo ""
    fi

    echo "Starting Futures Dashboard (Docker)..."
    docker compose up --build -d

    echo ""
    echo "Dashboard is running:"
    echo "  Streamlit:  http://localhost:8501"
    echo "  Trade API:  http://localhost:8000"
    echo "  Redis:      localhost:6379"
    echo ""
    echo "Logs: docker compose logs -f app"
    echo "Stop: ./run.sh --down"
}

# ---------------------------------------------------------------------------
# Local venv mode (default)
# ---------------------------------------------------------------------------
run_local() {
    VENV_DIR=".venv"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"

    # Install / update dependencies
    echo "Installing requirements ..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q

    # Launch the Streamlit app
    echo "Starting Streamlit ..."
    streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-}" in
    --docker)
        run_docker
        ;;
    --down)
        echo "Stopping Docker Compose services..."
        docker compose down
        ;;
    --help|-h)
        usage
        ;;
    "")
        run_local
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
