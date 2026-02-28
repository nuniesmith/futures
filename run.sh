#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Futures Trading Co-Pilot — Run Script
# =============================================================================
#
# Usage:
#   ./run.sh              Build, test, lint, then start Docker Compose
#   ./run.sh --local      Run locally with a Python virtual environment
#   ./run.sh --down       Stop Docker Compose services
#   ./run.sh --test       Run tests + lint only (no compose)
#   ./run.sh --monitoring Include Prometheus + Grafana
#   ./run.sh --help       Show this help message
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
ENV_FILE=".env"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo -e "${CYAN}[run]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ✓ ]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[fail]${NC} $*"; }

# Generate a cryptographically random string (URL-safe base64, no padding)
gen_secret() {
    python3 -c "import secrets; print(secrets.token_urlsafe(${1:-32}))"
}

usage() {
    echo "Usage: ./run.sh [--local | --down | --test | --monitoring | --help]"
    echo ""
    echo "  (no args)       Build, test, lint, then start Docker Compose"
    echo "  --local         Run locally with a Python virtual environment"
    echo "  --down          Stop Docker Compose services"
    echo "  --test          Run tests + lint only (skip Docker build)"
    echo "  --monitoring    Include Prometheus + Grafana (monitoring profile)"
    echo "  --help          Show this help message"
}

# ---------------------------------------------------------------------------
# Virtual-environment management
# ---------------------------------------------------------------------------

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment in ${VENV_DIR} ..."
        python3 -m venv "$VENV_DIR"
        ok "Virtual environment created"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    log "Updating pip ..."
    pip install --upgrade pip -q

    log "Installing project + dev dependencies from pyproject.toml ..."
    pip install -e ".[dev]" -q
    ok "Dependencies up to date"
}

# ---------------------------------------------------------------------------
# .env generation
# ---------------------------------------------------------------------------

ensure_env() {
    if [ -f "$ENV_FILE" ]; then
        ok ".env file already exists"
    else
        log "No .env file found — generating with secure random secrets ..."

        local pg_pass
        local redis_pass
        local secret_key
        pg_pass="$(gen_secret 32)"
        redis_pass="$(gen_secret 24)"
        secret_key="$(gen_secret 48)"

        cat > "$ENV_FILE" <<EOF
# =============================================================================
# Futures Trading Co-Pilot — Environment
# Generated on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# =============================================================================

# ---- Postgres ----
POSTGRES_USER=futures_user
POSTGRES_PASSWORD=${pg_pass}
POSTGRES_DB=futures_db

# ---- Redis ----
REDIS_PASSWORD=${redis_pass}

# ---- App Secret Key (sessions, CSRF, etc.) ----
SECRET_KEY=${secret_key}

# ---- Grafana (monitoring profile) ----
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# ---- API Keys (you must fill these in) ----

# Massive.com API key for real-time futures data (CME/CBOT/NYMEX/COMEX)
# Sign up at https://massive.com/dashboard → API Keys
# Without this key the app falls back to yfinance (delayed data)
MASSIVE_API_KEY=your_massive_api_key_here

# xAI Grok API key for the AI Analyst tab
# Get yours at https://console.x.ai
XAI_API_KEY=your_xai_api_key_here
EOF

        ok ".env generated with secure random secrets for Postgres, Redis, etc."
        warn "You still need to set your API keys:"
        warn "  • MASSIVE_API_KEY  — https://massive.com/dashboard"
        warn "  • XAI_API_KEY      — https://console.x.ai"
        echo ""
    fi

    # Always warn about placeholder API keys
    if grep -q "your_massive_api_key_here" "$ENV_FILE" 2>/dev/null; then
        warn "MASSIVE_API_KEY is still a placeholder — real-time data disabled (yfinance fallback)"
    fi
    if grep -q "your_xai_api_key_here" "$ENV_FILE" 2>/dev/null; then
        warn "XAI_API_KEY is still a placeholder — Grok AI Analyst tab will not work"
    fi
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

run_tests() {
    log "Running tests ..."
    if python -m pytest src/tests/ -x -q --tb=short; then
        ok "All tests passed"
    else
        err "Tests failed — aborting"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

run_lint() {
    log "Running ruff linter ..."
    if python -m ruff check src/; then
        ok "Linting passed"
    else
        err "Linting failed — aborting"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

run_docker() {
    local profile_flag=""
    if [ "${MONITORING:-false}" = "true" ]; then
        profile_flag="--profile monitoring"
    fi

    log "Building and starting Docker Compose services ..."
    # shellcheck disable=SC2086
    docker compose $profile_flag up --build -d

    echo ""
    ok "Services are running:"
    echo "    Dashboard:   http://localhost:8000"
    echo "    Postgres:    localhost:5432"
    echo "    Redis:       localhost:6379"
    if [ "${MONITORING:-false}" = "true" ]; then
        echo "    Prometheus:  http://localhost:9090"
        echo "    Grafana:     http://localhost:3000"
    fi
    echo ""
    echo "  Logs:  docker compose logs -f"
    echo "  Stop:  ./run.sh --down"
}

# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------

run_local() {
    ensure_venv
    ensure_env

    log "Starting data service locally (http://localhost:8000) ..."
    PYTHONPATH=src exec uvicorn lib.services.data.main:app \
        --host 0.0.0.0 --port 8000 --reload
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MONITORING="false"

# Parse flags
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --monitoring)
            MONITORING="true"
            shift
            ;;
        --local)
            POSITIONAL+=("local")
            shift
            ;;
        --down)
            POSITIONAL+=("down")
            shift
            ;;
        --test)
            POSITIONAL+=("test")
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Default action is "docker"
ACTION="${POSITIONAL[0]:-docker}"

case "$ACTION" in
    local)
        run_local
        ;;
    down)
        log "Stopping Docker Compose services ..."
        docker compose --profile monitoring down
        ok "All services stopped"
        ;;
    test)
        ensure_venv
        run_tests
        run_lint
        ok "All checks passed"
        ;;
    docker)
        # Full pipeline: venv → env → test → lint → build → up
        ensure_venv
        ensure_env
        echo ""
        run_tests
        echo ""
        run_lint
        echo ""
        run_docker
        ;;
esac
