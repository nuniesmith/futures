#!/usr/bin/env bash
# =============================================================================
# Sol Scalper — Project Management Script
# =============================================================================
#
# Usage:
#   ./run.sh              Run full pipeline: venv → deps → test → docker up
#   ./run.sh setup        Set up .venv and install dependencies only
#   ./run.sh test         Run pytest on src/tests
#   ./run.sh build        Build Docker image only
#   ./run.sh up           Start Docker Compose service
#   ./run.sh down         Stop Docker Compose service
#   ./run.sh logs         Follow service logs
#   ./run.sh status       Show service status
#   ./run.sh clean        Remove .venv and Docker artifacts
#   ./run.sh help         Show this help
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
PYTEST="${VENV_DIR}/bin/pytest"
REQUIREMENTS="${PROJECT_DIR}/requirements.txt"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"
SERVICE_NAME="sol-scalper"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { echo -e "${BLUE}ℹ ${NC} $*"; }
success() { echo -e "${GREEN}✅${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠️ ${NC} $*"; }
error()   { echo -e "${RED}❌${NC} $*"; }
header()  { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ---------------------------------------------------------------------------
# setup — create .venv, upgrade pip, install requirements
# ---------------------------------------------------------------------------
cmd_setup() {
    header "Setting up Python environment"

    # Find python3
    local py=""
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            py="$candidate"
            break
        fi
    done
    if [ -z "$py" ]; then
        error "Python 3 not found. Install python3 first."
        exit 1
    fi
    info "Using: $py ($($py --version 2>&1))"

    # Create venv if missing
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at ${VENV_DIR}..."
        "$py" -m venv "$VENV_DIR"
        success "Virtual environment created"
    else
        info "Virtual environment already exists at ${VENV_DIR}"
    fi

    # Upgrade pip
    info "Upgrading pip..."
    "$PIP" install --quiet --upgrade pip

    # Install dependencies
    if [ ! -f "$REQUIREMENTS" ]; then
        error "requirements.txt not found at ${REQUIREMENTS}"
        exit 1
    fi
    info "Installing dependencies from requirements.txt..."
    "$PIP" install --quiet -r "$REQUIREMENTS"

    # Install test dependencies (pytest + pytest-asyncio)
    info "Installing test dependencies..."
    "$PIP" install --quiet pytest pytest-asyncio

    success "All dependencies installed"
    echo ""
    "$PIP" list --format=columns 2>/dev/null | head -20
    echo "  ..."
}

# ---------------------------------------------------------------------------
# test — run pytest on src/tests
# ---------------------------------------------------------------------------
cmd_test() {
    header "Running tests"

    if [ ! -f "$PYTEST" ]; then
        warn "pytest not found in .venv — running setup first..."
        cmd_setup
    fi

    info "Running: pytest src/tests/ -v --tb=short"
    echo ""

    cd "$PROJECT_DIR"
    "$PYTEST" src/tests/ -v --tb=short -q
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        success "All tests passed!"
    else
        error "Tests failed (exit code: ${exit_code})"
        exit $exit_code
    fi

    return $exit_code
}

# ---------------------------------------------------------------------------
# build — build Docker image
# ---------------------------------------------------------------------------
cmd_build() {
    header "Building Docker image"

    if [ ! -f "$COMPOSE_FILE" ]; then
        error "docker-compose.yml not found at ${COMPOSE_FILE}"
        exit 1
    fi

    # Check for .env file
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        warn ".env file not found — Docker build will proceed but the"
        warn "container needs KUCOIN_API_KEY, KUCOIN_API_SECRET, and"
        warn "KUCOIN_PASSPHRASE to run."
    fi

    info "Building ${SERVICE_NAME}..."
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" build --no-cache
    success "Docker image built"
}

# ---------------------------------------------------------------------------
# up — start Docker Compose service
# ---------------------------------------------------------------------------
cmd_up() {
    header "Starting ${SERVICE_NAME}"

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        error ".env file required. Create it with your KuCoin credentials:"
        echo ""
        echo "  KUCOIN_API_KEY=your_key"
        echo "  KUCOIN_API_SECRET=your_secret"
        echo "  KUCOIN_PASSPHRASE=your_passphrase"
        echo "  DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/..."
        echo "  LEVERAGE=20"
        echo "  CAPITAL=30"
        echo ""
        exit 1
    fi

    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" up -d
    success "${SERVICE_NAME} is running"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
}

# ---------------------------------------------------------------------------
# down — stop Docker Compose service
# ---------------------------------------------------------------------------
cmd_down() {
    header "Stopping ${SERVICE_NAME}"
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down
    success "Stopped"
}

# ---------------------------------------------------------------------------
# logs — follow service logs
# ---------------------------------------------------------------------------
cmd_logs() {
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100 "$SERVICE_NAME"
}

# ---------------------------------------------------------------------------
# status — show service status
# ---------------------------------------------------------------------------
cmd_status() {
    header "Service Status"
    cd "$PROJECT_DIR"

    if docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "$SERVICE_NAME"; then
        docker compose -f "$COMPOSE_FILE" ps
    else
        warn "No running containers found"
    fi
}

# ---------------------------------------------------------------------------
# clean — remove .venv and docker artifacts
# ---------------------------------------------------------------------------
cmd_clean() {
    header "Cleaning up"

    if [ -d "$VENV_DIR" ]; then
        info "Removing .venv..."
        rm -rf "$VENV_DIR"
        success "Virtual environment removed"
    fi

    info "Removing Docker artifacts..."
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down --rmi local -v 2>/dev/null || true
    success "Clean complete"
}

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
cmd_help() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║     Sol Scalper — Project Manager        ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (no args)   Full pipeline: setup → test → build → up"
    echo "  setup       Create .venv, install dependencies"
    echo "  test        Run pytest on src/tests/"
    echo "  build       Build Docker image"
    echo "  up          Start Docker Compose service"
    echo "  down        Stop Docker Compose service"
    echo "  logs        Follow service logs"
    echo "  status      Show service status"
    echo "  clean       Remove .venv and Docker artifacts"
    echo "  help        Show this help"
    echo ""
}

# ---------------------------------------------------------------------------
# Full pipeline (default) — setup → test → build → up
# ---------------------------------------------------------------------------
cmd_full() {
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║     Sol Scalper — Full Pipeline          ║"
    echo "║     KuCoin SOLUSDTM Perpetual Futures    ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Step 1: Setup
    cmd_setup

    # Step 2: Tests
    cmd_test

    # Step 3: Build
    cmd_build

    # Step 4: Start
    cmd_up

    echo ""
    header "Pipeline complete!"
    success "Sol Scalper is live. Use './run.sh logs' to monitor."
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

case "${1:-}" in
    setup)    cmd_setup  ;;
    test)     cmd_test   ;;
    build)    cmd_build  ;;
    up)       cmd_up     ;;
    down)     cmd_down   ;;
    logs)     cmd_logs   ;;
    status)   cmd_status ;;
    clean)    cmd_clean  ;;
    help|-h|--help) cmd_help ;;
    "")       cmd_full   ;;
    *)
        error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
