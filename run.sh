#!/usr/bin/env bash
# =============================================================================
# Futures Trader — Project Management Script
# =============================================================================
# Multi-asset KuCoin perpetual futures trader + HTMX web dashboard
# Repository: nuniesmith/futures
#
# Usage:
#   ./run.sh              Full pipeline: venv → deps → lint → test → build → up
#   ./run.sh setup        Set up .venv and install dependencies only
#   ./run.sh test         Run pytest on src/tests
#   ./run.sh lint         Run ruff check + ruff format --check + mypy
#   ./run.sh sim          Start bot in simulation mode (local, no Docker)
#   ./run.sh live         Start bot in LIVE trading mode (real orders — be careful)
#   ./run.sh web          Start HTMX dashboard locally (uvicorn, port 8080)
#   ./run.sh web-up       Start web dashboard Docker service
#   ./run.sh web-down     Stop web dashboard Docker service
#   ./run.sh web-logs     Follow web dashboard container logs
#   ./run.sh web-hash-password   Generate bcrypt hash for WEB_PASSWORD_HASH
#   ./run.sh tailscale-serve     Expose dashboard at https://desktop.tailfef10.ts.net
#   ./run.sh tailscale-stop      Remove tailscale serve config for :8080
#   ./run.sh build        Build Docker images only
#   ./run.sh up           Start all Docker Compose services (futures + redis + web)
#   ./run.sh down         Stop all Docker Compose services
#   ./run.sh logs         Follow futures service logs
#   ./run.sh redis-logs   Follow redis service logs
#   ./run.sh status       Show service + Tailscale status
#   ./run.sh clean        Remove .venv and Docker artifacts
#   ./run.sh help         Show this help
#
# Tailscale:
#   Machine  : desktop  (100.109.182.42)
#   Domain   : desktop.tailfef10.ts.net
#   Dashboard: https://desktop.tailfef10.ts.net  (after tailscale-serve)
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
RUFF="${VENV_DIR}/bin/ruff"
MYPY="${VENV_DIR}/bin/mypy"
REQUIREMENTS="${PROJECT_DIR}/requirements.txt"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"
SERVICE_NAME="futures"
REDIS_SERVICE="redis"
WEB_SERVICE="web"

# Web dashboard
WEB_PORT="${WEB_PORT:-8080}"
TAILSCALE_HOST="desktop.tailfef10.ts.net"
TAILSCALE_IP="100.109.182.42"

export TZ="America/New_York"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { echo -e "${BLUE}ℹ ${NC} $*"; }
success() { echo -e "${GREEN}✅${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠️ ${NC} $*"; }
error()   { echo -e "${RED}❌${NC} $*"; }
header()  {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║         Futures Trader — Multi-Asset           ║${NC}"
    echo -e "${BOLD}${CYAN}║   KuCoin USDTM Perpetual Futures Scalper      ║${NC}"
    echo -e "${BOLD}${CYAN}║                                                ║${NC}"
    echo -e "${BOLD}${CYAN}║   BTC  ETH  SOL  DOGE  SUI  PEPE             ║${NC}"
    echo -e "${BOLD}${CYAN}║   AVAX  WIF  FARTCOIN  KCS  + 50 more        ║${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Tailscale helpers
# ---------------------------------------------------------------------------

# Returns 0 if tailscale CLI is available and daemon is running
_tailscale_available() {
    command -v tailscale &>/dev/null && tailscale status &>/dev/null 2>&1
}

# Returns 0 if tailscale serve is currently proxying the given port
_tailscale_serving_port() {
    local port="${1:-8080}"
    tailscale serve status 2>/dev/null | grep -q "proxy http://127.0.0.1:${port}"
}

# Print tailscale serve status block
_print_tailscale_status() {
    if ! _tailscale_available; then
        echo -e "  ${DIM}Tailscale: not found or not running${NC}"
        return
    fi

    local serve_output
    serve_output="$(tailscale serve status 2>/dev/null)" || serve_output=""

    if [ -z "$serve_output" ]; then
        echo -e "  ${YELLOW}Tailscale serve: nothing configured${NC}"
        echo -e "  ${DIM}Run: ./run.sh tailscale-serve  to expose dashboard${NC}"
        return
    fi

    # Check if :443 / https is in the serve output
    if echo "$serve_output" | grep -q "https://"; then
        local ts_url
        ts_url="$(echo "$serve_output" | grep -o 'https://[^ ]*' | head -1)"

        if _tailscale_serving_port "${WEB_PORT}"; then
            echo -e "  ${GREEN}● Tailscale HTTPS:${NC} ${BOLD}${ts_url}${NC}  ${GREEN}→ :${WEB_PORT}${NC}"
        else
            # HTTPS is active but pointing elsewhere
            local proxy_target
            proxy_target="$(echo "$serve_output" | grep 'proxy' | awk '{print $NF}' | head -1)"
            echo -e "  ${YELLOW}● Tailscale HTTPS:${NC} ${ts_url}  ${YELLOW}→ ${proxy_target}${NC}"
            echo -e "  ${DIM}  Dashboard not exposed — run: ./run.sh tailscale-serve${NC}"
        fi
    else
        echo -e "  ${YELLOW}Tailscale serve: active but no HTTPS${NC}"
        echo -e "$serve_output" | sed 's/^/    /'
    fi
}

# ---------------------------------------------------------------------------
# setup — create .venv, upgrade pip, install requirements
# ---------------------------------------------------------------------------
cmd_setup() {
    header "Setting up Python environment"

    local py=""
    for candidate in python3.13 python3.12 python3.11 python3; do
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

    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at ${VENV_DIR}..."
        "$py" -m venv "$VENV_DIR"
        success "Virtual environment created"
    else
        info "Virtual environment already exists at ${VENV_DIR}"
    fi

    info "Upgrading pip..."
    "$PIP" install --quiet --upgrade pip

    if [ -f "$REQUIREMENTS" ]; then
        info "Installing dependencies from requirements.txt..."
        "$PIP" install --quiet -r "$REQUIREMENTS"
    else
        warn "requirements.txt not found — installing core dependencies directly..."
        "$PIP" install --quiet ccxt optuna pandas aiohttp pyyaml python-dotenv redis \
            fastapi "uvicorn[standard]" jinja2 python-multipart bcrypt
    fi

    info "Installing dev dependencies..."
    "$PIP" install --quiet pytest pytest-asyncio ruff mypy fakeredis

    success "All dependencies installed"
    echo ""
    "$PIP" list --format=columns 2>/dev/null | head -30
    echo -e "  ${DIM}...${NC}"
}

# ---------------------------------------------------------------------------
# lint — ruff + mypy
# ---------------------------------------------------------------------------
cmd_lint() {
    header "Running linters"

    if [ ! -f "$RUFF" ]; then
        warn "ruff not found in .venv — running setup first..."
        cmd_setup
    fi

    local exit_code=0

    info "Running: ruff check src/"
    "$RUFF" check src/ --fix || exit_code=$?

    info "Running: ruff format --check src/"
    "$RUFF" format --check src/ || exit_code=$?

    if [ -f "$MYPY" ]; then
        info "Running: mypy src/"
        "$MYPY" src/ --ignore-missing-imports || exit_code=$?
    else
        warn "mypy not found — skipping type checks"
    fi

    echo ""
    if [ $exit_code -eq 0 ]; then
        success "All lint checks passed!"
    else
        error "Lint checks failed (exit code: ${exit_code})"
        exit $exit_code
    fi
}

# ---------------------------------------------------------------------------
# test — run pytest
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
}

# ---------------------------------------------------------------------------
# sim — start in simulation mode (local, no Docker)
# ---------------------------------------------------------------------------
cmd_sim() {
    header "Starting in SIMULATION mode"

    if [ ! -f "$PYTHON" ]; then
        warn "Python venv not found — running setup first..."
        cmd_setup
    fi

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        warn ".env file not found — KuCoin credentials are still needed for WS feeds."
    fi

    # When running locally (not in Docker), override REDIS_URL to point at
    # the Redis port exposed on the host.  The Docker service name "redis" is
    # only resolvable inside the futures_net Docker network.
    local redis_pass=""
    if [ -f "${PROJECT_DIR}/.env" ]; then
        redis_pass=$(grep -E '^REDIS_PASSWORD=' "${PROJECT_DIR}/.env" \
            | cut -d= -f2- | tr -d '"'"'" 2>/dev/null || true)
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠️  SIMULATION MODE — NO REAL ORDERS WILL EXECUTE${NC}"
    echo -e "${YELLOW}  Live WS data • Paper trades • PnL tracked in Redis${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd "$PROJECT_DIR"
    TRADING_MODE=sim \
    SIM_MODE=true \
    REDIS_URL="redis://127.0.0.1:6379/0" \
    REDIS_PASSWORD="${redis_pass}" \
        "$PYTHON" -m src.main
}

# ---------------------------------------------------------------------------
# live — start in LIVE trading mode (local, no Docker)
# ---------------------------------------------------------------------------
cmd_live() {
    header "Starting in LIVE trading mode"

    if [ ! -f "$PYTHON" ]; then
        warn "Python venv not found — running setup first..."
        cmd_setup
    fi

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        error ".env file required for live trading."
        exit 1
    fi

    # ── Credential checks ────────────────────────────────────────
    local missing=0
    for var in KUCOIN_API_KEY KUCOIN_API_SECRET KUCOIN_PASSPHRASE REDIS_PASSWORD XAI_API_KEY; do
        val=$(grep -E "^${var}=" "${PROJECT_DIR}/.env" | cut -d= -f2- | tr -d '"'"'" 2>/dev/null || true)
        if [ -z "$val" ]; then
            error "Missing required .env variable: ${var}"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo ""
        echo -e "  Set all required variables in ${CYAN}.env${NC} before going live."
        exit 1
    fi

    # ── Fetch real balance from KuCoin for display ──────────────────
    local capital=""
    local capital_note=""

    info "Fetching account balance from KuCoin..."
    capital=$(cd "$PROJECT_DIR" && "$PYTHON" <<'PYEOF' 2>/dev/null
import sys, os
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
import ccxt
ex = ccxt.kucoinfutures({
    'apiKey': os.environ.get('KUCOIN_API_KEY', ''),
    'secret': os.environ.get('KUCOIN_API_SECRET', ''),
    'password': os.environ.get('KUCOIN_PASSPHRASE', ''),
})
b = ex.fetch_balance()
usdt = b.get('USDT', {}).get('total') or b.get('total', {}).get('USDT')
if usdt is not None:
    print(f'{float(usdt):.2f}')
else:
    sys.exit(1)
PYEOF
) || capital=""

    if [ -n "$capital" ]; then
        capital_note=" (live balance)"
    else
        warn "Could not fetch live balance — using config value"
        capital=$(grep -E '^CAPITAL=' "${PROJECT_DIR}/.env" \
            | cut -d= -f2- | cut -d'#' -f1 | tr -d '"'"' \t" 2>/dev/null || true)
        if [ -z "$capital" ]; then
            capital=$(grep -E 'balance_usdt:' "${PROJECT_DIR}/config/futures.yaml" \
                | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || echo "30.0")
            capital_note=" (yaml default)"
        else
            capital_note=" (config)"
        fi
    fi

    local redis_pass
    redis_pass=$(grep -E '^REDIS_PASSWORD=' "${PROJECT_DIR}/.env" \
        | cut -d= -f2- | tr -d '"'"'" 2>/dev/null || true)

    # ── Hard confirmation gate ────────────────────────────────────
    echo ""
    # Build capital line with fixed-width box (48 chars inner width)
    local cap_line="  Capital  : \$${capital} USDT${capital_note}"
    local pad_len=$(( 48 - ${#cap_line} ))
    local padding=""
    local i=0
    while [ $i -lt $pad_len ]; do padding="${padding} "; i=$(( i + 1 )); done

    echo -e "${RED}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║        ⚠   LIVE TRADING MODE   ⚠              ║${NC}"
    echo -e "${RED}║                                                ║${NC}"
    echo -e "${RED}║  REAL ORDERS WILL BE PLACED ON KUCOIN          ║${NC}"
    echo -e "${RED}║  REAL MONEY IS AT RISK                         ║${NC}"
    echo -e "${RED}║                                                ║${NC}"
    echo -e "${RED}║${NC}${cap_line}${padding}${RED}║${NC}"
    echo -e "${RED}║  Exchange : KuCoin Futures (USDTM perpetuals)  ║${NC}"
    echo -e "${RED}║  Mode     : Isolated margin, one-way position  ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Redis will persist all trades at ${CYAN}redis://127.0.0.1:6379/0${NC}"
    echo -e "  Dashboard available at ${CYAN}http://localhost:8080${NC} (if running)"
    echo ""
    printf "  Type %s to confirm, anything else cancels: " "$(echo -e "${RED}LIVE${NC}")"
    read -r confirm
    echo ""

    if [ "$confirm" != "LIVE" ]; then
        warn "Cancelled — you must type exactly: LIVE"
        exit 0
    fi

    echo -e "${RED}  ● Starting LIVE bot in 3 seconds... Ctrl+C to abort${NC}"
    sleep 3

    cd "$PROJECT_DIR"
    TRADING_MODE=live \
    SIM_MODE=false \
    REDIS_URL="redis://127.0.0.1:6379/0" \
    REDIS_PASSWORD="${redis_pass}" \
        "$PYTHON" -m src.main
}

# ---------------------------------------------------------------------------
# web — start HTMX dashboard locally (no Docker)
# ---------------------------------------------------------------------------
cmd_web() {
    header "Starting HTMX Dashboard (local dev)"

    if [ ! -f "$PYTHON" ]; then
        warn "Python venv not found — running setup first..."
        cmd_setup
    fi

    # Same localhost override as cmd_sim — point at the Docker-exposed Redis port
    local redis_url="redis://127.0.0.1:6379/0"
    local redis_pass=""
    if [ -f "${PROJECT_DIR}/.env" ]; then
        redis_pass=$(grep -E '^REDIS_PASSWORD=' "${PROJECT_DIR}/.env" | cut -d= -f2- | tr -d '"'"'" 2>/dev/null || true)
    fi

    echo ""
    echo -e "  ${CYAN}Dashboard URL:${NC}  ${BOLD}http://localhost:${WEB_PORT}${NC}"

    # Show Tailscale URL if serve is active for this port
    if _tailscale_available && _tailscale_serving_port "${WEB_PORT}"; then
        echo -e "  ${GREEN}Tailscale URL:${NC}  ${BOLD}https://${TAILSCALE_HOST}${NC}  ${GREEN}(● live)${NC}"
    elif _tailscale_available; then
        echo -e "  ${DIM}Tailscale:      not serving :${WEB_PORT} — run: ./run.sh tailscale-serve${NC}"
    fi

    echo -e "  ${DIM}Redis:${NC}          ${redis_url}"
    echo ""
    echo -e "  ${DIM}Press Ctrl+C to stop${NC}"
    echo ""

    cd "$PROJECT_DIR"
    REDIS_URL="${redis_url}" \
    REDIS_PASSWORD="${redis_pass}" \
    WEB_PORT="${WEB_PORT}" \
    "${VENV_DIR}/bin/uvicorn" src.web.app:app \
        --host 0.0.0.0 \
        --port "${WEB_PORT}" \
        --reload \
        --reload-dir src/web \
        --log-level info
}

# ---------------------------------------------------------------------------
# web-up — start web dashboard as Docker service
# ---------------------------------------------------------------------------
cmd_web_up() {
    header "Starting Web Dashboard (Docker)"

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        error ".env file required. See ./run.sh help."
        exit 1
    fi

    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" up -d "$WEB_SERVICE"
    success "Web dashboard started"
    echo ""

    sleep 1
    docker compose -f "$COMPOSE_FILE" ps "$WEB_SERVICE"
    echo ""
    echo -e "  ${CYAN}Local:${NC}      http://localhost:${WEB_PORT}"

    if _tailscale_available && _tailscale_serving_port "${WEB_PORT}"; then
        echo -e "  ${GREEN}Tailscale:${NC}  https://${TAILSCALE_HOST}  ${GREEN}(● live)${NC}"
    else
        echo -e "  ${DIM}Tailscale:  run ./run.sh tailscale-serve to expose${NC}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# web-down — stop web dashboard Docker service
# ---------------------------------------------------------------------------
cmd_web_down() {
    header "Stopping Web Dashboard"
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" stop "$WEB_SERVICE"
    docker compose -f "$COMPOSE_FILE" rm -f "$WEB_SERVICE"
    success "Web dashboard stopped"
}

# ---------------------------------------------------------------------------
# web-logs — follow web container logs
# ---------------------------------------------------------------------------
cmd_web_logs() {
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100 "$WEB_SERVICE"
}

# ---------------------------------------------------------------------------
# web-hash-password — generate bcrypt hash for WEB_PASSWORD_HASH
# ---------------------------------------------------------------------------
cmd_web_hash_password() {
    header "Generate Dashboard Password Hash"

    if [ ! -f "$PYTHON" ]; then
        warn "Python venv not found — running setup first..."
        cmd_setup
    fi

    echo -e "  This generates a bcrypt hash to put in ${CYAN}WEB_PASSWORD_HASH${NC} in your .env"
    echo -e "  ${DIM}Leave blank for no auth (dev mode)${NC}"
    echo ""

    printf "  Enter password: "
    read -rs password
    echo ""

    if [ -z "$password" ]; then
        warn "Empty password — skipping"
        return
    fi

    local hash
    hash=$("$PYTHON" -c "import bcrypt, sys; pw=sys.argv[1].encode(); print(bcrypt.hashpw(pw, bcrypt.gensalt()).decode())" "$password" 2>/dev/null) || {
        error "bcrypt not installed. Run: ./run.sh setup"
        exit 1
    }

    echo ""
    echo -e "  ${GREEN}Hash generated:${NC}"
    echo ""
    echo -e "  ${BOLD}WEB_PASSWORD_HASH=${hash}${NC}"
    echo ""
    echo -e "  Add this line to your ${CYAN}.env${NC} file."
    echo -e "  Also set: ${CYAN}WEB_SESSION_SECRET=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p | tr -d '\n')${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# tailscale-serve — expose dashboard at https://desktop.tailfef10.ts.net
# ---------------------------------------------------------------------------
cmd_tailscale_serve() {
    header "Tailscale Serve — Expose Dashboard"

    if ! command -v tailscale &>/dev/null; then
        error "tailscale CLI not found"
        exit 1
    fi

    if ! tailscale status &>/dev/null; then
        error "Tailscale daemon is not running"
        exit 1
    fi

    echo -e "  Configuring Tailscale to serve the dashboard..."
    echo -e "  ${DIM}Local port: :${WEB_PORT}  →  https://${TAILSCALE_HOST}${NC}"
    echo ""

    # Remove any existing serve config on the default HTTPS port first
    # so we don't stack up duplicate rules
    tailscale serve --https=443 off 2>/dev/null || true

    # Proxy HTTPS :443 → local :8080 (background / persistent)
    tailscale serve --bg --https=443 "http://127.0.0.1:${WEB_PORT}"

    echo ""
    success "Tailscale serve configured"
    echo ""
    echo -e "  ${GREEN}● Dashboard URL:${NC} ${BOLD}https://${TAILSCALE_HOST}${NC}"
    echo -e "  ${GREEN}● Tailscale IP:${NC}  https://${TAILSCALE_IP}"
    echo ""
    echo -e "  ${DIM}Accessible from any device on your tailnet (rasp, etc.)${NC}"
    echo -e "  ${DIM}Run 'tailscale serve status' to verify${NC}"
    echo ""

    tailscale serve status
}

# ---------------------------------------------------------------------------
# tailscale-stop — remove tailscale serve config for :8080
# ---------------------------------------------------------------------------
cmd_tailscale_stop() {
    header "Tailscale Serve — Remove Dashboard"

    if ! command -v tailscale &>/dev/null; then
        error "tailscale CLI not found"
        exit 1
    fi

    tailscale serve --https=443 off 2>/dev/null && \
        success "Tailscale HTTPS serve removed" || \
        warn "Nothing to remove (or already gone)"

    echo ""
    tailscale serve status 2>/dev/null || echo "  (no serve config)"
}

# ---------------------------------------------------------------------------
# build — build Docker images
# ---------------------------------------------------------------------------
cmd_build() {
    header "Building Docker images"

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        warn ".env file not found — build proceeds but container will need credentials"
    fi

    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" build --no-cache
    success "Docker images built"
}

# ---------------------------------------------------------------------------
# up — start all Docker Compose services
# ---------------------------------------------------------------------------
cmd_up() {
    header "Starting Futures Trader + Dashboard"

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        error ".env file required. Copy from .env.example and fill in values."
        exit 1
    fi

    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" up -d
    success "All services started"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""

    if grep -qE "^(SIM_MODE=true|TRADING_MODE=sim)" "${PROJECT_DIR}/.env" 2>/dev/null; then
        echo -e "  ${YELLOW}Mode: SIMULATION (no real orders)${NC}"
    else
        echo -e "  ${GREEN}Mode: LIVE TRADING${NC}"
    fi

    echo -e "  ${BLUE}Timezone: America/New_York${NC}"
    echo ""
    echo -e "  ${CYAN}Dashboard:${NC} http://localhost:${WEB_PORT}"

    if _tailscale_available && _tailscale_serving_port "${WEB_PORT}"; then
        echo -e "  ${GREEN}Tailscale: https://${TAILSCALE_HOST}  (● live)${NC}"
    else
        echo -e "  ${DIM}Tailscale: run ./run.sh tailscale-serve to expose${NC}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# down — stop all Docker Compose services
# ---------------------------------------------------------------------------
cmd_down() {
    header "Stopping all services"
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" down
    success "All services stopped"
}

# ---------------------------------------------------------------------------
# logs — follow futures service logs
# ---------------------------------------------------------------------------
cmd_logs() {
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100 "$SERVICE_NAME"
}

# ---------------------------------------------------------------------------
# redis-logs — follow redis service logs
# ---------------------------------------------------------------------------
cmd_redis_logs() {
    cd "$PROJECT_DIR"
    docker compose -f "$COMPOSE_FILE" logs -f --tail=50 "$REDIS_SERVICE"
}

# ---------------------------------------------------------------------------
# status — show service + Tailscale status
# ---------------------------------------------------------------------------
cmd_status() {
    header "Service Status"
    cd "$PROJECT_DIR"

    # ── Docker services ──────────────────────────────────────────
    echo -e "${BOLD}  Docker Services${NC}"
    echo ""

    if docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -qE "(futures|redis|web)"; then
        docker compose -f "$COMPOSE_FILE" ps
        echo ""

        # Redis
        if docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "futures-redis.*healthy"; then
            echo -e "  ${GREEN}● Redis:   healthy${NC}"
        elif docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "futures-redis.*Up"; then
            echo -e "  ${YELLOW}● Redis:   running (health pending)${NC}"
        else
            echo -e "  ${RED}○ Redis:   not running${NC}"
        fi

        # Futures trader
        if docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "futures-trader.*Up"; then
            echo -e "  ${GREEN}● Trader:  running${NC}"
        else
            echo -e "  ${RED}○ Trader:  not running${NC}"
        fi

        # Web dashboard
        if docker compose -f "$COMPOSE_FILE" ps 2>/dev/null | grep -q "futures-web.*Up"; then
            echo -e "  ${GREEN}● Web:     running  →  http://localhost:${WEB_PORT}${NC}"
        else
            echo -e "  ${YELLOW}○ Web:     not running  (./run.sh web  to start locally)${NC}"
        fi
    else
        echo -e "  ${DIM}No Docker containers running${NC}"
        echo -e "  ${DIM}Use './run.sh up' or './run.sh web' (local)${NC}"
    fi

    echo ""

    # ── Tailscale ──────────────────────────────────────────────
    echo -e "${BOLD}  Tailscale${NC}"
    echo ""

    if ! command -v tailscale &>/dev/null; then
        echo -e "  ${DIM}tailscale CLI not found${NC}"
    elif ! tailscale status &>/dev/null 2>&1; then
        echo -e "  ${RED}○ Tailscale daemon not running${NC}"
    else
        # Node info
        local ts_name ts_ip
        ts_name="$(tailscale status --json 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('HostName','?'))" 2>/dev/null)" || ts_name="?"
        ts_ip="$(tailscale ip -4 2>/dev/null)" || ts_ip="?"

        echo -e "  ${GREEN}● Connected:${NC} ${BOLD}${ts_name}${NC}  ${DIM}(${ts_ip})${NC}"
        echo ""

        # Serve status
        local serve_raw
        serve_raw="$(tailscale serve status 2>/dev/null)" || serve_raw=""

        if [ -z "$serve_raw" ]; then
            echo -e "  ${YELLOW}  Serve:    nothing configured${NC}"
            echo -e "  ${DIM}  Run:      ./run.sh tailscale-serve${NC}"
        else
            # Check if our dashboard port is being served
            if echo "$serve_raw" | grep -q "proxy http://127.0.0.1:${WEB_PORT}"; then
                echo -e "  ${GREEN}  Serve:    ● https://${TAILSCALE_HOST}  →  :${WEB_PORT}${NC}"
                echo -e "  ${GREEN}  Also:     https://${TAILSCALE_IP}${NC}"
            else
                # HTTPS is active but not pointing at our port
                local current_target
                current_target="$(echo "$serve_raw" | grep 'proxy' | awk '{print $NF}' | head -1 || echo "unknown")"
                echo -e "  ${YELLOW}  Serve:    https://${TAILSCALE_HOST}  →  ${current_target}${NC}"
                echo -e "  ${YELLOW}  ⚠ Dashboard :${WEB_PORT} not exposed${NC}"
                echo -e "  ${DIM}  Run:      ./run.sh tailscale-serve  to fix${NC}"
            fi

            echo ""
            echo -e "  ${DIM}  Raw serve status:${NC}"
            echo "$serve_raw" | sed 's/^/    /'
        fi
    fi

    echo ""
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
    banner
    echo "Usage: ./run.sh [command]"
    echo ""
    echo -e "${BOLD}  Trading Bot${NC}"
    echo "  (no args)          Full pipeline: setup → lint → test → build → up"
    echo "  setup              Create .venv, install all dependencies"
    echo "  lint               Run ruff + mypy"
    echo "  test               Run pytest on src/tests/"
    echo "  sim                Start bot in simulation mode (local, no Docker)"
    echo "  live               Start bot in LIVE mode — real orders, confirmation required"
    echo "  build              Build Docker images"
    echo "  up                 Start all services (futures + redis + web)"
    echo "  down               Stop all services"
    echo "  logs               Follow futures container logs"
    echo "  redis-logs         Follow redis container logs"
    echo "  status             Show service + Tailscale status"
    echo "  clean              Remove .venv and Docker artifacts"
    echo ""
    echo -e "${BOLD}  Web Dashboard${NC}"
    echo "  web                Start dashboard locally (uvicorn, hot-reload)"
    echo "  web-up             Start dashboard Docker service"
    echo "  web-down           Stop dashboard Docker service"
    echo "  web-logs           Follow dashboard container logs"
    echo "  web-hash-password  Generate bcrypt hash for WEB_PASSWORD_HASH"
    echo ""
    echo -e "${BOLD}  Tailscale${NC}"
    echo "  tailscale-serve    Expose dashboard at https://${TAILSCALE_HOST}"
    echo "  tailscale-stop     Remove Tailscale HTTPS serve config"
    echo ""
    echo -e "${BOLD}  Environment (.env)${NC}"
    echo "  KUCOIN_API_KEY        KuCoin API key"
    echo "  KUCOIN_API_SECRET     KuCoin API secret"
    echo "  KUCOIN_PASSPHRASE     KuCoin passphrase"
    echo "  REDIS_PASSWORD        Redis auth password"
    echo "  TRADING_MODE          sim | live  (default: sim)"
    echo "  WEB_PORT              Dashboard port (default: 8080)"
    echo "  WEB_PASSWORD_HASH     bcrypt hash  (leave unset for no auth)"
    echo "  WEB_SESSION_SECRET    HMAC secret for session cookies"
    echo "  DISCORD_WEBHOOK_URL   Optional Discord alerts"
    echo ""
    echo -e "${BOLD}  Tailscale${NC}"
    echo "  Machine : desktop  (${TAILSCALE_IP})"
    echo "  Domain  : ${TAILSCALE_HOST}"
    echo "  Pi      : rasp.tailfef10.ts.net"
    echo ""
}

# ---------------------------------------------------------------------------
# Full pipeline (default)
# ---------------------------------------------------------------------------
cmd_full() {
    banner
    cmd_setup
    cmd_lint
    cmd_test
    cmd_build
    cmd_up
    echo ""
    header "Pipeline complete!"
    success "Futures Trader is live. Use './run.sh logs' to monitor."
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

case "${1:-}" in
    setup)                cmd_setup             ;;
    lint)                 cmd_lint              ;;
    test)                 cmd_test              ;;
    sim)                  cmd_sim               ;;
    live)                 cmd_live              ;;
    web)                  cmd_web               ;;
    web-up)               cmd_web_up            ;;
    web-down)             cmd_web_down          ;;
    web-logs)             cmd_web_logs          ;;
    web-hash-password)    cmd_web_hash_password ;;
    tailscale-serve)      cmd_tailscale_serve   ;;
    tailscale-stop)       cmd_tailscale_stop    ;;
    build)                cmd_build             ;;
    up)                   cmd_up                ;;
    down)                 cmd_down              ;;
    logs)                 cmd_logs              ;;
    redis-logs)           cmd_redis_logs        ;;
    status)               cmd_status            ;;
    clean)                cmd_clean             ;;
    help|-h|--help)       cmd_help              ;;
    "")                   cmd_full              ;;
    *)
        error "Unknown command: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
