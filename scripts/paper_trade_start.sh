#!/usr/bin/env bash
# =============================================================================
# Paper Trade Quick-Start — Monday Morning Checklist
# =============================================================================
#
# Usage:
#   cd futures
#   bash scripts/paper_trade_start.sh          # full check + start
#   bash scripts/paper_trade_start.sh --check  # check only, don't start
#   bash scripts/paper_trade_start.sh --stop   # stop everything
#
# Prerequisites:
#   - Docker + Docker Compose installed
#   - .env file configured (copy from .env.example)
#   - NinjaTrader 8 running with Bridge strategy on Sim101
#   - Git LFS migration done (optional but recommended)
#
# What this script does:
#   1. Validates environment (.env, Docker, model files)
#   2. Checks NinjaTrader Bridge connectivity
#   3. Builds and starts the Docker stack
#   4. Waits for all services to be healthy
#   5. Runs a quick smoke test
#   6. Tails engine logs filtered to ORB activity
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colours (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' DIM='' RESET=''
fi

# Counters
PASS=0
WARN=0
FAIL=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ok()   { ((PASS++)); echo -e "  ${GREEN}✓${RESET} $1"; }
warn() { ((WARN++)); echo -e "  ${YELLOW}⚠${RESET} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}✗${RESET} $1"; }
info() { echo -e "  ${CYAN}ℹ${RESET} $1"; }
header() { echo -e "\n${BOLD}── $1 ──${RESET}"; }

# ---------------------------------------------------------------------------
# Stop mode
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    echo -e "${BOLD}Stopping paper trade stack...${RESET}"
    docker compose down 2>/dev/null || true
    echo -e "${GREEN}Stack stopped.${RESET}"
    exit 0
fi

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# =============================================================================
# PHASE 1: Environment Checks
# =============================================================================
echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  📊  Paper Trade Quick-Start — Pre-Flight Checks${RESET}"
echo -e "${BOLD}================================================================${RESET}"

# ---------------------------------------------------------------------------
header "1. Environment & Configuration"
# ---------------------------------------------------------------------------

# .env file
if [ -f ".env" ]; then
    ok ".env file exists"

    # Check required vars
    if grep -q "POSTGRES_PASSWORD" .env 2>/dev/null; then
        ok "POSTGRES_PASSWORD is set"
    else
        fail "POSTGRES_PASSWORD not found in .env (required by docker-compose)"
    fi

    if grep -q "REDIS_PASSWORD" .env 2>/dev/null; then
        ok "REDIS_PASSWORD is set"
    else
        fail "REDIS_PASSWORD not found in .env (required by docker-compose)"
    fi

    # Check ORB_CNN_GATE is 0 (advisory mode)
    CNN_GATE=$(grep -oP 'ORB_CNN_GATE=\K.*' .env 2>/dev/null || echo "0")
    if [ "$CNN_GATE" = "0" ] || [ -z "$CNN_GATE" ]; then
        ok "ORB_CNN_GATE=0 (advisory mode — correct for paper trading)"
    else
        warn "ORB_CNN_GATE=$CNN_GATE — set to 0 for first week of paper trading"
    fi

    # Check ORB_FILTER_GATE
    FILTER_GATE=$(grep -oP 'ORB_FILTER_GATE=\K.*' .env 2>/dev/null || echo "majority")
    ok "ORB_FILTER_GATE=$FILTER_GATE"

else
    fail ".env file not found — copy from .env.example and configure"
    info "Run: cp .env.example .env && nano .env"
fi

# Docker
if command -v docker &>/dev/null; then
    ok "Docker is installed ($(docker --version | head -1))"
else
    fail "Docker is not installed"
fi

if docker compose version &>/dev/null 2>&1; then
    ok "Docker Compose is available"
else
    fail "Docker Compose is not available"
fi

# Check Docker daemon is running
if docker info &>/dev/null 2>&1; then
    ok "Docker daemon is running"
else
    fail "Docker daemon is not running — start Docker Desktop"
fi

# ---------------------------------------------------------------------------
header "2. Model & Dataset Files"
# ---------------------------------------------------------------------------

# CNN model
if [ -f "models/breakout_cnn_best.pt" ]; then
    MODEL_SIZE=$(du -h "models/breakout_cnn_best.pt" | cut -f1)
    MODEL_DATE=$(date -r "models/breakout_cnn_best.pt" "+%Y-%m-%d %H:%M" 2>/dev/null || echo "unknown")
    ok "CNN model: breakout_cnn_best.pt ($MODEL_SIZE, modified $MODEL_DATE)"
else
    warn "No CNN model found at models/breakout_cnn_best.pt"
    info "CNN inference will be skipped — ORB detection + filters still work"
    info "Model will be created by overnight retraining if dataset exists"
fi

# Dataset
if [ -f "dataset/labels.csv" ]; then
    LABEL_COUNT=$(tail -n +2 "dataset/labels.csv" | wc -l | tr -d ' ')
    ok "Dataset: labels.csv ($LABEL_COUNT samples)"
else
    warn "No dataset/labels.csv found"
    info "Dataset will be generated during off-hours by the engine"
fi

if [ -d "dataset/images" ]; then
    IMG_COUNT=$(find "dataset/images" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    ok "Dataset images: $IMG_COUNT PNG files"
else
    warn "No dataset/images/ directory"
fi

# Training history
if [ -f "models/training_history.csv" ]; then
    LAST_LINE=$(tail -1 "models/training_history.csv")
    ok "Training history exists (last entry: ${LAST_LINE:0:60}...)"
fi

# ---------------------------------------------------------------------------
header "3. NinjaTrader Bridge Connectivity"
# ---------------------------------------------------------------------------

# Check if NinjaTrader Bridge HTTP listener is running on port 8080
NT_PORT=8080
if curl -s --connect-timeout 2 "http://localhost:$NT_PORT/status" &>/dev/null; then
    ok "NinjaTrader Bridge listener responding on port $NT_PORT"
    # Try to get status JSON
    STATUS=$(curl -s --connect-timeout 2 "http://localhost:$NT_PORT/status" 2>/dev/null || echo "")
    if [ -n "$STATUS" ]; then
        # Extract account name if possible
        ACCOUNT=$(echo "$STATUS" | grep -oP '"account"\s*:\s*"\K[^"]+' 2>/dev/null || echo "unknown")
        POSITION_COUNT=$(echo "$STATUS" | grep -oP '"positionCount"\s*:\s*\K[0-9]+' 2>/dev/null || echo "?")
        info "Account: $ACCOUNT, Positions: $POSITION_COUNT"
    fi
else
    warn "NinjaTrader Bridge not responding on port $NT_PORT"
    info "Start NinjaTrader 8, add Bridge strategy to a chart on Sim101"
    info "Bridge listens on http://localhost:$NT_PORT by default"
    info "The engine will still run — it just won't push orders to NT"
fi

# Check if port 8000 (data service) is already in use
if curl -s --connect-timeout 2 "http://localhost:8000/health" &>/dev/null; then
    warn "Port 8000 already in use — data service may already be running"
    info "Run: docker compose down  (to stop existing stack)"
else
    ok "Port 8000 is available for the data service"
fi

# ---------------------------------------------------------------------------
header "4. Docker Images & Build"
# ---------------------------------------------------------------------------

# Check if images need rebuilding
ENGINE_DOCKERFILE="docker/engine/Dockerfile"
if [ -f "$ENGINE_DOCKERFILE" ]; then
    # Check if the Dockerfile has the gpu extras fix
    if grep -q '\.\[gpu\]' "$ENGINE_DOCKERFILE" 2>/dev/null; then
        ok "Engine Dockerfile includes GPU extras (.[gpu])"
    else
        warn "Engine Dockerfile may not include mplfinance/Pillow"
        info "Update: pip install --no-cache-dir \".[gpu]\" in $ENGINE_DOCKERFILE"
    fi
fi

# Check if scripts/ is still in .dockerignore
if [ -f ".dockerignore" ]; then
    if grep -q "^scripts/" ".dockerignore" 2>/dev/null; then
        warn "scripts/ is excluded in .dockerignore — daily_report won't be in container"
        info "Remove the 'scripts/' line from .dockerignore"
    else
        ok "scripts/ not excluded from Docker build context"
    fi
fi

# ---------------------------------------------------------------------------
header "5. Git LFS Status"
# ---------------------------------------------------------------------------

if command -v git-lfs &>/dev/null || command -v git lfs &>/dev/null; then
    ok "Git LFS is installed"
    if [ -f ".gitattributes" ]; then
        if grep -q "*.pt" ".gitattributes" 2>/dev/null; then
            ok ".gitattributes tracks *.pt with LFS"
        else
            warn ".gitattributes does not track *.pt — run scripts/migrate_git_lfs.sh"
        fi
    else
        warn "No .gitattributes file"
    fi
else
    info "Git LFS not installed — not required for paper trading"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "  ${GREEN}Passed: $PASS${RESET}  ${YELLOW}Warnings: $WARN${RESET}  ${RED}Failed: $FAIL${RESET}"
echo -e "${BOLD}================================================================${RESET}"

if [ $FAIL -gt 0 ]; then
    echo -e "\n${RED}${BOLD}  ✗ Fix the failures above before starting.${RESET}"
    echo ""
    exit 1
fi

if [ "$CHECK_ONLY" = true ]; then
    if [ $WARN -gt 0 ]; then
        echo -e "\n${YELLOW}  Warnings exist but are non-blocking. Ready to start.${RESET}"
    else
        echo -e "\n${GREEN}  All checks passed. Ready to start.${RESET}"
    fi
    echo ""
    exit 0
fi

# =============================================================================
# PHASE 2: Build & Launch
# =============================================================================
echo ""
echo -e "${BOLD}── Building and starting Docker stack... ──${RESET}"
echo ""

# Build (with progress)
docker compose build --progress=plain 2>&1 | tail -20
echo ""

# Start detached
docker compose up -d
echo ""

# ---------------------------------------------------------------------------
header "Waiting for services to be healthy..."
# ---------------------------------------------------------------------------

MAX_WAIT=120
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check Postgres
    PG_HEALTHY=$(docker compose ps --format json 2>/dev/null | grep -c '"healthy"' 2>/dev/null || echo "0")

    # Check if data service responds
    if curl -s --connect-timeout 2 "http://localhost:8000/health" &>/dev/null; then
        ok "Data service is healthy (http://localhost:8000)"
        break
    fi

    echo -e "  ${DIM}Waiting... (${ELAPSED}s / ${MAX_WAIT}s)${RESET}"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    warn "Timed out waiting for services (${MAX_WAIT}s)"
    info "Check logs: docker compose logs --tail=50"
else
    echo ""
fi

# ---------------------------------------------------------------------------
header "Service Status"
# ---------------------------------------------------------------------------

docker compose ps
echo ""

# ---------------------------------------------------------------------------
header "Quick Smoke Test"
# ---------------------------------------------------------------------------

# Health endpoint
HEALTH=$(curl -s --connect-timeout 5 "http://localhost:8000/health" 2>/dev/null || echo "")
if [ -n "$HEALTH" ]; then
    ok "GET /health → OK"
else
    warn "GET /health → no response yet"
fi

# Audit endpoint
AUDIT=$(curl -s --connect-timeout 5 "http://localhost:8000/audit/summary?days=1" 2>/dev/null || echo "")
if [ -n "$AUDIT" ]; then
    ok "GET /audit/summary → OK"
else
    warn "GET /audit/summary → no response"
fi

# Daily report endpoint
REPORT=$(curl -s --connect-timeout 5 "http://localhost:8000/audit/daily-report" 2>/dev/null || echo "")
if [ -n "$REPORT" ]; then
    ok "GET /audit/daily-report → OK"
else
    warn "GET /audit/daily-report → no response"
fi

# =============================================================================
# Ready!
# =============================================================================
echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}  🚀  Paper Trade Stack is Running!${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "  ${CYAN}Dashboard:${RESET}        http://localhost:8000"
echo -e "  ${CYAN}Health:${RESET}           http://localhost:8000/health"
echo -e "  ${CYAN}ORB Audit:${RESET}        http://localhost:8000/audit/orb"
echo -e "  ${CYAN}Daily Report:${RESET}     http://localhost:8000/audit/daily-report"
echo -e "  ${CYAN}Risk Events:${RESET}      http://localhost:8000/audit/risk"
echo -e "  ${CYAN}Audit Summary:${RESET}    http://localhost:8000/audit/summary?days=7"
echo ""
echo -e "  ${BOLD}Key Environment:${RESET}"
echo -e "    ORB_CNN_GATE=0     ${DIM}(advisory — CNN scores shown but don't block)${RESET}"
echo -e "    ORB_FILTER_GATE=${FILTER_GATE:-majority}  ${DIM}(filter mode for breakout quality)${RESET}"
echo ""
echo -e "  ${BOLD}What to Watch (09:30–11:00 ET):${RESET}"
echo -e "    docker compose logs -f engine 2>&1 | grep -E 'ORB|CNN|FILTER|BREAKOUT|orb'"
echo ""
echo -e "  ${BOLD}End-of-Day Report:${RESET}"
echo -e "    docker compose exec engine python scripts/daily_report.py"
echo -e "    ${DIM}# or from host:${RESET}"
echo -e "    PYTHONPATH=src python scripts/daily_report.py"
echo ""
echo -e "  ${BOLD}Multi-Day Summary:${RESET}"
echo -e "    docker compose exec engine python scripts/daily_report.py --days 5"
echo ""
echo -e "  ${BOLD}JSON Export:${RESET}"
echo -e "    curl -s http://localhost:8000/audit/daily-report?days=5 | python -m json.tool"
echo ""
echo -e "  ${BOLD}Stop:${RESET}"
echo -e "    docker compose down"
echo -e "    ${DIM}# or: bash scripts/paper_trade_start.sh --stop${RESET}"
echo ""
echo -e "${BOLD}================================================================${RESET}"
echo -e "  ${GREEN}Good luck trading! 📈${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""

# Optionally tail ORB logs
echo -e "${DIM}Tailing engine ORB logs (Ctrl+C to detach)...${RESET}"
echo ""
docker compose logs -f engine 2>&1 | grep --line-buffered -iE 'ORB|CNN|FILTER|BREAKOUT|breakout|orb_event|check_orb|SIGNAL' || true
