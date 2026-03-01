#!/usr/bin/env bash
# deploy_nt8.sh — Copy NinjaTrader 8 source files to the correct NT8 Custom directories.
#
# Usage:
#   ./scripts/deploy_nt8.sh          # default paths
#   NT8_CUSTOM="/some/path" ./scripts/deploy_nt8.sh   # override target
#
# Layout:
#   Bridge.cs    → Strategies  (strategy)
#   Ruby.cs      → Indicators  (indicator)
#   SignalBus.cs → root level  (shared bus)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src/ninjatrader"

# NT8 Custom directory (WSL path)
NT8_CUSTOM="${NT8_CUSTOM:-/mnt/c/Users/jordan/Documents/NinjaTrader 8/bin/Custom}"

# ── Colour helpers ─────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

ok()   { printf "${GREEN}  ✓ %s${NC}\n" "$*"; }
warn() { printf "${YELLOW}  ⚠ %s${NC}\n" "$*"; }
err()  { printf "${RED}  ✗ %s${NC}\n" "$*"; }

# ── Validation ─────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════╗"
echo "║   NinjaTrader 8 — Deploy CS Files            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Source:  $SRC_DIR"
echo "Target:  $NT8_CUSTOM"
echo ""

if [ ! -d "$NT8_CUSTOM" ]; then
    err "NT8 Custom directory not found: $NT8_CUSTOM"
    echo "    Make sure NinjaTrader 8 is installed and the path is correct."
    echo "    You can override it with: NT8_CUSTOM=\"/your/path\" $0"
    exit 1
fi

# Ensure target subdirectories exist
mkdir -p "$NT8_CUSTOM/Strategies"
mkdir -p "$NT8_CUSTOM/Indicators"

ERRORS=0

# ── Copy files ─────────────────────────────────────────────────────────────────

# Strategy: Bridge.cs → Strategies/
if [ -f "$SRC_DIR/Bridge.cs" ]; then
    cp "$SRC_DIR/Bridge.cs" "$NT8_CUSTOM/Strategies/Bridge.cs"
    ok "Bridge.cs      → Strategies/"
else
    err "Bridge.cs not found in $SRC_DIR"
    ERRORS=$((ERRORS + 1))
fi

# Indicator: Ruby.cs → Indicators/
if [ -f "$SRC_DIR/Ruby.cs" ]; then
    cp "$SRC_DIR/Ruby.cs" "$NT8_CUSTOM/Indicators/Ruby.cs"
    ok "Ruby.cs        → Indicators/"
else
    err "Ruby.cs not found in $SRC_DIR"
    ERRORS=$((ERRORS + 1))
fi

# Shared bus: SignalBus.cs → root (bin/Custom/)
if [ -f "$SRC_DIR/SignalBus.cs" ]; then
    cp "$SRC_DIR/SignalBus.cs" "$NT8_CUSTOM/SignalBus.cs"
    ok "SignalBus.cs   → Custom/ (root)"
else
    err "SignalBus.cs not found in $SRC_DIR"
    ERRORS=$((ERRORS + 1))
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}All files deployed successfully.${NC}"
    echo ""
    echo "Next step: open NinjaTrader 8 → Tools → NinjaScript Editor → right-click → Compile"
else
    echo -e "${RED}Completed with $ERRORS error(s).${NC}"
    exit 1
fi
