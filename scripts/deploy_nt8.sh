#!/usr/bin/env bash
# deploy_nt8.sh — Copy NinjaTrader 8 source files to the correct NT8 Custom directories.
#
# Usage:
#   ./scripts/deploy_nt8.sh                              # deploy all files
#   ./scripts/deploy_nt8.sh --dry-run                   # show what would be copied, do nothing
#   NT8_CUSTOM="/some/path" ./scripts/deploy_nt8.sh     # override target directory
#
# File → target mapping (mirrors NT8 namespace conventions):
#
#   RubyIndicator.cs      → Indicators/          (NinjaScript.Indicators)
#   BreakoutStrategy.cs   → Strategies/          (NinjaScript.Strategies)
#   MonitorConnection.cs  → Strategies/          (NinjaScript.Strategies)
#   BridgeOrderEngine.cs  → Strategies/          (NinjaScript.Strategies)
#   OrbCnnPredictor.cs    → Custom/ (root)       (NinjaScript — shared types)
#   SignalBus.cs          → Custom/ (root)       (NinjaScript — shared bus)
#
# NT8 compiles all files under bin/Custom/ into one assembly, so shared types
# (OrbCnnPredictor, SignalBus) must live at the root — not inside a subdirectory.

set -euo pipefail

# ── Argument parsing ───────────────────────────────────────────────────────────

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=true ;;
        *)
            printf '\033[0;31m  ✗ Unknown argument: %s\033[0m\n' "$arg"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/src/ninjatrader"

# NT8 Custom directory (WSL path — override with NT8_CUSTOM env var if needed)
NT8_CUSTOM="${NT8_CUSTOM:-/mnt/c/Users/jordan/Documents/NinjaTrader 8/bin/Custom}"

# ── Colour helpers ─────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { printf "${GREEN}  ✓ %s${NC}\n" "$*"; }
warn() { printf "${YELLOW}  ⚠ %s${NC}\n" "$*"; }
err()  { printf "${RED}  ✗ %s${NC}\n" "$*"; }
dry()  { printf "${CYAN}  ~ %s${NC}\n" "$*"; }

# ── Banner ─────────────────────────────────────────────────────────────────────

echo ""
printf "${BOLD}╔══════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}║   NinjaTrader 8 — Deploy NinjaScript Files       ║${NC}\n"
printf "${BOLD}╚══════════════════════════════════════════════════╝${NC}\n"
echo ""
echo "  Source : $SRC_DIR"
echo "  Target : $NT8_CUSTOM"
if $DRY_RUN; then
    printf "  ${CYAN}Mode   : DRY RUN — no files will be written${NC}\n"
fi
echo ""

# ── Validation ─────────────────────────────────────────────────────────────────

if [ ! -d "$SRC_DIR" ]; then
    err "Source directory not found: $SRC_DIR"
    exit 1
fi

if [ ! -d "$NT8_CUSTOM" ]; then
    err "NT8 Custom directory not found: $NT8_CUSTOM"
    echo ""
    echo "    Ensure NinjaTrader 8 is installed and the path is correct."
    echo "    Override with:  NT8_CUSTOM=\"/your/path\" $0"
    exit 1
fi

# ── Ensure subdirectories exist ────────────────────────────────────────────────

if ! $DRY_RUN; then
    mkdir -p "$NT8_CUSTOM/Strategies"
    mkdir -p "$NT8_CUSTOM/Indicators"
fi

# ── File map ───────────────────────────────────────────────────────────────────
# Format: "source_filename:relative_target_path"
# Targets are relative to $NT8_CUSTOM.

declare -a FILE_MAP=(
    # ── Shared types (NinjaScript root namespace) — must be at Custom/ root ──
    "SignalBus.cs:SignalBus.cs"
    "OrbCnnPredictor.cs:OrbCnnPredictor.cs"

    # ── Strategies (NinjaScript.Strategies namespace) ─────────────────────────
    "BridgeOrderEngine.cs:Strategies/BridgeOrderEngine.cs"
    "MonitorConnection.cs:Strategies/MonitorConnection.cs"
    "BreakoutStrategy.cs:Strategies/BreakoutStrategy.cs"

    # ── Indicators (NinjaScript.Indicators namespace) ─────────────────────────
    "RubyIndicator.cs:Indicators/RubyIndicator.cs"
)

# ── Copy loop ──────────────────────────────────────────────────────────────────

ERRORS=0
DEPLOYED=0
SKIPPED=0

for entry in "${FILE_MAP[@]}"; do
    src_name="${entry%%:*}"
    dst_rel="${entry##*:}"

    src_path="$SRC_DIR/$src_name"
    dst_path="$NT8_CUSTOM/$dst_rel"

    if [ ! -f "$src_path" ]; then
        err "$src_name  — source not found: $src_path"
        ERRORS=$((ERRORS + 1))
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Pad source name for aligned output (28 chars)
    label="$(printf '%-28s' "$src_name")"

    if $DRY_RUN; then
        dry "$label → $dst_rel"
    else
        cp "$src_path" "$dst_path"
        ok "$label → $dst_rel"
        DEPLOYED=$((DEPLOYED + 1))
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""

if $DRY_RUN; then
    printf "${CYAN}  Dry run complete — ${#FILE_MAP[@]} file(s) would be deployed.${NC}\n"
    echo "  Run without --dry-run to apply."
    echo ""
    exit 0
fi

if [ "$ERRORS" -eq 0 ]; then
    printf "${GREEN}${BOLD}  ✓ All ${DEPLOYED} file(s) deployed successfully.${NC}\n"
    echo ""
    echo "  Next steps:"
    echo "    1. Open NinjaTrader 8"
    echo "    2. Tools → NinjaScript Editor"
    echo "    3. Right-click the project → Compile"
    echo "    4. Confirm 0 errors in the output pane"
else
    printf "${RED}  Completed with ${ERRORS} error(s). ${DEPLOYED} file(s) deployed, ${SKIPPED} skipped.${NC}\n"
    echo ""
    exit 1
fi
