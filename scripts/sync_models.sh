#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# sync_models.sh — Pull latest CNN models from the orb GitHub repo
# =============================================================================
#
# Downloads the champion model files from:
#   https://github.com/nuniesmith/orb/tree/main/models
#
# Files synced:
#   breakout_cnn_best.pt          — PyTorch checkpoint (engine inference)
#   breakout_cnn_best.onnx        — ONNX export (NinjaTrader inference)
#   breakout_cnn_best_meta.json   — Model metadata (dashboard display)
#   feature_contract.json         — Feature/contract mapping
#
# Usage:
#   bash scripts/sync_models.sh              # download all model files
#   bash scripts/sync_models.sh --check      # check if models are current (no download)
#   bash scripts/sync_models.sh --pt-only    # download only the .pt file
#   bash scripts/sync_models.sh --restart    # download + restart engine container
#
# After syncing, restart the engine so it picks up the new model:
#   docker compose restart engine
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[sync]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ✓ ]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[fail]${NC} $*"; }

GITHUB_REPO="nuniesmith/orb"
GITHUB_BRANCH="main"
RAW_BASE="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/models"

# All model files to sync
ALL_FILES=(
    "breakout_cnn_best.pt"
    "breakout_cnn_best.onnx"
    "breakout_cnn_best_meta.json"
    "feature_contract.json"
)

# Lightweight files (always fetched for --check / metadata)
META_FILES=(
    "breakout_cnn_best_meta.json"
    "feature_contract.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: bash scripts/sync_models.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)     Download all model files from orb repo"
    echo "  --check       Check if local models are current (no download)"
    echo "  --pt-only     Download only the .pt checkpoint"
    echo "  --onnx-only   Download only the .onnx export"
    echo "  --restart     Download all + restart engine Docker container"
    echo "  --help        Show this help message"
}

ensure_model_dir() {
    mkdir -p "$MODEL_DIR"
}

download_file() {
    local filename="$1"
    local url="${RAW_BASE}/${filename}"
    local dest="${MODEL_DIR}/${filename}"
    local tmp="${dest}.tmp"

    log "Downloading ${filename}..."

    if curl -fSL --progress-bar -o "$tmp" "$url" 2>&1; then
        mv "$tmp" "$dest"
        local size
        size=$(du -h "$dest" 2>/dev/null | awk '{print $1}')
        ok "${filename} (${size})"
        return 0
    else
        rm -f "$tmp"
        err "Failed to download ${filename} from ${url}"
        return 1
    fi
}

check_file() {
    local filename="$1"
    local dest="${MODEL_DIR}/${filename}"

    if [ -f "$dest" ]; then
        local size modified
        size=$(du -h "$dest" 2>/dev/null | awk '{print $1}')
        modified=$(date -r "$dest" "+%Y-%m-%d %H:%M" 2>/dev/null || stat -c '%y' "$dest" 2>/dev/null | cut -d. -f1 || echo "unknown")
        ok "${filename}  ${DIM}(${size}, modified ${modified})${NC}"
        return 0
    else
        warn "${filename}  — not found locally"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_check() {
    log "Checking local model files in ${MODEL_DIR}/..."
    echo ""
    local missing=0
    for f in "${ALL_FILES[@]}"; do
        if ! check_file "$f"; then
            missing=$((missing + 1))
        fi
    done
    echo ""
    if [ "$missing" -gt 0 ]; then
        warn "${missing} file(s) missing — run: bash scripts/sync_models.sh"
        return 1
    else
        ok "All model files present"
        # Show meta info if available
        local meta="${MODEL_DIR}/breakout_cnn_best_meta.json"
        if [ -f "$meta" ] && command -v python3 >/dev/null 2>&1; then
            echo ""
            echo -e "${DIM}$(python3 -c "
import json, sys
try:
    d = json.load(open('$meta'))
    acc = d.get('val_accuracy', d.get('accuracy', '?'))
    prec = d.get('val_precision', d.get('precision', '?'))
    rec = d.get('val_recall', d.get('recall', '?'))
    print(f'  Champion: acc={acc}%  precision={prec}%  recall={rec}%')
except Exception:
    pass
" 2>/dev/null)${NC}"
        fi
        return 0
    fi
}

cmd_download() {
    local files=("$@")
    ensure_model_dir

    log "Pulling models from github.com/${GITHUB_REPO} (branch: ${GITHUB_BRANCH})..."
    echo ""

    local failed=0
    for f in "${files[@]}"; do
        if ! download_file "$f"; then
            failed=$((failed + 1))
        fi
    done

    echo ""
    if [ "$failed" -gt 0 ]; then
        err "${failed} file(s) failed to download"
        return 1
    else
        ok "All model files synced to ${MODEL_DIR}/"
        echo ""
        echo -e "  ${DIM}Restart the engine to pick up the new model:${NC}"
        echo "    docker compose restart engine"
        return 0
    fi
}

cmd_restart() {
    cmd_download "${ALL_FILES[@]}" || return 1
    echo ""
    log "Restarting engine container..."
    (cd "$PROJECT_ROOT" && docker compose restart engine)
    ok "Engine restarted with new model"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ACTION="download_all"
for arg in "$@"; do
    case "$arg" in
        --check)     ACTION="check" ;;
        --pt-only)   ACTION="pt_only" ;;
        --onnx-only) ACTION="onnx_only" ;;
        --restart)   ACTION="restart" ;;
        --help|-h)   usage; exit 0 ;;
        *)
            err "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

case "$ACTION" in
    check)
        cmd_check
        ;;
    download_all)
        cmd_download "${ALL_FILES[@]}"
        ;;
    pt_only)
        cmd_download "breakout_cnn_best.pt" "${META_FILES[@]}"
        ;;
    onnx_only)
        cmd_download "breakout_cnn_best.onnx" "${META_FILES[@]}"
        ;;
    restart)
        cmd_restart
        ;;
esac
