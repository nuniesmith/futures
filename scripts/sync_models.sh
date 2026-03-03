#!/usr/bin/env bash
# =============================================================================
# sync_models.sh — Pull trained CNN models from the GPU training machine
# =============================================================================
#
# Copies the contents of models/ from the GPU training machine back to the
# local models/ directory so the main stack picks up the new champion on
# the next engine restart (or live reload if hot-swap is enabled).
#
# The trainer service (docker-compose.train.yml) uses bind-mounts, so the
# models/ directory is accessible directly on the host filesystem.
# This means rsync works without any docker cp / volume-export dance.
#
# Usage:
#   bash scripts/sync_models.sh <gpu-host>
#   bash scripts/sync_models.sh <gpu-host> [remote-project-path]
#   bash scripts/sync_models.sh --local [tailscale-ip]
#
# Examples:
#   # Sync from Oryx (default Tailscale IP — same machine or any Tailscale peer):
#   bash scripts/sync_models.sh --local
#   bash scripts/sync_models.sh --local 100.113.72.63
#
#   # Sync from a different GPU machine by SSH:
#   bash scripts/sync_models.sh 192.168.1.50
#   bash scripts/sync_models.sh user@192.168.1.50
#   bash scripts/sync_models.sh user@192.168.1.50 /home/user/futures
#   bash scripts/sync_models.sh user@gpu-rig.local ~/projects/futures
#
#   # Named-volume export (legacy, for older compose setups):
#   bash scripts/sync_models.sh --export-volume <gpu-host>
#
# Prerequisites:
#   - rsync installed locally (and on the GPU machine for SSH mode).
#   - SSH key-based auth to the GPU machine for SSH mode (no password prompts).
#   - The trainer must be running with bind-mounts (default in train compose).
#
# What is synced:
#   - breakout_cnn_best.pt          Champion model weights
#   - breakout_cnn_best_meta.json   Champion metadata / metrics
#   - retrain_audit.jsonl           Full audit trail of every training run
#   - archive/                      Previous champion checkpoints
#
# What is NOT synced (excluded):
#   - *.tmp  *.lock  .retrain_lock  — transient pipeline artefacts
#   - dataset/                      — chart images stay on the GPU machine
#
# Exit codes:
#   0  All files transferred successfully (or nothing to transfer).
#   1  Missing argument / bad usage.
#   2  rsync or SSH error.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[sync]${RESET} $*"; }
success() { echo -e "${GREEN}[sync]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[sync]${RESET} $*"; }
die()     { echo -e "${RED}[sync] ERROR:${RESET} $*" >&2; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# Resolve the project root (the directory that contains this scripts/ folder)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_MODELS_DIR="${PROJECT_ROOT}/models"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
EXPORT_VOLUME=0
LOCAL_MODE=0
# Default Tailscale IP for the Oryx GPU machine — used in --local mode.
DEFAULT_TAILSCALE_IP="100.113.72.63"
DEFAULT_TAILSCALE_USER="${USER:-jordan}"
GPU_HOST=""
REMOTE_PROJECT_PATH=""

usage() {
    cat <<EOF
Usage:
  bash scripts/sync_models.sh --local [tailscale-ip]
  bash scripts/sync_models.sh <gpu-host> [remote-project-path]
  bash scripts/sync_models.sh --export-volume <gpu-host> [remote-project-path]

Options:
  --local [ip]      Sync from the Oryx GPU machine over Tailscale.
                    Uses bind-mounted ./models/ on the remote host directly
                    (no docker cp needed — trainer uses bind-mounts).
                    Default IP: ${DEFAULT_TAILSCALE_IP}
                    Override: --local 100.x.x.x

  --export-volume   First export the Docker named volume on the remote machine
                    (futures_trainer_models) into a bind-mounted directory,
                    then rsync that directory back locally.
                    Use this only for legacy named-volume setups.

  <gpu-host>        SSH target for non-local mode, e.g. 192.168.1.50,
                    user@hostname, or an alias from ~/.ssh/config.

  [remote-project-path]
                    Absolute path to the futures/ project on the GPU machine.
                    Defaults to: ~/futures

Examples:
  # Most common — sync from Oryx over Tailscale (bind-mounted models/):
  bash scripts/sync_models.sh --local
  bash scripts/sync_models.sh --local 100.113.72.63

  # SSH to a different GPU machine:
  bash scripts/sync_models.sh user@gpu-rig ~/projects/futures

  # Legacy named-volume export:
  bash scripts/sync_models.sh --export-volume user@gpu-rig /srv/futures
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            LOCAL_MODE=1
            shift
            # Optional: next arg is the Tailscale IP override
            if [[ $# -gt 0 && "$1" != --* ]]; then
                DEFAULT_TAILSCALE_IP="$1"
                shift
            fi
            ;;
        --export-volume)
            EXPORT_VOLUME=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            if [[ -z "${GPU_HOST}" ]]; then
                GPU_HOST="$1"
            elif [[ -z "${REMOTE_PROJECT_PATH}" ]]; then
                REMOTE_PROJECT_PATH="$1"
            else
                die "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

# In --local mode, build the SSH target from the Tailscale IP
if [[ "${LOCAL_MODE}" -eq 1 ]]; then
    GPU_HOST="${DEFAULT_TAILSCALE_USER}@${DEFAULT_TAILSCALE_IP}"
    REMOTE_PROJECT_PATH="${REMOTE_PROJECT_PATH:-~/futures}"
    info "Local/Tailscale mode: syncing from ${GPU_HOST}:${REMOTE_PROJECT_PATH}/models/"
elif [[ -z "${GPU_HOST}" ]]; then
    usage
fi

# Default remote project path
REMOTE_PROJECT_PATH="${REMOTE_PROJECT_PATH:-~/futures}"

# ---------------------------------------------------------------------------
# Ensure local models/ directory exists
# ---------------------------------------------------------------------------
mkdir -p "${LOCAL_MODELS_DIR}"
mkdir -p "${LOCAL_MODELS_DIR}/archive"

# ---------------------------------------------------------------------------
# Step 1 (optional): export the Docker named volume on the GPU machine
# (Not needed in --local / bind-mount mode — the host dir is the volume.)
# ---------------------------------------------------------------------------
if [[ "${LOCAL_MODE}" -eq 1 && "${EXPORT_VOLUME}" -eq 1 ]]; then
    warn "--export-volume is a no-op in --local mode (bind-mounts are already on the host)"
elif [[ "${EXPORT_VOLUME}" -eq 1 ]]; then
    info "Exporting Docker named volume ${BOLD}futures_trainer_models${RESET} on ${GPU_HOST} ..."

    # The remote command:
    #   - Creates the target directory inside the project.
    #   - Runs a throwaway alpine container that bind-mounts the named volume
    #     as /src and the project models/ dir as /dst, then copies everything.
    REMOTE_EXPORT_CMD="
        set -e
        TARGET=\"${REMOTE_PROJECT_PATH}/models\"
        mkdir -p \"\${TARGET}\"
        docker run --rm \
            -v futures_trainer_models:/src:ro \
            -v \"\${TARGET}\":/dst \
            alpine \
            sh -c 'cp -r /src/. /dst/ && echo \"Volume exported to \${TARGET}\"'
    "

    ssh "${GPU_HOST}" "bash -s" <<< "${REMOTE_EXPORT_CMD}" \
        || die "Failed to export Docker volume on ${GPU_HOST}" 2

    success "Volume exported to ${REMOTE_PROJECT_PATH}/models/ on ${GPU_HOST}"
fi

# ---------------------------------------------------------------------------
# Step 2: rsync models/ from the GPU machine to local
#
# In --local / bind-mount mode the remote path IS the host bind-mount:
#   <gpu-host>:~/futures/models/
# No docker cp or volume export needed.
# ---------------------------------------------------------------------------
REMOTE_MODELS="${GPU_HOST}:${REMOTE_PROJECT_PATH}/models/"

info "Syncing models from ${BOLD}${REMOTE_MODELS}${RESET} → ${BOLD}${LOCAL_MODELS_DIR}/${RESET}"
info "Excluding transient artefacts (*.tmp, *.lock, .retrain_lock)"
echo ""

# rsync flags:
#   -a   archive mode (preserves perms, timestamps, symlinks, etc.)
#   -v   verbose (list transferred files)
#   -z   compress during transfer
#   --progress           show per-file progress
#   --checksum           compare by checksum, not just size+mtime
#   --delete             remove local files that no longer exist on the remote
#                        (keeps local models/ in sync with remote, not additive)
#   --exclude            skip transient pipeline artefacts
rsync \
    -avz \
    --progress \
    --checksum \
    --delete \
    --exclude='*.tmp' \
    --exclude='*.lock' \
    --exclude='.retrain_lock' \
    --exclude='__pycache__/' \
    "${REMOTE_MODELS}" \
    "${LOCAL_MODELS_DIR}/" \
    || die "rsync failed (exit $?)" 2

echo ""
success "Sync complete."

# ---------------------------------------------------------------------------
# Step 3: Show a summary of what landed locally
# ---------------------------------------------------------------------------
echo ""
info "Local models/ contents:"
echo ""

CHAMPION="${LOCAL_MODELS_DIR}/breakout_cnn_best.pt"
META="${LOCAL_MODELS_DIR}/breakout_cnn_best_meta.json"
AUDIT="${LOCAL_MODELS_DIR}/retrain_audit.jsonl"

if [[ -f "${CHAMPION}" ]]; then
    CHAMPION_SIZE=$(du -sh "${CHAMPION}" | cut -f1)
    CHAMPION_MTIME=$(date -r "${CHAMPION}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || stat -c '%y' "${CHAMPION}" 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✔${RESET} breakout_cnn_best.pt          ${BOLD}${CHAMPION_SIZE}${RESET}  (${CHAMPION_MTIME})"
else
    echo -e "  ${YELLOW}✘${RESET} breakout_cnn_best.pt          not found"
fi

if [[ -f "${META}" ]]; then
    # Pretty-print key metrics from the metadata JSON if python is available
    if command -v python3 &>/dev/null; then
        METRICS=$(python3 - <<'PYEOF'
import json, sys, os
p = os.path.join(os.environ.get("LOCAL_MODELS_DIR", "models"), "breakout_cnn_best_meta.json")
try:
    d = json.load(open(p))
    acc = d.get("val_accuracy") or d.get("accuracy")
    prec = d.get("val_precision") or d.get("precision")
    rec = d.get("val_recall") or d.get("recall")
    ep = d.get("epochs_trained") or d.get("epoch")
    parts = []
    if acc is not None:  parts.append(f"acc={float(acc):.1f}%")
    if prec is not None: parts.append(f"prec={float(prec):.1f}%")
    if rec is not None:  parts.append(f"rec={float(rec):.1f}%")
    if ep is not None:   parts.append(f"epochs={ep}")
    print("  ".join(parts) if parts else "")
except Exception:
    pass
PYEOF
        )
        if [[ -n "${METRICS}" ]]; then
            echo -e "  ${GREEN}✔${RESET} breakout_cnn_best_meta.json   ${METRICS}"
        else
            echo -e "  ${GREEN}✔${RESET} breakout_cnn_best_meta.json"
        fi
    else
        echo -e "  ${GREEN}✔${RESET} breakout_cnn_best_meta.json"
    fi
else
    echo -e "  ${YELLOW}✘${RESET} breakout_cnn_best_meta.json   not found"
fi

if [[ -f "${AUDIT}" ]]; then
    AUDIT_LINES=$(wc -l < "${AUDIT}")
    echo -e "  ${GREEN}✔${RESET} retrain_audit.jsonl           ${AUDIT_LINES} run(s) recorded"
fi

ARCHIVE_COUNT=$(find "${LOCAL_MODELS_DIR}/archive" -name '*.pt' 2>/dev/null | wc -l)
if [[ "${ARCHIVE_COUNT}" -gt 0 ]]; then
    echo -e "  ${GREEN}✔${RESET} archive/                      ${ARCHIVE_COUNT} checkpoint(s)"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 4: Remind the operator to restart the engine so it picks up the model
# ---------------------------------------------------------------------------
echo -e "${YELLOW}Next step:${RESET} restart the engine container on the main machine so it"
echo -e "loads the updated model:"
echo ""
echo -e "  ${BOLD}docker compose restart engine${RESET}"
echo ""
echo -e "Or, if the engine supports live model hot-swap, no restart is needed."
echo ""

export LOCAL_MODELS_DIR
