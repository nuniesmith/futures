#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# verify_cicd.sh — Verify CI/CD secrets & deployment readiness
# =============================================================================
#
# Run this script on each machine to confirm the CI/CD pipeline can deploy
# to it successfully. Checks SSH access, Docker, Tailscale, secrets alignment,
# and model sync readiness.
#
# Usage:
#   bash scripts/verify_cicd.sh              # auto-detect role from hostname/IP
#   bash scripts/verify_cicd.sh --server     # force check as Ubuntu Server (prod)
#   bash scripts/verify_cicd.sh --trainer    # force check as GPU trainer rig
#   bash scripts/verify_cicd.sh --both       # check both roles (run from either)
#   bash scripts/verify_cicd.sh --secrets    # just print what GitHub secrets should be
#
# Expected GitHub Secrets:
#   PROD_TAILSCALE_IP       — Ubuntu Server Tailscale IP (100.122.184.58)
#   TRAINER_TAILSCALE_IP    — Home GPU rig Tailscale IP (100.113.72.63)
#   PROD_SSH_PORT           — SSH port (default: 22)
#   PROD_SSH_USER           — SSH user for deployments (default: actions)
#   PROD_SSH_KEY            — SSH private key for Ubuntu Server
#   TRAINER_SSH_KEY         — SSH private key for GPU rig
#   DOCKER_USERNAME         — Docker Hub username (nuniesmith)
#   DOCKER_TOKEN            — Docker Hub access token
#   TAILSCALE_OAUTH_CLIENT_ID — Tailscale OAuth client ID
#   TAILSCALE_OAUTH_SECRET    — Tailscale OAuth secret
#   DISCORD_WEBHOOK_ACTIONS   — Discord webhook for CI/CD notifications
#   API_KEY                   — Shared API key for data service auth
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Colors & helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()   { PASS=$((PASS + 1)); echo -e "  ${GREEN}✓${NC} $*"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}✗${NC} $*"; }
warn() { WARN=$((WARN + 1)); echo -e "  ${YELLOW}⚠${NC} $*"; }
info() { echo -e "  ${CYAN}ℹ${NC} $*"; }
header() { echo ""; echo -e "${BOLD}${CYAN}━━━ $* ━━━${NC}"; }

# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------

EXPECTED_PROD_IP="100.122.184.58"
EXPECTED_TRAINER_IP="100.113.72.63"
EXPECTED_SSH_USER="actions"
EXPECTED_PROJECT_PATH="$HOME/futures"
EXPECTED_DOCKER_IMAGE="nuniesmith/futures"

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

detect_role() {
    local ts_ip=""
    if command -v tailscale >/dev/null 2>&1; then
        ts_ip=$(tailscale ip -4 2>/dev/null || true)
    fi

    if [ "$ts_ip" = "$EXPECTED_PROD_IP" ]; then
        echo "server"
    elif [ "$ts_ip" = "$EXPECTED_TRAINER_IP" ]; then
        echo "trainer"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "trainer"
    else
        echo "server"
    fi
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

check_tailscale() {
    header "Tailscale"

    if ! command -v tailscale >/dev/null 2>&1; then
        fail "tailscale CLI not found — install: curl -fsSL https://tailscale.com/install.sh | sh"
        return
    fi
    ok "tailscale CLI installed"

    local status
    status=$(tailscale status --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('BackendState',''))" 2>/dev/null || echo "unknown")
    if [ "$status" = "Running" ]; then
        ok "Tailscale is running"
    else
        fail "Tailscale not running (state: ${status}) — run: sudo tailscale up"
        return
    fi

    local my_ip
    my_ip=$(tailscale ip -4 2>/dev/null || echo "unknown")
    if [ "$my_ip" = "unknown" ]; then
        fail "Cannot determine Tailscale IPv4 address"
    else
        ok "Tailscale IP: ${my_ip}"

        if [ "$ROLE" = "server" ] && [ "$my_ip" != "$EXPECTED_PROD_IP" ]; then
            warn "Expected PROD_TAILSCALE_IP=${EXPECTED_PROD_IP}, got ${my_ip}"
            warn "Update GitHub secret PROD_TAILSCALE_IP → ${my_ip}"
        elif [ "$ROLE" = "server" ] && [ "$my_ip" = "$EXPECTED_PROD_IP" ]; then
            ok "Matches expected PROD_TAILSCALE_IP (${EXPECTED_PROD_IP})"
        fi

        if [ "$ROLE" = "trainer" ] && [ "$my_ip" != "$EXPECTED_TRAINER_IP" ]; then
            warn "Expected TRAINER_TAILSCALE_IP=${EXPECTED_TRAINER_IP}, got ${my_ip}"
            warn "Update GitHub secret TRAINER_TAILSCALE_IP → ${my_ip}"
        elif [ "$ROLE" = "trainer" ] && [ "$my_ip" = "$EXPECTED_TRAINER_IP" ]; then
            ok "Matches expected TRAINER_TAILSCALE_IP (${EXPECTED_TRAINER_IP})"
        fi
    fi

    # Check if we can reach the other machine
    local other_ip=""
    local other_name=""
    if [ "$ROLE" = "server" ]; then
        other_ip="$EXPECTED_TRAINER_IP"
        other_name="Trainer (GPU rig)"
    else
        other_ip="$EXPECTED_PROD_IP"
        other_name="Server (Ubuntu)"
    fi

    if tailscale ping --timeout=3s "$other_ip" >/dev/null 2>&1; then
        ok "Can reach ${other_name} at ${other_ip} via Tailscale"
    else
        warn "Cannot reach ${other_name} at ${other_ip} — may be offline or IP changed"
    fi
}

check_ssh() {
    header "SSH Access"

    # Check if the expected deploy user exists
    if id "$EXPECTED_SSH_USER" >/dev/null 2>&1; then
        ok "User '${EXPECTED_SSH_USER}' exists"
    else
        fail "User '${EXPECTED_SSH_USER}' does not exist"
        info "Create it: sudo useradd -m -s /bin/bash ${EXPECTED_SSH_USER}"
        info "Then add SSH authorized_keys for CI/CD"
        return
    fi

    # Check SSH authorized_keys
    local auth_keys="/home/${EXPECTED_SSH_USER}/.ssh/authorized_keys"
    if [ -f "$auth_keys" ]; then
        local key_count
        key_count=$(grep -c "^ssh-" "$auth_keys" 2>/dev/null || echo "0")
        if [ "$key_count" -gt 0 ]; then
            ok "authorized_keys has ${key_count} key(s)"
        else
            fail "authorized_keys exists but has no keys"
        fi
    else
        fail "No authorized_keys file at ${auth_keys}"
        info "Add the CI/CD public key: ssh-copy-id or manual edit"
    fi

    # Check SSH config allows the user
    local sshd_port
    sshd_port=$(grep -E "^Port " /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "22")
    if [ -z "$sshd_port" ]; then
        sshd_port="22"
    fi
    ok "SSH port: ${sshd_port}"
    if [ "$sshd_port" != "22" ]; then
        info "Non-default SSH port — ensure PROD_SSH_PORT GitHub secret = ${sshd_port}"
    fi

    # Check if user can run docker without sudo
    if sudo -u "$EXPECTED_SSH_USER" docker ps >/dev/null 2>&1; then
        ok "User '${EXPECTED_SSH_USER}' can run Docker"
    elif groups "$EXPECTED_SSH_USER" 2>/dev/null | grep -q docker; then
        ok "User '${EXPECTED_SSH_USER}' is in docker group"
    else
        fail "User '${EXPECTED_SSH_USER}' cannot run Docker"
        info "Fix: sudo usermod -aG docker ${EXPECTED_SSH_USER}"
    fi
}

check_docker() {
    header "Docker"

    if ! command -v docker >/dev/null 2>&1; then
        fail "docker CLI not found"
        return
    fi
    ok "docker CLI installed ($(docker --version 2>/dev/null | head -1))"

    if ! command -v docker compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        fail "docker compose not available"
    else
        ok "docker compose available"
    fi

    if docker info >/dev/null 2>&1; then
        ok "Docker daemon is running"
    else
        fail "Docker daemon not running or current user lacks permissions"
        return
    fi

    # Check if we can pull from Docker Hub
    if docker manifest inspect "${EXPECTED_DOCKER_IMAGE}:data" >/dev/null 2>&1; then
        ok "Can access Docker Hub image ${EXPECTED_DOCKER_IMAGE}:data"
    else
        warn "Cannot verify Docker Hub image — may need DOCKER_USERNAME/DOCKER_TOKEN"
    fi

    # Check running containers
    local running
    running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "futures" || echo "0")
    if [ "$running" -gt 0 ]; then
        ok "${running} futures container(s) running"
        docker ps --format '    {{.Names}}: {{.Status}}' 2>/dev/null | grep "futures" || true
    else
        info "No futures containers currently running"
    fi

    # GPU check for trainer
    if [ "$ROLE" = "trainer" ]; then
        echo ""
        info "Checking GPU / NVIDIA runtime..."
        if command -v nvidia-smi >/dev/null 2>&1; then
            ok "nvidia-smi available"
            local gpu_name
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
            local gpu_mem
            gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
            ok "GPU: ${gpu_name} (${gpu_mem})"
        else
            fail "nvidia-smi not found — GPU training won't work"
        fi

        if docker info 2>/dev/null | grep -qi "nvidia"; then
            ok "NVIDIA Docker runtime available"
        else
            warn "NVIDIA Docker runtime not detected — check nvidia-container-toolkit"
            info "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    fi
}

check_project() {
    header "Project Directory"

    if [ -d "$EXPECTED_PROJECT_PATH" ]; then
        ok "Project exists at ${EXPECTED_PROJECT_PATH}"
    else
        fail "Project not found at ${EXPECTED_PROJECT_PATH}"
        info "Clone: git clone https://github.com/nuniesmith/futures.git ${EXPECTED_PROJECT_PATH}"
        return
    fi

    if [ -d "${EXPECTED_PROJECT_PATH}/.git" ]; then
        ok "Git repository initialized"
        local branch
        branch=$(cd "$EXPECTED_PROJECT_PATH" && git branch --show-current 2>/dev/null || echo "unknown")
        ok "Current branch: ${branch}"
        if [ "$branch" != "main" ]; then
            warn "Expected branch 'main', on '${branch}'"
        fi

        # Check for uncommitted changes
        local dirty
        dirty=$(cd "$EXPECTED_PROJECT_PATH" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        if [ "$dirty" = "0" ]; then
            ok "Working directory clean"
        else
            warn "${dirty} uncommitted change(s) — CI/CD deploy does git pull"
        fi
    else
        warn "Not a git repository — CI/CD deploy requires git"
    fi

    # Check docker-compose.yml
    if [ -f "${EXPECTED_PROJECT_PATH}/docker-compose.yml" ]; then
        ok "docker-compose.yml present"
    else
        fail "docker-compose.yml missing"
    fi

    # Check .env
    if [ -f "${EXPECTED_PROJECT_PATH}/.env" ]; then
        ok ".env file present"

        # Check key env vars (without revealing values)
        local env_file="${EXPECTED_PROJECT_PATH}/.env"
        for var in POSTGRES_PASSWORD API_KEY; do
            if grep -q "^${var}=" "$env_file" 2>/dev/null; then
                ok ".env has ${var}"
            else
                warn ".env missing ${var}"
            fi
        done

        # Check TRAINER_SERVICE_URL override
        if grep -q "^TRAINER_SERVICE_URL=" "$env_file" 2>/dev/null; then
            local trainer_url
            trainer_url=$(grep "^TRAINER_SERVICE_URL=" "$env_file" | cut -d= -f2-)
            ok ".env has TRAINER_SERVICE_URL=${trainer_url}"
        else
            info ".env does not override TRAINER_SERVICE_URL (will use default: http://100.113.72.63:8200)"
        fi
    else
        warn "No .env file — docker-compose.yml requires POSTGRES_PASSWORD at minimum"
        info "Create: cp .env.example .env && nano .env"
    fi
}

check_models() {
    header "Model Files"

    local model_dir="${EXPECTED_PROJECT_PATH}/models"
    if [ ! -d "$model_dir" ]; then
        warn "Models directory not found at ${model_dir}"
        return
    fi
    ok "Models directory exists"

    local files=("breakout_cnn_best.pt" "breakout_cnn_best_meta.json" "feature_contract.json")
    for f in "${files[@]}"; do
        local path="${model_dir}/${f}"
        if [ -f "$path" ]; then
            local size
            size=$(du -h "$path" 2>/dev/null | awk '{print $1}')
            # Check if it's an LFS pointer
            local file_bytes
            file_bytes=$(wc -c < "$path" | tr -d ' ')
            if [ "$file_bytes" -lt 1024 ] && head -1 "$path" 2>/dev/null | grep -q "^version https://git-lfs"; then
                fail "${f} is a Git LFS pointer (${size}) — run: bash scripts/sync_models.sh"
            else
                ok "${f} (${size})"
            fi
        else
            if [ "$f" = "breakout_cnn_best.pt" ] || [ "$f" = "feature_contract.json" ]; then
                fail "${f} — MISSING (required for inference)"
            else
                warn "${f} — not found"
            fi
        fi
    done

    # Show model metadata if available
    local meta="${model_dir}/breakout_cnn_best_meta.json"
    if [ -f "$meta" ] && command -v python3 >/dev/null 2>&1; then
        local model_info
        model_info=$(python3 -c "
import json, sys
try:
    d = json.load(open('${meta}'))
    acc = d.get('val_accuracy', d.get('accuracy', '?'))
    prec = d.get('val_precision', d.get('precision', '?'))
    rec = d.get('val_recall', d.get('recall', '?'))
    ver = d.get('version', d.get('contract_version', '?'))
    print(f'v{ver} — acc={acc}%  prec={prec}%  rec={rec}%')
except Exception:
    print('unreadable')
" 2>/dev/null || echo "unreadable")
        info "Champion model: ${model_info}"
    fi

    # Check sync_models.sh
    if [ -f "${EXPECTED_PROJECT_PATH}/scripts/sync_models.sh" ]; then
        ok "sync_models.sh present"
    else
        warn "sync_models.sh missing"
    fi
}

check_sync_models() {
    header "Model Sync Script"

    local script="${EXPECTED_PROJECT_PATH}/scripts/sync_models.sh"
    if [ ! -f "$script" ]; then
        warn "sync_models.sh not found — skipping"
        return
    fi

    # The script is platform-agnostic (no Pi-specific paths). Verify dependencies.
    for cmd in curl python3; do
        if command -v "$cmd" >/dev/null 2>&1; then
            ok "${cmd} available (needed by sync_models.sh)"
        else
            fail "${cmd} not found — required by sync_models.sh"
        fi
    done

    # Check if sha256sum or shasum is available
    if command -v sha256sum >/dev/null 2>&1; then
        ok "sha256sum available (for LFS integrity checks)"
    elif command -v shasum >/dev/null 2>&1; then
        ok "shasum available (for LFS integrity checks)"
    else
        warn "No sha256sum or shasum — LFS integrity checks will be skipped"
    fi

    # Dry run --check
    info "Running sync_models.sh --check..."
    if bash "$script" --check 2>&1 | sed 's/^/    /'; then
        ok "Model check passed"
    else
        warn "Model check reported issues — see above"
    fi
}

check_network() {
    header "Network & Ports"

    local ports_to_check=()
    if [ "$ROLE" = "server" ]; then
        ports_to_check=(8050 8180 8003 5432 6379 9090 3000)
    else
        ports_to_check=(8200)
    fi

    for port in "${ports_to_check[@]}"; do
        local service=""
        case "$port" in
            8050) service="data-service" ;;
            8180) service="web-dashboard" ;;
            8003) service="charting" ;;
            5432) service="postgres" ;;
            6379) service="redis" ;;
            9090) service="prometheus" ;;
            3000) service="grafana" ;;
            8200) service="trainer" ;;
        esac

        if ss -tlnp 2>/dev/null | grep -q ":${port} " || netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
            ok "Port ${port} (${service}) — listening"
        else
            info "Port ${port} (${service}) — not listening (container may be down)"
        fi
    done
}

check_compose_config() {
    header "Docker Compose Config Validation"

    local compose_file="${EXPECTED_PROJECT_PATH}/docker-compose.yml"
    if [ ! -f "$compose_file" ]; then
        fail "docker-compose.yml not found"
        return
    fi

    # Check TRAINER_SERVICE_URL is using env var (not hardcoded)
    if grep -q 'TRAINER_SERVICE_URL=\${TRAINER_SERVICE_URL:-' "$compose_file" 2>/dev/null; then
        ok "TRAINER_SERVICE_URL uses env var with fallback (not hardcoded)"
    elif grep -q 'TRAINER_SERVICE_URL=http://100\.' "$compose_file" 2>/dev/null; then
        fail "TRAINER_SERVICE_URL is hardcoded — should use \${TRAINER_SERVICE_URL:-...} pattern"
    else
        ok "TRAINER_SERVICE_URL configuration looks correct"
    fi

    # -----------------------------------------------------------------------
    # ENGINE_DATA_URL port check (trainer compose)
    #
    # The data service is exposed on host port 8050 (mapped 8050:8000).
    # The trainer's ENGINE_DATA_URL must point to :8050, NOT :8100.
    # This was a real bug — catch it before it wastes a training run.
    # -----------------------------------------------------------------------
    local trainer_compose="${EXPECTED_PROJECT_PATH}/docker-compose.trainer.yml"
    if [ -f "$trainer_compose" ]; then
        ok "docker-compose.trainer.yml present"

        # Check for the wrong port (8100) in ENGINE_DATA_URL
        if grep -q 'ENGINE_DATA_URL.*:8100' "$trainer_compose" 2>/dev/null; then
            fail "docker-compose.trainer.yml has ENGINE_DATA_URL on port 8100 — must be 8050 (data service port)"
            info "Fix: change :8100 → :8050 in ENGINE_DATA_URL default"
        elif grep -q 'ENGINE_DATA_URL.*:8050' "$trainer_compose" 2>/dev/null; then
            ok "docker-compose.trainer.yml ENGINE_DATA_URL uses correct port 8050"
        else
            warn "Could not verify ENGINE_DATA_URL port in docker-compose.trainer.yml"
        fi
    else
        warn "docker-compose.trainer.yml not found — trainer deploys may use main compose"
    fi

    # Also check .env override if present (trainer machine)
    if [ "$ROLE" = "trainer" ]; then
        local env_file="${EXPECTED_PROJECT_PATH}/.env"
        if [ -f "$env_file" ] && grep -q "^ENGINE_DATA_URL=" "$env_file" 2>/dev/null; then
            local engine_url
            engine_url=$(grep "^ENGINE_DATA_URL=" "$env_file" | cut -d= -f2-)
            if echo "$engine_url" | grep -q ':8100'; then
                fail ".env ENGINE_DATA_URL=${engine_url} uses wrong port 8100 — must be 8050"
                info "Fix: sed -i 's/:8100/:8050/' .env"
            elif echo "$engine_url" | grep -q ':8050'; then
                ok ".env ENGINE_DATA_URL=${engine_url} (correct port)"
            else
                warn ".env ENGINE_DATA_URL=${engine_url} — verify port matches data service (8050)"
            fi
        fi
    fi

    # Cross-check: verify the data service port mapping in main compose
    local data_port
    data_port=$(grep -A2 'data:' "$compose_file" 2>/dev/null | grep -oP '\d+:8000' | head -1 | cut -d: -f1 || echo "")
    if [ -z "$data_port" ]; then
        # Try a broader search for the data service port mapping
        data_port=$(awk '/^    data:/,/^    [a-z]/' "$compose_file" 2>/dev/null | grep -oP '"(\d+):8000"' | head -1 | tr -d '"' | cut -d: -f1 || echo "")
    fi
    if [ -n "$data_port" ]; then
        ok "Data service exposed on host port ${data_port}"
        if [ "$data_port" != "8050" ]; then
            warn "Data service port is ${data_port}, not 8050 — update ENGINE_DATA_URL in trainer compose accordingly"
        fi
    fi

    # Validate compose file
    if cd "$EXPECTED_PROJECT_PATH" && docker compose config --quiet 2>/dev/null; then
        ok "docker-compose.yml is valid"
    else
        warn "docker compose config reported warnings (may need .env values)"
    fi

    # Validate trainer compose file if present
    if [ -f "$trainer_compose" ]; then
        if cd "$EXPECTED_PROJECT_PATH" && docker compose -f docker-compose.trainer.yml config --quiet 2>/dev/null; then
            ok "docker-compose.trainer.yml is valid"
        else
            warn "docker-compose.trainer.yml config reported warnings (may need .env values)"
        fi
    fi
}

print_secrets_checklist() {
    header "GitHub Secrets Checklist"
    echo ""
    echo -e "  ${BOLD}Copy these values to: GitHub → Settings → Secrets and variables → Actions${NC}"
    echo ""

    local ts_ip=""
    if command -v tailscale >/dev/null 2>&1; then
        ts_ip=$(tailscale ip -4 2>/dev/null || echo "")
    fi

    local ssh_port
    ssh_port=$(grep -E "^Port " /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "22")
    if [ -z "$ssh_port" ]; then ssh_port="22"; fi

    echo -e "  ${DIM}─── Tailscale ───${NC}"
    if [ "$ROLE" = "server" ]; then
        echo -e "  PROD_TAILSCALE_IP       = ${CYAN}${ts_ip:-<run tailscale up>}${NC}"
    else
        echo -e "  TRAINER_TAILSCALE_IP    = ${CYAN}${ts_ip:-<run tailscale up>}${NC}"
    fi
    echo -e "  TAILSCALE_OAUTH_CLIENT_ID = ${CYAN}<from Tailscale admin console>${NC}"
    echo -e "  TAILSCALE_OAUTH_SECRET    = ${CYAN}<from Tailscale admin console>${NC}"
    echo ""
    echo -e "  ${DIM}─── SSH ───${NC}"
    echo -e "  PROD_SSH_PORT           = ${CYAN}${ssh_port}${NC}"
    echo -e "  PROD_SSH_USER           = ${CYAN}${EXPECTED_SSH_USER}${NC}"
    if [ "$ROLE" = "server" ]; then
        echo -e "  PROD_SSH_KEY            = ${CYAN}<private key for ${EXPECTED_SSH_USER}@this-machine>${NC}"
    else
        echo -e "  TRAINER_SSH_KEY         = ${CYAN}<private key for ${EXPECTED_SSH_USER}@this-machine>${NC}"
    fi
    echo ""
    echo -e "  ${DIM}─── Docker Hub ───${NC}"
    echo -e "  DOCKER_USERNAME         = ${CYAN}nuniesmith${NC}"
    echo -e "  DOCKER_TOKEN            = ${CYAN}<Docker Hub access token>${NC}"
    echo ""
    echo -e "  ${DIM}─── App ───${NC}"
    echo -e "  API_KEY                 = ${CYAN}<shared API key for data service>${NC}"
    echo -e "  DISCORD_WEBHOOK_ACTIONS = ${CYAN}<Discord webhook URL>${NC}"
    echo ""

    if [ -n "$ts_ip" ]; then
        if [ "$ROLE" = "server" ] && [ "$ts_ip" != "$EXPECTED_PROD_IP" ]; then
            echo -e "  ${YELLOW}⚠  PROD_TAILSCALE_IP in todo.md says ${EXPECTED_PROD_IP} but this machine is ${ts_ip}${NC}"
            echo -e "  ${YELLOW}   Update the GitHub secret if this is the production server.${NC}"
        fi
        if [ "$ROLE" = "trainer" ] && [ "$ts_ip" != "$EXPECTED_TRAINER_IP" ]; then
            echo -e "  ${YELLOW}⚠  TRAINER_TAILSCALE_IP in todo.md says ${EXPECTED_TRAINER_IP} but this machine is ${ts_ip}${NC}"
            echo -e "  ${YELLOW}   Update the GitHub secret if this is the GPU rig.${NC}"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: bash scripts/verify_cicd.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)     Auto-detect role and run all checks"
    echo "  --server      Force check as Ubuntu Server (production)"
    echo "  --trainer     Force check as GPU trainer rig"
    echo "  --both        Run checks for both roles"
    echo "  --secrets     Print GitHub secrets checklist only"
    echo "  --help        Show this help"
}

ROLE=""
MODE="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        --server)  ROLE="server"; MODE="single"; shift ;;
        --trainer) ROLE="trainer"; MODE="single"; shift ;;
        --both)    MODE="both"; shift ;;
        --secrets) MODE="secrets"; shift ;;
        --help)    usage; exit 0 ;;
        *)         echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Ruby Futures — CI/CD Verification          ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"

if [ "$MODE" = "secrets" ]; then
    ROLE=$(detect_role)
    print_secrets_checklist
    exit 0
fi

if [ "$MODE" = "auto" ] || [ "$MODE" = "single" ]; then
    if [ -z "$ROLE" ]; then
        ROLE=$(detect_role)
    fi

    echo ""
    echo -e "  ${BOLD}Role:${NC} ${CYAN}${ROLE}${NC}"
    if [ "$ROLE" = "server" ]; then
        echo -e "  ${DIM}Ubuntu Server — data + engine + web + monitoring${NC}"
    else
        echo -e "  ${DIM}Home GPU rig — trainer service${NC}"
    fi

    check_tailscale
    check_ssh
    check_docker
    check_project
    check_models
    check_sync_models
    check_network
    check_compose_config
    print_secrets_checklist
fi

if [ "$MODE" = "both" ]; then
    ROLE="server"
    echo ""
    echo -e "  ${BOLD}Checking as: ${CYAN}server${NC}"
    check_tailscale
    check_ssh
    check_docker
    check_project
    check_models
    check_network
    check_compose_config

    ROLE="trainer"
    echo ""
    echo -e "  ${BOLD}Checking as: ${CYAN}trainer${NC}"
    check_tailscale
    check_docker
    check_project
    check_network
    print_secrets_checklist
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

header "Summary"
echo ""
echo -e "  ${GREEN}✓ Passed:  ${PASS}${NC}"
echo -e "  ${YELLOW}⚠ Warnings: ${WARN}${NC}"
echo -e "  ${RED}✗ Failed:  ${FAIL}${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}${BOLD}⚡ ${FAIL} issue(s) must be fixed before CI/CD can deploy to this machine.${NC}"
    echo ""
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "  ${YELLOW}${BOLD}⚡ All critical checks passed, but ${WARN} warning(s) to review.${NC}"
    echo ""
    exit 0
else
    echo -e "  ${GREEN}${BOLD}⚡ All checks passed — this machine is ready for CI/CD deployment.${NC}"
    echo ""
    exit 0
fi
