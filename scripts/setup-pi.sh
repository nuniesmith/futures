#!/usr/bin/env bash
# =============================================================================
# Futures Trader — Raspberry Pi 4 Setup Script
# =============================================================================
# Prepares a fresh Raspberry Pi 4 (Ubuntu/Debian ARM64) for running the
# multi-asset KuCoin perpetual futures trader via Docker.
#
# What this script does:
#   1. Updates system packages
#   2. Sets timezone to America/New_York (EDT/EST)
#   3. Installs Docker Engine + Docker Compose plugin
#   4. Configures Docker for non-root usage
#   5. Tunes kernel parameters for Redis (vm.overcommit_memory, THP)
#   6. Installs Python 3.12+ and pip (for local dev/testing)
#   7. Sets up swap (important for Pi 4 with limited RAM)
#   8. Creates project directory structure
#   9. Validates everything works
#
# Usage:
#   chmod +x scripts/setup-pi.sh
#   ./scripts/setup-pi.sh
#
# Requirements:
#   - Raspberry Pi 4 (4GB+ RAM recommended)
#   - Ubuntu 22.04+ or Debian Bookworm (ARM64)
#   - Internet connection
#   - sudo access
#
# Repository: nuniesmith/futures
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIMEZONE="America/New_York"
SWAP_SIZE_MB=2048  # 2GB swap — helps with Docker builds on 4GB Pi
DOCKER_LOG_MAX_SIZE="10m"
DOCKER_LOG_MAX_FILE="3"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { echo -e "${BLUE}ℹ ${NC} $*"; }
success() { echo -e "${GREEN}✅${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠️ ${NC} $*"; }
error()   { echo -e "${RED}❌${NC} $*"; }

header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

confirm() {
    echo ""
    echo -e "${BOLD}$*${NC}"
    read -r -p "Continue? [Y/n] " response
    case "$response" in
        [nN][oO]|[nN])
            echo "Aborted."
            exit 0
            ;;
    esac
}

check_root() {
    if [ "$(id -u)" -eq 0 ]; then
        error "Do not run this script as root. Run as your normal user (it uses sudo when needed)."
        exit 1
    fi
}

check_arch() {
    local arch
    arch=$(uname -m)
    if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
        warn "This script is designed for ARM64 (Raspberry Pi 4)."
        warn "Detected architecture: $arch"
        confirm "You're not on ARM64. Some steps may not apply."
    fi
}

# ---------------------------------------------------------------------------
# Step 1: System update
# ---------------------------------------------------------------------------
step_update() {
    header "Step 1/9 — System Update"

    info "Updating package lists..."
    sudo apt-get update -qq

    info "Upgrading installed packages..."
    sudo apt-get upgrade -y -qq

    info "Installing essential tools..."
    sudo apt-get install -y -qq \
        curl \
        wget \
        git \
        ca-certificates \
        gnupg \
        lsb-release \
        apt-transport-https \
        software-properties-common \
        jq \
        htop \
        tmux \
        unzip

    success "System updated"
}

# ---------------------------------------------------------------------------
# Step 2: Timezone
# ---------------------------------------------------------------------------
step_timezone() {
    header "Step 2/9 — Timezone → ${TIMEZONE}"

    sudo timedatectl set-timezone "$TIMEZONE"
    info "Current time: $(date)"
    info "Timezone: $(timedatectl show --property=Timezone --value)"

    success "Timezone set to ${TIMEZONE} (EDT/EST)"
}

# ---------------------------------------------------------------------------
# Step 3: Docker Engine + Compose
# ---------------------------------------------------------------------------
step_docker() {
    header "Step 3/9 — Docker Engine + Compose"

    if command -v docker &>/dev/null; then
        local docker_version
        docker_version=$(docker --version 2>/dev/null || echo "unknown")
        info "Docker already installed: ${docker_version}"

        # Check for compose plugin
        if docker compose version &>/dev/null; then
            info "Docker Compose plugin: $(docker compose version)"
            success "Docker already set up"
            return
        else
            warn "Docker Compose plugin not found — installing..."
        fi
    fi

    info "Installing Docker via official convenience script..."

    # Remove old versions if present
    sudo apt-get remove -y -qq docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Install using the official get-docker script (handles ARM64 correctly)
    curl -fsSL https://get.docker.com | sudo sh

    # Install compose plugin if not bundled
    if ! docker compose version &>/dev/null; then
        info "Installing Docker Compose plugin..."
        sudo apt-get install -y -qq docker-compose-plugin
    fi

    success "Docker installed: $(docker --version)"
    success "Docker Compose: $(docker compose version)"
}

# ---------------------------------------------------------------------------
# Step 4: Docker non-root access
# ---------------------------------------------------------------------------
step_docker_user() {
    header "Step 4/9 — Docker Non-Root Access"

    local user
    user=$(whoami)

    if groups "$user" | grep -q docker; then
        info "User '${user}' is already in the docker group"
    else
        info "Adding user '${user}' to the docker group..."
        sudo usermod -aG docker "$user"
        warn "You may need to log out and back in for group changes to take effect."
        warn "Or run: newgrp docker"
    fi

    # Configure Docker daemon for Pi-friendly logging
    info "Configuring Docker daemon logging limits..."
    sudo mkdir -p /etc/docker

    # Only write if not already configured
    if [ ! -f /etc/docker/daemon.json ] || ! grep -q "max-size" /etc/docker/daemon.json 2>/dev/null; then
        echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "'"${DOCKER_LOG_MAX_SIZE}"'",
    "max-file": "'"${DOCKER_LOG_MAX_FILE}"'"
  },
  "storage-driver": "overlay2"
}' | sudo tee /etc/docker/daemon.json > /dev/null
        info "Docker daemon config written to /etc/docker/daemon.json"

        # Restart Docker to apply
        sudo systemctl restart docker
    else
        info "Docker daemon config already exists — skipping"
    fi

    # Enable Docker to start on boot
    sudo systemctl enable docker
    sudo systemctl enable containerd

    success "Docker configured for user '${user}'"
}

# ---------------------------------------------------------------------------
# Step 5: Kernel tuning for Redis
# ---------------------------------------------------------------------------
step_kernel_tuning() {
    header "Step 5/9 — Kernel Tuning for Redis"

    local sysctl_file="/etc/sysctl.d/99-futures-redis.conf"
    local thp_service="/etc/systemd/system/disable-thp.service"

    # vm.overcommit_memory = 1 (required by Redis for background saves)
    # net.core.somaxconn = 65535 (Redis recommends >= 511)
    info "Setting kernel parameters for Redis..."

    echo "# Futures Trader — Redis kernel tuning
# vm.overcommit_memory=1 prevents Redis background save failures
vm.overcommit_memory=1
# Higher connection backlog for Redis
net.core.somaxconn=65535" | sudo tee "$sysctl_file" > /dev/null

    sudo sysctl -p "$sysctl_file"
    success "vm.overcommit_memory=1 and somaxconn=65535 applied"

    # Disable Transparent Huge Pages (THP) — causes Redis latency spikes
    info "Disabling Transparent Huge Pages (THP)..."

    local thp_path="/sys/kernel/mm/transparent_hugepage/enabled"
    if [ -f "$thp_path" ]; then
        # Disable now
        echo never | sudo tee "$thp_path" > /dev/null 2>&1 || true
        echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null 2>&1 || true

        # Create systemd service to disable on boot
        echo "[Unit]
Description=Disable Transparent Huge Pages (for Redis)
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled && echo never > /sys/kernel/mm/transparent_hugepage/defrag'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target" | sudo tee "$thp_service" > /dev/null

        sudo systemctl daemon-reload
        sudo systemctl enable disable-thp.service
        sudo systemctl start disable-thp.service 2>/dev/null || true

        success "THP disabled (persists across reboots)"
    else
        info "THP not available on this kernel — skipping"
    fi
}

# ---------------------------------------------------------------------------
# Step 6: Python 3.12+
# ---------------------------------------------------------------------------
step_python() {
    header "Step 6/9 — Python 3.12+"

    # Check if a suitable Python is already installed
    local py=""
    for candidate in python3.13 python3.12; do
        if command -v "$candidate" &>/dev/null; then
            py="$candidate"
            break
        fi
    done

    if [ -n "$py" ]; then
        info "Found: $py ($($py --version 2>&1))"
    else
        # Check if default python3 is >= 3.12
        if command -v python3 &>/dev/null; then
            local ver
            ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
                py="python3"
                info "Found: python3 ($ver)"
            fi
        fi
    fi

    if [ -z "$py" ]; then
        info "Python 3.12+ not found — installing..."

        # Try deadsnakes PPA first (Ubuntu)
        if command -v add-apt-repository &>/dev/null; then
            sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev || {
                warn "Could not install from deadsnakes PPA"
                info "Falling back to system python3..."
                sudo apt-get install -y -qq python3 python3-venv python3-dev python3-pip
            }
        else
            # Debian — install system python3
            sudo apt-get install -y -qq python3 python3-venv python3-dev python3-pip
        fi
    fi

    # Ensure pip is available
    if ! command -v pip3 &>/dev/null; then
        info "Installing pip..."
        sudo apt-get install -y -qq python3-pip
    fi

    # Ensure venv module is available
    sudo apt-get install -y -qq python3-venv 2>/dev/null || true

    # Show final Python version
    local final_py
    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" &>/dev/null; then
            final_py="$candidate"
            break
        fi
    done
    success "Python ready: $($final_py --version 2>&1)"
}

# ---------------------------------------------------------------------------
# Step 7: Swap
# ---------------------------------------------------------------------------
step_swap() {
    header "Step 7/9 — Swap (${SWAP_SIZE_MB}MB)"

    # Check existing swap
    local current_swap
    current_swap=$(free -m | awk '/Swap:/ {print $2}')

    if [ "$current_swap" -ge "$SWAP_SIZE_MB" ]; then
        info "Swap already ${current_swap}MB (>= ${SWAP_SIZE_MB}MB target)"
        success "Swap is sufficient"
        return
    fi

    info "Current swap: ${current_swap}MB — setting up ${SWAP_SIZE_MB}MB swap file..."

    # Disable existing swap file if present
    if [ -f /swapfile ]; then
        sudo swapoff /swapfile 2>/dev/null || true
        sudo rm -f /swapfile
    fi

    # Create new swap file
    sudo fallocate -l "${SWAP_SIZE_MB}M" /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # Make persistent across reboots
    if ! grep -q "/swapfile" /etc/fstab; then
        echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
    fi

    # Set swappiness low (prefer RAM, only use swap when needed)
    echo "vm.swappiness=10" | sudo tee /etc/sysctl.d/99-swappiness.conf > /dev/null
    sudo sysctl vm.swappiness=10

    success "Swap configured: $(free -m | awk '/Swap:/ {print $2}')MB"
}

# ---------------------------------------------------------------------------
# Step 8: Project directory structure
# ---------------------------------------------------------------------------
step_project() {
    header "Step 8/9 — Project Structure"

    local project_dir
    project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    info "Project directory: ${project_dir}"

    # Create directories that should exist
    mkdir -p "${project_dir}/logs"
    mkdir -p "${project_dir}/scripts"

    # Check for .env
    if [ ! -f "${project_dir}/.env" ]; then
        if [ -f "${project_dir}/.env.example" ]; then
            warn ".env file not found"
            info "Creating .env from .env.example..."
            cp "${project_dir}/.env.example" "${project_dir}/.env"
            warn "Edit .env with your real credentials:"
            echo ""
            echo "  nano ${project_dir}/.env"
            echo ""
            echo "  Required:"
            echo "    KUCOIN_API_KEY=your_key"
            echo "    KUCOIN_API_SECRET=your_secret"
            echo "    KUCOIN_PASSPHRASE=your_passphrase"
            echo "    REDIS_PASSWORD=your_redis_password"
            echo ""
        else
            warn ".env.example not found — create .env manually"
        fi
    else
        info ".env file exists"
    fi

    # Make scripts executable
    chmod +x "${project_dir}/run.sh" 2>/dev/null || true
    chmod +x "${project_dir}/scripts/"*.sh 2>/dev/null || true

    success "Project structure ready"
}

# ---------------------------------------------------------------------------
# Step 9: Validate
# ---------------------------------------------------------------------------
step_validate() {
    header "Step 9/9 — Validation"

    local errors=0

    # Docker
    if command -v docker &>/dev/null; then
        success "Docker: $(docker --version)"
    else
        error "Docker not found"
        errors=$((errors + 1))
    fi

    # Docker Compose
    if docker compose version &>/dev/null; then
        success "Docker Compose: $(docker compose version)"
    else
        error "Docker Compose not found"
        errors=$((errors + 1))
    fi

    # Python
    local py=""
    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" &>/dev/null; then
            py="$candidate"
            break
        fi
    done
    if [ -n "$py" ]; then
        success "Python: $($py --version 2>&1)"
    else
        error "Python 3 not found"
        errors=$((errors + 1))
    fi

    # Git
    if command -v git &>/dev/null; then
        success "Git: $(git --version)"
    else
        error "Git not found"
        errors=$((errors + 1))
    fi

    # Timezone
    local tz
    tz=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "unknown")
    if [ "$tz" = "$TIMEZONE" ]; then
        success "Timezone: ${tz}"
    else
        warn "Timezone: ${tz} (expected ${TIMEZONE})"
    fi

    # Kernel params
    local overcommit
    overcommit=$(cat /proc/sys/vm/overcommit_memory 2>/dev/null || echo "?")
    if [ "$overcommit" = "1" ]; then
        success "vm.overcommit_memory: 1"
    else
        warn "vm.overcommit_memory: ${overcommit} (expected 1)"
    fi

    # THP
    local thp_status
    thp_status=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "N/A")
    if echo "$thp_status" | grep -q '\[never\]'; then
        success "THP: disabled"
    else
        warn "THP: ${thp_status} (expected [never])"
    fi

    # Swap
    local swap_mb
    swap_mb=$(free -m | awk '/Swap:/ {print $2}')
    if [ "$swap_mb" -ge 1024 ]; then
        success "Swap: ${swap_mb}MB"
    else
        warn "Swap: ${swap_mb}MB (recommend >= 2048MB)"
    fi

    # RAM
    local ram_mb
    ram_mb=$(free -m | awk '/Mem:/ {print $2}')
    info "RAM: ${ram_mb}MB total"

    # Disk
    local disk_avail
    disk_avail=$(df -h / | awk 'NR==2 {print $4}')
    info "Disk available: ${disk_avail}"

    # Architecture
    info "Architecture: $(uname -m)"
    info "Kernel: $(uname -r)"
    info "OS: $(lsb_release -ds 2>/dev/null || cat /etc/os-release 2>/dev/null | head -1 || echo 'unknown')"

    echo ""
    if [ $errors -gt 0 ]; then
        error "${errors} validation error(s) — fix the issues above and re-run."
        exit 1
    fi

    success "All checks passed!"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║    Futures Trader — Raspberry Pi 4 Setup       ║${NC}"
    echo -e "${BOLD}${CYAN}║    Multi-Asset KuCoin Perpetual Futures        ║${NC}"
    echo -e "${BOLD}${CYAN}║                                                ║${NC}"
    echo -e "${BOLD}${CYAN}║    Timezone: America/New_York (EDT)            ║${NC}"
    echo -e "${BOLD}${CYAN}║    Target:   Raspberry Pi 4 (ARM64)            ║${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════╝${NC}"
    echo ""

    check_root
    check_arch

    echo "This script will:"
    echo "  1. Update system packages"
    echo "  2. Set timezone to ${TIMEZONE}"
    echo "  3. Install Docker Engine + Compose"
    echo "  4. Configure Docker for non-root usage"
    echo "  5. Tune kernel for Redis (overcommit, THP)"
    echo "  6. Install Python 3.12+"
    echo "  7. Set up ${SWAP_SIZE_MB}MB swap"
    echo "  8. Prepare project structure"
    echo "  9. Validate everything"

    confirm "Ready to set up your Pi for futures trading?"

    local start_time
    start_time=$(date +%s)

    step_update
    step_timezone
    step_docker
    step_docker_user
    step_kernel_tuning
    step_python
    step_swap
    step_project
    step_validate

    local end_time elapsed
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo ""
    echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${GREEN}  Setup complete! (${elapsed}s)${NC}"
    echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Edit your credentials:"
    echo "     nano .env"
    echo ""
    echo "  2. Start in simulation mode first:"
    echo "     ./run.sh build"
    echo "     SIM_MODE=true ./run.sh up"
    echo "     ./run.sh logs"
    echo ""
    echo "  3. When ready for live trading:"
    echo "     # Edit .env → set TRADING_MODE=live"
    echo "     ./run.sh down && ./run.sh up"
    echo ""
    echo "  4. Monitor:"
    echo "     ./run.sh status"
    echo "     ./run.sh logs"
    echo ""

    # Remind about re-login for docker group
    if ! docker info &>/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Log out and back in (or run 'newgrp docker') for Docker access.${NC}"
        echo ""
    fi
}

main "$@"
