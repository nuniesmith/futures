#!/usr/bin/env bash
# =============================================================================
# FKS — Unified Project Management Script
# =============================================================================
#
# Usage:
#   ./run.sh <command> [flags]
#
# Top-level commands:
#   (no args) / check     Lint + type-check + test everything (Python & Rust)
#   lint                  Ruff check/fix + clippy (no tests)
#   fmt                   Auto-format Python (ruff) and Rust (cargo fmt)
#   test                  Run all Python + Rust tests
#   all                   Build base image, then start everything
#                         (core + ruby + trainer + monitoring)
#   start [prod]          Env setup → build → start all services (dev or prod)
#   up [prod] [svcs...]   Start services (already built)
#   down [prod] [-v]      Stop services (-v removes volumes)
#   restart [svcs...]     Restart services
#   logs [svc]            Follow service logs
#   status                Show service status
#   health                Health check all services
#   build [svcs...]       Build service images
#   build-ruby            Build FKS Python images (base cache → ruby)
#   build-ruby-trainer    Build GPU trainer image
#   build-redis           Build custom Redis image
#   retrain               Run 3-tier CNN retrain (default: --tier all)
#   setup-env             Generate or validate .env, fill missing secrets, prompt for API keys
#   generate-certs        Generate internal service TLS certs (skips if already present)
#   ssl-local             Force-regenerate internal service TLS certs
#   shell <svc>           Open a shell in a running container
#   ruby <cmd>            Ruby subsystem shortcuts (build/test/logs/shell/restart/up/down)
#   clean                 Remove stopped containers and dangling images
#   force-clean           ⚠️  Remove ALL FKS resources including volumes
#   network-cleanup       Fix Docker network conflicts
#   setup-kernel          Install sysctl tuning (fd limits, net, inotify) — run once per host
#   setup-hosts           Add fkstrading.local entries to /etc/hosts — run once per host
#   diagnose              Show detailed system diagnostics
#   web-hash-password     Generate bcrypt hash for WEB_PASSWORD_HASH
#   test-ruby [svc]       Run pytest in a Ruby container
#   help                  Show this message
#
# RustAssistant / OpenClaw commands:
#   ra up [--ollama] [--skip-build]   Start RA + OpenClaw stack
#   ra down [-v]                      Stop RA stack
#   ra status                         Show RA service status
#   ra logs [svc]                     Tail RA service logs
#   ra health                         Health check RA services
#   ra build [openclaw|app]           Build RA / OpenClaw images
#   ra shell [svc]                    Open shell in RA container (default: ra-app)
#   ra test [args]                    Run Rust tests (cargo test)
#   ra ci                             Full CI pipeline (fmt + clippy + test)
#   ra db <backup|shell|size|migrate>
#   ra diagnose                       RA-specific diagnostics
#   ra clean / ra force-clean         Clean RA Docker resources
#   ra dev                            Run RA server locally (cargo run)
#   ra watch                          Run RA with auto-reload (cargo-watch)
#   openclaw <cmd>                    Run OpenClaw CLI commands (passthrough)
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.yml"            # repo-root compose (single source of truth)
PROD_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

# docker compose invocation with consistent project name and env file
DC="docker compose -p fks --env-file $ENV_FILE"

# RA/OpenClaw docker compose (no project name override — uses compose default)
DC_RA="docker compose --env-file $ENV_FILE -f $COMPOSE_FILE"

# RA-specific paths & constants
OPENCLAW_BUILD_SCRIPT="docker/openclaw/build.sh"
RA_PORT="3500"
OPENCLAW_PORT="18789"

# Load .env for variable interpolation (TAILSCALE_IP, RA passwords, etc.)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE" 2>/dev/null || true
    set +a
fi
TAILSCALE_IP="${TAILSCALE_IP:-}"



# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()     { echo -e "${CYAN}[run]${NC} $*"; }
ok()      { echo -e "${GREEN}[  ✓ ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[ warn ]${NC} $*"; }
err()     { echo -e "${RED}[ fail ]${NC} $*"; }
header()  { echo -e "${BLUE}===================================================${NC}"; \
            echo -e "${BLUE}  $*${NC}"; \
            echo -e "${BLUE}===================================================${NC}"; }
info()    { echo -e "${BLUE}ℹ $*${NC}"; }

# ---------------------------------------------------------------------------
# Secret generators
# ---------------------------------------------------------------------------

generate_password() {
    # 40-char URL-safe password (no /, +, = which break DATABASE_URL)
    openssl rand -base64 48 | tr -d '/+=' | head -c 40 2>/dev/null \
        || python3 -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(40)))"
}

generate_secret() {
    # 32-char alphanumeric — URL-safe API key / short secret
    openssl rand -base64 32 | tr -d "=+/\n" | cut -c1-32 2>/dev/null \
        || python3 -c "import secrets; print(secrets.token_urlsafe(24)[:32])"
}

# ---------------------------------------------------------------------------
# Interactive API key prompt — asks user whether to enter secrets now or later
# ---------------------------------------------------------------------------

_prompt_api_keys() {
    local env_file="$1"
    [ -t 0 ] || return 0   # non-interactive — skip

    echo ""
    info "Would you like to enter your API keys now?"
    info "  (y) Enter keys interactively — paste each one into the terminal"
    info "  (n) Skip — edit .env manually later"
    echo ""
    printf "  Enter API keys now? [y/N]: "
    local answer=""
    read -r answer
    echo ""

    case "$answer" in
        [yY]|[yY][eE][sS])
            info "Enter each key when prompted (press Enter to skip any)."
            info "Keys are written directly to .env — they never appear in logs."
            echo ""

            _prompt_key() {
                local key="$1" desc="$2" val=""
                printf "  %s (%s): " "$key" "$desc"
                read -r val
                if [ -n "$val" ]; then
                    if grep -q "^${key}=" "$env_file"; then
                        sed -i "s|^${key}=.*|${key}=${val}|" "$env_file"
                    else
                        echo "${key}=${val}" >> "$env_file"
                    fi
                    ok "  Set $key"
                fi
            }

            echo "  ── Required for AI features ──"
            _prompt_key XAI_API_KEY "xAI / Grok API key — required for AI analyst"
            echo ""

            echo "  ── Trading (optional — skip if not trading yet) ──"
            _prompt_key KRAKEN_API_KEY "Kraken API key"
            _prompt_key KRAKEN_API_SECRET "Kraken API secret"
            _prompt_key MASSIVE_API_KEY "Massive.com futures data key"
            echo ""

            echo "  ── GitHub + Discord (optional — for RA / OpenClaw) ──"
            _prompt_key GITHUB_TOKEN "GitHub PAT (repo read scope)"
            _prompt_key DISCORD_BOT_TOKEN "Discord bot token for OpenClaw"
            echo ""

            echo "  ── Market data (optional — features degrade gracefully) ──"
            _prompt_key FINNHUB_API_KEY "Finnhub news key"
            _prompt_key CMC_API_KEY "CoinMarketCap key"
            _prompt_key WHALE_ALERT_API_KEY "Whale Alert key"
            _prompt_key ETHERSCAN_API_KEY "Etherscan key"
            _prompt_key CRYPTOCOMPARE_API_KEY "CryptoCompare key"
            echo ""

            ok "API keys saved to .env"
            ;;
        *)
            info "Skipped — edit .env manually to add API keys later"
            ;;
    esac
}

generate_fernet_key() {
    # 44-char URL-safe base64 Fernet symmetric key (32 random bytes).
    # Accepted directly by cryptography.fernet.Fernet without any derivation.
    # Format: [A-Za-z0-9_-]{43}= (always ends with exactly one '=')
    openssl rand -base64 32 | tr '+/' '-_' 2>/dev/null \
        || python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
}

generate_long_secret() {
    # 64-char alphanumeric — high-entropy session / HMAC secret
    openssl rand -base64 72 | tr -d "=+/\n" | cut -c1-64 2>/dev/null \
        || python3 -c "import secrets; print(secrets.token_urlsafe(48)[:64])"
}

# ---------------------------------------------------------------------------
# Web password helpers
# ---------------------------------------------------------------------------

_generate_bcrypt_hash() {
    # Safely hash a password with bcrypt, passing it via env var to avoid
    # shell quoting / injection issues with special characters.
    # Usage: hash=$(_generate_bcrypt_hash "$password")
    # Returns the hash on stdout, or empty string if bcrypt is unavailable.
    FKS_BCRYPT_PW="$1" python3 -c "
import os, bcrypt
pw = os.environ['FKS_BCRYPT_PW'].encode()
print(bcrypt.hashpw(pw, bcrypt.gensalt()).decode())
" 2>/dev/null || true
}

_prompt_web_password() {
    # Interactively prompt for a dashboard password + confirmation, generate a
    # bcrypt hash, and write it to WEB_PASSWORD_HASH in the given env file.
    # Skips silently if stdin is not a terminal or the user presses Enter blank.
    # Usage: _prompt_web_password "$env_file"
    local env_file="$1"
    local password="" confirm="" hash=""

    [ -t 0 ] || return 0   # non-interactive — skip
    grep -q "^WEB_PASSWORD_HASH=.\+" "$env_file" 2>/dev/null && return 0  # already set — skip

    echo ""
    info "Set a web dashboard password for WEB_PASSWORD_HASH"
    info "(Press Enter with no input to skip — leaves authentication disabled)"
    echo ""
    printf "  Password : "
    stty -echo 2>/dev/null || true
    read -r password
    stty echo 2>/dev/null || true
    printf "\n"

    if [ -z "$password" ]; then
        warn "No password entered — WEB_PASSWORD_HASH left empty (auth disabled)"
        return 0
    fi

    printf "  Confirm  : "
    stty -echo 2>/dev/null || true
    read -r confirm
    stty echo 2>/dev/null || true
    printf "\n"

    if [ "$password" != "$confirm" ]; then
        warn "Passwords do not match — WEB_PASSWORD_HASH left empty"
        return 0
    fi

    echo ""
    info "Generating bcrypt hash…"
    hash=$(_generate_bcrypt_hash "$password")

    if [ -z "$hash" ]; then
        warn "bcrypt unavailable — set WEB_PASSWORD_HASH manually after: pip install bcrypt"
        warn "  ./run.sh web-hash-password"
        return 0
    fi

    # Write into .env wrapped in single quotes so the bcrypt hash's literal
    # $ characters (e.g. $2b$12$...) are not expanded when the file is sourced.
    if grep -q "^WEB_PASSWORD_HASH=" "$env_file"; then
        sed -i "s|^WEB_PASSWORD_HASH=.*|WEB_PASSWORD_HASH='${hash}'|" "$env_file"
    else
        echo "WEB_PASSWORD_HASH='${hash}'" >> "$env_file"
    fi
    ok "WEB_PASSWORD_HASH set"
}

# ---------------------------------------------------------------------------
# Tailscale
# ---------------------------------------------------------------------------
get_tailscale_ip() {
    if command -v tailscale >/dev/null 2>&1; then
        local ip
        ip=$(tailscale ip -4 2>/dev/null || true)
        if [ -n "$ip" ]; then echo "$ip"; return; fi
    fi
    warn "Tailscale not available or not connected — Tailscale IP unknown"
    echo ""
}

# =============================================================================
# .env management
# =============================================================================

setup_env_file() {
    local env_file="$ENV_FILE"

    # ── First-time creation ─────────────────────────────────────────────────
    if [ ! -f "$env_file" ]; then
        header "Generating .env with secure secrets"

        # Auto-detect Tailscale IP
        local ts_ip=""
        ts_ip=$(get_tailscale_ip 2>/dev/null || true)
        if [ -z "$ts_ip" ]; then ts_ip="127.0.0.1"; fi

        # Generate shared RA proxy key (used by both RA_PROXY_API_KEYS and RA_API_KEY)
        local ra_proxy_key
        ra_proxy_key=$(generate_long_secret)

        cat > "$env_file" << EOF
# =============================================================================
# FKS Trading System — Environment
# Generated $(date -u +"%Y-%m-%dT%H:%M:%SZ") by ./run.sh setup-env
# Edit this file to fill in API keys and optional settings.
# Re-run ./run.sh setup-env at any time to fill missing secrets.
# =============================================================================


# =============================================================================
# SECTION 1 — SERVICE SECRETS  (auto-generated — never commit to git)
# =============================================================================

# --- Postgres (shared instance: janus_db + ruby_db) ---
POSTGRES_USER=fks_user
POSTGRES_PASSWORD=$(generate_password)
POSTGRES_DB=janus_db
POSTGRES_DATA_DB=ruby_db

# --- Redis ---
REDIS_PASSWORD=$(generate_password)

# --- QuestDB ---
QUESTDB_PG_USER=admin
QUESTDB_PG_PASSWORD=$(generate_password)

# --- Grafana ---
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(generate_password)

# --- Discord webhooks (optional) ---
DISCORD_WEBHOOK_ANALYSIS=
DISCORD_WEBHOOK_GENERAL=
DISCORD_WEBHOOK_SIGNALS=
DISCORD_BOT_TOKEN=

# --- Web dashboard auth ---
# WEB_PASSWORD_HASH: bcrypt hash of your login password.
#   ./run.sh setup-env  — prompts you interactively (recommended)
#   ./run.sh web-hash-password  — generate hash separately
# Leave empty to disable auth (Tailscale-only access).
WEB_PASSWORD_HASH=
WEB_SESSION_SECRET=$(generate_long_secret)
WEB_SESSION_TTL_DAYS=30

# --- API key encryption (Fernet) ---
# Master key for encrypting third-party credentials stored in Postgres.
# Rotate with care: changing this renders all DB-stored keys unreadable.
API_KEY_ENCRYPTION_SECRET=$(generate_fernet_key)

# --- Internal API key (data service + web endpoints) ---
API_KEY=$(generate_secret)

# --- Trainer API key ---
TRAINER_API_KEY=$(generate_secret)

# --- nginx internal token (nginx → backend trust header) ---
NGINX_INTERNAL_TOKEN=$(generate_secret)

# --- Tailscale (all ports bind to this IP — never the public interface) ---
TAILSCALE_IP=${ts_ip}


# =============================================================================
# SECTION 2 — JANUS (Rust trading engine)
# =============================================================================

EXECUTION_MODE=paper_trading
DATA_SOURCE=live
DATA_EXCHANGE=binance
DATA_WS_URL=wss://stream.binance.com:9443/ws
DATA_KLINE_INTERVALS=1m,5m

JANUS_ENABLE_BACKWARD=true
ENABLE_EXECUTION=true

OPTIMIZER_ENABLED=true
OPTIMIZE_ASSETS=BTC,ETH,SOL
OPTIMIZE_INTERVAL=6h
OPTIMIZE_TRIALS=100
OPTIMIZE_HISTORICAL_DAYS=30
OPTIMIZER_INSTANCE_ID=janus-dev

BRAIN_WIRE_KILL_SWITCH=false
BRAIN_AUTO_START_WATCHDOG=true

# --- Trade execution ---
EXECUTION_EXCHANGE=kraken
EXEC_ACCOUNT_TYPE=personal-crypto
EXECUTION_CONNECT_TIMEOUT=10
EXECUTION_REQUEST_TIMEOUT=30
EXECUTION_MAX_RETRIES=3
EXECUTION_RETRY_BACKOFF_MS=100
EXECUTION_DEFAULT_QUANTITY=0.001


# =============================================================================
# SECTION 3 — EXTERNAL API KEYS  (prefer WebUI Settings → API Keys)
# key_manager resolves: encrypted DB → env var → None (graceful disable)
# =============================================================================

# --- Broker: Kraken ---
KRAKEN_API_KEY=
KRAKEN_API_SECRET=
ENABLE_KRAKEN_CRYPTO=1

# --- Market data: Massive (futures bars) ---
MASSIVE_API_KEY=
MASSIVE_S3_KEY_ID=
MASSIVE_S3_SECRET=
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles

# --- Market data: supplementary ---
FINNHUB_API_KEY=
ALPHA_VANTAGE_API_KEY=

# --- TradingView (sessionid cookie from a logged-in TV session) ---
TV_SESSION_ID=

# --- AI / LLM (shared between Ruby + RA) ---
XAI_API_KEY=

# --- Reddit sentiment ---
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=fks-news-pipeline/1.0

# --- Copier integration ---
COPIER_URL=


# =============================================================================
# SECTION 4 — ON-CHAIN MONITORING & MARKET DATA  (all optional)
# =============================================================================

WHALE_ALERT_API_KEY=
ETHERSCAN_API_KEY=
SOLSCAN_API_KEY=
CRYPTOCOMPARE_API_KEY=
MEMPOOL_SPACE_URL=https://mempool.space/api

CHAIN_MIN_USD=1000000
CHAIN_POLL_WHALE=30
CHAIN_POLL_MEMPOOL=60
CHAIN_POLL_FLOW=3600

CMC_API_KEY=
CMC_POLL_QUOTES=900
CMC_POLL_GLOBAL=1800
CMC_POLL_FEAR_GREED=1800
CMC_POLL_LISTINGS=21600


# =============================================================================
# SECTION 5 — RUBY PYTHON SERVICES CONFIGURATION
# =============================================================================

# --- ORB / CNN signal gates ---
ORB_FILTER_GATE=majority
ORB_CNN_GATE=0

# --- DataFactory ---
FACTORY_ENABLED_CLASSES=futures_cme,crypto
GAP_SCAN_INTERVAL_SECS=300

# --- Bar backfill ---
BACKFILL_DAYS_BACK=365
BACKFILL_CHUNK_DAYS=30

# --- Paper trading / simulation ---
PAPER_TRADING_ENABLED=0
SIM_ENABLED=0
SIM_DATA_SOURCE=kraken
SIM_INITIAL_BALANCE=100000

# =============================================================================
# SECTION 6 — TRAINER  (GPU CNN retraining — --profile training)
# =============================================================================

CNN_RETRAIN_SYMBOLS=MGC,SIL,MES,MNQ,M2K,MYM
CNN_RETRAIN_DAYS_BACK=365
CNN_RETRAIN_BARS_SOURCE=engine
CNN_RETRAIN_EPOCHS=60
CNN_RETRAIN_BATCH_SIZE=64
CNN_RETRAIN_LR=0.0001
CNN_RETRAIN_PATIENCE=12
CNN_RETRAIN_MIN_ACC=80.0
CNN_RETRAIN_MIN_PRECISION=75.0
CNN_RETRAIN_MIN_RECALL=70.0


# =============================================================================
# SECTION 7 — RUSTASSISTANT  (AI Agent + LLM Proxy)
# Uses the shared FKS postgres (rustassistant db) and redis (db 1).
# =============================================================================

# --- RA proxy auth (shared by OpenClaw, futures app, Zed IDE) ---
RA_PROXY_API_KEYS=${ra_proxy_key}
RA_API_KEY=${ra_proxy_key}

# --- RA server config ---
RA_BASE_URL=http://fks_ra:3500
RA_REPO_ID=rustassistant

# --- GitHub integration ---
GITHUB_TOKEN=
GITHUB_USERNAME=nuniesmith
GITHUB_ORG=
GITHUB_WEBHOOK_SECRET=

# --- AI provider config ---
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4-1-fast-reasoning
XAI_MAX_TOKENS=4096
XAI_TEMPERATURE=0.3

# --- Ollama (local LLM — optional, --profile ollama) ---
OLLAMA_BASE_URL=http://fks_ra_ollama:11434
LOCAL_MODEL=qwen2.5-coder:7b
FORCE_REMOTE_MODEL=false

# --- RA paths ---
REPOS_BASE_PATH=/home/jordan/github/

# --- RA caching ---
CACHE_ENABLED=true
CACHE_PREFIX=rustassistant

# --- RA background services ---
REPO_SYNC_INTERVAL_SECS=300
AUTO_SCAN_ENABLED=true
AUTO_SCAN_INTERVAL=60
AUTO_SCAN_MAX_CONCURRENT=2

# --- RA cost control ---
DAILY_BUDGET_USD=5.00
WARN_AT_PERCENT=80
BLOCK_ON_EXCEED=true


# =============================================================================
# SECTION 8 — OPENCLAW  (Discord Bot)
# =============================================================================

OPENCLAW_BASE_IMAGE=nuniesmith/fks:openclaw-base
OPENCLAW_IMAGE=nuniesmith/fks:openclaw
OPENCLAW_GATEWAY_TOKEN=$(generate_secret)
OPENCLAW_CONFIG_DIR=./openclaw/config
OPENCLAW_WORKSPACE_DIR=./openclaw/workspace


# =============================================================================
# SECTION 9 — LOGGING & TIMEZONE
# =============================================================================

LOG_LEVEL=info
PYTHONUNBUFFERED=1
TZ=America/Toronto
EOF
        ok ".env generated with all service secrets (sections 1–9)"
        warn "Review .env and fill in your API keys before going live"

        # Ensure .env is gitignored
        if [ -f ".gitignore" ] && ! grep -q "^\.env$" .gitignore; then
            echo ".env" >> .gitignore
            info "Added .env to .gitignore"
        fi

        # ── Interactive API key prompt ──────────────────────────────────────
        _prompt_api_keys "$env_file"
    fi

    # ── Helpers (used for both new and existing files) ──────────────────────
    _env_get() { grep "^${1}=" "$env_file" | cut -d'=' -f2- | sed 's/#.*//' | tr -d ' '; }
    _env_set() {
        local key="$1" val="$2"
        if grep -q "^${key}=" "$env_file"; then
            sed -i "s|^${key}=.*|${key}=${val}|" "$env_file"
        else
            echo "${key}=${val}" >> "$env_file"
        fi
    }
    _env_del() {
        sed -i "/^${1}=/d" "$env_file"
    }

    # ── Validation / patch of existing file ────────────────────────────────
    local needs_update=false

    # --- Enforce canonical values ---
    local pg_db;  pg_db=$(_env_get POSTGRES_DB)
    local pg_usr; pg_usr=$(_env_get POSTGRES_USER)
    if [ "$pg_db" != "janus_db" ]; then
        _env_set POSTGRES_DB janus_db
        warn "Fixed POSTGRES_DB → janus_db"
        needs_update=true
    fi
    if [ "$pg_usr" != "fks_user" ]; then
        _env_set POSTGRES_USER fks_user
        warn "Fixed POSTGRES_USER → fks_user"
        needs_update=true
    fi

    # --- Migrate RUBY_POSTGRES_* → unified POSTGRES_* vars ---
    if grep -q "^RUBY_POSTGRES_USER=" "$env_file"; then
        _env_del RUBY_POSTGRES_USER
        _env_del RUBY_POSTGRES_PASSWORD
        _env_del RUBY_POSTGRES_DB
        warn "Removed stale RUBY_POSTGRES_* vars (now using POSTGRES_USER / POSTGRES_DATA_DB)"
        needs_update=true
    fi

    # --- Add POSTGRES_DATA_DB if missing ---
    if ! grep -q "^POSTGRES_DATA_DB=" "$env_file"; then
        _env_set POSTGRES_DATA_DB ruby_db
        warn "Added POSTGRES_DATA_DB=ruby_db"
        needs_update=true
    fi

    # --- Remove vars that docker-compose constructs itself ---
    for stale_var in DATABASE_URL GATEWAY_SECRET_KEY GATEWAY_JWT_SECRET \
                     DATA_SERVICE_JWT_SECRET DATA_SERVICE_JWT_EXPIRY \
                     GF_SECURITY_SECRET_KEY QUESTDB_PASSWORD QUESTDB_USER \
                     DISCORD_NOTIFICATIONS_ENABLED DISCORD_NOTIFY_ON_SIGNAL \
                     DISCORD_NOTIFY_ON_FILL DISCORD_NOTIFY_ON_ERROR \
                     REDDIT_POLL_INTERVAL REAL_ORDERS_ENABLED \
                     EXECUTION_ENDPOINT ENVIRONMENT RUST_LOG RUST_BACKTRACE \
                     CLOUDFLARE_API_KEY; do
        if grep -q "^${stale_var}=" "$env_file"; then
            _env_del "$stale_var"
            warn "Removed stale var: $stale_var"
            needs_update=true
        fi
    done

    # --- Fix stale signals repo name ---
    local sig_repo; sig_repo=$(_env_get GITHUB_SIGNALS_REPO || true)
    if [ "$sig_repo" = "nuniesmith/ruby-signals" ]; then
        _env_set GITHUB_SIGNALS_REPO nuniesmith/fks-signals
        warn "Fixed GITHUB_SIGNALS_REPO → nuniesmith/fks-signals"
        needs_update=true
    fi

    # --- Auto-fill missing/empty secrets ---
    _fill_password() {
        local key="$1"; local val; val=$(_env_get "$key")
        if [ -z "$val" ]; then
            _env_set "$key" "$(generate_password)"
            warn "Generated $key"
            needs_update=true
        fi
    }
    _fill_secret() {
        local key="$1"; local val; val=$(_env_get "$key")
        if [ -z "$val" ]; then
            _env_set "$key" "$(generate_secret)"
            warn "Generated $key"
            needs_update=true
        fi
    }
    _fill_long() {
        local key="$1"; local val; val=$(_env_get "$key")
        if [ -z "$val" ] || [ "${#val}" -lt 64 ]; then
            _env_set "$key" "$(generate_long_secret)"
            warn "Generated $key (64-char)"
            needs_update=true
        fi
    }
    _fill_fernet() {
        local key="$1"; local val; val=$(_env_get "$key")
        if [ -z "$val" ]; then
            _env_set "$key" "$(generate_fernet_key)"
            warn "Generated $key (Fernet key)"
            needs_update=true
        fi
    }

    _fill_password POSTGRES_PASSWORD
    _fill_password REDIS_PASSWORD
    _fill_password QUESTDB_PG_PASSWORD
    _fill_password GRAFANA_PASSWORD

    _fill_secret   API_KEY
    _fill_secret   TRAINER_API_KEY
    _fill_long     WEB_SESSION_SECRET
    _fill_fernet   API_KEY_ENCRYPTION_SECRET
    _fill_secret   NGINX_INTERNAL_TOKEN

    # --- RA secrets (Section 7) ---
    # RA proxy key — RA_PROXY_API_KEYS and RA_API_KEY must be the same value.
    local ra_proxy; ra_proxy=$(_env_get RA_PROXY_API_KEYS)
    if [ -z "$ra_proxy" ]; then
        ra_proxy=$(generate_long_secret)
        _env_set RA_PROXY_API_KEYS "$ra_proxy"
        _env_set RA_API_KEY "$ra_proxy"
        warn "Generated RA_PROXY_API_KEYS + RA_API_KEY (shared key)"
        needs_update=true
    else
        # Ensure RA_API_KEY matches RA_PROXY_API_KEYS
        local ra_api; ra_api=$(_env_get RA_API_KEY)
        if [ "$ra_api" != "$ra_proxy" ]; then
            _env_set RA_API_KEY "$ra_proxy"
            warn "Synced RA_API_KEY to match RA_PROXY_API_KEYS"
            needs_update=true
        fi
    fi

    # --- OpenClaw secret (Section 8) ---
    _fill_secret OPENCLAW_GATEWAY_TOKEN

    # --- Tailscale IP auto-detect ---
    local ts_val; ts_val=$(_env_get TAILSCALE_IP)
    if [ -z "$ts_val" ] || [ "$ts_val" = "100.x.x.x" ]; then
        local detected_ts=""
        detected_ts=$(get_tailscale_ip 2>/dev/null || true)
        if [ -n "$detected_ts" ]; then
            _env_set TAILSCALE_IP "$detected_ts"
            warn "Set TAILSCALE_IP=$detected_ts (auto-detected)"
            needs_update=true
        else
            warn "TAILSCALE_IP not set — using 127.0.0.1 (Tailscale not connected?)"
            _env_set TAILSCALE_IP "127.0.0.1"
            needs_update=true
        fi
    fi

    # --- Ensure RA config keys exist (value may be empty) ---
    for ra_key in RA_BASE_URL RA_REPO_ID GITHUB_TOKEN GITHUB_USERNAME \
                  XAI_BASE_URL XAI_MODEL OLLAMA_BASE_URL LOCAL_MODEL \
                  CACHE_ENABLED CACHE_PREFIX TZ; do
        if ! grep -q "^${ra_key}=" "$env_file"; then
            case "$ra_key" in
                RA_BASE_URL)     _env_set "$ra_key" "http://fks_ra:3500" ;;
                RA_REPO_ID)      _env_set "$ra_key" "rustassistant" ;;
                GITHUB_USERNAME) _env_set "$ra_key" "nuniesmith" ;;
                XAI_BASE_URL)    _env_set "$ra_key" "https://api.x.ai/v1" ;;
                XAI_MODEL)       _env_set "$ra_key" "grok-4-1-fast-reasoning" ;;
                OLLAMA_BASE_URL) _env_set "$ra_key" "http://fks_ra_ollama:11434" ;;
                LOCAL_MODEL)     _env_set "$ra_key" "qwen2.5-coder:7b" ;;
                CACHE_ENABLED)   _env_set "$ra_key" "true" ;;
                CACHE_PREFIX)    _env_set "$ra_key" "rustassistant" ;;
                TZ)              _env_set "$ra_key" "America/Toronto" ;;
                *)               echo "${ra_key}=" >> "$env_file" ;;
            esac
            warn "Added ${ra_key}"
            needs_update=true
        fi
    done

    # --- Ensure DISCORD_WEBHOOK_* key exists (value may be empty) ---
    if ! grep -q "^DISCORD_WEBHOOK_ANALYSIS=" "$env_file"; then
        echo "DISCORD_WEBHOOK_ANALYSIS=" >> "$env_file"
        warn "Added DISCORD_WEBHOOK_ANALYSIS (empty — set to your Discord webhook URL)"
        needs_update=true
    fi
    if ! grep -q "^DISCORD_WEBHOOK_GENERAL=" "$env_file"; then
        echo "DISCORD_WEBHOOK_GENERAL=" >> "$env_file"
        warn "Added DISCORD_WEBHOOK_GENERAL (empty — set to your Discord webhook URL)"
        needs_update=true
    fi
    if ! grep -q "^DISCORD_WEBHOOK_SIGNALS=" "$env_file"; then
        echo "DISCORD_WEBHOOK_SIGNALS=" >> "$env_file"
        warn "Added DISCORD_WEBHOOK_SIGNALS (empty — set to your Discord webhook URL)"
        needs_update=true
    fi
    if ! grep -q "^DISCORD_BOT_TOKEN=" "$env_file"; then
        echo "DISCORD_BOT_TOKEN=" >> "$env_file"
        warn "Added DISCORD_BOT_TOKEN (empty — set to your Discord bot token)"
        needs_update=true
    fi

    # --- OpenClaw config keys ---
    if ! grep -q "^OPENCLAW_BASE_IMAGE=" "$env_file"; then
        _env_set OPENCLAW_BASE_IMAGE "nuniesmith/fks:openclaw-base"
        warn "Added OPENCLAW_BASE_IMAGE"
        needs_update=true
    fi
    if ! grep -q "^OPENCLAW_IMAGE=" "$env_file"; then
        _env_set OPENCLAW_IMAGE "nuniesmith/fks:openclaw"
        warn "Added OPENCLAW_IMAGE"
        needs_update=true
    fi
    if ! grep -q "^OPENCLAW_CONFIG_DIR=" "$env_file"; then
        _env_set OPENCLAW_CONFIG_DIR "./openclaw/config"
        warn "Added OPENCLAW_CONFIG_DIR"
        needs_update=true
    fi
    if ! grep -q "^OPENCLAW_WORKSPACE_DIR=" "$env_file"; then
        _env_set OPENCLAW_WORKSPACE_DIR "./openclaw/workspace"
        warn "Added OPENCLAW_WORKSPACE_DIR"
        needs_update=true
    fi

    # --- Render alertmanager.yml from template ---
    local tmpl="infrastructure/configs/monitoring/alertmanager/alertmanager.yml.tmpl"
    local dest="infrastructure/configs/monitoring/alertmanager/alertmanager.yml"
    if [ -f "$tmpl" ]; then
        # Temporarily disable -u so any $-containing values (e.g. bcrypt hashes)
        # in the .env don't trigger "unbound variable" errors when sourced.
        set -a; set +u; source "$env_file"; set -u; set +a
        if command -v envsubst &>/dev/null; then
            envsubst '${DISCORD_WEBHOOK_GENERAL}' < "$tmpl" > "$dest"
        else
            local wh; wh=$(_env_get DISCORD_WEBHOOK_GENERAL)
            sed "s|\${DISCORD_WEBHOOK_GENERAL}|${wh}|g" "$tmpl" > "$dest"
        fi
    fi

    if [ "$needs_update" = true ]; then
        ok ".env updated"
    else
        ok ".env validated — all secrets present"
    fi

    # --- Warn about empty API keys that affect functionality ---
    _env_get MASSIVE_API_KEY | grep -q "^$" && warn "MASSIVE_API_KEY is empty — yfinance fallback active" || true
    _env_get FINNHUB_API_KEY | grep -q "^$" && warn "FINNHUB_API_KEY is empty" || true
    _env_get XAI_API_KEY     | grep -q "^$" && warn "XAI_API_KEY is empty — Grok AI tab + RA LLM proxy disabled" || true
    _env_get KRAKEN_API_KEY  | grep -q "^$" && warn "KRAKEN_API_KEY is empty — live trading disabled" || true
    _env_get GITHUB_TOKEN    | grep -q "^$" && warn "GITHUB_TOKEN is empty — RA repo sync disabled" || true
    _prompt_web_password "$env_file"
}

# =============================================================================
# Pre-flight checks
# =============================================================================

preflight_check() {
    header "Pre-flight Checks"
    local errors=0

    if ! docker info > /dev/null 2>&1; then
        err "Docker daemon is not running"; ((errors++))
    else
        ok "Docker daemon is running"
    fi

    if ! docker compose version > /dev/null 2>&1; then
        err "Docker Compose V2 is not available"; ((errors++))
    else
        ok "Docker Compose available"
    fi

    local available_gb
    available_gb=$(df / | tail -1 | awk '{print int($4/1024/1024)}')
    if [ "$available_gb" -lt 10 ]; then
        err "Low disk space: ${available_gb}GB (need ≥ 10GB)"
        warn "Run: docker system prune -af --volumes"
        ((errors++))
    else
        ok "Disk space: ${available_gb}GB available"
    fi

    local port_ok=true
    for entry in "9000:QuestDB" "6379:Redis"; do
        local port="${entry%:*}" svc="${entry#*:}"
        local ctr
        ctr=$(docker ps --filter "publish=$port" --format "{{.Names}}" 2>/dev/null)
        if [ -n "$ctr" ]; then
            err "Port $port ($svc) already in use by: $ctr"
            port_ok=false
            ((errors++))
        fi
    done
    [ "$port_ok" = true ] && ok "Critical ports are free"

    local orphans
    orphans=$(docker ps -a --filter "name=fks_" --filter "status=exited" --format "{{.Names}}" 2>/dev/null | wc -l)
    [ "$orphans" -gt 0 ] && warn "Found $orphans orphan container(s) — will be cleaned on start"

    if [ $errors -gt 0 ]; then
        err "Pre-flight failed with $errors error(s)"
        info "Run './run.sh diagnose' for details or './run.sh force-clean' to reset"
        return 1
    fi
    ok "Pre-flight passed"
}

# =============================================================================
# CNN model files check (Ruby)
# =============================================================================

ensure_models() {
    local missing=0
    for f in "models/breakout_cnn_best_meta.json" \
             "models/feature_contract.json"; do
        if [ ! -f "$f" ]; then
            err "Missing model file: $f"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        if [ "${EXECUTION_MODE:-paper_trading}" = "live" ]; then
            err "CNN model files missing — cannot start in live execution mode"
            exit 1
        else
            warn "CNN model files missing — ML predictions unavailable until models are present"
            warn "  Run './run.sh retrain' or copy pre-trained models into ./models/"
        fi
    else
        ok "CNN model files present"
    fi
}

# =============================================================================
# Volume bootstrap
# =============================================================================

ensure_volumes() {
    # All volumes declared as `external: true` in docker-compose.yml must exist
    # before `docker compose up` runs — Docker won't create them automatically.
    local external_volumes=(
        fks_postgres_data
        fks_redis_data
        fks_questdb_data
        fks_prometheus_data
        fks_grafana_data
        fks_alertmanager_data
    )
    local created=0
    for vol in "${external_volumes[@]}"; do
        if ! docker volume inspect "$vol" > /dev/null 2>&1; then
            docker volume create "$vol" > /dev/null
            log "Created volume: $vol"
            created=$((created + 1))
        fi
    done
    if [ "$created" -gt 0 ]; then
        ok "Created $created missing external volume(s)"
    else
        ok "All external volumes present"
    fi
}

# =============================================================================
# TLS certificate bootstrap
# =============================================================================
# Tailscale handles all external HTTPS.  These certs are for internal
# service-to-service use only (e.g. future mTLS between containers).
# Nginx runs HTTP-only; the fkstrading.local HTTPS block has been removed.
# =============================================================================

ensure_tls_certs() {
    local certs_dir="infrastructure/certs"
    local srv_crt="${certs_dir}/server.crt"
    local srv_key="${certs_dir}/server.key"

    # Already present — quick expiry check, warn only
    if [ -f "${srv_crt}" ] && [ -f "${srv_key}" ]; then
        if command -v openssl &>/dev/null; then
            local expiry
            expiry=$(openssl x509 -noout -enddate -in "${srv_crt}" 2>/dev/null | cut -d= -f2 || true)
            ok "Internal TLS certs present${expiry:+ (expires: ${expiry})}"
        else
            ok "Internal TLS certs present"
        fi
        return 0
    fi

    if ! command -v openssl &>/dev/null; then
        warn "openssl not found — skipping internal cert generation"
        return 0
    fi

    log "Generating internal self-signed TLS cert in ${certs_dir}/ ..."
    mkdir -p "${certs_dir}"

    if openssl req -x509 -newkey rsa:2048 \
            -keyout "${srv_key}" \
            -out "${srv_crt}" \
            -days 3650 \
            -nodes \
            -subj "/CN=fks-internal/O=FKS Trading/C=CA" \
            2>/dev/null; then
        chmod 600 "${srv_key}"
        ok "Internal TLS cert generated in ${certs_dir}/"
    else
        warn "Internal cert generation failed — non-fatal, continuing"
    fi
}

cmd_ssl_local() {
    # Force-regenerate the internal service cert.
    local certs_dir="infrastructure/certs"
    local srv_crt="${certs_dir}/server.crt"
    local srv_key="${certs_dir}/server.key"

    header "SSL/TLS — Regenerate Internal Service Certificates"

    if ! command -v openssl &>/dev/null; then
        err "openssl not found — cannot generate TLS certificates"
        exit 1
    fi

    mkdir -p "${certs_dir}"
    log "Generating internal self-signed TLS cert (force) ..."

    if openssl req -x509 -newkey rsa:2048 \
            -keyout "${srv_key}" \
            -out "${srv_crt}" \
            -days 3650 \
            -nodes \
            -subj "/CN=fks-internal/O=FKS Trading/C=CA" \
            2>/dev/null; then
        chmod 600 "${srv_key}"
        ok "Cert written to ${certs_dir}/"
    else
        err "Certificate generation failed"
        exit 1
    fi
}

# =============================================================================
# OpenClaw base image helper
# =============================================================================

_ensure_openclaw_base() {
    local base_tag="${OPENCLAW_BASE_IMAGE:-nuniesmith/fks:openclaw-base}"

    # Already present locally — nothing to do
    if docker image inspect "$base_tag" >/dev/null 2>&1; then
        ok "openclaw-base image present locally: $base_tag"
        return 0
    fi

    # Try to pull from Docker Hub first (fast path for pre-pushed images)
    log "openclaw-base not found locally — trying to pull $base_tag ..."
    if docker pull "$base_tag" 2>/dev/null; then
        ok "Pulled $base_tag from registry"
        return 0
    fi

    # Fall back: build from source using build.sh
    warn "Could not pull $base_tag — building openclaw-base from source (may take a while) ..."
    local build_script="infrastructure/docker/services/openclaw/build.sh"
    if [ ! -f "$build_script" ]; then
        err "OpenClaw build script not found: $build_script"
        return 1
    fi

    if OPENCLAW_BASE_TAG="$base_tag" bash "$build_script" --base-only; then
        ok "Built openclaw-base image: $base_tag"
        return 0
    else
        err "OpenClaw base build failed"
        return 1
    fi
}

# =============================================================================
# Build commands
# =============================================================================

cmd_build_ruby() {
    header "Building FKS Python Services"

    # Step 1 — pre-cache the dep builder layer (venv + all pip deps, no source).
    # Produces nuniesmith/fks:python-base for CI/CD layer reuse.
    log "Step 1/2 — build python-base (dep cache layer)..."
    if $DC --profile base build base; then
        ok "base (nuniesmith/fks:python-base) built"
    else
        err "base build failed — aborting"
        return 1
    fi

    # Step 2 — build the consolidated ruby container.
    # Runs data + engine + factory + web under supervisord in a single container.
    log "Step 2/2 — build ruby..."
    if $DC build ruby; then
        ok "ruby (nuniesmith/fks:ruby) built"
    else
        err "ruby build failed"
        return 1
    fi
}

cmd_build_ruby_trainer() {
    header "Building Ruby Trainer (GPU)"
    if $DC --profile training build trainer; then
        ok "Trainer image built"
    else
        err "Trainer build failed"
        return 1
    fi
}

cmd_retrain() {
    header "3-Tier CNN Retrain (all tiers)"

    # Check trainer is running
    if ! $DC exec trainer python -c "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')" 2>/dev/null; then
        warn "Trainer container not running — starting it..."
        $DC --profile training up -d trainer
        log "Waiting for trainer to become healthy..."
        sleep 15
    fi

    local trainer_url="http://localhost:8200"
    local extra_args=""

    # Pass through any extra args (e.g. --tier 2, --epochs 100, etc.)
    if [ $# -gt 0 ]; then
        extra_args="$*"
    else
        extra_args="--tier all --continue-on-failure"
    fi

    log "Running: python scripts/run_per_group_training.py --trainer-url $trainer_url $extra_args"
    python3 scripts/run_per_group_training.py --trainer-url "$trainer_url" $extra_args
    local rc=$?

    if [ $rc -eq 0 ]; then
        ok "Retrain complete"
    else
        err "Retrain failed (exit code $rc)"
    fi
    return $rc
}

cmd_build_redis() {
    header "Building Custom Redis Image"
    if $DC build redis; then
        ok "Redis image built"
    else
        err "Redis build failed"
        return 1
    fi
}

cmd_build() {
    local mode="${1:-dev}"
    shift || true
    header "Building Images (${mode})"
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" build "$@"
    else
        $DC -f "$COMPOSE_FILE" build "$@"
    fi
    ok "Build complete"
}

# =============================================================================
# Start / up / down / restart
# =============================================================================

cmd_all() {
    # One-shot: build everything then bring up every profile
    header "FKS — Start Everything"

    setup_env_file
    echo ""

    ensure_models
    echo ""

    log "Stopping any existing FKS containers..."
    $DC --profile training --profile monitoring down --remove-orphans --timeout 10 2>/dev/null || true
    ok "Existing containers stopped"
    echo ""

    preflight_check
    echo ""

    ensure_volumes
    echo ""

    ensure_tls_certs
    echo ""

    # Build the shared python-base first, then all Ruby services, then everything else
    cmd_build_ruby
    echo ""

    log "Building trainer image (GPU)..."
    cmd_build_ruby_trainer || warn "Trainer build failed — continuing without GPU trainer"
    echo ""

    log "Building remaining service images..."
    _ensure_openclaw_base && $DC build || {
        warn "openclaw-base not available — building all services except fks_openclaw"
        warn "  To build OpenClaw manually: ./run.sh ra build openclaw"
        _svcs=$(docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null \
            | grep -v '^fks_openclaw$' | grep -v '^fks_openclaw_cli$' | tr '\n' ' ')
        $DC build $_svcs
    }
    echo ""

    log "Bringing up all services (core + training + monitoring profiles)..."
    $DC --profile training --profile monitoring up -d

    local ts_ip
    ts_ip=$(get_tailscale_ip)

    echo ""
    ok "Everything is up:"
    echo "    Dashboard:    http://${ts_ip}:8180"
    echo "    Data API:     http://${ts_ip}:8050"
    echo "    Trainer:      http://${ts_ip}:8200"
    echo "    Grafana:      http://${ts_ip}:3000"
    echo "    Prometheus:   http://${ts_ip}:9090"
    echo "    QuestDB:      http://${ts_ip}:9000"
    echo ""
    info "Logs:  docker compose logs -f"
    info "Stop:  ./run.sh down"
}

cmd_start() {
    local mode="${1:-dev}"
    shift || true
    header "Starting FKS (${mode} mode)"

    # Stop existing containers cleanly
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" down --timeout 10 --remove-orphans 2>/dev/null || true
    else
        $DC -f "$COMPOSE_FILE" down --timeout 10 --remove-orphans 2>/dev/null || true
    fi
    ok "Stopped existing containers"
    echo ""

    preflight_check
    echo ""

    setup_env_file
    echo ""

    ensure_volumes
    echo ""

    if [ "$mode" = "prod" ]; then
        header "Pulling Production Images"
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" pull "$@"
        ok "Images pulled"
        echo ""
        header "Starting Production Services"
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d "$@"
    else
        echo ""

        log "Building remaining service images..."
        $DC -f "$COMPOSE_FILE" build "$@"
        ok "Build complete"
        echo ""
        header "Starting Development Services"
        $DC -f "$COMPOSE_FILE" up -d "$@"
    fi

    ok "Services started"
    info "Waiting for health checks..."
    sleep 5
    cmd_status "$mode"

    echo ""
    ok "FKS Trading System is ready!"
    echo ""
    info "Access:"
    echo "  Web UI:      http://localhost"
    echo "  Grafana:     http://localhost/grafana/"
    echo "  QuestDB:     http://localhost:9000"
    echo "  Prometheus:  http://localhost:9090"
    echo ""
    info "Logs:   ./run.sh logs"
    info "Health: ./run.sh health"
}

cmd_up() {
    local mode="${1:-dev}"
    shift || true
    ensure_volumes
    echo ""
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" up -d "$@"
    else
        $DC -f "$COMPOSE_FILE" up -d "$@"
    fi
    ok "Services up"
}

cmd_down() {
    local mode="${1:-dev}"
    shift || true
    header "Stopping Services (${mode})"
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" \
            --profile training --profile monitoring \
            down "$@" --remove-orphans || true
    else
        $DC -f "$COMPOSE_FILE" \
            --profile training --profile monitoring \
            down "$@" --remove-orphans || true
    fi
    ok "All services stopped"
}

cmd_restart() {
    local mode="${1:-dev}"
    shift || true
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" restart "$@"
    else
        $DC -f "$COMPOSE_FILE" restart "$@"
    fi
    ok "Restarted"
}

cmd_logs() {
    local mode="${1:-dev}"
    shift || true
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" logs -f "$@"
    else
        $DC -f "$COMPOSE_FILE" logs -f "$@"
    fi
}

cmd_status() {
    local mode="${1:-dev}"
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" ps
    else
        $DC -f "$COMPOSE_FILE" ps
    fi
}

# =============================================================================
# Health
# =============================================================================

cmd_health() {
    header "Service Health"
    local services
    services=$($DC -f "$COMPOSE_FILE" ps --services 2>/dev/null)
    local any_unhealthy=false
    while IFS= read -r svc; do
        [ -z "$svc" ] && continue
        local state
        state=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' \
                "fks_${svc}" 2>/dev/null || echo "not-running")
        case "$state" in
            healthy)           ok  "$svc — healthy" ;;
            no-healthcheck)    info "$svc — running (no healthcheck)" ;;
            starting)          warn "$svc — starting…" ;;
            *)                 err  "$svc — $state"; any_unhealthy=true ;;
        esac
    done <<< "$services"
    # Also probe key HTTP health endpoints
    echo ""
    info "HTTP endpoint checks:"
    local http_ok=true
    for entry in \
        "http://localhost:8050/health:ruby-data" \
        "http://localhost:8080/health:ruby-web" \
        "http://localhost:8080/health:ruby-web"; do
        local url="${entry%:*}" label="${entry##*:}"
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null || echo "000")
        if [ "$http_code" = "200" ]; then
            ok  "$label — HTTP 200 ($url)"
        else
            warn "$label — HTTP $http_code ($url)"
            http_ok=false
        fi
    done

    [ "$any_unhealthy" = true ] || [ "$http_ok" = false ] && return 1 || return 0
}

# =============================================================================
# Ruby subsystem
# =============================================================================

cmd_test_ruby() {
    local svc="${1:-engine}"
    header "Running pytest in $svc"
    $DC -f "$COMPOSE_FILE" exec "$svc" \
        python -m pytest src/tests/ -x -q --tb=short
}

cmd_lint_ruby() {
    header "Linting FKS Python Services"
    $DC -f "$COMPOSE_FILE" exec "${1:-data}" \
        sh -c "ruff check src/ && ruff format --check src/"
}

# =============================================================================
# Check / Lint / Fmt / Test  (local — no Docker required)
# =============================================================================

cmd_check() {
    header "Full Check — Python & Rust"
    local errors=0

    cmd_lint_python  || ((errors++))
    echo ""
    cmd_typecheck    || ((errors++))
    echo ""
    cmd_test_python  || ((errors++))
    echo ""
    cmd_lint_rust    || ((errors++))
    echo ""
    cmd_test_rust    || ((errors++))

    echo ""
    if [ "$errors" -gt 0 ]; then
        err "$errors stage(s) failed"
        return 1
    fi
    ok "All checks passed ✨"
}

# ── Python ──────────────────────────────────────────────────────────────────

cmd_lint_python() {
    header "Python — ruff fix + lint + format"
    local rc=0

    log "ruff check --fix (auto-fix safe issues)..."
    if ruff check --fix src/ scripts/ 2>/dev/null; then
        ok "ruff auto-fix complete"
    else
        warn "ruff auto-fix applied changes (or found unfixable issues)"
    fi

    log "ruff check (lint)..."
    if ruff check src/ scripts/; then
        ok "ruff lint passed"
    else
        err "ruff lint found errors"
        rc=1
    fi

    log "ruff format --check..."
    if ruff format --check src/ scripts/; then
        ok "ruff format check passed"
    else
        warn "ruff format: files need formatting — run './run.sh fmt' to fix"
        rc=1
    fi

    return $rc
}

cmd_typecheck() {
    header "Python — mypy"
    if mypy; then
        ok "mypy passed"
    else
        err "mypy found type errors"
        return 1
    fi
}

cmd_test_python() {
    header "Python — pytest"
    if python3 -m pytest -x -q --tb=short; then
        ok "pytest passed"
    else
        err "pytest failed"
        return 1
    fi
}

# ── Rust ────────────────────────────────────────────────────────────────────

cmd_lint_rust() {
    header "Rust — cargo clippy"
    local rc=0

    log "cargo fmt --check..."
    if cargo fmt --all -- --check; then
        ok "cargo fmt check passed"
    else
        warn "cargo fmt: files need formatting — run './run.sh fmt' to fix"
        rc=1
    fi

    log "cargo clippy..."
    if cargo clippy --workspace --all-targets -- -D warnings; then
        ok "cargo clippy passed"
    else
        err "cargo clippy found warnings/errors"
        rc=1
    fi

    return $rc
}

cmd_test_rust() {
    header "Rust — cargo test"
    local errors=0

    if cargo test --workspace; then
        ok "cargo test passed"
    else
        err "cargo test failed"
        ((errors++))
    fi

    # Run neuromorphic tests a second time with the cuda feature enabled so
    # that Device::cuda_if_available() actually returns a CUDA device and the
    # full-size ViViT forward pass (and any other GPU-gated tests) are
    # exercised.
    #
    # Requirements (both must be satisfied):
    #   1. nvidia-smi can enumerate at least one GPU
    #   2. nvcc is on PATH — candle-core/cuda links through cudarc which
    #      invokes nvcc at build time (only the CUDA toolkit is needed,
    #      not the full driver SDK).
    #
    # Quick install on Ubuntu/Debian if nvcc is missing:
    #   sudo apt-get install cuda-toolkit-12-8   # match your driver version
    #   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
    local _has_gpu=0
    local _has_nvcc=0

    if command -v nvidia-smi >/dev/null 2>&1 && \
       nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        _has_gpu=1
    fi

    if command -v nvcc >/dev/null 2>&1; then
        _has_nvcc=1
    # Also check the standard toolkit path in case it isn't on PATH yet.
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        _has_nvcc=1
    fi

    if [ "$_has_gpu" -eq 1 ] && [ "$_has_nvcc" -eq 1 ]; then
        local gpu nvcc_ver
        gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        nvcc_ver="$(nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}' | tr -d ,)"
        header "Rust — cargo test (janus-neuromorphic, CUDA) [${gpu} / nvcc ${nvcc_ver}]"
        if cargo test -p janus-neuromorphic --features cuda; then
            ok "neuromorphic CUDA tests passed"
        else
            err "neuromorphic CUDA tests failed"
            ((errors++))
        fi
    elif [ "$_has_gpu" -eq 1 ] && [ "$_has_nvcc" -eq 0 ]; then
        warn "GPU detected but nvcc not found — skipping neuromorphic CUDA tests"
        warn "Install the CUDA toolkit to enable them:"
        warn "  sudo apt-get install cuda-toolkit-12-8"
        warn "  export PATH=/usr/local/cuda/bin:\$PATH"
    else
        info "No CUDA GPU detected — skipping neuromorphic CUDA test pass"
    fi

    return $errors
}

# ── Format (write mode) ────────────────────────────────────────────────────

cmd_fmt() {
    header "Auto-format — Python & Rust"

    log "ruff format..."
    ruff format src/ scripts/ && ok "ruff format done" || warn "ruff format had issues"

    log "ruff check --fix..."
    ruff check --fix src/ scripts/ 2>/dev/null && ok "ruff fix done" || warn "ruff fix had issues"

    log "cargo fmt..."
    cargo fmt --all && ok "cargo fmt done" || warn "cargo fmt had issues"

    ok "Formatting complete"
}

# ── Convenience aliases ─────────────────────────────────────────────────────

cmd_lint() {
    header "Lint — Python & Rust"
    local errors=0

    cmd_lint_python || ((errors++))
    echo ""
    cmd_typecheck   || ((errors++))
    echo ""
    cmd_lint_rust   || ((errors++))

    echo ""
    if [ "$errors" -gt 0 ]; then
        err "$errors lint stage(s) failed"
        return 1
    fi
    ok "All lints passed"
}

cmd_test_all() {
    header "Test — Python & Rust"
    local errors=0

    cmd_test_python || ((errors++))
    echo ""
    cmd_test_rust   || ((errors++))

    echo ""
    if [ "$errors" -gt 0 ]; then
        err "$errors test stage(s) failed"
        return 1
    fi
    ok "All tests passed"
}

cmd_logs_ruby() {
    header "FKS Python Service Logs"
    $DC -f "$COMPOSE_FILE" logs -f ruby
}

cmd_local_ruby() {
    header "Starting Ruby Services Locally (venv)"
    # Start infrastructure deps first
    $DC -f "$COMPOSE_FILE" up -d postgres redis
    info "Waiting for postgres and redis to be healthy…"
    sleep 5
    info "Starting Ruby data service locally on port 8000"
    info "  cd src/ruby && python -m uvicorn entrypoints.data.main:app --host 0.0.0.0 --port 8000 --reload"
    info "Start each service in a separate terminal with the above pattern (web: port 8080, engine: no port)"
    warn "Use 'docker compose up -d ruby' to run via Docker instead"
}

# =============================================================================
# Shell
# =============================================================================

cmd_shell() {
    local mode="${1:-dev}"
    local svc="${2:-data}"
    if [ "$mode" = "prod" ]; then
        $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" exec "$svc" bash \
            || $DC -f "$COMPOSE_FILE" -f "$PROD_COMPOSE_FILE" exec "$svc" sh
    else
        $DC -f "$COMPOSE_FILE" exec "$svc" bash \
            || $DC -f "$COMPOSE_FILE" exec "$svc" sh
    fi
}

# =============================================================================
# Cleanup
# =============================================================================

cmd_clean() {
    header "Cleaning Docker Resources + Build Caches"
    docker container prune -f
    docker image prune -f
    ok "Cleaned stopped containers and dangling images"
    log "Cleaning Python build caches..."
    find . -name ".mypy_cache"  -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "__pycache__"  -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".ruff_cache"  -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.egg-info"   -type d -exec rm -rf {} + 2>/dev/null || true
    ok "Cleaned Python caches (.mypy_cache, __pycache__, .pytest_cache, .ruff_cache, *.egg-info)"
}

cmd_force_clean() {
    header "⚠️  Force Clean — Removing ALL FKS Resources"
    warn "This will destroy all FKS containers, images, and volumes."
    echo -n "Type 'yes' to confirm: "
    read -r confirm
    if [ "$confirm" != "yes" ]; then
        info "Cancelled"
        return
    fi
    $DC -f "$COMPOSE_FILE" --profile training --profile monitoring \
        down --volumes --remove-orphans --timeout 5 2>/dev/null || true
    docker ps -a --filter "name=fks_" -q | xargs -r docker rm -f || true
    docker images --filter "reference=nuniesmith/fks*" -q | xargs -r docker rmi -f || true
    docker images --filter "reference=fks:*" -q          | xargs -r docker rmi -f || true
    docker network ls --filter "name=fks" -q | xargs -r docker network rm || true
    docker network prune -f || true
    ok "Force clean complete"
}

cmd_network_cleanup() {
    header "Network Cleanup"
    $DC -f "$COMPOSE_FILE" down --remove-orphans --timeout 5 2>/dev/null || true
    docker ps -a --filter "name=fks_" -q | xargs -r docker rm -f || true
    docker network ls --filter "name=fks" -q | xargs -r docker network rm || true
    docker network prune -f || true
    ok "Network cleanup complete"
}

cmd_setup_kernel() {
    header "Host Kernel Tuning (sysctl)"
    local conf_src="infrastructure/configs/99-fks.sysctl.conf"
    local conf_dst="/etc/sysctl.d/99-fks.conf"

    if [ ! -f "$conf_src" ]; then
        err "Config not found: $conf_src"
        return 1
    fi

    info "Installing $conf_src → $conf_dst"
    if sudo cp "$conf_src" "$conf_dst"; then
        ok "Config installed"
    else
        err "sudo cp failed — re-run with sudo or copy manually:"
        info "  sudo cp $conf_src $conf_dst"
        return 1
    fi

    info "Applying settings (sudo sysctl --system)..."
    sudo sysctl --system 2>&1 | grep -E "fks|file-max|somaxconn|overcommit|inotify" || true

    echo ""
    ok "Kernel parameters applied:"
    for key in fs.file-max net.core.somaxconn vm.overcommit_memory \
                fs.inotify.max_user_instances fs.inotify.max_user_watches; do
        local val; val=$(sysctl -n "$key" 2>/dev/null || echo "unknown")
        echo "  $key = $val"
    done
    echo ""
    info "Settings will persist across reboots via $conf_dst"
}

cmd_diagnose() {
    header "System Diagnostics"
    echo "Docker version:"; docker version --format 'Client: {{.Client.Version}}  Server: {{.Server.Version}}' 2>/dev/null || docker version
    echo ""
    echo "Running containers:"; docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "FKS images:"; docker images --filter "reference=nuniesmith/fks*" --filter "reference=fks:*"
    echo ""
    echo "Volumes:"; docker volume ls --filter "name=fks"
    echo ""
    echo "Networks:"; docker network ls --filter "name=fks"
    echo ""
    echo "Port usage (selected):"
    for port in 80 443 5432 5433 6379 6380 8050 8180 8200 9000 9090 9095 3000 3010; do
        local owner; owner=$(ss -tlnp 2>/dev/null | grep ":${port} " | awk '{print $NF}' | head -1 || true)
        [ -n "$owner" ] && echo "  :$port — $owner" || true
    done
    echo ""
    echo "Disk:"; df -h /
}

# =============================================================================
# web-hash-password
# =============================================================================

cmd_web_hash_password() {
    header "Generate Web Dashboard Password Hash"
    echo ""
    info "Generates a bcrypt hash to set as WEB_PASSWORD_HASH in .env"
    echo ""

    local password=""
    if [ -t 0 ]; then
        printf "Enter password: "
        stty -echo 2>/dev/null || true
        read -r password
        stty echo 2>/dev/null || true
        echo ""
        echo ""
        printf "Confirm password: "
        stty -echo 2>/dev/null || true
        local confirm=""
        read -r confirm
        stty echo 2>/dev/null || true
        echo ""
        if [ "$password" != "$confirm" ]; then
            err "Passwords do not match"
            exit 1
        fi
    else
        read -r password
    fi

    if [ -z "$password" ]; then
        err "Password cannot be empty"
        exit 1
    fi

    local hash
    hash=$(_generate_bcrypt_hash "$password")

    if [ -n "$hash" ]; then
        echo ""
        ok "Password hash generated:"
        echo ""
        echo "    WEB_PASSWORD_HASH=${hash}"
        echo ""
        info "Add the line above to your .env file, or run:"
        info "  ./run.sh setup-env   (prompts for password automatically)"
    else
        err "Failed to generate hash — install bcrypt: pip install bcrypt"
        exit 1
    fi
}

# =============================================================================
# Usage
# =============================================================================

show_usage() {
    cat << EOF

${CYAN}FKS — Unified Management Script${NC}

${BLUE}Usage:${NC} ./run.sh <command> [options]

${BLUE}Quick start:${NC}
  ./run.sh                        Run full check (lint + type-check + test)
  ./run.sh all                    Build everything and start all services

${BLUE}Service management:${NC}
  ./run.sh start [prod]           Env setup → build → start
  ./run.sh up [prod] [svcs...]    Start (already built) services
  ./run.sh down [prod] [-v]       Stop services  (-v removes volumes)
  ./run.sh restart [svcs...]      Restart services
  ./run.sh logs [svc]             Follow logs (all services or one)
  ./run.sh status                 Show container status
  ./run.sh health                 Health check all services

${BLUE}Build:${NC}
  ./run.sh build [prod] [svcs...] Build images
  ./run.sh build-ruby-trainer     Build GPU trainer image
  ./run.sh build-redis            Build custom Redis image

${BLUE}Training:${NC}
  ./run.sh retrain                  Run 3-tier CNN retrain (default: --tier all)
  ./run.sh retrain --tier 2         Run only tier 2 (per-group) retrain

${BLUE}Ruby subsystem:${NC}
  ./run.sh ruby build             Build all Ruby images
  ./run.sh ruby build-trainer     Build trainer image
  ./run.sh ruby test [svc]        Run pytest (default: engine)
  ./run.sh ruby logs [svc]        Follow FKS Python service logs
  ./run.sh ruby shell [svc]       Open shell (default: data)
  ./run.sh ruby restart [svcs...] Restart Ruby services
  ./run.sh ruby up [svcs...]      Start Ruby services
  ./run.sh ruby down [svcs...]    Stop Ruby services

${BLUE}Environment:${NC}
  ./run.sh setup-env              Generate or validate .env, fill secrets, prompt for API keys

${BLUE}Code quality (local — no Docker):${NC}
  ./run.sh check                  Full check: lint + type-check + test (Python & Rust)
  ./run.sh lint                   Ruff + mypy + clippy (no tests)
  ./run.sh test                   pytest + cargo test
  ./run.sh fmt                    Auto-format Python (ruff) & Rust (cargo fmt)
  ./run.sh lint-python            Ruff check/fix only
  ./run.sh lint-rust              Cargo fmt --check + clippy only
  ./run.sh test-python            pytest only
  ./run.sh test-rust              cargo test only
  ./run.sh typecheck              mypy only

${BLUE}Utilities:${NC}
  ./run.sh shell <svc>            Open shell in a container
  ./run.sh web-hash-password      Generate bcrypt hash for WEB_PASSWORD_HASH
  ./run.sh test-ruby [svc]        Run pytest in a Ruby container (Docker)
  ./run.sh lint-ruby [svc]        Run ruff lint/format in a Ruby container (Docker)
  ./run.sh local-ruby             Start Ruby services locally (venv guide)
  ./run.sh clean                  Remove stopped containers + dangling images
  ./run.sh force-clean            ⚠️  Remove ALL FKS resources + volumes
  ./run.sh network-cleanup        Fix Docker network conflicts
  ./run.sh diagnose               Detailed system diagnostics
  ./run.sh help                   Show this message

${BLUE}RustAssistant / OpenClaw:${NC}
  ./run.sh ra up [--ollama]       Start RA + OpenClaw stack
  ./run.sh ra down                Stop RA stack
  ./run.sh ra status              Show RA service status
  ./run.sh ra logs [svc]          Tail RA logs
  ./run.sh ra health              Health check all RA services
  ./run.sh ra build [app|openclaw] Build RA images
  ./run.sh ra shell [svc]         Open shell (default: ra-app)
  ./run.sh ra test [args]         Run Rust tests (cargo test)
  ./run.sh ra ci                  Full CI: fmt + clippy + test
  ./run.sh ra db backup [path]    Dump RA database
  ./run.sh ra db shell            Open psql CLI
  ./run.sh ra db size             Show database size
  ./run.sh ra db migrate          Apply SQL migrations from src/sql/ra/
  ./run.sh openclaw <cmd>         Run OpenClaw CLI (passthrough)

${BLUE}Examples:${NC}
  ./run.sh                        # Run all checks (default action)
  ./run.sh check                  # Same as above
  ./run.sh fmt                    # Auto-format everything
  ./run.sh lint                   # Lint only (no tests)
  ./run.sh test                   # Tests only (no lint)
  ./run.sh all                    # Full build + start (first time / CI)
  ./run.sh start                  # Env → build → up (dev)
  ./run.sh start prod             # Env → pull → up (prod)
  ./run.sh ruby build             # Rebuild only Ruby images
  ./run.sh logs data              # Follow data logs
  ./run.sh down -v                # Stop and remove volumes
EOF
}

# =============================================================================
# Main
# =============================================================================

# Commands that need Docker vs commands that run locally
needs_docker() {
    case "$1" in
        check|lint|test|fmt|lint-python|lint-rust|test-python|test-rust|typecheck|help|--help|-h)
            return 1 ;;
        *)
            return 0 ;;
    esac
}

main() {
    # No args → run full check (no Docker needed)
    if [ $# -lt 1 ]; then
        cmd_check
        exit $?
    fi

    local command="$1"

    # Only require Docker for commands that actually use it
    if needs_docker "$command"; then
        if ! docker info > /dev/null 2>&1; then
            err "Docker is not running"
            exit 1
        fi
        if ! docker compose version > /dev/null 2>&1; then
            err "Docker Compose V2 is not available"
            exit 1
        fi
    fi

    shift || true

    # ── Cluster namespaces ──────────────────────────────────────────────────
    if [ "$command" = "ruby" ]; then
        local ruby_cmd="${1:-build}"
        shift || true
        case $ruby_cmd in
            build)          cmd_build_ruby ;;
            build-trainer)  cmd_build_ruby_trainer ;;
            test)           cmd_test_ruby "$@" ;;
            logs)           cmd_logs "dev" "${1:-web}" ;;
            shell)          cmd_shell "dev" "${1:-data}" ;;
            restart)        $DC -f "$COMPOSE_FILE" restart "${@:-ruby}" ;;
            up)             $DC -f "$COMPOSE_FILE" up -d "${@:-ruby}" ;;
            down)           $DC -f "$COMPOSE_FILE" stop "${@:-ruby}" ;;
            *)
                err "Unknown ruby command: $ruby_cmd"
                echo "Available: build, build-trainer, test, logs, shell, restart, up, down"
                exit 1
                ;;
        esac
        return 0
    fi

    # ── RA / OpenClaw namespace: ./run.sh ra <command> ──────────────────────
    if [ "$command" = "ra" ]; then
        local ra_cmd="${1:-status}"
        shift || true
        # RA services list (for targeted up/down/logs)
        local RA_SERVICES="fks_ra fks_openclaw fks_openclaw_cli"
        local RA_BUILD_SCRIPT="infrastructure/docker/services/openclaw/build.sh"
        local RA_PORT="3500"
        local OPENCLAW_PORT="18789"
        local TAILSCALE_IP="${TAILSCALE_IP:-127.0.0.1}"

        case $ra_cmd in
            up|start)
                header "Starting RustAssistant + OpenClaw stack"
                local ollama_flag=""
                local skip_build=false
                while [ $# -gt 0 ]; do
                    case "$1" in
                        --ollama)     ollama_flag="--profile ollama"; shift ;;
                        --skip-build) skip_build=true; shift ;;
                        *)            break ;;
                    esac
                done
                local ra_dc="$DC -f $COMPOSE_FILE $ollama_flag"
                if [ "$skip_build" = false ]; then
                    log "Building fks_ra image..."
                    $ra_dc build fks_ra 2>/dev/null || true
                fi
                if [ $# -gt 0 ]; then
                    $ra_dc up -d "$@"
                else
                    $ra_dc up -d $RA_SERVICES
                    if [ -n "$ollama_flag" ]; then
                        $ra_dc up -d fks_ra_ollama fks_ra_ollama_init
                    fi
                fi
                ok "RA stack started"
                echo ""
                info "Access points (bind: ${TAILSCALE_IP}):"
                echo "  RA API:            http://${TAILSCALE_IP}:${RA_PORT}/health"
                echo "  RA Proxy (OpenAI): http://${TAILSCALE_IP}:${RA_PORT}/v1/models"
                echo "  OpenClaw Gateway:  ws://${TAILSCALE_IP}:${OPENCLAW_PORT}"
                ;;
            down|stop)
                header "Stopping RustAssistant + OpenClaw stack"
                $DC -f "$COMPOSE_FILE" stop $RA_SERVICES fks_ra_ollama fks_ra_ollama_init 2>/dev/null || true
                ok "RA stack stopped"
                ;;
            restart)
                $DC -f "$COMPOSE_FILE" restart ${@:-$RA_SERVICES}
                ;;
            logs)
                $DC -f "$COMPOSE_FILE" logs -f --tail=100 ${@:-$RA_SERVICES}
                ;;
            status|ps)
                header "RustAssistant Service Status"
                $DC -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" \
                    fks_ra fks_ra_ollama fks_openclaw fks_openclaw_cli 2>/dev/null \
                    || $DC -f "$COMPOSE_FILE" ps
                ;;
            health)
                header "RustAssistant Health Checks"
                echo ""
                log "fks_ra (port ${RA_PORT})..."
                if curl -sf "http://${TAILSCALE_IP}:${RA_PORT}/health" 2>/dev/null; then
                    ok "fks_ra healthy"
                elif curl -sf "http://127.0.0.1:${RA_PORT}/health" 2>/dev/null; then
                    ok "fks_ra healthy (localhost)"
                else
                    err "fks_ra not responding"
                fi
                echo ""
                log "postgres (rustassistant db)..."
                if $DC -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "${POSTGRES_USER:-fks_user}" > /dev/null 2>&1; then
                    ok "postgres healthy (shared)"
                else
                    err "postgres not responding"
                fi
                echo ""
                log "redis (db 1)..."
                if $DC -f "$COMPOSE_FILE" exec -T redis redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q PONG; then
                    ok "redis healthy (shared)"
                else
                    err "redis not responding"
                fi
                echo ""
                log "fks_openclaw (port ${OPENCLAW_PORT})..."
                if curl -sf "http://127.0.0.1:${OPENCLAW_PORT}/healthz" > /dev/null 2>&1; then
                    ok "fks_openclaw healthy"
                else
                    err "fks_openclaw not responding"
                fi
                ;;
            build)
                local target="${1:-all}"
                shift 2>/dev/null || true
                case "$target" in
                    openclaw|oc)
                        header "Building OpenClaw image"
                        if [ -f "$RA_BUILD_SCRIPT" ]; then
                            bash "$RA_BUILD_SCRIPT" "$@"
                        else
                            warn "OpenClaw build script not found at $RA_BUILD_SCRIPT"
                            warn "Building via docker compose instead..."
                            $DC -f "$COMPOSE_FILE" build openclaw-gateway "$@"
                        fi
                        ;;
                    app|ra-app)
                        header "Building fks_ra image"
                        $DC -f "$COMPOSE_FILE" build fks_ra "$@"
                        ok "fks_ra image built"
                        ;;
                    all)
                        header "Building all RA images"
                        $DC -f "$COMPOSE_FILE" build fks_ra "$@"
                        if [ -f "$RA_BUILD_SCRIPT" ]; then
                            bash "$RA_BUILD_SCRIPT" "$@"
                        else
                            $DC -f "$COMPOSE_FILE" build fks_openclaw "$@"
                        fi
                        ok "All RA images built"
                        ;;
                    *)
                        $DC -f "$COMPOSE_FILE" build "$target" "$@"
                        ;;
                esac
                ;;
            shell)
                local svc="${1:-fks_ra}"
                log "Opening shell in $svc..."
                $DC -f "$COMPOSE_FILE" exec "$svc" /bin/bash 2>/dev/null \
                    || $DC -f "$COMPOSE_FILE" exec "$svc" /bin/sh 2>/dev/null \
                    || $DC -f "$COMPOSE_FILE" exec "$svc" sh
                ;;
            test)
                header "Running RA Rust tests"
                cargo test --manifest-path src/ra/Cargo.toml "$@"
                ;;
            ci)
                header "RA CI Pipeline (fmt → clippy → test)"
                log "Checking formatting..."
                cargo fmt --manifest-path src/ra/Cargo.toml -- --check && ok "Format OK" || err "Format failed"
                log "Running clippy..."
                cargo clippy --manifest-path src/ra/Cargo.toml --all-targets -- -D warnings && ok "Clippy OK" || err "Clippy failed"
                log "Running tests..."
                cargo test --manifest-path src/ra/Cargo.toml --lib --bins && ok "Tests OK" || err "Tests failed"
                ;;
            db)
                local db_cmd="${1:-help}"
                shift 2>/dev/null || true
                case "$db_cmd" in
                    backup)
                        local backup_dir="${1:-./backups}"
                        local ts; ts=$(date +%Y%m%d-%H%M%S)
                        mkdir -p "$backup_dir"
                        log "Dumping RA database..."
                        $DC -f "$COMPOSE_FILE" exec -T postgres pg_dump -U "${POSTGRES_USER:-fks_user}" rustassistant \
                            | gzip > "${backup_dir}/rustassistant-${ts}.sql.gz"
                        ok "Backup: ${backup_dir}/rustassistant-${ts}.sql.gz"
                        ;;
                    shell|psql)
                        $DC -f "$COMPOSE_FILE" exec postgres psql -U "${POSTGRES_USER:-fks_user}" -d rustassistant
                        ;;
                    size)
                        $DC -f "$COMPOSE_FILE" exec -T postgres psql -U "${POSTGRES_USER:-fks_user}" -d rustassistant \
                            -c "SELECT pg_size_pretty(pg_database_size('rustassistant')) AS db_size;"
                        ;;
                    migrate)
                        header "Running RA database migrations"
                        if $DC -f "$COMPOSE_FILE" exec -T postgres pg_isready -U "${POSTGRES_USER:-fks_user}" > /dev/null 2>&1; then
                            ok "postgres (shared) is ready"
                            log "Applying migrations from src/sql/ra/..."
                            local migration_count=0
                            for sql_file in src/sql/ra/*.sql; do
                                [ -f "$sql_file" ] || continue
                                local fname; fname=$(basename "$sql_file")
                                log "  Applying: $fname"
                                $DC -f "$COMPOSE_FILE" exec -T postgres \
                                    psql -U "${POSTGRES_USER:-fks_user}" -d rustassistant -f "/dev/stdin" < "$sql_file" 2>&1 \
                                    | grep -v "^$" | head -5
                                migration_count=$((migration_count + 1))
                            done
                            ok "Applied $migration_count migration file(s)"
                        else
                            err "postgres is not running — start with: ./run.sh up"
                        fi
                        ;;
                    *)
                        echo "RA database commands:"
                        echo "  ./run.sh ra db backup [path]   Dump RA database"
                        echo "  ./run.sh ra db shell           Open psql CLI"
                        echo "  ./run.sh ra db size            Show database size"
                        echo "  ./run.sh ra db migrate         Apply SQL migrations from src/sql/ra/"
                        ;;
                esac
                ;;
            *)
                err "Unknown ra command: $ra_cmd"
                echo ""
                echo "Available RA commands:"
                echo "  up [--ollama] [--skip-build]   Start RA + OpenClaw stack"
                echo "  down                           Stop RA stack"
                echo "  restart [svc...]               Restart RA services"
                echo "  logs [svc]                     Tail RA logs"
                echo "  status                         Show RA service status"
                echo "  health                         Health check all RA services"
                echo "  build [openclaw|app|all]       Build RA images"
                echo "  shell [svc]                    Open shell (default: fks_ra)"
                echo "  test [args]                    Run Rust tests"
                echo "  ci                             Full CI pipeline"
                echo "  db [backup|shell|size]         Database commands"
                exit 1
                ;;
        esac
        return 0
    fi

    # ── openclaw shortcut: ./run.sh openclaw <cmd> ──────────────────────────
    if [ "$command" = "openclaw" ] || [ "$command" = "oc" ]; then
        if [ $# -eq 0 ]; then
            info "OpenClaw CLI — pass any openclaw subcommand."
            echo ""
            echo "  ./run.sh openclaw --help"
            echo "  ./run.sh openclaw doctor"
            echo "  ./run.sh openclaw models status --plain"
            echo "  ./run.sh openclaw channels status"
            echo "  ./run.sh openclaw health"
            echo "  ./run.sh openclaw agent --message 'Hello' --to '#general'"
            return 0
        fi
        $DC -f "$COMPOSE_FILE" run --rm fks_openclaw_cli "$@"
        return 0
    fi

    # ── prod prefix: ./run.sh prod <command> ────────────────────────────────
    local mode="dev"
    if [ "$command" = "prod" ]; then
        mode="prod"
        command="${1:-start}"
        shift || true
    fi

    # ── Dispatch ────────────────────────────────────────────────────────────
    case $command in
        all)                cmd_all ;;
        start)              cmd_start "$mode" "$@" ;;
        up)                 cmd_up "$mode" "$@" ;;
        stop|down)          cmd_down "$mode" "$@" ;;
        restart)            cmd_restart "$mode" "$@" ;;
        logs)               cmd_logs "$mode" "$@" ;;
        status|ps)          cmd_status "$mode" ;;
        health)             cmd_health ;;
        build)              cmd_build "$mode" "$@" ;;
        build-ruby)         cmd_build_ruby ;;
        build-ruby-trainer) cmd_build_ruby_trainer ;;
        build-redis)        cmd_build_redis ;;
        retrain)            cmd_retrain "$@" ;;
        setup-env)          setup_env_file ;;
        generate-certs)     ensure_tls_certs ;;
        ssl-local)          cmd_ssl_local "$@" ;;
        shell|exec)         cmd_shell "$mode" "$@" ;;
        web-hash-password)  cmd_web_hash_password ;;
        check)              cmd_check ;;
        lint)               cmd_lint ;;
        test)               cmd_test_all ;;
        fmt|format)         cmd_fmt ;;
        lint-python)        cmd_lint_python ;;
        lint-rust)          cmd_lint_rust ;;
        test-python)        cmd_test_python ;;
        test-rust)          cmd_test_rust ;;
        typecheck|mypy)     cmd_typecheck ;;
        test-ruby)          cmd_test_ruby "$@" ;;
        lint-ruby)          cmd_lint_ruby "$@" ;;
        logs-ruby)          cmd_logs_ruby ;;
        local-ruby)         cmd_local_ruby ;;
        clean)              cmd_clean ;;
        force-clean)        cmd_force_clean ;;
        network-cleanup)    cmd_network_cleanup ;;
        setup-kernel)       cmd_setup_kernel ;;
        setup-hosts)        cmd_setup_hosts ;;
        diagnose|diag)      cmd_diagnose ;;
        help|--help|-h)     show_usage ;;
        *)
            err "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
