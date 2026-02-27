#!/usr/bin/env bash
# =============================================================================
# Futures Trading Co-Pilot — End-to-End Smoke Test (TASK-902)
# =============================================================================
# Run before every trading day to verify the full stack is healthy.
#
# Usage:
#   ./scripts/smoke_test.sh              # run all checks
#   ./scripts/smoke_test.sh --quick      # skip slow checks (SSE stream)
#   ./scripts/smoke_test.sh --verbose    # print response bodies
#
# Exit codes:
#   0  — all checks passed
#   1  — one or more checks failed
#
# Prerequisites:
#   - Docker Compose stack running: docker compose up -d --build
#   - curl, docker, jq (optional, for pretty JSON)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_SERVICE_URL="${DATA_SERVICE_URL:-http://localhost:8000}"
REDIS_CONTAINER="${REDIS_CONTAINER:-futures-redis-1}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-futures-postgres-1}"
ENGINE_CONTAINER="${ENGINE_CONTAINER:-futures-engine-1}"
DATA_CONTAINER="${DATA_CONTAINER:-futures-data-1}"

# Try alternate naming conventions (docker compose v2 uses project-service-N)
# We'll auto-detect below.

QUICK=false
VERBOSE=false

for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK=true ;;
        --verbose) VERBOSE=true ;;
        -h|--help)
            echo "Usage: $0 [--quick] [--verbose]"
            echo "  --quick    Skip slow checks (SSE stream hold test)"
            echo "  --verbose  Print response bodies on success"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Colors and formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

pass_check() {
    local name="$1"
    local detail="${2:-}"
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("${GREEN}✅ PASS${NC}  $name${detail:+ — $detail}")
    echo -e "${GREEN}✅ PASS${NC}  $name${detail:+ — $detail}"
}

fail_check() {
    local name="$1"
    local detail="${2:-}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("${RED}❌ FAIL${NC}  $name${detail:+ — $detail}")
    echo -e "${RED}❌ FAIL${NC}  $name${detail:+ — $detail}"
}

skip_check() {
    local name="$1"
    local detail="${2:-}"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    RESULTS+=("${YELLOW}⏭️  SKIP${NC}  $name${detail:+ — $detail}")
    echo -e "${YELLOW}⏭️  SKIP${NC}  $name${detail:+ — $detail}"
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

verbose_body() {
    if $VERBOSE; then
        echo -e "    ${CYAN}Response:${NC} $1"
    fi
}

# ---------------------------------------------------------------------------
# Auto-detect container names
# ---------------------------------------------------------------------------
detect_container() {
    local service_name="$1"
    local default_name="$2"

    # Try the default name first
    if docker inspect "$default_name" &>/dev/null 2>&1; then
        echo "$default_name"
        return
    fi

    # Try common docker compose naming patterns
    for pattern in \
        "futures-${service_name}-1" \
        "futures_${service_name}_1" \
        "futures-${service_name}1" \
        ; do
        if docker inspect "$pattern" &>/dev/null 2>&1; then
            echo "$pattern"
            return
        fi
    done

    # Try docker compose ps to find it
    local found
    found=$(docker compose ps --format '{{.Name}}' 2>/dev/null | grep -i "$service_name" | head -1) || true
    if [ -n "$found" ]; then
        echo "$found"
        return
    fi

    # Fall back to default
    echo "$default_name"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Futures Trading Co-Pilot — Smoke Test${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "  Target:  ${DATA_SERVICE_URL}"
echo -e "  Mode:    $(if $QUICK; then echo 'quick'; else echo 'full'; fi)"
echo -e "  Time:    $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Detect actual container names
info "Detecting container names..."
REDIS_CONTAINER=$(detect_container "redis" "$REDIS_CONTAINER")
POSTGRES_CONTAINER=$(detect_container "postgres" "$POSTGRES_CONTAINER")
ENGINE_CONTAINER=$(detect_container "engine" "$ENGINE_CONTAINER")
DATA_CONTAINER=$(detect_container "data" "$DATA_CONTAINER")
info "  Redis:    $REDIS_CONTAINER"
info "  Postgres: $POSTGRES_CONTAINER"
info "  Engine:   $ENGINE_CONTAINER"
info "  Data:     $DATA_CONTAINER"
echo ""

# ===========================================================================
# CHECK 1: Docker containers are running
# ===========================================================================
echo -e "${BOLD}--- Docker Containers ---${NC}"

check_container_running() {
    local name="$1"
    local label="$2"
    if docker inspect --format='{{.State.Running}}' "$name" 2>/dev/null | grep -q "true"; then
        local status
        status=$(docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null)
        pass_check "$label container running" "status=$status"
    else
        fail_check "$label container running" "container not found or not running"
    fi
}

check_container_running "$POSTGRES_CONTAINER" "Postgres"
check_container_running "$REDIS_CONTAINER" "Redis"
check_container_running "$DATA_CONTAINER" "Data-service"
check_container_running "$ENGINE_CONTAINER" "Engine"

# ===========================================================================
# CHECK 2: Postgres healthcheck
# ===========================================================================
echo ""
echo -e "${BOLD}--- Database Health ---${NC}"

if docker exec "$POSTGRES_CONTAINER" pg_isready -U futures_user -d futures_db &>/dev/null 2>&1; then
    pass_check "Postgres pg_isready"
else
    fail_check "Postgres pg_isready" "pg_isready returned non-zero or container not found"
fi

# ===========================================================================
# CHECK 3: Redis healthcheck
# ===========================================================================
if docker exec "$REDIS_CONTAINER" redis-cli ping 2>/dev/null | grep -q "PONG"; then
    pass_check "Redis ping → PONG"
else
    fail_check "Redis ping" "did not receive PONG"
fi

# Check for engine keys in Redis
ENGINE_FOCUS_EXISTS=$(docker exec "$REDIS_CONTAINER" redis-cli exists engine:daily_focus 2>/dev/null || echo "0")
if echo "$ENGINE_FOCUS_EXISTS" | grep -q "1"; then
    pass_check "Redis has engine:daily_focus key"
else
    # Not a hard failure — engine may not have run yet
    skip_check "Redis engine:daily_focus key" "key not present (engine may not have run yet)"
fi

ENGINE_STATUS_EXISTS=$(docker exec "$REDIS_CONTAINER" redis-cli exists engine:status 2>/dev/null || echo "0")
if echo "$ENGINE_STATUS_EXISTS" | grep -q "1"; then
    pass_check "Redis has engine:status key"
else
    skip_check "Redis engine:status key" "key not present"
fi

# ===========================================================================
# CHECK 4: Data-service /health endpoint
# ===========================================================================
echo ""
echo -e "${BOLD}--- Data Service Endpoints ---${NC}"

HEALTH_RESPONSE=$(curl -sf -m 10 "${DATA_SERVICE_URL}/health" 2>/dev/null) || HEALTH_RESPONSE=""
if [ -n "$HEALTH_RESPONSE" ]; then
    pass_check "GET /health returns 200"
    verbose_body "$HEALTH_RESPONSE"
else
    fail_check "GET /health" "no response or non-200 status"
fi

# ===========================================================================
# CHECK 5: Dashboard loads (GET /)
# ===========================================================================
DASHBOARD_RESPONSE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/" 2>/dev/null) || DASHBOARD_RESPONSE="000"
if [ "$DASHBOARD_RESPONSE" = "200" ]; then
    pass_check "GET / (dashboard) returns 200"
else
    fail_check "GET / (dashboard)" "HTTP $DASHBOARD_RESPONSE"
fi

# Check that dashboard contains expected HTML markers
DASHBOARD_BODY=$(curl -sf -m 10 "${DATA_SERVICE_URL}/" 2>/dev/null) || DASHBOARD_BODY=""
if echo "$DASHBOARD_BODY" | grep -q "Futures Trading Co-Pilot"; then
    pass_check "Dashboard contains title"
else
    fail_check "Dashboard title" "expected 'Futures Trading Co-Pilot' in response"
fi

if echo "$DASHBOARD_BODY" | grep -q "sse-connect"; then
    pass_check "Dashboard has SSE connection"
else
    fail_check "Dashboard SSE" "no sse-connect attribute found"
fi

# ===========================================================================
# CHECK 6: API /api/info endpoint
# ===========================================================================
INFO_RESPONSE=$(curl -sf -m 10 "${DATA_SERVICE_URL}/api/info" 2>/dev/null) || INFO_RESPONSE=""
if echo "$INFO_RESPONSE" | grep -q "futures-data-service"; then
    pass_check "GET /api/info returns service info"
    verbose_body "$INFO_RESPONSE"
else
    fail_check "GET /api/info" "unexpected response"
fi

# ===========================================================================
# CHECK 7: API /api/focus endpoint
# ===========================================================================
FOCUS_RESPONSE=$(curl -sf -m 10 "${DATA_SERVICE_URL}/api/focus" 2>/dev/null) || FOCUS_RESPONSE=""
if [ -n "$FOCUS_RESPONSE" ] && [ "$FOCUS_RESPONSE" != "null" ]; then
    # Check it's valid JSON
    if echo "$FOCUS_RESPONSE" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        pass_check "GET /api/focus returns valid JSON"
        verbose_body "$(echo "$FOCUS_RESPONSE" | head -c 200)..."
    else
        fail_check "GET /api/focus" "response is not valid JSON"
    fi
else
    skip_check "GET /api/focus" "empty or null (engine may not have computed focus yet)"
fi

# ===========================================================================
# CHECK 8: API /api/focus/html returns HTML
# ===========================================================================
FOCUS_HTML_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/focus/html" 2>/dev/null) || FOCUS_HTML_CODE="000"
if [ "$FOCUS_HTML_CODE" = "200" ]; then
    pass_check "GET /api/focus/html returns 200"
else
    fail_check "GET /api/focus/html" "HTTP $FOCUS_HTML_CODE"
fi

# ===========================================================================
# CHECK 9: API /api/time returns time fragment
# ===========================================================================
TIME_RESPONSE=$(curl -sf -m 10 "${DATA_SERVICE_URL}/api/time" 2>/dev/null) || TIME_RESPONSE=""
if [ -n "$TIME_RESPONSE" ]; then
    pass_check "GET /api/time returns content"
else
    fail_check "GET /api/time" "empty response"
fi

# ===========================================================================
# CHECK 10: API /api/positions/html
# ===========================================================================
POS_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/positions/html" 2>/dev/null) || POS_CODE="000"
if [ "$POS_CODE" = "200" ]; then
    pass_check "GET /api/positions/html returns 200"
else
    fail_check "GET /api/positions/html" "HTTP $POS_CODE"
fi

# ===========================================================================
# CHECK 11: API /api/risk/html
# ===========================================================================
RISK_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/risk/html" 2>/dev/null) || RISK_CODE="000"
if [ "$RISK_CODE" = "200" ]; then
    pass_check "GET /api/risk/html returns 200"
else
    fail_check "GET /api/risk/html" "HTTP $RISK_CODE"
fi

# ===========================================================================
# CHECK 12: API /api/grok/html
# ===========================================================================
GROK_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/grok/html" 2>/dev/null) || GROK_CODE="000"
if [ "$GROK_CODE" = "200" ]; then
    pass_check "GET /api/grok/html returns 200"
else
    fail_check "GET /api/grok/html" "HTTP $GROK_CODE"
fi

# ===========================================================================
# CHECK 13: SSE /sse/health endpoint
# ===========================================================================
echo ""
echo -e "${BOLD}--- SSE Subsystem ---${NC}"

SSE_HEALTH=$(curl -sf -m 10 "${DATA_SERVICE_URL}/sse/health" 2>/dev/null) || SSE_HEALTH=""
if echo "$SSE_HEALTH" | grep -q '"status"'; then
    pass_check "GET /sse/health returns status"
    verbose_body "$SSE_HEALTH"
else
    fail_check "GET /sse/health" "unexpected response"
fi

# ===========================================================================
# CHECK 14: SSE stream delivers events (connect, read first event, disconnect)
# ===========================================================================
if $QUICK; then
    skip_check "SSE stream test" "skipped in --quick mode"
else
    # Connect to SSE, read for up to 8 seconds, capture both headers and body.
    # NOTE: curl exits with code 28 on timeout, which is expected for SSE streams
    # that never close.  We write body to a tmpfile so the `|| true` doesn't
    # swallow the captured output (command substitution + || resets $()).
    SSE_HDRFILE=$(mktemp)
    SSE_BODYFILE=$(mktemp)
    curl -s -m 8 -N -D "$SSE_HDRFILE" -o "$SSE_BODYFILE" "${DATA_SERVICE_URL}/sse/dashboard" 2>/dev/null || true
    SSE_OUTPUT=$(cat "$SSE_BODYFILE" 2>/dev/null) || SSE_OUTPUT=""
    SSE_HEADERS=$(cat "$SSE_HDRFILE" 2>/dev/null) || SSE_HEADERS=""
    rm -f "$SSE_HDRFILE" "$SSE_BODYFILE"

    if echo "$SSE_OUTPUT" | grep -q "event:"; then
        # Count distinct event types received
        EVENT_TYPES=$(echo "$SSE_OUTPUT" | grep "^event:" | sort -u | wc -l)
        pass_check "SSE stream delivers events" "${EVENT_TYPES} event type(s) in 8s"
    else
        fail_check "SSE stream" "no 'event:' lines received in 8 seconds"
    fi

    # Check that SSE returns correct Content-Type header (from captured headers)
    if echo "$SSE_HEADERS" | grep -qi "text/event-stream"; then
        pass_check "SSE Content-Type is text/event-stream"
    else
        SSE_CT_VALUE=$(echo "$SSE_HEADERS" | grep -i "content-type" | head -1 | tr -d '\r')
        fail_check "SSE Content-Type" "expected text/event-stream, got: ${SSE_CT_VALUE:-<empty>}"
    fi
fi

# ===========================================================================
# CHECK 15: Engine health file
# ===========================================================================
echo ""
echo -e "${BOLD}--- Engine Health ---${NC}"

ENGINE_HEALTH=$(docker exec "$ENGINE_CONTAINER" cat /tmp/engine_health.json 2>/dev/null) || ENGINE_HEALTH=""
if [ -n "$ENGINE_HEALTH" ]; then
    if echo "$ENGINE_HEALTH" | grep -q '"healthy": true'; then
        ESESSION=$(echo "$ENGINE_HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session','?'))" 2>/dev/null) || ESESSION="?"
        pass_check "Engine health file (healthy=true)" "session=$ESESSION"
        verbose_body "$ENGINE_HEALTH"
    else
        fail_check "Engine health file" "healthy is not true: $ENGINE_HEALTH"
    fi
else
    fail_check "Engine health file" "/tmp/engine_health.json not found in container"
fi

# Check engine container logs for errors (last 50 lines)
# Filter out false positives like "Errors: 0", "0 errors", summary lines
ENGINE_ERRORS=$(docker logs --tail 50 "$ENGINE_CONTAINER" 2>&1 \
    | grep -i "error\|exception\|traceback" \
    | grep -icv "errors\?: *0\|0 *errors\?\|exc_info=\|error-like\|no.error" || true)
if [ "$ENGINE_ERRORS" -lt 3 ]; then
    pass_check "Engine logs (low error count)" "${ENGINE_ERRORS} error-like lines in last 50"
else
    fail_check "Engine logs" "${ENGINE_ERRORS} error-like lines in last 50 log lines"
fi

# ===========================================================================
# CHECK 16: Data-service container logs
# ===========================================================================
DATA_ERRORS=$(docker logs --tail 50 "$DATA_CONTAINER" 2>&1 \
    | grep -i "error\|exception\|traceback" \
    | grep -icv "errors\?: *0\|0 *errors\?\|exc_info=\|error-like\|no.error" || true)
if [ "$DATA_ERRORS" -lt 3 ]; then
    pass_check "Data-service logs (low error count)" "${DATA_ERRORS} error-like lines in last 50"
else
    fail_check "Data-service logs" "${DATA_ERRORS} error-like lines in last 50 log lines"
fi

# ===========================================================================
# CHECK 17: No-trade endpoint
# ===========================================================================
echo ""
echo -e "${BOLD}--- Additional Endpoints ---${NC}"

NOTRADE_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/no-trade" 2>/dev/null) || NOTRADE_CODE="000"
if [ "$NOTRADE_CODE" = "200" ]; then
    pass_check "GET /api/no-trade returns 200"
else
    fail_check "GET /api/no-trade" "HTTP $NOTRADE_CODE"
fi

# ===========================================================================
# CHECK 18: Alerts endpoint
# ===========================================================================
ALERTS_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/api/alerts/html" 2>/dev/null) || ALERTS_CODE="000"
if [ "$ALERTS_CODE" = "200" ]; then
    pass_check "GET /api/alerts/html returns 200"
else
    fail_check "GET /api/alerts/html" "HTTP $ALERTS_CODE"
fi

# ===========================================================================
# CHECK 19: Positions API
# ===========================================================================
POSITIONS_CODE=$(curl -sf -m 10 -o /dev/null -w '%{http_code}' "${DATA_SERVICE_URL}/positions/" 2>/dev/null) || POSITIONS_CODE="000"
if [ "$POSITIONS_CODE" = "200" ]; then
    pass_check "GET /positions/ returns 200"
else
    # Positions may return 404 if no data — that's acceptable
    if [ "$POSITIONS_CODE" = "404" ]; then
        skip_check "GET /positions/" "returned 404 (no position data yet)"
    else
        fail_check "GET /positions/" "HTTP $POSITIONS_CODE"
    fi
fi

# ===========================================================================
# Summary
# ===========================================================================
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  SMOKE TEST SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

for r in "${RESULTS[@]}"; do
    echo -e "  $r"
done

echo ""
echo -e "${BOLD}───────────────────────────────────────────────────────────${NC}"
echo -e "  Total:   ${TOTAL}"
echo -e "  ${GREEN}Passed:  ${PASS_COUNT}${NC}"
echo -e "  ${RED}Failed:  ${FAIL_COUNT}${NC}"
echo -e "  ${YELLOW}Skipped: ${SKIP_COUNT}${NC}"
echo -e "${BOLD}───────────────────────────────────────────────────────────${NC}"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ""
    echo -e "  ${GREEN}${BOLD}🎉 ALL CHECKS PASSED — Ready for trading!${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "  ${RED}${BOLD}⚠️  ${FAIL_COUNT} CHECK(S) FAILED — Review before trading.${NC}"
    echo ""
    exit 1
fi
