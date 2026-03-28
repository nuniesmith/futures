#!/bin/sh
set -e

# =============================================================================
# Futures Trader Redis Entrypoint — Kernel Tuning & Startup
# =============================================================================
# This entrypoint runs as root (before the official redis entrypoint drops to
# the `redis` user) and attempts to apply kernel parameter tuning that Redis
# recommends for production workloads.
#
# The following sysctls are now set via docker-compose `sysctls:` and no longer
# need to be written here:
#   - vm.overcommit_memory = 1
#   - net.core.somaxconn = 1024
#
# This entrypoint still handles:
#
#   1. Disable Transparent Huge Pages (THP)
#      THP causes latency spikes and increased memory usage with Redis.
#      Redis logs a warning at startup if THP is enabled.
#      This is a /sys knob, not a sysctl, so it cannot be set via
#      docker-compose `sysctls:`. If the write fails (no privilege),
#      Redis will still start but may log a cosmetic warning.
# =============================================================================

echo "=== Futures Trader Redis Startup ==="
echo "Applying kernel parameter tuning..."

WARNINGS=0

# ---------------------------------------------------------------------------
# 1. Disable Transparent Huge Pages (THP)
# ---------------------------------------------------------------------------
disable_thp() {
    local thp_enabled="/sys/kernel/mm/transparent_hugepage/enabled"
    local thp_defrag="/sys/kernel/mm/transparent_hugepage/defrag"

    if [ ! -f "$thp_enabled" ]; then
        echo "✓  Transparent Huge Pages: not available (OK)"
        return
    fi

    local current
    current=$(cat "$thp_enabled" 2>/dev/null || echo "")

    # Check if already disabled — the active value is shown in [brackets]
    if echo "$current" | grep -q '\[never\]'; then
        echo "✓  Transparent Huge Pages already disabled"
        return
    fi

    local thp_ok=true

    if echo never > "$thp_enabled" 2>/dev/null; then
        echo "✓  THP enabled set to [never]"
    else
        echo "⚠  Could not disable THP (enabled)"
        thp_ok=false
    fi

    if [ -f "$thp_defrag" ]; then
        if echo never > "$thp_defrag" 2>/dev/null; then
            echo "✓  THP defrag set to [never]"
        else
            echo "⚠  Could not disable THP (defrag)"
            thp_ok=false
        fi
    fi

    if [ "$thp_ok" = "false" ]; then
        echo "   To fix, disable THP on the Docker host:"
        echo "   echo never > /sys/kernel/mm/transparent_hugepage/enabled"
        echo "   echo never > /sys/kernel/mm/transparent_hugepage/defrag"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# ---------------------------------------------------------------------------
# Apply all tuning
# ---------------------------------------------------------------------------
disable_thp

echo ""
if [ "$WARNINGS" -gt 0 ]; then
    echo "⚠  Kernel tuning completed with $WARNINGS warning(s)"
    echo "   Redis will still start, but may log a cosmetic THP warning."
    echo "   To silence it, disable THP on the host or grant the container"
    echo "   write access to /sys/kernel/mm/transparent_hugepage/."
else
    echo "✓  All kernel parameters tuned successfully"
fi
echo ""

# ---------------------------------------------------------------------------
# Delegate to the official Redis entrypoint
# ---------------------------------------------------------------------------
# The official redis:7-alpine image includes docker-entrypoint.sh which:
#   - Switches to the `redis` user (via gosu) if running as root
#   - Handles signal forwarding for graceful shutdown
#   - Passes through all arguments to redis-server
#
# We exec into it so PID 1 becomes redis-server (proper signal handling).
# ---------------------------------------------------------------------------
echo "Delegating to official Redis entrypoint..."
echo "=== Futures Trader Redis Startup Complete ==="
echo ""

exec docker-entrypoint.sh "$@"
