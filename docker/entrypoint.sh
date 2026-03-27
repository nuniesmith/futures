#!/bin/sh
# ===================================================================
# Sol Scalper — Docker Entrypoint
#
# Reads all config from .env — no hardcoded values.
# Leverage and margin mode are set once at startup via the exchange
# API inside main.py itself (idempotent, safe on every restart).
# ===================================================================
set -e

LEVERAGE="${LEVERAGE:-5}"
MARGIN_MODE="${MARGIN_MODE:-isolated}"
CAPITAL="${CAPITAL:-20}"
TP_PCT="${TP_PCT:-0.004}"

echo "╔══════════════════════════════════════════╗"
echo "║       KuCoin SOLUSDTM Scalper            ║"
echo "║       Ruby Wave Edition                  ║"
echo "╚══════════════════════════════════════════╝"
echo "  Started    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Symbol     : SOLUSDTM (perpetual)"
echo "  Leverage   : ${LEVERAGE}x"
echo "  Margin     : ${MARGIN_MODE}"
echo "  Capital    : \$${CAPITAL}"
echo "  TP base    : ${TP_PCT} (adaptive at runtime)"
echo "  Position   : One-Way mode (required)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "${KUCOIN_API_KEY}" ] || [ -z "${KUCOIN_API_SECRET}" ] || [ -z "${KUCOIN_PASSPHRASE}" ]; then
    echo "❌  ERROR: KUCOIN_API_KEY / KUCOIN_API_SECRET / KUCOIN_PASSPHRASE not set in .env"
    exit 1
fi

if [ -z "${DISCORD_WEBHOOK_URL}" ]; then
    echo "  Discord    : not configured (set DISCORD_WEBHOOK_URL in .env to enable)"
else
    echo "  Discord    : ✅ webhook configured"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Launching main bot..."

exec python -m src.main
