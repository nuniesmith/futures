#!/bin/sh
# ===================================================================
# Sol Scalper — Docker Entrypoint
# Reads LEVERAGE and MARGIN_MODE from .env (no hardcoded values).
# The venv is already on PATH from the Dockerfile, so no `poetry run`.
# ===================================================================
set -e

LEVERAGE="${LEVERAGE:-5}"
MARGIN_MODE="${MARGIN_MODE:-isolated}"

echo "╔══════════════════════════════════════════╗"
echo "║       KuCoin SOLUSDTM Scalper            ║"
echo "╚══════════════════════════════════════════╝"
echo "  Started    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Leverage   : ${LEVERAGE}x"
echo "  Margin     : ${MARGIN_MODE}"
echo "  Capital    : \$${CAPITAL:-20}"
echo "  TP target  : ${TP_PCT:-0.004}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# One-time exchange setup (idempotent — safe to run on every restart)
echo "⚙️  Applying leverage & margin mode to exchange..."
python -c "
import asyncio, ccxt.async_support as ccxt, os
from dotenv import load_dotenv
load_dotenv()
ex = ccxt.kucoinfutures({
    'apiKey':    os.getenv('KUCOIN_API_KEY'),
    'secret':    os.getenv('KUCOIN_API_SECRET'),
    'password':  os.getenv('KUCOIN_PASSPHRASE'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
mode = os.getenv('MARGIN_MODE', 'isolated')
lev  = int(os.getenv('LEVERAGE', '5'))
async def setup():
    try:
        await ex.set_margin_mode(mode, 'SOLUSDTM')
        await ex.set_leverage(lev, 'SOLUSDTM')
        print(f'  ✅ {mode.upper()} margin @ {lev}x applied')
    except Exception as e:
        print(f'  ⚠️  Setup note (likely already set): {e}')
    finally:
        await ex.close()
asyncio.run(setup())
" || echo "  ⚠️  Exchange setup skipped — continuing"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Launching main bot..."
exec python -m src.main
