"""
KuCoin SOLUSDTM Perpetual Scalper
- EMA crossover + order-book imbalance signal
- Position stacking: up to 3 adds in same direction (1% risk each, 3% max)
- Static TP % target per trade
- Discord webhook alerts: opens, closes (with P&L %), errors, Optuna updates
- Optuna auto-optimization every 20 min on live streamed data (no storage)
"""

import asyncio
import os
import time
from collections import deque

import aiohttp
import ccxt.async_support as ccxt
import optuna
import pandas as pd
from dotenv import load_dotenv

# Silence Optuna logs (keeps console clean)
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()

# ========================= CONFIG =========================
SYMBOL = "SOLUSDTM"
LEVERAGE = int(os.getenv("LEVERAGE", "5"))  # 5–75
MARGIN_MODE = os.getenv("MARGIN_MODE", "isolated")  # isolated | cross
CAPITAL = float(os.getenv("CAPITAL", "20.0"))  # USD in futures wallet
RISK_PER_ADD = 0.01  # 1% per position add
MAX_STACK = 3  # max 3 adds in one direction
TP_PCT = float(os.getenv("TP_PCT", "0.004"))  # 0.4% take-profit target
ADD_THRESHOLD = 0.002  # price must move 0.2% against us to add
OPTUNA_INTERVAL = 1200  # re-optimize every 20 min

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")

API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

# ========================= STATE =========================
trades: deque = deque(maxlen=12000)
orderbook: dict = {"bids": [], "asks": []}

# Position stack tracking (in-memory only)
stack: dict = {
    "direction": None,  # "buy" | "sell" | None
    "count": 0,  # number of adds made (0–3)
    "prices": [],  # fill price of each add
    "sizes": [],  # contract count of each add
}

best_params: dict = {
    "fast": 8,
    "slow": 21,
    "imbalance_thresh": 1.42,
    "sl_pct": 0.0035,
}

# ========================= EXCHANGE =========================
exchange = ccxt.kucoinfutures(
    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)


# ========================= DISCORD =========================
COLORS = {
    "green": 0x00C853,
    "red": 0xD50000,
    "yellow": 0xFFD600,
    "blue": 0x2196F3,
    "grey": 0x9E9E9E,
}


async def discord(title: str, body: str, color: str = "blue") -> None:
    """Send an embed message to Discord. Silent no-op if webhook not configured."""
    if not DISCORD_WEBHOOK:
        return
    payload = {
        "embeds": [
            {
                "title": title,
                "description": body,
                "color": COLORS.get(color, COLORS["blue"]),
                "footer": {"text": f"SOL Scalper | {SYMBOL} | {LEVERAGE}x {MARGIN_MODE}"},
            }
        ]
    }
    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                DISCORD_WEBHOOK, json=payload, timeout=aiohttp.ClientTimeout(total=5)
            )
            if resp.status not in (200, 204):
                print(f"Discord HTTP {resp.status}")
    except Exception as exc:
        print(f"Discord error: {exc}")


# ========================= STACK HELPERS =========================
def stack_avg_price() -> float:
    if not stack["prices"]:
        return 0.0
    total_size = sum(stack["sizes"])
    return sum(p * s for p, s in zip(stack["prices"], stack["sizes"])) / total_size


def stack_total_size() -> float:
    return sum(stack["sizes"])


def stack_clear() -> None:
    stack["direction"] = None
    stack["count"] = 0
    stack["prices"].clear()
    stack["sizes"].clear()


# ========================= DATA STREAM =========================
async def handle_ws() -> None:
    global orderbook
    print("📡 Connecting KuCoin Futures WebSocket...")
    while True:
        try:
            trade_data = await exchange.watch_trades(SYMBOL)
            for t in trade_data:
                trades.append(
                    {
                        "timestamp": t["timestamp"],
                        "price": float(t["price"]),
                        "qty": float(t["amount"]),
                    }
                )

            book = await exchange.watch_order_book(SYMBOL, limit=20)
            orderbook = {
                "bids": book["bids"][:10],
                "asks": book["asks"][:10],
            }
        except Exception as exc:
            print(f"WS error (retrying): {exc}")
            await discord("⚠️ WebSocket Error", str(exc), "yellow")
            await asyncio.sleep(2)


# ========================= CANDLE BUILDER =========================
def build_candles() -> pd.DataFrame:
    if len(trades) < 60:
        return pd.DataFrame()
    df = pd.DataFrame(list(trades))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    ohlc = df["price"].resample("5s").ohlc()
    ohlc["volume"] = df["qty"].resample("5s").sum()
    return ohlc.dropna()


# ========================= OPTUNA =========================
def objective(trial: optuna.Trial) -> float:
    df = build_candles()
    if len(df) < 100:
        return -999.0

    fast = trial.suggest_int("fast", 5, 15)
    slow = trial.suggest_int("slow", 18, 35)
    trial.suggest_float("imbalance_thresh", 1.1, 1.8, step=0.05)
    trial.suggest_float("sl_pct", 0.002, 0.006, step=0.0005)

    df = df.copy()
    df["fast_ema"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["slow_ema"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = 0
    df.loc[df["fast_ema"] > df["slow_ema"], "signal"] = 1
    df.loc[df["fast_ema"] < df["slow_ema"], "signal"] = -1

    df["returns"] = df["close"].pct_change()
    df["pnl"] = df["signal"].shift(1) * df["returns"]
    total = float(df["pnl"].sum())
    return total if total != 0 else -999.0


def optimize_params() -> None:
    global best_params
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=70, timeout=14)
    best_params = study.best_params
    print(f"✅ OPTUNA → {best_params}")
    asyncio.get_event_loop().create_task(
        discord(
            "🔄 Optuna Updated Params",
            "\n".join(f"`{k}` → `{v:.4f}`" for k, v in best_params.items()),
            "blue",
        )
    )


# ========================= ORDER HELPERS =========================
def calc_size(price: float) -> float:
    """Contracts to risk 1% of capital at current price + leverage."""
    sl_dist = best_params["sl_pct"]
    raw = (CAPITAL * RISK_PER_ADD * LEVERAGE) / (sl_dist * price)
    return max(round(raw, 1), 0.1)


async def place_order(side: str, size: float, price: float, reduce_only: bool = False) -> bool:
    """Place a market order. Returns True on success."""
    params: dict = {"leverage": LEVERAGE}
    if reduce_only:
        params["reduceOnly"] = True
    try:
        await exchange.create_order(
            symbol=SYMBOL,
            type="market",
            side=side,
            amount=size,
            params=params,
        )
        return True
    except Exception as exc:
        print(f"Order error: {exc}")
        await discord("🛑 Order Failed", f"`{side} {size} contracts`\n```{exc}```", "red")
        return False


# ========================= CLOSE + P&L ALERT =========================
async def close_position(reason: str, current_price: float) -> None:
    """Close the entire stack and send Discord P&L alert."""
    direction = stack["direction"]
    avg = stack_avg_price()
    total_size = stack_total_size()

    if total_size == 0 or direction is None:
        stack_clear()
        return

    close_side = "sell" if direction == "buy" else "buy"
    success = await place_order(close_side, total_size, current_price, reduce_only=True)

    if success:
        # P&L calculation (leveraged)
        if direction == "buy":
            raw_pct = (current_price - avg) / avg
        else:
            raw_pct = (avg - current_price) / avg

        leveraged_pct = raw_pct * LEVERAGE * 100  # % on margin
        usdt_pnl = raw_pct * total_size * avg  # approximate USDT P&L

        emoji = "💰" if raw_pct >= 0 else "📉"
        color = "green" if raw_pct >= 0 else "red"
        sign = "+" if raw_pct >= 0 else ""

        await discord(
            f"{emoji} Trade Closed — {reason}",
            (
                f"**Direction:** `{direction.upper()}`\n"
                f"**Adds:** `{stack['count']}`  |  **Total:** `{total_size} contracts`\n"
                f"**Avg Entry:** `${avg:.4f}`  →  **Exit:** `${current_price:.4f}`\n"
                f"**P&L:** `{sign}{leveraged_pct:.2f}%` on margin  (`{sign}{usdt_pnl:.4f} USDT` est.)\n"
                f"**TP target was:** `{TP_PCT * 100:.1f}%`"
            ),
            color,
        )
        print(
            f"{emoji} CLOSED {direction} | avg={avg:.4f} exit={current_price:.4f} | {sign}{leveraged_pct:.2f}% leveraged"
        )

    stack_clear()


# ========================= TRADING LOOP =========================
async def trading_loop() -> None:
    last_optimize = time.time()
    last_status = time.time()

    await discord(
        "🚀 Bot Started",
        (
            f"**Symbol:** `{SYMBOL}`\n"
            f"**Leverage:** `{LEVERAGE}x`  |  **Margin:** `{MARGIN_MODE}`\n"
            f"**Capital:** `${CAPITAL}`  |  **Max stack:** `{MAX_STACK} adds`\n"
            f"**TP target:** `{TP_PCT * 100:.1f}%`  |  **Risk/add:** `{RISK_PER_ADD * 100:.0f}%`"
        ),
        "blue",
    )
    print(
        f"📈 SOLUSDTM Scalper | {LEVERAGE}x {MARGIN_MODE} | TP={TP_PCT * 100:.1f}% | Stack≤{MAX_STACK}"
    )

    while True:
        now = time.time()

        # ── Optuna re-optimize every 20 min ──────────────────────────────
        if now - last_optimize > OPTUNA_INTERVAL:
            optimize_params()
            last_optimize = now

        # ── Status heartbeat every 5 min ─────────────────────────────────
        if now - last_status > 300:
            candle_count = len(build_candles())
            trade_count = len(trades)
            print(
                f"💓 Heartbeat | trades={trade_count} candles={candle_count} stack={stack['count']}/{MAX_STACK} dir={stack['direction']}"
            )
            last_status = now

        # ── Need enough data ─────────────────────────────────────────────
        candles = build_candles()
        if len(candles) < 30:
            await asyncio.sleep(2)
            continue

        # ── EMAs ─────────────────────────────────────────────────────────
        fast_ema = candles["close"].ewm(span=best_params["fast"], adjust=False).mean().iloc[-1]
        slow_ema = candles["close"].ewm(span=best_params["slow"], adjust=False).mean().iloc[-1]

        # ── Book imbalance ────────────────────────────────────────────────
        bid_vol = sum(float(b[1]) for b in orderbook["bids"])
        ask_vol = sum(float(a[1]) for a in orderbook["asks"])
        imbalance = bid_vol / ask_vol if ask_vol > 0 else 1.0

        # ── Combo signal ──────────────────────────────────────────────────
        signal: str | None = None
        if fast_ema > slow_ema and imbalance > best_params["imbalance_thresh"]:
            signal = "buy"
        elif fast_ema < slow_ema and imbalance < (2.0 - best_params["imbalance_thresh"]):
            signal = "sell"

        # ── Current price ─────────────────────────────────────────────────
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            price = float(ticker["last"])
        except Exception as exc:
            print(f"Ticker error: {exc}")
            await asyncio.sleep(2)
            continue

        direction = stack["direction"]
        avg = stack_avg_price()

        # ── TP check (close whole stack) ──────────────────────────────────
        if direction == "buy" and avg > 0 and price >= avg * (1 + TP_PCT):
            await close_position("Take Profit ✅", price)
            await asyncio.sleep(2)
            continue

        if direction == "sell" and avg > 0 and price <= avg * (1 - TP_PCT):
            await close_position("Take Profit ✅", price)
            await asyncio.sleep(2)
            continue

        # ── SL check ─────────────────────────────────────────────────────
        sl_pct = best_params["sl_pct"]
        if direction == "buy" and avg > 0 and price <= avg * (1 - sl_pct):
            await close_position("Stop Loss 🛑", price)
            await asyncio.sleep(2)
            continue

        if direction == "sell" and avg > 0 and price >= avg * (1 + sl_pct):
            await close_position("Stop Loss 🛑", price)
            await asyncio.sleep(2)
            continue

        # ── Signal logic ──────────────────────────────────────────────────
        if signal is None:
            await asyncio.sleep(2)
            continue

        # Opposite signal → close stack and reverse
        if direction is not None and signal != direction:
            await close_position("Signal Reversal 🔄", price)
            # fall through to open opposite

        # Same direction → add if under stack limit and price has moved against us
        if direction == signal and stack["count"] >= MAX_STACK:
            await asyncio.sleep(2)
            continue

        # Check if this is a new add or first entry
        is_add = direction == signal and stack["count"] > 0
        if is_add:
            # Only add if price moved against us by ADD_THRESHOLD (DCA logic)
            if direction == "buy" and price >= avg * (1 - ADD_THRESHOLD):
                await asyncio.sleep(2)
                continue
            if direction == "sell" and price <= avg * (1 + ADD_THRESHOLD):
                await asyncio.sleep(2)
                continue

        # Place the entry/add order
        size = calc_size(price)
        success = await place_order(signal, size, price)

        if success:
            stack["direction"] = signal
            stack["count"] += 1
            stack["prices"].append(price)
            stack["sizes"].append(size)

            add_label = f"Add #{stack['count']}" if stack["count"] > 1 else "Entry #1"
            await discord(
                f"{'🟢' if signal == 'buy' else '🔴'} {add_label} — {signal.upper()}",
                (
                    f"**Size:** `{size} contracts`  |  **Price:** `${price:.4f}`\n"
                    f"**Stack:** `{stack['count']}/{MAX_STACK}`\n"
                    f"**Avg Entry:** `${stack_avg_price():.4f}`\n"
                    f"**TP target:** `${stack_avg_price() * (1 + TP_PCT if signal == 'buy' else 1 - TP_PCT):.4f}`\n"
                    f"**EMA fast/slow:** `{fast_ema:.4f}` / `{slow_ema:.4f}`\n"
                    f"**Imbalance:** `{imbalance:.2f}`"
                ),
                "green" if signal == "buy" else "red",
            )
            print(
                f"{'🟢' if signal == 'buy' else '🔴'} {add_label} {signal} {size}@{price:.4f} | avg={stack_avg_price():.4f} | imb={imbalance:.2f}"
            )

        await asyncio.sleep(2)


# ========================= MAIN =========================
async def main() -> None:
    # Set leverage + margin mode once on startup
    try:
        await exchange.set_margin_mode(MARGIN_MODE, SYMBOL)
        await exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"✅ {MARGIN_MODE.upper()} margin @ {LEVERAGE}x set")
    except Exception as exc:
        print(f"⚠️  Margin/leverage (may already be set): {exc}")

    asyncio.create_task(handle_ws())
    await trading_loop()


if __name__ == "__main__":
    asyncio.run(main())
