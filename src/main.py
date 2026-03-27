"""
KuCoin SOLUSDTM Perpetual Scalper — Ruby-Enhanced
================================================================
Ruby Pine Script v6.3 logic ported and integrated:

  Wave Analysis    → wave_ratio + cur_ratio gate position adds.
                     Only add #2/#3 when bull/bear wave momentum
                     confirms the direction. Core reason: SOL crypto
                     swings hard — you want wave energy behind you
                     before averaging in.

  Market Regime    → SMA200 slope + vol ratio → TRENDING/VOLATILE/
                     RANGING/NEUTRAL. Regime drives adaptive TP and
                     blocks counter-trend stacking.

  Vol Percentile   → Rolling ATR percentile (0.0–1.0). Widens TP
                     and add-threshold when volatility is high so
                     you don't get stopped on normal noise.

  Quality Score    → 0–100 pre-entry filter (AO + EMA alignment +
                     book imbalance + volume + regime). No trade
                     fires below quality_min.

  Awesome Oscillator → SMA(hl2,5) - SMA(hl2,34). Confirms macro
                     momentum direction before entry and adds.

  Adaptive TP      → TP_PCT scales with vol_pct and regime.
                     RANGING = tighter, VOLATILE/TRENDING = wider.
                     Baseline set in .env, adjusted at runtime.

Discord alerts include all Ruby state so you can see regime,
quality, wave_ratio, and AO in every trade notification.

FIXES v2:
  - handle_ws split into handle_trades + handle_orderbook (two
    separate tasks so orderbook updates never wait on trade feed).
  - optimize_params no longer uses get_event_loop(); discord
    notification fired by the async caller in trading_loop.
  - Added orderbook warm-up guard (skips signal check until both
    feeds are live).
  - src/__init__.py required for `python -m src.main`.
"""

import asyncio
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import ccxt.async_support as ccxt
import optuna
import pandas as pd
import yaml
from dotenv import load_dotenv

optuna.logging.set_verbosity(optuna.logging.WARNING)
load_dotenv()

# ================================================================
# CONFIG — loaded from config/sol.yaml with .env fallbacks
# ================================================================
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sol.yaml"


def _load_config() -> dict:
    """Load YAML config with ${ENV_VAR} expansion. Falls back to defaults."""
    defaults: dict = {
        "exchange": {"symbol": "SOLUSDTM", "leverage": 20, "margin_mode": "isolated"},
        "capital": {"balance_usdt": 30.0, "risk_per_add_pct": 0.01, "max_stack": 3},
        "strategy": {
            "fast_ema": 8,
            "slow_ema": 21,
            "tp_pct_base": 0.004,
            "sl_pct": 0.0035,
            "imbalance_thresh": 1.42,
            "add_threshold_base": 0.002,
        },
        "wave": {"quality_min": 50.0, "wave_pct_gate": 0.40},
        "optuna": {"interval_sec": 1200, "n_trials": 70, "timeout_sec": 14},
        "monitoring": {"heartbeat_interval_sec": 300, "loop_sleep_sec": 2},
        "candles": {"tick_buffer": 15000, "min_ticks": 60},
        "streams": {"orderbook_depth": 20, "orderbook_display": 10},
        "risk": {
            "max_daily_loss_pct": 0.05,
            "max_consecutive_losses": 4,
            "cooldown_after_loss_sec": 30,
            "daily_trade_limit": 50,
        },
        "volatility": {
            "entry_filter": {"min_percentile": 0.10, "max_percentile": 0.95},
        },
    }
    if not _CONFIG_PATH.exists():
        print(f"⚠️  Config not found at {_CONFIG_PATH}, using defaults + .env")
        return defaults
    try:
        raw = _CONFIG_PATH.read_text()

        def _expand(match: re.Match) -> str:
            return os.environ.get(match.group(1), match.group(0))

        expanded = re.sub(r"\$\{([^}]+)\}", _expand, raw)
        cfg = yaml.safe_load(expanded) or {}

        def _deep_merge(base: dict, override: dict) -> dict:
            merged = base.copy()
            for k, v in override.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = _deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        result = _deep_merge(defaults, cfg)
        print(f"✅ Config loaded from {_CONFIG_PATH}")
        return result
    except Exception as exc:
        print(f"⚠️  Config load error ({exc}), using defaults + .env")
        return defaults


CFG = _load_config()

# ── Flatten frequently-accessed values (.env overrides YAML) ──────
SYMBOL = CFG["exchange"]["symbol"]
LEVERAGE = int(os.getenv("LEVERAGE", str(CFG["exchange"]["leverage"])))
MARGIN_MODE = os.getenv("MARGIN_MODE", CFG["exchange"]["margin_mode"])
CAPITAL = float(os.getenv("CAPITAL", str(CFG["capital"]["balance_usdt"])))
TP_PCT_BASE = float(os.getenv("TP_PCT", str(CFG["strategy"]["tp_pct_base"])))
RISK_PER_ADD = CFG["capital"]["risk_per_add_pct"]
MAX_STACK = CFG["capital"]["max_stack"]
ADD_THRESHOLD_BASE = CFG["strategy"]["add_threshold_base"]
OPTUNA_INTERVAL = CFG["optuna"]["interval_sec"]
HEARTBEAT_INTERVAL = CFG["monitoring"]["heartbeat_interval_sec"]
LOOP_SLEEP = CFG["monitoring"]["loop_sleep_sec"]

# Ruby-ported thresholds (Optuna refines these live)
QUALITY_MIN_DEFAULT = CFG["wave"]["quality_min"]
WAVE_PCT_GATE_DEFAULT = CFG["wave"]["wave_pct_gate"]

# Risk management thresholds (ported from Ruby RiskManager)
MAX_DAILY_LOSS_PCT = CFG["risk"]["max_daily_loss_pct"]
MAX_CONSECUTIVE_LOSSES = CFG["risk"]["max_consecutive_losses"]
COOLDOWN_AFTER_LOSS = CFG["risk"]["cooldown_after_loss_sec"]
DAILY_TRADE_LIMIT = CFG["risk"]["daily_trade_limit"]
VOL_ENTRY_MIN = CFG["volatility"]["entry_filter"]["min_percentile"]
VOL_ENTRY_MAX = CFG["volatility"]["entry_filter"]["max_percentile"]
MIN_ORDER_USDT = float(CFG["capital"].get("min_order_usdt", 0.50))

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE")

# ================================================================
# IN-MEMORY STATE
# ================================================================
trades: deque = deque(maxlen=15000)
orderbook: dict = {"bids": [], "asks": []}

# Position stack — no database, purely in-memory
stack: dict = {
    "direction": None,  # "buy" | "sell" | None
    "count": 0,  # adds placed so far (max MAX_STACK)
    "prices": [],  # fill price of each add
    "sizes": [],  # contract count of each add
}

# Optuna best params (updated live every OPTUNA_INTERVAL seconds)
best_params: dict = {
    "fast": 8,
    "slow": 21,
    "imbalance_thresh": 1.42,
    "sl_pct": 0.0035,
    "quality_min": QUALITY_MIN_DEFAULT,
    "wave_pct_gate": WAVE_PCT_GATE_DEFAULT,
}

# Risk management state (ported from Ruby RiskManager)
risk_state: dict = {
    "daily_pnl": 0.0,  # cumulative P&L today (USDT)
    "daily_trades": 0,  # round-trips completed today
    "consecutive_losses": 0,  # streak of consecutive losing trades
    "last_loss_time": 0.0,  # timestamp of last stop-loss hit
    "day_start": "",  # for daily reset detection (set on first tick)
    "paused": False,  # True when risk limit hit
    "pause_reason": "",  # human-readable reason for pause
    "_alerted": False,  # True after Discord pause alert sent
}


# ── Wave analysis persistent state (ported from Ruby §4) ─────────
@dataclass
class WaveState:
    bull_waves: deque = field(default_factory=lambda: deque(maxlen=200))
    bear_waves: deque = field(default_factory=lambda: deque(maxlen=200))
    wr_history: deque = field(default_factory=lambda: deque(maxlen=200))
    mom_history: deque = field(default_factory=lambda: deque(maxlen=200))
    wave_ratio: float = 1.0  # bull_avg / |bear_avg|
    wr_pct: float = 0.5  # wave_ratio percentile in wr_history
    cur_ratio: float = 0.0  # current speed vs historical avg
    mom_pct: float = 0.5  # |cur_ratio| percentile in mom_history
    last_above: bool = False  # was close above EMA20 on previous bar?
    bias: str = "Neutral"  # "Bullish" | "Bearish" | "Neutral"


wave_state = WaveState()

# ================================================================
# EXCHANGE CLIENT
# ================================================================
exchange = ccxt.kucoinfutures(
    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)

# ================================================================
# DISCORD
# ================================================================
COLORS = {
    "green": 0x00C853,
    "red": 0xD50000,
    "yellow": 0xFFD600,
    "blue": 0x2196F3,
    "grey": 0x9E9E9E,
    "purple": 0x9C27B0,
}


async def discord(title: str, body: str, color: str = "blue") -> None:
    """Fire-and-forget Discord embed. Silent no-op if webhook not set."""
    if not DISCORD_WEBHOOK:
        return
    payload = {
        "embeds": [
            {
                "title": title,
                "description": body,
                "color": COLORS.get(color, COLORS["blue"]),
                "footer": {
                    "text": f"SOL Scalper | {SYMBOL} | {LEVERAGE}x {MARGIN_MODE}"
                },
            }
        ]
    }
    try:
        async with aiohttp.ClientSession() as session:
            r = await session.post(
                DISCORD_WEBHOOK,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            )
            if r.status not in (200, 204):
                print(f"Discord HTTP {r.status}")
    except Exception as exc:
        print(f"Discord error: {exc}")


# ================================================================
# STACK HELPERS
# ================================================================
def stack_avg_price() -> float:
    if not stack["prices"]:
        return 0.0
    total_sz = sum(stack["sizes"])
    return sum(p * s for p, s in zip(stack["prices"], stack["sizes"])) / total_sz


def stack_total_size() -> float:
    return sum(stack["sizes"])


def stack_clear() -> None:
    stack.update(direction=None, count=0, prices=[], sizes=[])


# ================================================================
# RISK MANAGEMENT (ported from Ruby RiskManager test suite)
# ================================================================
def risk_check_daily_reset() -> None:
    """Reset daily counters if the calendar day has changed."""
    today = time.strftime("%Y-%m-%d")
    if risk_state["day_start"] != today:
        risk_state["daily_pnl"] = 0.0
        risk_state["daily_trades"] = 0
        risk_state["paused"] = False
        risk_state["pause_reason"] = ""
        risk_state["_alerted"] = False
        risk_state["day_start"] = today
        print(f"📅 New trading day: {today} — counters reset")


def risk_record_trade(pnl_usdt: float, was_stop: bool) -> None:
    """Update risk state after a trade closes."""
    risk_state["daily_pnl"] += pnl_usdt
    risk_state["daily_trades"] += 1
    if pnl_usdt < 0:
        risk_state["consecutive_losses"] += 1
        if was_stop:
            risk_state["last_loss_time"] = time.time()
    else:
        risk_state["consecutive_losses"] = 0


def risk_can_trade() -> tuple[bool, str]:
    """Check all risk gates. Returns (allowed, reason)."""
    risk_check_daily_reset()

    # Daily loss limit
    max_loss = CAPITAL * MAX_DAILY_LOSS_PCT
    if risk_state["daily_pnl"] <= -max_loss:
        reason = f"Daily loss limit hit ({risk_state['daily_pnl']:.2f} USDT / -{max_loss:.2f} max)"
        risk_state["paused"] = True
        risk_state["pause_reason"] = reason
        return False, reason

    # Consecutive losses circuit breaker
    if risk_state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        reason = f"Consecutive losses: {risk_state['consecutive_losses']} (limit {MAX_CONSECUTIVE_LOSSES})"
        risk_state["paused"] = True
        risk_state["pause_reason"] = reason
        return False, reason

    # Cooldown after stop-loss
    if risk_state["last_loss_time"] > 0:
        elapsed = time.time() - risk_state["last_loss_time"]
        if elapsed < COOLDOWN_AFTER_LOSS:
            return (
                False,
                f"Post-SL cooldown: {COOLDOWN_AFTER_LOSS - elapsed:.0f}s remaining",
            )

    # Daily trade limit
    if risk_state["daily_trades"] >= DAILY_TRADE_LIMIT:
        reason = f"Daily trade limit reached ({DAILY_TRADE_LIMIT})"
        risk_state["paused"] = True
        risk_state["pause_reason"] = reason
        return False, reason

    # Clear paused state if we pass all gates
    if risk_state["paused"]:
        risk_state["paused"] = False
        risk_state["pause_reason"] = ""

    return True, ""


# ================================================================
# DATA STREAMS — split into two independent tasks
# FIX: previously one loop handled both sequentially, meaning
# orderbook updates only happened after new trades arrived.
# ================================================================
async def handle_trades() -> None:
    """Continuously streams trade ticks into the rolling deque."""
    print("📡 Trade feed connecting...")
    while True:
        try:
            trade_data = await exchange.watch_trades(SYMBOL)
            for t in trade_data:
                p = float(t["price"])
                trades.append(
                    {
                        "timestamp": t["timestamp"],
                        "price": p,
                        "qty": float(t["amount"]),
                        "open": p,
                        "high": p,
                        "low": p,
                    }
                )
        except Exception as exc:
            print(f"Trade WS error (retrying): {exc}")
            await asyncio.sleep(2)


async def handle_orderbook() -> None:
    """Continuously updates the top-10 order book snapshot."""
    global orderbook
    print("📡 Order book feed connecting...")
    while True:
        try:
            book = await exchange.watch_order_book(SYMBOL, limit=20)
            orderbook = {
                "bids": book["bids"][:10],
                "asks": book["asks"][:10],
            }
        except Exception as exc:
            print(f"Book WS error (retrying): {exc}")
            await discord("⚠️ OrderBook WS Error", str(exc), "yellow")
            await asyncio.sleep(2)


# ================================================================
# CANDLE BUILDER  — 5-second OHLCV from tick stream
# ================================================================
def build_candles() -> pd.DataFrame:
    if len(trades) < 60:
        return pd.DataFrame()
    df = pd.DataFrame(list(trades))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    ohlc = df["price"].resample("5s").ohlc()
    ohlc["high"] = df["price"].resample("5s").max()
    ohlc["low"] = df["price"].resample("5s").min()
    ohlc["volume"] = df["qty"].resample("5s").sum()
    ohlc["hl2"] = (ohlc["high"] + ohlc["low"]) / 2
    ohlc["open"] = df["price"].resample("5s").first()
    return ohlc.dropna()


# ================================================================
# RUBY INDICATORS — §2, §4, §5, §6, §7, §9 ported to Python
# ================================================================
def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder EMA)."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_ao(df: pd.DataFrame) -> float:
    """Awesome Oscillator — Ruby §2. ao = SMA(hl2,5) - SMA(hl2,34)."""
    if len(df) < 34:
        return 0.0
    ao = df["hl2"].rolling(5).mean() - df["hl2"].rolling(34).mean()
    return float(ao.iloc[-1])


def compute_vol_pct(df: pd.DataFrame) -> float:
    """Volatility percentile — Ruby §6. 0.0=calm, 1.0=extreme."""
    if len(df) < 20:
        return 0.5
    atr = _compute_atr(df, 14).dropna()
    recent = atr.values[-200:]
    cur = float(atr.iloc[-1])
    return float((recent < cur).sum()) / max(len(recent), 1)


def compute_regime(df: pd.DataFrame) -> tuple[str, float]:
    """
    Market regime — Ruby §5.
    TRENDING_UP / TRENDING_DOWN / VOLATILE / RANGING / NEUTRAL
    """
    if len(df) < 220:
        return "NEUTRAL", 0.0

    ma200 = df["close"].rolling(200).mean()
    avg_chg = ma200.diff().abs().rolling(100).mean()
    slope = (ma200 - ma200.shift(20)) / (avg_chg * 20 + 1e-10)
    slope_n = float(slope.iloc[-1])

    ret_s = df["close"].pct_change().rolling(100).std()
    ret_sma = ret_s.rolling(50).mean()
    vol_n = float(ret_s.iloc[-1] / (ret_sma.iloc[-1] + 1e-10))

    if slope_n > 1.0:
        return "TRENDING_UP", slope_n
    if slope_n < -1.0:
        return "TRENDING_DOWN", slope_n
    if vol_n > 1.5:
        return "VOLATILE", vol_n
    if vol_n < 0.8:
        return "RANGING", vol_n
    return "NEUTRAL", vol_n


def update_wave_state(df: pd.DataFrame, ws: WaveState) -> WaveState:
    """
    Wave analysis — Ruby §4, translated to Python.

    Ruby: speed accumulates (c_rma - o_rma) within each wave.
    Cross EMA20 up   → completed bear wave → record trough into bear_waves.
    Cross EMA20 down → completed bull wave → record peak into bull_waves.
    wave_ratio = bull_avg / |bear_avg|

    Stacking gate: wave_ratio > 1.0 → bulls harder → safe to add longs.
                   wave_ratio < 1.0 → bears harder → safe to add shorts.
    """
    if len(df) < 40:
        return ws

    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    c_rma = df["close"].ewm(alpha=1 / 10, adjust=False).mean()
    o_rma = df["open"].ewm(alpha=1 / 10, adjust=False).mean()
    speed_series = c_rma - o_rma

    above_now = bool(df["close"].iloc[-1] > ema20.iloc[-1])
    above_prev = bool(df["close"].iloc[-2] > ema20.iloc[-2])
    just_crossed = above_now != above_prev

    if just_crossed:
        lookback = min(50, len(speed_series) - 1)
        wave_speeds = speed_series.iloc[-lookback:].values
        if above_now:
            # Bear→Bull cross: record trough of completed bear wave
            ws.bear_waves.append(float(wave_speeds.min()))
        else:
            # Bull→Bear cross: record peak of completed bull wave
            ws.bull_waves.append(float(wave_speeds.max()))

    bull_list = list(ws.bull_waves)
    bear_list = list(ws.bear_waves)
    bull_avg = sum(bull_list) / len(bull_list) if bull_list else 0.0001
    bear_avg = abs(sum(bear_list) / len(bear_list)) if bear_list else 0.0001

    ws.wave_ratio = bull_avg / bear_avg if bear_avg > 0 else 1.0
    ws.bias = "Bullish" if bull_avg > bear_avg else "Bearish"

    cur_speed = float(speed_series.iloc[-1])
    if cur_speed >= 0:
        ws.cur_ratio = cur_speed / bull_avg if bull_avg > 0 else 1.0
    else:
        ws.cur_ratio = -(abs(cur_speed) / bear_avg) if bear_avg > 0 else -1.0

    ws.wr_history.append(ws.wave_ratio)
    ws.mom_history.append(abs(ws.cur_ratio))
    wr_arr = list(ws.wr_history)
    mom_arr = list(ws.mom_history)
    ws.wr_pct = sum(v < ws.wave_ratio for v in wr_arr) / max(len(wr_arr), 1)
    ws.mom_pct = sum(v < abs(ws.cur_ratio) for v in mom_arr) / max(len(mom_arr), 1)

    ws.last_above = above_now
    return ws


def compute_quality(
    df: pd.DataFrame,
    ao: float,
    vol_pct: float,
    fast_ema: float,
    slow_ema: float,
    imbalance: float,
    regime: str,
) -> float:
    """
    Quality Score 0–100 — Ruby §9, adapted for 24/7 crypto futures.

    1. AO confirms EMA direction          → 20 pts
    2. Close on correct side of fast EMA  → 15 pts
    3. Book imbalance confirms direction  → 20 pts  (replaces VWAP)
    4. Volume above vol-adjusted average  → 25 pts
    5. Regime aligns with direction       → 20 pts  (replaces ORB)
       (partial credit in RANGING/NEUTRAL → 10 pts)
    """
    if len(df) < 35:
        return 0.0

    close = float(df["close"].iloc[-1])
    volume = float(df["volume"].iloc[-1])
    vol_avg = float(df["volume"].rolling(20).mean().iloc[-1])
    bull_bias = fast_ema > slow_ema

    vol_mult = 1.8 if vol_pct >= 0.6 else 1.2 if vol_pct >= 0.4 else 0.8

    score = 0.0
    score += 20.0 if (bull_bias and ao > 0) or (not bull_bias and ao < 0) else 0.0
    score += (
        15.0
        if (bull_bias and close > fast_ema) or (not bull_bias and close < fast_ema)
        else 0.0
    )
    score += (
        20.0
        if (bull_bias and imbalance >= 1.2) or (not bull_bias and imbalance <= 0.8)
        else 0.0
    )
    score += 25.0 if volume > vol_avg * vol_mult else 0.0

    if bull_bias and regime == "TRENDING_UP":
        score += 20.0
    elif not bull_bias and regime == "TRENDING_DOWN":
        score += 20.0
    elif regime in ("NEUTRAL", "RANGING"):
        score += 10.0

    return min(score, 100.0)


def adaptive_tp(vol_pct: float, regime: str) -> float:
    """
    Scale TP_PCT_BASE by regime and vol percentile — Ruby §7.

    RANGING   → 0.7–1.0x  (collect small wins quickly)
    NEUTRAL   → 1.0–1.3x
    VOLATILE  → 1.2–1.7x  (give room so normal swings don't stop you out)
    TRENDING  → 1.4–2.0x  (ride the wave)
    """
    if regime in ("TRENDING_UP", "TRENDING_DOWN"):
        mult = 1.4 + vol_pct * 0.6
    elif regime == "VOLATILE":
        mult = 1.2 + vol_pct * 0.5
    elif regime == "RANGING":
        mult = 0.7 + vol_pct * 0.3
    else:
        mult = 1.0 + vol_pct * 0.3
    return TP_PCT_BASE * mult


def adaptive_add_threshold(vol_pct: float) -> float:
    """Price must retrace this fraction before we add another position."""
    return ADD_THRESHOLD_BASE * (1.0 + vol_pct)  # 0.20%–0.40%


def wave_gate_ok(ws: WaveState, direction: str, gate: float) -> bool:
    """
    Ruby §7 waveOK_L / waveOK_S — gates position adds.
    Prevents averaging into a stack when wave energy has flipped against you.
    """
    if direction == "buy":
        return ws.wr_pct >= gate and ws.cur_ratio > 0
    return ws.wr_pct <= (1.0 - gate) and ws.cur_ratio < 0


def regime_stack_ok(regime: str, direction: str) -> bool:
    """
    Ruby §10 regimeOK_L / regimeOK_S.
    Block counter-trend adds only — first entries allowed in any regime.
    """
    if direction == "buy" and regime == "TRENDING_DOWN":
        return False
    if direction == "sell" and regime == "TRENDING_UP":
        return False
    return True


# ================================================================
# OPTUNA — optimises Ruby-ported params live on streaming data
# FIX: discord notification moved to async caller; no more
#      get_event_loop() calls from sync context.
# ================================================================
def objective(trial: optuna.Trial) -> float:
    df = build_candles()
    if len(df) < 100:
        return -999.0

    fast = trial.suggest_int("fast", 5, 15)
    slow = trial.suggest_int("slow", 18, 35)
    _ = trial.suggest_float("imbalance_thresh", 1.1, 1.8, step=0.05)
    _ = trial.suggest_float("sl_pct", 0.002, 0.006, step=0.0005)
    _ = trial.suggest_float("quality_min", 30.0, 70.0, step=5.0)
    _ = trial.suggest_float("wave_pct_gate", 0.25, 0.65, step=0.05)

    df = df.copy()
    df["fast_ema"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["slow_ema"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["ao"] = df["hl2"].rolling(5).mean() - df["hl2"].rolling(34).mean()
    df["signal"] = 0
    df.loc[df["fast_ema"] > df["slow_ema"], "signal"] = 1
    df.loc[df["fast_ema"] < df["slow_ema"], "signal"] = -1

    ao_pass = ((df["signal"] == 1) & (df["ao"] > 0)) | (
        (df["signal"] == -1) & (df["ao"] < 0)
    )
    df.loc[~ao_pass, "signal"] = 0

    df["returns"] = df["close"].pct_change()
    df["pnl"] = df["signal"].shift(1) * df["returns"]
    total = float(df["pnl"].sum())
    return total if total != 0 else -999.0


def optimize_params() -> dict | None:
    """
    Runs Optuna study and updates best_params.
    Returns the new params dict so the async caller can send the
    Discord notification — avoids get_event_loop() in sync context.
    """
    global best_params
    df = build_candles()
    if len(df) < 100:
        print("⏭  Optuna skipped — not enough candle data yet")
        return None
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=70, timeout=14)
    best_params = study.best_params
    print(f"✅ OPTUNA → {best_params}")
    return best_params


# ================================================================
# ORDER HELPERS
# ================================================================
def calc_size(price: float) -> float:
    """Contracts to risk exactly 1% of capital at current price + leverage.

    Enforces a minimum notional value of MIN_ORDER_USDT (default $0.50)
    so that orders are large enough to actually fill on KuCoin.
    """
    sl_dist = best_params["sl_pct"]
    raw = (CAPITAL * RISK_PER_ADD * LEVERAGE) / (sl_dist * price)
    size = max(round(raw, 1), 0.1)
    # Ensure the order meets the minimum dollar threshold
    notional = size * price / LEVERAGE
    if notional < MIN_ORDER_USDT:
        size = round((MIN_ORDER_USDT * LEVERAGE) / price, 1)
        size = max(size, 0.1)
    return size


async def place_order(
    side: str, size: float, price: float, reduce_only: bool = False
) -> bool:
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
        await discord(
            "🛑 Order Failed", f"`{side} {size} contracts`\n```{exc}```", "red"
        )
        return False


# ================================================================
# CLOSE + P&L ALERT
# Prominent % display so you can track if returns stay consistent.
# ================================================================
async def close_position(
    reason: str, price: float, ruby_ctx: dict | None = None
) -> None:
    direction = stack["direction"]
    avg = stack_avg_price()
    total_size = stack_total_size()

    if total_size == 0 or direction is None:
        stack_clear()
        return

    close_side = "sell" if direction == "buy" else "buy"
    success = await place_order(close_side, total_size, price, reduce_only=True)

    if success:
        raw_pct = ((price - avg) / avg) if direction == "buy" else ((avg - price) / avg)
        lev_pct = raw_pct * LEVERAGE * 100
        usdt_pnl = raw_pct * total_size * avg
        sign = "+" if raw_pct >= 0 else ""
        emoji = "💰" if raw_pct >= 0 else "📉"
        clr = "green" if raw_pct >= 0 else "red"

        # ── Prominent P&L header line ──────────────────────────────
        pnl_header = (
            f"## {sign}{lev_pct:.2f}% on margin  "
            f"({sign}{usdt_pnl:.4f} USDT)\n"
            f"**Price:** `${avg:.4f}` → `${price:.4f}`  "
            f"|  **Adds:** `{stack['count']}/{MAX_STACK}`  "
            f"|  **Size:** `{total_size} contracts`\n"
            f"**Reason:** `{reason}`"
        )

        ctx_str = ""
        if ruby_ctx:
            ctx_str = (
                f"\n**Regime:** `{ruby_ctx.get('regime', '?')}`  "
                f"**WaveRatio:** `{ruby_ctx.get('wave_ratio', 0):.2f}` "
                f"(p{ruby_ctx.get('wr_pct', 0):.0%})\n"
                f"**AO:** `{ruby_ctx.get('ao', 0):.4f}`  "
                f"**VolPct:** `{ruby_ctx.get('vol_pct', 0):.0%}`  "
                f"**Quality:** `{ruby_ctx.get('quality', 0):.0f}/100`\n"
                f"**Adaptive TP used:** `{ruby_ctx.get('tp_pct', 0) * 100:.2f}%`"
            )

        await discord(
            f"{emoji} {sign}{lev_pct:.2f}%  |  Trade Closed",
            pnl_header + ctx_str,
            clr,
        )
        print(
            f"{emoji} CLOSED {direction} | avg={avg:.4f} exit={price:.4f} "
            f"| {sign}{lev_pct:.2f}% lev ({sign}{usdt_pnl:.4f} USDT) "
            f"| adds={stack['count']}"
        )

        # ── Update risk state ────────────────────────────────────────
        risk_record_trade(usdt_pnl, "Stop Loss" in reason)

    stack_clear()


# ================================================================
# TRADING LOOP
# ================================================================
async def trading_loop() -> None:
    global wave_state

    last_optimize = time.time()
    last_heartbeat = time.time()

    await discord(
        "🚀 SOL Scalper Started — Ruby Wave Edition",
        (
            f"**Symbol:** `{SYMBOL}`  |  **Leverage:** `{LEVERAGE}x`  "
            f"|  **Margin:** `{MARGIN_MODE}`\n"
            f"**Capital:** `${CAPITAL}`  |  **Stack:** max `{MAX_STACK}` adds "
            f"@ {RISK_PER_ADD * 100:.0f}% each\n"
            f"**TP base:** `{TP_PCT_BASE * 100:.2f}%` (adaptive via vol + regime)\n"
            f"**Ruby gates:** wave_ratio + AO + quality score + regime\n"
            f"**Position mode:** One-Way (required for reduceOnly closes)"
        ),
        "blue",
    )
    print(
        f"📈 {SYMBOL} | {LEVERAGE}x {MARGIN_MODE} | "
        f"TP_BASE={TP_PCT_BASE * 100:.2f}% | Stack≤{MAX_STACK} | "
        f"Ruby wave+regime+quality enabled"
    )

    while True:
        now = time.time()

        # ── Optuna re-optimize every 20 min ─────────────────────────
        if now - last_optimize > OPTUNA_INTERVAL:
            new_params = optimize_params()
            if new_params:
                # FIX: discord fired here (async), not inside sync optimize_params
                await discord(
                    "🔄 Optuna Updated Params",
                    "\n".join(
                        f"`{k}` → `{round(v, 4) if isinstance(v, float) else v}`"
                        for k, v in new_params.items()
                    ),
                    "purple",
                )
            last_optimize = now

        # ── Build candles ────────────────────────────────────────────
        candles = build_candles()
        if len(candles) < 35:
            await asyncio.sleep(2)
            continue

        # ── Guard: wait for order book to warm up ───────────────────
        # FIX: with split WS tasks the book may lag a few seconds on startup
        if not orderbook["bids"] or not orderbook["asks"]:
            await asyncio.sleep(2)
            continue

        # ── Compute Ruby indicators ──────────────────────────────────
        wave_state = update_wave_state(candles, wave_state)
        vol_pct = compute_vol_pct(candles)
        regime, _ = compute_regime(candles)
        ao = compute_ao(candles)

        fast_ema = float(
            candles["close"].ewm(span=best_params["fast"], adjust=False).mean().iloc[-1]
        )
        slow_ema = float(
            candles["close"].ewm(span=best_params["slow"], adjust=False).mean().iloc[-1]
        )

        bid_vol = sum(float(b[1]) for b in orderbook["bids"])
        ask_vol = sum(float(a[1]) for a in orderbook["asks"])
        imbalance = bid_vol / ask_vol if ask_vol > 0 else 1.0

        quality = compute_quality(
            candles, ao, vol_pct, fast_ema, slow_ema, imbalance, regime
        )
        tp_pct = adaptive_tp(vol_pct, regime)
        add_thresh = adaptive_add_threshold(vol_pct)
        qual_min = float(best_params.get("quality_min", QUALITY_MIN_DEFAULT))
        wave_gate = float(best_params.get("wave_pct_gate", WAVE_PCT_GATE_DEFAULT))

        ruby_ctx = {
            "regime": regime,
            "wave_ratio": wave_state.wave_ratio,
            "wr_pct": wave_state.wr_pct,
            "cur_ratio": wave_state.cur_ratio,
            "ao": ao,
            "vol_pct": vol_pct,
            "quality": quality,
            "tp_pct": tp_pct,
        }

        # ── Heartbeat console every 5 min ───────────────────────────
        if now - last_heartbeat > HEARTBEAT_INTERVAL:
            print(
                f"💓 regime={regime}  "
                f"wave={wave_state.wave_ratio:.2f}(p{wave_state.wr_pct:.0%})  "
                f"bias={wave_state.bias}  ao={ao:.4f}  "
                f"vol_pct={vol_pct:.0%}  quality={quality:.0f}  "
                f"tp={tp_pct * 100:.2f}%  imb={imbalance:.2f}  "
                f"stack={stack['count']}/{MAX_STACK}({stack['direction']})"
            )
            last_heartbeat = now

        # ── Current price ────────────────────────────────────────────
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            price = float(ticker["last"])
        except Exception as exc:
            print(f"Ticker error: {exc}")
            await asyncio.sleep(2)
            continue

        direction = stack["direction"]
        avg = stack_avg_price()

        # ── TP check ─────────────────────────────────────────────────
        if direction == "buy" and avg > 0 and price >= avg * (1 + tp_pct):
            await close_position("Take Profit ✅", price, ruby_ctx)
            await asyncio.sleep(2)
            continue

        if direction == "sell" and avg > 0 and price <= avg * (1 - tp_pct):
            await close_position("Take Profit ✅", price, ruby_ctx)
            await asyncio.sleep(2)
            continue

        # ── SL check ─────────────────────────────────────────────────
        sl_pct = best_params["sl_pct"]
        if direction == "buy" and avg > 0 and price <= avg * (1 - sl_pct):
            await close_position("Stop Loss 🛑", price, ruby_ctx)
            await asyncio.sleep(2)
            continue

        if direction == "sell" and avg > 0 and price >= avg * (1 + sl_pct):
            await close_position("Stop Loss 🛑", price, ruby_ctx)
            await asyncio.sleep(2)
            continue

        # ── Risk management gate (TP/SL above still active) ──────────
        can_trade, risk_reason = risk_can_trade()
        if not can_trade:
            if not risk_state["_alerted"]:
                await discord("⚠️ Trading Paused", risk_reason, "yellow")
                print(f"⚠️ RISK GATE: {risk_reason}")
                risk_state["_alerted"] = True
            await asyncio.sleep(LOOP_SLEEP)
            continue
        elif risk_state["_alerted"]:
            risk_state["_alerted"] = False
            await discord("✅ Trading Resumed", "Risk conditions cleared", "green")

        # ── Volatility filter (from Ruby should_filter_entry) ────────
        if vol_pct < VOL_ENTRY_MIN or vol_pct > VOL_ENTRY_MAX:
            await asyncio.sleep(LOOP_SLEEP)
            continue

        # ── Entry signal: EMA crossover + book imbalance ─────────────
        signal: str | None = None
        if fast_ema > slow_ema and imbalance > best_params["imbalance_thresh"]:
            signal = "buy"
        elif fast_ema < slow_ema and imbalance < (
            2.0 - best_params["imbalance_thresh"]
        ):
            signal = "sell"

        if signal is None:
            await asyncio.sleep(2)
            continue

        # ── Opposite signal: close stack ─────────────────────────────
        if direction is not None and signal != direction:
            await close_position("Signal Reversal 🔄", price, ruby_ctx)

        direction = stack["direction"]  # re-read after possible close

        # ── Stack full → hold ─────────────────────────────────────────
        if direction == signal and stack["count"] >= MAX_STACK:
            await asyncio.sleep(2)
            continue

        is_add = direction == signal and stack["count"] > 0

        # ── GATES ─────────────────────────────────────────────────────
        if is_add:
            # Ruby wave + regime gates for adds #2 and #3
            if not wave_gate_ok(ws=wave_state, direction=signal, gate=wave_gate):
                await asyncio.sleep(2)
                continue

            if not regime_stack_ok(regime=regime, direction=signal):
                await asyncio.sleep(2)
                continue

            # Price must have pulled back enough to get a better avg
            if signal == "buy" and price >= avg * (1 - add_thresh):
                await asyncio.sleep(2)
                continue
            if signal == "sell" and price <= avg * (1 + add_thresh):
                await asyncio.sleep(2)
                continue

        else:
            # First entry: quality + AO gate
            if quality < qual_min:
                await asyncio.sleep(2)
                continue

            if signal == "buy" and ao <= 0:
                await asyncio.sleep(2)
                continue
            if signal == "sell" and ao >= 0:
                await asyncio.sleep(2)
                continue

        # ── PLACE ORDER ───────────────────────────────────────────────
        size = calc_size(price)
        success = await place_order(signal, size, price)

        if success:
            stack["direction"] = signal
            stack["count"] += 1
            stack["prices"].append(price)
            stack["sizes"].append(size)

            avg_now = stack_avg_price()
            add_label = f"Add #{stack['count']}" if stack["count"] > 1 else "Entry #1"
            tp_target = avg_now * (1 + tp_pct if signal == "buy" else 1 - tp_pct)

            await discord(
                f"{'🟢' if signal == 'buy' else '🔴'} {add_label} — {signal.upper()}",
                (
                    f"**Size:** `{size} contracts`  |  **Price:** `${price:.4f}`\n"
                    f"**Stack:** `{stack['count']}/{MAX_STACK}`  "
                    f"|  **Avg entry:** `${avg_now:.4f}`\n"
                    f"**TP target:** `${tp_target:.4f}` (`{tp_pct * 100:.2f}%`)\n"
                    f"**Regime:** `{regime}`  |  **Quality:** `{quality:.0f}/100`  "
                    f"|  **VolPct:** `{vol_pct:.0%}`\n"
                    f"**WaveRatio:** `{wave_state.wave_ratio:.2f}` "
                    f"(p{wave_state.wr_pct:.0%})  "
                    f"**Bias:** `{wave_state.bias}`  "
                    f"**CurMom:** `{wave_state.cur_ratio:.2f}`\n"
                    f"**AO:** `{ao:.4f}`  |  **Imbalance:** `{imbalance:.2f}`"
                ),
                "green" if signal == "buy" else "red",
            )
            print(
                f"{'🟢' if signal == 'buy' else '🔴'} {add_label} {signal} "
                f"{size}@{price:.4f} | avg={avg_now:.4f} | "
                f"regime={regime} qual={quality:.0f} "
                f"wr={wave_state.wave_ratio:.2f}(p{wave_state.wr_pct:.0%}) "
                f"ao={ao:.4f} tp={tp_pct * 100:.2f}%"
            )

        await asyncio.sleep(2)


# ================================================================
# MAIN
# ================================================================
async def main() -> None:
    # Exchange setup (idempotent — safe on every restart)
    try:
        await exchange.set_margin_mode(MARGIN_MODE, SYMBOL)
        await exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"✅ {MARGIN_MODE.upper()} margin @ {LEVERAGE}x set")
    except Exception as exc:
        print(f"⚠️  Margin/leverage (may already be set): {exc}")

    # FIX: two independent WS tasks — book never waits for trade feed
    asyncio.create_task(handle_trades())
    asyncio.create_task(handle_orderbook())
    await trading_loop()


if __name__ == "__main__":
    asyncio.run(main())
