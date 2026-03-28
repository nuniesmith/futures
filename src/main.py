"""
Futures Multi-Asset Scalper — Entry Point
==========================================
Supervisor architecture: one async worker task per enabled asset.

Modes:
  sim  → live WS feeds, paper trades tracked in Redis (default)
  live → real orders via KuCoin API

Usage:
  python -m src.main              # loads config/futures.yaml
  TRADING_MODE=sim python -m src.main
"""

import asyncio
import json
import os
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import ccxt.pro as ccxt
import optuna
import pandas as pd

from src.logging_config import get_logger, setup_logging
from src.services.config_loader import AssetConfig, FuturesConfig, load_config
from src.services.redis_store import RedisStore
from src.services.report_generator import ReportGenerator

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)

# Thread pool for blocking Optuna work (shared across all workers)
_OPTUNA_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="optuna")


# ═══════════════════════════════════════════════════════════════════
# Wave State (from original)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class WaveState:
    """Persistent wave-analysis state — ported from Ruby §4."""

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
    rma_speed: float = 0.0  # latest RMA speed value


# ═══════════════════════════════════════════════════════════════════
# Indicator functions (preserved EXACTLY from original main.py)
# ═══════════════════════════════════════════════════════════════════


def build_candles(trades_deque: deque, min_ticks: int = 60) -> pd.DataFrame:
    """Build 5-second OHLCV candles from tick stream."""
    if len(trades_deque) < min_ticks:
        return pd.DataFrame()
    df = pd.DataFrame(list(trades_deque))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    ohlc = df["price"].resample("5s").ohlc()
    ohlc["high"] = df["price"].resample("5s").max()
    ohlc["low"] = df["price"].resample("5s").min()
    ohlc["volume"] = df["qty"].resample("5s").sum()
    ohlc["hl2"] = (ohlc["high"] + ohlc["low"]) / 2
    ohlc["open"] = df["price"].resample("5s").first()
    return ohlc.dropna()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder EMA)."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_ao(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> float:
    """Awesome Oscillator — Ruby §2. ao = SMA(hl2,fast) - SMA(hl2,slow)."""
    if len(df) < slow:
        return 0.0
    ao = df["hl2"].rolling(fast).mean() - df["hl2"].rolling(slow).mean()
    return float(ao.iloc[-1])


def compute_vol_pct(df: pd.DataFrame, atr_period: int = 14, lookback: int = 200) -> float:
    """Volatility percentile — Ruby §6. 0.0=calm, 1.0=extreme."""
    if len(df) < 20:
        return 0.5
    atr = _compute_atr(df, atr_period).dropna()
    recent = atr.values[-lookback:]
    cur = float(atr.iloc[-1])
    return float((recent < cur).sum()) / max(len(recent), 1)


def compute_regime(
    df: pd.DataFrame,
    sma_period: int = 200,
    slope_lookback: int = 20,
    slope_threshold: float = 1.0,
    vol_volatile: float = 1.5,
    vol_ranging: float = 0.8,
) -> tuple[str, float]:
    """
    Market regime — Ruby §5.
    TRENDING_UP / TRENDING_DOWN / VOLATILE / RANGING / NEUTRAL
    """
    min_bars = sma_period + slope_lookback
    if len(df) < min_bars:
        return "NEUTRAL", 0.0

    ma = df["close"].rolling(sma_period).mean()
    avg_chg = ma.diff().abs().rolling(100).mean()
    slope = (ma - ma.shift(slope_lookback)) / (avg_chg * slope_lookback + 1e-10)
    slope_n = float(slope.iloc[-1])

    ret_s = df["close"].pct_change().rolling(100).std()
    ret_sma = ret_s.rolling(50).mean()
    vol_n = float(ret_s.iloc[-1] / (ret_sma.iloc[-1] + 1e-10))

    if slope_n > slope_threshold:
        return "TRENDING_UP", slope_n
    if slope_n < -slope_threshold:
        return "TRENDING_DOWN", slope_n
    if vol_n > vol_volatile:
        return "VOLATILE", vol_n
    if vol_n < vol_ranging:
        return "RANGING", vol_n
    return "NEUTRAL", vol_n


def update_wave_state(
    df: pd.DataFrame, ws: WaveState, ema_period: int = 20, rma_alpha: float = 0.1
) -> WaveState:
    """
    Wave analysis — Ruby §4, translated to Python.

    Ruby: speed accumulates (c_rma - o_rma) within each wave.
    Cross EMA up   → completed bear wave → record trough into bear_waves.
    Cross EMA down → completed bull wave → record peak into bull_waves.
    wave_ratio = bull_avg / |bear_avg|

    Stacking gate: wave_ratio > 1.0 → bulls harder → safe to add longs.
                   wave_ratio < 1.0 → bears harder → safe to add shorts.
    """
    if len(df) < 40:
        return ws

    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    c_rma = df["close"].ewm(alpha=rma_alpha, adjust=False).mean()
    o_rma = df["open"].ewm(alpha=rma_alpha, adjust=False).mean()
    speed_series = c_rma - o_rma

    above_now = bool(df["close"].iloc[-1] > ema.iloc[-1])
    above_prev = bool(df["close"].iloc[-2] > ema.iloc[-2])
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
    ws.rma_speed = cur_speed
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
    quality_cfg=None,
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

    # Vol multiplier for volume comparison
    if quality_cfg is not None:
        vm_high = quality_cfg.vol_mult_high
        vm_mid = quality_cfg.vol_mult_mid
        vm_low = quality_cfg.vol_mult_low
        neutral_credit = quality_cfg.neutral_regime_credit
    else:
        vm_high, vm_mid, vm_low = 1.8, 1.2, 0.8
        neutral_credit = 10.0

    vol_mult = vm_high if vol_pct >= 0.6 else vm_mid if vol_pct >= 0.4 else vm_low

    score = 0.0
    score += 20.0 if (bull_bias and ao > 0) or (not bull_bias and ao < 0) else 0.0
    score += (
        15.0 if (bull_bias and close > fast_ema) or (not bull_bias and close < fast_ema) else 0.0
    )
    score += (
        20.0 if (bull_bias and imbalance >= 1.2) or (not bull_bias and imbalance <= 0.8) else 0.0
    )
    score += 25.0 if volume > vol_avg * vol_mult else 0.0

    if bull_bias and regime == "TRENDING_UP":
        score += 20.0
    elif not bull_bias and regime == "TRENDING_DOWN":
        score += 20.0
    elif regime in ("NEUTRAL", "RANGING"):
        score += float(neutral_credit)

    return min(score, 100.0)


def adaptive_tp(vol_pct: float, regime: str, tp_pct_base: float, vol_cfg=None) -> float:
    """
    Scale TP_PCT_BASE by regime and vol percentile — Ruby §7.

    RANGING   → 0.7–1.0x  (collect small wins quickly)
    NEUTRAL   → 1.0–1.3x
    VOLATILE  → 1.2–1.7x  (give room so normal swings don't stop you out)
    TRENDING  → 1.4–2.0x  (ride the wave)
    """
    if vol_cfg is not None and vol_cfg.tp_multipliers:
        mults = vol_cfg.tp_multipliers
        if regime in ("TRENDING_UP", "TRENDING_DOWN") and "trending" in mults:
            m = mults["trending"]
            mult = m.base + vol_pct * m.vol_scale
        elif regime == "VOLATILE" and "volatile" in mults:
            m = mults["volatile"]
            mult = m.base + vol_pct * m.vol_scale
        elif regime == "RANGING" and "ranging" in mults:
            m = mults["ranging"]
            mult = m.base + vol_pct * m.vol_scale
        elif "neutral" in mults:
            m = mults["neutral"]
            mult = m.base + vol_pct * m.vol_scale
        else:
            mult = 1.0 + vol_pct * 0.3
    else:
        # Fallback — same as original
        if regime in ("TRENDING_UP", "TRENDING_DOWN"):
            mult = 1.4 + vol_pct * 0.6
        elif regime == "VOLATILE":
            mult = 1.2 + vol_pct * 0.5
        elif regime == "RANGING":
            mult = 0.7 + vol_pct * 0.3
        else:
            mult = 1.0 + vol_pct * 0.3
    return tp_pct_base * mult


def adaptive_add_threshold(vol_pct: float, base: float) -> float:
    """Price must retrace this fraction before we add another position."""
    return base * (1.0 + vol_pct)  # 0.20%–0.40%


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


# ═══════════════════════════════════════════════════════════════════
# Optuna objective — runs in thread pool, must be pure / self-contained
# ═══════════════════════════════════════════════════════════════════


def _make_objective(
    trades_deque: deque,
    min_ticks: int,
    ao_fast: int,
    ao_slow: int,
    search_space: dict,
    fee_per_trade: float = 0.0007,
):
    """Return a closure suitable for ``study.optimize``.

    ``fee_per_trade`` is the one-way cost (taker + slippage).
    Each signal change incurs 2× this cost (close old + open new).
    """

    def objective(trial: optuna.Trial) -> float:
        df = build_candles(trades_deque, min_ticks)
        if len(df) < 100:
            return -999.0

        ss = search_space
        fast_range = ss.get("fast", {})
        slow_range = ss.get("slow", {})
        imb_range = ss.get("imbalance_thresh", {})
        sl_range = ss.get("sl_pct", {})
        qm_range = ss.get("quality_min", {})
        wp_range = ss.get("wave_pct_gate", {})

        fast = trial.suggest_int(
            "fast", int(fast_range.get("min", 5)), int(fast_range.get("max", 15))
        )
        slow = trial.suggest_int(
            "slow", int(slow_range.get("min", 18)), int(slow_range.get("max", 35))
        )
        _ = trial.suggest_float(
            "imbalance_thresh",
            float(imb_range.get("min", 1.1)),
            float(imb_range.get("max", 1.8)),
            step=float(imb_range.get("step", 0.05)),
        )
        _ = trial.suggest_float(
            "sl_pct",
            float(sl_range.get("min", 0.002)),
            float(sl_range.get("max", 0.006)),
            step=float(sl_range.get("step", 0.0005)),
        )
        _ = trial.suggest_float(
            "quality_min",
            float(qm_range.get("min", 30.0)),
            float(qm_range.get("max", 70.0)),
            step=float(qm_range.get("step", 5.0)),
        )
        _ = trial.suggest_float(
            "wave_pct_gate",
            float(wp_range.get("min", 0.25)),
            float(wp_range.get("max", 0.65)),
            step=float(wp_range.get("step", 0.05)),
        )

        df = df.copy()
        df["fast_ema"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["slow_ema"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["ao"] = df["hl2"].rolling(ao_fast).mean() - df["hl2"].rolling(ao_slow).mean()
        df["signal"] = 0
        df.loc[df["fast_ema"] > df["slow_ema"], "signal"] = 1
        df.loc[df["fast_ema"] < df["slow_ema"], "signal"] = -1

        ao_pass = ((df["signal"] == 1) & (df["ao"] > 0)) | ((df["signal"] == -1) & (df["ao"] < 0))
        df.loc[~ao_pass, "signal"] = 0

        df["returns"] = df["close"].pct_change()
        df["pnl"] = df["signal"].shift(1) * df["returns"]

        # Deduct round-trip fee on every signal change (entry + exit)
        df["sig_change"] = (df["signal"] != df["signal"].shift(1)).astype(int)
        n_trades = int(df["sig_change"].sum())
        fee_cost = n_trades * fee_per_trade * 2  # entry + exit
        total = float(df["pnl"].sum()) - fee_cost
        return total if total != 0 else -999.0

    return objective


def _run_optuna_sync(
    trades_deque: deque,
    min_ticks: int,
    ao_fast: int,
    ao_slow: int,
    search_space: dict,
    n_trials: int,
    timeout_sec: int,
    fee_per_trade: float = 0.0007,
) -> dict | None:
    """Blocking Optuna run — called from thread pool."""
    df = build_candles(trades_deque, min_ticks)
    if len(df) < 100:
        return None
    objective = _make_objective(
        trades_deque, min_ticks, ao_fast, ao_slow, search_space, fee_per_trade
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)
    return study.best_params


# ═══════════════════════════════════════════════════════════════════
# AssetWorker
# ═══════════════════════════════════════════════════════════════════


class AssetWorker:
    """Self-contained trading worker for one asset."""

    def __init__(
        self,
        asset: AssetConfig,
        config: FuturesConfig,
        exchange: ccxt.kucoinfutures,
        store: RedisStore,
        shutdown_event: asyncio.Event,
        contract_spec: dict | None = None,
    ):
        self.asset = asset
        self.config = config
        self.exchange = exchange
        self.store = store
        self.shutdown = shutdown_event
        self.tag = f"[{asset.base}]"
        self.asset_key = asset.key  # short name for Redis keys

        # Contract specification from exchange.load_markets()
        # Keys: contract_size, max_lots, lot_step, precision_amount
        self.contract_spec = contract_spec or {}

        # Order failure tracking — backoff after repeated errors
        self._consecutive_order_failures: int = 0
        self._order_backoff_until: float = 0.0  # timestamp

        # ── Per-worker state ──────────────────────────────────────
        self.trades: deque = deque(maxlen=config.candles.tick_buffer)
        self.orderbook: dict = {"bids": [], "asks": []}

        # Position stack — purely in-memory
        self.stack: dict = {
            "direction": None,  # "buy" | "sell" | None
            "count": 0,
            "prices": [],
            "sizes": [],
            "entry_time": 0.0,  # timestamp of first entry (for min_hold_sec)
        }

        # Optuna best params (updated live)
        sl_default = asset.sl_pct if asset.sl_pct else config.strategy.sl_pct
        self.best_params: dict = {
            "fast": config.strategy.fast_ema,
            "slow": config.strategy.slow_ema,
            "imbalance_thresh": config.strategy.imbalance_thresh,
            "sl_pct": sl_default,
            "quality_min": config.wave.quality_min,
            "wave_pct_gate": config.wave.wave_pct_gate,
        }

        self.wave_state = WaveState()

        # Risk management state (per-worker)
        self.risk_state: dict = {
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "consecutive_losses": 0,
            "last_loss_time": 0.0,
            "day_start": "",
            "paused": False,
            "pause_reason": "",
            "_alerted": False,
        }

        # Timing
        self.last_optimize = 0.0
        self.last_heartbeat = 0.0
        self.last_status_log = 0.0

        # Determine Optuna interval for this asset
        optuna_cfg = config.optuna
        if asset.key in optuna_cfg.fast_assets:
            self.optuna_interval = optuna_cfg.intervals.get(
                "fast_sec", optuna_cfg.intervals.get("default_sec", 1200)
            )
        elif asset.key in optuna_cfg.slow_assets:
            self.optuna_interval = optuna_cfg.intervals.get(
                "slow_sec", optuna_cfg.intervals.get("default_sec", 1200)
            )
        else:
            self.optuna_interval = optuna_cfg.intervals.get("default_sec", 1200)

    # ── Stack helpers ─────────────────────────────────────────────

    def _stack_avg_price(self) -> float:
        if not self.stack["prices"]:
            return 0.0
        total_sz = sum(self.stack["sizes"])
        if total_sz == 0:
            return 0.0
        return sum(p * s for p, s in zip(self.stack["prices"], self.stack["sizes"])) / total_sz

    def _stack_total_size(self) -> float:
        return sum(self.stack["sizes"])

    def _stack_clear(self) -> None:
        self.stack.update(direction=None, count=0, prices=[], sizes=[], entry_time=0.0)

    # ── Risk management ──────────────────────────────────────────

    def _risk_check_daily_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self.risk_state["day_start"] != today:
            self.risk_state["daily_pnl"] = 0.0
            self.risk_state["daily_trades"] = 0
            self.risk_state["paused"] = False
            self.risk_state["pause_reason"] = ""
            self.risk_state["_alerted"] = False
            self.risk_state["day_start"] = today
            logger.info("%s New trading day: %s — counters reset", self.tag, today)

    def _risk_record_trade(self, pnl_usdt: float, was_stop: bool) -> None:
        self.risk_state["daily_pnl"] += pnl_usdt
        self.risk_state["daily_trades"] += 1
        if pnl_usdt < 0:
            self.risk_state["consecutive_losses"] += 1
            if was_stop:
                self.risk_state["last_loss_time"] = time.time()
        else:
            self.risk_state["consecutive_losses"] = 0

    def _risk_can_trade(self) -> tuple[bool, str]:
        self._risk_check_daily_reset()
        cfg = self.config
        capital = cfg.capital.balance_usdt

        # Daily loss limit (per-asset uses asset-specific threshold)
        max_asset_loss = capital * cfg.risk.max_asset_daily_loss_pct
        if self.risk_state["daily_pnl"] <= -max_asset_loss:
            reason = (
                f"Asset daily loss limit hit "
                f"({self.risk_state['daily_pnl']:.2f} / -{max_asset_loss:.2f})"
            )
            self.risk_state["paused"] = True
            self.risk_state["pause_reason"] = reason
            return False, reason

        # Consecutive losses
        max_consec = cfg.risk.max_consecutive_losses
        if self.risk_state["consecutive_losses"] >= max_consec:
            reason = (
                f"Consecutive losses: {self.risk_state['consecutive_losses']} (limit {max_consec})"
            )
            self.risk_state["paused"] = True
            self.risk_state["pause_reason"] = reason
            return False, reason

        # Cooldown after stop-loss
        cooldown = cfg.risk.cooldown_after_loss_sec
        if self.risk_state["last_loss_time"] > 0:
            elapsed = time.time() - self.risk_state["last_loss_time"]
            if elapsed < cooldown:
                return False, f"Post-SL cooldown: {cooldown - elapsed:.0f}s remaining"

        # Per-asset daily trade limit
        asset_limit = cfg.risk.asset_trade_limit
        if self.risk_state["daily_trades"] >= asset_limit:
            reason = f"Asset trade limit reached ({asset_limit})"
            self.risk_state["paused"] = True
            self.risk_state["pause_reason"] = reason
            return False, reason

        # Clear pause
        if self.risk_state["paused"]:
            self.risk_state["paused"] = False
            self.risk_state["pause_reason"] = ""

        return True, ""

    # ── Order sizing ──────────────────────────────────────────────

    def _calc_size(self, price: float) -> float:
        """Calculate order size in **contracts** (lots).

        KuCoin Futures ``create_order(amount=…)`` expects the number of
        contracts, NOT the number of base-currency tokens.  Each contract
        represents ``contract_size`` units of the base currency.

        Formula
        -------
        risk_usd   = capital_slice × risk_per_add_pct
        notional   = risk_usd / sl_dist              (unleveraged exposure)
        raw_tokens = notional / price                 (base-currency qty)
        contracts  = raw_tokens / contract_size       (lot count)

        Leverage is factored in because we're risking a fraction of our
        *leveraged* capital slice (margin × leverage = effective exposure).
        """
        cfg = self.config
        capital = cfg.capital.balance_usdt * self.asset.margin_pct
        sl_dist = self.best_params["sl_pct"]
        leverage = self.asset.leverage
        risk_per_add = cfg.capital.risk_per_add_pct

        # Raw token-denominated size (same as before)
        raw_tokens = (capital * risk_per_add * leverage) / (sl_dist * price)

        # Convert tokens → contracts using exchange contract specification
        contract_size = self.contract_spec.get("contract_size", 1.0)
        lot_step = self.contract_spec.get("lot_step", 1.0)
        max_lots = self.contract_spec.get("max_lots", 1_000_000)

        raw_lots = raw_tokens / contract_size

        # Round to lot step (contracts are typically integers)
        if lot_step >= 1.0:
            lots = max(int(raw_lots), 1)
        else:
            lots = max(round(raw_lots / lot_step) * lot_step, lot_step)

        # Enforce minimum notional (in USDT)
        min_usdt = max(self.asset.min_order_usdt, cfg.capital.min_order_usdt)
        notional = lots * contract_size * price / leverage
        if notional < min_usdt:
            min_lots = (min_usdt * leverage) / (contract_size * price)
            if lot_step >= 1.0:
                lots = max(int(min_lots) + 1, 1)
            else:
                lots = max(round(min_lots / lot_step) * lot_step, lot_step)

        # ── Margin cap ────────────────────────────────────────────
        # Ensure this order's required margin fits within the asset's
        # capital slice.  Divide by max_stack so all potential stack
        # adds can co-exist without exhausting the account balance.
        max_stack = max(cfg.capital.max_stack, 1)
        max_margin_per_add = capital / max_stack
        max_notional_for_margin = max_margin_per_add * leverage
        margin_cap_lots = max_notional_for_margin / (contract_size * price)

        if lot_step >= 1.0:
            margin_cap_lots = int(margin_cap_lots)  # floor to integer
        else:
            margin_cap_lots = int(margin_cap_lots / lot_step) * lot_step

        if margin_cap_lots >= 1 and lots > margin_cap_lots:
            logger.info(
                "%s _calc_size: margin-capped %s → %s lots (slice=$%.2f, max_margin_per_add=$%.2f)",
                self.tag,
                lots,
                margin_cap_lots,
                capital,
                max_margin_per_add,
            )
            lots = margin_cap_lots
        elif margin_cap_lots < 1:
            # Even 1 lot exceeds the per-add margin budget.
            # Allow 1 lot minimum but warn — stacking will be limited.
            if lots > 1:
                lots = 1
            logger.info(
                "%s _calc_size: 1 lot exceeds margin cap "
                "(slice=$%.2f, max_margin=$%.2f, lot_margin=$%.2f); "
                "using min lot — stacking may be limited",
                self.tag,
                capital,
                max_margin_per_add,
                contract_size * price / leverage,
            )

        # Clamp to exchange max lot limit
        lots = min(lots, max_lots)

        if isinstance(lots, float):
            # Clean up floating-point dust for fractional lot steps
            precision = self.contract_spec.get("precision_amount", 0)
            if precision > 0:
                lots = round(lots, precision)

        logger.debug(
            "%s _calc_size: price=%.6f raw_tokens=%.2f contract_size=%.6f "
            "raw_lots=%.2f → lots=%s (max=%s)",
            self.tag,
            price,
            raw_tokens,
            contract_size,
            raw_lots,
            lots,
            max_lots,
        )
        return lots

    # ── Order execution ───────────────────────────────────────────

    async def _place_order(
        self, side: str, size: float, price: float, reduce_only: bool = False
    ) -> bool:
        """Place order — real in live mode, simulated in sim mode."""
        # Check order backoff (skip if we're in a failure cooldown)
        if self._order_backoff_until > 0:
            now = time.time()
            if now < self._order_backoff_until:
                remaining = self._order_backoff_until - now
                logger.debug(
                    "%s Order skipped — backoff %.0fs remaining (%d consecutive failures)",
                    self.tag,
                    remaining,
                    self._consecutive_order_failures,
                )
                return False
            # Backoff expired — reset and allow retry
            logger.info(
                "%s Order backoff expired — retrying (was %d consecutive failures)",
                self.tag,
                self._consecutive_order_failures,
            )
            self._consecutive_order_failures = 0
            self._order_backoff_until = 0.0

        if self.config.is_sim:
            # Simulate fill at last price + slippage
            slippage = self.config.simulation.slippage_pct
            if side == "buy":
                fill_price = price * (1 + slippage)
            else:
                fill_price = price * (1 - slippage)
            logger.info(
                "%s %s SIM %s %.1f @ %.6f (fill %.6f, slip %.4f%%)",
                self.tag,
                self.config.simulation.log_prefix,
                side,
                size,
                price,
                fill_price,
                slippage * 100,
            )
            return True
        else:
            # Live order via ccxt
            params: dict = {"leverage": self.asset.leverage}
            if reduce_only:
                params["reduceOnly"] = True
            try:
                await self.exchange.create_order(
                    symbol=self.asset.symbol,
                    type="market",
                    side=side,
                    amount=size,
                    params=params,
                )
                self._consecutive_order_failures = 0
                return True
            except Exception as exc:
                self._consecutive_order_failures += 1
                failures = self._consecutive_order_failures

                # Exponential backoff: 10s, 20s, 40s, 60s, 120s, max 300s
                backoff = min(10 * (2 ** (failures - 1)), 300)
                self._order_backoff_until = time.time() + backoff

                logger.error(
                    "%s Order error (attempt %d, backoff %ds): %s | "
                    "side=%s size=%s price=%.6f contract_size=%s",
                    self.tag,
                    failures,
                    backoff,
                    exc,
                    side,
                    size,
                    price,
                    self.contract_spec.get("contract_size", "?"),
                )

                # After 5 consecutive failures, pause the worker via risk gate
                if failures >= 5:
                    self.risk_state["paused"] = True
                    self.risk_state["pause_reason"] = (
                        f"Order failures: {failures} consecutive errors — last: {exc}"
                    )
                    logger.warning(
                        "%s PAUSED after %d consecutive order failures. "
                        "Check contract spec and sizing. Will retry in %ds.",
                        self.tag,
                        failures,
                        backoff,
                    )
                return False

    async def _close_position(
        self, reason: str, price: float, ruby_ctx: dict | None = None
    ) -> None:
        """Close the current stack position and record P&L."""
        direction = self.stack["direction"]
        avg = self._stack_avg_price()
        total_size = self._stack_total_size()

        if total_size == 0 or direction is None:
            self._stack_clear()
            return

        close_side = "sell" if direction == "buy" else "buy"
        success = await self._place_order(close_side, total_size, price, reduce_only=True)

        if success:
            raw_pct = ((price - avg) / avg) if direction == "buy" else ((avg - price) / avg)
            lev_pct = raw_pct * self.asset.leverage * 100
            usdt_pnl = raw_pct * total_size * avg
            sign = "+" if raw_pct >= 0 else ""

            logger.info(
                "%s CLOSED %s | avg=%.4f exit=%.4f | %s%.2f%% lev "
                "(%s%.4f USDT) | adds=%d | reason=%s",
                self.tag,
                direction,
                avg,
                price,
                sign,
                lev_pct,
                sign,
                usdt_pnl,
                self.stack["count"],
                reason,
            )

            # Record to Redis
            trade_info = {
                "direction": direction,
                "avg_entry": avg,
                "exit_price": price,
                "size": total_size,
                "adds": self.stack["count"],
                "reason": reason,
                "lev_pct": round(lev_pct, 4),
            }
            if ruby_ctx:
                trade_info["regime"] = ruby_ctx.get("regime", "?")
                trade_info["quality"] = ruby_ctx.get("quality", 0)
                trade_info["wave_ratio"] = ruby_ctx.get("wave_ratio", 0)
                trade_info["ao"] = ruby_ctx.get("ao", 0)
                trade_info["vol_pct"] = ruby_ctx.get("vol_pct", 0)

            await self.store.record_pnl(
                asset=self.asset_key,
                pnl_usdt=round(usdt_pnl, 6),
                pnl_pct=round(raw_pct, 6),
                trade_info=trade_info,
            )

            # Record order
            await self.store.record_order(
                asset=self.asset_key,
                order={
                    "side": close_side,
                    "size": total_size,
                    "price": price,
                    "type": "close",
                    "reason": reason,
                    "pnl_usdt": round(usdt_pnl, 6),
                    "timestamp": time.time(),
                },
            )

            # Update risk state
            self._risk_record_trade(usdt_pnl, "Stop Loss" in reason)

        self._stack_clear()

    # ── WebSocket feeds ───────────────────────────────────────────

    async def _handle_trades(self) -> None:
        """WS trade feed → tick deque. Reconnects automatically via ccxt."""
        logger.info("%s Trade feed connecting for %s...", self.tag, self.asset.symbol)
        reconnect_delay = self.config.streams.reconnect_delay_sec
        max_delay = self.config.streams.reconnect_max_delay_sec
        backoff = self.config.streams.reconnect_backoff_factor
        delay = reconnect_delay

        while not self.shutdown.is_set():
            try:
                trade_data = await self.exchange.watch_trades(self.asset.symbol)
                delay = reconnect_delay  # reset on success
                for t in trade_data:
                    p = float(t["price"])
                    self.trades.append(
                        {
                            "timestamp": t["timestamp"],
                            "price": p,
                            "qty": float(t["amount"]),
                            "open": p,
                            "high": p,
                            "low": p,
                        }
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "%s Trade WS error (retry in %.0fs): %s",
                    self.tag,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff, max_delay)

    async def _handle_orderbook(self) -> None:
        """WS orderbook feed → snapshot. Reconnects automatically via ccxt."""
        logger.info(
            "%s Orderbook feed connecting for %s...",
            self.tag,
            self.asset.symbol,
        )
        depth = self.config.streams.orderbook_depth
        display = self.config.streams.orderbook_display
        reconnect_delay = self.config.streams.reconnect_delay_sec
        max_delay = self.config.streams.reconnect_max_delay_sec
        backoff = self.config.streams.reconnect_backoff_factor
        delay = reconnect_delay

        while not self.shutdown.is_set():
            try:
                book = await self.exchange.watch_order_book(
                    self.asset.symbol,
                    limit=depth,
                )
                delay = reconnect_delay
                self.orderbook = {
                    "bids": book["bids"][:display],
                    "asks": book["asks"][:display],
                }
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "%s Book WS error (retry in %.0fs): %s",
                    self.tag,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff, max_delay)

    # ── Optuna optimization (dispatched to thread pool) ───────────

    async def _maybe_optimize(self) -> None:
        """Run Optuna if interval has elapsed. Non-blocking via thread pool."""
        now = time.time()
        if now - self.last_optimize < self.optuna_interval:
            return

        self.last_optimize = now
        cfg = self.config
        loop = asyncio.get_running_loop()

        # Fee model: taker + slippage per side
        fees = cfg.optuna.fees
        fee_per_trade = fees.get("taker_pct", 0.0006) + fees.get("slippage_pct", 0.0001)

        t_start = time.time()
        new_params = await loop.run_in_executor(
            _OPTUNA_POOL,
            _run_optuna_sync,
            self.trades,
            cfg.candles.min_ticks,
            cfg.awesome_oscillator.fast,
            cfg.awesome_oscillator.slow,
            cfg.optuna.search_space,
            cfg.optuna.n_trials,
            cfg.optuna.timeout_sec,
            fee_per_trade,
        )
        elapsed = time.time() - t_start

        if new_params:
            self.best_params.update(new_params)
            params_str = " ".join(
                f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in new_params.items()
            )
            logger.info(
                "%s Optuna complete (%d trials, %.1fs) → %s",
                self.tag,
                cfg.optuna.n_trials,
                elapsed,
                params_str,
            )
        else:
            logger.debug(
                "%s Optuna skipped — not enough candle data",
                self.tag,
            )

    # ── Heartbeat ─────────────────────────────────────────────────

    async def _send_heartbeat(
        self, candles_count: int = 0, regime: str = "?", quality: float = 0.0
    ) -> None:
        """Send heartbeat to Redis with current status."""
        status = {
            "ticks": len(self.trades),
            "candles": candles_count,
            "stack_dir": self.stack["direction"],
            "stack_count": self.stack["count"],
            "daily_pnl": round(self.risk_state["daily_pnl"], 4),
            "daily_trades": self.risk_state["daily_trades"],
            "regime": regime,
            "quality": round(quality, 1),
            "params": {
                k: round(v, 4) if isinstance(v, float) else v for k, v in self.best_params.items()
            },
        }
        try:
            await self.store.heartbeat(self.asset_key, status)
        except Exception as exc:
            logger.debug("%s Heartbeat error: %s", self.tag, exc)

    # ── Trading loop ──────────────────────────────────────────────

    async def _trading_loop(self) -> None:
        """Core loop — compute signals, manage positions, record to Redis."""
        cfg = self.config
        loop_sleep = cfg.monitoring.loop_sleep_sec
        heartbeat_interval = cfg.monitoring.heartbeat_interval_sec
        ao_fast = cfg.awesome_oscillator.fast
        ao_slow = cfg.awesome_oscillator.slow
        vol_entry_min = cfg.volatility.entry_filter_min_percentile
        vol_entry_max = cfg.volatility.entry_filter_max_percentile
        max_stack = cfg.capital.max_stack

        # Wait for initial tick data
        logger.info(
            "%s Waiting for tick data (need %d ticks)...",
            self.tag,
            cfg.candles.min_ticks,
        )
        while not self.shutdown.is_set() and len(self.trades) < cfg.candles.min_ticks:
            await asyncio.sleep(2)

        if self.shutdown.is_set():
            return

        logger.info("%s Tick data ready — entering trading loop", self.tag)

        while not self.shutdown.is_set():
            now = time.time()

            # ── Optuna re-optimize periodically ──────────────────
            try:
                await self._maybe_optimize()
            except Exception as exc:
                logger.warning("%s Optuna error: %s", self.tag, exc)

            # ── Build candles ────────────────────────────────────
            candles = build_candles(self.trades, cfg.candles.min_ticks)
            if len(candles) < 35:
                await asyncio.sleep(loop_sleep)
                continue

            # ── Guard: wait for order book to warm up ────────────
            if not self.orderbook["bids"] or not self.orderbook["asks"]:
                await asyncio.sleep(loop_sleep)
                continue

            # ── Compute indicators ───────────────────────────────
            self.wave_state = update_wave_state(
                candles,
                self.wave_state,
                ema_period=cfg.wave.ema_period,
                rma_alpha=cfg.wave.rma_alpha,
            )
            vol_pct = compute_vol_pct(
                candles,
                atr_period=cfg.volatility.atr_period,
                lookback=cfg.volatility.percentile_lookback,
            )
            regime, _ = compute_regime(
                candles,
                sma_period=cfg.regime.sma_period,
                slope_lookback=cfg.regime.slope_lookback,
                slope_threshold=cfg.regime.slope_trending_threshold,
                vol_volatile=cfg.regime.vol_volatile_threshold,
                vol_ranging=cfg.regime.vol_ranging_threshold,
            )
            ao = compute_ao(candles, fast=ao_fast, slow=ao_slow)

            fast_ema = float(
                candles["close"].ewm(span=self.best_params["fast"], adjust=False).mean().iloc[-1]
            )
            slow_ema = float(
                candles["close"].ewm(span=self.best_params["slow"], adjust=False).mean().iloc[-1]
            )

            bid_vol = sum(float(b[1]) for b in self.orderbook["bids"])
            ask_vol = sum(float(a[1]) for a in self.orderbook["asks"])
            imbalance = bid_vol / ask_vol if ask_vol > 0 else 1.0

            quality = compute_quality(
                candles,
                ao,
                vol_pct,
                fast_ema,
                slow_ema,
                imbalance,
                regime,
                quality_cfg=cfg.quality,
            )
            tp_pct = adaptive_tp(
                vol_pct,
                regime,
                cfg.strategy.tp_pct_base,
                vol_cfg=cfg.volatility,
            )
            # Enforce TP floor — TP must always exceed round-trip fees
            tp_pct = max(tp_pct, cfg.strategy.tp_pct_floor)
            add_thresh = adaptive_add_threshold(
                vol_pct,
                cfg.strategy.add_threshold_base,
            )
            qual_min = float(self.best_params.get("quality_min", cfg.wave.quality_min))
            wave_gate = float(self.best_params.get("wave_pct_gate", cfg.wave.wave_pct_gate))

            ruby_ctx = {
                "regime": regime,
                "wave_ratio": self.wave_state.wave_ratio,
                "wr_pct": self.wave_state.wr_pct,
                "cur_ratio": self.wave_state.cur_ratio,
                "ao": ao,
                "vol_pct": vol_pct,
                "quality": quality,
                "tp_pct": tp_pct,
                "imbalance": imbalance,
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
            }

            # ── Heartbeat / status log ───────────────────────────
            if now - self.last_heartbeat > heartbeat_interval:
                await self._send_heartbeat(
                    candles_count=len(candles),
                    regime=regime,
                    quality=quality,
                )
                logger.info(
                    "%s regime=%s wave=%.2f(p%.0f%%) bias=%s ao=%.4f "
                    "vol_pct=%.0f%% quality=%.0f tp=%.2f%% imb=%.2f "
                    "stack=%d/%d(%s) daily_pnl=%.4f trades=%d",
                    self.tag,
                    regime,
                    self.wave_state.wave_ratio,
                    self.wave_state.wr_pct * 100,
                    self.wave_state.bias,
                    ao,
                    vol_pct * 100,
                    quality,
                    tp_pct * 100,
                    imbalance,
                    self.stack["count"],
                    max_stack,
                    self.stack["direction"] or "flat",
                    self.risk_state["daily_pnl"],
                    self.risk_state["daily_trades"],
                )
                self.last_heartbeat = now

            # ── Current price ────────────────────────────────────
            # Use last trade price from our tick deque (no REST call needed)
            if not self.trades:
                await asyncio.sleep(loop_sleep)
                continue
            price = float(self.trades[-1]["price"])

            direction = self.stack["direction"]
            avg = self._stack_avg_price()

            # ── TP check ─────────────────────────────────────────
            if direction == "buy" and avg > 0 and price >= avg * (1 + tp_pct):
                await self._close_position("Take Profit ✅", price, ruby_ctx)
                await asyncio.sleep(loop_sleep)
                continue

            if direction == "sell" and avg > 0 and price <= avg * (1 - tp_pct):
                await self._close_position("Take Profit ✅", price, ruby_ctx)
                await asyncio.sleep(loop_sleep)
                continue

            # ── SL check ─────────────────────────────────────────
            sl_pct = self.best_params["sl_pct"]
            if direction == "buy" and avg > 0 and price <= avg * (1 - sl_pct):
                await self._close_position("Stop Loss 🛑", price, ruby_ctx)
                await asyncio.sleep(loop_sleep)
                continue

            if direction == "sell" and avg > 0 and price >= avg * (1 + sl_pct):
                await self._close_position("Stop Loss 🛑", price, ruby_ctx)
                await asyncio.sleep(loop_sleep)
                continue

            # ── Risk management gate (TP/SL above still active) ──
            can_trade, risk_reason = self._risk_can_trade()
            if not can_trade:
                if not self.risk_state["_alerted"]:
                    logger.warning(
                        "%s RISK GATE: %s",
                        self.tag,
                        risk_reason,
                    )
                    self.risk_state["_alerted"] = True
                await asyncio.sleep(loop_sleep)
                continue
            elif self.risk_state["_alerted"]:
                self.risk_state["_alerted"] = False
                logger.info("%s Risk conditions cleared — trading resumed", self.tag)

            # ── Volatility filter ────────────────────────────────
            if vol_pct < vol_entry_min or vol_pct > vol_entry_max:
                await asyncio.sleep(loop_sleep)
                continue

            # ── Entry signal: EMA crossover + book imbalance ─────
            signal: str | None = None
            imb_thresh = self.best_params["imbalance_thresh"]
            if fast_ema > slow_ema and imbalance > imb_thresh:
                signal = "buy"
            elif fast_ema < slow_ema and imbalance < (2.0 - imb_thresh):
                signal = "sell"

            if signal is None:
                await asyncio.sleep(loop_sleep)
                continue

            # ── Opposite signal: close stack (fee-aware) ─────────
            if direction is not None and signal != direction:
                raw_move = ((price - avg) / avg) if direction == "buy" else ((avg - price) / avg)
                hold_elapsed = now - self.stack["entry_time"]
                min_profit = cfg.strategy.min_profit_close_pct
                min_hold = cfg.strategy.min_hold_sec

                if raw_move >= min_profit:
                    # Move covers fees — close it
                    await self._close_position("Signal Reversal ✅", price, ruby_ctx)
                elif hold_elapsed < min_hold:
                    # Too soon + not profitable — ignore the reversal,
                    # let TP/SL handle the exit instead
                    logger.debug(
                        "%s Signal reversal BLOCKED — hold %.0fs < %.0fs, "
                        "move %.4f%% < %.4f%% min profit",
                        self.tag,
                        hold_elapsed,
                        min_hold,
                        raw_move * 100,
                        min_profit * 100,
                    )
                    await asyncio.sleep(loop_sleep)
                    continue
                elif raw_move < -cfg.strategy.sl_pct * 0.5:
                    # Reversal while already losing more than half the SL
                    # distance — cut it, the signal is confirming against us
                    await self._close_position("Signal Reversal 🔄 (losing)", price, ruby_ctx)
                else:
                    # Held long enough but move doesn't cover fees and
                    # we're not deeply underwater — let TP/SL handle it
                    logger.info(
                        "%s Signal reversal HELD — move %.4f%% < %.4f%% "
                        "min profit (held %.0fs). Waiting for TP/SL.",
                        self.tag,
                        raw_move * 100,
                        min_profit * 100,
                        hold_elapsed,
                    )
                    await asyncio.sleep(loop_sleep)
                    continue

            direction = self.stack["direction"]  # re-read after possible close

            # ── Stack full → hold ────────────────────────────────
            if direction == signal and self.stack["count"] >= max_stack:
                await asyncio.sleep(loop_sleep)
                continue

            is_add = direction == signal and self.stack["count"] > 0

            # ── GATES ────────────────────────────────────────────
            if is_add:
                # Wave + regime gates for adds #2 and #3
                if not wave_gate_ok(
                    ws=self.wave_state,
                    direction=signal,
                    gate=wave_gate,
                ):
                    await asyncio.sleep(loop_sleep)
                    continue

                if not regime_stack_ok(regime=regime, direction=signal):
                    await asyncio.sleep(loop_sleep)
                    continue

                # Price must have pulled back enough
                if signal == "buy" and price >= avg * (1 - add_thresh):
                    await asyncio.sleep(loop_sleep)
                    continue
                if signal == "sell" and price <= avg * (1 + add_thresh):
                    await asyncio.sleep(loop_sleep)
                    continue
            else:
                # First entry: quality + AO gate
                if quality < qual_min:
                    await asyncio.sleep(loop_sleep)
                    continue

                if signal == "buy" and ao <= 0:
                    await asyncio.sleep(loop_sleep)
                    continue
                if signal == "sell" and ao >= 0:
                    await asyncio.sleep(loop_sleep)
                    continue

            # ── PLACE ORDER / RECORD SIGNAL ──────────────────────
            size = self._calc_size(price)
            success = await self._place_order(signal, size, price)

            if success:
                if self.stack["count"] == 0:
                    self.stack["entry_time"] = now  # record time of first fill
                self.stack["direction"] = signal
                self.stack["count"] += 1
                self.stack["prices"].append(price)
                self.stack["sizes"].append(size)

                avg_now = self._stack_avg_price()
                add_label = f"Add #{self.stack['count']}" if self.stack["count"] > 1 else "Entry #1"
                tp_target = avg_now * (1 + tp_pct if signal == "buy" else 1 - tp_pct)

                logger.info(
                    "%s %s %s %.1f@%.4f | avg=%.4f | tp_target=%.4f | "
                    "regime=%s qual=%.0f wr=%.2f(p%.0f%%) ao=%.4f "
                    "tp=%.2f%%",
                    self.tag,
                    add_label,
                    signal,
                    size,
                    price,
                    avg_now,
                    tp_target,
                    regime,
                    quality,
                    self.wave_state.wave_ratio,
                    self.wave_state.wr_pct * 100,
                    ao,
                    tp_pct * 100,
                )

                # Record signal to Redis
                await self.store.record_signal(
                    self.asset_key,
                    {
                        "side": signal,
                        "price": price,
                        "size": size,
                        "add_label": add_label,
                        "avg_entry": avg_now,
                        "tp_target": tp_target,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                        "regime": regime,
                        "quality": quality,
                        "ao": ao,
                        "vol_pct": vol_pct,
                        "imbalance": imbalance,
                        "wave_ratio": self.wave_state.wave_ratio,
                        "wr_pct": self.wave_state.wr_pct,
                        "bias": self.wave_state.bias,
                        "cur_ratio": self.wave_state.cur_ratio,
                        "stack_count": self.stack["count"],
                        "mode": "sim" if cfg.is_sim else "live",
                    },
                )

                # Record entry order
                await self.store.record_order(
                    self.asset_key,
                    {
                        "side": signal,
                        "size": size,
                        "price": price,
                        "type": "entry",
                        "add_label": add_label,
                        "timestamp": time.time(),
                    },
                )

            await asyncio.sleep(loop_sleep)

    # ── Main entry point ──────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point — connect WS, enter trading loop."""
        logger.info(
            "%s Starting worker for %s (%s) @ %dx leverage",
            self.tag,
            self.asset.symbol,
            self.asset.base,
            self.asset.leverage,
        )

        # Set leverage on exchange (live mode only)
        # NOTE: KuCoin Futures ccxt does NOT support set_margin_mode()
        # for isolated margin — it must be configured on the exchange
        # web UI.  We only call set_leverage() here.
        if self.config.is_live:
            try:
                await self.exchange.set_leverage(
                    self.asset.leverage,
                    self.asset.symbol,
                )
                logger.info(
                    "%s Leverage set to %dx for %s",
                    self.tag,
                    self.asset.leverage,
                    self.asset.symbol,
                )
            except Exception as exc:
                exc_msg = str(exc)
                # ccxt raises this when it can't set marginMode — safe
                # to ignore because leverage is managed per-order via
                # the params dict in _place_order().
                if "marginMode" in exc_msg or "already" in exc_msg.lower():
                    logger.debug(
                        "%s Leverage note (non-fatal): %s",
                        self.tag,
                        exc_msg,
                    )
                else:
                    logger.warning(
                        "%s Leverage setup issue: %s",
                        self.tag,
                        exc,
                    )

        # Start WS feeds as subtasks
        trade_task = asyncio.create_task(
            self._handle_trades(),
            name=f"ws-trades-{self.asset_key}",
        )
        book_task = asyncio.create_task(
            self._handle_orderbook(),
            name=f"ws-book-{self.asset_key}",
        )

        try:
            await self._trading_loop()
        except asyncio.CancelledError:
            logger.info("%s Worker cancelled — shutting down", self.tag)
        except Exception as exc:
            logger.exception("%s Worker crashed: %s", self.tag, exc)
        finally:
            trade_task.cancel()
            book_task.cancel()
            # Wait for subtasks to finish cancellation
            for task in (trade_task, book_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("%s Worker stopped", self.tag)


# ═══════════════════════════════════════════════════════════════════
# Supervisor / main()
# ═══════════════════════════════════════════════════════════════════


async def _fetch_exchange_balance(exchange) -> float | None:
    """Fetch the USDT balance from KuCoin Futures account.

    Returns the total USDT balance, or None if the fetch fails.
    """
    try:
        balance = await exchange.fetch_balance()
        # ccxt normalises balances into balance['USDT'] = {free, used, total}
        usdt = balance.get("USDT", {})
        total = usdt.get("total")
        if total is not None:
            return float(total)
        # Fallback: some ccxt versions nest under balance['total']['USDT']
        total_map = balance.get("total", {})
        if "USDT" in total_map and total_map["USDT"] is not None:
            return float(total_map["USDT"])
        logger.warning("Could not parse USDT balance from exchange response")
        return None
    except Exception as exc:
        logger.warning("Failed to fetch exchange balance: %s", exc)
        return None


async def main() -> None:
    """Entry point — load config, start supervisor."""
    setup_logging()
    config = load_config()

    # ── Redis ─────────────────────────────────────────────────────
    store = RedisStore(
        redis_url=config.redis_url,
        password=config.redis_password or None,
        prefix=config.redis.key_prefix,
    )
    await store.connect()

    # ── Exchange (shared across all workers) ──────────────────────
    exchange = ccxt.kucoinfutures(
        {
            "apiKey": config.exchange.credentials.api_key,
            "secret": config.exchange.credentials.api_secret,
            "password": config.exchange.credentials.passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    # ── Load market specs (contract sizes, lot limits) ────────────
    try:
        await exchange.load_markets()
        logger.info("Loaded %d markets from KuCoin Futures", len(exchange.markets))
    except Exception as exc:
        logger.warning("Could not load markets: %s (will use defaults)", exc)

    # Build per-asset contract specs from exchange market data
    # ccxt keys markets by unified symbol (e.g. "BTC/USDT:USDT"),
    # but our config uses the exchange-native ID (e.g. "XBTUSDTM").
    # Build an id→market index so we can look up by native ID.
    markets_by_id: dict = {}
    for _mkt_key, _mkt_val in getattr(exchange, "markets", {}).items():
        eid = _mkt_val.get("id")
        if eid:
            markets_by_id[eid] = _mkt_val

    contract_specs: dict[str, dict] = {}
    for name, asset in config.enabled_assets.items():
        spec: dict = {
            "contract_size": 1.0,
            "lot_step": 1.0,
            "max_lots": 1_000_000,
            "precision_amount": 0,
        }
        if asset.symbol in markets_by_id:
            mkt = markets_by_id[asset.symbol]
            # contractSize: base-currency units per 1 contract
            cs = mkt.get("contractSize")
            if cs is not None and float(cs) > 0:
                spec["contract_size"] = float(cs)
            # Lot step (amount step / minimum lot increment)
            limits = mkt.get("limits", {})
            amount_limits = limits.get("amount", {})
            if amount_limits.get("min") is not None:
                spec["lot_step"] = float(amount_limits["min"])
            if amount_limits.get("max") is not None:
                spec["max_lots"] = float(amount_limits["max"])
            # Precision (decimal places for amount)
            precision = mkt.get("precision", {})
            if precision.get("amount") is not None:
                spec["precision_amount"] = int(precision["amount"])
            logger.info(
                "  %-10s %s  contract=%.6f %s  lot_step=%s  max=%s",
                name,
                asset.symbol,
                spec["contract_size"],
                asset.base,
                spec["lot_step"],
                spec["max_lots"],
            )
        else:
            logger.warning(
                "  %-10s %s  NOT FOUND in exchange markets — using defaults",
                name,
                asset.symbol,
            )
        contract_specs[name] = spec

    # ── Sync capital with real exchange balance (live mode) ───────
    config_capital = config.capital.balance_usdt
    if config.is_live:
        real_balance = await _fetch_exchange_balance(exchange)
        if real_balance is not None:
            config.capital.balance_usdt = real_balance
            logger.info(
                "Live balance from KuCoin: $%.2f USDT (config was $%.2f)",
                real_balance,
                config_capital,
            )
        else:
            logger.warning(
                "Could not fetch live balance — falling back to config capital $%.2f",
                config_capital,
            )
    else:
        logger.info("Sim mode — using config capital $%.2f", config_capital)

    logger.info("=" * 60)
    logger.info("  Futures Multi-Asset Scalper")
    logger.info("  Mode: %s", "SIMULATION" if config.is_sim else "LIVE")
    logger.info("  Assets: %d enabled", len(config.enabled_assets))
    for name, asset in config.enabled_assets.items():
        logger.info(
            "    %-10s %s  lev=%dx  margin=%.0f%%",
            name,
            asset.symbol,
            asset.leverage,
            asset.margin_pct * 100,
        )
    logger.info("  Capital: $%.2f", config.capital.balance_usdt)
    if config.is_live and config.capital.balance_usdt != config_capital:
        logger.info("  (synced from exchange — config default was $%.2f)", config_capital)
    logger.info("=" * 60)

    # ── Shutdown coordination ─────────────────────────────────────
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    # ── Spawn one worker per enabled asset ────────────────────────
    worker_tasks: list[asyncio.Task] = []
    worker_instances: list[AssetWorker] = []
    for name, asset in config.enabled_assets.items():
        worker = AssetWorker(
            asset,
            config,
            exchange,
            store,
            shutdown_event,
            contract_spec=contract_specs.get(name),
        )
        worker_instances.append(worker)
        task = asyncio.create_task(worker.run(), name=f"worker-{name}")
        worker_tasks.append(task)
        logger.info(
            "  Spawned worker: %s (%s) @ %dx leverage, margin=%.0f%%",
            name,
            asset.symbol,
            asset.leverage,
            asset.margin_pct * 100,
        )

    # ── Spawn report generator ────────────────────────────────────
    report_gen = ReportGenerator(redis_store=store)
    report_task = asyncio.create_task(
        report_gen.run_scheduled(),
        name="report-scheduler",
    )
    logger.info("  Spawned report scheduler")

    # ── Supervisor monitor loop ───────────────────────────────────
    worker_timeout = config.monitoring.worker_timeout_sec
    summary_interval = config.monitoring.summary_interval_sec
    last_summary = time.time()

    try:
        while not shutdown_event.is_set():
            now = time.time()

            # Check for crashed workers and log
            for i, task in enumerate(worker_tasks):
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        name = task.get_name()
                        logger.error(
                            "Worker %s crashed: %s — restarting",
                            name,
                            exc,
                        )
                        # Restart the crashed worker
                        w = worker_instances[i]
                        new_task = asyncio.create_task(
                            w.run(),
                            name=name,
                        )
                        worker_tasks[i] = new_task

            # Check heartbeats for stale workers
            try:
                heartbeats = await store.get_heartbeats()
                for asset_key, hb in heartbeats.items():
                    last_seen = hb.get("last_seen", 0)
                    age = now - last_seen
                    if age > worker_timeout:
                        logger.warning(
                            "Worker %s heartbeat stale (%.0fs ago, timeout=%ds)",
                            asset_key,
                            age,
                            worker_timeout,
                        )
            except Exception as exc:
                logger.debug("Heartbeat check error: %s", exc)

            # Aggregate portfolio summary
            if now - last_summary > summary_interval:
                try:
                    total_daily = await store.get_daily_pnl()
                    logger.info(
                        "── Portfolio Summary ── Daily PnL: $%.4f | Workers: %d running | Mode: %s",
                        total_daily,
                        sum(1 for t in worker_tasks if not t.done()),
                        "SIM" if config.is_sim else "LIVE",
                    )
                    # Per-asset breakdown
                    for name in config.enabled_assets:
                        asset_pnl = await store.get_daily_pnl(asset=name)
                        if asset_pnl != 0:
                            logger.info(
                                "  %-10s daily PnL: $%.4f",
                                name,
                                asset_pnl,
                            )
                except Exception as exc:
                    logger.debug("Summary error: %s", exc)
                last_summary = now

            # Sleep until next check (or shutdown)
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=min(
                        config.monitoring.heartbeat_interval_sec,
                        30,
                    ),
                )
            except asyncio.TimeoutError:
                pass  # normal — just loop again

    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutdown signal received — stopping all workers...")

        # Signal all workers to stop
        shutdown_event.set()

        # Close exchange FIRST — tears down WS connections before we
        # cancel tasks, so ccxt doesn't raise NetworkError on orphaned
        # futures that nobody is awaiting.
        try:
            await exchange.close()
        except Exception:
            pass

        # Small grace period for WS teardown to propagate
        await asyncio.sleep(0.2)

        # Cancel worker tasks
        for task in worker_tasks:
            task.cancel()

        # Cancel report scheduler
        report_task.cancel()

        # Wait for all tasks to finish (with timeout)
        all_tasks = worker_tasks + [report_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Suppress expected shutdown exceptions, log anything unexpected
        for task, result in zip(all_tasks, results):
            if result is None or isinstance(result, (asyncio.CancelledError,)):
                continue
            if isinstance(result, Exception):
                # ccxt NetworkError / ConnectionClosed during teardown — expected
                err_name = type(result).__name__
                if "NetworkError" in err_name or "ConnectionClosed" in err_name:
                    continue
                logger.warning(
                    "Unexpected error during shutdown of %s: %s",
                    task.get_name(),
                    result,
                )

        # Close Redis
        try:
            await store.close()
        except Exception:
            pass

        # Shutdown thread pool
        _OPTUNA_POOL.shutdown(wait=False)

        logger.info("=" * 60)
        logger.info("  Shutdown complete")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
