"""
ORB Simulator — Auto-Labeling Engine for CNN Training
======================================================
Replays historical 1-minute bar data through the exact same ORB detection +
Bridge-style bracket logic used in live trading, then assigns ground-truth
labels suitable for supervised learning.

Labels produced:
  - ``good_long``   — Long breakout hit TP1 before SL within holding window.
  - ``good_short``  — Short breakout hit TP1 before SL within holding window.
  - ``bad_long``    — Long breakout hit SL first (or timed out without TP).
  - ``bad_short``   — Short breakout hit SL first (or timed out without TP).
  - ``no_trade``    — No valid ORB breakout was detected in the window.

The simulator is intentionally conservative — it mirrors Bridge.cs bracket
sizing (ATR-based SL/TP) so that CNN training data reflects *real* execution
outcomes, not theoretical ones.

Public API:
    from lib.analysis.orb_simulator import (
        simulate_orb_outcome,
        simulate_batch,
        ORBSimResult,
        BracketConfig,
    )

    result = simulate_orb_outcome(bars_1m, symbol="MGC")
    #  result.label        → "good_long"
    #  result.direction     → "LONG"
    #  result.entry         → 2345.60
    #  result.to_dict()     → JSON-friendly dict for labels.csv

Design:
  - Pure functions — no Redis, no side-effects, fully testable.
  - Parameterised via BracketConfig so you can sweep SL/TP ratios.
  - Uses the same ATR computation as orb.py for consistency.
  - Thread-safe: no shared mutable state.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.orb_simulator")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BracketConfig:
    """Bracket parameters matching Bridge.cs risk logic.

    These map directly to the Bridge properties:
      - sl_atr_mult → StopLossATRMultiplier  (default 1.5)
      - tp1_atr_mult → Target1ATRMultiplier   (default 2.0)
      - tp2_atr_mult → Target2ATRMultiplier   (default 3.0, optional)
      - max_hold_bars → maximum bars to hold before labelling timeout
    """

    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 2.0
    tp2_atr_mult: float = 3.0
    max_hold_bars: int = 120  # ~2 hours of 1-min bars
    atr_period: int = 14

    # Opening range parameters
    or_start: dt_time = dt_time(9, 30)
    or_end: dt_time = dt_time(10, 0)
    or_minutes: int = 30
    min_or_bars: int = 5

    # Pre-market window (for NR7 / pm range extraction)
    pm_start: dt_time = dt_time(0, 0)
    pm_end: dt_time = dt_time(8, 20)

    # Breakout confirmation: require close beyond ORB level (not just wick)
    require_close_break: bool = True

    # Minimum ORB range (in ATR fraction) to avoid tiny-range noise
    min_or_range_atr_frac: float = 0.3


DEFAULT_BRACKET = BracketConfig()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ORBSimResult:
    """Result of a single ORB simulation."""

    # Label
    label: str = "no_trade"  # good_long, bad_long, good_short, bad_short, no_trade

    # Trade details
    symbol: str = ""
    direction: str = ""  # "LONG", "SHORT", or ""
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0

    # ORB details
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr: float = 0.0

    # Quality heuristic (0–100)
    quality_pct: int = 0

    # Outcome details
    outcome: str = ""  # "tp1_hit", "sl_hit", "timeout", "no_breakout", etc.
    pnl_r: float = 0.0  # P&L in R-multiples (1.0 = 1R win, -1.0 = 1R loss)
    hold_bars: int = 0  # how many bars the trade was held
    breakout_bar_idx: int = -1

    # Timing
    or_start_time: str = ""
    breakout_time: str = ""
    exit_time: str = ""
    simulated_at: str = ""

    # Pre-market context
    pm_high: float = 0.0
    pm_low: float = 0.0

    # NR7 flag (narrow range day)
    nr7: bool = False

    # Volume context
    breakout_volume_ratio: float = 0.0  # breakout bar vol / avg vol

    # Error
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON/CSV-friendly dict."""
        return {
            "label": self.label,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": round(self.entry, 6),
            "sl": round(self.sl, 6),
            "tp1": round(self.tp1, 6),
            "tp2": round(self.tp2, 6),
            "or_high": round(self.or_high, 6),
            "or_low": round(self.or_low, 6),
            "or_range": round(self.or_range, 6),
            "atr": round(self.atr, 6),
            "quality_pct": self.quality_pct,
            "outcome": self.outcome,
            "pnl_r": round(self.pnl_r, 3),
            "hold_bars": self.hold_bars,
            "or_start_time": self.or_start_time,
            "breakout_time": self.breakout_time,
            "exit_time": self.exit_time,
            "pm_high": round(self.pm_high, 6),
            "pm_low": round(self.pm_low, 6),
            "nr7": self.nr7,
            "breakout_volume_ratio": round(self.breakout_volume_ratio, 3),
            "error": self.error,
        }

    @property
    def is_winner(self) -> bool:
        return self.label.startswith("good_")

    @property
    def is_trade(self) -> bool:
        return self.label != "no_trade"


# ---------------------------------------------------------------------------
# ATR computation (matches orb.py and volatility.py)
# ---------------------------------------------------------------------------


def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """Compute ATR using simple moving average of True Range.

    Matches the implementation in orb.py for consistency.
    Returns 0.0 if insufficient data.
    """
    n = len(closes)
    if n < 2:
        return float(highs[0] - lows[0]) if n > 0 else 0.0

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    if n >= period:
        return float(np.mean(tr[-period:]))
    return float(np.mean(tr))


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _localize_to_est(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is tz-aware in Eastern Time."""
    df = df.copy()
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        if str(idx.tz) != str(_EST):
            df = df.tz_convert(_EST)
    else:
        with contextlib.suppress(Exception):
            df.index = idx.tz_localize(_EST)
    return df


def _safe_time(idx) -> dt_time | None:
    """Extract time from an index entry, returning None on failure."""
    try:
        return idx.time() if hasattr(idx, "time") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_orb_outcome(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
) -> ORBSimResult:
    """Simulate an ORB trade on a window of 1-minute bars.

    This replays the exact logic used by Ruby.cs (ORB detection) and
    Bridge.cs (bracket sizing) to produce a ground-truth label.

    Algorithm:
      1. Identify the opening range (OR) from bars in [or_start, or_end).
      2. Compute ATR from all available bars.
      3. Scan post-OR bars for the first close beyond OR high/low.
      4. Set entry, SL, TP1, TP2 using Bridge-style ATR multiples.
      5. Walk forward bar-by-bar to determine outcome.
      6. Assign label: good_long, bad_long, good_short, bad_short, no_trade.

    Args:
        bars_1m: 1-minute OHLCV DataFrame covering at least the OR period
                 and subsequent trading hours.  DatetimeIndex preferred.
        symbol: Instrument symbol for labelling.
        config: BracketConfig with SL/TP/holding parameters.
        bars_daily: Optional daily bars for NR7 detection (at least 7 rows).

    Returns:
        ORBSimResult with the label, trade details, and quality heuristic.
    """
    cfg = config or DEFAULT_BRACKET
    result = ORBSimResult(symbol=symbol, simulated_at=datetime.now(_EST).isoformat())

    # --- Validate input ---
    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data provided"
        return result

    required = {"High", "Low", "Close"}
    missing = required - set(bars_1m.columns)
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(bars_1m) < cfg.min_or_bars + 10:
        result.error = f"Insufficient bars ({len(bars_1m)})"
        return result

    # Localise index to Eastern Time
    df = _localize_to_est(bars_1m)
    df = df.sort_index()

    # Cast to float
    highs = df["High"].astype(float).values
    lows = df["Low"].astype(float).values
    closes = df["Close"].astype(float).values
    has_volume = "Volume" in df.columns
    volumes = df["Volume"].astype(float).values if has_volume else np.ones(len(df))
    has_open = "Open" in df.columns
    opens = df["Open"].astype(float).values if has_open else closes.copy()

    # --- Step 1: Identify Opening Range ---
    times = None
    with contextlib.suppress(Exception):
        times = df.index.time

    if times is None:
        result.error = "Cannot extract time from index"
        return result

    or_mask = (times >= cfg.or_start) & (times < cfg.or_end)
    or_indices = np.where(or_mask)[0]

    if len(or_indices) < cfg.min_or_bars:
        result.error = f"Only {len(or_indices)} bars in OR window (need {cfg.min_or_bars})"
        result.outcome = "insufficient_or_bars"
        return result

    or_high = float(highs[or_indices].max())
    or_low = float(lows[or_indices].min())
    or_range = or_high - or_low

    result.or_high = or_high
    result.or_low = or_low
    result.or_range = or_range

    with contextlib.suppress(Exception):
        result.or_start_time = str(df.index[or_indices[0]])

    if or_high <= 0 or or_low <= 0 or or_high <= or_low:
        result.error = f"Invalid OR: high={or_high}, low={or_low}"
        result.outcome = "invalid_or"
        return result

    # --- Step 2: Compute ATR ---
    atr = _compute_atr(highs, lows, closes, period=cfg.atr_period)
    result.atr = atr

    if atr <= 0:
        result.error = "ATR is zero"
        result.outcome = "zero_atr"
        return result

    # Check minimum OR range
    if or_range < atr * cfg.min_or_range_atr_frac:
        result.error = f"OR range {or_range:.4f} < {cfg.min_or_range_atr_frac}x ATR ({atr:.4f})"
        result.outcome = "or_too_narrow"
        return result

    # --- Pre-market range ---
    pm_mask = (times >= cfg.pm_start) & (times < cfg.pm_end)
    pm_indices = np.where(pm_mask)[0]
    if len(pm_indices) > 0:
        result.pm_high = float(highs[pm_indices].max())
        result.pm_low = float(lows[pm_indices].min())

    # --- NR7 detection ---
    if bars_daily is not None and len(bars_daily) >= 7:
        try:
            d_highs = bars_daily["High"].astype(float).values[-7:]
            d_lows = bars_daily["Low"].astype(float).values[-7:]
            daily_ranges = d_highs - d_lows
            today_range = daily_ranges[-1]
            result.nr7 = bool(today_range <= np.min(daily_ranges))
        except Exception:
            pass

    # --- Step 3: Scan for breakout ---
    # Post-OR bars: everything after or_end
    post_or_mask = times >= cfg.or_end
    post_or_indices = np.where(post_or_mask)[0]

    if len(post_or_indices) == 0:
        result.outcome = "no_post_or_bars"
        return result

    direction: str | None = None
    breakout_idx: int | None = None
    entry_price: float = 0.0

    for idx in post_or_indices:
        if cfg.require_close_break:
            bar_val = closes[idx]
        else:
            bar_val_high = highs[idx]
            bar_val_low = lows[idx]

        # Long breakout: bar breaks above OR high
        if cfg.require_close_break:
            if bar_val > or_high:
                direction = "LONG"
                # Entry is the worse of OR high and bar open (simulate fill)
                entry_price = max(or_high, opens[idx])
                breakout_idx = idx
                break
        else:
            if bar_val_high > or_high:
                direction = "LONG"
                entry_price = max(or_high, opens[idx])
                breakout_idx = idx
                break

        # Short breakout: bar breaks below OR low
        if cfg.require_close_break:
            if bar_val < or_low:
                direction = "SHORT"
                entry_price = min(or_low, opens[idx])
                breakout_idx = idx
                break
        else:
            if bar_val_low < or_low:
                direction = "SHORT"
                entry_price = min(or_low, opens[idx])
                breakout_idx = idx
                break

    if direction is None or breakout_idx is None:
        result.outcome = "no_breakout"
        return result

    result.direction = direction
    result.entry = entry_price
    result.breakout_bar_idx = breakout_idx

    with contextlib.suppress(Exception):
        result.breakout_time = str(df.index[breakout_idx])

    # Volume ratio at breakout bar
    avg_vol = float(np.mean(volumes[max(0, breakout_idx - 20) : breakout_idx])) if breakout_idx > 0 else 1.0
    if avg_vol > 0:
        result.breakout_volume_ratio = float(volumes[breakout_idx] / avg_vol)

    # --- Step 4: Compute brackets (Bridge-style) ---
    sl_dist = atr * cfg.sl_atr_mult
    tp1_dist = atr * cfg.tp1_atr_mult
    tp2_dist = atr * cfg.tp2_atr_mult

    if direction == "LONG":
        result.sl = entry_price - sl_dist
        result.tp1 = entry_price + tp1_dist
        result.tp2 = entry_price + tp2_dist
    else:
        result.sl = entry_price + sl_dist
        result.tp1 = entry_price - tp1_dist
        result.tp2 = entry_price - tp2_dist

    # --- Step 5: Walk forward to determine outcome ---
    max_exit_idx = min(breakout_idx + cfg.max_hold_bars, len(df))
    exit_bars = range(breakout_idx + 1, max_exit_idx)

    hit_tp1 = False
    hit_sl = False
    exit_idx = breakout_idx

    for bar_idx in exit_bars:
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]

        if direction == "LONG":
            # Check SL first (conservative — if both hit in same bar, SL wins)
            if bar_low <= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_high >= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break
        else:  # SHORT
            if bar_high >= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_low <= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break

    result.hold_bars = exit_idx - breakout_idx

    with contextlib.suppress(Exception):
        result.exit_time = str(df.index[exit_idx])

    # --- Step 6: Assign label ---
    if hit_tp1:
        result.label = f"good_{direction.lower()}"
        result.outcome = "tp1_hit"
        result.pnl_r = cfg.tp1_atr_mult / cfg.sl_atr_mult  # R-multiple
    elif hit_sl:
        result.label = f"bad_{direction.lower()}"
        result.outcome = "sl_hit"
        result.pnl_r = -1.0
    else:
        # Timeout — check if trade was in profit at expiry
        exit_close = closes[exit_idx] if exit_idx < len(closes) else entry_price
        if direction == "LONG":
            unrealised_r = (exit_close - entry_price) / sl_dist if sl_dist > 0 else 0.0
        else:
            unrealised_r = (entry_price - exit_close) / sl_dist if sl_dist > 0 else 0.0

        result.pnl_r = round(unrealised_r, 3)

        # Timeout with small gain is still "bad" for training purposes —
        # we only want clear TP1 hits as "good".
        result.label = f"bad_{direction.lower()}"
        result.outcome = "timeout"

    # --- Quality heuristic ---
    # Approximate the Ruby quality score: higher when OR is tight relative to
    # ATR, volume confirms, and NR7 is active.
    quality = 50.0  # base

    # OR range vs ATR: tighter OR + strong ATR = more coiled energy
    if or_range > 0 and atr > 0:
        range_ratio = atr / or_range
        quality += min(20.0, range_ratio * 10.0)

    # Volume confirmation
    if result.breakout_volume_ratio > 1.5:
        quality += 10.0
    elif result.breakout_volume_ratio > 1.2:
        quality += 5.0

    # NR7 bonus
    if result.nr7:
        quality += 15.0

    # Pre-market confluence
    if (
        direction == "LONG"
        and result.pm_high > 0
        and entry_price >= result.pm_high
        or direction == "SHORT"
        and result.pm_low > 0
        and entry_price <= result.pm_low
    ):
        quality += 5.0

    result.quality_pct = min(99, max(0, int(quality)))

    return result


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------


def simulate_batch(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    window_size: int = 240,
    step_size: int = 30,
    min_window_bars: int = 60,
) -> list[ORBSimResult]:
    """Run ORB simulation across sliding windows of bar data.

    This is the main entry point for the dataset generator.  It slides a
    window of ``window_size`` bars across the input data, stepping by
    ``step_size``, and simulates one ORB trade per window.

    This produces many training examples from a single day's data by
    varying the entry point within the session.

    Args:
        bars_1m: Full 1-minute OHLCV data (e.g. 60–120 trading days).
        symbol: Instrument symbol.
        config: BracketConfig for SL/TP/holding parameters.
        bars_daily: Daily bars for NR7 detection (optional).
        window_size: Number of 1-min bars per simulation window (default 240 = 4h).
        step_size: Step between windows (default 30 = 30 min).
        min_window_bars: Minimum bars in a window to attempt simulation.

    Returns:
        List of ORBSimResult (includes no_trade results for dataset balance).
    """
    cfg = config or DEFAULT_BRACKET
    results: list[ORBSimResult] = []

    if bars_1m is None or bars_1m.empty:
        return results

    n = len(bars_1m)

    for start in range(0, n - min_window_bars, step_size):
        end = min(start + window_size, n)
        window = bars_1m.iloc[start:end]

        if len(window) < min_window_bars:
            continue

        try:
            result = simulate_orb_outcome(
                window,
                symbol=symbol,
                config=cfg,
                bars_daily=bars_daily,
            )
            results.append(result)
        except Exception as exc:
            logger.debug("Simulation failed at offset %d: %s", start, exc)
            results.append(
                ORBSimResult(
                    symbol=symbol,
                    error=str(exc),
                    simulated_at=datetime.now(_EST).isoformat(),
                )
            )

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Batch simulation for %s: %d windows → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )

    return results


def simulate_day(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
) -> ORBSimResult:
    """Simulate a single day's ORB trade.

    Unlike ``simulate_batch`` which slides windows, this function treats the
    entire ``bars_1m`` input as one day and produces exactly one ORBSimResult.
    Use this when you've already sliced your data to one trading session.

    Args:
        bars_1m: One day of 1-minute OHLCV bars.
        symbol: Instrument symbol.
        config: BracketConfig.
        bars_daily: Daily bars for NR7 (optional).

    Returns:
        A single ORBSimResult.
    """
    return simulate_orb_outcome(
        bars_1m=bars_1m,
        symbol=symbol,
        config=config,
        bars_daily=bars_daily,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarise_results(results: list[ORBSimResult]) -> dict[str, Any]:
    """Compute aggregate statistics from a batch of simulation results.

    Useful for evaluating filter effectiveness or bracket parameter sweeps.

    Returns:
        Dict with trade count, win rate, avg R, profit factor, etc.
    """
    trades = [r for r in results if r.is_trade]
    winners = [r for r in trades if r.is_winner]
    losers = [r for r in trades if not r.is_winner]

    if not trades:
        return {
            "total_windows": len(results),
            "total_trades": 0,
            "no_trade_count": len(results),
            "win_rate": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "avg_hold_bars": 0.0,
            "avg_quality": 0.0,
        }

    total_win_r = sum(r.pnl_r for r in winners)
    total_loss_r = abs(sum(r.pnl_r for r in losers))
    profit_factor = total_win_r / total_loss_r if total_loss_r > 0 else float("inf")

    label_counts: dict[str, int] = {}
    for r in results:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1

    return {
        "total_windows": len(results),
        "total_trades": len(trades),
        "no_trade_count": len(results) - len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(trades) * 100, 1),
        "avg_r": round(sum(r.pnl_r for r in trades) / len(trades), 3),
        "total_r": round(sum(r.pnl_r for r in trades), 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_bars": round(sum(r.hold_bars for r in trades) / len(trades), 1),
        "avg_quality": round(sum(r.quality_pct for r in trades) / len(trades), 1),
        "label_distribution": label_counts,
        "long_trades": sum(1 for r in trades if r.direction == "LONG"),
        "short_trades": sum(1 for r in trades if r.direction == "SHORT"),
        "nr7_trades": sum(1 for r in trades if r.nr7),
        "nr7_win_rate": round(
            sum(1 for r in trades if r.nr7 and r.is_winner) / max(1, sum(1 for r in trades if r.nr7)) * 100,
            1,
        ),
    }


def results_to_dataframe(results: list[ORBSimResult]) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame for analysis."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)
