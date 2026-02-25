"""
Cumulative Volume Delta (CVD) approximation from OHLCV data.

Since Level 2 / tick data is not available via yfinance, we use the
standard OHLCV heuristic to approximate buy/sell volume:

    buy_volume  = volume × (close − low) / (high − low)
    sell_volume = volume − buy_volume
    delta       = buy_volume − sell_volume

Accuracy is ±15–25% versus true bid/ask delta but adds useful confluence
for confirming trends, detecting divergences, and identifying absorption.

Features:
  - CVD calculation with intraday anchoring (reset at market open)
  - CVD divergence detection (price vs CVD direction mismatch)
  - Volume absorption candle identification (high volume, small body near S/R)
  - Rolling CVD slope for momentum confirmation
  - Dashboard-ready indicator functions compatible with backtesting.py

Per the notes.md blueprint:
  - Track CVD divergences (price makes lower low, CVD makes higher low → bullish)
  - Volume absorption candles (high volume, small body near support → buyers absorbing)
  - Reset CVD at market open for intraday anchoring

Usage:
    from cvd import compute_cvd, detect_cvd_divergences, detect_absorption_candles

    df = get_data("GC=F", "5m", "5d")
    cvd_df = compute_cvd(df)
    divergences = detect_cvd_divergences(cvd_df, lookback=20)
    absorptions = detect_absorption_candles(cvd_df)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cvd")


# ---------------------------------------------------------------------------
# Core CVD calculation
# ---------------------------------------------------------------------------


def _estimate_buy_volume(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Estimate buy volume using the OHLCV heuristic.

    buy_volume = volume × (close − low) / (high − low)

    When high == low (doji / zero-range bar), split volume 50/50.
    """
    price_range = high - low
    # Avoid division by zero on zero-range bars
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (close - low) / safe_range
    buy_ratio = buy_ratio.fillna(0.5)  # 50/50 split for zero-range bars
    buy_ratio = buy_ratio.clip(0.0, 1.0)  # clamp to valid range
    return volume * buy_ratio


def compute_cvd(
    df: pd.DataFrame,
    anchor_daily: bool = True,
) -> pd.DataFrame:
    """Compute Cumulative Volume Delta from OHLCV data.

    Args:
        df: DataFrame with columns High, Low, Close, Volume (and index as datetime).
        anchor_daily: If True, reset CVD accumulation at each new trading day
                      (intraday anchoring). If False, compute continuous CVD.

    Returns:
        DataFrame with additional columns:
          - buy_volume: estimated buy volume per bar
          - sell_volume: estimated sell volume per bar
          - delta: per-bar volume delta (buy - sell)
          - cvd: cumulative volume delta
          - cvd_ema: smoothed CVD (EMA-10 of CVD for signal clarity)
          - cvd_slope: rolling slope of CVD (positive = buying pressure)
    """
    result = df.copy()

    high = pd.Series(result["High"].astype(float))
    low = pd.Series(result["Low"].astype(float))
    close = pd.Series(result["Close"].astype(float))
    volume = pd.Series(result["Volume"].astype(float))

    # Estimate buy/sell volume
    buy_vol = _estimate_buy_volume(high, low, close, volume)
    sell_vol = volume - buy_vol
    delta = buy_vol - sell_vol

    result["buy_volume"] = buy_vol
    result["sell_volume"] = sell_vol
    result["delta"] = delta

    # Cumulative volume delta with optional daily anchoring
    if anchor_daily and hasattr(result.index, "date"):
        # Group by trading date and cumsum within each day
        try:
            dates = result.index.to_series().dt.date
            cvd = delta.groupby(dates).cumsum()
        except Exception:
            cvd = delta.cumsum()
    else:
        cvd = delta.cumsum()

    result["cvd"] = cvd

    # Smoothed CVD (EMA-10) for cleaner signals
    result["cvd_ema"] = cvd.ewm(span=10, adjust=False).mean()

    # CVD slope: rolling 5-bar rate of change (normalized)
    cvd_diff = cvd.diff(5)
    # Normalize slope by rolling std to make it comparable across instruments
    cvd_std = cvd.rolling(20).std()
    result["cvd_slope"] = cvd_diff / (cvd_std + 1e-10)

    return result


# ---------------------------------------------------------------------------
# CVD Divergence detection
# ---------------------------------------------------------------------------


def _find_swing_points(
    series: pd.Series,
    lookback: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Find swing highs and swing lows in a series.

    A swing high is a point higher than `lookback` bars on each side.
    A swing low is a point lower than `lookback` bars on each side.

    Returns:
        (swing_highs, swing_lows) — each is a list of (index_position, value).
    """
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    highs = []
    lows = []

    for i in range(lookback, n - lookback):
        if np.isnan(values[i]):
            continue
        window = values[i - lookback : i + lookback + 1]
        if np.any(np.isnan(window)):
            continue
        if values[i] == float(np.nanmax(window)):
            highs.append((i, float(values[i])))
        if values[i] == float(np.nanmin(window)):
            lows.append((i, float(values[i])))

    return highs, lows


def detect_cvd_divergences(
    df: pd.DataFrame,
    lookback: int = 20,
    swing_period: int = 5,
    min_bars_between: int = 5,
) -> list[dict[str, Any]]:
    """Detect divergences between price and CVD.

    Divergence types:
      - Bullish divergence: Price makes lower low, CVD makes higher low
        → Hidden buying pressure, potential reversal up
      - Bearish divergence: Price makes higher high, CVD makes lower high
        → Hidden selling pressure, potential reversal down

    Args:
        df: DataFrame with 'Close' and 'cvd' columns (output of compute_cvd).
        lookback: Number of recent bars to scan for divergences.
        swing_period: Bars on each side to qualify as a swing point.
        min_bars_between: Minimum bars between two swing points for a valid divergence.

    Returns:
        List of dicts with keys:
          - type: "bullish" or "bearish"
          - bar_index: index position of the second (more recent) swing point
          - price_1, price_2: price at first and second swing points
          - cvd_1, cvd_2: CVD at first and second swing points
          - strength: rough measure of divergence magnitude (0-100)
    """
    if "cvd" not in df.columns or len(df) < lookback + swing_period * 2:
        return []

    # Only look at the recent portion
    recent = df.iloc[-lookback - swing_period * 2 :]
    close = recent["Close"].astype(float)
    cvd = recent["cvd"].astype(float)

    price_highs, price_lows = _find_swing_points(close, swing_period)
    cvd_highs, cvd_lows = _find_swing_points(cvd, swing_period)

    divergences: list[dict[str, Any]] = []

    # Bullish divergence: price lower low + CVD higher low
    for i in range(len(price_lows) - 1):
        p1_idx, p1_val = price_lows[i]
        p2_idx, p2_val = price_lows[i + 1]
        if p2_idx - p1_idx < min_bars_between:
            continue
        if p2_val >= p1_val:
            continue  # price must make lower low

        # Find corresponding CVD lows near these price lows
        cvd_at_p1 = cvd.iloc[p1_idx] if p1_idx < len(cvd) else np.nan
        cvd_at_p2 = cvd.iloc[p2_idx] if p2_idx < len(cvd) else np.nan

        if np.isnan(cvd_at_p1) or np.isnan(cvd_at_p2):
            continue
        if cvd_at_p2 <= cvd_at_p1:
            continue  # CVD must make higher low

        # Strength: how divergent are price and CVD moves?
        price_drop = abs(p2_val - p1_val) / (abs(p1_val) + 1e-10) * 100
        cvd_rise = abs(cvd_at_p2 - cvd_at_p1) / (abs(cvd_at_p1) + 1e-10) * 100
        strength = min((price_drop + cvd_rise) / 2, 100)

        # Map back to original DataFrame index
        orig_idx = recent.index[p2_idx] if p2_idx < len(recent) else None

        divergences.append(
            {
                "type": "bullish",
                "bar_index": p2_idx,
                "datetime": orig_idx,
                "price_1": round(p1_val, 4),
                "price_2": round(p2_val, 4),
                "cvd_1": round(cvd_at_p1, 2),
                "cvd_2": round(cvd_at_p2, 2),
                "strength": round(strength, 1),
            }
        )

    # Bearish divergence: price higher high + CVD lower high
    for i in range(len(price_highs) - 1):
        p1_idx, p1_val = price_highs[i]
        p2_idx, p2_val = price_highs[i + 1]
        if p2_idx - p1_idx < min_bars_between:
            continue
        if p2_val <= p1_val:
            continue  # price must make higher high

        cvd_at_p1 = cvd.iloc[p1_idx] if p1_idx < len(cvd) else np.nan
        cvd_at_p2 = cvd.iloc[p2_idx] if p2_idx < len(cvd) else np.nan

        if np.isnan(cvd_at_p1) or np.isnan(cvd_at_p2):
            continue
        if cvd_at_p2 >= cvd_at_p1:
            continue  # CVD must make lower high

        price_rise = abs(p2_val - p1_val) / (abs(p1_val) + 1e-10) * 100
        cvd_drop = abs(cvd_at_p1 - cvd_at_p2) / (abs(cvd_at_p1) + 1e-10) * 100
        strength = min((price_rise + cvd_drop) / 2, 100)

        orig_idx = recent.index[p2_idx] if p2_idx < len(recent) else None

        divergences.append(
            {
                "type": "bearish",
                "bar_index": p2_idx,
                "datetime": orig_idx,
                "price_1": round(p1_val, 4),
                "price_2": round(p2_val, 4),
                "cvd_1": round(cvd_at_p1, 2),
                "cvd_2": round(cvd_at_p2, 2),
                "strength": round(strength, 1),
            }
        )

    return divergences


# ---------------------------------------------------------------------------
# Absorption candle detection
# ---------------------------------------------------------------------------


def detect_absorption_candles(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.3,
    volume_mult: float = 1.5,
    volume_lookback: int = 20,
) -> pd.Series:
    """Detect volume absorption candles.

    An absorption candle is characterized by:
      1. High volume (> volume_mult × 20-bar average)
      2. Small body relative to range (body / range < body_ratio_threshold)
      3. This indicates one side absorbing the other's aggression

    The signal is:
      +1 = bullish absorption (close near high, buyers absorbing selling)
      -1 = bearish absorption (close near low, sellers absorbing buying)
       0 = no absorption

    Args:
        df: DataFrame with OHLCV data.
        body_ratio_threshold: Maximum body/range ratio to qualify (default 0.3).
        volume_mult: Minimum volume multiplier vs rolling average (default 1.5).
        volume_lookback: Rolling window for volume average (default 20).

    Returns:
        Series of absorption signals (+1, -1, 0).
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_price = df["Open"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    candle_range = high - low
    body = (close - open_price).abs()
    body_ratio = body / (candle_range + 1e-10)

    avg_volume = volume.rolling(volume_lookback).mean()
    high_volume = volume > (avg_volume * volume_mult)

    small_body = body_ratio < body_ratio_threshold
    is_absorption = high_volume & small_body & (candle_range > 0)

    # Determine direction: close position within the bar's range
    close_position = (close - low) / (candle_range + 1e-10)

    signal = pd.Series(0, index=df.index, dtype=int)
    # Bullish absorption: close in upper half (buyers absorbed selling pressure)
    signal.loc[is_absorption & (close_position > 0.5)] = 1
    # Bearish absorption: close in lower half (sellers absorbed buying pressure)
    signal.loc[is_absorption & (close_position <= 0.5)] = -1

    return signal


# ---------------------------------------------------------------------------
# CVD trend / momentum helpers
# ---------------------------------------------------------------------------


def cvd_confirms_trend(
    df: pd.DataFrame,
    direction: str = "long",
    slope_threshold: float = 0.5,
) -> bool:
    """Check if CVD slope confirms a directional bias.

    Args:
        df: DataFrame with 'cvd_slope' column (output of compute_cvd).
        direction: "long" or "short".
        slope_threshold: Minimum absolute slope value for confirmation.

    Returns:
        True if CVD slope confirms the given direction.
    """
    if "cvd_slope" not in df.columns or df.empty:
        return False

    current_slope = df["cvd_slope"].iloc[-1]
    if np.isnan(current_slope):
        return False

    if direction == "long":
        return float(current_slope) > slope_threshold
    elif direction == "short":
        return float(current_slope) < -slope_threshold
    return False


def cvd_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a dashboard-ready CVD summary for an instrument.

    Args:
        df: DataFrame with CVD columns (output of compute_cvd).

    Returns:
        Dict with CVD metrics for display.
    """
    if "cvd" not in df.columns or df.empty:
        return {
            "cvd_current": 0.0,
            "delta_current": 0.0,
            "cvd_slope": 0.0,
            "bias": "neutral",
            "bias_emoji": "⚪",
            "absorption": 0,
            "divergences": [],
        }

    cvd_val = float(df["cvd"].iloc[-1]) if not np.isnan(df["cvd"].iloc[-1]) else 0.0
    delta_val = (
        float(df["delta"].iloc[-1]) if not np.isnan(df["delta"].iloc[-1]) else 0.0
    )
    slope_val = (
        float(df["cvd_slope"].iloc[-1])
        if not np.isnan(df["cvd_slope"].iloc[-1])
        else 0.0
    )

    # Determine bias from slope
    if slope_val > 0.5:
        bias = "bullish"
        emoji = "🟢"
    elif slope_val < -0.5:
        bias = "bearish"
        emoji = "🔴"
    else:
        bias = "neutral"
        emoji = "🟡"

    # Check for absorption at latest bar
    absorption_signals = detect_absorption_candles(df)
    latest_absorption = (
        int(absorption_signals.iloc[-1]) if len(absorption_signals) > 0 else 0
    )

    # Check for recent divergences
    divergences = detect_cvd_divergences(df, lookback=30)

    return {
        "cvd_current": round(cvd_val, 2),
        "delta_current": round(delta_val, 2),
        "cvd_slope": round(slope_val, 3),
        "bias": bias,
        "bias_emoji": emoji,
        "absorption": latest_absorption,
        "divergences": divergences,
    }


# ---------------------------------------------------------------------------
# Indicator functions for backtesting.py compatibility
# ---------------------------------------------------------------------------


def _cvd_indicator(high, low, close, volume):
    """CVD indicator function for use with backtesting.py's self.I().

    Returns CVD as a numpy array.
    """
    h = pd.Series(high, dtype=float)
    lo = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)

    price_range = h - lo
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (c - lo) / safe_range
    buy_ratio = buy_ratio.fillna(0.5).clip(0.0, 1.0)

    buy_vol = v * buy_ratio
    sell_vol = v - buy_vol
    delta = buy_vol - sell_vol

    return delta.cumsum().values


def _delta_indicator(high, low, close, volume):
    """Per-bar volume delta indicator for backtesting.py's self.I().

    Returns delta as a numpy array.
    """
    h = pd.Series(high, dtype=float)
    lo = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)

    price_range = h - lo
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (c - lo) / safe_range
    buy_ratio = buy_ratio.fillna(0.5).clip(0.0, 1.0)

    buy_vol = v * buy_ratio
    sell_vol = v - buy_vol

    return (buy_vol - sell_vol).values


def _cvd_ema_indicator(high, low, close, volume, span: int = 10):
    """Smoothed CVD (EMA) indicator for backtesting.py's self.I().

    Returns EMA of CVD as a numpy array.
    """
    cvd = pd.Series(_cvd_indicator(high, low, close, volume))
    return cvd.ewm(span=span, adjust=False).mean().values


# ---------------------------------------------------------------------------
# DataFrame-to-display helpers
# ---------------------------------------------------------------------------


def divergences_to_dataframe(divergences: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert divergence list to a display-ready DataFrame.

    Returns a DataFrame with columns: Type, Datetime, Price Move,
    CVD Move, Strength.
    """
    if not divergences:
        return pd.DataFrame(
            columns=pd.Index(["Type", "Datetime", "Price Move", "CVD Move", "Strength"])
        )

    rows = []
    for d in divergences:
        price_move = f"{d['price_1']:.2f} → {d['price_2']:.2f}"
        cvd_move = f"{d['cvd_1']:.0f} → {d['cvd_2']:.0f}"
        rows.append(
            {
                "Type": d["type"].capitalize(),
                "Datetime": d.get("datetime", ""),
                "Price Move": price_move,
                "CVD Move": cvd_move,
                "Strength": f"{d['strength']:.0f}%",
            }
        )

    return pd.DataFrame(rows)
