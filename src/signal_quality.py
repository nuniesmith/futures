"""
FKS Signal Quality Score — ported from fks.pine and fks_info.pine.

This module computes the multi-factor signal quality score exactly as
implemented in your Pine Script indicators. The score combines 5 weighted
factors to produce a 0–1 quality metric that gates trade entries:

  Factor 1 (37.5%): Volatility sweet-spot — percentile between 0.2 and 0.7
  Factor 2 (25.0%): Normalized velocity — momentum aligned with trend
  Factor 3 (12.5%): Price acceleration / trend speed factor
  Factor 4 (12.5%): Candle pattern confirmation (bullish/bearish)
  Factor 5 (12.5%): HTF bias alignment (S/R or higher-timeframe)

The score adapts based on detected market phase:
  - UPTREND: rewards positive velocity, bullish candles, long bias
  - DOWNTREND: rewards negative velocity, bearish candles, short bias
  - RANGING: rewards low volatility, near-zero velocity, any pattern

Design decisions:
  - Stateless: operates on DataFrame + pre-computed analysis dicts
  - Reuses existing wave.py and volatility.py outputs (no redundant compute)
  - Candle pattern detection ported directly from fks.pine's
    f_detect_bullish_candle_pattern / f_detect_bearish_candle_pattern
  - Can be called on every 1m bar (via WebSocket) or per 5m refresh
  - Result includes individual factor scores for debugging/dashboard

Usage:
    from signal_quality import compute_signal_quality

    result = compute_signal_quality(
        df,
        wave_result=wave_result,   # from wave.calculate_wave_analysis()
        vol_result=vol_result,     # from volatility.kmeans_volatility_clusters()
    )
    # result = {
    #     "score": 0.72,
    #     "quality_pct": 72.0,
    #     "high_quality": True,
    #     "factors": {
    #         "vol_sweet_spot": 1.5,
    #         "velocity_aligned": 1.0,
    #         "trend_speed_factor": 0.5,
    #         "candle_confirmation": 0.5,
    #         "htf_bias": 0.5,
    #     },
    #     "market_context": "UPTREND",
    #     "trend_direction": "BULLISH",
    # }
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

logger = logging.getLogger("signal_quality")


# ---------------------------------------------------------------------------
# Candle pattern detection (exact port from fks.pine)
# ---------------------------------------------------------------------------


def _detect_bullish_candle(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    idx: int = -1,
) -> bool:
    """Detect bullish candlestick patterns at the given bar index.

    Port of fks.pine f_detect_bullish_candle_pattern():
      - Hammer: small body, long lower wick (>2× body), short upper wick
      - Bullish engulfing: green bar engulfs prior red bar
      - Pin bar: lower wick > 60% of range, small body, small upper wick
    """
    n = len(close)
    if n < 2:
        return False

    i = idx if idx >= 0 else n + idx
    if i < 1 or i >= n:
        return False

    body_size = abs(close[i] - open_[i])
    candle_range = high[i] - low[i]
    if candle_range == 0:
        candle_range = 0.00001

    lower_wick = min(open_[i], close[i]) - low[i]
    upper_wick = high[i] - max(open_[i], close[i])

    # Hammer pattern
    hammer = (
        body_size < candle_range * 0.3
        and lower_wick > body_size * 2
        and upper_wick < body_size
    )

    # Bullish engulfing
    bullish_engulfing = (
        close[i] > open_[i]
        and close[i - 1] < open_[i - 1]
        and close[i] > open_[i - 1]
        and open_[i] < close[i - 1]
    )

    # Pin bar (long lower wick rejection)
    pin_bar = (
        lower_wick > candle_range * 0.6
        and body_size < candle_range * 0.3
        and upper_wick < candle_range * 0.3
    )

    return hammer or bullish_engulfing or pin_bar


def _detect_bearish_candle(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    idx: int = -1,
) -> bool:
    """Detect bearish candlestick patterns at the given bar index.

    Port of fks.pine f_detect_bearish_candle_pattern():
      - Shooting star: small body, long upper wick (>2× body), short lower wick
      - Bearish engulfing: red bar engulfs prior green bar
      - Pin bar: upper wick > 60% of range, small body, small lower wick
    """
    n = len(close)
    if n < 2:
        return False

    i = idx if idx >= 0 else n + idx
    if i < 1 or i >= n:
        return False

    body_size = abs(close[i] - open_[i])
    candle_range = high[i] - low[i]
    if candle_range == 0:
        candle_range = 0.00001

    lower_wick = min(open_[i], close[i]) - low[i]
    upper_wick = high[i] - max(open_[i], close[i])

    # Shooting star
    shooting_star = (
        body_size < candle_range * 0.3
        and upper_wick > body_size * 2
        and lower_wick < body_size
    )

    # Bearish engulfing
    bearish_engulfing = (
        close[i] < open_[i]
        and close[i - 1] > open_[i - 1]
        and close[i] < open_[i - 1]
        and open_[i] > close[i - 1]
    )

    # Pin bar (long upper wick rejection)
    pin_bar = (
        upper_wick > candle_range * 0.6
        and body_size < candle_range * 0.3
        and lower_wick < candle_range * 0.3
    )

    return shooting_star or bearish_engulfing or pin_bar


# ---------------------------------------------------------------------------
# Normalized velocity (ported from fks.pine)
# ---------------------------------------------------------------------------


def _compute_normalized_velocity(
    close: ArrayLike,
    momentum_lookback: int = 3,
    stdev_lookback: int = 100,
) -> float:
    """Compute normalized price velocity.

    Port of fks.pine:
      price_velocity = change(close, momentum_lookback) / close[momentum_lookback]
      stdev_velocity = stdev(price_velocity, 100)
      normalized_velocity = price_velocity / stdev_velocity
    """
    n = len(close)
    if n < momentum_lookback + 2:
        return 0.0

    # Compute price velocity series
    velocities = np.zeros(n)
    for i in range(momentum_lookback, n):
        prev = close[i - momentum_lookback]
        if prev != 0:
            velocities[i] = (close[i] - prev) / prev

    # Standard deviation over lookback window
    window_start = max(0, n - stdev_lookback)
    vel_window = velocities[window_start:n]
    stdev = float(np.std(vel_window))

    if stdev == 0:
        return 0.0

    return float(velocities[-1] / stdev)


# ---------------------------------------------------------------------------
# Price acceleration (ported from fks.pine)
# ---------------------------------------------------------------------------


def _compute_price_acceleration(
    close: ArrayLike,
    momentum_lookback: int = 3,
    stdev_lookback: int = 100,
) -> float:
    """Compute normalized price acceleration (change of velocity).

    Port of fks.pine:
      price_velocity_raw = change(close, momentum_lookback) / close[momentum_lookback]
      price_acceleration_raw = change(price_velocity_raw, momentum_lookback)
      Normalize by stdev of velocity
    """
    n = len(close)
    if n < momentum_lookback * 2 + 2:
        return 0.0

    # Velocity series
    velocities = np.zeros(n)
    for i in range(momentum_lookback, n):
        prev = close[i - momentum_lookback]
        if prev != 0:
            velocities[i] = (close[i] - prev) / prev

    # Acceleration: change of velocity
    accelerations = np.zeros(n)
    for i in range(momentum_lookback * 2, n):
        accelerations[i] = velocities[i] - velocities[i - momentum_lookback]

    # Stdev of velocity for normalization
    window_start = max(0, n - stdev_lookback)
    vel_window = velocities[window_start:n]
    stdev = float(np.std(vel_window))

    if stdev == 0:
        return 0.0

    return float(accelerations[-1] / stdev)


# ---------------------------------------------------------------------------
# Trend speed factor (ported from fks.pine)
# ---------------------------------------------------------------------------


def _compute_trend_speed_factor(
    current_ratio: float,
    min_wave_ratio: float = 1.5,
) -> float:
    """Compute trend speed quality factor based on wave ratio.

    Port of fks.pine:
      if abs(current_ratio) > min_wave_ratio → 1.0
      if abs(current_ratio) > min_wave_ratio * 0.7 → 0.7
      if abs(current_ratio) > min_wave_ratio * 0.5 → 0.4
      else → 0.0
    """
    abs_ratio = abs(current_ratio)
    if abs_ratio > min_wave_ratio:
        return 1.0
    elif abs_ratio > min_wave_ratio * 0.7:
        return 0.7
    elif abs_ratio > min_wave_ratio * 0.5:
        return 0.4
    return 0.0


# ---------------------------------------------------------------------------
# Market trend detection (ported from fks.pine / fks_info.pine)
# ---------------------------------------------------------------------------


def _determine_trend_context(
    market_phase: str,
    ao_value: float = 0.0,
    rsi_value: float = 50.0,
) -> str:
    """Determine if we're in an uptrend, downtrend, or ranging context.

    Port of fks.pine:
      in_uptrend = market_phase == "UPTREND" or
                   (market_phase == "ACCUMULATION" and AO > 0 and RSI > 50)
      in_downtrend = market_phase == "DOWNTREND" or
                     (market_phase == "DISTRIBUTION" and AO < 0 and RSI < 50)
    """
    if market_phase == "UPTREND":
        return "UPTREND"
    if market_phase == "DOWNTREND":
        return "DOWNTREND"
    if market_phase == "ACCUMULATION" and ao_value > 0 and rsi_value > 50:
        return "UPTREND"
    if market_phase == "DISTRIBUTION" and ao_value < 0 and rsi_value < 50:
        return "DOWNTREND"
    return "RANGING"


# ---------------------------------------------------------------------------
# RSI computation
# ---------------------------------------------------------------------------


def _compute_rsi(close: ArrayLike, period: int = 14) -> float:
    """Compute RSI(period) for the latest bar. Uses Wilder's smoothing."""
    if len(close) < period + 1:
        return 50.0

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing for remaining bars
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------


def _compute_ao(
    high: ArrayLike, low: ArrayLike, fast: int = 5, slow: int = 34
) -> float:
    """Compute Awesome Oscillator: SMA(hl2, fast) - SMA(hl2, slow)."""
    if len(high) < slow:
        return 0.0

    hl2 = (high + low) / 2.0
    fast_sma = np.mean(hl2[-fast:])
    slow_sma = np.mean(hl2[-slow:])
    return float(fast_sma - slow_sma)


# ---------------------------------------------------------------------------
# Public API: Signal quality score
# ---------------------------------------------------------------------------


def compute_signal_quality(
    df: pd.DataFrame,
    wave_result: dict[str, Any] | None = None,
    vol_result: dict[str, Any] | None = None,
    quality_threshold: float = 0.6,
    min_wave_ratio: float = 1.5,
    momentum_lookback: int = 3,
) -> dict[str, Any]:
    """Compute the FKS multi-factor signal quality score.

    This is the exact port of the signal_quality_score calculation from
    fks.pine (lines ~960–1010) and fks_info.pine (lines ~610–650).

    The 5-factor weighted score adapts to market context:

    UPTREND context:
      (vol in sweet spot ? 1.5 : 0.5)        → 37.5% weight
      (velocity > 0 ? 1.0 : 0.0)             → 25.0% weight
      (acceleration > 0 ? 0.5 : 0.0)         → 12.5% weight  [fks.pine]
      OR (trend_speed_factor * 0.5)           → 12.5% weight  [fks.pine alt]
      (bullish candle ? 0.5 : 0.0)           → 12.5% weight
      (long bias ? 0.5 : 0.0)               → 12.5% weight

    DOWNTREND: mirrors with negative velocity, bearish candle, short bias
    RANGING: low vol, near-zero velocity, near-zero acceleration

    All divided by 4.0 to normalize to [0, 1].

    Args:
        df: OHLCV DataFrame with Open, High, Low, Close, Volume
        wave_result: Pre-computed result from calculate_wave_analysis()
        vol_result: Pre-computed result from kmeans_volatility_clusters()
        quality_threshold: Score threshold for "high quality" flag (default 0.6)
        min_wave_ratio: Minimum wave ratio for trend speed factor (default 1.5)
        momentum_lookback: Bars for velocity/acceleration calc (default 3)

    Returns:
        Dict with score, quality_pct, high_quality flag, factor breakdown,
        and market context info.
    """
    default_result: dict[str, Any] = {
        "score": 0.0,
        "quality_pct": 0.0,
        "high_quality": False,
        "factors": {
            "vol_sweet_spot": 0.0,
            "velocity_aligned": 0.0,
            "acceleration_aligned": 0.0,
            "trend_speed_factor": 0.0,
            "candle_confirmation": 0.0,
            "htf_bias": 0.0,
        },
        "market_context": "RANGING",
        "trend_direction": "NEUTRAL",
        "rsi": 50.0,
        "ao": 0.0,
        "normalized_velocity": 0.0,
        "price_acceleration": 0.0,
    }

    if df is None or len(df) == 0 or len(df) < 20:
        return default_result

    try:
        close = df["Close"].astype(float).values
        high = df["High"].astype(float).values
        low = df["Low"].astype(float).values
        open_ = df["Open"].astype(float).values
    except (KeyError, ValueError) as exc:
        logger.warning("Signal quality failed — missing OHLC columns: %s", exc)
        return default_result

    # --- Gather inputs from pre-computed results or compute fresh ---

    # Volatility percentile
    if vol_result:
        vol_percentile = vol_result.get("percentile", 0.5)
    else:
        vol_percentile = 0.5

    # Wave analysis
    if wave_result:
        current_ratio = wave_result.get("current_ratio", 0.0)
        market_phase = wave_result.get("market_phase", "ACCUMULATION")
        trend_speed = wave_result.get("trend_speed", 0.0)
        bias = wave_result.get("bias", "NEUTRAL")
    else:
        current_ratio = 0.0
        market_phase = "ACCUMULATION"
        trend_speed = 0.0
        bias = "NEUTRAL"

    # Compute RSI and AO for trend context
    rsi_value = _compute_rsi(close)
    ao_value = _compute_ao(high, low)

    # Determine market context
    market_context = _determine_trend_context(market_phase, ao_value, rsi_value)

    # Compute normalized velocity and acceleration
    normalized_velocity = _compute_normalized_velocity(close, momentum_lookback)
    price_acceleration = _compute_price_acceleration(close, momentum_lookback)

    # Trend speed factor from wave ratio
    tsf = _compute_trend_speed_factor(current_ratio, min_wave_ratio)

    # Candle pattern detection on last confirmed bar
    bullish_candle = _detect_bullish_candle(open_, high, low, close, idx=-1)
    bearish_candle = _detect_bearish_candle(open_, high, low, close, idx=-1)

    # HTF bias factor — use wave bias as proxy for S/R bias
    # (In Pine this uses long_bias_1m/5m from S/R levels; here we approximate
    #  with the wave dominance bias which captures the same directional info)
    htf_bias_factor = 0.0
    if market_context == "UPTREND" and bias == "BULLISH":
        htf_bias_factor = 1.0
    elif market_context == "UPTREND" and bias == "NEUTRAL":
        htf_bias_factor = 0.5
    elif market_context == "DOWNTREND" and bias == "BEARISH":
        htf_bias_factor = 1.0
    elif market_context == "DOWNTREND" and bias == "NEUTRAL":
        htf_bias_factor = 0.5
    elif market_context == "RANGING":
        # In ranging, any clear bias counts
        htf_bias_factor = 1.0 if bias != "NEUTRAL" else 0.5

    # --- Multi-factor signal quality calculation ---
    # Exact port of fks.pine lines ~960-1010

    score = 0.0
    factors: dict[str, float] = {}

    if market_context == "UPTREND":
        # Factor 1 (37.5%): Vol in sweet spot
        vol_factor = 1.5 if (0.2 < vol_percentile < 0.7) else 0.5
        # Factor 2 (25.0%): Positive velocity
        vel_factor = 1.0 if normalized_velocity > 0 else 0.0
        # Factor 3 (12.5%): Positive acceleration OR trend speed factor
        accel_factor = price_acceleration > 0
        tsf_component = max(0.5 if accel_factor else 0.0, tsf * 0.5)
        # Factor 4 (12.5%): Bullish candle confirmation
        candle_factor = 0.5 if bullish_candle else 0.0
        # Factor 5 (12.5%): HTF long bias
        htf_factor = htf_bias_factor * 0.5

        score = (
            vol_factor + vel_factor + tsf_component + candle_factor + htf_factor
        ) / 4.0

        factors = {
            "vol_sweet_spot": vol_factor,
            "velocity_aligned": vel_factor,
            "acceleration_aligned": 0.5 if accel_factor else 0.0,
            "trend_speed_factor": tsf * 0.5,
            "candle_confirmation": candle_factor,
            "htf_bias": htf_factor,
        }

    elif market_context == "DOWNTREND":
        # Factor 1 (37.5%): Vol in sweet spot
        vol_factor = 1.5 if (0.2 < vol_percentile < 0.7) else 0.5
        # Factor 2 (25.0%): Negative velocity
        vel_factor = 1.0 if normalized_velocity < 0 else 0.0
        # Factor 3 (12.5%): Negative acceleration OR trend speed factor
        accel_factor = price_acceleration < 0
        tsf_component = max(0.5 if accel_factor else 0.0, tsf * 0.5)
        # Factor 4 (12.5%): Bearish candle confirmation
        candle_factor = 0.5 if bearish_candle else 0.0
        # Factor 5 (12.5%): HTF short bias
        htf_factor = htf_bias_factor * 0.5

        score = (
            vol_factor + vel_factor + tsf_component + candle_factor + htf_factor
        ) / 4.0

        factors = {
            "vol_sweet_spot": vol_factor,
            "velocity_aligned": vel_factor,
            "acceleration_aligned": 0.5 if accel_factor else 0.0,
            "trend_speed_factor": tsf * 0.5,
            "candle_confirmation": candle_factor,
            "htf_bias": htf_factor,
        }

    else:
        # RANGING context — rewards low volatility and near-zero momentum
        # Factor 1 (37.5%): Very low volatility preferred
        vol_factor = 1.5 if vol_percentile < 0.3 else 0.5
        # Factor 2 (25.0%): Near-zero velocity
        vel_factor = 1.0 if abs(normalized_velocity) < 0.5 else 0.0
        # Factor 3 (12.5%): Near-zero acceleration
        accel_factor = abs(price_acceleration) < 0.2
        tsf_component = max(0.5 if accel_factor else 0.0, tsf * 0.5)
        # Factor 4 (12.5%): Any candle pattern
        candle_factor = 0.5 if (bullish_candle or bearish_candle) else 0.0
        # Factor 5 (12.5%): Any directional bias
        htf_factor = htf_bias_factor * 0.5

        score = (
            vol_factor + vel_factor + tsf_component + candle_factor + htf_factor
        ) / 4.0

        factors = {
            "vol_sweet_spot": vol_factor,
            "velocity_aligned": vel_factor,
            "acceleration_aligned": 0.5 if accel_factor else 0.0,
            "trend_speed_factor": tsf * 0.5,
            "candle_confirmation": candle_factor,
            "htf_bias": htf_factor,
        }

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # Determine trend direction label
    if trend_speed > 0:
        trend_direction = "BULLISH"
    elif trend_speed < 0:
        trend_direction = "BEARISH"
    else:
        trend_direction = "NEUTRAL"

    return {
        "score": round(score, 3),
        "quality_pct": round(score * 100, 1),
        "high_quality": score >= quality_threshold,
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "market_context": market_context,
        "trend_direction": trend_direction,
        "rsi": round(rsi_value, 1),
        "ao": round(ao_value, 4),
        "normalized_velocity": round(normalized_velocity, 4),
        "price_acceleration": round(price_acceleration, 4),
    }


# ---------------------------------------------------------------------------
# Premium setup detection (ported from fks.pine)
# ---------------------------------------------------------------------------


def is_premium_setup(
    quality_result: dict[str, Any],
    vol_result: dict[str, Any] | None = None,
    wave_result: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Check if current conditions qualify as a premium setup.

    Port of fks.pine:
      premium_buy_setup = final_buy_signal and signal_quality_score > 0.8 and
                          volatility_percentile > 0.2 and volatility_percentile < 0.8
                          and long_bias_5m
      premium_sell_setup = final_sell_signal and signal_quality_score > 0.8 and
                          volatility_percentile > 0.2 and volatility_percentile < 0.8
                          and short_bias_5m

    Returns (is_premium, direction) where direction is "LONG", "SHORT", or "NONE".
    """
    score = quality_result.get("score", 0.0)
    context = quality_result.get("market_context", "RANGING")

    if score <= 0.8:
        return False, "NONE"

    # Volatility check
    vol_pct = vol_result.get("percentile", 0.5) if vol_result else 0.5
    if vol_pct <= 0.2 or vol_pct >= 0.8:
        return False, "NONE"

    # Bias alignment
    bias = wave_result.get("bias", "NEUTRAL") if wave_result else "NEUTRAL"

    if context == "UPTREND" and bias == "BULLISH":
        return True, "LONG"
    elif context == "DOWNTREND" and bias == "BEARISH":
        return True, "SHORT"

    return False, "NONE"


# ---------------------------------------------------------------------------
# Summary text for Grok / dashboard
# ---------------------------------------------------------------------------


def signal_quality_summary(result: dict[str, Any]) -> str:
    """One-line summary suitable for Grok prompts or dashboard captions."""
    score = result.get("score", 0)
    pct = result.get("quality_pct", 0)
    ctx = result.get("market_context", "?")
    direction = result.get("trend_direction", "?")
    hq = "HIGH QUALITY" if result.get("high_quality", False) else "below threshold"
    return (
        f"Signal quality: {pct}% ({hq}) — "
        f"context={ctx}, direction={direction}, "
        f"RSI={result.get('rsi', '?')}, AO={result.get('ao', '?')}"
    )
