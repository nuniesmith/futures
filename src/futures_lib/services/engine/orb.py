"""
Opening Range Breakout (ORB) Detector — TASK-801
==================================================
Detects the opening range (first 30 minutes after 9:30 ET equity open)
and flags breakouts when price exceeds OR_high + 0.5×ATR or drops below
OR_low - 0.5×ATR.

The opening range is defined as the highest high and lowest low of all
1-minute bars between 09:30:00 and 09:59:59 Eastern Time.

Public API:
    result = detect_opening_range_breakout(bars_1m, symbol="MGC")
    #  result = ORBResult(...)
    #  result.to_dict()  → JSON-friendly dict

    publish_orb_alert(result)   → push to Redis for SSE/dashboard

Usage from engine scheduler:
    from services.engine.orb import detect_opening_range_breakout, publish_orb_alert

    result = detect_opening_range_breakout(bars_1m, symbol="MGC")
    if result.breakout_detected:
        publish_orb_alert(result)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("engine.orb")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Opening range window: 09:30–10:00 ET (first 30 minutes of equity session)
OR_START = dt_time(9, 30)
OR_END = dt_time(10, 0)

# ATR look-back period (number of bars)
ATR_PERIOD = 14

# Breakout threshold multiplier applied to ATR
BREAKOUT_ATR_MULTIPLIER = 0.5

# Minimum number of bars required in the opening range to be valid
MIN_OR_BARS = 5

# Maximum number of bars we'd expect in a 30-min window of 1-min data
MAX_OR_BARS = 35  # allow a few extra for edge cases

# Redis keys
REDIS_KEY_ORB = "engine:orb"
REDIS_KEY_ORB_TS = "engine:orb:ts"
REDIS_PUBSUB_ORB = "dashboard:orb"
REDIS_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ORBResult:
    """Result of an Opening Range Breakout evaluation."""

    symbol: str = ""
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr_value: float = 0.0
    breakout_threshold: float = 0.0

    # Breakout detection
    breakout_detected: bool = False
    direction: str = ""  # "LONG", "SHORT", or ""
    trigger_price: float = 0.0
    breakout_bar_time: str = ""

    # Upper/lower breakout levels
    long_trigger: float = 0.0
    short_trigger: float = 0.0

    # Status
    or_complete: bool = False  # True after 10:00 ET
    or_bar_count: int = 0
    evaluated_at: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "type": "ORB",
            "symbol": self.symbol,
            "or_high": round(self.or_high, 4),
            "or_low": round(self.or_low, 4),
            "or_range": round(self.or_range, 4),
            "atr_value": round(self.atr_value, 4),
            "breakout_threshold": round(self.breakout_threshold, 4),
            "breakout_detected": self.breakout_detected,
            "direction": self.direction,
            "trigger_price": round(self.trigger_price, 4),
            "breakout_bar_time": self.breakout_bar_time,
            "long_trigger": round(self.long_trigger, 4),
            "short_trigger": round(self.short_trigger, 4),
            "or_complete": self.or_complete,
            "or_bar_count": self.or_bar_count,
            "evaluated_at": self.evaluated_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD
) -> float:
    """Compute the Average True Range for the given bar data.

    Uses the standard Wilder ATR calculation:
        TR = max(H-L, abs(H-prevC), abs(L-prevC))
        ATR = SMA(TR, period)

    Returns 0.0 if insufficient data.
    """
    n = len(closes)
    if n < period + 1:
        # Not enough data for a proper ATR; use simple H-L range average
        if n >= 2:
            return float(np.mean(highs[:n] - lows[:n]))
        return 0.0

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # Simple moving average of the last `period` true ranges
    if n >= period:
        atr = float(np.mean(tr[-period:]))
    else:
        atr = float(np.mean(tr))

    return atr


def _localize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is tz-aware in Eastern Time.

    Handles:
      - Already localized to US/Eastern or America/New_York → no-op
      - Localized to another tz → convert to Eastern
      - Naive (no tz) → localize as Eastern
    """
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        # Already tz-aware — convert to Eastern
        return df.tz_convert(_EST) if str(idx.tz) != str(_EST) else df  # type: ignore[attr-defined]
    else:
        # Naive — assume Eastern
        try:
            df.index = idx.tz_localize(_EST)  # type: ignore[attr-defined]
        except Exception:
            pass
        return df


def compute_opening_range(bars_1m: pd.DataFrame) -> tuple[float, float, int, bool]:
    """Extract the opening range (high, low) from 1-minute bars.

    Args:
        bars_1m: DataFrame with DatetimeIndex and columns: High, Low, Close.
                 Must contain bars from today's session.

    Returns:
        (or_high, or_low, bar_count, is_complete)
        is_complete is True if we have bars past OR_END (10:00 ET).
    """
    if bars_1m is None or bars_1m.empty:
        return 0.0, 0.0, 0, False

    df = _localize_index(bars_1m.copy())

    # Filter to bars within the opening range window
    times = df.index.time  # type: ignore[attr-defined]
    or_mask = (times >= OR_START) & (times < OR_END)
    or_bars = df[or_mask]

    if or_bars.empty:
        return 0.0, 0.0, 0, False

    or_high = float(or_bars["High"].max())  # type: ignore[arg-type, call-overload]
    or_low = float(or_bars["Low"].min())  # type: ignore[arg-type, call-overload]
    bar_count = len(or_bars)

    # Check if we have bars past the OR window (session progressed beyond 10:00)
    is_complete = bool(np.any(times >= OR_END))

    return or_high, or_low, bar_count, is_complete


def detect_opening_range_breakout(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    atr_period: int = ATR_PERIOD,
    breakout_multiplier: float = BREAKOUT_ATR_MULTIPLIER,
    now_fn: Optional[Callable[..., Any]] = None,
) -> ORBResult:
    """Detect an Opening Range Breakout from 1-minute bar data.

    Algorithm:
      1. Compute the opening range (OR) from bars between 09:30–10:00 ET.
      2. Compute ATR(14) from all available bars for volatility context.
      3. Define breakout levels:
           - Long trigger  = OR_high + (ATR × multiplier)
           - Short trigger = OR_low  - (ATR × multiplier)
      4. Scan bars after 10:00 ET for a close beyond either trigger.
      5. Return the first breakout found (or no breakout).

    Args:
        bars_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
        symbol: Instrument symbol for labelling (e.g. "MGC", "MNQ").
        atr_period: ATR look-back period (default 14).
        breakout_multiplier: ATR multiplier for breakout threshold (default 0.5).
        now_fn: Optional clock function for testability.

    Returns:
        ORBResult with breakout_detected=True/False and all details.
    """
    now = (now_fn or (lambda: datetime.now(tz=_EST)))()
    evaluated_at = now.isoformat()

    result = ORBResult(symbol=symbol, evaluated_at=evaluated_at)

    # --- Validate input ---
    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data provided"
        return result

    required_cols = {"High", "Low", "Close"}
    missing = required_cols - set(bars_1m.columns)
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(bars_1m) < MIN_OR_BARS:
        result.error = f"Insufficient bars ({len(bars_1m)} < {MIN_OR_BARS})"
        return result

    # --- Compute opening range ---
    or_high, or_low, or_bar_count, or_complete = compute_opening_range(bars_1m)

    result.or_high = or_high
    result.or_low = or_low
    result.or_range = or_high - or_low if or_high > or_low else 0.0
    result.or_bar_count = or_bar_count
    result.or_complete = or_complete

    if or_bar_count < MIN_OR_BARS:
        result.error = (
            f"Opening range has only {or_bar_count} bars (need >= {MIN_OR_BARS})"
        )
        return result

    if or_high <= 0 or or_low <= 0 or or_high <= or_low:
        result.error = f"Invalid opening range: high={or_high}, low={or_low}"
        return result

    # --- Compute ATR ---
    highs = bars_1m["High"].values.astype(float)
    lows = bars_1m["Low"].values.astype(float)
    closes = bars_1m["Close"].values.astype(float)

    atr = compute_atr(highs, lows, closes, period=atr_period)
    result.atr_value = atr

    if atr <= 0:
        result.error = "ATR is zero — cannot compute breakout thresholds"
        return result

    # --- Compute breakout levels ---
    threshold = atr * breakout_multiplier
    result.breakout_threshold = threshold
    result.long_trigger = or_high + threshold
    result.short_trigger = or_low - threshold

    # --- If OR isn't complete yet, just return the levels (no scan) ---
    if not or_complete:
        # OR is still forming — return the current range and triggers
        return result

    # --- Scan post-OR bars for breakout ---
    df = _localize_index(bars_1m.copy())
    times = df.index.time  # type: ignore[attr-defined]
    post_or_mask = times >= OR_END
    post_or_bars = df[post_or_mask]

    if post_or_bars.empty:
        # No post-OR bars yet — check not possible
        return result

    for idx, row in post_or_bars.iterrows():
        close = float(row["Close"])  # type: ignore[arg-type, call-overload]

        # Long breakout: close above OR_high + threshold
        if close > result.long_trigger:
            result.breakout_detected = True
            result.direction = "LONG"
            result.trigger_price = close
            result.breakout_bar_time = str(idx)
            break

        # Short breakout: close below OR_low - threshold
        if close < result.short_trigger:
            result.breakout_detected = True
            result.direction = "SHORT"
            result.trigger_price = close
            result.breakout_bar_time = str(idx)
            break

    if result.breakout_detected:
        logger.info(
            "ORB detected: %s %s @ %.4f (OR %.4f–%.4f, ATR %.4f, threshold %.4f)",
            result.direction,
            symbol,
            result.trigger_price,
            result.or_low,
            result.or_high,
            atr,
            threshold,
        )

    return result


# ---------------------------------------------------------------------------
# Multi-asset convenience
# ---------------------------------------------------------------------------


def scan_orb_all_assets(
    bars_by_symbol: dict[str, pd.DataFrame],
    atr_period: int = ATR_PERIOD,
    breakout_multiplier: float = BREAKOUT_ATR_MULTIPLIER,
) -> list[ORBResult]:
    """Run ORB detection across multiple symbols.

    Args:
        bars_by_symbol: Dict mapping symbol → 1-minute DataFrame.

    Returns:
        List of ORBResult for each symbol (including non-breakouts).
    """
    results = []
    for symbol, bars in bars_by_symbol.items():
        try:
            result = detect_opening_range_breakout(
                bars,
                symbol=symbol,
                atr_period=atr_period,
                breakout_multiplier=breakout_multiplier,
            )
            results.append(result)
        except Exception as exc:
            logger.error("ORB scan failed for %s: %s", symbol, exc)
            results.append(ORBResult(symbol=symbol, error=str(exc)))
    return results


# ---------------------------------------------------------------------------
# Redis publishing
# ---------------------------------------------------------------------------


def publish_orb_alert(result: ORBResult) -> bool:
    """Publish an ORB alert to Redis for SSE/dashboard consumption.

    Writes to:
      - ``engine:orb``              — full result JSON (TTL 300s)
      - ``engine:orb:ts``           — timestamp of last publish
      - Redis PubSub ``dashboard:orb`` — trigger for SSE event

    Returns True on success.
    """
    try:
        from src.futures_lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module for ORB publish")
        return False

    payload = result.to_dict()
    try:
        payload_json = json.dumps(payload, default=str)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize ORB result: %s", exc)
        return False

    try:
        cache_set(REDIS_KEY_ORB, payload_json.encode(), ttl=REDIS_TTL)
        cache_set(
            REDIS_KEY_ORB_TS,
            datetime.now(tz=_EST).isoformat().encode(),
            ttl=REDIS_TTL,
        )

        if REDIS_AVAILABLE and _r is not None:
            try:
                _r.publish(REDIS_PUBSUB_ORB, payload_json)
            except Exception as exc:
                logger.debug("ORB PubSub publish failed (non-fatal): %s", exc)

        logger.info(
            "ORB alert published: %s %s (OR %.4f–%.4f)",
            result.direction,
            result.symbol,
            result.or_low,
            result.or_high,
        )
        return True

    except Exception as exc:
        logger.error("Failed to publish ORB to Redis: %s", exc)
        return False


def clear_orb_alert() -> bool:
    """Clear any active ORB alert from Redis (e.g. end of day)."""
    try:
        from src.futures_lib.core.cache import cache_set
    except ImportError:
        return False

    try:
        cache_set(REDIS_KEY_ORB, b"", ttl=1)
        cache_set(REDIS_KEY_ORB_TS, b"", ttl=1)
        return True
    except Exception:
        return False
