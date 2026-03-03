"""
Opening Range Breakout (ORB) Detector — Multi-Session
==================================================
Detects opening range breakouts across multiple trading sessions:

  1. **London Open** (03:00–03:30 ET / 08:00–08:30 UTC)
     The primary ORB session for futures. London open drives the
     majority of daily range establishment for metals, indices, and
     energy futures. This is where institutional order flow begins.

  2. **US Equity Open** (09:30–10:00 ET)
     The traditional ORB window for equity-index futures (MES, MNQ).
     Still useful as a secondary confirmation or for equity-correlated
     instruments.

Each session is defined by an ORBSession dataclass with its own
start/end times, ATR period, and breakout multiplier. The detector
runs independently for each session and publishes results to
session-specific Redis keys.

Public API:
    result = detect_opening_range_breakout(bars_1m, symbol="MGC", session=LONDON_SESSION)
    result = detect_opening_range_breakout(bars_1m, symbol="MNQ", session=US_SESSION)
    results = detect_all_sessions(bars_1m, symbol="MGC")

    publish_orb_alert(result)   → push to Redis for SSE/dashboard

Usage from engine scheduler:
    from lib.services.engine.orb import (
        detect_opening_range_breakout,
        detect_all_sessions,
        publish_orb_alert,
        LONDON_SESSION,
        US_SESSION,
        ORB_SESSIONS,
    )

    # Single session
    result = detect_opening_range_breakout(bars_1m, symbol="MGC", session=LONDON_SESSION)

    # All sessions
    for result in detect_all_sessions(bars_1m, symbol="MGC"):
        if result.breakout_detected:
            publish_orb_alert(result)
"""

import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("engine.orb")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Session Definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ORBSession:
    """Definition of an Opening Range session window.

    Each session has its own time window, ATR parameters, and Redis keys.
    Frozen so instances are hashable and can be used as dict keys.
    """

    name: str  # Human-readable name
    key: str  # Short key for Redis/logs ("london", "us")
    or_start: dt_time  # Opening range start (ET)
    or_end: dt_time  # Opening range end (ET)
    scan_end: dt_time  # Stop scanning for breakouts after this time (ET)
    atr_period: int = 14  # ATR look-back period
    breakout_multiplier: float = 0.5  # ATR multiplier for breakout threshold
    min_bars: int = 5  # Minimum bars required in OR window
    max_bars: int = 35  # Maximum expected bars in OR window
    description: str = ""


# London Open: 03:00–03:30 ET (08:00–08:30 UTC)
# Primary session — institutional flow, daily range establishment.
# Scan for breakouts until 03:00 ET (gives 1.5 hours post-OR).
LONDON_SESSION = ORBSession(
    name="London Open",
    key="london",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    scan_end=dt_time(5, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description="London open session (03:00–03:30 ET / 08:00–08:30 UTC)",
)

# US Equity Open: 09:30–10:00 ET
# Secondary session — equity cash open.
# Scan for breakouts until 11:00 ET (gives 1 hour post-OR).
US_SESSION = ORBSession(
    name="US Equity Open",
    key="us",
    or_start=dt_time(9, 30),
    or_end=dt_time(10, 0),
    scan_end=dt_time(11, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description="US equity cash open (09:30–10:00 ET)",
)

# All sessions in priority order (London first — it's the primary session)
ORB_SESSIONS: list[ORBSession] = [LONDON_SESSION, US_SESSION]

# Legacy aliases for backward compatibility
OR_START = US_SESSION.or_start
OR_END = US_SESSION.or_end
ATR_PERIOD = 14
BREAKOUT_ATR_MULTIPLIER = 0.5
MIN_OR_BARS = 5
MAX_OR_BARS = 35


# ---------------------------------------------------------------------------
# Redis keys — session-aware
# ---------------------------------------------------------------------------


def _redis_key_orb(session: ORBSession) -> str:
    """Redis key for a session's ORB result."""
    return f"engine:orb:{session.key}"


def _redis_key_orb_ts(session: ORBSession) -> str:
    """Redis key for a session's ORB timestamp."""
    return f"engine:orb:{session.key}:ts"


def _redis_pubsub_orb(session: ORBSession) -> str:
    """Redis pub/sub channel for a session's ORB alerts."""
    return f"dashboard:orb:{session.key}"


# Legacy keys (used by dashboard for backward compat — stores "best" result)
REDIS_KEY_ORB = "engine:orb"
REDIS_KEY_ORB_TS = "engine:orb:ts"
REDIS_PUBSUB_ORB = "dashboard:orb"
REDIS_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ORBResult:
    """Result of an Opening Range Breakout evaluation for a specific session."""

    symbol: str = ""
    session_name: str = ""  # "London Open", "US Equity Open"
    session_key: str = ""  # "london", "us"
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
    or_complete: bool = False  # True after OR window ends
    or_bar_count: int = 0
    evaluated_at: str = ""
    error: str = ""

    # CNN enrichment (populated by engine main after inference)
    cnn_prob: float | None = None
    cnn_confidence: str = ""
    cnn_signal: bool | None = None

    # Filter enrichment (populated by engine main after filters)
    filter_passed: bool | None = None
    filter_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d = {
            "type": "ORB",
            "symbol": self.symbol,
            "session_name": self.session_name,
            "session_key": self.session_key,
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
        # Include CNN data if present
        if self.cnn_prob is not None:
            d["cnn_prob"] = round(self.cnn_prob, 4)
            d["cnn_confidence"] = self.cnn_confidence
            d["cnn_signal"] = bool(self.cnn_signal) if self.cnn_signal is not None else False  # type: ignore[assignment]
        # Include filter data if present
        if self.filter_passed is not None:
            d["filter_passed"] = bool(self.filter_passed)
            d["filter_summary"] = self.filter_summary
        return d


@dataclass
class MultiSessionORBResult:
    """Aggregated ORB results across all sessions for a single symbol."""

    symbol: str = ""
    sessions: dict[str, ORBResult] = field(default_factory=dict)
    evaluated_at: str = ""

    @property
    def has_any_breakout(self) -> bool:
        return any(r.breakout_detected for r in self.sessions.values())

    @property
    def active_breakouts(self) -> list[ORBResult]:
        return [r for r in self.sessions.values() if r.breakout_detected]

    @property
    def best_breakout(self) -> ORBResult | None:
        """Return the breakout with the largest OR range (most significant)."""
        breakouts = self.active_breakouts
        if not breakouts:
            return None
        return max(breakouts, key=lambda r: r.or_range)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "evaluated_at": self.evaluated_at,
            "has_any_breakout": self.has_any_breakout,
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_atr(
    highs: "np.ndarray | Any",
    lows: "np.ndarray | Any",
    closes: "np.ndarray | Any",
    period: int = ATR_PERIOD,
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
        hl = float(highs[i] - lows[i])
        hc = float(abs(highs[i] - closes[i - 1]))
        lc = float(abs(lows[i] - closes[i - 1]))
        tr[i] = max(hl, hc, lc)  # noqa: E501

    # Simple moving average of the last `period` true ranges
    atr = float(np.mean(tr[-period:])) if n >= period else float(np.mean(tr))

    return atr


def _localize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is tz-aware in Eastern Time.

    Handles:
      - Already localized to US/Eastern or America/New_York → no-op
      - Localized to another tz → convert to Eastern
      - Naive (no tz) → localize as Eastern
    """
    idx: Any = df.index
    if getattr(idx, "tz", None) is not None:
        # Already tz-aware — convert to Eastern
        return df.tz_convert(_EST) if str(idx.tz) != str(_EST) else df
    else:
        # Naive — assume Eastern
        with contextlib.suppress(Exception):
            df.index = idx.tz_localize(_EST)
        return df


def compute_opening_range(
    bars_1m: pd.DataFrame | None,
    session: ORBSession | None = None,
) -> tuple[float, float, int, bool]:
    """Extract the opening range (high, low) from 1-minute bars.

    Args:
        bars_1m: DataFrame with DatetimeIndex and columns: High, Low, Close.
                 Must contain bars from today's session.
        session: ORB session definition. Defaults to US_SESSION for
                 backward compatibility.

    Returns:
        (or_high, or_low, bar_count, is_complete)
        is_complete is True if we have bars past the OR end time.
    """
    if session is None:
        session = US_SESSION

    if bars_1m is None or bars_1m.empty:
        return 0.0, 0.0, 0, False

    df = _localize_index(bars_1m.copy())

    # Filter to bars within the opening range window
    idx: Any = df.index
    times = idx.time
    or_mask = (times >= session.or_start) & (times < session.or_end)
    or_bars: pd.DataFrame = df.loc[or_mask]

    if or_bars.empty:
        return 0.0, 0.0, 0, False

    or_high = float(or_bars["High"].max())  # type: ignore[arg-type]
    or_low = float(or_bars["Low"].min())  # type: ignore[arg-type]
    bar_count: int = len(or_bars)

    # Check if we have bars past the OR window (session progressed beyond OR end)
    is_complete = bool(np.any(times >= session.or_end))

    return or_high, or_low, bar_count, is_complete


def detect_opening_range_breakout(
    bars_1m: pd.DataFrame | None,
    symbol: str = "",
    session: ORBSession | None = None,
    atr_period: int | None = None,
    breakout_multiplier: float | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> ORBResult:
    """Detect an Opening Range Breakout from 1-minute bar data.

    Algorithm:
      1. Compute the opening range (OR) from bars within the session window.
      2. Compute ATR from all available bars for volatility context.
      3. Define breakout levels:
           - Long trigger  = OR_high + (ATR * multiplier)
           - Short trigger = OR_low  - (ATR * multiplier)
      4. Scan bars after OR end (up to scan_end) for a close beyond either trigger.
      5. Return the first breakout found (or no breakout).

    Args:
        bars_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
        symbol: Instrument symbol for labelling (e.g. "MGC", "MNQ").
        session: ORB session definition. Defaults to US_SESSION.
        atr_period: ATR look-back period (overrides session default).
        breakout_multiplier: ATR multiplier (overrides session default).
        now_fn: Optional clock function for testability.

    Returns:
        ORBResult with breakout_detected=True/False and all details.
    """
    if session is None:
        session = US_SESSION

    _atr_period = atr_period if atr_period is not None else session.atr_period
    _breakout_mult = breakout_multiplier if breakout_multiplier is not None else session.breakout_multiplier

    now = (now_fn or (lambda: datetime.now(tz=_EST)))()
    evaluated_at = now.isoformat()

    result = ORBResult(
        symbol=symbol,
        session_name=session.name,
        session_key=session.key,
        evaluated_at=evaluated_at,
    )

    # --- Validate input ---
    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data provided"
        return result

    required_cols = {"High", "Low", "Close"}
    missing = required_cols - set(bars_1m.columns)
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(bars_1m) < session.min_bars:
        result.error = f"Insufficient bars ({len(bars_1m)} < {session.min_bars})"
        return result

    # --- Compute opening range ---
    or_high, or_low, or_bar_count, or_complete = compute_opening_range(bars_1m, session=session)

    result.or_high = or_high
    result.or_low = or_low
    result.or_range = or_high - or_low if or_high > or_low else 0.0
    result.or_bar_count = or_bar_count
    result.or_complete = or_complete

    if or_bar_count < session.min_bars:
        result.error = f"{session.name} opening range has only {or_bar_count} bars (need >= {session.min_bars})"
        return result

    if or_high <= 0 or or_low <= 0 or or_high <= or_low:
        result.error = f"Invalid opening range: high={or_high}, low={or_low}"
        return result

    # --- Compute ATR ---
    highs = bars_1m["High"].values.astype(float)
    lows = bars_1m["Low"].values.astype(float)
    closes = bars_1m["Close"].values.astype(float)

    atr = compute_atr(highs, lows, closes, period=_atr_period)
    result.atr_value = atr

    if atr <= 0:
        result.error = "ATR is zero — cannot compute breakout thresholds"
        return result

    # --- Compute breakout levels ---
    threshold = atr * _breakout_mult
    result.breakout_threshold = threshold
    result.long_trigger = or_high + threshold
    result.short_trigger = or_low - threshold

    # --- If OR isn't complete yet, just return the levels (no scan) ---
    if not or_complete:
        # OR is still forming — return the current range and triggers
        return result

    # --- Scan post-OR bars for breakout ---
    df = _localize_index(bars_1m.copy())
    _idx: Any = df.index
    times = _idx.time
    # Only scan bars between OR end and scan end
    post_or_mask = (times >= session.or_end) & (times <= session.scan_end)
    post_or_bars: pd.DataFrame = df.loc[post_or_mask]

    if post_or_bars.empty:
        # No post-OR bars yet — check not possible
        return result

    for idx_label, row in post_or_bars.iterrows():
        close: float = float(row["Close"])  # type: ignore[arg-type]

        # Long breakout: close above OR_high + threshold
        if close > result.long_trigger:
            result.breakout_detected = True
            result.direction = "LONG"
            result.trigger_price = close
            result.breakout_bar_time = str(idx_label)
            break

        # Short breakout: close below OR_low - threshold
        if close < result.short_trigger:
            result.breakout_detected = True
            result.direction = "SHORT"
            result.trigger_price = close
            result.breakout_bar_time = str(idx_label)
            break

    if result.breakout_detected:
        logger.info(
            "ORB [%s] detected: %s %s @ %.4f (OR %.4f–%.4f, ATR %.4f, threshold %.4f)",
            session.name,
            result.direction,
            symbol,
            result.trigger_price,
            result.or_low,
            result.or_high,
            atr,
            threshold,
        )

    return result


def detect_all_sessions(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    sessions: list[ORBSession] | None = None,
    now_fn: Callable[..., Any] | None = None,
) -> MultiSessionORBResult:
    """Run ORB detection for all sessions on a single symbol.

    Args:
        bars_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
        symbol: Instrument symbol.
        sessions: List of sessions to check. Defaults to ORB_SESSIONS.
        now_fn: Optional clock function for testability.

    Returns:
        MultiSessionORBResult containing results for each session.
    """
    if sessions is None:
        sessions = ORB_SESSIONS

    now = (now_fn or (lambda: datetime.now(tz=_EST)))()
    multi = MultiSessionORBResult(
        symbol=symbol,
        evaluated_at=now.isoformat(),
    )

    for session in sessions:
        try:
            result = detect_opening_range_breakout(
                bars_1m,
                symbol=symbol,
                session=session,
                now_fn=now_fn,
            )
            multi.sessions[session.key] = result
        except Exception as exc:
            logger.error(
                "ORB [%s] detection failed for %s: %s",
                session.name,
                symbol,
                exc,
            )
            multi.sessions[session.key] = ORBResult(
                symbol=symbol,
                session_name=session.name,
                session_key=session.key,
                error=str(exc),
                evaluated_at=now.isoformat(),
            )

    return multi


# ---------------------------------------------------------------------------
# Multi-asset convenience
# ---------------------------------------------------------------------------


def scan_orb_all_assets(
    bars_by_symbol: dict[str, pd.DataFrame],
    session: ORBSession | None = None,
    atr_period: int | None = None,
    breakout_multiplier: float | None = None,
) -> list[ORBResult]:
    """Run ORB detection across multiple symbols for a single session.

    Args:
        bars_by_symbol: Dict mapping symbol → 1-minute DataFrame.
        session: ORB session. Defaults to US_SESSION.
        atr_period: Override ATR period.
        breakout_multiplier: Override breakout multiplier.

    Returns:
        List of ORBResult for each symbol (including non-breakouts).
    """
    results = []
    for symbol, bars in bars_by_symbol.items():
        try:
            result = detect_opening_range_breakout(
                bars,
                symbol=symbol,
                session=session,
                atr_period=atr_period,
                breakout_multiplier=breakout_multiplier,
            )
            results.append(result)
        except Exception as exc:
            logger.error("ORB scan failed for %s: %s", symbol, exc)
            results.append(ORBResult(symbol=symbol, error=str(exc)))
    return results


def scan_orb_all_sessions_all_assets(
    bars_by_symbol: dict[str, pd.DataFrame],
    sessions: list[ORBSession] | None = None,
) -> dict[str, MultiSessionORBResult]:
    """Run ORB detection across multiple symbols and all sessions.

    Args:
        bars_by_symbol: Dict mapping symbol → 1-minute DataFrame.
        sessions: Sessions to check. Defaults to ORB_SESSIONS.

    Returns:
        Dict mapping symbol → MultiSessionORBResult.
    """
    results = {}
    for symbol, bars in bars_by_symbol.items():
        try:
            results[symbol] = detect_all_sessions(bars, symbol=symbol, sessions=sessions)
        except Exception as exc:
            logger.error("Multi-session ORB scan failed for %s: %s", symbol, exc)
    return results


# ---------------------------------------------------------------------------
# Session-awareness helpers
# ---------------------------------------------------------------------------


def get_active_sessions(now: datetime | None = None) -> list[ORBSession]:
    """Return sessions that are currently in their OR formation or scan window.

    Useful for the scheduler to know which sessions need checking right now.

    Args:
        now: Current time (tz-aware). Defaults to now in ET.

    Returns:
        List of ORBSession objects that are currently active.
    """
    if now is None:
        now = datetime.now(tz=_EST)

    now_et = now.astimezone(_EST) if now.tzinfo else now.replace(tzinfo=_EST)
    t = now_et.time()

    active = []
    for session in ORB_SESSIONS:
        # A session is "active" from its OR start through scan_end
        if session.or_start <= t <= session.scan_end:
            active.append(session)

    return active


def is_any_session_active(now: datetime | None = None) -> bool:
    """Check if any ORB session is currently active."""
    return len(get_active_sessions(now)) > 0


def get_session_status(now: datetime | None = None) -> dict[str, str]:
    """Return a status dict for each session.

    Possible statuses: "waiting", "forming", "scanning", "complete"
    """
    if now is None:
        now = datetime.now(tz=_EST)

    now_et = now.astimezone(_EST) if now.tzinfo else now.replace(tzinfo=_EST)
    t = now_et.time()

    statuses = {}
    for session in ORB_SESSIONS:
        if t < session.or_start:
            statuses[session.key] = "waiting"
        elif session.or_start <= t < session.or_end:
            statuses[session.key] = "forming"
        elif session.or_end <= t <= session.scan_end:
            statuses[session.key] = "scanning"
        else:
            statuses[session.key] = "complete"

    return statuses


# ---------------------------------------------------------------------------
# Redis publishing
# ---------------------------------------------------------------------------


def publish_orb_alert(result: ORBResult, session: ORBSession | None = None) -> bool:
    """Publish an ORB alert to Redis for SSE/dashboard consumption.

    Writes to:
      - ``engine:orb:{session_key}``     — session-specific result (TTL 300s)
      - ``engine:orb:{session_key}:ts``  — timestamp of last publish
      - ``engine:orb``                   — legacy combined key (best breakout)
      - ``engine:orb:ts``                — legacy timestamp
      - Redis PubSub ``dashboard:orb:{session_key}`` — session-specific trigger
      - Redis PubSub ``dashboard:orb``   — legacy trigger

    Returns True on success.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module for ORB publish")
        return False

    # Determine the session from the result if not provided
    if session is None:
        # Try to find matching session from result's session_key
        for s in ORB_SESSIONS:
            if s.key == result.session_key:
                session = s
                break

    payload = result.to_dict()
    try:
        payload_json = json.dumps(payload, default=str)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize ORB result: %s", exc)
        return False

    now_iso = datetime.now(tz=_EST).isoformat().encode()

    try:
        # Write to session-specific keys
        if session is not None:
            cache_set(_redis_key_orb(session), payload_json.encode(), ttl=REDIS_TTL)
            cache_set(_redis_key_orb_ts(session), now_iso, ttl=REDIS_TTL)

        # Write to legacy combined key (for backward compat with dashboard)
        cache_set(REDIS_KEY_ORB, payload_json.encode(), ttl=REDIS_TTL)
        cache_set(REDIS_KEY_ORB_TS, now_iso, ttl=REDIS_TTL)

        if REDIS_AVAILABLE and _r is not None:
            try:
                # Session-specific pub/sub
                if session is not None:
                    _r.publish(_redis_pubsub_orb(session), payload_json)
                # Legacy pub/sub
                _r.publish(REDIS_PUBSUB_ORB, payload_json)
            except Exception as exc:
                logger.debug("ORB PubSub publish failed (non-fatal): %s", exc)

        logger.info(
            "ORB [%s] alert published: %s %s (OR %.4f–%.4f)",
            result.session_name or "unknown",
            result.direction,
            result.symbol,
            result.or_low,
            result.or_high,
        )
        return True

    except Exception as exc:
        logger.error("Failed to publish ORB to Redis: %s", exc)
        return False


def publish_multi_session_orb(multi: MultiSessionORBResult) -> bool:
    """Publish all session results from a MultiSessionORBResult.

    Publishes each session result to its own Redis key, plus writes
    a combined payload to ``engine:orb:multi:{symbol}`` for the
    dashboard's multi-session panel.

    Returns True if all publishes succeed.
    """
    try:
        from lib.core.cache import cache_set
    except ImportError:
        logger.error("Cannot import cache module for multi-session ORB publish")
        return False

    success = True

    # Publish individual session results
    for session_key, result in multi.sessions.items():
        session_obj = None
        for s in ORB_SESSIONS:
            if s.key == session_key:
                session_obj = s
                break

        if not publish_orb_alert(result, session=session_obj):
            success = False

    # Publish combined multi-session payload
    try:
        combined_payload = json.dumps(multi.to_dict(), default=str)
        cache_set(
            f"engine:orb:multi:{multi.symbol}",
            combined_payload.encode(),
            ttl=REDIS_TTL,
        )
    except Exception as exc:
        logger.error("Failed to publish multi-session ORB: %s", exc)
        success = False

    return success


def clear_orb_alert(session: ORBSession | None = None) -> bool:
    """Clear any active ORB alert from Redis (e.g. end of day).

    If session is None, clears all sessions and the legacy key.
    """
    try:
        from lib.core.cache import cache_set
    except ImportError:
        return False

    try:
        if session is not None:
            cache_set(_redis_key_orb(session), b"", ttl=1)
            cache_set(_redis_key_orb_ts(session), b"", ttl=1)
        else:
            # Clear all sessions
            for s in ORB_SESSIONS:
                cache_set(_redis_key_orb(s), b"", ttl=1)
                cache_set(_redis_key_orb_ts(s), b"", ttl=1)
            # Clear legacy keys
            cache_set(REDIS_KEY_ORB, b"", ttl=1)
            cache_set(REDIS_KEY_ORB_TS, b"", ttl=1)
        return True
    except Exception:
        return False
