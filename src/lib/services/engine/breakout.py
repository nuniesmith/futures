"""
Generalised Breakout Detection Engine
======================================
Replaces all ORB-specific breakout handlers with a single, type-aware
``detect_range_breakout()`` function that works for every range type:

  - **ORB**  — Opening Range Breakout (existing, session-parameterised)
  - **PDR**  — Previous Day Range breakout (daily high/low anchor)
  - **IB**   — Initial Balance breakout (first 60 min of RTH, 09:30–10:30 ET)
  - **CONS** — Consolidation / Squeeze breakout (ATR/Bollinger contraction)

Design principles
-----------------
1. **Single entry-point**: ``detect_range_breakout(bars, symbol, config)``
   returns a ``BreakoutResult`` regardless of type.  Callers don't need to
   know how each range is built — they set ``BreakoutType`` in the config.

2. **Composable config**: ``RangeConfig`` is a frozen dataclass that holds
   all thresholds (depth, body, OR-size floor/cap, ATR period, etc.) plus
   the ``BreakoutType``.  Callers can start from pre-built ``DEFAULT_CONFIGS``
   and override fields with ``dataclasses.replace()``.

3. **No side-effects**: detection is pure — no Redis, no DB writes.  The
   engine main / handler layer is responsible for publishing, persisting, and
   filtering.

4. **Backward-compatible**: the existing ``ORBResult`` / ``detect_opening_range_breakout``
   pipeline is untouched.  This module adds new types alongside the ORB
   infrastructure rather than replacing it until a later migration step.

Breakout types
--------------
ORB   Opening Range — already handled by orb.py; mirrored here so callers
      can use a single dispatch path via ``detect_range_breakout``.

PDR   Previous Day Range — the prior trading session's high and low define
      the range.  A breakout occurs when price closes beyond either level
      by at least ``min_depth_atr_pct × ATR``.  Strongest at London open
      and US open when yesterday's levels act as magnets / targets.

IB    Initial Balance — the high/low of the first 60 minutes of the RTH
      session (09:30–10:30 ET) defines the range.  Breakouts after 10:30
      are the "B-type" IB breakout from the Dalton/Steidlmayer market-profile
      playbook.  Very high win-rate on trend days when IB is narrow.

CONS  Consolidation / Squeeze — uses a Bollinger Band / ATR contraction
      detector to find periods of range compression, then detects the
      expansion bar.  Parametrised by ``squeeze_atr_mult`` (BB width as a
      fraction of ATR) and ``squeeze_lookback`` (bars to confirm compression).

Public API
----------
    from lib.services.engine.breakout import (
        BreakoutType,
        RangeConfig,
        BreakoutResult,
        detect_range_breakout,
        DEFAULT_CONFIGS,
    )

    config = DEFAULT_CONFIGS[BreakoutType.IB]
    result = detect_range_breakout(bars_1m, symbol="MES=F", config=config)
    if result.breakout_detected:
        print(result.direction, result.trigger_price)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.core.breakout_types import BreakoutType as TrainingBreakoutType
from lib.core.breakout_types import get_range_config

logger = logging.getLogger("engine.breakout")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ===========================================================================
# Enumerations
# ===========================================================================


class BreakoutType(StrEnum):
    """All supported breakout range types (engine runtime).

    This is the *engine-side* enum used for live detection and logging.
    For CNN training ordinals and ONNX feature values, use the canonical
    ``lib.core.breakout_types.BreakoutType`` (IntEnum) instead.

    Use ``to_training_type()`` / ``from_training_type()`` to convert.
    """

    ORB = "ORB"  # Opening Range Breakout (session-parameterised)
    PDR = "PDR"  # Previous Day Range
    IB = "IB"  # Initial Balance (first 60 min RTH)
    CONS = "CONS"  # Consolidation / Squeeze expansion
    WEEKLY = "WEEKLY"  # Prior week's high/low
    MONTHLY = "MONTHLY"  # Prior month's high/low
    ASIAN = "ASIAN"  # Asian session range (19:00–02:00 ET)
    BBSQUEEZE = "BBSQUEEZE"  # Bollinger Band squeeze → expansion
    VA = "VA"  # Value Area (VAH/VAL from volume profile)
    INSIDE = "INSIDE"  # Inside Day breakout
    GAP = "GAP"  # Gap Rejection / fill breakout
    PIVOT = "PIVOT"  # Classic floor pivot R1/S1
    FIB = "FIB"  # Fibonacci 38.2%–61.8% retracement zone


# ---------------------------------------------------------------------------
# Mapping between engine StrEnum ↔ training IntEnum
# ---------------------------------------------------------------------------

_ENGINE_TO_TRAINING: dict[BreakoutType, TrainingBreakoutType] = {
    BreakoutType.ORB: TrainingBreakoutType.ORB,
    BreakoutType.PDR: TrainingBreakoutType.PrevDay,
    BreakoutType.IB: TrainingBreakoutType.InitialBalance,
    BreakoutType.CONS: TrainingBreakoutType.Consolidation,
    BreakoutType.WEEKLY: TrainingBreakoutType.Weekly,
    BreakoutType.MONTHLY: TrainingBreakoutType.Monthly,
    BreakoutType.ASIAN: TrainingBreakoutType.Asian,
    BreakoutType.BBSQUEEZE: TrainingBreakoutType.BollingerSqueeze,
    BreakoutType.VA: TrainingBreakoutType.ValueArea,
    BreakoutType.INSIDE: TrainingBreakoutType.InsideDay,
    BreakoutType.GAP: TrainingBreakoutType.GapRejection,
    BreakoutType.PIVOT: TrainingBreakoutType.PivotPoints,
    BreakoutType.FIB: TrainingBreakoutType.Fibonacci,
}

_TRAINING_TO_ENGINE: dict[TrainingBreakoutType, BreakoutType] = {v: k for k, v in _ENGINE_TO_TRAINING.items()}


def to_training_type(bt: BreakoutType) -> TrainingBreakoutType:
    """Convert an engine ``BreakoutType`` to the canonical training ``IntEnum``.

    This is needed when passing breakout type ordinals to the CNN model
    or writing to the feature contract.
    """
    return _ENGINE_TO_TRAINING[bt]


def from_training_type(tbt: TrainingBreakoutType) -> BreakoutType:
    """Convert a training ``BreakoutType`` (IntEnum) back to the engine ``StrEnum``."""
    return _TRAINING_TO_ENGINE[tbt]


def breakout_type_ordinal(bt: BreakoutType) -> float:
    """Return the normalised CNN ordinal [0, 1] for an engine ``BreakoutType``."""
    return get_range_config(to_training_type(bt)).breakout_type_ord


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass(frozen=True)
class RangeConfig:
    """Configuration for a single breakout-type detector.

    All threshold values carry sensible defaults tuned for micro CME
    futures on 1-minute bars.  Override via ``dataclasses.replace()``.
    """

    # --- Identity ---
    breakout_type: BreakoutType = BreakoutType.ORB
    label: str = ""  # human-readable name for logging / alerts

    # --- ATR ---
    atr_period: int = 14  # look-back bars for ATR computation
    atr_multiplier: float = 0.5  # breakout threshold = atr_multiplier × ATR

    # --- Range quality gates ---
    min_depth_atr_pct: float = 0.15  # close must clear level by ≥ N×ATR
    min_body_ratio: float = 0.55  # body/range ≥ this on breakout bar
    max_range_atr_ratio: float = 2.0  # range cap  (skip if too wide)
    min_range_atr_ratio: float = 0.05  # range floor (skip if too narrow)
    min_bars: int = 3  # minimum bars to form range

    # --- IB-specific ---
    ib_duration_minutes: int = 60  # Initial Balance window length (min)
    ib_start_time: dt_time = dt_time(9, 30)  # RTH open

    # --- PDR-specific ---
    pdr_session_start: dt_time = dt_time(18, 0)  # Globex-day start (ET)
    pdr_session_end: dt_time = dt_time(17, 0)  # Globex-day end (ET)

    # --- Consolidation / Squeeze ---
    squeeze_lookback: int = 20  # bars to measure contraction over
    squeeze_atr_mult: float = 1.5  # BB must be < mult×ATR to qualify as squeeze
    squeeze_bb_period: int = 20  # Bollinger Band period
    squeeze_bb_std: float = 2.0  # Bollinger Band std-dev multiplier
    squeeze_min_bars: int = 5  # min consecutive squeeze bars needed

    def __str__(self) -> str:
        name = self.label or self.breakout_type.value
        return f"RangeConfig[{name}]"


# ---------------------------------------------------------------------------
# Pre-built defaults — one per BreakoutType
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[BreakoutType, RangeConfig] = {
    BreakoutType.ORB: RangeConfig(
        breakout_type=BreakoutType.ORB,
        label="Opening Range Breakout",
        atr_period=14,
        atr_multiplier=0.5,
        min_depth_atr_pct=0.15,
        min_body_ratio=0.55,
        max_range_atr_ratio=1.8,
        min_range_atr_ratio=0.05,
        min_bars=5,
    ),
    BreakoutType.PDR: RangeConfig(
        breakout_type=BreakoutType.PDR,
        label="Previous Day Range",
        atr_period=14,
        atr_multiplier=0.4,
        min_depth_atr_pct=0.20,  # PDR levels need meaningful penetration
        min_body_ratio=0.50,
        max_range_atr_ratio=3.0,  # daily ranges can be wide
        min_range_atr_ratio=0.10,
        min_bars=1,
    ),
    BreakoutType.IB: RangeConfig(
        breakout_type=BreakoutType.IB,
        label="Initial Balance",
        atr_period=14,
        atr_multiplier=0.5,
        min_depth_atr_pct=0.15,
        min_body_ratio=0.52,
        max_range_atr_ratio=2.5,
        min_range_atr_ratio=0.05,
        min_bars=10,  # need ≥10 bars in 60-min IB window
        ib_duration_minutes=60,
        ib_start_time=dt_time(9, 30),
    ),
    BreakoutType.CONS: RangeConfig(
        breakout_type=BreakoutType.CONS,
        label="Consolidation/Squeeze",
        atr_period=14,
        atr_multiplier=0.6,  # squeeze breakouts tend to be explosive
        min_depth_atr_pct=0.18,
        min_body_ratio=0.55,
        max_range_atr_ratio=1.2,  # squeeze range must be narrow
        min_range_atr_ratio=0.02,
        min_bars=5,
        squeeze_lookback=20,
        squeeze_atr_mult=1.5,
        squeeze_bb_period=20,
        squeeze_bb_std=2.0,
        squeeze_min_bars=5,
    ),
}


# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class BreakoutResult:
    """Result of ``detect_range_breakout()`` for any BreakoutType.

    Mirrors the fields of ``ORBResult`` so the same publishing / filtering /
    persistence pipeline can consume both without branching.
    """

    # --- Identity ---
    symbol: str = ""
    breakout_type: BreakoutType = BreakoutType.ORB
    label: str = ""  # human-readable label from config

    # --- Range ---
    range_high: float = 0.0
    range_low: float = 0.0
    range_size: float = 0.0  # range_high − range_low
    atr_value: float = 0.0
    breakout_threshold: float = 0.0  # threshold = atr × multiplier

    # --- Breakout ---
    breakout_detected: bool = False
    direction: str = ""  # "LONG", "SHORT", or ""
    trigger_price: float = 0.0
    breakout_bar_time: str = ""

    # --- Levels ---
    long_trigger: float = 0.0  # range_high + threshold
    short_trigger: float = 0.0  # range_low  − threshold

    # --- Range formation ---
    range_complete: bool = False  # True once the range window has closed
    range_bar_count: int = 0
    evaluated_at: str = ""
    error: str = ""

    # --- Quality gate results ---
    depth_ok: bool | None = None
    body_ratio_ok: bool | None = None
    range_size_ok: bool | None = None
    breakout_bar_depth: float = 0.0
    breakout_bar_body_ratio: float = 0.0

    # --- Filter enrichment (set by engine after apply_all_filters) ---
    filter_passed: bool | None = None
    filter_summary: str = ""

    # --- MTF enrichment (set by engine after MTF analyzer) ---
    mtf_score: float | None = None  # 0.0 – 1.0 aggregate MTF score
    mtf_direction: str = ""  # "bullish", "bearish", "neutral"
    macd_slope: float | None = None  # MACD histogram slope (signed)
    macd_divergence: bool | None = None  # True if price/MACD diverge

    # --- CNN enrichment (set by engine after inference) ---
    cnn_prob: float | None = None
    cnn_confidence: str = ""
    cnn_signal: bool | None = None

    # --- Squeeze-specific (CONS type only) ---
    squeeze_detected: bool = False
    squeeze_bar_count: int = 0
    squeeze_bb_width: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0

    # --- PDR-specific ---
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    prev_day_range: float = 0.0

    # --- IB-specific ---
    ib_high: float = 0.0
    ib_low: float = 0.0
    ib_complete: bool = False

    # --- Extra metadata ---
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "type": self.breakout_type.value,
            "label": self.label,
            "symbol": self.symbol,
            "range_high": round(self.range_high, 4),
            "range_low": round(self.range_low, 4),
            "range_size": round(self.range_size, 4),
            "atr_value": round(self.atr_value, 4),
            "breakout_threshold": round(self.breakout_threshold, 4),
            "breakout_detected": self.breakout_detected,
            "direction": self.direction,
            "trigger_price": round(self.trigger_price, 4),
            "breakout_bar_time": self.breakout_bar_time,
            "long_trigger": round(self.long_trigger, 4),
            "short_trigger": round(self.short_trigger, 4),
            "range_complete": self.range_complete,
            "range_bar_count": self.range_bar_count,
            "evaluated_at": self.evaluated_at,
            "error": self.error,
            # Quality gates
            "depth_ok": self.depth_ok,
            "body_ratio_ok": self.body_ratio_ok,
            "range_size_ok": self.range_size_ok,
            "breakout_bar_depth": round(self.breakout_bar_depth, 6),
            "breakout_bar_body_ratio": round(self.breakout_bar_body_ratio, 4),
        }
        # Optional enrichment
        if self.filter_passed is not None:
            d["filter_passed"] = bool(self.filter_passed)
            d["filter_summary"] = self.filter_summary
        if self.mtf_score is not None:
            d["mtf_score"] = round(self.mtf_score, 4)
            d["mtf_direction"] = self.mtf_direction
        if self.macd_slope is not None:
            d["macd_slope"] = round(self.macd_slope, 6)
        if self.macd_divergence is not None:
            d["macd_divergence"] = bool(self.macd_divergence)
        if self.cnn_prob is not None:
            d["cnn_prob"] = round(self.cnn_prob, 4)
            d["cnn_confidence"] = self.cnn_confidence
            d["cnn_signal"] = bool(self.cnn_signal) if self.cnn_signal is not None else False
        # Type-specific
        if self.breakout_type == BreakoutType.CONS:
            d["squeeze_detected"] = self.squeeze_detected
            d["squeeze_bar_count"] = self.squeeze_bar_count
            d["squeeze_bb_width"] = round(self.squeeze_bb_width, 4)
            d["bb_upper"] = round(self.bb_upper, 4)
            d["bb_lower"] = round(self.bb_lower, 4)
        if self.breakout_type == BreakoutType.PDR:
            d["prev_day_high"] = round(self.prev_day_high, 4)
            d["prev_day_low"] = round(self.prev_day_low, 4)
            d["prev_day_range"] = round(self.prev_day_range, 4)
        if self.breakout_type == BreakoutType.IB:
            d["ib_high"] = round(self.ib_high, 4)
            d["ib_low"] = round(self.ib_low, 4)
            d["ib_complete"] = self.ib_complete
        if self.extra:
            d["extra"] = self.extra
        return d


# ===========================================================================
# Internal helpers
# ===========================================================================


def _compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR on a DataFrame with High / Low / Close columns.

    Returns 0.0 if there is insufficient data.
    """
    n = len(bars)
    if n < 2:
        return 0.0

    highs = bars["High"].astype(float).values
    lows = bars["Low"].astype(float).values
    closes = bars["Close"].astype(float).values

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            float(highs[i] - lows[i]),
            abs(float(highs[i] - closes[i - 1])),
            abs(float(lows[i] - closes[i - 1])),
        )

    if n < period + 1:
        return float(np.mean(tr))

    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period, n):
        atr = alpha * tr[i] + (1.0 - alpha) * atr
    return atr


def _localize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Ensure the bar index is a tz-aware DatetimeIndex in ET.

    Works whether the index is UTC, naive, or already ET.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        return bars

    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(_ET)
    return bars.set_index(idx)


def _check_bar_quality(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    level: float,
    direction: str,
    atr: float,
    config: RangeConfig,
) -> tuple[bool, bool, float, float]:
    """Check depth and body-ratio quality gates for a candidate breakout bar.

    Returns:
        (depth_ok, body_ok, depth_value, body_ratio)
    """
    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return False, False, 0.0, 0.0

    body = abs(bar_close - bar_open)
    body_ratio = body / bar_range

    depth = bar_close - level if direction == "LONG" else level - bar_close

    min_depth = config.min_depth_atr_pct * atr if atr > 0 else 0.0
    depth_ok = depth >= min_depth
    body_ok = body_ratio >= config.min_body_ratio

    return depth_ok, body_ok, max(depth, 0.0), body_ratio


# ===========================================================================
# Range builders  (one per BreakoutType)
# ===========================================================================


def _build_orb_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    session_start: dt_time,
    session_end: dt_time,
) -> tuple[float, float, int, bool]:
    """Extract the opening-range high/low from bars within the OR window.

    Returns:
        (or_high, or_low, bar_count, complete)
    ``complete`` is True once the current bar timestamp is past session_end.
    """
    bars_et = _localize_bars(bars)
    now_et = bars_et.index[-1].to_pydatetime().time() if len(bars_et) > 0 else dt_time(0, 0)

    idx_time = pd.DatetimeIndex(bars_et.index).time
    mask = (idx_time >= session_start) & (idx_time < session_end)
    or_bars = bars_et.loc[mask]

    if len(or_bars) < config.min_bars:
        return 0.0, 0.0, len(or_bars), False

    or_high = float(or_bars["High"].max())
    or_low = float(or_bars["Low"].min())
    complete = now_et >= session_end

    return or_high, or_low, len(or_bars), complete


def _build_pdr_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool, float, float, float]:
    """Identify the previous Globex day's high/low from intraday 1m bars.

    Strategy: split bars at config.pdr_session_start (18:00 ET by default).
    Everything before today's session start is "previous day".

    Returns:
        (pdr_high, pdr_low, bar_count, complete, prev_high, prev_low, prev_range)
    ``complete`` is always True for PDR (the range is already formed).
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0

    # Identify the current Globex-day start: the most recent bar whose time
    # equals or is just after pdr_session_start.
    session_start = config.pdr_session_start  # 18:00 ET

    # Find the latest 18:00 ET boundary within the bar history
    idx_time_pdr = pd.DatetimeIndex(bars_et.index).time
    today_session_starts = bars_et.index[idx_time_pdr == session_start]

    if len(today_session_starts) == 0:
        # Fallback: use calendar midnight boundary
        today_et = bars_et.index[-1].to_pydatetime().date()
        cutoff = pd.Timestamp(today_et, tz=_ET)
        prev_bars = bars_et[bars_et.index < cutoff]
        bars_et[bars_et.index >= cutoff]
    else:
        # The most recent session-start boundary
        latest_start = today_session_starts[-1]
        prev_bars = bars_et[bars_et.index < latest_start]
        bars_et[bars_et.index >= latest_start]

    if len(prev_bars) < config.min_bars:
        # Fall back to daily bars if available: use the penultimate day's H/L
        # (caller may pass daily bars via bars_daily kwarg in the outer layer)
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    prev_high = float(prev_bars["High"].max())  # type: ignore[arg-type]
    prev_low = float(prev_bars["Low"].min())  # type: ignore[arg-type]
    prev_range = prev_high - prev_low
    bar_count = len(prev_bars)

    return prev_high, prev_low, bar_count, True, prev_high, prev_low, prev_range


def _build_ib_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool]:
    """Build the Initial Balance range (first ``ib_duration_minutes`` of RTH).

    Returns:
        (ib_high, ib_low, bar_count, complete)
    ``complete`` is True once the current time is past the IB end time.
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 1:
        return 0.0, 0.0, 0, False

    ib_start = config.ib_start_time
    ib_end_minutes = ib_start.hour * 60 + ib_start.minute + config.ib_duration_minutes
    ib_end_h, ib_end_m = divmod(ib_end_minutes, 60)
    ib_end = dt_time(int(ib_end_h), int(ib_end_m))

    idx_time_ib = pd.DatetimeIndex(bars_et.index).time
    mask = (idx_time_ib >= ib_start) & (idx_time_ib < ib_end)
    ib_bars = bars_et.loc[mask]

    bar_count = len(ib_bars)
    now_et = bars_et.index[-1].to_pydatetime().time()
    complete = now_et >= ib_end

    if bar_count < config.min_bars:
        return 0.0, 0.0, bar_count, complete

    ib_high = float(ib_bars["High"].max())  # type: ignore[arg-type]
    ib_low = float(ib_bars["Low"].min())  # type: ignore[arg-type]

    return ib_high, ib_low, bar_count, complete


def _build_consolidation_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    atr: float,
) -> tuple[float, float, int, bool, float, float, float, int, float]:
    """Detect a Bollinger Band / ATR consolidation squeeze and extract its range.

    A "squeeze" is present when the BB bandwidth (upper − lower) is smaller
    than ``squeeze_atr_mult × ATR`` for at least ``squeeze_min_bars``
    consecutive bars.

    Returns:
        (cons_high, cons_low, bar_count, squeeze_detected,
         bb_upper, bb_lower, bb_width, squeeze_bar_count, current_bb_width)
    """
    if len(bars) < config.squeeze_bb_period + 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0, 0.0

    close = bars["Close"].astype(float)
    n = config.squeeze_bb_period
    std_mult = config.squeeze_bb_std

    # Rolling Bollinger Bands
    bb_mid = close.rolling(n).mean()
    bb_std = close.rolling(n).std(ddof=0)
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    bb_width = (bb_upper - bb_lower).fillna(0.0)

    threshold = config.squeeze_atr_mult * atr

    # Count consecutive squeeze bars at the end of the series
    squeeze_flags = bb_width < threshold
    squeeze_bar_count = 0
    for i in range(len(squeeze_flags) - 1, -1, -1):
        if squeeze_flags.iloc[i]:
            squeeze_bar_count += 1
        else:
            break

    squeeze_detected = squeeze_bar_count >= config.squeeze_min_bars
    current_bb_width = float(bb_width.iloc[-1]) if len(bb_width) > 0 else 0.0
    current_bb_upper = float(bb_upper.iloc[-1]) if len(bb_upper) > 0 else 0.0
    current_bb_lower = float(bb_lower.iloc[-1]) if len(bb_lower) > 0 else 0.0

    if not squeeze_detected:
        return 0.0, 0.0, 0, False, current_bb_upper, current_bb_lower, current_bb_width, 0, current_bb_width

    # The consolidation range is the BB upper/lower at the squeeze boundary
    squeeze_start_idx = len(bars) - squeeze_bar_count
    squeeze_slice = bars.iloc[squeeze_start_idx:]
    cons_high = float(squeeze_slice["High"].max())  # type: ignore[arg-type]
    cons_low = float(squeeze_slice["Low"].min())  # type: ignore[arg-type]
    bar_count = len(squeeze_slice)

    return (
        cons_high,
        cons_low,
        bar_count,
        True,
        current_bb_upper,
        current_bb_lower,
        current_bb_width,
        squeeze_bar_count,
        current_bb_width,
    )


# ===========================================================================
# Breakout scanner (shared logic for all types)
# ===========================================================================


def _scan_for_breakout(
    bars: pd.DataFrame,
    range_high: float,
    range_low: float,
    atr: float,
    config: RangeConfig,
    scan_start_time: dt_time | None = None,
) -> tuple[bool, str, float, str, float, float]:
    """Scan bars after range formation for a breakout close beyond H/L.

    Only bars whose ET wall-clock time is >= scan_start_time (if supplied)
    are considered.

    Returns:
        (detected, direction, trigger_price, bar_time_str, depth, body_ratio)
    """
    if range_high <= 0 or range_low <= 0 or range_high <= range_low:
        return False, "", 0.0, "", 0.0, 0.0

    threshold = atr * config.atr_multiplier
    long_trigger = range_high + threshold
    short_trigger = range_low - threshold

    bars_et = _localize_bars(bars)

    for ts, row in bars_et.iterrows():
        bar_time = (
            ts.to_pydatetime().time()
            if hasattr(ts, "to_pydatetime")
            else (ts.time() if hasattr(ts, "time") else dt_time(0, 0))
        )
        if scan_start_time is not None and bar_time < scan_start_time:
            continue

        bar_open = float(row["Open"] if "Open" in row.index else row["Close"] if "Close" in row.index else 0.0)
        bar_high = float(row["High"] if "High" in row.index else 0.0)
        bar_low = float(row["Low"] if "Low" in row.index else 0.0)
        bar_close = float(row["Close"] if "Close" in row.index else 0.0)

        if bar_close <= 0:
            continue

        direction = ""
        level = 0.0

        if bar_close > long_trigger:
            direction = "LONG"
            level = long_trigger
        elif bar_close < short_trigger:
            direction = "SHORT"
            level = short_trigger

        if direction:
            depth_ok, body_ok, depth, body_ratio = _check_bar_quality(
                bar_open,
                bar_high,
                bar_low,
                bar_close,
                level,
                direction,
                atr,
                config,
            )
            if depth_ok and body_ok:
                bar_time_str = str(ts.isoformat()) if hasattr(ts, "isoformat") else str(ts)
                return True, direction, bar_close, bar_time_str, depth, body_ratio

    return False, "", 0.0, "", 0.0, 0.0


# ===========================================================================
# Main public function
# ===========================================================================


def detect_range_breakout(
    bars: pd.DataFrame,
    symbol: str,
    config: RangeConfig | None = None,
    *,
    # ORB-specific overrides
    orb_session_start: dt_time | None = None,
    orb_session_end: dt_time | None = None,
    orb_scan_start: dt_time | None = None,
    # PDR override: supply pre-computed daily high/low
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    # IB override: supply pre-computed IB high/low
    ib_high: float | None = None,
    ib_low: float | None = None,
) -> BreakoutResult:
    """Unified breakout detector for ORB, PDR, IB, and Consolidation types.

    Args:
        bars: 1-minute OHLCV DataFrame (tz-aware or UTC).  Must have columns
              ``Open``, ``High``, ``Low``, ``Close`` and a DatetimeIndex.
        symbol: Instrument symbol (for logging / result annotation).
        config: ``RangeConfig`` describing the breakout type and thresholds.
                Defaults to ``DEFAULT_CONFIGS[BreakoutType.ORB]`` if omitted.
        orb_session_start: ET time for ORB range start (ORB type only).
        orb_session_end: ET time for ORB range end (ORB type only).
        orb_scan_start: ET time after which breakout scanning begins
                        (defaults to orb_session_end).
        prev_day_high: Override PDR high (skips internal range building).
        prev_day_low: Override PDR low.
        ib_high: Override IB high (skips internal range building).
        ib_low: Override IB low.

    Returns:
        ``BreakoutResult`` populated with range, quality gate verdicts,
        and breakout detection state.  Never raises — errors surface in
        ``result.error``.
    """
    from datetime import datetime as _dt

    if config is None:
        config = DEFAULT_CONFIGS[BreakoutType.ORB]

    now_str = _dt.now(tz=_ET).isoformat()
    result = BreakoutResult(
        symbol=symbol,
        breakout_type=config.breakout_type,
        label=config.label or config.breakout_type.value,
        evaluated_at=now_str,
    )

    if bars is None or bars.empty:
        result.error = "No bar data supplied"
        return result

    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(bars.columns):
        result.error = f"Missing columns: {required_cols - set(bars.columns)}"
        return result

    # --- ATR ---
    atr = _compute_atr(bars, period=config.atr_period)
    result.atr_value = round(atr, 6)

    # --- Build range (type-specific) ---
    try:
        btype = config.breakout_type

        if btype == BreakoutType.ORB:
            s_start = orb_session_start or dt_time(9, 30)
            s_end = orb_session_end or dt_time(10, 0)
            scan_start = orb_scan_start or s_end

            r_high, r_low, bar_count, complete = _build_orb_range(bars, config, s_start, s_end)
            result.range_complete = complete
            result.range_bar_count = bar_count

        elif btype == BreakoutType.PDR:
            if prev_day_high is not None and prev_day_low is not None:
                r_high, r_low = prev_day_high, prev_day_low
                bar_count = 1
                complete = True
                result.prev_day_high = r_high
                result.prev_day_low = r_low
                result.prev_day_range = r_high - r_low
            else:
                (r_high, r_low, bar_count, complete, pdr_high, pdr_low, pdr_range) = _build_pdr_range(bars, config)
                result.prev_day_high = pdr_high
                result.prev_day_low = pdr_low
                result.prev_day_range = pdr_range

            result.range_complete = complete
            result.range_bar_count = bar_count
            scan_start = None  # PDR: always scan latest bars

        elif btype == BreakoutType.IB:
            if ib_high is not None and ib_low is not None:
                r_high, r_low = ib_high, ib_low
                bar_count = 0
                complete = True
            else:
                r_high, r_low, bar_count, complete = _build_ib_range(bars, config)

            result.ib_high = r_high
            result.ib_low = r_low
            result.ib_complete = complete
            result.range_complete = complete
            result.range_bar_count = bar_count

            # IB breakout scan starts after IB window closes
            ib_end_min = config.ib_start_time.hour * 60 + config.ib_start_time.minute + config.ib_duration_minutes
            ib_end_h, ib_end_m = divmod(int(ib_end_min), 60)
            scan_start = dt_time(ib_end_h, ib_end_m)

        elif btype == BreakoutType.CONS:
            (
                r_high,
                r_low,
                bar_count,
                squeeze_detected,
                bb_upper,
                bb_lower,
                bb_width,
                squeeze_bar_count,
                current_bb_width,
            ) = _build_consolidation_range(bars, config, atr)
            result.squeeze_detected = squeeze_detected
            result.squeeze_bar_count = squeeze_bar_count
            result.squeeze_bb_width = round(current_bb_width, 4)
            result.bb_upper = round(bb_upper, 4)
            result.bb_lower = round(bb_lower, 4)
            result.range_complete = squeeze_detected
            result.range_bar_count = bar_count
            scan_start = None  # scan the very latest bar for the expansion

            if not squeeze_detected:
                result.error = "No squeeze detected — cannot form consolidation range"
                return result

        else:
            result.error = f"Unknown BreakoutType: {btype}"
            return result

    except Exception as exc:
        result.error = f"Range build error: {exc}"
        logger.warning("detect_range_breakout[%s] range build error for %s: %s", btype, symbol, exc)
        return result

    # --- Populate common range fields ---
    result.range_high = round(r_high, 4)
    result.range_low = round(r_low, 4)
    result.range_size = round(r_high - r_low, 4)

    threshold = atr * config.atr_multiplier
    result.breakout_threshold = round(threshold, 4)
    result.long_trigger = round(r_high + threshold, 4)
    result.short_trigger = round(r_low - threshold, 4)

    # --- Range size quality gate ---
    if atr > 0:
        range_atr_ratio = result.range_size / atr
        size_ok = config.min_range_atr_ratio <= range_atr_ratio <= config.max_range_atr_ratio
    else:
        size_ok = result.range_size > 0

    result.range_size_ok = size_ok

    if not size_ok:
        logger.debug(
            "detect_range_breakout[%s] %s: range size %.4f / ATR %.4f = %.2f out of [%.2f, %.2f]",
            btype,
            symbol,
            result.range_size,
            atr,
            result.range_size / atr if atr > 0 else 0,
            config.min_range_atr_ratio,
            config.max_range_atr_ratio,
        )
        return result  # breakout_detected stays False

    if r_high <= 0 or r_low <= 0 or r_high <= r_low:
        return result

    # --- Scan for breakout ---
    try:
        detected, direction, trigger, bar_time_str, depth, body_ratio = _scan_for_breakout(
            bars, r_high, r_low, atr, config, scan_start_time=scan_start
        )
    except Exception as exc:
        result.error = f"Breakout scan error: {exc}"
        logger.warning("detect_range_breakout[%s] scan error for %s: %s", btype, symbol, exc)
        return result

    result.breakout_detected = detected
    result.direction = direction
    result.trigger_price = round(trigger, 4)
    result.breakout_bar_time = bar_time_str
    result.breakout_bar_depth = round(depth, 6)
    result.breakout_bar_body_ratio = round(body_ratio, 4)

    if detected:
        # Recompute quality gate verdicts now that we have bar data
        result.depth_ok = depth >= (config.min_depth_atr_pct * atr if atr > 0 else 0)
        result.body_ratio_ok = body_ratio >= config.min_body_ratio

        logger.info(
            "🔔 %s BREAKOUT [%s]: %s %s @ %.4f (range %.4f–%.4f, ATR %.4f)",
            config.breakout_type.value,
            config.label or btype,
            direction,
            symbol,
            trigger,
            r_low,
            r_high,
            atr,
        )
    else:
        logger.debug(
            "detect_range_breakout[%s] %s: no breakout (range %.4f–%.4f)",
            btype,
            symbol,
            r_low,
            r_high,
        )

    return result


# ===========================================================================
# Convenience helpers for the engine handler layer
# ===========================================================================


def detect_pdr_breakout(
    bars: pd.DataFrame,
    symbol: str,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect a Previous Day Range breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.PDR]
    return detect_range_breakout(
        bars,
        symbol,
        cfg,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
    )


def detect_ib_breakout(
    bars: pd.DataFrame,
    symbol: str,
    ib_high: float | None = None,
    ib_low: float | None = None,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect an Initial Balance breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.IB]
    return detect_range_breakout(
        bars,
        symbol,
        cfg,
        ib_high=ib_high,
        ib_low=ib_low,
    )


def detect_consolidation_breakout(
    bars: pd.DataFrame,
    symbol: str,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect a Consolidation/Squeeze breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.CONS]
    return detect_range_breakout(bars, symbol, cfg)


def detect_all_breakout_types(
    bars: pd.DataFrame,
    symbol: str,
    types: list[BreakoutType] | None = None,
    configs: dict[BreakoutType, RangeConfig] | None = None,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: dt_time | None = None,
    orb_session_end: dt_time | None = None,
) -> dict[BreakoutType, BreakoutResult]:
    """Run all (or a subset of) breakout type detectors for a single symbol.

    Args:
        bars: 1-minute OHLCV DataFrame.
        symbol: Instrument symbol.
        types: List of ``BreakoutType`` values to check.  Defaults to all four.
        configs: Override configs per type.  Falls back to ``DEFAULT_CONFIGS``.
        prev_day_high: Pre-computed PDR high (optional).
        prev_day_low: Pre-computed PDR low (optional).
        ib_high: Pre-computed IB high (optional).
        ib_low: Pre-computed IB low (optional).
        orb_session_start: ET time for ORB session start.
        orb_session_end: ET time for ORB session end.

    Returns:
        Dict mapping each ``BreakoutType`` to its ``BreakoutResult``.
    """
    if types is None:
        types = list(BreakoutType)
    merged_configs = {**DEFAULT_CONFIGS, **(configs or {})}

    results: dict[BreakoutType, BreakoutResult] = {}

    for btype in types:
        cfg = merged_configs.get(btype, DEFAULT_CONFIGS.get(btype, RangeConfig(breakout_type=btype)))
        try:
            result = detect_range_breakout(
                bars,
                symbol,
                cfg,
                orb_session_start=orb_session_start,
                orb_session_end=orb_session_end,
                prev_day_high=prev_day_high,
                prev_day_low=prev_day_low,
                ib_high=ib_high,
                ib_low=ib_low,
            )
        except Exception as exc:
            logger.warning("detect_all_breakout_types[%s] error for %s: %s", btype, symbol, exc)
            result = BreakoutResult(
                symbol=symbol,
                breakout_type=btype,
                label=cfg.label,
                error=str(exc),
                evaluated_at=datetime.now(tz=_ET).isoformat(),
            )
        results[btype] = result

    return results
