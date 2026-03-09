"""
Breakout CNN — Hybrid EfficientNetV2 + Tabular Model for ORB Pattern Recognition
==================================================================================
Classifies Ruby-style chart snapshots as "good breakout" (high-probability
follow-through) or "bad breakout" (likely to fail / chop) using a hybrid
architecture that combines:

  1. **EfficientNetV2-S** (image backbone) — pre-trained on ImageNet, fine-tuned
     on Ruby-rendered candlestick charts.  Extracts 1280-dim visual features from
     the exact same chart images a human trader sees (ORB box, VWAP, EMA9,
     quality badge, volume panel).

  2. **Tabular head** — a small feed-forward network that ingests numeric features
     the image cannot easily capture (quality %, volume ratio, ATR %, CVD delta,
     NR7 flag, direction bias).

  3. **Classifier** — merges the two feature vectors and outputs a probability
     of "clean breakout" (0.0–1.0).

Dependencies:
  - torch >= 2.0
  - torchvision >= 0.15
  - Pillow
  - pandas, numpy (already in project)

All torch imports are guarded so the module can be imported on machines
without CUDA (it will log a warning and the functions will return None /
raise informative errors).

Public API:
    from lib.analysis.breakout_cnn import (
        HybridBreakoutCNN,
        BreakoutDataset,
        train_model,
        predict_breakout,
        predict_breakout_batch,
        get_device,
        DEFAULT_THRESHOLD,
        TABULAR_FEATURES,
    )

Design:
  - Training produces a ``.pt`` state-dict file under ``models/``.
  - Inference loads the latest model automatically (or a specific path).
  - Thread-safe: model loading uses a module-level lock.
  - Graceful degradation: if torch is missing, all functions return None
    with a logged warning — the rest of the engine keeps running.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    # Always visible to the type checker — these are never actually
    # executed at runtime when TYPE_CHECKING is True.
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.breakout_cnn")

# ---------------------------------------------------------------------------
# Guard torch imports — allow the module to be imported without GPU/torch
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore[no-redef]
    import torch.nn as nn  # type: ignore[no-redef]
    import torchvision.models as models  # type: ignore[no-redef]
    import torchvision.transforms as T  # type: ignore[no-redef]
    from PIL import Image  # type: ignore[no-redef]
    from torch.utils.data import DataLoader, Dataset  # type: ignore[no-redef]

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch / torchvision not installed — CNN features disabled.  "
        "Install with: pip install torch torchvision Pillow"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of tabular feature names expected by the model.
# The dataset and inference code must provide them in exactly this order.
#
# v7 contract (24 features) — extends v6's 18 features with 6 new slots
# from the Daily Strategy layer and crypto momentum analysis.
# Aligns with NinjaTrader BreakoutStrategy PrepareCnnTabular() and
# OrbCnnPredictor.NormaliseTabular() in C#.
# This is the canonical contract for Python training and NT8 inference.
#
# Index  Feature                  Raw source / notes
# ─────  ───────────────────────  ──────────────────────────────────────────
#  [0]   quality_pct_norm        quality_pct / 100  →  [0, 1]
#  [1]   volume_ratio            breakout bar vol / rolling avg vol
#  [2]   atr_pct                 ATR / close price  (fraction, not %)
#  [3]   cvd_delta               cumulative signed vol / total vol  [-1, 1]
#  [4]   nr7_flag                1 if session range is narrowest of 7
#  [5]   direction_flag          1 = LONG, 0 = SHORT
#  [6]   session_ordinal         Globex day cycle ordinal  [0, 1]
#  [7]   london_overlap_flag     1 if bar hour 08:00–09:00 ET
#  [8]   or_range_atr_ratio      ORB range / ATR  (raw, normalised later)
#  [9]   premarket_range_ratio   premarket range / ORB range  (raw)
#  [10]  bar_of_day              minutes since Globex open / 1380  [0, 1]
#  [11]  day_of_week             weekday Mon=0..Fri=4  / 4  →  [0, 1]
#  [12]  vwap_distance           (price − vwap) / ATR  (raw, normalised later)
#  [13]  asset_class_id          asset class ordinal / 4  →  [0, 1]
#  ── v6 additions (slots 14–17) ──────────────────────────────────────────
#  [14]  breakout_type_ord       BreakoutType.value / 12  →  [0, 1]
#  [15]  asset_volatility_class  low=0.0 / med=0.5 / high=1.0
#  [16]  hour_of_day             ET hour / 23  →  [0, 1]
#  [17]  tp3_atr_mult_norm       TP3 ATR multiplier / 5.0  →  [0, 1]
#  ── v7 additions (slots 18–23) — Daily Strategy layer features ──────────
#  [18]  daily_bias_direction    from bias_analyzer: -1→0, 0→0.5, +1→1.0
#  [19]  daily_bias_confidence   0.0–1.0 scalar from bias analyzer
#  [20]  prior_day_pattern       ordinal of yesterday's candle pattern / 9
#  [21]  weekly_range_position   price position in prior week H/L  [0, 1]
#  [22]  monthly_trend_score     20-day EMA slope [-1,+1] → [0, 1]
#  [23]  crypto_momentum_score   from crypto_momentum: [-1,+1] → [0, 1]
#  ── v7.1 additions (slots 24–27) — Phase 4B sub-feature decomposition ──
#  [24]  breakout_type_category  time-based=0, range-based=0.5, squeeze=1.0
#  [25]  session_overlap_flag    1.0 if London+NY overlap window, else 0.0
#  [26]  atr_trend               ATR expanding=1.0, contracting=0.0 (10-bar)
#  [27]  volume_trend            5-bar volume slope, normalised [0, 1]
TABULAR_FEATURES: list[str] = [
    "quality_pct_norm",  # [0]  quality / 100
    "volume_ratio",  # [1]  breakout bar vol / avg vol
    "atr_pct",  # [2]  ATR / price
    "cvd_delta",  # [3]  CVD signed ratio [-1, 1]
    "nr7_flag",  # [4]  1 if NR7 session
    "direction_flag",  # [5]  1=LONG 0=SHORT
    "session_ordinal",  # [6]  Globex day position [0, 1]
    "london_overlap_flag",  # [7]  1 if 08:00–09:00 ET overlap
    "or_range_atr_ratio",  # [8]  ORB range / ATR
    "premarket_range_ratio",  # [9]  premarket range / ORB range
    "bar_of_day",  # [10] minutes-since-open / 1380
    "day_of_week",  # [11] Mon=0..Fri=4  / 4
    "vwap_distance",  # [12] (price-vwap) / ATR
    "asset_class_id",  # [13] asset class ordinal / 4
    # ── v6 additions ─────────────────────────────────────────────────────
    "breakout_type_ord",  # [14] BreakoutType ordinal / 12
    "asset_volatility_class",  # [15] low=0 / med=0.5 / high=1
    "hour_of_day",  # [16] ET hour / 23
    "tp3_atr_mult_norm",  # [17] TP3 multiplier / 5
    # ── v7 additions — Daily Strategy layer ──────────────────────────────
    "daily_bias_direction",  # [18] -1→0, 0→0.5, +1→1.0
    "daily_bias_confidence",  # [19] 0.0–1.0 from bias analyzer
    "prior_day_pattern",  # [20] candle pattern ordinal / 9
    "weekly_range_position",  # [21] price in week range [0, 1]
    "monthly_trend_score",  # [22] EMA slope [-1,+1] → [0, 1]
    "crypto_momentum_score",  # [23] crypto lead [-1,+1] → [0, 1]
    # ── v7.1 additions — Phase 4B sub-feature decomposition ──────────────
    "breakout_type_category",  # [24] time=0, range=0.5, squeeze=1.0
    "session_overlap_flag",  # [25] 1.0 if London+NY overlap
    "atr_trend",  # [26] expanding=1.0, contracting=0.0
    "volume_trend",  # [27] 5-bar vol slope [0, 1]
]

NUM_TABULAR = len(TABULAR_FEATURES)

# Feature contract version — must match NinjaTrader BreakoutStrategy.
# v7.1 adds 4 sub-features (Phase 4B): breakout_type_category,
# session_overlap_flag, atr_trend, volume_trend.
FEATURE_CONTRACT_VERSION = 7

# Asset class ordinal map — mirrors GetAssetClassNorm() in BreakoutStrategy.cs.
# 0=equity_index, 1=fx, 2=metals_energy, 3=treasuries_ags, 4=crypto
# Normalised as ordinal / 4.0 so the value sits in [0, 1].
ASSET_CLASS_ORDINALS: dict[str, float] = {
    # Equity index micros
    "MES": 0.0 / 4,
    "MNQ": 0.0 / 4,
    "M2K": 0.0 / 4,
    "MYM": 0.0 / 4,
    # FX
    "6E": 1.0 / 4,
    "6B": 1.0 / 4,
    "6J": 1.0 / 4,
    "6A": 1.0 / 4,
    "6C": 1.0 / 4,
    "6S": 1.0 / 4,
    "M6E": 1.0 / 4,
    "M6B": 1.0 / 4,
    # Metals / Energy
    "MGC": 2.0 / 4,
    "SIL": 2.0 / 4,
    "MHG": 2.0 / 4,
    "MCL": 2.0 / 4,
    "MNG": 2.0 / 4,
    # Treasuries / Ags
    "ZN": 3.0 / 4,
    "ZB": 3.0 / 4,
    "ZC": 3.0 / 4,
    "ZS": 3.0 / 4,
    "ZW": 3.0 / 4,
    # Crypto (CME futures)
    "MBT": 4.0 / 4,
    "MET": 4.0 / 4,
    # Crypto (short aliases — used in training symbol lists)
    "BTC": 4.0 / 4,
    "ETH": 4.0 / 4,
    "SOL": 4.0 / 4,
    "LINK": 4.0 / 4,
    "AVAX": 4.0 / 4,
    "DOT": 4.0 / 4,
    "ADA": 4.0 / 4,
    "MATIC": 4.0 / 4,
    "XRP": 4.0 / 4,
    # Crypto (Kraken internal tickers — KRAKEN: prefix)
    "KRAKEN:XBTUSD": 4.0 / 4,
    "KRAKEN:ETHUSD": 4.0 / 4,
    "KRAKEN:SOLUSD": 4.0 / 4,
    "KRAKEN:LINKUSD": 4.0 / 4,
    "KRAKEN:AVAXUSD": 4.0 / 4,
    "KRAKEN:DOTUSD": 4.0 / 4,
    "KRAKEN:ADAUSD": 4.0 / 4,
    "KRAKEN:MATICUSD": 4.0 / 4,
    "KRAKEN:XRPUSD": 4.0 / 4,
}

# Keep BREAKOUT_TYPE_ORDINALS for generate_feature_contract / model_info consumers
# that still reference it by name.  These are not used in the v4 tabular vector.
BREAKOUT_TYPE_ORDINALS: dict[str, float] = {
    "ORB": 0.0 / 12,
    "PDR": 1.0 / 12,
    "IB": 2.0 / 12,
    "CONS": 3.0 / 12,
    "WEEKLY": 4.0 / 12,
    "MONTHLY": 5.0 / 12,
    "ASIAN": 6.0 / 12,
    "BBSQUEEZE": 7.0 / 12,
    "VA": 8.0 / 12,
    "INSIDE": 9.0 / 12,
    "GAP": 10.0 / 12,
    "PIVOT": 11.0 / 12,
    "FIB": 12.0 / 12,
}

# Keep ASSET_VOLATILITY_CLASS for any callers that still reference it by name.
# Not used in the v4 tabular vector (replaced by asset_class_id).
ASSET_VOLATILITY_CLASS: dict[str, float] = {
    "ZN": 0.0,
    "M6B": 0.0,
    "M6E": 0.0,
    "MES": 0.5,
    "MYM": 0.5,
    "MGC": 0.5,
    "MCL": 0.5,
    "SIL": 0.5,
    "M2K": 0.5,
    "MNQ": 1.0,
    "MBT": 1.0,
    "MHG": 1.0,
    # Crypto short aliases
    "BTC": 1.0,
    "ETH": 1.0,
    "SOL": 1.0,
    "LINK": 1.0,
    "AVAX": 1.0,
    "DOT": 1.0,
    "ADA": 1.0,
    "MATIC": 1.0,
    "XRP": 1.0,
    # Crypto Kraken internal tickers
    "KRAKEN:XBTUSD": 1.0,
    "KRAKEN:ETHUSD": 1.0,
    "KRAKEN:SOLUSD": 1.0,
    "KRAKEN:LINKUSD": 1.0,
    "KRAKEN:AVAXUSD": 1.0,
    "KRAKEN:DOTUSD": 1.0,
    "KRAKEN:ADAUSD": 1.0,
    "KRAKEN:MATICUSD": 1.0,
    "KRAKEN:XRPUSD": 1.0,
}


def get_breakout_type_ordinal(breakout_type: str) -> float:
    """Return the normalised ordinal [0, 1] for a BreakoutType string.

    Accepts both upper-case ("ORB", "PDR") and lower-case / mixed variants.
    Falls back to 0.0 (ORB) for unknown types.
    """
    return BREAKOUT_TYPE_ORDINALS.get(str(breakout_type).upper().strip(), 0.0)


# ---------------------------------------------------------------------------
# v7.1 feature helpers — Phase 4B sub-feature decomposition
# ---------------------------------------------------------------------------

# Breakout type → category mapping.
# Categories capture *how* the range is formed:
#   time-based (0.0):    range defined by a fixed time window
#   range-based (0.5):   range defined by prior price levels / structures
#   squeeze-based (1.0): range detected algorithmically via compression
_BREAKOUT_TYPE_CATEGORY: dict[str, float] = {
    # Time-based — fixed session/time windows define the range
    "ORB": 0.0,
    "ASIAN": 0.0,
    "IB": 0.0,
    # Range-based — prior price levels / structures define the range
    "PDR": 0.5,
    "WEEKLY": 0.5,
    "MONTHLY": 0.5,
    "VA": 0.5,
    "INSIDE": 0.5,
    "GAP": 0.5,
    "PIVOT": 0.5,
    "FIB": 0.5,
    # Squeeze-based — algorithmically detected compression
    "CONS": 1.0,
    "BBSQUEEZE": 1.0,
}


def get_breakout_type_category(breakout_type: str) -> float:
    """Return the breakout type category for the CNN tabular vector.

    Categories decompose breakout_type_ord into a coarser grouping that
    captures *how* the range is formed:
      - time-based (0.0):    ORB, Asian, IB — fixed time window
      - range-based (0.5):   PDR, Weekly, Monthly, VA, Inside, Gap, Pivot, Fib
      - squeeze-based (1.0): Consolidation, BollingerSqueeze

    This helps the CNN learn that squeeze breakouts share characteristics
    (explosive expansion) regardless of the specific detection algorithm,
    while time-based breakouts share session-dependency characteristics.

    Args:
        breakout_type: Short name (``"ORB"``, ``"PDR"``, ``"CONS"``, …)
                       or canonical name (``"PrevDay"``, ``"Consolidation"``, …).

    Returns:
        Float in {0.0, 0.5, 1.0}.  Falls back to 0.5 (range-based) for
        unknown types.
    """
    upper = str(breakout_type).upper().strip()
    if upper in _BREAKOUT_TYPE_CATEGORY:
        return _BREAKOUT_TYPE_CATEGORY[upper]
    # Try canonical name → short name mapping
    _CANONICAL_TO_SHORT = {
        "PREVDAY": "PDR",
        "INITIALBALANCE": "IB",
        "CONSOLIDATION": "CONS",
        "BOLLINGERSQUEEZE": "BBSQUEEZE",
        "VALUEAREA": "VA",
        "INSIDEDAY": "INSIDE",
        "GAPREJECTION": "GAP",
        "PIVOTPOINTS": "PIVOT",
        "FIBONACCI": "FIB",
    }
    short = _CANONICAL_TO_SHORT.get(upper, upper)
    return _BREAKOUT_TYPE_CATEGORY.get(short, 0.5)


def get_session_overlap_flag(
    session_key: str | None = None,
    *,
    bar_hour_et: int | None = None,
) -> float:
    """Return 1.0 if the bar is in the London+NY overlap window, else 0.0.

    The London-NY overlap (roughly 08:00–12:00 ET / 13:00–17:00 UTC) is the
    highest-volume window of the trading day.  Breakouts during this window
    have significantly higher follow-through rates because of the liquidity
    depth from both European and American institutional flow.

    This sub-feature enriches ``session_ordinal`` by explicitly flagging the
    overlap, which the model can use as a multiplicative signal quality boost.

    Detection logic (any of these → 1.0):
      - ``session_key == "london_ny"``
      - ``bar_hour_et`` is in [8, 9, 10, 11] (08:00–11:59 ET)

    Args:
        session_key: ORBSession.key (e.g. ``"london_ny"``).
        bar_hour_et: Hour of the bar in US/Eastern (0–23).

    Returns:
        1.0 if in the overlap window, 0.0 otherwise.
    """
    if session_key is not None and session_key.lower().strip() == "london_ny":
        return 1.0
    if bar_hour_et is not None and 8 <= bar_hour_et <= 11:
        return 1.0
    return 0.0


def get_atr_trend(
    bars: "pd.DataFrame | None" = None,
    *,
    lookback: int = 10,
) -> float:
    """Return whether ATR is expanding (1.0) or contracting (0.0) over the last N bars.

    Computes the per-bar true range for the most recent ``lookback`` bars,
    then compares the mean of the second half to the first half.  If the
    recent half has higher average TR, ATR is expanding (→ 1.0); if lower,
    it's contracting (→ 0.0).  Equal → 0.5.

    This sub-feature enriches ``atr_pct`` (which is a point-in-time snapshot)
    by adding the *direction* of volatility change.  Expanding ATR into a
    breakout is bullish for continuation; contracting ATR suggests the move
    may be exhausting.

    Args:
        bars:     1-minute OHLCV DataFrame (needs at least ``lookback`` rows).
        lookback: Number of bars to measure the TR trend over (default 10).

    Returns:
        Float in [0.0, 1.0].  1.0 = expanding, 0.0 = contracting, 0.5 = flat.
        Falls back to 0.5 if insufficient data.
    """
    if bars is None or len(bars) < lookback + 1:
        return 0.5

    try:
        tail = bars.iloc[-(lookback + 1) :]
        highs = tail["High"].astype(float).values
        lows = tail["Low"].astype(float).values
        closes = tail["Close"].astype(float).values

        # Compute per-bar true range (skip first bar — no prior close)
        tr = np.empty(lookback)
        for i in range(lookback):
            j = i + 1  # offset into tail (skip row 0 which is the "prior close" anchor)
            tr[i] = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j - 1]),
                abs(lows[j] - closes[j - 1]),
            )

        half = lookback // 2
        first_half_mean = float(np.mean(tr[:half]))
        second_half_mean = float(np.mean(tr[half:]))

        if first_half_mean <= 0:
            return 0.5

        ratio = second_half_mean / first_half_mean
        # Map ratio to [0, 1]: ratio > 1 = expanding, ratio < 1 = contracting
        # Use a sigmoid-like clamp: ratio 0.5 → 0.0, ratio 1.0 → 0.5, ratio 1.5 → 1.0
        normalised = max(0.0, min(1.0, (ratio - 0.5) / 1.0))
        return normalised
    except Exception:
        return 0.5


def get_volume_trend(
    bars: "pd.DataFrame | None" = None,
    *,
    lookback: int = 5,
) -> float:
    """Return the normalised volume trend over the last N bars [0, 1].

    Computes a simple linear slope of volume over the last ``lookback`` bars,
    then normalises to [0, 1] where:
      - 0.0 = volume declining sharply
      - 0.5 = volume flat
      - 1.0 = volume rising sharply

    This sub-feature enriches ``volume_ratio`` (which is a single-bar snapshot)
    by adding the *trend* of volume leading into the breakout.  Rising volume
    into a breakout is bullish for continuation; declining volume suggests a
    false breakout / low-conviction move.

    Args:
        bars:     1-minute OHLCV DataFrame (needs at least ``lookback`` rows).
        lookback: Number of bars to measure the volume trend over (default 5).

    Returns:
        Float in [0.0, 1.0].  0.5 = flat, >0.5 = rising, <0.5 = falling.
        Falls back to 0.5 if insufficient data or no Volume column.
    """
    if bars is None or len(bars) < lookback:
        return 0.5

    try:
        if "Volume" not in bars.columns:
            return 0.5

        vol = bars["Volume"].astype(float).values[-lookback:]

        # Skip if all zero (no volume data)
        if np.sum(vol) <= 0:
            return 0.5

        # Simple linear regression slope: y = vol, x = 0..lookback-1
        x = np.arange(lookback, dtype=float)
        x_mean = x.mean()
        vol_mean = vol.mean()

        if vol_mean <= 0:
            return 0.5

        numerator = float(np.sum((x - x_mean) * (vol - vol_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))

        if denominator <= 0:
            return 0.5

        slope = numerator / denominator

        # Normalise slope relative to mean volume:
        # slope_pct = slope / vol_mean gives the fractional change per bar.
        # A slope_pct of +0.2 means volume is growing 20% per bar — very strong.
        # Map [-0.3, +0.3] → [0.0, 1.0] with clamp.
        slope_pct = slope / vol_mean
        normalised = max(0.0, min(1.0, (slope_pct + 0.3) / 0.6))
        return normalised
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# v7 feature helpers — Daily Strategy layer
# ---------------------------------------------------------------------------


def get_daily_bias_direction(
    asset_name: str,
    bars_daily: "pd.DataFrame | None" = None,
    *,
    _bias_cache: dict[str, Any] | None = None,
) -> float:
    """Return normalised daily bias direction for the CNN tabular vector.

    Maps BiasDirection to [0, 1]:  SHORT → 0.0, NEUTRAL → 0.5, LONG → 1.0.
    Falls back to 0.5 (neutral) if the bias analyzer is unavailable or errors.

    Args:
        asset_name: Human-readable asset name (e.g. ``"Gold"``, ``"S&P"``).
        bars_daily: Daily OHLCV bars (≥10 rows preferred).
        _bias_cache: Optional pre-computed DailyBias dict keyed by asset name.
    """
    if _bias_cache is not None and asset_name in _bias_cache:
        bias = _bias_cache[asset_name]
        return getattr(bias, "direction_feature", 0.5)
    try:
        from lib.strategies.daily.bias_analyzer import compute_daily_bias

        if bars_daily is not None and not bars_daily.empty:
            bias = compute_daily_bias(asset_name, bars_daily)
            return bias.direction_feature
    except Exception:
        pass
    return 0.5


def get_daily_bias_confidence(
    asset_name: str,
    bars_daily: "pd.DataFrame | None" = None,
    *,
    _bias_cache: dict[str, Any] | None = None,
) -> float:
    """Return daily bias confidence [0, 1] for the CNN tabular vector.

    Falls back to 0.0 if unavailable.
    """
    if _bias_cache is not None and asset_name in _bias_cache:
        bias = _bias_cache[asset_name]
        return getattr(bias, "confidence_feature", 0.0)
    try:
        from lib.strategies.daily.bias_analyzer import compute_daily_bias

        if bars_daily is not None and not bars_daily.empty:
            bias = compute_daily_bias(asset_name, bars_daily)
            return bias.confidence_feature
    except Exception:
        pass
    return 0.0


def get_prior_day_pattern(
    asset_name: str,
    bars_daily: "pd.DataFrame | None" = None,
    *,
    _bias_cache: dict[str, Any] | None = None,
) -> float:
    """Return normalised prior-day candle pattern ordinal [0, 1].

    Uses the ordinal encoding from ``bias_analyzer.CANDLE_PATTERN_ORDINAL``:
        inside=0, doji=1, engulfing_bull=2, engulfing_bear=3, hammer=4,
        shooting_star=5, strong_close_up=6, strong_close_down=7,
        outside_day=8, neutral=9.
    Normalised as ordinal / 9.

    Falls back to 1.0 (neutral=9/9) if unavailable.
    """
    if _bias_cache is not None and asset_name in _bias_cache:
        bias = _bias_cache[asset_name]
        return getattr(bias, "candle_pattern_feature", 1.0)
    try:
        from lib.strategies.daily.bias_analyzer import compute_daily_bias

        if bars_daily is not None and not bars_daily.empty:
            bias = compute_daily_bias(asset_name, bars_daily)
            return bias.candle_pattern_feature
    except Exception:
        pass
    return 1.0  # neutral pattern


def get_weekly_range_position(
    asset_name: str,
    bars_daily: "pd.DataFrame | None" = None,
    *,
    _bias_cache: dict[str, Any] | None = None,
) -> float:
    """Return where price sits within the prior week's H/L range [0, 1].

    0.0 = at weekly low, 1.0 = at weekly high.
    Falls back to 0.5 (midpoint) if unavailable.
    """
    if _bias_cache is not None and asset_name in _bias_cache:
        bias = _bias_cache[asset_name]
        return getattr(bias, "weekly_range_feature", 0.5)
    try:
        from lib.strategies.daily.bias_analyzer import compute_daily_bias

        if bars_daily is not None and not bars_daily.empty:
            bias = compute_daily_bias(asset_name, bars_daily)
            return bias.weekly_range_feature
    except Exception:
        pass
    return 0.5


def get_monthly_trend_score(
    asset_name: str,
    bars_daily: "pd.DataFrame | None" = None,
    *,
    _bias_cache: dict[str, Any] | None = None,
) -> float:
    """Return normalised monthly trend score [0, 1].

    Raw score is the slope of the 20-day EMA on daily bars, in [-1, +1].
    Mapped to [0, 1] as (raw + 1) / 2.
    Falls back to 0.5 (flat) if unavailable.
    """
    if _bias_cache is not None and asset_name in _bias_cache:
        bias = _bias_cache[asset_name]
        return getattr(bias, "monthly_trend_feature", 0.5)
    try:
        from lib.strategies.daily.bias_analyzer import compute_daily_bias

        if bars_daily is not None and not bars_daily.empty:
            bias = compute_daily_bias(asset_name, bars_daily)
            return bias.monthly_trend_feature
    except Exception:
        pass
    return 0.5


def get_crypto_momentum_score(
    ticker: str,
    bars_1m: "pd.DataFrame | None" = None,
) -> float:
    """Return normalised crypto momentum score [0, 1] for the CNN.

    Raw score from ``crypto_momentum.CryptoMomentumSignal.to_tabular_feature()``
    is in [-1, +1].  Mapped to [0, 1] as (raw + 1) / 2.

    For non-crypto assets, this captures whether crypto momentum is leading
    the broader risk-on/risk-off move.  Returns 0.5 (neutral) if unavailable.
    """
    try:
        from lib.analysis.crypto_momentum import compute_crypto_momentum

        signal = compute_crypto_momentum(ticker)
        if signal is not None:
            raw = signal.to_tabular_feature()  # [-1, +1]
            return max(0.0, min(1.0, (raw + 1.0) / 2.0))
    except Exception:
        pass
    return 0.5


def get_asset_class_id(ticker: str) -> float:
    """Return the asset class ordinal / 4 for *ticker*, matching C# GetAssetClassNorm().

    Classes:
        0.00 — equity index (MES, MNQ, M2K, MYM)
        0.25 — FX          (6E, 6B, 6J, 6A, 6C, 6S)
        0.50 — metals/energy (MGC, SIL, MHG, MCL, MNG)
        0.75 — treasuries/ags (ZN, ZB, ZC, ZS, ZW)
        1.00 — crypto      (MBT, MET, BTC, ETH, …)

    Falls back to 0.0 (equity index) for unknown tickers, matching C#.
    """
    return ASSET_CLASS_ORDINALS.get(str(ticker).upper().strip(), 0.0)


def get_asset_volatility_class(ticker: str) -> float:
    """Return the legacy volatility class [0.0/0.5/1.0] for *ticker*.

    Kept for backward compatibility.  New code should use get_asset_class_id().
    """
    return ASSET_VOLATILITY_CLASS.get(str(ticker).upper().strip(), 0.5)


# Image pre-processing — matches ImageNet stats used by EfficientNetV2
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference threshold — probability above this → "send signal"
DEFAULT_THRESHOLD = 0.82

# Per-session inference thresholds.
#
# Rationale: overnight sessions (CME open, Sydney, Tokyo, Shanghai) have
# thinner markets and noisier price action than the primary London / US
# sessions, so we require lower confidence to avoid over-filtering the
# smaller signal pool while still maintaining signal quality.  Daytime
# sessions (Frankfurt, London, London-NY, US, CME Settlement) keep the
# full 0.82 bar.
#
# These are *starting* values — tune after 2 weeks of paper-trade data
# by comparing CNN probability distributions per session in Grafana and
# adjusting to match the 58-65% win-rate target.
#
# Keys match ORBSession.key values in orb.py / SESSION_BY_KEY.
SESSION_THRESHOLDS: dict[str, float] = {
    # ── Overnight sessions (18:00–03:00 ET, thin markets) ──────────────
    "cme": 0.75,  # CME Globex re-open 18:00 ET — first bars, wide spread
    "sydney": 0.72,  # Sydney/ASX 18:30 ET — thinnest session
    "tokyo": 0.74,  # Tokyo/TSE 19:00 ET — narrow range, metals/JPY
    "shanghai": 0.74,  # Shanghai/HK 21:00 ET — copper/gold driver
    # ── Primary daytime sessions ────────────────────────────────────────
    "frankfurt": 0.80,  # Frankfurt/Xetra 03:00 ET — pre-London, good vol
    "london": 0.82,  # London Open 03:00 ET — PRIMARY, highest conviction
    "london_ny": 0.82,  # London-NY Crossover 08:00 ET — highest volume
    "us": 0.82,  # US Equity Open 09:30 ET — classic ORB session
    "cme_settle": 0.78,  # CME Settlement 14:00 ET — metals/energy only
}


def get_session_threshold(session_key: str | None) -> float:
    """Return the CNN inference threshold for *session_key*.

    Falls back to ``DEFAULT_THRESHOLD`` (0.82) for unknown or None keys.
    This is the single authoritative lookup used by both
    ``predict_breakout()`` and ``predict_breakout_batch()``.

    Args:
        session_key: ORBSession.key string (e.g. "london", "tokyo", "us").
                     None or empty string returns DEFAULT_THRESHOLD.

    Returns:
        Float probability threshold in [0, 1].

    Example::

        >>> get_session_threshold("tokyo")
        0.74
        >>> get_session_threshold("london")
        0.82
        >>> get_session_threshold(None)
        0.82
    """
    if not session_key:
        return DEFAULT_THRESHOLD
    return SESSION_THRESHOLDS.get(session_key.lower().strip(), DEFAULT_THRESHOLD)


# Ordinal session encoding — maps ORBSession.key → float in [0, 1].
# Encodes the session's position in the 24-hour Globex day cycle so the
# tabular head can learn time-of-day patterns.  Used by BreakoutDataset
# and _normalise_tabular_for_inference in place of the old binary
# session_flag (1.0 = US, 0.0 = London).
#
# Ordered chronologically within the Globex day (18:00 ET start):
#   cme(18:00) → sydney(18:30) → tokyo(19:00) → shanghai(21:00) →
#   frankfurt(03:00) → london(03:00) → london_ny(08:00) →
#   us(09:30) → cme_settle(14:00)
SESSION_ORDINAL: dict[str, float] = {
    "cme": 0.0 / 8,  # 18:00 ET — position 0
    "sydney": 1.0 / 8,  # 18:30 ET — position 1
    "tokyo": 2.0 / 8,  # 19:00 ET — position 2
    "shanghai": 3.0 / 8,  # 21:00 ET — position 3
    "frankfurt": 4.0 / 8,  # 03:00 ET — position 4
    "london": 5.0 / 8,  # 03:00 ET — position 5
    "london_ny": 6.0 / 8,  # 08:00 ET — position 6
    "us": 7.0 / 8,  # 09:30 ET — position 7
    "cme_settle": 8.0 / 8,  # 14:00 ET — position 8
}

# Backward-compat aliases so old callers that pass session_flag=1.0 (US)
# or session_flag=0.0 (London) still get sensible ordinal values.
_SESSION_FLAG_COMPAT = {1.0: SESSION_ORDINAL["us"], 0.0: SESSION_ORDINAL["london"]}


def get_session_ordinal(session_key: str | None) -> float:
    """Return the ordinal encoding [0, 1] for *session_key*.

    Falls back to the US session ordinal (0.875) for unknown keys.
    Accepts the legacy float values ``1.0`` (US) and ``0.0`` (London)
    for backward compatibility with old callers.

    Args:
        session_key: ORBSession.key string, or None.

    Returns:
        Float in [0.0, 1.0] representing session position in the Globex day.
    """
    if session_key is None:
        return SESSION_ORDINAL["us"]
    # Legacy float-as-string passthrough (e.g. "1.0", "0.0")
    with contextlib.suppress(ValueError):
        fval = float(session_key)
        if fval in _SESSION_FLAG_COMPAT:
            return _SESSION_FLAG_COMPAT[fval]
    return SESSION_ORDINAL.get(str(session_key).lower().strip(), SESSION_ORDINAL["us"])


# Model output directory
DEFAULT_MODEL_DIR = "models"
MODEL_PREFIX = "breakout_cnn_"

# Thread lock for model loading
_model_lock = threading.Lock()
_cached_model: Any | None = None
_cached_model_path: str | None = None
_cached_model_mtime: float = 0.0


def invalidate_model_cache() -> bool:
    """Invalidate the cached CNN model so the next inference reloads from disk.

    Thread-safe.  Returns True if a cached model was actually evicted,
    False if the cache was already empty.

    Called by the engine's hot-reload watcher when it detects that the
    champion model file has changed on disk (new ``st_mtime``).
    """
    global _cached_model, _cached_model_path, _cached_model_mtime
    with _model_lock:
        had_model = _cached_model is not None
        _cached_model = None
        _cached_model_path = None
        _cached_model_mtime = 0.0
    if had_model:
        logger.info("CNN model cache invalidated — next inference will reload from disk")
    return had_model


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_inference_transform():
    """Standard image transform for inference (resize + normalise)."""
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_training_transform():
    """Image transform for training with mild augmentation.

    Augmentations are intentionally light — we want the CNN to learn from
    the chart structure (candlestick bodies, ORB box, overlays), not from
    random crops or heavy colour jitter that would destroy that information.

    Augmentation strategy:
      - Mild random crop (±16 px): simulates slight chart zoom variation
      - No horizontal flip: chart direction (left→right time) must be preserved
      - Mild brightness/contrast jitter (±8%): handles monitor calibration
        differences and theme variations (dark vs light dashboard)
      - Mild saturation jitter (±5%): Kraken/crypto pairs use different
        colour palettes from futures charts
      - Random rotation (±1.5°): simulates slight chart panel tilt in screenshots
      - Random erasing (p=0.05, max 10% area): simulates minor UI overlays
        or partial occlusions on the dashboard
    """
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
            T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.0),  # disabled — chart direction matters
            T.RandomRotation(degrees=1.5, fill=0),  # tiny tilt only
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # Random erasing AFTER ToTensor (operates on tensor, not PIL)
            # Simulates minor UI overlays / partial occlusion.
            # p=0.05: only 5% of images affected; scale capped at 10% area.
            T.RandomErasing(p=0.05, scale=(0.01, 0.10), ratio=(0.3, 3.3), value=0),
        ]
    )


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return the best available device string ('cuda', 'mps', or 'cpu')."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class BreakoutDataset(Dataset[Any]):  # type: ignore[no-redef]
        """PyTorch Dataset for breakout chart images + tabular features.

        Expects a CSV with columns:
          - image_path: path to the PNG chart snapshot
          - label: "good_long", "good_short", "bad_long", "bad_short"
          - quality_pct, volume_ratio, atr_pct, cvd_delta, nr7_flag, direction
            (the tabular features)

        The binary target is:
          - 1 if label starts with "good_" (clean breakout)
          - 0 otherwise (failed breakout)
        """

        def __init__(
            self,
            csv_path: str,
            transform=None,
            image_root: str | None = None,
        ):
            self.df = pd.read_csv(csv_path)
            self.transform = transform or get_inference_transform()
            self.image_root = image_root

            if "label" not in self.df.columns:
                raise ValueError("CSV must have a 'label' column")

            # --- Pre-validate: aggressively remove rows without usable images ---
            initial_count = len(self.df)

            # 1. Drop rows where image_path is NaN or empty string
            self.df = self.df.dropna(subset=["image_path"])
            self.df = self.df[self.df["image_path"].astype(str).str.strip().ne("")]
            dropped_empty = initial_count - len(self.df)

            # 2. Verify image files actually exist on disk
            def _resolve(p: str) -> str:
                p = str(p).strip()
                if self.image_root and not os.path.isabs(p):
                    return os.path.join(self.image_root, p)
                return p

            exists_mask = self.df["image_path"].apply(lambda p: os.path.isfile(_resolve(str(p))))
            dropped_missing = int((~exists_mask).sum())
            self.df = self.df[exists_mask]

            # 3. Reset index so iloc is contiguous
            self.df = self.df.reset_index(drop=True)

            if dropped_empty > 0 or dropped_missing > 0:
                logger.warning(
                    "BreakoutDataset: dropped %d empty-path + %d missing-file rows from %s (kept %d / %d)",
                    dropped_empty,
                    dropped_missing,
                    csv_path,
                    len(self.df),
                    initial_count,
                )

            logger.info("BreakoutDataset loaded: %d samples from %s", len(self.df), csv_path)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]

            # --- Image ---
            img_path = str(row["image_path"]).strip()
            if self.image_root and not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root, img_path)

            valid = True  # tracks whether this sample should be used
            try:
                img = Image.open(img_path).convert("RGB")

                # Detect degenerate / near-blank images: if the image has
                # essentially zero variance it contains no chart information
                # (e.g. a solid-colour fallback from a render failure).
                img_arr = np.asarray(img)
                if img_arr.std() < 2.0:
                    logger.warning(
                        "Near-blank image detected (std=%.2f): %s — marking invalid",
                        img_arr.std(),
                        img_path,
                    )
                    valid = False
            except Exception as exc:
                logger.warning("Failed to load image %s: %s — marking invalid", img_path, exc)
                # Create a dummy image so tensor pipeline doesn't crash;
                # the valid=False flag tells the collate_fn to drop this sample.
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
                valid = False

            if self.transform:
                img = self.transform(img)

            # ── Tabular features v6 — 18 features matching C# PrepareCnnTabular ──
            #
            # Raw values are normalised here using the same transforms as
            # OrbCnnPredictor.NormaliseTabular() in BreakoutStrategy.cs so that
            # Python training and NT8 inference are byte-for-byte identical.

            # [0] quality_pct_norm — quality / 100, clamp [0, 1]
            _qual = max(0.0, min(1.0, float(row.get("quality_pct", 50)) / 100.0))

            # [1] volume_ratio — log-scale: min(log1p(raw) / log1p(10), 1.0)
            _vol_raw = max(float(row.get("volume_ratio", 1.0)), 0.01)
            _vol_norm = min(np.log1p(_vol_raw) / np.log1p(10.0), 1.0)

            # [2] atr_pct — ×100 then clamp [0, 1]
            _atr_norm = min(float(row.get("atr_pct", 0.0)) * 100.0, 1.0)

            # [3] cvd_delta — clamp [-1, 1]
            _cvd_norm = max(-1.0, min(1.0, float(row.get("cvd_delta", 0.0))))

            # [4] nr7_flag — passthrough 0 or 1
            _nr7 = float(row.get("nr7_flag", 0))

            # [5] direction_flag — 1=LONG 0=SHORT
            _dir = 1.0 if str(row.get("direction", "")).upper().startswith("L") else 0.0

            # [6] session_ordinal — Globex day position [0, 1]; infer from breakout_time
            _hour = 10  # default to US open hour
            _session_ord = SESSION_ORDINAL["us"]
            try:
                _bt = str(row.get("breakout_time", ""))
                if _bt and " " in _bt:
                    _hour = int(_bt.split(" ")[1].split(":")[0])
                    if _hour < 3:
                        _session_ord = SESSION_ORDINAL["shanghai"]
                    elif _hour < 8:
                        _session_ord = SESSION_ORDINAL["london"]
                    elif _hour < 9:
                        _session_ord = SESSION_ORDINAL["london_ny"]
                    else:
                        _session_ord = SESSION_ORDINAL["us"]
            except Exception:
                pass

            # [7] london_overlap_flag — 08:00–09:00 ET
            _london_overlap = 1.0 if 8 <= _hour <= 9 else 0.0

            # [8] or_range_atr_ratio — ORB range / ATR; clamp(raw, 0, 3) / 3
            _orb_range = float(row.get("range_size", row.get("or_range", 0.0)))
            _atr_val = float(row.get("atr_value", 0.0))
            if _atr_val <= 0:
                # Derive atr_value from atr_pct × a proxy price (1.0 → atr_pct already fractional)
                _atr_val = float(row.get("atr_pct", 0.0))
            _or_range_atr = min(max(_orb_range / _atr_val, 0.0), 3.0) / 3.0 if _atr_val > 0 else 0.0

            # [9] premarket_range_ratio — premarket range / ORB range; clamp(raw, 0, 5) / 5
            _pm_range = float(row.get("premarket_range", 0.0))
            _pm_ratio = min(max(_pm_range / _orb_range, 0.0), 5.0) / 5.0 if _orb_range > 0 else 0.0

            # [10] bar_of_day — minutes since Globex open (18:00 ET) / 1380; clamp [0, 1]
            _bar_min = int(row.get("bar_of_day_minutes", -1))
            if _bar_min < 0:
                # Derive from _hour: minutes since 18:00 ET
                _bar_min = (_hour + 6) * 60 if _hour < 18 else (_hour - 18) * 60
            _bar_of_day = max(0.0, min(1.0, _bar_min / 1380.0))

            # [11] day_of_week — Mon=0..Fri=4 / 4; already normalised in CSV or default 0.5
            _dow = float(row.get("day_of_week_norm", 0.5))

            # [12] vwap_distance — (price − vwap) / ATR; clamp(raw, -3, 3) / 3
            _vwap_dist_raw = float(row.get("vwap_distance", 0.0))
            _vwap_dist = max(-3.0, min(3.0, _vwap_dist_raw)) / 3.0

            # [13] asset_class_id — ordinal / 4 matching C# GetAssetClassNorm()
            _ticker = str(row.get("ticker", row.get("symbol", ""))).upper().strip()
            _asset_cls = get_asset_class_id(_ticker)

            # ── v6 additions ──────────────────────────────────────────────

            # [14] breakout_type_ord — BreakoutType ordinal / 12 → [0, 1]
            _bt_ord_val = 0.0
            try:
                _bt_raw = str(row.get("breakout_type", row.get("breakout_type_ord", "ORB")))
                # Try numeric ordinal first (from CSV)
                try:
                    _bt_ord_val = max(0.0, min(1.0, float(row.get("breakout_type_ord", -1))))
                    if _bt_ord_val < 0:
                        raise ValueError
                except (ValueError, TypeError):
                    _bt_ord_val = get_breakout_type_ordinal(_bt_raw)
            except Exception:
                _bt_ord_val = 0.0

            # [15] asset_volatility_class — low=0.0, med=0.5, high=1.0
            _vol_cls = get_asset_volatility_class(_ticker)

            # [16] hour_of_day — ET hour / 23 → [0, 1]
            _hour_of_day = max(0.0, min(1.0, _hour / 23.0))

            # [17] tp3_atr_mult_norm — TP3 multiplier / 5.0 → [0, 1]
            _tp3_mult = 0.0
            try:
                _tp3_raw = float(row.get("tp3_atr_mult", 0.0))
                _tp3_mult = max(0.0, min(1.0, _tp3_raw / 5.0))
            except (ValueError, TypeError):
                pass

            tabular = torch.tensor(
                [
                    _qual,  # [0]  quality_pct_norm
                    _vol_norm,  # [1]  volume_ratio (log-scaled)
                    _atr_norm,  # [2]  atr_pct (×100, clamped)
                    _cvd_norm,  # [3]  cvd_delta
                    _nr7,  # [4]  nr7_flag
                    _dir,  # [5]  direction_flag
                    _session_ord,  # [6]  session_ordinal
                    _london_overlap,  # [7]  london_overlap_flag
                    _or_range_atr,  # [8]  or_range_atr_ratio
                    _pm_ratio,  # [9]  premarket_range_ratio
                    _bar_of_day,  # [10] bar_of_day
                    _dow,  # [11] day_of_week
                    _vwap_dist,  # [12] vwap_distance
                    _asset_cls,  # [13] asset_class_id
                    _bt_ord_val,  # [14] breakout_type_ord
                    _vol_cls,  # [15] asset_volatility_class
                    _hour_of_day,  # [16] hour_of_day
                    _tp3_mult,  # [17] tp3_atr_mult_norm
                ],
                dtype=torch.float32,
            )

            # Guard against NaN / Inf from corrupt data
            if torch.isnan(tabular).any() or torch.isinf(tabular).any():
                logger.warning("NaN/Inf in tabular features at row %d — zeroing", idx)
                tabular = torch.zeros(NUM_TABULAR, dtype=torch.float32)

            # --- Label ---
            label_str = str(row.get("label", "bad"))
            target = 1 if label_str.startswith("good") else 0

            return img, tabular, torch.tensor(target, dtype=torch.long), torch.tensor(valid, dtype=torch.bool)

        @staticmethod
        def skip_invalid_collate(batch):
            """Custom collate_fn that drops samples flagged as invalid.

            Each sample is a tuple ``(img, tabular, target, valid_flag)``.
            Only samples where ``valid_flag`` is True are kept.  If the entire
            batch is invalid we return ``None`` — the training loop must check
            for this and skip the batch.
            """
            # batch is a list of (img, tabular, target, valid) tuples
            filtered = [s for s in batch if s[3].item()]
            if not filtered:
                return None  # entire batch was invalid — caller must skip

            imgs = torch.stack([s[0] for s in filtered])
            tabs = torch.stack([s[1] for s in filtered])
            targets = torch.stack([s[2] for s in filtered])
            return imgs, tabs, targets


else:
    # Stub when torch is not available
    class BreakoutDataset:  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create BreakoutDataset")

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise RuntimeError("PyTorch is not installed")

        @staticmethod
        def skip_invalid_collate(batch: Any) -> Any:
            """Stub — PyTorch is not installed."""
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Number of distinct breakout types — used to size the embedding table.
# Mirrors len(BreakoutType) in lib.core.breakout_types.
NUM_BREAKOUT_TYPES = 13

# Learned embedding dimension for breakout type.
# Replaces the single scalar ``breakout_type_ord`` feature with a richer
# NUM_BREAKOUT_TYPES × BREAKOUT_EMBED_DIM lookup table that the model
# trains end-to-end.  The tabular vector's ``breakout_type_ord`` slot [14]
# is still consumed (so the input dimension stays at NUM_TABULAR = 18) but
# when ``use_type_embedding=True`` that slot is ignored and the embedding
# replaces it in the combined representation.
BREAKOUT_EMBED_DIM = 8


if _TORCH_AVAILABLE:

    class HybridBreakoutCNN(nn.Module):  # type: ignore[no-redef]
        """Hybrid image + tabular model for breakout classification (v6 contract).

        Architecture:
          Image branch:   EfficientNetV2-S (pre-trained ImageNet) → 1280-dim
          Tabular branch: Linear(NUM_TABULAR→128) → BN → ReLU → Dropout →
                          Linear(128→64) → BN → ReLU → Linear(64→32)
          Classifier:     Linear(1280+32→512) → BN → ReLU → Dropout →
                          Linear(512→128) → ReLU → Dropout → Linear(128→2)

        NUM_TABULAR = 18 (v6 contract, identical to C# PrepareCnnTabular).

        The model outputs raw logits for 2 classes:
          - Class 0: bad breakout (fail / chop)
          - Class 1: good breakout (clean follow-through)

        Apply ``torch.softmax(output, dim=1)[:, 1]`` to get P(good breakout).
        """

        def __init__(
            self,
            num_tabular: int = NUM_TABULAR,
            dropout: float = 0.4,
            pretrained: bool = True,
        ):
            super().__init__()
            self.num_tabular = num_tabular

            # --- Image backbone: EfficientNetV2-S ---
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)
            backbone.classifier = nn.Identity()  # type: ignore[assignment]
            self.cnn = backbone
            self._cnn_out_dim = 1280

            # --- Tabular branch (deeper than v5/v6, BatchNorm for stability) ---
            self.tabular_head = nn.Sequential(
                nn.Linear(num_tabular, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
            )

            # --- Classifier ---
            combined_dim = self._cnn_out_dim + 32
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 2),
            )

        def forward(
            self,
            image: torch.Tensor,
            tabular: torch.Tensor,
            type_ids: torch.Tensor | None = None,  # kept for API compat, unused
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                image:   (B, 3, 224, 224) normalised image tensor.
                tabular: (B, NUM_TABULAR) float tensor — v6 18-feature vector.
                type_ids: ignored (kept for backward API compatibility).

            Returns:
                (B, 2) logits tensor.
            """
            img_features = self.cnn(image)  # (B, 1280)
            tab_features = self.tabular_head(tabular)  # (B, 32)
            combined = torch.cat([img_features, tab_features], dim=1)  # (B, 1312)
            return self.classifier(combined)  # (B, 2)

        def freeze_backbone(self) -> None:
            """Freeze the CNN backbone (first N epochs of fine-tuning)."""
            for param in self.cnn.parameters():
                param.requires_grad = False
            logger.info("CNN backbone frozen")

        def unfreeze_backbone(self) -> None:
            """Unfreeze the CNN backbone for full fine-tuning."""
            for param in self.cnn.parameters():
                param.requires_grad = True
            logger.info("CNN backbone unfrozen")

else:

    class HybridBreakoutCNN:  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create HybridBreakoutCNN")

        def freeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")

        def unfreeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainResult(NamedTuple):
    """Return value from :func:`train_model`.

    Attributes:
        model_path:     Absolute path to the saved ``.pt`` checkpoint.
        best_epoch:     1-based epoch index that achieved the best validation
                        accuracy (None if no validation was performed).
        epochs_trained: Total number of epochs completed.
    """

    model_path: str
    best_epoch: int | None
    epochs_trained: int


def _detect_docker() -> bool:
    """Return True if we appear to be running inside a Docker container."""
    try:
        if os.path.exists("/.dockerenv"):
            return True
        with open("/proc/1/cgroup") as f:
            return any("docker" in line or "containerd" in line for line in f)
    except Exception:
        return False


def _safe_num_workers(requested: int) -> int:
    """Clamp DataLoader num_workers to 0 inside Docker when /dev/shm is small.

    PyTorch multiprocess DataLoader workers communicate via shared memory.
    Docker's default /dev/shm is only 64 MB, which causes:
        RuntimeError: unable to allocate shared memory(shm)
    Setting num_workers=0 forces single-process loading (slower but safe).
    """
    if requested == 0:
        return 0

    if not _detect_docker():
        return requested

    # Check /dev/shm size — need at least 512 MB for multi-worker loading
    try:
        stat = os.statvfs("/dev/shm")
        shm_bytes = stat.f_frsize * stat.f_blocks
        shm_mb = shm_bytes / (1024 * 1024)
        if shm_mb >= 512:
            logger.info("Docker detected with %.0f MB /dev/shm — using %d DataLoader workers", shm_mb, requested)
            return requested
        else:
            logger.warning(
                "Docker detected with only %.0f MB /dev/shm (need ≥512 MB) — "
                "forcing num_workers=0 to avoid shared memory crash. "
                "Fix: add 'shm_size: 2gb' to docker-compose.yml",
                shm_mb,
            )
            return 0
    except Exception:
        logger.warning("Docker detected but cannot check /dev/shm — forcing num_workers=0 for safety")
        return 0


def train_model(
    data_csv: str,
    val_csv: str | None = None,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    freeze_epochs: int = 2,
    model_dir: str = DEFAULT_MODEL_DIR,
    image_root: str | None = None,
    num_workers: int = 4,
    save_best: bool = True,
) -> TrainResult | None:
    """Train the HybridBreakoutCNN model.

    Two-phase training:
      1. Freeze CNN backbone for ``freeze_epochs`` epochs — trains only the
         tabular head and classifier on your data.
      2. Unfreeze backbone and fine-tune everything at a lower LR.

    Args:
        data_csv: Path to training CSV (see BreakoutDataset for format).
        val_csv: Optional validation CSV.  If None, 15% of training data
                 is held out automatically.
        epochs: Total training epochs (default 8).
        batch_size: Batch size (default 32).
        lr: Learning rate (default 3e-4).
        weight_decay: AdamW weight decay (default 1e-5).
        freeze_epochs: Number of epochs to freeze the CNN backbone (default 2).
        model_dir: Directory to save the trained model (default "models").
        image_root: Optional root directory to prepend to image_path values.
        num_workers: DataLoader workers (default 4).
        save_best: If True and val_csv is provided, save the best model by
                   validation accuracy instead of the final epoch.

    Returns:
        :class:`TrainResult` with ``model_path``, ``best_epoch``, and
        ``epochs_trained``, or ``None`` if training failed.
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch is not installed — cannot train model")
        return None

    device = torch.device(get_device())
    logger.info("Training on device: %s", device)

    # Clamp num_workers for Docker shared memory safety
    num_workers = _safe_num_workers(num_workers)

    # --- Datasets ---
    train_transform = get_training_transform()
    val_transform = get_inference_transform()

    train_dataset = BreakoutDataset(data_csv, transform=train_transform, image_root=image_root)

    if val_csv:
        val_dataset = BreakoutDataset(val_csv, transform=val_transform, image_root=image_root)
    else:
        # Auto-split: 85% train / 15% val
        n = len(train_dataset)
        n_val = max(1, int(n * 0.15))
        n_train = n - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
        logger.info("Auto-split: %d train / %d val", n_train, n_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )

    # --- Model ---
    model = HybridBreakoutCNN(pretrained=True)
    logger.info("HybridBreakoutCNN v6: %d tabular features", NUM_TABULAR)
    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # --- Loss ---
    # Use label smoothing to prevent overconfident predictions
    criterion: Any = nn.CrossEntropyLoss(label_smoothing=0.05)

    # --- Training loop ---
    best_val_acc = 0.0
    best_model_path: str | None = None
    best_epoch: int | None = None
    epochs_completed: int = 0
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        # Phase management: freeze/unfreeze backbone
        if epoch < freeze_epochs:
            if epoch == 0:
                model.freeze_backbone()
        elif epoch == freeze_epochs:
            model.unfreeze_backbone()
            # Lower LR for backbone fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            # skip_invalid_collate returns None when every sample was invalid
            if batch is None:
                continue
            imgs, tabs, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs, tabs)
            loss = criterion(outputs, labels)

            # NaN guard — skip batch if loss explodes
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss at batch %d — skipping", batch_idx)
                continue

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1) * 100

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, tabs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(imgs, tabs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100

        current_lr = optimizer.param_groups[0]["lr"]
        phase = "frozen" if epoch < freeze_epochs else "fine-tune"

        logger.info(
            "Epoch %d/%d [%s] — Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | LR: %.2e",
            epoch + 1,
            epochs,
            phase,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            current_lr,
        )

        # --- Save best model ---
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1  # 1-based
            best_model_path = os.path.join(
                model_dir, f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_acc{val_acc:.0f}.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            logger.info("New best model saved: %s (val_acc=%.1f%%)", best_model_path, val_acc)

        epochs_completed = epoch + 1

    # --- Save final model (if not saving best, or as fallback) ---
    final_path = os.path.join(model_dir, f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved: %s", final_path)

    result_path = best_model_path if (save_best and best_model_path) else final_path
    logger.info(
        "Training complete — best val accuracy: %.1f%% — model: %s (best_epoch=%s, epochs_trained=%d)",
        best_val_acc,
        result_path,
        best_epoch,
        epochs_completed,
    )

    # Invalidate cached model so next inference picks up the new one
    invalidate_model_cache()

    return TrainResult(
        model_path=result_path,
        best_epoch=best_epoch,
        epochs_trained=epochs_completed,
    )


def evaluate_model(
    model_path: str,
    val_csv: str,
    image_root: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, Any] | None:
    """Evaluate a trained model checkpoint against a validation CSV.

    Computes accuracy, precision, and recall (macro-averaged) on the
    validation set.  Intended to be called by the trainer server after
    :func:`train_model` completes, so the pipeline can gate promotion on
    concrete metrics.

    Args:
        model_path: Path to a ``.pt`` state-dict checkpoint.
        val_csv: Path to the validation CSV (same format as training CSV).
        image_root: Optional root directory prepended to ``image_path``
                    values in the CSV.
        batch_size: Evaluation batch size (default 32).
        num_workers: DataLoader workers (default 4).

    Returns:
        Dict with keys ``val_accuracy``, ``val_precision``, ``val_recall``
        (all 0.0–1.0 floats), or ``None`` if evaluation failed.
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch is not installed — cannot evaluate model")
        return None

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        device = torch.device(get_device())
        num_workers = _safe_num_workers(num_workers)

        # Load model — auto-detect num_tabular from checkpoint
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)  # type: ignore[call-overload]
        tab_key = "tabular_head.0.weight"
        num_tab = state_dict[tab_key].shape[1] if tab_key in state_dict else NUM_TABULAR
        model = HybridBreakoutCNN(pretrained=False, num_tabular=num_tab)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        # Build validation loader
        val_transform = get_inference_transform()
        val_dataset = BreakoutDataset(val_csv, transform=val_transform, image_root=image_root)

        if len(val_dataset) == 0:
            logger.warning("Validation dataset is empty — cannot evaluate")
            return None

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=BreakoutDataset.skip_invalid_collate,
        )

        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, tabs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(imgs, tabs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        if not all_labels:
            logger.warning("No valid samples evaluated — cannot compute metrics")
            return None

        acc = accuracy_score(all_labels, all_preds)
        # Use zero_division=0.0 so we don't crash when a class is absent
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0.0)  # type: ignore[arg-type]
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0.0)  # type: ignore[arg-type]

        logger.info(
            "Evaluation complete — accuracy: %.1f%%, precision: %.1f%%, recall: %.1f%% (%d samples)",
            acc * 100,
            prec * 100,
            rec * 100,
            len(all_labels),
        )

        return {
            "val_accuracy": round(float(acc), 4),
            "val_precision": round(float(prec), 4),
            "val_recall": round(float(rec), 4),
            "num_samples": len(all_labels),
        }

    except Exception as exc:
        logger.error("Model evaluation failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_best_model(model_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Find the best available model in *model_dir*.

    Selection priority:
      1. ``breakout_cnn_best.pt`` (the promoted champion) — always preferred
         when it exists.
      2. The checkpoint with the highest ``val_accuracy`` recorded in its
         companion ``<stem>_meta.json`` sidecar file.
      3. The most recently modified ``.pt`` file (mtime fallback when no
         meta JSON is available for any checkpoint).

    Args:
        model_dir: Directory to search (default ``"models"``).

    Returns:
        Absolute path to the chosen model file, or ``None`` if no models
        are found.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return None

    # 1. Prefer the promoted champion if it exists
    champion = model_path / "breakout_cnn_best.pt"
    if champion.is_file():
        logger.debug("Model selection: using champion %s", champion)
        return str(champion)

    # 2. Scan all breakout_cnn_*.pt checkpoints
    pt_files = list(model_path.glob(f"{MODEL_PREFIX}*.pt"))
    if not pt_files:
        return None

    # Try to rank by val_accuracy from companion meta JSON sidecars.
    # A meta JSON for checkpoint "breakout_cnn_20260101_020000_acc87.pt"
    # would be named "breakout_cnn_20260101_020000_acc87_meta.json".
    # Also check the global "breakout_cnn_best_meta.json" as a fallback.
    global_meta_path = model_path / "breakout_cnn_best_meta.json"

    def _val_accuracy(pt: Path) -> float:
        # Sidecar meta: same stem + _meta.json
        sidecar = pt.with_name(pt.stem + "_meta.json")
        for candidate in (sidecar, global_meta_path):
            if candidate.is_file():
                try:
                    import json as _json

                    meta = _json.loads(candidate.read_text())
                    return float(meta.get("val_accuracy", 0.0))
                except Exception:
                    pass
        # Try to parse accuracy from filename: "..._accNN.pt"
        import re as _re

        m = _re.search(r"_acc(\d+(?:\.\d+)?)", pt.stem)
        if m:
            return float(m.group(1))
        return 0.0

    scored = [(pt, _val_accuracy(pt)) for pt in pt_files]

    # Check if any checkpoint has a meaningful accuracy score
    if any(score > 0.0 for _, score in scored):
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        logger.debug(
            "Model selection: chose %s (val_acc=%.1f%%) from %d checkpoints",
            best.name,
            scored[0][1],
            len(pt_files),
        )
        return str(best)

    # 3. Fallback: most recently modified file
    pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.debug("Model selection: fallback to newest mtime %s", pt_files[0].name)
    return str(pt_files[0])


def _find_latest_model(model_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Alias for :func:`_find_best_model` kept for backwards compatibility."""
    return _find_best_model(model_dir)


def _build_model_from_checkpoint(state_dict: dict) -> Any:
    """Instantiate a HybridBreakoutCNN whose architecture matches *state_dict*.

    Detects whether the checkpoint was trained with ``use_type_embedding=True``
    by checking for the ``type_embedding.weight`` key in the state dict.

    Returns a model with weights loaded (eval mode, not moved to device yet).
    """
    if not _TORCH_AVAILABLE:
        return None

    use_type_emb = "type_embedding.weight" in state_dict

    # Infer num_tabular from first tabular_head Linear weight shape.
    # For scalar-only models the key is ``tabular_head.0.weight`` with shape
    # (64, num_tabular).  For embedding models it is (64, num_tabular-1).
    num_tabular = NUM_TABULAR
    tab_key = "tabular_head.0.weight"
    if tab_key in state_dict:
        in_features = state_dict[tab_key].shape[1]
        num_tabular = in_features + 1 if use_type_emb else in_features

    model = HybridBreakoutCNN(
        num_tabular=num_tabular,
        pretrained=False,  # weights come from checkpoint
        use_type_embedding=use_type_emb,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_model(
    model_path: str | None = None,
    device: str | None = None,
) -> Any | None:
    """Load a HybridBreakoutCNN model from disk.

    Uses a module-level cache to avoid reloading on every inference call.
    Thread-safe via _model_lock.

    Args:
        model_path: Explicit path to a .pt file.  If None, finds the latest.
        device: Device to load onto.  If None, auto-detects.

    Returns:
        The loaded model in eval mode, or None if loading failed.
    """
    if not _TORCH_AVAILABLE:
        return None

    global _cached_model, _cached_model_path

    if model_path is None:
        model_path = _find_latest_model()

    if model_path is None:
        logger.warning("No trained model found in %s", DEFAULT_MODEL_DIR)
        return None

    with _model_lock:
        # Return cached model if same path
        if _cached_model is not None and _cached_model_path == model_path:
            return _cached_model

        dev = torch.device(device or get_device())

        try:
            try:
                state_dict = torch.load(model_path, map_location=dev, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location=dev)  # type: ignore[call-overload]
            tab_key = "tabular_head.0.weight"
            num_tab = state_dict[tab_key].shape[1] if tab_key in state_dict else NUM_TABULAR
            model = HybridBreakoutCNN(pretrained=False, num_tabular=num_tab)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model = model.to(dev)

            _cached_model = model
            _cached_model_path = model_path

            logger.info("Model loaded: %s → %s", model_path, dev)
            return model

        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)
            return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _normalise_tabular_for_inference(raw_features: Sequence[float]) -> list[float]:
    """Normalise a raw tabular feature vector for inference.

    Applies the same transforms as OrbCnnPredictor.NormaliseTabular() in C#
    so that Python inference and NT8 inference are identical.

    v7.1 input order (28 features — must match TABULAR_FEATURES exactly):
        [0]  quality_pct_norm      — quality / 100, already in [0, 1]
        [1]  volume_ratio          — raw ratio (log-normalised here)
        [2]  atr_pct               — ATR / price fraction (×100 here)
        [3]  cvd_delta             — signed vol ratio, clamped [-1, 1]
        [4]  nr7_flag              — 0 or 1 passthrough
        [5]  direction_flag        — 1=LONG 0=SHORT passthrough
        [6]  session_ordinal       — Globex day position [0, 1] passthrough
        [7]  london_overlap_flag   — 0 or 1 passthrough
        [8]  or_range_atr_ratio    — raw ORB/ATR  (clamp(0,3)/3 applied here)
        [9]  premarket_range_ratio — raw PM/ORB  (clamp(0,5)/5 applied here)
        [10] bar_of_day            — already normalised [0, 1] passthrough
        [11] day_of_week           — already normalised [0, 1] passthrough
        [12] vwap_distance         — raw (price-vwap)/ATR (clamp(-3,3)/3 here)
        [13] asset_class_id        — ordinal/4 already [0, 1] passthrough
        [14] breakout_type_ord     — BreakoutType ordinal/12 [0, 1] passthrough
        [15] asset_volatility_class — low=0/med=0.5/high=1.0 passthrough
        [16] hour_of_day           — ET hour/23 [0, 1] passthrough
        [17] tp3_atr_mult_norm     — TP3 mult/5.0 [0, 1] passthrough
        [18] daily_bias_direction  — SHORT=0, NEUTRAL=0.5, LONG=1.0 passthrough
        [19] daily_bias_confidence — [0, 1] passthrough
        [20] prior_day_pattern     — ordinal/9 [0, 1] passthrough
        [21] weekly_range_position — [0, 1] passthrough
        [22] monthly_trend_score   — [0, 1] passthrough
        [23] crypto_momentum_score — [0, 1] passthrough
        [24] breakout_type_category — {0, 0.5, 1.0} passthrough
        [25] session_overlap_flag  — 0 or 1 passthrough
        [26] atr_trend             — [0, 1] passthrough
        [27] volume_trend          — [0, 1] passthrough

    For backward compat, 8-feature (v5), 14-feature (v4), 18-feature (v6),
    and 24-feature (v7) vectors are zero-padded to 28 with sensible defaults
    before normalisation.

    Returns a list of NUM_TABULAR floats ready for the model tabular input tensor.
    """
    f = list(raw_features)

    # Backward compat padding — extend shorter vectors with neutral defaults
    if len(f) == 8:
        # v5 (8 features) → pad to 28
        # [8..13] v4 extras, [14..17] v6 extras, [18..23] v7 extras, [24..27] v7.1 extras
        f.extend(
            [
                1.0,
                0.0,
                0.5,
                0.5,
                0.0,
                0.0,  # [8..13]
                0.0,
                0.5,
                0.5,
                0.0,  # [14..17]
                0.5,
                0.0,
                1.0,
                0.5,
                0.5,
                0.5,  # [18..23] v7 neutral defaults
                0.5,
                0.0,
                0.5,
                0.5,  # [24..27] v7.1 neutral defaults
            ]
        )
    elif len(f) == 14:
        # v4 (14 features) → pad [14..27]
        f.extend(
            [
                0.0,
                0.5,
                0.5,
                0.0,  # [14..17] v6 defaults
                0.5,
                0.0,
                1.0,
                0.5,
                0.5,
                0.5,  # [18..23] v7 neutral defaults
                0.5,
                0.0,
                0.5,
                0.5,  # [24..27] v7.1 neutral defaults
            ]
        )
    elif len(f) == 18:
        # v6 (18 features) → pad [18..27]
        f.extend(
            [
                0.5,
                0.0,
                1.0,
                0.5,
                0.5,
                0.5,  # [18..23] v7 neutral defaults
                0.5,
                0.0,
                0.5,
                0.5,  # [24..27] v7.1 neutral defaults
            ]
        )
    elif len(f) == 24:
        # v7 (24 features) → pad [24..27] with v7.1 neutral defaults
        f.extend([0.5, 0.0, 0.5, 0.5])

    if len(f) != NUM_TABULAR:
        raise ValueError(
            f"Expected {NUM_TABULAR} tabular features (v7.1), 24 (v7 compat), "
            f"18 (v6 compat), 14 (v4 compat), or 8 (v5 compat); "
            f"got {len(f)}. Required order: {TABULAR_FEATURES}"
        )

    # [0] quality_pct_norm — clamp [0, 1]
    quality_norm = max(0.0, min(1.0, f[0]))

    # [1] volume_ratio — log-scale: min(log1p(raw) / log1p(10), 1.0)
    vol_raw = max(f[1], 0.01)
    vol_norm = min(float(np.log1p(vol_raw) / np.log1p(10.0)), 1.0)

    # [2] atr_pct — ×100 then clamp [0, 1]
    atr_norm = max(0.0, min(1.0, f[2] * 100.0))

    # [3] cvd_delta — clamp [-1, 1]
    cvd_norm = max(-1.0, min(1.0, f[3]))

    # [4] nr7_flag       — passthrough
    # [5] direction_flag — passthrough
    # [6] session_ord    — passthrough [0, 1]
    # [7] london_overlap — passthrough

    # [8] or_range_atr_ratio — clamp(raw, 0, 3) / 3
    or_range_atr = max(0.0, min(3.0, f[8])) / 3.0

    # [9] premarket_range_ratio — clamp(raw, 0, 5) / 5
    pm_ratio = max(0.0, min(5.0, f[9])) / 5.0

    # [10] bar_of_day  — already [0, 1], passthrough with clamp
    bar_of_day = max(0.0, min(1.0, f[10]))

    # [11] day_of_week — already [0, 1], passthrough with clamp
    dow = max(0.0, min(1.0, f[11]))

    # [12] vwap_distance — clamp(raw, -3, 3) / 3  → [-1, 1]
    vwap_dist = max(-3.0, min(3.0, f[12])) / 3.0

    # [13] asset_class_id — already [0, 1], passthrough with clamp
    asset_cls = max(0.0, min(1.0, f[13]))

    # [14] breakout_type_ord — already [0, 1], passthrough with clamp
    bt_ord = max(0.0, min(1.0, f[14]))

    # [15] asset_volatility_class — already [0, 1], passthrough with clamp
    vol_class = max(0.0, min(1.0, f[15]))

    # [16] hour_of_day — already [0, 1], passthrough with clamp
    hour_of_day = max(0.0, min(1.0, f[16]))

    # [17] tp3_atr_mult_norm — already [0, 1], passthrough with clamp
    tp3_norm = max(0.0, min(1.0, f[17]))

    return [
        quality_norm,  # [0]
        vol_norm,  # [1]
        atr_norm,  # [2]
        cvd_norm,  # [3]
        f[4],  # [4] nr7_flag
        f[5],  # [5] direction_flag
        f[6],  # [6] session_ordinal
        f[7],  # [7] london_overlap_flag
        or_range_atr,  # [8]
        pm_ratio,  # [9]
        bar_of_day,  # [10]
        dow,  # [11]
        vwap_dist,  # [12]
        asset_cls,  # [13]
        bt_ord,  # [14] breakout_type_ord
        vol_class,  # [15] asset_volatility_class
        hour_of_day,  # [16] hour_of_day
        tp3_norm,  # [17] tp3_atr_mult_norm
        # ── v7 features (slots 18–23) — passthrough with clamp [0, 1] ────
        max(0.0, min(1.0, f[18])),  # [18] daily_bias_direction
        max(0.0, min(1.0, f[19])),  # [19] daily_bias_confidence
        max(0.0, min(1.0, f[20])),  # [20] prior_day_pattern
        max(0.0, min(1.0, f[21])),  # [21] weekly_range_position
        max(0.0, min(1.0, f[22])),  # [22] monthly_trend_score
        max(0.0, min(1.0, f[23])),  # [23] crypto_momentum_score
        # ── v7.1 sub-features (slots 24–27) — passthrough with clamp ─────
        max(0.0, min(1.0, f[24])),  # [24] breakout_type_category
        max(0.0, min(1.0, f[25])),  # [25] session_overlap_flag
        max(0.0, min(1.0, f[26])),  # [26] atr_trend
        max(0.0, min(1.0, f[27])),  # [27] volume_trend
    ]


def predict_breakout(
    image_path: str,
    tabular_features: Sequence[float],
    model_path: str | None = None,
    threshold: float | None = None,
    session_key: str | None = None,
) -> dict[str, Any] | None:
    """Predict whether a chart snapshot shows a high-quality breakout.

    Args:
        image_path: Path to the PNG chart snapshot.
        tabular_features: List/tuple of 18 floats in TABULAR_FEATURES order
            (v6 contract).  8-feature (v5) and 14-feature (v4) vectors are
            accepted for backward compatibility and zero-padded automatically.

            v6 features:
              [0]  quality_pct_norm      — quality_pct / 100 (0.0–1.0)
              [1]  volume_ratio          — breakout bar vol / 20-bar avg
              [2]  atr_pct               — ATR as fraction of price
              [3]  cvd_delta             — normalised CVD delta (-1 to 1)
              [4]  nr7_flag              — 1.0 if NR7 day, 0.0 otherwise
              [5]  direction_flag        — 1.0 for LONG, 0.0 for SHORT
              [6]  session_ordinal       — Globex day position [0, 1]
              [7]  london_overlap_flag   — 1.0 if 08:00–09:00 ET
              [8]  or_range_atr_ratio    — ORB range / ATR (raw)
              [9]  premarket_range_ratio — premarket range / ORB range (raw)
              [10] bar_of_day            — minutes since Globex open / 1380
              [11] day_of_week           — Mon=0..Fri=4 / 4
              [12] vwap_distance         — (price-vwap) / ATR (raw)
              [13] asset_class_id        — asset class ordinal / 4
              [14] breakout_type_ord     — BreakoutType ordinal / 12
              [15] asset_volatility_class — low=0 / med=0.5 / high=1.0
              [16] hour_of_day           — ET hour / 23
              [17] tp3_atr_mult_norm     — TP3 ATR mult / 5.0

        model_path: Explicit model path (default: latest in models/).
        threshold: Probability threshold for "signal" verdict.
                   When None (default), the per-session threshold from
                   ``SESSION_THRESHOLDS`` is used via ``session_key``.
                   Passing an explicit float overrides the session default.
        session_key: ORBSession.key (e.g. "london", "tokyo", "us").
                     Used to look up the per-session threshold and for
                     logging.  Ignored if *threshold* is explicitly set.

    Returns:
        Dict with:
          - prob: float (0.0–1.0) — probability of clean breakout
          - signal: bool — True if prob >= threshold
          - confidence: str — "high", "medium", or "low"
          - threshold: float — the threshold that was applied
          - session_key: str — which session threshold was used
          - model_path: str — which model was used
        Or None if inference failed.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — cannot run inference")
        return None

    model = _load_model(model_path)
    if model is None:
        return None

    # Resolve the effective threshold — explicit arg wins, then per-session,
    # then global default.
    effective_threshold = threshold if threshold is not None else get_session_threshold(session_key)

    device = next(model.parameters()).device
    transform = get_inference_transform()

    try:
        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        assert transform is not None, "Transform must not be None"
        img_tensor = transform(img).unsqueeze(0).to(device)  # type: ignore[union-attr]  # (1, 3, 224, 224)

        # Normalise tabular features (same transforms as training dataset)
        tab_list = _normalise_tabular_for_inference(tabular_features)
        if len(tab_list) != NUM_TABULAR:
            logger.error("Expected %d tabular features, got %d", NUM_TABULAR, len(tab_list))
            return None

        tab_tensor = torch.tensor([tab_list], dtype=torch.float32).to(device)  # (1, 8)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor, tab_tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=1)
            prob_good = float(probs[0, 1].item())

        # Confidence bucketing — relative to the effective threshold so that
        # a "high" confidence call means the same quality bar regardless of
        # which session we're in.
        if prob_good >= effective_threshold + 0.08:
            confidence = "high"
        elif prob_good >= effective_threshold - 0.04:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "prob": round(prob_good, 4),
            "signal": prob_good >= effective_threshold,
            "confidence": confidence,
            "threshold": effective_threshold,
            "session_key": session_key or "",
            "model_path": _cached_model_path or "",
        }

    except Exception as exc:
        logger.error("Inference failed for %s: %s", image_path, exc, exc_info=True)
        return None


def predict_breakout_batch(
    image_paths: Sequence[str],
    tabular_features_batch: Sequence[Sequence[float]],
    model_path: str | None = None,
    threshold: float | None = None,
    session_key: str | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any] | None]:
    """Batch inference for multiple chart snapshots.

    More efficient than calling ``predict_breakout`` in a loop because it
    batches the GPU forward passes.

    Args:
        image_paths: List of PNG paths.
        tabular_features_batch: List of tabular feature vectors (one per image).
        model_path: Explicit model path (default: latest).
        threshold: Signal threshold.  When None (default), the per-session
                   threshold from ``SESSION_THRESHOLDS`` is used via
                   ``session_key``.  Passing an explicit float overrides it.
        session_key: ORBSession.key (e.g. "london", "tokyo", "us") used to
                     look up the per-session threshold.  Ignored if
                     *threshold* is explicitly set.
        batch_size: Max images per GPU forward pass.

    Returns:
        List of result dicts (same format as predict_breakout), or None entries
        for images that failed to load.
    """
    if not _TORCH_AVAILABLE:
        return [None] * len(image_paths)

    if len(image_paths) != len(tabular_features_batch):
        logger.error("Mismatched lengths: %d images vs %d tabular", len(image_paths), len(tabular_features_batch))
        return [None] * len(image_paths)

    model = _load_model(model_path)
    if model is None:
        return [None] * len(image_paths)

    # Resolve the effective threshold once for the whole batch.
    effective_threshold = threshold if threshold is not None else get_session_threshold(session_key)

    device = next(model.parameters()).device
    transform = get_inference_transform()
    results: list[dict[str, Any] | None] = [None] * len(image_paths)

    # Process in batches
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_indices = list(range(batch_start, batch_end))

        img_tensors = []
        tab_tensors = []
        valid_indices = []

        for i in batch_indices:
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                assert transform is not None
                img_t = transform(img)
                tab_list = _normalise_tabular_for_inference(tabular_features_batch[i])
                if len(tab_list) != NUM_TABULAR:
                    continue
                tab_t = torch.tensor(tab_list, dtype=torch.float32)

                img_tensors.append(img_t)
                tab_tensors.append(tab_t)
                valid_indices.append(i)
            except Exception as exc:
                logger.debug("Failed to load image %s: %s", image_paths[i], exc)

        if not valid_indices:
            continue

        img_batch = torch.stack(img_tensors).to(device)  # (B, 3, 224, 224)
        tab_batch = torch.stack(tab_tensors).to(device)  # (B, 6)

        with torch.no_grad():
            logits = model(img_batch, tab_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # (B,)

        for j, global_idx in enumerate(valid_indices):
            prob_good = float(probs[j].item())
            # Relative confidence bucketing — mirrors predict_breakout so
            # "high"/"medium"/"low" mean the same quality bar across sessions.
            if prob_good >= effective_threshold + 0.08:
                confidence = "high"
            elif prob_good >= effective_threshold - 0.04:
                confidence = "medium"
            else:
                confidence = "low"

            results[global_idx] = {
                "prob": round(prob_good, 4),
                "signal": prob_good >= effective_threshold,
                "confidence": confidence,
                "threshold": effective_threshold,
                "session_key": session_key or "",
                "model_path": _cached_model_path or "",
            }

    return results


# ---------------------------------------------------------------------------
# Model info / diagnostics
# ---------------------------------------------------------------------------


def generate_feature_contract(output_path: str | None = None) -> dict[str, Any]:
    """Generate and optionally write the ``feature_contract.json`` v6 file.

    The contract encodes every parameter needed by consumers (engine, NT8
    C# OrbCnnPredictor, rb trainer) to correctly prepare the tabular feature
    vector and interpret model outputs.  The v6 schema matches the 18-feature
    vector built by BreakoutStrategy.PrepareCnnTabular() in C#.

    Structure::

        {
          "version":          6,
          "num_tabular":      18,
          "tabular_features": [...],
          "default_threshold": 0.82,
          "image_size":       224,
          "imagenet_mean":    [...],
          "imagenet_std":     [...],
          "session_thresholds":   {...},
          "session_ordinals":     {...},
          "asset_class_map":      {...},
          "breakout_type_ordinals": {...},
          "asset_volatility_classes": {...},
          "breakout_types":       {...},
          "generated_at":     "<ISO timestamp>",
        }

    Args:
        output_path: If given, write the JSON to this path (creates parent
            directories as needed).  If ``None``, only return the dict.

    Returns:
        The contract as a Python dict (always returned regardless of
        ``output_path``).
    """
    import json as _json
    from datetime import datetime as _dt

    # Import breakout type helpers for the full contract
    try:
        from lib.core.breakout_types import to_feature_contract_dict as _bt_dict

        _breakout_types_section = _bt_dict()
    except ImportError:
        _breakout_types_section = {}

    contract: dict[str, Any] = {
        "version": FEATURE_CONTRACT_VERSION,
        "num_tabular": NUM_TABULAR,
        "tabular_features": TABULAR_FEATURES,
        "default_threshold": DEFAULT_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        # Per-session CNN thresholds — must match C# CnnSessionThresholds
        "session_thresholds": SESSION_THRESHOLDS,
        # Per-session ordinals — must match C# CnnSessionThresholds._ordinals
        "session_ordinals": SESSION_ORDINAL,
        # Asset class map — must match C# GetAssetClassNorm()
        "asset_class_map": ASSET_CLASS_ORDINALS,
        # v6: breakout type ordinals — must match C# BreakoutType enum
        "breakout_type_ordinals": BREAKOUT_TYPE_ORDINALS,
        # v6: asset volatility classes — must match C# GetVolatilityClass()
        "asset_volatility_classes": ASSET_VOLATILITY_CLASS,
        # v6: full breakout type configs (ordinals, bracket params, box styles)
        "breakout_types": _breakout_types_section,
        # v7: daily strategy layer feature descriptions
        "v7_feature_descriptions": {
            "daily_bias_direction": "Daily bias from bias_analyzer: SHORT=0.0, NEUTRAL=0.5, LONG=1.0",
            "daily_bias_confidence": "Confidence of daily bias analysis, 0.0-1.0 scalar",
            "prior_day_pattern": (
                "Yesterday's candle pattern ordinal / 9: inside=0, doji=1, "
                "engulfing_bull=2, engulfing_bear=3, hammer=4, shooting_star=5, "
                "strong_close_up=6, strong_close_down=7, outside_day=8, neutral=9"
            ),
            "weekly_range_position": "Price position within prior week's H/L range: 0.0=at low, 1.0=at high",
            "monthly_trend_score": "Normalised slope of 20-day EMA on daily bars: [-1,+1] mapped to [0,1]",
            "crypto_momentum_score": "Crypto momentum leading indicator: [-1,+1] mapped to [0,1], 0.5=neutral",
        },
        # v7.1: Phase 4B sub-feature decomposition descriptions
        "v71_sub_feature_descriptions": {
            "breakout_type_category": (
                "Coarse grouping of breakout_type_ord: time-based=0.0 (ORB, Asian, IB), "
                "range-based=0.5 (PDR, Weekly, Monthly, VA, Inside, Gap, Pivot, Fib), "
                "squeeze-based=1.0 (Consolidation, BollingerSqueeze)"
            ),
            "session_overlap_flag": (
                "1.0 if the bar is in the London+NY overlap window (08:00-12:00 ET), "
                "0.0 otherwise. Captures the highest-volume intraday window."
            ),
            "atr_trend": (
                "ATR direction over last 10 bars: 1.0=expanding (volatility increasing), "
                "0.0=contracting (volatility decreasing), 0.5=flat. "
                "Enriches atr_pct with trend context."
            ),
            "volume_trend": (
                "5-bar volume slope normalised to [0,1]: 1.0=rising sharply, "
                "0.5=flat, 0.0=declining sharply. Enriches volume_ratio with "
                "trend context - rising volume into breakout is bullish for continuation."
            ),
        },
        # v7.1: breakout type category mapping
        "breakout_type_categories": {
            "time_based": ["ORB", "Asian", "InitialBalance"],
            "range_based": [
                "PrevDay",
                "Weekly",
                "Monthly",
                "ValueArea",
                "InsideDay",
                "GapRejection",
                "PivotPoints",
                "Fibonacci",
            ],
            "squeeze_based": ["Consolidation", "BollingerSqueeze"],
        },
        "generated_at": _dt.now(tz=__import__("datetime").timezone.utc).isoformat(),
    }

    if output_path:
        import os as _os

        _os.makedirs(_os.path.dirname(_os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            _json.dump(contract, fh, indent=2)

    return contract


def export_onnx_model(
    pt_path: str,
    onnx_path: str,
    opset: int = 17,
    image_size: int = IMAGE_SIZE,
    num_tabular: int = NUM_TABULAR,
) -> str:
    """Export a trained HybridBreakoutCNN ``.pt`` checkpoint to ONNX format.

    The exported graph accepts two named inputs:

    * ``image``   — float32 (1, 3, 224, 224) pre-processed with ImageNet stats.
    * ``tabular`` — float32 (1, NUM_TABULAR) v6 normalised feature vector (18 features).

    Output:

    * ``logits``  — float32 (1, 2).  Apply softmax(axis=1)[0,1] → P(good).

    The output ONNX file is validated with onnx.checker before returning.
    Dynamic batch axes are exported so NT8's OrbCnnPredictor (batch_size=1)
    and any batch-inference caller both work without re-export.

    Args:
        pt_path:     Path to the ``.pt`` state-dict checkpoint.
        onnx_path:   Destination ``.onnx`` file (parent dirs created).
        opset:       ONNX opset version (default 17).
        image_size:  Spatial input size (default 224).
        num_tabular: Tabular feature count (default NUM_TABULAR=14).

    Returns:
        Absolute path to the written ``.onnx`` file.

    Raises:
        RuntimeError: torch or onnx not installed, or checkpoint load failed.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed — cannot export ONNX model.  Install with: pip install torch torchvision"
        )

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("The 'onnx' package is required.  Install with: pip install onnx onnxruntime") from exc

    import os as _os

    _os.makedirs(_os.path.dirname(_os.path.abspath(onnx_path)), exist_ok=True)

    device = torch.device("cpu")  # always export on CPU for portability

    # ── Load checkpoint ────────────────────────────────────────────────────
    logger.info("ONNX export: loading checkpoint %s", pt_path)
    try:
        state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(pt_path, map_location=device)  # type: ignore[call-overload]

    # ── Detect num_tabular from checkpoint ────────────────────────────────
    tab_key = "tabular_head.0.weight"
    if tab_key in state_dict:
        detected = int(state_dict[tab_key].shape[1])
        if detected != num_tabular:
            logger.warning(
                "ONNX export: checkpoint tabular dim=%d, expected %d — using %d",
                detected,
                num_tabular,
                detected,
            )
            num_tabular = detected

    # ── Instantiate & load model ──────────────────────────────────────────
    model = HybridBreakoutCNN(num_tabular=num_tabular, pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info("ONNX export: model loaded  num_tabular=%d", num_tabular)

    # ── Thin export wrapper (clean two-input graph) ────────────────────────
    class _ExportWrapper(nn.Module):
        def __init__(self, inner: HybridBreakoutCNN) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
            return self.inner(image, tabular)

    export_model: nn.Module = _ExportWrapper(model)
    export_model.eval()

    # ── Dummy inputs ──────────────────────────────────────────────────────
    dummy_image = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    dummy_tabular = torch.zeros(1, num_tabular, dtype=torch.float32)

    # ── Export ────────────────────────────────────────────────────────────
    logger.info("ONNX export: tracing → %s  (opset=%d)", onnx_path, opset)

    # dynamo=False forces the legacy TorchScript trace exporter so that
    # onnxscript (an optional PyTorch 2.4+ dependency) is never required.
    # The dynamo path became the default in PyTorch 2.4 and pulls in
    # onnxscript internally — but that package is not installed in the
    # trainer image and is not needed for a simple two-input CNN export.
    _export_kwargs: dict[str, Any] = dict(
        opset_version=opset,
        input_names=["image", "tabular"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "tabular": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # PyTorch >= 2.5 exposes dynamo= as an explicit kwarg; 2.4 does not
    # but still respects the internal _exporter_type env var.  Try the
    # kwarg first; fall back gracefully on older builds.
    try:
        _export_kwargs["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(export_model, (dummy_image, dummy_tabular), onnx_path, **_export_kwargs)
    except TypeError:
        # dynamo= kwarg not recognised (PyTorch < 2.5) — drop it and
        # re-run; the legacy exporter is the only path available anyway.
        del _export_kwargs["dynamo"]
        with torch.no_grad():
            torch.onnx.export(export_model, (dummy_image, dummy_tabular), onnx_path, **_export_kwargs)

    # ── Validate ──────────────────────────────────────────────────────────
    import onnx as _onnx

    onnx_model = _onnx.load(onnx_path)
    _onnx.checker.check_model(onnx_model)

    size_mb = _os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(
        "ONNX export complete: %s  (%.1f MB, opset=%d, num_tabular=%d)",
        onnx_path,
        size_mb,
        opset,
        num_tabular,
    )

    return _os.path.abspath(onnx_path)


def get_type_embedding_weights(model_path: str | None = None) -> dict[str, Any] | None:
    """Return the learned BreakoutType embedding matrix from a checkpoint.

    Returns None for v4 models (no type embedding) or if torch is unavailable.
    Kept for backward compatibility with callers that checked for embeddings
    in older v6 checkpoints.
    """
    if not _TORCH_AVAILABLE:
        return None

    mp = model_path or _find_best_model()
    if not mp or not os.path.isfile(mp):
        return None

    try:
        try:
            sd = torch.load(mp, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(mp, map_location="cpu")  # type: ignore[call-overload]

        if "type_embedding.weight" not in sd:
            # v4 models have no type embedding — this is expected
            return None

        emb_weight = sd["type_embedding.weight"].numpy()
        _ordinal_to_name = {int(round(v * 12)): k for k, v in BREAKOUT_TYPE_ORDINALS.items()}
        result: dict[str, Any] = {}
        for i, vec in enumerate(emb_weight):
            name = _ordinal_to_name.get(i, f"type_{i}")
            result[name] = vec.tolist()
        return result
    except Exception as exc:
        logger.warning("Failed to extract type embedding weights: %s", exc)
        return None


def model_info(model_path: str | None = None) -> dict[str, Any]:
    """Return diagnostic information about the current or specified model.

    Useful for the dashboard / health checks.
    """
    if not _TORCH_AVAILABLE:
        return {"available": False, "error": "PyTorch not installed"}

    path = model_path or _find_latest_model()
    if path is None:
        return {"available": False, "error": "No trained model found"}

    try:
        file_stat = os.stat(path)
        size_mb = file_stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    except OSError:
        size_mb = 0.0
        modified = ""

    info = {
        "available": True,
        "model_path": path,
        "size_mb": round(size_mb, 1),
        "modified": modified,
        "device": get_device(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "image_size": IMAGE_SIZE,
        "num_tabular_features": NUM_TABULAR,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "tabular_features": TABULAR_FEATURES,
        "default_threshold": DEFAULT_THRESHOLD,
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    return info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    """Simple CLI for training and inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Breakout CNN — Train or Predict")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_parser = sub.add_parser("train", help="Train the CNN model")
    train_parser.add_argument("--csv", required=True, help="Path to training CSV")
    train_parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--freeze-epochs", type=int, default=2)
    train_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--image-root", default=None)
    train_parser.add_argument(
        "--type-embedding",
        action="store_true",
        default=False,
        help="Legacy flag — ignored for v6 models (no type embedding used).",
    )
    train_parser.add_argument("--workers", type=int, default=4)

    # Predict
    pred_parser = sub.add_parser("predict", help="Predict on a single image")
    pred_parser.add_argument("--image", required=True, help="Path to chart PNG")
    pred_parser.add_argument(
        "--features",
        nargs=NUM_TABULAR,
        type=float,
        required=True,
        help=f"Tabular features: {', '.join(TABULAR_FEATURES)}",
    )
    pred_parser.add_argument("--model", default=None, help="Model path (default: latest)")
    pred_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    # Info
    sub.add_parser("info", help="Show model info")

    # Embedding inspection
    sub.add_parser(
        "embedding",
        help="Print learned BreakoutType embedding weights from the current champion model",
    )

    # Contract
    contract_parser = sub.add_parser("contract", help="Generate feature_contract.json v6 (18 features)")
    contract_parser.add_argument(
        "--output",
        default="feature_contract.json",
        help="Output path for feature_contract.json (default: ./feature_contract.json)",
    )
    contract_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the contract to stdout without writing a file",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.command == "train":
        result = train_model(
            data_csv=args.csv,
            val_csv=args.val_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            model_dir=args.model_dir,
            image_root=args.image_root,
            num_workers=args.workers,
        )
        if result:
            print(f"\nModel saved to: {result.model_path}")
            print(f"Best epoch: {result.best_epoch}  |  Epochs trained: {result.epochs_trained}")
        else:
            print("\nTraining failed")
            exit(1)

    elif args.command == "predict":
        result = predict_breakout(
            image_path=args.image,
            tabular_features=args.features,
            model_path=args.model,
            threshold=args.threshold,
        )
        if result:
            signal_str = "SIGNAL" if result["signal"] else "NO SIGNAL"
            print(
                f"\n{signal_str} — P(good breakout) = {result['prob']:.4f} "
                f"(threshold={result['threshold']}, confidence={result['confidence']})"
            )
        else:
            print("\nPrediction failed")
            exit(1)

    elif args.command == "embedding":
        weights = get_type_embedding_weights()
        if weights is None:
            print("No type embedding found in checkpoint (model not trained with --type-embedding).")
        else:
            import json as _json

            print(_json.dumps(weights, indent=2))

    elif args.command == "info":
        info = model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

    elif args.command == "contract":
        import json as _json

        if args.print_only:
            contract = generate_feature_contract(output_path=None)
            print(_json.dumps(contract, indent=2))
        else:
            contract = generate_feature_contract(output_path=args.output)
            print(f"✅ feature_contract.json v{contract['version']} written to: {args.output}")
            print(f"   tabular features : {contract['num_tabular']}")
            print(f"   breakout types   : {len(contract.get('breakout_types', {}))}")
            print(f"   sessions         : {len(contract.get('session_thresholds', {}))}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
