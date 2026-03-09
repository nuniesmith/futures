"""
Opening Range Breakout (ORB) Detector — Multi-Session, Full 24h Coverage
=========================================================================
Detects opening range breakouts across **nine** trading sessions spanning
the full 24-hour futures cycle, starting at the CME Globex re-open at
18:00 ET each day.

Session order (by ET clock, new-day starting 18:00 ET):

  1. **CME Globex Open**  18:00–18:30 ET  (prev-day; after settlement break)
     First bars of the new Globex trading day. Clean overnight anchor
     for all CME micro contracts.  wraps_midnight=True.

  2. **Sydney / Asia Open**  18:30–19:00 ET  (prev-day; ASX/SFE open ~19:00 ET)
     Thin metals, energy, and MBT.  wraps_midnight=True.

  3. **Tokyo Open**  19:00–19:30 ET  (prev-day; TSE open 19:00 ET EST / 09:00 JST)
     Narrow-range session. Strongest on metals and JPY-correlated FX.
     wraps_midnight=True.

  4. **Shanghai/Hong Kong Open**  21:00–21:30 ET  (prev-day; CSI/HKEX 09:30 CST)
     Copper (MHG/HG) and gold sentiment driver from SHFE.
     wraps_midnight=True.

  5. **Frankfurt / Xetra Open**  03:00–03:30 ET  (08:00–08:30 CET)
     Pre-London institutional flow; sets European equity and EUR/USD tone.
     Strongest on 6E, MES/MNQ (DAX correlation), MGC.

  6. **London Open**  03:00–03:30 ET  (08:00–08:30 UTC)  ← PRIMARY
     Highest-conviction session. Institutional order flow drives the
     daily range for metals, energy, FX futures, and indices.

  7. **London–NY Crossover**  08:00–08:30 ET  (13:00–13:30 UTC)
     Overlap window — highest intraday volume and tightest spreads.
     Best assets: 6E, MES/MNQ, MGC.

  8. **US Equity Open**  09:30–10:00 ET
     Classic Toby Crabel ORB for MES/MNQ.  Also covers MGC.

  9. **CME Settlement / Late Session**  14:00–14:30 ET
     Metals/energy settlement window.  Gold (MGC) and crude (MCL)
     typically see directional resolution before the 17:00 close.

DST Handling
------------
All ``or_start`` / ``or_end`` / ``scan_end`` times are stored in **ET wall-clock**
time (America/New_York).  ``ZoneInfo("America/New_York")`` handles EST↔EDT
transitions automatically, so the UTC equivalent shifts by 1 hour during
summer (EDT = UTC-4) vs winter (EST = UTC-5).  No manual offset needed.

The ``get_session_for_utc()`` helper converts a UTC datetime to ET and
checks membership in the appropriate session window.

Quality filters (applied inside detect_opening_range_breakout):
  - **Depth filter**: breakout bar close must penetrate the OR level
    by at least ``min_depth_atr_pct`` × ATR (default 0.15×).
  - **Body-ratio filter**: breakout bar body ≥ ``min_body_ratio`` of range.
  - **OR-size cap**: if OR range > ``max_or_atr_ratio`` × ATR, skip.
  - **OR-size floor**: if OR range < ``min_or_atr_ratio`` × ATR, skip.

Public API:
    result = detect_opening_range_breakout(bars_1m, symbol="MGC", session=LONDON_SESSION)
    results = detect_all_sessions(bars_1m, symbol="MGC")
    publish_orb_alert(result)   → push to Redis for SSE/dashboard

    # DST-safe session lookup
    session = get_session_for_utc(datetime.now(UTC))
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
_UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# Session Definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ORBSession:
    """Definition of an Opening Range session window.

    Each session has its own time window, ATR parameters, Redis keys,
    and quality-gate thresholds for depth, body, and OR-size checks.
    Frozen so instances are hashable and can be used as dict keys.

    All times (or_start, or_end, scan_end) are **ET wall-clock** times
    (America/New_York).  DST transitions (EST↔EDT) are handled automatically
    by Python's ZoneInfo — no manual UTC offset needed.
    """

    name: str  # Human-readable name
    key: str  # Short key for Redis/logs ("london", "us", "tokyo", …)
    or_start: dt_time  # Opening range start (ET wall-clock)
    or_end: dt_time  # Opening range end (ET wall-clock)
    scan_end: dt_time  # Stop scanning for breakouts after this time (ET)
    atr_period: int = 14  # ATR look-back period
    breakout_multiplier: float = 0.5  # ATR multiplier for breakout threshold
    min_bars: int = 5  # Minimum bars required in OR window
    max_bars: int = 35  # Maximum expected bars in OR window
    description: str = ""

    # --- Quality gate thresholds ---
    # Depth: breakout bar close must clear the OR level by at least
    # this fraction of ATR (e.g. 0.15 = 15% of ATR beyond the level).
    # Set to 0.0 to disable.
    min_depth_atr_pct: float = 0.15

    # Body ratio: breakout candle body / total bar range.
    # A value of 0.55 means the close-open body must be ≥ 55% of the
    # high-low range — eliminates doji/shooting-star breakouts.
    # Set to 0.0 to disable.
    min_body_ratio: float = 0.55

    # OR-size cap: if OR range > max_or_atr_ratio × ATR, skip — range
    # is too wide (news spike, thin market) for reliable ORB.
    # Set to 0.0 to disable.
    max_or_atr_ratio: float = 1.8

    # OR-size floor: if OR range < min_or_atr_ratio × ATR, skip — too
    # narrow / compressed (usually pre-event squeeze).
    # Set to 0.0 to disable.
    min_or_atr_ratio: float = 0.05

    # Whether this session wraps past midnight (all overnight sessions
    # 18:00–03:00 ET start in the previous calendar day ET).  When True,
    # bar filtering must look back into the previous day's bars.
    wraps_midnight: bool = False

    # --- Dataset generation ---
    # Whether to include this session in CNN dataset generation.
    # Set False for very thin sessions where signal quality is too low
    # to produce useful training labels.
    include_in_dataset: bool = True


# ===========================================================================
# Session Definitions — Full 24-Hour Cycle (ET wall-clock, DST-aware)
# ===========================================================================
# The Globex trading day starts at 18:00 ET (after the 17:00–18:00
# settlement break).  Sessions below are listed in chronological order
# within that 18:00 ET → next-day 17:00 ET cycle.
#
# DST note: all times are ET wall-clock.  In summer (EDT, UTC-4) all
# UTC equivalents shift 1 hour earlier vs winter (EST, UTC-5).  Python's
# ZoneInfo("America/New_York") handles this automatically — no adjustments
# needed here.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. CME Globex Re-Open  18:00–18:30 ET  ← START OF GLOBEX DAY
# Futures exchange re-opens after the 17:00–18:00 ET daily settlement
# break. First bars of the new trading day — clean overnight anchor for
# all CME micro products.  wraps_midnight=True.
# ---------------------------------------------------------------------------
CME_OPEN_SESSION = ORBSession(
    name="CME Globex Open",
    key="cme",
    or_start=dt_time(18, 0),
    or_end=dt_time(18, 30),
    scan_end=dt_time(20, 0),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=3,
    max_bars=35,
    description=(
        "CME Globex re-open after settlement break (18:00–18:30 ET / 23:00–23:30 UTC EST | 22:00–22:30 UTC EDT)"
    ),
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.6,
    min_or_atr_ratio=0.04,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 2. Sydney / ASX Open  18:30–19:00 ET  (ASX opens ~19:00 ET EST)
# Australian Securities Exchange open; thin metals + MBT in overnight
# Globex.  wraps_midnight=True.
# ---------------------------------------------------------------------------
SYDNEY_SESSION = ORBSession(
    name="Sydney Open",
    key="sydney",
    or_start=dt_time(18, 30),
    or_end=dt_time(19, 0),
    scan_end=dt_time(20, 30),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Sydney / ASX open (18:30–19:00 ET / 23:30–00:00 UTC EST | 22:30–23:00 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 3. Tokyo / TSE Open  19:00–19:30 ET  (09:00 JST = 19:00 ET EST / 18:00 ET EDT)
# Tokyo Stock Exchange open.  Narrow-range session; strongest for metals
# and JPY/AUD-correlated FX futures.  wraps_midnight=True.
# ---------------------------------------------------------------------------
TOKYO_SESSION = ORBSession(
    name="Tokyo Open",
    key="tokyo",
    or_start=dt_time(19, 0),
    or_end=dt_time(19, 30),
    scan_end=dt_time(21, 0),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Tokyo / TSE open (19:00–19:30 ET / 00:00–00:30 UTC EST | 23:00–23:30 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.4,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 4. Shanghai / Hong Kong Open  21:00–21:30 ET
# CSI 300 / HKEX open (09:30 CST / HKT).  Copper (MHG) and gold (MGC)
# sentiment driver via SHFE open-price auction.  wraps_midnight=True.
# ---------------------------------------------------------------------------
SHANGHAI_SESSION = ORBSession(
    name="Shanghai/HK Open",
    key="shanghai",
    or_start=dt_time(21, 0),
    or_end=dt_time(21, 30),
    scan_end=dt_time(23, 0),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Shanghai/HK open — CSI 300 / HKEX (21:00–21:30 ET / 02:00–02:30 UTC EST | 01:00–01:30 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 5. Frankfurt / Xetra Open  03:00–03:30 ET  (08:00–08:30 CET / 09:00 CEST)
# Pre-London institutional flow; sets European equity and EUR/USD tone.
# Fires at the same ET time as London open — treated as a separate
# session key for asset filtering (DAX-correlated symbols).
# ---------------------------------------------------------------------------
FRANKFURT_SESSION = ORBSession(
    name="Frankfurt/Xetra Open",
    key="frankfurt",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    scan_end=dt_time(4, 30),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=4,
    max_bars=35,
    description=("Frankfurt / Xetra open (03:00–03:30 ET / 08:00–08:30 UTC EST | 07:00–07:30 UTC EDT)"),
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.7,
    min_or_atr_ratio=0.04,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 6. London Open  03:00–03:30 ET  (08:00–08:30 UTC)  ← PRIMARY SESSION
# Highest-conviction session. Institutional order flow drives the daily
# range for metals, energy, FX futures, and indices.
# ---------------------------------------------------------------------------
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
    description=("London open session (03:00–03:30 ET / 08:00–08:30 UTC EST | 07:00–07:30 UTC EDT)"),
    min_depth_atr_pct=0.15,
    min_body_ratio=0.55,
    max_or_atr_ratio=1.8,
    min_or_atr_ratio=0.05,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 7. London–NY Crossover  08:00–08:30 ET  (13:00–13:30 UTC)
# Both exchanges fully active. Highest intraday volume, tightest spreads.
# Best assets: 6E (EUR), MES, MNQ, MGC.
# ---------------------------------------------------------------------------
LONDON_NY_SESSION = ORBSession(
    name="London-NY Crossover",
    key="london_ny",
    or_start=dt_time(8, 0),
    or_end=dt_time(8, 30),
    scan_end=dt_time(10, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description=("London-NY crossover (08:00–08:30 ET / 13:00–13:30 UTC EST | 12:00–12:30 UTC EDT)"),
    min_depth_atr_pct=0.18,
    min_body_ratio=0.58,
    max_or_atr_ratio=2.0,
    min_or_atr_ratio=0.06,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 8. US Equity Open  09:30–10:00 ET
# Classic Toby Crabel ORB for MES/MNQ.  Also covers MGC during the
# gold-index correlation window.
# ---------------------------------------------------------------------------
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
    min_depth_atr_pct=0.15,
    min_body_ratio=0.55,
    max_or_atr_ratio=1.8,
    min_or_atr_ratio=0.05,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 9. CME Settlement / Late Session  14:00–14:30 ET
# Metals and energy settlement window.  Gold (MGC) and crude (MCL) often
# see directional resolution and range extension before the 17:00 close.
# ---------------------------------------------------------------------------
CME_SETTLEMENT_SESSION = ORBSession(
    name="CME Settlement",
    key="cme_settle",
    or_start=dt_time(14, 0),
    or_end=dt_time(14, 30),
    scan_end=dt_time(15, 30),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=3,
    max_bars=35,
    description="CME metals/energy settlement (14:00–14:30 ET)",
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.7,
    min_or_atr_ratio=0.04,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# All sessions — chronological order within the Globex day (18:00 ET start)
# ---------------------------------------------------------------------------
ORB_SESSIONS: list[ORBSession] = [
    CME_OPEN_SESSION,  # 18:00 ET  — start of Globex day
    SYDNEY_SESSION,  # 18:30 ET
    TOKYO_SESSION,  # 19:00 ET
    SHANGHAI_SESSION,  # 21:00 ET
    FRANKFURT_SESSION,  # 03:00 ET  — pre-London
    LONDON_SESSION,  # 03:00 ET  — primary
    LONDON_NY_SESSION,  # 08:00 ET
    US_SESSION,  # 09:30 ET
    CME_SETTLEMENT_SESSION,  # 14:00 ET
]

# Convenience lookup: session key → ORBSession
SESSION_BY_KEY: dict[str, ORBSession] = {s.key: s for s in ORB_SESSIONS}

# ---------------------------------------------------------------------------
# Per-session asset focus lists
# ---------------------------------------------------------------------------
# Maps session key → list of Yahoo tickers relevant for that session.
# The ORB check loop filters assets to this list per session, avoiding e.g.
# checking MES during Tokyo (near-zero volume) or 6E during US Equity Open
# (FX already moved 5 hours earlier).
#
# Ticker values must match ASSETS dict in lib/core/models.py.
#
# Extended symbol set (all micro CME contracts + FX futures):
#   MGC=F  Micro Gold          MES=F  Micro S&P 500
#   MCL=F  Micro Crude Oil     MNQ=F  Micro Nasdaq-100
#   MHG=F  Micro Copper        M2K=F  Micro Russell 2000
#   SIL=F  Micro Silver        MYM=F  Micro Dow Jones
#   MBT=F  Micro Bitcoin       6E=F   Euro FX
#   6B=F   British Pound       6J=F   Japanese Yen
#   6A=F   Australian Dollar   6C=F   Canadian Dollar
SESSION_ASSETS: dict[str, list[str]] = {
    # CME Globex re-open (18:00 ET): all CME micros — new trading day starts here.
    # FX included because overnight gaps in 6E/6J drive early direction.
    "cme": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
        "MBT=F",
    ],
    # Sydney / ASX (18:30 ET): thin overnight; metals, energy, AUD-correlated FX, MBT.
    "sydney": [
        "MGC=F",
        "MCL=F",
        "SIL=F",
        "6A=F",
        "6J=F",
        "MBT=F",
    ],
    # Tokyo / TSE (19:00 ET): metals, JPY/AUD FX, thin index futures.
    "tokyo": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "6J=F",
        "6A=F",
    ],
    # Shanghai / HK (21:00 ET): copper and gold dominant; CNH-proxy via 6J.
    "shanghai": [
        "MGC=F",
        "MHG=F",
        "MCL=F",
        "SIL=F",
        "6J=F",
    ],
    # Frankfurt / Xetra (03:00 ET): EUR FX, DAX-correlated index futures, metals.
    "frankfurt": [
        "MGC=F",
        "MCL=F",
        "MES=F",
        "MNQ=F",
        "MYM=F",
        "6E=F",
        "6B=F",
    ],
    # London Open (03:00 ET): primary session — all major CME contracts + FX.
    "london": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
    ],
    # London-NY Crossover (08:00 ET): highest-conviction; full universe.
    "london_ny": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
        "MBT=F",
    ],
    # US Equity Open (09:30 ET): index futures primary; gold correlation window.
    "us": [
        "MGC=F",
        "MCL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
    ],
    # CME Settlement (14:00 ET): metals and energy resolution before close.
    "cme_settle": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
    ],
}


# ---------------------------------------------------------------------------
# Crypto-specific ORB sessions
# ---------------------------------------------------------------------------
# Crypto markets are 24/7. The two highest-volume windows (by on-chain and
# exchange data) are:
#   • UTC 00:00–00:30 = 19:00–19:30 ET EST / 20:00–20:30 ET EDT
#     → Coincides with the Asia open auction; BTC/ETH see a reliable
#       volume surge as the new UTC calendar day begins.
#   • UTC 12:00–12:30 = 07:00–07:30 ET EST / 08:00–08:30 ET EDT
#     → London morning session for crypto; overlaps with European
#       institutional flow and pre-US-open positioning.
#
# These sessions use wider ATR thresholds (breakout_multiplier=0.65) and a
# looser OR-size cap (max_or_atr_ratio=2.5) to account for crypto's higher
# intraday volatility relative to CME micro futures.
# Both sessions are marked wraps_midnight=False because the ET window is
# always on the same calendar day.
# ---------------------------------------------------------------------------

# UTC 00:00 ≡ 19:00 ET EST / 20:00 ET EDT
# We store the ET wall-clock time. ZoneInfo handles the EST/EDT shift.
CRYPTO_UTC_MIDNIGHT_SESSION = ORBSession(
    name="Crypto UTC Midnight",
    key="crypto_utc0",
    or_start=dt_time(19, 0),  # 19:00 ET (EST=UTC-5: 00:00 UTC)
    or_end=dt_time(19, 30),
    scan_end=dt_time(21, 0),
    atr_period=14,
    breakout_multiplier=0.65,  # wider for crypto volatility
    min_bars=3,
    max_bars=35,
    description=("Crypto UTC midnight session (19:00–19:30 ET EST / 20:00–20:30 ET EDT = 00:00–00:30 UTC)"),
    min_depth_atr_pct=0.10,  # looser depth gate for crypto
    min_body_ratio=0.45,  # looser body gate for crypto wicks
    max_or_atr_ratio=2.5,  # crypto can gap wide
    min_or_atr_ratio=0.03,
    wraps_midnight=True,  # 19:00 ET is in the previous calendar day
    include_in_dataset=True,
)

# UTC 12:00 ≡ 07:00 ET EST / 08:00 ET EDT
CRYPTO_UTC_NOON_SESSION = ORBSession(
    name="Crypto UTC Noon",
    key="crypto_utc12",
    or_start=dt_time(7, 0),  # 07:00 ET (EST=UTC-5: 12:00 UTC)
    or_end=dt_time(7, 30),
    scan_end=dt_time(9, 0),
    atr_period=14,
    breakout_multiplier=0.65,
    min_bars=3,
    max_bars=35,
    description=("Crypto UTC noon session (07:00–07:30 ET EST / 08:00–08:30 ET EDT = 12:00–12:30 UTC)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.45,
    max_or_atr_ratio=2.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# Per-symbol parameter overrides for crypto ORB detection
# ---------------------------------------------------------------------------
# Crypto assets have much larger ATR values (BTC ~$1000+/day) and wider
# intraday ranges than CME micro futures.  These per-symbol multipliers
# are applied inside detect_opening_range_breakout() when the caller
# passes the symbol name.
#
# Structure: {ticker: {"breakout_multiplier": float, "max_or_atr_ratio": float,
#                       "min_depth_atr_pct": float, "min_body_ratio": float}}
# ---------------------------------------------------------------------------
CRYPTO_SYMBOL_OVERRIDES: dict[str, dict[str, float]] = {
    # Bitcoin — highest volatility; BTC ATR is typically 1-3% of price.
    # Wider multiplier and OR-size cap to avoid false-positive rejections.
    "KRAKEN:XBT/USD": {
        "breakout_multiplier": 0.70,
        "max_or_atr_ratio": 3.0,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.08,
        "min_body_ratio": 0.40,
    },
    # Ethereum — slightly tighter than BTC
    "KRAKEN:ETH/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    # Solana — highly volatile; wider range acceptable
    "KRAKEN:SOL/USD": {
        "breakout_multiplier": 0.68,
        "max_or_atr_ratio": 3.0,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.08,
        "min_body_ratio": 0.40,
    },
    # Mid-cap alts (LINK, AVAX, DOT, ADA, MATIC, XRP) — moderate volatility
    "KRAKEN:LINK/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:AVAX/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:DOT/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:ADA/USD": {
        "breakout_multiplier": 0.60,
        "max_or_atr_ratio": 2.5,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
    "KRAKEN:MATIC/USD": {
        "breakout_multiplier": 0.62,
        "max_or_atr_ratio": 2.6,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
    "KRAKEN:XRP/USD": {
        "breakout_multiplier": 0.60,
        "max_or_atr_ratio": 2.5,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
}


def get_symbol_session_overrides(symbol: str, session: "ORBSession") -> dict[str, float]:
    """Return per-symbol parameter overrides for a given session.

    For Kraken crypto tickers the returned dict may contain
    ``breakout_multiplier``, ``max_or_atr_ratio``, ``min_or_atr_ratio``,
    ``min_depth_atr_pct``, and ``min_body_ratio`` keys that should
    override the session defaults.

    For non-crypto symbols (or when no override is registered) an empty
    dict is returned so callers can use session defaults unchanged.

    Args:
        symbol: Instrument ticker (e.g. "KRAKEN:XBT/USD", "MGC=F").
        session: The ORBSession being evaluated.

    Returns:
        Dict of parameter overrides, or {} if no overrides are defined.
    """
    # Direct match (exact ticker)
    if symbol in CRYPTO_SYMBOL_OVERRIDES:
        return dict(CRYPTO_SYMBOL_OVERRIDES[symbol])

    # Prefix match for KRAKEN: tickers not in the table — use BTC defaults
    # as a safe conservative fallback for any unknown crypto asset.
    if symbol.startswith("KRAKEN:"):
        return {
            "breakout_multiplier": 0.65,
            "max_or_atr_ratio": 2.8,
            "min_or_atr_ratio": 0.03,
            "min_depth_atr_pct": 0.09,
            "min_body_ratio": 0.42,
        }

    return {}


# ---------------------------------------------------------------------------
# Kraken crypto tickers — injected into session asset lists when enabled.
# Crypto markets are 24/7 so they're relevant in every session window.
# Gated by ENABLE_KRAKEN_CRYPTO env var (same flag as models.py).
# ---------------------------------------------------------------------------
_KRAKEN_CRYPTO_TICKERS: list[str] = []

try:
    from lib.core.models import ENABLE_KRAKEN_CRYPTO

    if ENABLE_KRAKEN_CRYPTO:
        from lib.integrations.kraken_client import KRAKEN_PAIRS

        _KRAKEN_CRYPTO_TICKERS.extend(p["internal_ticker"] for p in KRAKEN_PAIRS.values())
except ImportError:
    pass

# Inject crypto tickers into every session that makes sense.
# Crypto is 24/7, so all sessions benefit from scanning them.
# We add them at the end of each list so CME futures remain first priority.
if _KRAKEN_CRYPTO_TICKERS:
    for _sk in SESSION_ASSETS:
        SESSION_ASSETS[_sk] = SESSION_ASSETS[_sk] + _KRAKEN_CRYPTO_TICKERS

    # Also populate the dedicated crypto session asset lists.
    # These sessions only run when crypto is enabled.
    SESSION_ASSETS["crypto_utc0"] = list(_KRAKEN_CRYPTO_TICKERS)
    SESSION_ASSETS["crypto_utc12"] = list(_KRAKEN_CRYPTO_TICKERS)

    # Register the crypto sessions so they are included in ORB_SESSIONS
    # and SESSION_BY_KEY.  Append after the existing nine sessions so
    # existing session ordering is unchanged.
    ORB_SESSIONS.append(CRYPTO_UTC_MIDNIGHT_SESSION)
    ORB_SESSIONS.append(CRYPTO_UTC_NOON_SESSION)
    SESSION_BY_KEY["crypto_utc0"] = CRYPTO_UTC_MIDNIGHT_SESSION
    SESSION_BY_KEY["crypto_utc12"] = CRYPTO_UTC_NOON_SESSION


def get_session_assets(session: "ORBSession") -> list[str]:
    """Return the list of Yahoo tickers relevant for *session*.

    Falls back to all configured assets if the session has no specific list.

    Args:
        session: The ORBSession to look up.

    Returns:
        List of Yahoo ticker strings (e.g. ``["MGC=F", "ES=F"]``).
    """
    if session.key in SESSION_ASSETS:
        return SESSION_ASSETS[session.key]
    # Fallback: return all asset tickers from the models config
    try:
        from lib.core.models import ASSETS

        return list(ASSETS.values())
    except Exception:
        return []


def get_session_for_utc(utc_dt: datetime) -> ORBSession | None:
    """Return the ORBSession that is currently active for the given UTC datetime.

    Converts *utc_dt* to ET wall-clock time (ZoneInfo("America/New_York"))
    and checks each session's OR start → scan_end window.  DST transitions
    are handled automatically — no manual UTC offset needed.

    For ``wraps_midnight`` sessions the wall-clock window may straddle
    00:00 ET (e.g. CME open 18:00–20:00 ET).  These sessions are always
    active when the ET time is within [or_start, scan_end] regardless of
    calendar date.

    Returns the *first* matching session in ORB_SESSIONS priority order,
    or ``None`` if no session is currently active.

    Args:
        utc_dt: A tz-aware datetime in UTC (or any tz — will be converted).

    Example::

        from datetime import datetime, timezone
        from lib.services.engine.rb.orb import get_session_for_utc

        now_utc = datetime.now(timezone.utc)
        session = get_session_for_utc(now_utc)
        if session:
            print(f"Active session: {session.name}")
    """
    # Convert to ET wall-clock — ZoneInfo handles EST (UTC-5) / EDT (UTC-4)
    et_dt = utc_dt.astimezone(_EST)
    et_time = et_dt.time()

    for session in ORB_SESSIONS:
        start = session.or_start
        end = session.scan_end

        if start <= end:
            # Normal (no midnight wrap): window is e.g. 03:00–05:00
            if start <= et_time <= end:
                return session
        else:
            # Wraps midnight: window straddles 00:00 e.g. 18:00–02:00
            # (scan_end < or_start means the window crosses midnight)
            if et_time >= start or et_time <= end:
                return session

    return None


def get_active_session_keys(utc_dt: datetime | None = None) -> list[str]:
    """Return keys of ALL sessions whose windows overlap the given UTC time.

    Unlike ``get_session_for_utc()`` (which returns only the first match),
    this returns every session that is simultaneously active — useful when
    Frankfurt and London overlap at 03:00–03:30 ET.

    Args:
        utc_dt: UTC datetime to check.  Defaults to ``datetime.now(UTC)``.

    Returns:
        List of session key strings (may be empty).
    """
    if utc_dt is None:
        utc_dt = datetime.now(tz=_UTC)

    et_dt = utc_dt.astimezone(_EST)
    et_time = et_dt.time()

    active: list[str] = []
    for session in ORB_SESSIONS:
        start = session.or_start
        end = session.scan_end
        if start <= end:
            if start <= et_time <= end:
                active.append(session.key)
        else:
            if et_time >= start or et_time <= end:
                active.append(session.key)
    return active


# ---------------------------------------------------------------------------
# Legacy aliases for backward compatibility
# ---------------------------------------------------------------------------
OR_START = US_SESSION.or_start
OR_END = US_SESSION.or_end
ATR_PERIOD = 14
BREAKOUT_ATR_MULTIPLIER = 0.5
MIN_OR_BARS = 5
MAX_OR_BARS = 35

# Convenience groupings used by the scheduler
OVERNIGHT_SESSIONS: list[ORBSession] = [s for s in ORB_SESSIONS if s.wraps_midnight]
DAYTIME_SESSIONS: list[ORBSession] = [s for s in ORB_SESSIONS if not s.wraps_midnight]

# Sessions included in CNN dataset generation
DATASET_SESSIONS: list[ORBSession] = [s for s in ORB_SESSIONS if s.include_in_dataset]


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

    # Quality gate results (populated by detect_opening_range_breakout)
    depth_ok: bool | None = None  # True if depth filter passed
    body_ratio_ok: bool | None = None  # True if body-ratio filter passed
    or_size_ok: bool | None = None  # True if OR-size cap/floor passed
    breakout_bar_depth: float = 0.0  # Actual penetration beyond OR level
    breakout_bar_body_ratio: float = 0.0  # Actual body/range ratio of breakout bar

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
            # Quality gate results
            "depth_ok": self.depth_ok,
            "body_ratio_ok": self.body_ratio_ok,
            "or_size_ok": self.or_size_ok,
            "breakout_bar_depth": round(self.breakout_bar_depth, 6),
            "breakout_bar_body_ratio": round(self.breakout_bar_body_ratio, 4),
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

    Handles sessions that ``wraps_midnight`` (Sydney, Tokyo, CME open):
    those sessions start in the *previous* calendar day in ET, so we
    look for bars on either today or yesterday within the OR time window.

    Args:
        bars_1m: DataFrame with DatetimeIndex and columns: High, Low, Close.
                 Should contain at least 24 hours of 1-minute bars so that
                 overnight sessions have data.
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
    idx: Any = df.index
    times = idx.time

    if session.wraps_midnight:
        # For overnight sessions the OR window is e.g. 17:00–17:30 ET which
        # lives in the *previous* calendar day relative to "today" in ET.
        # We filter purely by wall-clock time regardless of date so that the
        # most recent occurrence of that window is used.
        or_mask = (times >= session.or_start) & (times < session.or_end)
        # is_complete: any bar at or after or_end on the same calendar day
        # as the last OR bar, or any bar today past midnight.
        or_bars_all: pd.DataFrame = df.loc[or_mask]
        if or_bars_all.empty:
            return 0.0, 0.0, 0, False
        # Use the most recent contiguous block (last date that has OR bars)
        last_or_date = or_bars_all.index[-1].date()
        or_bars: pd.DataFrame = or_bars_all[or_bars_all.index.date == last_or_date]

        # is_complete: bars after or_end on that date, or any bar on the
        # *next* calendar day (i.e., today if OR was yesterday)
        import datetime as _dt

        next_date = last_or_date + _dt.timedelta(days=1)
        post_or = df[
            ((df.index.date == last_or_date) & (times >= session.or_end)) | (df.index.date == next_date)  # type: ignore[operator]
        ]
        is_complete = not post_or.empty
    else:
        # Normal intraday session — filter by today's wall-clock time only
        or_mask = (times >= session.or_start) & (times < session.or_end)
        or_bars = df.loc[or_mask]
        if or_bars.empty:
            return 0.0, 0.0, 0, False
        is_complete = bool(np.any(times >= session.or_end))

    if or_bars.empty:
        return 0.0, 0.0, 0, False

    or_high = float(or_bars["High"].max())  # type: ignore[arg-type]
    or_low = float(or_bars["Low"].min())  # type: ignore[arg-type]
    bar_count: int = len(or_bars)

    return or_high, or_low, bar_count, is_complete


def _check_or_size(
    or_range: float,
    atr: float,
    session: ORBSession,
) -> tuple[bool, str]:
    """Check whether the opening range size is within acceptable bounds.

    Args:
        or_range: Height of the opening range (or_high - or_low).
        atr: Current ATR value.
        session: Session whose thresholds to apply.

    Returns:
        (passed, reason_string)  — reason is empty when passed=True.
    """
    if atr <= 0:
        return True, ""  # can't evaluate without ATR; defer to caller

    ratio = or_range / atr

    if session.max_or_atr_ratio > 0 and ratio > session.max_or_atr_ratio:
        return False, (f"OR too wide: {or_range:.4f} = {ratio:.2f}× ATR (cap {session.max_or_atr_ratio:.2f}×)")
    if session.min_or_atr_ratio > 0 and ratio < session.min_or_atr_ratio:
        return False, (f"OR too narrow: {or_range:.4f} = {ratio:.2f}× ATR (floor {session.min_or_atr_ratio:.2f}×)")
    return True, ""


def _check_breakout_bar_quality(
    row: "pd.Series",
    direction: str,
    or_high: float,
    or_low: float,
    atr: float,
    session: ORBSession,
) -> tuple[bool, bool, float, float]:
    """Evaluate depth and body-ratio quality of a candidate breakout bar.

    Args:
        row: The breakout bar (must have Open, High, Low, Close).
        direction: ``"LONG"`` or ``"SHORT"``.
        or_high: Opening range high.
        or_low: Opening range low.
        atr: Current ATR value.
        session: Session whose quality thresholds to apply.

    Returns:
        (depth_ok, body_ratio_ok, depth_value, body_ratio_value)
        depth_value  — absolute penetration beyond OR level (always ≥ 0).
        body_ratio_value — |close - open| / (high - low), clamped [0, 1].
    """
    bar_open: float = float(row.get("Open", row.get("open", 0.0)))
    bar_high: float = float(row.get("High", row.get("high", 0.0)))
    bar_low: float = float(row.get("Low", row.get("low", 0.0)))
    bar_close: float = float(row.get("Close", row.get("close", 0.0)))

    # Depth: how far the close penetrated beyond the OR level
    depth = max(bar_close - or_high, 0.0) if direction == "LONG" else max(or_low - bar_close, 0.0)

    # Body ratio: fraction of bar range covered by the candle body
    bar_range = bar_high - bar_low
    body_ratio = abs(bar_close - bar_open) / bar_range if bar_range > 0 else 0.0
    body_ratio = min(body_ratio, 1.0)

    # Evaluate thresholds
    min_depth = atr * session.min_depth_atr_pct if atr > 0 else 0.0
    depth_ok = (session.min_depth_atr_pct <= 0) or (depth >= min_depth)
    body_ok = (session.min_body_ratio <= 0) or (body_ratio >= session.min_body_ratio)

    return depth_ok, body_ok, depth, body_ratio


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
      3. Apply OR-size quality gate (too wide or too narrow → skip).
      4. Define breakout levels:
           - Long trigger  = OR_high + (ATR × multiplier)
           - Short trigger = OR_low  − (ATR × multiplier)
      5. Scan bars after OR end (up to scan_end) for a close beyond either trigger.
      6. For the first candidate bar, apply depth and body-ratio quality gates.
         If they fail, continue scanning for the next candidate bar.
      7. Return the first bar that passes all quality gates (or no breakout).

    The quality gate results (depth_ok, body_ratio_ok, or_size_ok) are always
    populated on the returned ORBResult so the caller can log/audit them even
    when breakout_detected=False.

    Args:
        bars_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
                 Should cover at least the current session day.  For overnight
                 sessions (wraps_midnight=True), pass 2 days of bars.
        symbol: Instrument symbol for labelling (e.g. "MGC", "MNQ").
        session: ORB session definition. Defaults to US_SESSION.
        atr_period: ATR look-back period (overrides session default).
        breakout_multiplier: ATR multiplier (overrides session default).
        now_fn: Optional clock function for testability.

    Returns:
        ORBResult with breakout_detected=True/False and all quality details.
    """
    if session is None:
        session = US_SESSION

    _atr_period = atr_period if atr_period is not None else session.atr_period

    # Apply per-symbol overrides (crypto assets need wider thresholds).
    # Caller-supplied breakout_multiplier always takes highest precedence;
    # symbol overrides are second; session defaults are the fallback.
    _sym_overrides = get_symbol_session_overrides(symbol, session)
    _breakout_mult = (
        breakout_multiplier
        if breakout_multiplier is not None
        else _sym_overrides.get("breakout_multiplier", session.breakout_multiplier)
    )

    # Build an effective session object that merges symbol-level overrides
    # into the OR-size and quality-gate fields so _check_or_size() and
    # _check_breakout_bar_quality() automatically use the right thresholds.
    if _sym_overrides:
        import dataclasses as _dc

        session = _dc.replace(
            session,
            max_or_atr_ratio=_sym_overrides.get("max_or_atr_ratio", session.max_or_atr_ratio),
            min_or_atr_ratio=_sym_overrides.get("min_or_atr_ratio", session.min_or_atr_ratio),
            min_depth_atr_pct=_sym_overrides.get("min_depth_atr_pct", session.min_depth_atr_pct),
            min_body_ratio=_sym_overrides.get("min_body_ratio", session.min_body_ratio),
        )

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

    # --- Compute breakout levels (always set so callers can read them) ---
    threshold = atr * _breakout_mult
    result.breakout_threshold = threshold
    result.long_trigger = or_high + threshold
    result.short_trigger = or_low - threshold

    # --- OR-size quality gate ---
    or_size_ok, or_size_reason = _check_or_size(result.or_range, atr, session)
    result.or_size_ok = or_size_ok
    if not or_size_ok:
        result.error = f"OR-size gate: {or_size_reason}"
        logger.debug(
            "ORB [%s] %s skipped — %s",
            session.name,
            symbol,
            or_size_reason,
        )
        return result

    # --- If OR isn't complete yet, just return the levels (no scan) ---
    if not or_complete:
        return result

    # --- Scan post-OR bars for breakout ---
    df = _localize_index(bars_1m.copy())
    _idx: Any = df.index
    times = _idx.time

    if session.wraps_midnight:
        # For overnight sessions, post-OR bars can be on the next calendar day
        import datetime as _dt

        # Find the date of the most recent OR bar
        _or_mask = (times >= session.or_start) & (times < session.or_end)
        _or_dates = df.loc[_or_mask].index
        if _or_dates.empty:
            return result
        _last_or_date = _or_dates[-1].date()
        _next_date = _last_or_date + _dt.timedelta(days=1)

        post_or_bars: pd.DataFrame = df[
            ((df.index.date == _last_or_date) & (times >= session.or_end) & (times <= session.scan_end))  # type: ignore[operator]
            | ((df.index.date == _next_date) & (times <= session.scan_end))  # type: ignore[operator]
        ]
    else:
        post_or_mask = (times >= session.or_end) & (times <= session.scan_end)
        post_or_bars = df.loc[post_or_mask]

    if post_or_bars.empty:
        return result

    # --- Scan for first qualifying breakout ---
    for idx_label, row in post_or_bars.iterrows():
        close: float = float(row["Close"])  # type: ignore[arg-type]

        candidate_direction: str = ""
        if close > result.long_trigger:
            candidate_direction = "LONG"
        elif close < result.short_trigger:
            candidate_direction = "SHORT"
        else:
            continue

        # Apply depth and body-ratio quality gates
        depth_ok, body_ok, depth_val, body_val = _check_breakout_bar_quality(
            row,
            candidate_direction,
            or_high,
            or_low,
            atr,
            session,
        )

        # Always record the quality metrics on first candidate for auditing
        if result.depth_ok is None:
            result.depth_ok = depth_ok
            result.body_ratio_ok = body_ok
            result.breakout_bar_depth = depth_val
            result.breakout_bar_body_ratio = body_val

        if not depth_ok:
            logger.debug(
                "ORB [%s] %s %s candidate rejected: depth %.4f < min %.4f (%.2f× ATR)",
                session.name,
                candidate_direction,
                symbol,
                depth_val,
                atr * session.min_depth_atr_pct,
                session.min_depth_atr_pct,
            )
            continue  # look for a deeper bar

        if not body_ok:
            logger.debug(
                "ORB [%s] %s %s candidate rejected: body ratio %.2f < min %.2f",
                session.name,
                candidate_direction,
                symbol,
                body_val,
                session.min_body_ratio,
            )
            continue  # look for a cleaner bar

        # All quality gates passed — confirmed breakout
        result.breakout_detected = True
        result.direction = candidate_direction
        result.trigger_price = close
        result.breakout_bar_time = str(idx_label)
        result.depth_ok = depth_ok
        result.body_ratio_ok = body_ok
        result.breakout_bar_depth = depth_val
        result.breakout_bar_body_ratio = body_val
        break

    if result.breakout_detected:
        logger.info(
            "ORB [%s] detected: %s %s @ %.4f (OR %.4f–%.4f, ATR %.4f, threshold %.4f, depth %.4f, body_ratio %.2f)",
            session.name,
            result.direction,
            symbol,
            result.trigger_price,
            result.or_low,
            result.or_high,
            atr,
            threshold,
            result.breakout_bar_depth,
            result.breakout_bar_body_ratio,
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

    For overnight sessions (wraps_midnight=True) whose OR window is before
    midnight ET, we check whether the current time is within the scan window
    which may extend past midnight into the next calendar day.

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
        if session.wraps_midnight:
            # Overnight session: active from or_start (e.g. 17:00) to
            # scan_end (e.g. 19:00).  scan_end is always on the same
            # calendar day as or_start (no midnight crossing in scan window).
            if session.or_start <= t <= session.scan_end:
                active.append(session)
        else:
            # Normal session: active from or_start through scan_end same day
            if session.or_start <= t <= session.scan_end:
                active.append(session)

    return active


def is_any_session_active(now: datetime | None = None) -> bool:
    """Check if any ORB session is currently active."""
    return len(get_active_sessions(now)) > 0


def get_session_status(now: datetime | None = None) -> dict[str, str]:
    """Return a status dict for each session.

    Possible statuses: "waiting", "forming", "scanning", "complete"

    For overnight (wraps_midnight) sessions, status is relative to the
    most recent occurrence of that session window.
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


def get_session_by_key(key: str) -> ORBSession | None:
    """Return the ORBSession with the given key, or None if not found.

    Args:
        key: Session key string, e.g. ``"london"``, ``"tokyo"``, ``"us"``.

    Returns:
        Matching ORBSession, or None.
    """
    for session in ORB_SESSIONS:
        if session.key == key:
            return session
    return None


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

    Also pushes the signal to the TradingView signal store + GitHub signals.csv
    (Phase TV-A) so TradingView's ``request.seed()`` can auto-draw engine levels.

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
