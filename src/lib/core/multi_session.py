"""
Multi-Session Support — All 9 Globex Sessions with Bracket Parameters
=======================================================================
Single source of truth for every session used by the ``rb`` platform.
Both the Python dataset generator / CNN pipeline **and** the C# NinjaTrader
strategy read from this contract.

Sessions defined (chronological Globex-day order, 18:00 ET start):
  1. ``cme``        — CME Globex Re-open      18:00–18:30 ET
  2. ``sydney``     — Sydney / ASX            18:30–19:00 ET
  3. ``tokyo``      — Tokyo / TSE             19:00–19:30 ET
  4. ``shanghai``   — Shanghai / HK           21:00–21:30 ET
  5. ``frankfurt``  — Frankfurt / Xetra       03:00–03:30 ET
  6. ``london``     — London Open             03:00–03:30 ET
  7. ``london_ny``  — London-NY Crossover     08:00–08:30 ET
  8. ``us``         — US Equity Open          09:30–10:00 ET
  9. ``cme_settle`` — CME Settlement          14:00–14:30 ET

Each session carries:
  - Open/close time for the opening range window (ET)
  - Pre-market end time (used to compute premarket_range_ratio)
  - Default bracket parameters (SL/TP/max-hold)
  - CNN inference threshold (per ``breakout_cnn.SESSION_THRESHOLDS``)
  - Session ordinal [0, 1] for the CNN tabular feature
  - Metadata (display name, applies_to asset classes, overnight flag)

Usage::

    from lib.core.multi_session import (
        get_session,
        all_sessions,
        session_keys,
        ORBSession,
        SESSION_BY_KEY,
    )

    sess = get_session("london")
    print(sess.or_start)           # datetime.time(3, 0)
    print(sess.cnn_threshold)      # 0.82
    print(sess.session_ordinal)    # 0.625

    for sess in all_sessions():
        print(sess.key, sess.display_name)

Design:
  - ``ORBSession`` is a frozen dataclass so instances are hashable and safe
    to use as dict keys.
  - ``SESSION_BY_KEY`` is the module-level registry — callers should use
    ``get_session()`` for safe lookup with a clear error message.
  - All times are ``datetime.time`` objects in US/Eastern (ET) — callers are
    responsible for attaching the correct tzinfo when building full datetimes.
  - The ordering in ``ALL_SESSION_KEYS`` mirrors the C# ``ORBSession`` list
    in ``orb.py`` / ``BreakoutStrategy.cs`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Any

# ---------------------------------------------------------------------------
# Ordered session key list — chronological Globex-day (18:00 ET start)
# ---------------------------------------------------------------------------

ALL_SESSION_KEYS: list[str] = [
    "cme",  # 18:00–18:30 ET
    "sydney",  # 18:30–19:00 ET
    "tokyo",  # 19:00–19:30 ET
    "shanghai",  # 21:00–21:30 ET
    "frankfurt",  # 03:00–03:30 ET
    "london",  # 03:00–03:30 ET  (primary)
    "london_ny",  # 08:00–08:30 ET
    "us",  # 09:30–10:00 ET  (primary)
    "cme_settle",  # 14:00–14:30 ET
]


# ---------------------------------------------------------------------------
# ORBSession dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ORBSession:
    """All parameters for one named Globex session.

    Attributes:
        key:                 Short identifier used everywhere in Python and C#
                             code (e.g. ``"london"``, ``"us"``).
        display_name:        Human-readable label for logs and the dashboard.
        or_start:            Opening-range window start (ET).
        or_end:              Opening-range window end (ET).
        pm_end:              Pre-market window end (ET) — used to compute the
                             ``premarket_range_ratio`` tabular feature.
                             For overnight sessions this equals ``or_start``.
        wraps_midnight:      True if the session straddles the ET midnight
                             boundary (i.e. 18:00–23:59 → 00:00–…).  The
                             dataset generator uses this to correctly slice
                             bar data across date boundaries.
        is_overnight:        True for sessions that run during thin overnight
                             markets (CME, Sydney, Tokyo, Shanghai).  These
                             use lower CNN thresholds.
        sl_atr_mult:         Default stop-loss ATR multiplier.
        tp1_atr_mult:        Default TP1 ATR multiplier.
        tp2_atr_mult:        Default TP2 ATR multiplier.
        max_hold_bars:       Maximum bars to stay in trade before forced exit.
        cnn_threshold:       CNN inference probability threshold for this
                             session.  Overnight sessions use lower values to
                             avoid over-filtering a thinner signal pool.
        session_ordinal:     Normalised position in the 24-h Globex day
                             [0.0, 1.0].  Matches ``SESSION_ORDINAL`` in
                             ``breakout_cnn.py`` and ``feature_contract.json``.
        applies_to:          Asset class tags this session is most relevant
                             for.  ``"all"`` means no filtering.
        description:         One-line summary for documentation.
        extra:               Reserved dict for future per-session parameters.
    """

    key: str
    display_name: str
    or_start: dt_time
    or_end: dt_time
    pm_end: dt_time
    wraps_midnight: bool
    is_overnight: bool
    sl_atr_mult: float
    tp1_atr_mult: float
    tp2_atr_mult: float
    max_hold_bars: int
    cnn_threshold: float
    session_ordinal: float
    applies_to: list[str]
    description: str
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------
#
# Bracket parameters (SL/TP/max_hold) are the *default* values used by
# ``_bracket_configs_for_session()`` in ``dataset_generator.py`` and by the
# NT8 strategy when no per-symbol override is configured.
#
# CNN thresholds match ``breakout_cnn.SESSION_THRESHOLDS`` exactly — keep
# them in sync if you tune one side.

_CME_SESSION = ORBSession(
    key="cme",
    display_name="CME Globex Re-open",
    or_start=dt_time(18, 0),
    or_end=dt_time(18, 30),
    pm_end=dt_time(18, 0),  # no pre-market concept at open of Globex day
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.75,
    session_ordinal=0.0 / 8.0,  # 0.000
    applies_to=["all"],
    description="CME Globex session re-opens at 18:00 ET — first bars of the new trading day.",
    extra={
        "globex_day_start": True,
        "typical_spread_wide": True,
    },
)

_SYDNEY_SESSION = ORBSession(
    key="sydney",
    display_name="Sydney / ASX",
    or_start=dt_time(18, 30),
    or_end=dt_time(19, 0),
    pm_end=dt_time(18, 30),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.72,  # thinnest session — lowest threshold
    session_ordinal=1.0 / 8.0,  # 0.125
    applies_to=["fx", "metals"],
    description="Sydney / ASX session open at 18:30 ET — thinnest overnight session.",
    extra={
        "active_pairs": ["6A=F", "6J=F"],
    },
)

_TOKYO_SESSION = ORBSession(
    key="tokyo",
    display_name="Tokyo / TSE",
    or_start=dt_time(19, 0),
    or_end=dt_time(19, 30),
    pm_end=dt_time(19, 0),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.74,
    session_ordinal=2.0 / 8.0,  # 0.250
    applies_to=["fx", "metals"],
    description="Tokyo / TSE session at 19:00 ET — narrow ranges, metals and JPY driver.",
    extra={
        "active_pairs": ["6J=F", "MGC=F", "SIL=F"],
    },
)

_SHANGHAI_SESSION = ORBSession(
    key="shanghai",
    display_name="Shanghai / HK",
    or_start=dt_time(21, 0),
    or_end=dt_time(21, 30),
    pm_end=dt_time(21, 0),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.74,
    session_ordinal=3.0 / 8.0,  # 0.375
    applies_to=["metals", "energy"],
    description="Shanghai / Hong Kong session at 21:00 ET — copper and gold driver.",
    extra={
        "active_pairs": ["MHG=F", "MGC=F", "MCL=F"],
    },
)

_FRANKFURT_SESSION = ORBSession(
    key="frankfurt",
    display_name="Frankfurt / Xetra",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    pm_end=dt_time(3, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.80,
    session_ordinal=4.0 / 8.0,  # 0.500
    applies_to=["fx", "equity_index"],
    description="Frankfurt / Xetra open at 03:00 ET — pre-London, good volume on EUR pairs.",
    extra={
        "active_pairs": ["6E=F", "6B=F", "MES=F"],
        "pre_london": True,
    },
)

_LONDON_SESSION = ORBSession(
    key="london",
    display_name="London Open",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    pm_end=dt_time(3, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.82,  # primary session — highest conviction bar
    session_ordinal=5.0 / 8.0,  # 0.625
    applies_to=["all"],
    description="London Open at 03:00 ET — PRIMARY session, highest volume and conviction.",
    extra={
        "primary_session": True,
        "overlaps_frankfurt": True,
    },
)

_LONDON_NY_SESSION = ORBSession(
    key="london_ny",
    display_name="London-NY Crossover",
    or_start=dt_time(8, 0),
    or_end=dt_time(8, 30),
    pm_end=dt_time(8, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.5,  # highest volume window — wider TP target
    max_hold_bars=120,
    cnn_threshold=0.82,
    session_ordinal=6.0 / 8.0,  # 0.750
    applies_to=["all"],
    description="London-NY Crossover at 08:00 ET — highest total volume of the Globex day.",
    extra={
        "highest_volume": True,
        "overlap_window_start": dt_time(8, 0),
        "overlap_window_end": dt_time(10, 0),
    },
)

_US_SESSION = ORBSession(
    key="us",
    display_name="US Equity Open",
    or_start=dt_time(9, 30),
    or_end=dt_time(10, 0),
    pm_end=dt_time(8, 20),  # pre-market ends at 08:20 ET (gap-up/down reference)
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.82,  # primary session
    session_ordinal=7.0 / 8.0,  # 0.875
    applies_to=["all"],
    description="US Equity Open at 09:30 ET — classic ORB session for equity indices.",
    extra={
        "primary_session": True,
        "rth_open": True,
        "premarket_ref_start": dt_time(4, 0),  # used for pm_high / pm_low computation
    },
)

_CME_SETTLE_SESSION = ORBSession(
    key="cme_settle",
    display_name="CME Settlement",
    or_start=dt_time(14, 0),
    or_end=dt_time(14, 30),
    pm_end=dt_time(8, 20),  # same pre-market reference as US session
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=1.5,  # tighter targets near settlement
    tp2_atr_mult=2.5,
    max_hold_bars=60,  # shorter hold — settlement can reverse quickly
    cnn_threshold=0.78,
    session_ordinal=8.0 / 8.0,  # 1.000
    applies_to=["metals", "energy"],
    description="CME Settlement at 14:00 ET — metals and energy settlement window.",
    extra={
        "settlement_window": True,
        "active_pairs": ["MGC=F", "MCL=F", "MHG=F", "MNG=F"],
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SESSION_BY_KEY: dict[str, ORBSession] = {
    "cme": _CME_SESSION,
    "sydney": _SYDNEY_SESSION,
    "tokyo": _TOKYO_SESSION,
    "shanghai": _SHANGHAI_SESSION,
    "frankfurt": _FRANKFURT_SESSION,
    "london": _LONDON_SESSION,
    "london_ny": _LONDON_NY_SESSION,
    "us": _US_SESSION,
    "cme_settle": _CME_SETTLE_SESSION,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_session(key: str) -> ORBSession:
    """Return the ``ORBSession`` for *key*.

    Args:
        key: Session key string (e.g. ``"london"``, ``"us"``).
             Case-insensitive.

    Returns:
        ``ORBSession`` frozen dataclass instance.

    Raises:
        KeyError: If *key* is not a recognised session key, with a helpful
                  message listing all valid keys.

    Example::

        >>> from multi_session import get_session
        >>> sess = get_session("london")
        >>> sess.or_start
        datetime.time(3, 0)
        >>> sess.cnn_threshold
        0.82
    """
    _key = key.strip().lower()
    if _key not in SESSION_BY_KEY:
        raise KeyError(f"Unknown session key {key!r}. Valid keys: {ALL_SESSION_KEYS}")
    return SESSION_BY_KEY[_key]


def all_sessions() -> list[ORBSession]:
    """Return all ``ORBSession`` objects in Globex-day chronological order.

    Example::

        >>> from multi_session import all_sessions
        >>> for s in all_sessions():
        ...     print(s.key, s.session_ordinal)
        cme 0.0
        sydney 0.125
        ...
    """
    return [SESSION_BY_KEY[k] for k in ALL_SESSION_KEYS]


def session_keys() -> list[str]:
    """Return all session keys in Globex-day chronological order."""
    return list(ALL_SESSION_KEYS)


def overnight_sessions() -> list[ORBSession]:
    """Return only the overnight sessions (is_overnight=True).

    These sessions use lower CNN thresholds due to thin markets.
    """
    return [s for s in all_sessions() if s.is_overnight]


def daytime_sessions() -> list[ORBSession]:
    """Return only the daytime sessions (is_overnight=False)."""
    return [s for s in all_sessions() if not s.is_overnight]


def sessions_for_asset_class(asset_class: str) -> list[ORBSession]:
    """Return sessions relevant to *asset_class*.

    Args:
        asset_class: One of ``"fx"``, ``"metals"``, ``"energy"``,
                     ``"equity_index"``, ``"crypto"``, or ``"all"``.

    Returns:
        List of ``ORBSession`` objects whose ``applies_to`` list contains
        *asset_class* or ``"all"``.

    Example::

        >>> from multi_session import sessions_for_asset_class
        >>> [s.key for s in sessions_for_asset_class("metals")]
        ['sydney', 'tokyo', 'shanghai', 'frankfurt', 'london', 'london_ny', 'us', 'cme_settle']
    """
    _ac = asset_class.strip().lower()
    return [s for s in all_sessions() if "all" in s.applies_to or _ac in s.applies_to]


def to_bracket_params(session: ORBSession) -> dict[str, Any]:
    """Serialise a session's bracket parameters to a plain dict.

    This is the format consumed by ``orb_simulator.BracketConfig`` and the
    NT8 ``SessionBracket`` struct in ``BreakoutStrategy.cs``.

    Returns::

        {
            "key":           "london",
            "or_start":      "03:00",
            "or_end":        "03:30",
            "pm_end":        "03:00",
            "sl_atr_mult":   1.5,
            "tp1_atr_mult":  2.0,
            "tp2_atr_mult":  3.0,
            "max_hold_bars": 120,
        }
    """
    return {
        "key": session.key,
        "or_start": session.or_start.strftime("%H:%M"),
        "or_end": session.or_end.strftime("%H:%M"),
        "pm_end": session.pm_end.strftime("%H:%M"),
        "sl_atr_mult": session.sl_atr_mult,
        "tp1_atr_mult": session.tp1_atr_mult,
        "tp2_atr_mult": session.tp2_atr_mult,
        "max_hold_bars": session.max_hold_bars,
    }


def to_feature_contract_dict() -> dict[str, Any]:
    """Serialise all sessions to a dict for ``feature_contract.json``.

    Inserted under the ``"sessions"`` key so the C# consumer can verify
    ordinals and thresholds at load time.

    Example output::

        {
          "us":        {"ordinal": 7, "session_ordinal": 0.875, "cnn_threshold": 0.82, ...},
          "london":    {"ordinal": 5, "session_ordinal": 0.625, "cnn_threshold": 0.82, ...},
          ...
        }
    """
    return {
        sess.key: {
            "ordinal": ALL_SESSION_KEYS.index(sess.key),
            "session_ordinal": round(sess.session_ordinal, 6),
            "cnn_threshold": sess.cnn_threshold,
            "or_start": sess.or_start.strftime("%H:%M"),
            "or_end": sess.or_end.strftime("%H:%M"),
            "pm_end": sess.pm_end.strftime("%H:%M"),
            "is_overnight": sess.is_overnight,
            "wraps_midnight": sess.wraps_midnight,
            "display_name": sess.display_name,
            "applies_to": sess.applies_to,
        }
        for sess in all_sessions()
    }
