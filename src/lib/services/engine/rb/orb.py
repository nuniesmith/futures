"""
Opening Range Breakout — Compatibility Shim
=============================================
This module is a **backward-compatibility alias** for the ``rb.open`` package.

The full ORB implementation was previously at
``lib.services.engine.rb.open.main`` (monolithic) and is now split across
five focused sub-modules inside ``lib.services.engine.rb.open``:

    sessions.py   — ORBSession dataclass, session instances, helpers
    assets.py     — SESSION_ASSETS, crypto overrides, asset helpers
    models.py     — ORBResult, MultiSessionORBResult
    detector.py   — compute_atr, detect_opening_range_breakout, etc.
    publisher.py  — Redis publish/clear helpers

All symbols are re-exported here unchanged so that existing call sites of
the form::

    from lib.services.engine.rb.orb import LONDON_SESSION
    from lib.services.engine.rb.orb import detect_opening_range_breakout

continue to work without modification.  New code should import directly
from the package::

    from lib.services.engine.rb.open import LONDON_SESSION
"""

from __future__ import annotations

from lib.services.engine.rb.open import (  # noqa: F401
    ATR_PERIOD,
    BREAKOUT_ATR_MULTIPLIER,
    CME_OPEN_SESSION,
    CME_SETTLEMENT_SESSION,
    CRYPTO_SYMBOL_OVERRIDES,
    CRYPTO_UTC_MIDNIGHT_SESSION,
    CRYPTO_UTC_NOON_SESSION,
    DATASET_SESSIONS,
    DAYTIME_SESSIONS,
    FRANKFURT_SESSION,
    LONDON_NY_SESSION,
    LONDON_SESSION,
    MAX_OR_BARS,
    MIN_OR_BARS,
    OR_END,
    OR_START,
    ORB_SESSIONS,
    OVERNIGHT_SESSIONS,
    REDIS_KEY_ORB,
    REDIS_KEY_ORB_TS,
    REDIS_PUBSUB_ORB,
    REDIS_TTL,
    SESSION_ASSETS,
    SESSION_BY_KEY,
    SHANGHAI_SESSION,
    SYDNEY_SESSION,
    TOKYO_SESSION,
    US_SESSION,
    MultiSessionORBResult,
    ORBResult,
    ORBSession,
    clear_orb_alert,
    compute_atr,
    compute_opening_range,
    detect_all_sessions,
    detect_opening_range_breakout,
    get_active_session_keys,
    get_active_sessions,
    get_session_assets,
    get_session_by_key,
    get_session_for_utc,
    get_session_status,
    get_symbol_session_overrides,
    is_any_session_active,
    publish_multi_session_orb,
    publish_orb_alert,
    scan_orb_all_assets,
    scan_orb_all_sessions_all_assets,
)
