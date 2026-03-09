"""
rb.orb — Backward-Compatibility Shim
======================================
The Opening Range Breakout (ORB) sub-system was reorganised into the
``rb.open`` package.  This shim re-exports every public symbol from
``rb.open`` so that existing import sites of the form::

    from lib.trading.strategies.rb.orb import LONDON_SESSION
    from lib.trading.strategies.rb.orb import SESSION_ASSETS, ORB_SESSIONS

continue to work without modification.

Do **not** add new logic here.  All implementation lives in ``rb.open``.
"""

from lib.trading.strategies.rb.open import (  # noqa: F401
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
    # Legacy scalar aliases
    OR_START,
    ORB_SESSIONS,
    OVERNIGHT_SESSIONS,
    # --- Publisher ---
    REDIS_KEY_ORB,
    REDIS_KEY_ORB_TS,
    REDIS_PUBSUB_ORB,
    REDIS_TTL,
    # --- Assets ---
    SESSION_ASSETS,
    SESSION_BY_KEY,
    SHANGHAI_SESSION,
    SYDNEY_SESSION,
    TOKYO_SESSION,
    US_SESSION,
    MultiSessionORBResult,
    # --- Models ---
    ORBResult,
    # --- Sessions ---
    ORBSession,
    clear_orb_alert,
    # --- Detector ---
    compute_atr,
    compute_opening_range,
    detect_all_sessions,
    detect_opening_range_breakout,
    get_active_session_keys,
    get_active_sessions,
    get_session_assets,
    get_session_by_key,
    # Session helpers
    get_session_for_utc,
    get_session_status,
    get_symbol_session_overrides,
    is_any_session_active,
    publish_multi_session_orb,
    publish_orb_alert,
    scan_orb_all_assets,
    scan_orb_all_sessions_all_assets,
)
