"""
futures_lib.core — Core infrastructure modules.

Re-exports the public API from each sub-module so callers can do:

    from src.futures_lib.core import cache_get, cache_set, ASSETS, init_db
"""

from src.futures_lib.core.alerts import AlertDispatcher, get_dispatcher, send_risk_alert
from src.futures_lib.core.cache import (
    REDIS_AVAILABLE,
    TTL_DAILY,
    TTL_INTRADAY,
    _cache_key,
    _df_to_bytes,
    cache_get,
    cache_set,
    clear_cached_optimization,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    get_data_source,
    set_cached_indicator,
    set_cached_optimization,
)
from src.futures_lib.core.logging_config import get_logger, setup_logging
from src.futures_lib.core.models import (
    ACCOUNT_PROFILES,
    ASSETS,
    CONTRACT_MODE,
    CONTRACT_SPECS,
    DB_PATH,
    STATUS_CLOSED,
    STATUS_OPEN,
    TICKER_TO_NAME,
    cancel_trade,
    close_trade,
    create_trade,
    get_all_trades,
    get_closed_trades,
    get_daily_journal,
    get_journal_stats,
    get_max_contracts_for_profile,
    get_open_trades,
    get_today_pnl,
    init_db,
    save_daily_journal,
)

__all__ = [
    # alerts
    "AlertDispatcher",
    "get_dispatcher",
    "send_risk_alert",
    # cache
    "REDIS_AVAILABLE",
    "TTL_DAILY",
    "TTL_INTRADAY",
    "_cache_key",
    "_df_to_bytes",
    "cache_get",
    "cache_set",
    "clear_cached_optimization",
    "flush_all",
    "get_cached_indicator",
    "get_cached_optimization",
    "get_daily",
    "get_data",
    "get_data_source",
    "set_cached_indicator",
    "set_cached_optimization",
    # logging
    "get_logger",
    "setup_logging",
    # models
    "ACCOUNT_PROFILES",
    "ASSETS",
    "CONTRACT_MODE",
    "CONTRACT_SPECS",
    "DB_PATH",
    "STATUS_CLOSED",
    "STATUS_OPEN",
    "TICKER_TO_NAME",
    "cancel_trade",
    "close_trade",
    "create_trade",
    "get_all_trades",
    "get_closed_trades",
    "get_daily_journal",
    "get_journal_stats",
    "get_max_contracts_for_profile",
    "get_open_trades",
    "get_today_pnl",
    "init_db",
    "save_daily_journal",
]
