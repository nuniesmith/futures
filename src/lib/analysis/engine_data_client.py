"""
Engine Data Client — re-export shim.

The canonical implementation has moved to:
    lib.services.data.engine_data_client

This module re-exports everything so existing ``from lib.analysis.engine_data_client import ...``
imports continue to work without change.
"""

from lib.services.data.engine_data_client import (  # noqa: F401
    EngineDataClient,
    StaticBarProvider,
    _cache_get,
    _cache_key,
    _cache_set,
    clear_cache,
    get_bars,
    get_bars_bulk,
    get_client,
    get_daily_bars,
    get_htf_bars,
    get_snapshot,
    get_symbols,
    reset_client,
)

__all__ = [
    "EngineDataClient",
    "StaticBarProvider",
    "get_client",
    "reset_client",
    "clear_cache",
    "get_bars",
    "get_daily_bars",
    "get_htf_bars",
    "get_snapshot",
    "get_symbols",
    "get_bars_bulk",
]
