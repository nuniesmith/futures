"""
Redis caching layer for market data and computed indicators.

Falls back to in-memory dict if Redis is unavailable, so the app
still works without Docker / Redis running.
"""

import hashlib
import json
import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def _flatten_columns(df: "pd.DataFrame | None") -> pd.DataFrame:
    """Flatten MultiIndex columns returned by newer yfinance versions."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


try:
    import redis  # type: ignore[import-unresolved]

    _redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _r: "redis.Redis[bytes] | None" = redis.from_url(_redis_url, decode_responses=False)
    _r.ping()  # type: ignore[union-attr]
    REDIS_AVAILABLE = True
except Exception:
    _r = None
    REDIS_AVAILABLE = False

# Fallback in-memory cache when Redis is not available
_mem_cache: dict = {}

# Default TTLs in seconds
TTL_INTRADAY = 60  # 1-min / 5-min OHLCV
TTL_DAILY = 300  # daily bars (pivots, etc.)
TTL_INDICATOR = 120  # computed indicators (ATR, EMA, VWAP)
TTL_OPTIMIZATION = 3600  # optimization results (1 hour)


def _cache_key(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts)
    return "futures:" + hashlib.md5(raw.encode()).hexdigest()


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    return (df.to_json(date_format="iso") or "").encode()


def _bytes_to_df(raw: bytes) -> pd.DataFrame:
    df = pd.read_json(raw.decode())
    # Restore DatetimeIndex if the index looks like timestamps
    if not df.empty and df.index.dtype == "int64":
        pass  # leave numeric index as-is
    elif not df.empty:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------------
# Low-level get / set
# ---------------------------------------------------------------------------


def cache_get(key: str) -> bytes | None:
    if REDIS_AVAILABLE and _r is not None:
        return _r.get(key)
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    if datetime.utcnow().timestamp() > entry["expires"]:
        del _mem_cache[key]
        return None
    return entry["data"]


def cache_set(key: str, data: bytes, ttl: int) -> None:
    if REDIS_AVAILABLE and _r is not None:
        _r.setex(key, ttl, data)
    else:
        _mem_cache[key] = {
            "data": data,
            "expires": datetime.utcnow().timestamp() + ttl,
        }


# ---------------------------------------------------------------------------
# Market data fetching with cache
# ---------------------------------------------------------------------------


def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data, cached in Redis for TTL_INTRADAY seconds."""
    key = _cache_key("ohlcv", ticker, interval, period)
    cached = cache_get(key)
    if cached is not None:
        return _bytes_to_df(cached)

    df = _flatten_columns(
        yf.download(
            ticker, interval=interval, period=period, prepost=True, auto_adjust=True
        )
    )
    if not df.empty:
        cache_set(key, _df_to_bytes(df), TTL_INTRADAY)
    return df


def get_daily(ticker: str, period: str = "10d") -> pd.DataFrame:
    """Fetch daily bars, cached longer since they change less often."""
    key = _cache_key("daily", ticker, period)
    cached = cache_get(key)
    if cached is not None:
        return _bytes_to_df(cached)

    df = _flatten_columns(
        yf.download(ticker, interval="1d", period=period, auto_adjust=True)
    )
    if not df.empty:
        cache_set(key, _df_to_bytes(df), TTL_DAILY)
    return df


# ---------------------------------------------------------------------------
# Indicator caching
# ---------------------------------------------------------------------------


def get_cached_indicator(
    name: str, ticker: str, interval: str, period: str
) -> dict | None:
    """Return cached indicator dict or None."""
    key = _cache_key("ind", name, ticker, interval, period)
    cached = cache_get(key)
    if cached is not None:
        return json.loads(cached.decode())
    return None


def set_cached_indicator(
    name: str, ticker: str, interval: str, period: str, payload: dict
) -> None:
    key = _cache_key("ind", name, ticker, interval, period)
    cache_set(key, json.dumps(payload).encode(), TTL_INDICATOR)


# ---------------------------------------------------------------------------
# Optimization results caching
# ---------------------------------------------------------------------------


def get_cached_optimization(ticker: str, interval: str, period: str) -> dict | None:
    key = _cache_key("opt", ticker, interval, period)
    cached = cache_get(key)
    if cached is not None:
        return json.loads(cached.decode())
    return None


def set_cached_optimization(
    ticker: str, interval: str, period: str, result: dict
) -> None:
    key = _cache_key("opt", ticker, interval, period)
    cache_set(key, json.dumps(result).encode(), TTL_OPTIMIZATION)


def flush_all() -> None:
    """Clear all cached data (used by refresh button)."""
    if REDIS_AVAILABLE and _r is not None:
        for key in _r.scan_iter("futures:*"):
            _r.delete(key)
    else:
        _mem_cache.clear()
