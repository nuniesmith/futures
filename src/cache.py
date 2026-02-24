"""
Redis caching layer for market data and computed indicators.

Falls back to in-memory dict if Redis is unavailable, so the app
still works without Docker / Redis running.
"""

import hashlib
import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import yfinance as yf


def _flatten_columns(df: "pd.DataFrame | None") -> pd.DataFrame:
    """Flatten MultiIndex columns returned by newer yfinance versions."""
    if df is None or df.empty:
        return pd.DataFrame()
    # At this point df is guaranteed to be a non-empty DataFrame
    result: pd.DataFrame = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        # Flatten to single level: take first level names only
        result.columns = pd.Index(
            [col[0] if isinstance(col, tuple) else col for col in result.columns]
        )
    # Remove duplicate columns (keep first occurrence)
    mask = ~pd.Index(result.columns).duplicated(keep="first")
    result = result.loc[:, mask]
    # Reset column names to plain strings to avoid any leftover index weirdness
    result.columns = pd.Index([str(c) for c in result.columns])
    # Drop rows with NaN in OHLCV columns (yfinance sometimes returns partial rows)
    ohlcv = [
        c for c in ("Open", "High", "Low", "Close", "Volume") if c in result.columns
    ]
    if ohlcv:
        result = result.dropna(subset=ohlcv)
    return result


try:
    import redis  # type: ignore[import-unresolved]

    _redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _r: "redis.Redis | None" = redis.from_url(_redis_url, decode_responses=False)
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
    # Safety net: deduplicate columns before serialising to JSON
    if df.columns.duplicated().any():
        mask = ~pd.Index(df.columns).duplicated(keep="first")
        df = df.loc[:, mask]
    return (df.to_json(date_format="iso") or "").encode()


def _bytes_to_df(raw: bytes) -> pd.DataFrame:
    df = pd.read_json(StringIO(raw.decode()))
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
        result = _r.get(key)
        if isinstance(result, bytes):
            return result
        return None
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    if datetime.now().timestamp() > entry["expires"]:
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
# Yahoo Finance interval → max period limits
# ---------------------------------------------------------------------------
# Yahoo enforces these caps on intraday data. Requesting beyond them returns
# an empty frame or a "possibly delisted" error.
_YF_MAX_PERIOD: dict[str, list[str]] = {
    # interval → ordered list of allowed periods (largest last)
    "1m": ["1d", "5d"],
    "2m": ["1d", "5d", "15d", "1mo"],
    "5m": ["1d", "5d", "15d", "1mo"],
    "15m": ["1d", "5d", "15d", "1mo"],
    "30m": ["1d", "5d", "15d", "1mo"],
    "60m": ["1d", "5d", "15d", "1mo", "3mo", "6mo"],
    "1h": ["1d", "5d", "15d", "1mo", "3mo", "6mo"],
    "90m": ["1d", "5d", "15d", "1mo", "3mo", "6mo"],
    "1d": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "5d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1mo": ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "3mo": ["1y", "2y", "5y", "10y", "max"],
}

# Numeric ordering so we can compare periods
_PERIOD_RANK: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "15d": 15,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "max": 99999,
}


def _clamp_period(interval: str, period: str) -> str:
    """Return the largest Yahoo-allowed period that is ≤ the requested one.

    If the requested period exceeds the interval's max, it is clamped down
    and a message is printed so the user knows.
    """
    allowed = _YF_MAX_PERIOD.get(interval)
    if allowed is None:
        # Unknown interval – pass through and let Yahoo decide
        return period

    req_rank = _PERIOD_RANK.get(period, 90)

    # If the requested period is within the allowed list, use it directly
    if period in allowed:
        return period

    # Otherwise find the largest allowed period that doesn't exceed the request
    best = allowed[0]  # smallest fallback
    for p in allowed:
        if _PERIOD_RANK.get(p, 0) <= req_rank:
            best = p

    if best != period:
        print(
            f"[cache] Clamped period {period!r} → {best!r} for interval {interval!r} "
            f"(Yahoo limit)"
        )
    return best


# ---------------------------------------------------------------------------
# Market data fetching with cache
# ---------------------------------------------------------------------------


def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data, cached in Redis for TTL_INTRADAY seconds.

    Automatically clamps the period to Yahoo Finance's maximum for the
    requested interval to avoid empty responses.
    """
    period = _clamp_period(interval, period)
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


def clear_cached_optimization(ticker: str, interval: str, period: str) -> None:
    """Remove cached optimization result so the next run re-optimizes."""
    key = _cache_key("opt", ticker, interval, period)
    if REDIS_AVAILABLE and _r is not None:
        _r.delete(key)
    elif key in _mem_cache:
        del _mem_cache[key]


def flush_all() -> None:
    """Clear all cached data (used by refresh button)."""
    if REDIS_AVAILABLE and _r is not None:
        for key in _r.scan_iter("futures:*"):
            _r.delete(key)
    else:
        _mem_cache.clear()
