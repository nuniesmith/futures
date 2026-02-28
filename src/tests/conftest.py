"""
Shared pytest fixtures for the futures test suite.

Provides synthetic OHLCV DataFrames that mirror real market data shapes
so every test module can exercise indicators, strategies, and detectors
without hitting the network.
"""

import os

# ---------------------------------------------------------------------------
# Disable Redis connections during tests so alert/cache tests don't hang
# waiting for a Redis server that isn't running locally.
# ---------------------------------------------------------------------------
os.environ.setdefault("DISABLE_REDIS", "1")

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure the `src/` package is importable from tests regardless of how
# pytest is invoked (repo root, tests/ dir, or via CI).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_timestamps(
    n: int, freq: str = "5min", start: str = "2025-01-06 03:00"
) -> pd.DatetimeIndex:
    """Generate a tz-aware (US/Eastern) DatetimeIndex for *n* bars."""
    return pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")


def _random_walk_ohlcv(
    n: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.005,
    freq: str = "5min",
    seed: int = 42,
    volume_mean: int = 1000,
) -> pd.DataFrame:
    """Build a realistic-ish OHLCV DataFrame via geometric random walk.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a tz-aware DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    # Derive O/H/L from Close with small random spread
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.5, 0.5, n) * spread

    # Ensure H >= max(O, C) and L <= min(O, C)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(volume_mean, n).astype(float)
    volume = np.maximum(volume, 1)  # no zero-volume bars

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _trending_ohlcv(
    n: int = 300,
    start_price: float = 5000.0,
    trend: float = 0.001,
    volatility: float = 0.003,
    freq: str = "5min",
    seed: int = 123,
) -> pd.DataFrame:
    """Build a clearly trending OHLCV DataFrame (positive drift).

    Useful for testing trend-following indicators / strategies.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.005, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(800, n).astype(float)
    volume = np.maximum(volume, 1)

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _gappy_ohlcv(
    n: int = 300,
    start_price: float = 2700.0,
    gap_bars: list[int] | None = None,
    gap_size: float = 0.01,
    freq: str = "5min",
    seed: int = 77,
) -> pd.DataFrame:
    """Build OHLCV with intentional gaps (for FVG / sweep testing).

    At each bar index in *gap_bars*, an upward or downward gap is injected
    so that candle-1's high < candle-3's low (or vice-versa).
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.003, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    # Inject gaps
    if gap_bars is None:
        gap_bars = [50, 100, 150, 200]

    for gb in gap_bars:
        if gb + 2 >= n or gb < 2:
            continue
        direction = rng.choice([-1, 1])
        shift = close[gb] * gap_size * direction
        for k in range(gb, n):
            close[k] += shift
            high[k] += shift
            low[k] += shift
            opn[k] += shift

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(1200, n).astype(float)
    volume = np.maximum(volume, 1)

    # Inject volume spikes near gap bars (simulate event/impulsive bars)
    for gb in gap_bars:
        if gb < n:
            volume[gb] *= 4
            if gb + 1 < n:
                volume[gb + 1] *= 3

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Generic 500-bar random-walk OHLCV DataFrame."""
    return _random_walk_ohlcv(n=500, seed=42)


@pytest.fixture()
def short_ohlcv_df() -> pd.DataFrame:
    """Short 50-bar DataFrame (edge-case / minimum-data tests)."""
    return _random_walk_ohlcv(n=50, seed=99, start_price=50.0)


@pytest.fixture()
def trending_df() -> pd.DataFrame:
    """300-bar trending DataFrame (positive drift)."""
    return _trending_ohlcv(n=300, seed=123)


@pytest.fixture()
def gappy_df() -> pd.DataFrame:
    """300-bar DataFrame with intentional price gaps for ICT tests."""
    return _gappy_ohlcv(n=300, seed=77)


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    """Empty OHLCV DataFrame for guard-clause tests."""
    return pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    """5-bar DataFrame (below most minimum-bar thresholds)."""
    return _random_walk_ohlcv(n=5, seed=11, start_price=20.0)


@pytest.fixture()
def gold_like_df() -> pd.DataFrame:
    """500-bar DataFrame at Gold-like price levels (~2700)."""
    return _random_walk_ohlcv(n=500, seed=55, start_price=2700.0, volatility=0.003)


@pytest.fixture()
def es_like_df() -> pd.DataFrame:
    """500-bar DataFrame at ES-like price levels (~5500)."""
    return _random_walk_ohlcv(n=500, seed=66, start_price=5500.0, volatility=0.002)
