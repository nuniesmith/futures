"""
Tests for src.indicators.helpers — functional indicator wrappers.

Covers: ema, sma, rsi, rsi_scalar, atr, atr_scalar, macd, bollinger, vwap,
        awesome_oscillator.
"""

import numpy as np
import pandas as pd

from src.indicators.helpers import (
    atr,
    atr_scalar,
    awesome_oscillator,
    bollinger,
    ema,
    macd,
    rsi,
    rsi_scalar,
    sma,
    vwap,
)

# ---------------------------------------------------------------------------
# Local synthetic data helpers (self-contained — no conftest dependency)
# ---------------------------------------------------------------------------


def _make_close(n=100, start=150.0, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    return pd.Series(start * np.exp(np.cumsum(returns)))


def _make_ohlcv(n=100, start=150.0, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    volume = rng.poisson(1000, n).astype(float)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


def _make_uptrend(n=100, start=100.0):
    """Strictly monotonically increasing prices."""
    return pd.Series(np.linspace(start, start + n, n))


def _make_downtrend(n=100, start=200.0):
    """Strictly monotonically decreasing prices."""
    return pd.Series(np.linspace(start, start - n, n))


# ===================================================================
# EMA tests
# ===================================================================


class TestEma:
    def test_returns_series(self):
        close = _make_close()
        result = ema(close, period=10)
        assert isinstance(result, pd.Series)

    def test_correct_length(self):
        close = _make_close(n=80)
        result = ema(close, period=10)
        assert len(result) == len(close)

    def test_tracks_uptrend(self):
        """EMA of an uptrend should end higher than it starts."""
        close = _make_uptrend(n=50)
        result = ema(close, period=10)
        assert result.iloc[-1] > result.iloc[0]

    def test_accepts_numpy_array(self):
        arr = _make_close(n=60).values
        result = ema(arr, period=12)
        assert isinstance(result, pd.Series)
        assert len(result) == 60

    def test_short_period_closer_to_price(self):
        """A shorter EMA period should track the price more tightly."""
        close = _make_close(n=100)
        fast = ema(close, period=5)
        slow = ema(close, period=50)
        # Mean absolute deviation from price should be smaller for fast EMA
        fast_mad = (close - fast).abs().mean()
        slow_mad = (close - slow).abs().mean()
        assert fast_mad < slow_mad


# ===================================================================
# SMA tests
# ===================================================================


class TestSma:
    def test_returns_series(self):
        close = _make_close()
        result = sma(close, period=10)
        assert isinstance(result, pd.Series)

    def test_correct_rolling_mean(self):
        """SMA of a constant series equals the constant."""
        constant = pd.Series([50.0] * 30)
        result = sma(constant, period=10)
        # After warm-up, all values should equal 50
        np.testing.assert_allclose(result.dropna().values, 50.0, atol=1e-10)

    def test_nan_prefix(self):
        """First (period - 1) values should be NaN."""
        close = _make_close(n=50)
        result = sma(close, period=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_correct_length(self):
        close = _make_close(n=75)
        result = sma(close, period=15)
        assert len(result) == 75


# ===================================================================
# RSI tests
# ===================================================================


class TestRsi:
    def test_returns_series(self):
        close = _make_close()
        result = rsi(close, period=14)
        assert isinstance(result, pd.Series)

    def test_values_in_range(self):
        """RSI values should be in [0, 100]."""
        close = _make_close(n=200)
        result = rsi(close, period=14)
        valid = result.dropna()
        assert (valid >= 0).all(), f"Min RSI: {valid.min()}"
        assert (valid <= 100).all(), f"Max RSI: {valid.max()}"

    def test_pure_uptrend_high_rsi(self):
        """Pure uptrend should produce RSI > 70."""
        close = _make_uptrend(n=100)
        result = rsi(close, period=14)
        # After warm-up, RSI should be very high
        assert result.iloc[-1] > 70

    def test_pure_downtrend_low_rsi(self):
        """Pure downtrend should produce RSI < 30."""
        close = _make_downtrend(n=100)
        result = rsi(close, period=14)
        assert result.iloc[-1] < 30

    def test_correct_length(self):
        close = _make_close(n=60)
        result = rsi(close, period=14)
        assert len(result) == 60


# ===================================================================
# RSI scalar tests
# ===================================================================


class TestRsiScalar:
    def test_returns_float(self):
        close = _make_close(n=50).values
        result = rsi_scalar(close, period=14)
        assert isinstance(result, float)

    def test_correct_range(self):
        close = _make_close(n=200).values
        result = rsi_scalar(close, period=14)
        assert 0.0 <= result <= 100.0

    def test_insufficient_data_returns_50(self):
        """With fewer bars than period+1, should return 50.0."""
        close = np.array([100.0, 101.0, 102.0])
        result = rsi_scalar(close, period=14)
        assert result == 50.0

    def test_pure_uptrend_returns_100(self):
        """All gains, no losses → RSI = 100."""
        close = np.linspace(100, 200, 50)
        result = rsi_scalar(close, period=14)
        assert result == 100.0

    def test_pure_downtrend_returns_near_zero(self):
        close = np.linspace(200, 100, 50)
        result = rsi_scalar(close, period=14)
        assert result < 5.0


# ===================================================================
# ATR tests
# ===================================================================


class TestAtr:
    def test_returns_series(self):
        df = _make_ohlcv()
        result = atr(df, period=14)
        assert isinstance(result, pd.Series)

    def test_positive_values(self):
        """ATR should always be positive (after warm-up)."""
        df = _make_ohlcv(n=100)
        result = atr(df, period=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_correct_length(self):
        df = _make_ohlcv(n=80)
        result = atr(df, period=14)
        assert len(result) == 80

    def test_accepts_capitalized_columns(self):
        """Should handle 'High', 'Low', 'Close' column names."""
        df = _make_ohlcv(n=60)
        df_cap = df.rename(columns={"high": "High", "low": "Low", "close": "Close"})
        result = atr(df_cap, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == 60


# ===================================================================
# ATR scalar tests
# ===================================================================


class TestAtrScalar:
    def test_returns_float(self):
        df = _make_ohlcv(n=50)
        result = atr_scalar(
            df["high"].values, df["low"].values, df["close"].values, period=14
        )
        assert isinstance(result, float)

    def test_positive_result(self):
        df = _make_ohlcv(n=100)
        result = atr_scalar(
            df["high"].values, df["low"].values, df["close"].values, period=14
        )
        assert result > 0.0

    def test_insufficient_data_returns_zero(self):
        """With too few bars, should return 0.0."""
        h = np.array([101.0, 102.0])
        lo = np.array([99.0, 98.0])
        c = np.array([100.0, 101.0])
        result = atr_scalar(h, lo, c, period=14)
        assert result == 0.0


# ===================================================================
# MACD tests
# ===================================================================


class TestMacd:
    def test_returns_dict_with_required_keys(self):
        close = _make_close(n=100)
        result = macd(close)
        assert isinstance(result, dict)
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result

    def test_all_values_are_series(self):
        close = _make_close(n=100)
        result = macd(close)
        for key in ("macd_line", "signal_line", "histogram"):
            assert isinstance(result[key], pd.Series), f"{key} is not pd.Series"

    def test_histogram_equals_macd_minus_signal(self):
        close = _make_close(n=200)
        result = macd(close)
        diff = result["macd_line"] - result["signal_line"]
        np.testing.assert_allclose(result["histogram"].values, diff.values, atol=1e-10)

    def test_correct_length(self):
        close = _make_close(n=60)
        result = macd(close, fast=12, slow=26, signal=9)
        for key in ("macd_line", "signal_line", "histogram"):
            assert len(result[key]) == 60

    def test_accepts_numpy_array(self):
        arr = _make_close(n=80).values
        result = macd(arr)
        assert isinstance(result["macd_line"], pd.Series)


# ===================================================================
# Bollinger Bands tests
# ===================================================================


class TestBollinger:
    def test_returns_dict_with_required_keys(self):
        close = _make_close(n=100)
        result = bollinger(close)
        assert isinstance(result, dict)
        for key in ("upper", "lower", "middle", "bandwidth", "percent_b"):
            assert key in result, f"Missing key: {key}"

    def test_upper_gt_middle_gt_lower(self):
        """After warm-up, upper > middle > lower must hold."""
        close = _make_close(n=100)
        result = bollinger(close, period=20, std_dev=2.0)
        valid_idx = result["middle"].dropna().index
        upper = result["upper"].loc[valid_idx]
        middle = result["middle"].loc[valid_idx]
        lower = result["lower"].loc[valid_idx]
        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_correct_length(self):
        close = _make_close(n=50)
        result = bollinger(close, period=20)
        for key in ("upper", "lower", "middle"):
            assert len(result[key]) == 50

    def test_constant_price_zero_bandwidth(self):
        """Constant price → std = 0 → bandwidth ≈ 0 and upper ≈ lower ≈ middle."""
        constant = pd.Series([100.0] * 40)
        result = bollinger(constant, period=20, std_dev=2.0)
        valid = result["bandwidth"].dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-8)


# ===================================================================
# VWAP tests
# ===================================================================


class TestVwap:
    def test_returns_series(self):
        df = _make_ohlcv(n=100)
        result = vwap(df)
        assert isinstance(result, pd.Series)

    def test_correct_length(self):
        df = _make_ohlcv(n=75)
        result = vwap(df)
        assert len(result) == 75

    def test_within_high_low_range(self):
        """VWAP should be bounded by the cumulative price range."""
        df = _make_ohlcv(n=100)
        result = vwap(df)
        valid = result.dropna()
        # VWAP should be between global min(low) and max(high)
        assert valid.min() >= df["low"].min() * 0.999  # tiny tolerance
        assert valid.max() <= df["high"].max() * 1.001

    def test_accepts_capitalized_columns(self):
        df = _make_ohlcv(n=50)
        df_cap = df.rename(
            columns={"high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
        result = vwap(df_cap)
        assert isinstance(result, pd.Series)
        assert len(result) == 50


# ===================================================================
# Awesome Oscillator tests
# ===================================================================


class TestAwesomeOscillator:
    def test_returns_float(self):
        df = _make_ohlcv(n=50)
        result = awesome_oscillator(df["high"].values, df["low"].values)
        assert isinstance(result, float)

    def test_insufficient_data_returns_zero(self):
        """With fewer than 34 bars, should return 0.0."""
        h = np.array([101.0] * 10)
        lo = np.array([99.0] * 10)
        result = awesome_oscillator(h, lo)
        assert result == 0.0

    def test_uptrend_positive_ao(self):
        """In a strong uptrend, the fast SMA of hl2 should exceed the slow SMA."""
        n = 60
        high = np.linspace(100, 200, n) + 1.0
        low = np.linspace(100, 200, n) - 1.0
        result = awesome_oscillator(high, low, fast=5, slow=34)
        assert result > 0.0

    def test_downtrend_negative_ao(self):
        """In a strong downtrend, fast SMA of hl2 should be below slow SMA."""
        n = 60
        high = np.linspace(200, 100, n) + 1.0
        low = np.linspace(200, 100, n) - 1.0
        result = awesome_oscillator(high, low, fast=5, slow=34)
        assert result < 0.0

    def test_accepts_pandas_series(self):
        df = _make_ohlcv(n=50)
        result = awesome_oscillator(df["high"], df["low"])
        assert isinstance(result, float)
