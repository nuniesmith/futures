"""
Tests for src.analysis.signal_quality — multi-factor signal quality scoring.

Covers: compute_signal_quality, is_premium_setup, signal_quality_summary.
"""

import numpy as np
import pandas as pd

from src.analysis.signal_quality import (
    compute_signal_quality,
    is_premium_setup,
    signal_quality_summary,
)

# ---------------------------------------------------------------------------
# Local synthetic data helpers (self-contained — no conftest dependency)
# ---------------------------------------------------------------------------


def _make_ohlcv_cap(n=500, start=150.0, seed=42):
    """Build OHLCV DataFrame with CAPITALIZED columns (Open, High, Low, Close, Volume)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(1000, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _make_uptrend_ohlcv(n=300, start=100.0, seed=10):
    """Build a clearly trending-up OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    drift = 0.003
    returns = rng.normal(drift, 0.002, n)
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0.3, 1, n) * spread
    low = close - rng.uniform(0, 0.7, n) * spread
    opn = close - rng.uniform(0, 0.5, n) * spread  # open < close → bullish bars
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(1200, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _make_downtrend_ohlcv(n=300, start=200.0, seed=20):
    """Build a clearly trending-down OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    drift = -0.003
    returns = rng.normal(drift, 0.002, n)
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 0.7, n) * spread
    low = close - rng.uniform(0.3, 1, n) * spread
    opn = close + rng.uniform(0, 0.5, n) * spread  # open > close → bearish bars
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(1200, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _make_ranging_ohlcv(n=300, start=150.0, seed=55):
    """Build a low-volatility, sideways OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.001, n)
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.0005, 0.002, n)
    high = close + spread
    low = close - spread
    opn = close + rng.uniform(-0.2, 0.2, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(800, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


# ===================================================================
# compute_signal_quality tests
# ===================================================================


class TestComputeSignalQuality:
    """Tests for the main compute_signal_quality function."""

    def test_returns_dict_with_required_keys(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        required_keys = {
            "score",
            "quality_pct",
            "high_quality",
            "factors",
            "market_context",
            "trend_direction",
            "rsi",
            "ao",
            "normalized_velocity",
            "price_acceleration",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_score_in_zero_one(self):
        """Signal quality score must be in [0, 1]."""
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert 0.0 <= result["score"] <= 1.0

    def test_quality_pct_in_zero_hundred(self):
        """quality_pct is score * 100, so must be in [0, 100]."""
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert 0.0 <= result["quality_pct"] <= 100.0

    def test_quality_pct_matches_score(self):
        """quality_pct should equal score * 100 (within rounding)."""
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert abs(result["quality_pct"] - result["score"] * 100) < 0.2

    def test_high_quality_bool(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert isinstance(result["high_quality"], bool)

    def test_high_quality_matches_threshold(self):
        """high_quality should be True when score >= threshold."""
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df, quality_threshold=0.0)
        assert result["high_quality"] is True

    def test_factors_dict_has_expected_keys(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        factors = result["factors"]
        assert isinstance(factors, dict)
        expected_factor_keys = {
            "vol_sweet_spot",
            "velocity_aligned",
            "candle_confirmation",
            "htf_bias",
        }
        assert expected_factor_keys.issubset(factors.keys()), (
            f"Missing factor keys: {expected_factor_keys - factors.keys()}"
        )

    def test_market_context_valid_label(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert result["market_context"] in {"UPTREND", "DOWNTREND", "RANGING"}

    def test_trend_direction_valid_label(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert result["trend_direction"] in {"BULLISH", "BEARISH", "NEUTRAL"}

    def test_rsi_in_valid_range(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert 0.0 <= result["rsi"] <= 100.0

    def test_ao_is_float(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        assert isinstance(result["ao"], float)

    # ── Empty / small DataFrame guard clause tests ──────────────────

    def test_empty_dataframe_returns_defaults(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = compute_signal_quality(df)
        assert result["score"] == 0.0
        assert result["quality_pct"] == 0.0
        assert result["high_quality"] is False
        assert result["market_context"] == "RANGING"
        assert result["trend_direction"] == "NEUTRAL"

    def test_none_input_returns_defaults(self):
        result = compute_signal_quality(None)
        assert result["score"] == 0.0
        assert result["high_quality"] is False

    def test_small_dataframe_returns_defaults(self):
        """DataFrames with fewer than 20 rows should return safe defaults."""
        df = _make_ohlcv_cap(n=10)
        result = compute_signal_quality(df)
        assert result["score"] == 0.0
        assert result["quality_pct"] == 0.0

    def test_missing_columns_returns_defaults(self):
        """DataFrame without proper OHLC columns should return defaults."""
        df = pd.DataFrame({"foo": range(50), "bar": range(50)})
        result = compute_signal_quality(df)
        assert result["score"] == 0.0

    # ── Pre-computed results integration ────────────────────────────

    def test_wave_result_accepted(self):
        """Passing a pre-computed wave_result should not crash."""
        df = _make_ohlcv_cap(n=300)
        wave_result = {
            "current_ratio": 1.5,
            "market_phase": "UPTREND",
            "trend_speed": 2.0,
            "bias": "BULLISH",
        }
        result = compute_signal_quality(df, wave_result=wave_result)
        assert 0.0 <= result["score"] <= 1.0

    def test_vol_result_accepted(self):
        """Passing a pre-computed vol_result should not crash."""
        df = _make_ohlcv_cap(n=300)
        vol_result = {"percentile": 0.45}
        result = compute_signal_quality(df, vol_result=vol_result)
        assert 0.0 <= result["score"] <= 1.0

    def test_uptrend_with_bullish_wave_boosts_score(self):
        """Uptrend data with bullish wave result should produce a reasonable score."""
        df = _make_uptrend_ohlcv(n=300)
        wave_result = {
            "current_ratio": 2.0,
            "market_phase": "UPTREND",
            "trend_speed": 3.0,
            "bias": "BULLISH",
        }
        vol_result = {"percentile": 0.45}
        result = compute_signal_quality(
            df, wave_result=wave_result, vol_result=vol_result
        )
        assert result["score"] > 0.0

    def test_vol_sweet_spot_factor_higher_when_in_range(self):
        """Vol percentile between 0.2 and 0.7 should give higher vol_sweet_spot."""
        df = _make_ohlcv_cap(n=300)
        wave_result = {
            "current_ratio": 1.5,
            "market_phase": "UPTREND",
            "trend_speed": 1.0,
            "bias": "BULLISH",
        }
        r_in = compute_signal_quality(
            df, wave_result=wave_result, vol_result={"percentile": 0.45}
        )
        r_out = compute_signal_quality(
            df, wave_result=wave_result, vol_result={"percentile": 0.05}
        )
        assert r_in["factors"]["vol_sweet_spot"] >= r_out["factors"]["vol_sweet_spot"]

    def test_custom_quality_threshold(self):
        """Custom quality_threshold should affect high_quality flag."""
        df = _make_ohlcv_cap(n=300)
        r_low = compute_signal_quality(df, quality_threshold=0.0)
        r_high = compute_signal_quality(df, quality_threshold=1.0)
        # With threshold 0.0, everything is high quality
        assert r_low["high_quality"] is True
        # With threshold 1.0, almost nothing is high quality
        assert r_high["high_quality"] is False

    def test_different_data_produces_different_scores(self):
        """Different input data should generally produce different scores."""
        df1 = _make_uptrend_ohlcv(n=300, seed=10)
        df2 = _make_downtrend_ohlcv(n=300, seed=20)
        r1 = compute_signal_quality(df1)
        r2 = compute_signal_quality(df2)
        # Scores or context should differ
        assert (
            r1["score"] != r2["score"]
            or r1["market_context"] != r2["market_context"]
            or r1["trend_direction"] != r2["trend_direction"]
        )


# ===================================================================
# is_premium_setup tests
# ===================================================================


class TestIsPremiumSetup:
    """Tests for the is_premium_setup gate function."""

    def test_returns_tuple_of_bool_and_str(self):
        quality_result = {"score": 0.5, "market_context": "RANGING"}
        result = is_premium_setup(quality_result)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_premium, direction = result
        assert isinstance(is_premium, bool)
        assert isinstance(direction, str)

    def test_low_score_not_premium(self):
        """Score <= 0.8 should never be premium."""
        quality_result = {"score": 0.5, "market_context": "UPTREND"}
        is_premium, direction = is_premium_setup(quality_result)
        assert is_premium is False
        assert direction == "NONE"

    def test_high_score_uptrend_bullish_is_premium_long(self):
        """Score > 0.8 + UPTREND context + BULLISH bias + good vol → premium LONG."""
        quality_result = {"score": 0.85, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.5}
        wave_result = {"bias": "BULLISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is True
        assert direction == "LONG"

    def test_high_score_downtrend_bearish_is_premium_short(self):
        """Score > 0.8 + DOWNTREND context + BEARISH bias + good vol → premium SHORT."""
        quality_result = {"score": 0.85, "market_context": "DOWNTREND"}
        vol_result = {"percentile": 0.5}
        wave_result = {"bias": "BEARISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is True
        assert direction == "SHORT"

    def test_extreme_vol_blocks_premium(self):
        """Extreme volatility (percentile >= 0.8) should block premium even with high score."""
        quality_result = {"score": 0.9, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.85}
        wave_result = {"bias": "BULLISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is False
        assert direction == "NONE"

    def test_very_low_vol_blocks_premium(self):
        """Very low volatility (percentile <= 0.2) should block premium."""
        quality_result = {"score": 0.9, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.15}
        wave_result = {"bias": "BULLISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is False
        assert direction == "NONE"

    def test_misaligned_bias_blocks_premium(self):
        """UPTREND with BEARISH bias should not be premium."""
        quality_result = {"score": 0.9, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.5}
        wave_result = {"bias": "BEARISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is False
        assert direction == "NONE"

    def test_neutral_bias_blocks_premium(self):
        """NEUTRAL bias should not be premium even with high score."""
        quality_result = {"score": 0.9, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.5}
        wave_result = {"bias": "NEUTRAL"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is False
        assert direction == "NONE"

    def test_ranging_context_not_premium(self):
        """RANGING context should not produce premium setups."""
        quality_result = {"score": 0.9, "market_context": "RANGING"}
        vol_result = {"percentile": 0.5}
        wave_result = {"bias": "BULLISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=wave_result
        )
        assert is_premium is False
        assert direction == "NONE"

    def test_no_vol_result_uses_default(self):
        """Without vol_result, should default to percentile=0.5 (within sweet spot)."""
        quality_result = {"score": 0.85, "market_context": "UPTREND"}
        wave_result = {"bias": "BULLISH"}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=None, wave_result=wave_result
        )
        assert is_premium is True
        assert direction == "LONG"

    def test_no_wave_result_defaults_neutral(self):
        """Without wave_result, bias defaults to NEUTRAL → not premium."""
        quality_result = {"score": 0.85, "market_context": "UPTREND"}
        vol_result = {"percentile": 0.5}
        is_premium, direction = is_premium_setup(
            quality_result, vol_result=vol_result, wave_result=None
        )
        assert is_premium is False
        assert direction == "NONE"


# ===================================================================
# signal_quality_summary tests
# ===================================================================


class TestSignalQualitySummary:
    """Tests for the signal_quality_summary formatter."""

    def test_returns_string(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert isinstance(text, str)

    def test_contains_quality_pct(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "%" in text

    def test_contains_context(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "context=" in text

    def test_contains_direction(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "direction=" in text

    def test_contains_rsi(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "RSI=" in text

    def test_contains_ao(self):
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "AO=" in text

    def test_high_quality_indicated_in_text(self):
        """If high_quality is True, summary should mention 'HIGH QUALITY'."""
        result = {
            "score": 0.8,
            "quality_pct": 80.0,
            "high_quality": True,
            "market_context": "UPTREND",
            "trend_direction": "BULLISH",
            "rsi": 65.0,
            "ao": 0.5,
        }
        text = signal_quality_summary(result)
        assert "HIGH QUALITY" in text

    def test_below_threshold_indicated_in_text(self):
        """If high_quality is False, summary should mention 'below threshold'."""
        result = {
            "score": 0.3,
            "quality_pct": 30.0,
            "high_quality": False,
            "market_context": "RANGING",
            "trend_direction": "NEUTRAL",
            "rsi": 50.0,
            "ao": 0.0,
        }
        text = signal_quality_summary(result)
        assert "below threshold" in text

    def test_works_with_default_result(self):
        """Should work even with the default/empty result dict."""
        default = {
            "score": 0.0,
            "quality_pct": 0.0,
            "high_quality": False,
            "market_context": "RANGING",
            "trend_direction": "NEUTRAL",
            "rsi": 50.0,
            "ao": 0.0,
        }
        text = signal_quality_summary(default)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_summary_is_single_line(self):
        """Summary text should be a single line (no newlines)."""
        df = _make_ohlcv_cap(n=300)
        result = compute_signal_quality(df)
        text = signal_quality_summary(result)
        assert "\n" not in text
