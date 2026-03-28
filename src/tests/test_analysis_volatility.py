"""
Tests for src.analysis.volatility — K-Means adaptive volatility clustering.

Covers: kmeans_volatility_clusters, volatility_summary_text, should_filter_entry.
"""

import numpy as np
import pandas as pd

from analysis.volatility import (
    kmeans_volatility_clusters,
    should_filter_entry,
    volatility_summary_text,
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
    return pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume})


def _make_volatile_ohlcv(n=500, start=150.0, seed=99):
    """Build OHLCV with high volatility for testing HIGH cluster detection."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.025, n)  # 5x normal volatility
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.005, 0.020, n)
    high = close + spread
    low = close - spread
    opn = close + rng.uniform(-0.5, 0.5, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(2000, n).astype(float)
    return pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume})


def _make_calm_ohlcv(n=500, start=150.0, seed=77):
    """Build OHLCV with very low volatility for testing LOW cluster detection."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.0005, n)  # very calm
    close = start * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.0001, 0.0005, n)
    high = close + spread
    low = close - spread
    opn = close + rng.uniform(-0.1, 0.1, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(500, n).astype(float)
    return pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume})


# ===================================================================
# kmeans_volatility_clusters tests
# ===================================================================


class TestKmeansVolatilityClusters:
    """Tests for the main kmeans_volatility_clusters function."""

    def test_returns_dict_with_required_keys(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        required_keys = {
            "cluster",
            "adaptive_atr",
            "percentile",
            "position_multiplier",
            "sl_multiplier",
            "raw_atr",
            "vol_status",
            "volatility_regime",
            "centroids",
            "strategy_hint",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_cluster_is_valid_label(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert result["cluster"] in {"LOW", "MEDIUM", "HIGH"}

    def test_percentile_in_zero_one(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert 0.0 <= result["percentile"] <= 1.0

    def test_position_multiplier_valid_values(self):
        """Position multiplier must be one of the defined set."""
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert result["position_multiplier"] in {0.6, 1.0, 1.2}

    def test_sl_multiplier_valid_values(self):
        """SL multiplier must be one of the defined set."""
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert result["sl_multiplier"] in {0.8, 1.0, 1.2}

    def test_adaptive_atr_positive(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert result["adaptive_atr"] > 0.0

    def test_raw_atr_positive(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        assert result["raw_atr"] > 0.0

    def test_centroids_dict_has_three_clusters(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        centroids = result["centroids"]
        assert isinstance(centroids, dict)
        assert set(centroids.keys()) == {"LOW", "MEDIUM", "HIGH"}
        # Centroids should be sorted: LOW < MEDIUM < HIGH
        assert centroids["LOW"] <= centroids["MEDIUM"] <= centroids["HIGH"]

    def test_volatility_regime_is_string(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        valid_regimes = {"VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"}
        assert result["volatility_regime"] in valid_regimes

    def test_strategy_hint_is_string(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        valid_hints = {
            "AVOID TIGHT STOPS",
            "BREAKOUT WATCH",
            "WIDER STOPS",
            "TIGHTER STOPS",
            "NORMAL STOPS",
        }
        assert result["strategy_hint"] in valid_hints

    def test_empty_dataframe_returns_defaults(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = kmeans_volatility_clusters(df)
        assert result["cluster"] == "MEDIUM"
        assert result["percentile"] == 0.5
        assert result["position_multiplier"] == 1.0

    def test_none_input_returns_defaults(self):
        result = kmeans_volatility_clusters(None)
        assert result["cluster"] == "MEDIUM"
        assert result["percentile"] == 0.5

    def test_small_dataframe_returns_defaults(self):
        """DataFrames with fewer than 20 rows should return safe defaults."""
        df = _make_ohlcv_cap(n=10)
        result = kmeans_volatility_clusters(df)
        assert result["cluster"] == "MEDIUM"
        assert result["position_multiplier"] == 1.0

    def test_different_seeds_produce_different_results(self):
        """Ensure the function responds to different data inputs."""
        df1 = _make_ohlcv_cap(n=300, seed=42)
        df2 = _make_volatile_ohlcv(n=300, seed=99)
        r1 = kmeans_volatility_clusters(df1)
        r2 = kmeans_volatility_clusters(df2)
        # They should have different raw ATR values at minimum
        assert r1["raw_atr"] != r2["raw_atr"]

    def test_high_vol_cluster_lower_position_multiplier(self):
        """HIGH volatility cluster should get conservative position sizing (0.6x)."""
        # Build data with extreme final-bar volatility spike
        df = _make_volatile_ohlcv(n=500, seed=11)
        result = kmeans_volatility_clusters(df)
        if result["cluster"] == "HIGH":
            assert result["position_multiplier"] == 0.6
            assert result["sl_multiplier"] == 0.8

    def test_low_vol_cluster_higher_position_multiplier(self):
        """LOW volatility cluster should get aggressive position sizing (1.2x)."""
        df = _make_calm_ohlcv(n=500, seed=33)
        result = kmeans_volatility_clusters(df)
        if result["cluster"] == "LOW":
            assert result["position_multiplier"] == 1.2
            assert result["sl_multiplier"] == 1.2

    def test_custom_atr_len(self):
        """Custom ATR period should still produce valid results."""
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df, atr_len=7)
        assert result["cluster"] in {"LOW", "MEDIUM", "HIGH"}
        assert result["raw_atr"] > 0.0

    def test_custom_training_period(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df, training_period=100)
        assert result["cluster"] in {"LOW", "MEDIUM", "HIGH"}


# ===================================================================
# volatility_summary_text tests
# ===================================================================


class TestVolatilitySummaryText:
    """Tests for the volatility_summary_text formatter."""

    def test_returns_string(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        text = volatility_summary_text(result)
        assert isinstance(text, str)

    def test_contains_cluster_label(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        text = volatility_summary_text(result)
        assert result["cluster"] in text

    def test_contains_atr_value(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        text = volatility_summary_text(result)
        assert "ATR=" in text

    def test_contains_hint(self):
        df = _make_ohlcv_cap(n=300)
        result = kmeans_volatility_clusters(df)
        text = volatility_summary_text(result)
        assert "hint=" in text

    def test_works_with_default_result(self):
        """Should work even with the default/empty result dict."""
        default = {
            "cluster": "MEDIUM",
            "adaptive_atr": 0.0,
            "raw_atr": 0.0,
            "percentile": 0.5,
            "position_multiplier": 1.0,
            "strategy_hint": "NORMAL STOPS",
        }
        text = volatility_summary_text(default)
        assert isinstance(text, str)
        assert "MEDIUM" in text


# ===================================================================
# should_filter_entry tests
# ===================================================================


class TestShouldFilterEntry:
    """Tests for the should_filter_entry vol-based entry filter."""

    def test_returns_tuple_of_bool_and_str(self):
        result = {"percentile": 0.5}
        filtered, reason = should_filter_entry(result)
        assert isinstance(filtered, bool)
        assert isinstance(reason, str)

    def test_normal_percentile_not_filtered(self):
        """Percentile in the middle range should not filter."""
        result = {"percentile": 0.5}
        filtered, reason = should_filter_entry(result)
        assert filtered is False
        assert reason == ""

    def test_very_low_percentile_filtered(self):
        """Volatility below min_percentile should be filtered out."""
        result = {"percentile": 0.1}
        filtered, reason = should_filter_entry(result)
        assert filtered is True
        assert "too low" in reason.lower()

    def test_very_high_percentile_filtered(self):
        """Extreme volatility above max_percentile should be filtered out."""
        result = {"percentile": 0.95}
        filtered, reason = should_filter_entry(result)
        assert filtered is True
        assert "extreme" in reason.lower() or "volatility" in reason.lower()

    def test_custom_thresholds(self):
        """Custom min/max percentile thresholds should be respected."""
        result = {"percentile": 0.15}
        # Default min is 0.2, so 0.15 is filtered
        filtered, _ = should_filter_entry(result, min_percentile=0.2)
        assert filtered is True
        # But with a lower min, it passes
        filtered, _ = should_filter_entry(result, min_percentile=0.1)
        assert filtered is False

    def test_edge_at_min_threshold_not_filtered(self):
        """Percentile exactly at min_percentile should not be filtered."""
        result = {"percentile": 0.2}
        filtered, _ = should_filter_entry(result, min_percentile=0.2)
        assert filtered is False

    def test_edge_at_max_threshold_not_filtered(self):
        """Percentile exactly at max_percentile should not be filtered."""
        result = {"percentile": 0.9}
        filtered, _ = should_filter_entry(result, max_percentile=0.9)
        assert filtered is False

    def test_missing_percentile_key_uses_default(self):
        """If 'percentile' key is missing, should default to 0.5 (not filtered)."""
        result = {}
        filtered, _reason = should_filter_entry(result)
        assert filtered is False

    def test_integration_with_full_result(self):
        """End-to-end: cluster result fed directly into should_filter_entry."""
        df = _make_ohlcv_cap(n=300)
        vol_result = kmeans_volatility_clusters(df)
        filtered, reason = should_filter_entry(vol_result)
        assert isinstance(filtered, bool)
        assert isinstance(reason, str)
