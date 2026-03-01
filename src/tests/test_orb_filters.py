"""
Tests for ORB Quality Filters — lib.analysis.orb_filters
==========================================================
Covers all six filters and the composite ``apply_all_filters`` function.

Each filter is tested for:
  - Pass / fail logic with realistic inputs
  - Edge cases (missing data, empty DataFrames, boundary times)
  - Score boost values
  - Correct reason strings

The composite function is tested for:
  - "all" gate mode (every hard filter must pass)
  - "majority" gate mode (> 50% of hard filters pass)
  - Soft filter (NR7) never rejecting
  - Quality boost aggregation
  - Enable/disable toggles
"""

from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.analysis.orb_filters import (
    FilterVerdict,
    ORBFilterResult,
    apply_all_filters,
    check_lunch_filter,
    check_multi_tf_bias,
    check_nr7,
    check_premarket_range,
    check_session_window,
    check_vwap_confluence,
    compute_session_vwap,
    extract_premarket_range,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_bars(ranges: list[float], base_price: float = 100.0) -> pd.DataFrame:
    """Create a simple daily OHLCV DataFrame from a list of daily ranges.

    Each day: Open = base_price, High = Open + range/2, Low = Open - range/2,
              Close = Open, Volume = 1000.
    """
    n = len(ranges)
    dates = pd.date_range("2026-02-20", periods=n, freq="B", tz=_EST)
    data = {
        "Open": [base_price] * n,
        "High": [base_price + r / 2 for r in ranges],
        "Low": [base_price - r / 2 for r in ranges],
        "Close": [base_price] * n,
        "Volume": [1000] * n,
    }
    return pd.DataFrame(data, index=dates)


def _make_1m_bars(
    n: int = 120,
    start_hour: int = 6,
    start_minute: int = 0,
    base_price: float = 2340.0,
    vol: float = 500.0,
    date_str: str = "2026-02-27",
) -> pd.DataFrame:
    """Create 1-minute OHLCV bars starting at a given ET hour."""
    start = pd.Timestamp(f"{date_str} {start_hour:02d}:{start_minute:02d}:00", tz=_EST)
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.RandomState(42)
    closes = base_price + np.cumsum(rng.randn(n) * 0.5)
    data = {
        "Open": closes - rng.rand(n) * 0.2,
        "High": closes + rng.rand(n) * 0.8,
        "Low": closes - rng.rand(n) * 0.8,
        "Close": closes,
        "Volume": vol + rng.rand(n) * 200,
    }
    return pd.DataFrame(data, index=idx)


def _make_htf_bars(
    n: int = 50,
    trend: str = "up",
    base_price: float = 2340.0,
) -> pd.DataFrame:
    """Create 15-minute OHLCV bars with a clear trend direction."""
    idx = pd.date_range("2026-02-27 06:00", periods=n, freq="15min", tz=_EST)
    if trend == "up":
        closes = base_price + np.arange(n) * 0.5
    elif trend == "down":
        closes = base_price - np.arange(n) * 0.5
    else:
        closes = np.full(n, base_price)

    data = {
        "Open": closes - 0.1,
        "High": closes + 0.3,
        "Low": closes - 0.3,
        "Close": closes,
        "Volume": np.full(n, 1000.0),
    }
    return pd.DataFrame(data, index=idx)


def _et_time(hour: int, minute: int = 0) -> datetime:
    """Create a tz-aware datetime for today at the given ET hour:minute."""
    return datetime(2026, 2, 27, hour, minute, 0, tzinfo=_EST)


# ============================================================================
# 1. NR7 Filter
# ============================================================================


class TestNR7:
    def test_nr7_detected(self):
        """When today's range is the narrowest of 7 days, NR7 fires."""
        ranges = [5.0, 6.0, 4.5, 7.0, 3.8, 4.2, 2.0]  # last is narrowest
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=7)
        assert v.passed is True  # NR7 never rejects
        assert v.score_boost > 0
        assert "NR7 active" in v.reason

    def test_nr7_not_detected(self):
        """When today's range is NOT the narrowest, no boost."""
        ranges = [2.0, 6.0, 4.5, 7.0, 3.8, 4.2, 5.0]  # last is NOT narrowest
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=7)
        assert v.passed is True
        assert v.score_boost == 0.0
        assert "Not NR7" in v.reason

    def test_nr7_tied_range(self):
        """When today's range ties the minimum, NR7 should still fire."""
        ranges = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=7)
        assert v.passed is True
        assert v.score_boost > 0
        assert "NR7 active" in v.reason

    def test_nr7_insufficient_bars(self):
        """With fewer bars than lookback, filter is skipped."""
        bars = _make_daily_bars([3.0, 4.0, 5.0])
        v = check_nr7(bars, lookback=7)
        assert v.passed is True
        assert v.score_boost == 0.0
        assert "skipped" in v.reason.lower()

    def test_nr7_empty_dataframe(self):
        v = check_nr7(pd.DataFrame())
        assert v.passed is True

    def test_nr7_none_input(self):
        v = check_nr7(None)
        assert v.passed is True

    def test_nr7_custom_lookback(self):
        ranges = [10.0, 8.0, 6.0, 4.0, 2.0]
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=5)
        assert v.score_boost > 0  # last is narrowest of 5

    def test_nr7_custom_boost(self):
        ranges = [5.0, 6.0, 4.5, 7.0, 3.8, 4.2, 2.0]
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=7, boost_pct=0.50)
        assert v.score_boost == 0.50

    def test_nr7_zero_range_days(self):
        """All zero-range days should be handled gracefully."""
        ranges = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bars = _make_daily_bars(ranges)
        v = check_nr7(bars, lookback=7)
        assert v.passed is True
        assert "skipped" in v.reason.lower() or "NR7" in v.reason

    def test_nr7_missing_columns(self):
        """Missing High/Low columns should be handled gracefully."""
        df = pd.DataFrame({"Close": [1, 2, 3, 4, 5, 6, 7]})
        v = check_nr7(df, lookback=7)
        assert v.passed is True
        assert "missing" in v.reason.lower() or "columns" in v.reason.lower()


# ============================================================================
# 2. Pre-Market Range Filter
# ============================================================================


class TestPreMarketRange:
    def test_long_clears_pm_high(self):
        v = check_premarket_range("LONG", trigger_price=2350.0, premarket_high=2345.0)
        assert v.passed is True
        assert "clears PM high" in v.reason

    def test_long_below_pm_high(self):
        v = check_premarket_range("LONG", trigger_price=2340.0, premarket_high=2345.0)
        assert v.passed is False
        assert "below PM high" in v.reason

    def test_short_clears_pm_low(self):
        v = check_premarket_range("SHORT", trigger_price=2330.0, premarket_low=2335.0)
        assert v.passed is True
        assert "clears PM low" in v.reason

    def test_short_above_pm_low(self):
        v = check_premarket_range("SHORT", trigger_price=2340.0, premarket_low=2335.0)
        assert v.passed is False
        assert "above PM low" in v.reason

    def test_long_no_pm_high(self):
        """When pre-market high is not available, filter is skipped."""
        v = check_premarket_range("LONG", trigger_price=2350.0, premarket_high=None)
        assert v.passed is True
        assert "skipped" in v.reason.lower()

    def test_short_no_pm_low(self):
        v = check_premarket_range("SHORT", trigger_price=2330.0, premarket_low=None)
        assert v.passed is True

    def test_tolerance(self):
        """Trigger price very close to PM high should still pass with tolerance."""
        # tolerance_pct=0.001 means 0.1% — so 2345 * 0.999 = 2342.655
        v = check_premarket_range("LONG", trigger_price=2344.0, premarket_high=2345.0, tolerance_pct=0.001)
        assert v.passed is True

    def test_unknown_direction(self):
        v = check_premarket_range("NEUTRAL", trigger_price=2340.0, premarket_high=2345.0)
        assert v.passed is True
        assert "skipped" in v.reason.lower()

    def test_score_boost_on_clearance(self):
        """Trigger price well above PM high should get a score boost."""
        v = check_premarket_range("LONG", trigger_price=2350.0, premarket_high=2345.0)
        assert v.score_boost > 0

    def test_score_boost_on_short_clearance(self):
        v = check_premarket_range("SHORT", trigger_price=2330.0, premarket_low=2335.0)
        assert v.score_boost > 0

    def test_zero_pm_high(self):
        v = check_premarket_range("LONG", trigger_price=2350.0, premarket_high=0.0)
        assert v.passed is True

    def test_case_insensitive_direction(self):
        v = check_premarket_range("long", trigger_price=2350.0, premarket_high=2345.0)
        assert v.passed is True

    def test_direction_with_whitespace(self):
        v = check_premarket_range("  SHORT  ", trigger_price=2330.0, premarket_low=2335.0)
        assert v.passed is True


# ============================================================================
# 3. Session Window Filter
# ============================================================================


class TestSessionWindow:
    def test_inside_default_window(self):
        signal_time = _et_time(9, 0)
        v = check_session_window(signal_time)
        assert v.passed is True
        assert "within window" in v.reason

    def test_before_window(self):
        signal_time = _et_time(7, 0)
        v = check_session_window(signal_time)
        assert v.passed is False
        assert "outside" in v.reason

    def test_after_window(self):
        signal_time = _et_time(12, 0)
        v = check_session_window(signal_time)
        assert v.passed is False

    def test_at_window_start(self):
        signal_time = _et_time(8, 20)
        v = check_session_window(signal_time)
        assert v.passed is True

    def test_at_window_end(self):
        signal_time = _et_time(10, 30)
        v = check_session_window(signal_time)
        assert v.passed is True

    def test_custom_windows(self):
        """Custom window: only 13:30–14:00 allowed."""
        windows = [(dt_time(13, 30), dt_time(14, 0))]
        signal_time = _et_time(13, 45)
        v = check_session_window(signal_time, allowed_windows=windows)
        assert v.passed is True

        v2 = check_session_window(_et_time(9, 0), allowed_windows=windows)
        assert v2.passed is False

    def test_multiple_windows(self):
        """Multiple disjoint windows."""
        windows = [
            (dt_time(8, 0), dt_time(9, 0)),
            (dt_time(14, 0), dt_time(15, 0)),
        ]
        assert check_session_window(_et_time(8, 30), allowed_windows=windows).passed is True
        assert check_session_window(_et_time(14, 30), allowed_windows=windows).passed is True
        assert check_session_window(_et_time(12, 0), allowed_windows=windows).passed is False

    def test_naive_datetime_assumed_et(self):
        """Naive datetime (no tz) should be treated as Eastern."""
        naive = datetime(2026, 2, 27, 9, 30, 0)  # no tz
        v = check_session_window(naive)
        assert v.passed is True

    def test_utc_datetime_converted(self):
        """UTC datetime should be converted to ET before checking."""
        utc_tz = ZoneInfo("UTC")
        # 14:00 UTC = 09:00 ET (during EST, UTC-5)
        utc_time = datetime(2026, 2, 27, 14, 0, 0, tzinfo=utc_tz)
        v = check_session_window(utc_time)
        assert v.passed is True


# ============================================================================
# 4. Lunch / Dead-Zone Filter
# ============================================================================


class TestLunchFilter:
    def test_during_lunch(self):
        signal_time = _et_time(11, 30)
        v = check_lunch_filter(signal_time)
        assert v.passed is False
        assert "dead zone" in v.reason

    def test_before_lunch(self):
        v = check_lunch_filter(_et_time(9, 30))
        assert v.passed is True

    def test_after_lunch(self):
        v = check_lunch_filter(_et_time(14, 0))
        assert v.passed is True

    def test_at_lunch_start(self):
        v = check_lunch_filter(_et_time(10, 30))
        assert v.passed is False

    def test_at_lunch_end(self):
        v = check_lunch_filter(_et_time(13, 30))
        assert v.passed is False

    def test_custom_lunch_window(self):
        """Narrow lunch window: 12:00–12:30."""
        v = check_lunch_filter(_et_time(12, 15), lunch_start=dt_time(12, 0), lunch_end=dt_time(12, 30))
        assert v.passed is False

        v2 = check_lunch_filter(_et_time(11, 0), lunch_start=dt_time(12, 0), lunch_end=dt_time(12, 30))
        assert v2.passed is True

    def test_naive_datetime(self):
        naive = datetime(2026, 2, 27, 12, 0, 0)
        v = check_lunch_filter(naive)
        assert v.passed is False


# ============================================================================
# 5. Multi-TF EMA Bias
# ============================================================================


class TestMultiTFBias:
    def test_long_with_uptrend(self):
        bars = _make_htf_bars(50, trend="up")
        v = check_multi_tf_bias(bars, "LONG")
        assert v.passed is True
        assert "agrees" in v.reason.lower()

    def test_long_with_downtrend(self):
        bars = _make_htf_bars(50, trend="down")
        v = check_multi_tf_bias(bars, "LONG")
        assert v.passed is False
        assert "counter-trend" in v.reason.lower()

    def test_short_with_downtrend(self):
        bars = _make_htf_bars(50, trend="down")
        v = check_multi_tf_bias(bars, "SHORT")
        assert v.passed is True

    def test_short_with_uptrend(self):
        bars = _make_htf_bars(50, trend="up")
        v = check_multi_tf_bias(bars, "SHORT")
        assert v.passed is False
        assert "counter-trend" in v.reason.lower()

    def test_flat_trend_long(self):
        """Flat trend should pass for both directions."""
        bars = _make_htf_bars(50, trend="flat")
        v = check_multi_tf_bias(bars, "LONG")
        assert v.passed is True

    def test_flat_trend_short(self):
        bars = _make_htf_bars(50, trend="flat")
        v = check_multi_tf_bias(bars, "SHORT")
        assert v.passed is True

    def test_no_htf_bars(self):
        v = check_multi_tf_bias(None, "LONG")
        assert v.passed is True
        assert "skipped" in v.reason.lower()

    def test_empty_htf_bars(self):
        v = check_multi_tf_bias(pd.DataFrame(), "LONG")
        assert v.passed is True

    def test_insufficient_bars(self):
        bars = _make_htf_bars(5, trend="up")  # 5 bars < ema_period(34) + slope_bars(3) + 1
        v = check_multi_tf_bias(bars, "LONG")
        assert v.passed is True
        assert "skipped" in v.reason.lower()

    def test_score_boost_on_aligned_trend(self):
        bars = _make_htf_bars(50, trend="up")
        v = check_multi_tf_bias(bars, "LONG")
        assert v.score_boost > 0

    def test_custom_ema_period(self):
        """Shorter EMA period should work with fewer bars."""
        bars = _make_htf_bars(20, trend="up")
        v = check_multi_tf_bias(bars, "LONG", ema_period=9, slope_bars=2)
        assert v.passed is True

    def test_unknown_direction(self):
        bars = _make_htf_bars(50, trend="up")
        v = check_multi_tf_bias(bars, "HOLD")
        assert v.passed is True
        assert "skipped" in v.reason.lower()


# ============================================================================
# 6. VWAP Confluence
# ============================================================================


class TestVWAPConfluence:
    def test_long_above_vwap(self):
        v = check_vwap_confluence(None, "LONG", trigger_price=2350.0, vwap=2340.0)
        assert v.passed is True
        assert v.score_boost > 0

    def test_long_below_vwap(self):
        v = check_vwap_confluence(None, "LONG", trigger_price=2330.0, vwap=2340.0)
        assert v.passed is False
        assert "below VWAP" in v.reason

    def test_short_below_vwap(self):
        v = check_vwap_confluence(None, "SHORT", trigger_price=2330.0, vwap=2340.0)
        assert v.passed is True
        assert v.score_boost > 0

    def test_short_above_vwap(self):
        v = check_vwap_confluence(None, "SHORT", trigger_price=2350.0, vwap=2340.0)
        assert v.passed is False
        assert "above VWAP" in v.reason

    def test_no_vwap_available(self):
        v = check_vwap_confluence(pd.DataFrame(), "LONG", trigger_price=2350.0)
        assert v.passed is True
        assert "unavailable" in v.reason.lower()

    def test_compute_vwap_from_bars(self):
        """When no VWAP is pre-computed, it should be computed from bars."""
        bars = _make_1m_bars(60, start_hour=9, start_minute=30, base_price=2340.0)
        vwap = compute_session_vwap(bars)
        assert vwap is not None
        assert 2300.0 < vwap < 2380.0  # sanity check

    def test_unknown_direction(self):
        v = check_vwap_confluence(None, "NEUTRAL", trigger_price=2350.0, vwap=2340.0)
        assert v.passed is True


# ============================================================================
# 7. VWAP Computation
# ============================================================================


class TestComputeSessionVWAP:
    def test_basic(self):
        bars = _make_1m_bars(30, base_price=100.0, vol=1000.0)
        vwap = compute_session_vwap(bars)
        assert vwap is not None
        assert 90.0 < vwap < 120.0

    def test_empty_bars(self):
        assert compute_session_vwap(pd.DataFrame()) is None

    def test_none_bars(self):
        assert compute_session_vwap(None) is None

    def test_single_bar(self):
        assert compute_session_vwap(_make_1m_bars(1)) is None

    def test_zero_volume(self):
        bars = _make_1m_bars(10)
        bars["Volume"] = 0
        assert compute_session_vwap(bars) is None


# ============================================================================
# 8. Pre-Market Range Extraction
# ============================================================================


class TestExtractPremarketRange:
    def test_extract_from_bars(self):
        bars = _make_1m_bars(n=200, start_hour=4, base_price=2340.0)
        pm_high, pm_low = extract_premarket_range(bars)
        assert pm_high is not None
        assert pm_low is not None
        assert pm_high > pm_low

    def test_empty_bars(self):
        pm_high, pm_low = extract_premarket_range(pd.DataFrame())
        assert pm_high is None
        assert pm_low is None

    def test_none_bars(self):
        pm_high, pm_low = extract_premarket_range(None)
        assert pm_high is None
        assert pm_low is None

    def test_no_bars_in_pm_window(self):
        """Bars only during regular session should return None."""
        bars = _make_1m_bars(60, start_hour=10, base_price=2340.0)
        pm_high, pm_low = extract_premarket_range(bars)
        assert pm_high is None
        assert pm_low is None


# ============================================================================
# 9. Composite: apply_all_filters
# ============================================================================


class TestApplyAllFilters:
    """Tests for the composite filter orchestrator."""

    def _good_params(self) -> dict:
        """Return params that should pass ALL filters."""
        return dict(
            direction="LONG",
            trigger_price=2350.0,
            signal_time=_et_time(9, 30),
            bars_daily=_make_daily_bars([5, 6, 4.5, 7, 3.8, 4.2, 2.0]),
            bars_1m=_make_1m_bars(120, start_hour=4, base_price=2340.0),
            bars_htf=_make_htf_bars(50, trend="up", base_price=2340.0),
            premarket_high=2345.0,
            premarket_low=2330.0,
            vwap=2340.0,
        )

    def test_all_filters_pass(self):
        result = apply_all_filters(**self._good_params())
        assert isinstance(result, ORBFilterResult)
        assert result.passed is True
        assert result.filters_passed == result.filters_total
        assert "PASS" in result.summary

    def test_result_to_dict(self):
        result = apply_all_filters(**self._good_params())
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "verdicts" in d
        assert isinstance(d["verdicts"], list)

    def test_session_window_rejects(self):
        """Signal outside session window should fail in 'all' mode."""
        params = self._good_params()
        params["signal_time"] = _et_time(14, 0)  # after hours
        result = apply_all_filters(**params)
        assert result.passed is False
        assert "Session Window" in result.summary or "Lunch" in result.summary

    def test_lunch_filter_rejects(self):
        params = self._good_params()
        params["signal_time"] = _et_time(12, 0)  # lunch zone
        # Also need to widen session window to include lunch time
        result = apply_all_filters(
            **params,
            allowed_windows=[(dt_time(8, 0), dt_time(16, 0))],
        )
        assert result.passed is False
        failed_names = [v.name for v in result.verdicts if not v.passed]
        assert "Lunch Filter" in failed_names

    def test_premarket_rejects_long(self):
        params = self._good_params()
        params["trigger_price"] = 2330.0  # below PM high of 2345
        result = apply_all_filters(**params)
        assert result.passed is False
        failed_names = [v.name for v in result.verdicts if not v.passed]
        assert "Pre-Market Range" in failed_names

    def test_multi_tf_rejects_counter_trend(self):
        params = self._good_params()
        params["bars_htf"] = _make_htf_bars(50, trend="down")  # LONG vs DOWN
        result = apply_all_filters(**params)
        assert result.passed is False
        failed_names = [v.name for v in result.verdicts if not v.passed]
        assert "Multi-TF Bias" in failed_names

    def test_vwap_rejects(self):
        params = self._good_params()
        params["vwap"] = 2360.0  # LONG trigger at 2350 < VWAP 2360
        result = apply_all_filters(**params)
        assert result.passed is False
        failed_names = [v.name for v in result.verdicts if not v.passed]
        assert "VWAP Confluence" in failed_names

    def test_nr7_never_rejects(self):
        """NR7 is a soft filter — it should never cause a rejection."""
        params = self._good_params()
        # Force NR7 to NOT fire (today is widest range)
        params["bars_daily"] = _make_daily_bars([2.0, 3.0, 2.5, 4.0, 3.0, 2.8, 10.0])
        result = apply_all_filters(**params)
        # NR7 should be in verdicts and passed
        nr7_verdicts = [v for v in result.verdicts if v.name == "NR7"]
        assert len(nr7_verdicts) == 1
        assert nr7_verdicts[0].passed is True

    def test_nr7_boost_included(self):
        """When NR7 fires, quality_boost should include the NR7 boost."""
        params = self._good_params()
        result = apply_all_filters(**params)
        # NR7 should fire (day 7 range = 2.0, narrowest)
        assert result.quality_boost > 0

    def test_majority_gate_mode(self):
        """In 'majority' mode, failing one hard filter should still pass."""
        params = self._good_params()
        params["vwap"] = 2360.0  # Only VWAP will fail
        result = apply_all_filters(**params, gate_mode="majority")
        # 4 hard filters pass, 1 fails → majority passes
        assert result.passed is True

    def test_majority_mode_many_failures(self):
        """In 'majority' mode, failing most hard filters should reject."""
        params = self._good_params()
        params["trigger_price"] = 2330.0  # fails PM range + VWAP
        params["bars_htf"] = _make_htf_bars(50, trend="down")  # fails multi-TF
        params["signal_time"] = _et_time(12, 0)  # fails lunch + session window
        result = apply_all_filters(
            **params,
            gate_mode="majority",
            allowed_windows=[(dt_time(8, 0), dt_time(16, 0))],  # widen so only lunch fails
        )
        assert result.passed is False

    def test_disable_session_window(self):
        """Disabling session window should allow after-hours signals."""
        params = self._good_params()
        params["signal_time"] = _et_time(14, 0)
        result = apply_all_filters(
            **params,
            enable_session_window=False,
            enable_lunch_filter=False,
        )
        assert result.passed is True

    def test_disable_all_hard_filters(self):
        """With all hard filters disabled, should always pass."""
        result = apply_all_filters(
            direction="LONG",
            trigger_price=0.0,
            signal_time=_et_time(3, 0),
            enable_session_window=False,
            enable_lunch_filter=False,
            enable_premarket=False,
            enable_multi_tf=False,
            enable_vwap=False,
            enable_nr7=False,
        )
        assert result.passed is True
        assert result.filters_total == 0

    def test_only_nr7_enabled(self):
        """Only soft filter enabled → always passes."""
        result = apply_all_filters(
            direction="LONG",
            trigger_price=2350.0,
            signal_time=_et_time(9, 30),
            bars_daily=_make_daily_bars([5, 6, 4.5, 7, 3.8, 4.2, 2.0]),
            enable_session_window=False,
            enable_lunch_filter=False,
            enable_premarket=False,
            enable_multi_tf=False,
            enable_vwap=False,
            enable_nr7=True,
        )
        assert result.passed is True
        assert result.filters_total == 1
        assert result.quality_boost > 0  # NR7 fires

    def test_short_direction(self):
        """Test filter composition with SHORT direction."""
        result = apply_all_filters(
            direction="SHORT",
            trigger_price=2325.0,
            signal_time=_et_time(9, 30),
            bars_daily=_make_daily_bars([5, 6, 4.5, 7, 3.8, 4.2, 2.0]),
            bars_htf=_make_htf_bars(50, trend="down"),
            premarket_low=2330.0,
            vwap=2340.0,
            enable_session_window=True,
            enable_lunch_filter=True,
            enable_premarket=True,
            enable_multi_tf=True,
            enable_vwap=True,
        )
        assert result.passed is True

    def test_quality_boost_aggregation(self):
        """Quality boost should be the sum of all individual boosts."""
        params = self._good_params()
        result = apply_all_filters(**params)
        individual_boosts = sum(v.score_boost for v in result.verdicts)
        assert abs(result.quality_boost - individual_boosts) < 1e-6

    def test_filters_passed_count(self):
        params = self._good_params()
        result = apply_all_filters(**params)
        passed_count = sum(1 for v in result.verdicts if v.passed)
        assert result.filters_passed == passed_count

    def test_summary_string_format(self):
        params = self._good_params()
        result = apply_all_filters(**params)
        assert "PASS" in result.summary or "REJECT" in result.summary

    def test_custom_ema_period(self):
        """Custom EMA period should be passed through to the multi-TF filter."""
        params = self._good_params()
        params["bars_htf"] = _make_htf_bars(20, trend="up")
        result = apply_all_filters(**params, ema_period=9)
        # Should pass since we have enough bars for EMA(9)
        multi_tf = [v for v in result.verdicts if v.name == "Multi-TF Bias"]
        assert len(multi_tf) == 1
        assert multi_tf[0].passed is True


# ============================================================================
# 10. FilterVerdict and ORBFilterResult dataclasses
# ============================================================================


class TestDataclasses:
    def test_filter_verdict_str_pass(self):
        v = FilterVerdict(name="Test", passed=True, reason="All good")
        s = str(v)
        assert "✅" in s
        assert "Test" in s

    def test_filter_verdict_str_fail(self):
        v = FilterVerdict(name="Test", passed=False, reason="Bad thing")
        s = str(v)
        assert "❌" in s

    def test_orb_filter_result_defaults(self):
        r = ORBFilterResult()
        assert r.passed is False
        assert r.verdicts == []
        assert r.quality_boost == 0.0

    def test_orb_filter_result_to_dict_roundtrip(self):
        v = FilterVerdict(name="NR7", passed=True, reason="Active", score_boost=0.2)
        r = ORBFilterResult(
            passed=True,
            verdicts=[v],
            filters_passed=1,
            filters_total=1,
            quality_boost=0.2,
            summary="PASS",
        )
        d = r.to_dict()
        assert d["passed"] is True
        assert len(d["verdicts"]) == 1
        assert d["verdicts"][0]["name"] == "NR7"
        assert d["quality_boost"] == 0.2
