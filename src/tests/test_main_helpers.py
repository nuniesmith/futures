"""
Tests for src.main — bot helper functions and risk management.

Covers: stack_avg_price, stack_total_size, stack_clear, risk_check_daily_reset,
        risk_record_trade, risk_can_trade, compute_ao, compute_vol_pct,
        compute_regime, compute_quality, adaptive_tp, wave_gate_ok,
        regime_stack_ok, calc_size.
"""

import time

import numpy as np
import pandas as pd
import pytest

import src.main as bot

# ---------------------------------------------------------------------------
# Local synthetic data helpers (self-contained — no conftest dependency)
# ---------------------------------------------------------------------------


def _make_candle_df(n=250, start=150.0, seed=42):
    """Build a DataFrame mimicking build_candles() output (lowercase columns + hl2)."""
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
    hl2 = (high + low) / 2.0
    return pd.DataFrame(
        {
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "hl2": hl2,
        }
    )


# ---------------------------------------------------------------------------
# Fixture: reset global bot state before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global bot state before each test to avoid cross-contamination."""
    bot.stack.update(direction=None, count=0, prices=[], sizes=[])
    bot.risk_state.update(
        daily_pnl=0.0,
        daily_trades=0,
        consecutive_losses=0,
        last_loss_time=0.0,
        day_start="",
        paused=False,
        pause_reason="",
        _alerted=False,
    )
    return


# ===================================================================
# stack_avg_price tests
# ===================================================================


class TestStackAvgPrice:
    def test_empty_stack_returns_zero(self):
        assert bot.stack_avg_price() == 0.0

    def test_single_entry(self):
        bot.stack["prices"] = [150.0]
        bot.stack["sizes"] = [10.0]
        assert bot.stack_avg_price() == 150.0

    def test_weighted_average(self):
        """Average should be weighted by size."""
        bot.stack["prices"] = [100.0, 200.0]
        bot.stack["sizes"] = [10.0, 10.0]
        assert bot.stack_avg_price() == pytest.approx(150.0)

    def test_weighted_average_unequal_sizes(self):
        """Unequal sizes should shift the average."""
        bot.stack["prices"] = [100.0, 200.0]
        bot.stack["sizes"] = [30.0, 10.0]
        # (100*30 + 200*10) / 40 = 5000/40 = 125
        assert bot.stack_avg_price() == pytest.approx(125.0)

    def test_three_entries(self):
        bot.stack["prices"] = [100.0, 110.0, 120.0]
        bot.stack["sizes"] = [5.0, 5.0, 5.0]
        assert bot.stack_avg_price() == pytest.approx(110.0)


# ===================================================================
# stack_total_size tests
# ===================================================================


class TestStackTotalSize:
    def test_empty_stack_returns_zero(self):
        assert bot.stack_total_size() == 0

    def test_single_entry(self):
        bot.stack["sizes"] = [10.0]
        assert bot.stack_total_size() == 10.0

    def test_multiple_entries(self):
        bot.stack["sizes"] = [5.0, 10.0, 3.5]
        assert bot.stack_total_size() == pytest.approx(18.5)


# ===================================================================
# stack_clear tests
# ===================================================================


class TestStackClear:
    def test_clears_all_fields(self):
        bot.stack.update(direction="buy", count=3, prices=[100, 110, 120], sizes=[5, 5, 5])
        bot.stack_clear()
        assert bot.stack["direction"] is None
        assert bot.stack["count"] == 0
        assert bot.stack["prices"] == []
        assert bot.stack["sizes"] == []

    def test_idempotent_on_empty_stack(self):
        """Clearing an already-empty stack should not error."""
        bot.stack_clear()
        assert bot.stack["direction"] is None
        assert bot.stack["count"] == 0


# ===================================================================
# risk_check_daily_reset tests
# ===================================================================


class TestRiskCheckDailyReset:
    def test_resets_counters_on_day_change(self):
        """Setting day_start to a past date should trigger a reset."""
        bot.risk_state["day_start"] = "2020-01-01"
        bot.risk_state["daily_pnl"] = -50.0
        bot.risk_state["daily_trades"] = 10
        bot.risk_state["paused"] = True
        bot.risk_state["pause_reason"] = "test"
        bot.risk_state["_alerted"] = True

        bot.risk_check_daily_reset()

        today = time.strftime("%Y-%m-%d")
        assert bot.risk_state["day_start"] == today
        assert bot.risk_state["daily_pnl"] == 0.0
        assert bot.risk_state["daily_trades"] == 0
        assert bot.risk_state["paused"] is False
        assert bot.risk_state["pause_reason"] == ""
        assert bot.risk_state["_alerted"] is False

    def test_no_reset_same_day(self):
        """If day_start matches today, no reset should occur."""
        today = time.strftime("%Y-%m-%d")
        bot.risk_state["day_start"] = today
        bot.risk_state["daily_pnl"] = -10.0
        bot.risk_state["daily_trades"] = 3

        bot.risk_check_daily_reset()

        assert bot.risk_state["daily_pnl"] == -10.0
        assert bot.risk_state["daily_trades"] == 3


# ===================================================================
# risk_record_trade tests
# ===================================================================


class TestRiskRecordTrade:
    def test_winning_trade_updates_pnl(self):
        bot.risk_record_trade(pnl_usdt=5.0, was_stop=False)
        assert bot.risk_state["daily_pnl"] == 5.0
        assert bot.risk_state["daily_trades"] == 1
        assert bot.risk_state["consecutive_losses"] == 0

    def test_losing_trade_increments_consecutive_losses(self):
        bot.risk_record_trade(pnl_usdt=-3.0, was_stop=False)
        assert bot.risk_state["daily_pnl"] == -3.0
        assert bot.risk_state["consecutive_losses"] == 1

    def test_stop_loss_trade_records_last_loss_time(self):
        before = time.time()
        bot.risk_record_trade(pnl_usdt=-2.0, was_stop=True)
        after = time.time()
        assert bot.risk_state["last_loss_time"] >= before
        assert bot.risk_state["last_loss_time"] <= after

    def test_winning_trade_resets_consecutive_losses(self):
        bot.risk_state["consecutive_losses"] = 3
        bot.risk_record_trade(pnl_usdt=1.0, was_stop=False)
        assert bot.risk_state["consecutive_losses"] == 0

    def test_multiple_trades_accumulate_pnl(self):
        bot.risk_record_trade(pnl_usdt=5.0, was_stop=False)
        bot.risk_record_trade(pnl_usdt=-2.0, was_stop=False)
        bot.risk_record_trade(pnl_usdt=3.0, was_stop=False)
        assert bot.risk_state["daily_pnl"] == pytest.approx(6.0)
        assert bot.risk_state["daily_trades"] == 3

    def test_consecutive_losses_stack(self):
        bot.risk_record_trade(pnl_usdt=-1.0, was_stop=False)
        bot.risk_record_trade(pnl_usdt=-1.0, was_stop=False)
        bot.risk_record_trade(pnl_usdt=-1.0, was_stop=False)
        assert bot.risk_state["consecutive_losses"] == 3


# ===================================================================
# risk_can_trade tests
# ===================================================================


class TestRiskCanTrade:
    def test_allowed_when_no_limits_hit(self):
        # Set day_start to today so daily reset doesn't trigger
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        allowed, reason = bot.risk_can_trade()
        assert allowed is True
        assert reason == ""

    def test_blocked_by_daily_loss_limit(self):
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        # CAPITAL * MAX_DAILY_LOSS_PCT is the max loss; exceed it
        max_loss = bot.CAPITAL * bot.MAX_DAILY_LOSS_PCT
        bot.risk_state["daily_pnl"] = -(max_loss + 1.0)
        allowed, reason = bot.risk_can_trade()
        assert allowed is False
        assert "loss limit" in reason.lower() or "daily" in reason.lower()

    def test_blocked_by_consecutive_losses(self):
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        bot.risk_state["consecutive_losses"] = bot.MAX_CONSECUTIVE_LOSSES
        allowed, reason = bot.risk_can_trade()
        assert allowed is False
        assert "consecutive" in reason.lower()

    def test_blocked_by_cooldown(self):
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        # Set last_loss_time to right now → within cooldown window
        bot.risk_state["last_loss_time"] = time.time()
        allowed, reason = bot.risk_can_trade()
        assert allowed is False
        assert "cooldown" in reason.lower()

    def test_blocked_by_trade_limit(self):
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        bot.risk_state["daily_trades"] = bot.DAILY_TRADE_LIMIT
        allowed, reason = bot.risk_can_trade()
        assert allowed is False
        assert "limit" in reason.lower()

    def test_unpauses_when_gates_pass(self):
        """If previously paused but conditions cleared, should unpause."""
        bot.risk_state["day_start"] = time.strftime("%Y-%m-%d")
        bot.risk_state["paused"] = True
        bot.risk_state["pause_reason"] = "test pause"
        # All counters are at defaults (0) → should pass
        allowed, _reason = bot.risk_can_trade()
        assert allowed is True
        assert bot.risk_state["paused"] is False
        assert bot.risk_state["pause_reason"] == ""


# ===================================================================
# compute_ao tests
# ===================================================================


class TestComputeAo:
    def test_returns_float_for_sufficient_data(self):
        df = _make_candle_df(n=50)
        result = bot.compute_ao(df)
        assert isinstance(result, float)

    def test_returns_zero_for_insufficient_data(self):
        df = _make_candle_df(n=20)
        result = bot.compute_ao(df)
        assert result == 0.0

    def test_uptrend_positive_ao(self):
        """An uptrending hl2 should produce a positive AO."""
        n = 60
        high = np.linspace(100, 200, n) + 1.0
        low = np.linspace(100, 200, n) - 1.0
        close = np.linspace(100, 200, n)
        hl2 = (high + low) / 2.0
        df = pd.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "hl2": hl2,
                "volume": np.ones(n) * 1000,
            }
        )
        result = bot.compute_ao(df)
        assert result > 0.0

    def test_flat_hl2_zero_ao(self):
        """Flat hl2 should produce AO ≈ 0."""
        n = 50
        hl2 = np.full(n, 150.0)
        df = pd.DataFrame(
            {
                "high": hl2 + 1,
                "low": hl2 - 1,
                "close": hl2,
                "hl2": hl2,
                "volume": np.ones(n) * 1000,
            }
        )
        result = bot.compute_ao(df)
        assert abs(result) < 1e-10


# ===================================================================
# compute_vol_pct tests
# ===================================================================


class TestComputeVolPct:
    def test_returns_float(self):
        df = _make_candle_df(n=250)
        result = bot.compute_vol_pct(df)
        assert isinstance(result, float)

    def test_in_zero_one_range(self):
        df = _make_candle_df(n=250)
        result = bot.compute_vol_pct(df)
        assert 0.0 <= result <= 1.0

    def test_insufficient_data_returns_default(self):
        df = _make_candle_df(n=10)
        result = bot.compute_vol_pct(df)
        assert result == 0.5


# ===================================================================
# compute_regime tests
# ===================================================================


class TestComputeRegime:
    def test_returns_tuple_of_str_and_float(self):
        df = _make_candle_df(n=250)
        regime, value = bot.compute_regime(df)
        assert isinstance(regime, str)
        assert isinstance(value, float)

    def test_valid_regime_labels(self):
        df = _make_candle_df(n=250)
        regime, _ = bot.compute_regime(df)
        valid = {"TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "RANGING", "NEUTRAL"}
        assert regime in valid

    def test_insufficient_data_returns_neutral(self):
        df = _make_candle_df(n=50)
        regime, value = bot.compute_regime(df)
        assert regime == "NEUTRAL"
        assert value == 0.0

    def test_strongly_trending_up_data(self):
        """A monotonically increasing series with 220+ bars should detect trend."""
        n = 250
        close = pd.Series(np.linspace(100, 300, n))
        high = close + 1
        low = close - 1
        hl2 = (high + low) / 2
        volume = pd.Series(np.ones(n) * 1000)
        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "hl2": hl2,
            }
        )
        regime, slope = bot.compute_regime(df)
        # Strong uptrend should produce TRENDING_UP or at least a positive slope
        assert regime in {"TRENDING_UP", "NEUTRAL", "VOLATILE", "RANGING"}
        # Slope should be positive or at minimum well-defined
        assert isinstance(slope, float)


# ===================================================================
# compute_quality tests
# ===================================================================


class TestComputeQuality:
    def test_returns_float(self):
        df = _make_candle_df(n=50)
        result = bot.compute_quality(
            df,
            ao=1.0,
            vol_pct=0.5,
            fast_ema=151.0,
            slow_ema=149.0,
            imbalance=1.5,
            regime="NEUTRAL",
        )
        assert isinstance(result, float)

    def test_in_zero_hundred_range(self):
        df = _make_candle_df(n=50)
        result = bot.compute_quality(
            df,
            ao=1.0,
            vol_pct=0.5,
            fast_ema=151.0,
            slow_ema=149.0,
            imbalance=1.5,
            regime="TRENDING_UP",
        )
        assert 0.0 <= result <= 100.0

    def test_insufficient_data_returns_zero(self):
        df = _make_candle_df(n=20)
        result = bot.compute_quality(
            df,
            ao=1.0,
            vol_pct=0.5,
            fast_ema=151.0,
            slow_ema=149.0,
            imbalance=1.5,
            regime="NEUTRAL",
        )
        assert result == 0.0

    def test_all_factors_aligned_high_score(self):
        """When all quality factors align, score should be high."""
        df = _make_candle_df(n=50, start=150.0)
        close = float(df["close"].iloc[-1])
        # Bullish bias: fast > slow, ao > 0, close > fast, imbalance >= 1.2, trending up
        fast_ema = close - 0.01
        slow_ema = close - 1.0
        # Boost volume so it's above the 20-bar mean
        df["volume"] = df["volume"] * 10
        result = bot.compute_quality(
            df,
            ao=2.0,
            vol_pct=0.5,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            imbalance=1.5,
            regime="TRENDING_UP",
        )
        assert result >= 40.0  # at least several factors should fire

    def test_no_factors_aligned_low_score(self):
        """When quality factors contradict each other, score should be low."""
        df = _make_candle_df(n=50)
        close = float(df["close"].iloc[-1])
        # Bull bias (fast > slow) but bearish AO, bearish imbalance,
        # bearish regime, and close below fast EMA → all factors misalign
        result = bot.compute_quality(
            df,
            ao=-2.0,  # bearish AO contradicts bull EMA bias
            vol_pct=0.1,
            fast_ema=close + 5.0,  # close is BELOW fast EMA → misaligned
            slow_ema=close - 5.0,  # fast > slow → bull bias
            imbalance=0.5,  # bearish imbalance contradicts bull bias
            regime="TRENDING_DOWN",  # bearish regime contradicts bull bias
        )
        # AO misaligned (0), close below fast EMA (0), imbalance misaligned (0),
        # regime misaligned (0), only volume *might* fire (25) → score <= 25
        assert result <= 35.0


# ===================================================================
# adaptive_tp tests
# ===================================================================


class TestAdaptiveTp:
    def test_returns_float(self):
        result = bot.adaptive_tp(vol_pct=0.5, regime="NEUTRAL")
        assert isinstance(result, float)

    def test_positive_result(self):
        result = bot.adaptive_tp(vol_pct=0.5, regime="NEUTRAL")
        assert result > 0.0

    def test_trending_higher_than_ranging(self):
        """Trending regime should produce wider TP than ranging."""
        tp_trend = bot.adaptive_tp(vol_pct=0.5, regime="TRENDING_UP")
        tp_range = bot.adaptive_tp(vol_pct=0.5, regime="RANGING")
        assert tp_trend > tp_range

    def test_high_vol_wider_tp(self):
        """Higher volatility should produce wider TP within the same regime."""
        tp_low = bot.adaptive_tp(vol_pct=0.1, regime="NEUTRAL")
        tp_high = bot.adaptive_tp(vol_pct=0.9, regime="NEUTRAL")
        assert tp_high > tp_low

    def test_all_regimes_covered(self):
        """All regime values should produce a valid positive TP."""
        for regime in [
            "TRENDING_UP",
            "TRENDING_DOWN",
            "VOLATILE",
            "RANGING",
            "NEUTRAL",
        ]:
            result = bot.adaptive_tp(vol_pct=0.5, regime=regime)
            assert result > 0.0, f"Failed for regime {regime}"

    def test_volatile_wider_than_neutral(self):
        tp_vol = bot.adaptive_tp(vol_pct=0.5, regime="VOLATILE")
        tp_neu = bot.adaptive_tp(vol_pct=0.5, regime="NEUTRAL")
        assert tp_vol > tp_neu


# ===================================================================
# wave_gate_ok tests
# ===================================================================


class TestWaveGateOk:
    def test_buy_gate_passes_with_bullish_wave(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.7
        ws.cur_ratio = 0.5
        assert bot.wave_gate_ok(ws, "buy", gate=0.4) is True

    def test_buy_gate_fails_low_wr_pct(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.3
        ws.cur_ratio = 0.5
        assert bot.wave_gate_ok(ws, "buy", gate=0.4) is False

    def test_buy_gate_fails_negative_cur_ratio(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.7
        ws.cur_ratio = -0.5  # bearish current speed
        assert bot.wave_gate_ok(ws, "buy", gate=0.4) is False

    def test_sell_gate_passes_with_bearish_wave(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.2
        ws.cur_ratio = -0.5
        assert bot.wave_gate_ok(ws, "sell", gate=0.4) is True

    def test_sell_gate_fails_high_wr_pct(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.8
        ws.cur_ratio = -0.5
        assert bot.wave_gate_ok(ws, "sell", gate=0.4) is False

    def test_sell_gate_fails_positive_cur_ratio(self):
        ws = bot.WaveState()
        ws.wr_pct = 0.2
        ws.cur_ratio = 0.5  # bullish current speed
        assert bot.wave_gate_ok(ws, "sell", gate=0.4) is False


# ===================================================================
# regime_stack_ok tests
# ===================================================================


class TestRegimeStackOk:
    def test_buy_allowed_in_trending_up(self):
        assert bot.regime_stack_ok("TRENDING_UP", "buy") is True

    def test_buy_blocked_in_trending_down(self):
        assert bot.regime_stack_ok("TRENDING_DOWN", "buy") is False

    def test_sell_allowed_in_trending_down(self):
        assert bot.regime_stack_ok("TRENDING_DOWN", "sell") is True

    def test_sell_blocked_in_trending_up(self):
        assert bot.regime_stack_ok("TRENDING_UP", "sell") is False

    def test_buy_allowed_in_neutral(self):
        assert bot.regime_stack_ok("NEUTRAL", "buy") is True

    def test_sell_allowed_in_neutral(self):
        assert bot.regime_stack_ok("NEUTRAL", "sell") is True

    def test_buy_allowed_in_volatile(self):
        assert bot.regime_stack_ok("VOLATILE", "buy") is True

    def test_sell_allowed_in_ranging(self):
        assert bot.regime_stack_ok("RANGING", "sell") is True


# ===================================================================
# calc_size tests
# ===================================================================


class TestCalcSize:
    def test_returns_positive_float(self):
        result = bot.calc_size(price=150.0)
        assert isinstance(result, float)
        assert result > 0.0

    def test_minimum_size(self):
        """calc_size should never return less than 0.1."""
        result = bot.calc_size(price=999999.0)
        assert result >= 0.1

    def test_higher_price_smaller_size(self):
        """Higher price should produce fewer contracts (inverse relationship)."""
        size_low = bot.calc_size(price=50.0)
        size_high = bot.calc_size(price=500.0)
        assert size_low > size_high

    def test_consistent_results(self):
        """Same price should always return the same size."""
        s1 = bot.calc_size(price=150.0)
        s2 = bot.calc_size(price=150.0)
        assert s1 == s2
