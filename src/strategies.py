"""
Strategy module for futures backtesting.

Provides seven strategy classes with trend filters and ATR-based risk management:
  1. TrendEMACross    — EMA crossover filtered by a longer trend EMA, ATR SL/TP
  2. RSIReversal      — RSI mean-reversion entries with trend filter, ATR SL/TP
  3. BreakoutStrategy  — Breakout of recent high/low with volume filter, ATR SL/TP
  4. VWAPReversion    — Mean-reversion around daily VWAP with trend filter
  5. ORBStrategy      — Opening Range Breakout of first N bars each session
  6. MACDMomentum     — MACD crossover with histogram acceleration filter

All strategies are compatible with the `backtesting.py` library and expose
class-level parameters that Optuna can tune.

The VolumeProfileStrategy is imported from volume_profile.py and registered
here for unified optimizer access.
"""

import math

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

# ---------------------------------------------------------------------------
# Indicator helper functions (compatible with backtesting.py's self.I())
# ---------------------------------------------------------------------------


def _passthrough(arr):
    """Identity function for pre-computed indicator arrays."""
    return arr


def _ema(series, length: int):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=length, adjust=False).mean()


def _sma(series, length: int):
    """Simple Moving Average."""
    return pd.Series(series).rolling(length).mean()


def _atr(high, low, close, length: int = 14):
    """Average True Range."""
    h, lo, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = h - lo
    tr2 = (h - c.shift(1)).abs()
    tr3 = (lo - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def _rsi(series, length: int = 14):
    """Relative Strength Index."""
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=length, adjust=False).mean()
    avg_loss = loss.ewm(span=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    return 100 - (100 / (1 + rs))


def _rolling_max(series, length: int):
    """Rolling maximum (for breakout high detection)."""
    return pd.Series(series).rolling(length).max()


def _rolling_min(series, length: int):
    """Rolling minimum (for breakout low detection)."""
    return pd.Series(series).rolling(length).min()


def _macd_line(series, fast: int = 12, slow: int = 26):
    """MACD line (fast EMA - slow EMA)."""
    s = pd.Series(series)
    return s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()


def _macd_signal(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD signal line."""
    macd = _macd_line(series, fast, slow)
    return macd.ewm(span=signal, adjust=False).mean()


def _macd_histogram(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD histogram (MACD line - signal line)."""
    macd = _macd_line(series, fast, slow)
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


# ---------------------------------------------------------------------------
# Strategy 1 — Trend-Filtered EMA Cross
# ---------------------------------------------------------------------------


class TrendEMACross(Strategy):
    """EMA crossover with a longer-term trend filter and ATR-based SL/TP.

    Only takes LONG trades when price is above the trend EMA, and SHORT
    trades when price is below it.  Exits on the opposite crossover OR
    when the ATR stop-loss / take-profit is hit — whichever comes first.

    Optimisable parameters
    ----------------------
    n1           : int    fast EMA period           (5 – 20)
    n2           : int    slow EMA period            (15 – 50)
    trend_period : int    trend-direction EMA period (40 – 120)
    atr_period   : int    ATR look-back              (10 – 20)
    atr_sl_mult  : float  SL distance = ATR × this   (1.0 – 3.0)
    atr_tp_mult  : float  TP distance = ATR × this   (1.5 – 5.0)
    trade_size   : float  fraction of equity per trade (0.05 – 0.30)
    """

    # Defaults (will be overridden by the optimizer)
    n1: int = 9
    n2: int = 21
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(_ema, close, self.n1, name=f"EMA{self.n1}")
        self.ema_slow = self.I(_ema, close, self.n2, name=f"EMA{self.n2}")
        self.ema_trend = self.I(
            _ema, close, self.trend_period, name=f"Trend{self.trend_period}"
        )
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        if _is_nan(trend):
            return

        # --- Exit logic (signal-based; SL/TP handled by broker) ---
        if self.position:
            if self.position.is_long and crossover(
                list(self.ema_slow), list(self.ema_fast)
            ):
                self.position.close()
            elif self.position.is_short and crossover(
                list(self.ema_fast), list(self.ema_slow)
            ):
                self.position.close()
            return  # no new entries while in a position

        # --- Entry logic ---
        if crossover(list(self.ema_fast), list(self.ema_slow)) and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossover(list(self.ema_slow), list(self.ema_fast)) and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 2 — RSI Mean-Reversion
# ---------------------------------------------------------------------------


class RSIReversal(Strategy):
    """RSI mean-reversion entries with trend filter and ATR-based SL/TP.

    Enters LONG when RSI crosses up from oversold (and price > trend EMA).
    Enters SHORT when RSI crosses down from overbought (and price < trend EMA).
    Exits when RSI reaches the opposite extreme, or SL/TP is hit.

    Optimisable parameters
    ----------------------
    rsi_period     : int    RSI look-back               (7 – 21)
    rsi_oversold   : int    oversold threshold           (20 – 40)
    rsi_overbought : int    overbought threshold         (60 – 80)
    trend_period   : int    trend-direction EMA period   (40 – 120)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.0
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.rsi = self.I(_rsi, close, self.rsi_period, name=f"RSI{self.rsi_period}")
        self.ema_trend = self.I(
            _ema, close, self.trend_period, name=f"Trend{self.trend_period}"
        )
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Need at least 2 RSI values for crossover detection
        if len(self.rsi) < 2 or _is_nan(self.rsi[-1]) or _is_nan(self.rsi[-2]):
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        if _is_nan(trend):
            return

        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]

        # --- Exit: RSI reaches opposite extreme ---
        if self.position:
            if self.position.is_long and rsi_now >= self.rsi_overbought:
                self.position.close()
            elif self.position.is_short and rsi_now <= self.rsi_oversold:
                self.position.close()
            return

        # --- Entry ---
        # Long: RSI crosses UP through oversold threshold, price above trend
        if (
            rsi_prev <= self.rsi_oversold
            and rsi_now > self.rsi_oversold
            and price > trend
        ):
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Short: RSI crosses DOWN through overbought threshold, price below trend
        elif (
            rsi_prev >= self.rsi_overbought
            and rsi_now < self.rsi_overbought
            and price < trend
        ):
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 3 — Breakout
# ---------------------------------------------------------------------------


class BreakoutStrategy(Strategy):
    """Price breakout of recent high/low with volume filter and ATR SL/TP.

    Enters LONG when Close breaks above the rolling highest-high of the
    prior `lookback` bars AND volume exceeds its moving average × vol_mult.
    Mirror logic for shorts.  Exits purely on SL/TP — no signal-based exit.

    Optimisable parameters
    ----------------------
    lookback       : int    high/low look-back bars      (10 – 50)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (2.0 – 6.0)
    vol_sma_period : int    volume SMA look-back          (10 – 30)
    vol_mult       : float  volume filter multiplier      (1.0 – 2.0)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    lookback: int = 20
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0
    vol_sma_period: int = 20
    vol_mult: float = 1.2
    trade_size: float = 0.10

    def init(self):
        self.highest = self.I(_rolling_max, self.data.High, self.lookback, name="HH")
        self.lowest = self.I(_rolling_min, self.data.Low, self.lookback, name="LL")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(
            _sma, self.data.Volume, self.vol_sma_period, name="VolSMA"
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Need prior-bar rolling extremes (avoid look-ahead)
        if len(self.highest) < 2 or _is_nan(self.highest[-2]):
            return
        if len(self.lowest) < 2 or _is_nan(self.lowest[-2]):
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]

        # Volume gate — skip if below-average volume
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # Let SL/TP manage exits — no signal-based close
        if self.position:
            return

        prior_high = self.highest[-2]  # rolling max as of the previous bar
        prior_low = self.lowest[-2]

        # Breakout long
        if price > prior_high:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Breakout short
        elif price < prior_low:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 4 — VWAP Mean-Reversion
# ---------------------------------------------------------------------------


class VWAPReversion(Strategy):
    """Mean-reversion around daily VWAP with trend filter and ATR SL/TP.

    Enters LONG when price crosses back above VWAP after being below it
    (pullback buy in an uptrend).  Enters SHORT when price crosses below
    VWAP after being above it (rally sell in a downtrend).

    Designed for intraday futures where VWAP acts as a magnet / fair-value
    anchor.  Requires a trend EMA to filter direction and volume confirmation.

    Optimisable parameters
    ----------------------
    trend_period   : int    trend-direction EMA period   (40 – 120)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (1.5 – 5.0)
    vol_sma_period : int    volume SMA look-back          (10 – 30)
    vol_mult       : float  volume filter multiplier      (0.8 – 1.5)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.0
    vol_sma_period: int = 20
    vol_mult: float = 1.0
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_trend = self.I(
            _ema, close, self.trend_period, name=f"Trend{self.trend_period}"
        )
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(
            _sma, self.data.Volume, self.vol_sma_period, name="VolSMA"
        )

        # Pre-compute daily-resetting VWAP
        df = self.data.df
        idx = df.index
        high = df["High"]
        low = df["Low"]
        close_s = df["Close"]
        volume = df["Volume"]
        typical = (high + low + close_s) / 3
        tpv = typical * volume

        try:
            dates = idx.to_series().dt.date
            cum_tpv = tpv.groupby(dates.values).cumsum()
            cum_vol = volume.groupby(dates.values).cumsum()
        except AttributeError:
            # Non-datetime index — running cumulative VWAP
            cum_tpv = tpv.cumsum()
            cum_vol = volume.cumsum()

        vwap_arr = (cum_tpv / (cum_vol + 1e-10)).values
        self.vwap = self.I(_passthrough, vwap_arr, name="VWAP")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        if len(self.vwap) < 2 or _is_nan(self.vwap[-1]) or _is_nan(self.vwap[-2]):
            return

        price = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        vwap_now = self.vwap[-1]
        vwap_prev = self.vwap[-2]
        trend = self.ema_trend[-1]

        if _is_nan(trend) or _is_nan(vwap_now):
            return

        # Volume filter
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # --- Exit: price reverts back through VWAP ---
        if self.position:
            if self.position.is_long and price < vwap_now:
                self.position.close()
            elif self.position.is_short and price > vwap_now:
                self.position.close()
            return

        # --- Entry: VWAP crossover with trend filter ---
        crossed_above = prev_close <= vwap_prev and price > vwap_now
        crossed_below = prev_close >= vwap_prev and price < vwap_now

        if crossed_above and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossed_below and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 5 — Opening Range Breakout
# ---------------------------------------------------------------------------


class ORBStrategy(Strategy):
    """Opening Range Breakout — classic morning intraday strategy.

    Computes the high/low of the first `orb_bars` bars each session day.
    After the opening range is established:
      - LONG when price breaks above the opening range high
      - SHORT when price breaks below the opening range low

    Ideal for the first 2-3 hours of futures trading where morning
    momentum drives directional moves off the opening range.

    Optimisable parameters
    ----------------------
    orb_bars       : int    bars forming the opening range (3 – 12)
    atr_period     : int    ATR look-back                  (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this       (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this       (1.5 – 5.0)
    vol_sma_period : int    volume SMA look-back            (10 – 30)
    vol_mult       : float  volume filter multiplier        (0.8 – 1.5)
    trade_size     : float  fraction of equity per trade    (0.05 – 0.30)
    """

    orb_bars: int = 6  # 6 × 5min = 30 minute opening range
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    vol_sma_period: int = 20
    vol_mult: float = 1.0
    trade_size: float = 0.10

    def init(self):
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(
            _sma, self.data.Volume, self.vol_sma_period, name="VolSMA"
        )

        # Pre-compute ORB levels per session day
        df = self.data.df
        idx = df.index
        h = df["High"]
        low_s = df["Low"]
        orb_bars = int(self.orb_bars)

        orb_h = np.full(len(idx), np.nan)
        orb_l = np.full(len(idx), np.nan)

        try:
            dates = idx.to_series().dt.date.values
            unique_dates = sorted(set(dates))
            for date_val in unique_dates:
                day_positions = np.where(dates == date_val)[0]
                if len(day_positions) < orb_bars + 1:
                    continue
                range_positions = day_positions[:orb_bars]
                trade_positions = day_positions[orb_bars:]
                range_high = h.iloc[range_positions].max()
                range_low = low_s.iloc[range_positions].min()
                orb_h[trade_positions] = range_high
                orb_l[trade_positions] = range_low
        except AttributeError:
            # Non-datetime index — use rolling lookback as fallback
            orb_h = pd.Series(h).rolling(orb_bars).max().shift(1).values
            orb_l = pd.Series(low_s).rolling(orb_bars).min().shift(1).values

        self.orb_high = self.I(_passthrough, orb_h, name="ORB_H")
        self.orb_low = self.I(_passthrough, orb_l, name="ORB_L")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        orb_h = self.orb_high[-1]
        orb_l = self.orb_low[-1]
        if _is_nan(orb_h) or _is_nan(orb_l):
            return  # Still in opening range formation

        price = self.data.Close[-1]

        # Volume gate
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # SL/TP manage exits — no signal-based close
        if self.position:
            return

        # Breakout long
        if price > orb_h:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Breakout short
        elif price < orb_l:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 6 — MACD Momentum
# ---------------------------------------------------------------------------


class MACDMomentum(Strategy):
    """MACD crossover with histogram acceleration and trend filter.

    Enters LONG when MACD crosses above its signal line, histogram is
    accelerating (building momentum), and price is above the trend EMA.
    Mirror logic for shorts.  Exits on the opposite MACD crossover.

    Good for catching medium-momentum intraday moves where the initial
    impulse is confirmed by accelerating MACD histogram.

    Optimisable parameters
    ----------------------
    macd_fast      : int    fast EMA period for MACD      (8 – 16)
    macd_slow      : int    slow EMA period for MACD      (20 – 34)
    macd_signal    : int    signal line EMA period         (6 – 12)
    trend_period   : int    trend-direction EMA period     (40 – 120)
    atr_period     : int    ATR look-back                  (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this       (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this       (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade    (0.05 – 0.30)
    """

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.macd = self.I(
            _macd_line, close, self.macd_fast, self.macd_slow, name="MACD"
        )
        self.signal = self.I(
            _macd_signal,
            close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            name="Signal",
        )
        self.histogram = self.I(
            _macd_histogram,
            close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            name="Hist",
        )
        self.ema_trend = self.I(
            _ema, close, self.trend_period, name=f"Trend{self.trend_period}"
        )
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        if len(self.histogram) < 2:
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        hist_now = self.histogram[-1]
        hist_prev = self.histogram[-2]

        if _is_nan(trend) or _is_nan(hist_now) or _is_nan(hist_prev):
            return

        # Histogram acceleration: momentum is building, not fading
        hist_growing = hist_now > hist_prev

        # --- Exit: opposite MACD crossover ---
        if self.position:
            if self.position.is_long and crossover(list(self.signal), list(self.macd)):
                self.position.close()
            elif self.position.is_short and crossover(
                list(self.macd), list(self.signal)
            ):
                self.position.close()
            return

        # --- Entry: MACD crossover + histogram acceleration + trend filter ---
        if (
            crossover(list(self.macd), list(self.signal))
            and hist_growing
            and price > trend
        ):
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif (
            crossover(list(self.signal), list(self.macd))
            and not hist_growing
            and price < trend
        ):
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Legacy compatibility: plain EMA Cross (no filters, no stops)
# ---------------------------------------------------------------------------


class PlainEMACross(Strategy):
    """Original bare EMA crossover — kept for A/B comparison only."""

    n1: int = 9
    n2: int = 21
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema1 = self.I(_ema, close, self.n1, name=f"EMA{self.n1}")
        self.ema2 = self.I(_ema, close, self.n2, name=f"EMA{self.n2}")

    def next(self):
        if self.position:
            if self.position.is_long and crossover(list(self.ema2), list(self.ema1)):
                self.position.close()
            elif self.position.is_short and crossover(list(self.ema1), list(self.ema2)):
                self.position.close()
            return
        if crossover(list(self.ema1), list(self.ema2)):
            self.buy(size=self.trade_size)
        elif crossover(list(self.ema2), list(self.ema1)):
            self.sell(size=self.trade_size)


# ---------------------------------------------------------------------------
# Strategy registry — used by the optimizer / engine
# ---------------------------------------------------------------------------

# Import VolumeProfileStrategy from its dedicated module
_VP_AVAILABLE = False
_VolumeProfileStrategy = None
_suggest_volume_profile_params = None
try:
    from volume_profile import VolumeProfileStrategy as _VPS
    from volume_profile import suggest_volume_profile_params as _svpp

    _VolumeProfileStrategy = _VPS
    _suggest_volume_profile_params = _svpp
    _VP_AVAILABLE = True
except ImportError:
    pass

STRATEGY_CLASSES = {
    "TrendEMA": TrendEMACross,
    "RSI": RSIReversal,
    "Breakout": BreakoutStrategy,
    "VWAP": VWAPReversion,
    "ORB": ORBStrategy,
    "MACD": MACDMomentum,
    "PlainEMA": PlainEMACross,
}

if _VP_AVAILABLE and _VolumeProfileStrategy is not None:
    STRATEGY_CLASSES["VolumeProfile"] = _VolumeProfileStrategy

# Human-readable labels
STRATEGY_LABELS = {
    "TrendEMA": "Trend-Filtered EMA Cross",
    "RSI": "RSI Mean-Reversion",
    "Breakout": "Breakout + Volume",
    "VWAP": "VWAP Reversion",
    "ORB": "Opening Range Breakout",
    "MACD": "MACD Momentum",
    "VolumeProfile": "Volume Profile (POC/VA)",
    "PlainEMA": "Plain EMA Cross (legacy)",
}


def suggest_params(trial, strategy_key: str) -> dict:
    """Ask Optuna to suggest hyper-parameters for the given strategy.

    Returns a dict that can be unpacked into the Strategy's class attributes.
    """
    params: dict = {}

    if strategy_key == "VolumeProfile":
        if _VP_AVAILABLE and _suggest_volume_profile_params is not None:
            return _suggest_volume_profile_params(trial)
        # Fallback if volume_profile module not available
        return {"trade_size": trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)}

    elif strategy_key == "TrendEMA":
        params["n1"] = trial.suggest_int("n1", 5, 20)
        params["n2"] = trial.suggest_int("n2", max(params["n1"] + 5, 15), 55)
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "RSI":
        params["rsi_period"] = trial.suggest_int("rsi_period", 7, 21)
        params["rsi_oversold"] = trial.suggest_int("rsi_oversold", 20, 40)
        params["rsi_overbought"] = trial.suggest_int(
            "rsi_overbought",
            max(params["rsi_oversold"] + 20, 60),
            80,
        )
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "Breakout":
        params["lookback"] = trial.suggest_int("lookback", 10, 50)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 2.0, 6.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 1.0, 2.0, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "VWAP":
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 0.8, 1.5, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "ORB":
        params["orb_bars"] = trial.suggest_int("orb_bars", 3, 12)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 0.8, 1.5, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "MACD":
        params["macd_fast"] = trial.suggest_int("macd_fast", 8, 16)
        params["macd_slow"] = trial.suggest_int(
            "macd_slow", max(params["macd_fast"] + 8, 20), 34
        )
        params["macd_signal"] = trial.suggest_int("macd_signal", 6, 12)
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "PlainEMA":
        params["n1"] = trial.suggest_int("n1", 5, 20)
        params["n2"] = trial.suggest_int("n2", max(params["n1"] + 5, 15), 55)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    return params


def make_strategy(strategy_key: str, params: dict) -> type:
    """Create a new Strategy *subclass* with the given params baked in.

    We create a fresh subclass each time so that class-attribute mutations
    in one backtest don't leak into another.
    """
    base_cls = STRATEGY_CLASSES[strategy_key]

    # Build a new class dynamically with params as class attributes
    attrs = dict(params)  # copy
    new_cls = type(f"{base_cls.__name__}_Configured", (base_cls,), attrs)
    return new_cls


# ---------------------------------------------------------------------------
# Scoring helper — used by the optimizer
# ---------------------------------------------------------------------------

_PENALTY = -100.0  # returned for invalid / degenerate runs


def _safe_float(val, fallback: float = 0.0) -> float:
    """Extract a float from backtest stats, returning fallback for NaN/missing."""
    try:
        f = float(val)
        return fallback if math.isnan(f) else f
    except (TypeError, ValueError):
        return fallback


def score_backtest(stats, min_trades: int = 3) -> float:
    """Compute a risk-adjusted score from backtest stats.

    Designed for funded-account (TPT) trading where drawdown control is
    paramount and consistent win rates matter more than occasional big wins.

    Scoring components:
      - Base: Sharpe (40%) + Sortino (30%) + normalised Profit Factor (30%)
      - Drawdown penalty: progressive, severe above 6%
      - Win rate bonus: rewards consistency above 45%
      - Expectancy bonus: rewards positive per-trade edge
      - Trade count bonus: prefers statistically significant sample sizes
    """
    n_trades = int(stats["# Trades"])
    if n_trades < min_trades:
        return _PENALTY

    sharpe = float(stats["Sharpe Ratio"])
    if _is_nan(sharpe):
        return _PENALTY

    max_dd = abs(float(stats["Max. Drawdown [%]"]))
    wr = _safe_float(stats["Win Rate [%]"])
    pf = _safe_float(stats.get("Profit Factor", 0))
    sortino = _safe_float(stats.get("Sortino Ratio", sharpe))
    expectancy = _safe_float(stats.get("Expectancy [%]", 0))

    # Base: weighted combination of risk-adjusted metrics
    pf_norm = min(pf / 3.0, 1.0) * 2.0 if pf > 0 else 0.0
    score = 0.4 * sharpe + 0.3 * sortino + 0.3 * pf_norm

    # Drawdown penalty — progressive and severe for funded accounts
    if max_dd > 3:
        score -= (max_dd - 3) * 0.08
    if max_dd > 6:
        score -= (max_dd - 6) * 0.15
    if max_dd > 10:
        score -= (max_dd - 10) * 0.30

    # Win rate bonus (consistent winners)
    if wr > 45:
        score += (wr - 45) * 0.015
    if wr > 60:
        score += (wr - 60) * 0.01  # diminishing returns above 60%

    # Expectancy bonus (per-trade edge)
    if expectancy > 0:
        score += min(expectancy * 0.1, 0.5)

    # Trade count: prefer statistical significance
    if n_trades >= 8:
        score += 0.1
    if n_trades >= 15:
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _is_nan(x) -> bool:
    """Robust NaN check for floats and numpy scalars."""
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True
