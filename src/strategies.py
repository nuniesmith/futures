"""
Strategy module for futures backtesting.

Provides three strategy classes with trend filters and ATR-based risk management:
  1. TrendEMACross  — EMA crossover filtered by a longer trend EMA, ATR SL/TP
  2. RSIReversal    — RSI mean-reversion entries with trend filter, ATR SL/TP
  3. BreakoutStrategy — Breakout of recent high/low with volume filter, ATR SL/TP

All strategies are compatible with the `backtesting.py` library and expose
class-level parameters that Optuna can tune.
"""

import math

import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

# ---------------------------------------------------------------------------
# Indicator helper functions (compatible with backtesting.py's self.I())
# ---------------------------------------------------------------------------


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

STRATEGY_CLASSES = {
    "TrendEMA": TrendEMACross,
    "RSI": RSIReversal,
    "Breakout": BreakoutStrategy,
    "PlainEMA": PlainEMACross,
}

# Human-readable labels
STRATEGY_LABELS = {
    "TrendEMA": "Trend-Filtered EMA Cross",
    "RSI": "RSI Mean-Reversion",
    "Breakout": "Breakout + Volume",
    "PlainEMA": "Plain EMA Cross (legacy)",
}


def suggest_params(trial, strategy_key: str) -> dict:
    """Ask Optuna to suggest hyper-parameters for the given strategy.

    Returns a dict that can be unpacked into the Strategy's class attributes.
    """
    params: dict = {}

    if strategy_key == "TrendEMA":
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


def score_backtest(stats, min_trades: int = 3) -> float:
    """Compute a risk-adjusted score from backtest stats.

    Primary metric : Sharpe Ratio
    Penalties      : < min_trades, NaN Sharpe, drawdown > 8%
    """
    n_trades = int(stats["# Trades"])
    if n_trades < min_trades:
        return _PENALTY

    sharpe = float(stats["Sharpe Ratio"])
    if _is_nan(sharpe):
        return _PENALTY

    max_dd = abs(float(stats["Max. Drawdown [%]"]))

    # Start from Sharpe, penalise large drawdowns
    score = sharpe
    if max_dd > 8:
        score -= (max_dd - 8) * 0.15

    # Small bonus for win-rate above 40 %
    wr = float(stats["Win Rate [%]"])
    if not _is_nan(wr) and wr > 40:
        score += (wr - 40) * 0.01

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
