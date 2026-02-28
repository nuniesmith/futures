"""
Volume Profile analysis for CME futures.

Provides institutional-grade support/resistance via:
  - Point of Control (POC): price level with highest traded volume
  - Value Area High/Low (VAH/VAL): enclosing 70% of session volume
  - High/Low Volume Nodes (HVN/LVN): significant volume clusters/gaps
  - Naked POC tracking: unfilled POCs from prior sessions

Three trading setups (per todo.md blueprint):
  1. POC Mean Reversion — enter toward POC when price moves 30+ points away
  2. Value Area Rejection — bullish/bearish engulfing at VAL/VAH boundary
  3. Naked POC Magnet — trade toward unfilled POCs from prior sessions

Usage:
    from lib.volume_profile import (
        compute_volume_profile,
        compute_session_profiles,
        find_naked_pocs,
        VolumeProfileStrategy,
    )

    # Single session profile
    profile = compute_volume_profile(df, n_bins=50)
    print(profile["poc"], profile["vah"], profile["val"])

    # Multi-session with naked POC tracking
    sessions = compute_session_profiles(df, n_bins=50)
    naked = find_naked_pocs(sessions, current_price=2650.0)
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Strategy

logger = logging.getLogger("volume_profile")


# ---------------------------------------------------------------------------
# Core Volume Profile computation
# ---------------------------------------------------------------------------


def compute_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
) -> dict[str, Any]:
    """Compute the volume profile for a set of OHLCV bars.

    Distributes each bar's volume across price bins proportional to
    the bar's overlap with each bin. This is more accurate than simply
    assigning all volume to the close price.

    Args:
        df: DataFrame with columns High, Low, Close, Volume.
        n_bins: Number of price bins to divide the range into.
        value_area_pct: Fraction of total volume to include in the
            value area (default 70%).

    Returns:
        Dict with keys:
          - poc: float — Point of Control price
          - vah: float — Value Area High
          - val: float — Value Area Low
          - poc_volume: float — volume at the POC bin
          - total_volume: float — total volume across all bins
          - bin_edges: np.ndarray — edges of the price bins
          - bin_centers: np.ndarray — center prices of each bin
          - bin_volumes: np.ndarray — volume in each bin
          - hvn: list[float] — High Volume Node prices
          - lvn: list[float] — Low Volume Node prices
    """
    if df.empty or len(df) < 2:
        return _empty_profile()

    high = np.asarray(df["High"].astype(float).values, dtype=np.float64)
    low = np.asarray(df["Low"].astype(float).values, dtype=np.float64)
    volume = np.asarray(df["Volume"].astype(float).values, dtype=np.float64)

    price_min = float(np.nanmin(low))
    price_max = float(np.nanmax(high))

    if price_max <= price_min or np.isnan(price_min) or np.isnan(price_max):
        return _empty_profile()

    # Add small padding to avoid edge issues
    padding = (price_max - price_min) * 0.001
    price_min -= padding
    price_max += padding

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    _bin_width = bin_edges[1] - bin_edges[0]  # noqa: F841 — used for reference
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(n_bins, dtype=np.float64)

    # Distribute each bar's volume across overlapping bins
    for i in range(len(df)):
        bar_low = low[i]
        bar_high = high[i]
        bar_vol = volume[i]

        if np.isnan(bar_low) or np.isnan(bar_high) or np.isnan(bar_vol):
            continue
        if bar_vol <= 0 or bar_high <= bar_low:
            continue

        bar_range = bar_high - bar_low

        for j in range(n_bins):
            bin_lo = bin_edges[j]
            bin_hi = bin_edges[j + 1]

            # Calculate overlap between bar range and bin
            overlap_lo = max(bar_low, bin_lo)
            overlap_hi = min(bar_high, bin_hi)

            if overlap_hi > overlap_lo:
                overlap_pct = (overlap_hi - overlap_lo) / bar_range
                bin_volumes[j] += bar_vol * overlap_pct

    total_volume = bin_volumes.sum()
    if total_volume <= 0:
        return _empty_profile()

    # POC: bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])
    poc_volume = float(bin_volumes[poc_idx])

    # Value Area: expand outward from POC until value_area_pct is captured
    vah, val = _compute_value_area(
        bin_centers, bin_volumes, poc_idx, total_volume, value_area_pct
    )

    # High/Low Volume Nodes
    hvn, lvn = _find_volume_nodes(bin_centers, bin_volumes, total_volume, n_bins)

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "poc_volume": poc_volume,
        "total_volume": total_volume,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_volumes": bin_volumes,
        "hvn": hvn,
        "lvn": lvn,
    }


def _compute_value_area(
    bin_centers: np.ndarray,
    bin_volumes: np.ndarray,
    poc_idx: int,
    total_volume: float,
    value_area_pct: float,
) -> tuple[float, float]:
    """Expand outward from the POC bin to capture value_area_pct of volume.

    At each step, compare the volume of the next bin above vs below the
    current area. Add whichever is larger. Continue until the target
    percentage of total volume is enclosed.

    Returns (vah, val) — the high and low boundaries of the value area.
    """
    n = len(bin_centers)
    target_volume = total_volume * value_area_pct
    area_volume = bin_volumes[poc_idx]
    upper = poc_idx
    lower = poc_idx

    while area_volume < target_volume:
        can_go_up = upper + 1 < n
        can_go_down = lower - 1 >= 0

        if not can_go_up and not can_go_down:
            break

        vol_up = bin_volumes[upper + 1] if can_go_up else -1
        vol_down = bin_volumes[lower - 1] if can_go_down else -1

        if vol_up >= vol_down:
            upper += 1
            area_volume += bin_volumes[upper]
        else:
            lower -= 1
            area_volume += bin_volumes[lower]

    vah = float(bin_centers[upper])
    val = float(bin_centers[lower])
    return vah, val


def _find_volume_nodes(
    bin_centers: np.ndarray,
    bin_volumes: np.ndarray,
    total_volume: float,
    n_bins: int,
) -> tuple[list[float], list[float]]:
    """Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN).

    HVN: bins with volume > 1.5× average bin volume
    LVN: bins with volume < 0.5× average bin volume (gaps in profile)
    """
    avg_vol = total_volume / max(n_bins, 1)
    hvn_threshold = avg_vol * 1.5
    lvn_threshold = avg_vol * 0.5

    hvn = [
        float(bin_centers[i]) for i in range(n_bins) if bin_volumes[i] > hvn_threshold
    ]
    lvn = [
        float(bin_centers[i])
        for i in range(n_bins)
        if 0 < bin_volumes[i] < lvn_threshold
    ]

    return hvn, lvn


def _empty_profile() -> dict[str, Any]:
    """Return an empty profile dict when computation isn't possible."""
    return {
        "poc": 0.0,
        "vah": 0.0,
        "val": 0.0,
        "poc_volume": 0.0,
        "total_volume": 0.0,
        "bin_edges": np.array([]),
        "bin_centers": np.array([]),
        "bin_volumes": np.array([]),
        "hvn": [],
        "lvn": [],
    }


# ---------------------------------------------------------------------------
# Multi-session profiles
# ---------------------------------------------------------------------------


def compute_session_profiles(
    df: pd.DataFrame,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
    max_sessions: int = 10,
) -> list[dict[str, Any]]:
    """Compute volume profiles for each trading session (day).

    Splits the data by calendar date and computes a separate profile
    for each day. Returns a list of profile dicts, most recent last,
    with an additional "date" key.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns.
        n_bins: Number of price bins per session.
        value_area_pct: Value area percentage.
        max_sessions: Maximum number of sessions to return.

    Returns:
        List of profile dicts, each with an additional "date" key.
    """
    if df.empty:
        return []

    # Group by date
    try:
        idx = df.index.to_series()
        if hasattr(idx.dt, "date"):
            dates = idx.dt.date
        else:
            dates = pd.to_datetime(idx).dt.date
    except Exception:
        return []

    profiles = []
    unique_dates = sorted(dates.unique())

    for d in unique_dates[-max_sessions:]:
        day_mask = dates == d
        day_df = df.loc[day_mask]
        if len(day_df) < 5:
            continue

        profile = compute_volume_profile(day_df, n_bins, value_area_pct)
        profile["date"] = d
        profile["open"] = (
            float(day_df["Close"].iloc[0]) if "Close" in day_df.columns else 0.0
        )
        profile["close"] = (
            float(day_df["Close"].iloc[-1]) if "Close" in day_df.columns else 0.0
        )
        profiles.append(profile)

    return profiles


def find_naked_pocs(
    session_profiles: list[dict[str, Any]],
    current_price: float,
    max_distance_points: float = 100.0,
) -> list[dict[str, Any]]:
    """Find unfilled (naked) POCs from prior sessions.

    A POC is "naked" if price has not traded through it in any subsequent
    session. These act as magnets — price tends to revisit them.

    Args:
        session_profiles: List of session profile dicts (from compute_session_profiles).
        current_price: The current price to measure distance from.
        max_distance_points: Only return naked POCs within this many points.

    Returns:
        List of dicts with "date", "poc", "distance", "direction" keys,
        sorted by absolute distance (nearest first).
    """
    if not session_profiles or len(session_profiles) < 2:
        return []

    naked_pocs = []

    # Check each session's POC against all subsequent sessions
    for i in range(len(session_profiles) - 1):
        poc = session_profiles[i]["poc"]
        if poc <= 0:
            continue

        # Check if any subsequent session traded through this POC
        was_filled = False
        for j in range(i + 1, len(session_profiles)):
            subsequent = session_profiles[j]
            # A POC is filled if subsequent session's price range includes it
            try:
                sub_low = subsequent["val"]  # use VAL as proxy for session low
                sub_high = subsequent["vah"]  # use VAH as proxy for session high
                # More accurate: check actual data range
                if (
                    subsequent.get("bin_edges") is not None
                    and len(subsequent["bin_edges"]) > 1
                ):
                    sub_low = float(subsequent["bin_edges"][0])
                    sub_high = float(subsequent["bin_edges"][-1])

                if sub_low <= poc <= sub_high:
                    was_filled = True
                    break
            except (KeyError, TypeError, IndexError):
                continue

        if not was_filled:
            distance = current_price - poc
            if abs(distance) <= max_distance_points:
                naked_pocs.append(
                    {
                        "date": session_profiles[i].get("date"),
                        "poc": poc,
                        "distance": round(distance, 2),
                        "abs_distance": round(abs(distance), 2),
                        "direction": "above" if distance > 0 else "below",
                    }
                )

    naked_pocs.sort(key=lambda x: x["abs_distance"])
    return naked_pocs


# ---------------------------------------------------------------------------
# Indicator helpers for backtesting.py strategies
# ---------------------------------------------------------------------------


def _passthrough(arr):
    """Identity function for pre-computed indicator arrays."""
    return arr


def _ema(series, length: int):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=length, adjust=False).mean()


def _atr(high, low, close, length: int = 14):
    """Average True Range."""
    h, lo, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr1 = h - lo
    tr2 = (h - c.shift(1)).abs()
    tr3 = (lo - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def _rolling_poc(high, low, close, volume, lookback: int = 100, n_bins: int = 30):
    """Compute a rolling POC over a lookback window.

    Returns a pandas Series with the POC price at each bar, calculated
    over the trailing `lookback` bars.
    """
    h = np.asarray(pd.Series(high).values, dtype=np.float64)
    lo_arr = np.asarray(pd.Series(low).values, dtype=np.float64)
    c = np.asarray(pd.Series(close).values, dtype=np.float64)
    v = np.asarray(pd.Series(volume).values, dtype=np.float64)
    n = len(h)
    poc_series = np.full(n, np.nan)

    for i in range(lookback, n):
        start = i - lookback
        window_high = h[start:i]
        window_low = lo_arr[start:i]
        window_vol = v[start:i]

        price_min = np.nanmin(window_low)
        price_max = np.nanmax(window_high)
        if price_max <= price_min:
            poc_series[i] = c[i]
            continue

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_volumes = np.zeros(n_bins)

        for j in range(lookback):
            bar_lo = window_low[j]
            bar_hi = window_high[j]
            bar_vol = window_vol[j]
            if np.isnan(bar_lo) or np.isnan(bar_hi) or np.isnan(bar_vol):
                continue
            if bar_vol <= 0 or bar_hi <= bar_lo:
                continue
            bar_range = bar_hi - bar_lo
            for k in range(n_bins):
                ov_lo = max(bar_lo, bin_edges[k])
                ov_hi = min(bar_hi, bin_edges[k + 1])
                if ov_hi > ov_lo:
                    bin_volumes[k] += bar_vol * (ov_hi - ov_lo) / bar_range

        poc_idx = int(np.argmax(bin_volumes))
        poc_series[i] = bin_centers[poc_idx]

    return pd.Series(poc_series, index=pd.Series(close).index)


def _rolling_vah_val(
    high,
    low,
    close,
    volume,
    lookback: int = 100,
    n_bins: int = 30,
    value_area_pct: float = 0.70,
):
    """Compute rolling VAH and VAL over a lookback window.

    Returns two pandas Series: (vah_series, val_series).
    """
    h = np.asarray(pd.Series(high).values, dtype=np.float64)
    lo_arr = np.asarray(pd.Series(low).values, dtype=np.float64)
    c = np.asarray(pd.Series(close).values, dtype=np.float64)
    v = np.asarray(pd.Series(volume).values, dtype=np.float64)
    n = len(h)
    vah_series = np.full(n, np.nan)
    val_series = np.full(n, np.nan)

    for i in range(lookback, n):
        start = i - lookback
        window_high = h[start:i]
        window_low = lo_arr[start:i]
        window_vol = v[start:i]

        price_min = np.nanmin(window_low)
        price_max = np.nanmax(window_high)
        if price_max <= price_min:
            vah_series[i] = c[i]
            val_series[i] = c[i]
            continue

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_volumes = np.zeros(n_bins)

        for j in range(lookback):
            bar_lo = window_low[j]
            bar_hi = window_high[j]
            bar_vol = window_vol[j]
            if np.isnan(bar_lo) or np.isnan(bar_hi) or np.isnan(bar_vol):
                continue
            if bar_vol <= 0 or bar_hi <= bar_lo:
                continue
            bar_range = bar_hi - bar_lo
            for k in range(n_bins):
                ov_lo = max(bar_lo, bin_edges[k])
                ov_hi = min(bar_hi, bin_edges[k + 1])
                if ov_hi > ov_lo:
                    bin_volumes[k] += bar_vol * (ov_hi - ov_lo) / bar_range

        total_vol = bin_volumes.sum()
        if total_vol <= 0:
            vah_series[i] = c[i]
            val_series[i] = c[i]
            continue

        poc_idx = int(np.argmax(bin_volumes))
        target_vol = total_vol * value_area_pct
        area_vol = bin_volumes[poc_idx]
        upper = poc_idx
        lower = poc_idx

        while area_vol < target_vol:
            can_up = upper + 1 < n_bins
            can_down = lower - 1 >= 0
            if not can_up and not can_down:
                break
            vol_up = bin_volumes[upper + 1] if can_up else -1
            vol_down = bin_volumes[lower - 1] if can_down else -1
            if vol_up >= vol_down:
                upper += 1
                area_vol += bin_volumes[upper]
            else:
                lower -= 1
                area_vol += bin_volumes[lower]

        vah_series[i] = bin_centers[upper]
        val_series[i] = bin_centers[lower]

    idx = pd.Series(close).index
    return pd.Series(vah_series, index=idx), pd.Series(val_series, index=idx)


# ---------------------------------------------------------------------------
# Strategy: Volume Profile (POC Mean Reversion + Value Area Rejection)
# ---------------------------------------------------------------------------


class VolumeProfileStrategy(Strategy):
    """Volume Profile strategy combining POC reversion and Value Area rejection.

    Setup 1 — POC Mean Reversion:
        When price moves significantly away from the rolling POC (> atr_dist_mult × ATR),
        enter toward the POC. Price reverts to POC ~75% of the time in ranging markets.

    Setup 2 — Value Area Rejection:
        When price dips below VAL and closes back above it (bullish),
        or pushes above VAH and closes back below it (bearish),
        enter in the rejection direction targeting POC then opposite VA boundary.

    Parameters (Optuna-tunable):
        vp_lookback: int — bars for rolling volume profile (50–200)
        vp_bins: int — number of price bins (20–60)
        atr_period: int — ATR period for distance/SL/TP
        atr_dist_mult: float — ATR multiple for POC distance threshold
        atr_sl_mult: float — ATR multiple for stop loss
        atr_tp_mult: float — ATR multiple for take profit
        trade_size: float — position size as fraction of equity
    """

    # Tunable parameters with defaults
    vp_lookback: int = 100
    vp_bins: int = 30
    atr_period: int = 14
    atr_dist_mult: float = 1.5  # enter when price is this many ATRs from POC
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        # ATR for distance thresholds and SL/TP
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

        # Rolling POC
        self.poc = self.I(
            _rolling_poc,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.vp_lookback,
            self.vp_bins,
            name="POC",
        )

        # Rolling VAH/VAL — compute together, store as separate indicators
        vah_arr, val_arr = _rolling_vah_val(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.vp_lookback,
            self.vp_bins,
        )
        self.vah = self.I(_passthrough, vah_arr.values, name="VAH")
        self.val_line = self.I(_passthrough, val_arr.values, name="VAL")

        # Trend filter: 50-period EMA
        self.trend_ema = self.I(_ema, pd.Series(self.data.Close), 50, name="EMA50")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        poc = self.poc[-1]
        vah = self.vah[-1]
        val = self.val_line[-1]
        trend = self.trend_ema[-1]

        # Skip if indicators aren't ready
        if (
            math.isnan(atr_val)
            or math.isnan(poc)
            or math.isnan(vah)
            or math.isnan(val)
            or math.isnan(trend)
            or atr_val <= 0
        ):
            return

        # If already in a position, manage it (SL/TP handled by order placement)
        if self.position:
            return

        dist_from_poc = price - poc
        atr_threshold = atr_val * self.atr_dist_mult

        sl_dist = atr_val * self.atr_sl_mult
        tp_dist = atr_val * self.atr_tp_mult

        # --- Setup 1: POC Mean Reversion ---
        # Price is far above POC → short toward POC (if below trend = bearish bias)
        if dist_from_poc > atr_threshold and price < trend:
            sl = price + sl_dist
            tp = price - tp_dist
            self.sell(size=self.trade_size, sl=sl, tp=tp)
            return

        # Price is far below POC → long toward POC (if above trend = bullish bias)
        if dist_from_poc < -atr_threshold and price > trend:
            sl = price - sl_dist
            tp = price + tp_dist
            self.buy(size=self.trade_size, sl=sl, tp=tp)
            return

        # --- Setup 2: Value Area Rejection ---
        if len(self.data.Close) < 3:
            return

        prev_close = self.data.Close[-2]
        prev_low = self.data.Low[-2]
        prev_high = self.data.High[-2]
        prev_open = self.data.Open[-2]

        # Bullish rejection: previous bar dipped below VAL, current close above VAL
        # and previous bar shows bullish character (close > open = bullish candle)
        if prev_low < val and price > val and prev_close > prev_open:
            sl = price - sl_dist
            tp = poc  # target POC first
            # Ensure TP is actually above entry
            if tp <= price:
                tp = price + tp_dist
            self.buy(size=self.trade_size, sl=sl, tp=tp)
            return

        # Bearish rejection: previous bar pushed above VAH, current close below VAH
        # and previous bar shows bearish character (close < open = bearish candle)
        if prev_high > vah and price < vah and prev_close < prev_open:
            sl = price + sl_dist
            tp = poc  # target POC first
            # Ensure TP is actually below entry
            if tp >= price:
                tp = price - tp_dist
            self.sell(size=self.trade_size, sl=sl, tp=tp)
            return


# ---------------------------------------------------------------------------
# Optuna parameter suggestion for optimizer integration
# ---------------------------------------------------------------------------


def suggest_volume_profile_params(trial) -> dict:
    """Ask Optuna to suggest parameters for VolumeProfileStrategy.

    Compatible with the optimizer in engine.py.
    """
    return {
        "vp_lookback": trial.suggest_int("vp_lookback", 50, 200, step=10),
        "vp_bins": trial.suggest_int("vp_bins", 20, 60, step=5),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "atr_dist_mult": trial.suggest_float("atr_dist_mult", 1.0, 3.0, step=0.25),
        "atr_sl_mult": trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25),
        "atr_tp_mult": trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25),
        "trade_size": trial.suggest_float("trade_size", 0.05, 0.30, step=0.05),
    }


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------


def profile_to_dataframe(profile: dict[str, Any]) -> pd.DataFrame:
    """Convert a volume profile to a DataFrame for display/plotting.

    Returns a DataFrame with columns: Price, Volume, IsPOC, InValueArea.
    """
    if not profile or len(profile.get("bin_centers", [])) == 0:
        return pd.DataFrame(
            columns=pd.Index(["Price", "Volume", "IsPOC", "InValueArea"])
        )

    centers = profile["bin_centers"]
    volumes = profile["bin_volumes"]
    poc = profile["poc"]
    vah = profile["vah"]
    val = profile["val"]

    rows = []
    for i, (price, vol) in enumerate(zip(centers, volumes)):
        rows.append(
            {
                "Price": round(float(price), 2),
                "Volume": float(vol),
                "IsPOC": abs(float(price) - poc) < (centers[1] - centers[0]) * 0.6
                if len(centers) > 1
                else False,
                "InValueArea": val <= float(price) <= vah,
            }
        )

    return pd.DataFrame(rows)


def format_profile_summary(profile: dict[str, Any]) -> str:
    """One-line summary of a volume profile for display."""
    if not profile or profile["poc"] == 0:
        return "No volume profile data"
    return (
        f"POC: {profile['poc']:.2f} | "
        f"VAH: {profile['vah']:.2f} | "
        f"VAL: {profile['val']:.2f} | "
        f"HVN: {len(profile.get('hvn', []))} | "
        f"LVN: {len(profile.get('lvn', []))}"
    )
