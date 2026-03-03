#!/usr/bin/env python
"""
Generate Synthetic 1-Minute Bar Data for Backtest Testing
==========================================================
Produces realistic-looking 1-minute OHLCV CSVs so you can run
``scripts/backtest_filters.py`` without needing live market data
or a Redis/Massive connection.

The generator creates bars that mimic real futures price action:
  - Overnight/pre-market session (00:00–09:30 ET) with lower volume
  - Opening range (09:30–10:00 ET) with a defined OR high/low
  - Post-OR session (10:00–16:00 ET) with occasional breakouts
  - Lunch chop zone (11:00–13:00 ET) with reduced volatility
  - Some days produce clean breakouts (follow-through), others fail
  - NR7 days are sprinkled in (~15% of days)
  - Volume spikes on breakout bars

Usage:
    # From project root:
    python scripts/generate_sample_bars.py --symbols MGC MES MNQ --days 60

    # Custom output directory:
    python scripts/generate_sample_bars.py --symbols MGC --days 30 --output data/bars

    # Then run the backtest:
    PYTHONPATH=src python scripts/backtest_filters.py --symbols MGC MES MNQ --source csv --csv-dir data/bars -v

Output:
    data/bars/{SYMBOL}_1m.csv   — 1-minute OHLCV bars
    data/bars/{SYMBOL}_daily.csv — daily OHLCV bars (for NR7)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Price profiles for different instruments
# ---------------------------------------------------------------------------

INSTRUMENT_PROFILES = {
    "MGC": {
        "name": "Micro Gold",
        "base_price": 2350.0,
        "tick_size": 0.10,
        "avg_daily_range": 25.0,  # typical daily H-L in points
        "avg_or_range": 8.0,  # typical opening range in points
        "avg_volume_per_bar": 120,
        "breakout_rate": 0.55,  # % of days that produce a breakout
        "follow_through_rate": 0.45,  # % of breakouts that hit TP1
    },
    "MES": {
        "name": "Micro E-mini S&P 500",
        "base_price": 5900.0,
        "tick_size": 0.25,
        "avg_daily_range": 50.0,
        "avg_or_range": 15.0,
        "avg_volume_per_bar": 350,
        "breakout_rate": 0.60,
        "follow_through_rate": 0.42,
    },
    "MNQ": {
        "name": "Micro E-mini Nasdaq",
        "base_price": 20500.0,
        "tick_size": 0.25,
        "avg_daily_range": 200.0,
        "avg_or_range": 60.0,
        "avg_volume_per_bar": 280,
        "breakout_rate": 0.58,
        "follow_through_rate": 0.40,
    },
    "MCL": {
        "name": "Micro Crude Oil",
        "base_price": 72.0,
        "tick_size": 0.01,
        "avg_daily_range": 1.5,
        "avg_or_range": 0.5,
        "avg_volume_per_bar": 90,
        "breakout_rate": 0.52,
        "follow_through_rate": 0.43,
    },
}


def _round_to_tick(price: float, tick_size: float) -> float:
    """Round price to the nearest valid tick."""
    return round(round(price / tick_size) * tick_size, 6)


def _generate_random_walk(
    n: int,
    start: float,
    volatility: float,
    drift: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a random walk price series."""
    if rng is None:
        rng = np.random.default_rng()
    steps = rng.normal(loc=drift, scale=volatility, size=n)
    return start + np.cumsum(steps)


def generate_day_bars(
    day_date: date,
    profile: dict,
    rng: np.random.Generator,
    is_nr7: bool = False,
    force_breakout: bool | None = None,
    force_follow_through: bool | None = None,
    trend_bias: float = 0.0,
) -> pd.DataFrame:
    """Generate 1-minute bars for a single trading day.

    Args:
        day_date: The calendar date.
        profile: Instrument profile dict.
        rng: Numpy random generator.
        is_nr7: If True, compress the daily range.
        force_breakout: Override random breakout decision.
        force_follow_through: Override random follow-through decision.
        trend_bias: +1 = bullish day, -1 = bearish day, 0 = neutral.

    Returns:
        DataFrame with DatetimeIndex (ET) and OHLCV columns.
    """
    base = profile["base_price"]
    tick = profile["tick_size"]
    daily_range = profile["avg_daily_range"]
    or_range = profile["avg_or_range"]
    avg_vol = profile["avg_volume_per_bar"]

    # Daily variation: drift the base price slightly each day
    day_offset = rng.normal(0, daily_range * 0.3)
    day_base = base + day_offset

    # NR7: compress range to ~60% of normal
    range_mult = 0.55 + rng.uniform(0, 0.15) if is_nr7 else 0.8 + rng.uniform(0, 0.5)
    effective_range = daily_range * range_mult

    # Decide if breakout and follow-through
    has_breakout = force_breakout if force_breakout is not None else (rng.random() < profile["breakout_rate"])
    has_follow_through = (
        force_follow_through if force_follow_through is not None else (rng.random() < profile["follow_through_rate"])
    )

    # Determine breakout direction
    if trend_bias > 0.3:
        direction = "LONG"
    elif trend_bias < -0.3:
        direction = "SHORT"
    else:
        direction = "LONG" if rng.random() < 0.5 else "SHORT"

    rows = []

    # --- Pre-market session: 00:00 – 09:30 ET ---
    pm_start = datetime(day_date.year, day_date.month, day_date.day, 0, 0, tzinfo=_EST)
    pm_bars = 9 * 60 + 30  # minutes

    pm_vol_base = max(10, avg_vol * 0.15)
    pm_volatility = effective_range * 0.002
    pm_prices = _generate_random_walk(pm_bars, day_base, pm_volatility, rng=rng)

    for i in range(pm_bars):
        t = pm_start + timedelta(minutes=i)
        mid = pm_prices[i]
        spread = rng.uniform(0.3, 1.5) * tick * 4
        o = _round_to_tick(mid + rng.normal(0, spread * 0.3), tick)
        c = _round_to_tick(mid + rng.normal(0, spread * 0.3), tick)
        h = _round_to_tick(max(o, c) + abs(rng.normal(0, spread * 0.5)), tick)
        l = _round_to_tick(min(o, c) - abs(rng.normal(0, spread * 0.5)), tick)
        v = max(1, int(rng.poisson(pm_vol_base * (1 + 0.3 * rng.random()))))
        rows.append({"Date": t, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    # Pre-market high/low (for filter testing)
    pm_highs = [r["High"] for r in rows]
    pm_lows = [r["Low"] for r in rows]
    pm_high = max(pm_highs) if pm_highs else day_base + or_range
    pm_low = min(pm_lows) if pm_lows else day_base - or_range

    # --- Opening Range: 09:30 – 10:00 ET ---
    or_start_time = datetime(day_date.year, day_date.month, day_date.day, 9, 30, tzinfo=_EST)
    or_bars_count = 30
    or_mid = rows[-1]["Close"] if rows else day_base
    or_half = or_range * range_mult * 0.5

    or_high = or_mid + or_half
    or_low = or_mid - or_half
    or_volatility = or_range * range_mult * 0.02

    # Walk within the OR range
    or_price = or_mid
    or_vol_base = avg_vol * 1.5  # higher volume at open

    for i in range(or_bars_count):
        t = or_start_time + timedelta(minutes=i)
        step = rng.normal(0, or_volatility)
        or_price = np.clip(or_price + step, or_low + tick, or_high - tick)
        spread = rng.uniform(0.5, 1.5) * tick * 3
        o = _round_to_tick(or_price + rng.normal(0, spread * 0.2), tick)
        c = _round_to_tick(or_price + rng.normal(0, spread * 0.2), tick)
        h = _round_to_tick(min(or_high, max(o, c) + abs(rng.normal(0, spread * 0.4))), tick)
        l = _round_to_tick(max(or_low, min(o, c) - abs(rng.normal(0, spread * 0.4))), tick)
        v = max(1, int(rng.poisson(or_vol_base)))
        rows.append({"Date": t, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    # --- Post-OR session: 10:00 – 16:00 ET ---
    post_or_start = datetime(day_date.year, day_date.month, day_date.day, 10, 0, tzinfo=_EST)
    post_or_bars = 6 * 60  # 6 hours

    post_price = rows[-1]["Close"]
    post_volatility = effective_range * 0.008

    # Breakout parameters
    breakout_triggered = False
    breakout_bar = rng.integers(5, 40) if has_breakout else -1
    atr_estimate = effective_range * 0.06  # rough ATR for bracket sizing

    for i in range(post_or_bars):
        t = post_or_start + timedelta(minutes=i)
        hour = t.hour

        # Lunch chop: reduce volatility 11:00–13:00
        if 11 <= hour < 13:
            vol_mult = 0.4
            vol_vol = avg_vol * 0.5
        elif hour >= 15:
            vol_mult = 0.6
            vol_vol = avg_vol * 0.7
        else:
            vol_mult = 1.0
            vol_vol = avg_vol * 1.0

        effective_vol = post_volatility * vol_mult

        # Breakout mechanics
        drift = 0.0
        vol_spike = 1.0

        if has_breakout and i == breakout_bar:
            breakout_triggered = True
            # Strong move in breakout direction
            if direction == "LONG":
                drift = or_range * 0.15
            else:
                drift = -or_range * 0.15
            vol_spike = 2.5 + rng.random()

        elif breakout_triggered and i > breakout_bar:
            bars_since = i - breakout_bar

            if has_follow_through:
                # Sustained drift in breakout direction, fading over time
                fade = max(0.1, 1.0 - bars_since / 120)
                if direction == "LONG":
                    drift = or_range * 0.02 * fade
                else:
                    drift = -or_range * 0.02 * fade
                vol_spike = max(1.0, 1.5 - bars_since / 60)
            else:
                # Failed breakout: reverse after ~15–30 bars
                if bars_since < rng.integers(15, 35):
                    # Initial continuation (fake)
                    if direction == "LONG":
                        drift = or_range * 0.005
                    else:
                        drift = -or_range * 0.005
                else:
                    # Reversal
                    if direction == "LONG":
                        drift = -or_range * 0.025
                    else:
                        drift = or_range * 0.025
                vol_spike = max(1.0, 1.2 - bars_since / 80)

        step = rng.normal(drift, effective_vol)
        post_price += step

        spread = rng.uniform(0.5, 2.0) * tick * 3
        o_val = _round_to_tick(post_price + rng.normal(0, spread * 0.2), tick)
        c_val = _round_to_tick(post_price + rng.normal(0, spread * 0.2), tick)
        h_val = _round_to_tick(max(o_val, c_val) + abs(rng.normal(0, spread * 0.5)), tick)
        l_val = _round_to_tick(min(o_val, c_val) - abs(rng.normal(0, spread * 0.5)), tick)
        v_val = max(1, int(rng.poisson(vol_vol * vol_spike)))

        rows.append({"Date": t, "Open": o_val, "High": h_val, "Low": l_val, "Close": c_val, "Volume": v_val})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], utc=False)
    df = df.set_index("Date")

    return df


def generate_bars(
    symbol: str,
    days: int = 60,
    seed: int | None = None,
    output_dir: str = "data/bars",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate multi-day 1-minute bars and daily bars for a symbol.

    Args:
        symbol: Instrument symbol (must be in INSTRUMENT_PROFILES, or a
                generic profile is used).
        days: Number of trading days to generate.
        seed: Random seed for reproducibility.
        output_dir: Directory to save CSVs.

    Returns:
        (bars_1m, bars_daily) DataFrames.
    """
    rng = np.random.default_rng(seed)

    profile = INSTRUMENT_PROFILES.get(
        symbol.upper(),
        {
            "name": symbol,
            "base_price": 1000.0,
            "tick_size": 0.01,
            "avg_daily_range": 15.0,
            "avg_or_range": 5.0,
            "avg_volume_per_bar": 100,
            "breakout_rate": 0.55,
            "follow_through_rate": 0.42,
        },
    )

    # Generate trading days (skip weekends)
    start_date = date.today() - timedelta(days=int(days * 1.5))
    trading_days = []
    d = start_date
    while len(trading_days) < days:
        if d.weekday() < 5:  # Mon–Fri
            trading_days.append(d)
        d += timedelta(days=1)

    # Decide NR7 days (~15% of days, in clusters)
    nr7_flags = [False] * days
    nr7_cluster_start = rng.choice(range(7, days - 3), size=max(1, days // 8), replace=False)
    for start_idx in nr7_cluster_start:
        cluster_len = rng.integers(1, 3)
        for offset in range(cluster_len):
            idx = start_idx + offset
            if idx < days:
                nr7_flags[idx] = True

    # Multi-day trend bias (mean-reverting random walk)
    trend = 0.0
    trends = []
    for _ in range(days):
        trend += rng.normal(-trend * 0.1, 0.3)
        trend = np.clip(trend, -1.0, 1.0)
        trends.append(trend)

    # Vary follow-through rate so some periods are better than others
    all_1m = []
    daily_rows = []

    for i, day_d in enumerate(trading_days):
        # Gradually drift base price to simulate trending market
        profile_copy = dict(profile)
        profile_copy["base_price"] = profile["base_price"] + trends[i] * profile["avg_daily_range"] * 2

        day_df = generate_day_bars(
            day_date=day_d,
            profile=profile_copy,
            rng=rng,
            is_nr7=nr7_flags[i],
            trend_bias=trends[i],
        )
        all_1m.append(day_df)

        # Daily bar summary
        daily_rows.append(
            {
                "Date": pd.Timestamp(day_d, tz=_EST),
                "Open": day_df["Open"].iloc[0],
                "High": day_df["High"].max(),
                "Low": day_df["Low"].min(),
                "Close": day_df["Close"].iloc[-1],
                "Volume": day_df["Volume"].sum(),
            }
        )

    bars_1m = pd.concat(all_1m)
    bars_daily = pd.DataFrame(daily_rows).set_index("Date")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)

    path_1m = os.path.join(output_dir, f"{symbol.upper()}_1m.csv")
    path_daily = os.path.join(output_dir, f"{symbol.upper()}_daily.csv")

    bars_1m.to_csv(path_1m)
    bars_daily.to_csv(path_daily)

    return bars_1m, bars_daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 1-minute bar data for backtest testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_sample_bars.py --symbols MGC MES MNQ --days 60
  python scripts/generate_sample_bars.py --symbols MGC --days 30 --output data/bars --seed 42

Then run the filter backtest:
  PYTHONPATH=src python scripts/backtest_filters.py --symbols MGC MES MNQ --source csv --csv-dir data/bars -v
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["MGC", "MES", "MNQ"],
        help="Symbols to generate (default: MGC MES MNQ)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of trading days to generate (default: 60)",
    )
    parser.add_argument(
        "--output",
        default="data/bars",
        help="Output directory for CSVs (default: data/bars)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Synthetic Bar Data Generator")
    print("=" * 60)

    for symbol in args.symbols:
        print(f"\n📊 Generating {args.days} days of 1-minute bars for {symbol}...")

        bars_1m, bars_daily = generate_bars(
            symbol=symbol,
            days=args.days,
            seed=args.seed,
            output_dir=args.output,
        )

        bars_per_day = len(bars_1m) / args.days
        print(f"  ✅ {len(bars_1m):,} bars ({bars_per_day:.0f}/day)")
        print(f"     Price range: {bars_1m['Low'].min():.2f} – {bars_1m['High'].max():.2f}")
        print(f"     Date range:  {bars_1m.index.min()} → {bars_1m.index.max()}")
        print(f"     Daily bars:  {len(bars_daily)} days")
        print(f"     Saved to:    {args.output}/{symbol}_1m.csv")
        print(f"                  {args.output}/{symbol}_daily.csv")

    print(f"\n{'=' * 60}")
    print(f"  Done! Now run the backtest:")
    print(f"  PYTHONPATH=src python scripts/backtest_filters.py \\")
    print(f"    --symbols {' '.join(args.symbols)} --source csv --csv-dir {args.output} -v")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
