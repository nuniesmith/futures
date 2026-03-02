#!/usr/bin/env python
"""
ORB Filter Backtest Comparison
================================
Compares ORB trade outcomes WITH and WITHOUT the deterministic filter gate
across historical 1-minute bar data.

For each trading day in the dataset, the script:
  1. Detects the ORB and simulates the trade (entry, SL, TP1 via Bridge brackets).
  2. Runs the same trade through the quality filter gate (NR7, pre-market range,
     session window, lunch filter, multi-TF EMA bias, VWAP confluence).
  3. Collects results into two buckets: BASELINE (all trades) and FILTERED
     (only trades that pass the filter gate).
  4. Prints a side-by-side comparison table.

Usage:
    # From project root, using CSV bars:
    python scripts/backtest_filters.py --symbols MGC MES MNQ --source csv --csv-dir data/bars

    # Using Redis cache:
    python scripts/backtest_filters.py --symbols MGC --source cache --days 60

    # Using Massive API:
    python scripts/backtest_filters.py --symbols MGC MES --source massive --days 90

    # Gate mode: majority (pass if >50% of hard filters pass, instead of all):
    python scripts/backtest_filters.py --symbols MGC --source csv --gate-mode majority

    # Export per-trade detail to CSV:
    python scripts/backtest_filters.py --symbols MGC --source csv --export results.csv

Requirements:
    - pandas, numpy (already in project)
    - Access to 1-minute bars via one of: CSV files, Redis cache, or Massive API
    - No GPU, no torch, no mplfinance needed.

Run from the project root so that ``src/`` is importable:
    cd /path/to/futures
    PYTHONPATH=src python scripts/backtest_filters.py ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the import path so lib.* resolves
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd

from lib.analysis.orb_filters import (
    ORBFilterResult,
    apply_all_filters,
    check_lunch_filter,
    check_multi_tf_bias,
    check_nr7,
    check_premarket_range,
    check_session_window,
    check_vwap_confluence,
    extract_premarket_range,
)
from lib.analysis.orb_simulator import (
    BracketConfig,
    ORBSimResult,
    simulate_orb_outcome,
    summarise_results,
)
from lib.services.engine.orb import LONDON_SESSION, US_SESSION, ORBSession

logger = logging.getLogger("backtest_filters")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Data loading (reuse dataset_generator loaders with fallback)
# ---------------------------------------------------------------------------


def _load_bars(
    symbol: str,
    source: str = "csv",
    days: int = 90,
    csv_dir: str = "data/bars",
) -> Optional[pd.DataFrame]:
    """Load 1-minute bars with a fallback chain.

    Tries the dataset_generator's ``load_bars`` first (which has its own
    fallback logic).  If that module is unavailable or fails, falls back
    to a simple CSV reader.
    """
    try:
        from lib.analysis.dataset_generator import load_bars

        df = load_bars(symbol, source=source, days=days, csv_dir=csv_dir)
        if df is not None and not df.empty:
            return df
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("dataset_generator.load_bars failed for %s: %s", symbol, exc)

    # Direct CSV fallback
    for name in [f"{symbol}_1m.csv", f"{symbol.lower()}_1m.csv", f"{symbol.upper()}_1m.csv"]:
        path = os.path.join(csv_dir, name)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                _normalise_columns(df)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                logger.info("Loaded %d bars for %s from %s", len(df), symbol, path)
                return df
            except Exception as exc:
                logger.warning("CSV load failed for %s: %s", path, exc)

    return None


def _load_daily_bars(
    symbol: str,
    source: str = "csv",
    csv_dir: str = "data/bars",
) -> Optional[pd.DataFrame]:
    """Load or derive daily bars for NR7 detection."""
    try:
        from lib.analysis.dataset_generator import load_daily_bars

        df = load_daily_bars(symbol, source=source, csv_dir=csv_dir)
        if df is not None and not df.empty:
            return df
    except ImportError:
        pass
    except Exception:
        pass

    return None


def _normalise_columns(df: pd.DataFrame) -> None:
    """In-place column rename to canonical OHLCV names."""
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("open", "o"):
            col_map[col] = "Open"
        elif cl in ("high", "h"):
            col_map[col] = "High"
        elif cl in ("low", "l"):
            col_map[col] = "Low"
        elif cl in ("close", "c"):
            col_map[col] = "Close"
        elif cl in ("volume", "vol", "v"):
            col_map[col] = "Volume"
    if col_map:
        df.rename(columns=col_map, inplace=True)


# ---------------------------------------------------------------------------
# Session splitting
# ---------------------------------------------------------------------------


def _ensure_est(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a tz-aware DatetimeIndex in US/Eastern."""
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        out.index = pd.to_datetime(idx)
        idx = out.index
    if idx.tz is None:
        try:
            out.index = idx.tz_localize(_EST)
        except Exception:
            out.index = idx.tz_localize("UTC").tz_convert(_EST)
    elif str(idx.tz) != str(_EST):
        out.index = idx.tz_convert(_EST)
    return out


def split_into_sessions(
    bars_1m: pd.DataFrame,
    session_start: dt_time = dt_time(0, 0),
    session_end: dt_time = dt_time(23, 59),
) -> dict[date, pd.DataFrame]:
    """Split a multi-day 1m DataFrame into per-calendar-day slices.

    Returns a dict mapping ``date → DataFrame`` (sorted by date).
    Days with fewer than 30 bars are dropped.
    """
    df = _ensure_est(bars_1m).sort_index()
    sessions: dict[date, pd.DataFrame] = {}

    for day_date, group in df.groupby(df.index.date):
        if len(group) >= 30:
            sessions[day_date] = group

    return dict(sorted(sessions.items()))


# ---------------------------------------------------------------------------
# Per-day backtest logic
# ---------------------------------------------------------------------------


@dataclass
class DayResult:
    """Result of one trading day's ORB simulation + filter evaluation."""

    day: date
    symbol: str

    # Simulation result (always present if a trade was detected)
    sim: Optional[ORBSimResult] = None

    # Filter evaluation (None if sim produced no trade)
    filter_result: Optional[ORBFilterResult] = None

    @property
    def has_trade(self) -> bool:
        return self.sim is not None and self.sim.is_trade

    @property
    def is_winner(self) -> bool:
        return self.has_trade and self.sim.is_winner

    @property
    def filter_passed(self) -> bool:
        if self.filter_result is None:
            return True  # no filter run → pass by default
        return self.filter_result.passed

    @property
    def pnl_r(self) -> float:
        return self.sim.pnl_r if self.has_trade else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "date": str(self.day),
            "symbol": self.symbol,
            "has_trade": self.has_trade,
            "direction": self.sim.direction if self.has_trade else "",
            "label": self.sim.label if self.sim else "no_trade",
            "outcome": self.sim.outcome if self.sim else "",
            "pnl_r": self.pnl_r,
            "quality_pct": self.sim.quality_pct if self.has_trade else 0,
            "hold_bars": self.sim.hold_bars if self.has_trade else 0,
            "entry": self.sim.entry if self.has_trade else 0.0,
            "or_high": self.sim.or_high if self.sim else 0.0,
            "or_low": self.sim.or_low if self.sim else 0.0,
            "atr": self.sim.atr if self.sim else 0.0,
            "nr7": self.sim.nr7 if self.sim else False,
            "breakout_volume_ratio": self.sim.breakout_volume_ratio if self.has_trade else 0.0,
            "filter_passed": self.filter_passed,
            "filter_summary": self.filter_result.summary if self.filter_result else "",
            "filters_passed_count": self.filter_result.filters_passed if self.filter_result else 0,
            "filters_total_count": self.filter_result.filters_total if self.filter_result else 0,
            "quality_boost": self.filter_result.quality_boost if self.filter_result else 0.0,
        }
        # Add individual filter verdicts
        if self.filter_result:
            for v in self.filter_result.verdicts:
                d[f"filter_{v.name.lower().replace(' ', '_').replace('-', '_')}"] = v.passed
        return d


def backtest_day(
    day_bars: pd.DataFrame,
    symbol: str,
    day_date: date,
    bracket_config: BracketConfig,
    bars_daily: Optional[pd.DataFrame] = None,
    gate_mode: str = "all",
    orb_session: Optional[ORBSession] = None,
) -> DayResult:
    """Run ORB simulation + filter evaluation on one day's data.

    Args:
        day_bars: 1-minute OHLCV for a single trading day.
        symbol: Instrument symbol.
        day_date: The calendar date.
        bracket_config: Bridge-style bracket parameters.
        bars_daily: Daily bars for NR7 detection (at least 7 rows).
        gate_mode: Filter gate mode — "all" or "majority".
        orb_session: ORBSession to evaluate (LONDON_SESSION or US_SESSION).
                     Defaults to US_SESSION if None.  Controls the session-aware
                     filter windows and pre-market extraction end time.

    Returns:
        DayResult with both simulation and filter outcomes.
    """
    if orb_session is None:
        orb_session = US_SESSION
    result = DayResult(day=day_date, symbol=symbol)

    # --- Run ORB simulation ---
    sim = simulate_orb_outcome(
        bars_1m=day_bars,
        symbol=symbol,
        config=bracket_config,
        bars_daily=bars_daily,
    )
    result.sim = sim

    if not sim.is_trade:
        return result

    # --- Run filter evaluation ---
    # Session-aware filter configuration:
    #   London (03:00–05:00 ET): premarket 00:00–03:00, no lunch filter.
    #   US     (08:20–10:30 ET): premarket 00:00–08:20, lunch filter active.
    _session_key = orb_session.key if orb_session else "us"

    if _session_key == "london":
        _filter_allowed_windows: list[tuple[dt_time, dt_time]] = [
            (dt_time(3, 0), dt_time(5, 0)),
        ]
        _pm_end = dt_time(3, 0)
        _enable_lunch = False
    else:
        _filter_allowed_windows = [
            (dt_time(8, 20), dt_time(10, 30)),
        ]
        _pm_end = dt_time(8, 20)
        _enable_lunch = True

    # Extract pre-market range with session-aware end time
    pm_high, pm_low = extract_premarket_range(day_bars, pm_end=_pm_end)

    # Derive HTF bars (15m) by resampling
    bars_htf = None
    try:
        bars_htf = (
            day_bars.resample("15min")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )
        if bars_htf.empty:
            bars_htf = None
    except Exception:
        pass

    # Determine signal time from breakout
    signal_time = datetime.now(tz=_EST)
    if sim.breakout_time:
        try:
            parsed = pd.Timestamp(sim.breakout_time)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize(_EST)
            signal_time = parsed.to_pydatetime()
        except Exception:
            pass

    filter_result = apply_all_filters(
        direction=sim.direction,
        trigger_price=sim.entry,
        signal_time=signal_time,
        bars_daily=bars_daily,
        bars_1m=day_bars,
        bars_htf=bars_htf,
        premarket_high=pm_high,
        premarket_low=pm_low,
        orb_high=sim.or_high,
        orb_low=sim.or_low,
        gate_mode=gate_mode,
        allowed_windows=_filter_allowed_windows,
        enable_lunch_filter=_enable_lunch,
    )
    result.filter_result = filter_result

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@dataclass
class BacktestStats:
    """Aggregate statistics for a set of day results."""

    label: str
    total_days: int = 0
    trade_days: int = 0
    no_trade_days: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    avg_quality: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    nr7_trades: int = 0
    nr7_winners: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_drawdown_r: float = 0.0


def compute_stats(results: list[DayResult], label: str = "") -> BacktestStats:
    """Compute aggregate statistics from a list of DayResults."""
    stats = BacktestStats(label=label)
    stats.total_days = len(results)

    trades = [r for r in results if r.has_trade]
    stats.trade_days = len(trades)
    stats.no_trade_days = stats.total_days - stats.trade_days

    if not trades:
        return stats

    winners = [r for r in trades if r.is_winner]
    losers = [r for r in trades if not r.is_winner]

    stats.winners = len(winners)
    stats.losers = len(losers)
    stats.win_rate = round(len(winners) / len(trades) * 100, 1) if trades else 0.0

    pnls = [r.pnl_r for r in trades]
    stats.avg_r = round(sum(pnls) / len(pnls), 3) if pnls else 0.0
    stats.total_r = round(sum(pnls), 2)

    total_win_r = sum(r.pnl_r for r in winners)
    total_loss_r = abs(sum(r.pnl_r for r in losers))
    stats.profit_factor = round(total_win_r / total_loss_r, 2) if total_loss_r > 0 else float("inf")

    stats.avg_hold_bars = round(sum(r.sim.hold_bars for r in trades) / len(trades), 1)
    stats.avg_quality = round(sum(r.sim.quality_pct for r in trades) / len(trades), 1)

    stats.long_trades = sum(1 for r in trades if r.sim.direction == "LONG")
    stats.short_trades = sum(1 for r in trades if r.sim.direction == "SHORT")
    stats.nr7_trades = sum(1 for r in trades if r.sim.nr7)
    stats.nr7_winners = sum(1 for r in trades if r.sim.nr7 and r.is_winner)

    # Consecutive streaks
    streak_w = 0
    streak_l = 0
    max_w = 0
    max_l = 0
    for r in trades:
        if r.is_winner:
            streak_w += 1
            streak_l = 0
            max_w = max(max_w, streak_w)
        else:
            streak_l += 1
            streak_w = 0
            max_l = max(max_l, streak_l)
    stats.max_consecutive_wins = max_w
    stats.max_consecutive_losses = max_l

    # Max drawdown in R
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in trades:
        equity += r.pnl_r
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
    stats.max_drawdown_r = round(max_dd, 2)

    return stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_COL_W = 22


def _fmt_row(label: str, baseline: Any, filtered: Any, delta: str = "") -> str:
    """Format one row of the comparison table."""
    return f"  {label:<26s} {str(baseline):>{_COL_W}s} {str(filtered):>{_COL_W}s} {delta:>{_COL_W}s}"


def print_comparison(baseline: BacktestStats, filtered: BacktestStats, symbol: str = "") -> None:
    """Print a formatted side-by-side comparison table."""
    header = f"\n{'=' * 100}"
    header += f"\n  ORB FILTER BACKTEST — {symbol or 'ALL SYMBOLS'}"
    header += f"\n{'=' * 100}"
    print(header)

    col_header = _fmt_row("Metric", "BASELINE (no filter)", "FILTERED", "Delta")
    print(col_header)
    print("  " + "-" * 96)

    def _delta(a: float, b: float, fmt: str = "+.1f", suffix: str = "") -> str:
        d = b - a
        if d == 0:
            return "—"
        return f"{d:{fmt}}{suffix}"

    def _pct_delta(a: float, b: float) -> str:
        if a == 0:
            return "—"
        d = ((b - a) / abs(a)) * 100
        return f"{d:+.1f}%"

    print(_fmt_row("Total days", baseline.total_days, filtered.total_days, ""))
    print(
        _fmt_row(
            "Trade days",
            f"{baseline.trade_days}",
            f"{filtered.trade_days}",
            _delta(baseline.trade_days, filtered.trade_days, "+d"),
        )
    )
    print(
        _fmt_row(
            "No-trade days",
            f"{baseline.no_trade_days}",
            f"{filtered.no_trade_days}",
            _delta(baseline.no_trade_days, filtered.no_trade_days, "+d"),
        )
    )
    print(
        _fmt_row(
            "Trades filtered out",
            "—",
            f"{baseline.trade_days - filtered.trade_days}",
            f"{((baseline.trade_days - filtered.trade_days) / max(1, baseline.trade_days)) * 100:.0f}% removed",
        )
    )
    print()

    print(
        _fmt_row(
            "Winners",
            f"{baseline.winners}",
            f"{filtered.winners}",
            _delta(baseline.winners, filtered.winners, "+d"),
        )
    )
    print(
        _fmt_row(
            "Losers",
            f"{baseline.losers}",
            f"{filtered.losers}",
            _delta(baseline.losers, filtered.losers, "+d"),
        )
    )
    print(
        _fmt_row(
            "Win Rate",
            f"{baseline.win_rate:.1f}%",
            f"{filtered.win_rate:.1f}%",
            _delta(baseline.win_rate, filtered.win_rate, "+.1f", "%"),
        )
    )
    print()

    print(
        _fmt_row(
            "Total R",
            f"{baseline.total_r:+.2f}R",
            f"{filtered.total_r:+.2f}R",
            _delta(baseline.total_r, filtered.total_r, "+.2f") + "R",
        )
    )
    print(
        _fmt_row(
            "Avg R per trade",
            f"{baseline.avg_r:+.3f}R",
            f"{filtered.avg_r:+.3f}R",
            _delta(baseline.avg_r, filtered.avg_r, "+.3f") + "R",
        )
    )
    print(
        _fmt_row(
            "Profit Factor",
            f"{baseline.profit_factor:.2f}",
            f"{filtered.profit_factor:.2f}",
            _delta(baseline.profit_factor, filtered.profit_factor, "+.2f"),
        )
    )
    print(
        _fmt_row(
            "Max Drawdown",
            f"{baseline.max_drawdown_r:.2f}R",
            f"{filtered.max_drawdown_r:.2f}R",
            _delta(baseline.max_drawdown_r, filtered.max_drawdown_r, "+.2f") + "R",
        )
    )
    print()

    print(
        _fmt_row(
            "Avg Hold (bars)",
            f"{baseline.avg_hold_bars:.1f}",
            f"{filtered.avg_hold_bars:.1f}",
            _delta(baseline.avg_hold_bars, filtered.avg_hold_bars, "+.1f"),
        )
    )
    print(
        _fmt_row(
            "Avg Quality %",
            f"{baseline.avg_quality:.1f}",
            f"{filtered.avg_quality:.1f}",
            _delta(baseline.avg_quality, filtered.avg_quality, "+.1f"),
        )
    )
    print(
        _fmt_row(
            "Long / Short",
            f"{baseline.long_trades}L / {baseline.short_trades}S",
            f"{filtered.long_trades}L / {filtered.short_trades}S",
            "",
        )
    )
    print(
        _fmt_row(
            "NR7 trades (winners)",
            f"{baseline.nr7_trades} ({baseline.nr7_winners}W)",
            f"{filtered.nr7_trades} ({filtered.nr7_winners}W)",
            "",
        )
    )
    print(
        _fmt_row(
            "Max consec. wins",
            f"{baseline.max_consecutive_wins}",
            f"{filtered.max_consecutive_wins}",
            "",
        )
    )
    print(
        _fmt_row(
            "Max consec. losses",
            f"{baseline.max_consecutive_losses}",
            f"{filtered.max_consecutive_losses}",
            "",
        )
    )
    print("=" * 100)


def print_per_filter_breakdown(results: list[DayResult]) -> None:
    """Show how many trades each individual filter rejected."""
    trades = [r for r in results if r.has_trade and r.filter_result is not None]
    if not trades:
        return

    # Collect filter names from the first result that has verdicts
    filter_names: list[str] = []
    for r in trades:
        if r.filter_result and r.filter_result.verdicts:
            filter_names = [v.name for v in r.filter_result.verdicts]
            break

    if not filter_names:
        return

    print(f"\n{'=' * 70}")
    print("  PER-FILTER REJECTION BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"  {'Filter':<25s} {'Passed':>10s} {'Rejected':>10s} {'Reject %':>10s}")
    print("  " + "-" * 60)

    for fname in filter_names:
        passed = 0
        rejected = 0
        for r in trades:
            if r.filter_result:
                for v in r.filter_result.verdicts:
                    if v.name == fname:
                        if v.passed:
                            passed += 1
                        else:
                            rejected += 1
                        break

        total = passed + rejected
        pct = (rejected / total * 100) if total > 0 else 0.0
        print(f"  {fname:<25s} {passed:>10d} {rejected:>10d} {pct:>9.1f}%")

    print(f"{'=' * 70}")


def print_equity_curve(results: list[DayResult], label: str = "") -> None:
    """Print a simple text-based equity curve."""
    trades = [r for r in results if r.has_trade]
    if not trades:
        return

    print(f"\n  Equity Curve ({label}):")
    equity = 0.0
    peak = 0.0
    for r in trades:
        equity += r.pnl_r
        peak = max(peak, equity)
        bar_len = int(abs(equity) * 4)
        marker = "█" * min(bar_len, 60)
        sign = "+" if equity >= 0 else ""
        dd = peak - equity
        dd_str = f" (DD: {dd:.1f}R)" if dd > 0.5 else ""
        win_marker = "✓" if r.is_winner else "✗"
        print(f"    {r.day}  {win_marker}  {sign}{equity:>7.2f}R  {marker}{dd_str}")


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    symbols: list[str],
    source: str = "csv",
    days: int = 90,
    csv_dir: str = "data/bars",
    gate_mode: str = "all",
    bracket_config: Optional[BracketConfig] = None,
    verbose: bool = False,
    export_path: Optional[str] = None,
) -> dict[str, tuple[BacktestStats, BacktestStats]]:
    """Run the full filter backtest comparison for the given symbols.

    Returns:
        Dict mapping symbol → (baseline_stats, filtered_stats).
        Also includes a combined "ALL" key if multiple symbols.
    """
    cfg = bracket_config or BracketConfig()
    all_results: dict[str, list[DayResult]] = {}
    combined_results: list[DayResult] = []

    for symbol in symbols:
        print(f"\n📊 Loading bars for {symbol}...")

        bars_1m = _load_bars(symbol, source=source, days=days, csv_dir=csv_dir)
        if bars_1m is None or bars_1m.empty:
            print(f"  ⚠️  No data found for {symbol} — skipping")
            print(f"       Searched: source={source}, csv_dir={csv_dir}")
            print(f"       Expected CSV: {csv_dir}/{symbol}_1m.csv")
            continue

        print(f"  Loaded {len(bars_1m)} bars ({bars_1m.index.min()} → {bars_1m.index.max()})")

        # Load daily bars for NR7
        bars_daily = _load_daily_bars(symbol, source=source, csv_dir=csv_dir)
        if bars_daily is not None:
            print(f"  Daily bars: {len(bars_daily)} days (for NR7)")
        else:
            print(f"  No daily bars — NR7 will be skipped")

        # Split into per-day sessions
        sessions = split_into_sessions(bars_1m)
        print(f"  Sessions: {len(sessions)} trading days")

        if not sessions:
            print(f"  ⚠️  No valid sessions — skipping")
            continue

        # Run backtest for each day
        day_results: list[DayResult] = []
        for day_date, day_bars in sessions.items():
            # Derive daily bars up to this day for rolling NR7
            daily_slice = None
            if bars_daily is not None:
                try:
                    daily_slice = bars_daily[bars_daily.index.date <= day_date].tail(10)
                    if len(daily_slice) < 7:
                        daily_slice = None
                except Exception:
                    daily_slice = None

            dr = backtest_day(
                day_bars=day_bars,
                symbol=symbol,
                day_date=day_date,
                bracket_config=cfg,
                bars_daily=daily_slice,
                gate_mode=gate_mode,
            )
            day_results.append(dr)

            if verbose and dr.has_trade:
                status = "✅" if dr.is_winner else "❌"
                filt = "PASS" if dr.filter_passed else "REJECT"
                print(
                    f"    {day_date}  {status}  {dr.sim.direction:<5s}  "
                    f"{dr.pnl_r:+.2f}R  Q:{dr.sim.quality_pct}%  "
                    f"Filter: {filt}"
                )

        all_results[symbol] = day_results
        combined_results.extend(day_results)

        # Compute per-symbol stats
        baseline_results = [r for r in day_results if r.has_trade]
        filtered_results = [r for r in day_results if r.has_trade and r.filter_passed]

        baseline_stats = compute_stats(baseline_results, label=f"{symbol} BASELINE")
        filtered_stats = compute_stats(filtered_results, label=f"{symbol} FILTERED")

        print_comparison(baseline_stats, filtered_stats, symbol=symbol)
        print_per_filter_breakdown(day_results)

        if verbose:
            print_equity_curve(filtered_results, label=f"{symbol} FILTERED")

    # Combined stats (if multiple symbols)
    output: dict[str, tuple[BacktestStats, BacktestStats]] = {}

    for symbol, results in all_results.items():
        baseline = compute_stats([r for r in results if r.has_trade], f"{symbol} BASELINE")
        filtered = compute_stats([r for r in results if r.has_trade and r.filter_passed], f"{symbol} FILTERED")
        output[symbol] = (baseline, filtered)

    if len(symbols) > 1 and combined_results:
        baseline_all = compute_stats([r for r in combined_results if r.has_trade], "ALL BASELINE")
        filtered_all = compute_stats([r for r in combined_results if r.has_trade and r.filter_passed], "ALL FILTERED")
        output["ALL"] = (baseline_all, filtered_all)
        print_comparison(baseline_all, filtered_all, symbol="ALL SYMBOLS COMBINED")
        print_per_filter_breakdown(combined_results)

    # Export CSV if requested
    if export_path and combined_results:
        rows = [r.to_dict() for r in combined_results]
        df = pd.DataFrame(rows)
        df.to_csv(export_path, index=False)
        print(f"\n📁 Per-trade results exported to: {export_path}")

    # Final summary
    print(f"\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'=' * 100}")
    for sym, (bl, fl) in output.items():
        if bl.trade_days == 0:
            continue
        removed = bl.trade_days - fl.trade_days
        removed_pct = (removed / bl.trade_days * 100) if bl.trade_days > 0 else 0
        wr_delta = fl.win_rate - bl.win_rate
        pf_delta = fl.profit_factor - bl.profit_factor
        print(
            f"  {sym:<8s}  Trades: {bl.trade_days:>3d} → {fl.trade_days:>3d} "
            f"({removed_pct:>4.0f}% removed)  "
            f"WR: {bl.win_rate:>5.1f}% → {fl.win_rate:>5.1f}% ({wr_delta:>+5.1f}%)  "
            f"PF: {bl.profit_factor:>5.2f} → {fl.profit_factor:>5.2f} ({pf_delta:>+5.2f})  "
            f"R: {bl.total_r:>+6.2f} → {fl.total_r:>+6.2f}"
        )
    print(f"{'=' * 100}\n")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ORB Filter Backtest Comparison — measure filter impact on trade quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest MGC using CSV bars
  python scripts/backtest_filters.py --symbols MGC --source csv --csv-dir data/bars

  # Backtest multiple symbols via Redis cache
  python scripts/backtest_filters.py --symbols MGC MES MNQ --source cache

  # Use majority gate mode (more permissive)
  python scripts/backtest_filters.py --symbols MGC --source csv --gate-mode majority

  # Export detailed per-trade results
  python scripts/backtest_filters.py --symbols MGC --source csv --export results.csv

  # Verbose mode (show every trade)
  python scripts/backtest_filters.py --symbols MGC --source csv -v
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["MGC"],
        help="Symbols to backtest (default: MGC)",
    )
    parser.add_argument(
        "--source",
        choices=["csv", "cache", "massive"],
        default="csv",
        help="Bar data source (default: csv)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of history to load (default: 90)",
    )
    parser.add_argument(
        "--csv-dir",
        default="data/bars",
        help="Directory for CSV bar files (default: data/bars)",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["all", "majority"],
        default="all",
        help="Filter gate mode: 'all' = every hard filter must pass, 'majority' = >50%% must pass (default: all)",
    )
    parser.add_argument(
        "--sl-mult",
        type=float,
        default=1.5,
        help="Stop-loss ATR multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--tp1-mult",
        type=float,
        default=2.0,
        help="TP1 ATR multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export per-trade detail to CSV (e.g. --export results.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show each trade as it's processed",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    bracket_config = BracketConfig(
        sl_atr_mult=args.sl_mult,
        tp1_atr_mult=args.tp1_mult,
    )

    print("\n" + "=" * 100)
    print("  ORB FILTER BACKTEST COMPARISON")
    print("=" * 100)
    print(f"  Symbols:     {', '.join(args.symbols)}")
    print(f"  Source:      {args.source} (csv_dir={args.csv_dir})")
    print(f"  Days:        {args.days}")
    print(f"  Gate mode:   {args.gate_mode}")
    print(f"  Brackets:    SL={args.sl_mult}x ATR, TP1={args.tp1_mult}x ATR")
    if args.export:
        print(f"  Export:      {args.export}")
    print("=" * 100)

    run_backtest(
        symbols=args.symbols,
        source=args.source,
        days=args.days,
        csv_dir=args.csv_dir,
        gate_mode=args.gate_mode,
        bracket_config=bracket_config,
        verbose=args.verbose,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
