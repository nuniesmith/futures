#!/usr/bin/env python
"""
Walk-Forward Backtest
=====================
Time-series cross-validation of the full ORB pipeline (filters + CNN gate)
across all 9 Globex-day sessions.  Unlike a single-window backtest, this
script avoids look-ahead bias by:

  1. Splitting the historical bar data into chronological folds.
  2. On each fold: tuning the CNN threshold on the TRAIN window, then
     evaluating the filter gate + CNN gate on the out-of-sample TEST window.
  3. Aggregating per-fold metrics into a final walk-forward equity curve and
     summary table.

Two modes:
  - **Expanding window** (default): train window grows with each fold; the
    test window slides forward by one period.  Best for datasets where more
    history is always better.
  - **Rolling window**: fixed-size train window that also slides forward.
    Best for detecting regime changes (recent data weighted equally to older).

Session coverage:
  All 9 sessions are evaluated by default (--session all).  Each session uses
  its own per-session CNN threshold from SESSION_THRESHOLDS in breakout_cnn.py,
  which is further tuned on the fold's train window if --tune-threshold is set.

Threshold tuning (--tune-threshold):
  On each fold's TRAIN window, sweeps CNN thresholds from 0.60 to 0.95 in 0.01
  steps and picks the value that maximises profit factor on filtered trades.
  The tuned threshold is then applied to the TEST window for that fold.
  This gives a realistic estimate of how well per-session threshold tuning
  generalises out-of-sample.

Usage:
    cd futures

    # Expanding-window walk-forward, all sessions, DB bars:
    PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \\
        --symbols MGC MES MNQ \\
        --source db --days 180 \\
        --session all \\
        --folds 6

    # Rolling-window, London + US only, with threshold tuning:
    PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \\
        --symbols MGC 6E \\
        --source db --days 180 \\
        --session london us \\
        --mode rolling --train-days 60 --test-days 30 \\
        --tune-threshold \\
        --export wf_results.csv

    # Quick smoke-test (2 folds, US session only):
    PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \\
        --symbols MGC --source db --days 90 --session us --folds 2

Environment Variables:
    DATABASE_URL    Postgres DSN (if set, bars are loaded from Postgres)
    DB_PATH         SQLite path (fallback when DATABASE_URL is unset)
    PYTHONPATH      Must include the project src/ directory
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

# Import the per-day backtest machinery from backtest_filters
from scripts.backtest_filters import (
    _ALL_SESSION_KEYS,
    BacktestStats,
    DayResult,
    _load_bars,
    _load_daily_bars,
    _resolve_sessions,
    _session_bracket,
    backtest_day,
    compute_stats,
    split_into_sessions,
)

from lib.analysis.breakout_cnn import (
    DEFAULT_THRESHOLD,
    SESSION_THRESHOLDS,
    get_session_threshold,
)
from lib.analysis.orb_simulator import BracketConfig
from lib.services.engine.orb import (
    ORB_SESSIONS,
    SESSION_BY_KEY,
    ORBSession,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("walk_forward_backtest")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FoldSpec:
    """Date boundaries for a single walk-forward fold."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_index:02d}: "
            f"train [{self.train_start} → {self.train_end}] "
            f"test  [{self.test_start} → {self.test_end}]"
        )


@dataclass
class FoldResult:
    """Aggregated metrics for one fold, one session, one symbol."""

    fold_index: int
    session_key: str
    symbol: str

    # Tuned threshold (may equal the default if tuning was skipped)
    tuned_threshold: float = DEFAULT_THRESHOLD

    # Out-of-sample (test window) stats
    test_baseline: BacktestStats | None = None
    test_filtered: BacktestStats | None = None
    test_filtered_cnn: BacktestStats | None = None

    # Raw day results (test window, for equity curve construction)
    test_day_results: list[DayResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _st(s: BacktestStats | None, prefix: str) -> dict[str, Any]:
            if s is None:
                return {
                    f"{prefix}_trades": 0,
                    f"{prefix}_win_rate": 0.0,
                    f"{prefix}_profit_factor": 0.0,
                    f"{prefix}_total_r": 0.0,
                    f"{prefix}_avg_r": 0.0,
                    f"{prefix}_max_dd_r": 0.0,
                }
            return {
                f"{prefix}_trades": s.trade_days,
                f"{prefix}_win_rate": round(s.win_rate, 2),
                f"{prefix}_profit_factor": round(s.profit_factor, 3),
                f"{prefix}_total_r": round(s.total_r, 3),
                f"{prefix}_avg_r": round(s.avg_r, 3),
                f"{prefix}_max_dd_r": round(s.max_drawdown_r, 3),
            }

        d: dict[str, Any] = {
            "fold": self.fold_index,
            "session": self.session_key,
            "symbol": self.symbol,
            "tuned_threshold": round(self.tuned_threshold, 3),
        }
        d.update(_st(self.test_baseline, "base"))
        d.update(_st(self.test_filtered, "filt"))
        d.update(_st(self.test_filtered_cnn, "cnn"))
        return d


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def _generate_folds_expanding(
    all_dates: list[date],
    n_folds: int,
    min_train_days: int = 20,
    test_days: int = 30,
) -> list[FoldSpec]:
    """Generate expanding-window fold specs.

    The test window always has *test_days* calendar days.  The train window
    starts at the beginning of the dataset and grows by one test period per fold.

    Args:
        all_dates: Sorted list of calendar dates with bar data.
        n_folds: Number of folds to produce.
        min_train_days: Minimum calendar days required in the train window.
        test_days: Calendar days per test window.

    Returns:
        List of FoldSpec objects in chronological order.
    """
    if not all_dates:
        return []

    total_span = (all_dates[-1] - all_dates[0]).days
    required_span = min_train_days + n_folds * test_days
    if total_span < required_span:
        logger.warning(
            "Date span %d days is less than required %d days for %d folds — reducing fold count",
            total_span,
            required_span,
            n_folds,
        )
        n_folds = max(1, (total_span - min_train_days) // test_days)

    folds: list[FoldSpec] = []
    data_start = all_dates[0]

    for i in range(n_folds):
        test_end_offset = (n_folds - i) * test_days
        test_start_offset = test_end_offset - test_days

        test_start = all_dates[-1] - timedelta(days=test_end_offset)
        test_end = all_dates[-1] - timedelta(days=test_start_offset)

        if test_start <= data_start + timedelta(days=min_train_days):
            break

        folds.append(
            FoldSpec(
                fold_index=i + 1,
                train_start=data_start,
                train_end=test_start - timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
            )
        )

    return sorted(folds, key=lambda f: f.fold_index)


def _generate_folds_rolling(
    all_dates: list[date],
    train_days: int = 60,
    test_days: int = 30,
) -> list[FoldSpec]:
    """Generate rolling-window fold specs.

    Both train and test windows have fixed sizes and slide forward together.
    Produces as many non-overlapping folds as the date range allows.

    Args:
        all_dates: Sorted list of calendar dates with bar data.
        train_days: Fixed calendar days in each train window.
        test_days: Fixed calendar days in each test window.

    Returns:
        List of FoldSpec objects in chronological order.
    """
    if not all_dates:
        return []

    folds: list[FoldSpec] = []
    step = test_days
    fold_index = 1
    data_start = all_dates[0]
    data_end = all_dates[-1]

    cursor = data_start
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        if test_end > data_end:
            break

        folds.append(
            FoldSpec(
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_index += 1
        cursor += timedelta(days=step)

    return folds


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------


def _tune_cnn_threshold(
    train_results: list[DayResult],
    session_key: str,
    sweep_start: float = 0.60,
    sweep_end: float = 0.95,
    sweep_step: float = 0.01,
) -> float:
    """Find the CNN threshold that maximises profit factor on *train_results*.

    Sweeps *sweep_start* → *sweep_end* in *sweep_step* increments.  For each
    candidate threshold we compute the profit factor of trades that pass both
    the deterministic filter gate AND the CNN gate at that threshold.

    If no threshold produces at least 5 trades, falls back to the default
    per-session threshold from SESSION_THRESHOLDS.

    Args:
        train_results: List of DayResult objects from the training window.
        session_key: ORBSession.key — used for the fallback default.
        sweep_start: Lower bound of the threshold sweep.
        sweep_end: Upper bound of the threshold sweep.
        sweep_step: Step size for the sweep.

    Returns:
        Best threshold float, or the session default if tuning is not possible.
    """
    # Only use results that have a CNN probability recorded
    cnn_results = [r for r in train_results if r.has_trade and r.filter_passed and r.cnn_prob is not None]
    if len(cnn_results) < 10:
        default = get_session_threshold(session_key)
        logger.debug(
            "Threshold tuning skipped for session '%s' — only %d results with CNN prob (need ≥10). Using default %.3f.",
            session_key,
            len(cnn_results),
            default,
        )
        return default

    best_threshold = get_session_threshold(session_key)
    best_pf = 0.0
    best_n_trades = 0

    thresh = sweep_start
    while thresh <= sweep_end + 1e-9:
        candidates = [r for r in cnn_results if r.cnn_prob is not None and r.cnn_prob >= thresh]
        if len(candidates) < 5:
            thresh = round(thresh + sweep_step, 3)
            continue
        stats = compute_stats(candidates, label="tune")
        pf = stats.profit_factor
        # Prefer higher profit factor; break ties by more trades
        if pf > best_pf or (abs(pf - best_pf) < 0.001 and len(candidates) > best_n_trades):
            best_pf = pf
            best_threshold = thresh
            best_n_trades = len(candidates)
        thresh = round(thresh + sweep_step, 3)

    logger.debug(
        "Threshold tuned for session '%s': %.3f → PF=%.3f (%d train trades)",
        session_key,
        best_threshold,
        best_pf,
        best_n_trades,
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------


def _slice_by_dates(
    sessions_map: dict[date, pd.DataFrame],
    start: date,
    end: date,
) -> dict[date, pd.DataFrame]:
    """Return the subset of *sessions_map* whose date falls within [start, end]."""
    return {d: df for d, df in sessions_map.items() if start <= d <= end}


def _run_fold_session(
    fold: FoldSpec,
    symbol: str,
    sessions_map: dict[date, pd.DataFrame],
    bars_daily: pd.DataFrame | None,
    orb_session: ORBSession,
    base_cfg: BracketConfig,
    gate_mode: str,
    tune_threshold: bool,
    cnn_gate: bool,
    fixed_cnn_threshold: float | None,
) -> FoldResult:
    """Evaluate one fold for one session for one symbol.

    Runs:
      1. (Optional) Threshold tuning on the train window.
      2. Backtest evaluation on the test window.

    Returns a FoldResult.
    """
    session_key = orb_session.key
    result = FoldResult(
        fold_index=fold.fold_index,
        session_key=session_key,
        symbol=symbol,
        tuned_threshold=fixed_cnn_threshold or get_session_threshold(session_key),
    )

    session_cfg = _session_bracket(orb_session, base_cfg)

    # --- Step 1: Optional threshold tuning on train window ---
    if tune_threshold and cnn_gate and fixed_cnn_threshold is None:
        train_days_map = _slice_by_dates(sessions_map, fold.train_start, fold.train_end)
        train_results: list[DayResult] = []

        for day_date, day_bars in train_days_map.items():
            daily_slice = _daily_slice(bars_daily, day_date)
            dr = backtest_day(
                day_bars=day_bars,
                symbol=symbol,
                day_date=day_date,
                bracket_config=session_cfg,
                bars_daily=daily_slice,
                gate_mode=gate_mode,
                orb_session=orb_session,
                cnn_gate=cnn_gate,
            )
            train_results.append(dr)

        result.tuned_threshold = _tune_cnn_threshold(train_results, session_key)

    # --- Step 2: Out-of-sample evaluation on test window ---
    test_days_map = _slice_by_dates(sessions_map, fold.test_start, fold.test_end)
    test_results: list[DayResult] = []

    effective_threshold = result.tuned_threshold

    for day_date, day_bars in test_days_map.items():
        daily_slice = _daily_slice(bars_daily, day_date)
        dr = backtest_day(
            day_bars=day_bars,
            symbol=symbol,
            day_date=day_date,
            bracket_config=session_cfg,
            bars_daily=daily_slice,
            gate_mode=gate_mode,
            orb_session=orb_session,
            cnn_gate=cnn_gate,
            cnn_threshold_override=effective_threshold if cnn_gate else None,
        )
        test_results.append(dr)

    result.test_day_results = test_results

    trades = [r for r in test_results if r.has_trade]
    filtered = [r for r in trades if r.filter_passed]
    cnn_filtered = [r for r in filtered if r.filter_and_cnn_passed]

    result.test_baseline = compute_stats(trades, label=f"fold{fold.fold_index}/{session_key}/base")
    result.test_filtered = compute_stats(filtered, label=f"fold{fold.fold_index}/{session_key}/filt")
    result.test_filtered_cnn = (
        compute_stats(cnn_filtered, label=f"fold{fold.fold_index}/{session_key}/cnn") if cnn_gate else None
    )

    return result


def _daily_slice(
    bars_daily: pd.DataFrame | None,
    day_date: date,
    n: int = 10,
    min_n: int = 7,
) -> pd.DataFrame | None:
    """Return the last *n* rows of *bars_daily* up to and including *day_date*."""
    if bars_daily is None:
        return None
    try:
        dti = pd.DatetimeIndex(bars_daily.index)
        sliced = bars_daily[dti.date <= day_date].tail(n)
        return sliced if len(sliced) >= min_n else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main walk-forward runner
# ---------------------------------------------------------------------------


def run_walk_forward(
    symbols: list[str],
    source: str = "db",
    days: int = 180,
    csv_dir: str = "data/bars",
    orb_sessions: list[ORBSession] | None = None,
    n_folds: int = 6,
    mode: str = "expanding",
    train_days: int = 60,
    test_days: int = 30,
    gate_mode: str = "majority",
    bracket_config: BracketConfig | None = None,
    cnn_gate: bool = False,
    tune_threshold: bool = False,
    fixed_cnn_threshold: float | None = None,
    verbose: bool = False,
    export_path: str | None = None,
) -> list[FoldResult]:
    """Run the full walk-forward backtest.

    Args:
        symbols: Instrument symbols to test.
        source: Bar data source — "db", "csv", "cache", or "massive".
        days: Total days of history to load.
        csv_dir: CSV bar directory (only used when source="csv").
        orb_sessions: ORBSession objects to evaluate. Defaults to [US_SESSION].
        n_folds: Number of folds for expanding-window mode.
        mode: "expanding" or "rolling".
        train_days: Fixed train window size (rolling mode) or minimum initial
                    train window (expanding mode).
        test_days: Test window size in calendar days.
        gate_mode: Filter gate mode — "all" or "majority".
        bracket_config: Base BracketConfig (OR times overridden per session).
        cnn_gate: Enable CNN gate evaluation.
        tune_threshold: Tune CNN threshold on each fold's train window.
        fixed_cnn_threshold: Fixed override for all sessions (disables tuning).
        verbose: Print per-trade detail.
        export_path: Write per-trade CSV to this path.

    Returns:
        List of FoldResult objects (one per fold × session × symbol).
    """
    base_cfg = bracket_config or BracketConfig()
    sessions_to_run = orb_sessions or [SESSION_BY_KEY["us"]]
    all_fold_results: list[FoldResult] = []

    for symbol in symbols:
        print(f"\n{'─' * 80}")
        print(f"  Loading bars: {symbol} (source={source}, days={days})")
        print(f"{'─' * 80}")

        bars_1m = _load_bars(symbol, source=source, days=days, csv_dir=csv_dir)
        if bars_1m is None or bars_1m.empty:
            print(f"  ⚠️  No bars found for {symbol} — skipping")
            continue

        bars_daily = _load_daily_bars(symbol, source=source, csv_dir=csv_dir)
        print(f"  1-min bars: {len(bars_1m)}  ({bars_1m.index.min()} → {bars_1m.index.max()})")
        if bars_daily is not None:
            print(f"  Daily bars: {len(bars_daily)} (for NR7)")

        sessions_map = split_into_sessions(bars_1m)
        if not sessions_map:
            print(f"  ⚠️  No valid trading days for {symbol} — skipping")
            continue

        all_dates = sorted(sessions_map.keys())
        print(f"  Trading days available: {len(all_dates)}")

        # Build fold specs
        if mode == "rolling":
            folds = _generate_folds_rolling(all_dates, train_days=train_days, test_days=test_days)
        else:
            folds = _generate_folds_expanding(
                all_dates,
                n_folds=n_folds,
                min_train_days=train_days,
                test_days=test_days,
            )

        if not folds:
            print(f"  ⚠️  Not enough data to generate folds for {symbol} — skipping")
            continue

        print(f"  Folds: {len(folds)} ({mode})")
        for f in folds:
            print(f"    {f}")

        for orb_session in sessions_to_run:
            session_key = orb_session.key
            print(f"\n  ── Session: {orb_session.name} [{session_key}] ──")

            symbol_session_results: list[FoldResult] = []

            for fold in folds:
                fr = _run_fold_session(
                    fold=fold,
                    symbol=symbol,
                    sessions_map=sessions_map,
                    bars_daily=bars_daily,
                    orb_session=orb_session,
                    base_cfg=base_cfg,
                    gate_mode=gate_mode,
                    tune_threshold=tune_threshold,
                    cnn_gate=cnn_gate,
                    fixed_cnn_threshold=fixed_cnn_threshold,
                )
                symbol_session_results.append(fr)
                all_fold_results.append(fr)

                if verbose:
                    _print_fold_summary(fr, cnn_gate=cnn_gate)

            # Print the per-session walk-forward summary
            _print_walk_forward_table(symbol_session_results, symbol=symbol, session_key=session_key, cnn_gate=cnn_gate)

    # Aggregate across all sessions / symbols
    _print_aggregate_summary(all_fold_results, sessions_to_run, cnn_gate=cnn_gate)

    # Export
    if export_path and all_fold_results:
        _export_results(all_fold_results, export_path)

    return all_fold_results


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_fold_summary(fr: FoldResult, cnn_gate: bool = False) -> None:
    """Print a one-line summary for a single fold result."""
    bl = fr.test_baseline
    fl = fr.test_filtered
    cn = fr.test_filtered_cnn

    bl_str = (
        f"{bl.trade_days:>3d}T  WR:{bl.win_rate:>5.1f}%  PF:{bl.profit_factor:>5.2f}  R:{bl.total_r:>+6.2f}"
        if bl
        else "  no trades"
    )
    fl_str = (
        f"{fl.trade_days:>3d}T  WR:{fl.win_rate:>5.1f}%  PF:{fl.profit_factor:>5.2f}  R:{fl.total_r:>+6.2f}"
        if fl
        else "  no trades"
    )
    cn_str = (
        f"  CNN→ {cn.trade_days:>3d}T  WR:{cn.win_rate:>5.1f}%  PF:{cn.profit_factor:>5.2f}  R:{cn.total_r:>+6.2f}"
        if (cnn_gate and cn)
        else ""
    )

    print(
        f"    Fold {fr.fold_index:02d} [{fr.session_key:>10s}]  "
        f"BASE: {bl_str}  |  FILT: {fl_str}{cn_str}" + (f"  thresh={fr.tuned_threshold:.3f}" if cnn_gate else "")
    )


def _print_walk_forward_table(
    fold_results: list[FoldResult],
    symbol: str,
    session_key: str,
    cnn_gate: bool = False,
) -> None:
    """Print the fold-by-fold walk-forward table for one symbol/session."""
    if not fold_results:
        return

    w = 110
    print(f"\n  {'═' * w}")
    print(f"  Walk-Forward Results — {symbol} / {session_key}")
    print(f"  {'═' * w}")

    hdr = f"  {'Fold':>4}  {'Trades':>6}  {'Filt':>6}  {'WR%':>6}  {'PF':>6}  {'TotalR':>8}  {'AvgR':>7}  {'MaxDD':>7}"
    if cnn_gate:
        hdr += f"  {'CNN_T':>5}  {'CNN_N':>5}  {'CNN_WR':>7}  {'CNN_PF':>7}  {'CNN_R':>8}"
    print(hdr)
    print(f"  {'─' * (w - 2)}")

    cumulative_r_filt = 0.0
    cumulative_r_cnn = 0.0

    for fr in fold_results:
        bl = fr.test_baseline
        fl = fr.test_filtered
        cn = fr.test_filtered_cnn

        n_base = bl.trade_days if bl else 0
        n_filt = fl.trade_days if fl else 0
        wr = fl.win_rate if fl else 0.0
        pf = fl.profit_factor if fl else 0.0
        tr = fl.total_r if fl else 0.0
        ar = fl.avg_r if fl else 0.0
        dd = fl.max_drawdown_r if fl else 0.0
        cumulative_r_filt += tr

        row = (
            f"  {fr.fold_index:>4d}  {n_base:>6d}  {n_filt:>6d}  "
            f"{wr:>6.1f}  {pf:>6.2f}  {tr:>+8.3f}  {ar:>+7.3f}  {dd:>+7.3f}"
        )
        if cnn_gate:
            n_cnn = cn.trade_days if cn else 0
            cnn_wr = cn.win_rate if cn else 0.0
            cnn_pf = cn.profit_factor if cn else 0.0
            cnn_tr = cn.total_r if cn else 0.0
            cumulative_r_cnn += cnn_tr
            row += f"  {fr.tuned_threshold:>5.3f}  {n_cnn:>5d}  {cnn_wr:>7.1f}  {cnn_pf:>7.2f}  {cnn_tr:>+8.3f}"
        print(row)

    # Totals row
    print(f"  {'─' * (w - 2)}")
    all_filt = [r for fr in fold_results for r in fr.test_day_results if r.has_trade and r.filter_passed]
    all_cnn = [r for fr in fold_results for r in fr.test_day_results if r.has_trade and r.filter_and_cnn_passed]

    if all_filt:
        tot = compute_stats(all_filt, label="TOTAL")
        tot_row = (
            f"  {'ALL':>4}  {'':>6}  {tot.trade_days:>6d}  "
            f"{tot.win_rate:>6.1f}  {tot.profit_factor:>6.2f}  "
            f"{tot.total_r:>+8.3f}  {tot.avg_r:>+7.3f}  {tot.max_drawdown_r:>+7.3f}"
        )
        if cnn_gate and all_cnn:
            tot_cnn = compute_stats(all_cnn, label="CNN_TOTAL")
            tot_row += (
                f"  {'':>5}  {tot_cnn.trade_days:>5d}  "
                f"{tot_cnn.win_rate:>7.1f}  {tot_cnn.profit_factor:>7.2f}  "
                f"{tot_cnn.total_r:>+8.3f}"
            )
        print(tot_row)

    print(f"  {'═' * w}\n")


def _print_aggregate_summary(
    all_fold_results: list[FoldResult],
    sessions: list[ORBSession],
    cnn_gate: bool = False,
) -> None:
    """Print the final aggregate summary across all symbols, sessions, and folds."""
    if not all_fold_results:
        print("\n  No results to summarise.")
        return

    print(f"\n{'═' * 110}")
    print("  WALK-FORWARD AGGREGATE SUMMARY")
    print(f"{'═' * 110}")
    print(
        f"  {'Session':<12}  {'Symbol':<8}  "
        f"{'Folds':>5}  {'Trades':>6}  {'Filt':>6}  "
        f"{'WR%':>6}  {'PF':>6}  {'TotalR':>8}  {'AvgR':>7}  {'MaxDD':>7}"
        + ("  {'CNN_WR':>7}  {'CNN_PF':>7}  {'CNN_R':>8}" if cnn_gate else "")
    )
    print(f"  {'─' * 106}")

    # Group by (session_key, symbol)
    grouped: dict[tuple[str, str], list[FoldResult]] = defaultdict(list)
    for fr in all_fold_results:
        grouped[(fr.session_key, fr.symbol)].append(fr)

    grand_filt: list[DayResult] = []
    grand_cnn: list[DayResult] = []

    for (session_key, symbol), frs in sorted(grouped.items()):
        filt_all = [r for fr in frs for r in fr.test_day_results if r.has_trade and r.filter_passed]
        cnn_all = [r for fr in frs for r in fr.test_day_results if r.has_trade and r.filter_and_cnn_passed]
        base_all = [r for fr in frs for r in fr.test_day_results if r.has_trade]

        grand_filt.extend(filt_all)
        grand_cnn.extend(cnn_all)

        if not filt_all:
            print(f"  {session_key:<12}  {symbol:<8}  {len(frs):>5d}  — no filtered trades —")
            continue

        st = compute_stats(filt_all, label=f"{session_key}/{symbol}")
        row = (
            f"  {session_key:<12}  {symbol:<8}  "
            f"{len(frs):>5d}  {len(base_all):>6d}  {len(filt_all):>6d}  "
            f"{st.win_rate:>6.1f}  {st.profit_factor:>6.2f}  "
            f"{st.total_r:>+8.3f}  {st.avg_r:>+7.3f}  {st.max_drawdown_r:>+7.3f}"
        )
        if cnn_gate and cnn_all:
            cnn_st = compute_stats(cnn_all, label=f"{session_key}/{symbol}/cnn")
            row += f"  {cnn_st.win_rate:>7.1f}  {cnn_st.profit_factor:>7.2f}  {cnn_st.total_r:>+8.3f}"
        print(row)

    # Grand total
    if grand_filt:
        print(f"  {'─' * 106}")
        gt = compute_stats(grand_filt, label="GRAND_TOTAL")
        grand_row = (
            f"  {'ALL':<12}  {'ALL':<8}  "
            f"{'':>5}  {'':>6}  {len(grand_filt):>6d}  "
            f"{gt.win_rate:>6.1f}  {gt.profit_factor:>6.2f}  "
            f"{gt.total_r:>+8.3f}  {gt.avg_r:>+7.3f}  {gt.max_drawdown_r:>+7.3f}"
        )
        if cnn_gate and grand_cnn:
            gt_cnn = compute_stats(grand_cnn, label="GRAND_CNN")
            grand_row += f"  {gt_cnn.win_rate:>7.1f}  {gt_cnn.profit_factor:>7.2f}  {gt_cnn.total_r:>+8.3f}"
        print(grand_row)

    print(f"{'═' * 110}\n")

    # Per-session CNN threshold summary
    if cnn_gate:
        print("  Per-session tuned thresholds (average across folds):")
        thresh_by_session: dict[str, list[float]] = defaultdict(list)
        for fr in all_fold_results:
            thresh_by_session[fr.session_key].append(fr.tuned_threshold)
        for skey in _ALL_SESSION_KEYS:
            vals = thresh_by_session.get(skey, [])
            if vals:
                avg = sum(vals) / len(vals)
                default = get_session_threshold(skey)
                delta = avg - default
                print(f"    {skey:<12}  avg={avg:.3f}  default={default:.3f}  delta={delta:+.3f}  folds={len(vals)}")
        print()


def _export_results(fold_results: list[FoldResult], path: str) -> None:
    """Write per-trade detail for all folds to a CSV file."""
    rows: list[dict[str, Any]] = []
    for fr in fold_results:
        for dr in fr.test_day_results:
            if not dr.has_trade:
                continue
            row = dr.to_dict()
            row["fold"] = fr.fold_index
            row["tuned_threshold"] = fr.tuned_threshold
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"\n📁 Per-trade walk-forward results exported to: {path}")
    else:
        print("\n⚠️  No trades to export.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SESSION_CHOICES = ["all", "both"] + _ALL_SESSION_KEYS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward backtest: time-series CV across all 9 Globex sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["MGC"],
        metavar="SYM",
        help="Symbols to test (default: MGC)",
    )
    p.add_argument(
        "--source",
        choices=["db", "csv", "cache", "massive"],
        default="db",
        help="Bar data source (default: db)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=180,
        help="Total days of history to load (default: 180)",
    )
    p.add_argument(
        "--csv-dir",
        default="data/bars",
        metavar="DIR",
        help="CSV bar directory (only used with --source csv, default: data/bars)",
    )

    # Session
    p.add_argument(
        "--session",
        nargs="+",
        choices=_SESSION_CHOICES,
        default=["us"],
        metavar="SESSION",
        help=(
            f"ORB session(s) to evaluate. Choices: {_SESSION_CHOICES}. "
            "Use 'all' for all 9 sessions. Multiple values accepted. "
            "Default: us."
        ),
    )

    # Fold configuration
    p.add_argument(
        "--mode",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Walk-forward mode: 'expanding' (growing train window) or 'rolling' (fixed). Default: expanding.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=6,
        metavar="N",
        help="Number of folds (expanding mode only, default: 6)",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=60,
        metavar="N",
        help=("Minimum train window in calendar days (expanding) or fixed train window size (rolling). Default: 60."),
    )
    p.add_argument(
        "--test-days",
        type=int,
        default=30,
        metavar="N",
        help="Test window size in calendar days (default: 30)",
    )

    # Gate / bracket
    p.add_argument(
        "--gate-mode",
        choices=["all", "majority"],
        default="majority",
        help="Filter gate mode (default: majority)",
    )
    p.add_argument(
        "--sl-mult",
        type=float,
        default=1.5,
        metavar="X",
        help="Stop-loss ATR multiplier (default: 1.5)",
    )
    p.add_argument(
        "--tp1-mult",
        type=float,
        default=2.0,
        metavar="X",
        help="TP1 ATR multiplier (default: 2.0)",
    )

    # CNN gate
    p.add_argument(
        "--cnn-gate",
        action="store_true",
        help="Enable CNN gate evaluation using per-session thresholds",
    )
    p.add_argument(
        "--tune-threshold",
        action="store_true",
        help=(
            "Tune the CNN threshold on each fold's train window before "
            "evaluating on the test window (implies --cnn-gate)"
        ),
    )
    p.add_argument(
        "--cnn-threshold",
        type=float,
        default=None,
        metavar="PROB",
        help="Fixed CNN threshold override for all sessions (disables per-session defaults and tuning)",
    )

    # Output
    p.add_argument(
        "--export",
        default=None,
        metavar="PATH",
        help="Export per-trade walk-forward results to CSV",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-fold trade-level detail",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Resolve sessions — handle "all" and multiple values
    session_keys: list[str] = []
    for s in args.session:
        if s == "all":
            session_keys = _ALL_SESSION_KEYS[:]
            break
        elif s == "both":
            for k in ("london", "us"):
                if k not in session_keys:
                    session_keys.append(k)
        elif s not in session_keys:
            session_keys.append(s)

    orb_sessions = [SESSION_BY_KEY[k] for k in session_keys if k in SESSION_BY_KEY]
    if not orb_sessions:
        print("ERROR: no valid sessions resolved from --session args", file=sys.stderr)
        return 1

    cnn_gate = args.cnn_gate or args.tune_threshold

    bracket_config = BracketConfig(
        sl_atr_mult=args.sl_mult,
        tp1_atr_mult=args.tp1_mult,
    )

    # Print header
    sep = "═" * 110
    print(f"\n{sep}")
    print("  WALK-FORWARD BACKTEST")
    print(sep)
    print(f"  Symbols:       {', '.join(args.symbols)}")
    print(f"  Source:        {args.source}" + (f" ({args.csv_dir})" if args.source == "csv" else ""))
    print(f"  History:       {args.days} days")
    print(f"  Sessions:      {', '.join(s.key for s in orb_sessions)}")
    print(f"  Mode:          {args.mode}")
    if args.mode == "expanding":
        print(f"  Folds:         {args.folds}  (test_days={args.test_days}, min_train={args.train_days})")
    else:
        print(f"  Windows:       train={args.train_days}d  test={args.test_days}d (rolling)")
    print(f"  Gate mode:     {args.gate_mode}")
    print(f"  Brackets:      SL={args.sl_mult}x ATR, TP1={args.tp1_mult}x ATR")
    if cnn_gate:
        if args.cnn_threshold is not None:
            print(f"  CNN gate:      enabled  threshold={args.cnn_threshold:.3f} (fixed override)")
        elif args.tune_threshold:
            print("  CNN gate:      enabled  threshold=tuned-per-fold")
        else:
            print(
                "  CNN gate:      enabled  thresholds=" + ", ".join(f"{k}:{v}" for k, v in SESSION_THRESHOLDS.items())
            )
    else:
        print("  CNN gate:      disabled")
    if args.export:
        print(f"  Export:        {args.export}")
    print(sep)

    results = run_walk_forward(
        symbols=args.symbols,
        source=args.source,
        days=args.days,
        csv_dir=args.csv_dir,
        orb_sessions=orb_sessions,
        n_folds=args.folds,
        mode=args.mode,
        train_days=args.train_days,
        test_days=args.test_days,
        gate_mode=args.gate_mode,
        bracket_config=bracket_config,
        cnn_gate=cnn_gate,
        tune_threshold=args.tune_threshold,
        fixed_cnn_threshold=args.cnn_threshold,
        verbose=args.verbose,
        export_path=args.export,
    )

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
