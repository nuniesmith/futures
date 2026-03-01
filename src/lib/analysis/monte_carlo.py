"""
Monte Carlo simulation for backtesting robustness testing.

Implements trade-level bootstrap Monte Carlo (the gold standard per todo.md)
to stress-test strategy equity curves and estimate realistic drawdown
distributions. Also provides Probability of Backtest Overfitting (PBO)
estimation via combinatorial cross-validation.

Key features:
  - Trade-level bootstrap: resample N trades with replacement, build
    synthetic equity curves, repeat 10,000 times
  - Percentile bands: 5th, 25th, 50th, 75th, 95th equity confidence cones
  - Drawdown distribution: 95th-percentile max drawdown for risk planning
  - Live performance tracking: check if live results fall within MC bands
  - PBO estimation: partition time series, enumerate train/test splits,
    compute fraction where best IS config underperforms median OOS

Usage:
    from lib.monte_carlo import (
        run_monte_carlo,
        compute_confidence_cones,
        estimate_pbo,
        check_live_performance,
        mc_results_to_dataframe,
    )

    # Run Monte Carlo on backtest trade results
    trades = [10.5, -5.2, 20.1, -8.0, 15.3, ...]  # per-trade P&L list
    mc = run_monte_carlo(trades, n_simulations=10000, initial_equity=150000)
    cones = compute_confidence_cones(mc)

    # Check if live performance is within expected bands
    live_equity = [150000, 150500, 149800, 151200, ...]
    status = check_live_performance(live_equity, cones)

    # Estimate PBO
    pbo = estimate_pbo(equity_series, n_partitions=10)
"""

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("monte_carlo")


# ---------------------------------------------------------------------------
# Monte Carlo simulation (trade-level bootstrap)
# ---------------------------------------------------------------------------


def run_monte_carlo(
    trade_pnls: list[float],
    n_simulations: int = 10000,
    initial_equity: float = 150000.0,
    seed: Optional[int] = 42,
) -> dict[str, Any]:
    """Run Monte Carlo simulation via trade-level bootstrap.

    Samples N trades with replacement from backtest results, builds a
    new equity curve, and repeats `n_simulations` times. This is the
    gold standard for robustness testing per the todo.md blueprint.

    Args:
        trade_pnls: List of per-trade P&L values from backtesting.
        n_simulations: Number of Monte Carlo iterations (default 10,000).
        initial_equity: Starting account equity.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
          - equity_curves: np.ndarray of shape (n_simulations, n_trades+1)
          - final_equities: np.ndarray of shape (n_simulations,)
          - max_drawdowns: np.ndarray of shape (n_simulations,)
          - max_drawdown_pcts: np.ndarray of shape (n_simulations,)
          - sharpe_ratios: np.ndarray of shape (n_simulations,)
          - win_rates: np.ndarray of shape (n_simulations,)
          - n_trades: int
          - n_simulations: int
          - initial_equity: float
          - original_pnls: list[float]

    Caveat: Monte Carlo assumes trade independence. If the strategy
    produces correlated win/loss streaks (trending strategies), MC can
    be overly optimistic about drawdown distributions.
    """
    if not trade_pnls or len(trade_pnls) < 3:
        return _empty_mc_result(initial_equity)

    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls, dtype=np.float64)
    n_trades = len(pnls)

    # Pre-allocate arrays
    equity_curves = np.zeros((n_simulations, n_trades + 1), dtype=np.float64)
    equity_curves[:, 0] = initial_equity

    final_equities = np.zeros(n_simulations, dtype=np.float64)
    max_drawdowns = np.zeros(n_simulations, dtype=np.float64)
    max_drawdown_pcts = np.zeros(n_simulations, dtype=np.float64)
    sharpe_ratios = np.zeros(n_simulations, dtype=np.float64)
    win_rates = np.zeros(n_simulations, dtype=np.float64)

    for i in range(n_simulations):
        # Sample trades with replacement
        sampled_indices = rng.integers(0, n_trades, size=n_trades)
        sampled_pnls = pnls[sampled_indices]

        # Build equity curve
        cumulative = np.cumsum(sampled_pnls)
        equity_curves[i, 1:] = initial_equity + cumulative

        # Final equity
        final_equities[i] = equity_curves[i, -1]

        # Max drawdown (absolute)
        running_max = np.maximum.accumulate(equity_curves[i])
        drawdowns = running_max - equity_curves[i]
        max_drawdowns[i] = np.max(drawdowns)

        # Max drawdown (percentage)
        dd_pct = drawdowns / np.maximum(running_max, 1e-10) * 100
        max_drawdown_pcts[i] = np.max(dd_pct)

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(sampled_pnls) > 1:
            mean_pnl = np.mean(sampled_pnls)
            std_pnl = np.std(sampled_pnls, ddof=1)
            if std_pnl > 0:
                # Daily Sharpe × sqrt(252) — but we don't know trades/day
                # Use per-trade Sharpe × sqrt(n_trades) as proxy
                sharpe_ratios[i] = (mean_pnl / std_pnl) * math.sqrt(min(n_trades, 252))
            else:
                sharpe_ratios[i] = 0.0
        else:
            sharpe_ratios[i] = 0.0

        # Win rate
        wins = np.sum(sampled_pnls > 0)
        win_rates[i] = wins / n_trades * 100 if n_trades > 0 else 0.0

    return {
        "equity_curves": equity_curves,
        "final_equities": final_equities,
        "max_drawdowns": max_drawdowns,
        "max_drawdown_pcts": max_drawdown_pcts,
        "sharpe_ratios": sharpe_ratios,
        "win_rates": win_rates,
        "n_trades": n_trades,
        "n_simulations": n_simulations,
        "initial_equity": initial_equity,
        "original_pnls": list(trade_pnls),
    }


def _empty_mc_result(initial_equity: float) -> dict[str, Any]:
    """Return an empty MC result when there aren't enough trades."""
    return {
        "equity_curves": np.array([[initial_equity]]),
        "final_equities": np.array([initial_equity]),
        "max_drawdowns": np.array([0.0]),
        "max_drawdown_pcts": np.array([0.0]),
        "sharpe_ratios": np.array([0.0]),
        "win_rates": np.array([0.0]),
        "n_trades": 0,
        "n_simulations": 0,
        "initial_equity": initial_equity,
        "original_pnls": [],
    }


# ---------------------------------------------------------------------------
# Confidence cones (percentile bands)
# ---------------------------------------------------------------------------


def compute_confidence_cones(
    mc_result: dict[str, Any],
    percentiles: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Extract percentile bands from Monte Carlo equity curves.

    Produces equity confidence cones at each trade step for visualization.
    The 95th-percentile max drawdown becomes the risk planning figure.

    Args:
        mc_result: Output from run_monte_carlo().
        percentiles: List of percentiles to compute (default: 5, 25, 50, 75, 95).

    Returns:
        Dict with:
          - percentile_curves: dict mapping percentile → equity array
          - drawdown_percentiles: dict mapping percentile → max drawdown value
          - summary: dict with key statistics
    """
    if percentiles is None:
        percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]

    curves = mc_result["equity_curves"]
    n_steps = curves.shape[1]

    # Equity percentile curves (at each trade step)
    percentile_curves = {}
    for p in percentiles:
        percentile_curves[p] = np.percentile(curves, p, axis=0)

    # Drawdown distribution percentiles
    dd_abs = mc_result["max_drawdowns"]
    dd_pct = mc_result["max_drawdown_pcts"]
    drawdown_percentiles = {}
    drawdown_pct_percentiles = {}
    for p in percentiles:
        drawdown_percentiles[p] = float(np.percentile(dd_abs, p))
        drawdown_pct_percentiles[p] = float(np.percentile(dd_pct, p))

    # Final equity distribution
    final_eq = mc_result["final_equities"]
    initial = mc_result["initial_equity"]

    # Return distribution
    returns = (final_eq - initial) / initial * 100

    summary = {
        "median_final_equity": float(np.median(final_eq)),
        "mean_final_equity": float(np.mean(final_eq)),
        "worst_case_equity_5pct": float(np.percentile(final_eq, 5)),
        "best_case_equity_95pct": float(np.percentile(final_eq, 95)),
        "median_return_pct": float(np.median(returns)),
        "worst_case_return_5pct": float(np.percentile(returns, 5)),
        "median_max_drawdown": float(np.median(dd_abs)),
        "worst_case_drawdown_95pct": float(np.percentile(dd_abs, 95)),
        "median_max_drawdown_pct": float(np.median(dd_pct)),
        "worst_case_drawdown_pct_95pct": float(np.percentile(dd_pct, 95)),
        "prob_profitable": float(np.mean(final_eq > initial) * 100),
        "prob_loss_gt_5pct": float(np.mean(returns < -5) * 100),
        "prob_loss_gt_10pct": float(np.mean(returns < -10) * 100),
        "median_sharpe": float(np.median(mc_result["sharpe_ratios"])),
        "median_win_rate": float(np.median(mc_result["win_rates"])),
        "n_trades": mc_result["n_trades"],
        "n_simulations": mc_result["n_simulations"],
    }

    return {
        "percentile_curves": percentile_curves,
        "drawdown_percentiles": drawdown_percentiles,
        "drawdown_pct_percentiles": drawdown_pct_percentiles,
        "percentiles": percentiles,
        "summary": summary,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Live performance tracking
# ---------------------------------------------------------------------------


def check_live_performance(
    live_equity: list[float],
    cones: dict[str, Any],
) -> dict[str, Any]:
    """Check if live equity curve falls within Monte Carlo confidence bands.

    If performance falls outside the 5th percentile band, the strategy
    may be broken and should be reviewed.

    Args:
        live_equity: List of equity values from live trading.
        cones: Output from compute_confidence_cones().

    Returns:
        Dict with:
          - status: "normal", "warning", "critical"
          - current_percentile: approximate percentile of current equity
          - below_5th: bool — True if below 5th percentile at any point
          - deviation_points: list of indices where equity exits bands
    """
    if not live_equity or len(live_equity) < 2:
        return {
            "status": "normal",
            "current_percentile": 50.0,
            "below_5th": False,
            "deviation_points": [],
            "message": "Insufficient live data for comparison",
        }

    p_curves = cones["percentile_curves"]
    n_live = len(live_equity)
    n_mc = cones["n_steps"]

    # Only compare up to the available MC steps
    compare_len = min(n_live, n_mc)

    band_5 = p_curves[5.0][:compare_len]
    band_25 = p_curves[25.0][:compare_len]
    band_50 = p_curves[50.0][:compare_len]
    band_75 = p_curves[75.0][:compare_len]
    band_95 = p_curves[95.0][:compare_len]
    live = np.array(live_equity[:compare_len])

    # Find deviation points (where live exits 5th-95th band)
    below_5th = live < band_5
    above_95th = live > band_95
    deviation_points = list(np.where(below_5th | above_95th)[0])

    # Estimate current percentile by interpolation
    current_live = live[-1]
    current_percentiles = {
        5: band_5[-1],
        25: band_25[-1],
        50: band_50[-1],
        75: band_75[-1],
        95: band_95[-1],
    }

    current_pct = _estimate_percentile(current_live, current_percentiles)

    # Determine status
    any_below_5th = bool(np.any(below_5th))

    if current_pct < 5:
        status = "critical"
        message = (
            f"CRITICAL: Live equity is below the 5th percentile MC band. "
            f"Strategy may be broken. Current ~{current_pct:.0f}th percentile."
        )
    elif current_pct < 25 or any_below_5th:
        status = "warning"
        message = (
            f"WARNING: Live equity is in the lower quartile of MC projections. "
            f"Current ~{current_pct:.0f}th percentile. Monitor closely."
        )
    else:
        status = "normal"
        message = (
            f"Live equity is within normal MC range. "
            f"Current ~{current_pct:.0f}th percentile."
        )

    return {
        "status": status,
        "current_percentile": round(current_pct, 1),
        "below_5th": any_below_5th,
        "deviation_points": deviation_points,
        "message": message,
    }


def _estimate_percentile(value: float, percentile_values: dict) -> float:
    """Estimate the percentile of a value given known percentile values.

    Uses linear interpolation between known percentile bands.
    """
    sorted_pcts = sorted(percentile_values.keys())
    sorted_vals = [percentile_values[p] for p in sorted_pcts]

    if value <= sorted_vals[0]:
        return float(sorted_pcts[0]) * value / max(sorted_vals[0], 1e-10)
    if value >= sorted_vals[-1]:
        return min(
            99.0,
            float(sorted_pcts[-1])
            + (value - sorted_vals[-1])
            / max(sorted_vals[-1] - sorted_vals[-2], 1e-10)
            * 5,
        )

    for i in range(len(sorted_pcts) - 1):
        if sorted_vals[i] <= value <= sorted_vals[i + 1]:
            denom = sorted_vals[i + 1] - sorted_vals[i]
            if denom <= 0:
                return float(sorted_pcts[i])
            frac = (value - sorted_vals[i]) / denom
            return float(sorted_pcts[i]) + frac * (sorted_pcts[i + 1] - sorted_pcts[i])

    return 50.0


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (PBO)
# ---------------------------------------------------------------------------


def estimate_pbo(
    strategy_scores: list[list[float]],
    n_partitions: int = 10,
) -> dict[str, Any]:
    """Estimate the Probability of Backtest Overfitting (PBO).

    Implements the methodology from Bailey and Lopez de Prado:
      1. Partition the performance data into S equal subsets
      2. Enumerate all C(S, S/2) train/test split combinations
      3. For each split: find the best strategy in-sample, measure it OOS
      4. PBO = fraction of splits where best IS underperforms median OOS

    Args:
        strategy_scores: A list of M strategy score sequences, each of length T.
            strategy_scores[i] = [score_t0, score_t1, ..., score_tT]
            where score_t is the per-period return or score for strategy i.
            Each inner list must have the same length.
        n_partitions: Number of time partitions S (even number, default 10).

    Returns:
        Dict with:
          - pbo: float in [0, 1] — probability of backtest overfitting
          - interpretation: str — "low" (<0.05), "moderate" (0.05-0.5), "high" (>0.5)
          - n_combinations: int — number of train/test splits evaluated
          - logit_distribution: list[float] — logit(rank) values for each split
    """
    if not strategy_scores or len(strategy_scores) < 2:
        return {
            "pbo": 0.0,
            "interpretation": "insufficient_data",
            "n_combinations": 0,
            "logit_distribution": [],
            "message": "Need at least 2 strategies to estimate PBO",
        }

    # Convert to numpy array: shape (n_strategies, T)
    scores = np.array(strategy_scores, dtype=np.float64)
    n_strategies, T = scores.shape

    if T < n_partitions:
        n_partitions = max(2, T // 2) * 2  # ensure even, at least 2

    # Ensure n_partitions is even
    if n_partitions % 2 != 0:
        n_partitions += 1

    # Partition into S subsets of roughly equal size
    partition_size = T // n_partitions
    if partition_size < 1:
        return {
            "pbo": 0.0,
            "interpretation": "insufficient_data",
            "n_combinations": 0,
            "logit_distribution": [],
            "message": f"Not enough data points ({T}) for {n_partitions} partitions",
        }

    # Compute per-partition performance for each strategy
    # Shape: (n_strategies, n_partitions)
    partition_perfs = np.zeros((n_strategies, n_partitions))
    for s in range(n_partitions):
        start = s * partition_size
        end = start + partition_size
        partition_perfs[:, s] = scores[:, start:end].sum(axis=1)

    # Enumerate all C(S, S/2) combinations
    half = n_partitions // 2
    from itertools import combinations

    all_combos = list(combinations(range(n_partitions), half))
    n_combinations = len(all_combos)

    # Limit to manageable number if too many combinations
    max_combos = 5000
    if n_combinations > max_combos:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_combinations, size=max_combos, replace=False)
        all_combos = [all_combos[i] for i in sorted(indices)]
        n_combinations = max_combos

    overfit_count = 0
    logit_distribution = []

    for train_partitions in all_combos:
        test_partitions = tuple(
            s for s in range(n_partitions) if s not in train_partitions
        )

        # In-sample performance per strategy
        is_perfs = partition_perfs[:, list(train_partitions)].sum(axis=1)

        # Out-of-sample performance per strategy
        oos_perfs = partition_perfs[:, list(test_partitions)].sum(axis=1)

        # Find best strategy in-sample
        best_is_idx = int(np.argmax(is_perfs))
        best_is_oos = oos_perfs[best_is_idx]

        # Median OOS performance across all strategies
        median_oos = float(np.median(oos_perfs))

        # Is the best IS strategy below median OOS? → overfit
        if best_is_oos < median_oos:
            overfit_count += 1

        # Compute logit of the rank
        # Rank of best-IS strategy among all strategies OOS
        rank = int(np.sum(oos_perfs <= best_is_oos))
        relative_rank = rank / max(n_strategies, 1)
        # Clamp to avoid log(0) or log(inf)
        relative_rank = max(0.01, min(0.99, relative_rank))
        logit_val = math.log(relative_rank / (1.0 - relative_rank))
        logit_distribution.append(logit_val)

    pbo = overfit_count / max(n_combinations, 1)

    if pbo < 0.05:
        interpretation = "low"
    elif pbo < 0.50:
        interpretation = "moderate"
    else:
        interpretation = "high"

    return {
        "pbo": round(pbo, 4),
        "interpretation": interpretation,
        "n_combinations": n_combinations,
        "logit_distribution": logit_distribution,
        "overfit_count": overfit_count,
        "n_strategies": n_strategies,
        "n_partitions": n_partitions,
        "message": _pbo_message(pbo, interpretation),
    }


def _pbo_message(pbo: float, interpretation: str) -> str:
    """Generate a human-readable PBO interpretation."""
    if interpretation == "low":
        return (
            f"PBO = {pbo:.2%} — LOW overfitting risk. "
            f"The strategy selection is robust across different data splits."
        )
    elif interpretation == "moderate":
        return (
            f"PBO = {pbo:.2%} — MODERATE overfitting risk. "
            f"Some data-snooping detected. Consider reducing the parameter "
            f"search space or using more data."
        )
    elif interpretation == "high":
        return (
            f"PBO = {pbo:.2%} — HIGH overfitting risk. "
            f"The strategy is likely curve-fit to historical data. "
            f"Do NOT deploy to live trading without significant changes."
        )
    return f"PBO = {pbo:.2%} — insufficient data for reliable assessment."


# ---------------------------------------------------------------------------
# DataFrame formatters for display
# ---------------------------------------------------------------------------


def mc_results_to_dataframe(mc_result: dict[str, Any]) -> pd.DataFrame:
    """Convert MC simulation statistics to a display DataFrame."""
    cones = compute_confidence_cones(mc_result)
    summary = cones["summary"]

    rows = [
        {"Metric": "Simulations", "Value": f"{summary['n_simulations']:,}"},
        {"Metric": "Trades Sampled", "Value": f"{summary['n_trades']}"},
        {
            "Metric": "Median Final Equity",
            "Value": f"${summary['median_final_equity']:,.0f}",
        },
        {
            "Metric": "Worst Case (5th pct)",
            "Value": f"${summary['worst_case_equity_5pct']:,.0f}",
        },
        {
            "Metric": "Best Case (95th pct)",
            "Value": f"${summary['best_case_equity_95pct']:,.0f}",
        },
        {
            "Metric": "Median Return",
            "Value": f"{summary['median_return_pct']:+.2f}%",
        },
        {
            "Metric": "Worst Return (5th pct)",
            "Value": f"{summary['worst_case_return_5pct']:+.2f}%",
        },
        {
            "Metric": "Prob. Profitable",
            "Value": f"{summary['prob_profitable']:.1f}%",
        },
        {
            "Metric": "Prob. Loss > 5%",
            "Value": f"{summary['prob_loss_gt_5pct']:.1f}%",
        },
        {
            "Metric": "Prob. Loss > 10%",
            "Value": f"{summary['prob_loss_gt_10pct']:.1f}%",
        },
        {
            "Metric": "Median Max Drawdown",
            "Value": f"${summary['median_max_drawdown']:,.0f}",
        },
        {
            "Metric": "Worst Drawdown (95th pct)",
            "Value": f"${summary['worst_case_drawdown_95pct']:,.0f}",
        },
        {
            "Metric": "Worst Drawdown % (95th pct)",
            "Value": f"{summary['worst_case_drawdown_pct_95pct']:.2f}%",
        },
        {
            "Metric": "Median Sharpe",
            "Value": f"{summary['median_sharpe']:.2f}",
        },
        {
            "Metric": "Median Win Rate",
            "Value": f"{summary['median_win_rate']:.1f}%",
        },
    ]

    return pd.DataFrame(rows)


def cone_curves_to_dataframe(cones: dict[str, Any]) -> pd.DataFrame:
    """Convert confidence cone curves to a DataFrame for plotting.

    Returns a DataFrame with columns: Trade, P5, P25, P50, P75, P95.
    """
    p_curves = cones["percentile_curves"]
    n_steps = cones["n_steps"]

    data = {"Trade": list(range(n_steps))}
    for p in cones["percentiles"]:
        data[f"P{int(p)}"] = p_curves[p].tolist()

    return pd.DataFrame(data)


def drawdown_distribution_to_dataframe(
    mc_result: dict[str, Any],
    n_bins: int = 50,
) -> pd.DataFrame:
    """Create a histogram-ready DataFrame of max drawdown distribution.

    Returns a DataFrame with columns: Drawdown, Count.
    """
    dd = mc_result["max_drawdown_pcts"]
    if len(dd) == 0:
        return pd.DataFrame(columns=pd.Index(["Drawdown %", "Count"]))

    counts, bin_edges = np.histogram(dd, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return pd.DataFrame({"Drawdown %": np.round(bin_centers, 2), "Count": counts})
