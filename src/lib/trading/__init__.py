"""
lib.trading — Trading engine, strategies, and cost models.

Re-exports the public API from each sub-module so callers can do:

    from lib.trading import get_engine, DashboardEngine, STRATEGY_CLASSES
"""

from lib.trading.costs import (
    estimate_trade_costs,
    get_cost_model,
    slippage_commission_rate,
)
from lib.trading.engine import (
    DashboardEngine,
    get_engine,
    run_backtest,
    run_optimization,
)
from lib.trading.strategies import (
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    _safe_float,
    make_strategy,
    score_backtest,
    suggest_params,
)

__all__ = [
    # costs
    "estimate_trade_costs",
    "get_cost_model",
    "slippage_commission_rate",
    # engine
    "DashboardEngine",
    "get_engine",
    "run_backtest",
    "run_optimization",
    # strategies
    "STRATEGY_CLASSES",
    "STRATEGY_LABELS",
    "_safe_float",
    "make_strategy",
    "score_backtest",
    "suggest_params",
]
