"""
lib.trading — Trading engine, strategies, and cost models.

Re-exports the public API from each sub-module so callers can do:

    from src.lib.trading import get_engine, DashboardEngine, STRATEGY_CLASSES
"""

from src.lib.trading.costs import (
    estimate_trade_costs,
    get_cost_model,
    slippage_commission_rate,
)
from src.lib.trading.engine import (
    DashboardEngine,
    get_engine,
    run_backtest,
    run_optimization,
)
from src.lib.trading.strategies import (
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
