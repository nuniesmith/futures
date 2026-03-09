"""
Backward-compatibility shim — ``lib.trading.strategies``
========================================================
This module has moved to ``lib.strategies.strategy_defs`` (Phase 1G).
All public symbols are re-exported here so existing callers continue to
work without changes.  New code should import from
``lib.strategies.strategy_defs`` directly.
"""

from lib.strategies.strategy_defs import (  # noqa: F401
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    BreakoutStrategy,
    EventReaction,
    ICTTrendEMA,
    MACDMomentum,
    ORBStrategy,
    PlainEMACross,
    PullbackEMA,
    RSIReversal,
    TrendEMACross,
    VWAPReversion,
    _safe_float,
    make_strategy,
    score_backtest,
    suggest_params,
)

__all__ = [
    "BreakoutStrategy",
    "EventReaction",
    "ICTTrendEMA",
    "MACDMomentum",
    "ORBStrategy",
    "PlainEMACross",
    "PullbackEMA",
    "RSIReversal",
    "STRATEGY_CLASSES",
    "STRATEGY_LABELS",
    "TrendEMACross",
    "VWAPReversion",
    "_safe_float",
    "make_strategy",
    "score_backtest",
    "suggest_params",
]
