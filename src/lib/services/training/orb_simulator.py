"""
orb_simulator — backward-compatibility shim
============================================
This module has been renamed to ``lib.services.training.rb_simulator``
(Phase 1F).

All public symbols are re-exported from the new module so existing imports
continue to work unchanged:

    from lib.services.training.orb_simulator import ORBSimResult, simulate_batch
    # ↑ still works — delegates to rb_simulator

Migrate callers to:

    from lib.services.training.rb_simulator import RBSimResult, simulate_batch

This shim will be removed in a future cleanup pass once all callers are
updated.
"""

from lib.services.training.rb_simulator import (  # noqa: F401
    DEFAULT_BRACKET,
    BracketConfig,
    ORBSimResult,
    RBSimResult,
    _compute_atr,
    _localize_to_est,
    _simulate_range_outcome,
    results_to_dataframe,
    simulate_batch,
    simulate_batch_asian,
    simulate_batch_bollinger_squeeze,
    simulate_batch_consolidation,
    simulate_batch_fibonacci,
    simulate_batch_gap_rejection,
    simulate_batch_ib,
    simulate_batch_inside_day,
    simulate_batch_monthly,
    simulate_batch_pivot_points,
    simulate_batch_prev_day,
    simulate_batch_value_area,
    simulate_batch_weekly,
    simulate_day,
    simulate_orb_outcome,
    simulate_rb_outcome,
    summarise_results,
)

__all__ = [
    "BracketConfig",
    "DEFAULT_BRACKET",
    "ORBSimResult",
    "RBSimResult",
    "results_to_dataframe",
    "simulate_batch",
    "simulate_batch_asian",
    "simulate_batch_bollinger_squeeze",
    "simulate_batch_consolidation",
    "simulate_batch_fibonacci",
    "simulate_batch_gap_rejection",
    "simulate_batch_ib",
    "simulate_batch_inside_day",
    "simulate_batch_monthly",
    "simulate_batch_pivot_points",
    "simulate_batch_prev_day",
    "simulate_batch_value_area",
    "simulate_batch_weekly",
    "simulate_day",
    "simulate_orb_outcome",
    "simulate_rb_outcome",
    "summarise_results",
]
