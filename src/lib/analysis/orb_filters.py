"""
orb_filters — backward-compatibility shim
==========================================
This module has been renamed to ``lib.analysis.breakout_filters`` (Phase 1E).

All public symbols are re-exported from the new module so existing imports
continue to work unchanged:

    from lib.analysis.orb_filters import ORBFilterResult, apply_all_filters
    # ↑ still works — delegates to breakout_filters

Migrate callers to:

    from lib.analysis.breakout_filters import BreakoutFilterResult, apply_all_filters

This shim will be removed in a future cleanup pass once all callers are updated.
"""

from lib.analysis.breakout_filters import (  # noqa: F401
    BreakoutFilterResult,
    FilterVerdict,
    ORBFilterResult,
    apply_all_filters,
    check_lunch_filter,
    check_mtf_analyzer,
    check_multi_tf_bias,
    check_nr7,
    check_premarket_range,
    check_session_window,
    check_vwap_confluence,
    compute_session_vwap,
    extract_premarket_range,
)

__all__ = [
    "BreakoutFilterResult",
    "FilterVerdict",
    "ORBFilterResult",
    "apply_all_filters",
    "check_lunch_filter",
    "check_mtf_analyzer",
    "check_multi_tf_bias",
    "check_nr7",
    "check_premarket_range",
    "check_session_window",
    "check_vwap_confluence",
    "compute_session_vwap",
    "extract_premarket_range",
]
