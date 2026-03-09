"""
Strategies Package
==================
Trading strategy modules for the futures co-pilot.

Sub-packages:
  - daily/   — Daily bias analysis, trade plan generation, swing detection
  - rb/      — Range Breakout scalping system (all 13 breakout types)

The ``rb`` package is the canonical public façade for the Range Breakout
system.  It re-exports the most commonly used symbols from the core type
definitions, the detection layer, range builders, and the publisher pipeline::

    from lib.strategies.rb import (
        # Core types
        BreakoutType,
        RangeConfig,
        get_range_config,
        # Detection
        detect_range_breakout,
        detect_breakout_for_type,
        detect_all_breakout_types,
        detect_breakouts_filtered,
        BreakoutResult,
        # Range builders (pure, no side-effects)
        compute_atr,
        localize_bars,
        build_orb_range,
        build_pdr_range,
        build_ib_range,
        build_consolidation_range,
        build_weekly_range,
        build_monthly_range,
        build_asian_range,
        build_bbsqueeze_range,
        build_va_range,
        build_inside_day_range,
        build_gap_rejection_range,
        build_pivot_range,
        build_fibonacci_range,
        # Publisher / orchestration
        publish_breakout_result,
        persist_breakout_result,
        dispatch_to_position_manager,
        send_breakout_alert,
        publish_pipeline,
        # Handler pipeline
        handle_breakout_check,
    )

The ``daily`` package provides higher-timeframe analysis that feeds into
the CNN feature vector (v7+) and the dashboard daily plan view::

    from lib.strategies.daily import (
        compute_daily_bias,
        DailyBias,
        BiasDirection,
    )

Phase 1G additions (RB System Refactor):
  - ``rb.range_builders`` — all 13 ``build_*_range()`` functions extracted
    from ``engine/breakout.py``, plus canonical ``compute_atr()`` and
    ``localize_bars()`` shared helpers.
  - ``rb.detector`` — unified detection façade with ``detect_breakout_for_type()``
    and ``detect_breakouts_filtered()`` convenience wrappers.
  - ``rb.publisher`` — Redis pub + alerting with ``publish_pipeline()``
    convenience and ``get_alert_template()`` for alert formatting.

Phase 4B additions (CNN Sub-Feature Decomposition):
  - ``breakout_type_category`` — coarse grouping: time=0, range=0.5, squeeze=1.0
  - ``session_overlap_flag`` — 1.0 if London+NY overlap window
  - ``atr_trend`` — ATR expanding=1.0, contracting=0.0 (10-bar lookback)
  - ``volume_trend`` — 5-bar volume slope normalised [0, 1]
  These sub-features are computed in ``lib.analysis.breakout_cnn`` and wired
  into ``dataset_generator._build_row()`` and ``feature_contract.json`` v7.1.
"""
