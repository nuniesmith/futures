"""
lib.analysis — Market analysis and signal modules.

Re-exports the public API from each sub-module so callers can do:

    from lib.analysis import compute_cvd, detect_fvgs, kmeans_volatility_clusters
    from lib.analysis import apply_all_filters, render_ruby_snapshot, predict_breakout
"""

from lib.analysis.confluence import (
    MultiTimeframeFilter,
    check_confluence,
    get_recommended_timeframes,
)
from lib.analysis.cvd import (
    compute_cvd,
    cvd_summary,
    detect_absorption_candles,
    detect_cvd_divergences,
)
from lib.analysis.ict import (
    detect_breaker_blocks,
    detect_fvgs,
    detect_order_blocks,
    ict_summary,
)
from lib.analysis.monte_carlo import (
    compute_confidence_cones,
    run_monte_carlo,
)
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
from lib.analysis.regime import (
    RegimeDetector,
    detect_regime_hmm,
    fit_detector,
)
from lib.analysis.scorer import (
    EVENT_CATALOG,
    PreMarketScorer,
    results_to_dataframe,
    score_instruments,
)
from lib.analysis.signal_quality import compute_signal_quality
from lib.analysis.volatility import kmeans_volatility_clusters
from lib.analysis.volume_profile import (
    compute_session_profiles,
    compute_volume_profile,
    find_naked_pocs,
    format_profile_summary,
    profile_to_dataframe,
)
from lib.analysis.wave_analysis import calculate_wave_analysis

# Optional imports — these require extra dependencies (torch, mplfinance)
# that may not be installed in all environments.  We import them lazily
# so the rest of lib.analysis works without GPU/torch.
try:
    from lib.analysis.chart_renderer import (
        RenderConfig,
        cleanup_inference_images,
        render_batch_snapshots,
        render_ruby_snapshot,
        render_snapshot_for_inference,
    )
except ImportError:
    render_ruby_snapshot = None  # type: ignore[assignment,misc]
    render_batch_snapshots = None  # type: ignore[assignment,misc]
    render_snapshot_for_inference = None  # type: ignore[assignment,misc]
    cleanup_inference_images = None  # type: ignore[assignment,misc]
    RenderConfig = None  # type: ignore[assignment,misc]

_breakout_cnn_default_threshold: float = 0.82

try:
    from lib.analysis.breakout_cnn import DEFAULT_THRESHOLD as _cnn_threshold
    from lib.analysis.breakout_cnn import (
        HybridBreakoutCNN,
        model_info,
        predict_breakout,
        predict_breakout_batch,
        train_model,
    )

    _breakout_cnn_default_threshold = _cnn_threshold
except ImportError:
    predict_breakout = None  # type: ignore[assignment,misc]
    predict_breakout_batch = None  # type: ignore[assignment,misc]
    train_model = None  # type: ignore[assignment,misc]
    model_info = None  # type: ignore[assignment,misc]
    HybridBreakoutCNN = None  # type: ignore[assignment,misc]

DEFAULT_THRESHOLD: float = _breakout_cnn_default_threshold

try:
    from lib.analysis.orb_simulator import (
        BracketConfig,
        ORBSimResult,
        simulate_batch,
        simulate_day,
        simulate_orb_outcome,
        summarise_results,
    )
except ImportError:
    simulate_orb_outcome = None  # type: ignore[assignment,misc]
    simulate_batch = None  # type: ignore[assignment,misc]
    simulate_day = None  # type: ignore[assignment,misc]
    summarise_results = None  # type: ignore[assignment,misc]
    ORBSimResult = None  # type: ignore[assignment,misc]
    BracketConfig = None  # type: ignore[assignment,misc]

try:
    from lib.analysis.dataset_generator import (
        DatasetConfig,
        DatasetStats,
        generate_dataset,
        split_dataset,
        validate_dataset,
    )
except ImportError:
    generate_dataset = None  # type: ignore[assignment,misc]
    split_dataset = None  # type: ignore[assignment,misc]
    validate_dataset = None  # type: ignore[assignment,misc]
    DatasetConfig = None  # type: ignore[assignment,misc]
    DatasetStats = None  # type: ignore[assignment,misc]

__all__ = [
    # confluence
    "MultiTimeframeFilter",
    "check_confluence",
    "get_recommended_timeframes",
    # cvd
    "compute_cvd",
    "cvd_summary",
    "detect_absorption_candles",
    "detect_cvd_divergences",
    # ict
    "detect_breaker_blocks",
    "detect_fvgs",
    "detect_order_blocks",
    "ict_summary",
    # monte_carlo
    "compute_confidence_cones",
    "run_monte_carlo",
    # orb_filters
    "ORBFilterResult",
    "apply_all_filters",
    "check_lunch_filter",
    "check_multi_tf_bias",
    "check_nr7",
    "check_premarket_range",
    "check_session_window",
    "check_vwap_confluence",
    "extract_premarket_range",
    # regime
    "RegimeDetector",
    "detect_regime_hmm",
    "fit_detector",
    # scorer
    "EVENT_CATALOG",
    "PreMarketScorer",
    "results_to_dataframe",
    "score_instruments",
    # signal_quality
    "compute_signal_quality",
    # volatility
    "kmeans_volatility_clusters",
    # volume_profile
    "compute_session_profiles",
    "compute_volume_profile",
    "find_naked_pocs",
    "format_profile_summary",
    "profile_to_dataframe",
    # wave_analysis
    "calculate_wave_analysis",
    # chart_renderer (optional)
    "RenderConfig",
    "render_ruby_snapshot",
    "render_batch_snapshots",
    "render_snapshot_for_inference",
    "cleanup_inference_images",
    # breakout_cnn (optional)
    "HybridBreakoutCNN",
    "predict_breakout",
    "predict_breakout_batch",
    "train_model",
    "model_info",
    "DEFAULT_THRESHOLD",
    # orb_simulator (optional)
    "ORBSimResult",
    "BracketConfig",
    "simulate_orb_outcome",
    "simulate_batch",
    "simulate_day",
    "summarise_results",
    # dataset_generator (optional)
    "DatasetConfig",
    "DatasetStats",
    "generate_dataset",
    "split_dataset",
    "validate_dataset",
]
