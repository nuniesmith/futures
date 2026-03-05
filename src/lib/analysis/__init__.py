"""
lib.analysis — Market analysis and signal modules.

Re-exports the public API from each sub-module so callers can do:

    from lib.analysis import compute_cvd, detect_fvgs, kmeans_volatility_clusters
    from lib.analysis import apply_all_filters, predict_breakout
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

# Optional import — breakout_cnn requires torch which may not be installed
# in all environments.  Only inference functions are re-exported here;
# training lives in the orb repo.
_breakout_cnn_default_threshold: float = 0.82


try:
    from lib.analysis.breakout_cnn import DEFAULT_THRESHOLD as _cnn_threshold
    from lib.analysis.breakout_cnn import (
        predict_breakout,
        predict_breakout_batch,
    )

    _breakout_cnn_default_threshold = _cnn_threshold
except ImportError:
    predict_breakout = None  # type: ignore[assignment,misc]
    predict_breakout_batch = None  # type: ignore[assignment,misc]

DEFAULT_THRESHOLD: float = _breakout_cnn_default_threshold

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
    # breakout_cnn (optional — inference only)
    "predict_breakout",
    "predict_breakout_batch",
    "DEFAULT_THRESHOLD",
]
