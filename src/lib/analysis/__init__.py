"""
lib.analysis — Market analysis and signal modules.

Re-exports the public API from each sub-module so callers can do:

    from lib.analysis import compute_cvd, detect_fvgs, kmeans_volatility_clusters
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
]
