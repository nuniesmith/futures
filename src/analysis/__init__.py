"""
src.analysis — Market analysis and signal modules.

Submodules:
    cvd             — Cumulative Volume Delta computation and divergence detection
    signal_quality  — Multi-factor signal quality scoring
    volatility      — K-Means adaptive volatility clustering
    wave_analysis   — Wave dominance tracking and trend analysis

Consumers should import directly from the relevant sub-module:

    from analysis.cvd import compute_cvd, detect_cvd_divergences
    from analysis.signal_quality import compute_signal_quality
    from analysis.volatility import kmeans_volatility_clusters
    from analysis.wave_analysis import calculate_wave_analysis
"""

from analysis.cvd import compute_cvd, detect_cvd_divergences
from analysis.signal_quality import compute_signal_quality
from analysis.volatility import kmeans_volatility_clusters, volatility_summary_text
from analysis.wave_analysis import calculate_wave_analysis, wave_summary_text

__all__ = [
    "calculate_wave_analysis",
    "compute_cvd",
    "compute_signal_quality",
    "detect_cvd_divergences",
    "kmeans_volatility_clusters",
    "volatility_summary_text",
    "wave_summary_text",
]
