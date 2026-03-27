"""
src.analysis — Market analysis and signal modules.

Submodules:
    cvd             — Cumulative Volume Delta computation and divergence detection
    signal_quality  — Multi-factor signal quality scoring
    volatility      — K-Means adaptive volatility clustering
    wave_analysis   — Wave dominance tracking and trend analysis

Consumers should import directly from the relevant sub-module:

    from src.analysis.cvd import compute_cvd, detect_cvd_divergences
    from src.analysis.signal_quality import compute_signal_quality
    from src.analysis.volatility import kmeans_volatility_clusters
    from src.analysis.wave_analysis import calculate_wave_analysis
"""

from src.analysis.cvd import compute_cvd, detect_cvd_divergences
from src.analysis.signal_quality import compute_signal_quality
from src.analysis.volatility import kmeans_volatility_clusters, volatility_summary_text
from src.analysis.wave_analysis import calculate_wave_analysis, wave_summary_text

__all__ = [
    "compute_cvd",
    "detect_cvd_divergences",
    "compute_signal_quality",
    "kmeans_volatility_clusters",
    "volatility_summary_text",
    "calculate_wave_analysis",
    "wave_summary_text",
]
