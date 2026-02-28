"""
lib — Shared library for the Futures Trading Co-Pilot.

All business logic modules live under organised sub-packages:

    # Core infrastructure
    from src.lib.core.cache import cache_get, cache_set
    from src.lib.core.models import init_db, ASSETS
    from src.lib.core.alerts import get_dispatcher
    from src.lib.core.logging_config import setup_logging, get_logger

    # Analysis modules
    from src.lib.analysis.volatility import kmeans_volatility_clusters
    from src.lib.analysis.wave_analysis import calculate_wave_analysis
    from src.lib.analysis.ict import detect_fvgs, detect_order_blocks
    from src.lib.analysis.cvd import compute_cvd
    from src.lib.analysis.confluence import check_confluence
    from src.lib.analysis.regime import RegimeDetector
    from src.lib.analysis.scorer import PreMarketScorer
    from src.lib.analysis.signal_quality import compute_signal_quality
    from src.lib.analysis.volume_profile import compute_volume_profile
    from src.lib.analysis.monte_carlo import run_monte_carlo

    # Trading modules
    from src.lib.trading.engine import get_engine, DashboardEngine
    from src.lib.trading.strategies import STRATEGY_CLASSES, make_strategy
    from src.lib.trading.costs import get_cost_model

    # External integrations
    from src.lib.integrations.grok_helper import GrokSession
    from src.lib.integrations.massive_client import get_massive_provider

Services (data-service, engine, web) are sub-packages:

    from src.lib.services.data.main import app
    from src.lib.services.engine.focus import compute_daily_focus
    from src.lib.services.web.app import main

Install in editable mode for development:

    pip install -e .
"""
