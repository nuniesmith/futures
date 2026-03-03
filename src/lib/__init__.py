"""
lib — Shared library for the Futures Trading Co-Pilot.

All business logic modules live under organised sub-packages:

    # Core infrastructure
    from lib.core.cache import cache_get, cache_set
    from lib.core.models import init_db, ASSETS
    from lib.core.alerts import get_dispatcher
    from lib.core.logging_config import setup_logging, get_logger

    # Analysis modules
    from lib.analysis.volatility import kmeans_volatility_clusters
    from lib.analysis.wave_analysis import calculate_wave_analysis
    from lib.analysis.ict import detect_fvgs, detect_order_blocks
    from lib.analysis.cvd import compute_cvd
    from lib.analysis.confluence import check_confluence
    from lib.analysis.regime import RegimeDetector
    from lib.analysis.scorer import PreMarketScorer
    from lib.analysis.signal_quality import compute_signal_quality
    from lib.analysis.volume_profile import compute_volume_profile
    from lib.analysis.monte_carlo import run_monte_carlo

    # Trading modules
    from lib.trading.engine import get_engine, DashboardEngine
    from lib.trading.strategies import STRATEGY_CLASSES, make_strategy
    from lib.trading.costs import get_cost_model

    # External integrations
    from lib.integrations.grok_helper import GrokSession
    from lib.integrations.massive_client import get_massive_provider

Services (data-service, engine, web) are sub-packages:

    from lib.services.data.main import app
    from lib.services.engine.focus import compute_daily_focus
    from lib.services.web.app import main

Install in editable mode for development:

    pip install -e .
"""
