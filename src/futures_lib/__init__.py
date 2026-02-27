"""
futures_lib — Shared library for the Futures Trading Co-Pilot.

All business logic modules live under organised sub-packages:

    # Core infrastructure
    from src.futures_lib.core.cache import cache_get, cache_set
    from src.futures_lib.core.models import init_db, ASSETS
    from src.futures_lib.core.alerts import get_dispatcher
    from src.futures_lib.core.logging_config import setup_logging, get_logger

    # Analysis modules
    from src.futures_lib.analysis.volatility import kmeans_volatility_clusters
    from src.futures_lib.analysis.wave_analysis import calculate_wave_analysis
    from src.futures_lib.analysis.ict import detect_fvgs, detect_order_blocks
    from src.futures_lib.analysis.cvd import compute_cvd
    from src.futures_lib.analysis.confluence import check_confluence
    from src.futures_lib.analysis.regime import RegimeDetector
    from src.futures_lib.analysis.scorer import PreMarketScorer
    from src.futures_lib.analysis.signal_quality import compute_signal_quality
    from src.futures_lib.analysis.volume_profile import compute_volume_profile
    from src.futures_lib.analysis.monte_carlo import run_monte_carlo

    # Trading modules
    from src.futures_lib.trading.engine import get_engine, DashboardEngine
    from src.futures_lib.trading.strategies import STRATEGY_CLASSES, make_strategy
    from src.futures_lib.trading.costs import get_cost_model

    # External integrations
    from src.futures_lib.integrations.grok_helper import GrokSession
    from src.futures_lib.integrations.massive_client import get_massive_provider

Services (data-service, engine, web) are sub-packages:

    from src.futures_lib.services.data.main import app
    from src.futures_lib.services.engine.focus import compute_daily_focus
    from src.futures_lib.services.web.app import main

Install in editable mode for development:

    pip install -e .
"""
