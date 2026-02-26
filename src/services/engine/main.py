"""
Engine Service — Background computation worker
================================================
Runs the DashboardEngine as a standalone service, separate from data-service.

Responsibilities:
  - FKS computation (wave, volatility, signal quality, regime, ICT, CVD)
  - Optuna optimization and walk-forward backtesting
  - Massive WebSocket live feed management
  - Session-aware scheduling (pre-market, active, off-hours)
  - Writes all results to Redis for data-service API to serve

The data-service becomes a thin API layer that reads from Redis.

Usage:
    python -m src.services.engine.main

Docker:
    CMD ["python", "-m", "src.services.engine.main"]
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Path setup for bare imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
for _p in (_src_dir, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logging_config import get_logger, setup_logging

setup_logging(service="engine")
logger = get_logger("engine_service")

_EST = ZoneInfo("America/New_York")
HEALTH_FILE = "/tmp/engine_health.json"


def _write_health(healthy: bool, status: str, **extras):
    """Write health status to a file for Docker healthcheck."""
    data = {
        "healthy": healthy,
        "status": status,
        "timestamp": datetime.now(tz=_EST).isoformat(),
        **extras,
    }
    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _get_session_mode() -> str:
    """Determine current trading session based on ET time."""
    hour = datetime.now(tz=_EST).hour
    if 0 <= hour < 5:
        return "pre-market"
    elif 5 <= hour < 12:
        return "active"
    else:
        return "off-hours"


def main():
    logger.info("=" * 60)
    logger.info("  Engine Service starting up")
    logger.info("=" * 60)

    # Configuration from environment
    account_size = int(os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000")))
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # Import and start the engine
    from engine import get_engine
    from cache import get_data_source

    engine = get_engine(
        account_size=account_size,
        interval=interval,
        period=period,
    )

    session = _get_session_mode()
    logger.info(
        "Engine started: account=$%s  interval=%s  period=%s  session=%s  data_source=%s",
        f"{account_size:,}",
        interval,
        period,
        session,
        get_data_source(),
    )

    _write_health(True, "running", session=session)

    logger.info("=" * 60)
    logger.info("  Engine Service ready — session: %s", session.upper())
    logger.info("=" * 60)

    # Publish engine status to Redis periodically
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not shutdown:
            # Update session mode and health file
            current_session = _get_session_mode()
            _write_health(True, "running", session=current_session)

            # Publish engine status to Redis for data-service to read
            try:
                from cache import cache_set

                status = engine.get_status()
                status["session_mode"] = current_session
                cache_set(
                    "engine:status",
                    json.dumps(status, default=str).encode(),
                    ttl=60,
                )

                # Publish backtest results
                bt = engine.get_backtest_results()
                if bt:
                    cache_set(
                        "engine:backtest_results",
                        json.dumps(bt, default=str).encode(),
                        ttl=300,
                    )

                # Publish strategy history
                sh = engine.get_strategy_history()
                if sh:
                    cache_set(
                        "engine:strategy_history",
                        json.dumps(sh, default=str).encode(),
                        ttl=300,
                    )

                # Publish live feed status
                lf = engine.get_live_feed_status()
                cache_set(
                    "engine:live_feed_status",
                    json.dumps(lf, default=str).encode(),
                    ttl=30,
                )
            except Exception as exc:
                logger.debug("Failed to publish engine status to Redis: %s", exc)

            time.sleep(10)  # Update status every 10 seconds

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    # Shutdown
    logger.info("=" * 60)
    logger.info("  Engine Service shutting down")
    logger.info("=" * 60)

    _write_health(False, "shutting_down")

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.stop())
        loop.close()
    except Exception as exc:
        logger.warning("Engine stop error (non-fatal): %s", exc)

    _write_health(False, "stopped")
    logger.info("Engine Service stopped")


if __name__ == "__main__":
    main()
