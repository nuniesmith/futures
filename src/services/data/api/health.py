"""
Health check and metrics API router.

Provides:
    GET /health   — Service health check (Redis, engine, live feed)
    GET /metrics  — Lightweight operational metrics
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.health")

router = APIRouter(tags=["health"])


def _check_redis() -> dict:
    """Check Redis connectivity."""
    try:
        from cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            return {"status": "ok", "connected": True}
        return {"status": "unavailable", "connected": False}
    except Exception as exc:
        return {"status": "error", "connected": False, "error": str(exc)}


def _get_engine_or_none():
    """Try to get the engine singleton without raising."""
    try:
        from engine import get_engine

        return get_engine()
    except Exception:
        return None


@router.get("/health")
def health():
    """Service health check.

    Returns the status of all critical subsystems:
    - Redis cache connectivity
    - Engine running state
    - Massive WebSocket live feed
    - Data source (Massive vs yfinance)
    - Database path
    """
    redis_status = _check_redis()
    engine = _get_engine_or_none()

    engine_status = "not_initialized"
    live_feed_status = {"status": "unknown"}
    data_source = "unknown"

    if engine is not None:
        try:
            status = engine.get_status()
            engine_status = status.get("engine", "unknown")
            live_feed_status = status.get("live_feed", {"status": "unknown"})
            data_source = live_feed_status.get("data_source", "unknown")
        except Exception as exc:
            engine_status = f"error: {exc}"

    db_path = os.getenv("DB_PATH", "futures_journal.db")

    return {
        "status": "ok" if engine_status == "running" else "degraded",
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "components": {
            "redis": redis_status,
            "engine": {"status": engine_status},
            "live_feed": live_feed_status,
            "data_source": data_source,
            "database": {"path": db_path},
        },
    }


@router.get("/metrics")
def metrics():
    """Lightweight operational metrics.

    Returns counts and timing information useful for monitoring
    the data service without overwhelming detail.
    """
    engine = _get_engine_or_none()

    result = {
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "engine_running": False,
        "data_refresh": {},
        "optimization": {},
        "backtest": {},
        "live_feed": {},
        "backtest_results_count": 0,
        "tracked_assets_count": 0,
    }

    if engine is not None:
        try:
            status = engine.get_status()
            result["engine_running"] = status.get("engine") == "running"
            result["data_refresh"] = status.get("data_refresh", {})
            result["optimization"] = status.get("optimization", {})
            result["backtest"] = status.get("backtest", {})
            result["live_feed"] = status.get("live_feed", {})

            try:
                result["backtest_results_count"] = len(engine.get_backtest_results())
            except Exception:
                pass

            try:
                from models import ASSETS

                result["tracked_assets_count"] = len(ASSETS)
            except Exception:
                pass
        except Exception as exc:
            result["error"] = str(exc)

    # Redis key count (if available)
    try:
        from cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            futures_keys = list(_r.scan_iter("futures:*", count=1000))
            result["redis_cached_keys"] = len(futures_keys)
    except Exception:
        result["redis_cached_keys"] = -1

    return result
