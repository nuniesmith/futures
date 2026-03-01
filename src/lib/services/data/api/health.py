"""
Health check, metrics, and backfill status API router.

Provides:
    GET /health              — Service health check (Redis, engine, live feed)
    GET /metrics             — Lightweight operational metrics
    GET /backfill/status     — Historical data backfill status (bar counts, date ranges)
    GET /backfill/gaps/{sym} — Gap analysis for a specific symbol's stored bars
"""

import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.health")

router = APIRouter(tags=["health"])


def _check_redis() -> dict[str, Any]:
    """Check Redis connectivity."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            return {"status": "ok", "connected": True}
        return {"status": "unavailable", "connected": False}
    except Exception as exc:
        return {"status": "error", "connected": False, "error": str(exc)}


def _get_engine_or_none():
    """Try to get the engine singleton without raising."""
    try:
        from lib.trading.engine import get_engine

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
                from lib.core.models import ASSETS

                result["tracked_assets_count"] = len(ASSETS)
            except Exception:
                pass
        except Exception as exc:
            result["error"] = str(exc)

    # Redis key count (if available)
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            futures_keys = list(_r.scan_iter("futures:*", count=1000))
            result["redis_cached_keys"] = len(futures_keys)
    except Exception:
        result["redis_cached_keys"] = -1

    return result


# ---------------------------------------------------------------------------
# Backfill status endpoints (TASK-204)
# ---------------------------------------------------------------------------


@router.get("/backfill/status")
def backfill_status():
    """Return the current historical data backfill status.

    Shows per-symbol bar counts, date ranges, and total stored bars.
    Useful for monitoring whether backfill is running and how much
    data is available for optimization and backtesting.
    """
    try:
        from lib.services.engine.backfill import get_backfill_status

        status = get_backfill_status()
        return {
            "status": "ok",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            **status,
        }
    except ImportError:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "message": "Backfill module not available",
            "symbols": [],
            "total_bars": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "error": str(exc),
            "symbols": [],
            "total_bars": 0,
        }


@router.get("/backfill/gaps/{symbol}")
def backfill_gaps(symbol: str, days_back: int = 30):
    """Analyse gaps in stored historical data for a specific symbol.

    Args:
        symbol: Ticker symbol (e.g. ``MGC=F``). URL-encode the ``=`` as ``%3D``.
        days_back: Number of calendar days to analyse (default 30).

    Returns:
        Gap report with total bars, expected bars, coverage percentage,
        and a list of significant gaps (>30 minutes).
    """
    try:
        from lib.services.engine.backfill import get_gap_report

        report = get_gap_report(symbol, days_back=days_back)
        return {
            "status": "ok",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            **report,
        }
    except ImportError:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "message": "Backfill module not available",
            "symbol": symbol,
            "total_bars": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "symbol": symbol,
            "error": str(exc),
        }
