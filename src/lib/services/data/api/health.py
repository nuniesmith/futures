"""
Health check, metrics, and backfill status API router.

Provides:
    GET /health              — Service health check (Redis, Postgres, engine, live feed)
    GET /metrics             — Lightweight operational metrics
    GET /backfill/status     — Historical data backfill status (bar counts, date ranges)
    GET /backfill/gaps/{sym} — Gap analysis for a specific symbol's stored bars
"""

import contextlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.health")

router = APIRouter(tags=["health"])

# Maximum age before a model is considered stale (hours)
_MODEL_STALE_HOURS = 26


def _check_model_health() -> dict[str, Any]:
    """Check CNN model existence and freshness.

    Returns a dict with:
      - ``status``: ``"ok"`` | ``"stale"`` | ``"missing"``
      - ``available``: bool
      - ``champion_path``: str | None
      - ``size_mb``: float
      - ``last_retrain``: ISO timestamp from promotion meta, or None
      - ``last_retrain_ago``: human-readable age string, or None
      - ``val_accuracy``: float from last promotion meta, or None
      - ``stale``: bool — True when model hasn't been retrained in > 26 h
    """
    result: dict[str, Any] = {
        "status": "missing",
        "available": False,
        "champion_path": None,
        "size_mb": 0.0,
        "last_retrain": None,
        "last_retrain_ago": None,
        "val_accuracy": None,
        "precision": None,
        "recall": None,
        "stale": False,
        "total_checkpoints": 0,
        # When was sync_models.sh last run? (mtime of meta.json sidecar)
        "last_sync_time": None,
        "last_sync_ago": None,
        # Is the ONNX export present alongside the champion?
        "onnx_available": False,
    }

    # Locate models/ directory — works both in Docker (/app/models) and bare-metal
    _model_dir_candidates = [
        Path("/app/models"),
        Path(__file__).resolve().parents[5] / "models",
        Path(__file__).resolve().parents[4] / "models",
    ]
    model_dir: Path | None = None
    for _c in _model_dir_candidates:
        if _c.is_dir():
            model_dir = _c
            break

    if model_dir is None:
        return result

    # Count all checkpoints
    all_pt = list(model_dir.glob("breakout_cnn_*.pt"))
    result["total_checkpoints"] = len(all_pt)

    # Check for the ONNX export
    onnx_path = model_dir / "breakout_cnn_best.onnx"
    result["onnx_available"] = onnx_path.is_file()

    # Check for the champion model
    champion = model_dir / "breakout_cnn_best.pt"
    if not champion.is_file():
        # Fall back to newest checkpoint by mtime
        if all_pt:
            champion = max(all_pt, key=lambda p: p.stat().st_mtime)
        else:
            return result  # no models at all

    result["available"] = True
    result["champion_path"] = str(champion)
    stat = champion.stat()
    result["size_mb"] = round(stat.st_size / (1024 * 1024), 1)

    now_et = datetime.now(tz=_EST)

    # last_sync_time — mtime of the meta.json sidecar written by sync_models.sh.
    # This tells us when the operator last pulled from the orb repo, which may
    # be more recent than the model's own promoted_at training timestamp.
    meta_path = model_dir / "breakout_cnn_best_meta.json"
    if meta_path.is_file():
        sync_dt = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=_EST)
        result["last_sync_time"] = sync_dt.isoformat()
        sync_delta = now_et - sync_dt
        sync_hours = sync_delta.total_seconds() / 3600
        if sync_hours < 1:
            result["last_sync_ago"] = f"{int(sync_delta.total_seconds() / 60)}m ago"
        elif sync_hours < 24:
            result["last_sync_ago"] = f"{sync_hours:.1f}h ago"
        else:
            result["last_sync_ago"] = f"{sync_delta.days}d ago"

    # Load promotion metadata for accurate retrain timestamp + accuracy
    promoted_at: datetime | None = None

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            promoted_str = meta.get("promoted_at")
            if promoted_str:
                promoted_at = datetime.fromisoformat(promoted_str)
                if promoted_at.tzinfo is None:
                    promoted_at = promoted_at.replace(tzinfo=_EST)
                result["last_retrain"] = promoted_at.isoformat()

                delta = now_et - promoted_at
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    result["last_retrain_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
                elif hours < 24:
                    result["last_retrain_ago"] = f"{hours:.1f}h ago"
                else:
                    result["last_retrain_ago"] = f"{delta.days}d ago"

            val_acc = meta.get("val_accuracy")
            if val_acc is not None:
                result["val_accuracy"] = round(float(val_acc), 1)
            precision = meta.get("precision")
            if precision is not None:
                result["precision"] = round(float(precision), 3)
            recall = meta.get("recall")
            if recall is not None:
                result["recall"] = round(float(recall), 3)

        except Exception as exc:
            logger.debug("_check_model_health: could not read meta JSON: %s", exc)

    # Fall back to file mtime if no promotion metadata
    if promoted_at is None:
        promoted_at = datetime.fromtimestamp(stat.st_mtime, tz=_EST)
        result["last_retrain"] = promoted_at.isoformat()
        delta = now_et - promoted_at
        hours = delta.total_seconds() / 3600
        if hours < 1:
            result["last_retrain_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
        elif hours < 24:
            result["last_retrain_ago"] = f"{hours:.1f}h ago"
        else:
            result["last_retrain_ago"] = f"{delta.days}d ago"

    # Staleness check
    stale_threshold = timedelta(hours=_MODEL_STALE_HOURS)
    is_stale = (now_et - promoted_at) > stale_threshold
    result["stale"] = is_stale
    result["status"] = "stale" if is_stale else "ok"

    return result


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


def _check_postgres() -> dict[str, Any]:
    """Check Postgres connectivity."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("postgresql"):
        return {"status": "not_configured", "connected": False}
    try:
        from lib.core.models import _get_conn

        conn = _get_conn()
        try:
            conn.execute("SELECT 1")
            return {"status": "ok", "connected": True}
        finally:
            conn.close()
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
    - Postgres database connectivity
    - Engine running state
    - Massive WebSocket live feed
    - Data source (Massive vs yfinance)
    - CNN model existence, accuracy, and staleness
    - Database path
    """
    redis_status = _check_redis()
    postgres_status = _check_postgres()
    model_status = _check_model_health()
    engine = _get_engine_or_none()

    engine_status = "not_initialized"
    live_feed_status: dict[str, Any] = {"status": "unknown"}
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

    # Overall status: degraded if engine not running, model missing, or model stale
    overall_ok = engine_status == "running" and model_status["available"] and not model_status["stale"]

    # --- Update Prometheus gauges ---
    with contextlib.suppress(Exception):
        from lib.services.data.api.metrics import (
            update_engine_up,
            update_model_stale,
            update_postgres_status,
            update_redis_status,
        )

        update_redis_status(redis_status.get("connected", False))
        update_postgres_status(postgres_status.get("connected", False))
        update_engine_up(engine_status == "running")
        update_model_stale(model_status.get("stale", False))

    return {
        "status": "ok" if overall_ok else "degraded",
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "components": {
            "redis": redis_status,
            "postgres": postgres_status,
            "engine": {"status": engine_status},
            "live_feed": live_feed_status,
            "data_source": data_source,
            "model": model_status,
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

            with contextlib.suppress(Exception):
                result["backtest_results_count"] = len(engine.get_backtest_results())

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
