"""
System Health API Router
==========================
Provides live health indicators for core services (Data, Engine, Redis,
Postgres, CNN model) displayed in the dashboard header bar.

The NT8/NinjaTrader-specific deployment, installer generation, and Bridge
health-probing code has been removed.  TradingView integration is handled
by the ``tradingview.py`` router instead.

Endpoints:
    GET  /api/nt8/panel/html     — Empty placeholder (legacy HTMX target)
    GET  /api/nt8/health/html    — Health status bar HTML fragment (polled)
    GET  /api/nt8/health         — Health status JSON

.. note::
   Route paths still use ``/api/nt8/`` to avoid breaking existing HTMX
   ``hx-get`` attributes in the dashboard HTML.  A future rename to
   ``/api/health/`` can be done once all dashboard references are updated.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("api.system_health")

router = APIRouter(tags=["System Health"])

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


def _cnn_model_on_disk() -> bool:
    """Return True if a usable CNN model file exists on disk.

    Checks (in priority order):
      1. ``CNN_MODEL_PATH`` env var override (explicit path to any model file)
      2. ``models/breakout_cnn_best.pt``  — champion PyTorch checkpoint
         (written by the trainer after every successful promotion)
      3. ``models/breakout_cnn_best.onnx`` — exported ONNX model
         (only present after an explicit ONNX export step)

    The .pt file is the primary indicator because the engine hot-reloads
    it directly; the ONNX is optional and used by external consumers.
    """
    override = os.getenv("CNN_MODEL_PATH", "")
    if override:
        return os.path.isfile(override)

    here = os.path.dirname(__file__)
    root = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))

    # Primary: champion .pt checkpoint (always present after training)
    pt_path = os.path.join(root, "models", "breakout_cnn_best.pt")
    if os.path.isfile(pt_path):
        return True

    # Secondary: exported ONNX (present after explicit export)
    onnx_path = os.path.join(root, "models", "breakout_cnn_best.onnx")
    return os.path.isfile(onnx_path)


def _compute_health() -> dict[str, Any]:
    """Compute full system health status from all available cache sources.

    Returns a dict with:
        --- Service-level indicators ---
        data_service_up: bool     — always True if this code is running
        engine_up: bool           — engine status == "running"
        redis_up: bool            — Redis ping succeeds
        postgres_up: bool         — Postgres SELECT 1 succeeds

        --- Broker / TradingView indicators ---
        broker_connected: bool    — heartbeat received within TTL
        broker_state: str         — "Realtime", "Connected", "disconnected", etc.
        broker_version: str       — connector version
        broker_account: str       — e.g. "Tradovate-Sim101"
        broker_age_seconds: float — seconds since last heartbeat
        positions_count: int      — open positions
        risk_blocked: bool        — risk enforcement blocking

        --- CNN model indicators ---
        cnn_model_on_disk: bool   — champion model file present on disk
    """
    result: dict[str, Any] = {
        # Service-level health
        "data_service_up": True,  # If we're computing this, the data service is up
        "engine_up": False,
        "redis_up": False,
        "postgres_up": False,
        # Broker / TradingView health
        "broker_connected": False,
        "bridge_connected": False,  # Legacy alias for dashboard compat
        "broker_state": "disconnected",
        "bridge_state": "disconnected",  # Legacy alias
        "broker_version": "",
        "bridge_version": "",  # Legacy alias
        "broker_account": "",
        "bridge_account": "",  # Legacy alias
        "broker_age_seconds": -1,
        "bridge_age_seconds": -1,  # Legacy alias
        "positions_count": 0,
        "risk_blocked": False,
        "last_heartbeat": None,
        # CNN model
        "cnn_model_on_disk": False,
        # Legacy fields kept for backward compat with dashboard JS
        "ruby_attached": False,
        "signalbus_active": False,
        "signalbus_pending": 0,
        "breakout_instruments": 0,
    }

    # --- Service-level checks ---

    # Redis
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            result["redis_up"] = True
    except Exception:
        pass

    # Postgres
    try:
        database_url = os.getenv("DATABASE_URL", "")
        if database_url.startswith("postgresql"):
            from lib.core.models import _get_conn

            conn = _get_conn()
            try:
                conn.execute("SELECT 1")
                result["postgres_up"] = True
            finally:
                conn.close()
    except Exception:
        pass

    # Engine (check Redis-cached engine status)
    try:
        from lib.core.cache import cache_get as _cg

        raw_status = _cg("engine:status")
        if raw_status:
            eng = json.loads(raw_status)
            result["engine_up"] = eng.get("engine") == "running"
    except Exception:
        pass

    # --- Broker heartbeat (primary freshness signal) ---
    heartbeat = None
    try:
        from lib.core.cache import cache_get

        raw = cache_get("futures:broker_heartbeat:current")
        # Fall back to legacy key if new key has no data
        if not raw:
            raw = cache_get("futures:bridge_heartbeat:current")
        if raw:
            heartbeat = json.loads(raw)
    except Exception:
        pass

    if heartbeat:
        received_at = heartbeat.get("received_at", "")
        account = heartbeat.get("account", "")
        state = heartbeat.get("state", "unknown")
        version = heartbeat.get("broker_version", heartbeat.get("bridge_version", ""))

        result["broker_account"] = account
        result["bridge_account"] = account
        result["broker_state"] = state
        result["bridge_state"] = state
        result["broker_version"] = version
        result["bridge_version"] = version
        result["positions_count"] = heartbeat.get("positions", 0)
        result["risk_blocked"] = heartbeat.get("riskBlocked", False)
        result["last_heartbeat"] = received_at

        if received_at:
            try:
                dt = datetime.fromisoformat(received_at)
                age = (datetime.now(tz=_EST) - dt).total_seconds()
                result["broker_age_seconds"] = round(age, 1)
                result["bridge_age_seconds"] = round(age, 1)
                # Consider alive if heartbeat < 60s old
                connected = age < 60
                result["broker_connected"] = connected
                result["bridge_connected"] = connected
            except Exception:
                pass

    # --- CNN model file presence ---
    try:
        result["cnn_model_on_disk"] = _cnn_model_on_disk()
    except Exception:
        result["cnn_model_on_disk"] = False

    return result


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------


def _render_health_dot(label: str, is_up: bool, title_up: str, title_down: str) -> str:
    """Render a single health indicator dot with label."""
    if is_up:
        bg = "#22c55e"
        title = title_up
        text_color = "#d4d4d8"
    else:
        bg = "#ef4444"
        title = title_down
        text_color = "#71717a"

    return f"""
        <span style="display:inline-flex;align-items:center;gap:4px;cursor:default" title="{title}">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{bg}"></span>
            <span style="font-size:11px;color:{text_color}">{label}</span>
        </span>"""


def _render_health_bar(health: dict[str, Any]) -> str:
    """Render health indicators as a compact HTML fragment.

    Shows colored dots for Data, Engine, Redis, Postgres (service-level)
    and a CNN badge.  Designed to sit in the dashboard header bar.
    """
    # --- Service-level indicators ---
    data_ok = health.get("data_service_up", True)
    engine_ok = health.get("engine_up", False)
    redis_ok = health.get("redis_up", False)
    postgres_ok = health.get("postgres_up", False)

    # --- CNN model ---
    cnn_on_disk = health.get("cnn_model_on_disk", False)

    # Service dots
    data_dot = _render_health_dot("Data", data_ok, "Data Service: Running", "Data Service: Down")
    engine_dot = _render_health_dot("Engine", engine_ok, "Engine: Running", "Engine: Not running")
    redis_dot = _render_health_dot("Redis", redis_ok, "Redis: Connected", "Redis: Disconnected")
    pg_dot = _render_health_dot("Postgres", postgres_ok, "Postgres: Connected", "Postgres: Disconnected")

    # CNN badge — purple when model is on disk, grey when missing
    if cnn_on_disk:
        cnn_title = "CNN model ready"
        cnn_bg = "rgba(88,28,135,0.5)"
        cnn_border = "rgba(126,34,206,0.6)"
        cnn_color = "#d8b4fe"
        cnn_label = "CNN ✓"
    else:
        cnn_title = "CNN model not found — run: bash scripts/sync_models.sh"
        cnn_bg = "rgba(39,39,42,0.8)"
        cnn_border = "#3f3f46"
        cnn_color = "#71717a"
        cnn_label = "CNN –"

    cnn_badge = f"""<span style="padding:2px 6px;background:{cnn_bg};border:1px solid {cnn_border};
                     border-radius:4px;font-size:10px;color:{cnn_color};font-weight:600;
                     letter-spacing:0.025em;cursor:default" title="{cnn_title}">{cnn_label}</span>"""

    return f"""
    <span style="display:inline-flex;align-items:center;gap:10px">
        {data_dot}
        {engine_dot}
        {redis_dot}
        {pg_dot}
        <span style="margin-left:4px">{cnn_badge}</span>
    </span>
    """


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Primary routes (clean paths)
# ---------------------------------------------------------------------------


@router.get("/api/health/html", response_class=HTMLResponse)
def get_health_html():
    """Return system health indicators as an HTML fragment (polled by HTMX)."""
    health = _compute_health()
    return HTMLResponse(content=_render_health_bar(health))


@router.get("/api/health")
def get_health():
    """Return system health status as JSON."""
    health = _compute_health()
    return JSONResponse(content=health)


# ---------------------------------------------------------------------------
# Legacy routes — kept so old HTMX references don't 404
# ---------------------------------------------------------------------------


@router.get("/api/nt8/panel/html", response_class=HTMLResponse)
def get_panel_html_legacy():
    """Return an empty HTML fragment (legacy toolbar placeholder)."""
    return HTMLResponse(content="")


@router.get("/api/nt8/health/html", response_class=HTMLResponse)
def get_health_html_legacy():
    """Legacy route — redirects to /api/health/html."""
    return get_health_html()


@router.get("/api/nt8/health")
def get_health_legacy():
    """Legacy route — redirects to /api/health."""
    return get_health()
