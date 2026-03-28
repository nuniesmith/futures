"""
Futures HTMX Dashboard — FastAPI Web App
=========================================
Lightweight web dashboard for monitoring the futures bot.
Reads directly from Redis — no data service proxy needed.

Stack: FastAPI + Jinja2 + HTMX (no heavy JS)

Routes:
    GET /                  — Dashboard (live worker cards + stats)
    GET /assets            — Asset registry browser (grouped by category)
    GET /signals           — Recent signals feed with asset filter
    GET /reports           — Grok AI reports viewer (day/week/month)
    GET /pnl               — PnL summary with per-asset breakdown
    GET /api/health        — JSON health check

HTMX partials (polled every 5 s by dashboard):
    GET /partials/workers  — Worker status cards
    GET /partials/stats    — Stats summary bar
    GET /partials/signals  — Signals table rows
    GET /partials/report   — Report content block
    GET /partials/pnl      — PnL summary block

Environment:
    REDIS_URL           redis://localhost:6379/0
    REDIS_PASSWORD      redis auth password
    WEB_HOST            0.0.0.0
    WEB_PORT            8080
    WEB_PASSWORD_HASH   bcrypt hash (optional — leave unset for no auth)
    WEB_SESSION_SECRET  HMAC secret for session cookies
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.logging_config import get_logger, setup_logging
from src.services.asset_registry import ASSET_REGISTRY, get_categories, list_by_category
from src.services.redis_store import RedisStore
from src.web.auth import SessionAuthMiddleware, is_auth_enabled
from src.web.auth import router as auth_router

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD") or None
WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

setup_logging(service="web")
logger = get_logger("web")

# ---------------------------------------------------------------------------
# Redis store (singleton, initialised at startup)
# ---------------------------------------------------------------------------

_store: RedisStore | None = None


def get_store() -> RedisStore:
    if _store is None:
        raise RuntimeError("Redis store not yet initialised")
    return _store


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    global _store
    _store = RedisStore(redis_url=REDIS_URL, password=REDIS_PASSWORD)
    await _store.connect()
    logger.info("Futures dashboard started — redis=%s port=%s", REDIS_URL, WEB_PORT)
    yield
    if _store:
        await _store.close()
    logger.info("Futures dashboard stopped")


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Futures Dashboard",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.include_router(auth_router)

if is_auth_enabled():
    app.add_middleware(SessionAuthMiddleware)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Jinja2 globals
# ---------------------------------------------------------------------------


def _pnl_class(val: float) -> str:
    if val > 0:
        return "pos"
    if val < 0:
        return "neg"
    return "zero"


def _fmt_pnl(val: float) -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.4f}"


def _age(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    return f"{int(diff / 3600)}h ago"


def _heartbeat_alive(ts: float, ttl: int = 120) -> bool:
    return (time.time() - ts) < ttl


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


templates.env.globals.update(
    pnl_class=_pnl_class,
    fmt_pnl=_fmt_pnl,
    age=_age,
    heartbeat_alive=_heartbeat_alive,
    fmt_pct=_fmt_pct,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health", include_in_schema=False)
async def health() -> JSONResponse:
    store = get_store()
    return JSONResponse(
        {
            "status": "ok",
            "redis": store.connected,
            "timestamp": time.time(),
        }
    )


# ---------------------------------------------------------------------------
# Dashboard — GET /
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    store = get_store()
    heartbeats = await store.get_heartbeats()
    stats = await store.get_aggregate_stats(days=1)
    daily_pnl = await store.get_daily_pnl()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "heartbeats": heartbeats,
            "stats": stats,
            "daily_pnl": daily_pnl,
            "active_page": "dashboard",
        },
    )


@app.get("/partials/workers", response_class=HTMLResponse, include_in_schema=False)
async def partial_workers(request: Request) -> HTMLResponse:
    store = get_store()
    heartbeats = await store.get_heartbeats()
    return templates.TemplateResponse(
        request,
        "partials/workers.html",
        {"heartbeats": heartbeats},
    )


@app.get("/partials/stats", response_class=HTMLResponse, include_in_schema=False)
async def partial_stats(request: Request) -> HTMLResponse:
    store = get_store()
    stats = await store.get_aggregate_stats(days=1)
    daily_pnl = await store.get_daily_pnl()
    return templates.TemplateResponse(
        request,
        "partials/stats.html",
        {"stats": stats, "daily_pnl": daily_pnl},
    )


# ---------------------------------------------------------------------------
# Assets — GET /assets
# ---------------------------------------------------------------------------


@app.get("/assets", response_class=HTMLResponse)
async def assets_page(request: Request) -> HTMLResponse:
    """Show all contracts from the registry grouped by category."""
    categories = get_categories()
    assets_by_category: dict[str, list[dict[str, Any]]] = {}
    for cat in categories:
        assets_by_category[cat] = [
            {"key": key, "spec": spec} for key, spec in list_by_category(cat).items()
        ]
    store = get_store()
    disabled = await store.get_ui_disabled_assets()
    return templates.TemplateResponse(
        request,
        "assets.html",
        {
            "assets_by_category": assets_by_category,
            "categories": categories,
            "disabled_assets": disabled,
            "active_page": "assets",
        },
    )


# ---------------------------------------------------------------------------
# Asset toggle — POST /assets/{key}/toggle
# ---------------------------------------------------------------------------


@app.post("/assets/{key}/toggle", response_class=HTMLResponse, include_in_schema=False)
async def toggle_asset(request: Request, key: str) -> HTMLResponse:
    """Toggle a single asset's enabled/disabled state in Redis.

    Returns an HTMX fragment — just the updated toggle button — so the
    page doesn't need a full reload.  The bot picks up changes on next
    restart (config is re-read from Redis UI overrides at startup).
    """
    spec = ASSET_REGISTRY.get(key)
    if spec is None:
        return HTMLResponse(content="<span class='err'>Unknown asset</span>", status_code=404)

    store = get_store()
    now_enabled = await store.toggle_ui_asset(key)
    return templates.TemplateResponse(
        request,
        "partials/asset_toggle.html",
        {"key": key, "spec": spec, "enabled": now_enabled},
    )


# ---------------------------------------------------------------------------
# Signals — GET /signals
# ---------------------------------------------------------------------------


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request, asset: str = "", limit: int = 100) -> HTMLResponse:
    store = get_store()
    signals = await _fetch_signals(store, asset=asset, limit=limit)
    heartbeats = await store.get_heartbeats()
    return templates.TemplateResponse(
        request,
        "signals.html",
        {
            "signals": signals,
            "selected_asset": asset,
            "known_assets": sorted(heartbeats.keys()),
            "limit": limit,
            "active_page": "signals",
        },
    )


@app.get("/partials/signals", response_class=HTMLResponse, include_in_schema=False)
async def partial_signals(request: Request, asset: str = "", limit: int = 100) -> HTMLResponse:
    store = get_store()
    signals = await _fetch_signals(store, asset=asset, limit=limit)
    return templates.TemplateResponse(
        request,
        "partials/signals_table.html",
        {"signals": signals},
    )


async def _fetch_signals(store: RedisStore, asset: str = "", limit: int = 100) -> list[dict]:
    """Fetch and merge signals across one or all assets."""
    if asset:
        sigs = await store.get_signals(asset, limit=limit)
        for s in sigs:
            s.setdefault("_asset", asset)
        return sigs

    heartbeats = await store.get_heartbeats()
    all_sigs: list[dict] = []
    for a in heartbeats:
        sigs = await store.get_signals(a, limit=20)
        for s in sigs:
            s.setdefault("_asset", a)
        all_sigs.extend(sigs)

    all_sigs.sort(key=lambda s: s.get("timestamp", 0), reverse=True)
    return all_sigs[:limit]


# ---------------------------------------------------------------------------
# Reports — GET /reports
# ---------------------------------------------------------------------------


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request, period: str = "day", date: str = "") -> HTMLResponse:
    store = get_store()
    available_dates = await store.list_report_dates(period)
    if date and date != "latest":
        report = await store.get_report_by_date(period, date)
        selected_date = date
    else:
        report = await store.get_latest_report(period)
        selected_date = available_dates[0] if available_dates else ""
    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            "report": report,
            "period": period,
            "selected_date": selected_date,
            "available_dates": available_dates,
            "active_page": "reports",
        },
    )


@app.get("/partials/report", response_class=HTMLResponse, include_in_schema=False)
async def partial_report(request: Request, period: str = "day", date: str = "") -> HTMLResponse:
    store = get_store()
    available_dates = await store.list_report_dates(period)
    if date and date != "latest":
        report = await store.get_report_by_date(period, date)
        selected_date = date
    else:
        report = await store.get_latest_report(period)
        selected_date = available_dates[0] if available_dates else ""
    return templates.TemplateResponse(
        request,
        "partials/report_content.html",
        {
            "report": report,
            "period": period,
            "selected_date": selected_date,
            "available_dates": available_dates,
        },
    )


# ---------------------------------------------------------------------------
# PnL — GET /pnl
# ---------------------------------------------------------------------------


@app.get("/pnl", response_class=HTMLResponse)
async def pnl_page(request: Request, days: int = 7) -> HTMLResponse:
    store = get_store()
    stats = await store.get_aggregate_stats(days=days)
    history = await store.get_pnl_history(days=days)
    daily_pnl = await store.get_daily_pnl()
    return templates.TemplateResponse(
        request,
        "pnl.html",
        {
            "stats": stats,
            "history": history,
            "daily_pnl": daily_pnl,
            "days": days,
            "active_page": "pnl",
        },
    )


@app.get("/partials/pnl", response_class=HTMLResponse, include_in_schema=False)
async def partial_pnl(request: Request, days: int = 7) -> HTMLResponse:
    store = get_store()
    stats = await store.get_aggregate_stats(days=days)
    history = await store.get_pnl_history(days=days)
    return templates.TemplateResponse(
        request,
        "partials/pnl_summary.html",
        {"stats": stats, "history": history, "days": days},
    )
