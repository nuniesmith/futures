"""
Futures Data Service — FastAPI + Background Engine
===================================================
Dedicated microservice that runs all heavy computation:
  - Massive WebSocket listener (real-time 1m bars)
  - DashboardEngine (5m refresh, optimization, backtesting)
  - Ruby modules (volatility, wave, signal quality, regime, CVD, ICT)
  - REST API + HTMX dashboard

Usage (from project root):
    PYTHONPATH=src uvicorn lib.services.data.main:app --host 0.0.0.0 --port 8000

Docker:
    ENV PYTHONPATH="/app/src"
    CMD ["uvicorn", "lib.services.data.main:app", ...]
"""

import json
import math
import os
from contextlib import asynccontextmanager
from typing import Any

# ---------------------------------------------------------------------------
# All imports use fully-qualified `lib.*` paths.
# PYTHONPATH only needs /app/src so that `lib` is discoverable.
# ---------------------------------------------------------------------------
from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402


# ---------------------------------------------------------------------------
# Custom JSON encoder that replaces inf / NaN with null instead of crashing.
# Backtesting and optimization routines can produce inf Sharpe ratios or
# NaN win-rates, which the stdlib json encoder rejects.
# ---------------------------------------------------------------------------
class _SafeFloatEncoder(json.JSONEncoder):
    """JSON encoder that converts inf/-inf/NaN to None."""

    def default(self, o: Any) -> Any:
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(_sanitize(o))


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse subclass that handles inf/NaN floats gracefully."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=_SafeFloatEncoder,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


# ---------------------------------------------------------------------------
# Logging — structured via structlog
# ---------------------------------------------------------------------------
from lib.core.logging_config import (  # noqa: E402  # pylint: disable=wrong-import-position
    get_logger,
    setup_logging,
)

setup_logging(service="data-service")
logger = get_logger("data_service")

# ---------------------------------------------------------------------------
# Import routers — these live under src/services/data/api/
# Bare imports like `from cache import ...` resolve via PYTHONPATH (/app/src).
# ---------------------------------------------------------------------------
from lib.core.models import init_db  # noqa: E402
from lib.services.data.api.actions import (  # noqa: E402
    router as actions_router,
)
from lib.services.data.api.actions import (  # noqa: E402
    set_engine as actions_set_engine,
)
from lib.services.data.api.analysis import (  # noqa: E402
    router as analysis_router,
)
from lib.services.data.api.analysis import (  # noqa: E402
    set_engine as analysis_set_engine,
)
from lib.services.data.api.audit import router as audit_router  # noqa: E402
from lib.services.data.api.auth import require_api_key  # noqa: E402
from lib.services.data.api.bars import (  # noqa: E402
    router as bars_router,
)
from lib.services.data.api.bars import (  # noqa: E402
    startup_warm_caches,
)
from lib.services.data.api.cnn import router as cnn_router  # noqa: E402
from lib.services.data.api.dashboard import (  # noqa: E402
    router as dashboard_router,
)
from lib.services.data.api.grok import router as grok_router  # noqa: E402
from lib.services.data.api.grok import set_engine as grok_set_engine  # noqa: E402
from lib.services.data.api.health import (  # noqa: E402
    router as health_router,
)
from lib.services.data.api.journal import (  # noqa: E402
    router as journal_router,
)
from lib.services.data.api.kraken import router as kraken_router  # noqa: E402
from lib.services.data.api.market_data import (  # noqa: E402
    router as market_data_router,
)
from lib.services.data.api.metrics import PrometheusMiddleware  # noqa: E402
from lib.services.data.api.metrics import (  # noqa: E402
    router as metrics_router,
)
from lib.services.data.api.nt8_deploy import (  # noqa: E402
    router as nt8_deploy_router,
)
from lib.services.data.api.positions import (  # noqa: E402
    router as positions_router,
)
from lib.services.data.api.rate_limit import (  # noqa: E402
    setup_rate_limiting,
)
from lib.services.data.api.risk import router as risk_router  # noqa: E402
from lib.services.data.api.sar import router as sar_router  # noqa: E402
from lib.services.data.api.settings import (  # noqa: E402
    router as settings_router,
)
from lib.services.data.api.sse import router as sse_router  # noqa: E402
from lib.services.data.api.trades import (  # noqa: E402
    router as trades_router,
)
from lib.services.data.api.trainer import (  # noqa: E402
    router as trainer_router,
)

# ---------------------------------------------------------------------------
# Engine mode: embedded (legacy, all-in-one) or remote (reads from Redis)
# Set ENGINE_MODE=remote when running engine as separate container.
# ---------------------------------------------------------------------------
_ENGINE_MODE = os.getenv("ENGINE_MODE", "embedded")  # "embedded" or "remote"

_engine = None


class _RemoteEngineProxy:
    """Lightweight proxy that reads engine state from Redis.

    When the engine runs in a separate container, it publishes status,
    backtest results, and strategy history to Redis keys.  This proxy
    reads those keys so the API routers work without modification.
    """

    def __init__(self):
        self.interval = os.getenv("ENGINE_INTERVAL", "5m")
        self.period = os.getenv("ENGINE_PERIOD", "5d")

    def _redis_get_json(self, key: str, default: Any = None) -> Any:
        try:
            from lib.core.cache import cache_get  # noqa: E402

            raw = cache_get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return default

    def get_status(self) -> dict[str, Any]:
        return self._redis_get_json(
            "engine:status",
            {
                "engine": "remote",
                "data_refresh": {"last": None, "status": "unknown"},
                "optimization": {"last": None, "status": "unknown"},
                "backtest": {"last": None, "status": "unknown"},
                "live_feed": {"status": "unknown"},
            },
        )

    def get_backtest_results(self) -> list[Any]:
        return self._redis_get_json("engine:backtest_results", [])

    def get_strategy_history(self) -> dict[str, Any]:
        return self._redis_get_json("engine:strategy_history", {})

    def get_live_feed_status(self) -> dict[str, Any]:
        return self._redis_get_json(
            "engine:live_feed_status",
            {
                "status": "unknown",
                "connected": False,
                "data_source": "unknown",
            },
        )

    def force_refresh(self) -> None:
        try:
            from lib.core.cache import flush_all

            flush_all()
        except Exception:
            pass

    def start_live_feed(self) -> bool:
        return False

    async def stop_live_feed(self) -> None:
        pass

    def upgrade_live_feed(self) -> None:
        pass

    def downgrade_live_feed(self) -> None:
        pass

    def update_settings(self, **kwargs) -> None:
        pass

    async def stop(self) -> None:
        pass


def get_current_engine():
    """Return the running engine instance (or remote proxy). Used by health router."""
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialised — service is still starting up")
    return _engine


# ---------------------------------------------------------------------------
# Lifespan: start engine + background tasks on startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine

    logger.info("=" * 60)
    logger.info("  Data Service starting up (engine_mode=%s)", _ENGINE_MODE)
    logger.info("=" * 60)

    # 1. Initialise the database
    try:
        init_db()
        logger.info(
            "Database initialised (DB_PATH=%s)",
            os.getenv("DB_PATH", "futures_journal.db"),
        )
    except Exception as exc:
        logger.error("Database init failed: %s", exc)

    # 2. Read configuration from environment
    account_size = int(os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000")))
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # 3. Start engine (embedded) or connect proxy (remote)
    if _ENGINE_MODE == "remote":
        _engine = _RemoteEngineProxy()
        logger.info("Using remote engine proxy (reads from Redis)")
    else:
        from lib.trading.engine import get_engine

        _engine = get_engine(
            account_size=account_size,
            interval=interval,
            period=period,
        )
        logger.info(
            "Embedded engine started: account=$%s  interval=%s  period=%s",
            f"{account_size:,}",
            interval,
            period,
        )

    app.state.engine = _engine

    # 4. Inject engine into routers that need it
    analysis_set_engine(_engine)
    actions_set_engine(_engine)
    grok_set_engine(_engine)

    # 5. Log data source
    try:
        from lib.core.cache import get_data_source

        ds = get_data_source()
        logger.info("Primary data source: %s", ds)
    except Exception:
        logger.info("Primary data source: yfinance (default)")

    # 5b. Start Kraken WebSocket feed for live crypto data (if enabled)
    _kraken_feed = None
    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO

        if ENABLE_KRAKEN_CRYPTO:
            from lib.integrations.kraken_client import start_kraken_feed

            _kraken_feed = start_kraken_feed()
            logger.info(
                "Kraken WebSocket feed started: %d pairs",
                len(_kraken_feed._pairs),
            )
        else:
            logger.info("Kraken crypto disabled (ENABLE_KRAKEN_CRYPTO=0)")
    except ImportError:
        logger.debug("Kraken client not available — crypto feed disabled")
    except Exception as exc:
        logger.warning("Kraken feed startup failed (non-fatal): %s", exc)

    # 6. Warm Redis bar caches from Postgres so the dataset generator and
    #    engine have data available immediately without waiting for a fill.
    try:
        startup_warm_caches(days_back=7)
    except Exception as exc:
        logger.warning("Startup cache warm failed (non-fatal): %s", exc)

    logger.info("=" * 60)
    logger.info("  Data Service ready — accepting requests")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("=" * 60)
    logger.info("  Data Service shutting down")
    logger.info("=" * 60)

    # Stop Kraken WebSocket feed
    if _kraken_feed is not None:
        try:
            await _kraken_feed.stop()
            logger.info("Kraken feed stopped cleanly")
        except Exception as exc:
            logger.warning("Kraken feed stop error (non-fatal): %s", exc)

    if _engine is not None:
        try:
            await _engine.stop()
            logger.info("Engine stopped cleanly")
        except Exception as exc:
            logger.warning("Engine stop error (non-fatal): %s", exc)

    logger.info("Data Service stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Futures Data Service",
    description=(
        "Background data service for the Futures Trading Co-Pilot. "
        "Runs the DashboardEngine, Massive WS listener, and all FKS "
        "computation modules. Exposes REST endpoints and an HTMX dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Structured error responses — consistent JSON shape for all error types
# ---------------------------------------------------------------------------
# Every error response follows:
#   { "error": "<short_code>", "detail": "<human message>", "status": <int> }
# This replaces FastAPI's default {"detail": ...} shape so clients can
# rely on a single schema for error handling.
# ---------------------------------------------------------------------------


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle all HTTP exceptions (404, 403, 405, 422, 500, etc.)."""
    status = exc.status_code
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    # Map common status codes to short error codes
    code_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }
    error_code = code_map.get(status, f"http_{status}")

    return JSONResponse(
        status_code=status,
        content={
            "error": error_code,
            "detail": detail,
            "status": status,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic / query-param validation errors with structured JSON."""
    errors = exc.errors()
    # Build a human-readable summary from the validation error list
    messages = []
    for err in errors:
        loc = " → ".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid")
        messages.append(f"{loc}: {msg}" if loc else msg)

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "; ".join(messages) if messages else "Request validation failed",
            "status": 422,
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — log and return structured 500."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": f"Internal server error: {type(exc).__name__}",
            "status": 500,
        },
    )


# CORS — allow local dev origins + Tailscale IPs
# NOTE: allow_credentials must be False when using wildcard origins.
# The SSE StreamingResponse also must NOT set Access-Control-Allow-Origin
# manually — let this middleware handle it consistently to avoid the
# "wildcard + credentials" conflict that makes browsers drop EventSource.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware — records request count + latency (TASK-704)
app.add_middleware(PrometheusMiddleware)

# Rate limiting (TASK-703) — slowapi-based per-client rate limits
setup_rate_limiting(app)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
# Dashboard: / (HTML page), /api/focus, /api/focus/html, /api/time, etc.
# NOTE: dashboard_router is mounted WITHOUT a prefix so GET / serves the HTML
# dashboard and /api/focus, /api/focus/html etc. are top-level paths.
app.include_router(dashboard_router, tags=["Dashboard"])

# SSE: /sse/dashboard (live event stream), /sse/health
# NOTE: sse_router is mounted WITHOUT a prefix so /sse/dashboard is top-level.
app.include_router(sse_router, tags=["SSE"])

# Analysis: /analysis/latest, /analysis/latest/{ticker}, /analysis/status, etc.
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])

# Actions: /actions/force_refresh, /actions/optimize_now, /actions/update_settings, etc.
app.include_router(actions_router, prefix="/actions", tags=["Actions"])

# Positions: /positions/, /positions/update, etc.  (NinjaTrader bridge)
app.include_router(positions_router, prefix="/positions", tags=["Positions"])

# Trades: /trades, /trades/{id}/close, /log_trade, etc.  (trade CRUD)
app.include_router(trades_router, prefix="", tags=["Trades"])

# Risk: /risk/status, /risk/check, /risk/history  (risk engine API)
app.include_router(risk_router, prefix="/risk", tags=["Risk"])

# Audit: /audit/risk, /audit/orb, /audit/summary  (persistent event history)
app.include_router(audit_router, prefix="/audit", tags=["Audit"])

# Journal: /journal/save, /journal/entries, /journal/stats, /journal/today
app.include_router(journal_router, prefix="/journal", tags=["Journal"])

# Market Data: /data/ohlcv, /data/daily, /data/source  (OHLCV proxy for thin client)
app.include_router(market_data_router, prefix="/data", tags=["Market Data"])

# Bars: /bars/{symbol}, /bars/bulk, /bars/status, /bars/assets
#       /bars/{symbol}/gaps, /bars/{symbol}/fill, /bars/fill/all, /bars/fill/status
# Historical bar store — serves data from Postgres with auto gap-fill from Massive.
# Primary data source for CNN dataset generation, engine backtesting, and web charts.
app.include_router(bars_router, tags=["Bars"])

# Health: /health, /metrics  (no prefix — top-level)
app.include_router(health_router, tags=["Health"])

# Prometheus metrics: /metrics/prometheus  (TASK-704)
app.include_router(metrics_router, tags=["Metrics"])

# NT8 Deploy: /api/nt8/installer, /api/nt8/panel/html, /api/nt8/deploy
# NOTE: nt8_deploy_router is mounted WITHOUT a prefix — routes are defined with /api/nt8/ paths.
app.include_router(nt8_deploy_router, tags=["NT8 Deploy"])

# CNN: /cnn/status, /cnn/retrain, /cnn/retrain/status, /cnn/history, /cnn/status/html
# NOTE: cnn_router is mounted WITHOUT a prefix — routes are defined with /cnn/ paths.
app.include_router(cnn_router, tags=["CNN"])

# Kraken: /kraken/health, /kraken/status, /kraken/pairs, /kraken/ticker/{pair},
#          /kraken/tickers, /kraken/ohlcv/{pair}, /kraken/health/html
# NOTE: kraken_router is mounted WITHOUT a prefix — routes are defined with /kraken/ paths.
app.include_router(kraken_router, tags=["Kraken"])

# Grok: /api/grok/latest, /api/grok/briefing, /api/grok/trigger/briefing,
#        /api/grok/trigger/update, /sse/grok/briefing, /sse/grok/update
# NOTE: grok_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(grok_router, tags=["Grok"])

# SAR Sync: /sar/sync, /sar/state, /sar/state/{asset}
# Receives stop-and-reverse reversal events from the NinjaTrader BreakoutStrategy
# (POST /sar/sync from 100.127.182.112) and exposes current per-instrument SAR
# direction to the engine and dashboard.
app.include_router(sar_router, prefix="/sar", tags=["SAR Sync"])

# Trainer: /trainer (HTML page), /trainer/api/* (proxy to trainer service),
#          /trainer/config, /trainer/service_status
# NOTE: trainer_router is mounted WITHOUT a prefix — routes are defined with /trainer/ paths.
app.include_router(trainer_router, tags=["Trainer"])

# Settings: /settings (HTML page)
# NOTE: settings_router is mounted WITHOUT a prefix — route is defined with /settings path.
app.include_router(settings_router, tags=["Settings"])


# ---------------------------------------------------------------------------
# Root endpoint — now served by dashboard_router (GET / returns HTML dashboard)
# The old JSON root is moved to /api/info for programmatic consumers.
# ---------------------------------------------------------------------------
@app.get("/api/info")
def api_info():
    """Service info and links to docs (formerly GET /)."""
    return {
        "service": "futures-data-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "dashboard": "/",
            "focus_json": "/api/focus",
            "focus_html": "/api/focus/html",
            "sse_dashboard": "/sse/dashboard",
            "sse_health": "/sse/health",
            "analysis": "/analysis/latest",
            "status": "/analysis/status",
            "force_refresh": "/actions/force_refresh",
            "positions": "/positions/",
            "trades": "/trades",
            "journal": "/journal/entries",
            "market_data": "/data/ohlcv",
            "daily_data": "/data/daily",
            "data_source": "/data/source",
            "health": "/health",
            "metrics": "/metrics",
            "nt8_installer": "/api/nt8/installer",
            "nt8_deploy": "/api/nt8/deploy",
            "nt8_health": "/api/nt8/health",
            "nt8_health_html": "/api/nt8/health/html",
            "cnn_status": "/cnn/status",
            "cnn_status_html": "/cnn/status/html",
            "cnn_retrain": "/cnn/retrain",
            "cnn_retrain_status": "/cnn/retrain/status",
            "cnn_retrain_log": "/cnn/retrain/log",
            "cnn_retrain_cancel": "/cnn/retrain/cancel",
            "cnn_history": "/cnn/history",
            "cnn_sync": "/cnn/sync",
            "cnn_sync_status": "/cnn/sync/status",
            "cnn_watcher_status": "/cnn/watcher/status",
            "kraken_health": "/kraken/health",
            "kraken_status": "/kraken/status",
            "kraken_pairs": "/kraken/pairs",
            "kraken_ticker": "/kraken/ticker/{pair}",
            "kraken_tickers": "/kraken/tickers",
            "kraken_ohlcv": "/kraken/ohlcv/{pair}",
            "kraken_health_html": "/kraken/health/html",
            "trainer_page": "/trainer",
            "trainer_config": "/trainer/config",
            "trainer_service_status": "/trainer/service_status",
            "trainer_api": "/trainer/api/{path}",
            "bars": "/bars/{symbol}",
            "bars_bulk": "/bars/bulk",
            "bars_status": "/bars/status",
            "bars_assets": "/bars/assets",
            "bars_gaps": "/bars/{symbol}/gaps",
            "bars_fill": "/bars/{symbol}/fill",
            "bars_fill_all": "/bars/fill/all",
            "bars_fill_status": "/bars/fill/status",
        },
    }


# ---------------------------------------------------------------------------
# Favicon — return 204 No Content to suppress browser 404 errors.
# The HTML dashboard uses an inline SVG data-URI favicon in <link rel="icon">,
# but browsers still request /favicon.ico automatically on first load.
# ---------------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Run directly: python -m src.services.data.main
# or:           python src/services/data/main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DATA_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("DATA_SERVICE_PORT", "8000"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
