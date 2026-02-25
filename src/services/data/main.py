"""
Futures Data Service — FastAPI + Background Engine
===================================================
Dedicated microservice that runs all heavy computation:
  - Massive WebSocket listener (real-time 1m bars)
  - DashboardEngine (5m refresh, optimization, backtesting)
  - FKS modules (volatility, wave, signal quality, regime, CVD, ICT)
  - REST API for the Streamlit UI to consume

The Streamlit app becomes a pure thin client that reads from this
service's API + Redis cache for ultra-low latency.

Usage (from project root):
    PYTHONPATH=src/services/data/core:src uvicorn src.services.data.main:app --host 0.0.0.0 --port 8000

Docker:
    ENV PYTHONPATH="/app/src/services/data/core:/app/src"
    CMD ["uvicorn", "src.services.data.main:app", ...]
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Ensure core modules are importable via bare imports (e.g. `from cache import ...`)
# AND that the src/ root is importable for shared modules like grok_helper, scorer, etc.
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(_this_dir, "core")
_src_dir = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))

for _p in (_core_dir, _src_dir, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="[DATA-SVC] %(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_service")

# ---------------------------------------------------------------------------
# Import core modules (these use bare imports like `from cache import ...`
# which resolve because core/ is on sys.path)
# ---------------------------------------------------------------------------
from api.actions import router as actions_router  # noqa: E402
from api.actions import set_engine as actions_set_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Import routers — these live under src/services/data/api/
# They use `from core.X import ...` style, which works because _this_dir
# is on sys.path and core/ is a proper package.
# ---------------------------------------------------------------------------
from api.analysis import router as analysis_router  # noqa: E402
from api.analysis import set_engine as analysis_set_engine  # noqa: E402
from api.health import router as health_router  # noqa: E402
from api.journal import router as journal_router  # noqa: E402
from api.positions import router as positions_router  # noqa: E402
from api.trades import router as trades_router  # noqa: E402

from engine import DashboardEngine, get_engine  # noqa: E402
from models import init_db  # noqa: E402

# ---------------------------------------------------------------------------
# Engine singleton — shared across all routers
# ---------------------------------------------------------------------------
_engine: DashboardEngine | None = None


def get_current_engine() -> DashboardEngine:
    """Return the running engine instance. Used by health router."""
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
    logger.info("  Data Service starting up")
    logger.info("=" * 60)

    # 1. Initialise the database (SQLite for now, Postgres later)
    try:
        init_db()
        logger.info(
            "Database initialised (DB_PATH=%s)",
            os.getenv("DB_PATH", "futures_journal.db"),
        )
    except Exception as exc:
        logger.error("Database init failed: %s", exc)

    # 2. Read configuration from environment
    account_size = int(
        os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000"))
    )
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # 3. Create and start the background engine
    #    get_engine() returns the singleton — it auto-starts a daemon thread
    _engine = get_engine(
        account_size=account_size,
        interval=interval,
        period=period,
    )
    app.state.engine = _engine

    # 4. Inject engine into routers that need it
    analysis_set_engine(_engine)
    actions_set_engine(_engine)

    logger.info(
        "Engine started: account=$%s  interval=%s  period=%s",
        f"{account_size:,}",
        interval,
        period,
    )

    # 5. Log data source
    try:
        from cache import get_data_source

        ds = get_data_source()
        logger.info("Primary data source: %s", ds)
    except Exception:
        logger.info("Primary data source: yfinance (default)")

    logger.info("=" * 60)
    logger.info("  Data Service ready — accepting requests")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("=" * 60)
    logger.info("  Data Service shutting down")
    logger.info("=" * 60)

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
        "computation modules. Exposes REST endpoints for the Streamlit UI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit (typically on port 8501) and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://streamlit-app:8501",
        "http://127.0.0.1:8501",
        "*",  # For development; tighten in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
# Analysis: /analysis/latest, /analysis/latest/{ticker}, /analysis/status, etc.
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])

# Actions: /actions/force_refresh, /actions/optimize_now, /actions/update_settings, etc.
app.include_router(actions_router, prefix="/actions", tags=["Actions"])

# Positions: /positions/, /positions/update, etc.  (NinjaTrader bridge)
app.include_router(positions_router, prefix="/positions", tags=["Positions"])

# Trades: /trades, /trades/{id}/close, /log_trade, etc.  (trade CRUD)
app.include_router(trades_router, prefix="", tags=["Trades"])

# Journal: /journal/save, /journal/entries, /journal/stats, /journal/today
app.include_router(journal_router, prefix="/journal", tags=["Journal"])

# Health: /health, /metrics  (no prefix — top-level)
app.include_router(health_router, tags=["Health"])


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Service info and links to docs."""
    return {
        "service": "futures-data-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "analysis": "/analysis/latest",
            "status": "/analysis/status",
            "force_refresh": "/actions/force_refresh",
            "positions": "/positions/",
            "trades": "/trades",
            "journal": "/journal/entries",
            "health": "/health",
            "metrics": "/metrics",
        },
    }


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
