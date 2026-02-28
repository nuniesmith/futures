"""
FastAPI server for receiving trade data from NinjaTrader and external tools.

.. deprecated::
    This module is **DEPRECATED** and kept only for backwards compatibility
    with existing tests (``tests/test_positions.py``).  All functionality has
    been migrated to the new 4-service architecture under
    ``src/services/data/api/`` (trades, positions, journal, actions, health).

    New code should use the data-service endpoints instead:
        - Trades:     src/services/data/api/trades.py
        - Positions:  src/services/data/api/positions.py
        - Journal:    src/services/data/api/journal.py
        - Actions:    src/services/data/api/actions.py
        - Health:     src/services/data/api/health.py

    This file will be removed once tests are migrated to the new service.

Run the server:
    python api_server.py

Listens on port 8000. Supports:
    POST /trades              - Create a new OPEN trade
    POST /trades/{id}/close   - Close a trade with exit price
    POST /trades/{id}/cancel  - Cancel an open trade
    POST /log_trade           - Legacy endpoint (creates and immediately closes)
    GET  /trades              - List trades (filterable by status, account_size)
    GET  /trades/open         - List open trades
    GET  /trades/{id}         - Get single trade
    POST /update_positions    - NinjaTrader live position bridge (push)
    GET  /positions           - Current live positions from NinjaTrader
    GET  /health              - Health check
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")

# Ensure sibling modules are importable

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.futures_lib.core.cache import _cache_key, cache_get, cache_set  # noqa: E402
from src.futures_lib.core.models import (  # noqa: E402
    ACCOUNT_PROFILES,
    CONTRACT_SPECS,
    DB_PATH,
    STATUS_CLOSED,
    STATUS_OPEN,
    cancel_trade,
    close_trade,
    create_trade,
    get_all_trades,
    get_closed_trades,
    get_open_trades,
    get_today_pnl,
    init_db,
)

logger = logging.getLogger("api_server")

app = FastAPI(
    title="Futures Dashboard Trade API",
    description=(
        "REST API for trade management and NinjaTrader live position bridge — "
        "supports $50k / $100k / $150k accounts"
    ),
    version="3.0.0",
)

# Initialise DB on startup
init_db()


# ---------------------------------------------------------------------------
# Cache key & TTL for live positions from NinjaTrader
# ---------------------------------------------------------------------------
_POSITIONS_CACHE_KEY = _cache_key("live_positions", "current")
_POSITIONS_TTL = 7200  # 2 hours — positions persist across brief disconnects


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateTradeRequest(BaseModel):
    account_size: int = Field(
        150000,
        description="Account size: 50000, 100000, or 150000",
    )
    asset: str = Field(..., description="Asset name, e.g. 'Gold', 'E-mini S&P'")
    direction: str = Field(..., description="LONG or SHORT")
    entry: float = Field(..., description="Entry price")
    sl: Optional[float] = Field(None, description="Stop loss price")
    tp: Optional[float] = Field(None, description="Take profit price")
    contracts: int = Field(1, ge=1, description="Number of contracts")
    strategy: str = Field("", description="Strategy name")
    notes: str = Field("", description="Trade notes")


class CloseTradeRequest(BaseModel):
    close_price: float = Field(..., description="Exit price")


class LegacyTradeRequest(BaseModel):
    """Backwards-compatible with the original /log_trade endpoint."""

    asset: str
    direction: str
    entry: float
    exit_price: float
    contracts: int = 1
    pnl: float = 0.0
    strategy: str = ""
    notes: str = ""


class TradeResponse(BaseModel):
    id: int
    created_at: str
    account_size: int
    asset: str
    direction: str
    entry: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    contracts: int
    status: str
    close_price: Optional[float] = None
    close_time: Optional[str] = None
    pnl: Optional[float] = None
    rr: Optional[float] = None
    notes: str = ""
    strategy: str = ""


# ---------------------------------------------------------------------------
# NinjaTrader Live Position Bridge models
# ---------------------------------------------------------------------------


class NTPosition(BaseModel):
    """A single live position pushed from NinjaTrader."""

    symbol: str = Field(..., description="Instrument full name, e.g. 'MESZ5'")
    side: str = Field(..., description="Long or Short")
    quantity: float = Field(..., description="Number of contracts")
    avgPrice: float = Field(..., description="Average fill price")
    unrealizedPnL: float = Field(0.0, description="Current unrealized P&L in USD")
    lastUpdate: Optional[str] = Field(None, description="ISO timestamp of last update")


class NTPositionsPayload(BaseModel):
    """Payload pushed by the LivePositionBridge NinjaTrader indicator."""

    account: str = Field(..., description="NinjaTrader account name, e.g. 'Sim101'")
    positions: List[NTPosition] = Field(
        default_factory=list, description="List of open positions"
    )
    timestamp: Optional[str] = Field(None, description="UTC timestamp from NT")


class NTPositionsResponse(BaseModel):
    """Response returned by GET /positions."""

    account: str = ""
    positions: List[NTPosition] = []
    timestamp: str = ""
    received_at: str = ""
    has_positions: bool = False
    total_unrealized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Helper to read a single trade
# ---------------------------------------------------------------------------


def _get_trade_by_id(trade_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM trades_v2 WHERE id = ?", (trade_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/trades", response_model=TradeResponse, status_code=201)
def api_create_trade(req: CreateTradeRequest):
    """Create a new OPEN trade."""
    if req.asset not in CONTRACT_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown asset '{req.asset}'. Valid: {list(CONTRACT_SPECS.keys())}",
        )

    acct_sizes = [p["size"] for p in ACCOUNT_PROFILES.values()]
    if req.account_size not in acct_sizes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid account_size. Valid: {acct_sizes}",
        )

    trade_id = create_trade(
        account_size=req.account_size,
        asset=req.asset,
        direction=req.direction.upper(),
        entry=req.entry,
        sl=req.sl or 0.0,
        tp=req.tp or 0.0,
        contracts=req.contracts,
        strategy=req.strategy,
        notes=req.notes,
    )
    return _get_trade_by_id(trade_id)


@app.post("/trades/{trade_id}/close", response_model=TradeResponse)
def api_close_trade(trade_id: int, req: CloseTradeRequest):
    """Close an open trade with the given exit price. Calculates P&L automatically."""
    try:
        close_trade(trade_id, req.close_price)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _get_trade_by_id(trade_id)


@app.post("/trades/{trade_id}/cancel")
def api_cancel_trade(trade_id: int):
    """Cancel an open trade (never filled)."""
    trade = _get_trade_by_id(trade_id)
    if trade["status"] != STATUS_OPEN:
        raise HTTPException(
            status_code=400,
            detail=f"Trade {trade_id} is {trade['status']}, cannot cancel",
        )
    cancel_trade(trade_id)
    return {"status": "cancelled", "trade_id": trade_id}


@app.get("/trades", response_model=list[TradeResponse])
def api_list_trades(
    status: Optional[str] = Query(
        None, description="Filter by status: OPEN, CLOSED, CANCELLED"
    ),
    account_size: Optional[int] = Query(None, description="Filter by account size"),
    limit: int = Query(50, ge=1, le=500),
):
    """List trades with optional filters."""
    if status and status.upper() == STATUS_OPEN:
        trades = get_open_trades(account_size)
    elif status and status.upper() == STATUS_CLOSED:
        trades = get_closed_trades(account_size)
    else:
        trades = get_all_trades(account_size)

    return trades[:limit]


@app.get("/trades/open", response_model=list[TradeResponse])
def api_open_trades(
    account_size: Optional[int] = Query(None, description="Filter by account size"),
):
    """List all open trades."""
    return get_open_trades(account_size)


@app.get("/trades/{trade_id}", response_model=TradeResponse)
def api_get_trade(trade_id: int):
    """Get a single trade by ID."""
    return _get_trade_by_id(trade_id)


@app.get("/today/pnl")
def api_today_pnl(
    account_size: Optional[int] = Query(None, description="Filter by account size"),
):
    """Get today's realised P&L."""
    pnl = get_today_pnl(account_size)
    return {"date": datetime.now(tz=_EST).strftime("%Y-%m-%d"), "pnl": pnl}


# ---------------------------------------------------------------------------
# Legacy endpoint (backwards compatible)
# ---------------------------------------------------------------------------


@app.post("/log_trade")
def log_trade(trade: LegacyTradeRequest):
    """Legacy endpoint: creates a trade and immediately closes it.

    Maintained for backwards compatibility with NinjaTrader integrations.
    """
    trade_id = create_trade(
        account_size=150_000,
        asset=trade.asset,
        direction=trade.direction.upper(),
        entry=trade.entry,
        sl=0.0,
        tp=0.0,
        contracts=trade.contracts,
        strategy=trade.strategy,
        notes=trade.notes,
    )

    # Immediately close with the provided exit price and P&L
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE trades_v2
           SET status = ?, close_price = ?, close_time = ?, pnl = ?, rr = 0
           WHERE id = ?""",
        (
            STATUS_CLOSED,
            trade.exit_price,
            datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S"),
            trade.pnl,
            trade_id,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "status": "logged",
        "trade_id": trade_id,
        "asset": trade.asset,
        "pnl": trade.pnl,
    }


# ---------------------------------------------------------------------------
# NinjaTrader Live Position Bridge
# ---------------------------------------------------------------------------


@app.post("/update_positions")
def api_update_positions(payload: NTPositionsPayload):
    """Receive live positions from NinjaTrader's LivePositionBridge indicator.

    The NinjaTrader indicator POSTs here on every position change (open,
    close, partial fill, PnL tick).  The payload is stored in the cache
    so the dashboard can read it instantly.

    Payload example (from NT indicator):
        {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 5,
                    "avgPrice": 6045.25,
                    "unrealizedPnL": 125.00,
                    "lastUpdate": "2025-01-15T14:30:00Z"
                }
            ],
            "timestamp": "2025-01-15T14:30:00Z"
        }
    """
    received_at = datetime.now(tz=_EST).isoformat()

    # Build the stored object
    stored = {
        "account": payload.account,
        "positions": [p.model_dump() for p in payload.positions],
        "timestamp": payload.timestamp or received_at,
        "received_at": received_at,
    }

    cache_set(_POSITIONS_CACHE_KEY, json.dumps(stored).encode(), _POSITIONS_TTL)

    total_pnl = sum(p.unrealizedPnL for p in payload.positions)
    open_count = len([p for p in payload.positions if p.quantity > 0])

    logger.info(
        "Position update from %s: %d open, unrealized P&L: $%.2f",
        payload.account,
        open_count,
        total_pnl,
    )

    return {
        "status": "ok",
        "positions_received": len(payload.positions),
        "open_positions": open_count,
        "total_unrealized_pnl": round(total_pnl, 2),
        "received_at": received_at,
    }


@app.get("/positions", response_model=NTPositionsResponse)
def api_get_positions():
    """Get current live positions from NinjaTrader.

    Returns the most recent position snapshot pushed by the
    LivePositionBridge indicator.  If no data has been received
    (or the cache has expired), returns an empty response.
    """
    raw = cache_get(_POSITIONS_CACHE_KEY)
    if raw is None:
        return NTPositionsResponse()

    try:
        data = json.loads(raw.decode())
        positions = [NTPosition(**p) for p in data.get("positions", [])]
        total_pnl = sum(p.unrealizedPnL for p in positions)

        return NTPositionsResponse(
            account=data.get("account", ""),
            positions=positions,
            timestamp=data.get("timestamp", ""),
            received_at=data.get("received_at", ""),
            has_positions=len(positions) > 0,
            total_unrealized_pnl=round(total_pnl, 2),
        )
    except Exception as exc:
        logger.error("Failed to parse cached positions: %s", exc)
        return NTPositionsResponse()


@app.delete("/positions")
def api_clear_positions():
    """Clear cached live positions (e.g. end of day reset).

    Useful when NinjaTrader is closed but stale position data
    remains in cache.
    """
    from src.futures_lib.core.cache import REDIS_AVAILABLE

    if REDIS_AVAILABLE:
        from src.futures_lib.core.cache import _r

        if _r is not None:
            _r.delete(_POSITIONS_CACHE_KEY)
    else:
        from src.futures_lib.core.cache import _mem_cache

        _mem_cache.pop(_POSITIONS_CACHE_KEY, None)

    return {"status": "cleared", "timestamp": datetime.now(tz=_EST).isoformat()}


# ---------------------------------------------------------------------------
# Helper: read live positions from cache (importable by other modules)
# ---------------------------------------------------------------------------


def get_live_positions() -> dict[str, Any]:
    """Read the latest NinjaTrader positions from cache.

    Returns a dict with keys: account, positions (list of dicts),
    timestamp, received_at, has_positions, total_unrealized_pnl.

    Importable by app.py and grok_helper.py without going through HTTP.
    """
    raw = cache_get(_POSITIONS_CACHE_KEY)
    if raw is None:
        return {
            "account": "",
            "positions": [],
            "timestamp": "",
            "received_at": "",
            "has_positions": False,
            "total_unrealized_pnl": 0.0,
        }

    try:
        data = json.loads(raw.decode())
        positions = data.get("positions", [])
        total_pnl = sum(p.get("unrealizedPnL", 0) for p in positions)
        return {
            "account": data.get("account", ""),
            "positions": positions,
            "timestamp": data.get("timestamp", ""),
            "received_at": data.get("received_at", ""),
            "has_positions": len(positions) > 0,
            "total_unrealized_pnl": round(total_pnl, 2),
        }
    except Exception:
        return {
            "account": "",
            "positions": [],
            "timestamp": "",
            "received_at": "",
            "has_positions": False,
            "total_unrealized_pnl": 0.0,
        }


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------


@app.get("/accounts")
def api_accounts():
    """List available account profiles and their risk parameters."""
    return ACCOUNT_PROFILES


@app.get("/assets")
def api_assets():
    """List available assets and their contract specifications."""
    return CONTRACT_SPECS


@app.get("/health")
def health():
    """Health check."""
    positions = get_live_positions()
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "nt_bridge": {
            "connected": positions["has_positions"],
            "account": positions["account"],
            "open_positions": len(positions["positions"]),
            "last_update": positions["received_at"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
