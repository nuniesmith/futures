"""
Trades & Positions API router.

Migrated from the original src/api_server.py — all trade CRUD endpoints
and the NinjaTrader live position bridge live here now.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from core.cache import REDIS_AVAILABLE, _cache_key, cache_get, cache_set
from core.models import (
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
)
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.trades")

router = APIRouter(tags=["trades"])

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
# Helpers
# ---------------------------------------------------------------------------


def _get_trade_by_id(trade_id: int) -> dict:
    """Look up a trade by ID from all trades, raise 404 if not found."""
    all_trades = get_all_trades()
    for t in all_trades:
        if t["id"] == trade_id:
            return t
    raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")


# ---------------------------------------------------------------------------
# Trade CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/trades", response_model=TradeResponse, status_code=201)
def api_create_trade(req: CreateTradeRequest):
    """Open a new trade."""
    trade_id = create_trade(
        account_size=req.account_size,
        asset=req.asset,
        direction=req.direction.upper(),
        entry=req.entry,
        sl=req.sl,
        tp=req.tp,
        contracts=req.contracts,
        strategy=req.strategy,
        notes=req.notes,
    )
    trade = _get_trade_by_id(trade_id)
    return TradeResponse(**trade)


@router.post("/trades/{trade_id}/close", response_model=TradeResponse)
def api_close_trade(trade_id: int, req: CloseTradeRequest):
    """Close an open trade with exit price."""
    close_trade(trade_id, req.close_price)
    trade = _get_trade_by_id(trade_id)
    return TradeResponse(**trade)


@router.post("/trades/{trade_id}/cancel")
def api_cancel_trade(trade_id: int):
    """Cancel an open trade (no fill)."""
    cancel_trade(trade_id)
    trade = _get_trade_by_id(trade_id)
    return {
        "status": "cancelled",
        "trade_id": trade_id,
        "trade": trade,
    }


@router.get("/trades", response_model=List[TradeResponse])
def api_list_trades(
    status: Optional[str] = Query(None, description="Filter: open, closed"),
    account_size: Optional[int] = Query(None, description="Filter by account size"),
):
    """List trades, optionally filtered by status and account size."""
    if status == "open":
        trades = get_open_trades(account_size=account_size or 150_000)
    elif status == "closed":
        trades = get_closed_trades(account_size=account_size or 150_000)
    else:
        trades = get_all_trades()
        if account_size:
            trades = [t for t in trades if t.get("account_size") == account_size]

    return [TradeResponse(**t) for t in trades]


@router.get("/trades/open", response_model=List[TradeResponse])
def api_open_trades(
    account_size: int = Query(150_000, description="Account size"),
):
    """List currently open trades."""
    trades = get_open_trades(account_size=account_size)
    return [TradeResponse(**t) for t in trades]


@router.get("/trades/{trade_id}", response_model=TradeResponse)
def api_get_trade(trade_id: int):
    """Get a single trade by ID."""
    return TradeResponse(**_get_trade_by_id(trade_id))


@router.get("/trades/today/pnl")
def api_today_pnl(account_size: int = Query(150_000)):
    """Get today's net P&L."""
    pnl = get_today_pnl(account_size)
    return {"date": datetime.now(tz=_EST).strftime("%Y-%m-%d"), "pnl": pnl}


# ---------------------------------------------------------------------------
# Legacy endpoint (backwards compatibility)
# ---------------------------------------------------------------------------


@router.post("/log_trade")
def log_trade(req: LegacyTradeRequest):
    """Legacy: create and immediately close a trade in one call.

    Kept for backwards compatibility with older NinjaTrader scripts.
    """
    trade_id = create_trade(
        account_size=150_000,
        asset=req.asset,
        direction=req.direction.upper(),
        entry=req.entry,
        sl=None,
        tp=None,
        contracts=req.contracts,
        strategy=req.strategy,
        notes=req.notes,
    )
    close_trade(trade_id, req.exit_price)
    trade = _get_trade_by_id(trade_id)
    return {
        "status": "logged",
        "trade_id": trade_id,
        "pnl": trade.get("pnl", 0),
        "trade": trade,
    }


# ---------------------------------------------------------------------------
# NinjaTrader live position bridge
# ---------------------------------------------------------------------------


@router.post("/update_positions")
def api_update_positions(payload: NTPositionsPayload):
    """Receive live position snapshot from NinjaTrader's LivePositionBridge.

    The NinjaTrader indicator POSTs here every few seconds with the
    current state of all open positions in the account.
    """
    received_at = datetime.now(tz=_EST).isoformat()

    data = {
        "account": payload.account,
        "positions": [p.model_dump() for p in payload.positions],
        "timestamp": payload.timestamp or received_at,
        "received_at": received_at,
    }

    cache_set(
        _POSITIONS_CACHE_KEY,
        json.dumps(data).encode(),
        _POSITIONS_TTL,
    )

    logger.info(
        "Position update: account=%s positions=%d total_pnl=%.2f",
        payload.account,
        len(payload.positions),
        sum(p.unrealizedPnL for p in payload.positions),
    )

    return {
        "status": "received",
        "account": payload.account,
        "positions_count": len(payload.positions),
        "received_at": received_at,
    }


@router.get("/positions", response_model=NTPositionsResponse)
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


@router.delete("/positions")
def api_clear_positions():
    """Clear cached live positions (e.g. end of day reset).

    Useful when NinjaTrader is closed but stale position data
    remains in cache.
    """
    if REDIS_AVAILABLE:
        from core.cache import _r

        if _r is not None:
            _r.delete(_POSITIONS_CACHE_KEY)
    else:
        from core.cache import _mem_cache

        _mem_cache.pop(_POSITIONS_CACHE_KEY, None)

    return {"status": "cleared", "timestamp": datetime.now(tz=_EST).isoformat()}


# ---------------------------------------------------------------------------
# Helper: read live positions from cache (importable by other modules)
# ---------------------------------------------------------------------------


def get_live_positions() -> dict:
    """Read the latest NinjaTrader positions from cache.

    Returns a dict with keys: account, positions (list of dicts),
    timestamp, received_at, has_positions, total_unrealized_pnl.

    Importable by other modules without going through HTTP.
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


@router.get("/accounts")
def api_accounts():
    """List available account profiles and their risk parameters."""
    return ACCOUNT_PROFILES


@router.get("/assets")
def api_assets():
    """List available assets and their contract specifications."""
    return CONTRACT_SPECS
