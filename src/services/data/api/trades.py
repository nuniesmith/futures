"""
Trades API router — trade CRUD endpoints.

Provides endpoints for creating, closing, cancelling, and listing trades.
Includes a legacy /log_trade endpoint for backwards compatibility with
older NinjaTrader scripts.

Position management (NinjaTrader live bridge) is handled by positions.py.
Asset/account info endpoints are handled by analysis.py.
"""

import logging
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from models import (
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

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.trades")

router = APIRouter(tags=["trades"])


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
