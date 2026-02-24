"""
FastAPI server for receiving trade data from NinjaTrader and external tools.

Run alongside the Streamlit dashboard:
    python api_server.py

Listens on port 8000. Supports:
    POST /trades          - Create a new OPEN trade
    POST /trades/{id}/close - Close a trade with exit price
    POST /trades/{id}/cancel - Cancel an open trade
    POST /log_trade       - Legacy endpoint (creates and immediately closes)
    GET  /trades          - List trades (filterable by status, account_size)
    GET  /trades/open     - List open trades
    GET  /trades/{id}     - Get single trade
    GET  /health          - Health check
"""

import os
import sqlite3
import sys
from datetime import datetime
from typing import Optional

# Ensure sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from models import (
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

app = FastAPI(
    title="Futures Dashboard Trade API",
    description="REST API for trade management — supports $50k / $100k / $150k accounts",
    version="2.0.0",
)

# Initialise DB on startup
init_db()


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
# Helper to read a single trade
# ---------------------------------------------------------------------------


def _get_trade_by_id(trade_id: int) -> dict:
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
        df = get_open_trades(account_size)
    elif status and status.upper() == STATUS_CLOSED:
        df = get_closed_trades(account_size)
    else:
        df = get_all_trades(account_size)

    df = df.head(limit)
    # Replace NaN with None for JSON serialization
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


@app.get("/trades/open", response_model=list[TradeResponse])
def api_open_trades(
    account_size: Optional[int] = Query(None, description="Filter by account size"),
):
    """List all open trades."""
    df = get_open_trades(account_size)
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


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
    return {"date": datetime.now().strftime("%Y-%m-%d"), "pnl": pnl}


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
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
