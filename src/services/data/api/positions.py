"""
Positions API router — NinjaTrader live position bridge.

Handles receiving live position snapshots from NinjaTrader's
LivePositionBridge indicator and serving them to the Streamlit UI.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from pydantic import BaseModel, Field

from cache import REDIS_AVAILABLE, _cache_key, cache_get, cache_set

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.positions")

router = APIRouter(tags=["Positions"])

# ---------------------------------------------------------------------------
# Cache key & TTL for live positions from NinjaTrader
# ---------------------------------------------------------------------------
_POSITIONS_CACHE_KEY = _cache_key("live_positions", "current")
_POSITIONS_TTL = 7200  # 2 hours — positions persist across brief disconnects


# ---------------------------------------------------------------------------
# Request / Response models
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
    """Response returned by GET /."""

    account: str = ""
    positions: List[NTPosition] = []
    timestamp: str = ""
    received_at: str = ""
    has_positions: bool = False
    total_unrealized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/update")
def update_positions(payload: NTPositionsPayload):
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


@router.get("/", response_model=NTPositionsResponse)
def get_positions():
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


@router.delete("/")
def clear_positions():
    """Clear cached live positions (e.g. end of day reset).

    Useful when NinjaTrader is closed but stale position data
    remains in cache.
    """
    if REDIS_AVAILABLE:
        from cache import _r

        if _r is not None:
            _r.delete(_POSITIONS_CACHE_KEY)
    else:
        from cache import _mem_cache

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
