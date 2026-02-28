"""
Positions API router — NinjaTrader live position bridge (v2).

Handles receiving live position snapshots from NinjaTrader's Bridge
strategy and serving them to the dashboard.  Also provides proxy
endpoints that let the web dashboard send commands *back* to
NinjaTrader (execute signal, flatten, cancel orders).

Bridge v2 additions:
  - POST /positions/heartbeat    — Bridge keep-alive with account summary
  - POST /positions/execute      — Proxy: forward a trade signal to Bridge
  - POST /positions/flatten      — Proxy: flatten all positions via Bridge
  - POST /positions/cancel_orders — Proxy: cancel working orders via Bridge
  - GET  /positions/bridge_status — Read Bridge /status (account info)
  - GET  /positions/bridge_orders — Read Bridge /orders (recent order events)

Risk enforcement:
  When NinjaTrader pushes a position snapshot via POST /positions/update,
  the router syncs the positions into the local RiskManager and evaluates
  all risk rules.  The response includes risk status fields so the NT8
  Bridge knows immediately if trading limits have been hit.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.lib.core.cache import (  # noqa: PLC2701
    REDIS_AVAILABLE,
    _cache_key,
    cache_get,
    cache_set,
)

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.positions")

router = APIRouter(tags=["Positions"])

# ---------------------------------------------------------------------------
# Cache key & TTL for live positions from NinjaTrader
# ---------------------------------------------------------------------------
_POSITIONS_CACHE_KEY = _cache_key("live_positions", "current")
_POSITIONS_TTL = 7200  # 2 hours — positions persist across brief disconnects

# Bridge heartbeat cache (separate from positions — shorter TTL)
_HEARTBEAT_CACHE_KEY = _cache_key("bridge_heartbeat", "current")
_HEARTBEAT_TTL = 60  # 1 minute — if no heartbeat within 60s, bridge is "stale"

# NinjaTrader Bridge listener URL (the Bridge runs an HttpListener on this port)
_BRIDGE_HOST = os.getenv("NT_BRIDGE_HOST", "localhost")
_BRIDGE_PORT = int(os.getenv("NT_BRIDGE_PORT", "8080"))
_BRIDGE_BASE_URL = f"http://{_BRIDGE_HOST}:{_BRIDGE_PORT}"
_BRIDGE_TIMEOUT = 5.0  # seconds


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
    instrument: Optional[str] = Field(
        None, description="Master instrument name, e.g. 'MES'"
    )
    tickSize: Optional[float] = Field(None, description="Tick size for this instrument")
    pointValue: Optional[float] = Field(
        None, description="Point value for this instrument"
    )
    lastUpdate: Optional[str] = Field(None, description="ISO timestamp of last update")


class NTPendingOrder(BaseModel):
    """A working/accepted order from NinjaTrader."""

    orderId: str = Field("", description="NinjaTrader order ID")
    name: str = Field("", description="Order name/label")
    instrument: str = Field("", description="Instrument full name")
    action: str = Field("", description="Buy, Sell, SellShort, BuyToCover")
    type: str = Field("", description="Market, Limit, StopMarket, etc.")
    quantity: int = Field(0, description="Order quantity")
    limitPrice: float = Field(0.0, description="Limit price (0 if not applicable)")
    stopPrice: float = Field(0.0, description="Stop price (0 if not applicable)")
    state: str = Field("", description="Order state: Working, Accepted, etc.")


class NTPositionsPayload(BaseModel):
    """Payload pushed by the Bridge NinjaTrader strategy (v2).

    Bridge v2 adds: cashBalance, realizedPnL, pendingOrders,
    totalUnrealizedPnL, riskBlocked, riskBlockReason, bridge_version.
    """

    account: str = Field(..., description="NinjaTrader account name, e.g. 'Sim101'")
    positions: List[NTPosition] = Field(
        default_factory=list, description="List of open positions"
    )
    pendingOrders: List[NTPendingOrder] = Field(
        default_factory=list, description="Working/accepted orders"
    )
    timestamp: Optional[str] = Field(None, description="UTC timestamp from NT")
    cashBalance: float = Field(0.0, description="Account cash balance")
    realizedPnL: float = Field(0.0, description="Today's realized P&L")
    totalUnrealizedPnL: float = Field(0.0, description="Sum of all unrealized P&L")
    riskBlocked: bool = Field(
        False, description="True if Bridge risk enforcement is blocking new trades"
    )
    riskBlockReason: str = Field("", description="Reason for risk block (if any)")
    bridge_version: str = Field("1.0", description="Bridge version string")


class NTHeartbeatPayload(BaseModel):
    """Lightweight heartbeat from the Bridge strategy."""

    account: str = Field(..., description="NinjaTrader account name")
    state: str = Field("", description="Strategy state (e.g. Realtime)")
    connected: bool = Field(True, description="Whether the account is connected")
    positions: int = Field(0, description="Number of open positions")
    cashBalance: float = Field(0.0, description="Account cash balance")
    riskBlocked: bool = Field(False, description="Whether risk enforcement is blocking")
    bridge_version: str = Field("1.0", description="Bridge version")
    listenerPort: int = Field(8080, description="Bridge HTTP listener port")
    timestamp: Optional[str] = Field(None, description="UTC timestamp")


class NTPositionsResponse(BaseModel):
    """Response returned by GET /."""

    account: str = ""
    positions: List[NTPosition] = []
    timestamp: str = ""
    received_at: str = ""
    has_positions: bool = False
    total_unrealized_pnl: float = 0.0
    cash_balance: float = 0.0
    realized_pnl: float = 0.0
    pending_orders: List[NTPendingOrder] = []
    bridge_connected: bool = False
    bridge_version: str = ""


class ExecuteSignalRequest(BaseModel):
    """Request body for sending a trade signal to NinjaTrader via the Bridge."""

    direction: str = Field(..., description="'long' or 'short'")
    quantity: int = Field(
        1, ge=1, description="Number of contracts (will be risk-sized by Bridge)"
    )
    order_type: str = Field("market", description="'market', 'limit', or 'stop'")
    limit_price: float = Field(0.0, description="Limit price (for limit/stop orders)")
    stop_loss: float = Field(
        0.0, description="Exact stop loss price (0 = use Bridge default)"
    )
    take_profit: float = Field(
        0.0, description="Exact take profit price (0 = use Bridge default)"
    )
    tp2: float = Field(0.0, description="Second take profit target (0 = none)")
    strategy: str = Field("", description="Strategy name for logging")
    asset: str = Field("", description="Asset name for logging")
    signal_id: str = Field(
        "", description="Unique signal ID for tracking (auto-generated if empty)"
    )
    enforce_risk: bool = Field(
        True,
        description="If True, run a pre-flight risk check before forwarding to Bridge",
    )


class FlattenRequest(BaseModel):
    """Request body for flattening all positions via the Bridge."""

    reason: str = Field(
        "dashboard", description="Reason for flattening (for audit trail)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: forward HTTP requests to the NinjaTrader Bridge listener
# ---------------------------------------------------------------------------


def _get_bridge_url() -> str:
    """Return the base URL for the NinjaTrader Bridge HTTP listener.

    Reads from the latest heartbeat (which includes the actual listener
    port) or falls back to the environment variable / default.
    """
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw:
            hb = json.loads(raw)
            port = hb.get("listenerPort", _BRIDGE_PORT)
            return f"http://{_BRIDGE_HOST}:{port}"
    except Exception:
        pass
    return _BRIDGE_BASE_URL


def _is_bridge_alive() -> bool:
    """Check whether we've received a heartbeat recently."""
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw is None:
            return False
        hb = json.loads(raw)
        received = hb.get("received_at", "")
        if not received:
            return False
        dt = datetime.fromisoformat(received)
        age = (datetime.now(tz=_EST) - dt).total_seconds()
        return age < _HEARTBEAT_TTL
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Endpoints — NinjaTrader → Python (push)
# ---------------------------------------------------------------------------


@router.post("/update")
def update_positions(payload: NTPositionsPayload):
    """Receive live position snapshot from NinjaTrader's Bridge strategy.

    The Bridge POSTs here on every position change (open, close,
    partial fill, P&L tick).  Bridge v2 also sends account balance,
    realized P&L, pending orders, and risk-block state.

    The response includes risk evaluation fields so the Bridge
    knows immediately if any risk limits have been reached.
    """
    received_at = datetime.now(tz=_EST).isoformat()

    position_dicts = [p.model_dump() for p in payload.positions]
    pending_order_dicts = [o.model_dump() for o in payload.pendingOrders]

    data = {
        "account": payload.account,
        "positions": position_dicts,
        "pendingOrders": pending_order_dicts,
        "timestamp": payload.timestamp or received_at,
        "received_at": received_at,
        "cashBalance": payload.cashBalance,
        "realizedPnL": payload.realizedPnL,
        "totalUnrealizedPnL": payload.totalUnrealizedPnL,
        "riskBlocked": payload.riskBlocked,
        "riskBlockReason": payload.riskBlockReason,
        "bridge_version": payload.bridge_version,
    }

    cache_set(
        _POSITIONS_CACHE_KEY,
        json.dumps(data).encode(),
        _POSITIONS_TTL,
    )

    total_pnl = sum(p.unrealizedPnL for p in payload.positions)
    open_count = len([p for p in payload.positions if p.quantity > 0])

    logger.info(
        "Position update: account=%s positions=%d total_pnl=%.2f balance=%.2f bridge=%s",
        payload.account,
        open_count,
        total_pnl,
        payload.cashBalance,
        payload.bridge_version,
    )

    # --- Risk evaluation ---
    risk_status: Dict[str, Any] = {}
    try:
        from src.lib.services.data.api.risk import evaluate_position_risk

        risk_status = evaluate_position_risk(position_dicts)

        if not risk_status.get("can_trade", True):
            logger.warning(
                "⚠️ Risk block after position sync: %s (daily P&L $%.2f)",
                risk_status.get("block_reason", ""),
                risk_status.get("daily_pnl", 0.0),
            )
        for warning in risk_status.get("warnings", []):
            logger.warning("⚠️ Risk warning: %s", warning)
    except Exception as exc:
        logger.debug("Risk evaluation skipped (non-fatal): %s", exc)

    return {
        "status": "received",
        "account": payload.account,
        "positions_count": len(payload.positions),
        "open_positions": open_count,
        "total_unrealized_pnl": round(total_pnl, 2),
        "received_at": received_at,
        "risk": risk_status,
    }


@router.post("/heartbeat")
def receive_heartbeat(payload: NTHeartbeatPayload):
    """Receive a keep-alive heartbeat from the NinjaTrader Bridge.

    The Bridge sends this every ~15 seconds so the dashboard knows
    the connection is alive even when there are no position changes.
    The heartbeat also carries the Bridge's listener port so the
    execute/flatten proxy endpoints know where to forward requests.
    """
    received_at = datetime.now(tz=_EST).isoformat()

    hb_data = {
        "account": payload.account,
        "state": payload.state,
        "connected": payload.connected,
        "positions": payload.positions,
        "cashBalance": payload.cashBalance,
        "riskBlocked": payload.riskBlocked,
        "bridge_version": payload.bridge_version,
        "listenerPort": payload.listenerPort,
        "received_at": received_at,
        "timestamp": payload.timestamp or received_at,
    }

    cache_set(
        _HEARTBEAT_CACHE_KEY,
        json.dumps(hb_data).encode(),
        _HEARTBEAT_TTL,
    )

    logger.debug(
        "Bridge heartbeat: account=%s state=%s positions=%d port=%d",
        payload.account,
        payload.state,
        payload.positions,
        payload.listenerPort,
    )

    return {
        "status": "ok",
        "received_at": received_at,
    }


# ---------------------------------------------------------------------------
# Endpoints — Python → NinjaTrader (proxy to Bridge listener)
# ---------------------------------------------------------------------------


@router.post("/execute")
def execute_signal(req: ExecuteSignalRequest):
    """Send a trade signal to NinjaTrader via the Bridge's HTTP listener.

    This is the primary way the web dashboard triggers order execution.
    The signal is forwarded to the Bridge's ``/execute_signal`` endpoint,
    which queues it for main-thread execution inside NinjaTrader.

    Optionally runs a pre-flight risk check before forwarding.
    """
    if not _is_bridge_alive():
        raise HTTPException(
            status_code=503,
            detail="NinjaTrader Bridge is not connected (no recent heartbeat)",
        )

    # --- Optional pre-flight risk check ---
    if req.enforce_risk:
        try:
            from src.lib.services.data.api.risk import check_trade_entry_risk

            allowed, reason, details = check_trade_entry_risk(
                symbol=req.asset or "UNKNOWN",
                side=req.direction.upper(),
                size=req.quantity,
            )
            if not allowed:
                return {
                    "status": "rejected",
                    "reason": f"Risk check failed: {reason}",
                    "risk_details": details,
                }
        except Exception as exc:
            logger.debug("Pre-flight risk check unavailable (non-fatal): %s", exc)

    # --- Forward to NinjaTrader Bridge ---
    bridge_url = _get_bridge_url()
    signal_payload = req.model_dump()

    try:
        with httpx.Client(timeout=_BRIDGE_TIMEOUT) as client:
            resp = client.post(f"{bridge_url}/execute_signal", json=signal_payload)
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = bridge_url
            return result
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to NinjaTrader Bridge at {bridge_url}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"NinjaTrader Bridge at {bridge_url} timed out",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bridge communication error: {exc}",
        )


@router.post("/flatten")
def flatten_all(req: FlattenRequest):
    """Flatten all positions by forwarding to the Bridge's /flatten endpoint.

    Closes all open positions at market and cancels all working orders.
    """
    if not _is_bridge_alive():
        raise HTTPException(
            status_code=503,
            detail="NinjaTrader Bridge is not connected (no recent heartbeat)",
        )

    bridge_url = _get_bridge_url()

    try:
        with httpx.Client(timeout=_BRIDGE_TIMEOUT) as client:
            resp = client.post(
                f"{bridge_url}/flatten",
                json={"reason": req.reason},
            )
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = bridge_url
            logger.warning("🔴 FLATTEN ALL sent to Bridge — reason: %s", req.reason)
            return result
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to NinjaTrader Bridge at {bridge_url}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bridge communication error: {exc}",
        )


@router.post("/cancel_orders")
def cancel_orders():
    """Cancel all working orders by forwarding to the Bridge's /cancel_orders endpoint."""
    if not _is_bridge_alive():
        raise HTTPException(
            status_code=503,
            detail="NinjaTrader Bridge is not connected (no recent heartbeat)",
        )

    bridge_url = _get_bridge_url()

    try:
        with httpx.Client(timeout=_BRIDGE_TIMEOUT) as client:
            resp = client.post(f"{bridge_url}/cancel_orders")
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = bridge_url
            logger.info("Cancel all orders sent to Bridge")
            return result
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to NinjaTrader Bridge at {bridge_url}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bridge communication error: {exc}",
        )


@router.get("/bridge_status")
def get_bridge_status():
    """Read the NinjaTrader Bridge's /status endpoint.

    Returns account info, position count, balance, risk state,
    and Bridge configuration.  Also includes heartbeat age to
    indicate connection freshness.
    """
    # Return cached heartbeat data + live status if Bridge is reachable
    heartbeat_data = {}
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw:
            heartbeat_data = json.loads(raw)
    except Exception:
        pass

    bridge_alive = _is_bridge_alive()

    # Try to fetch live status from Bridge
    bridge_status = {}
    if bridge_alive:
        bridge_url = _get_bridge_url()
        try:
            with httpx.Client(timeout=_BRIDGE_TIMEOUT) as client:
                resp = client.get(f"{bridge_url}/status")
                resp.raise_for_status()
                bridge_status = resp.json()
        except Exception as exc:
            logger.debug("Could not fetch Bridge /status: %s", exc)

    return {
        "bridge_alive": bridge_alive,
        "heartbeat": heartbeat_data,
        "live_status": bridge_status,
        "bridge_url": _get_bridge_url(),
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


@router.get("/bridge_orders")
def get_bridge_orders():
    """Read the NinjaTrader Bridge's /orders endpoint.

    Returns recent order events (fills, rejects, cancellations)
    tracked by the Bridge for audit and display on the dashboard.
    """
    if not _is_bridge_alive():
        return {
            "bridge_alive": False,
            "events": [],
            "error": "NinjaTrader Bridge is not connected",
        }

    bridge_url = _get_bridge_url()
    try:
        with httpx.Client(timeout=_BRIDGE_TIMEOUT) as client:
            resp = client.get(f"{bridge_url}/orders")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {
            "bridge_alive": True,
            "events": [],
            "error": f"Could not fetch orders: {exc}",
        }


# ---------------------------------------------------------------------------
# Endpoints — Dashboard reads (GET)
# ---------------------------------------------------------------------------


@router.get("/", response_model=NTPositionsResponse)
def get_positions():
    """Get current live positions from NinjaTrader.

    Returns the most recent position snapshot pushed by the
    Bridge strategy.  If no data has been received (or the cache
    has expired), returns an empty response.

    Bridge v2 payloads include account balance, realized P&L,
    and pending orders alongside the position list.
    """
    raw = cache_get(_POSITIONS_CACHE_KEY)
    if raw is None:
        return NTPositionsResponse()

    try:
        data = json.loads(raw.decode())
        positions = [NTPosition(**p) for p in data.get("positions", [])]
        pending = [NTPendingOrder(**o) for o in data.get("pendingOrders", [])]
        total_pnl = sum(p.unrealizedPnL for p in positions)

        return NTPositionsResponse(
            account=data.get("account", ""),
            positions=positions,
            timestamp=data.get("timestamp", ""),
            received_at=data.get("received_at", ""),
            has_positions=len(positions) > 0,
            total_unrealized_pnl=round(total_pnl, 2),
            cash_balance=data.get("cashBalance", 0.0),
            realized_pnl=data.get("realizedPnL", 0.0),
            pending_orders=pending,
            bridge_connected=_is_bridge_alive(),
            bridge_version=data.get("bridge_version", ""),
        )
    except Exception as exc:
        logger.error("Failed to parse cached positions: %s", exc)
        return NTPositionsResponse()


@router.delete("/")
def clear_positions():
    """Clear cached live positions and heartbeat (e.g. end of day reset).

    Useful when NinjaTrader is closed but stale position data
    remains in cache.
    """
    keys_to_clear = [_POSITIONS_CACHE_KEY, _HEARTBEAT_CACHE_KEY]

    if REDIS_AVAILABLE:
        from src.lib.core.cache import _r

        if _r is not None:
            for key in keys_to_clear:
                _r.delete(key)
    else:
        from src.lib.core.cache import _mem_cache

        for key in keys_to_clear:
            _mem_cache.pop(key, None)

    return {"status": "cleared", "timestamp": datetime.now(tz=_EST).isoformat()}


# ---------------------------------------------------------------------------
# Helper: read live positions from cache (importable by other modules)
# ---------------------------------------------------------------------------


def get_live_positions() -> dict[str, Any]:
    """Read the latest NinjaTrader positions from cache.

    Returns a dict with keys: account, positions (list of dicts),
    timestamp, received_at, has_positions, total_unrealized_pnl,
    cash_balance, realized_pnl, pending_orders, bridge_connected,
    bridge_version.

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
            "cash_balance": 0.0,
            "realized_pnl": 0.0,
            "pending_orders": [],
            "bridge_connected": False,
            "bridge_version": "",
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
            "cash_balance": data.get("cashBalance", 0.0),
            "realized_pnl": data.get("realizedPnL", 0.0),
            "pending_orders": data.get("pendingOrders", []),
            "bridge_connected": _is_bridge_alive(),
            "bridge_version": data.get("bridge_version", ""),
        }
    except Exception:
        return {
            "account": "",
            "positions": [],
            "timestamp": "",
            "received_at": "",
            "has_positions": False,
            "total_unrealized_pnl": 0.0,
            "cash_balance": 0.0,
            "realized_pnl": 0.0,
            "pending_orders": [],
            "bridge_connected": False,
            "bridge_version": "",
        }
