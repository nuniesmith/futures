"""
DOM (Depth of Market) API Routes — Placeholder
================================================
Provides REST and SSE endpoints for real-time depth-of-market data,
powering the DOM ladder widget on the dashboard.

Endpoints:
    GET /api/dom/snapshot?symbol=MES  — Current DOM state (mock data)
    GET /api/dom/config               — DOM display configuration
    GET /sse/dom?symbol=MES           — SSE stream of DOM updates (mock)

The snapshot and SSE endpoints return Level-2 order-book data structured
as bid/ask price ladders with size at each level.  When wired to the
live Rithmic Position Engine, these will reflect real-time market depth.

Usage:
    from lib.services.data.api.dom import router, sse_router

    app.include_router(router)
    app.include_router(sse_router)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "router",
    "sse_router",
]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/dom", tags=["dom"])
sse_router = APIRouter(tags=["dom-sse"])


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_SYMBOL = "MES"
_DEFAULT_LEVELS = 10
_TICK_SIZES: dict[str, float] = {
    "MES": 0.25,
    "MNQ": 0.25,
    "MGC": 0.10,
    "MCL": 0.01,
    "ES": 0.25,
    "NQ": 0.25,
    "GC": 0.10,
    "CL": 0.01,
}
_MOCK_BASE_PRICES: dict[str, float] = {
    "MES": 5420.00,
    "MNQ": 18950.00,
    "MGC": 2345.00,
    "MCL": 78.50,
    "ES": 5420.00,
    "NQ": 18950.00,
    "GC": 2345.00,
    "CL": 78.50,
}

_SSE_UPDATE_INTERVAL_S = 1.0
_SSE_HEARTBEAT_INTERVAL_S = 15.0


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


def _build_mock_snapshot(
    symbol: str,
    levels: int = _DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Build a mock DOM snapshot for *symbol*.

    Returns a dict with ``bids``, ``asks``, ``last``, ``spread``, and
    metadata fields ready for JSON serialisation.
    """
    # TODO: Wire to live Rithmic Position Engine data
    tick = _TICK_SIZES.get(symbol, 0.25)
    base = _MOCK_BASE_PRICES.get(symbol, 5000.00)

    best_bid = base - tick
    best_ask = base

    bids: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []

    for i in range(levels):
        bids.append(
            {
                "price": round(best_bid - i * tick, 4),
                "size": max(5, 60 - i * 4),
                "cumulative_size": 0,  # filled by the UI or a post-processing step
            }
        )
        asks.append(
            {
                "price": round(best_ask + i * tick, 4),
                "size": max(5, 55 - i * 4),
                "cumulative_size": 0,
            }
        )

    # Compute cumulative sizes
    running = 0
    for lvl in bids:
        running += lvl["size"]
        lvl["cumulative_size"] = running
    running = 0
    for lvl in asks:
        running += lvl["size"]
        lvl["cumulative_size"] = running

    bid_total = sum(b["size"] for b in bids)
    ask_total = sum(a["size"] for a in asks)

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "last": round(best_bid + tick / 2, 4),
        "spread": round(best_ask - best_bid, 4),
        "bid_total": bid_total,
        "ask_total": ask_total,
        "imbalance_ratio": round(bid_total / max(ask_total, 1), 3),
        "levels": levels,
        "timestamp": time.time(),
        "source": "mock",
    }


def _build_mock_config() -> dict[str, Any]:
    """Return DOM display configuration for the dashboard widget."""
    # TODO: Wire to user preferences / persisted settings
    return {
        "default_symbol": _DEFAULT_SYMBOL,
        "default_levels": _DEFAULT_LEVELS,
        "available_symbols": sorted(_TICK_SIZES.keys()),
        "tick_sizes": _TICK_SIZES,
        "color_scheme": {
            "bid_fill": "#0d6efd33",
            "ask_fill": "#dc354533",
            "bid_text": "#0d6efd",
            "ask_text": "#dc3545",
            "last_highlight": "#ffc107",
            "spread_bar": "#6c757d",
        },
        "update_interval_ms": int(_SSE_UPDATE_INTERVAL_S * 1000),
        "max_levels": 20,
        "show_cumulative": True,
        "show_imbalance_bar": True,
    }


# ---------------------------------------------------------------------------
# REST routes
# ---------------------------------------------------------------------------


@router.get("/snapshot")
async def dom_snapshot(
    symbol: str = Query(default=_DEFAULT_SYMBOL, description="Instrument symbol"),
    levels: int = Query(default=_DEFAULT_LEVELS, ge=1, le=50, description="Price levels per side"),
) -> dict[str, Any]:
    """Return a point-in-time DOM snapshot for *symbol*.

    Returns Level-2 bid/ask ladders with size, cumulative size,
    spread, and imbalance ratio.
    """
    # TODO: Wire to RithmicPositionEngine.get_l2() for live data
    logger.debug("dom_snapshot", symbol=symbol, levels=levels)
    return _build_mock_snapshot(symbol, levels=levels)


@router.get("/config")
async def dom_config() -> dict[str, Any]:
    """Return DOM display configuration for the dashboard widget.

    Includes colour scheme, available symbols, tick sizes, and
    default rendering options.
    """
    # TODO: Wire to user preferences / persisted settings
    logger.debug("dom_config")
    return _build_mock_config()


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


async def _dom_event_generator(
    symbol: str,
    levels: int,
    request: Request,
) -> AsyncGenerator[str]:
    """Yield SSE-formatted DOM updates at a regular interval.

    Each event is a JSON-encoded DOM snapshot.  A heartbeat comment
    (``:``) is sent periodically to keep the connection alive even if
    no data changes.
    """
    # TODO: Wire to live Rithmic stream — replace mock polling with
    #       real-time L2 push from RithmicPositionEngine
    logger.info("dom SSE stream started", symbol=symbol, levels=levels)
    last_heartbeat = time.monotonic()

    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("dom SSE client disconnected", symbol=symbol)
                break

            # Build a mock snapshot (in production, this would come from
            # a Redis pub/sub channel or a direct stream callback)
            snapshot = _build_mock_snapshot(symbol, levels=levels)
            payload = json.dumps(snapshot, default=str)

            yield f"event: dom-update\ndata: {payload}\n\n"

            # Heartbeat (SSE comment) to keep proxies / load-balancers happy
            now = time.monotonic()
            if now - last_heartbeat >= _SSE_HEARTBEAT_INTERVAL_S:
                yield ": heartbeat\n\n"
                last_heartbeat = now

            await asyncio.sleep(_SSE_UPDATE_INTERVAL_S)

    except asyncio.CancelledError:
        logger.info("dom SSE stream cancelled", symbol=symbol)
    except Exception:
        logger.exception("dom SSE stream error", symbol=symbol)
    finally:
        logger.info("dom SSE stream closed", symbol=symbol)


@sse_router.get("/sse/dom")
async def sse_dom(
    request: Request,
    symbol: str = Query(default=_DEFAULT_SYMBOL, description="Instrument symbol"),
    levels: int = Query(default=_DEFAULT_LEVELS, ge=1, le=50, description="Price levels per side"),
) -> StreamingResponse:
    """SSE endpoint streaming real-time DOM updates.

    Connect via ``EventSource("/sse/dom?symbol=MES")`` in the browser.
    Events are named ``dom-update`` and contain a JSON DOM snapshot.

    The stream sends an update roughly every second (configurable via
    ``dom_config.update_interval_ms``).
    """
    # TODO: Wire to live Rithmic stream when creds available
    logger.info("sse_dom connection opened", symbol=symbol, levels=levels)

    return StreamingResponse(
        _dom_event_generator(symbol, levels, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
