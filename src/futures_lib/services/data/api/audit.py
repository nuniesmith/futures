"""
Audit API router — persistent risk/ORB event history endpoints.

Provides durable audit trail endpoints backed by the database (Postgres
or SQLite), complementing the in-memory / Redis-based risk history in
``risk.py``.

Endpoints:
  - GET  /audit/risk          — query persisted risk events
  - GET  /audit/orb           — query persisted ORB events
  - GET  /audit/summary       — aggregated summary for last N days
  - POST /audit/risk          — manually record a risk event (internal use)
  - POST /audit/orb           — manually record an ORB event (internal use)

These tables are created automatically by ``models.init_db()`` on startup.
The engine service writes events via ``models.record_risk_event()`` and
``models.record_orb_event()`` during its CHECK_RISK_RULES and CHECK_ORB
handlers.  This router provides read access for dashboards and external
consumers, plus write access for manual / programmatic event injection.

Usage:
    # In main.py:
    from src.futures_lib.services.data.api.audit import router as audit_router
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure bare imports resolve

logger = logging.getLogger("api.audit")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Audit"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RiskEventCreate(BaseModel):
    """Request body for manually recording a risk event."""

    event_type: str = Field(
        ..., description="Event type: 'block', 'warning', 'clear', 'circuit_breaker'"
    )
    symbol: str = Field("", description="Instrument symbol, e.g. 'MGC'")
    side: str = Field("", description="LONG or SHORT")
    reason: str = Field("", description="Human-readable reason")
    daily_pnl: float = Field(0.0, description="Daily P&L at time of event")
    open_trades: int = Field(0, ge=0, description="Open trade count")
    account_size: int = Field(0, ge=0, description="Account size")
    risk_pct: float = Field(0.0, ge=0, description="Risk as % of account")
    session: str = Field("", description="Session mode (pre_market, active, off_hours)")
    metadata: Optional[dict] = Field(None, description="Optional extra data")


class ORBEventCreate(BaseModel):
    """Request body for manually recording an ORB event."""

    symbol: str = Field(..., description="Instrument symbol")
    or_high: float = Field(0.0, description="Opening range high")
    or_low: float = Field(0.0, description="Opening range low")
    or_range: float = Field(0.0, description="OR high - OR low")
    atr_value: float = Field(0.0, description="ATR value")
    breakout_detected: bool = Field(False, description="Was a breakout detected?")
    direction: str = Field("", description="LONG, SHORT, or empty")
    trigger_price: float = Field(0.0, description="Breakout trigger price")
    long_trigger: float = Field(0.0, description="Upper breakout level")
    short_trigger: float = Field(0.0, description="Lower breakout level")
    bar_count: int = Field(0, ge=0, description="Bars in the opening range")
    session: str = Field("", description="Session mode")
    metadata: Optional[dict] = Field(None, description="Optional extra data")


class RiskEventResponse(BaseModel):
    """A single risk event from the audit trail."""

    id: Optional[int] = None
    timestamp: str = ""
    event_type: str = ""
    symbol: str = ""
    side: str = ""
    reason: str = ""
    daily_pnl: float = 0.0
    open_trades: int = 0
    account_size: int = 0
    risk_pct: float = 0.0
    session: str = ""
    metadata_json: str = "{}"


class ORBEventResponse(BaseModel):
    """A single ORB event from the audit trail."""

    id: Optional[int] = None
    timestamp: str = ""
    symbol: str = ""
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr_value: float = 0.0
    breakout_detected: int = 0
    direction: str = ""
    trigger_price: float = 0.0
    long_trigger: float = 0.0
    short_trigger: float = 0.0
    bar_count: int = 0
    session: str = ""
    metadata_json: str = "{}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/risk")
def get_risk_events(
    limit: int = Query(50, ge=1, le=1000, description="Max events to return"),
    event_type: Optional[str] = Query(
        None, description="Filter by event type (block, warning, clear)"
    ),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    since: Optional[str] = Query(
        None, description="ISO timestamp — only return events after this time"
    ),
):
    """Query persisted risk events from the database.

    Returns the most recent events matching the filters, ordered by
    timestamp descending (most recent first).

    These events are written by the engine's CHECK_RISK_RULES handler
    and by the risk pre-flight check endpoint.
    """
    try:
        from src.futures_lib.core.models import get_risk_events as _get_risk_events

        events = _get_risk_events(
            limit=limit,
            event_type=event_type,
            symbol=symbol,
            since=since,
        )

        # Parse metadata_json for each event
        for ev in events:
            if "metadata_json" in ev:
                try:
                    ev["metadata"] = json.loads(ev["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    ev["metadata"] = {}

        return {
            "events": events,
            "count": len(events),
            "limit": limit,
            "filters": {
                "event_type": event_type,
                "symbol": symbol,
                "since": since,
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        logger.error("Failed to query risk events: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to query risk events: {exc}"
        )


@router.get("/orb")
def get_orb_events(
    limit: int = Query(50, ge=1, le=1000, description="Max events to return"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    breakout_only: bool = Query(
        False, description="Only return events where breakout was detected"
    ),
    since: Optional[str] = Query(
        None, description="ISO timestamp — only return events after this time"
    ),
):
    """Query persisted ORB events from the database.

    Returns the most recent ORB evaluations matching the filters,
    ordered by timestamp descending (most recent first).

    These events are written by the engine's CHECK_ORB handler.
    """
    try:
        from src.futures_lib.core.models import get_orb_events as _get_orb_events

        events = _get_orb_events(
            limit=limit,
            symbol=symbol,
            breakout_only=breakout_only,
            since=since,
        )

        # Parse metadata_json and convert breakout_detected to bool
        for ev in events:
            if "metadata_json" in ev:
                try:
                    ev["metadata"] = json.loads(ev["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    ev["metadata"] = {}
            # Convert integer breakout_detected to boolean for API consumers
            if "breakout_detected" in ev:
                ev["breakout_detected_bool"] = bool(ev["breakout_detected"])

        return {
            "events": events,
            "count": len(events),
            "limit": limit,
            "filters": {
                "symbol": symbol,
                "breakout_only": breakout_only,
                "since": since,
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        logger.error("Failed to query ORB events: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to query ORB events: {exc}"
        )


@router.get("/summary")
def get_audit_summary(
    days: int = Query(7, ge=1, le=365, description="Number of days to summarise"),
):
    """Get an aggregated summary of audit events for the last N days.

    Returns counts and breakdowns for both risk and ORB events,
    useful for dashboard widgets and reporting.
    """
    try:
        from src.futures_lib.core.models import get_audit_summary as _get_audit_summary

        summary = _get_audit_summary(days_back=days)
        summary["timestamp"] = datetime.now(tz=_EST).isoformat()
        return summary
    except Exception as exc:
        logger.error("Failed to build audit summary: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to build audit summary: {exc}"
        )


@router.post("/risk", status_code=201)
def create_risk_event(req: RiskEventCreate):
    """Manually record a risk event for the audit trail.

    This endpoint is primarily for internal use — the engine records
    events automatically.  Use this for manual annotations, testing,
    or external system integrations.
    """
    try:
        from src.futures_lib.core.models import record_risk_event

        row_id = record_risk_event(
            event_type=req.event_type,
            symbol=req.symbol,
            side=req.side,
            reason=req.reason,
            daily_pnl=req.daily_pnl,
            open_trades=req.open_trades,
            account_size=req.account_size,
            risk_pct=req.risk_pct,
            session=req.session,
            metadata=req.metadata,
        )

        if row_id is None:
            raise HTTPException(status_code=500, detail="Failed to insert risk event")

        logger.info(
            "Risk event recorded: id=%s type=%s symbol=%s",
            row_id,
            req.event_type,
            req.symbol,
        )

        return {
            "id": row_id,
            "event_type": req.event_type,
            "symbol": req.symbol,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record risk event: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to record risk event: {exc}"
        )


@router.post("/orb", status_code=201)
def create_orb_event(req: ORBEventCreate):
    """Manually record an ORB event for the audit trail.

    This endpoint is primarily for internal use — the engine records
    events automatically.  Use this for manual annotations, testing,
    or external system integrations.
    """
    try:
        from src.futures_lib.core.models import record_orb_event

        row_id = record_orb_event(
            symbol=req.symbol,
            or_high=req.or_high,
            or_low=req.or_low,
            or_range=req.or_range,
            atr_value=req.atr_value,
            breakout_detected=req.breakout_detected,
            direction=req.direction,
            trigger_price=req.trigger_price,
            long_trigger=req.long_trigger,
            short_trigger=req.short_trigger,
            bar_count=req.bar_count,
            session=req.session,
            metadata=req.metadata,
        )

        if row_id is None:
            raise HTTPException(status_code=500, detail="Failed to insert ORB event")

        logger.info(
            "ORB event recorded: id=%s symbol=%s breakout=%s direction=%s",
            row_id,
            req.symbol,
            req.breakout_detected,
            req.direction,
        )

        return {
            "id": row_id,
            "symbol": req.symbol,
            "breakout_detected": req.breakout_detected,
            "direction": req.direction,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record ORB event: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to record ORB event: {exc}"
        )
