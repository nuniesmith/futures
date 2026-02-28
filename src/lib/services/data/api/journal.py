"""
Journal API router — daily P&L journal endpoints.

Provides endpoints for saving end-of-day journal entries,
retrieving journal history, and computing journal statistics.
"""

from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.lib.core.models import (
    get_daily_journal,
    get_journal_stats,
    save_daily_journal,
)

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["journal"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class JournalEntryRequest(BaseModel):
    """Request body for saving a daily journal entry."""

    trade_date: str = Field(
        ...,
        description="Date string in YYYY-MM-DD format",
    )
    account_size: int = Field(
        150000,
        description="Account size: 50000, 100000, or 150000",
    )
    gross_pnl: float = Field(0.0, description="Gross P&L for the day")
    net_pnl: float = Field(0.0, description="Net P&L after commissions")
    commissions: float = Field(0.0, description="Total commissions paid")
    num_contracts: int = Field(0, description="Total contracts traded")
    instruments: str = Field("", description="Comma-separated instrument names")
    notes: str = Field("", description="Free-form notes about the trading day")


class JournalEntryResponse(BaseModel):
    """Response after saving a journal entry."""

    status: str
    trade_date: str
    net_pnl: float
    timestamp: str


class JournalStatsResponse(BaseModel):
    """Aggregated journal statistics.

    Field names match the dict returned by models.get_journal_stats().
    """

    total_days: int = 0
    win_days: int = 0
    loss_days: int = 0
    break_even_days: int = 0
    win_rate: float = 0.0
    total_net: float = 0.0
    total_gross: float = 0.0
    total_commissions: float = 0.0
    avg_daily_net: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    current_streak: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/save", response_model=JournalEntryResponse)
def save_journal_entry(entry: JournalEntryRequest):
    """Save or update a daily journal entry.

    If an entry for the given date already exists, it will be updated
    (SQLite UPSERT via INSERT OR REPLACE on the unique trade_date column).
    """
    try:
        save_daily_journal(
            trade_date=entry.trade_date,
            account_size=entry.account_size,
            gross_pnl=entry.gross_pnl,
            net_pnl=entry.net_pnl,
            num_contracts=entry.num_contracts,
            instruments=entry.instruments,
            notes=entry.notes,
        )
        return JournalEntryResponse(
            status="saved",
            trade_date=entry.trade_date,
            net_pnl=entry.net_pnl,
            timestamp=datetime.now(tz=_EST).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save journal: {exc}")


@router.get("/entries")
def get_journal_entries(
    limit: int = Query(30, ge=1, le=365, description="Number of recent entries"),
    account_size: Optional[int] = Query(
        None, description="Filter by account size (50000, 100000, 150000)"
    ),
):
    """Retrieve recent daily journal entries.

    Returns a list of journal entry dicts, most recent first.
    """
    try:
        df = get_daily_journal(limit=limit)

        # Convert DataFrame to list of dicts for JSON serialization
        if hasattr(df, "to_dict"):
            entries = df.to_dict(orient="records") if not df.empty else []
        else:
            entries = list(df) if len(df) > 0 else []

        # Optional filter by account size
        if account_size is not None:
            entries = [e for e in entries if e.get("account_size") == account_size]

        return {
            "entries": entries,
            "count": len(entries),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve journal: {exc}"
        )


@router.get("/stats", response_model=JournalStatsResponse)
def get_stats(
    account_size: Optional[int] = Query(
        None, description="Filter stats by account size"
    ),
):
    """Get aggregated journal statistics.

    Computes win rate, streaks, averages, and totals across
    all recorded journal entries.
    """
    try:
        stats = get_journal_stats()
        return JournalStatsResponse(**stats)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to compute journal stats: {exc}"
        )


@router.get("/today")
def get_today_entry():
    """Get today's journal entry if it exists.

    Convenience endpoint that checks if a journal entry has already
    been saved for the current trading day.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        df = get_daily_journal(limit=1)

        # Convert DataFrame to list of dicts for iteration
        if hasattr(df, "to_dict"):
            entries = df.to_dict(orient="records") if not df.empty else []
        else:
            entries = list(df) if len(df) > 0 else []

        for entry in entries:
            if entry.get("trade_date") == today_str:
                return {
                    "exists": True,
                    "entry": entry,
                    "timestamp": datetime.now(tz=_EST).isoformat(),
                }
        return {
            "exists": False,
            "entry": None,
            "trade_date": today_str,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to check today's journal: {exc}"
        )
