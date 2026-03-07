"""
SAR (Stop-and-Reverse) Sync API
================================
Receives live SAR reversal events pushed by the NinjaTrader BreakoutStrategy
and exposes the current per-instrument SAR state for monitoring.

The C# ``TryReversePosition()`` method fires a fire-and-forget
``POST /sar/sync`` after every reversal so the Python ``PositionManager``
can mirror the C# ``ReversalState`` without polling NT8.

Endpoints
---------
POST /sar/sync
    Receive a SAR reversal event from NinjaTrader.
    Updates Redis and optionally calls PositionManager.load_state()
    to force a re-sync of active positions.

GET  /sar/state
    Return the latest SAR state for all tracked instruments (from Redis).

GET  /sar/state/{asset}
    Return the SAR state for a single instrument.

DELETE /sar/state/{asset}
    Clear the SAR state for a single instrument (e.g. end-of-day reset).

Network topology
----------------
  NT8 Windows server  100.127.182.112  →  POST http://100.100.84.48:8000/sar/sync
  Pi (Docker engine)  100.100.84.48    ←  reads Redis, drives PositionManager
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lib.core.cache import cache_get, cache_set

if TYPE_CHECKING:
    from lib.services.engine.position_manager import PositionManager  # noqa: F401

logger = logging.getLogger("api.sar")

router = APIRouter(tags=["SAR Sync"])

# ---------------------------------------------------------------------------
# Cache config
# ---------------------------------------------------------------------------

_SAR_KEY_PREFIX = "sar:state:"  # sar:state:{ASSET}  →  JSON blob
_SAR_TTL = 86_400  # 24 h — positions survive overnight
_SAR_ALL_ASSETS_KEY = "sar:assets"  # JSON list of known asset names


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SarSyncPayload(BaseModel):
    """Payload pushed by NinjaTrader ``TryReversePosition()`` / ``FireEntry()``."""

    asset: str = Field(..., description="Instrument root name, e.g. 'MGC'")
    direction: str = Field(..., description="New active direction: 'long', 'short', or '' (flat)")
    signal_id: str = Field("", description="SignalId of the new position (matches _positionPhases key)")
    reversal_count: int = Field(0, ge=0, description="Number of reversals made so far for this asset")
    entry_price: float = Field(0.0, description="Entry price of the new position")
    atr_at_entry: float = Field(0.0, description="ATR at entry time (used for R-multiple calculation)")
    sl_price: float = Field(0.0, description="Initial stop-loss price of the new position")
    timestamp: str = Field("", description="UTC ISO timestamp from NinjaTrader bar time")
    source: str = Field("NinjaTrader", description="Originating system (always 'NinjaTrader' from C#)")


class SarStateResponse(BaseModel):
    """SAR state for a single instrument as stored in Redis."""

    asset: str
    direction: str  # "long", "short", or "" (flat)
    signal_id: str
    reversal_count: int
    entry_price: float
    atr_at_entry: float
    sl_price: float
    timestamp: str  # UTC ISO of last reversal/entry from NT8
    received_at: str  # UTC ISO of when the Pi received the push
    source: str


class SarAllStateResponse(BaseModel):
    """SAR state for all tracked instruments."""

    assets: list[SarStateResponse]
    total: int
    active: int  # number of non-flat positions
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sar_cache_key(asset: str) -> str:
    return f"{_SAR_KEY_PREFIX}{asset.upper().strip()}"


def _read_sar_state(asset: str) -> dict[str, Any] | None:
    """Read SAR state for one asset from Redis.  Returns None if absent."""
    try:
        raw = cache_get(_sar_cache_key(asset))
        if raw is None:
            return None
        raw_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return json.loads(raw_str)  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("_read_sar_state(%s) error: %s", asset, exc)
        return None


def _write_sar_state(asset: str, state: dict[str, Any]) -> None:
    """Persist SAR state for one asset to Redis."""
    key = _sar_cache_key(asset)
    cache_set(key, json.dumps(state).encode("utf-8"), _SAR_TTL)


def _register_asset(asset: str) -> None:
    """Add asset to the global known-assets list (for GET /sar/state)."""
    try:
        raw = cache_get(_SAR_ALL_ASSETS_KEY)
        assets: list[str] = []
        if raw:
            raw_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            loaded: Any = json.loads(raw_str)
            if isinstance(loaded, list):
                assets = [str(a) for a in loaded]
        asset_upper = asset.upper().strip()
        if asset_upper not in assets:
            assets.append(asset_upper)
            cache_set(_SAR_ALL_ASSETS_KEY, json.dumps(assets).encode("utf-8"), _SAR_TTL * 30)
    except Exception as exc:
        logger.debug("_register_asset(%s) error: %s", asset, exc)


def _get_all_asset_names() -> list[str]:
    """Return the list of assets that have ever pushed a SAR sync."""
    try:
        raw = cache_get(_SAR_ALL_ASSETS_KEY)
        if not raw:
            return []
        raw_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        loaded: Any = json.loads(raw_str)
        if isinstance(loaded, list):
            return [str(a) for a in loaded]
        return []
    except Exception:
        return []


def _notify_position_manager(payload: SarSyncPayload) -> None:
    """Best-effort: inform the PositionManager of the new C# SAR state.

    When a reversal arrives from NT8 we update the in-process PositionManager
    so its Redis-persisted state reflects the C# position.  This prevents the
    Python engine from re-entering a position that NT8 has already opened (or
    reversed), avoiding duplicate orders on the next PositionManager tick.

    Any failure here is logged at DEBUG and swallowed — the primary sync
    mechanism is the Redis cache written by this endpoint; PositionManager
    will reconcile on the next ``process_signal`` call regardless.
    """
    try:
        # Lazy import — engine may not be running in the same process
        import importlib  # noqa: PLC0415

        engine_main = importlib.import_module("lib.services.engine.main")
        pm: PositionManager | None = getattr(engine_main, "_position_manager", None)
        if pm is None:
            return

        asset_upper = payload.asset.upper().strip()
        direction = payload.direction.lower().strip()

        if not direction:
            # NT8 flattened the position — close it in Python PM if it exists.
            # Call the private _close_position directly so only the one ticker
            # is affected (close_all would wipe every asset).
            existing = pm.get_position(asset_upper)
            if existing is not None:
                logger.info("[SAR sync] NT8 reports %s flat — closing Python PM position", asset_upper)
                try:
                    pm._close_position(  # type: ignore[attr-defined]
                        existing,
                        reason="NT8-flat (SAR sync)",
                        close_price=existing.current_price,
                    )
                    pm.save_state()
                except Exception as close_exc:
                    logger.debug("[SAR sync] _close_position(%s) error: %s", asset_upper, close_exc)
        else:
            # NT8 opened / reversed — reload state from Redis so PM reflects reality.
            # load_state() merges Redis keys into _positions without wiping intact positions.
            _ = pm.load_state()
            logger.info(
                "[SAR sync] NT8 %s %s reversal#%d — PM state reloaded",
                asset_upper,
                direction,
                payload.reversal_count,
            )
    except Exception as exc:
        logger.debug("[SAR sync] _notify_position_manager error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sync", summary="Receive SAR reversal event from NinjaTrader")
def sar_sync(payload: SarSyncPayload):
    """Receive a stop-and-reverse event from the NT8 BreakoutStrategy.

    Called by ``TryReversePosition()`` (and ``FireEntry()`` for fresh entries)
    in ``BreakoutStrategy.cs`` immediately after a reversal order is submitted.

    Updates the Redis SAR state for the instrument so:
      - The Python PositionManager knows the C# position direction.
      - The dashboard can display the current SAR direction per instrument.
      - The engine avoids opening a conflicting position on the next tick.

    Returns 200 on success.  The C# caller is fire-and-forget so errors
    here are logged but never retried — Redis durability is the safety net.
    """
    received_at: str = datetime.now(UTC).isoformat()

    asset = payload.asset.upper().strip()
    if not asset:
        raise HTTPException(status_code=422, detail="asset field is required and must not be empty")

    direction = payload.direction.lower().strip()
    if direction not in ("long", "short", ""):
        raise HTTPException(
            status_code=422, detail=f"direction must be 'long', 'short', or '' (flat); got '{payload.direction}'"
        )

    state: dict[str, Any] = {
        "asset": asset,
        "direction": direction,
        "signal_id": payload.signal_id,
        "reversal_count": payload.reversal_count,
        "entry_price": payload.entry_price,
        "atr_at_entry": payload.atr_at_entry,
        "sl_price": payload.sl_price,
        "timestamp": payload.timestamp,
        "received_at": received_at,
        "source": payload.source,
    }

    _write_sar_state(asset, state)
    _register_asset(asset)

    logger.info(
        "[SAR sync] %s → %s  signal=%s  reversal#%d  entry=%.4f  sl=%.4f  (from %s)",
        asset,
        direction if direction else "FLAT",
        payload.signal_id or "(none)",
        payload.reversal_count,
        payload.entry_price,
        payload.sl_price,
        payload.source,
    )

    # Best-effort PM notification (non-blocking)
    _notify_position_manager(payload)

    return {
        "status": "ok",
        "asset": asset,
        "direction": direction,
        "reversal_count": payload.reversal_count,
        "received_at": received_at,
    }


@router.get(
    "/state",
    response_model=SarAllStateResponse,
    summary="Get SAR state for all tracked instruments",
)
def get_all_sar_state():
    """Return the latest SAR state for every instrument that has ever synced.

    Reads the per-asset Redis keys populated by ``POST /sar/sync``.
    Instruments that have never pushed a sync event are absent from the list.
    """
    asset_names = _get_all_asset_names()
    states: list[SarStateResponse] = []

    for name in asset_names:
        data = _read_sar_state(name)
        if data is None:
            continue
        try:
            states.append(SarStateResponse(**data))
        except Exception as exc:
            logger.debug("get_all_sar_state parse error for %s: %s", name, exc)

    active = sum(1 for s in states if s.direction in ("long", "short"))

    return SarAllStateResponse(
        assets=states,
        total=len(states),
        active=active,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/state/{asset}",
    response_model=SarStateResponse,
    summary="Get SAR state for a single instrument",
)
def get_sar_state(asset: str):
    """Return the SAR state for a single instrument.

    Args:
        asset: Instrument root name, case-insensitive (e.g. ``MGC``, ``mes``).

    Raises:
        404 if no SAR sync has been received for this instrument yet.
    """
    data = _read_sar_state(asset)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SAR state found for '{asset.upper()}'. "
            f"Has NinjaTrader pushed a sync event for this instrument?",
        )
    try:
        return SarStateResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAR state parse error for '{asset}': {exc}") from exc


@router.delete(
    "/state/{asset}",
    summary="Clear SAR state for a single instrument",
)
def clear_sar_state(asset: str):
    """Clear the cached SAR state for a single instrument.

    Useful for end-of-day resets or when NinjaTrader is restarted and the
    cached direction is stale.  The next ``POST /sar/sync`` from NT8 will
    repopulate the state.
    """
    asset_upper = asset.upper().strip()
    key = _sar_cache_key(asset_upper)

    try:
        from lib.core.cache import REDIS_AVAILABLE as _redis_avail  # noqa: PLC0415
        from lib.core.cache import _mem_cache, _r  # noqa: PLC0415

        if _redis_avail and _r is not None:
            _r.delete(key)  # type: ignore[union-attr]
        else:
            _mem_cache.pop(key, None)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("clear_sar_state(%s) Redis delete error: %s", asset_upper, exc)

    logger.info("[SAR sync] State cleared for %s", asset_upper)
    return {
        "status": "cleared",
        "asset": asset_upper,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.delete(
    "/state",
    summary="Clear SAR state for all instruments",
)
def clear_all_sar_state():
    """Clear the SAR state cache for all instruments.

    Intended for end-of-day maintenance.  All per-asset keys and the
    asset registry are removed.  The next NT8 push will rebuild the state.
    """
    asset_names = _get_all_asset_names()
    cleared: list[str] = []

    try:
        from lib.core.cache import REDIS_AVAILABLE as _redis_avail  # noqa: PLC0415
        from lib.core.cache import _mem_cache, _r  # noqa: PLC0415
    except Exception:
        _redis_avail = False
        _r = None
        _mem_cache: dict[str, Any] = {}  # type: ignore[assignment]

    for name in asset_names:
        key = _sar_cache_key(name)
        try:
            if _redis_avail and _r is not None:
                _r.delete(key)  # type: ignore[union-attr]
            else:
                _mem_cache.pop(key, None)  # type: ignore[union-attr]
            cleared.append(name)
        except Exception as exc:
            logger.warning("clear_all_sar_state key=%s error: %s", key, exc)

    # Clear the asset registry too
    try:
        if _redis_avail and _r is not None:
            _r.delete(_SAR_ALL_ASSETS_KEY)  # type: ignore[union-attr]
        else:
            _mem_cache.pop(_SAR_ALL_ASSETS_KEY, None)  # type: ignore[union-attr]
    except Exception:
        pass

    logger.info("[SAR sync] All SAR state cleared (%d assets)", len(cleared))
    return {
        "status": "cleared",
        "assets_cleared": cleared,
        "count": len(cleared),
        "timestamp": datetime.now(UTC).isoformat(),
    }
