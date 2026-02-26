"""
SSE (Server-Sent Events) Router — TASK-302
============================================
Streams live dashboard updates to the browser via SSE.

Architecture:
    Engine → XADD dashboard:stream:focus (durable) + PUBLISH dashboard:live (trigger)
    Data-service SSE → on connect: XREVRANGE last 8 messages (catch-up),
                       then subscribe to pub/sub for live updates
    Browser → hx-ext="sse" sse-connect="/sse/dashboard" with per-asset event names

Endpoints:
    GET /sse/dashboard  — Main SSE stream (focus updates, alerts, heartbeat)

Event types sent to browser:
    - focus-update       — Full focus payload (all assets)
    - {symbol}-update    — Per-asset update (e.g. mgc-update, mnq-update)
    - no-trade-alert     — No-trade condition triggered
    - session-change     — Session mode changed (pre-market/active/off-hours)
    - positions-update   — Live positions changed
    - heartbeat          — Keep-alive every 30 seconds
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("api.sse")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["SSE"])

# ---------------------------------------------------------------------------
# Throttle settings — max 1 update per asset per N seconds
# ---------------------------------------------------------------------------
_THROTTLE_SECONDS = 7.0
_HEARTBEAT_INTERVAL = 30.0
_CATCHUP_COUNT = 8

# Track last send time per event type for throttling
_last_sent: dict[str, float] = {}


def _should_throttle(event_key: str) -> bool:
    """Return True if we should skip this event due to throttling."""
    now = time.monotonic()
    last = _last_sent.get(event_key, 0)
    if now - last < _THROTTLE_SECONDS:
        return True
    _last_sent[event_key] = now
    return False


def _format_sse(
    data: str,
    event: Optional[str] = None,
    id: Optional[str] = None,
    retry: Optional[int] = None,
) -> str:
    """Format a single SSE message according to the spec.

    See: https://html.spec.whatwg.org/multipage/server-sent-events.html
    """
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    # Data can be multi-line; each line needs its own "data:" prefix
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")  # blank line terminates the event
    lines.append("")
    return "\n".join(lines)


def _make_heartbeat_event() -> str:
    """Create a heartbeat SSE event with current server time."""
    now = datetime.now(tz=_EST)
    payload = json.dumps(
        {
            "type": "heartbeat",
            "time_et": now.strftime("%H:%M:%S ET"),
            "timestamp": now.isoformat(),
        }
    )
    return _format_sse(data=payload, event="heartbeat")


def _make_session_event(session_mode: str) -> str:
    """Create a session-change SSE event."""
    now = datetime.now(tz=_EST)
    emoji = {
        "pre_market": "\U0001f319",
        "active": "\U0001f7e2",
        "off_hours": "\u2699\ufe0f",
    }.get(session_mode, "")
    payload = json.dumps(
        {
            "type": "session-change",
            "session": session_mode,
            "emoji": emoji,
            "timestamp": now.isoformat(),
        }
    )
    return _format_sse(data=payload, event="session-change")


# ---------------------------------------------------------------------------
# Redis helpers — async wrappers around the sync redis client
# ---------------------------------------------------------------------------


def _get_redis():
    """Get the Redis client from cache module, or None."""
    try:
        from cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            return _r
    except ImportError:
        pass
    return None


def _get_catchup_messages(count: int = _CATCHUP_COUNT) -> list[dict]:
    """Read the last N messages from the Redis Stream for catch-up.

    Returns list of dicts with keys: id, data, ts.
    """
    r = _get_redis()
    if r is None:
        return []

    try:
        # XREVRANGE returns newest first; we want oldest first for the client
        raw = r.xrevrange("dashboard:stream:focus", count=count)
        if not raw:
            return []

        messages = []
        for msg_id, fields in reversed(raw):
            # msg_id is bytes, fields is dict of bytes
            entry = {
                "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                "data": fields.get(b"data", b"{}").decode()
                if isinstance(fields.get(b"data", b"{}"), bytes)
                else str(fields.get(b"data", "{}")),
                "ts": fields.get(b"ts", b"").decode()
                if isinstance(fields.get(b"ts", b""), bytes)
                else str(fields.get(b"ts", "")),
            }
            messages.append(entry)
        return messages
    except Exception as exc:
        logger.debug("Failed to read catchup from Redis Stream: %s", exc)
        return []


def _get_focus_from_cache() -> Optional[str]:
    """Read the current focus JSON from Redis cache."""
    try:
        from cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_positions_from_cache() -> Optional[str]:
    """Read current positions from Redis cache."""
    try:
        from cache import _cache_key, cache_get

        key = _cache_key("live_positions", "current")
        raw = cache_get(key)
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_engine_status() -> Optional[dict]:
    """Read engine status from Redis."""
    try:
        from cache import cache_get

        raw = cache_get("engine:status")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main SSE generator
# ---------------------------------------------------------------------------


async def _dashboard_event_generator(request: Request) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for the dashboard.

    Flow:
    1. Send retry directive (auto-reconnect after 3s)
    2. Send catch-up messages from Redis Stream
    3. If Redis pub/sub available, subscribe and forward live events
    4. Otherwise, fall back to polling Redis every 5 seconds
    5. Send heartbeat every 30 seconds
    """

    # 1. Retry directive — tells browser to reconnect after 3 seconds on disconnect
    yield _format_sse(data="connected", event="connected", retry=3000)

    # 2. Catch-up: send last N focus updates from Redis Stream
    catchup = _get_catchup_messages()
    if catchup:
        for msg in catchup:
            try:
                data_str = msg["data"]
                yield _format_sse(data=data_str, event="focus-update", id=msg["id"])

                # Also emit per-asset events from the catchup data
                try:
                    focus = json.loads(data_str)
                    for asset in focus.get("assets", []):
                        symbol = asset.get("symbol", "").lower().replace(" ", "_")
                        if symbol:
                            asset_json = json.dumps(asset, default=str)
                            yield _format_sse(
                                data=asset_json,
                                event=f"{symbol}-update",
                                id=msg["id"],
                            )

                    # No-trade catchup
                    if focus.get("no_trade"):
                        yield _format_sse(
                            data=json.dumps(
                                {
                                    "no_trade": True,
                                    "reason": focus.get("no_trade_reason", ""),
                                }
                            ),
                            event="no-trade-alert",
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            except Exception as exc:
                logger.debug("Catchup message error: %s", exc)
    else:
        # No stream data — try the cache key directly
        cached = _get_focus_from_cache()
        if cached:
            yield _format_sse(data=cached, event="focus-update")

    # Send initial positions
    pos = _get_positions_from_cache()
    if pos:
        yield _format_sse(data=pos, event="positions-update")

    # Send initial session info
    status = _get_engine_status()
    if status:
        session_mode = status.get("session_mode", "unknown")
        yield _make_session_event(session_mode)

    # 3. Try Redis pub/sub for live events, fall back to polling
    r = _get_redis()
    pubsub = None
    use_pubsub = False

    if r is not None:
        try:
            pubsub = r.pubsub()
            # Subscribe to all dashboard channels using pattern
            pubsub.psubscribe("dashboard:*")
            use_pubsub = True
            logger.info("SSE client connected (pub/sub mode)")
        except Exception as exc:
            logger.debug("Pub/sub subscribe failed, falling back to polling: %s", exc)
            use_pubsub = False

    if not use_pubsub:
        logger.info("SSE client connected (polling mode)")

    last_heartbeat = time.monotonic()
    last_focus_hash = ""
    last_positions_hash = ""
    last_session = ""

    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.debug("SSE client disconnected")
                break

            now = time.monotonic()

            # --- Heartbeat ---
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield _make_heartbeat_event()
                last_heartbeat = now

            if use_pubsub and pubsub is not None:
                # ---- Pub/sub mode: check for messages ----
                try:
                    message = pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=0.1
                    )
                    if message and message["type"] in ("message", "pmessage"):
                        channel = message.get("channel", b"")
                        if isinstance(channel, bytes):
                            channel = channel.decode()
                        data = message.get("data", b"")
                        if isinstance(data, bytes):
                            data = data.decode()

                        if channel == "dashboard:live":
                            # Full focus update
                            if not _should_throttle("focus-update"):
                                yield _format_sse(data=data, event="focus-update")

                                # Emit per-asset events
                                try:
                                    focus = json.loads(data)
                                    for asset in focus.get("assets", []):
                                        symbol = (
                                            asset.get("symbol", "")
                                            .lower()
                                            .replace(" ", "_")
                                        )
                                        if symbol and not _should_throttle(
                                            f"{symbol}-update"
                                        ):
                                            asset_json = json.dumps(asset, default=str)
                                            yield _format_sse(
                                                data=asset_json,
                                                event=f"{symbol}-update",
                                            )
                                except (json.JSONDecodeError, TypeError):
                                    pass

                        elif channel.startswith("dashboard:asset:"):
                            # Per-asset update
                            symbol = channel.split(":")[-1]
                            event_name = f"{symbol}-update"
                            if not _should_throttle(event_name):
                                yield _format_sse(data=data, event=event_name)

                        elif channel == "dashboard:no_trade":
                            yield _format_sse(data=data, event="no-trade-alert")

                        elif channel == "dashboard:session":
                            yield _format_sse(data=data, event="session-change")

                        elif channel == "dashboard:positions":
                            if not _should_throttle("positions-update"):
                                yield _format_sse(data=data, event="positions-update")

                except Exception as exc:
                    logger.debug("Pub/sub read error: %s", exc)
                    # Don't break — keep trying

                # Also check for session changes via engine status (pubsub might miss it)
                try:
                    status = _get_engine_status()
                    if status:
                        current_session = status.get("session_mode", "")
                        if current_session and current_session != last_session:
                            last_session = current_session
                            yield _make_session_event(current_session)
                except Exception:
                    pass

            else:
                # ---- Polling mode: check Redis cache periodically ----
                try:
                    # Check focus data
                    cached = _get_focus_from_cache()
                    if cached:
                        focus_hash = str(hash(cached))
                        if focus_hash != last_focus_hash:
                            last_focus_hash = focus_hash
                            if not _should_throttle("focus-update"):
                                yield _format_sse(data=cached, event="focus-update")

                                # Per-asset events
                                try:
                                    focus = json.loads(cached)
                                    for asset in focus.get("assets", []):
                                        symbol = (
                                            asset.get("symbol", "")
                                            .lower()
                                            .replace(" ", "_")
                                        )
                                        if symbol and not _should_throttle(
                                            f"{symbol}-update"
                                        ):
                                            asset_json = json.dumps(asset, default=str)
                                            yield _format_sse(
                                                data=asset_json,
                                                event=f"{symbol}-update",
                                            )

                                    if focus.get("no_trade"):
                                        yield _format_sse(
                                            data=json.dumps(
                                                {
                                                    "no_trade": True,
                                                    "reason": focus.get(
                                                        "no_trade_reason", ""
                                                    ),
                                                }
                                            ),
                                            event="no-trade-alert",
                                        )
                                except (json.JSONDecodeError, TypeError):
                                    pass

                    # Check positions
                    pos = _get_positions_from_cache()
                    if pos:
                        pos_hash = str(hash(pos))
                        if pos_hash != last_positions_hash:
                            last_positions_hash = pos_hash
                            if not _should_throttle("positions-update"):
                                yield _format_sse(data=pos, event="positions-update")

                    # Check session
                    status = _get_engine_status()
                    if status:
                        current_session = status.get("session_mode", "")
                        if current_session and current_session != last_session:
                            last_session = current_session
                            yield _make_session_event(current_session)

                except Exception as exc:
                    logger.debug("Polling read error: %s", exc)

            # Small sleep to avoid busy-looping
            # Pub/sub mode: 0.25s (messages arrive via subscription)
            # Polling mode: 5s (we poll cache keys)
            sleep_time = 0.25 if use_pubsub else 5.0
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.debug("SSE generator cancelled")
    except GeneratorExit:
        logger.debug("SSE generator exited")
    except Exception as exc:
        logger.error("SSE generator error: %s", exc, exc_info=True)
    finally:
        # Clean up pub/sub subscription
        if pubsub is not None:
            try:
                pubsub.punsubscribe("dashboard:*")
                pubsub.close()
            except Exception:
                pass
        logger.debug("SSE connection closed")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/sse/dashboard")
async def sse_dashboard(request: Request):
    """Server-Sent Events endpoint for live dashboard updates.

    Connect from HTMX:
        <div hx-ext="sse" sse-connect="/sse/dashboard">
            <div sse-swap="focus-update" hx-swap="innerHTML">...</div>
            <div sse-swap="mgc-update" hx-swap="innerHTML">...</div>
            <div sse-swap="no-trade-alert" hx-swap="innerHTML">...</div>
            <div sse-swap="heartbeat">...</div>
        </div>

    Or from JavaScript:
        const es = new EventSource('/sse/dashboard');
        es.addEventListener('focus-update', (e) => { ... });
        es.addEventListener('mgc-update', (e) => { ... });
        es.addEventListener('no-trade-alert', (e) => { ... });
        es.addEventListener('heartbeat', (e) => { ... });

    Events:
        - connected         — Initial connection confirmation
        - focus-update      — Full focus payload (JSON)
        - {symbol}-update   — Per-asset update (JSON), e.g. mgc-update
        - no-trade-alert    — No-trade condition (JSON)
        - session-change    — Session mode changed (JSON)
        - positions-update  — Live positions changed (JSON)
        - heartbeat         — Keep-alive with server time (JSON)

    Catch-up: On connect, the last 8 focus updates from the Redis Stream
    are sent immediately so the client doesn't miss anything.

    Throttling: Max 1 update per asset per 7 seconds to avoid overwhelming
    the browser during high-frequency engine recomputation.

    Auto-reconnect: The retry directive tells the browser to reconnect
    after 3 seconds if the connection drops. HTMX handles this natively.
    """
    return StreamingResponse(
        _dashboard_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/sse/health")
async def sse_health():
    """Health check for SSE subsystem.

    Returns the status of Redis connectivity and stream stats.
    """
    r = _get_redis()
    redis_ok = r is not None

    stream_length = 0
    if redis_ok and r is not None:
        try:
            info = r.xinfo_stream("dashboard:stream:focus")
            stream_length = info.get(b"length", info.get("length", 0))
        except Exception:
            pass

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "stream_length": stream_length,
        "mode": "pubsub" if redis_ok else "polling",
        "throttle_seconds": _THROTTLE_SECONDS,
        "heartbeat_interval": _HEARTBEAT_INTERVAL,
        "catchup_count": _CATCHUP_COUNT,
    }
