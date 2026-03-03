"""
Dashboard API Router (TASK-301 / TASK-303 / TASK-501)
=======================================================
Serves the HTMX dashboard and HTML fragment endpoints.

Day 4 additions (TASK-501):
  - Positions panel now shows risk status, total risk % of account,
    and red warning banner when total risk > 5%.
  - Risk status bar sourced from engine:risk_status Redis key.
  - Grok compact update panel (TASK-601) shows latest ≤8-line summary.

Endpoints:
    GET /                     — Full HTML dashboard page
    GET /api/focus            — All asset focus data as JSON
    GET /api/focus/html       — All asset cards as HTML fragments
    GET /api/focus/{symbol}   — Single asset card HTML fragment
    GET /api/positions/html   — Live positions panel HTML
    GET /api/risk/html        — Risk status panel HTML
    GET /api/grok/html        — Grok compact update panel HTML
    GET /api/alerts/html      — Alerts panel HTML
    GET /api/time             — Formatted time string with session indicator
    GET /api/no-trade         — No-trade banner HTML (if applicable)
"""

import contextlib
import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

logger = logging.getLogger("api.dashboard")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_focus_data() -> dict[str, Any] | None:
    """Read daily focus payload from Redis.

    Primary source: ``engine:daily_focus`` key (written by the engine with
    a 5-minute TTL).  During off-hours or after a restart the key may have
    expired, so we fall back to the most recent entry in the durable Redis
    Stream ``dashboard:stream:focus`` which has no TTL.
    """
    # 1. Try the primary cache key (fast, most common path)
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            parsed: object = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("Failed to read focus from Redis key: %s", exc)

    # 2. Fallback: read the latest entry from the Redis Stream
    try:
        from lib.core.cache import REDIS_AVAILABLE

        if not REDIS_AVAILABLE:
            return None

        # Try to get a Redis client; the function may not exist in all versions
        client: Any = None
        try:
            from lib.core.cache import get_redis_client as _get_client  # type: ignore[attr-defined]

            client = _get_client()
        except (ImportError, AttributeError):
            pass

        if client is None:
            return None

        entries_raw: Any = client.xrevrange("dashboard:stream:focus", count=1)
        if not entries_raw:
            return None

        entry: Any = entries_raw[0]
        fields: Any = entry[1]
        raw_data: Any = fields.get(b"data") if fields else None
        if raw_data is not None:
            decoded: str = raw_data.decode() if isinstance(raw_data, bytes) else str(raw_data)
            parsed_stream: object = json.loads(decoded)
            if isinstance(parsed_stream, dict):
                return parsed_stream  # type: ignore[return-value]

    except Exception as exc:
        logger.debug("Failed to read focus from Redis stream: %s", exc)

    return None


def _get_session_info() -> dict[str, str]:
    """Get current session mode and display info.

    Boundaries (all Eastern Time):
      - Pre-market:          00:00–03:00
      - Active / London Open: 03:00–08:00
      - Active / US Open:     08:00–12:00
      - Off-hours:           12:00–00:00
    """
    now = datetime.now(tz=_EST)
    hour = now.hour

    if 0 <= hour < 3:
        mode = "pre-market"
        emoji = "🌙"
        label = "PRE-MARKET"
        css_class = "text-purple-400"
    elif 3 <= hour < 8:
        mode = "active"
        emoji = "🟢"
        label = "LONDON OPEN"
        css_class = "text-green-400"
    elif 8 <= hour < 12:
        mode = "active"
        emoji = "🟢"
        label = "US OPEN"
        css_class = "text-green-400"
    else:
        mode = "off-hours"
        emoji = "⚙️"
        label = "OFF-HOURS"
        css_class = "text-zinc-400"

    return {
        "mode": mode,
        "emoji": emoji,
        "label": label,
        "css_class": css_class,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%A, %B %d, %Y"),
        "time_et": now.strftime("%I:%M:%S %p ET"),
    }


def _get_positions() -> list[dict[str, Any]]:
    """Read live positions from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("positions:current")
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("positions", [])
    except Exception:
        pass
    # Try the positions router's hashed cache key as fallback
    try:
        from lib.core.cache import cache_get as cg2
        from lib.services.data.api.positions import _POSITIONS_CACHE_KEY

        raw2 = cg2(_POSITIONS_CACHE_KEY)
        if raw2:
            data2 = json.loads(raw2)
            return data2.get("positions", [])
    except Exception:
        pass
    return []


def _get_risk_status() -> dict[str, Any] | None:
    """Read risk manager status from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:risk_status")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _get_orb_data() -> dict[str, Any] | None:
    """Read the latest ORB (Opening Range Breakout) result from cache (TASK-801).

    Returns a dict with keys:
      - "london": dict | None  — London Open ORB result (03:00–03:30 ET)
      - "us": dict | None      — US Equity Open ORB result (09:30–10:00 ET)
      - "best": dict | None    — whichever session has a breakout (or latest)
    """
    try:
        from lib.core.cache import cache_get

        sessions: dict[str, Any] = {}

        # Fetch London Open ORB
        raw_london = cache_get("engine:orb:london")
        if raw_london:
            data = raw_london.decode() if isinstance(raw_london, bytes) else str(raw_london)
            if data:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    sessions["london"] = json.loads(data)

        # Fetch US Equity Open ORB
        raw_us = cache_get("engine:orb:us")
        if raw_us:
            data = raw_us.decode() if isinstance(raw_us, bytes) else str(raw_us)
            if data:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    sessions["us"] = json.loads(data)

        # Fallback: try the legacy combined key
        if not sessions:
            raw = cache_get("engine:orb")
            if raw:
                data = raw.decode() if isinstance(raw, bytes) else str(raw)
                if data:
                    try:
                        legacy = json.loads(data)
                        # Slot into the appropriate session based on session_key
                        sk = legacy.get("session_key", "us")
                        sessions[sk] = legacy
                    except (json.JSONDecodeError, TypeError):
                        pass

        if not sessions:
            return None

        # Determine the "best" result: prefer one with a breakout
        best = None
        for key in ("london", "us"):
            s = sessions.get(key)
            if s and s.get("breakout_detected"):
                best = s
                break
        # If no breakout, pick the first non-empty, non-error session
        if best is None:
            for key in ("london", "us"):
                s = sessions.get(key)
                if s and not s.get("error"):
                    best = s
                    break
        # Last resort: any session
        if best is None:
            for s in sessions.values():
                if s:
                    best = s
                    break

        return {
            "london": sessions.get("london"),
            "us": sessions.get("us"),
            "best": best,
        }
    except Exception:
        pass
    return None


def _get_grok_update() -> dict[str, Any] | None:
    """Read latest Grok compact update from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:grok_update")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# HTML rendering helpers (inline Jinja2-style templates)
# We render HTML directly to avoid template file dependencies during
# initial bootstrap. These can be migrated to Jinja2 files later.
# ---------------------------------------------------------------------------


def _render_asset_card(asset: dict[str, Any]) -> str:
    """Render a single asset focus card as an HTML fragment."""
    symbol = asset.get("symbol", "?")
    bias = asset.get("bias", "NEUTRAL")
    bias_emoji = asset.get("bias_emoji", "⚪")
    last_price = asset.get("last_price", 0)
    quality_pct = asset.get("quality_pct", 0)
    wave_ratio = asset.get("wave_ratio", 1.0)
    vol_cluster = asset.get("vol_cluster", "MEDIUM")
    vol_pct = asset.get("vol_percentile", 0.5)
    entry_low = asset.get("entry_low", 0)
    entry_high = asset.get("entry_high", 0)
    stop = asset.get("stop", 0)
    tp1 = asset.get("tp1", 0)
    tp2 = asset.get("tp2", 0)
    position_size = asset.get("position_size", 0)
    risk_dollars = asset.get("risk_dollars", 0)
    trend_dir = asset.get("trend_direction", "NEUTRAL ↔️")
    _dominance = asset.get("dominance_text", "Neutral")  # noqa: F841
    _market_phase = asset.get("market_phase", "UNKNOWN")  # noqa: F841
    notes = asset.get("notes", "")
    skip = asset.get("skip", False)

    # Color scheme based on bias
    if bias == "LONG":
        border_color = "border-green-500"
        bias_bg = "bg-green-900/40"
        bias_text = "text-green-400"
    elif bias == "SHORT":
        border_color = "border-red-500"
        bias_bg = "bg-red-900/40"
        bias_text = "text-red-400"
    else:
        border_color = "border-zinc-600"
        bias_bg = "bg-zinc-800/40"
        bias_text = "text-zinc-400"

    # Quality bar color
    if quality_pct >= 70:
        q_color = "bg-green-500"
    elif quality_pct >= 55:
        q_color = "bg-yellow-500"
    else:
        q_color = "bg-red-500"

    # Opacity for skipped assets
    opacity = "opacity-50" if skip else ""

    symbol_lower = symbol.lower().replace(" ", "_").replace("&", "")

    return f"""
    <div id="asset-card-{symbol_lower}"
         class="border {border_color} rounded-lg p-4 {bias_bg} {opacity} transition-all duration-300"
         data-quality="{quality_pct}"
         data-wave="{wave_ratio}"
         data-bias="{bias}"
         data-symbol="{symbol_lower}"
         hx-swap-oob="true"
         _="on load if my @data-quality as Number < 55 add .opacity-50 to me
              else remove .opacity-50 from me end">

        <!-- Header -->
        <div class="flex items-center justify-between mb-3">
            <div class="flex items-center gap-2">
                <span class="text-2xl">{bias_emoji}</span>
                <h3 class="text-lg font-bold text-white">{symbol}</h3>
            </div>
            <div class="text-right">
                <div class="text-xl font-mono font-bold text-white">{last_price:,.2f}</div>
                <div class="text-xs {bias_text} font-semibold">{bias}</div>
            </div>
        </div>

        <!-- Quality & Wave -->
        <div class="grid grid-cols-2 gap-2 mb-3">
            <div>
                <div class="text-xs text-zinc-400 mb-1">Quality</div>
                <div class="w-full bg-zinc-700 rounded-full h-2">
                    <div class="{q_color} h-2 rounded-full transition-all duration-500"
                         style="width: {min(quality_pct, 100):.0f}%"></div>
                </div>
                <div class="text-xs text-zinc-300 mt-0.5">{quality_pct:.0f}%</div>
            </div>
            <div>
                <div class="text-xs text-zinc-400 mb-1">Wave Ratio</div>
                <div class="text-lg font-mono font-bold text-white">{wave_ratio:.2f}x</div>
            </div>
        </div>

        <!-- Levels -->
        <div class="grid grid-cols-3 gap-1 mb-3 text-xs">
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">Entry</div>
                <div class="text-zinc-200 font-mono">{entry_low:,.2f}</div>
                <div class="text-zinc-400 font-mono">– {entry_high:,.2f}</div>
            </div>
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">Stop</div>
                <div class="text-red-400 font-mono">{stop:,.2f}</div>
            </div>
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">TP1 / TP2</div>
                <div class="text-green-400 font-mono">{tp1:,.2f}</div>
                <div class="text-green-300 font-mono">{tp2:,.2f}</div>
            </div>
        </div>

        <!-- Meta row -->
        <div class="flex items-center justify-between text-xs text-zinc-400">
            <span>{trend_dir}</span>
            <span>Vol: {vol_cluster} ({vol_pct:.0%})</span>
            <span>{position_size} micros / ${risk_dollars:,.0f} risk</span>
        </div>

        <!-- Notes -->
        {"<div class='mt-2 text-xs text-yellow-400 italic'>" + notes + "</div>" if notes else ""}
    </div>
    """


def _render_no_trade_banner(reason: str) -> str:
    """Render the NO TRADE warning banner."""
    return f"""
    <div id="no-trade-banner"
         class="bg-red-900/60 border border-red-500 rounded-lg p-4 text-center animate-pulse"
         hx-swap-oob="true">
        <div class="text-3xl mb-2">⛔</div>
        <div class="text-xl font-bold text-red-300">NO TRADE TODAY</div>
        <div class="text-sm text-red-400 mt-1">{reason}</div>
    </div>
    """


def _render_positions_panel(
    positions: list[dict[str, Any]],
    risk_status: dict[str, Any] | None = None,
) -> str:
    """Render live positions panel with risk status as HTML fragment (TASK-501).

    Shows:
      - Each position: symbol, LONG/SHORT, qty, avg price, unrealized P&L
      - Total risk % prominently (sum of position risks / account value)
      - Red warning banner if total risk > 5% of account
      - Risk rules status bar (daily P&L, open trades, block reason)
    """
    # ---- Risk status bar (always shown) ----
    risk_bar = ""
    if risk_status:
        daily_pnl = risk_status.get("daily_pnl", 0)
        daily_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"
        open_count = risk_status.get("open_trade_count", 0)
        max_trades = risk_status.get("max_open_trades", 2)
        risk_pct = risk_status.get("risk_pct_of_account", 0)
        can_trade = risk_status.get("can_trade", True)
        block_reason = risk_status.get("block_reason", "")
        consecutive = risk_status.get("consecutive_losses", 0)
        is_overnight = risk_status.get("is_overnight_warning", False)

        # Risk percentage color + warning threshold
        if risk_pct > 5:
            risk_pct_color = "text-red-400 font-bold"
            risk_pct_emoji = "🔴"
        elif risk_pct > 3:
            risk_pct_color = "text-yellow-400"
            risk_pct_emoji = "🟡"
        else:
            risk_pct_color = "text-green-400"
            risk_pct_emoji = "🟢"

        # Risk warning banner (> 5% of account)
        risk_warning_html = ""
        if risk_pct > 5:
            risk_warning_html = f"""
            <div class="bg-red-900/60 border border-red-500 rounded px-3 py-2 mb-2 text-center">
                <span class="text-red-300 text-xs font-bold">
                    ⚠️ TOTAL RISK {risk_pct:.1f}% OF ACCOUNT — EXCEEDS 5% LIMIT
                </span>
            </div>
            """

        # Overnight warning
        overnight_html = ""
        if is_overnight:
            overnight_html = """
            <div class="bg-yellow-900/60 border border-yellow-600 rounded px-3 py-2 mb-2 text-center">
                <span class="text-yellow-300 text-xs font-bold">
                    ⏰ SESSION ENDING — Close or protect open positions
                </span>
            </div>
            """

        # Trade block banner
        block_html = ""
        if not can_trade and block_reason:
            block_html = f"""
            <div class="bg-red-900/40 border border-red-700 rounded px-3 py-1.5 mb-2">
                <span class="text-red-400 text-xs">🚫 {block_reason}</span>
            </div>
            """

        risk_bar = f"""
        {risk_warning_html}{overnight_html}{block_html}
        <div class="grid grid-cols-4 gap-2 mb-3 text-xs">
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">Daily P&L</div>
                <div class="{daily_color} font-mono font-bold">${daily_pnl:,.2f}</div>
            </div>
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">Trades</div>
                <div class="text-zinc-200 font-mono">{open_count}/{max_trades}</div>
            </div>
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">Risk %</div>
                <div class="{risk_pct_color} font-mono">{risk_pct_emoji} {risk_pct:.1f}%</div>
            </div>
            <div class="bg-zinc-800/60 rounded p-1.5 text-center">
                <div class="text-zinc-500">L-Streak</div>
                <div class="{"text-red-400" if consecutive > 1 else "text-zinc-200"} font-mono">{consecutive}</div>
            </div>
        </div>
        """

    # ---- Empty state ----
    if not positions:
        return f"""
        <div id="positions-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">LIVE POSITIONS</h3>
            {risk_bar}
            <div class="flex items-center gap-2 text-zinc-500">
                <span class="text-green-500">✓</span>
                <span>No open positions</span>
            </div>
        </div>
        """

    # ---- Position rows ----
    rows = ""
    total_pnl = 0.0
    for pos in positions:
        sym = pos.get("symbol", pos.get("instrument", "?"))
        side = pos.get("side", pos.get("direction", "?"))
        qty = pos.get("quantity", pos.get("contracts", 0))
        avg_price = pos.get("avgPrice", pos.get("avg_price", pos.get("entry", 0)))
        unrealized = pos.get("unrealizedPnL", pos.get("unrealized_pnl", pos.get("pnl", 0)))
        total_pnl += unrealized

        side_color = "text-green-400" if side.upper() in ("LONG", "BUY") else "text-red-400"
        pnl_color = "text-green-400" if unrealized >= 0 else "text-red-400"

        rows += f"""
        <tr class="border-b border-zinc-800">
            <td class="py-1 text-white font-mono text-sm">{sym}</td>
            <td class="py-1 {side_color} text-sm font-semibold">{side.upper()}</td>
            <td class="py-1 text-zinc-300 text-sm text-center">{qty}</td>
            <td class="py-1 text-zinc-300 font-mono text-sm text-right">{avg_price:,.2f}</td>
            <td class="py-1 {pnl_color} font-mono text-sm text-right">${unrealized:,.2f}</td>
        </tr>
        """

    total_color = "text-green-400" if total_pnl >= 0 else "text-red-400"

    return f"""
    <div id="positions-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-zinc-400">LIVE POSITIONS</h3>
            <span class="{total_color} font-mono font-bold">${total_pnl:,.2f}</span>
        </div>
        {risk_bar}
        <table class="w-full">
            <thead>
                <tr class="text-xs text-zinc-500 border-b border-zinc-700">
                    <th class="text-left py-1">Symbol</th>
                    <th class="text-left py-1">Side</th>
                    <th class="text-center py-1">Qty</th>
                    <th class="text-right py-1">Avg Price</th>
                    <th class="text-right py-1">P&L</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def _render_risk_panel(risk_status: dict[str, Any] | None) -> str:
    """Render standalone risk status panel as HTML fragment."""
    if not risk_status:
        return """
        <div id="risk-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">RISK STATUS</h3>
            <div class="text-zinc-500 text-sm">Waiting for risk engine...</div>
        </div>
        """

    daily_pnl = risk_status.get("daily_pnl", 0)
    max_daily = risk_status.get("max_daily_loss", -500)
    risk_per_trade = risk_status.get("max_risk_per_trade", 375)
    can_trade = risk_status.get("can_trade", True)
    block_reason = risk_status.get("block_reason", "")
    risk_pct = risk_status.get("risk_pct_of_account", 0)
    rules = risk_status.get("rules", {})

    status_emoji = "🟢" if can_trade else "🔴"
    status_text = "CLEAR" if can_trade else "BLOCKED"
    status_color = "text-green-400" if can_trade else "text-red-400"
    pnl_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"

    # Daily P&L progress bar (relative to max daily loss)
    pnl_pct = min(abs(daily_pnl / max_daily) * 100, 100) if max_daily != 0 else 0
    pnl_bar_color = "bg-green-500" if daily_pnl >= 0 else ("bg-red-500" if pnl_pct > 60 else "bg-yellow-500")

    block_html = ""
    if not can_trade:
        block_html = f"""
        <div class="bg-red-900/40 border border-red-700 rounded px-3 py-1.5 mt-2">
            <span class="text-red-400 text-xs">🚫 {block_reason}</span>
        </div>
        """

    return f"""
    <div id="risk-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-zinc-400">RISK STATUS</h3>
            <span class="{status_color} text-sm font-bold">{status_emoji} {status_text}</span>
        </div>

        <div class="space-y-2 text-xs">
            <div>
                <div class="flex justify-between text-zinc-400 mb-0.5">
                    <span>Daily P&L</span>
                    <span class="{pnl_color} font-mono">${daily_pnl:,.2f} / ${max_daily:,.2f}</span>
                </div>
                <div class="w-full bg-zinc-700 rounded-full h-1.5">
                    <div class="{pnl_bar_color} h-1.5 rounded-full transition-all duration-500"
                         style="width: {pnl_pct:.0f}%"></div>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-2">
                <div class="text-zinc-400">Max risk/trade: <span class="text-zinc-200 font-mono">${risk_per_trade:,.0f}</span></div>
                <div class="text-zinc-400">Exposure: <span class="text-zinc-200 font-mono">{risk_pct:.1f}%</span></div>
                <div class="text-zinc-400">Cutoff: <span class="text-zinc-200 font-mono">{rules.get("no_entry_after", "10:00")} ET</span></div>
                <div class="text-zinc-400">Close by: <span class="text-zinc-200 font-mono">{rules.get("session_end", "12:00")} ET</span></div>
            </div>
        </div>
        {block_html}
    </div>
    """


def _render_grok_panel(grok_data: dict[str, Any] | None) -> str:
    """Render the Grok compact update panel as HTML fragment (TASK-601)."""
    if not grok_data:
        return """
        <div id="grok-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">🤖 GROK UPDATE</h3>
            <div class="text-zinc-500 text-sm">Waiting for next update...</div>
        </div>
        """

    text = grok_data.get("text", "")
    time_et = grok_data.get("time_et", "")

    # Convert the compact text to styled HTML lines
    lines_html = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines_html += '<div class="h-1"></div>'
            continue
        if stripped.upper().startswith("DO NOW"):
            # Highlight the DO NOW line
            lines_html += f'<div class="text-yellow-300 font-bold text-sm mt-1">{stripped}</div>'
        else:
            # Asset status line — detect emoji for coloring
            css = "text-zinc-200"
            if "🟢" in stripped:
                css = "text-green-300"
            elif "🔴" in stripped:
                css = "text-red-300"
            elif "⚪" in stripped:
                css = "text-zinc-400"
            lines_html += f'<div class="{css} font-mono text-xs">{stripped}</div>'

    return f"""
    <div id="grok-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-zinc-400">🤖 GROK UPDATE</h3>
            <span class="text-xs text-zinc-500">{time_et}</span>
        </div>
        <div class="space-y-0.5">
            {lines_html}
        </div>
    </div>
    """


def _render_orb_session_card(session_data: dict[str, Any] | None, session_label: str, session_times: str) -> str:
    """Render a single ORB session sub-card (London or US)."""
    if not session_data:
        return f"""
        <div class="bg-zinc-800/40 rounded px-3 py-2">
            <div class="flex items-center justify-between mb-1">
                <span class="text-zinc-400 text-xs font-semibold">{session_label}</span>
                <span class="text-zinc-600 text-[10px]">{session_times}</span>
            </div>
            <div class="text-zinc-600 text-xs">Waiting for opening range...</div>
        </div>
        """

    or_high = session_data.get("or_high", 0)
    or_low = session_data.get("or_low", 0)
    or_range = session_data.get("or_range", 0)
    atr_value = session_data.get("atr_value", 0)
    long_trigger = session_data.get("long_trigger", 0)
    short_trigger = session_data.get("short_trigger", 0)
    breakout = session_data.get("breakout_detected", False)
    direction = session_data.get("direction", "")
    trigger_price = session_data.get("trigger_price", 0)
    symbol = session_data.get("symbol", "")
    or_complete = session_data.get("or_complete", False)
    error = session_data.get("error", "")
    cnn_prob = session_data.get("cnn_prob")
    cnn_confidence = session_data.get("cnn_confidence", "")
    filter_passed = session_data.get("filter_passed")
    filter_summary = session_data.get("filter_summary", "")

    if error:
        return f"""
        <div class="bg-zinc-800/40 rounded px-3 py-2">
            <div class="flex items-center justify-between mb-1">
                <span class="text-zinc-400 text-xs font-semibold">{session_label}</span>
                <span class="text-zinc-600 text-[10px]">{session_times}</span>
            </div>
            <div class="text-zinc-500 text-xs">{error}</div>
        </div>
        """

    # Status for this session
    if breakout:
        if direction == "LONG":
            s_emoji = "🟢"
            s_color = "text-green-400"
            border = "border-green-600/40"
        else:
            s_emoji = "🔴"
            s_color = "text-red-400"
            border = "border-red-600/40"
    elif or_complete:
        s_emoji = "⏳"
        s_color = "text-yellow-400"
        border = "border-zinc-700/40"
    else:
        s_emoji = "🔄"
        s_color = "text-zinc-500"
        border = "border-zinc-700/40"

    # Breakout banner
    breakout_html = ""
    if breakout:
        bo_color = "green" if direction == "LONG" else "red"
        breakout_html = f"""
            <div class="bg-{bo_color}-900/40 border border-{bo_color}-600/50 rounded px-2 py-1 mb-1.5 text-center animate-pulse">
                <span class="{s_color} text-xs font-bold">
                    {s_emoji} {direction} @ {trigger_price:,.2f} — {symbol}
                </span>
            </div>
        """

    # CNN badge
    cnn_html = ""
    if cnn_prob is not None:
        cnn_pct = cnn_prob * 100
        if cnn_pct >= 65:
            cnn_color = "text-green-400"
        elif cnn_pct >= 45:
            cnn_color = "text-yellow-400"
        else:
            cnn_color = "text-red-400"
        cnn_html = f"""<span class="{cnn_color} text-[10px] font-mono ml-1" title="CNN P(good)={cnn_prob:.3f} ({cnn_confidence})">🧠 {cnn_pct:.0f}%</span>"""

    # Filter badge
    filter_html = ""
    if filter_passed is not None:
        if filter_passed:
            filter_html = f"""<span class="text-green-500 text-[10px] ml-1" title="{filter_summary}">✅</span>"""
        else:
            filter_html = f"""<span class="text-red-500 text-[10px] ml-1" title="{filter_summary}">🚫</span>"""

    # Status text
    if breakout:
        status_text = f"{direction} BREAKOUT"
    elif or_complete:
        status_text = "Watching"
    elif or_high > 0:
        status_text = "Forming"
    else:
        status_text = "Waiting"

    return f"""
    <div class="bg-zinc-800/40 border {border} rounded px-3 py-2">
        <div class="flex items-center justify-between mb-1">
            <span class="text-zinc-400 text-xs font-semibold">{session_label}{cnn_html}{filter_html}</span>
            <span class="{s_color} text-[10px] font-bold">{s_emoji} {status_text}</span>
        </div>
        {breakout_html}
        <div class="grid grid-cols-3 gap-x-2 gap-y-0.5 text-[10px]">
            <div class="text-zinc-500">High: <span class="text-green-300 font-mono">{or_high:,.2f}</span></div>
            <div class="text-zinc-500">Low: <span class="text-red-300 font-mono">{or_low:,.2f}</span></div>
            <div class="text-zinc-500">Range: <span class="text-zinc-300 font-mono">{or_range:,.2f}</span></div>
            <div class="text-zinc-500">ATR: <span class="text-zinc-300 font-mono">{atr_value:,.2f}</span></div>
            <div class="text-zinc-500">L↑: <span class="text-green-400 font-mono">{long_trigger:,.2f}</span></div>
            <div class="text-zinc-500">S↓: <span class="text-red-400 font-mono">{short_trigger:,.2f}</span></div>
        </div>
    </div>
    """


def _render_orb_panel(orb_data: dict[str, Any] | None) -> str:
    """Render the Opening Range Breakout panel as HTML fragment (TASK-801).

    Now supports multi-session display: London Open (03:00 ET) and
    US Equity Open (09:30 ET) are shown as separate sub-cards within
    the same panel.
    """
    if not orb_data:
        return """
        <div id="orb-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">📊 OPENING RANGE</h3>
            <div class="text-zinc-500 text-sm">Waiting for ORB sessions...</div>
            <div class="text-zinc-600 text-xs mt-1">London 03:00 ET · US 09:30 ET</div>
        </div>
        """

    # Multi-session format: orb_data has "london", "us", "best" keys
    london_data = orb_data.get("london") if isinstance(orb_data, dict) else None
    us_data = orb_data.get("us") if isinstance(orb_data, dict) else None
    best = orb_data.get("best") if isinstance(orb_data, dict) else None

    # Backward compat: if orb_data doesn't have session keys, treat it as
    # a single legacy result and slot it based on session_key
    if london_data is None and us_data is None and best is None:
        sk = orb_data.get("session_key", "us")
        if sk == "london":
            london_data = orb_data
        else:
            us_data = orb_data
        best = orb_data

    # Determine overall panel status from the best result
    has_breakout = best.get("breakout_detected", False) if best else False
    direction = best.get("direction", "") if best else ""

    if has_breakout:
        border_color = "border-green-500" if direction == "LONG" else "border-red-500"
    else:
        border_color = "border-zinc-700"

    # Time display from best result
    time_str = ""
    evaluated_at = best.get("evaluated_at", "") if best else ""
    if evaluated_at and "T" in evaluated_at:
        try:
            dt = datetime.fromisoformat(evaluated_at)
            time_str = dt.strftime("%I:%M %p")
        except Exception:
            time_str = evaluated_at

    # Render session sub-cards
    london_html = _render_orb_session_card(london_data, "🇬🇧 London Open", "03:00–03:30 ET")
    us_html = _render_orb_session_card(us_data, "🇺🇸 US Equity Open", "09:30–10:00 ET")

    return f"""
    <div id="orb-panel" class="bg-zinc-900/60 border {border_color} rounded-lg p-4"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold text-zinc-400">📊 OPENING RANGE</h3>
            <span class="text-zinc-600 text-xs">{time_str}</span>
        </div>
        <div class="space-y-2">
            {london_html}
            {us_html}
        </div>
    </div>
    """


def _render_full_dashboard(focus_data: dict[str, Any] | None, session: dict[str, str]) -> str:
    """Render the complete dashboard HTML page."""
    # Asset cards grid
    cards_html = ""
    if focus_data and focus_data.get("assets"):
        for asset in focus_data["assets"]:
            cards_html += _render_asset_card(asset)
    else:
        cards_html = """
        <div class="col-span-2 text-center py-12 text-zinc-500">
            <div class="text-4xl mb-4">📊</div>
            <div class="text-lg">Waiting for engine to compute daily focus...</div>
            <div class="text-sm mt-2">Data will appear automatically when ready.</div>
        </div>
        """

    # No-trade banner
    no_trade_html = ""
    if focus_data and focus_data.get("no_trade"):
        no_trade_html = _render_no_trade_banner(str(focus_data.get("no_trade_reason", "Low-conviction day")))

    # Determine if we're in off-hours (hide trading panels)
    is_off_hours = session.get("mode") == "off-hours"

    # Positions panel with risk status (TASK-501)
    positions = _get_positions()
    risk_status = _get_risk_status()
    positions_html = _render_positions_panel(positions, risk_status=risk_status)

    # Risk status panel
    risk_html = _render_risk_panel(risk_status)

    # Grok compact update panel (TASK-601)
    grok_data = _get_grok_update()
    grok_html = _render_grok_panel(grok_data)

    # Opening Range Breakout panel (TASK-801)
    orb_data = _get_orb_data()
    orb_html = _render_orb_panel(orb_data)

    # Focus summary
    total = focus_data.get("total_assets", 0) if focus_data else 0
    tradeable = focus_data.get("tradeable_assets", 0) if focus_data else 0
    computed = focus_data.get("computed_at", "—") if focus_data else "—"
    if isinstance(computed, str) and "T" in computed:
        try:
            dt = datetime.fromisoformat(computed)
            computed = dt.strftime("%I:%M %p ET")
        except Exception:
            pass

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Futures Trading Co-Pilot</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">

    <!-- Suppress Tailwind CDN production warning before it loads -->
    <script>
        (function() {{
            var origWarn = console.warn;
            console.warn = function() {{
                if (arguments.length > 0 && typeof arguments[0] === 'string' && arguments[0].includes('cdn.tailwindcss.com')) return;
                origWarn.apply(console, arguments);
            }};
        }})();
    </script>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- HTMX (polling/ajax only — SSE is handled by native EventSource below) -->
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>

    <!-- Hyperscript -->
    <script src="https://unpkg.com/hyperscript.org@0.9.14"></script>

    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .glow-green {{ box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); }}
        .glow-red {{ box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }}
        .glow-purple {{ box-shadow: 0 0 15px rgba(168, 85, 247, 0.2); }}
        /* Flash animation for live updates */
        @keyframes sse-flash {{
            0% {{ outline: 2px solid rgba(34, 197, 94, 0.8); outline-offset: -2px; }}
            100% {{ outline: 2px solid transparent; outline-offset: -2px; }}
        }}
        .sse-updated {{
            animation: sse-flash 1.2s ease-out;
        }}
        /* SSE connection indicator */
        #sse-status-dot.connected {{ color: #22c55e; }}
        #sse-status-dot.disconnected {{ color: #ef4444; }}
        #sse-status-dot.connecting {{ color: #eab308; }}
    </style>
</head>
<body class="bg-zinc-950 text-white min-h-screen">

    <!-- Main content container (SSE managed by native JS EventSource below) -->
    <div id="sse-container">

    <div class="max-w-7xl mx-auto px-4 py-4">
        <!-- Header — 3-column: title | NT8 health | clock + NT8 tools -->
        <header class="mb-6 border-b border-zinc-800 pb-4">
            <div class="flex items-center justify-between">
                <!-- Left: Title + SSE -->
                <div class="flex-shrink-0">
                    <h1 class="text-2xl font-bold">
                        <span class="text-zinc-400">Futures</span> Trading Co-Pilot
                    </h1>
                    <div class="text-sm text-zinc-500 mt-1">
                        {session["date"]}
                        <span id="sse-status-dot" class="connecting ml-2" title="SSE: connecting...">●</span>
                        <span id="sse-status-text" class="text-xs text-zinc-600">connecting</span>
                    </div>
                </div>

                <!-- Center: System Health Indicators (polled every 10s) -->
                <div id="nt8-health-bar"
                     class="hidden md:flex items-center"
                     hx-get="/api/nt8/health/html"
                     hx-trigger="load, every 10s"
                     hx-swap="innerHTML">
                    <div class="flex items-center gap-2">
                        <div class="flex items-center gap-2 pr-2 border-r border-zinc-700">
                            <div class="flex items-center gap-1.5 cursor-default" title="Data Service: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Data</span>
                            </div>
                            <div class="flex items-center gap-1.5 cursor-default" title="Engine: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Engine</span>
                            </div>
                            <div class="flex items-center gap-1.5 cursor-default" title="Redis: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Redis</span>
                            </div>
                            <div class="flex items-center gap-1.5 cursor-default" title="Postgres: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Postgres</span>
                            </div>
                        </div>
                        <div class="flex items-center gap-2 pl-1">
                            <div class="flex items-center gap-1.5 cursor-default" title="Bridge: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Bridge</span>
                            </div>
                            <div class="flex items-center gap-1.5 cursor-default" title="Ruby: checking...">
                                <span class="relative flex h-2.5 w-2.5">
                                    <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-zinc-600 ring-2 ring-zinc-600/30"></span>
                                </span>
                                <span class="text-[11px] text-zinc-500">Ruby</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right: Clock + NT8 Tools Dropdown -->
                <div class="flex items-center gap-3">
                    <div class="text-right">
                        <div id="clock" class="text-2xl font-mono font-bold {session["css_class"]}">
                            {session["time_et"]}
                        </div>
                        <div id="session-badge" class="text-sm font-semibold mt-1 {session["css_class"]}">
                            {session["emoji"]} {session["label"]}
                        </div>
                    </div>

                    <!-- NT8 Tools Dropdown -->
                    <div id="nt8-toolbar-container"
                         hx-get="/api/nt8/panel/html"
                         hx-trigger="load"
                         hx-swap="innerHTML">
                    </div>
                </div>
            </div>
        </header>

        <!-- No Trade Banner — updated via JS htmx:sseMessage handler (fetches /api/no-trade) -->
        <div id="no-trade-container"
             _="on `no-trade-alert` add .glow-red to me then wait 2s then remove .glow-red from me">
            {no_trade_html}
        </div>

        <!-- Focus Summary Bar -->
        <div id="focus-summary"
             class="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 rounded-lg px-4 py-2 mb-4">
            <div class="flex items-center gap-4 text-sm">
                <span class="text-zinc-400">TODAY'S FOCUS</span>
                <span id="focus-count" class="text-zinc-300">{tradeable}/{total} tradeable</span>
                <span id="focus-updated" class="text-zinc-500">Updated: {computed}</span>
            </div>
            <div class="flex items-center gap-2">
                <button hx-get="/api/focus/html"
                        hx-target="#focus-grid"
                        hx-swap="innerHTML"
                        hx-indicator="#refresh-spinner"
                        class="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded text-xs text-zinc-300
                               transition-colors duration-200 border border-zinc-700">
                    ↻ Refresh
                </button>
                <span id="refresh-spinner" class="htmx-indicator text-zinc-500 text-xs">Loading...</span>
            </div>
        </div>

        <!-- Main Grid: Focus Cards + Sidebar -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <!-- Focus Cards (2/3 width) -->
            <div class="lg:col-span-2">
                <!-- SSE full focus swap target (fallback: HTMX polling every 30s) -->
                <div id="focus-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4"
                     hx-get="/api/focus/html"
                     hx-trigger="every 30s"
                     hx-swap="innerHTML">
                    {cards_html}
                </div>
            </div>

            <!-- Sidebar (1/3 width) -->
            <div class="space-y-4">
                <!-- Positions Panel — SSE live + HTMX polling fallback (TASK-501) -->
                <div id="positions-container"
                     hx-get="/api/positions/html"
                     hx-trigger="every 10s"
                     hx-swap="innerHTML">
                    {positions_html}
                </div>

                <!-- Risk Status Panel (TASK-502) — hidden during off-hours -->
                <div id="risk-container"
                     class="{"hidden" if is_off_hours else ""}"
                     hx-get="/api/risk/html"
                     hx-trigger="every 15s"
                     hx-swap="innerHTML">
                    {risk_html}
                </div>

                <!-- Grok Compact Update Panel (TASK-601) — hidden during off-hours -->
                <div id="grok-container"
                     class="{"hidden" if is_off_hours else ""}"
                     hx-get="/api/grok/html"
                     hx-trigger="every 60s"
                     hx-swap="innerHTML">
                    {grok_html}
                </div>

                <!-- Opening Range Breakout Panel (TASK-801) — hidden during off-hours -->
                <div id="orb-container"
                     class="{"hidden" if is_off_hours else ""}"
                     hx-get="/api/orb/html"
                     hx-trigger="every 30s"
                     hx-swap="innerHTML">
                    {orb_html}
                </div>

                <!-- CNN Model Panel — always visible (retrain can run anytime) -->
                <div id="cnn-panel"
                     class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
                     hx-get="/cnn/status/html"
                     hx-trigger="load, every 15s"
                     hx-swap="innerHTML">
                    <h3 class="text-sm font-semibold text-zinc-400 mb-2">🧠 CNN MODEL</h3>
                    <div class="text-zinc-500 text-sm">Loading model status...</div>
                </div>

                <!-- Alerts Panel — hidden during off-hours -->
                <div id="alerts-panel"
                     class="{"hidden" if is_off_hours else ""}"
                     hx-get="/api/alerts/html"
                     hx-trigger="every 30s"
                     hx-swap="innerHTML">
                    <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
                        <h3 class="text-sm font-semibold text-zinc-400 mb-2">ALERTS</h3>
                        <div class="text-zinc-500 text-sm">No alerts</div>
                    </div>
                </div>

                <!-- Engine Status -->
                <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
                    <h3 class="text-sm font-semibold text-zinc-400 mb-2">ENGINE STATUS</h3>
                    <div id="engine-status"
                         hx-get="/api/time"
                         hx-trigger="every 5s"
                         hx-swap="innerHTML"
                         class="text-xs text-zinc-500">
                        Connecting...
                    </div>
                </div>

                <!-- SSE Connection Health -->
                <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
                    <h3 class="text-sm font-semibold text-zinc-400 mb-2">LIVE FEED</h3>
                    <div id="sse-heartbeat" class="text-xs text-zinc-500">
                        Waiting for heartbeat...
                    </div>
                    <div id="sse-last-update" class="text-xs text-zinc-600 mt-1">
                        —
                    </div>
                </div>

                <!-- Off-Hours Info Panel — only shown during off-hours -->
                {
        ""
        if not is_off_hours
        else '''
                <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
                    <h3 class="text-sm font-semibold text-zinc-400 mb-2">📅 NEXT SESSION</h3>
                    <div class="text-xs text-zinc-500 space-y-1">
                        <div>🌙 Pre-market: <span class="text-purple-400">00:00–03:00 ET</span></div>
                        <div>🟢 London Open: <span class="text-green-400">03:00–08:00 ET</span></div>
                        <div>🟢 US Open: <span class="text-green-400">08:00–12:00 ET</span></div>
                        <div>⚙️ Off-hours: <span class="text-zinc-400">12:00–00:00 ET</span></div>
                        <div class="mt-2 pt-2 border-t border-zinc-700 text-zinc-400">
                            Engine is running backfill, optimization, and next-day prep.
                            Trading panels will appear when the next session begins.
                        </div>
                    </div>
                </div>
                '''
    }

            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-8 pt-4 border-t border-zinc-800 text-center text-xs text-zinc-600">
            Pilot v1.0 — Session rules: Pre-market 00–03 | Active 03–12 | Off-hours 12–00 ET
            | <a href="/sse/health" class="underline hover:text-zinc-400">SSE Health</a>
            | <a href="/api/info" class="underline hover:text-zinc-400">API Info</a>
        </footer>
    </div>

    </div><!-- end sse-container -->

    <!-- Live clock JS (updates every second, no page refresh needed) -->
    <script>
        function updateClock() {{
            const now = new Date();
            const et = now.toLocaleTimeString('en-US', {{
                timeZone: 'America/New_York',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: true
            }}) + ' ET';
            const el = document.getElementById('clock');
            if (el) el.textContent = et;

            // Update session badge
            const etHour = parseInt(now.toLocaleString('en-US', {{
                timeZone: 'America/New_York', hour: 'numeric', hour12: false
            }}));
            const badge = document.getElementById('session-badge');
            if (badge) {{
                if (etHour >= 0 && etHour < 3) {{
                    badge.innerHTML = '🌙 PRE-MARKET';
                    badge.className = 'text-sm font-semibold mt-1 text-purple-400';
                    if (el) el.className = 'text-2xl font-mono font-bold text-purple-400';
                }} else if (etHour >= 3 && etHour < 8) {{
                    badge.innerHTML = '🟢 LONDON OPEN';
                    badge.className = 'text-sm font-semibold mt-1 text-green-400';
                    if (el) el.className = 'text-2xl font-mono font-bold text-green-400';
                }} else if (etHour >= 8 && etHour < 12) {{
                    badge.innerHTML = '🟢 US OPEN';
                    badge.className = 'text-sm font-semibold mt-1 text-green-400';
                    if (el) el.className = 'text-2xl font-mono font-bold text-green-400';
                }} else {{
                    badge.innerHTML = '⚙️ OFF-HOURS';
                    badge.className = 'text-sm font-semibold mt-1 text-zinc-400';
                    if (el) el.className = 'text-2xl font-mono font-bold text-zinc-400';
                }}
            }}
        }}
        setInterval(updateClock, 1000);
        updateClock();
    </script>

    <!-- SSE via native EventSource — no htmx-ext-sse needed.
         Opening the EventSource AFTER window.load guarantees Firefox
         won't log "connection interrupted while the page was loading". -->
    <script>
        (function() {{
            var _es = null;           // current EventSource instance
            var _reconnectMs = 3000;  // base reconnect delay
            var _maxReconnect = 30000;
            var _curDelay = _reconnectMs;

            function _setStatus(state) {{
                var dot = document.getElementById('sse-status-dot');
                var txt = document.getElementById('sse-status-text');
                if (state === 'connected') {{
                    if (dot) {{ dot.className = 'connected ml-2'; dot.title = 'SSE: connected'; }}
                    if (txt) {{ txt.textContent = 'live'; txt.className = 'text-xs text-green-600'; }}
                }} else if (state === 'connecting') {{
                    if (dot) {{ dot.className = 'connecting ml-2'; dot.title = 'SSE: connecting...'; }}
                    if (txt) {{ txt.textContent = 'connecting'; txt.className = 'text-xs text-yellow-600'; }}
                }} else {{
                    if (dot) {{ dot.className = 'disconnected ml-2'; dot.title = 'SSE: disconnected'; }}
                    if (txt) {{ txt.textContent = 'reconnecting...'; txt.className = 'text-xs text-red-600'; }}
                }}
            }}

            function _connect() {{
                if (_es) {{
                    try {{ _es.close(); }} catch(e) {{}}
                }}
                _setStatus('connecting');
                _es = new EventSource('/sse/dashboard');

                _es.onopen = function() {{
                    _setStatus('connected');
                    _curDelay = _reconnectMs;  // reset backoff on success
                }};

                _es.onerror = function() {{
                    _setStatus('disconnected');
                    // EventSource auto-reconnects, but if it moves to CLOSED
                    // state we need to manually reconnect with backoff.
                    if (_es && _es.readyState === EventSource.CLOSED) {{
                        setTimeout(function() {{
                            _curDelay = Math.min(_curDelay * 1.5, _maxReconnect);
                            _connect();
                        }}, _curDelay);
                    }}
                }};

                // --- Connected confirmation ---
                _es.addEventListener('connected', function() {{
                    _setStatus('connected');
                }});

                // --- Focus update: refresh summary bar + cards ---
                _es.addEventListener('focus-update', function(e) {{
                    try {{
                        var focus = JSON.parse(e.data);
                        var countEl = document.getElementById('focus-count');
                        if (countEl) {{
                            var tradeable = focus.tradeable_assets || 0;
                            var total = focus.total_assets || 0;
                            countEl.textContent = tradeable + '/' + total + ' tradeable';
                        }}
                        var updEl = document.getElementById('focus-updated');
                        if (updEl) {{
                            var ts = focus.computed_at || '';
                            if (ts) {{
                                var d = new Date(ts);
                                updEl.textContent = 'Updated: ' + d.toLocaleTimeString('en-US', {{
                                    timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true
                                }}) + ' ET';
                            }}
                        }}
                        var lastUpd = document.getElementById('sse-last-update');
                        if (lastUpd) {{
                            lastUpd.textContent = 'Last focus: ' + new Date().toLocaleTimeString();
                        }}
                        // Refresh focus grid HTML via HTMX
                        if (typeof htmx !== 'undefined') {{
                            htmx.ajax('GET', '/api/focus/html', {{target: '#focus-grid', swap: 'innerHTML'}});
                        }}
                    }} catch(err) {{ /* ignore parse errors */ }}
                }});

                // --- Heartbeat ---
                _es.addEventListener('heartbeat', function(e) {{
                    try {{
                        var hb = JSON.parse(e.data);
                        var hbEl = document.getElementById('sse-heartbeat');
                        if (hbEl) {{
                            hbEl.innerHTML = '<span class="text-green-500">●</span> Connected — ' + (hb.time_et || '');
                        }}
                    }} catch(err) {{}}
                    if (typeof htmx !== 'undefined') {{
                        htmx.ajax('GET', '/api/nt8/health/html', {{target: '#nt8-health-bar', swap: 'innerHTML'}});
                    }}
                }});

                // --- Session change ---
                _es.addEventListener('session-change', function(e) {{
                    try {{
                        var sc = JSON.parse(e.data);
                        var badge = document.getElementById('session-badge');
                        if (badge && sc.emoji && sc.session) {{
                            var label = sc.session.replace('_', '-').toUpperCase();
                            badge.innerHTML = sc.emoji + ' ' + label;
                        }}
                    }} catch(err) {{}}
                }});

                // --- No-trade alert ---
                _es.addEventListener('no-trade-alert', function(e) {{
                    try {{
                        var nt = JSON.parse(e.data);
                        if (nt.no_trade && typeof htmx !== 'undefined') {{
                            htmx.ajax('GET', '/api/no-trade', {{target: '#no-trade-container', swap: 'innerHTML'}});
                            var ntc = document.getElementById('no-trade-container');
                            if (ntc) {{ ntc.dispatchEvent(new CustomEvent('no-trade-alert', {{bubbles: false}})); }}
                        }}
                    }} catch(err) {{}}
                }});

                // --- Positions update ---
                _es.addEventListener('positions-update', function() {{
                    if (typeof htmx !== 'undefined') {{
                        htmx.ajax('GET', '/api/positions/html', {{target: '#positions-container', swap: 'innerHTML'}});
                        htmx.ajax('GET', '/api/nt8/health/html', {{target: '#nt8-health-bar', swap: 'innerHTML'}});
                    }}
                }});

                // --- Grok update (TASK-602) ---
                _es.addEventListener('grok-update', function() {{
                    if (typeof htmx !== 'undefined') {{
                        htmx.ajax('GET', '/api/grok/html', {{target: '#grok-container', swap: 'innerHTML'}});
                    }}
                    var lastUpd = document.getElementById('sse-last-update');
                    if (lastUpd) {{
                        lastUpd.textContent = 'Last Grok: ' + new Date().toLocaleTimeString();
                    }}
                }});

                // --- Risk update ---
                _es.addEventListener('risk-update', function() {{
                    if (typeof htmx !== 'undefined') {{
                        htmx.ajax('GET', '/api/risk/html', {{target: '#risk-container', swap: 'innerHTML'}});
                    }}
                }});

                // --- ORB update (TASK-801) ---
                _es.addEventListener('orb-update', function(e) {{
                    if (typeof htmx !== 'undefined') {{
                        htmx.ajax('GET', '/api/orb/html', {{target: '#orb-container', swap: 'innerHTML'}});
                    }}
                    try {{
                        var orbData = JSON.parse(e.data);
                        if (orbData.breakout_detected) {{
                            var orbPanel = document.getElementById('orb-container');
                            if (orbPanel) {{
                                orbPanel.classList.add('sse-updated');
                                setTimeout(function() {{ orbPanel.classList.remove('sse-updated'); }}, 2000);
                            }}
                        }}
                    }} catch(err) {{}}
                }});

                // --- Per-asset updates (e.g. gold-update, nasdaq-update) ---
                // We listen for any message event and check the type
                _es.onmessage = function(e) {{
                    // onmessage only fires for events without a named type,
                    // which shouldn't happen here. Ignore.
                }};
            }}

            // Helper: register per-asset listeners after we know which
            // assets exist (read from DOM).  Called once after connect.
            function _registerAssetListeners() {{
                var cards = document.querySelectorAll('[id^="asset-card-"]');
                cards.forEach(function(card) {{
                    var sym = card.id.replace('asset-card-', '');
                    if (sym && _es) {{
                        _es.addEventListener(sym + '-update', function() {{
                            if (typeof htmx !== 'undefined') {{
                                htmx.ajax('GET', '/api/focus/' + encodeURIComponent(sym), {{target: card, swap: 'outerHTML'}});
                            }}
                            setTimeout(function() {{
                                var updated = document.getElementById('asset-card-' + sym);
                                if (updated) {{
                                    updated.classList.add('sse-updated');
                                    setTimeout(function() {{ updated.classList.remove('sse-updated'); }}, 1500);
                                }}
                            }}, 100);
                        }});
                    }}
                }});
            }}

            // ---------------------------------------------------------------
            // Wait for window.load to guarantee all external scripts are done
            // and Firefox considers the page fully loaded.  Only THEN open
            // the EventSource — this eliminates the "connection interrupted
            // while the page was loading" console error.
            // ---------------------------------------------------------------
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    _connect();
                    _registerAssetListeners();
                }}, 100);
            }});
        }})();
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    """Serve the full HTML dashboard page."""
    focus_data = _get_focus_data()
    session = _get_session_info()
    html = _render_full_dashboard(focus_data, session)
    return HTMLResponse(content=html)


@router.get("/api/focus")
def get_focus():
    """Return all asset focus data as JSON (for programmatic consumers)."""
    focus_data = _get_focus_data()
    if focus_data is None:
        return JSONResponse(
            content={
                "assets": [],
                "no_trade": False,
                "no_trade_reason": "",
                "computed_at": None,
                "message": "Focus data not yet computed. Engine may still be starting.",
            }
        )
    return JSONResponse(content=focus_data)


@router.get("/api/focus/html", response_class=HTMLResponse)
def get_focus_html(request: Request):
    """Return all asset cards as HTML fragments (for HTMX swap).

    When called by HTMX polling and there is no data (Redis key expired),
    return 204 No Content so HTMX keeps the existing DOM intact instead
    of replacing visible cards with a "waiting" placeholder.
    """
    focus_data = _get_focus_data()
    is_htmx = request.headers.get("HX-Request") == "true"

    if not focus_data or not focus_data.get("assets"):
        if is_htmx:
            # 204 tells HTMX "nothing new, keep what you have"
            return Response(status_code=204)
        return HTMLResponse(
            content="""
            <div class="col-span-2 text-center py-12 text-zinc-500">
                <div class="text-4xl mb-4">📊</div>
                <div class="text-lg">Waiting for engine to compute daily focus...</div>
                <div class="text-sm mt-2">Data will appear automatically when ready.</div>
            </div>
            """
        )

    html = ""
    for asset in focus_data["assets"]:
        html += _render_asset_card(asset)
    return HTMLResponse(content=html)


@router.get("/api/focus/{symbol}", response_class=HTMLResponse)
def get_focus_symbol(request: Request, symbol: str):
    """Return a single asset card as HTML fragment."""
    focus_data = _get_focus_data()
    is_htmx = request.headers.get("HX-Request") == "true"
    if not focus_data:
        if is_htmx:
            return Response(status_code=204)
        return HTMLResponse(content="<div class='text-zinc-500'>No data</div>")

    # Find matching asset (case-insensitive)
    for asset in focus_data.get("assets", []):
        asset_symbol = asset.get("symbol", "").lower().replace(" ", "_")
        if asset_symbol == symbol.lower().replace(" ", "_"):
            return HTMLResponse(content=_render_asset_card(asset))

    return HTMLResponse(content=f"<div class='text-zinc-500'>Asset '{symbol}' not found</div>")


@router.get("/api/positions/html", response_class=HTMLResponse)
def get_positions_html():
    """Return live positions panel with risk status as HTML fragment.

    Always returns content (positions panel renders even when empty),
    so no 204 guard needed here.
    """
    positions = _get_positions()
    risk_status = _get_risk_status()
    return HTMLResponse(content=_render_positions_panel(positions, risk_status=risk_status))


@router.get("/api/risk/html", response_class=HTMLResponse)
def get_risk_html():
    """Return risk status panel as HTML fragment (TASK-502).

    The risk panel always renders a container even when status is None
    (shows 'Waiting for risk engine...'), so always return HTML.
    """
    risk_status = _get_risk_status()
    return HTMLResponse(content=_render_risk_panel(risk_status))


@router.get("/api/grok/html", response_class=HTMLResponse)
def get_grok_html():
    """Return Grok compact update panel as HTML fragment (TASK-601)."""
    grok_data = _get_grok_update()
    return HTMLResponse(content=_render_grok_panel(grok_data))


@router.get("/api/orb/html")
def get_orb_html():
    """Return Opening Range Breakout panel as HTML fragment (TASK-801)."""
    orb_data = _get_orb_data()
    return HTMLResponse(content=_render_orb_panel(orb_data))


@router.get("/api/alerts/html", response_class=HTMLResponse)
def get_alerts_html():
    """Return alerts panel as HTML fragment."""
    # Read alerts from Redis if available
    alerts = []
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:alerts")
        if raw:
            alerts = json.loads(raw)
    except Exception:
        pass

    if not alerts:
        return HTMLResponse(
            content="""
            <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-zinc-400 mb-2">ALERTS</h3>
                <div class="text-zinc-500 text-sm">No alerts</div>
            </div>
            """
        )

    rows = ""
    for alert in alerts[-10:]:  # Show last 10
        msg = alert.get("message", alert.get("title", "Alert"))
        _ts = alert.get("timestamp", "")  # noqa: F841
        level = alert.get("level", "info")
        color = {
            "warning": "text-yellow-400",
            "error": "text-red-400",
            "success": "text-green-400",
        }.get(level, "text-zinc-400")

        rows += f'<div class="{color} text-xs py-1 border-b border-zinc-800">{msg}</div>'

    return HTMLResponse(
        content=f"""
        <div class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">ALERTS</h3>
            {rows}
        </div>
        """
    )


@router.get("/api/time", response_class=HTMLResponse)
def get_time():
    """Return formatted time + engine status as HTML fragment."""
    session = _get_session_info()

    # Try to get engine status
    engine_info = ""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:status")
        if raw:
            status = json.loads(raw)
            engine_state = status.get("engine", "unknown")
            session_mode = status.get("session_mode", "unknown")
            data_refresh = status.get("data_refresh", {})
            last_refresh = data_refresh.get("last", "—")

            engine_info = f"""
            <div class="space-y-1">
                <div>Engine: <span class="text-green-400">{engine_state}</span></div>
                <div>Session: <span class="{session["css_class"]}">{session_mode}</span></div>
                <div>Last refresh: {last_refresh}</div>
                <div>{session["time_et"]}</div>
            </div>
            """
        else:
            engine_info = f"""
            <div class="space-y-1">
                <div>Engine: <span class="text-yellow-400">connecting...</span></div>
                <div>{session["time_et"]}</div>
            </div>
            """
    except Exception:
        engine_info = f"<div>{session['time_et']}</div>"

    return HTMLResponse(content=engine_info)


@router.get("/api/no-trade", response_class=HTMLResponse)
def get_no_trade():
    """Return no-trade banner HTML if applicable, empty otherwise."""
    focus_data = _get_focus_data()
    if focus_data and focus_data.get("no_trade"):
        return HTMLResponse(
            content=_render_no_trade_banner(str(focus_data.get("no_trade_reason", "Low-conviction day")))
        )
    return HTMLResponse(content='<div id="no-trade-banner"></div>')
