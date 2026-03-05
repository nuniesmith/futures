"""
Dashboard API Router
=====================
ORB-centric trading dashboard for NinjaTrader + Ruby indicator workflow.

Layout:
  - Header bar: service health dots + market session clock strip
  - Session strip: live visual timeline of all major futures sessions
    with overlap highlighting and current-time cursor
  - Main (2/3): ORB detection cards per symbol, with CNN probability,
    filter results, and NT8/Ruby metric validation side-by-side
  - Sidebar (1/3): live positions + P&L, risk status, market events feed,
    Grok brief, CNN model status
  - Footer: session schedule reference

Endpoints:
    GET /                       — Full HTML dashboard page
    GET /api/focus              — All asset focus data as JSON
    GET /api/focus/html         — All asset cards as HTML fragments
    GET /api/focus/{symbol}     — Single asset card HTML fragment
    GET /api/positions/html     — Live positions panel HTML
    GET /api/risk/html          — Risk status panel HTML
    GET /api/grok/html          — Grok compact update panel HTML
    GET /api/alerts/html        — Alerts panel HTML
    GET /api/orb/html           — ORB panel HTML
    GET /api/market-session/html — Session clock strip HTML
    GET /api/time               — Formatted time string with session indicator
    GET /api/no-trade           — No-trade banner HTML (if applicable)
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
      - Pre-market:           00:00–03:00
      - Active / London Open: 03:00–08:00
      - Active / US Open:     08:00–12:00
      - Off-hours:            12:00–00:00
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


# ---------------------------------------------------------------------------
# Market session definitions (all times in ET, displayed on the strip)
# Each session: (label, short, start_hour_ET, end_hour_ET, color_class, bg_class)
# ---------------------------------------------------------------------------
_SESSIONS = [
    # (label,          short,   start, end,  bar_color,           text_color,       overlap_note)
    ("Sydney", "SYD", 17, 2, "bg-slate-600", "text-slate-300", ""),
    ("Tokyo", "TYO", 19, 4, "bg-indigo-700", "text-indigo-200", ""),
    ("London", "LON", 3, 12, "bg-blue-700", "text-blue-200", ""),
    ("US Equity", "US", 9, 16, "bg-emerald-700", "text-emerald-200", ""),
    ("US Futures", "CME", 18, 17, "bg-teal-800", "text-teal-300", ""),  # ~23h
]

# ORB window markers shown on the session strip
_ORB_WINDOWS = [
    ("London ORB", 3, 3.5, "border-blue-400"),
    ("US ORB", 9.5, 10.0, "border-emerald-400"),
]


def _render_session_strip() -> str:
    """Render the horizontal market session timeline strip.

    Shows a 24-hour bar (00:00–24:00 ET) with coloured session blocks,
    overlap highlights, ORB window markers, and a live cursor for now.
    Updated client-side via JS every minute.
    """
    # Build session blocks as percentage offsets (each hour = 100/24 %)
    HOUR_PCT = 100.0 / 24.0

    def _pct(h: float) -> str:
        return f"{h * HOUR_PCT:.3f}%"

    # Session blocks HTML
    # Sydney wraps midnight: render two segments
    # We use inline style for pixel-perfect positioning

    overlap_highlights = ""
    # London + US overlap: 09:30–12:00 ET  (9.5–12)
    overlap_highlights += f"""
        <div class="absolute top-0 bottom-0 border-l border-r border-yellow-400/30 bg-yellow-400/10"
             style="left:{_pct(9.5)};width:{_pct(2.5)}"
             title="London/US Overlap 09:30–12:00 ET"></div>
    """

    # ORB markers
    orb_markers = ""
    for orb_label, start_h, end_h, border_cls in _ORB_WINDOWS:
        orb_markers += f"""
        <div class="absolute top-0 bottom-0 {border_cls} border-l-2 border-r-2 bg-white/5"
             style="left:{_pct(start_h)};width:{_pct(end_h - start_h)}"
             title="{orb_label}">
            <span class="absolute top-0.5 left-0.5 text-[8px] text-white/60 leading-none whitespace-nowrap">ORB</span>
        </div>
        """

    # Hour tick marks — every 3h on desktop, label only every 6h on mobile
    ticks = ""
    for h in range(0, 25, 3):
        label_h = h % 24
        # On small screens only render the 00/06/12/18 labels to avoid crowding
        label_mobile_class = "" if label_h % 6 == 0 else "hidden sm:block"
        ticks += f"""
        <div class="absolute top-0 bottom-0 border-l border-zinc-700/50"
             style="left:{_pct(h)}">
            <span class="absolute -bottom-4 text-[9px] text-zinc-600 -translate-x-1/2 {label_mobile_class}">{label_h:02d}</span>
        </div>
        """

    # Session label bars — stacked in two rows to avoid overlap
    # Row 0: Sydney, Tokyo, CME (background/futures)
    # Row 1: London, US Equity (foreground/primary)
    row0 = [
        ("Sydney", 17, 26, "#1e293b", "#94a3b8"),
        ("Tokyo", 19, 28, "#1e1b4b", "#a5b4fc"),
        ("CME 23h", 18, 41, "#042f2e", "#5eead4"),
    ]
    row1 = [("London", 3, 12, "#1e3a5f", "#93c5fd"), ("US", 9, 16, "#052e16", "#6ee7b7")]

    def _bar(label: str, s: int, e: int, bg: str, fg: str, row: int) -> str:
        # clamp to 0–24
        s24 = s % 24
        width = (e - s) if e <= 24 else (24 - s)
        width = min(width, 24 - s24)
        top = "1px" if row == 0 else "13px"
        height = "10px"
        return (
            f'<div class="absolute rounded-sm flex items-center px-1 overflow-hidden"'
            f' style="left:{_pct(s24)};width:{_pct(width)};top:{top};height:{height};'
            f'background:{bg};border:1px solid {fg}33" title="{label}">'
            f'<span style="color:{fg};font-size:7px;white-space:nowrap;line-height:1">{label}</span>'
            f"</div>"
        )

    bars_html = ""
    for label, s, e, bg, fg in row0:
        bars_html += _bar(label, s, e, bg, fg, 0)
    for label, s, e, bg, fg in row1:
        bars_html += _bar(label, s, e, bg, fg, 1)

    return f"""
    <div id="session-strip"
         class="bg-zinc-900/80 border border-zinc-800 rounded-lg px-2 sm:px-4 pt-3 pb-5 sm:pb-6 mb-3 sm:mb-4 relative"
         hx-get="/api/market-session/html"
         hx-trigger="every 60s"
         hx-swap="outerHTML">
        <div class="flex items-center justify-between mb-2">
            <span class="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Market Sessions (ET)</span>
            <div class="hidden sm:flex items-center gap-3 text-[9px] text-zinc-600">
                <span><span class="inline-block w-2 h-2 rounded-sm bg-blue-700 mr-1"></span>London</span>
                <span><span class="inline-block w-2 h-2 rounded-sm bg-emerald-700 mr-1"></span>US Equity</span>
                <span><span class="inline-block w-2 h-2 rounded-sm bg-yellow-400/30 border border-yellow-400/40 mr-1"></span>Overlap</span>
                <span><span class="inline-block w-1 h-2 border-l-2 border-blue-400 mr-1"></span>ORB Window</span>
            </div>
        </div>
        <!-- Timeline bar — min-width keeps it readable on narrow screens -->
        <div class="relative h-6 w-full min-w-[320px]" id="session-bar-inner">
            <!-- Background -->
            <div class="absolute inset-0 bg-zinc-800/60 rounded"></div>
            {overlap_highlights}
            {bars_html}
            {orb_markers}
            {ticks}
            <!-- Live cursor — positioned by JS -->
            <div id="session-cursor"
                 class="absolute top-0 bottom-0 w-0.5 bg-white/80 z-10 pointer-events-none"
                 style="left:0%">
                <div class="absolute -top-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-white rounded-full"></div>
            </div>
        </div>
        <!-- Open/closed badges — updated by JS -->
        <div id="session-badges" class="flex flex-wrap gap-1 sm:gap-1.5 mt-2 sm:mt-3">
            <span class="text-[9px] text-zinc-600 self-center">Loading sessions...</span>
        </div>
    </div>
    """


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
         class="border {border_color} rounded-lg p-3 sm:p-4 {bias_bg} {opacity} transition-all duration-300 t-panel"
         data-quality="{quality_pct}"
         data-wave="{wave_ratio}"
         data-bias="{bias}"
         data-symbol="{symbol_lower}"
         hx-swap-oob="true"
         _="on load if my @data-quality as Number < 55 add .opacity-50 to me
              else remove .opacity-50 from me end">

        <!-- Header -->
        <div class="flex items-center justify-between mb-2 sm:mb-3">
            <div class="flex items-center gap-2 min-w-0">
                <span class="text-xl sm:text-2xl shrink-0">{bias_emoji}</span>
                <h3 class="text-base sm:text-lg font-bold t-text truncate">{symbol}</h3>
            </div>
            <div class="text-right shrink-0 ml-2">
                <div class="text-lg sm:text-xl font-mono font-bold t-text">{last_price:,.2f}</div>
                <div class="text-xs {bias_text} font-semibold">{bias}</div>
            </div>
        </div>

        <!-- Quality & Wave -->
        <div class="grid grid-cols-2 gap-2 mb-2 sm:mb-3">
            <div>
                <div class="text-xs t-text-muted mb-1">Quality</div>
                <div class="w-full t-bar rounded-full h-2">
                    <div class="{q_color} h-2 rounded-full transition-all duration-500"
                         style="width: {min(quality_pct, 100):.0f}%"></div>
                </div>
                <div class="text-xs t-text-secondary mt-0.5">{quality_pct:.0f}%</div>
            </div>
            <div>
                <div class="text-xs t-text-muted mb-1">Wave Ratio</div>
                <div class="text-lg font-mono font-bold t-text">{wave_ratio:.2f}x</div>
            </div>
        </div>

        <!-- Levels -->
        <div class="grid grid-cols-3 gap-1 mb-2 sm:mb-3 text-xs">
            <div class="t-panel-inner rounded p-1 sm:p-1.5 text-center">
                <div class="t-text-muted text-[10px] sm:text-xs">Entry</div>
                <div class="t-text-secondary font-mono text-[10px] sm:text-xs">{entry_low:,.2f}</div>
                <div class="t-text-muted font-mono text-[10px] sm:text-xs">– {entry_high:,.2f}</div>
            </div>
            <div class="t-panel-inner rounded p-1 sm:p-1.5 text-center">
                <div class="t-text-muted text-[10px] sm:text-xs">Stop</div>
                <div class="text-red-400 font-mono text-[10px] sm:text-xs">{stop:,.2f}</div>
            </div>
            <div class="t-panel-inner rounded p-1 sm:p-1.5 text-center">
                <div class="t-text-muted text-[10px] sm:text-xs">TP1 / TP2</div>
                <div class="text-green-400 font-mono text-[10px] sm:text-xs">{tp1:,.2f}</div>
                <div class="text-green-300 font-mono text-[10px] sm:text-xs">{tp2:,.2f}</div>
            </div>
        </div>

        <!-- Meta row — wraps gracefully on mobile -->
        <div class="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] sm:text-xs t-text-muted">
            <span>{trend_dir}</span>
            <span>Vol: {vol_cluster} ({vol_pct:.0%})</span>
            <span>{position_size} micros / ${risk_dollars:,.0f} risk</span>
        </div>

        <!-- Notes -->
        {"<div class='mt-2 text-xs text-yellow-400 italic'>" + notes + "</div>" if notes else ""}
    </div>
    """


def _render_no_trade_banner(reason: str = "Low-conviction day") -> str:
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
    """Render condensed live positions + daily P&L panel."""
    daily_pnl = 0.0
    can_trade = True
    block_reason = ""
    consecutive = 0
    open_count = 0
    max_trades = 2
    risk_pct = 0.0

    block_html = ""
    if risk_status:
        daily_pnl = risk_status.get("daily_pnl", 0)
        can_trade = risk_status.get("can_trade", True)
        block_reason = risk_status.get("block_reason", "")
        consecutive = risk_status.get("consecutive_losses", 0)
        open_count = risk_status.get("open_trade_count", 0)
        max_trades = risk_status.get("max_open_trades", 2)
        risk_pct = risk_status.get("risk_pct_of_account", 0)
        if not can_trade and block_reason:
            block_html = f"""
            <div class="bg-red-900/40 border border-red-700 rounded px-2 py-1 mb-2">
                <span class="text-red-400 text-xs">🚫 {block_reason}</span>
            </div>
            """

    daily_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"
    pnl_sign = "+" if daily_pnl >= 0 else ""
    risk_color = "text-red-400 font-bold" if risk_pct > 5 else ("text-yellow-400" if risk_pct > 3 else "text-green-400")
    trade_color = "text-yellow-400" if open_count >= max_trades else "text-zinc-200"
    streak_color = "text-red-400" if consecutive > 1 else "text-zinc-400"

    # Stats row
    stats_html = f"""
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 mb-2 text-center">
        <div class="t-panel-inner rounded py-1">
            <div class="text-[9px] t-text-muted">Daily P&L</div>
            <div class="{daily_color} font-mono text-xs font-bold">{pnl_sign}${daily_pnl:,.0f}</div>
        </div>
        <div class="t-panel-inner rounded py-1">
            <div class="text-[9px] t-text-muted">Positions</div>
            <div class="{trade_color} font-mono text-xs">{open_count}/{max_trades}</div>
        </div>
        <div class="t-panel-inner rounded py-1">
            <div class="text-[9px] t-text-muted">Exposure</div>
            <div class="{risk_color} font-mono text-xs">{risk_pct:.1f}%</div>
        </div>
        <div class="t-panel-inner rounded py-1">
            <div class="text-[9px] t-text-muted">L-Streak</div>
            <div class="{streak_color} font-mono text-xs">{consecutive}</div>
        </div>
    </div>
    """

    if not positions:
        return f"""
        <div id="positions-panel" class="t-panel border t-border rounded-lg p-3"
             hx-swap-oob="true">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-2">Positions &amp; P&amp;L</h3>
            {block_html}{stats_html}
            <div class="text-xs t-text-faint flex items-center gap-1.5">
                <span class="text-green-500">✓</span> No open positions
            </div>
        </div>
        """

    rows = ""
    total_pnl = 0.0
    for pos in positions:
        sym = pos.get("symbol", pos.get("instrument", "?"))
        side = pos.get("side", pos.get("direction", "?")).upper()
        qty = pos.get("quantity", pos.get("contracts", 0))
        avg_price = pos.get("avgPrice", pos.get("avg_price", pos.get("entry", 0)))
        unrealized = pos.get("unrealizedPnL", pos.get("unrealized_pnl", pos.get("pnl", 0)))
        total_pnl += unrealized
        side_color = "text-green-400" if side in ("LONG", "BUY") else "text-red-400"
        pnl_color = "text-green-400" if unrealized >= 0 else "text-red-400"
        rows += f"""
        <tr class="border-b t-border-subtle">
            <td class="py-0.5 t-text font-mono text-xs">{sym}</td>
            <td class="py-0.5 {side_color} text-xs font-semibold">{side}</td>
            <td class="py-0.5 t-text-secondary text-xs text-center">{qty}</td>
            <td class="py-0.5 t-text-muted font-mono text-xs text-right hidden sm:table-cell">{avg_price:,.2f}</td>
            <td class="py-0.5 {pnl_color} font-mono text-xs text-right font-bold">${unrealized:,.2f}</td>
        </tr>
        """

    total_color = "text-green-400" if total_pnl >= 0 else "text-red-400"
    return f"""
    <div id="positions-panel" class="t-panel border t-border rounded-lg p-3"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Positions &amp; P&amp;L</h3>
            <span class="{total_color} font-mono text-xs font-bold">Open: ${total_pnl:,.2f}</span>
        </div>
        {block_html}{stats_html}
        <div class="overflow-x-auto -mx-1 px-1">
        <table class="w-full min-w-[200px]">
            <thead>
                <tr class="text-[9px] t-text-faint border-b t-border">
                    <th class="text-left pb-0.5">Sym</th>
                    <th class="text-left pb-0.5">Side</th>
                    <th class="text-center pb-0.5">Qty</th>
                    <th class="text-right pb-0.5 hidden sm:table-cell">Avg</th>
                    <th class="text-right pb-0.5">P&amp;L</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>
    """


def _render_risk_panel(risk_status: dict[str, Any] | None) -> str:
    """Render condensed risk rules panel."""
    if not risk_status:
        return """
        <div id="risk-panel" class="t-panel border t-border rounded-lg p-3"
             hx-swap-oob="true">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">Risk Rules</h3>
            <div class="t-text-faint text-xs">Waiting for risk engine...</div>
        </div>
        """

    daily_pnl = risk_status.get("daily_pnl", 0)
    max_daily = risk_status.get("max_daily_loss", -500)
    risk_per_trade = risk_status.get("max_risk_per_trade", 375)
    can_trade = risk_status.get("can_trade", True)
    rules = risk_status.get("rules", {})

    status_color = "text-green-400" if can_trade else "text-red-400"
    status_dot = "bg-green-500" if can_trade else "bg-red-500"
    status_text = "CLEAR" if can_trade else "BLOCKED"
    pnl_color = "text-green-400" if daily_pnl >= 0 else "text-red-400"
    pnl_pct = min(abs(daily_pnl / max_daily) * 100, 100) if max_daily != 0 else 0
    bar_color = "bg-green-500" if daily_pnl >= 0 else ("bg-red-500" if pnl_pct > 60 else "bg-yellow-500")

    return f"""
    <div id="risk-panel" class="t-panel border t-border rounded-lg p-3"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Risk Rules</h3>
            <span class="flex items-center gap-1 {status_color} text-xs font-bold">
                <span class="w-1.5 h-1.5 rounded-full {status_dot} inline-block"></span>
                {status_text}
            </span>
        </div>
        <div class="mb-1.5">
            <div class="flex justify-between text-[10px] t-text-muted mb-0.5">
                <span>Daily P&amp;L</span>
                <span class="{pnl_color} font-mono">${daily_pnl:,.0f} / ${max_daily:,.0f}</span>
            </div>
            <div class="w-full t-bar rounded-full h-1">
                <div class="{bar_color} h-1 rounded-full" style="width:{pnl_pct:.0f}%"></div>
            </div>
        </div>
        <div class="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] t-text-muted">
            <div>Max/trade: <span class="t-text-secondary font-mono">${risk_per_trade:,.0f}</span></div>
            <div>No entry: <span class="t-text-secondary font-mono">{rules.get("no_entry_after", "10:00")} ET</span></div>
            <div>Close by: <span class="t-text-secondary font-mono">{rules.get("session_end", "12:00")} ET</span></div>
        </div>
    </div>
    """


def _render_grok_panel(grok_data: dict[str, Any] | None) -> str:
    """Render condensed Grok AI brief panel."""
    if not grok_data:
        return """
        <div id="grok-panel" class="t-panel border t-border rounded-lg p-3"
             hx-swap-oob="true">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">🤖 AI Brief</h3>
            <div class="t-text-faint text-xs">Waiting for next update...</div>
        </div>
        """

    text = grok_data.get("text", "")
    time_et = grok_data.get("time_et", "")

    lines_html = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("DO NOW"):
            lines_html += f'<div class="text-yellow-300 font-bold text-[10px] mt-1 border-l-2 border-yellow-400 pl-1.5">{stripped}</div>'
        else:
            css = "t-text-muted"
            if "🟢" in stripped:
                css = "text-green-300"
            elif "🔴" in stripped:
                css = "text-red-300"
            lines_html += f'<div class="{css} font-mono text-[10px]">{stripped}</div>'

    return f"""
    <div id="grok-panel" class="t-panel border t-border rounded-lg p-3"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-1.5">
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🤖 AI Brief</h3>
            <span class="text-[9px] t-text-faint">{time_et}</span>
        </div>
        <div class="space-y-0.5 max-h-28 overflow-y-auto">
            {lines_html}
        </div>
    </div>
    """


def _render_orb_session_card(session_data: dict[str, Any] | None, session_label: str, session_times: str) -> str:
    """Render a single ORB session sub-card (London or US)."""
    if not session_data:
        return f"""
        <div class="t-panel-inner rounded px-3 py-2">
            <div class="flex items-center justify-between mb-1">
                <span class="t-text-muted text-xs font-semibold">{session_label}</span>
                <span class="t-text-faint text-[10px]">{session_times}</span>
            </div>
            <div class="t-text-faint text-xs">Waiting for opening range...</div>
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
        <div class="t-panel-inner rounded px-3 py-2">
            <div class="flex items-center justify-between mb-1">
                <span class="t-text-muted text-xs font-semibold">{session_label}</span>
                <span class="t-text-faint text-[10px]">{session_times}</span>
            </div>
            <div class="t-text-muted text-xs">{error}</div>
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
    <div class="t-panel-inner border {border} rounded px-3 py-2">
        <div class="flex items-center justify-between mb-1">
            <span class="t-text-muted text-xs font-semibold">{session_label}{cnn_html}{filter_html}</span>
            <span class="{s_color} text-[10px] font-bold">{s_emoji} {status_text}</span>
        </div>
        {breakout_html}
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-2 gap-y-0.5 text-[10px]">
            <div class="t-text-muted">High: <span class="text-green-300 font-mono">{or_high:,.2f}</span></div>
            <div class="t-text-muted">Low: <span class="text-red-300 font-mono">{or_low:,.2f}</span></div>
            <div class="t-text-muted">Range: <span class="t-text-secondary font-mono">{or_range:,.2f}</span></div>
            <div class="t-text-muted">ATR: <span class="t-text-secondary font-mono">{atr_value:,.2f}</span></div>
            <div class="t-text-muted">L↑: <span class="text-green-400 font-mono">{long_trigger:,.2f}</span></div>
            <div class="t-text-muted">S↓: <span class="text-red-400 font-mono">{short_trigger:,.2f}</span></div>
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
        <div id="orb-panel" class="t-panel border t-border rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold t-text-muted mb-2">📊 OPENING RANGE</h3>
            <div class="t-text-muted text-sm">Waiting for ORB sessions...</div>
            <div class="t-text-faint text-xs mt-1">London 03:00 ET · US 09:30 ET</div>
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
    <div id="orb-panel" class="t-panel border {border_color} rounded-lg p-4"
         hx-swap-oob="true">
        <div class="flex items-center justify-between mb-2">
            <h3 class="text-sm font-semibold t-text-muted">📊 OPENING RANGE</h3>
            <span class="t-text-faint text-xs">{time_str}</span>
        </div>
        <div class="space-y-2">
            {london_html}
            {us_html}
        </div>
    </div>
    """


def _render_market_events_panel() -> str:
    """Render a live market events / activity feed panel for the sidebar."""
    return """
    <div id="market-events-panel" class="t-panel border t-border rounded-lg p-3">
        <div class="flex items-center justify-between mb-1.5">
            <h3 class="text-xs font-semibold text-zinc-400 uppercase tracking-wide">Market Events</h3>
            <span class="text-[9px] text-zinc-600" id="events-ts">—</span>
        </div>
        <div id="market-events-feed" class="space-y-1 max-h-36 overflow-y-auto text-[10px]">
            <div class="text-zinc-600">Listening for ORB signals, fills, and alerts...</div>
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
        <div class="col-span-2 text-center py-12 t-text-muted">
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

    # Positions panel with risk status
    positions = _get_positions()
    risk_status = _get_risk_status()
    positions_html = _render_positions_panel(positions, risk_status=risk_status)

    # Risk panel
    risk_html = _render_risk_panel(risk_status)

    # Grok brief panel
    grok_data = _get_grok_update()
    grok_html = _render_grok_panel(grok_data)

    # ORB panel
    orb_data = _get_orb_data()
    orb_html = _render_orb_panel(orb_data)

    # Session strip
    session_strip_html = _render_session_strip()

    # Market events panel
    market_events_html = _render_market_events_panel()

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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>ORB Co-Pilot</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
    <!-- Apply saved theme BEFORE paint to prevent flash -->
    <script>
        (function() {{
            var t = localStorage.getItem('theme');
            if (t === 'light') document.documentElement.classList.remove('dark');
            else document.documentElement.classList.add('dark');
        }})();
    </script>
    <script>
        (function() {{
            var origWarn = console.warn;
            console.warn = function() {{
                if (arguments.length > 0 && typeof arguments[0] === 'string' && arguments[0].includes('cdn.tailwindcss.com')) return;
                origWarn.apply(console, arguments);
            }};
        }})();
    </script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{ darkMode: 'class' }};
    </script>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/hyperscript.org@0.9.14"></script>
    <style>
        /* ── Theme CSS custom properties ────────────────────────────── */
        /* Every panel renderer uses these variables so we only need   */
        /* to flip values here rather than adding dark: classes to     */
        /* hundreds of inline HTML strings.                             */
        :root {{
            --bg-body:        #f4f4f5;   /* zinc-100 */
            --bg-panel:       rgba(255,255,255,0.80);
            --bg-panel-inner: rgba(244,244,245,0.60);  /* zinc-100/60 */
            --bg-input:       #e4e4e7;   /* zinc-200 */
            --bg-bar:         #d4d4d8;   /* zinc-300 */
            --border-panel:   #d4d4d8;   /* zinc-300 */
            --border-subtle:  #e4e4e7;   /* zinc-200 */
            --text-primary:   #18181b;   /* zinc-900 */
            --text-secondary: #3f3f46;   /* zinc-700 */
            --text-muted:     #71717a;   /* zinc-500 */
            --text-faint:     #a1a1aa;   /* zinc-400 */
            --scrollbar-track: #f4f4f5;
            --scrollbar-thumb: #a1a1aa;
        }}
        .dark {{
            --bg-body:        #09090b;   /* zinc-950 */
            --bg-panel:       rgba(24,24,27,0.60);     /* zinc-900/60 */
            --bg-panel-inner: rgba(39,39,42,0.40);     /* zinc-800/40 */
            --bg-input:       #27272a;   /* zinc-800 */
            --bg-bar:         #3f3f46;   /* zinc-700 */
            --border-panel:   #3f3f46;   /* zinc-700 */
            --border-subtle:  #27272a;   /* zinc-800 */
            --text-primary:   #ffffff;
            --text-secondary: #d4d4d8;   /* zinc-300 */
            --text-muted:     #71717a;   /* zinc-500 */
            --text-faint:     #52525b;   /* zinc-600 */
            --scrollbar-track: #18181b;
            --scrollbar-thumb: #3f3f46;
        }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            -webkit-text-size-adjust: 100%;
            background: var(--bg-body);
            color: var(--text-primary);
        }}
        * {{ box-sizing: border-box; }}

        /* Panels — used by every _render_*_panel function */
        .t-panel {{
            background: var(--bg-panel);
            border-color: var(--border-panel);
        }}
        .t-panel-inner {{
            background: var(--bg-panel-inner);
        }}
        .t-input {{
            background: var(--bg-input);
        }}
        .t-bar {{
            background: var(--bg-bar);
        }}
        .t-border {{
            border-color: var(--border-panel);
        }}
        .t-border-subtle {{
            border-color: var(--border-subtle);
        }}
        .t-text {{
            color: var(--text-primary);
        }}
        .t-text-secondary {{
            color: var(--text-secondary);
        }}
        .t-text-muted {{
            color: var(--text-muted);
        }}
        .t-text-faint {{
            color: var(--text-faint);
        }}

        @media (max-width: 640px) {{
            .mobile-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
            .mobile-hide {{ display: none !important; }}
            .mobile-full {{ width: 100% !important; }}
        }}
        .glow-green {{ box-shadow: 0 0 12px rgba(34,197,94,0.25); }}
        .glow-red   {{ box-shadow: 0 0 12px rgba(239,68,68,0.25); }}
        .glow-blue  {{ box-shadow: 0 0 12px rgba(59,130,246,0.25); }}
        @keyframes sse-flash {{
            0%   {{ outline: 2px solid rgba(34,197,94,0.9); outline-offset:-2px; }}
            100% {{ outline: 2px solid transparent; outline-offset:-2px; }}
        }}
        .sse-updated {{ animation: sse-flash 1.2s ease-out; }}
        @keyframes breakout-pulse {{
            0%,100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }}
            50%      {{ box-shadow: 0 0 0 6px rgba(34,197,94,0); }}
        }}
        .breakout-active {{ animation: breakout-pulse 2s ease-in-out infinite; }}
        #sse-status-dot.connected    {{ color: #22c55e; }}
        #sse-status-dot.disconnected {{ color: #ef4444; }}
        #sse-status-dot.connecting   {{ color: #eab308; }}
        /* Validation match/mismatch row colours */
        .val-match    {{ background: rgba(34,197,94,0.06); }}
        .val-mismatch {{ background: rgba(239,68,68,0.06); }}
        /* Slim scrollbar */
        ::-webkit-scrollbar {{ width:4px; height:4px; }}
        ::-webkit-scrollbar-track {{ background: var(--scrollbar-track); }}
        ::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius:2px; }}
    </style>
</head>
<body class="min-h-screen transition-colors duration-200">
<div id="sse-container">
<div class="max-w-screen-2xl mx-auto px-2 sm:px-3 py-2 sm:py-3">

    <!-- ═══════════════════════════════════════════════════════════════
         HEADER: compact single row — logo | health bar | clock | tools
    ═══════════════════════════════════════════════════════════════════ -->
    <header class="mb-2 sm:mb-3 border-b t-border-subtle pb-2 sm:pb-2.5">
        <!-- Top row: title | clock -->
        <div class="flex items-center justify-between">
            <!-- Left: title + SSE dot -->
            <div class="flex items-center gap-2 min-w-0">
                <div>
                    <span class="text-base sm:text-lg font-bold t-text leading-none">ORB Co-Pilot</span>
                    <div class="text-[10px] t-text-faint mt-0.5">
                        <span class="hidden sm:inline">{session["date"]} · </span>
                        <span id="sse-status-dot" class="connecting" title="SSE">●</span>
                        <span id="sse-status-text" class="t-text-faint">connecting</span>
                    </div>
                </div>
            </div>

            <!-- Right: theme toggle + clock + session + NT8 tools -->
            <div class="flex items-center gap-1.5 sm:gap-2 shrink-0">
                <button id="theme-toggle"
                        onclick="toggleTheme()"
                        class="p-1 rounded-md t-input t-text-muted border t-border
                               hover:opacity-80 transition-colors text-sm leading-none"
                        title="Toggle dark/light theme">
                    <span id="theme-icon">☀️</span>
                </button>
                <div class="text-right">
                    <div id="clock" class="text-lg sm:text-xl font-mono font-bold {session["css_class"]} leading-none">
                        {session["time_et"]}
                    </div>
                    <div id="session-badge" class="text-[10px] sm:text-[11px] font-semibold {session["css_class"]} mt-0.5 text-right">
                        {session["emoji"]} {session["label"]}
                    </div>
                </div>
                <div id="nt8-toolbar-container"
                     hx-get="/api/nt8/panel/html"
                     hx-trigger="load"
                     hx-swap="innerHTML">
                </div>
            </div>
        </div>

        <!-- Bottom row (tablet+): health indicators -->
        <div id="nt8-health-bar"
             class="hidden md:flex items-center gap-1 flex-wrap mt-2 t-text-secondary"
             hx-get="/api/nt8/health/html"
             hx-trigger="load, every 10s"
             hx-swap="innerHTML">
            <!-- skeleton dots shown until HTMX loads real content -->
            <div class="flex items-center gap-1 px-2 py-0.5 rounded t-panel-inner border t-border">
                <span class="w-1.5 h-1.5 rounded-full t-text-faint"></span>
                <span class="text-[10px] t-text-faint">Loading...</span>
            </div>
        </div>
    </header>

    <!-- ═══════════════════════════════════════════════════════════════
         SESSION TIMELINE STRIP
    ═══════════════════════════════════════════════════════════════════ -->
    <div class="mobile-scroll">
        {session_strip_html}
    </div>

    <!-- No-trade banner -->
    <div id="no-trade-container"
         _="on `no-trade-alert` add .glow-red to me then wait 2s then remove .glow-red from me">
        {no_trade_html}
    </div>

    <!-- Focus summary bar -->
    <div id="focus-summary"
         class="flex flex-wrap items-center justify-between gap-y-1 t-panel border t-border rounded-lg px-3 py-1.5 mb-2 sm:mb-3">
        <div class="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
            <span class="t-text-muted uppercase tracking-wide font-semibold">Today's Focus</span>
            <span id="focus-count" class="t-text-secondary font-mono">{tradeable}/{total} tradeable</span>
            <span id="focus-updated" class="t-text-faint hidden sm:inline">Updated: {computed}</span>
        </div>
        <div class="flex items-center gap-2">
            <button hx-get="/api/focus/html"
                    hx-target="#focus-grid"
                    hx-swap="innerHTML"
                    hx-indicator="#refresh-spinner"
                    class="px-2 py-0.5 t-input hover:opacity-80 rounded text-[11px] t-text-muted
                           border t-border transition-colors">↻ Refresh</button>
            <span id="refresh-spinner" class="htmx-indicator t-text-faint text-xs">…</span>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════
         MAIN GRID  — 3 cols: ORB signals (2) | Sidebar (1)
    ═══════════════════════════════════════════════════════════════════ -->
    <div class="grid grid-cols-1 xl:grid-cols-3 gap-2 sm:gap-3">

        <!-- ── LEFT/CENTRE: ORB detection + asset cards ─────────────── -->
        <div class="xl:col-span-2 space-y-2 sm:space-y-3">

            <!-- ORB Panel — primary focus, full width, always visible -->
            <div id="orb-container" class="t-panel border t-border rounded-lg"
                 hx-get="/api/orb/html"
                 hx-trigger="every 20s"
                 hx-swap="innerHTML">
                {orb_html}
            </div>

            <!-- ORB Signal History — collapsible -->
            <details class="group">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide
                                flex items-center gap-1 py-1 select-none hover:opacity-80">
                    <span class="transition-transform group-open:rotate-90">▶</span>
                    ORB Signal History
                </summary>
                <div id="orb-history-container"
                     hx-get="/api/orb/history/html"
                     hx-trigger="revealed"
                     hx-swap="innerHTML">
                    <div class="t-panel border t-border rounded-lg p-4 text-center t-text-faint text-xs">
                        Loading history...
                    </div>
                </div>
            </details>

            <!-- NT8 / Ruby Validation Panel -->
            <div id="nt8-validation-panel"
                 class="t-panel border t-border rounded-lg p-3"
                 hx-get="/api/nt8/health/html?detail=1"
                 hx-trigger="load, every 15s"
                 hx-swap="outerHTML">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">
                        NinjaTrader / Ruby Validation
                    </h3>
                    <span class="text-[9px] t-text-faint">Compares engine metrics vs NT8 bridge</span>
                </div>
                <div class="text-xs t-text-faint">Loading NT8 data...</div>
            </div>

            <!-- Asset Focus Cards -->
            <div>
                <div class="flex items-center justify-between mb-1.5">
                    <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Asset Focus</h3>
                </div>
                <div id="focus-grid" class="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3"
                     hx-get="/api/focus/html"
                     hx-trigger="every 30s"
                     hx-swap="innerHTML">
                    {cards_html}
                </div>
            </div>
        </div>

        <!-- ── SIDEBAR ───────────────────────────────────────────────── -->
        <div class="space-y-2 sm:space-y-2.5">

            <!-- Positions & P&L -->
            <div id="positions-container"
                 hx-get="/api/positions/html"
                 hx-trigger="every 10s"
                 hx-swap="innerHTML">
                {positions_html}
            </div>

            <!-- Risk Rules — always visible -->
            <div id="risk-container"
                 hx-get="/api/risk/html"
                 hx-trigger="every 15s"
                 hx-swap="innerHTML">
                {risk_html}
            </div>

            <!-- Market Events feed -->
            {market_events_html}

            <!-- Grok AI Brief -->
            <div id="grok-container"
                 hx-get="/api/grok/html"
                 hx-trigger="every 60s"
                 hx-swap="innerHTML">
                {grok_html}
            </div>

            <!-- CNN Model — always visible -->
            <div id="cnn-panel"
                 class="t-panel border t-border rounded-lg p-3"
                 hx-get="/cnn/status/html"
                 hx-trigger="load, every 15s"
                 hx-swap="innerHTML">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">🧠 CNN Model</h3>
                <div class="t-text-faint text-xs">Loading...</div>
            </div>

            <!-- Alerts -->
            <div id="alerts-panel"
                 hx-get="/api/alerts/html"
                 hx-trigger="every 30s"
                 hx-swap="innerHTML">
                <div class="t-panel border t-border rounded-lg p-3">
                    <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">Alerts</h3>
                    <div class="t-text-faint text-xs">No alerts</div>
                </div>
            </div>

            <!-- Engine + SSE status — compact -->
            <div class="t-panel-inner border t-border-subtle rounded-lg p-2.5 space-y-1">
                <div class="flex items-center justify-between">
                    <span class="text-[10px] t-text-faint uppercase tracking-wide">Engine</span>
                    <div id="engine-status"
                         hx-get="/api/time"
                         hx-trigger="every 5s"
                         hx-swap="innerHTML"
                         class="text-[10px] t-text-muted">—</div>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-[10px] t-text-faint uppercase tracking-wide">Live Feed</span>
                    <div id="sse-heartbeat" class="text-[10px] t-text-muted">—</div>
                </div>
                <div id="sse-last-update" class="text-[9px] t-text-faint">—</div>
            </div>

        </div>
    </div>

    <!-- Footer -->
    <footer class="mt-4 pt-2 border-t t-border-subtle text-center text-[10px] t-text-faint px-2">
        <span class="hidden sm:inline">ORB Co-Pilot — Pre-market 00–03 ET | London 03–08 ET | US 08–12 ET | Off-hours 12–00 ET
        &nbsp;·&nbsp;</span>
        <a href="/sse/health" class="underline hover:opacity-80">SSE</a>
        &nbsp;·&nbsp;<a href="/api/info" class="underline hover:opacity-80">API</a>
    </footer>
</div>
</div><!-- end sse-container -->

<!-- ═══════════════════════════════════════════════════
     SESSION CURSOR: moves the timeline cursor every minute
     and renders open/closed session badges
═══════════════════════════════════════════════════════ -->
<script>
(function() {{
    var HOUR_PCT = 100.0 / 24.0;

    // Sessions: [label, startH(ET), endH(ET), openColor, closedColor]
    var SESSIONS = [
        ['Sydney',    17, 26, '#22d3ee', '#3f3f46'],
        ['Tokyo',     19, 28, '#818cf8', '#3f3f46'],
        ['London',     3, 12, '#60a5fa', '#3f3f46'],
        ['US Equity',  9, 16, '#34d399', '#3f3f46'],
        ['CME 23h',   18, 41, '#2dd4bf', '#3f3f46'],
    ];

    function _etHour() {{
        var now = new Date();
        var etStr = now.toLocaleString('en-US', {{timeZone:'America/New_York',hour:'numeric',minute:'numeric',hour12:false}});
        var parts = etStr.split(':');
        return parseInt(parts[0]) + parseInt(parts[1]||0)/60;
    }}

    function _isOpen(startH, endH, h) {{
        if (endH > 24) {{
            // wraps midnight
            return h >= startH || h < (endH - 24);
        }}
        return h >= startH && h < endH;
    }}

    function updateStrip() {{
        var h = _etHour();
        var cursor = document.getElementById('session-cursor');
        if (cursor) {{
            var pct = (h * HOUR_PCT);
            cursor.style.left = pct.toFixed(2) + '%';
        }}

        // Render badges
        var badgesEl = document.getElementById('session-badges');
        if (!badgesEl) return;
        var html = '';
        for (var i=0; i<SESSIONS.length; i++) {{
            var s = SESSIONS[i];
            var open = _isOpen(s[1], s[2], h);
            var color = open ? s[3] : '#52525b';
            var bgColor = open ? 'rgba(255,255,255,0.05)' : 'transparent';
            var border = open ? '1px solid ' + s[3] + '44' : '1px solid #3f3f46';
            html += '<span style="color:' + color + ';background:' + bgColor + ';border:' + border + ';font-size:9px;padding:1px 6px;border-radius:9999px;white-space:nowrap">';
            html += (open ? '● ' : '○ ') + s[0];
            html += '</span>';
        }}
        // Overlap badge
        if (_isOpen(9, 12, h)) {{
            html += '<span style="color:#fbbf24;background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);font-size:9px;padding:1px 6px;border-radius:9999px">⚡ London/US Overlap</span>';
        }}
        badgesEl.innerHTML = html;
    }}

    updateStrip();
    setInterval(updateStrip, 60000);
}})();
</script>

<!-- ═══════════════════════════════════════════════════
     LIVE CLOCK
═══════════════════════════════════════════════════════ -->
<script>
function updateClock() {{
    var now = new Date();
    var et = now.toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true}}) + ' ET';
    var el = document.getElementById('clock');
    if (el) el.textContent = et;
    var etHour = parseInt(now.toLocaleString('en-US',{{timeZone:'America/New_York',hour:'numeric',hour12:false}}));
    var badge = document.getElementById('session-badge');
    if (badge) {{
        if (etHour>=0&&etHour<3)  {{ badge.innerHTML='🌙 PRE-MARKET';  badge.className='text-[11px] font-semibold text-purple-400 mt-0.5 text-right'; if(el) el.className='text-xl font-mono font-bold text-purple-400 leading-none'; }}
        else if (etHour>=3&&etHour<8)  {{ badge.innerHTML='🟢 LONDON';  badge.className='text-[11px] font-semibold text-green-400 mt-0.5 text-right'; if(el) el.className='text-xl font-mono font-bold text-green-400 leading-none'; }}
        else if (etHour>=8&&etHour<12) {{ badge.innerHTML='🟢 US OPEN'; badge.className='text-[11px] font-semibold text-green-400 mt-0.5 text-right'; if(el) el.className='text-xl font-mono font-bold text-green-400 leading-none'; }}
        else                           {{ badge.innerHTML='⚙️ OFF-HRS';  badge.className='text-[11px] font-semibold text-zinc-400 mt-0.5 text-right'; if(el) el.className='text-xl font-mono font-bold text-zinc-400 leading-none'; }}
    }}
}}
setInterval(updateClock, 1000);
updateClock();
</script>

<!-- ═══════════════════════════════════════════════════
     MARKET EVENTS FEED — appended to by SSE handlers
═══════════════════════════════════════════════════════ -->
<script>
var _events = [];
function _pushEvent(emoji, msg, color) {{
    var now = new Date().toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}});
    _events.unshift({{t:now,e:emoji,m:msg,c:color||'text-zinc-400'}});
    if (_events.length>40) _events.pop();
    var feed = document.getElementById('market-events-feed');
    if (!feed) return;
    var ts = document.getElementById('events-ts');
    if (ts) ts.textContent = now;
    feed.innerHTML = _events.slice(0,12).map(function(ev){{
        return '<div class="flex items-start gap-1.5 py-0.5 border-b border-zinc-800/40 last:border-0">'
             + '<span class="text-[9px] text-zinc-600 font-mono shrink-0 mt-px">' + ev.t + '</span>'
             + '<span class="' + ev.c + ' text-[10px] leading-tight">' + ev.e + ' ' + ev.m + '</span>'
             + '</div>';
    }}).join('');
}}
</script>

<!-- ═══════════════════════════════════════════════════
     THEME TOGGLE
═══════════════════════════════════════════════════════ -->
<script>
function toggleTheme() {{
    var html = document.documentElement;
    var icon = document.getElementById('theme-icon');
    if (html.classList.contains('dark')) {{
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
        if (icon) icon.textContent = '🌙';
    }} else {{
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
        if (icon) icon.textContent = '☀️';
    }}
}}
// Set correct icon on load
(function() {{
    var icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = document.documentElement.classList.contains('dark') ? '☀️' : '🌙';
}})();
</script>

<!-- ═══════════════════════════════════════════════════
     SSE — native EventSource
═══════════════════════════════════════════════════════ -->
<script>
(function() {{
    var _es = null;
    var _reconnectMs = 3000;
    var _maxReconnect = 30000;
    var _curDelay = _reconnectMs;

    function _setStatus(state) {{
        var dot = document.getElementById('sse-status-dot');
        var txt = document.getElementById('sse-status-text');
        if (state==='connected')   {{ if(dot){{dot.className='connected ml-1.5';dot.title='live';}} if(txt){{txt.textContent='live';txt.className='text-zinc-600';}} }}
        else if(state==='connecting') {{ if(dot){{dot.className='connecting ml-1.5';}} if(txt){{txt.textContent='connecting';txt.className='text-zinc-700';}} }}
        else {{ if(dot){{dot.className='disconnected ml-1.5';}} if(txt){{txt.textContent='reconnecting';txt.className='text-zinc-700';}} }}
    }}

    function _connect() {{
        if (_es) {{ try{{_es.close();}}catch(e){{}} }}
        _setStatus('connecting');
        _es = new EventSource('/sse/dashboard');

        _es.onopen = function() {{ _setStatus('connected'); _curDelay=_reconnectMs; }};
        _es.onerror = function() {{
            _setStatus('disconnected');
            if (_es && _es.readyState===EventSource.CLOSED) {{
                setTimeout(function(){{ _curDelay=Math.min(_curDelay*1.5,_maxReconnect); _connect(); }}, _curDelay);
            }}
        }};

        _es.addEventListener('connected', function() {{ _setStatus('connected'); }});

        // --- Focus update ---
        _es.addEventListener('focus-update', function(e) {{
            try {{
                var focus = JSON.parse(e.data);
                var countEl = document.getElementById('focus-count');
                if (countEl) {{
                    countEl.textContent = (focus.tradeable_assets||0) + '/' + (focus.total_assets||0) + ' tradeable';
                }}
                var updEl = document.getElementById('focus-updated');
                if (updEl && focus.computed_at) {{
                    var d = new Date(focus.computed_at);
                    updEl.textContent = 'Updated: ' + d.toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',hour12:true}}) + ' ET';
                }}
                if (typeof htmx!=='undefined') {{
                    htmx.ajax('GET','/api/focus/html',{{target:'#focus-grid',swap:'innerHTML'}});
                }}
            }} catch(err) {{}}
        }});

        // --- Heartbeat ---
        _es.addEventListener('heartbeat', function(e) {{
            try {{
                var hb = JSON.parse(e.data);
                var hbEl = document.getElementById('sse-heartbeat');
                if (hbEl) hbEl.innerHTML = '<span style="color:#22c55e">●</span> ' + (hb.time_et||'');
            }} catch(err) {{}}
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/nt8/health/html',{{target:'#nt8-health-bar',swap:'innerHTML'}});
            }}
        }});

        // --- Session change ---
        _es.addEventListener('session-change', function(e) {{
            try {{
                var sc = JSON.parse(e.data);
                var badge = document.getElementById('session-badge');
                if (badge && sc.emoji && sc.session) badge.innerHTML = sc.emoji + ' ' + sc.session.replace('_','-').toUpperCase();
            }} catch(err) {{}}
        }});

        // --- No-trade ---
        _es.addEventListener('no-trade-alert', function(e) {{
            try {{
                var nt = JSON.parse(e.data);
                if (nt.no_trade && typeof htmx!=='undefined') {{
                    htmx.ajax('GET','/api/no-trade',{{target:'#no-trade-container',swap:'innerHTML'}});
                }}
            }} catch(err) {{}}
        }});

        // --- Positions update ---
        _es.addEventListener('positions-update', function() {{
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/positions/html',{{target:'#positions-container',swap:'innerHTML'}});
                htmx.ajax('GET','/api/nt8/health/html',{{target:'#nt8-health-bar',swap:'innerHTML'}});
            }}
            _pushEvent('📋','Position update','text-blue-400');
        }});

        // --- Grok update ---
        _es.addEventListener('grok-update', function() {{
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/grok/html',{{target:'#grok-container',swap:'innerHTML'}});
            }}
            var lastUpd = document.getElementById('sse-last-update');
            if (lastUpd) lastUpd.textContent = 'Grok: ' + new Date().toLocaleTimeString();
        }});

        // --- Risk update ---
        _es.addEventListener('risk-update', function() {{
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/risk/html',{{target:'#risk-container',swap:'innerHTML'}});
            }}
        }});

        // --- ORB update ---
        _es.addEventListener('orb-update', function(e) {{
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/orb/html',{{target:'#orb-container',swap:'innerHTML'}});
            }}
            try {{
                var orbData = JSON.parse(e.data);
                var dir = orbData.direction||'';
                var sym = orbData.symbol||'';
                if (orbData.breakout_detected) {{
                    var color = dir==='LONG' ? 'text-green-400' : 'text-red-400';
                    var emoji = dir==='LONG' ? '🟢' : '🔴';
                    _pushEvent(emoji, 'ORB breakout ' + dir + ' — ' + sym, color);
                    // pulse the ORB container
                    var orbEl = document.getElementById('orb-container');
                    if (orbEl) {{ orbEl.classList.add('breakout-active'); setTimeout(function(){{orbEl.classList.remove('breakout-active');}},6000); }}
                }} else {{
                    _pushEvent('📐','ORB update — ' + sym,'text-zinc-400');
                }}
            }} catch(err) {{}}
        }});

        // --- Per-asset listeners ---
        _es.onmessage = function(e) {{}};
    }}

    function _registerAssetListeners() {{
        document.querySelectorAll('[id^="asset-card-"]').forEach(function(card) {{
            var sym = card.id.replace('asset-card-','');
            if (sym && _es) {{
                _es.addEventListener(sym+'-update', function() {{
                    if (typeof htmx!=='undefined') {{
                        htmx.ajax('GET','/api/focus/'+encodeURIComponent(sym),{{target:card,swap:'outerHTML'}});
                    }}
                    setTimeout(function(){{
                        var u = document.getElementById('asset-card-'+sym);
                        if (u) {{ u.classList.add('sse-updated'); setTimeout(function(){{u.classList.remove('sse-updated');}},1500); }}
                    }},100);
                }});
            }}
        }});
    }}

    window.addEventListener('load', function() {{
        setTimeout(function() {{ _connect(); _registerAssetListeners(); }}, 100);
    }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
@router.get("/api/market-session/html", response_class=HTMLResponse)
def get_market_session_html():
    """Return the session strip HTML fragment for HTMX polling."""
    return HTMLResponse(_render_session_strip())


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


@router.get("/api/orb/history/html", response_class=HTMLResponse)
def get_orb_history_html(
    session: str | None = None,
    symbol: str | None = None,
    days: int = 7,
    breakout_only: bool = False,
):
    """Return per-session ORB signal history as an HTML table + summary.

    Query params:
        session      — Filter by session key (e.g. "london", "us")
        symbol       — Filter by symbol (e.g. "MGC=F")
        days         — Lookback window in calendar days (default 7)
        breakout_only — If true, only show events with a breakout detected
    """
    from datetime import timedelta

    since = (datetime.now(tz=_EST) - timedelta(days=days)).isoformat()

    try:
        from lib.core.models import get_orb_events as _get_orb_events

        events = _get_orb_events(
            limit=200,
            symbol=symbol,
            breakout_only=breakout_only,
            since=since,
        )
    except Exception:
        events = []

    # Optionally filter by session (stored in the 'session' column or metadata)
    if session:
        filtered = []
        for ev in events:
            ev_session = ev.get("session", "")
            # Also check metadata for session_key
            meta = {}
            if ev.get("metadata_json"):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    meta = json.loads(ev["metadata_json"])
            sk = meta.get("session_key", ev_session)
            if session.lower() in (ev_session.lower(), sk.lower()):
                filtered.append(ev)
        events = filtered

    # Summary stats
    total = len(events)
    breakouts = sum(1 for e in events if e.get("breakout_detected"))
    longs = sum(1 for e in events if e.get("direction") == "LONG")
    shorts = sum(1 for e in events if e.get("direction") == "SHORT")
    bo_rate = f"{breakouts / total * 100:.0f}%" if total > 0 else "—"

    # Filter tabs
    session_filter = session or "all"
    tab_classes_all = (
        "t-text font-bold border-b-2 border-blue-500" if session_filter == "all" else "t-text-muted hover:opacity-80"
    )
    tab_classes_lon = (
        "t-text font-bold border-b-2 border-blue-500" if session_filter == "london" else "t-text-muted hover:opacity-80"
    )
    tab_classes_us = (
        "t-text font-bold border-b-2 border-blue-500" if session_filter == "us" else "t-text-muted hover:opacity-80"
    )

    # Checkbox state
    bo_checked = "checked" if breakout_only else ""

    # Build table rows
    rows_html = ""
    for ev in events[:50]:  # Cap at 50 rows
        ts_raw = ev.get("timestamp", "")
        ts_display = ts_raw
        if ts_raw and "T" in ts_raw:
            try:
                dt = datetime.fromisoformat(ts_raw)
                ts_display = dt.strftime("%m/%d %H:%M")
            except Exception:
                pass

        sym = ev.get("symbol", "?")
        bd = bool(ev.get("breakout_detected"))
        direction = ev.get("direction", "")
        trigger = ev.get("trigger_price", 0)
        or_range = ev.get("or_range", 0)
        atr = ev.get("atr_value", 0)
        ev_session = ev.get("session", "")

        # Parse metadata for CNN/filter info
        meta = {}
        if ev.get("metadata_json"):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                meta = json.loads(ev["metadata_json"])

        cnn_prob = meta.get("cnn_prob")
        filter_passed = meta.get("filter_passed")
        sk = meta.get("session_key", ev_session)

        # Row styling
        if bd and direction == "LONG":
            row_bg = "background: rgba(34,197,94,0.08);"
            dir_html = '<span class="text-green-400 font-bold">🟢 LONG</span>'
        elif bd and direction == "SHORT":
            row_bg = "background: rgba(239,68,68,0.08);"
            dir_html = '<span class="text-red-400 font-bold">🔴 SHORT</span>'
        elif bd:
            row_bg = "background: rgba(234,179,8,0.08);"
            dir_html = f'<span class="text-yellow-400">{direction or "—"}</span>'
        else:
            row_bg = ""
            dir_html = '<span class="t-text-faint">—</span>'

        # CNN badge
        cnn_html = ""
        if cnn_prob is not None:
            pct = cnn_prob * 100
            c = "text-green-400" if pct >= 65 else ("text-yellow-400" if pct >= 45 else "text-red-400")
            cnn_html = f'<span class="{c} font-mono">{pct:.0f}%</span>'
        else:
            cnn_html = '<span class="t-text-faint">—</span>'

        # Filter badge
        if filter_passed is True:
            filt_html = '<span class="text-green-500">✅</span>'
        elif filter_passed is False:
            filt_html = '<span class="text-red-500">🚫</span>'
        else:
            filt_html = '<span class="t-text-faint">—</span>'

        # Session badge
        if "london" in sk.lower():
            sess_badge = '<span class="text-blue-400 text-[10px]">🇬🇧 LON</span>'
        elif "us" in sk.lower():
            sess_badge = '<span class="text-emerald-400 text-[10px]">🇺🇸 US</span>'
        else:
            sess_badge = f'<span class="t-text-faint text-[10px]">{sk[:6]}</span>'

        rows_html += f"""
        <tr class="border-b t-border-subtle text-[10px]" style="{row_bg}">
            <td class="py-1 px-1.5 t-text-muted font-mono whitespace-nowrap">{ts_display}</td>
            <td class="py-1 px-1.5">{sess_badge}</td>
            <td class="py-1 px-1.5 t-text-secondary font-mono">{sym}</td>
            <td class="py-1 px-1.5">{dir_html}</td>
            <td class="py-1 px-1.5 t-text-secondary font-mono text-right">{trigger:,.2f}</td>
            <td class="py-1 px-1.5 t-text-muted font-mono text-right">{or_range:,.2f}</td>
            <td class="py-1 px-1.5 t-text-muted font-mono text-right">{atr:,.2f}</td>
            <td class="py-1 px-1.5 text-center">{cnn_html}</td>
            <td class="py-1 px-1.5 text-center">{filt_html}</td>
        </tr>"""

    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="9" class="py-6 text-center t-text-faint text-xs">
                No ORB events found for the selected filters.
            </td>
        </tr>"""

    return HTMLResponse(
        content=f"""
    <div class="t-panel border t-border rounded-lg p-4">
        <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold t-text-muted">📊 ORB Signal History</h3>
            <span class="t-text-faint text-[10px]">Last {days} days · {total} events</span>
        </div>

        <!-- Summary stats -->
        <div class="grid grid-cols-4 gap-2 mb-3">
            <div class="t-panel-inner rounded p-2 text-center">
                <div class="text-lg font-bold t-text font-mono">{total}</div>
                <div class="text-[10px] t-text-muted">Total</div>
            </div>
            <div class="t-panel-inner rounded p-2 text-center">
                <div class="text-lg font-bold text-green-400 font-mono">{breakouts}</div>
                <div class="text-[10px] t-text-muted">Breakouts</div>
            </div>
            <div class="t-panel-inner rounded p-2 text-center">
                <div class="text-lg font-bold text-blue-400 font-mono">{bo_rate}</div>
                <div class="text-[10px] t-text-muted">BO Rate</div>
            </div>
            <div class="t-panel-inner rounded p-2 text-center">
                <div class="text-[11px] font-mono t-text-secondary">{longs}L / {shorts}S</div>
                <div class="text-[10px] t-text-muted">Direction</div>
            </div>
        </div>

        <!-- Session filter tabs -->
        <div class="flex items-center gap-3 mb-2 text-[11px] border-b t-border-subtle pb-1.5">
            <a hx-get="/api/orb/history/html?days={days}&breakout_only={"true" if breakout_only else "false"}"
               hx-target="#orb-history-container" hx-swap="innerHTML"
               class="cursor-pointer pb-0.5 {tab_classes_all}">All</a>
            <a hx-get="/api/orb/history/html?session=london&days={days}&breakout_only={"true" if breakout_only else "false"}"
               hx-target="#orb-history-container" hx-swap="innerHTML"
               class="cursor-pointer pb-0.5 {tab_classes_lon}">🇬🇧 London</a>
            <a hx-get="/api/orb/history/html?session=us&days={days}&breakout_only={"true" if breakout_only else "false"}"
               hx-target="#orb-history-container" hx-swap="innerHTML"
               class="cursor-pointer pb-0.5 {tab_classes_us}">🇺🇸 US</a>
            <label class="ml-auto flex items-center gap-1 cursor-pointer t-text-muted">
                <input type="checkbox" {bo_checked}
                       hx-get="/api/orb/history/html?{"session=" + session + "&" if session else ""}days={days}&breakout_only={{this.checked}}"
                       hx-target="#orb-history-container" hx-swap="innerHTML"
                       hx-trigger="change"
                       class="rounded">
                <span class="text-[10px]">Breakouts only</span>
            </label>
        </div>

        <!-- Table -->
        <div class="overflow-x-auto max-h-72 overflow-y-auto">
            <table class="w-full text-left">
                <thead>
                    <tr class="border-b t-border text-[9px] t-text-faint uppercase tracking-wider">
                        <th class="py-1 px-1.5">Time</th>
                        <th class="py-1 px-1.5">Session</th>
                        <th class="py-1 px-1.5">Symbol</th>
                        <th class="py-1 px-1.5">Signal</th>
                        <th class="py-1 px-1.5 text-right">Trigger</th>
                        <th class="py-1 px-1.5 text-right">Range</th>
                        <th class="py-1 px-1.5 text-right">ATR</th>
                        <th class="py-1 px-1.5 text-center">CNN</th>
                        <th class="py-1 px-1.5 text-center">Filter</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    )


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
