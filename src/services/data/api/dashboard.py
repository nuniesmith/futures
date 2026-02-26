"""
Dashboard API Router (TASK-301 / TASK-303)
============================================
Serves the HTMX dashboard and HTML fragment endpoints.

Endpoints:
    GET /                     — Full HTML dashboard page
    GET /api/focus            — All asset focus data as JSON
    GET /api/focus/html       — All asset cards as HTML fragments
    GET /api/focus/{symbol}   — Single asset card HTML fragment
    GET /api/positions/html   — Live positions panel HTML
    GET /api/alerts/html      — Alerts panel HTML
    GET /api/time             — Formatted time string with session indicator
    GET /api/no-trade         — No-trade banner HTML (if applicable)
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("api.dashboard")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_focus_data() -> Optional[dict[str, Any]]:
    """Read daily focus payload from Redis."""
    try:
        from cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.debug("Failed to read focus from Redis: %s", exc)
    return None


def _get_session_info() -> dict[str, str]:
    """Get current session mode and display info."""
    now = datetime.now(tz=_EST)
    hour = now.hour

    if 0 <= hour < 5:
        mode = "pre-market"
        emoji = "🌙"
        label = "PRE-MARKET"
        css_class = "text-purple-400"
    elif 5 <= hour < 12:
        mode = "active"
        emoji = "🟢"
        label = "ACTIVE"
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
        from cache import cache_get

        raw = cache_get("positions:current")
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("positions", [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# HTML rendering helpers (inline Jinja2-style templates)
# We render HTML directly to avoid template file dependencies during
# initial bootstrap. These can be migrated to Jinja2 files later.
# ---------------------------------------------------------------------------


def _render_asset_card(asset: dict) -> str:
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
    dominance = asset.get("dominance_text", "Neutral")
    market_phase = asset.get("market_phase", "UNKNOWN")
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


def _render_positions_panel(positions: list[dict[str, Any]]) -> str:
    """Render live positions panel as HTML fragment."""
    if not positions:
        return """
        <div id="positions-panel" class="bg-zinc-900/60 border border-zinc-700 rounded-lg p-4"
             hx-swap-oob="true">
            <h3 class="text-sm font-semibold text-zinc-400 mb-2">LIVE POSITIONS</h3>
            <div class="flex items-center gap-2 text-zinc-500">
                <span class="text-green-500">✓</span>
                <span>No open positions</span>
            </div>
        </div>
        """

    rows = ""
    total_pnl = 0.0
    for pos in positions:
        sym = pos.get("symbol", pos.get("instrument", "?"))
        side = pos.get("side", pos.get("direction", "?"))
        qty = pos.get("quantity", pos.get("contracts", 0))
        avg_price = pos.get("avg_price", pos.get("entry", 0))
        unrealized = pos.get("unrealized_pnl", pos.get("pnl", 0))
        total_pnl += unrealized

        side_color = (
            "text-green-400" if side.upper() in ("LONG", "BUY") else "text-red-400"
        )
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


def _render_full_dashboard(focus_data: Optional[dict], session: dict) -> str:
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
        no_trade_html = _render_no_trade_banner(
            focus_data.get("no_trade_reason", "Low-conviction day")
        )

    # Positions panel
    positions = _get_positions()
    positions_html = _render_positions_panel(positions)

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

    # Build per-asset SSE swap targets for granular live updates
    asset_sse_targets = ""
    if focus_data and focus_data.get("assets"):
        for asset in focus_data["assets"]:
            sym = asset.get("symbol", "").lower().replace(" ", "_").replace("&", "")
            if sym:
                asset_sse_targets += (
                    f'<div id="sse-asset-{sym}" sse-swap="{sym}-update" '
                    f'hx-swap="innerHTML" hx-target="#asset-card-{sym}" '
                    f'style="display:none;"></div>\n                    '
                )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Futures Trading Co-Pilot</title>

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- HTMX + SSE Extension + Idiomorph (smooth DOM merge) -->
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"></script>

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

    <!-- SSE Connection Wrapper — streams live events from engine via data-service -->
    <div id="sse-container"
         hx-ext="sse"
         sse-connect="/sse/dashboard"
         sse-close="close">

    <div class="max-w-7xl mx-auto px-4 py-4">
        <!-- Header -->
        <header class="flex items-center justify-between mb-6 border-b border-zinc-800 pb-4">
            <div>
                <h1 class="text-2xl font-bold">
                    <span class="text-zinc-400">Futures</span> Trading Co-Pilot
                </h1>
                <div class="text-sm text-zinc-500 mt-1">
                    {session["date"]}
                    <span id="sse-status-dot" class="connecting ml-2" title="SSE: connecting...">●</span>
                    <span id="sse-status-text" class="text-xs text-zinc-600">connecting</span>
                </div>
            </div>
            <div class="text-right">
                <div id="clock" class="text-2xl font-mono font-bold {session["css_class"]}">
                    {session["time_et"]}
                </div>
                <div id="session-badge" class="text-sm font-semibold mt-1 {session["css_class"]}">
                    {session["emoji"]} {session["label"]}
                </div>
            </div>
        </header>

        <!-- No Trade Banner — live-swapped via SSE no-trade-alert event -->
        <div id="no-trade-container"
             sse-swap="no-trade-alert"
             hx-swap="innerHTML"
             _="on no-trade-alert add .glow-red to me then wait 2s then remove .glow-red from me">
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
                <!-- Positions Panel — SSE live + HTMX polling fallback -->
                <div id="positions-panel"
                     hx-get="/api/positions/html"
                     hx-trigger="every 10s"
                     hx-swap="innerHTML">
                    {positions_html}
                </div>

                <!-- Alerts Panel -->
                <div id="alerts-panel"
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
            </div>
        </div>

        <!-- Footer -->
        <footer class="mt-8 pt-4 border-t border-zinc-800 text-center text-xs text-zinc-600">
            Futures Trading Co-Pilot v1.0 — Session rules: Pre-market 00–05 | Active 05–12 | Off-hours 12–00 ET
            | <a href="/sse/health" class="underline hover:text-zinc-400">SSE Health</a>
            | <a href="/api/info" class="underline hover:text-zinc-400">API Info</a>
        </footer>
    </div>

    <!-- Hidden SSE swap targets for per-asset events -->
    <div style="display:none;">
        {asset_sse_targets}
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
                if (etHour >= 0 && etHour < 5) {{
                    badge.innerHTML = '🌙 PRE-MARKET';
                    badge.className = 'text-sm font-semibold mt-1 text-purple-400';
                    if (el) el.className = 'text-2xl font-mono font-bold text-purple-400';
                }} else if (etHour >= 5 && etHour < 12) {{
                    badge.innerHTML = '🟢 ACTIVE';
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

    <!-- SSE Event Handlers — process JSON events and update DOM -->
    <script>
        // SSE connection status tracking
        const sseContainer = document.getElementById('sse-container');
        if (sseContainer) {{
            // HTMX fires these events for SSE connections
            sseContainer.addEventListener('htmx:sseOpen', function() {{
                const dot = document.getElementById('sse-status-dot');
                const txt = document.getElementById('sse-status-text');
                if (dot) {{ dot.className = 'connected ml-2'; dot.title = 'SSE: connected'; }}
                if (txt) {{ txt.textContent = 'live'; txt.className = 'text-xs text-green-600'; }}
            }});
            sseContainer.addEventListener('htmx:sseError', function() {{
                const dot = document.getElementById('sse-status-dot');
                const txt = document.getElementById('sse-status-text');
                if (dot) {{ dot.className = 'disconnected ml-2'; dot.title = 'SSE: disconnected'; }}
                if (txt) {{ txt.textContent = 'reconnecting...'; txt.className = 'text-xs text-red-600'; }}
            }});
            sseContainer.addEventListener('htmx:sseClose', function() {{
                const dot = document.getElementById('sse-status-dot');
                const txt = document.getElementById('sse-status-text');
                if (dot) {{ dot.className = 'disconnected ml-2'; dot.title = 'SSE: closed'; }}
                if (txt) {{ txt.textContent = 'disconnected'; txt.className = 'text-xs text-red-600'; }}
            }});
        }}

        // Listen for SSE events via the native EventSource (backup for custom processing)
        // HTMX handles the actual SSE connection; we use custom event listeners for
        // processing JSON payloads into DOM updates.
        document.body.addEventListener('htmx:sseMessage', function(evt) {{
            const eventName = evt.detail.type || '';
            const data = evt.detail.data || '';

            // --- Focus update: refresh summary bar + flash cards ---
            if (eventName === 'focus-update') {{
                try {{
                    const focus = JSON.parse(data);
                    // Update summary counts
                    const countEl = document.getElementById('focus-count');
                    if (countEl) {{
                        const tradeable = focus.tradeable_assets || 0;
                        const total = focus.total_assets || 0;
                        countEl.textContent = tradeable + '/' + total + ' tradeable';
                    }}
                    // Update timestamp
                    const updEl = document.getElementById('focus-updated');
                    if (updEl) {{
                        const ts = focus.computed_at || '';
                        if (ts) {{
                            const d = new Date(ts);
                            updEl.textContent = 'Updated: ' + d.toLocaleTimeString('en-US', {{
                                timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true
                            }}) + ' ET';
                        }}
                    }}
                    // Update last-update indicator
                    const lastUpd = document.getElementById('sse-last-update');
                    if (lastUpd) {{
                        lastUpd.textContent = 'Last focus: ' + new Date().toLocaleTimeString();
                    }}
                    // Trigger HTMX refresh of focus grid to pick up new HTML
                    htmx.trigger('#focus-grid', 'htmx:load');
                    htmx.ajax('GET', '/api/focus/html', {{target: '#focus-grid', swap: 'innerHTML'}});
                }} catch(e) {{ /* ignore parse errors */ }}
            }}

            // --- Heartbeat: update heartbeat display ---
            if (eventName === 'heartbeat') {{
                try {{
                    const hb = JSON.parse(data);
                    const hbEl = document.getElementById('sse-heartbeat');
                    if (hbEl) {{
                        hbEl.innerHTML = '<span class="text-green-500">●</span> Connected — ' + (hb.time_et || '');
                    }}
                }} catch(e) {{}}
            }}

            // --- Session change: update badge ---
            if (eventName === 'session-change') {{
                try {{
                    const sc = JSON.parse(data);
                    const badge = document.getElementById('session-badge');
                    if (badge && sc.emoji && sc.session) {{
                        const label = sc.session.replace('_', '-').toUpperCase();
                        badge.innerHTML = sc.emoji + ' ' + label;
                    }}
                }} catch(e) {{}}
            }}

            // --- No-trade alert: show banner ---
            if (eventName === 'no-trade-alert') {{
                try {{
                    const nt = JSON.parse(data);
                    if (nt.no_trade) {{
                        // Fetch rendered banner HTML from server
                        htmx.ajax('GET', '/api/no-trade', {{target: '#no-trade-container', swap: 'innerHTML'}});
                    }}
                }} catch(e) {{}}
            }}

            // --- Positions update: refresh panel ---
            if (eventName === 'positions-update') {{
                htmx.ajax('GET', '/api/positions/html', {{target: '#positions-panel', swap: 'innerHTML'}});
            }}

            // --- Per-asset updates: flash the specific card ---
            if (eventName.endsWith('-update') && eventName !== 'focus-update' && eventName !== 'positions-update') {{
                const symbol = eventName.replace('-update', '').replace(' ', '_').replace('&', '');
                const card = document.getElementById('asset-card-' + symbol);
                if (card) {{
                    // Fetch updated card HTML from server
                    htmx.ajax('GET', '/api/focus/' + encodeURIComponent(symbol), {{target: card, swap: 'outerHTML'}});
                    // Flash animation
                    setTimeout(function() {{
                        const updated = document.getElementById('asset-card-' + symbol);
                        if (updated) {{
                            updated.classList.add('sse-updated');
                            setTimeout(function() {{ updated.classList.remove('sse-updated'); }}, 1500);
                        }}
                    }}, 100);
                }}
            }}
        }});
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
def get_focus_html():
    """Return all asset cards as HTML fragments (for HTMX swap)."""
    focus_data = _get_focus_data()
    if not focus_data or not focus_data.get("assets"):
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
def get_focus_symbol(symbol: str):
    """Return a single asset card as HTML fragment."""
    focus_data = _get_focus_data()
    if not focus_data:
        return HTMLResponse(content="<div class='text-zinc-500'>No data</div>")

    # Find matching asset (case-insensitive)
    for asset in focus_data.get("assets", []):
        asset_symbol = asset.get("symbol", "").lower().replace(" ", "_")
        if asset_symbol == symbol.lower().replace(" ", "_"):
            return HTMLResponse(content=_render_asset_card(asset))

    return HTMLResponse(
        content=f"<div class='text-zinc-500'>Asset '{symbol}' not found</div>"
    )


@router.get("/api/positions/html", response_class=HTMLResponse)
def get_positions_html():
    """Return live positions panel as HTML fragment."""
    positions = _get_positions()
    return HTMLResponse(content=_render_positions_panel(positions))


@router.get("/api/alerts/html", response_class=HTMLResponse)
def get_alerts_html():
    """Return alerts panel as HTML fragment."""
    # Read alerts from Redis if available
    alerts = []
    try:
        from cache import cache_get

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
        ts = alert.get("timestamp", "")
        level = alert.get("level", "info")
        color = {
            "warning": "text-yellow-400",
            "error": "text-red-400",
            "success": "text-green-400",
        }.get(level, "text-zinc-400")

        rows += (
            f'<div class="{color} text-xs py-1 border-b border-zinc-800">{msg}</div>'
        )

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
        from cache import cache_get

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
            content=_render_no_trade_banner(
                focus_data.get("no_trade_reason", "Low-conviction day")
            )
        )
    return HTMLResponse(content='<div id="no-trade-banner"></div>')
