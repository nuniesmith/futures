"""
Dashboard API Router
=====================
ORB-centric futures trading co-pilot dashboard.

Layout:
  - Header bar: service health dots + market session clock strip
  - Session strip: live visual timeline of all major futures sessions
    with overlap highlighting and current-time cursor
  - Main (2/3): ORB detection cards per symbol, with CNN probability
    and filter results
  - Sidebar (1/3): live positions + P&L, risk status, CNN model,
    market events feed, Grok brief, alerts, engine status
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
    GET /api/regime/html        — HMM regime state panel HTML fragment
"""

import contextlib
import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Request
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
        color_hex = "#c084fc"
    elif 3 <= hour < 8:
        mode = "active"
        emoji = "🟢"
        label = "LONDON OPEN"
        css_class = "text-green-400"
        color_hex = "#4ade80"
    elif 8 <= hour < 12:
        mode = "active"
        emoji = "🟢"
        label = "US OPEN"
        css_class = "text-green-400"
        color_hex = "#4ade80"
    else:
        mode = "off-hours"
        emoji = "⚙️"
        label = "OFF-HOURS"
        css_class = "text-zinc-400"
        color_hex = "#a1a1aa"

    return {
        "mode": mode,
        "emoji": emoji,
        "label": label,
        "css_class": css_class,
        "color_hex": color_hex,
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


def _get_bridge_info() -> dict[str, Any]:
    """Read NT8 Bridge liveness from the heartbeat cache key.

    Returns a dict with: connected (bool), age_seconds (float), account (str).
    Safe to call even when Redis is unavailable — returns disconnected state.
    """
    try:
        from lib.core.cache import _cache_key, cache_get

        key = _cache_key("bridge_heartbeat", "current")
        raw = cache_get(key)
        if raw is None:
            return {"connected": False, "age_seconds": -1.0, "account": ""}
        hb = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        received = hb.get("received_at", "")
        age = -1.0
        connected = False
        if received:
            try:
                from zoneinfo import ZoneInfo

                dt = datetime.fromisoformat(received)
                age = (datetime.now(tz=ZoneInfo("America/New_York")) - dt).total_seconds()
                connected = age < 60.0
            except Exception:
                pass
        return {
            "connected": connected,
            "age_seconds": round(age, 1),
            "account": hb.get("account", ""),
        }
    except Exception:
        return {"connected": False, "age_seconds": -1.0, "account": ""}


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


# ---------------------------------------------------------------------------
# Volume Profile helpers
# ---------------------------------------------------------------------------


def _fetch_bars_for_vp(symbol: str, days_back: int = 5) -> "Any":
    """Fetch stored 1m bars for volume profile computation."""
    try:
        from lib.services.data.api.bars import _fetch_stored_bars

        return _fetch_stored_bars(symbol, interval="1m", days_back=days_back)
    except Exception:
        return None


def _render_vp_svg(
    bin_centers: "Any",
    bin_volumes: "Any",
    poc: float,
    vah: float,
    val: float,
    hvn: list[float],
    lvn: list[float],
    current_price: float = 0.0,
    naked_pocs: list[dict] | None = None,
    width: int = 260,
    height: int = 280,
) -> str:
    """Render a horizontal volume-profile histogram as an inline SVG.

    The chart is drawn with price on the Y-axis (highest at top) and
    volume on the X-axis.  Key levels (POC, VAH, VAL, naked POCs) are
    annotated with horizontal lines and labels.
    """
    import math as _math

    import numpy as np

    if bin_centers is None or len(bin_centers) == 0:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" fill="#71717a" font-size="11" '
            f'text-anchor="middle">No data</text></svg>'
        )

    centers = np.asarray(bin_centers, dtype=np.float64)
    volumes = np.asarray(bin_volumes, dtype=np.float64)
    max_vol = float(volumes.max()) if volumes.max() > 0 else 1.0

    n = len(centers)
    price_min = float(centers[0])
    price_max = float(centers[-1])
    price_range = price_max - price_min if price_max > price_min else 1.0

    # Layout constants
    PAD_L = 4
    PAD_R = 52  # right side for price labels
    PAD_T = 8
    PAD_B = 8
    bar_area_w = width - PAD_L - PAD_R
    bar_area_h = height - PAD_T - PAD_B
    bin_h = bar_area_h / max(n, 1)

    def price_to_y(price: float) -> float:
        """Map price to SVG y (high price = low y = top of chart)."""
        frac = (price - price_min) / price_range
        return PAD_T + bar_area_h - frac * bar_area_h

    def vol_to_w(vol: float) -> float:
        return max(1.0, vol / max_vol * bar_area_w)

    # Build bar elements
    bars_svg = []
    for i in range(n):
        bh = _math.ceil(bin_h) + 1  # +1 to avoid gaps
        by = PAD_T + bar_area_h - (i + 1) * bin_h
        bw = vol_to_w(float(volumes[i]))
        price = float(centers[i])

        # Color: POC=amber, HVN=blue, VA=green tint, LVN=dim, default=steel
        is_poc = abs(price - poc) < price_range / max(n, 1)
        in_va = val <= price <= vah
        is_hvn = any(abs(price - h) < price_range / max(n, 1) for h in hvn)
        is_lvn = any(abs(price - lv) < price_range / max(n, 1) for lv in lvn)

        if is_poc:
            fill = "#f59e0b"
            opacity = "0.90"
        elif in_va:
            fill = "#3b82f6" if is_hvn else "#1d4ed8"
            opacity = "0.70" if is_hvn else "0.45"
        elif is_lvn:
            fill = "#3f3f46"
            opacity = "0.50"
        else:
            fill = "#52525b"
            opacity = "0.65"

        bars_svg.append(
            f'<rect x="{PAD_L}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'fill="{fill}" fill-opacity="{opacity}"/>'
        )

    # Key-level horizontal lines + right-side labels
    def h_line(price: float, color: str, label: str, dash: str = "") -> str:
        y = price_to_y(price)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        label_x = PAD_L + bar_area_w + 2
        return (
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + bar_area_w}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="1.2"{dash_attr}/>'
            f'<text x="{label_x}" y="{y + 3.5:.1f}" fill="{color}" '
            f'font-size="8" font-family="monospace">{label}</text>'
        )

    lines_svg = []

    # VAH / VAL shading rectangle
    vah_y = price_to_y(vah)
    val_y = price_to_y(val)
    lines_svg.append(
        f'<rect x="{PAD_L}" y="{vah_y:.1f}" width="{bar_area_w}" '
        f'height="{val_y - vah_y:.1f}" fill="#1d4ed8" fill-opacity="0.08"/>'
    )

    lines_svg.append(h_line(poc, "#f59e0b", f"POC {poc:.1f}"))
    lines_svg.append(h_line(vah, "#3b82f6", f"VAH {vah:.1f}", "3 2"))
    lines_svg.append(h_line(val, "#3b82f6", f"VAL {val:.1f}", "3 2"))

    # Naked POC lines
    if naked_pocs:
        for np_entry in naked_pocs[:3]:
            np_price = float(np_entry.get("poc", 0))
            if price_min <= np_price <= price_max:
                np_date = str(np_entry.get("date", ""))[-5:]  # MM-DD
                lines_svg.append(h_line(np_price, "#c084fc", f"nPOC {np_date}", "4 3"))

    # Current price line
    if current_price and price_min <= current_price <= price_max:
        y_cp = price_to_y(current_price)
        lines_svg.append(
            f'<line x1="{PAD_L}" y1="{y_cp:.1f}" x2="{PAD_L + bar_area_w}" y2="{y_cp:.1f}" '
            f'stroke="#22c55e" stroke-width="1.5" stroke-dasharray="2 2"/>'
            f'<text x="{PAD_L + bar_area_w + 2}" y="{y_cp + 3.5:.1f}" fill="#22c55e" '
            f'font-size="8" font-family="monospace">{current_price:.1f}</text>'
        )

    svg_body = "\n".join(bars_svg) + "\n" + "\n".join(lines_svg)
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="overflow:visible">\n{svg_body}\n</svg>'
    )


def _render_volume_profile_panel(
    symbol: str,
    days_back: int = 5,
    bins: int = 40,
) -> str:
    """Render a volume profile chart panel for *symbol*.

    Shows the last *days_back* sessions aggregated into a single profile
    plus the most-recent session's profile side-by-side.  Naked POC
    levels from prior sessions are overlaid in purple.
    """
    try:
        from lib.analysis.volume_profile import (
            compute_session_profiles,
            compute_volume_profile,
            find_naked_pocs,
        )

        df = _fetch_bars_for_vp(symbol, days_back=days_back)
    except Exception as exc:
        return f'<div class="t-text-faint text-xs text-center py-4">VP unavailable: {exc}</div>'

    if df is None or df.empty:
        return '<div class="t-text-faint text-xs text-center py-4">No bar data for VP</div>'

    # Composite profile (all bars)
    composite = compute_volume_profile(df, n_bins=bins)

    # Session-by-session profiles (for naked POC tracking)
    session_profiles = compute_session_profiles(df, n_bins=bins, max_sessions=10)

    # Current price estimate = last close
    current_price = 0.0
    with contextlib.suppress(Exception):
        current_price = float(df["Close"].iloc[-1])

    # Naked POCs
    naked_pocs = find_naked_pocs(session_profiles, current_price=current_price, max_distance_points=200.0)

    # Composite profile SVG
    svg_composite = _render_vp_svg(
        bin_centers=composite.get("bin_centers"),
        bin_volumes=composite.get("bin_volumes"),
        poc=composite.get("poc", 0.0),
        vah=composite.get("vah", 0.0),
        val=composite.get("val", 0.0),
        hvn=composite.get("hvn", []),
        lvn=composite.get("lvn", []),
        current_price=current_price,
        naked_pocs=naked_pocs,
        width=220,
        height=260,
    )

    # Most-recent session profile
    svg_latest = ""
    latest_label = "—"
    if session_profiles:
        latest = session_profiles[-1]
        latest_label = str(latest.get("date", ""))
        svg_latest = _render_vp_svg(
            bin_centers=latest.get("bin_centers"),
            bin_volumes=latest.get("bin_volumes"),
            poc=latest.get("poc", 0.0),
            vah=latest.get("vah", 0.0),
            val=latest.get("val", 0.0),
            hvn=latest.get("hvn", []),
            lvn=latest.get("lvn", []),
            current_price=current_price,
            naked_pocs=[],
            width=220,
            height=260,
        )

    # Naked POC summary list
    naked_html = ""
    if naked_pocs:
        items = []
        for npoc in naked_pocs[:5]:
            dist = npoc.get("distance", 0.0)
            direction = "↑" if dist > 0 else "↓"
            color = "#c084fc"
            items.append(
                f'<div style="display:flex;justify-content:space-between;font-size:9px">'
                f'<span style="color:{color};font-family:monospace">{npoc["poc"]:.2f}</span>'
                f'<span style="color:#a1a1aa">{npoc.get("date", "")}</span>'
                f'<span style="color:{color}">{direction}{abs(dist):.1f}pts</span>'
                f"</div>"
            )
        naked_html = (
            '<div style="margin-top:6px;padding-top:6px;border-top:1px solid var(--border-subtle)">'
            '<div style="font-size:9px;color:var(--text-faint);margin-bottom:3px">🎯 Naked POCs</div>'
            + "".join(items)
            + "</div>"
        )
    else:
        naked_html = '<div style="font-size:9px;color:var(--text-faint);margin-top:4px">No naked POCs in range</div>'

    # VP stats summary pills
    poc_v = composite.get("poc", 0.0)
    vah_v = composite.get("vah", 0.0)
    val_v = composite.get("val", 0.0)
    stats_html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;margin-bottom:6px;text-align:center">
        <div style="background:rgba(245,158,11,0.1);border-radius:4px;padding:2px 4px">
            <div style="font-size:8px;color:#a1a1aa">POC</div>
            <div style="font-size:9px;color:#f59e0b;font-family:monospace">{poc_v:.2f}</div>
        </div>
        <div style="background:rgba(59,130,246,0.1);border-radius:4px;padding:2px 4px">
            <div style="font-size:8px;color:#a1a1aa">VAH</div>
            <div style="font-size:9px;color:#60a5fa;font-family:monospace">{vah_v:.2f}</div>
        </div>
        <div style="background:rgba(59,130,246,0.1);border-radius:4px;padding:2px 4px">
            <div style="font-size:8px;color:#a1a1aa">VAL</div>
            <div style="font-size:9px;color:#60a5fa;font-family:monospace">{val_v:.2f}</div>
        </div>
    </div>
    """

    # Symbol selector (common futures)
    _vp_symbols = ["MGC=F", "MES=F", "MNQ=F", "MCL=F", "M2K=F", "MYM=F", "MHG=F", "SIL=F"]
    sym_opts = "".join(f'<option value="{s}" {"selected" if s == symbol else ""}>{s}</option>' for s in _vp_symbols)
    # Add Kraken tickers if available
    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO

        if ENABLE_KRAKEN_CRYPTO:
            from lib.integrations.kraken_client import KRAKEN_PAIRS

            for _pair_info in list(KRAKEN_PAIRS.values())[:5]:
                _tk = _pair_info.get("internal_ticker", "")
                if _tk and _tk not in _vp_symbols:
                    sym_opts += f'<option value="{_tk}" {"selected" if _tk == symbol else ""}>{_tk}</option>'
    except Exception:
        pass

    return f"""
    <div id="vp-panel-inner">
        <!-- Header with symbol selector -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <div style="font-size:9px;color:var(--text-faint)">{days_back}d composite · {bins} bins</div>
            <div style="display:flex;align-items:center;gap:4px">
                <select id="vp-symbol-select" style="font-size:9px;background:var(--bg-input);color:var(--text-secondary);border:1px solid var(--border-panel);border-radius:3px;padding:1px 3px"
                    hx-get="/api/volume-profile/html"
                    hx-trigger="change"
                    hx-target="#vp-panel-inner"
                    hx-swap="outerHTML"
                    name="symbol">
                    {sym_opts}
                </select>
                <select style="font-size:9px;background:var(--bg-input);color:var(--text-secondary);border:1px solid var(--border-panel);border-radius:3px;padding:1px 3px"
                    hx-get="/api/volume-profile/html"
                    hx-trigger="change"
                    hx-target="#vp-panel-inner"
                    hx-swap="outerHTML"
                    name="days">
                    <option value="3" {"selected" if days_back == 3 else ""}>3d</option>
                    <option value="5" {"selected" if days_back == 5 else ""}>5d</option>
                    <option value="10" {"selected" if days_back == 10 else ""}>10d</option>
                </select>
            </div>
        </div>

        {stats_html}

        <!-- Charts: composite + latest session -->
        <div style="display:flex;gap:8px;overflow-x:auto">
            <div>
                <div style="font-size:8px;color:var(--text-faint);margin-bottom:2px;text-align:center">
                    Composite ({days_back}d)
                </div>
                {svg_composite}
            </div>
            {"<div><div style='font-size:8px;color:var(--text-faint);margin-bottom:2px;text-align:center'>" + latest_label + "</div>" + svg_latest + "</div>" if svg_latest else ""}
        </div>

        <!-- Legend -->
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">
            <span style="font-size:8px;color:#f59e0b">■ POC</span>
            <span style="font-size:8px;color:#3b82f6">■ Value Area</span>
            <span style="font-size:8px;color:#c084fc">— Naked POC</span>
            <span style="font-size:8px;color:#22c55e">— Current</span>
        </div>

        {naked_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Performance chart helpers
# ---------------------------------------------------------------------------


def _render_equity_curve_svg(
    dates: list[str],
    cumulative_pnl: list[float],
    width: int = 300,
    height: int = 100,
) -> str:
    """Render a compact equity curve line chart as inline SVG."""
    if not dates or not cumulative_pnl or len(cumulative_pnl) < 2:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" fill="#71717a" font-size="11" '
            f'text-anchor="middle">No data</text></svg>'
        )

    n = len(cumulative_pnl)
    PAD_L, PAD_R, PAD_T, PAD_B = 32, 8, 8, 18
    chart_w = width - PAD_L - PAD_R
    chart_h = height - PAD_T - PAD_B

    y_min = min(cumulative_pnl)
    y_max = max(cumulative_pnl)
    y_range = y_max - y_min if y_max != y_min else 1.0

    def px(i: int) -> float:
        return PAD_L + i / max(n - 1, 1) * chart_w

    def py(v: float) -> float:
        return PAD_T + chart_h - (v - y_min) / y_range * chart_h

    # Zero line
    zero_y = py(0.0)
    zero_line = ""
    if y_min < 0 < y_max:
        zero_line = (
            f'<line x1="{PAD_L}" y1="{zero_y:.1f}" x2="{PAD_L + chart_w}" y2="{zero_y:.1f}" '
            f'stroke="#52525b" stroke-width="0.8" stroke-dasharray="3 2"/>'
        )

    # Area fill (gradient from first point)
    points_for_area = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(cumulative_pnl))
    final_x = px(n - 1)
    base_y = py(0.0) if y_min < 0 < y_max else PAD_T + chart_h
    area_color = "#22c55e" if cumulative_pnl[-1] >= 0 else "#ef4444"
    area_poly = (
        f'<polygon points="{PAD_L},{base_y:.1f} {points_for_area} {final_x:.1f},{base_y:.1f}" '
        f'fill="{area_color}" fill-opacity="0.12"/>'
    )

    # Line path
    path_d = " ".join(("M" if i == 0 else "L") + f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(cumulative_pnl))
    line_color = "#22c55e" if cumulative_pnl[-1] >= 0 else "#ef4444"
    line_path = f'<path d="{path_d}" stroke="{line_color}" stroke-width="1.5" fill="none"/>'

    # Y-axis labels (min/max/current)
    def money(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:,.0f}"

    y_labels = (
        f'<text x="{PAD_L - 2}" y="{PAD_T + 8:.1f}" fill="#71717a" font-size="7" text-anchor="end">{money(y_max)}</text>'
        f'<text x="{PAD_L - 2}" y="{PAD_T + chart_h:.1f}" fill="#71717a" font-size="7" text-anchor="end">{money(y_min)}</text>'
        f'<text x="{PAD_L + chart_w}" y="{py(cumulative_pnl[-1]) - 2:.1f}" fill="{line_color}" font-size="7" text-anchor="end">{money(cumulative_pnl[-1])}</text>'
    )

    # X-axis date labels (first + last)
    x_labels = ""
    if dates:
        first_label = dates[0][-5:] if len(dates[0]) >= 5 else dates[0]
        last_label = dates[-1][-5:] if len(dates[-1]) >= 5 else dates[-1]
        x_labels = (
            f'<text x="{PAD_L}" y="{height - 2}" fill="#71717a" font-size="7" text-anchor="start">{first_label}</text>'
            f'<text x="{PAD_L + chart_w}" y="{height - 2}" fill="#71717a" font-size="7" text-anchor="end">{last_label}</text>'
        )

    svg_body = "\n".join([zero_line, area_poly, line_path, y_labels, x_labels])
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n{svg_body}\n</svg>'


def _render_winrate_bar_svg(
    dates: list[str],
    win_flags: list[bool],
    rolling_window: int = 10,
    width: int = 300,
    height: int = 60,
) -> str:
    """Render a rolling win-rate sparkline as inline SVG."""
    if not win_flags or len(win_flags) < 2:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" fill="#71717a" font-size="11" '
            f'text-anchor="middle">No data</text></svg>'
        )

    # Compute rolling win rate
    rolling: list[float] = []
    for i in range(len(win_flags)):
        start = max(0, i - rolling_window + 1)
        window = win_flags[start : i + 1]
        rolling.append(sum(window) / len(window) * 100.0)

    n = len(rolling)
    PAD_L, PAD_R, PAD_T, PAD_B = 26, 8, 4, 14
    chart_w = width - PAD_L - PAD_R
    chart_h = height - PAD_T - PAD_B

    def px(i: int) -> float:
        return PAD_L + i / max(n - 1, 1) * chart_w

    def py(v: float) -> float:
        return PAD_T + chart_h - v / 100.0 * chart_h

    # 50% reference line
    mid_y = py(50.0)
    mid_line = (
        f'<line x1="{PAD_L}" y1="{mid_y:.1f}" x2="{PAD_L + chart_w}" y2="{mid_y:.1f}" '
        f'stroke="#52525b" stroke-width="0.8" stroke-dasharray="3 2"/>'
        f'<text x="{PAD_L - 2}" y="{mid_y + 3:.1f}" fill="#52525b" font-size="7" text-anchor="end">50%</text>'
    )

    # Area
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(rolling))
    area_poly = (
        f'<polygon points="{PAD_L},{PAD_T + chart_h:.1f} {pts} {px(n - 1):.1f},{PAD_T + chart_h:.1f}" '
        f'fill="#3b82f6" fill-opacity="0.15"/>'
    )

    # Line
    path_d = " ".join(("M" if i == 0 else "L") + f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(rolling))
    cur_color = "#22c55e" if rolling[-1] >= 50 else "#f87171"
    line_path = f'<path d="{path_d}" stroke="{cur_color}" stroke-width="1.5" fill="none"/>'

    # Labels
    cur_wr = rolling[-1]
    y_labels = (
        f'<text x="{PAD_L - 2}" y="{PAD_T + 8}" fill="#71717a" font-size="7" text-anchor="end">100%</text>'
        f'<text x="{PAD_L - 2}" y="{PAD_T + chart_h}" fill="#71717a" font-size="7" text-anchor="end">0%</text>'
        f'<text x="{PAD_L + chart_w}" y="{py(cur_wr) - 2:.1f}" fill="{cur_color}" font-size="7" text-anchor="end">{cur_wr:.0f}%</text>'
    )
    x_labels = ""
    if dates:
        first_label = dates[0][-5:] if len(dates[0]) >= 5 else dates[0]
        last_label = dates[-1][-5:] if len(dates[-1]) >= 5 else dates[-1]
        x_labels = (
            f'<text x="{PAD_L}" y="{height - 2}" fill="#71717a" font-size="7" text-anchor="start">{first_label}</text>'
            f'<text x="{PAD_L + chart_w}" y="{height - 2}" fill="#71717a" font-size="7" text-anchor="end">{last_label}</text>'
        )

    svg_body = "\n".join([mid_line, area_poly, line_path, y_labels, x_labels])
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n{svg_body}\n</svg>'


def _render_monthly_bar_svg(
    labels: list[str],
    values: list[float],
    width: int = 300,
    height: int = 70,
) -> str:
    """Render a monthly P&L bar chart as inline SVG."""
    if not labels or not values:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" fill="#71717a" font-size="11" '
            f'text-anchor="middle">No data</text></svg>'
        )

    n = len(values)
    PAD_L, PAD_R, PAD_T, PAD_B = 32, 8, 8, 14
    chart_w = width - PAD_L - PAD_R
    chart_h = height - PAD_T - PAD_B

    y_max = max(abs(v) for v in values) or 1.0
    y_min_v = min(values)
    y_max_v = max(values)

    # Zero baseline
    zero_frac = abs(y_min_v) / (abs(y_min_v) + abs(y_max_v)) if y_min_v < 0 else 0.0
    zero_y = PAD_T + chart_h * (1.0 - zero_frac)

    bar_w = max(2.0, chart_w / max(n, 1) * 0.75)
    gap = chart_w / max(n, 1)

    bars_svg = []
    for i, v in enumerate(values):
        bar_x = PAD_L + i * gap + (gap - bar_w) / 2
        frac = abs(v) / max(y_max, 1.0)
        bar_h = max(1.0, frac * chart_h * (abs(y_min_v) + abs(y_max_v)) / (2 * y_max))
        if v >= 0:
            bar_y = zero_y - bar_h
            fill = "#22c55e"
        else:
            bar_y = zero_y
            fill = "#ef4444"
        bars_svg.append(
            f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{fill}" fill-opacity="0.75"/>'
        )
        # Month label
        lbl = labels[i][-7:] if len(labels[i]) > 7 else labels[i]
        bars_svg.append(
            f'<text x="{bar_x + bar_w / 2:.1f}" y="{height - 2}" fill="#71717a" '
            f'font-size="6" text-anchor="middle">{lbl}</text>'
        )

    zero_line = (
        f'<line x1="{PAD_L}" y1="{zero_y:.1f}" x2="{PAD_L + chart_w}" y2="{zero_y:.1f}" '
        f'stroke="#52525b" stroke-width="0.8"/>'
    )
    y_labels = (
        f'<text x="{PAD_L - 2}" y="{PAD_T + 8}" fill="#71717a" font-size="7" text-anchor="end">${y_max_v:,.0f}</text>'
    )
    if y_min_v < 0:
        y_labels += f'<text x="{PAD_L - 2}" y="{PAD_T + chart_h}" fill="#71717a" font-size="7" text-anchor="end">${y_min_v:,.0f}</text>'

    svg_body = "\n".join(bars_svg) + "\n" + zero_line + "\n" + y_labels
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n{svg_body}\n</svg>'


def _render_performance_panel(days_back: int = 90) -> str:
    """Render historical performance charts panel (equity curve, win rate, monthly bars).

    Data source: ``daily_journal`` table via ``get_daily_journal()``.
    """
    try:
        from lib.core.models import get_daily_journal, get_journal_stats
    except ImportError as exc:
        return f'<div class="t-text-faint text-xs text-center py-4">Stats unavailable: {exc}</div>'

    try:
        df = get_daily_journal(limit=days_back)
        stats = get_journal_stats()
    except Exception as exc:
        return f'<div class="t-text-faint text-xs text-center py-4">DB error: {exc}</div>'

    if df is None or (hasattr(df, "empty") and df.empty):
        return '<div class="t-text-faint text-xs text-center py-4">No journal data yet</div>'

    try:
        records = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)
    except Exception:
        records = []

    if not records:
        return '<div class="t-text-faint text-xs text-center py-4">No journal entries found</div>'

    # Sort ascending by date
    records = sorted(records, key=lambda r: str(r.get("trade_date", "")))

    dates = [str(r.get("trade_date", "")) for r in records]
    net_pnls = [float(r.get("net_pnl", 0.0)) for r in records]
    win_flags = [p > 0 for p in net_pnls]
    cumulative = []
    running = 0.0
    for p in net_pnls:
        running += p
        cumulative.append(running)

    # Monthly aggregates
    monthly: dict[str, float] = {}
    for r in records:
        d = str(r.get("trade_date", ""))
        month_key = d[:7] if len(d) >= 7 else d  # YYYY-MM
        monthly[month_key] = monthly.get(month_key, 0.0) + float(r.get("net_pnl", 0.0))

    month_labels = sorted(monthly.keys())
    month_vals = [monthly[m] for m in month_labels]

    # Render charts
    equity_svg = _render_equity_curve_svg(dates, cumulative, width=300, height=100)
    winrate_svg = _render_winrate_bar_svg(dates, win_flags, rolling_window=10, width=300, height=60)
    monthly_svg = _render_monthly_bar_svg(month_labels, month_vals, width=300, height=70)

    # Key stats
    total_days = stats.get("total_days", 0)
    win_rate = stats.get("win_rate", 0.0)
    total_net = stats.get("total_net", 0.0)
    best_day = stats.get("best_day", 0.0)
    worst_day = stats.get("worst_day", 0.0)
    avg_daily = stats.get("avg_daily_net", 0.0)
    streak = stats.get("current_streak", 0)
    net_color = "#22c55e" if total_net >= 0 else "#ef4444"
    wr_color = "#22c55e" if win_rate >= 50 else "#f87171"
    streak_color = "#22c55e" if streak > 0 else ("#ef4444" if streak < 0 else "#a1a1aa")
    streak_str = f"+{streak}W" if streak > 0 else (f"{streak}L" if streak < 0 else "—")

    days_opts = "".join(
        f'<option value="{d}" {"selected" if d == days_back else ""}>{d}d</option>' for d in [30, 60, 90, 180, 365]
    )

    return f"""
    <div id="perf-panel-inner">
        <!-- Controls -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <div style="font-size:9px;color:var(--text-faint)">{total_days} trading days recorded</div>
            <select style="font-size:9px;background:var(--bg-input);color:var(--text-secondary);border:1px solid var(--border-panel);border-radius:3px;padding:1px 3px"
                hx-get="/api/performance/html"
                hx-trigger="change"
                hx-target="#perf-panel-inner"
                hx-swap="outerHTML"
                name="days">
                {days_opts}
            </select>
        </div>

        <!-- Key metrics row -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:8px;text-align:center">
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Total Net</div>
                <div style="font-size:10px;font-family:monospace;color:{net_color};font-weight:700">
                    {"+" if total_net >= 0 else ""}${total_net:,.0f}
                </div>
            </div>
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Win Rate</div>
                <div style="font-size:10px;font-family:monospace;color:{wr_color};font-weight:700">{win_rate:.1f}%</div>
            </div>
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Streak</div>
                <div style="font-size:10px;font-family:monospace;color:{streak_color};font-weight:700">{streak_str}</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:8px;text-align:center">
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Avg/Day</div>
                <div style="font-size:10px;font-family:monospace;color:var(--text-secondary)">
                    {"+" if avg_daily >= 0 else ""}${avg_daily:,.0f}
                </div>
            </div>
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Best Day</div>
                <div style="font-size:10px;font-family:monospace;color:#22c55e">+${best_day:,.0f}</div>
            </div>
            <div style="background:var(--bg-inner);border-radius:4px;padding:3px 2px">
                <div style="font-size:8px;color:var(--text-faint)">Worst Day</div>
                <div style="font-size:10px;font-family:monospace;color:#f87171">${worst_day:,.0f}</div>
            </div>
        </div>

        <!-- Equity Curve -->
        <div style="margin-bottom:8px">
            <div style="font-size:8px;color:var(--text-faint);margin-bottom:2px">📈 Equity Curve (Cumulative Net P&amp;L)</div>
            {equity_svg}
        </div>

        <!-- Rolling Win Rate -->
        <div style="margin-bottom:8px">
            <div style="font-size:8px;color:var(--text-faint);margin-bottom:2px">🎯 Rolling Win Rate (10-day)</div>
            {winrate_svg}
        </div>

        <!-- Monthly P&L bars -->
        <div>
            <div style="font-size:8px;color:var(--text-faint);margin-bottom:2px">📅 Monthly Net P&amp;L</div>
            {monthly_svg}
        </div>
    </div>
    """


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
    target1_dollars = asset.get("target1_dollars", 0)
    target2_dollars = asset.get("target2_dollars", 0)
    price_decimals = asset.get("price_decimals", 4)
    trend_dir = asset.get("trend_direction", "NEUTRAL ↔️")
    _dominance = asset.get("dominance_text", "Neutral")  # noqa: F841
    _market_phase = asset.get("market_phase", "UNKNOWN")  # noqa: F841
    notes = asset.get("notes", "")
    skip = asset.get("skip", False)

    # Build a locale-aware price formatter that respects the tick precision
    def _fmt(v: float) -> str:
        if price_decimals <= 2:
            return f"{v:,.{price_decimals}f}"
        # For high-precision prices (forex), use fixed notation without thousands sep
        return f"{v:.{price_decimals}f}"

    # Dollar estimate badges — only show when > $0
    def _dollar_badge(amount: float, color: str = "green") -> str:
        if amount <= 0:
            return ""
        color_map = {
            "green": "rgba(34,197,94,0.15);color:#4ade80",
            "red": "rgba(239,68,68,0.15);color:#f87171",
        }
        style = color_map.get(color, color_map["green"])
        return (
            f'<span style="font-size:8px;padding:1px 3px;border-radius:3px;'
            f'background:{style};margin-left:2px">~${amount:,.0f}</span>'
        )

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
                <div class="t-text-secondary font-mono text-[10px] sm:text-xs">{_fmt(entry_low)}</div>
                <div class="t-text-muted font-mono text-[10px] sm:text-xs">– {_fmt(entry_high)}</div>
            </div>
            <div class="t-panel-inner rounded p-1 sm:p-1.5 text-center">
                <div class="t-text-muted text-[10px] sm:text-xs">Stop</div>
                <div class="text-red-400 font-mono text-[10px] sm:text-xs">{_fmt(stop)}</div>
                <div class="text-red-300 font-mono" style="font-size:8px">-${risk_dollars:,.0f}</div>
            </div>
            <div class="t-panel-inner rounded p-1 sm:p-1.5 text-center">
                <div class="t-text-muted text-[10px] sm:text-xs">TP1 / TP2</div>
                <div class="text-green-400 font-mono text-[10px] sm:text-xs">
                    {_fmt(tp1)}{_dollar_badge(target1_dollars)}
                </div>
                <div class="text-green-300 font-mono text-[10px] sm:text-xs">
                    {_fmt(tp2)}{_dollar_badge(target2_dollars)}
                </div>
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
    bridge_connected: bool = False,
    bridge_age_seconds: float = -1,
    bridge_account: str = "",
) -> str:
    """Render condensed live positions + daily P&L panel with Bridge status and action buttons."""
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

    # Bridge status dot + age
    if bridge_connected:
        age_str = f"{bridge_age_seconds:.0f}s ago" if bridge_age_seconds >= 0 else ""
        acct_str = f" · {bridge_account}" if bridge_account else ""
        bridge_dot_html = f'<span style="color:#22c55e;font-size:9px" title="Bridge connected{acct_str} · last heartbeat {age_str}">● BRIDGE</span>'
    else:
        bridge_dot_html = (
            '<span style="color:#71717a;font-size:9px" title="Bridge offline — no heartbeat">○ BRIDGE</span>'
        )

    # Action buttons (disabled when bridge is offline)
    btn_disabled = "" if bridge_connected else ' style="opacity:0.4;pointer-events:none" disabled'
    btn_title_flatten = "Flatten all positions via Bridge" if bridge_connected else "Bridge offline"
    btn_title_cancel = "Cancel all working orders via Bridge" if bridge_connected else "Bridge offline"
    action_buttons_html = f"""
    <div class="flex items-center gap-1 mt-1 mb-2">
        <button
            hx-post="/api/positions/flatten"
            hx-confirm="Flatten ALL positions immediately?"
            hx-target="#positions-panel"
            hx-swap="outerHTML"
            title="{btn_title_flatten}"
            class="co-btn"
            style="font-size:10px;padding:2px 8px;background:rgba(239,68,68,0.15);border-color:rgba(239,68,68,0.4);color:#f87171"
            {btn_disabled}>
            ⚡ Flatten All
        </button>
        <button
            hx-post="/api/positions/cancel_orders"
            hx-confirm="Cancel ALL working orders?"
            hx-target="#positions-panel"
            hx-swap="outerHTML"
            title="{btn_title_cancel}"
            class="co-btn"
            style="font-size:10px;padding:2px 8px;background:rgba(250,204,21,0.10);border-color:rgba(250,204,21,0.3);color:#facc15"
            {btn_disabled}>
            ✕ Cancel Orders
        </button>
    </div>
    """

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
            <div class="flex items-center justify-between mb-2">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Positions &amp; P&amp;L</h3>
                {bridge_dot_html}
            </div>
            {block_html}{stats_html}{action_buttons_html}
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
            <div class="flex items-center gap-2">
                {bridge_dot_html}
                <span class="{total_color} font-mono text-xs font-bold">Open: ${total_pnl:,.2f}</span>
            </div>
        </div>
        {block_html}{stats_html}{action_buttons_html}
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
    """Render condensed Grok AI brief panel with streaming controls.

    The panel has two zones:
    - **Cached brief** — the last completed update, rendered from Redis.
    - **Stream area** — hidden by default; shown while a Grok SSE stream
      is active so tokens appear progressively as they arrive.

    Two action buttons live in the header:
    - 📋 Brief  → opens /sse/grok/briefing  (morning briefing)
    - ⚡ Update → opens /sse/grok/update    (compact live update)

    The JS that drives the stream is emitted inline once and reused
    across re-renders via the ``_grokStreamInit`` guard flag.
    """
    if not grok_data:
        cached_html = '<div class="t-text-faint text-xs" id="grok-cached-text">Waiting for next update...</div>'
        time_et = ""
        update_type = ""
    else:
        text = grok_data.get("text", "")
        time_et = grok_data.get("time_et", "")
        update_type = grok_data.get("type", "")

        lines_html = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("DO NOW"):
                lines_html += (
                    f'<div class="text-yellow-300 font-bold text-[10px] mt-1 border-l-2 border-yellow-400 pl-1.5">'
                    f"{stripped}</div>"
                )
            else:
                css = "t-text-muted"
                if "🟢" in stripped:
                    css = "text-green-300"
                elif "🔴" in stripped:
                    css = "text-red-300"
                lines_html += f'<div class="{css} font-mono text-[10px]">{stripped}</div>'

        type_badge = ""
        if update_type == "briefing":
            type_badge = '<span style="font-size:9px;padding:1px 5px;border-radius:9999px;background:rgba(96,165,250,0.15);color:#60a5fa;margin-left:4px">briefing</span>'
        elif update_type == "live_update":
            type_badge = '<span style="font-size:9px;padding:1px 5px;border-radius:9999px;background:rgba(52,211,153,0.15);color:#34d399;margin-left:4px">live</span>'

        cached_html = f"""
        <div id="grok-cached-text" style="display:flex;align-items:flex-start;gap:4px">
            {type_badge}
            <div class="space-y-0.5 w-full">{lines_html}</div>
        </div>
        """

    return f"""
    <div id="grok-panel" class="t-panel border t-border rounded-lg p-3"
         style="border-left:3px solid rgba(251,191,36,0.4)"
         hx-swap-oob="true">

        <!-- Header row: title + time + stream buttons -->
        <div class="flex items-center justify-between mb-1.5">
            <div class="flex items-center gap-1.5">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🤖 AI Analyst</h3>
                <span id="grok-stream-status" style="font-size:9px;display:none;color:#fbbf24">● streaming</span>
            </div>
            <div class="flex items-center gap-1">
                <span class="text-[9px] t-text-faint" id="grok-time-label">{time_et}</span>
                <button onclick="grokStream('briefing')"
                        title="Stream morning briefing"
                        style="font-size:9px;padding:1px 6px;border-radius:4px;background:rgba(96,165,250,0.15);border:1px solid rgba(96,165,250,0.3);color:#93c5fd;cursor:pointer"
                        id="grok-btn-brief">📋 Brief</button>
                <button onclick="grokStream('update')"
                        title="Stream live update"
                        style="font-size:9px;padding:1px 6px;border-radius:4px;background:rgba(52,211,153,0.15);border:1px solid rgba(52,211,153,0.3);color:#6ee7b7;cursor:pointer"
                        id="grok-btn-update">⚡ Update</button>
            </div>
        </div>

        <!-- Cached text area (shown when not streaming) -->
        <div id="grok-cached-area" class="max-h-36 overflow-y-auto">
            {cached_html}
        </div>

        <!-- Streaming area (hidden until a stream starts) -->
        <div id="grok-stream-area"
             style="display:none;max-height:14rem;overflow-y:auto;border-top:1px solid var(--border-subtle);margin-top:6px;padding-top:6px">
            <div id="grok-stream-text"
                 style="font-size:10px;color:var(--text-secondary);font-family:monospace;white-space:pre-wrap;line-height:1.55"></div>
            <div id="grok-stream-cursor"
                 style="display:inline-block;width:7px;height:12px;background:#fbbf24;vertical-align:text-bottom;animation:pulse 0.8s steps(1) infinite"></div>
        </div>

        <!-- Error toast -->
        <div id="grok-stream-error"
             style="display:none;margin-top:4px;padding:3px 8px;border-radius:4px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);font-size:9px;color:#fca5a5"></div>
    </div>

    <script>
    (function() {{
        // Guard: only initialise once even if the panel is re-rendered by HTMX
        if (window._grokStreamInit) return;
        window._grokStreamInit = true;

        var _grokEs = null;

        window.grokStream = function(type) {{
            // Close any existing stream first
            if (_grokEs) {{ try {{ _grokEs.close(); }} catch(e) {{}} _grokEs = null; }}

            var url = type === 'briefing' ? '/sse/grok/briefing' : '/sse/grok/update?update_number=1';
            var streamArea  = document.getElementById('grok-stream-area');
            var streamText  = document.getElementById('grok-stream-text');
            var streamCursor = document.getElementById('grok-stream-cursor');
            var cachedArea  = document.getElementById('grok-cached-area');
            var statusDot   = document.getElementById('grok-stream-status');
            var errDiv      = document.getElementById('grok-stream-error');
            var timeLabel   = document.getElementById('grok-time-label');
            var btnBrief    = document.getElementById('grok-btn-brief');
            var btnUpdate   = document.getElementById('grok-btn-update');

            if (!streamArea || !streamText) return;

            // Reset UI
            streamText.textContent = '';
            if (streamCursor) streamCursor.style.display = 'inline-block';
            streamArea.style.display = 'block';
            if (cachedArea) cachedArea.style.display = 'none';
            if (statusDot) statusDot.style.display = 'inline';
            if (errDiv) errDiv.style.display = 'none';
            if (btnBrief)  btnBrief.disabled  = true;
            if (btnUpdate) btnUpdate.disabled = true;

            _grokEs = new EventSource(url);
            var accumulated = '';

            _grokEs.addEventListener('grok-start', function(e) {{
                if (timeLabel) timeLabel.textContent = 'streaming…';
            }});

            _grokEs.addEventListener('grok-token', function(e) {{
                accumulated += e.data;
                if (streamText) {{
                    streamText.textContent = accumulated;
                    // Auto-scroll to bottom
                    var parent = streamArea;
                    if (parent) parent.scrollTop = parent.scrollHeight;
                }}
            }});

            _grokEs.addEventListener('grok-heartbeat', function(e) {{
                // keep-alive — no visible action needed
            }});

            _grokEs.addEventListener('grok-done', function(e) {{
                _grokEs.close(); _grokEs = null;
                if (streamCursor) streamCursor.style.display = 'none';
                if (statusDot) statusDot.style.display = 'none';
                if (btnBrief)  btnBrief.disabled  = false;
                if (btnUpdate) btnUpdate.disabled = false;

                // Parse completion meta
                try {{
                    var meta = JSON.parse(e.data);
                    var now = new Date().toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',hour12:false}});
                    if (timeLabel) timeLabel.textContent = now + ' ET';
                }} catch(ex) {{}}

                // After a short delay, swap the streamed text back into the
                // cached area so it persists after the next HTMX re-render
                setTimeout(function() {{
                    if (cachedArea) {{
                        var badge = type === 'briefing'
                            ? '<span style="font-size:9px;padding:1px 5px;border-radius:9999px;background:rgba(96,165,250,0.15);color:#60a5fa;margin-right:4px">briefing</span>'
                            : '<span style="font-size:9px;padding:1px 5px;border-radius:9999px;background:rgba(52,211,153,0.15);color:#34d399;margin-right:4px">live</span>';
                        cachedArea.innerHTML = '<div id="grok-cached-text" style="display:flex;align-items:flex-start;gap:4px">'
                            + badge
                            + '<div style="font-size:10px;color:var(--text-secondary);font-family:monospace;white-space:pre-wrap;line-height:1.55">'
                            + accumulated.replace(/</g,'&lt;').replace(/>/g,'&gt;')
                            + '</div></div>';
                        cachedArea.style.display = 'block';
                        streamArea.style.display = 'none';
                    }}
                }}, 800);
            }});

            _grokEs.addEventListener('grok-error', function(e) {{
                _grokEs.close(); _grokEs = null;
                if (streamCursor) streamCursor.style.display = 'none';
                if (statusDot) statusDot.style.display = 'none';
                if (btnBrief)  btnBrief.disabled  = false;
                if (btnUpdate) btnUpdate.disabled = false;
                if (cachedArea) cachedArea.style.display = 'block';
                streamArea.style.display = 'none';
                try {{
                    var errData = JSON.parse(e.data);
                    if (errDiv) {{
                        errDiv.textContent = '⚠ ' + (errData.error || 'Unknown error');
                        errDiv.style.display = 'block';
                    }}
                }} catch(ex) {{}}
            }});

            _grokEs.onerror = function() {{
                if (_grokEs && _grokEs.readyState === EventSource.CLOSED) {{
                    _grokEs = null;
                    if (streamCursor) streamCursor.style.display = 'none';
                    if (statusDot) statusDot.style.display = 'none';
                    if (btnBrief)  btnBrief.disabled  = false;
                    if (btnUpdate) btnUpdate.disabled = false;
                    // Only hide stream area if we got no content
                    if (!accumulated) {{
                        if (cachedArea) cachedArea.style.display = 'block';
                        streamArea.style.display = 'none';
                        if (errDiv) {{
                            errDiv.textContent = '⚠ Connection lost — check API key / network';
                            errDiv.style.display = 'block';
                        }}
                    }}
                }}
            }};
        }};
    }})();
    </script>
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
            <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Market Events</h3>
            <span class="text-[9px] t-text-faint" id="events-ts">—</span>
        </div>
        <div id="market-events-feed" class="space-y-1 max-h-36 overflow-y-auto text-[10px]">
            <div class="t-text-faint">Listening for ORB signals, fills, and alerts...</div>
        </div>
    </div>
    """


def _render_full_dashboard(focus_data: dict[str, Any] | None, session: dict[str, str]) -> str:
    """Render the complete dashboard HTML page.

    Self-contained CSS — no Tailwind CDN dependency.  All utility classes
    are defined in the embedded <style> block so the dashboard renders
    correctly even when there is no internet access on the host.
    """
    # Asset cards grid
    cards_html = ""
    if focus_data and focus_data.get("assets"):
        for asset in focus_data["assets"]:
            cards_html += _render_asset_card(asset)
    else:
        cards_html = """
        <div class="col-span-2" style="text-align:center;padding:3rem 0">
            <div style="font-size:2.5rem;margin-bottom:1rem">📊</div>
            <div class="t-text-muted" style="font-size:1.1rem">Waiting for engine to compute daily focus...</div>
            <div class="t-text-faint" style="font-size:0.85rem;margin-top:0.5rem">Data will appear automatically when ready.</div>
        </div>
        """

    # No-trade banner
    no_trade_html = ""
    if focus_data and focus_data.get("no_trade"):
        no_trade_html = _render_no_trade_banner(str(focus_data.get("no_trade_reason", "Low-conviction day")))

    # Positions panel with risk status + Bridge liveness
    positions = _get_positions()
    risk_status = _get_risk_status()
    _bridge_info = _get_bridge_info()
    positions_html = _render_positions_panel(
        positions,
        risk_status=risk_status,
        bridge_connected=_bridge_info["connected"],
        bridge_age_seconds=_bridge_info["age_seconds"],
        bridge_account=_bridge_info["account"],
    )

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
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Futures Trading Co-Pilot</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
    <!-- Apply saved theme AND dashboard mode BEFORE paint to prevent flash -->
    <script>
        (function() {{
            var t = localStorage.getItem('theme');
            if (t === 'light') document.documentElement.classList.remove('dark');
            else document.documentElement.classList.add('dark');
        }})();
        // Apply dashboard mode before body renders to avoid review-only panel flash
        (function() {{
            var saved = localStorage.getItem('dashMode');
            if (!saved) {{
                // Auto-detect from ET hour
                try {{
                    var etStr = new Date().toLocaleString('en-US', {{
                        timeZone: 'America/New_York', hour: 'numeric', hour12: false
                    }});
                    var h = parseInt(etStr, 10);
                    saved = (h >= 3 && h < 16) ? 'trading' : 'review';
                }} catch(e) {{
                    saved = 'trading';
                }}
            }}
            // Patch body class before it is painted — body tag already has mode-trading
            // as a fallback; override if saved preference differs
            document.addEventListener('DOMContentLoaded', function() {{
                document.body.classList.remove('mode-trading', 'mode-review');
                document.body.classList.add(saved === 'review' ? 'mode-review' : 'mode-trading');
            }});
        }})();
    </script>
    <!-- HTMX for live fragment swaps -->
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/hyperscript.org@0.9.14"></script>
    <style>
        /* ═══════════════════════════════════════════════════════════
           SELF-CONTAINED CSS — Futures Trading Co-Pilot
           No Tailwind CDN needed.  All utility classes are defined
           below so the dashboard works on air-gapped hosts.
        ═══════════════════════════════════════════════════════════ */

        /* ── Reset ────────────────────────────────────────────── */
        *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
        html {{ -webkit-text-size-adjust: 100%; }}

        /* ── Theme variables ──────────────────────────────────── */
        :root {{
            --bg-body:        #f4f4f5;
            --bg-panel:       rgba(255,255,255,0.80);
            --bg-panel-inner: rgba(244,244,245,0.60);
            --bg-input:       #e4e4e7;
            --bg-bar:         #d4d4d8;
            --border-panel:   #d4d4d8;
            --border-subtle:  #e4e4e7;
            --text-primary:   #18181b;
            --text-secondary: #3f3f46;
            --text-muted:     #71717a;
            --text-faint:     #a1a1aa;
            --scrollbar-track: #f4f4f5;
            --scrollbar-thumb: #a1a1aa;
        }}
        .dark {{
            --bg-body:        #09090b;
            --bg-panel:       rgba(24,24,27,0.60);
            --bg-panel-inner: rgba(39,39,42,0.40);
            --bg-input:       #27272a;
            --bg-bar:         #3f3f46;
            --border-panel:   #3f3f46;
            --border-subtle:  #27272a;
            --text-primary:   #ffffff;
            --text-secondary: #d4d4d8;
            --text-muted:     #71717a;
            --text-faint:     #52525b;
            --scrollbar-track: #18181b;
            --scrollbar-thumb: #3f3f46;
        }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-body);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}

        /* ── Theme utility classes (used by panel renderers) ─── */
        .t-panel        {{ background: var(--bg-panel); border-color: var(--border-panel); }}
        .t-panel-inner  {{ background: var(--bg-panel-inner); }}
        .t-input        {{ background: var(--bg-input); }}
        .t-bar          {{ background: var(--bg-bar); }}
        .t-border       {{ border-color: var(--border-panel); }}
        .t-border-subtle {{ border-color: var(--border-subtle); }}
        .t-text         {{ color: var(--text-primary); }}
        .t-text-secondary {{ color: var(--text-secondary); }}
        .t-text-muted   {{ color: var(--text-muted); }}
        .t-text-faint   {{ color: var(--text-faint); }}

        /* ── Layout utilities ─────────────────────────────────── */
        .flex {{ display: flex; }}
        .inline-flex {{ display: inline-flex; }}
        .grid {{ display: grid; }}
        .block {{ display: block; }}
        .inline-block {{ display: inline-block; }}
        .hidden {{ display: none; }}
        .relative {{ position: relative; }}
        .absolute {{ position: absolute; }}
        .items-center {{ align-items: center; }}
        .items-start {{ align-items: flex-start; }}
        .justify-between {{ justify-content: space-between; }}
        .justify-center {{ justify-content: center; }}
        .flex-wrap {{ flex-wrap: wrap; }}
        .shrink-0 {{ flex-shrink: 0; }}
        .min-w-0 {{ min-width: 0; }}
        .w-full {{ width: 100%; }}
        .overflow-hidden {{ overflow: hidden; }}
        .overflow-x-auto {{ overflow-x: auto; }}
        .overflow-y-auto {{ overflow-y: auto; }}
        .truncate {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .whitespace-nowrap {{ white-space: nowrap; }}
        .select-none {{ user-select: none; }}
        .pointer-events-none {{ pointer-events: none; }}
        .cursor-pointer {{ cursor: pointer; }}

        /* ── Gaps ──────────────────────────────────────────────── */
        .gap-0\\.5 {{ gap: 0.125rem; }}
        .gap-1    {{ gap: 0.25rem; }}
        .gap-1\\.5 {{ gap: 0.375rem; }}
        .gap-2    {{ gap: 0.5rem; }}
        .gap-3    {{ gap: 0.75rem; }}
        .gap-4    {{ gap: 1rem; }}
        .gap-x-2  {{ column-gap: 0.5rem; }}
        .gap-x-3  {{ column-gap: 0.75rem; }}
        .gap-y-0\\.5 {{ row-gap: 0.125rem; }}
        .gap-y-1  {{ row-gap: 0.25rem; }}

        /* ── Spacing ───────────────────────────────────────────── */
        .p-1    {{ padding: 0.25rem; }}
        .p-1\\.5 {{ padding: 0.375rem; }}
        .p-2    {{ padding: 0.5rem; }}
        .p-2\\.5 {{ padding: 0.625rem; }}
        .p-3    {{ padding: 0.75rem; }}
        .p-4    {{ padding: 1rem; }}
        .px-1   {{ padding-left: 0.25rem; padding-right: 0.25rem; }}
        .px-1\\.5 {{ padding-left: 0.375rem; padding-right: 0.375rem; }}
        .px-2   {{ padding-left: 0.5rem; padding-right: 0.5rem; }}
        .px-3   {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
        .px-4   {{ padding-left: 1rem; padding-right: 1rem; }}
        .py-0\\.5 {{ padding-top: 0.125rem; padding-bottom: 0.125rem; }}
        .py-1   {{ padding-top: 0.25rem; padding-bottom: 0.25rem; }}
        .py-1\\.5 {{ padding-top: 0.375rem; padding-bottom: 0.375rem; }}
        .py-2   {{ padding-top: 0.5rem; padding-bottom: 0.5rem; }}
        .py-3   {{ padding-top: 0.75rem; padding-bottom: 0.75rem; }}
        .py-6   {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}
        .py-12  {{ padding-top: 3rem; padding-bottom: 3rem; }}
        .pb-0\\.5 {{ padding-bottom: 0.125rem; }}
        .pb-1\\.5 {{ padding-bottom: 0.375rem; }}
        .pb-2   {{ padding-bottom: 0.5rem; }}
        .pt-2   {{ padding-top: 0.5rem; }}
        .pt-3   {{ padding-top: 0.75rem; }}
        .mb-0\\.5 {{ margin-bottom: 0.125rem; }}
        .mb-1   {{ margin-bottom: 0.25rem; }}
        .mb-1\\.5 {{ margin-bottom: 0.375rem; }}
        .mb-2   {{ margin-bottom: 0.5rem; }}
        .mb-3   {{ margin-bottom: 0.75rem; }}
        .mb-4   {{ margin-bottom: 1rem; }}
        .mt-0\\.5 {{ margin-top: 0.125rem; }}
        .mt-1   {{ margin-top: 0.25rem; }}
        .mt-2   {{ margin-top: 0.5rem; }}
        .mt-3   {{ margin-top: 0.75rem; }}
        .mt-4   {{ margin-top: 1rem; }}
        .ml-1   {{ margin-left: 0.25rem; }}
        .ml-1\\.5 {{ margin-left: 0.375rem; }}
        .ml-2   {{ margin-left: 0.5rem; }}
        .ml-auto {{ margin-left: auto; }}
        .mr-1   {{ margin-right: 0.25rem; }}
        .-mx-1  {{ margin-left: -0.25rem; margin-right: -0.25rem; }}

        /* ── Sizing ────────────────────────────────────────────── */
        .h-1    {{ height: 0.25rem; }}
        .h-2    {{ height: 0.5rem; }}
        .h-6    {{ height: 1.5rem; }}
        .w-0\\.5 {{ width: 0.125rem; }}
        .w-1    {{ width: 0.25rem; }}
        .w-1\\.5 {{ width: 0.375rem; }}
        .w-2    {{ width: 0.5rem; }}
        .max-h-28 {{ max-height: 7rem; }}
        .max-h-36 {{ max-height: 9rem; }}
        .max-h-72 {{ max-height: 18rem; }}
        .min-w-\\[200px\\] {{ min-width: 200px; }}
        .min-w-\\[320px\\] {{ min-width: 320px; }}

        /* ── Typography ────────────────────────────────────────── */
        .text-\\[9px\\]  {{ font-size: 9px; line-height: 1.2; }}
        .text-\\[10px\\] {{ font-size: 10px; line-height: 1.3; }}
        .text-\\[11px\\] {{ font-size: 11px; line-height: 1.3; }}
        .text-xs       {{ font-size: 0.75rem; line-height: 1rem; }}
        .text-sm       {{ font-size: 0.875rem; line-height: 1.25rem; }}
        .text-base     {{ font-size: 1rem; line-height: 1.5rem; }}
        .text-lg       {{ font-size: 1.125rem; line-height: 1.75rem; }}
        .text-xl       {{ font-size: 1.25rem; line-height: 1.75rem; }}
        .text-2xl      {{ font-size: 1.5rem; line-height: 2rem; }}
        .text-3xl      {{ font-size: 1.875rem; line-height: 2.25rem; }}
        .text-4xl      {{ font-size: 2.25rem; line-height: 2.5rem; }}
        .font-mono     {{ font-family: 'SF Mono', 'Cascadia Code', 'Consolas', 'Liberation Mono', monospace; }}
        .font-bold     {{ font-weight: 700; }}
        .font-semibold {{ font-weight: 600; }}
        .italic        {{ font-style: italic; }}
        .uppercase     {{ text-transform: uppercase; }}
        .tracking-wide {{ letter-spacing: 0.025em; }}
        .tracking-wider {{ letter-spacing: 0.05em; }}
        .leading-none  {{ line-height: 1; }}
        .leading-tight {{ line-height: 1.25; }}
        .text-center   {{ text-align: center; }}
        .text-left     {{ text-align: left; }}
        .text-right    {{ text-align: right; }}
        .underline     {{ text-decoration: underline; }}
        .no-underline  {{ text-decoration: none; }}

        /* ── Colors (static — used in panel renderers) ─────── */
        .text-white     {{ color: #ffffff; }}
        .text-green-300 {{ color: #86efac; }}
        .text-green-400 {{ color: #4ade80; }}
        .text-green-500 {{ color: #22c55e; }}
        .text-red-300   {{ color: #fca5a5; }}
        .text-red-400   {{ color: #f87171; }}
        .text-red-500   {{ color: #ef4444; }}
        .text-yellow-300 {{ color: #fde047; }}
        .text-yellow-400 {{ color: #facc15; }}
        .text-blue-400  {{ color: #60a5fa; }}
        .text-purple-400 {{ color: #c084fc; }}
        .text-emerald-400 {{ color: #34d399; }}
        .text-zinc-200  {{ color: #e4e4e7; }}
        .text-zinc-400  {{ color: #a1a1aa; }}
        .text-zinc-500  {{ color: #71717a; }}
        .text-zinc-600  {{ color: #52525b; }}
        .text-zinc-700  {{ color: #3f3f46; }}

        .bg-green-500   {{ background-color: #22c55e; }}
        .bg-green-900\\/40 {{ background: rgba(20,83,45,0.4); }}
        .bg-red-500     {{ background-color: #ef4444; }}
        .bg-red-900\\/40 {{ background: rgba(127,29,29,0.4); }}
        .bg-red-900\\/60 {{ background: rgba(127,29,29,0.6); }}
        .bg-yellow-500  {{ background-color: #eab308; }}
        .bg-yellow-400\\/10 {{ background: rgba(250,204,21,0.1); }}
        .bg-zinc-800\\/40 {{ background: rgba(39,39,42,0.4); }}
        .bg-zinc-800\\/60 {{ background: rgba(39,39,42,0.6); }}
        .bg-zinc-900\\/60 {{ background: rgba(24,24,27,0.6); }}
        .bg-zinc-900\\/80 {{ background: rgba(24,24,27,0.8); }}
        .bg-white\\/5     {{ background: rgba(255,255,255,0.05); }}
        .bg-white\\/60    {{ background: rgba(255,255,255,0.60); }}
        .bg-white\\/80    {{ background: rgba(255,255,255,0.80); }}
        .bg-white         {{ background: #ffffff; }}
        .bg-blue-700      {{ background-color: #1d4ed8; }}
        .bg-emerald-700   {{ background-color: #047857; }}
        .bg-slate-600     {{ background-color: #475569; }}
        .bg-indigo-700    {{ background-color: #4338ca; }}
        .bg-teal-800      {{ background-color: #115e59; }}

        .text-white\\/60   {{ color: rgba(255,255,255,0.6); }}
        .text-slate-300   {{ color: #cbd5e1; }}
        .text-indigo-200  {{ color: #c7d2fe; }}
        .text-blue-200    {{ color: #bfdbfe; }}
        .text-emerald-200 {{ color: #a7f3d0; }}
        .text-teal-300    {{ color: #5eead4; }}

        .border-green-500 {{ border-color: #22c55e; }}
        .border-green-600\\/40 {{ border-color: rgba(22,163,74,0.4); }}
        .border-green-600\\/50 {{ border-color: rgba(22,163,74,0.5); }}
        .border-red-500   {{ border-color: #ef4444; }}
        .border-red-600\\/40 {{ border-color: rgba(220,38,38,0.4); }}
        .border-red-600\\/50 {{ border-color: rgba(220,38,38,0.5); }}
        .border-red-700   {{ border-color: #b91c1c; }}
        .border-yellow-400\\/30 {{ border-color: rgba(250,204,21,0.3); }}
        .border-yellow-400\\/40 {{ border-color: rgba(250,204,21,0.4); }}
        .border-zinc-600  {{ border-color: #52525b; }}
        .border-zinc-700  {{ border-color: #3f3f46; }}
        .border-zinc-700\\/40 {{ border-color: rgba(63,63,70,0.4); }}
        .border-zinc-700\\/50 {{ border-color: rgba(63,63,70,0.5); }}
        .border-zinc-800  {{ border-color: #27272a; }}
        .border-zinc-800\\/40 {{ border-color: rgba(39,39,42,0.4); }}
        .border-blue-400  {{ border-color: #60a5fa; }}
        .border-blue-500  {{ border-color: #3b82f6; }}
        .border-emerald-400 {{ border-color: #34d399; }}

        /* ── Border / Rounded ──────────────────────────────────── */
        .border    {{ border-width: 1px; border-style: solid; }}
        .border-b  {{ border-bottom-width: 1px; border-bottom-style: solid; }}
        .border-t  {{ border-top-width: 1px; border-top-style: solid; }}
        .border-l  {{ border-left-width: 1px; border-left-style: solid; }}
        .border-r  {{ border-right-width: 1px; border-right-style: solid; }}
        .border-l-2 {{ border-left-width: 2px; border-left-style: solid; }}
        .border-r-2 {{ border-right-width: 2px; border-right-style: solid; }}
        .border-b-2 {{ border-bottom-width: 2px; border-bottom-style: solid; }}
        .rounded    {{ border-radius: 0.25rem; }}
        .rounded-sm {{ border-radius: 0.125rem; }}
        .rounded-md {{ border-radius: 0.375rem; }}
        .rounded-lg {{ border-radius: 0.5rem; }}
        .rounded-full {{ border-radius: 9999px; }}

        /* ── Grid ──────────────────────────────────────────────── */
        .grid-cols-1 {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
        .grid-cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .grid-cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        .grid-cols-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
        .col-span-2 {{ grid-column: span 2 / span 2; }}

        /* ── Transitions / Animations ──────────────────────────── */
        .transition-all     {{ transition: all 0.15s ease; }}
        .transition-colors  {{ transition: color 0.15s ease, background-color 0.15s ease, border-color 0.15s ease; }}
        .transition-transform {{ transition: transform 0.15s ease; }}
        .duration-200 {{ transition-duration: 200ms; }}
        .duration-500 {{ transition-duration: 500ms; }}
        @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.5; }} }}
        .animate-pulse {{ animation: pulse 2s cubic-bezier(0.4,0,0.6,1) infinite; }}
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

        /* ── Positioning ───────────────────────────────────────── */
        .inset-0  {{ top:0; right:0; bottom:0; left:0; }}
        .top-0    {{ top: 0; }}
        .bottom-0 {{ bottom: 0; }}
        .left-0   {{ left: 0; }}
        .right-0  {{ right: 0; }}
        .-top-1   {{ top: -0.25rem; }}
        .top-0\\.5 {{ top: 2px; }}
        .left-0\\.5 {{ left: 2px; }}
        .left-1\\/2 {{ left: 50%; }}
        .-translate-x-1\\/2 {{ transform: translateX(-50%); }}

        /* ── Effects ───────────────────────────────────────────── */
        .glow-green {{ box-shadow: 0 0 12px rgba(34,197,94,0.25); }}
        .glow-red   {{ box-shadow: 0 0 12px rgba(239,68,68,0.25); }}
        .glow-blue  {{ box-shadow: 0 0 12px rgba(59,130,246,0.25); }}
        .opacity-50 {{ opacity: 0.5; }}
        .hover\\:opacity-80:hover {{ opacity: 0.8; }}
        .z-10 {{ z-index: 10; }}
        .self-center {{ align-self: center; }}

        /* ── Status dots ───────────────────────────────────────── */
        #sse-status-dot.connected    {{ color: #22c55e; }}
        #sse-status-dot.disconnected {{ color: #ef4444; }}
        #sse-status-dot.connecting   {{ color: #eab308; }}
        .val-match    {{ background: rgba(34,197,94,0.06); }}
        .val-mismatch {{ background: rgba(239,68,68,0.06); }}

        /* ── Trading / Review mode visibility ─────────────────── */
        /* Default: Trading Mode (active session focus) */
        body.mode-review  .trading-only {{ display: none !important; }}
        body.mode-trading .review-only  {{ display: none !important; }}

        /* Mode toggle button states */
        .mode-btn {{ cursor: pointer; border-radius: 9999px; padding: 2px 10px; font-size: 11px; font-weight: 600; border: 1px solid transparent; transition: all 0.15s ease; }}
        .mode-btn.active-trading {{ background: rgba(34,197,94,0.15); border-color: rgba(34,197,94,0.4); color: #4ade80; }}
        .mode-btn.active-review  {{ background: rgba(96,165,250,0.15); border-color: rgba(96,165,250,0.4); color: #60a5fa; }}
        .mode-btn:not(.active-trading):not(.active-review) {{ background: var(--bg-input); border-color: var(--border-panel); color: var(--text-muted); }}

        /* ── Table ─────────────────────────────────────────────── */
        table {{ border-collapse: collapse; }}
        td, th {{ border-color: inherit; }}

        /* ── HTMX indicator ────────────────────────────────────── */
        .htmx-indicator {{ display: none; }}
        .htmx-request .htmx-indicator, .htmx-request.htmx-indicator {{ display: inline; }}

        /* ── Scrollbar ─────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width:4px; height:4px; }}
        ::-webkit-scrollbar-track {{ background: var(--scrollbar-track); }}
        ::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius:2px; }}

        /* ── Space-y utility (vertical spacing between children) ─ */
        .space-y-0\\.5 > * + * {{ margin-top: 0.125rem; }}
        .space-y-1 > * + * {{ margin-top: 0.25rem; }}
        .space-y-2 > * + * {{ margin-top: 0.5rem; }}
        .space-y-2\\.5 > * + * {{ margin-top: 0.625rem; }}
        .space-y-3 > * + * {{ margin-top: 0.75rem; }}

        /* ── Details/summary ───────────────────────────────────── */
        details > summary {{ list-style: none; }}
        details > summary::-webkit-details-marker {{ display: none; }}
        details[open] > summary .rotate-on-open {{ transform: rotate(90deg); }}

        /* ── Health dot bar (Co-Pilot header) ──────────────────── */
        .health-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
        .health-dot.green {{ background: #22c55e; }}
        .health-dot.red   {{ background: #ef4444; }}
        .health-dot.gray  {{ background: #52525b; }}
        .health-dot.yellow {{ background: #eab308; }}
        .health-label {{ font-size: 11px; margin-right: 12px; vertical-align: middle; }}

        /* ── Co-Pilot clock ─────────────────────────────────────── */
        .copilot-clock {{
            font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}

        /* ── Buttons ───────────────────────────────────────────── */
        .co-btn {{
            background: var(--bg-input);
            border: 1px solid var(--border-panel);
            border-radius: 0.375rem;
            padding: 4px 10px;
            font-size: 11px;
            color: var(--text-secondary);
            cursor: pointer;
        }}
        .co-btn:hover {{ opacity: 0.8; }}

        /* ── Container ─────────────────────────────────────────── */
        .container {{
            max-width: 1536px;
            margin: 0 auto;
            padding: 0.5rem 0.75rem;
        }}

        /* ── Responsive grid ───────────────────────────────────── */
        @media (min-width: 640px) {{
            .sm\\:grid-cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .sm\\:grid-cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .sm\\:grid-cols-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
            .sm\\:inline {{ display: inline; }}
            .sm\\:flex {{ display: flex; }}
            .sm\\:block {{ display: block; }}
            .sm\\:table-cell {{ display: table-cell; }}
            .hidden.sm\\:flex {{ display: flex !important; }}
            .hidden.sm\\:inline {{ display: inline !important; }}
            .sm\\:px-3 {{ padding-left: 0.75rem; padding-right: 0.75rem; }}
            .sm\\:px-4 {{ padding-left: 1rem; padding-right: 1rem; }}
            .sm\\:py-3 {{ padding-top: 0.75rem; padding-bottom: 0.75rem; }}
            .sm\\:p-4 {{ padding: 1rem; }}
            .sm\\:p-1\\.5 {{ padding: 0.375rem; }}
            .sm\\:text-xs {{ font-size: 0.75rem; }}
            .sm\\:text-lg {{ font-size: 1.125rem; }}
            .sm\\:text-xl {{ font-size: 1.25rem; }}
            .sm\\:text-2xl {{ font-size: 1.5rem; }}
            .sm\\:text-\\[11px\\] {{ font-size: 11px; }}
            .sm\\:gap-1\\.5 {{ gap: 0.375rem; }}
            .sm\\:gap-2 {{ gap: 0.5rem; }}
            .sm\\:gap-3 {{ gap: 0.75rem; }}
            .sm\\:mb-3 {{ margin-bottom: 0.75rem; }}
            .sm\\:mb-4 {{ margin-bottom: 1rem; }}
            .sm\\:pb-2\\.5 {{ padding-bottom: 0.625rem; }}
            .sm\\:pb-6 {{ padding-bottom: 1.5rem; }}
            .sm\\:mt-3 {{ margin-top: 0.75rem; }}
            .sm\\:space-y-2\\.5 > * + * {{ margin-top: 0.625rem; }}
            .sm\\:space-y-3 > * + * {{ margin-top: 0.75rem; }}
            .sm\\:flex {{ display: flex; }}
        }}
        @media (min-width: 768px) {{
            .md\\:flex {{ display: flex; }}
        }}
        @media (min-width: 1280px) {{
            .xl\\:grid-cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .xl\\:col-span-2 {{ grid-column: span 2 / span 2; }}
        }}
        @media (max-width: 639px) {{
            .hidden.sm\\:inline {{ display: none !important; }}
            .hidden.sm\\:flex {{ display: none !important; }}
            .hidden.sm\\:table-cell {{ display: none !important; }}
            .mobile-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
            .mobile-hide {{ display: none !important; }}
        }}

        /* ── Link styling ──────────────────────────────────────── */
        a {{ color: inherit; }}

        /* ── Top nav bar ───────────────────────────────────────── */
        .co-nav {{
            display: flex;
            align-items: center;
            gap: 0;
            padding: 0 1rem;
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border-subtle);
            height: 42px;
            position: sticky;
            top: 0;
            z-index: 200;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            margin-bottom: 0.75rem;
        }}
        .co-nav-brand {{
            font-weight: 700;
            font-size: 0.9rem;
            color: var(--text);
            text-decoration: none;
            margin-right: 1.25rem;
            letter-spacing: -0.02em;
            white-space: nowrap;
        }}
        .co-nav-tab {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 12px;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 500;
            transition: background .12s, color .12s;
            white-space: nowrap;
        }}
        .co-nav-tab:hover {{
            background: var(--bg-input);
            color: var(--text);
        }}
        .co-nav-tab.active {{
            background: var(--bg-input);
            color: var(--text);
            font-weight: 650;
        }}
        .co-nav-right {{
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
    </style>
</head>
<body class="mode-trading">

<!-- ═══════════════════════════════════════════════════════════════
     TOP NAV BAR
═══════════════════════════════════════════════════════════════════ -->
<nav class="co-nav">
    <a class="co-nav-brand" href="/">📈 Co-Pilot</a>
    <a class="co-nav-tab active" href="/">📊 Dashboard</a>
    <a class="co-nav-tab" href="/orb-history">📅 ORB History</a>
    <a class="co-nav-tab" href="/trainer">🧠 Trainer</a>
    <a class="co-nav-tab" href="/journal/page">📓 Journal</a>
    <a class="co-nav-tab" href="/settings">⚙️ Settings</a>
    <div class="co-nav-right">
        <span id="nav-sse-dot" style="font-size:10px;color:#52525b" title="SSE connection">●</span>
    </div>
</nav>

<div id="sse-container">
<div class="container">

    <!-- ═══════════════════════════════════════════════════════════════
         HEADER — Futures Trading Co-Pilot
         Left: Title + date + live badge
         Centre: Health dot bar
         Right: Clock + session badge + theme toggle
    ═══════════════════════════════════════════════════════════════════ -->
    <header style="margin-bottom:0.75rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border-subtle)">
        <div class="flex items-center justify-between" style="flex-wrap:wrap;gap:0.5rem">
            <!-- LEFT: Title + date -->
            <div class="min-w-0">
                <div style="font-size:1.35rem;font-weight:700;line-height:1.2">
                    <span style="color:#a78bfa">Futures</span> <span class="t-text">Trading Co-Pilot</span>
                </div>
                <div style="font-size:11px;margin-top:2px" class="t-text-muted">
                    {session["date"]}
                    <span style="margin-left:6px">
                        <span id="sse-status-dot" class="connecting" title="SSE" style="font-size:10px">●</span>
                        <span id="sse-status-text" class="t-text-faint" style="font-size:10px">connecting</span>
                    </span>
                </div>
            </div>

            <!-- CENTRE: Health indicators (loaded via HTMX) -->
            <div id="health-bar"
                 class="flex items-center"
                 style="gap:3px;flex-wrap:wrap"
                 hx-get="/api/nt8/health/html"
                 hx-trigger="load, every 10s"
                 hx-swap="innerHTML">
                <span class="health-dot gray"></span><span class="health-label t-text-faint">Loading...</span>
            </div>

            <!-- RIGHT: Clock + session badge + theme toggle -->
            <div class="flex items-center" style="gap:0.75rem">
                <div style="text-align:right">
                    <div id="clock" class="copilot-clock" style="color:{session["color_hex"]}">
                        {session["time_et"]}
                    </div>
                    <div id="session-badge" style="font-size:11px;font-weight:600;text-align:right;margin-top:2px;color:{session["color_hex"]}">
                        {session["emoji"]} {session["label"]}
                    </div>
                </div>
                <!-- Mode toggle: Trading vs Review -->
                <div class="flex items-center" style="gap:4px;background:var(--bg-input);border:1px solid var(--border-panel);border-radius:9999px;padding:2px 4px">
                    <button id="mode-btn-trading"
                            onclick="setDashboardMode('trading')"
                            class="mode-btn active-trading"
                            title="Trading Mode — actionable info only">
                        ⚡ Trading
                    </button>
                    <button id="mode-btn-review"
                            onclick="setDashboardMode('review')"
                            class="mode-btn"
                            title="Review Mode — all panels expanded">
                        🔍 Review
                    </button>
                </div>
                <button id="theme-toggle"
                        onclick="toggleTheme()"
                        class="co-btn"
                        title="Toggle dark/light theme"
                        style="padding:4px 8px;font-size:14px;line-height:1">
                    <span id="theme-icon">☀️</span>
                </button>
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
         class="t-panel border t-border rounded-lg flex items-center justify-between"
         style="padding:6px 12px;margin-bottom:0.75rem;flex-wrap:wrap;gap:0.25rem 0">
        <div class="flex items-center" style="gap:0.75rem;flex-wrap:wrap">
            <span class="t-text-muted uppercase tracking-wide font-semibold text-xs">Today's Focus</span>
            <span id="focus-count" class="t-text-secondary font-mono text-xs">{tradeable}/{total} tradeable</span>
            <span id="focus-updated" class="t-text-faint text-xs hidden sm:inline">Updated: {computed}</span>
        </div>
        <div class="flex items-center gap-2">
            <button hx-get="/api/focus/html"
                    hx-target="#focus-grid"
                    hx-swap="innerHTML"
                    hx-indicator="#refresh-spinner"
                    class="co-btn" style="font-size:11px">↻ Refresh</button>
            <span id="refresh-spinner" class="htmx-indicator t-text-faint text-xs">…</span>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════
         MAIN GRID  — 3 cols: Content (2) | Sidebar (1)
    ═══════════════════════════════════════════════════════════════════ -->
    <div class="grid grid-cols-1 xl:grid-cols-3 gap-3">

        <!-- ── LEFT/CENTRE: ORB + asset cards ───────────────────── -->
        <div class="xl:col-span-2 space-y-3">

            <!-- ORB Panel — primary focus -->
            <div id="orb-container" class="t-panel border t-border rounded-lg"
                 hx-get="/api/orb/html"
                 hx-trigger="every 20s"
                 hx-swap="innerHTML">
                {orb_html}
            </div>

            <!-- ORB Signal History — collapsible -->
            <details>
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    ORB Signal History
                </summary>
                <div id="orb-history-container"
                     hx-get="/api/orb/history/html"
                     hx-trigger="revealed"
                     hx-swap="innerHTML">
                    <div class="t-panel border t-border rounded-lg p-4 t-text-faint text-xs" style="text-align:center">
                        Loading history...
                    </div>
                </div>
            </details>

            <!-- Asset Focus Cards -->
            <div>
                <div class="flex items-center justify-between mb-1.5">
                    <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">Asset Focus</h3>
                </div>
                <div id="focus-grid" class="grid grid-cols-1 sm:grid-cols-2 gap-3"
                     hx-get="/api/focus/html"
                     hx-trigger="every 30s"
                     hx-swap="innerHTML">
                    {cards_html}
                </div>
            </div>
        </div>

        <!-- ── SIDEBAR ───────────────────────────────────────────── -->
        <div class="space-y-2.5">

            <!-- Positions & P&L -->
            <div id="positions-container"
                 hx-get="/api/positions/html"
                 hx-trigger="every 10s"
                 hx-swap="innerHTML">
                {positions_html}
            </div>

            <!-- Risk Rules -->
            <div id="risk-container"
                 hx-get="/api/risk/html"
                 hx-trigger="every 15s"
                 hx-swap="innerHTML">
                {risk_html}
            </div>

            <!-- CNN Model -->
            <div id="cnn-panel"
                 class="t-panel border t-border rounded-lg p-3"
                 hx-get="/cnn/status/html"
                 hx-trigger="load, every 15s"
                 hx-swap="innerHTML">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">🧠 CNN Model</h3>
                <div class="t-text-faint text-xs">Loading...</div>
            </div>

            <!-- CNN Dataset Preview — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Dataset Preview
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(139,92,246,0.5)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🖼️ Training Snapshots</h3>
                    </div>
                    <div id="cnn-dataset-preview-container"
                         hx-get="/cnn/dataset/preview"
                         hx-trigger="revealed"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Kraken Crypto health + prices -->
            <div id="kraken-panel"
                 class="t-panel border t-border rounded-lg p-3"
                 style="border-left:3px solid #f7931a"
                 hx-get="/kraken/health/html"
                 hx-trigger="load, every 10s"
                 hx-swap="innerHTML">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">🪙 Crypto (Kraken)</h3>
                <div class="t-text-faint text-xs">Loading...</div>
            </div>

            <!-- Kraken Candlestick Chart — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Crypto Chart
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(247,147,26,0.5)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">📈 Crypto Chart</h3>
                    </div>
                    <div id="kraken-chart-container"
                         hx-get="/kraken/chart/html"
                         hx-trigger="revealed"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Kraken Account (private API) -->
            <details>
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Kraken Account
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(247,147,26,0.4)">
                    <div id="kraken-account-container"
                         hx-get="/kraken/account/html"
                         hx-trigger="revealed, every 30s"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Crypto/Futures Correlation — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Correlation
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(99,102,241,0.5)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🔗 Crypto / Futures Correlation</h3>
                    </div>
                    <div id="correlation-container"
                         hx-get="/kraken/correlation/html"
                         hx-trigger="revealed, every 300s"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Volume Profile — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Volume Profile
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(245,158,11,0.5)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">📊 Volume Profile</h3>
                    </div>
                    <div id="vp-container"
                         hx-get="/api/volume-profile/html"
                         hx-trigger="revealed"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Performance Charts — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Performance
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(34,197,94,0.4)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">📈 Performance</h3>
                    </div>
                    <div id="perf-container"
                         hx-get="/api/performance/html"
                         hx-trigger="revealed"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Trade Journal — review only -->
            <details class="review-only">
                <summary class="cursor-pointer t-text-muted text-xs font-semibold uppercase tracking-wide flex items-center gap-1"
                         style="padding:4px 0">
                    <span class="rotate-on-open" style="transition:transform .15s">▶</span>
                    Trade Journal
                </summary>
                <div class="t-panel border t-border rounded-lg p-3"
                     style="border-left:3px solid rgba(167,139,250,0.5)">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">📓 Trade Journal</h3>
                    </div>
                    <div id="journal-panel-inner"
                         hx-get="/journal/html"
                         hx-trigger="revealed"
                         hx-swap="innerHTML">
                        <div class="t-text-faint text-xs text-center py-3">Loading...</div>
                    </div>
                </div>
            </details>

            <!-- Grok AI Brief — manual pull only in Trading Mode, auto-refresh in Review Mode -->
            <div id="grok-container"
                 hx-get="/api/grok/html"
                 hx-trigger="load"
                 hx-swap="innerHTML">
                {grok_html}
            </div>

            <!-- Market Events feed -->
            {market_events_html}

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

            <!-- Market Regime (HMM) — review only -->
            <div id="regime-container"
                 class="review-only"
                 hx-get="/api/regime/html"
                 hx-trigger="load, every 60s"
                 hx-swap="innerHTML">
                <div class="t-panel border t-border rounded-lg p-3" style="border-left:3px solid #7c3aed">
                    <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">🧮 Market Regime</h3>
                    <div class="t-text-faint text-xs">Loading...</div>
                </div>
            </div>

            <!-- Engine Status -->
            <div class="t-panel border t-border rounded-lg p-3">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">ENGINE STATUS</h3>
                <div id="engine-status"
                     hx-get="/api/time"
                     hx-trigger="every 5s"
                     hx-swap="innerHTML"
                     class="text-xs t-text-muted">—</div>
            </div>

            <!-- Live Feed -->
            <div class="t-panel border t-border rounded-lg p-3">
                <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide mb-1">LIVE FEED</h3>
                <div class="flex items-center justify-between">
                    <div id="sse-heartbeat" class="text-xs t-text-muted">Waiting for heartbeat...</div>
                </div>
                <div id="sse-last-update" class="text-[9px] t-text-faint mt-1">—</div>
            </div>

        </div>
    </div>

    <!-- Footer -->
    <footer style="margin-top:1.5rem;padding-top:0.5rem;border-top:1px solid var(--border-subtle);text-align:center">
        <span class="t-text-faint" style="font-size:10px">
            Pilot v1.0 — Session rules: Pre-market 00–03 | Active 03–12 | Off-hours 12–00 ET
            &nbsp;|&nbsp;
            <a href="/sse/health" class="underline hover:opacity-80">SSE Health</a>
            &nbsp;|&nbsp;
            <a href="/api/info" class="underline hover:opacity-80">API Info</a>
        </span>
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
        if (endH > 24) return h >= startH || h < (endH - 24);
        return h >= startH && h < endH;
    }}

    function updateStrip() {{
        var h = _etHour();
        var cursor = document.getElementById('session-cursor');
        if (cursor) cursor.style.left = (h * HOUR_PCT).toFixed(2) + '%';

        var badgesEl = document.getElementById('session-badges');
        if (!badgesEl) return;
        var html = '';
        for (var i=0; i<SESSIONS.length; i++) {{
            var s = SESSIONS[i];
            var open = _isOpen(s[1], s[2], h);
            var color = open ? s[3] : '#52525b';
            var bgColor = open ? 'rgba(255,255,255,0.05)' : 'transparent';
            var border = open ? '1px solid ' + s[3] + '44' : '1px solid #3f3f46';
            html += '<span style="color:' + color + ';background:' + bgColor + ';border:' + border + ';font-size:9px;padding:1px 6px;border-radius:9999px;white-space:nowrap;display:inline-block;margin:1px">';
            html += (open ? '● ' : '○ ') + s[0];
            html += '</span>';
        }}
        if (_isOpen(9, 12, h)) {{
            html += '<span style="color:#fbbf24;background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);font-size:9px;padding:1px 6px;border-radius:9999px;display:inline-block;margin:1px">⚡ London/US Overlap</span>';
        }}
        badgesEl.innerHTML = html;
    }}

    updateStrip();
    setInterval(updateStrip, 60000);
}})();
</script>

<!-- ═══════════════════════════════════════════════════
     LIVE CLOCK — updates every second
═══════════════════════════════════════════════════════ -->
<script>
function updateClock() {{
    var now = new Date();
    var h = now.toLocaleString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}});
    var parts = h.split(':');
    // Build the big formatted clock: HH:MM:SS  PM  ET
    var ampm = parseInt(parts[0]) >= 12 ? 'PM' : 'AM';
    var h12 = parseInt(parts[0]) % 12 || 12;
    var clockStr = (h12 < 10 ? '0' : '') + h12 + ':' + parts[1] + ':' + parts[2] + '  ' + ampm + '  ET';
    var el = document.getElementById('clock');
    if (el) el.textContent = clockStr;

    var etHour = parseInt(parts[0]);
    var badge = document.getElementById('session-badge');
    var colors = {{
        pre:    '#c084fc',
        london: '#4ade80',
        us:     '#4ade80',
        off:    '#a1a1aa'
    }};
    if (badge) {{
        var c, txt;
        if (etHour>=0&&etHour<3)       {{ c=colors.pre; txt='🌙 PRE-MARKET'; }}
        else if (etHour>=3&&etHour<8)   {{ c=colors.london; txt='🟢 LONDON'; }}
        else if (etHour>=8&&etHour<12)  {{ c=colors.us; txt='🟢 US OPEN'; }}
        else                            {{ c=colors.off; txt='⚙️ OFF-HOURS'; }}
        badge.innerHTML = txt;
        badge.style.color = c;
        if (el) el.style.color = c;
    }}
}}
setInterval(updateClock, 1000);
updateClock();
</script>

<!-- ═══════════════════════════════════════════════════
     MARKET EVENTS FEED
═══════════════════════════════════════════════════════ -->
<script>
var _events = [];
function _pushEvent(emoji, msg, color) {{
    var now = new Date().toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}});
    _events.unshift({{t:now,e:emoji,m:msg,c:color||''}});
    if (_events.length>40) _events.pop();
    var feed = document.getElementById('market-events-feed');
    if (!feed) return;
    var ts = document.getElementById('events-ts');
    if (ts) ts.textContent = now;
    feed.innerHTML = _events.slice(0,12).map(function(ev){{
        var sc = ev.c ? 'color:' + ev.c : '';
        return '<div style="display:flex;align-items:flex-start;gap:6px;padding:2px 0;border-bottom:1px solid var(--border-subtle)">'
             + '<span style="font-size:9px;color:var(--text-faint);font-family:monospace;flex-shrink:0;margin-top:1px">' + ev.t + '</span>'
             + '<span style="font-size:10px;line-height:1.3;' + sc + '">' + ev.e + ' ' + ev.m + '</span>'
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
(function() {{
    var icon = document.getElementById('theme-icon');
    if (icon) icon.textContent = document.documentElement.classList.contains('dark') ? '☀️' : '🌙';
}})();
</script>

<!-- ═══════════════════════════════════════════════════
     TRADING / REVIEW MODE TOGGLE
     - Trading Mode: active session focus — hides review-only panels
     - Review Mode:  all panels visible — for post-session analysis
     - Default: Trading Mode during 03:00–16:00 ET, Review Mode otherwise
     - Persisted in localStorage as 'dashMode'
═══════════════════════════════════════════════════════ -->
<script>
(function() {{
    // Determine default mode based on current ET hour
    function _defaultMode() {{
        try {{
            var etStr = new Date().toLocaleString('en-US', {{
                timeZone: 'America/New_York',
                hour: 'numeric',
                hour12: false
            }});
            var h = parseInt(etStr, 10);
            // Active session hours 03:00–16:00 → Trading; off-hours → Review
            return (h >= 3 && h < 16) ? 'trading' : 'review';
        }} catch(e) {{
            return 'trading';
        }}
    }}

    // Apply the given mode to the document
    function _applyMode(mode) {{
        var body = document.body;
        if (mode === 'review') {{
            body.classList.remove('mode-trading');
            body.classList.add('mode-review');
        }} else {{
            body.classList.remove('mode-review');
            body.classList.add('mode-trading');
        }}

        // Update button states
        var btnT = document.getElementById('mode-btn-trading');
        var btnR = document.getElementById('mode-btn-review');
        if (btnT && btnR) {{
            if (mode === 'trading') {{
                btnT.classList.add('active-trading');
                btnT.classList.remove('active-review');
                btnR.classList.remove('active-review');
                btnR.classList.remove('active-trading');
            }} else {{
                btnR.classList.add('active-review');
                btnR.classList.remove('active-trading');
                btnT.classList.remove('active-trading');
                btnT.classList.remove('active-review');
            }}
        }}

        // In Review Mode, trigger Grok auto-refresh by adding hx-trigger polling.
        // In Trading Mode, the Grok container only loads once (manual pull via buttons).
        var grokContainer = document.getElementById('grok-container');
        if (grokContainer) {{
            if (mode === 'review') {{
                grokContainer.setAttribute('hx-trigger', 'every 60s');
                if (window.htmx) window.htmx.process(grokContainer);
            }} else {{
                grokContainer.setAttribute('hx-trigger', 'load');
                if (window.htmx) window.htmx.process(grokContainer);
            }}
        }}
    }}

    // Global setter — called by the toggle buttons
    window.setDashboardMode = function(mode) {{
        localStorage.setItem('dashMode', mode);
        _applyMode(mode);
    }};

    // On page load: restore saved preference or auto-detect from ET hour
    var saved = localStorage.getItem('dashMode');
    var initial = saved || _defaultMode();
    _applyMode(initial);
}})();
</script>

<!-- ═══════════════════════════════════════════════════
     SSE — native EventSource with reconnect logic
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
        var navDot = document.getElementById('nav-sse-dot');
        if (state==='connected') {{
            if(dot){{dot.className='connected';dot.title='live';}}
            if(txt){{txt.textContent='live';}}
            if(navDot){{navDot.style.color='#22c55e';navDot.title='SSE live';}}
        }} else if(state==='connecting') {{
            if(dot){{dot.className='connecting';}}
            if(txt){{txt.textContent='connecting';}}
            if(navDot){{navDot.style.color='#eab308';navDot.title='SSE connecting';}}
        }} else {{
            if(dot){{dot.className='disconnected';}}
            if(txt){{txt.textContent='reconnecting';}}
            if(navDot){{navDot.style.color='#ef4444';navDot.title='SSE disconnected';}}
        }}
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

        // Focus update
        _es.addEventListener('focus-update', function(e) {{
            try {{
                var focus = JSON.parse(e.data);
                var countEl = document.getElementById('focus-count');
                if (countEl) countEl.textContent = (focus.tradeable_assets||0) + '/' + (focus.total_assets||0) + ' tradeable';
                var updEl = document.getElementById('focus-updated');
                if (updEl && focus.computed_at) {{
                    var d = new Date(focus.computed_at);
                    updEl.textContent = 'Updated: ' + d.toLocaleTimeString('en-US',{{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',hour12:true}}) + ' ET';
                }}
                if (typeof htmx!=='undefined') htmx.ajax('GET','/api/focus/html',{{target:'#focus-grid',swap:'innerHTML'}});
            }} catch(err) {{}}
        }});

        // Heartbeat
        _es.addEventListener('heartbeat', function(e) {{
            try {{
                var hb = JSON.parse(e.data);
                var hbEl = document.getElementById('sse-heartbeat');
                if (hbEl) hbEl.innerHTML = '<span style="color:#22c55e">●</span> Connected — ' + (hb.time_et||'');
                var lastEl = document.getElementById('sse-last-update');
                if (lastEl) lastEl.textContent = 'Last focus: ' + new Date().toLocaleTimeString();
            }} catch(err) {{}}
            if (typeof htmx!=='undefined') htmx.ajax('GET','/api/nt8/health/html',{{target:'#health-bar',swap:'innerHTML'}});
        }});

        // Session change
        _es.addEventListener('session-change', function(e) {{
            try {{
                var sc = JSON.parse(e.data);
                var badge = document.getElementById('session-badge');
                if (badge && sc.emoji && sc.session) badge.innerHTML = sc.emoji + ' ' + sc.session.replace('_','-').toUpperCase();
            }} catch(err) {{}}
        }});

        // No-trade
        _es.addEventListener('no-trade-alert', function(e) {{
            try {{
                var nt = JSON.parse(e.data);
                if (nt.no_trade && typeof htmx!=='undefined') htmx.ajax('GET','/api/no-trade',{{target:'#no-trade-container',swap:'innerHTML'}});
            }} catch(err) {{}}
        }});

        // Positions update
        _es.addEventListener('positions-update', function() {{
            if (typeof htmx!=='undefined') {{
                htmx.ajax('GET','/api/positions/html',{{target:'#positions-container',swap:'innerHTML'}});
                htmx.ajax('GET','/api/nt8/health/html',{{target:'#health-bar',swap:'innerHTML'}});
            }}
            _pushEvent('📋','Position update','#60a5fa');
        }});

        // Bridge status — online/offline state change
        _es.addEventListener('bridge-status', function(e) {{
            try {{
                var bs = JSON.parse(e.data);
                // Always refresh the positions panel so the bridge dot + button state updates
                if (typeof htmx!=='undefined') {{
                    htmx.ajax('GET','/api/positions/html',{{target:'#positions-container',swap:'innerHTML'}});
                    htmx.ajax('GET','/api/nt8/health/html',{{target:'#health-bar',swap:'innerHTML'}});
                }}
                if (bs.connected) {{
                    var acct = bs.account ? ' (' + bs.account + ')' : '';
                    _pushEvent('🟢','NT8 Bridge connected' + acct,'#22c55e');
                }} else {{
                    _pushEvent('🔴','NT8 Bridge offline — no heartbeat','#f87171');
                }}
            }} catch(err) {{}}
        }});

        // Grok update
        _es.addEventListener('grok-update', function() {{
            if (typeof htmx!=='undefined') htmx.ajax('GET','/api/grok/html',{{target:'#grok-container',swap:'innerHTML'}});
        }});

        // Risk update
        _es.addEventListener('risk-update', function() {{
            if (typeof htmx!=='undefined') htmx.ajax('GET','/api/risk/html',{{target:'#risk-container',swap:'innerHTML'}});
        }});

        // ORB update
        _es.addEventListener('orb-update', function(e) {{
            if (typeof htmx!=='undefined') htmx.ajax('GET','/api/orb/html',{{target:'#orb-container',swap:'innerHTML'}});
            try {{
                var orbData = JSON.parse(e.data);
                var dir = orbData.direction||'';
                var sym = orbData.symbol||'';
                if (orbData.breakout_detected) {{
                    var color = dir==='LONG' ? '#4ade80' : '#f87171';
                    var emoji = dir==='LONG' ? '🟢' : '🔴';
                    _pushEvent(emoji, 'ORB breakout ' + dir + ' — ' + sym, color);
                    var orbEl = document.getElementById('orb-container');
                    if (orbEl) {{ orbEl.classList.add('breakout-active'); setTimeout(function(){{orbEl.classList.remove('breakout-active');}},6000); }}
                }} else {{
                    _pushEvent('📐','ORB update — ' + sym, '');
                }}
            }} catch(err) {{}}
        }});

        _es.onmessage = function(e) {{}};
    }}

    function _registerAssetListeners() {{
        document.querySelectorAll('[id^="asset-card-"]').forEach(function(card) {{
            var sym = card.id.replace('asset-card-','');
            if (sym && _es) {{
                _es.addEventListener(sym+'-update', function() {{
                    if (typeof htmx!=='undefined') htmx.ajax('GET','/api/focus/'+encodeURIComponent(sym),{{target:card,swap:'outerHTML'}});
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


@router.get("/api/market-session/html", response_class=HTMLResponse)
def get_market_session_html():
    """Return the session strip HTML fragment for HTMX polling."""
    return HTMLResponse(_render_session_strip())


@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    """Serve the full HTML dashboard page."""
    focus_data = _get_focus_data()
    session = _get_session_info()
    html = _render_full_dashboard(focus_data, session)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


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
    """Return live positions panel with risk status and Bridge state as HTML fragment.

    Always returns content (positions panel renders even when empty),
    so no 204 guard needed here.
    """
    positions = _get_positions()
    risk_status = _get_risk_status()
    bridge_info = _get_bridge_info()
    return HTMLResponse(
        content=_render_positions_panel(
            positions,
            risk_status=risk_status,
            bridge_connected=bridge_info["connected"],
            bridge_age_seconds=bridge_info["age_seconds"],
            bridge_account=bridge_info["account"],
        )
    )


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


# ---------------------------------------------------------------------------
# All 13 breakout types — colour palette for pills, tabs, and row badges
# ---------------------------------------------------------------------------
_ALL_BTYPE_LABELS = [
    "ALL",
    "ORB",
    "PDR",
    "IB",
    "CONS",
    "WEEKLY",
    "MONTHLY",
    "ASIAN",
    "BBSQUEEZE",
    "VA",
    "INSIDE",
    "GAP",
    "PIVOT",
    "FIB",
]

_BTYPE_COLORS: dict[str, tuple[str, str]] = {
    # (background-rgba, foreground-hex)
    "ORB": ("rgba(96,165,250,0.15)", "#60a5fa"),  # blue
    "PDR": ("rgba(192,132,252,0.15)", "#c084fc"),  # purple
    "IB": ("rgba(52,211,153,0.15)", "#34d399"),  # emerald
    "CONS": ("rgba(251,191,36,0.15)", "#fbbf24"),  # amber
    "WEEKLY": ("rgba(20,184,166,0.15)", "#2dd4bf"),  # teal
    "MONTHLY": ("rgba(251,146,60,0.15)", "#fb923c"),  # orange
    "ASIAN": ("rgba(248,113,113,0.15)", "#f87171"),  # red
    "BBSQUEEZE": ("rgba(232,121,249,0.15)", "#e879f9"),  # magenta
    "VA": ("rgba(163,230,53,0.15)", "#a3e635"),  # lime
    "INSIDE": ("rgba(132,204,22,0.15)", "#84cc16"),  # lime-darker
    "GAP": ("rgba(249,115,22,0.15)", "#f97316"),  # coral
    "PIVOT": ("rgba(56,189,248,0.15)", "#38bdf8"),  # steel-blue
    "FIB": ("rgba(234,179,8,0.15)", "#eab308"),  # gold
}

# Session display map: key → (tab_key, tab_text)
_SESSION_TABS: list[tuple[str, str]] = [
    ("all", "All"),
    ("cme", "🕕 CME"),
    ("sydney", "🇦🇺 SYD"),
    ("tokyo", "🇯🇵 TYO"),
    ("shanghai", "🇨🇳 SHA"),
    ("frankfurt", "🇩🇪 FRA"),
    ("london", "🇬🇧 LON"),
    ("london_ny", "🌐 LN-NY"),
    ("us", "🇺🇸 US"),
    ("cme_settle", "📊 SETTLE"),
]


@router.get("/api/orb/history/html", response_class=HTMLResponse)
def get_orb_history_html(
    session: str | None = None,
    symbol: str | None = None,
    days: int = 7,
    breakout_only: bool = False,
    btype: str | None = None,
):
    """Return per-session signal history as an HTML table + summary.

    Query params:
        session       — Filter by session key (e.g. "london", "us", "tokyo")
        symbol        — Filter by symbol (e.g. "MGC=F")
        days          — Lookback window in calendar days (default 7)
        breakout_only — If true, only show events with a breakout detected
        btype         — Filter by breakout type: ALL | ORB | PDR | IB | CONS |
                        WEEKLY | MONTHLY | ASIAN | BBSQUEEZE | VA | INSIDE |
                        GAP | PIVOT | FIB
    """
    from datetime import timedelta

    since = (datetime.now(tz=_EST) - timedelta(days=days)).isoformat()

    # Normalise btype for DB query (None / "ALL" → no filter)
    _btype_db = btype.upper() if btype and btype.upper() != "ALL" else None

    try:
        from lib.core.models import get_orb_events as _get_orb_events

        events = _get_orb_events(
            limit=500,
            symbol=symbol,
            breakout_only=breakout_only,
            since=since,
            breakout_type=_btype_db,
        )
    except Exception:
        events = []

    # Optionally filter by session (stored in the 'session' column or metadata)
    if session and session.lower() != "all":
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

    # (DB already filtered by btype — no additional client-side filter needed)

    # Summary stats
    total = len(events)
    breakouts = sum(1 for e in events if e.get("breakout_detected"))
    longs = sum(1 for e in events if e.get("direction") == "LONG")
    shorts = sum(1 for e in events if e.get("direction") == "SHORT")
    bo_rate = f"{breakouts / total * 100:.0f}%" if total > 0 else "—"

    # btype_filter drives the active pill tab highlight in the UI.
    # It comes directly from the ?btype= query param (already applied above).
    btype_filter: str = btype or ""

    # Summary: per-type counts for the stats bar
    type_counts: dict[str, int] = {}
    for ev in events:
        bt = ev.get("breakout_type") or "ORB"
        type_counts[bt] = type_counts.get(bt, 0) + 1

    # ── Session filter tabs ────────────────────────────────────────────────
    session_filter = (session or "all").lower()

    def _sess_tab(key: str, label: str) -> str:
        active = session_filter == key
        bo_param = f"&breakout_only={'true' if breakout_only else 'false'}"
        days_param = f"&days={days}"
        btype_param = f"&btype={btype}" if btype and btype.upper() != "ALL" else ""
        sess_param = "" if key == "all" else f"&session={key}"
        href = f"/api/orb/history/html?{days_param}{bo_param}{sess_param}{btype_param}"
        active_cls = "t-text font-bold border-b-2 border-blue-500" if active else "t-text-muted hover:opacity-80"
        return (
            f'<a hx-get="{href}" hx-target="#orb-history-container" hx-swap="innerHTML" '
            f'class="cursor-pointer pb-0.5 {active_cls} whitespace-nowrap">{label}</a>'
        )

    session_tabs_html = " ".join(_sess_tab(k, lbl) for k, lbl in _SESSION_TABS)

    # ── Breakout-type filter pills — all 13 types ─────────────────────────
    _bt_active = btype_filter.upper() if btype_filter else "ALL"

    def _bt_tab(label: str) -> str:
        active = label == _bt_active
        _fg = _BTYPE_COLORS.get(label, ("rgba(161,161,170,0.15)", "#a1a1aa"))[1]
        _bg_active = _BTYPE_COLORS.get(label, ("rgba(255,255,255,0.08)", "#ffffff"))[0]
        base_style = "cursor:pointer;padding:1px 7px;border-radius:9999px;font-size:9px;white-space:nowrap"
        if active:
            bg = f"background:{_bg_active};border:1px solid {_fg}55;color:{_fg}"
        else:
            bg = "background:transparent;border:1px solid transparent;color:var(--text-muted)"
        # Build href — preserve current session + breakout_only + days
        sess_param = f"&session={session}" if session and session.lower() != "all" else ""
        bo_param = f"&breakout_only={'true' if breakout_only else 'false'}"
        days_param = f"&days={days}"
        type_param = "" if label == "ALL" else f"&btype={label}"
        href = f"/api/orb/history/html?{days_param}{bo_param}{sess_param}{type_param}"
        # Show count badge when data is present
        cnt = type_counts.get(label, 0)
        cnt_badge = f' <span style="opacity:.6">({cnt})</span>' if cnt and label != "ALL" else ""
        return (
            f'<a hx-get="{href}" hx-target="#orb-history-container" hx-swap="innerHTML" '
            f'style="{base_style};{bg}">{label}{cnt_badge}</a>'
        )

    bt_tabs_html = " ".join(_bt_tab(b) for b in _ALL_BTYPE_LABELS)

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
        ev_btype = (ev.get("breakout_type") or "ORB").upper()
        mtf_score_val = ev.get("mtf_score")
        macd_slope_val = ev.get("macd_slope")
        divergence_val = ev.get("divergence") or ""

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

        # Breakout type badge — full 13-type colour map
        _bt_pill_color = _BTYPE_COLORS.get(ev_btype, ("rgba(161,161,170,0.15)", "#a1a1aa"))
        btype_html = (
            f'<span style="font-size:8px;padding:0 4px;border-radius:3px;'
            f'background:{_bt_pill_color[0]};color:{_bt_pill_color[1]};white-space:nowrap">'
            f"{ev_btype}</span>"
        )

        # MTF score badge
        if mtf_score_val is not None:
            try:
                ms = float(mtf_score_val)
                ms_pct = int(ms * 100)
                ms_color = "#4ade80" if ms >= 0.65 else ("#fbbf24" if ms >= 0.45 else "#f87171")
                # Show MACD slope arrow if available
                slope_arrow = ""
                if macd_slope_val is not None:
                    with contextlib.suppress(Exception):
                        slope_arrow = " ↑" if float(macd_slope_val) > 0 else " ↓"
                # Divergence indicator
                div_icon = ""
                if divergence_val == "confirming":
                    div_icon = " ✓"
                elif divergence_val == "opposing":
                    div_icon = " ✗"
                mtf_html = (
                    f'<span style="font-size:9px;font-family:monospace;color:{ms_color}">'
                    f"{ms_pct}%{slope_arrow}{div_icon}</span>"
                )
            except Exception:
                mtf_html = '<span class="t-text-faint">—</span>'
        else:
            mtf_html = '<span class="t-text-faint">—</span>'

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

        # Session badge — all 9 sessions
        _sk_lower = sk.lower()
        if "london_ny" in _sk_lower or "ln_ny" in _sk_lower:
            sess_badge = '<span class="text-indigo-400 text-[10px]">🌐 LN-NY</span>'
        elif "london" in _sk_lower:
            sess_badge = '<span class="text-blue-400 text-[10px]">🇬🇧 LON</span>'
        elif "frankfurt" in _sk_lower:
            sess_badge = '<span class="text-yellow-400 text-[10px]">🇩🇪 FRA</span>'
        elif "tokyo" in _sk_lower:
            sess_badge = '<span class="text-red-300 text-[10px]">🇯🇵 TYO</span>'
        elif "shanghai" in _sk_lower:
            sess_badge = '<span class="text-red-400 text-[10px]">🇨🇳 SHA</span>'
        elif "sydney" in _sk_lower:
            sess_badge = '<span class="text-cyan-400 text-[10px]">🇦🇺 SYD</span>'
        elif "cme_settle" in _sk_lower or "settle" in _sk_lower:
            sess_badge = '<span class="text-orange-300 text-[10px]">📊 SET</span>'
        elif "cme" in _sk_lower:
            sess_badge = '<span class="text-purple-400 text-[10px]">🕕 CME</span>'
        elif "us" in _sk_lower:
            sess_badge = '<span class="text-emerald-400 text-[10px]">🇺🇸 US</span>'
        elif "crypto" in _sk_lower:
            sess_badge = '<span class="text-orange-400 text-[10px]">₿ CRY</span>'
        else:
            sess_badge = f'<span class="t-text-faint text-[10px]">{sk[:6]}</span>'

        rows_html += f"""
        <tr class="border-b t-border-subtle text-[10px]" style="{row_bg}">
            <td class="py-1 px-1.5 t-text-muted font-mono whitespace-nowrap">{ts_display}</td>
            <td class="py-1 px-1.5">{sess_badge}</td>
            <td class="py-1 px-1.5 t-text-secondary font-mono">{sym}</td>
            <td class="py-1 px-1.5">{btype_html}</td>
            <td class="py-1 px-1.5">{dir_html}</td>
            <td class="py-1 px-1.5 t-text-secondary font-mono text-right">{trigger:,.2f}</td>
            <td class="py-1 px-1.5 t-text-muted font-mono text-right">{or_range:,.2f}</td>
            <td class="py-1 px-1.5 t-text-muted font-mono text-right">{atr:,.2f}</td>
            <td class="py-1 px-1.5 text-center">{mtf_html}</td>
            <td class="py-1 px-1.5 text-center">{cnn_html}</td>
            <td class="py-1 px-1.5 text-center">{filt_html}</td>
        </tr>"""

    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="11" class="py-6 text-center t-text-faint text-xs">
                No events found for the selected filters.
            </td>
        </tr>"""

    # Per-type summary pills for the stats bar — all 13 types
    type_pills_html = ""
    for bt_key in _ALL_BTYPE_LABELS[1:]:  # skip "ALL"
        count = type_counts.get(bt_key, 0)
        if count == 0:
            continue
        pill_bg, pill_fg = _BTYPE_COLORS.get(bt_key, ("rgba(161,161,170,0.12)", "#a1a1aa"))
        type_pills_html += (
            f'<span style="font-size:9px;padding:1px 6px;border-radius:9999px;'
            f'background:{pill_bg};color:{pill_fg};white-space:nowrap">'
            f"{bt_key}: {count}</span> "
        )

    return HTMLResponse(
        content=f"""
    <div class="t-panel border t-border rounded-lg p-4">
        <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold t-text-muted">📊 Signal History</h3>
            <div class="flex items-center gap-2">
                {type_pills_html}
                <span class="t-text-faint text-[10px]">Last {days}d · {total} events</span>
            </div>
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

        <!-- Session filter tabs — all 9 Globex sessions -->
        <div class="overflow-x-auto mb-1.5">
            <div class="flex items-center gap-2 text-[10px] border-b t-border-subtle pb-1.5 min-w-max">
                {session_tabs_html}
                <label class="ml-auto flex items-center gap-1 cursor-pointer t-text-muted pl-2">
                    <input type="checkbox" {bo_checked}
                           hx-get="/api/orb/history/html?{"session=" + session + "&" if session and session.lower() != "all" else ""}days={days}&breakout_only={{this.checked}}"
                           hx-target="#orb-history-container" hx-swap="innerHTML"
                           hx-trigger="change"
                           class="rounded">
                    <span class="text-[10px]">BO only</span>
                </label>
            </div>
        </div>

        <!-- Breakout-type filter pills — all 13 types -->
        <div class="flex items-center gap-1 mb-2 flex-wrap">
            {bt_tabs_html}
        </div>

        <!-- Table -->
        <div class="overflow-x-auto max-h-72 overflow-y-auto">
            <table class="w-full text-left">
                <thead>
                    <tr class="border-b t-border text-[9px] t-text-faint uppercase tracking-wider">
                        <th class="py-1 px-1.5">Time</th>
                        <th class="py-1 px-1.5">Session</th>
                        <th class="py-1 px-1.5">Symbol</th>
                        <th class="py-1 px-1.5">Type</th>
                        <th class="py-1 px-1.5">Signal</th>
                        <th class="py-1 px-1.5 text-right">Trigger</th>
                        <th class="py-1 px-1.5 text-right">Range</th>
                        <th class="py-1 px-1.5 text-right">ATR</th>
                        <th class="py-1 px-1.5 text-center" title="MTF score · MACD slope · Divergence">MTF</th>
                        <th class="py-1 px-1.5 text-center">CNN</th>
                        <th class="py-1 px-1.5 text-center">Filter</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        <div class="mt-1.5 text-[9px] t-text-faint">
            MTF col: score% · ↑↓ MACD slope · ✓ confirming divergence · ✗ opposing
        </div>
    </div>
    """
    )


@router.get("/api/alerts/html", response_class=HTMLResponse)
def _get_gap_alerts() -> dict:
    """Read the latest gap alert payload from Redis.

    Returns the dict published by ``_check_and_alert_gaps()`` in the engine,
    or an empty dict when no gaps have been detected yet.
    """
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:gap_alerts")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def get_alerts_html():
    """Return alerts panel as HTML fragment.

    Now includes:
      - Engine-published alert messages (``engine:alerts`` key)
      - Data-gap warnings from the post-backfill gap scan (``engine:gap_alerts`` key)
    """
    # Read engine alerts from Redis
    alerts = []
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:alerts")
        if raw:
            alerts = json.loads(raw)
    except Exception:
        pass

    # Read gap alerts
    gap_data = _get_gap_alerts()
    gap_alerts = gap_data.get("alerts", [])
    gap_checked_at = gap_data.get("checked_at", "")
    gap_threshold = gap_data.get("threshold_minutes", 30)

    has_alerts = bool(alerts) or bool(gap_alerts)

    if not has_alerts:
        return HTMLResponse(
            content="""
            <div class="t-panel border t-border rounded-lg p-4">
                <h3 class="text-sm font-semibold t-text-muted mb-2">ALERTS</h3>
                <div class="t-text-faint text-sm">No alerts</div>
            </div>
            """
        )

    # ── Engine alert rows ────────────────────────────────────────────────────
    rows = ""
    for alert in alerts[-10:]:
        msg = alert.get("message", alert.get("title", "Alert"))
        _ts = alert.get("timestamp", "")  # noqa: F841
        level = alert.get("level", "info")
        color = {
            "warning": "text-yellow-400",
            "error": "text-red-400",
            "success": "text-green-400",
        }.get(level, "t-text-muted")
        rows += f'<div class="{color} text-xs py-1 border-b t-border-subtle">{msg}</div>'

    # ── Gap alert rows ────────────────────────────────────────────────────────
    gap_rows = ""
    if gap_alerts:
        # Format the checked_at timestamp nicely
        checked_str = ""
        if gap_checked_at:
            try:
                dt = datetime.fromisoformat(gap_checked_at)
                checked_str = dt.strftime("%H:%M ET")
            except Exception:
                checked_str = gap_checked_at

        for ga in gap_alerts[:8]:  # cap at 8 rows
            sym = ga.get("symbol", "?")
            g_count = ga.get("gap_count", 0)
            worst = ga.get("worst_gap_minutes", 0)
            cov = ga.get("coverage_pct", 0)

            # Severity colouring: >4h is critical, >1h is warning
            if worst >= 240:
                badge_color = "#ef4444"
                badge_bg = "rgba(239,68,68,0.12)"
                badge_border = "rgba(239,68,68,0.35)"
                icon = "🔴"
            elif worst >= 60:
                badge_color = "#fbbf24"
                badge_bg = "rgba(251,191,36,0.12)"
                badge_border = "rgba(251,191,36,0.35)"
                icon = "🟡"
            else:
                badge_color = "#fb923c"
                badge_bg = "rgba(251,146,60,0.12)"
                badge_border = "rgba(251,146,60,0.35)"
                icon = "🟠"

            # Format worst-gap duration
            dur_str = f"{worst // 60}h {worst % 60}m" if worst >= 60 else f"{worst}m"

            gap_rows += f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:4px 6px;margin-bottom:3px;
            background:{badge_bg};border:1px solid {badge_border};border-radius:4px">
    <div style="display:flex;align-items:center;gap:5px">
        <span style="font-size:9px">{icon}</span>
        <span style="font-size:10px;font-weight:600;color:{badge_color};font-family:monospace">{sym}</span>
        <span style="font-size:9px;color:var(--text-muted)">{g_count} gap(s)</span>
    </div>
    <div style="text-align:right">
        <span style="font-size:9px;font-family:monospace;color:{badge_color}">worst {dur_str}</span>
        <span style="font-size:8px;color:var(--text-faint);margin-left:4px">{cov:.0f}% cov</span>
    </div>
</div>"""

        gap_section = f"""
<div style="margin-top:6px;padding-top:5px;border-top:1px solid var(--border-subtle,#27272a)">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:9px;font-weight:600;color:var(--text-muted);text-transform:uppercase;
                     letter-spacing:0.05em">⚠️ Data Gaps (≥{gap_threshold}m)</span>
        <span style="font-size:8px;color:var(--text-faint)">{checked_str}</span>
    </div>
    {gap_rows}
    <div style="font-size:8px;color:var(--text-faint);margin-top:2px">
        Set <code>BACKFILL_GAP_ALERT_MINUTES</code> env var to adjust threshold
    </div>
</div>"""
    else:
        gap_section = ""

    return HTMLResponse(
        content=f"""
        <div class="t-panel border t-border rounded-lg p-4">
            <h3 class="text-sm font-semibold t-text-muted mb-2">ALERTS</h3>
            {rows}
            {gap_section}
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


@router.get("/api/volume-profile/html", response_class=HTMLResponse)
def get_volume_profile_html(
    symbol: str = Query(default="MGC=F", description="Ticker symbol"),
    days: int = Query(default=5, ge=1, le=30, description="Days of bar history"),
    bins: int = Query(default=40, ge=20, le=80, description="Number of price bins"),
):
    """Return a volume profile chart panel as an HTML fragment.

    Renders a horizontal histogram with POC, VAH, VAL overlays and naked
    POC markers from prior sessions.  Uses stored 1-minute bars from the
    historical_bars DB table.

    Query params:
        symbol — Yahoo-style ticker (default: MGC=F)
        days   — days of bar history to use (default: 5)
        bins   — number of price bins for the profile (default: 40)
    """
    html = _render_volume_profile_panel(symbol=symbol, days_back=days, bins=bins)
    return HTMLResponse(content=html)


@router.get("/api/performance/html", response_class=HTMLResponse)
def get_performance_html(
    days: int = Query(default=90, ge=7, le=365, description="Days of journal history"),
):
    """Return historical performance charts as an HTML fragment.

    Renders an equity curve, rolling win-rate sparkline, and monthly P&L
    bar chart sourced from the daily_journal table.

    Query params:
        days — journal lookback window in trading days (default: 90)
    """
    html = _render_performance_panel(days_back=days)
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Regime panel renderer
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    "trending": ("#4ade80", "rgba(74,222,128,0.12)", "rgba(74,222,128,0.35)"),  # green
    "volatile": ("#fbbf24", "rgba(251,191,36,0.12)", "rgba(251,191,36,0.35)"),  # amber
    "choppy": ("#f87171", "rgba(248,113,113,0.12)", "rgba(248,113,113,0.35)"),  # red
}

_REGIME_LABELS = {
    "trending": "Trending",
    "volatile": "Volatile",
    "choppy": "Choppy",
}

_REGIME_EMOJI = {
    "trending": "📈",
    "volatile": "⚡",
    "choppy": "🔀",
}

_REGIME_MULT_LABEL = {
    "trending": "Full size (1.0×)",
    "volatile": "Half size (0.5×)",
    "choppy": "Quarter size (0.25×)",
}


def _render_regime_panel() -> str:
    """Render the HMM regime detection sidebar panel.

    Reads the consolidated ``engine:regime_states`` Redis key written by
    ``_publish_regime_states()`` in the engine.  Falls back gracefully when
    the engine hasn't run yet or hmmlearn is not installed.

    Returns a self-contained HTML string suitable for HTMX swap into
    ``#regime-container``.
    """
    regime_map: dict[str, dict] = {}
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:regime_states")
        if raw:
            regime_map = json.loads(raw)
    except Exception:
        pass

    now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

    if not regime_map:
        return f"""
<div class="t-panel border t-border rounded-lg p-3" style="border-left:3px solid #7c3aed">
    <div class="flex items-center justify-between mb-1">
        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🧮 Market Regime</h3>
        <span class="t-text-faint" style="font-size:9px">{now_str}</span>
    </div>
    <div class="t-text-faint text-xs text-center py-3">
        Waiting for engine regime data…<br>
        <span style="font-size:9px">Requires hmmlearn + ≥200 bars of history</span>
    </div>
</div>"""

    # Build per-symbol rows
    rows_html = ""
    # Sort: trending first, then volatile, then choppy; secondary by confidence desc
    _order = {"trending": 0, "volatile": 1, "choppy": 2}
    sorted_items = sorted(
        regime_map.items(),
        key=lambda kv: (_order.get(kv[1].get("regime", "choppy"), 2), -float(kv[1].get("confidence", 0))),
    )

    for sym, info in sorted_items:
        regime = info.get("regime", "choppy")
        confidence = float(info.get("confidence", 0.0))
        confident = bool(info.get("confident", False))
        multiplier = float(info.get("position_multiplier", 0.25))
        persistence = int(info.get("persistence", 0))
        proba = info.get("probabilities", {})

        fg, bg, border_col = _REGIME_COLORS.get(regime, ("#a1a1aa", "rgba(161,161,170,0.12)", "rgba(161,161,170,0.3)"))
        label = _REGIME_LABELS.get(regime, regime.title())
        emoji = _REGIME_EMOJI.get(regime, "❓")
        mult_label = _REGIME_MULT_LABEL.get(regime, f"{multiplier:.2f}×")

        # Confidence bar width (capped at 100%)
        conf_pct = min(100, int(confidence * 100))
        conf_color = "#4ade80" if confidence >= 0.7 else ("#fbbf24" if confidence >= 0.5 else "#f87171")

        # Probability mini-bars for trending / volatile / choppy
        prob_bars = ""
        for r_key, r_label in [("trending", "T"), ("volatile", "V"), ("choppy", "C")]:
            p = float(proba.get(r_key, 0.0))
            r_fg = _REGIME_COLORS.get(r_key, ("#a1a1aa", "", ""))[0]
            prob_bars += (
                f'<div style="display:flex;align-items:center;gap:3px;font-size:8px">'
                f'<span style="color:{r_fg};width:8px;flex-shrink:0">{r_label}</span>'
                f'<div style="flex:1;height:3px;background:var(--bg-bar);border-radius:2px">'
                f'<div style="width:{int(p * 100)}%;height:100%;background:{r_fg};border-radius:2px"></div>'
                f"</div>"
                f'<span style="color:var(--text-faint);width:22px;text-align:right">{int(p * 100)}%</span>'
                f"</div>"
            )

        # Uncertain badge when confidence is below threshold
        uncertain_badge = (
            ""
            if confident
            else '<span style="font-size:8px;color:#fbbf24;margin-left:4px" title="Below confidence threshold">?</span>'
        )

        rows_html += f"""
        <div style="background:{bg};border:1px solid {border_col};border-radius:6px;padding:6px 8px;margin-bottom:5px">
            <div class="flex items-center justify-between mb-1">
                <div class="flex items-center gap-1">
                    <span style="font-size:11px">{emoji}</span>
                    <span style="font-size:11px;font-weight:600;color:{fg}">{label}</span>
                    {uncertain_badge}
                </div>
                <div class="flex items-center gap-2">
                    <span style="font-size:9px;color:var(--text-muted);font-family:monospace">{mult_label}</span>
                    <span style="font-size:9px;color:var(--text-faint);font-family:monospace">{sym}</span>
                </div>
            </div>
            <!-- Confidence bar -->
            <div style="display:flex;align-items:center;gap:4px;margin-bottom:4px">
                <span style="font-size:8px;color:var(--text-faint);width:14px;flex-shrink:0">conf</span>
                <div style="flex:1;height:4px;background:var(--bg-bar);border-radius:2px">
                    <div style="width:{conf_pct}%;height:100%;background:{conf_color};border-radius:2px"></div>
                </div>
                <span style="font-size:8px;color:{conf_color};font-family:monospace;width:28px;text-align:right">{conf_pct}%</span>
            </div>
            <!-- Probability breakdown -->
            <div style="display:flex;flex-direction:column;gap:1px">
                {prob_bars}
            </div>
            <div style="font-size:8px;color:var(--text-faint);margin-top:3px;text-align:right">
                persistence: {persistence} bars
            </div>
        </div>"""

    # Summary: dominant regime across all symbols
    regime_counts: dict[str, int] = {}
    for info in regime_map.values():
        r = info.get("regime", "choppy")
        regime_counts[r] = regime_counts.get(r, 0) + 1

    dominant = max(regime_counts, key=regime_counts.get) if regime_counts else "choppy"  # type: ignore[arg-type]
    dom_fg = _REGIME_COLORS.get(dominant, ("#a1a1aa", "", ""))[0]
    dom_emoji = _REGIME_EMOJI.get(dominant, "❓")
    dom_label = _REGIME_LABELS.get(dominant, dominant.title())

    summary_pills = " ".join(
        f'<span style="font-size:9px;padding:1px 5px;border-radius:9999px;'
        f"background:{_REGIME_COLORS.get(r, ('#a1a1aa', 'rgba(161,161,170,0.12)', ''))[1]};"
        f'color:{_REGIME_COLORS.get(r, ("#a1a1aa", "", ""))[0]}">'
        f"{_REGIME_EMOJI.get(r, '❓')} {r.title()}: {cnt}</span>"
        for r, cnt in sorted(regime_counts.items())
    )

    return f"""
<div class="t-panel border t-border rounded-lg p-3" style="border-left:3px solid #7c3aed">
    <div class="flex items-center justify-between mb-2">
        <h3 class="text-xs font-semibold t-text-muted uppercase tracking-wide">🧮 Market Regime</h3>
        <span class="t-text-faint" style="font-size:9px">{now_str}</span>
    </div>
    <!-- Dominant regime summary -->
    <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-1">
            <span style="font-size:16px">{dom_emoji}</span>
            <span style="font-size:13px;font-weight:700;color:{dom_fg}">{dom_label}</span>
        </div>
        <div class="flex items-center gap-1 flex-wrap" style="justify-content:flex-end">
            {summary_pills}
        </div>
    </div>
    <!-- Per-symbol rows -->
    <div>
        {rows_html}
    </div>
    <div style="font-size:9px;color:var(--text-faint);margin-top:4px;border-top:1px solid var(--border-subtle);padding-top:4px">
        HMM · 3-state Gaussian · forward algorithm (no look-ahead)
    </div>
</div>"""


@router.get("/api/regime/html", response_class=HTMLResponse)
def get_regime_html():
    """Return the HMM regime detection panel as an HTML fragment."""
    return HTMLResponse(content=_render_regime_panel())


@router.get("/api/no-trade", response_class=HTMLResponse)
def get_no_trade():
    """Return no-trade banner HTML if applicable, empty otherwise."""
    focus_data = _get_focus_data()
    if focus_data and focus_data.get("no_trade"):
        return HTMLResponse(
            content=_render_no_trade_banner(str(focus_data.get("no_trade_reason", "Low-conviction day")))
        )
    return HTMLResponse(content='<div id="no-trade-banner"></div>')


# ---------------------------------------------------------------------------
# ORB History — standalone full-page view
# ---------------------------------------------------------------------------

_ORB_HISTORY_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>
<title>ORB Signal History — Futures Co-Pilot</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📅</text></svg>"/>
<script>(function(){{var t=localStorage.getItem('theme');if(t==='light')document.documentElement.classList.remove('dark');else document.documentElement.classList.add('dark');}})();</script>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
/* ── Reset & theme ── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f4f4f5;--bg-panel:rgba(255,255,255,0.85);--bg-inner:rgba(244,244,245,0.6);
  --bg-input:#e4e4e7;--border:#d4d4d8;--border-s:#e4e4e7;
  --text:#18181b;--text2:#3f3f46;--muted:#71717a;--faint:#a1a1aa;
}}
.dark{{
  --bg:#09090b;--bg-panel:rgba(24,24,27,0.7);--bg-inner:rgba(39,39,42,0.5);
  --bg-input:#27272a;--border:#3f3f46;--border-s:#27272a;
  --text:#f4f4f5;--text2:#d4d4d8;--muted:#71717a;--faint:#52525b;
}}
body{{font-family:ui-monospace,'Cascadia Code','Fira Code',monospace;background:var(--bg);color:var(--text);min-height:100vh;font-size:13px}}

/* ── Nav bar ── */
.nav{{display:flex;align-items:center;gap:0;padding:0 1rem;background:var(--bg-panel);
     border-bottom:1px solid var(--border);height:44px;position:sticky;top:0;z-index:100;backdrop-filter:blur(10px)}}
.nav-brand{{font-weight:700;font-size:0.9rem;color:var(--text);text-decoration:none;margin-right:1.5rem;letter-spacing:-0.02em}}
.nav-tab{{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:6px;
         text-decoration:none;color:var(--muted);font-size:0.78rem;font-weight:500;transition:all .15s;white-space:nowrap}}
.nav-tab:hover{{background:var(--bg-inner);color:var(--text)}}
.nav-tab.active{{background:var(--bg-input);color:var(--text);font-weight:600}}
.nav-right{{margin-left:auto;display:flex;align-items:center;gap:8px}}
.theme-btn{{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 8px;
           color:var(--muted);cursor:pointer;font-size:0.75rem;transition:all .15s;font-family:inherit}}
.theme-btn:hover{{color:var(--text);border-color:var(--text)}}

/* ── Layout ── */
.page{{padding:1rem;max-width:1600px;margin:0 auto}}

/* ── Tailwind-compatible utility shims for the HTMX fragment ── */
.t-panel{{background:var(--bg-panel)}}
.t-panel-inner{{background:var(--bg-inner)}}
.t-border{{border-color:var(--border)}}
.t-border-subtle{{border-color:var(--border-s)}}
.t-text{{color:var(--text)}}
.t-text-secondary{{color:var(--text2)}}
.t-text-muted{{color:var(--muted)}}
.t-text-faint{{color:var(--faint)}}
.border{{border-width:1px;border-style:solid}}
.rounded-lg{{border-radius:10px}}
.rounded{{border-radius:6px}}
.p-4{{padding:1rem}}
.p-2{{padding:0.5rem}}
.p-1.5{{padding:0.375rem}}
.px-1.5{{padding-left:0.375rem;padding-right:0.375rem}}
.py-1{{padding-top:0.25rem;padding-bottom:0.25rem}}
.py-6{{padding-top:1.5rem;padding-bottom:1.5rem}}
.mb-1.5{{margin-bottom:0.375rem}}
.mb-2{{margin-bottom:0.5rem}}
.mb-3{{margin-bottom:0.75rem}}
.mt-1.5{{margin-top:0.375rem}}
.flex{{display:flex}}
.items-center{{align-items:center}}
.justify-between{{justify-content:space-between}}
.gap-1{{gap:0.25rem}}
.gap-2{{gap:0.5rem}}
.grid{{display:grid}}
.grid-cols-4{{grid-template-columns:repeat(4,1fr)}}
.text-center{{text-align:center}}
.text-right{{text-align:right}}
.text-left{{text-align:left}}
.font-bold{{font-weight:700}}
.font-semibold{{font-weight:600}}
.font-mono{{font-family:ui-monospace,'Cascadia Code',monospace}}
.uppercase{{text-transform:uppercase}}
.tracking-wide{{letter-spacing:.05em}}
.tracking-wider{{letter-spacing:.08em}}
.whitespace-nowrap{{white-space:nowrap}}
.overflow-x-auto{{overflow-x:auto}}
.overflow-y-auto{{overflow-y:auto}}
.max-h-72{{max-height:18rem}}
.w-full{{width:100%}}
.min-w-max{{min-width:max-content}}
.pb-1.5{{padding-bottom:0.375rem}}
.pl-2{{padding-left:0.5rem}}
.ml-auto{{margin-left:auto}}
.text-sm{{font-size:0.875rem}}
.text-xs{{font-size:0.75rem}}
.text-lg{{font-size:1.125rem}}
.text-\\[10px\\]{{font-size:10px}}
.text-\\[11px\\]{{font-size:11px}}
.text-\\[9px\\]{{font-size:9px}}
.text-\\[8px\\]{{font-size:8px}}
.border-b{{border-bottom-width:1px;border-bottom-style:solid;border-bottom-color:var(--border)}}
.border-b-2{{border-bottom-width:2px;border-bottom-style:solid}}
.border-blue-500{{border-bottom-color:#3b82f6}}
.cursor-pointer{{cursor:pointer}}
.rounded{{border-radius:4px}}
.text-green-400{{color:#4ade80}}
.text-red-400{{color:#f87171}}
.text-yellow-400{{color:#fbbf24}}
.text-blue-400{{color:#60a5fa}}
.text-emerald-400{{color:#34d399}}
.text-indigo-400{{color:#818cf8}}
.text-cyan-400{{color:#22d3ee}}
.text-orange-400{{color:#fb923c}}
.text-orange-300{{color:#fdba74}}
.text-red-300{{color:#fca5a5}}
.text-purple-400{{color:#c084fc}}
.text-green-500{{color:#22c55e}}
.text-red-500{{color:#ef4444}}
.flex-wrap{{flex-wrap:wrap}}
table{{width:100%;border-collapse:collapse}}
th{{padding:4px 6px;text-align:left;border-bottom:1px solid var(--border);
    font-weight:600;text-transform:uppercase;font-size:0.62rem;letter-spacing:.05em;color:var(--faint)}}
td{{padding:4px 6px;border-bottom:1px solid var(--border-s)}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--bg-inner)}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
</style>
</head>
<body>

<!-- Nav bar -->
<nav class="nav">
  <a class="nav-brand" href="/">📈 Co-Pilot</a>
  <a class="nav-tab" href="/">📊 Dashboard</a>
  <a class="nav-tab active" href="/orb-history">📅 ORB History</a>
  <a class="nav-tab" href="/trainer">🧠 Trainer</a>
  <a class="nav-tab" href="/journal/page">📓 Journal</a>
  <a class="nav-tab" href="/settings">⚙️ Settings</a>
  <div class="nav-right">
    <button class="theme-btn" onclick="toggleTheme()">☀/🌙</button>
  </div>
</nav>

<div class="page">

  <!-- Days selector -->
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
    <span style="font-size:0.75rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em">Lookback:</span>
    <div style="display:flex;gap:4px" id="days-pills">
      <a style="padding:3px 12px;border-radius:9999px;font-size:0.74rem;cursor:pointer;text-decoration:none;
                background:var(--bg-input);border:1px solid var(--border);color:var(--text)"
         hx-get="/api/orb/history/html?days=1" hx-target="#orb-history-container" hx-swap="innerHTML">1d</a>
      <a style="padding:3px 12px;border-radius:9999px;font-size:0.74rem;cursor:pointer;text-decoration:none;
                background:var(--bg-input);border:1px solid var(--border);color:var(--text)"
         hx-get="/api/orb/history/html?days=3" hx-target="#orb-history-container" hx-swap="innerHTML">3d</a>
      <a style="padding:3px 12px;border-radius:9999px;font-size:0.74rem;cursor:pointer;text-decoration:none;
                background:#2563eb;border:1px solid #1d4ed8;color:#fff"
         hx-get="/api/orb/history/html?days=7" hx-target="#orb-history-container" hx-swap="innerHTML">7d</a>
      <a style="padding:3px 12px;border-radius:9999px;font-size:0.74rem;cursor:pointer;text-decoration:none;
                background:var(--bg-input);border:1px solid var(--border);color:var(--text)"
         hx-get="/api/orb/history/html?days=14" hx-target="#orb-history-container" hx-swap="innerHTML">14d</a>
      <a style="padding:3px 12px;border-radius:9999px;font-size:0.74rem;cursor:pointer;text-decoration:none;
                background:var(--bg-input);border:1px solid var(--border);color:var(--text)"
         hx-get="/api/orb/history/html?days=30" hx-target="#orb-history-container" hx-swap="innerHTML">30d</a>
    </div>
  </div>

  <!-- ORB history fragment — loaded on page reveal -->
  <div id="orb-history-container"
       hx-get="/api/orb/history/html?days=7"
       hx-trigger="load"
       hx-swap="innerHTML">
    <div style="padding:2rem;text-align:center;color:var(--muted);font-size:0.8rem">Loading signal history…</div>
  </div>

</div>

<script>
function toggleTheme() {
  var html = document.documentElement;
  if (html.classList.contains('dark')) {
    html.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  } else {
    html.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  }
}

// Highlight the active days pill after HTMX swaps
document.body.addEventListener('htmx:afterRequest', function(evt) {
  var url = evt.detail.requestConfig && evt.detail.requestConfig.path;
  if (!url) return;
  var m = url.match(/days=(\\d+)/);
  if (!m) return;
  var days = m[1];
  document.querySelectorAll('#days-pills a').forEach(function(a) {
    var aMatch = (a.getAttribute('hx-get') || '').match(/days=(\\d+)/);
    if (aMatch && aMatch[1] === days) {
      a.style.background = '#2563eb';
      a.style.borderColor = '#1d4ed8';
      a.style.color = '#fff';
    } else {
      a.style.background = 'var(--bg-input)';
      a.style.borderColor = 'var(--border)';
      a.style.color = 'var(--text)';
    }
  });
});
</script>
</body>
</html>
"""


@router.get("/orb-history", response_class=HTMLResponse)
def orb_history_page():
    """Serve the standalone ORB Signal History full-page view."""
    return HTMLResponse(content=_ORB_HISTORY_PAGE_HTML)
