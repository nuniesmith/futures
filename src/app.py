"""
Futures Trading Co-Pilot
========================
Focused single-page dashboard for live trading with NinjaTrader.
No clutter, no unnecessary settings — just what matters.

Key features:
  - ONE setting: account size (controls all risk parameters)
  - Pre-market: scanner, scores, ICT levels, CVD, Grok morning briefing
  - Live trading: "Positions Open" toggle → Grok 15-min updates
  - End of day: simple journal (gross PnL + net PnL → commissions auto-calculated)
  - Auto-refresh every 5 minutes
  - Micro contracts focus (10-20 units)

All heavy computation runs in the background engine.
This page only reads cached results — loads instantly.
"""

import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")

# Ensure sibling modules are importable when run as `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from api_server import get_live_positions  # noqa: E402
from cache import (  # noqa: E402
    REDIS_AVAILABLE,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    get_data_source,
    set_cached_indicator,
)
from confluence import check_confluence, get_recommended_timeframes  # noqa: E402
from costs import estimate_trade_costs, get_cost_model  # noqa: E402
from cvd import compute_cvd, cvd_summary, detect_cvd_divergences  # noqa: E402
from engine import DashboardEngine, get_engine  # noqa: E402
from grok_helper import (  # noqa: E402
    GrokSession,
    _escape_dollars,
    format_market_context,
    run_morning_briefing,
)
from ict import (  # noqa: E402
    ict_summary,
    levels_to_dataframe,
)
from massive_client import is_massive_available  # noqa: E402
from models import (  # noqa: E402
    ACCOUNT_PROFILES,
    ASSETS,
    CONTRACT_SPECS,
    get_daily_journal,
    get_journal_stats,
    get_max_contracts_for_profile,
    init_db,
    save_daily_journal,
)
from scorer import (  # noqa: E402
    EVENT_CATALOG,
    PreMarketScorer,
    score_instruments,
)
from scorer import results_to_dataframe as scorer_to_dataframe  # noqa: E402

# ---------------------------------------------------------------------------
# Initialise database
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Technical indicator helpers (kept minimal)
# ---------------------------------------------------------------------------


def ema(series, length):
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def atr(high, low, close, length=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def compute_vwap(df):
    """Compute daily-resetting VWAP for a DataFrame with OHLCV."""
    out = df.copy()
    out["date"] = out.index.date
    out["typical"] = (out["High"] + out["Low"] + out["Close"]) / 3
    out["tpv"] = out["typical"] * out["Volume"]
    out["cum_tpv"] = out.groupby("date")["tpv"].cumsum()
    out["cum_vol"] = out.groupby("date")["Volume"].cumsum()
    out["VWAP"] = out["cum_tpv"] / out["cum_vol"]
    return out


def compute_pivots(daily_df):
    """Compute prior-day pivot levels from daily bars."""
    if daily_df is None or len(daily_df) < 2:
        return None
    prev = daily_df.iloc[-2]
    ph, pl, pc = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (ph + pl + pc) / 3
    return {
        "Prior High": ph,
        "Prior Low": pl,
        "Prior Close": pc,
        "Pivot": pp,
        "R1": 2 * pp - pl,
        "S1": 2 * pp - ph,
        "R2": pp + (ph - pl),
        "S2": pp - (ph - pl),
    }


def _compute_1m_momentum(df_1m):
    """Compute 1-minute momentum metrics from the last N bars.

    Returns a dict with micro-trend direction, momentum slope,
    last-5-bar range, and 1m volume surge indicator.
    """
    if df_1m is None or df_1m.empty or len(df_1m) < 10:
        return {"1m Trend": "—", "1m Mom": 0.0, "1m Range": 0.0, "1m VolSurge": False}

    tail = df_1m.tail(10)
    closes = tail["Close"]

    # Micro-trend: EMA5 vs EMA10 on 1m
    ema5 = closes.ewm(span=5, adjust=False).mean().iloc[-1]
    ema10 = closes.ewm(span=10, adjust=False).mean().iloc[-1]
    if ema5 > ema10 * 1.0001:
        trend = "🟢 Up"
    elif ema5 < ema10 * 0.9999:
        trend = "🔴 Down"
    else:
        trend = "⚪ Flat"

    # Momentum: slope of last 5 closes (normalised by ATR)
    last5 = closes.tail(5)
    if len(last5) >= 2:
        slope = (float(last5.iloc[-1]) - float(last5.iloc[0])) / max(len(last5) - 1, 1)
    else:
        slope = 0.0

    # Range of last 5 bars
    bar_range = float(tail.tail(5)["High"].max() - tail.tail(5)["Low"].min())

    # Volume surge: last bar volume > 2× average of prior 9
    vol = tail["Volume"]
    if len(vol) >= 2:
        avg_vol = float(vol.iloc[:-1].mean()) if len(vol) > 1 else 1
        last_vol = float(vol.iloc[-1])
        vol_surge = last_vol > 2 * avg_vol and avg_vol > 0
    else:
        vol_surge = False

    return {
        "1m Trend": trend,
        "1m Mom": round(slope, 3),
        "1m Range": round(bar_range, 2),
        "1m VolSurge": vol_surge,
    }


def build_scanner_row(name, ticker, interval, period):
    """Build one scanner row, using indicator cache when possible."""
    cached = get_cached_indicator("scanner", ticker, interval, period)
    if cached is not None:
        return cached

    df = get_data(ticker, interval, period)
    if df.empty:
        return None
    last = float(df["Close"].iloc[-1])

    daily = get_daily(ticker)
    pivots = compute_pivots(daily)
    if daily is not None and len(daily) >= 2:
        prior_close = float(daily["Close"].iloc[-2])
    elif len(df) > 1:
        prior_close = float(df["Close"].iloc[0])
    else:
        prior_close = last
    pct_chg = (last - prior_close) / prior_close * 100 if prior_close != 0 else 0.0

    vdf = compute_vwap(df)
    cum_vol_last = float(vdf["cum_vol"].iloc[-1])
    vwap_val = (
        float(vdf["cum_tpv"].iloc[-1]) / cum_vol_last if cum_vol_last != 0 else last
    )

    pivot = pivots["Pivot"] if pivots else last
    r1 = pivots["R1"] if pivots else last
    s1 = pivots["S1"] if pivots else last
    r2 = pivots["R2"] if pivots else last
    s2 = pivots["S2"] if pivots else last
    dist_pivot = last - pivot

    atr_series = atr(df["High"], df["Low"], df["Close"], length=14)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
    raw_vol = df["Volume"].iloc[-1]
    vol = int(raw_vol) if not pd.isna(raw_vol) else 0

    # 1m momentum overlay
    df_1m = get_data(ticker, "1m", "1d")
    mom = _compute_1m_momentum(df_1m)

    row = {
        "Asset": name,
        "1m": mom["1m Trend"],
        "% Chg": round(pct_chg, 2),
        "Last": round(last, 2),
        "VWAP": round(vwap_val, 2),
        "% VWAP": round((last - vwap_val) / vwap_val * 100, 2) if vwap_val else 0.0,
        "Pivot": round(pivot, 2),
        "S1": round(s1, 2),
        "R1": round(r1, 2),
        "S2": round(s2, 2),
        "R2": round(r2, 2),
        "Dist Pivot": round(dist_pivot, 2),
        "ATR": round(atr_val, 2),
        "Vol": vol,
        "1m Mom": mom["1m Mom"],
        "🔊": "⚡" if mom["1m VolSurge"] else "",
    }
    set_cached_indicator("scanner", ticker, interval, period, row)
    return row


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Futures Co-Pilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hard-coded optimal params — engine auto-optimizes in background
INTERVAL = "5m"
PERIOD = "5d"
INTERVAL_1M = "1m"
PERIOD_1M = "1d"
ALL_ASSETS = list(ASSETS.keys())

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Minimal: Account Size + API Key only
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")

    # Account selector — the ONLY setting that matters
    account_key = st.radio(
        "Account Size",
        list(ACCOUNT_PROFILES.keys()),
        index=0,
        format_func=lambda k: ACCOUNT_PROFILES[k]["label"],
    )
    acct = ACCOUNT_PROFILES[account_key]
    account_size = acct["size"]
    risk_dollars = acct["risk_dollars"]
    max_contracts = get_max_contracts_for_profile(account_key)

    st.divider()
    st.caption("**Risk Parameters** (auto-calculated)")
    st.markdown(
        f"- Risk/trade: **\\${risk_dollars:,}**\n"
        f"- Max contracts: **{max_contracts}** micro\n"
        f"- Soft stop: **\\${abs(acct['soft_stop']):,}**\n"
        f"- Hard stop: **\\${abs(acct['hard_stop']):,}**\n"
        f"- Daily drawdown: **\\${acct['eod_dd']:,}**"
    )

    st.divider()

    # API keys
    env_key = os.getenv("XAI_API_KEY", "")
    if env_key:
        st.session_state.grok_key = env_key
        st.success("Grok API: ✅ Connected")
    else:
        api_key = st.text_input("Grok API Key", type="password", key="grok_key_input")
        if api_key:
            st.session_state.grok_key = api_key
            st.success("API key set")
        else:
            st.warning("Enter Grok API key")

    # Data source status
    st.divider()
    st.markdown("**📡 Data Source**")
    data_source = get_data_source()
    massive_available = is_massive_available()
    if massive_available:
        st.success(f"🟢 **{data_source}** — Real-time futures data")
    else:
        st.info(f"🟡 **{data_source}** — Set `MASSIVE_API_KEY` for real-time data")

    st.divider()
    cache_badge = "Redis" if REDIS_AVAILABLE else "In-Memory"
    st.caption(f"Cache: {cache_badge} · Data: {data_source} · {INTERVAL}/{PERIOD}")
    st.caption("Engine auto-optimizes strategies, intervals, and params in background.")

# ═══════════════════════════════════════════════════════════════════════════
# START ENGINE + LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
engine: DashboardEngine = get_engine(account_size, INTERVAL, PERIOD)

# Load market data for all assets — 5m (big picture) + 1m (live action)
data = {}
data_1m = {}
for name in ALL_ASSETS:
    ticker = ASSETS[name]
    df = get_data(ticker, INTERVAL, PERIOD)
    if not df.empty:
        data[name] = df
    df_1m = get_data(ticker, INTERVAL_1M, PERIOD_1M)
    if not df_1m.empty:
        data_1m[name] = df_1m

# Session time info
now_est = datetime.now(tz=_EST)
current_hour = now_est.hour
session_active = 3 <= current_hour < 12
pre_market = 3 <= current_hour < 9
market_open = 9 <= current_hour < 12
session_warning = 10 <= current_hour < 12

# Initialize Grok session in session_state
if "grok_session" not in st.session_state:
    st.session_state.grok_session = GrokSession()
grok_session: GrokSession = st.session_state.grok_session


# ═══════════════════════════════════════════════════════════════════════════
# TOP BAR — Session Status + Positions Toggle
# ═══════════════════════════════════════════════════════════════════════════
top_left, top_center, top_right = st.columns([2, 3, 2])

with top_left:
    st.markdown("## 🎯 Futures Co-Pilot")

with top_center:
    if session_active and not session_warning:
        st.success(
            f"🟢 Session OPEN — {now_est.strftime('%H:%M')} EST"
            f" {'(Pre-Market)' if pre_market else '(Market Hours)'}"
        )
    elif session_warning:
        st.warning(
            f"🟡 WIND DOWN — {now_est.strftime('%H:%M')} EST — close positions by noon"
        )
    else:
        st.error(f"🔴 Session CLOSED — {now_est.strftime('%H:%M')} EST")

    # Live feed status badge
    feed_status = engine.get_live_feed_status()
    feed_src = feed_status.get("data_source", "yfinance")
    if feed_status.get("connected"):
        bars_n = feed_status.get("bars", 0)
        trades_n = feed_status.get("trades", 0)
        st.caption(
            f"📡 Live: **{feed_src}** · 🟢 Connected · "
            f"Bars: {bars_n:,} · Trades: {trades_n:,}"
        )
    elif feed_src == "Massive":
        st.caption(f"📡 **{feed_src}** · 🟡 REST polling (WS connecting…)")
    else:
        st.caption(f"📡 **{feed_src}** · polling every 60s")

with top_right:
    acct_col, toggle_col = st.columns(2)
    with acct_col:
        st.metric("Account", f"${account_size:,}")
    with toggle_col:
        positions_open = st.toggle(
            "📊 Positions Open",
            value=st.session_state.get("positions_open", False),
            key="positions_open",
            help="Toggle ON when you have positions in NinjaTrader. Enables Grok 15-min updates.",
        )

# Handle toggle state changes
if positions_open and not grok_session.is_active:
    grok_session.activate()
    # Upgrade live feed to per-second aggregates for tighter updates
    engine.upgrade_live_feed()
elif not positions_open and grok_session.is_active:
    grok_session.deactivate()
    # Downgrade back to per-minute aggregates to save bandwidth
    engine.downgrade_live_feed()

# ═══════════════════════════════════════════════════════════════════════════
# LIVE POSITIONS PANEL (from NinjaTrader bridge)
# ═══════════════════════════════════════════════════════════════════════════
live_pos = get_live_positions()
if live_pos["has_positions"]:
    st.divider()
    pos_header, pos_pnl = st.columns([3, 1])
    with pos_header:
        st.markdown("### 📊 Live Positions — NinjaTrader")
    with pos_pnl:
        total_pnl = live_pos["total_unrealized_pnl"]
        pnl_color = "🟢" if total_pnl >= 0 else "🔴"
        st.metric(
            "Unrealized P&L",
            f"{pnl_color} ${total_pnl:+,.2f}",
        )

    pos_rows = []
    for p in live_pos["positions"]:
        upnl = p.get("unrealizedPnL", 0)
        pos_rows.append(
            {
                "Symbol": p.get("symbol", "?"),
                "Side": p.get("side", "?"),
                "Qty": int(p.get("quantity", 0)),
                "Avg Price": f"{p.get('avgPrice', 0):,.2f}",
                "Unrealized P&L": f"${upnl:+,.2f}",
                "Status": "🟢" if upnl >= 0 else "🔴",
            }
        )
    if pos_rows:
        df_pos = pd.DataFrame(pos_rows)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)

    received = live_pos.get("received_at", "")
    acct_name = live_pos.get("account", "")
    st.caption(
        f"Account: **{acct_name}** · Last update: {received} · "
        f"Auto-pushed from NinjaTrader LivePositionBridge"
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Build scanner dataframe (used by multiple sections)
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for name in ALL_ASSETS:
    ticker = ASSETS[name]
    row = build_scanner_row(name, ticker, INTERVAL, PERIOD)
    if row:
        rows.append(row)
df_scan = pd.DataFrame(rows) if rows else pd.DataFrame()
if not df_scan.empty:
    df_scan = df_scan.sort_values("% Chg", key=abs, ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Build all analysis summaries (used by Grok + display)
# ═══════════════════════════════════════════════════════════════════════════
# ICT summaries — use 1m data for sharper level detection, fall back to 5m
ict_summaries = {}
for name in ALL_ASSETS:
    ict_df = data_1m.get(name) if name in data_1m else data.get(name)
    if ict_df is not None and not ict_df.empty:
        try:
            ict_summaries[name] = ict_summary(ict_df)
        except Exception:
            pass

# Confluence results — use proper multi-TF data per asset
confluence_results = {}
for name in ALL_ASSETS:
    try:
        htf_iv, setup_iv, entry_iv = get_recommended_timeframes(name)
        ticker = ASSETS[name]

        # Fetch each timeframe; fall back gracefully
        _tf_period_map = {
            "1m": "1d",
            "5m": "5d",
            "15m": "15d",
            "30m": "1mo",
            "1h": "1mo",
            "60m": "1mo",
        }
        entry_df = get_data(ticker, entry_iv, _tf_period_map.get(entry_iv, "5d"))
        setup_df = get_data(ticker, setup_iv, _tf_period_map.get(setup_iv, "5d"))
        htf_df = get_data(ticker, htf_iv, _tf_period_map.get(htf_iv, "5d"))

        # Fall back to whatever we have if a TF failed
        fallback = data.get(name, pd.DataFrame())
        if entry_df.empty:
            entry_df = data_1m.get(name, fallback)
        if setup_df.empty:
            setup_df = fallback
        if htf_df.empty:
            htf_df = fallback

        if not entry_df.empty and len(entry_df) >= 20:
            conf = check_confluence(
                htf_df=htf_df,
                setup_df=setup_df,
                entry_df=entry_df,
                asset_name=name,
            )
            confluence_results[name] = conf
    except Exception:
        pass

# CVD summaries — use 1m data for sharper delta readings
cvd_summaries = {}
for name in ALL_ASSETS:
    cvd_df = data_1m.get(name) if name in data_1m else data.get(name)
    if cvd_df is not None and not cvd_df.empty:
        try:
            cvd_result = compute_cvd(cvd_df)
            if not cvd_result.empty:
                cvd_summaries[name] = cvd_summary(cvd_result)
        except Exception:
            pass

# Pre-market scorer results
scorer_results = []
if data:
    daily_data_dict = {}
    for name in ALL_ASSETS:
        ticker = ASSETS[name]
        daily_df = get_daily(ticker)
        if daily_df is not None and not daily_df.empty:
            daily_data_dict[name] = daily_df
    try:
        scorer_results = score_instruments(
            intraday_data=data,
            daily_data=daily_data_dict,
        )
    except Exception:
        scorer_results = []


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — MARKET SCANNER (auto-refresh)
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("📡 Market Scanner")

scan_col, score_col = st.columns([3, 2])

with scan_col:

    @st.fragment(run_every=timedelta(seconds=60))
    def _live_scanner():
        """Auto-refreshing scanner table."""
        live_rows = []
        for name in ALL_ASSETS:
            ticker = ASSETS[name]
            row = build_scanner_row(name, ticker, INTERVAL, PERIOD)
            if row:
                live_rows.append(row)
        df_live = pd.DataFrame(live_rows) if live_rows else pd.DataFrame()
        if not df_live.empty:
            df_live = df_live.sort_values("% Chg", key=abs, ascending=False)
            st.dataframe(
                df_live.style.background_gradient(subset=["% VWAP"], cmap="RdYlGn"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Last refresh: {datetime.now(tz=_EST).strftime('%H:%M:%S')} EST · Auto-updates every 60s"
            )
        else:
            st.info("Waiting for market data...")

    _live_scanner()

with score_col:
    st.markdown("**Pre-Market Scores**")
    if scorer_results:
        scorer_df = scorer_to_dataframe(scorer_results)
        st.dataframe(
            scorer_df.style.background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )
        scorer_obj = PreMarketScorer()
        focus_assets = scorer_obj.get_focus_assets(scorer_results, max_focus=3)
        if focus_assets:
            focus_text = ", ".join(f"**{a}**" for a in focus_assets)
            st.success(f"🎯 Focus: {focus_text}")
    else:
        st.info("Scores will appear once data loads.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1b — ⚡ LIVE MINUTE VIEW (auto-refreshes every 30s)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⚡ Live Minute View")


@st.fragment(run_every=timedelta(seconds=30))
def _live_minute_view():
    """Auto-refreshing 1m chart + momentum panel for the top-scored assets."""
    # Pick focus assets: up to 3 based on pre-market score or just first 3
    if scorer_results:
        scorer_obj = PreMarketScorer()
        focus = scorer_obj.get_focus_assets(scorer_results, max_focus=3)
    else:
        focus = ALL_ASSETS[:3]

    if not focus:
        st.info("Waiting for data to determine focus assets...")
        return

    cols = st.columns(len(focus))
    for idx, name in enumerate(focus):
        ticker = ASSETS.get(name)
        if not ticker:
            continue
        df_1m_live = get_data(ticker, "1m", "1d")
        if df_1m_live is None or df_1m_live.empty:
            with cols[idx]:
                st.caption(f"**{name}** — no 1m data")
            continue

        mom = _compute_1m_momentum(df_1m_live)
        tail = df_1m_live.tail(60)  # last 60 minutes
        last_price = float(tail["Close"].iloc[-1])

        with cols[idx]:
            # Header with trend + price
            trend_icon = mom["1m Trend"]
            surge_icon = " ⚡" if mom["1m VolSurge"] else ""
            st.markdown(f"**{name}** {trend_icon}{surge_icon}  **{last_price:,.2f}**")

            # Mini 1m candlestick chart
            fig_1m = go.Figure(
                data=[
                    go.Candlestick(
                        x=tail.index,
                        open=tail["Open"],
                        high=tail["High"],
                        low=tail["Low"],
                        close=tail["Close"],
                        name=name,
                        increasing_line_color="#00D4AA",
                        decreasing_line_color="#FF4444",
                    )
                ]
            )
            # Add VWAP on 1m
            vdf_1m = compute_vwap(tail)
            fig_1m.add_trace(
                go.Scatter(
                    x=vdf_1m.index,
                    y=vdf_1m["VWAP"],
                    name="VWAP",
                    line=dict(color="#9c27b0", width=1.5, dash="dash"),
                )
            )
            fig_1m.update_layout(
                height=220,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                showlegend=False,
                margin=dict(l=0, r=0, t=5, b=5),
                xaxis=dict(showticklabels=True, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(
                fig_1m,
                use_container_width=True,
                key=f"1m_chart_{name}_{time.time():.0f}",
            )

            # Momentum metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.caption(f"Mom: **{mom['1m Mom']:+.3f}**")
            with m2:
                st.caption(f"Range: **{mom['1m Range']:.2f}**")
            with m3:
                last_vol = (
                    int(tail["Volume"].iloc[-1])
                    if not pd.isna(tail["Volume"].iloc[-1])
                    else 0
                )
                st.caption(f"Vol: **{last_vol:,}**")

    st.caption(
        f"🔄 Auto-refresh every 30s · Last: {datetime.now(tz=_EST).strftime('%H:%M:%S')} EST · "
        f"Showing 1m bars (last 60 min) for focus assets"
    )


_live_minute_view()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — KEY LEVELS & ANALYSIS (ICT + CVD + Confluence)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

levels_col, cvd_col, conf_col = st.columns([2, 2, 1])

with levels_col:
    st.subheader("🏦 Key ICT Levels")
    for name in ALL_ASSETS:
        if name in ict_summaries:
            summary = ict_summaries[name]
            stats = summary.get("stats", {})
            nearest = summary.get("nearest_levels", {})

            # Compact display
            above = nearest.get("above", {})
            below = nearest.get("below", {})
            above_str = (
                f"↑ {above.get('label', '—')} @ {above.get('price', '—')}"
                if above
                else "↑ —"
            )
            below_str = (
                f"↓ {below.get('label', '—')} @ {below.get('price', '—')}"
                if below
                else "↓ —"
            )

            fvg_count = stats.get("unfilled_fvgs", 0)
            ob_count = stats.get("active_obs", 0)
            sweep_count = stats.get("recent_sweeps", 0)

            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{name}** · {summary.get('current_price', 0):.2f}")
                with c2:
                    st.caption(
                        f"{above_str} · {below_str} · "
                        f"FVGs: {fvg_count} · OBs: {ob_count} · Sweeps: {sweep_count}"
                    )

with cvd_col:
    st.subheader("📊 Volume Delta (CVD)")
    for name in ALL_ASSETS:
        if name in cvd_summaries:
            s = cvd_summaries[name]
            bias = s.get("bias", "neutral")
            bias_emoji = s.get("bias_emoji", "⚪")
            slope = s.get("cvd_slope", 0)
            delta = s.get("delta_current", 0)

            slope_dir = "Buying" if slope > 0 else "Selling"
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**{name}** {bias_emoji}")
                with c2:
                    st.caption(f"Bias: {bias.title()}")
                with c3:
                    st.caption(f"Slope: {slope:+.3f} ({slope_dir})")

with conf_col:
    st.subheader("🎯 Confluence")
    for name in ALL_ASSETS:
        if name in confluence_results:
            conf = confluence_results[name]
            score = conf.get("score", 0)
            direction = conf.get("direction", "neutral")
            tradeable = conf.get("tradeable", False)
            emoji = "🟢" if score >= 3 else "🟡" if score >= 2 else "🔴"
            trade_badge = " **TRADE**" if tradeable else ""
            htf_iv, setup_iv, entry_iv = get_recommended_timeframes(name)
            st.markdown(
                f"{emoji} **{name}**: {score}/3 {direction.upper()}{trade_badge}"
            )
            st.caption(f"  {htf_iv}/{setup_iv}/{entry_iv}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OPTIMIZED STRATEGIES (from engine)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

with st.expander("🔬 Engine — Optimized Strategies & Backtests", expanded=False):
    opt_col, bt_col = st.columns(2)

    with opt_col:
        st.markdown("**Auto-Optimized Strategies**")
        opt_rows = []
        for name in ALL_ASSETS:
            ticker = ASSETS[name]
            opt = get_cached_optimization(ticker, INTERVAL, PERIOD)
            if opt:
                opt_rows.append(
                    {
                        "Asset": name,
                        "Strategy": opt.get("strategy_label", opt.get("strategy", "?")),
                        "Return": f"{opt.get('return_pct', '?')}%",
                        "Sharpe": opt.get("sharpe", "?"),
                        "Win%": f"{opt.get('win_rate', '?')}%",
                        "Confidence": opt.get("confidence", "?"),
                        "Regime": opt.get("regime", "?"),
                    }
                )
        if opt_rows:
            st.dataframe(
                pd.DataFrame(opt_rows), use_container_width=True, hide_index=True
            )
        else:
            st.caption(
                "Engine is running initial optimization... check back in a few minutes."
            )

    with bt_col:
        st.markdown("**Latest Backtests**")
        bt_results = engine.get_backtest_results()
        if bt_results:
            bt_df = pd.DataFrame(bt_results)
            display_cols = [
                c
                for c in ["Asset", "Return %", "Win Rate %", "Sharpe", "# Trades"]
                if c in bt_df.columns
            ]
            if display_cols:
                st.dataframe(
                    bt_df[display_cols], use_container_width=True, hide_index=True
                )
        else:
            st.caption("Backtests running in background...")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — GROK AI (Morning Briefing + Live Updates)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

grok_col_main, grok_col_live = st.columns([3, 2])

# Build market context for Grok (used by both briefing and live updates)
market_context = format_market_context(
    engine=engine,
    scanner_df=df_scan,
    account_size=account_size,
    risk_dollars=risk_dollars,
    max_contracts=max_contracts,
    contract_specs=CONTRACT_SPECS,
    selected_assets=ALL_ASSETS,
    ict_summaries=ict_summaries,
    confluence_results=confluence_results,
    cvd_summaries=cvd_summaries,
    scorer_results=scorer_results,
    live_positions=live_pos,
)

with grok_col_main:
    st.subheader("🤖 Grok Morning Briefing")

    if st.button("Generate Morning Game Plan", type="primary", key="run_briefing"):
        api_key = st.session_state.get("grok_key", "")
        if not api_key:
            st.error("Enter your Grok API key in the sidebar first.")
        else:
            with st.spinner("Generating morning game plan with Grok..."):
                result = run_morning_briefing(market_context, api_key)
                if result:
                    st.session_state["morning_briefing"] = result
                    grok_session.set_morning_briefing(result)
                else:
                    st.error(
                        "Failed to get response from Grok. Check API key and try again."
                    )

    # Display morning briefing
    if "morning_briefing" in st.session_state:
        with st.container(border=True):
            st.markdown(_escape_dollars(st.session_state["morning_briefing"]))

with grok_col_live:
    st.subheader("📡 Grok Live Updates")

    if positions_open:
        # Show session info
        grok_summary = grok_session.get_session_summary()
        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            st.metric("Updates", grok_summary["total_updates"])
        with info_c2:
            last_time = grok_summary.get("last_update")
            st.metric("Last", last_time or "Pending...")
        with info_c3:
            st.metric("Est. Cost", f"${grok_summary['estimated_cost']:.4f}")

        # Check if we need a new update
        api_key = st.session_state.get("grok_key", "")
        if api_key and grok_session.needs_update():
            with st.spinner("Running 15-min Grok analysis..."):
                update_result = grok_session.run_update(market_context, api_key)

        # Force first update button
        if grok_summary["total_updates"] == 0:
            if st.button("▶ Run First Update Now", key="force_first_update"):
                if api_key:
                    grok_session.last_update_time = 0  # force
                    with st.spinner("Running first Grok update..."):
                        grok_session.run_update(market_context, api_key)
                    st.rerun()
                else:
                    st.error("Enter Grok API key first.")

        # Display latest update
        latest = grok_session.get_latest_update()
        if latest:
            with st.container(border=True):
                st.caption(f"Update #{latest['number']} — {latest['time']}")
                st.markdown(_escape_dollars(latest["text"]))
        else:
            st.info(
                "⏳ First update coming soon... (every 15 min while positions are open)"
            )

        # Show update history in expander
        if len(grok_session.updates) > 1:
            with st.expander(
                f"📋 Update History ({len(grok_session.updates)} updates)"
            ):
                for update in reversed(grok_session.updates[:-1]):
                    st.caption(f"**Update #{update['number']}** — {update['time']}")
                    st.markdown(_escape_dollars(update["text"]))
                    st.divider()
    else:
        st.info(
            "Toggle **📊 Positions Open** when you enter trades in NinjaTrader.\n\n"
            "Grok will review your game plan every 15 minutes with:\n"
            "- Setup status updates\n"
            "- CVD/volume shifts\n"
            "- ICT level reactions\n"
            "- Risk check-ins\n\n"
            "Cost: ~\\$0.007 per update (~\\$0.02/day)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUICK CHARTS (expandable, not the focus)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

with st.expander("📈 Charts", expanded=True):
    chart_left, chart_right = st.columns([1, 4])
    with chart_left:
        chart_asset = st.selectbox("Asset", ALL_ASSETS, key="chart_asset")
        chart_tf = st.radio(
            "Timeframe",
            ["1m", "5m", "15m"],
            index=0,
            key="chart_tf",
            help="1m for live action, 5m/15m for bigger picture",
        )
    with chart_right:
        # Select data based on chosen timeframe
        _chart_tf_period = {"1m": "1d", "5m": "5d", "15m": "15d"}
        chart_ticker = ASSETS[chart_asset]
        if chart_tf == "1m" and chart_asset in data_1m:
            df_chart = data_1m[chart_asset].copy()
        elif chart_tf == "5m" and chart_asset in data:
            df_chart = data[chart_asset].copy()
        else:
            df_chart = get_data(
                chart_ticker, chart_tf, _chart_tf_period.get(chart_tf, "5d")
            )
            if df_chart is not None:
                df_chart = df_chart.copy()
            else:
                df_chart = pd.DataFrame()

        if not df_chart.empty:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df_chart.index,
                        open=df_chart["Open"],
                        high=df_chart["High"],
                        low=df_chart["Low"],
                        close=df_chart["Close"],
                        name="Price",
                        increasing_line_color="#00D4AA",
                        decreasing_line_color="#FF4444",
                    )
                ]
            )

            # Add VWAP
            vdf = compute_vwap(df_chart)
            fig.add_trace(
                go.Scatter(
                    x=vdf.index,
                    y=vdf["VWAP"],
                    name="VWAP",
                    line=dict(color="#9c27b0", width=2, dash="dash"),
                )
            )

            # Add optimized EMAs
            opt = get_cached_optimization(chart_ticker, INTERVAL, PERIOD)
            ema_n1 = opt["n1"] if opt else 9
            ema_n2 = opt["n2"] if opt else 21
            df_chart["EMA_fast"] = ema(df_chart["Close"], length=ema_n1)
            df_chart["EMA_slow"] = ema(df_chart["Close"], length=ema_n2)
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=df_chart["EMA_fast"],
                    name=f"EMA{ema_n1}",
                    line=dict(color="#ff9800"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_chart.index,
                    y=df_chart["EMA_slow"],
                    name=f"EMA{ema_n2}",
                    line=dict(color="#2196f3"),
                )
            )

            # Add pivot levels
            daily = get_daily(chart_ticker)
            pivots = compute_pivots(daily)
            if pivots:
                colors = {
                    "Prior High": "red",
                    "Prior Low": "green",
                    "Pivot": "yellow",
                    "R1": "orange",
                    "S1": "lime",
                    "R2": "darkorange",
                    "S2": "lawngreen",
                }
                for label, val in pivots.items():
                    if label == "Prior Close":
                        continue
                    fig.add_hline(
                        y=val,
                        line_dash="dot",
                        line_color=colors.get(label, "gray"),
                        annotation_text=label,
                        annotation_position="bottom right",
                        line_width=1.5,
                    )

            fig.update_layout(
                height=500,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0),
                title=f"{chart_asset} — {chart_tf}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this asset/timeframe.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — END-OF-DAY JOURNAL
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📓 End-of-Day Journal")

journal_tab_entry, journal_tab_history, journal_tab_stats = st.tabs(
    ["📝 Today's Entry", "📋 History", "📊 Stats"]
)

with journal_tab_entry:
    with st.form("daily_journal_form", clear_on_submit=False):
        st.markdown(
            "Enter your day's results from NinjaTrader. "
            "Commissions are auto-calculated from Gross - Net."
        )

        j_date = st.date_input(
            "Trade Date",
            value=date.today(),
            key="journal_date",
        )

        j_col1, j_col2, j_col3 = st.columns(3)
        with j_col1:
            j_gross = st.number_input(
                "Gross PnL ($)",
                value=0.0,
                step=10.0,
                format="%.2f",
                key="journal_gross",
            )
        with j_col2:
            j_net = st.number_input(
                "Net PnL ($)",
                value=0.0,
                step=10.0,
                format="%.2f",
                key="journal_net",
            )
        with j_col3:
            j_commissions = j_gross - j_net
            color = "🟢" if j_net > 0 else "🔴" if j_net < 0 else "⚪"
            st.metric(
                "Commissions",
                f"${j_commissions:,.2f}",
            )
            st.caption(f"Day result: {color} ${j_net:,.2f}")

        j_detail_col1, j_detail_col2 = st.columns(2)
        with j_detail_col1:
            j_contracts = st.number_input(
                "Total Contracts Traded",
                value=0,
                min_value=0,
                step=1,
                key="journal_contracts",
            )
        with j_detail_col2:
            j_instruments = st.text_input(
                "Instruments Traded",
                value="",
                placeholder="e.g. MES, MNQ, MGC",
                key="journal_instruments",
            )

        j_notes = st.text_area(
            "Notes / Lessons",
            placeholder="What went well? What to improve? Key takeaways...",
            key="journal_notes",
        )

        submitted = st.form_submit_button("💾 Save Day", type="primary")
        if submitted:
            trade_date_str = j_date.strftime("%Y-%m-%d")
            row_id = save_daily_journal(
                trade_date=trade_date_str,
                account_size=account_size,
                gross_pnl=j_gross,
                net_pnl=j_net,
                num_contracts=j_contracts,
                instruments=j_instruments,
                notes=j_notes,
            )
            st.success(f"✅ Journal saved for {trade_date_str} (#{row_id})")

with journal_tab_history:
    journal_df = get_daily_journal(limit=30, account_size=account_size)
    if not journal_df.empty:
        display_cols = [
            "trade_date",
            "gross_pnl",
            "net_pnl",
            "commissions",
            "num_contracts",
            "instruments",
            "notes",
        ]
        available = [c for c in display_cols if c in journal_df.columns]
        display_df = journal_df[available].copy()

        # Rename columns for cleaner display
        rename_map = {
            "trade_date": "Date",
            "gross_pnl": "Gross P&L",
            "net_pnl": "Net P&L",
            "commissions": "Commissions",
            "num_contracts": "Contracts",
            "instruments": "Instruments",
            "notes": "Notes",
        }
        col_rename: dict[str, str] = {
            k: v for k, v in rename_map.items() if k in display_df.columns
        }
        display_df.columns = [col_rename.get(c, c) for c in display_df.columns]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Cumulative P&L chart
        if len(journal_df) > 1:
            sorted_j = journal_df.sort_values("trade_date")
            sorted_j["cumulative_net"] = sorted_j["net_pnl"].cumsum()

            pnl_fig = go.Figure()
            pnl_fig.add_trace(
                go.Scatter(
                    x=sorted_j["trade_date"],
                    y=sorted_j["cumulative_net"],
                    mode="lines+markers",
                    name="Cumulative Net P&L",
                    line=dict(color="#00D4AA", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 170, 0.1)",
                )
            )
            pnl_fig.add_hline(y=0, line_dash="dot", line_color="gray")
            pnl_fig.update_layout(
                height=300,
                template="plotly_dark",
                yaxis_title="Cumulative Net P&L ($)",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(pnl_fig, use_container_width=True)
    else:
        st.info("No journal entries yet. Start recording your daily results!")

with journal_tab_stats:
    stats = get_journal_stats(account_size=account_size)
    if stats["total_days"] > 0:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Trading Days", stats["total_days"])
        with s2:
            win_emoji = "🟢" if stats["win_rate"] >= 50 else "🔴"
            st.metric("Win Rate", f"{win_emoji} {stats['win_rate']}%")
        with s3:
            streak_emoji = (
                "🔥"
                if stats["current_streak"] > 0
                else "❄️"
                if stats["current_streak"] < 0
                else "➖"
            )
            streak_val = abs(stats["current_streak"])
            streak_type = (
                "W"
                if stats["current_streak"] > 0
                else "L"
                if stats["current_streak"] < 0
                else ""
            )
            st.metric("Streak", f"{streak_emoji} {streak_val}{streak_type}")

        with s4:
            total_net = stats["total_net"]
            net_emoji = "🟢" if total_net > 0 else "🔴"
            st.metric("Total Net P&L", f"{net_emoji} ${total_net:,.2f}")

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            avg_dn = stats["avg_daily_net"]
            st.metric("Avg Daily Net", f"${avg_dn:,.2f}")
        with s6:
            best = stats["best_day"]
            st.metric("Best Day", f"${best:,.2f}")
        with s7:
            worst = stats["worst_day"]
            st.metric("Worst Day", f"${worst:,.2f}")
        with s8:
            total_comm = stats["total_commissions"]
            st.metric("Total Commissions", f"${total_comm:,.2f}")

        st.divider()
        win_d = stats["win_days"]
        loss_d = stats["loss_days"]
        be_d = stats["break_even_days"]
        gross_t = stats["total_gross"]
        st.caption(
            f"Win Days: {win_d} · Loss Days: {loss_d} · "
            f"Break Even: {be_d} · "
            f"Gross Total: ${gross_t:,.2f}"
        )
    else:
        st.info("Start logging daily results to see your stats build up here.")


# ═══════════════════════════════════════════════════════════════════════════
# BOTTOM: ENGINE STATUS (tiny footer)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

with st.expander("⚙️ Engine Status", expanded=False):
    status = engine.get_status()
    eng_c1, eng_c2, eng_c3, eng_c4 = st.columns(4)
    with eng_c1:
        eng_status = status.get("engine", "unknown")
        st.metric("Engine", "🟢 Running" if eng_status == "running" else "🔴 Stopped")
    with eng_c2:
        data_status = status.get("data_refresh", {})
        st.metric("Data", data_status.get("status", "idle").title())
    with eng_c3:
        opt_status = status.get("optimization", {})
        st.metric("Optimizer", opt_status.get("status", "idle").title())
    with eng_c4:
        bt_status = status.get("backtest", {})
        st.metric("Backtest", bt_status.get("status", "idle").title())

    # Live feed & data source row
    live_feed_info = status.get("live_feed", {})
    feed_c1, feed_c2, feed_c3, feed_c4 = st.columns(4)
    with feed_c1:
        ds = live_feed_info.get("data_source", "yfinance")
        ds_emoji = "🟢" if ds == "Massive" else "🟡"
        st.metric("Data Source", f"{ds_emoji} {ds}")
    with feed_c2:
        feed_st = live_feed_info.get("status", "off")
        feed_conn = live_feed_info.get("connected", False)
        if feed_conn:
            st.metric("Live Feed", "🟢 Connected")
        elif feed_st == "running":
            st.metric("Live Feed", "🟡 Connecting")
        elif feed_st == "unavailable":
            st.metric("Live Feed", "⚪ No API Key")
        else:
            st.metric("Live Feed", "🔴 Off")
    with feed_c3:
        bars_received = live_feed_info.get("bars", 0)
        st.metric("WS Bars", f"{bars_received:,}")
    with feed_c4:
        trades_received = live_feed_info.get("trades", 0)
        st.metric("WS Trades", f"{trades_received:,}")

    feed_err = live_feed_info.get("error")
    if feed_err:
        st.warning(f"Live feed error: {feed_err}")

    if st.button("🔄 Force Data Refresh", key="force_refresh"):
        flush_all()
        engine.force_refresh()
        st.rerun()

    st.caption(
        "The engine auto-optimizes strategies hourly, refreshes data every 60s, "
        "and backtests every 10 min. All params are auto-tuned — no manual settings needed."
    )
    if live_feed_info.get("data_source") == "Massive":
        st.caption(
            "📡 Real-time data via Massive.com (formerly Polygon.io) — "
            "CME/CBOT/NYMEX/COMEX direct feeds. "
            "WebSocket streams live bars & trades into cache automatically."
        )
    else:
        st.caption(
            "💡 Set MASSIVE_API_KEY in your .env for real-time CME futures data, "
            "WebSocket live streaming, and accurate pre-market levels."
        )


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (5 minutes)
# ═══════════════════════════════════════════════════════════════════════════
# We use st.fragment for individual component refresh (scanner: 60s)
# The full page auto-reruns every 5 minutes to pick up new engine results
# and trigger Grok updates when positions are open.


@st.fragment(run_every=timedelta(minutes=5))
def _auto_refresh_trigger():
    """Silent 5-minute auto-refresh to pick up engine updates and trigger Grok."""
    # This fragment runs every 5 minutes silently.
    # When positions are open, Grok checks happen on the main render
    # cycle triggered by this rerun.
    pass


_auto_refresh_trigger()
