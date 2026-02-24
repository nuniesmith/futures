"""
Futures Intraday Dashboard
==========================
Guided morning workflow → Focus assets → Limit orders → Active trade management.
Multi-account support: $50k / $100k / $150k TakeProfit Trader accounts.
Background engine keeps data fresh, optimises, and backtests automatically.
"""

import json
import os
import re
import sys
from datetime import date, datetime, timedelta

# Ensure sibling modules are importable when run as `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from backtesting import Backtest

from cache import (
    REDIS_AVAILABLE,
    clear_cached_optimization,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    set_cached_indicator,
)
from engine import DashboardEngine, get_engine, run_optimization
from models import (
    ACCOUNT_PROFILES,
    ASSETS,
    CONTRACT_SPECS,
    STATUS_CLOSED,
    STATUS_OPEN,
    calc_max_contracts,
    calc_pnl,
    cancel_trade,
    close_trade,
    create_trade,
    get_all_trades,
    get_open_trades,
    get_today_pnl,
    get_today_trades,
    init_db,
)
from strategies import (
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    make_strategy,
)

# ---------------------------------------------------------------------------
# Initialise database
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Technical indicator helpers
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
    if len(daily_df) < 2:
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


# ---------------------------------------------------------------------------
# Scanner row builder
# ---------------------------------------------------------------------------


def build_scanner_row(name, ticker, interval, period):
    """Build one scanner row, using indicator cache when possible."""
    cached = get_cached_indicator("scanner", ticker, interval, period)
    if cached is not None:
        return cached

    df = get_data(ticker, interval, period)
    if df.empty:
        return None
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    pct_chg = (last - prev) / prev * 100

    vdf = compute_vwap(df)
    cum_vol_last = float(vdf["cum_vol"].iloc[-1])
    vwap_val = (
        float(vdf["cum_tpv"].iloc[-1]) / cum_vol_last if cum_vol_last != 0 else last
    )

    daily = get_daily(ticker)
    pivots = compute_pivots(daily)
    pivot = pivots["Pivot"] if pivots else last
    dist_pivot = last - pivot

    atr_series = atr(df["High"], df["Low"], df["Close"], length=14)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
    raw_vol = df["Volume"].iloc[-1]
    vol = int(raw_vol) if not pd.isna(raw_vol) else 0

    row = {
        "Asset": name,
        "% Overnight": round(pct_chg, 2),
        "Last": round(last, 2),
        "VWAP": round(vwap_val, 2),
        "% from VWAP": round((last - vwap_val) / vwap_val * 100, 2),
        "Pivot": round(pivot, 2),
        "Dist to Pivot": round(dist_pivot, 2),
        "ATR": round(atr_val, 2),
        "Volume": vol,
    }
    set_cached_indicator("scanner", ticker, interval, period, row)
    return row


# ---------------------------------------------------------------------------
# Grok AI helpers
# ---------------------------------------------------------------------------


def _escape_dollars(text: str) -> str:
    """Escape bare $ signs so Streamlit doesn't render them as LaTeX."""
    text = text.replace("$$", "\x00DBL\x00")
    text = re.sub(r"\$([0-9,.\-+])", r"\\$\1", text)
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    text = text.replace("\x00DBL\x00", "$$")
    return text


def _call_grok(prompt, max_tokens=2000, temperature=0.3):
    key = st.session_state.get("grok_key")
    if not key:
        st.error("Enter your Grok API key in the sidebar or set XAI_API_KEY env var.")
        return None
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "grok-4-1-fast-reasoning",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_trade_ideas(text: str) -> list[dict]:
    """Extract JSON trade ideas block from Grok response."""
    patterns = [
        r"```json\s*(\[.*?\])\s*```",
        r"```\s*(\[.*?\])\s*```",
        r"(\[\s*\{.*?\}\s*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                ideas = json.loads(match.group(1))
                if isinstance(ideas, list):
                    return ideas
            except json.JSONDecodeError:
                continue
    return []


# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Futures Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — Account & Settings
# ---------------------------------------------------------------------------
st.sidebar.title("Futures Dashboard")

# Account selector
account_key = st.sidebar.radio(
    "Account Size",
    list(ACCOUNT_PROFILES.keys()),
    index=2,
    format_func=lambda k: ACCOUNT_PROFILES[k]["label"],
    horizontal=True,
)
acct = ACCOUNT_PROFILES[account_key]
account_size = acct["size"]
risk_dollars = acct["risk_dollars"]
max_contracts = acct["max_contracts"]

st.sidebar.success(
    f"Risk/trade: **${risk_dollars:,}** · Max contracts: **{max_contracts}** · "
    f"Soft: **${abs(acct['soft_stop']):,}** · Hard: **${abs(acct['hard_stop']):,}**"
)

# Asset selection
selected_assets = st.sidebar.multiselect(
    "Assets to Track", list(ASSETS.keys()), default=list(ASSETS.keys())
)

interval = st.sidebar.selectbox("Chart Interval", ["1m", "5m"], index=1)
period = st.sidebar.selectbox("Data Period", ["5d", "10d", "1mo", "3mo"], index=1)

# API key
env_key = os.getenv("XAI_API_KEY", "")
if env_key:
    st.session_state.grok_key = env_key
else:
    api_key = st.sidebar.text_input(
        "Grok API Key", type="password", key="grok_key_input"
    )
    if api_key:
        st.session_state.grok_key = api_key

# Engine status badge
cache_badge = "Redis" if REDIS_AVAILABLE else "In-Memory"
st.sidebar.caption(f"Cache: {cache_badge}")

# ---------------------------------------------------------------------------
# Start background engine
# ---------------------------------------------------------------------------
engine: DashboardEngine = get_engine(account_size, interval, period)

# ---------------------------------------------------------------------------
# Load data for selected assets
# ---------------------------------------------------------------------------
data = {}
for name in selected_assets:
    ticker = ASSETS[name]
    df = get_data(ticker, interval, period)
    if not df.empty:
        data[name] = df

# ---------------------------------------------------------------------------
# Daily P&L guard (shown in sidebar)
# ---------------------------------------------------------------------------
today_pnl = get_today_pnl(account_size)
open_trades_df = get_open_trades(account_size)

st.sidebar.divider()
st.sidebar.metric("Today's Realised P&L", f"${today_pnl:,.0f}")
st.sidebar.metric("Open Trades", len(open_trades_df))

if today_pnl <= acct["hard_stop"]:
    st.sidebar.error("HARD STOP HIT — NO MORE TRADES")
elif today_pnl <= acct["soft_stop"]:
    st.sidebar.warning("Soft stop reached — tighten up!")
elif today_pnl >= abs(acct["soft_stop"]):
    st.sidebar.success("Great day! Consider locking profits.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_brief, tab_trades, tab_charts, tab_journal, tab_backtest, tab_engine = st.tabs(
    [
        "🌅 Morning Brief",
        "📊 Active Trades",
        "📈 Charts",
        "📓 Journal",
        "🔬 Backtester",
        "⚙️ Engine",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — MORNING BRIEF
# ═══════════════════════════════════════════════════════════════════════════
with tab_brief:
    st.header("Morning Brief")
    st.warning(acct["playbook_note"])

    # Step 1: Scanner
    st.subheader("1 · Market Scanner")
    if st.button("Refresh Scanner Data", key="refresh_scanner"):
        flush_all()
        engine.force_refresh()
        st.rerun()

    rows = []
    for name in selected_assets:
        ticker = ASSETS[name]
        row = build_scanner_row(name, ticker, interval, period)
        if row:
            rows.append(row)

    df_scan = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df_scan.empty:
        df_scan = df_scan.sort_values("% Overnight", key=abs, ascending=False)
        st.dataframe(
            df_scan.style.background_gradient(subset=["% from VWAP"], cmap="RdYlGn"),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No scanner data. Select assets and refresh.")

    st.divider()

    # Step 2: AI Analysis
    st.subheader("2 · Grok AI Analysis")
    if st.button("Generate Morning Game Plan", type="primary", key="run_brief"):
        scan_text = (
            df_scan.to_string(index=False) if not df_scan.empty else "No scanner data"
        )

        # Correlation matrix
        corr_text = "Not enough data"
        if len(data) >= 2:
            closes = pd.DataFrame({n: d["Close"] for n, d in data.items()})
            returns = closes.pct_change(fill_method=None).dropna()
            corr = returns.corr().round(2)
            corr_text = corr.to_string()

        # Optimisation results
        opt_text_parts = []
        for name in selected_assets:
            ticker = ASSETS[name]
            opt = get_cached_optimization(ticker, interval, period)
            if opt:
                opt_text_parts.append(
                    f"  {name}: n1={opt['n1']}, n2={opt['n2']}, size={opt.get('size', '?')}, return={opt['return_pct']}%"
                )
        opt_text = "\n".join(opt_text_parts) if opt_text_parts else "Not yet run"

        # Backtest results
        bt_results = engine.get_backtest_results()
        bt_text_parts = []
        for r in bt_results:
            bt_text_parts.append(
                f"  {r['Asset']}: return={r['Return %']}%, win_rate={r['Win Rate %']}%, "
                f"sharpe={r['Sharpe']}, trades={r['# Trades']}"
            )
        bt_text = "\n".join(bt_text_parts) if bt_text_parts else "Not yet run"

        prompt = f"""You are a strict TPT-funded futures trader managing a USD {account_size:,} account.

Rules you MUST follow:
- Max {max_contracts} contracts per trade (25% rule)
- Risk exactly 1% (USD {risk_dollars:,}) per trade
- Only early-morning trades (3 AM - noon EST), close everything by noon
- Daily P&L limits: soft stop -USD {abs(acct["soft_stop"]):,}, hard stop -USD {abs(acct["hard_stop"]):,}
- Focus on correlations (Gold/Silver/Copper/Oil/ES/NQ)

FORMATTING RULE: NEVER use bare $ signs for dollar amounts — always write "USD" instead.
Do not use LaTeX or math notation. Use plain text only.

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M")} EST
Account: USD {account_size:,}

Scanner data:
{scan_text}

Correlation matrix (5-min returns):
{corr_text}

Auto-optimized EMA parameters:
{opt_text}

Backtest results (last {period}):
{bt_text}

Give me a complete morning game plan:
1. Overall market bias & rank 1-5 best assets to focus on today
2. For each focus asset: direction, entry zone, SL, TP, contracts, RR, rationale
3. Key correlations to watch and how to use them
4. News/events awareness
5. Risk reminders

After your analysis, provide your trade ideas as a JSON code block using this EXACT format
(entry_low/entry_high define the limit order zone):
```json
[
  {{
    "asset": "Gold",
    "direction": "LONG",
    "entry_low": 2650.00,
    "entry_high": 2655.00,
    "sl": 2640.00,
    "tp": 2670.00,
    "contracts": 3,
    "rationale": "VWAP pullback, above pivot"
  }}
]
```"""

        with st.spinner("Generating morning game plan with Grok..."):
            try:
                result = _call_grok(prompt, max_tokens=3000)
                if result:
                    st.session_state["morning_analysis"] = result
                    st.session_state["morning_trade_ideas"] = _parse_trade_ideas(result)
            except Exception as e:
                st.error(f"API error: {e}")

    # Display analysis
    if "morning_analysis" in st.session_state:
        analysis = st.session_state["morning_analysis"]
        # Display narrative (strip the JSON block for cleaner reading)
        narrative = re.sub(r"```json\s*\[.*?\]\s*```", "", analysis, flags=re.DOTALL)
        narrative = re.sub(r"```\s*\[.*?\]\s*```", "", narrative, flags=re.DOTALL)
        st.markdown(_escape_dollars(narrative.strip()))

    st.divider()

    # Step 3: Focus Assets & Trade Ideas
    st.subheader("3 · Focus Assets & Limit Orders")

    trade_ideas = st.session_state.get("morning_trade_ideas", [])

    if trade_ideas:
        st.caption(
            f"{len(trade_ideas)} trade idea(s) generated. "
            "Review and adjust before creating orders."
        )

        for idx, idea in enumerate(trade_ideas):
            asset_name = idea.get("asset", "Unknown")
            direction = idea.get("direction", "LONG").upper()
            spec = CONTRACT_SPECS.get(asset_name)

            if spec is None:
                st.warning(f"Unknown asset: {asset_name} — skipping")
                continue

            with st.expander(
                f"{'🟢' if direction == 'LONG' else '🔴'} "
                f"{direction} {asset_name} — {idea.get('rationale', '')}",
                expanded=True,
            ):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    entry_low = st.number_input(
                        "Entry Low",
                        value=float(idea.get("entry_low", 0)),
                        step=spec["tick"],
                        key=f"idea_elow_{idx}",
                    )
                with c2:
                    entry_high = st.number_input(
                        "Entry High",
                        value=float(idea.get("entry_high", 0)),
                        step=spec["tick"],
                        key=f"idea_ehigh_{idx}",
                    )
                with c3:
                    sl = st.number_input(
                        "Stop Loss",
                        value=float(idea.get("sl", 0)),
                        step=spec["tick"],
                        key=f"idea_sl_{idx}",
                    )
                with c4:
                    tp = st.number_input(
                        "Take Profit",
                        value=float(idea.get("tp", 0)),
                        step=spec["tick"],
                        key=f"idea_tp_{idx}",
                    )

                mid_entry = (
                    (entry_low + entry_high) / 2
                    if entry_low and entry_high
                    else entry_low
                )
                rec_contracts = calc_max_contracts(
                    mid_entry, sl, asset_name, risk_dollars, max_contracts
                )
                idea_contracts = idea.get("contracts", rec_contracts)

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    contracts = st.number_input(
                        "Contracts",
                        min_value=1,
                        max_value=max_contracts,
                        value=min(int(idea_contracts), max_contracts),
                        key=f"idea_ct_{idx}",
                    )
                with cc2:
                    risk_per = (
                        abs(mid_entry - sl) * spec["point"] * contracts
                        if mid_entry and sl
                        else 0
                    )
                    st.metric("Risk", f"${risk_per:,.0f}")
                with cc3:
                    rr = (
                        abs(tp - mid_entry) / abs(mid_entry - sl)
                        if sl and mid_entry and sl != mid_entry
                        else 0
                    )
                    st.metric("R:R", f"1:{rr:.1f}")

                notes = st.text_input(
                    "Notes", value=idea.get("rationale", ""), key=f"idea_notes_{idx}"
                )

                if st.button(
                    f"Create {direction} Order → Active Trades",
                    key=f"idea_create_{idx}",
                    type="primary",
                ):
                    if today_pnl <= acct["hard_stop"]:
                        st.error("Hard stop hit — cannot create new trades today.")
                    else:
                        trade_id = create_trade(
                            account_size=account_size,
                            asset=asset_name,
                            direction=direction,
                            entry=mid_entry,
                            sl=sl,
                            tp=tp,
                            contracts=int(contracts),
                            strategy="Morning Brief",
                            notes=notes or "",
                        )
                        st.success(
                            f"Order created (#{trade_id}): {direction} {contracts}x {asset_name} "
                            f"@ {mid_entry:.2f} | SL {sl:.2f} | TP {tp:.2f}"
                        )
                        st.rerun()
    else:
        st.info(
            "No trade ideas yet. Run the Morning Game Plan above, or create a manual order below."
        )

    # Manual order entry
    st.divider()
    st.subheader("4 · Manual Order Entry")
    with st.form("manual_order", clear_on_submit=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            m_asset = st.selectbox("Asset", selected_assets, key="m_asset")
            m_direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
        with mc2:
            m_spec = CONTRACT_SPECS.get(m_asset, {"tick": 0.01, "point": 1})
            last_price = (
                float(data[m_asset]["Close"].iloc[-1]) if m_asset in data else 0.0
            )
            m_entry = st.number_input(
                "Entry Price", value=last_price, step=m_spec["tick"]
            )
            m_strategy = st.selectbox(
                "Strategy", ["VWAP Pullback", "Pivot Break", "EMA Crossover", "Manual"]
            )

        mc3, mc4, mc5 = st.columns(3)
        with mc3:
            atr_val = 0.0
            if m_asset in data:
                dfm = data[m_asset]
                atr_val = float(atr(dfm["High"], dfm["Low"], dfm["Close"], 14).iloc[-1])
            m_sl_dist = st.number_input(
                "SL Distance (ATR)", value=round(atr_val, 4), step=m_spec["tick"]
            )
            m_sl = (
                (m_entry - m_sl_dist)
                if m_direction == "Long"
                else (m_entry + m_sl_dist)
            )
            st.caption(f"SL: {m_sl:.4f}")
        with mc4:
            m_rr = st.number_input("R:R Target", value=2.0, step=0.5, min_value=1.0)
            m_tp = (
                m_entry + m_sl_dist * m_rr
                if m_direction == "Long"
                else m_entry - m_sl_dist * m_rr
            )
            st.caption(f"TP: {m_tp:.4f}")
        with mc5:
            m_contracts = calc_max_contracts(
                m_entry, m_sl, m_asset, risk_dollars, max_contracts
            )
            m_contracts = st.number_input(
                "Contracts", min_value=1, max_value=max_contracts, value=m_contracts
            )

        m_notes = st.text_input("Notes", key="m_notes")
        submitted = st.form_submit_button("Create Order", type="primary")

        if submitted:
            if today_pnl <= acct["hard_stop"]:
                st.error("Hard stop hit — cannot create new trades today.")
            else:
                tid = create_trade(
                    account_size=account_size,
                    asset=m_asset,
                    direction=m_direction.upper(),
                    entry=m_entry,
                    sl=m_sl,
                    tp=m_tp,
                    contracts=int(m_contracts),
                    strategy=m_strategy,
                    notes=m_notes or "",
                )
                st.success(
                    f"Order #{tid}: {m_direction.upper()} {m_contracts}x {m_asset} "
                    f"@ {m_entry:.2f} | SL {m_sl:.2f} | TP {m_tp:.2f}"
                )
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ACTIVE TRADES
# ═══════════════════════════════════════════════════════════════════════════
with tab_trades:
    st.header("Active Trades")

    open_df = get_open_trades(account_size)

    if open_df.empty:
        st.info("No open trades. Create orders from the Morning Brief tab.")
    else:
        # Summary metrics
        total_risk = 0.0
        for _, _r in open_df.iterrows():
            _spec = CONTRACT_SPECS.get(str(_r["asset"]))
            if _spec and float(_r["sl"] or 0):
                total_risk += (
                    abs(float(_r["entry"]) - float(_r["sl"]))
                    * _spec["point"]
                    * int(_r["contracts"])
                )

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Open Positions", len(open_df))
        sc2.metric("Total Risk Exposure", f"${total_risk:,.0f}")
        sc3.metric("Today's Realised", f"${today_pnl:,.0f}")
        sc4.metric(
            "Remaining Risk Budget",
            f"${abs(acct['hard_stop']) - abs(today_pnl):,.0f}"
            if today_pnl < 0
            else f"${abs(acct['hard_stop']):,.0f}",
        )

        st.divider()

        # Individual trade cards
        for _, trow in open_df.iterrows():
            trade_id = int(trow["id"])
            asset_name = str(trow["asset"])
            direction = str(trow["direction"])
            entry = float(trow["entry"])
            sl = float(trow["sl"]) if bool(pd.notna(trow["sl"])) else 0.0
            tp = float(trow["tp"]) if bool(pd.notna(trow["tp"])) else 0.0
            contracts = int(trow["contracts"])
            spec = CONTRACT_SPECS.get(asset_name)
            ticker = ASSETS.get(asset_name, "")
            trade_strategy = str(trow.get("strategy", ""))
            trade_created = str(trow["created_at"])
            trade_notes = str(trow.get("notes", ""))

            # Current price
            current_price: float | None = None
            if asset_name in data and not data[asset_name].empty:
                current_price = float(data[asset_name]["Close"].iloc[-1])

            # Unrealised P&L
            unrealised = 0.0
            if current_price is not None and spec:
                unrealised = calc_pnl(
                    asset_name, direction, entry, current_price, contracts
                )

            emoji = "🟢" if direction.upper() == "LONG" else "🔴"
            pnl_color = "normal" if unrealised >= 0 else "inverse"

            with st.container(border=True):
                tc1, tc2, tc3, tc4, tc5 = st.columns([2, 1, 1, 1, 2])
                with tc1:
                    st.markdown(
                        f"**{emoji} #{trade_id} — {direction} {contracts}x {asset_name}**"
                    )
                    st.caption(
                        f"Entry: {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | "
                        f"Strategy: {trade_strategy} | {trade_created}"
                    )
                with tc2:
                    if current_price is not None:
                        st.metric("Current", f"{current_price:.2f}")
                    else:
                        st.metric("Current", "N/A")
                with tc3:
                    st.metric(
                        "Unrealised",
                        f"${unrealised:,.0f}",
                        delta=f"${unrealised:,.0f}",
                        delta_color=pnl_color,
                    )
                with tc4:
                    if current_price is not None and sl and entry != sl:
                        if direction.upper() == "LONG":
                            cur_rr = (current_price - entry) / abs(entry - sl)
                        else:
                            cur_rr = (entry - current_price) / abs(sl - entry)
                        st.metric("R Multiple", f"{cur_rr:+.2f}R")
                    else:
                        st.metric("R Multiple", "—")
                with tc5:
                    close_col, cancel_col = st.columns(2)
                    with close_col:
                        close_px = st.number_input(
                            "Close @",
                            value=current_price if current_price is not None else entry,
                            step=spec["tick"] if spec else 0.01,
                            key=f"close_px_{trade_id}",
                            label_visibility="collapsed",
                        )
                        if st.button(
                            "Close Trade", key=f"close_{trade_id}", type="primary"
                        ):
                            result = close_trade(trade_id, close_px)
                            st.success(f"Closed #{trade_id}: P&L ${result['pnl']:,.2f}")
                            st.rerun()
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel_{trade_id}"):
                            cancel_trade(trade_id)
                            st.info(f"Cancelled #{trade_id}")
                            st.rerun()

                if trade_notes:
                    st.caption(f"Notes: {trade_notes}")

    # Daily risk summary
    st.divider()
    st.subheader("Daily Risk Summary")
    dr1, dr2, dr3 = st.columns(3)
    dr1.metric("Realised P&L", f"${today_pnl:,.0f}")
    dr2.metric("Soft Stop", f"${acct['soft_stop']:,}")
    dr3.metric("Hard Stop", f"${acct['hard_stop']:,}")

    if today_pnl <= acct["hard_stop"]:
        st.error("HARD STOP HIT — NO MORE TRADES TODAY. Walk away.")
    elif today_pnl <= acct["soft_stop"]:
        st.warning("Soft stop reached — reduce size or stop trading.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.header("Charts & Correlations")

    show_vwap = st.checkbox("Show VWAP", value=True)
    show_pivots = st.checkbox("Show Pivots", value=True)
    show_ema = st.checkbox("Show EMAs", value=True)

    col_chart, col_corr = st.columns([3, 1])

    with col_chart:
        asset_chart = st.selectbox("Chart Asset", selected_assets, key="chart_asset")
        if asset_chart in data and not data[asset_chart].empty:
            df = data[asset_chart].copy()

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Price",
                    )
                ]
            )

            if show_ema:
                # Use optimized params if available
                ticker = ASSETS[asset_chart]
                opt = get_cached_optimization(ticker, interval, period)
                ema_n1 = opt["n1"] if opt else 9
                ema_n2 = opt["n2"] if opt else 21

                df["EMA_fast"] = ema(df["Close"], length=ema_n1)
                df["EMA_slow"] = ema(df["Close"], length=ema_n2)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["EMA_fast"],
                        name=f"EMA{ema_n1}",
                        line=dict(color="#ff9800"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["EMA_slow"],
                        name=f"EMA{ema_n2}",
                        line=dict(color="#2196f3"),
                    )
                )

            if show_vwap:
                vdf = compute_vwap(df)
                fig.add_trace(
                    go.Scatter(
                        x=vdf.index,
                        y=vdf["VWAP"],
                        name="VWAP",
                        line=dict(color="#9c27b0", width=2, dash="dash"),
                    )
                )

            if show_pivots:
                daily = get_daily(ASSETS[asset_chart])
                pivots = compute_pivots(daily)
                if pivots:
                    colors = {
                        "Prior High": "red",
                        "Prior Low": "green",
                        "Prior Close": "white",
                        "Pivot": "yellow",
                        "R1": "orange",
                        "S1": "lime",
                        "R2": "darkorange",
                        "S2": "lawngreen",
                    }
                    for label, val in pivots.items():
                        fig.add_hline(
                            y=val,
                            line_dash="dot",
                            line_color=colors.get(label, "gray"),
                            annotation_text=label,
                            annotation_position="bottom right",
                            line_width=1.5,
                        )

            # Mark open trades on chart
            open_for_asset = (
                open_df[open_df["asset"] == asset_chart]
                if not open_df.empty
                else pd.DataFrame()
            )
            for _, t in open_for_asset.iterrows():
                fig.add_hline(
                    y=t["entry"],
                    line_dash="solid",
                    line_color="cyan",
                    annotation_text=f"Entry #{int(t['id'])}",
                    line_width=2,
                )
                if bool(pd.notna(t["sl"])) and float(t["sl"]):
                    fig.add_hline(
                        y=t["sl"],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="SL",
                        line_width=1,
                    )
                if bool(pd.notna(t["tp"])) and float(t["tp"]):
                    fig.add_hline(
                        y=t["tp"],
                        line_dash="dash",
                        line_color="lime",
                        annotation_text="TP",
                        line_width=1,
                    )

            fig.update_layout(
                height=700,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, width="stretch")

    with col_corr:
        st.subheader("Correlations")
        if len(data) >= 2:
            closes = pd.DataFrame({n: d["Close"] for n, d in data.items()})
            returns = closes.pct_change(fill_method=None).dropna()
            corr = returns.corr().round(2)
            st.dataframe(
                corr.style.background_gradient(cmap="RdYlGn"),
                width="stretch",
            )
        else:
            st.info("Select 2+ assets")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — JOURNAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_journal:
    st.header("Trade Journal")

    journal_scope = st.radio(
        "Scope", ["Today", "This Week", "All Time"], horizontal=True, key="j_scope"
    )

    if journal_scope == "Today":
        dfj = get_today_trades(account_size)
    elif journal_scope == "This Week":
        week_start = (
            datetime.now() - timedelta(days=datetime.now().weekday())
        ).strftime("%Y-%m-%d")
        all_trades = get_all_trades(account_size)
        dfj = (
            all_trades[all_trades["created_at"] >= week_start]
            if not all_trades.empty
            else pd.DataFrame()
        )
    else:
        dfj = get_all_trades(account_size)

    if not dfj.empty:
        # Stats
        closed_df = pd.DataFrame(dfj[dfj["status"] == STATUS_CLOSED])
        total_trades = len(closed_df)
        wins = (
            len(closed_df[closed_df["pnl"] > 0])
            if "pnl" in closed_df.columns and total_trades > 0
            else 0
        )
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_pnl = float(closed_df["pnl"].sum()) if "pnl" in closed_df.columns else 0.0
        avg_rr = (
            float(closed_df["rr"].mean())
            if "rr" in closed_df.columns and total_trades > 0
            else 0.0
        )

        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("Closed Trades", total_trades)
        j2.metric("Win Rate", f"{win_rate:.1f}%")
        j3.metric("Total P&L", f"${total_pnl:,.0f}")
        j4.metric("Avg R:R", f"{avg_rr:.2f}")
        j5.metric(
            "Open",
            len(dfj[dfj["status"] == STATUS_OPEN]),
        )

        # Status color coding
        def _status_color(val):
            if val == STATUS_OPEN:
                return "color: #2196f3"
            elif val == STATUS_CLOSED:
                return "color: #4caf50"
            return ""

        display_cols = [
            "id",
            "created_at",
            "asset",
            "direction",
            "entry",
            "sl",
            "tp",
            "contracts",
            "status",
            "close_price",
            "close_time",
            "pnl",
            "rr",
            "strategy",
            "notes",
        ]
        available_cols = [c for c in display_cols if c in dfj.columns]
        dfj_display = pd.DataFrame(dfj[available_cols])
        styled = dfj_display.style.map(_status_color, subset=["status"])
        st.dataframe(styled, width="stretch", hide_index=True)

        # P&L over time chart
        if not closed_df.empty and "close_time" in closed_df.columns:
            pnl_ts = closed_df.copy()
            pnl_ts["close_time"] = pd.to_datetime(pnl_ts["close_time"])
            pnl_ts = pnl_ts.sort_values("close_time")
            pnl_ts["cumulative_pnl"] = pnl_ts["pnl"].cumsum()

            pnl_fig = go.Figure()
            pnl_fig.add_trace(
                go.Scatter(
                    x=pnl_ts["close_time"],
                    y=pnl_ts["cumulative_pnl"],
                    mode="lines+markers",
                    name="Cumulative P&L",
                    fill="tozeroy",
                    line=dict(color="#2196f3"),
                )
            )
            pnl_fig.update_layout(
                template="plotly_dark",
                height=300,
                yaxis_title="Cumulative P&L ($)",
                xaxis_title="Time",
            )
            st.plotly_chart(pnl_fig, width="stretch")

        # Export
        st.divider()
        ex1, ex2 = st.columns(2)
        with ex1:
            csv_data = dfj.to_csv(index=False).encode()
            st.download_button(
                "Export as CSV",
                csv_data,
                f"trades_{journal_scope.lower().replace(' ', '_')}.csv",
                "text/csv",
            )
        with ex2:
            json_data = dfj.to_json(orient="records", date_format="iso") or ""
            st.download_button(
                "Export as JSON",
                json_data,
                f"trades_{journal_scope.lower().replace(' ', '_')}.json",
                "application/json",
            )
    else:
        st.info("No trades found for this scope.")

    # Grok AI Review section
    st.divider()
    st.subheader("AI Trade Review")

    review_scope = st.radio(
        "Review Period",
        ["Today", "Specific Date"],
        horizontal=True,
        key="review_scope",
    )

    review_date = date.today()
    if review_scope == "Specific Date":
        review_date = st.date_input(
            "Select date", value=date.today(), max_value=date.today()
        )

    if st.button(f"Generate AI Review for {review_date}", key="ai_review"):
        d_str = review_date.strftime("%Y-%m-%d")
        all_t = get_all_trades(account_size)
        day_trades = (
            all_t[
                all_t["created_at"].str.startswith(d_str)
                | all_t["close_time"].fillna("").str.startswith(d_str)
            ]
            if not all_t.empty
            else pd.DataFrame()
        )
        trades_text = (
            day_trades.to_string(index=False)
            if not day_trades.empty
            else "No trades on this day"
        )
        day_pnl = (
            float(day_trades[day_trades["status"] == STATUS_CLOSED]["pnl"].sum())
            if not day_trades.empty and "pnl" in day_trades.columns
            else 0.0
        )

        prompt = f"""You are reviewing my trading day for a USD {account_size:,} TPT account.
Rules: 1% risk (USD {risk_dollars:,}), max {max_contracts} contracts, early-morning only,
close by noon, daily limits -USD {abs(acct["soft_stop"]):,} soft / -USD {abs(acct["hard_stop"]):,} hard.

FORMATTING RULE: NEVER use bare $ signs — always write "USD". No LaTeX.

Date: {d_str}
Total P&L: USD {day_pnl:,.0f}
Trades:
{trades_text}

Provide a professional daily review:
1. Performance summary (win rate, avg RR, total risk taken)
2. What worked well
3. What went wrong / rule breaks
4. Key lessons & 1-2 improvements for tomorrow
5. TPT compliance check

Bullet points only, honest, constructive."""

        with st.spinner("Generating review..."):
            try:
                result = _call_grok(prompt, max_tokens=1500, temperature=0.4)
                if result:
                    st.markdown(_escape_dollars(result))
            except Exception as e:
                st.error(f"API error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("Backtester")

    # Show latest engine backtest results
    bt_results = engine.get_backtest_results()
    if bt_results:
        st.subheader("Auto-Backtest Results (Background Engine)")
        df_bt_all = pd.DataFrame(bt_results)
        display_cols_bt = [
            "Asset",
            "Strategy",
            "n1",
            "n2",
            "Size",
            "Params",
            "Return %",
            "Buy & Hold %",
            "# Trades",
            "Win Rate %",
            "Max DD %",
            "Sharpe",
            "Profit Factor",
            "Expectancy %",
            "Final Equity $",
        ]
        available_bt = [c for c in display_cols_bt if c in df_bt_all.columns]
        # Convert n1/n2 to strings so Arrow doesn't choke on mixed int/None
        df_bt_display = df_bt_all[available_bt].copy()
        if "n1" in df_bt_display.columns:
            df_bt_display["n1"] = pd.Series(df_bt_display["n1"]).apply(
                lambda v: str(int(v)) if bool(pd.notna(v)) else "—"
            )
        if "n2" in df_bt_display.columns:
            df_bt_display["n2"] = pd.Series(df_bt_display["n2"]).apply(
                lambda v: str(int(v)) if bool(pd.notna(v)) else "—"
            )
        # Only apply gradient to columns that have numeric, non-NaN data
        gradient_cols: list[str] = []
        for gc in ["Return %", "Win Rate %", "Sharpe"]:
            if gc in df_bt_display.columns and bool(
                pd.Series(df_bt_display[gc]).notna().any()
            ):
                gradient_cols.append(gc)
        styler = df_bt_display.style
        for gc in gradient_cols:
            styler = styler.background_gradient(subset=[gc], cmap="RdYlGn")
        fmt: dict = {
            "Return %": "{:+.2f}",
            "Buy & Hold %": "{:+.2f}",
            "Max DD %": "{:.2f}",
            "Sharpe": "{:.2f}",
            "Profit Factor": "{:.2f}",
            "Expectancy %": "{:.3f}",
            "Final Equity $": "${:,.0f}",
            "Size": "{:.3f}",
            "Win Rate %": "{:.1f}",
        }
        # Only format columns that actually exist
        fmt = {k: v for k, v in fmt.items() if k in df_bt_display.columns}
        styled_bt = styler.format(fmt)
        st.dataframe(styled_bt, width="stretch", hide_index=True)

        # Portfolio summary
        avg_ret = df_bt_all["Return %"].mean()
        avg_wr = df_bt_all["Win Rate %"].mean()
        total_tr = df_bt_all["# Trades"].sum()
        avg_sharpe = df_bt_all["Sharpe"].mean()
        worst_dd = df_bt_all["Max DD %"].min()
        winners = len(df_bt_all[df_bt_all["Return %"] > 0])
        losers = len(df_bt_all[df_bt_all["Return %"] <= 0])

        ps1, ps2, ps3, ps4 = st.columns(4)
        ps1.metric("Avg Return", f"{avg_ret:+.2f}%")
        ps2.metric("Avg Win Rate", f"{avg_wr:.1f}%")
        ps3.metric("Total Trades", int(total_tr))
        ps4.metric("Profitable Assets", f"{winners}/{winners + losers}")

        # Bar chart
        comp_fig = go.Figure()
        comp_fig.add_trace(
            go.Bar(
                x=df_bt_all["Asset"],
                y=df_bt_all["Return %"],
                name="EMA Strategy",
                marker_color="#2196f3",
            )
        )
        if "Buy & Hold %" in df_bt_all.columns:
            comp_fig.add_trace(
                go.Bar(
                    x=df_bt_all["Asset"],
                    y=df_bt_all["Buy & Hold %"],
                    name="Buy & Hold",
                    marker_color="#ff9800",
                )
            )
        comp_fig.update_layout(
            barmode="group",
            template="plotly_dark",
            height=400,
            yaxis_title="Return %",
        )
        st.plotly_chart(comp_fig, width="stretch")
    else:
        st.info(
            "Backtest results will appear here once the engine completes its first cycle."
        )

    st.divider()

    # Single asset backtest
    st.subheader("Single Asset Backtest")

    bt_col1, bt_col2 = st.columns([1, 1])
    with bt_col1:
        asset_bt = st.selectbox("Asset", selected_assets, key="bt_asset")
    with bt_col2:
        # Strategy selector — show all available strategies
        strat_options = list(STRATEGY_LABELS.keys())
        strat_labels_list = [STRATEGY_LABELS[k] for k in strat_options]
        bt_strat_idx = st.selectbox(
            "Strategy",
            range(len(strat_options)),
            format_func=lambda i: strat_labels_list[i],
            key="bt_strat",
        )
        bt_strat_key = strat_options[bt_strat_idx]

    opt_params = None
    if asset_bt in ASSETS:
        opt_params = get_cached_optimization(ASSETS[asset_bt], interval, period)

    use_opt = False
    if opt_params:
        opt_strat = opt_params.get("strategy_label", opt_params.get("strategy", "EMA"))
        opt_sharpe = opt_params.get("sharpe", "?")
        opt_ret = opt_params.get("return_pct", "?")
        opt_detail_parts = []
        if "params" in opt_params:
            for pk, pv in opt_params["params"].items():
                opt_detail_parts.append(f"{pk}={pv}")
        else:
            opt_detail_parts.append(f"n1={opt_params.get('n1', '?')}")
            opt_detail_parts.append(f"n2={opt_params.get('n2', '?')}")
            opt_detail_parts.append(f"size={opt_params.get('size', '?')}")
        st.success(
            f"**Best strategy: {opt_strat}** — Sharpe={opt_sharpe}, Return={opt_ret}% · "
            + " · ".join(opt_detail_parts)
        )
        use_opt = st.checkbox(
            "Use optimized strategy & parameters", value=True, key="bt_use_opt"
        )
    else:
        st.info(
            "No optimized params yet — using defaults. Engine will optimize automatically."
        )

    # Strategy-specific parameter inputs
    st.caption(f"**{STRATEGY_LABELS[bt_strat_key]}** parameters:")

    # Determine defaults (from optimizer if matching strategy, else class defaults)
    opt_p = {}
    if use_opt and opt_params and opt_params.get("strategy") == bt_strat_key:
        opt_p = opt_params.get("params", {})

    strat_cls = STRATEGY_CLASSES[bt_strat_key]

    if bt_strat_key in ("TrendEMA", "PlainEMA"):
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_n1 = st.number_input(
                "Fast EMA",
                min_value=2,
                max_value=50,
                value=int(opt_p.get("n1", getattr(strat_cls, "n1", 9))),
                key="bt_n1",
            )
        with bc2:
            bt_n2 = st.number_input(
                "Slow EMA",
                min_value=5,
                max_value=100,
                value=int(opt_p.get("n2", getattr(strat_cls, "n2", 21))),
                key="bt_n2",
            )
        with bc3:
            bt_size = st.number_input(
                "Size",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                value=float(
                    opt_p.get("trade_size", getattr(strat_cls, "trade_size", 0.10))
                ),
                key="bt_size",
            )
        with bc4:
            if bt_strat_key == "TrendEMA":
                bt_trend = st.number_input(
                    "Trend EMA",
                    min_value=20,
                    max_value=200,
                    value=int(
                        opt_p.get(
                            "trend_period", getattr(strat_cls, "trend_period", 50)
                        )
                    ),
                    key="bt_trend",
                )
            else:
                bt_trend = 50  # unused for PlainEMA but keeps variable bound
        manual_params = {"n1": bt_n1, "n2": bt_n2, "trade_size": bt_size}
        if bt_strat_key == "TrendEMA":
            manual_params["trend_period"] = bt_trend
            bc5, bc6, bc7 = st.columns(3)
            with bc5:
                manual_params["atr_period"] = st.number_input(
                    "ATR Period",
                    min_value=5,
                    max_value=30,
                    value=int(opt_p.get("atr_period", 14)),
                    key="bt_atr_p",
                )
            with bc6:
                manual_params["atr_sl_mult"] = st.number_input(
                    "ATR SL ×",
                    min_value=0.5,
                    max_value=5.0,
                    step=0.25,
                    value=float(opt_p.get("atr_sl_mult", 1.5)),
                    key="bt_sl_m",
                )
            with bc7:
                manual_params["atr_tp_mult"] = st.number_input(
                    "ATR TP ×",
                    min_value=0.5,
                    max_value=8.0,
                    step=0.25,
                    value=float(opt_p.get("atr_tp_mult", 2.5)),
                    key="bt_tp_m",
                )

    elif bt_strat_key == "RSI":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_rsi_p = st.number_input(
                "RSI Period",
                min_value=5,
                max_value=30,
                value=int(opt_p.get("rsi_period", 14)),
                key="bt_rsi_p",
            )
        with bc2:
            bt_rsi_os = st.number_input(
                "Oversold",
                min_value=10,
                max_value=45,
                value=int(opt_p.get("rsi_oversold", 30)),
                key="bt_rsi_os",
            )
        with bc3:
            bt_rsi_ob = st.number_input(
                "Overbought",
                min_value=55,
                max_value=90,
                value=int(opt_p.get("rsi_overbought", 70)),
                key="bt_rsi_ob",
            )
        with bc4:
            bt_size = st.number_input(
                "Size",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                value=float(opt_p.get("trade_size", 0.10)),
                key="bt_size",
            )
        bc5, bc6, bc7, bc8 = st.columns(4)
        with bc5:
            bt_trend = st.number_input(
                "Trend EMA",
                min_value=20,
                max_value=200,
                value=int(opt_p.get("trend_period", 50)),
                key="bt_trend",
            )
        with bc6:
            bt_atr_p = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=30,
                value=int(opt_p.get("atr_period", 14)),
                key="bt_atr_p",
            )
        with bc7:
            bt_sl_m = st.number_input(
                "ATR SL ×",
                min_value=0.5,
                max_value=5.0,
                step=0.25,
                value=float(opt_p.get("atr_sl_mult", 1.5)),
                key="bt_sl_m",
            )
        with bc8:
            bt_tp_m = st.number_input(
                "ATR TP ×",
                min_value=0.5,
                max_value=8.0,
                step=0.25,
                value=float(opt_p.get("atr_tp_mult", 2.0)),
                key="bt_tp_m",
            )
        manual_params = {
            "rsi_period": bt_rsi_p,
            "rsi_oversold": bt_rsi_os,
            "rsi_overbought": bt_rsi_ob,
            "trend_period": bt_trend,
            "atr_period": bt_atr_p,
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "trade_size": bt_size,
        }

    elif bt_strat_key == "Breakout":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_lookback = st.number_input(
                "Lookback",
                min_value=5,
                max_value=100,
                value=int(opt_p.get("lookback", 20)),
                key="bt_lookback",
            )
        with bc2:
            bt_vol_mult = st.number_input(
                "Vol ×",
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                value=float(opt_p.get("vol_mult", 1.2)),
                key="bt_vol_mult",
            )
        with bc3:
            bt_size = st.number_input(
                "Size",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                value=float(opt_p.get("trade_size", 0.10)),
                key="bt_size",
            )
        with bc4:
            bt_atr_p = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=30,
                value=int(opt_p.get("atr_period", 14)),
                key="bt_atr_p",
            )
        bc5, bc6, bc7 = st.columns(3)
        with bc5:
            bt_sl_m = st.number_input(
                "ATR SL ×",
                min_value=0.5,
                max_value=5.0,
                step=0.25,
                value=float(opt_p.get("atr_sl_mult", 1.5)),
                key="bt_sl_m",
            )
        with bc6:
            bt_tp_m = st.number_input(
                "ATR TP ×",
                min_value=0.5,
                max_value=8.0,
                step=0.25,
                value=float(opt_p.get("atr_tp_mult", 3.0)),
                key="bt_tp_m",
            )
        with bc7:
            bt_vol_sma = st.number_input(
                "Vol SMA",
                min_value=5,
                max_value=50,
                value=int(opt_p.get("vol_sma_period", 20)),
                key="bt_vol_sma",
            )
        manual_params = {
            "lookback": bt_lookback,
            "atr_period": bt_atr_p,
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "vol_sma_period": bt_vol_sma,
            "vol_mult": bt_vol_mult,
            "trade_size": bt_size,
        }

    else:
        # Fallback for any unhandled strategy key
        manual_params = {"trade_size": 0.10}

    if asset_bt in data and not data[asset_bt].empty:
        df_bt = data[asset_bt].copy()

        # Build strategy class with the selected params
        configured_cls = make_strategy(bt_strat_key, manual_params)
        bt = Backtest(
            df_bt,
            configured_cls,
            cash=account_size,
            commission=0.0002,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()

        # Show key metrics in columns
        sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
        sm1.metric("Return %", f"{float(stats['Return [%]']):+.2f}")
        _sharpe_raw = stats["Sharpe Ratio"]
        _sharpe_ok = bool(pd.notna(_sharpe_raw))
        sm2.metric(
            "Sharpe",
            f"{float(_sharpe_raw):.2f}" if _sharpe_ok else "—",
        )
        sm3.metric("# Trades", int(stats["# Trades"]))
        sm4.metric(
            "Win Rate",
            f"{float(stats['Win Rate [%]']):.1f}%"
            if int(stats["# Trades"]) > 0
            else "—",
        )
        sm5.metric("Max DD", f"{float(stats['Max. Drawdown [%]']):.2f}%")
        sm6.metric("B&H Return", f"{float(stats['Buy & Hold Return [%]']):+.2f}%")

        # Full stats table
        with st.expander("Full Backtest Stats"):
            stats_display = stats.apply(lambda x: str(x))
            st.dataframe(stats_display.to_frame(name="Value"), width="stretch")

        if st.button("Force Re-Optimize This Asset", key="bt_force_opt"):
            ticker = ASSETS[asset_bt]
            clear_cached_optimization(ticker, interval, period)
            with st.spinner(f"Optimizing {asset_bt} (3 strategies × 30 trials)..."):
                result = run_optimization(ticker, interval, period, account_size)
                if result:
                    strat_name = result.get("strategy_label", "?")
                    st.success(
                        f"Done! Best: **{strat_name}** — "
                        f"Sharpe={result.get('sharpe', '?')}, "
                        f"Return={result['return_pct']}%"
                    )
                    st.rerun()
    else:
        st.info("Select an asset with data to backtest.")

    st.divider()

    # Optimization results table
    st.subheader("Current Optimization Cache")
    opt_rows = []
    for name in selected_assets:
        ticker = ASSETS[name]
        cached = get_cached_optimization(ticker, interval, period)
        if cached:
            strat_label = cached.get("strategy_label", cached.get("strategy", "EMA"))
            params_str = ""
            if "params" in cached:
                params_str = ", ".join(f"{k}={v}" for k, v in cached["params"].items())
            else:
                params_str = f"n1={cached.get('n1', '?')}, n2={cached.get('n2', '?')}"
            opt_rows.append(
                {
                    "Asset": name,
                    "Strategy": strat_label,
                    "Sharpe": cached.get("sharpe", "?"),
                    "Return %": cached["return_pct"],
                    "Score": cached.get("score", "?"),
                    "Params": params_str,
                    "Updated": cached["updated"],
                }
            )
    if opt_rows:
        st.dataframe(pd.DataFrame(opt_rows), width="stretch", hide_index=True)
    else:
        st.info("Engine will populate optimization results automatically.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ENGINE STATUS
# ═══════════════════════════════════════════════════════════════════════════
with tab_engine:
    st.header("Background Engine")
    st.write(
        "The engine runs in the background, keeping data fresh, running Optuna "
        "optimizations when cache expires, and backtesting all assets with optimal parameters."
    )

    status = engine.get_status()

    # Engine state
    engine_state = status.get("engine", "unknown")
    if engine_state == "running":
        st.success("Engine is running")
    else:
        st.error("Engine is stopped")
        if st.button("Start Engine"):
            engine.start()
            st.rerun()

    st.divider()

    # Status cards
    e1, e2, e3 = st.columns(3)

    with e1:
        st.subheader("Data Refresh")
        dr = status.get("data_refresh", {})
        st.metric("Status", dr.get("status", "idle").upper())
        st.metric("Last Refresh", dr.get("last", "Never"))
        if dr.get("error"):
            st.error(dr["error"])
        st.caption(f"Interval: every {engine.DATA_REFRESH_INTERVAL}s")

    with e2:
        st.subheader("Optimization")
        opt_s = status.get("optimization", {})
        st.metric("Status", opt_s.get("status", "idle").upper())
        st.metric("Last Run", opt_s.get("last", "Never"))
        if opt_s.get("progress"):
            st.info(f"Working on: {opt_s['progress']}")
        if opt_s.get("error"):
            st.error(opt_s["error"])
        st.caption(f"Interval: every {engine.OPTIMIZATION_INTERVAL}s (1 hour)")

    with e3:
        st.subheader("Backtesting")
        bt_s = status.get("backtest", {})
        st.metric("Status", bt_s.get("status", "idle").upper())
        st.metric("Last Run", bt_s.get("last", "Never"))
        if bt_s.get("progress"):
            st.info(f"Working on: {bt_s['progress']}")
        if bt_s.get("error"):
            st.error(bt_s["error"])
        st.caption(f"Interval: every {engine.BACKTEST_INTERVAL}s (10 min)")

    st.divider()

    # Manual controls
    st.subheader("Manual Controls")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        if st.button("Force Data Refresh", type="primary"):
            engine.force_refresh()
            st.success("Data refresh triggered")
    with mc2:
        if st.button("Clear All Cache"):
            flush_all()
            st.success("Cache cleared")
            st.rerun()
    with mc3:
        if st.button("Force Full Re-Optimization"):
            flush_all()
            engine.force_refresh()
            st.success("Cache cleared, engine will re-optimize on next cycle")

    # Settings
    st.divider()
    st.subheader("Engine Settings")
    st.json(
        {
            "account_size": engine.account_size,
            "interval": engine.interval,
            "period": engine.period,
            "data_refresh_interval_s": engine.DATA_REFRESH_INTERVAL,
            "optimization_interval_s": engine.OPTIMIZATION_INTERVAL,
            "backtest_interval_s": engine.BACKTEST_INTERVAL,
            "assets_tracked": list(ASSETS.keys()),
            "cache_backend": "Redis" if REDIS_AVAILABLE else "In-Memory",
        }
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    f"Futures Dashboard · {acct['label']} · "
    f"Cache: {cache_badge} · "
    f"Engine: {status.get('engine', 'unknown')} · "
    f"Data: Yahoo Finance · Close all by noon EST"
)
