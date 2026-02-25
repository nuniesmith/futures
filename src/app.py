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
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")

# Ensure sibling modules are importable when run as `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from backtesting import Backtest  # noqa: E402

from alerts import get_dispatcher, send_signal  # noqa: E402
from cache import (  # noqa: E402
    REDIS_AVAILABLE,
    clear_cached_optimization,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    set_cached_indicator,
)
from confluence import (  # noqa: E402
    check_confluence,
    get_recommended_timeframes,
)
from costs import (  # noqa: E402
    estimate_trade_costs,
    get_cost_model,
    should_use_full_contracts,
    slippage_commission_rate,
)
from cvd import (  # noqa: E402
    compute_cvd,
    cvd_summary,
    detect_absorption_candles,
    detect_cvd_divergences,
)
from engine import (  # noqa: E402
    OPTIMIZER_STRATEGIES,
    TRAIN_RATIO,
    TRIALS_PER_STRATEGY,
    DashboardEngine,
    filter_session_hours,
    get_engine,
    run_optimization,
)
from models import (  # noqa: E402
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
from monte_carlo import (  # noqa: E402
    compute_confidence_cones,
    cone_curves_to_dataframe,
    drawdown_distribution_to_dataframe,
    mc_results_to_dataframe,
    run_monte_carlo,
)
from scorer import (  # noqa: E402
    EVENT_CATALOG,
    PreMarketScorer,
    score_instruments,
)
from scorer import (  # noqa: E402
    results_to_dataframe as scorer_to_dataframe,
)
from strategies import (  # noqa: E402
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    make_strategy,
)
from volume_profile import (  # noqa: E402
    compute_session_profiles,
    compute_volume_profile,
    find_naked_pocs,
    profile_to_dataframe,
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

    # Use prior daily close for true overnight % change (not previous 5-min bar)
    daily = get_daily(ticker)
    pivots = compute_pivots(daily)
    if daily is not None and len(daily) >= 2:
        prior_close = float(daily["Close"].iloc[-2])
    elif len(df) > 1:
        prior_close = float(df["Close"].iloc[0])  # fallback: first bar of period
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

    row = {
        "Asset": name,
        "% Overnight": round(pct_chg, 2),
        "Last": round(last, 2),
        "VWAP": round(vwap_val, 2),
        "% from VWAP": round((last - vwap_val) / vwap_val * 100, 2),
        "Pivot": round(pivot, 2),
        "S2": round(s2, 2),
        "S1": round(s1, 2),
        "R1": round(r1, 2),
        "R2": round(r2, 2),
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


def _validate_trade_ideas(
    ideas: list[dict], scanner_df: pd.DataFrame, max_atr_mult: float = 5.0
) -> tuple[list[dict], list[str]]:
    """Sanity-check Grok trade ideas against actual scanner prices.

    Compares each idea's entry_low, entry_high, sl, and tp against the
    scanner's "Last" price for that asset.  Any price more than
    max_atr_mult × ATR away from Last is flagged as a hallucination and
    the idea is discarded.

    Returns:
        (valid_ideas, warnings)  — cleaned list and human-readable warnings.
    """
    if scanner_df.empty or not ideas:
        return ideas, []

    # Build lookup: asset name → {last, atr}
    price_lookup: dict[str, dict[str, float]] = {}
    for _, row in scanner_df.iterrows():
        asset = str(row.get("Asset") or "")
        _last_raw = row.get("Last")
        last = float(_last_raw) if _last_raw is not None else 0.0
        _atr_raw = row.get("ATR")
        atr_val = float(_atr_raw) if _atr_raw is not None else 0.0
        if asset and last > 0:
            price_lookup[asset] = {
                "last": last,
                "atr": atr_val if atr_val > 0 else last * 0.01,
            }

    valid: list[dict] = []
    warnings: list[str] = []

    for idea in ideas:
        asset_name = idea.get("asset", "")
        ref = price_lookup.get(asset_name)
        if ref is None:
            # Unknown asset — keep but warn
            warnings.append(f"⚠️ {asset_name}: not in scanner, cannot validate prices.")
            valid.append(idea)
            continue

        last = ref["last"]
        atr_val = ref["atr"]
        threshold = max_atr_mult * atr_val

        # Check all price fields
        bad_fields: list[str] = []
        for field in ("entry_low", "entry_high", "sl", "tp"):
            val = idea.get(field)
            if val is not None:
                try:
                    price = float(val)
                except (TypeError, ValueError):
                    bad_fields.append(f"{field}=INVALID")
                    continue
                distance = abs(price - last)
                if distance > threshold:
                    bad_fields.append(
                        f"{field}={price:.2f} (off by {distance:.2f}, "
                        f"Last={last:.2f}, max allowed ±{threshold:.2f})"
                    )

        if bad_fields:
            detail = "; ".join(bad_fields)
            warnings.append(
                f"🚫 {asset_name} DISCARDED — hallucinated prices: {detail}"
            )
        else:
            valid.append(idea)

    return valid, warnings


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
    index=0,
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

# Only show periods that Yahoo Finance supports for the selected interval
_VALID_PERIODS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "5d", "10d", "15d", "1mo"],
}
_period_options = _VALID_PERIODS.get(interval, ["5d", "10d", "15d", "1mo", "3mo"])
_period_default = min(2, len(_period_options) - 1)
period = st.sidebar.selectbox("Data Period", _period_options, index=_period_default)

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
# Session time guard — no trading outside the window
# ---------------------------------------------------------------------------
now_est = datetime.now(tz=_EST)
current_hour = now_est.hour
session_active = 3 <= current_hour < 12
session_warning = 10 <= current_hour < 12  # wind-down period

st.sidebar.divider()
if session_active and not session_warning:
    st.sidebar.success(f"Session OPEN — {now_est.strftime('%H:%M')} EST")
elif session_warning:
    st.sidebar.warning(
        f"WIND DOWN — {now_est.strftime('%H:%M')} EST — close positions by noon"
    )
else:
    st.sidebar.error(
        f"Session CLOSED — {now_est.strftime('%H:%M')} EST — no new trades"
    )

# ---------------------------------------------------------------------------
# Daily P&L guard (shown in sidebar)
# ---------------------------------------------------------------------------
today_pnl = get_today_pnl(account_size)
open_trades_df = get_open_trades(account_size)

st.sidebar.metric("Today's Realised P&L", f"${today_pnl:,.0f}")
st.sidebar.metric("Open Trades", len(open_trades_df))

if today_pnl <= acct["hard_stop"]:
    st.sidebar.error("HARD STOP HIT — NO MORE TRADES")
elif today_pnl <= acct["soft_stop"]:
    st.sidebar.warning("Soft stop reached — tighten up!")
elif today_pnl >= abs(acct["soft_stop"]):
    st.sidebar.success("Great day! Consider locking profits.")

# Warn about open positions outside session
if not session_active and len(open_trades_df) > 0:
    st.sidebar.error(
        f"FLATTEN NOW — {len(open_trades_df)} open position(s) outside session hours!"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(
    tab_brief,
    tab_trades,
    tab_charts,
    tab_signals,
    tab_journal,
    tab_backtest,
    tab_engine,
) = st.tabs(
    [
        "🌅 Morning Brief",
        "📊 Active Trades",
        "📈 Charts",
        "🔔 Signals",
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

    # Pre-Market Composite Scorer
    st.divider()
    st.subheader("1b · Pre-Market Composite Score")
    st.caption(
        "Weighted composite score (0–100) based on: "
        "Normalized ATR (30%), Relative Volume (25%), Overnight Gap (15%), "
        "Economic Catalyst (20%), Momentum (10%)"
    )

    # Event selector — let the user flag today's active events
    available_events = sorted(EVENT_CATALOG.keys())
    active_events = st.multiselect(
        "Today's Economic Events",
        available_events,
        default=[],
        key="active_events",
        help="Select events scheduled for today to boost catalyst scores for affected instruments",
    )

    if data:
        # Build daily data dict for the scorer
        daily_data_dict = {}
        for name in selected_assets:
            ticker = ASSETS[name]
            daily_df = get_daily(ticker)
            if daily_df is not None and not daily_df.empty:
                daily_data_dict[name] = daily_df

        scorer_results = score_instruments(
            intraday_data=data,
            daily_data=daily_data_dict,
            active_events=active_events if active_events else None,
        )

        if scorer_results:
            # Traffic-light table
            scorer_df = scorer_to_dataframe(scorer_results)
            st.dataframe(
                scorer_df.style.background_gradient(
                    subset=["Score", "NATR", "RVOL", "Catalyst", "Momentum"],
                    cmap="RdYlGn",
                ),
                width="stretch",
                hide_index=True,
            )

            # Focus recommendation
            scorer_obj = PreMarketScorer()
            focus_assets = scorer_obj.get_focus_assets(scorer_results, max_focus=3)
            if focus_assets:
                focus_text = ", ".join(f"**{a}**" for a in focus_assets)
                st.success(f"🎯 Today's Focus: {focus_text}")
            else:
                st.info(
                    "No instruments meet the minimum score threshold (40) for focus today."
                )

            # Expandable detail cards
            with st.expander("📊 Score Breakdown Details"):
                for r in scorer_results:
                    emoji = r["signal_emoji"]
                    asset = r["asset"]
                    score = r["composite_score"]
                    st.markdown(f"**{emoji} {asset}** — {score:.0f}/100")
                    detail_cols = st.columns(5)
                    with detail_cols[0]:
                        st.metric(
                            "NATR",
                            f"{r['natr_score']:.0f}",
                            help=f"Ratio: {r['natr_detail']['ratio']:.2f}×",
                        )
                    with detail_cols[1]:
                        st.metric(
                            "RVOL",
                            f"{r['rvol_score']:.0f}",
                            help=f"{r['rvol_detail']['rvol']:.1f}×",
                        )
                    with detail_cols[2]:
                        st.metric(
                            "Gap",
                            f"{r['gap_score']:.0f}",
                            help=f"{r['gap_detail']['gap_pct']:.2f}%",
                        )
                    with detail_cols[3]:
                        events_str = (
                            ", ".join(r["catalyst_detail"]["matching_events"]) or "None"
                        )
                        st.metric(
                            "Catalyst", f"{r['catalyst_score']:.0f}", help=events_str
                        )
                    with detail_cols[4]:
                        st.metric(
                            "Momentum",
                            f"{r['momentum_score']:.0f}",
                            help=r["momentum_detail"]["direction"],
                        )
    else:
        st.info("Load market data to compute pre-market scores.")

    st.divider()

    # Step 2: AI Analysis
    st.subheader("2 · Grok AI Analysis")
    if st.button("Generate Morning Game Plan", type="primary", key="run_brief"):
        scan_text = (
            df_scan.to_string(index=False) if not df_scan.empty else "No scanner data"
        )

        # Build contract specs text so the LLM knows tick sizes and price scales
        specs_parts = []
        for asset_name, spec in CONTRACT_SPECS.items():
            data_ticker = spec.get("data_ticker", spec["ticker"])
            # Include the current price from scanner for explicit anchoring
            scan_price = "N/A"
            if not df_scan.empty:
                match = df_scan.loc[df_scan["Asset"] == asset_name, "Last"]
                if not match.empty:
                    scan_price = str(match.iloc[0])
            specs_parts.append(
                f"  {asset_name} ({data_ticker}): "
                f"current_price={scan_price}, "
                f"tick_size={spec['tick']}, "
                f"point_value=USD {spec['point']}/point, "
                f"margin=USD {spec['margin']:,}"
            )
        specs_text = "\n".join(specs_parts)

        # Correlation matrix
        corr_text = "Not enough data"
        if len(data) >= 2:
            closes = pd.DataFrame({n: d["Close"] for n, d in data.items()})
            returns = closes.pct_change(fill_method=None).dropna()
            corr = returns.corr().round(2)
            corr_text = corr.to_string()

        # Optimisation results (with walk-forward and confidence data)
        opt_text_parts = []
        for name in selected_assets:
            ticker = ASSETS[name]
            opt = get_cached_optimization(ticker, interval, period)
            if opt:
                strat_label = opt.get("strategy_label", opt.get("strategy", "?"))
                confidence = opt.get("confidence", "?")
                regime = opt.get("regime", "?")
                wf = "yes" if opt.get("walk_forward") else "no"
                regime_method = opt.get("regime_method", "atr")
                pos_mult = opt.get("position_multiplier", 1.0)
                opt_text_parts.append(
                    f"  {name}: strategy={strat_label}, return={opt['return_pct']}%, "
                    f"sharpe={opt.get('sharpe', '?')}, win_rate={opt.get('win_rate', '?')}%, "
                    f"confidence={confidence}, regime={regime} ({regime_method}), "
                    f"pos_multiplier={pos_mult:.2f}x, walk_forward={wf}"
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
- No new entries after 10 AM EST, only manage existing positions
- Daily P&L limits: soft stop -USD {abs(acct["soft_stop"]):,}, hard stop -USD {abs(acct["hard_stop"]):,}
- Focus on correlations (Gold/Silver/Copper/Oil/ES/NQ)
- All positions MUST be flat before end of day — no overnight holds

FORMATTING RULE: NEVER use bare $ signs for dollar amounts — always write "USD" instead.
Do not use LaTeX or math notation. Use plain text only.

CRITICAL — CONTRACT SPECIFICATIONS (use these for correct price scales):
{specs_text}

PRICE ANCHORING RULES — READ CAREFULLY:
- The "Last" column in scanner data is the CURRENT PRICE for each asset. ALL entry zones, SL, and TP MUST be based on this price.
- Gold (GC=F) trades around USD 2,500-3,500 per oz. Silver (SI=F) around USD 28-40 per oz. Copper (HG=F) around USD 4-6 per lb.
- Crude Oil (CL=F) trades around USD 55-85 per barrel. S&P (ES=F) around USD 5,000-6,500. Nasdaq (NQ=F) around USD 18,000-22,000.
- Entry zones must be within 1-2x ATR of the "Last" price. SL within 1-2x ATR. TP within 2-4x ATR.
- Use the "Pivot", "S1", "S2", "R1", "R2" columns as key support/resistance levels for entry/SL/TP zones.
- NEVER confuse "Dist to Pivot", "ATR", or "% from VWAP" with the actual price. These are supplementary metrics only.
- Contracts formula: contracts = floor(USD {risk_dollars:,} / (SL_distance × point_value)). Clamp to max {max_contracts}.

Current time: {datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M")} EST
Account: USD {account_size:,}
Session status: {"ACTIVE — primary entry window" if 3 <= current_hour < 10 else "WIND DOWN — manage only" if 10 <= current_hour < 12 else "CLOSED — no trading"}

Scanner data (columns: Asset, % Overnight, Last=CURRENT PRICE, VWAP, % from VWAP, Pivot, S2, S1, R1, R2, Dist to Pivot, ATR, Volume):
{scan_text}

Correlation matrix (5-min returns):
{corr_text}

Optimized strategy results (walk-forward validated):
{opt_text}

Backtest results (last {period}, session-hours only):
{bt_text}

Give me a complete morning game plan:
1. Overall market bias & rank 1-5 best assets to focus on today
2. For each focus asset: direction, entry zone, SL, TP, contracts, RR, rationale
   - The entry zone MUST be near the "Last" price from the scanner (within 1-2x ATR)
   - Use Pivot/S1/S2/R1/R2 levels for support and resistance zones
   - Calculate contracts using: floor(USD {risk_dollars:,} / (SL_distance × point_value))
   - Pay attention to which strategies the optimizer chose and their confidence levels
   - Prioritise assets where the optimizer has "high" confidence
3. Key correlations to watch and how to use them
4. News/events awareness
5. Risk reminders (including session time rules)

After your analysis, provide your trade ideas as a JSON code block using this EXACT format
(entry_low/entry_high define the limit order zone — must be near the current "Last" price):
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
                    raw_ideas = _parse_trade_ideas(result)
                    validated, val_warnings = _validate_trade_ideas(raw_ideas, df_scan)
                    st.session_state["morning_trade_ideas"] = validated
                    if val_warnings:
                        st.session_state["trade_idea_warnings"] = val_warnings
                    else:
                        st.session_state.pop("trade_idea_warnings", None)
            except Exception as e:
                st.error(f"API error: {e}")

    # Display analysis
    if "morning_analysis" in st.session_state:
        analysis = st.session_state["morning_analysis"]
        # Display narrative (strip the JSON block for cleaner reading)
        narrative = re.sub(r"```json\s*\[.*?\]\s*```", "", analysis, flags=re.DOTALL)
        narrative = re.sub(r"```\s*\[.*?\]\s*```", "", narrative, flags=re.DOTALL)
        st.markdown(_escape_dollars(narrative.strip()))

        # Show validation warnings if any trade ideas were discarded
        if "trade_idea_warnings" in st.session_state:
            with st.expander("⚠️ Price Validation Warnings", expanded=True):
                for w in st.session_state["trade_idea_warnings"]:
                    st.warning(w)
                st.caption(
                    "Trade ideas with prices far from the actual 'Last' price "
                    "(> 5× ATR) were automatically discarded to prevent "
                    "hallucinated entries."
                )

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

    # --- Volume Profile Section ---
    st.divider()
    st.subheader("Volume Profile")
    st.caption(
        "Institutional-grade support/resistance via Point of Control (POC), "
        "Value Area High (VAH), and Value Area Low (VAL). "
        "Price reverts to POC ~75% of the time in ranging markets."
    )

    vp_asset = st.selectbox("Volume Profile Asset", selected_assets, key="vp_asset")

    if vp_asset in data and not data[vp_asset].empty:
        vp_df = data[vp_asset]
        vp_col1, vp_col2 = st.columns([3, 1])

        with vp_col1:
            # Compute volume profile for the full dataset
            profile = compute_volume_profile(vp_df, n_bins=50)

            if profile["poc"] > 0:
                # Build horizontal bar chart for the volume profile
                vp_display = profile_to_dataframe(profile)

                # Create a combined chart: candlestick + volume profile bars
                vp_fig = go.Figure()

                # Add candlestick
                vp_fig.add_trace(
                    go.Candlestick(
                        x=vp_df.index,
                        open=vp_df["Open"],
                        high=vp_df["High"],
                        low=vp_df["Low"],
                        close=vp_df["Close"],
                        name="Price",
                    )
                )

                # Add POC, VAH, VAL lines
                vp_fig.add_hline(
                    y=profile["poc"],
                    line_dash="solid",
                    line_color="#FFD700",
                    line_width=2,
                    annotation_text=f"POC {profile['poc']:.2f}",
                    annotation_position="bottom right",
                )
                vp_fig.add_hline(
                    y=profile["vah"],
                    line_dash="dash",
                    line_color="#FF6B6B",
                    line_width=1.5,
                    annotation_text=f"VAH {profile['vah']:.2f}",
                    annotation_position="bottom right",
                )
                vp_fig.add_hline(
                    y=profile["val"],
                    line_dash="dash",
                    line_color="#00D4AA",
                    line_width=1.5,
                    annotation_text=f"VAL {profile['val']:.2f}",
                    annotation_position="bottom right",
                )

                # Add HVN markers
                for hvn_price in profile.get("hvn", [])[:5]:
                    vp_fig.add_hline(
                        y=hvn_price,
                        line_dash="dot",
                        line_color="rgba(255, 215, 0, 0.3)",
                        line_width=1,
                    )

                vp_fig.update_layout(
                    height=500,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    title=f"{vp_asset} — Volume Profile",
                )
                st.plotly_chart(vp_fig, width="stretch")

                # Volume profile horizontal histogram
                if not vp_display.empty:
                    vp_hist_fig = go.Figure()
                    colors = [
                        "#FFD700"
                        if bool(row["IsPOC"])
                        else "#00D4AA"
                        if bool(row["InValueArea"])
                        else "#555555"
                        for _, row in vp_display.iterrows()
                    ]
                    vp_hist_fig.add_trace(
                        go.Bar(
                            y=vp_display["Price"],
                            x=vp_display["Volume"],
                            orientation="h",
                            marker_color=colors,
                            name="Volume",
                        )
                    )
                    vp_hist_fig.update_layout(
                        height=400,
                        template="plotly_dark",
                        yaxis_title="Price",
                        xaxis_title="Volume",
                        title="Volume Distribution by Price",
                    )
                    st.plotly_chart(vp_hist_fig, width="stretch")
            else:
                st.info("Insufficient data for volume profile calculation.")

        with vp_col2:
            st.markdown("**Key Levels**")
            if profile["poc"] > 0:
                st.metric("POC", f"{profile['poc']:.2f}")
                st.metric("VAH", f"{profile['vah']:.2f}")
                st.metric("VAL", f"{profile['val']:.2f}")

                current_vp_price = float(vp_df["Close"].iloc[-1])
                dist_to_poc = current_vp_price - profile["poc"]
                st.metric(
                    "Dist to POC",
                    f"{dist_to_poc:+.2f}",
                    delta=f"{dist_to_poc:+.2f}",
                    delta_color="normal" if abs(dist_to_poc) < 10 else "inverse",
                )

                st.markdown("**HVN** (Support/Resistance)")
                for hvn_price in profile.get("hvn", [])[:5]:
                    st.caption(f"  {hvn_price:.2f}")

                st.markdown("**LVN** (Price Gaps)")
                for lvn_price in profile.get("lvn", [])[:5]:
                    st.caption(f"  {lvn_price:.2f}")

            # Naked POC tracking
            st.markdown("---")
            st.markdown("**Naked POCs**")
            st.caption("Unfilled POCs from prior sessions — price magnets")
            session_profiles = compute_session_profiles(vp_df, n_bins=50)
            if session_profiles and profile["poc"] > 0:
                naked = find_naked_pocs(
                    session_profiles,
                    current_price=float(vp_df["Close"].iloc[-1]),
                    max_distance_points=200.0,
                )
                if naked:
                    for np_item in naked[:5]:
                        direction_emoji = (
                            "⬆️" if np_item["direction"] == "below" else "⬇️"
                        )
                        st.caption(
                            f"{direction_emoji} {np_item['poc']:.2f} "
                            f"({np_item['distance']:+.2f} pts, {np_item['date']})"
                        )
                else:
                    st.caption("No naked POCs within range")
            else:
                st.caption("Need multi-session data")
    else:
        st.info("Select an asset with data to view volume profile.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — SIGNALS (CVD, Confluence, Alerts)
# ═══════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.header("Signals & Confluence")

    # ── CVD (Cumulative Volume Delta) ──────────────────────────────────────
    st.subheader("📊 Cumulative Volume Delta (CVD)")
    st.caption(
        "Approximated from OHLCV data. Buy volume = V × (C−L)/(H−L). "
        "Divergences between price and CVD reveal hidden accumulation/distribution."
    )

    cvd_asset = st.selectbox("CVD Asset", selected_assets, key="cvd_asset")

    if cvd_asset in data and not data[cvd_asset].empty:
        cvd_df = data[cvd_asset]
        cvd_result = compute_cvd(cvd_df)
        summary = cvd_summary(cvd_result) if not cvd_result.empty else {}

        # Key metrics
        cvd_m1, cvd_m2, cvd_m3, cvd_m4 = st.columns(4)
        bias_emoji = summary.get("bias_emoji", "⚪")
        cvd_m1.metric(
            "CVD Bias", f"{bias_emoji} {summary.get('bias', 'neutral').title()}"
        )
        cvd_m2.metric("CVD Current", f"{summary.get('cvd_current', 0):,.0f}")
        cvd_m3.metric("Delta (Latest Bar)", f"{summary.get('delta_current', 0):,.0f}")
        cvd_m4.metric(
            "CVD Slope",
            f"{summary.get('cvd_slope', 0):+.3f}",
            delta=f"{'Buying' if summary.get('cvd_slope', 0) > 0 else 'Selling'} pressure",
            delta_color="normal" if summary.get("cvd_slope", 0) > 0 else "inverse",
        )

        # CVD chart
        cvd_fig = go.Figure()
        cvd_fig.add_trace(
            go.Scatter(
                x=cvd_result.index,
                y=cvd_result["cvd"],
                mode="lines",
                name="CVD",
                line=dict(color="#00D4AA", width=2),
            )
        )
        if "cvd_ema" in cvd_result.columns:
            cvd_fig.add_trace(
                go.Scatter(
                    x=cvd_result.index,
                    y=cvd_result["cvd_ema"],
                    mode="lines",
                    name="CVD EMA",
                    line=dict(color="#FFD700", width=1, dash="dash"),
                )
            )
        cvd_fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        cvd_fig.update_layout(
            height=350,
            template="plotly_dark",
            yaxis_title="Cumulative Delta",
            title=f"{cvd_asset} — Cumulative Volume Delta",
        )
        st.plotly_chart(cvd_fig, use_container_width=True)

        # Divergences
        divergences = detect_cvd_divergences(cvd_df)
        if divergences:
            st.markdown(f"**{len(divergences)} CVD Divergence(s) Detected**")
            for div in divergences[-5:]:
                div_emoji = "🟢" if div["type"] == "bullish" else "🔴"
                st.caption(
                    f"{div_emoji} **{div['type'].title()}** divergence "
                    f"at bar {div.get('bar_index', '?')} — "
                    f"Price {'lower low' if div['type'] == 'bullish' else 'higher high'}, "
                    f"CVD {'higher low' if div['type'] == 'bullish' else 'lower high'}"
                )
        else:
            st.caption("No CVD divergences detected in current data.")

        # Absorption candles
        absorption_series = pd.Series(detect_absorption_candles(cvd_result))
        absorption_nonzero = pd.Series(absorption_series[absorption_series != 0])
        n_absorptions = len(absorption_nonzero)
        if n_absorptions > 0:
            st.markdown(f"**{n_absorptions} Absorption Candle(s)**")
            tail = absorption_nonzero.tail(5)
            for i in range(len(tail)):
                idx_val = tail.index[i]
                sig_val = int(tail.iloc[i])
                ab_emoji = "🟢" if sig_val > 0 else "🔴"
                ab_type = "Bullish Absorption" if sig_val > 0 else "Bearish Absorption"
                st.caption(f"{ab_emoji} {ab_type} at {idx_val}")
        else:
            st.caption("No absorption candles detected.")
    else:
        st.info("Select an asset with data to view CVD analysis.")

    st.divider()

    # ── Multi-Timeframe Confluence ─────────────────────────────────────────
    st.subheader("🎯 Multi-Timeframe Confluence")
    st.caption(
        "Three-layer filter: Higher Timeframe (bias) → Setup (pattern) → Entry (timing). "
        "Score 3/3 = high-conviction trade. Only trade on full alignment."
    )

    conf_asset = st.selectbox("Confluence Asset", selected_assets, key="conf_asset")

    if conf_asset in data and not data[conf_asset].empty:
        # Get recommended timeframes for this instrument
        recommended_tfs = get_recommended_timeframes(conf_asset)
        st.caption(
            f"Recommended timeframes for {conf_asset}: "
            f"HTF={recommended_tfs[0]}, "
            f"Setup={recommended_tfs[1]}, "
            f"Entry={recommended_tfs[2]}"
        )

        # Use whatever data we have as the entry timeframe
        entry_df = data[conf_asset]

        # Check confluence
        try:
            conf_result = check_confluence(
                entry_df=entry_df,
                setup_df=entry_df,  # Same data for now (single-TF approximation)
                htf_df=entry_df,
            )

            # Display confluence score
            conf_score = conf_result.get("score", 0)
            bias = conf_result.get("direction", "neutral")

            cc1, cc2, cc3, cc4 = st.columns(4)
            score_emoji = "🟢" if conf_score >= 3 else "🟡" if conf_score >= 2 else "🔴"
            cc1.metric("Confluence Score", f"{score_emoji} {conf_score}/3")
            cc2.metric("Bias", bias.upper())

            htf_data = conf_result.get("htf", {})
            setup_data = conf_result.get("setup", {})
            htf_dir = (
                htf_data.get("direction", "—") if isinstance(htf_data, dict) else "—"
            )
            setup_dir = (
                setup_data.get("direction", "—")
                if isinstance(setup_data, dict)
                else "—"
            )
            cc3.metric(
                "HTF",
                f"{'✅' if htf_dir in ('bullish', 'bearish') else '❌'} {htf_dir}",
            )
            cc4.metric(
                "Setup",
                f"{'✅' if setup_dir in ('bullish', 'bearish') else '❌'} {setup_dir}",
            )

            # Detailed breakdown
            with st.expander("Confluence Breakdown"):
                st.json(
                    {
                        k: v
                        for k, v in conf_result.items()
                        if not isinstance(v, (pd.DataFrame, pd.Series))
                    }
                )

            # Confluence table for all assets
            st.markdown("**All-Asset Confluence Summary**")
            conf_rows = []
            for asset_name in selected_assets:
                if asset_name in data and not data[asset_name].empty:
                    try:
                        asset_conf = check_confluence(
                            entry_df=data[asset_name],
                            setup_df=data[asset_name],
                            htf_df=data[asset_name],
                        )
                        a_score = asset_conf.get("score", 0)
                        a_emoji = (
                            "🟢" if a_score >= 3 else "🟡" if a_score >= 2 else "🔴"
                        )
                        a_htf = asset_conf.get("htf", {})
                        a_setup = asset_conf.get("setup", {})
                        a_entry = asset_conf.get("entry", {})
                        conf_rows.append(
                            {
                                "Asset": asset_name,
                                "Score": f"{a_emoji} {a_score}/3",
                                "Bias": asset_conf.get("direction", "neutral").upper(),
                                "HTF": a_htf.get("direction", "—")
                                if isinstance(a_htf, dict)
                                else "—",
                                "Setup": a_setup.get("direction", "—")
                                if isinstance(a_setup, dict)
                                else "—",
                                "Entry": a_entry.get("direction", "—")
                                if isinstance(a_entry, dict)
                                else "—",
                            }
                        )
                    except Exception:
                        conf_rows.append(
                            {
                                "Asset": asset_name,
                                "Score": "—",
                                "Bias": "—",
                                "HTF": "—",
                                "Setup": "—",
                                "Entry": "—",
                            }
                        )
            if conf_rows:
                st.dataframe(pd.DataFrame(conf_rows), width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Confluence analysis unavailable: {e}")
    else:
        st.info("Select an asset with data to evaluate confluence.")

    st.divider()

    # ── Alert Dispatcher Status ────────────────────────────────────────────
    st.subheader("🔔 Alerts & Notifications")
    st.caption(
        "Multi-channel alert dispatcher (Slack, Discord, Telegram). "
        "Configure via environment variables. Redis-backed dedup with 5-min cooldown."
    )

    dispatcher = get_dispatcher()
    channels = dispatcher.channels_configured

    al1, al2, al3 = st.columns(3)
    al1.metric(
        "Slack",
        "✅ Connected" if "Slack" in channels else "❌ Not configured",
    )
    al2.metric(
        "Discord",
        "✅ Connected" if "Discord" in channels else "❌ Not configured",
    )
    al3.metric(
        "Telegram",
        "✅ Connected" if "Telegram" in channels else "❌ Not configured",
    )

    if not dispatcher.has_channels:
        st.info(
            "No alert channels configured. Set environment variables to enable:\n"
            "- `SLACK_WEBHOOK_URL` for Slack\n"
            "- `DISCORD_WEBHOOK_URL` for Discord\n"
            "- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` for Telegram"
        )
    else:
        stats = dispatcher.get_stats()
        st.caption(
            f"Alerts sent: {stats.get('total_sent', 0)} · "
            f"Deduplicated: {stats.get('deduplicated', 0)}"
        )

    # Manual test alert
    if dispatcher.has_channels:
        if st.button("Send Test Alert", key="test_alert"):
            try:
                send_signal(
                    signal_key="test_alert",
                    title="🔔 Test Alert",
                    message="This is a test alert from the Futures Dashboard.",
                    asset="TEST",
                    strategy="Test Alert",
                    direction="LONG",
                )
                st.success("Test alert sent!")
            except Exception as e:
                st.error(f"Alert failed: {e}")

    # Recent alerts
    recent = dispatcher.get_recent_alerts()
    if recent:
        with st.expander(f"Recent Alerts ({len(recent)})"):
            for alert in recent[-10:]:
                st.caption(
                    f"[{alert.get('timestamp', '?')}] "
                    f"{alert.get('channel', '?')}: {alert.get('message', '?')[:100]}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — JOURNAL
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
            datetime.now(tz=_EST) - timedelta(days=datetime.now(tz=_EST).weekday())
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
# TAB 6 — BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("Backtester")

    # Show latest engine backtest results
    bt_results = engine.get_backtest_results()
    if bt_results:
        st.subheader("Auto-Backtest Results (Session Hours, Walk-Forward Validated)")
        df_bt_all = pd.DataFrame(bt_results)
        display_cols_bt = [
            "Asset",
            "Strategy",
            "Confidence",
            "Regime",
            "Params",
            "Return %",
            "Buy & Hold %",
            "# Trades",
            "Win Rate %",
            "Max DD %",
            "Sharpe",
            "Sortino",
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
                name="Optimized Strategy",
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
        opt_confidence = opt_params.get("confidence", "?")
        opt_regime = opt_params.get("regime", "?")
        opt_detail_parts = []
        if "params" in opt_params:
            for pk, pv in opt_params["params"].items():
                opt_detail_parts.append(f"{pk}={pv}")
        else:
            opt_detail_parts.append(f"n1={opt_params.get('n1', '?')}")
            opt_detail_parts.append(f"n2={opt_params.get('n2', '?')}")
            opt_detail_parts.append(f"size={opt_params.get('size', '?')}")
        opt_pos_mult = opt_params.get("position_multiplier", 1.0)
        st.success(
            f"**Best strategy: {opt_strat}** — Sharpe={opt_sharpe}, Return={opt_ret}%, "
            f"Confidence={opt_confidence}, Regime={opt_regime} (sizing: {opt_pos_mult:.2f}x) · "
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

    elif bt_strat_key == "VWAP":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_trend = st.number_input(
                "Trend EMA",
                min_value=20,
                max_value=200,
                value=int(opt_p.get("trend_period", 50)),
                key="bt_trend",
            )
        with bc2:
            bt_size = st.number_input(
                "Size",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                value=float(opt_p.get("trade_size", 0.10)),
                key="bt_size",
            )
        with bc3:
            bt_vol_mult = st.number_input(
                "Vol ×",
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                value=float(opt_p.get("vol_mult", 1.0)),
                key="bt_vol_mult",
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
                value=float(opt_p.get("atr_tp_mult", 2.0)),
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
            "trend_period": bt_trend,
            "atr_period": bt_atr_p,
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "vol_sma_period": bt_vol_sma,
            "vol_mult": bt_vol_mult,
            "trade_size": bt_size,
        }

    elif bt_strat_key == "ORB":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_orb_bars = st.number_input(
                "ORB Bars",
                min_value=2,
                max_value=24,
                value=int(opt_p.get("orb_bars", 6)),
                key="bt_orb_bars",
                help="Number of bars forming the opening range (e.g. 6 × 5min = 30 min)",
            )
        with bc2:
            bt_vol_mult = st.number_input(
                "Vol ×",
                min_value=0.5,
                max_value=3.0,
                step=0.1,
                value=float(opt_p.get("vol_mult", 1.0)),
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
                value=float(opt_p.get("atr_tp_mult", 2.5)),
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
            "orb_bars": bt_orb_bars,
            "atr_period": bt_atr_p,
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "vol_sma_period": bt_vol_sma,
            "vol_mult": bt_vol_mult,
            "trade_size": bt_size,
        }

    elif bt_strat_key == "MACD":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_macd_fast = st.number_input(
                "MACD Fast",
                min_value=4,
                max_value=20,
                value=int(opt_p.get("macd_fast", 12)),
                key="bt_macd_fast",
            )
        with bc2:
            bt_macd_slow = st.number_input(
                "MACD Slow",
                min_value=15,
                max_value=50,
                value=int(opt_p.get("macd_slow", 26)),
                key="bt_macd_slow",
            )
        with bc3:
            bt_macd_sig = st.number_input(
                "Signal",
                min_value=3,
                max_value=20,
                value=int(opt_p.get("macd_signal", 9)),
                key="bt_macd_sig",
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
                value=float(opt_p.get("atr_tp_mult", 2.5)),
                key="bt_tp_m",
            )
        manual_params = {
            "macd_fast": bt_macd_fast,
            "macd_slow": bt_macd_slow,
            "macd_signal": bt_macd_sig,
            "trend_period": bt_trend,
            "atr_period": bt_atr_p,
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "trade_size": bt_size,
        }

    elif bt_strat_key == "PullbackEMA":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_ema_fast = st.number_input(
                "Fast EMA",
                min_value=3,
                max_value=20,
                value=int(opt_p.get("ema_fast", 9)),
                key="bt_ema_fast",
            )
        with bc2:
            bt_ema_mid = st.number_input(
                "Mid EMA",
                min_value=10,
                max_value=40,
                value=int(opt_p.get("ema_mid", 21)),
                key="bt_ema_mid",
            )
        with bc3:
            bt_ema_slow = st.number_input(
                "Slow EMA",
                min_value=30,
                max_value=100,
                value=int(opt_p.get("ema_slow", 50)),
                key="bt_ema_slow",
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
            bt_rsi_p = st.number_input(
                "RSI Period",
                min_value=5,
                max_value=30,
                value=int(opt_p.get("rsi_period", 14)),
                key="bt_rsi_p",
            )
        with bc6:
            bt_rsi_lim = st.number_input(
                "RSI Limit",
                min_value=20,
                max_value=70,
                value=int(opt_p.get("rsi_limit", 45)),
                key="bt_rsi_lim",
                help="RSI must be below this for longs (above 100−this for shorts)",
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
                value=float(opt_p.get("atr_tp_mult", 2.5)),
                key="bt_tp_m",
            )
        manual_params = {
            "ema_fast": bt_ema_fast,
            "ema_mid": bt_ema_mid,
            "ema_slow": bt_ema_slow,
            "rsi_period": bt_rsi_p,
            "rsi_limit": bt_rsi_lim,
            "atr_period": int(opt_p.get("atr_period", 14)),
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "trade_size": bt_size,
        }

    elif bt_strat_key == "EventReaction":
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bt_vol_spike = st.number_input(
                "Vol Spike ×",
                min_value=1.0,
                max_value=5.0,
                step=0.25,
                value=float(opt_p.get("vol_spike_mult", 2.0)),
                key="bt_vol_spike",
                help="Volume must exceed average × this to detect an event bar",
            )
        with bc2:
            bt_move_atr = st.number_input(
                "Move ATR ×",
                min_value=0.25,
                max_value=3.0,
                step=0.25,
                value=float(opt_p.get("move_atr_mult", 1.0)),
                key="bt_move_atr",
                help="Price move must exceed ATR × this to qualify as event bar",
            )
        with bc3:
            bt_wait = st.number_input(
                "Wait Bars",
                min_value=1,
                max_value=10,
                value=int(opt_p.get("wait_bars", 2)),
                key="bt_wait",
                help="Bars to wait after spike before entering",
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
            bt_vol_conf = st.number_input(
                "Vol Confirm ×",
                min_value=0.5,
                max_value=4.0,
                step=0.25,
                value=float(opt_p.get("vol_confirm", 1.5)),
                key="bt_vol_conf",
                help="Post-wait volume must exceed average × this to confirm entry",
            )
        with bc6:
            bt_vol_sma = st.number_input(
                "Vol SMA",
                min_value=5,
                max_value=50,
                value=int(opt_p.get("vol_sma_period", 20)),
                key="bt_vol_sma",
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
                value=float(opt_p.get("atr_tp_mult", 3.0)),
                key="bt_tp_m",
            )
        manual_params = {
            "vol_spike_mult": bt_vol_spike,
            "move_atr_mult": bt_move_atr,
            "wait_bars": bt_wait,
            "vol_confirm": bt_vol_conf,
            "vol_sma_period": bt_vol_sma,
            "atr_period": int(opt_p.get("atr_period", 14)),
            "atr_sl_mult": bt_sl_m,
            "atr_tp_mult": bt_tp_m,
            "trade_size": bt_size,
        }

    else:
        # Fallback for any unhandled strategy key (VolumeProfile, PlainEMA, etc.)
        manual_params = {"trade_size": 0.10}

    if asset_bt in data and not data[asset_bt].empty:
        df_bt = filter_session_hours(data[asset_bt].copy())

        # Build strategy class with the selected params
        configured_cls = make_strategy(bt_strat_key, manual_params)
        from models import CONTRACT_MODE

        bt_comm = slippage_commission_rate(asset_bt, CONTRACT_MODE)
        bt = Backtest(
            df_bt,
            configured_cls,
            cash=account_size,
            commission=bt_comm,
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
            with st.spinner(
                f"Optimizing {asset_bt} (6 strategies × 30 trials, walk-forward)..."
            ):
                result = run_optimization(ticker, interval, period, account_size)
                if result:
                    strat_name = result.get("strategy_label", "?")
                    confidence = result.get("confidence", "?")
                    st.success(
                        f"Done! Best: **{strat_name}** — "
                        f"Sharpe={result.get('sharpe', '?')}, "
                        f"Return={result['return_pct']}%, "
                        f"Confidence={confidence}"
                    )
                    st.rerun()
        # --- Monte Carlo Simulation ---
        st.divider()
        st.subheader("Monte Carlo Robustness Test")
        st.caption(
            "Trade-level bootstrap Monte Carlo: resample trades with replacement "
            "10,000 times to build equity confidence cones and estimate realistic "
            "drawdown distributions."
        )

        n_trades_bt = int(stats["# Trades"])
        if n_trades_bt >= 3:
            # Extract per-trade P&L from backtest results
            trades_series = stats.get("_trades")
            trade_pnls_list = []

            if (
                trades_series is not None
                and hasattr(trades_series, "__len__")
                and len(trades_series) > 0
            ):
                try:
                    trade_pnls_list = [
                        float(t.PnL) for t in trades_series if hasattr(t, "PnL")
                    ]
                except Exception:
                    pass

            # Fallback: estimate from aggregate stats if trade-level data unavailable
            if not trade_pnls_list:
                avg_win = (
                    float(stats.get("Avg. Trade [%]", 0) or 0) * account_size / 100
                )
                wr = (
                    float(stats.get("Win Rate [%]", 0) or 0) / 100
                    if n_trades_bt > 0
                    else 0.5
                )
                # Synthesize approximate trades
                import random

                random.seed(42)
                for _ in range(n_trades_bt):
                    if random.random() < wr:
                        trade_pnls_list.append(abs(avg_win) * random.uniform(0.5, 2.0))
                    else:
                        trade_pnls_list.append(-abs(avg_win) * random.uniform(0.5, 2.0))

            mc_sims = st.slider(
                "Simulations",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                key="mc_sims",
            )

            if st.button("Run Monte Carlo", key="run_mc", type="primary"):
                with st.spinner(f"Running {mc_sims:,} Monte Carlo simulations..."):
                    mc_result = run_monte_carlo(
                        trade_pnls_list,
                        n_simulations=mc_sims,
                        initial_equity=float(account_size),
                    )
                    cones = compute_confidence_cones(mc_result)
                    st.session_state["mc_result"] = mc_result
                    st.session_state["mc_cones"] = cones

            if "mc_result" in st.session_state and "mc_cones" in st.session_state:
                mc_result = st.session_state["mc_result"]
                cones = st.session_state["mc_cones"]
                summary = cones["summary"]

                # Key metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(
                    "Prob. Profitable",
                    f"{summary['prob_profitable']:.1f}%",
                )
                mc2.metric(
                    "Median Return",
                    f"{summary['median_return_pct']:+.2f}%",
                )
                mc3.metric(
                    "Worst DD (95th pct)",
                    f"${summary['worst_case_drawdown_95pct']:,.0f}",
                )
                mc4.metric(
                    "Worst Return (5th pct)",
                    f"{summary['worst_case_return_5pct']:+.2f}%",
                )

                # Equity confidence cone chart
                cone_df = cone_curves_to_dataframe(cones)
                if not cone_df.empty:
                    cone_fig = go.Figure()

                    # Shaded band: 5th to 95th percentile
                    if "P5" in cone_df.columns and "P95" in cone_df.columns:
                        cone_fig.add_trace(
                            go.Scatter(
                                x=cone_df["Trade"],
                                y=cone_df["P95"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                            )
                        )
                        cone_fig.add_trace(
                            go.Scatter(
                                x=cone_df["Trade"],
                                y=cone_df["P5"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(0, 212, 170, 0.15)",
                                name="5th–95th Percentile",
                            )
                        )

                    # Shaded band: 25th to 75th percentile
                    if "P25" in cone_df.columns and "P75" in cone_df.columns:
                        cone_fig.add_trace(
                            go.Scatter(
                                x=cone_df["Trade"],
                                y=cone_df["P75"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                            )
                        )
                        cone_fig.add_trace(
                            go.Scatter(
                                x=cone_df["Trade"],
                                y=cone_df["P25"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(0, 212, 170, 0.3)",
                                name="25th–75th Percentile",
                            )
                        )

                    # Median line
                    if "P50" in cone_df.columns:
                        cone_fig.add_trace(
                            go.Scatter(
                                x=cone_df["Trade"],
                                y=cone_df["P50"],
                                mode="lines",
                                line=dict(color="#00D4AA", width=2),
                                name="Median (50th)",
                            )
                        )

                    # Starting equity reference
                    cone_fig.add_hline(
                        y=account_size,
                        line_dash="dot",
                        line_color="white",
                        line_width=1,
                        annotation_text="Initial Equity",
                    )

                    cone_fig.update_layout(
                        height=400,
                        template="plotly_dark",
                        yaxis_title="Equity ($)",
                        xaxis_title="Trade #",
                        title="Monte Carlo Equity Confidence Cones",
                    )
                    st.plotly_chart(cone_fig, width="stretch")

                # Drawdown distribution histogram
                dd_hist_df = drawdown_distribution_to_dataframe(mc_result)
                if not dd_hist_df.empty:
                    dd_fig = go.Figure()
                    dd_fig.add_trace(
                        go.Bar(
                            x=dd_hist_df["Drawdown %"],
                            y=dd_hist_df["Count"],
                            marker_color="#FF6B6B",
                            name="Max Drawdown Distribution",
                        )
                    )
                    # Add 95th percentile line
                    dd_95 = summary["worst_case_drawdown_pct_95pct"]
                    dd_fig.add_vline(
                        x=dd_95,
                        line_dash="dash",
                        line_color="#FFD700",
                        annotation_text=f"95th pct: {dd_95:.1f}%",
                    )
                    dd_fig.update_layout(
                        height=300,
                        template="plotly_dark",
                        xaxis_title="Max Drawdown %",
                        yaxis_title="Frequency",
                        title="Max Drawdown Distribution",
                    )
                    st.plotly_chart(dd_fig, width="stretch")

                # Full stats table
                with st.expander("Full Monte Carlo Statistics"):
                    mc_stats_df = mc_results_to_dataframe(mc_result)
                    st.dataframe(mc_stats_df, width="stretch", hide_index=True)
        else:
            st.info(
                f"Need at least 3 trades for Monte Carlo (got {n_trades_bt}). "
                "Try a longer data period or different strategy."
            )

        # --- Cost Model Section ---
        st.divider()
        st.subheader("Trade Cost Analysis")
        st.caption(
            "Realistic CME futures slippage and commission model. "
            "1-tick slippage per side during RTH, time-of-day multipliers, "
            "and instrument-specific break-even calculations."
        )

        from models import CONTRACT_MODE

        cost_asset = asset_bt
        cost_model = get_cost_model(cost_asset, CONTRACT_MODE)
        costs_rth = estimate_trade_costs(cost_asset, 1, "rth", CONTRACT_MODE)
        costs_eth = estimate_trade_costs(cost_asset, 1, "eth", CONTRACT_MODE)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Tick Value", f"${cost_model['tick_value']:.2f}")
        cc2.metric("Slippage/Side", f"${cost_model['slippage_per_side']:.2f}")
        cc3.metric("RT Cost (RTH)", f"${costs_rth['total_cost']:.2f}")
        cc4.metric("Break-Even", f"{costs_rth['break_even_ticks']:.1f} ticks")

        with st.expander("Cost Model Details"):
            st.markdown(
                f"**{cost_asset}** ({cost_model['ticker']}) — {CONTRACT_MODE.upper()} contracts"
            )
            cost_cols = st.columns(2)
            with cost_cols[0]:
                st.markdown("**RTH (Regular Trading Hours)**")
                st.json(
                    {
                        "Slippage (2 sides)": f"${costs_rth['slippage_total']:.2f}",
                        "Commission + Fees": f"${costs_rth['commission_total']:.2f}",
                        "Total Cost": f"${costs_rth['total_cost']:.2f}",
                        "Break-Even Move": f"{costs_rth['break_even_move']:.4f} pts",
                        "Break-Even Ticks": f"{costs_rth['break_even_ticks']:.1f}",
                    }
                )
            with cost_cols[1]:
                st.markdown("**ETH (Overnight/Extended)**")
                st.json(
                    {
                        "Slippage (2 sides)": f"${costs_eth['slippage_total']:.2f}",
                        "Commission + Fees": f"${costs_eth['commission_total']:.2f}",
                        "Total Cost": f"${costs_eth['total_cost']:.2f}",
                        "Break-Even Move": f"{costs_eth['break_even_move']:.4f} pts",
                        "Break-Even Ticks": f"{costs_eth['break_even_ticks']:.1f}",
                    }
                )

            # Full-size vs micro comparison
            st.markdown("**Micro vs Full-Size Cost Comparison**")
            for check_size in [5, 10, 15, 20]:
                advice = should_use_full_contracts(cost_asset, check_size)
                emoji = "✅" if advice["recommend_full"] else "➖"
                st.caption(
                    f"{emoji} {check_size} micros: {advice['reason']} "
                    f"(micro=${advice['micro_cost']:.2f}"
                    + (
                        f", full=${advice['full_cost']:.2f}"
                        if advice["full_cost"]
                        else ""
                    )
                    + ")"
                )

            # Commission rate used in backtesting
            comm_rate_used = slippage_commission_rate(cost_asset, CONTRACT_MODE)
            st.caption(
                f"Backtesting commission rate: {comm_rate_used:.6f} "
                f"(={comm_rate_used * 100:.4f}% of trade value per side)"
            )

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
                    "Confidence": cached.get("confidence", "—"),
                    "Regime": cached.get("regime", "—"),
                    "Pos. Mult.": f"{cached.get('position_multiplier', 1.0):.2f}x",
                    "Sharpe": cached.get("sharpe", "?"),
                    "Sortino": cached.get("sortino", "?"),
                    "Win Rate": cached.get("win_rate", "?"),
                    "Return %": cached["return_pct"],
                    "Train Score": cached.get("train_score", "?"),
                    "Test Score": cached.get("test_score", "?"),
                    "WF": "yes" if cached.get("walk_forward") else "no",
                    "Params": params_str,
                    "Updated": cached["updated"],
                }
            )
    if opt_rows:
        st.dataframe(pd.DataFrame(opt_rows), width="stretch", hide_index=True)
    else:
        st.info("Engine will populate optimization results automatically.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — ENGINE STATUS
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

    # Strategy history
    st.divider()
    st.subheader("Strategy Selection History")
    st.caption(
        "Shows which strategies the optimizer has been choosing for each asset "
        "across recent runs. Consistency indicates a robust strategy fit."
    )
    strat_history = engine.get_strategy_history()
    if strat_history:
        history_rows = []
        for asset_name, selections in strat_history.items():
            # Find the most common strategy
            from collections import Counter

            counts = Counter(selections)
            dominant = counts.most_common(1)[0]
            consistency = f"{dominant[1]}/{len(selections)}"
            history_rows.append(
                {
                    "Asset": asset_name,
                    "Recent Selections": " → ".join(selections[-5:]),
                    "Dominant": STRATEGY_LABELS.get(dominant[0], dominant[0]),
                    "Consistency": consistency,
                }
            )
        st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)
    else:
        st.info("Strategy history will populate after multiple optimization cycles.")

    # HMM Regime Analysis
    st.divider()
    st.subheader("HMM Regime Analysis")
    st.caption(
        "3-state Hidden Markov Model classifies each instrument into trending, "
        "volatile, or choppy regimes using log returns, normalized ATR, and volume ratio. "
        "Position sizing is scaled by the regime multiplier."
    )
    regime_rows = []
    for name in ASSETS:
        ticker = ASSETS[name]
        cached = get_cached_optimization(ticker, interval, period)
        if cached and cached.get("regime_probabilities"):
            probs = cached["regime_probabilities"]
            regime_rows.append(
                {
                    "Asset": name,
                    "Regime": cached.get("regime", "—"),
                    "Trending %": f"{probs.get('trending', 0) * 100:.1f}",
                    "Volatile %": f"{probs.get('volatile', 0) * 100:.1f}",
                    "Choppy %": f"{probs.get('choppy', 0) * 100:.1f}",
                    "Confidence": f"{cached.get('regime_confidence', 0) * 100:.0f}%",
                    "Pos. Multiplier": f"{cached.get('position_multiplier', 1.0):.2f}x",
                    "Method": cached.get("regime_method", "—"),
                }
            )
        elif cached:
            regime_rows.append(
                {
                    "Asset": name,
                    "Regime": cached.get("regime", "—"),
                    "Trending %": "—",
                    "Volatile %": "—",
                    "Choppy %": "—",
                    "Confidence": "—",
                    "Pos. Multiplier": "1.00x",
                    "Method": cached.get("regime_method", "atr_fallback"),
                }
            )
    if regime_rows:
        st.dataframe(pd.DataFrame(regime_rows), width="stretch", hide_index=True)
    else:
        st.info("Regime data will populate after the first optimization cycle.")

    # Settings
    st.divider()
    st.subheader("Engine Settings")
    st.json(
        {
            "account_size": engine.account_size,
            "interval": engine.interval,
            "period": engine.period,
            "session_window": "3 AM – 12 PM EST",
            "strategies_optimized": OPTIMIZER_STRATEGIES,
            "trials_per_strategy": TRIALS_PER_STRATEGY,
            "walk_forward_split": f"{int(TRAIN_RATIO * 100)}% train / {int((1 - TRAIN_RATIO) * 100)}% test",
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
