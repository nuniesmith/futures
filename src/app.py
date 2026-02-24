import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

# Ensure sibling modules (cache.py, etc.) are importable when run as
# `streamlit run src/app.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from cache import (
    REDIS_AVAILABLE,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    set_cached_indicator,
    set_cached_optimization,
)

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
# Scanner row builder (cached per asset)
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

    atr_val = float(atr(df["High"], df["Low"], df["Close"], length=14).iloc[-1])
    vol = int(df["Volume"].iloc[-1])

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
# Auto-optimization runner
# ---------------------------------------------------------------------------


def run_optimization_for_asset(ticker, interval, period, account_size):
    """Run Optuna optimization for a single asset, cached for 1 hour."""
    cached = get_cached_optimization(ticker, interval, period)
    if cached is not None:
        return cached

    df = get_data(ticker, interval, period)
    if df.empty:
        return None

    df_opt = df.copy()

    def objective(trial):
        n1 = trial.suggest_int("n1", 5, 20)
        n2 = trial.suggest_int("n2", 15, 50)

        class OptStrat(Strategy):
            def init(self):
                self.ema1 = self.I(ema, pd.Series(self.data.Close), n1)
                self.ema2 = self.I(ema, pd.Series(self.data.Close), n2)

            def next(self):
                if crossover(list(self.ema1), list(self.ema2)):
                    self.buy()
                elif crossover(list(self.ema2), list(self.ema1)):
                    self.sell()

        bt = Backtest(
            df_opt, OptStrat, cash=account_size, commission=0.0002, finalize_trades=True
        )
        return float(bt.run()["Return [%]"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    result = {
        "ticker": ticker,
        "n1": study.best_params["n1"],
        "n2": study.best_params["n2"],
        "return_pct": round(study.best_value, 2),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    set_cached_optimization(ticker, interval, period, result)
    return result


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Futures Morning Dashboard", layout="wide")
st.title("Futures Intraday Dashboard - $150k TakeProfit Trader")

cache_badge = "Redis" if REDIS_AVAILABLE else "In-Memory"
st.caption(f"Cache: {cache_badge}")

st.warning(
    "TPT PLAYBOOK RULE: Use MAX 3-4 contracts on $150k (25% rule). "
    "Oversizing is the #1 reason traders fail. "
    "Daily Loss Removed is your edge - respect EOD DD $4,500."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Account & Risk")
account_size = st.sidebar.number_input("Account Size $", value=150000, step=1000)
risk_pct = st.sidebar.slider("Risk per Trade %", 0.5, 2.0, 1.0) / 100
risk_dollars = account_size * risk_pct
st.sidebar.success(f"Risk per trade: **${risk_dollars:,.0f}** (25% rule enforced)")

assets = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Crude Oil": "CL=F",
    "E-mini S&P": "ES=F",
    "E-mini Nasdaq": "NQ=F",
}
selected_assets = st.sidebar.multiselect(
    "Assets to Track", list(assets.keys()), default=list(assets.keys())
)

interval = st.sidebar.selectbox("Chart Interval", ["1m", "5m"], index=1)
period = st.sidebar.selectbox("Data Period", ["5d", "10d"], index=0)

# ---------------------------------------------------------------------------
# Load data via Redis cache
# ---------------------------------------------------------------------------
data = {}
for name, ticker in assets.items():
    if name in selected_assets:
        data[name] = get_data(ticker, interval, period)

# ---------------------------------------------------------------------------
# SQLite journal
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "futures_journal.db")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, date TEXT, asset TEXT, direction TEXT,
            entry REAL, sl REAL, tp REAL, contracts INTEGER,
            exit_price REAL, pnl REAL, rr REAL, notes TEXT, strategy TEXT
        )"""
    )
    conn.commit()
    conn.close()


_init_db()


def log_trade(trade_tuple):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)", trade_tuple
    )
    conn.commit()
    conn.close()


def get_journal():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM trades ORDER BY date DESC", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Contract specs
# ---------------------------------------------------------------------------
contract_specs = {
    "Gold": {"point": 100, "tick": 0.1},
    "Silver": {"point": 5000, "tick": 0.005},
    "Copper": {"point": 250, "tick": 0.0005},
    "Crude Oil": {"point": 1000, "tick": 0.01},
    "E-mini S&P": {"point": 50, "tick": 0.25},
    "E-mini Nasdaq": {"point": 20, "tick": 0.25},
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Morning Scanner",
        "Charts & Correlations",
        "Trade Planner",
        "Journal",
        "Grok AI Analyst",
        "Backtester",
        "Auto-Optimizer",
    ]
)

# ===== TAB 1: Morning Scanner ==============================================
with tab1:
    st.header("3-5 AM Daily Scanner")
    if st.button("Refresh All Data (3-5 AM EST)"):
        flush_all()
        st.rerun()

    rows = []
    for name in selected_assets:
        ticker = assets[name]
        row = build_scanner_row(name, ticker, interval, period)
        if row:
            rows.append(row)

    df_scan = pd.DataFrame(rows)
    if not df_scan.empty:
        df_scan = df_scan.sort_values("% Overnight", key=abs, ascending=False)
        st.dataframe(
            df_scan.style.background_gradient(subset=["% from VWAP"], cmap="RdYlGn"),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Today's Top Focus")
        st.write("**Focus today:** " + ", ".join(df_scan.head(5)["Asset"].tolist()))
    else:
        st.info("No data loaded yet. Select assets and refresh.")

# ===== TAB 2: Charts with VWAP + Pivots ====================================
with tab2:
    st.header("Live Charts + Correlations")
    show_vwap = st.checkbox("Show Daily VWAP (resets at midnight)", value=True)
    show_pivots = st.checkbox("Show Prior-Day Pivots + Levels", value=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        asset = st.selectbox("Main Chart", selected_assets)
        if asset in data and not data[asset].empty:
            df = data[asset].copy()

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

            df["EMA9"] = ema(df["Close"], length=9)
            df["EMA21"] = ema(df["Close"], length=21)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["EMA9"], name="EMA9", line=dict(color="#ff9800")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["EMA21"], name="EMA21", line=dict(color="#2196f3")
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
                daily = get_daily(assets[asset])
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

            fig.update_layout(
                height=700,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Correlation Matrix")
        corr = None
        if len(data) >= 2:
            closes = pd.DataFrame({n: d["Close"] for n, d in data.items()})
            returns = closes.pct_change(fill_method=None).dropna()
            corr = returns.corr().round(2)
            st.dataframe(
                corr.style.background_gradient(cmap="RdYlGn"),
                width="stretch",
            )
        else:
            st.info("Select 2+ assets for correlations")

# ===== TAB 3: Trade Planner ================================================
with tab3:
    st.header("Trade Planner - 1% Risk + Levels")
    asset_plan = st.selectbox("Asset", selected_assets, key="plan")

    if asset_plan in data and not data[asset_plan].empty:
        dfp = data[asset_plan]
        last_price = float(dfp["Close"].iloc[-1])

        vdf = compute_vwap(dfp)
        cum_vol = float(vdf["cum_vol"].iloc[-1])
        current_vwap = (
            float(vdf["cum_tpv"].iloc[-1]) / cum_vol if cum_vol != 0 else last_price
        )

        daily = get_daily(assets[asset_plan])
        pivots = compute_pivots(daily)
        pivot = pivots["Pivot"] if pivots else last_price

        st.metric("Current Price", f"{last_price:.2f}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "VWAP", f"{current_vwap:.2f}", f"{last_price - current_vwap:+.2f}"
            )
        with col_b:
            st.metric("Daily Pivot", f"{pivot:.2f}", f"{last_price - pivot:+.2f}")

        if last_price > max(current_vwap, pivot):
            st.subheader("Bias: BULLISH (above VWAP + Pivot)")
        else:
            st.subheader("Bias: BEARISH (below VWAP or Pivot)")

        # Show cached optimal params if available
        opt_result = get_cached_optimization(assets[asset_plan], interval, period)
        if opt_result:
            st.info(
                f"Optimal EMA params: n1={opt_result['n1']}, "
                f"n2={opt_result['n2']} "
                f"(return {opt_result['return_pct']}%, "
                f"updated {opt_result['updated']})"
            )

        strategy_type = st.selectbox(
            "Strategy", ["VWAP Pullback", "Prior Day Pivot Break", "EMA Crossover"]
        )
        direction = st.radio("Direction", ["Long", "Short"])
        spec = contract_specs[asset_plan]

        entry = st.number_input(
            "Limit Entry Price", value=float(last_price), step=spec["tick"]
        )
        atr_val = float(atr(dfp["High"], dfp["Low"], dfp["Close"], length=14).iloc[-1])
        sl_mult = st.slider("SL ATR Multiplier", 0.5, 2.0, 1.0)
        sl_dist = atr_val * sl_mult
        sl = entry - sl_dist if direction == "Long" else entry + sl_dist
        tp_mult = st.slider("TP RR", 1.5, 4.0, 2.0)
        tp = (
            entry + (entry - sl) * tp_mult
            if direction == "Long"
            else entry - (sl - entry) * tp_mult
        )

        risk_per_contract = abs(entry - sl) * spec["point"]
        max_contracts = (
            int(risk_dollars // risk_per_contract) if risk_per_contract > 0 else 1
        )
        max_contracts = min(max_contracts, 4)  # HARD 25% RULE

        st.error("TPT 25% RULE: Max 4 contracts on $150k (recommended 3)")
        st.metric("Recommended Contracts", max_contracts)
        st.write(
            f"**SL:** {sl:.2f}  |  **TP:** {tp:.2f}  |  **Risk:** ${risk_dollars:,.0f}"
        )

        if max_contracts > 0:
            be_offset = 5 / (max_contracts * spec["point"])
            be_price = entry + be_offset if direction == "Long" else entry - be_offset
            st.write(f"**Breakeven (after ~$5 commissions):** {be_price:.2f}")

        if st.button("Save Trade to Journal"):
            pnl_est = (
                (tp - entry if direction == "Long" else entry - tp)
                * spec["point"]
                * max_contracts
            )
            log_trade(
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    asset_plan,
                    direction,
                    entry,
                    sl,
                    tp,
                    max_contracts,
                    None,
                    None,
                    pnl_est,
                    tp_mult,
                    "",
                    strategy_type,
                )
            )
            st.success(
                f"Logged: {direction} {asset_plan} @ {entry} | "
                f"SL {sl:.2f} | TP {tp:.2f} | {max_contracts} contracts"
            )

# ===== TAB 4: Journal + Daily P&L ==========================================
with tab4:
    st.header("Trade Journal + Daily P&L Tracker")

    today_str = date.today().strftime("%Y-%m-%d")
    dfj = get_journal()
    today_trades = (
        dfj[dfj["date"].str.startswith(today_str)] if not dfj.empty else pd.DataFrame()
    )

    realized_pnl = (
        float(today_trades["pnl"].sum())
        if not today_trades.empty and "pnl" in today_trades.columns
        else 0.0
    )

    unrealized = st.number_input(
        "Today's Unrealized P&L (from open positions)", value=0.0, step=50.0
    )
    total_today = realized_pnl + unrealized

    col1, col2, col3 = st.columns(3)
    col1.metric("Today's Realized P&L", f"${realized_pnl:,.0f}")
    col2.metric(
        "Today's Total P&L",
        f"${total_today:,.0f}",
        delta=f"{total_today / 1500:.1f}x risk" if risk_dollars > 0 else "",
    )
    col3.metric("Trades Today", len(today_trades))

    if total_today <= -2250:
        st.error("HARD STOP: -$2,250 MAX LOSS HIT - NO MORE TRADES TODAY")
    elif total_today <= -1500:
        st.warning("SOFT WARNING: -$1,500 reached - tighten up!")
    if total_today >= 2250:
        st.success("Excellent day! Consider locking profits and stopping.")

    st.divider()

    if not dfj.empty:
        st.subheader("All-Time Stats")
        col_a, col_b, col_c, col_d = st.columns(4)
        wins = len(dfj[dfj["pnl"] > 0]) if "pnl" in dfj.columns else 0
        winrate = wins / len(dfj) * 100 if len(dfj) else 0
        total_pnl = float(dfj["pnl"].sum()) if "pnl" in dfj.columns else 0.0
        col_a.metric("Total Trades", len(dfj))
        col_b.metric("Win Rate", f"{winrate:.1f}%")
        col_c.metric("Total PNL", f"${total_pnl:,.0f}")
        col_d.metric(
            "Avg RR",
            f"{dfj['rr'].mean():.2f}" if "rr" in dfj.columns else "N/A",
        )
        st.dataframe(dfj, width="stretch")
    else:
        st.info("No trades yet - log from Planner tab")

    st.divider()
    st.subheader("Export")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        if not dfj.empty:
            today_df = dfj[dfj["date"].str.startswith(today_str)]
            csv_today = today_df.to_csv(index=False).encode()
            st.download_button(
                "Export TODAY as CSV",
                csv_today,
                f"trades_{today_str}.csv",
                "text/csv",
            )
    with col_exp2:
        if not dfj.empty:
            week_start = (
                datetime.now() - timedelta(days=datetime.now().weekday())
            ).strftime("%Y-%m-%d")
            week_df = dfj[dfj["date"] >= week_start]
            csv_week = week_df.to_csv(index=False).encode()
            st.download_button(
                "Export THIS WEEK as CSV",
                csv_week,
                f"trades_week_{week_start}.csv",
                "text/csv",
            )
    with col_exp3:
        if not dfj.empty:
            json_str = dfj.to_json(orient="records", date_format="iso") or ""
            st.download_button(
                "Export ALL as JSON",
                json_str,
                "full_journal.json",
                "application/json",
            )

# ===== TAB 5: Grok AI Analyst ==============================================
with tab5:
    st.header("Grok AI Analyst - Pre & Post Market")

    # Read API key from environment first, allow override in UI
    env_key = os.getenv("XAI_API_KEY", "")
    if env_key:
        st.session_state.grok_key = env_key
        st.success("API key loaded from environment")
    else:
        api_key = st.text_input(
            "xAI Grok API Key (or set XAI_API_KEY env var)",
            type="password",
            key="grok_key_input",
        )
        if api_key:
            st.session_state.grok_key = api_key

    def _call_grok(prompt, max_tokens=1500, temperature=0.3):
        key = st.session_state.get("grok_key")
        if not key:
            st.error("Enter your Grok API key above or set XAI_API_KEY")
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
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # --- Pre-Market ---
    st.subheader("1. Pre-Market Morning Analysis (3-5 AM)")
    if st.button("Run Pre-Market Analysis (Full Game Plan)", type="primary"):
        scan_text = (
            df_scan.to_string(index=False)
            if "df_scan" in dir() and not df_scan.empty
            else "No scanner data"
        )
        corr_text = (
            corr.to_string()  # type: ignore[possibly-undefined]
            if "corr" in dir() and corr is not None
            else "No correlations"
        )

        prompt = f"""You are a strict TPT-funded futures trader managing a $150k account.
Rules you MUST follow:
- Max 3-4 contracts per trade (25% rule)
- Risk exactly 1% ($1,500) per trade
- Only early-morning trades (3 AM - noon EST), close everything by noon
- Daily P&L limits: soft stop -$1,500, hard stop -$2,250
- Focus on correlations (Gold/Silver/Copper/Oil/ES/NQ)

Current time: 3-5 AM EST
Scanner data:
{scan_text}

Correlation matrix (5-min returns):
{corr_text}

Today's key levels (VWAP + pivots) are already plotted in charts.

Give me a complete morning game plan:
1. Overall market bias & 1-5 best assets to focus on
2. 1-5 specific trade ideas (asset, direction, limit entry zone, SL, TP, contracts, RR, rationale)
3. Key correlations to watch
4. Any news/events to be aware of
5. Risk reminders for today

Be concise, bullet-point only, actionable. No fluff."""

        with st.spinner("Asking Grok for morning game plan..."):
            try:
                result = _call_grok(prompt)
                if result:
                    st.markdown(result)
            except Exception as e:
                st.error(f"API error: {e}")

    st.divider()

    # --- Post-Market ---
    st.subheader("2. Post-Market Daily Review (After Noon)")
    if st.button("Run Post-Market Daily Review (Today)"):
        today_str_pm = date.today().strftime("%Y-%m-%d")
        journal_df = get_journal()
        today_trades_pm = (
            journal_df[journal_df["date"].str.startswith(today_str_pm)]
            if not journal_df.empty
            else pd.DataFrame()
        )
        trades_text = (
            today_trades_pm.to_string(index=False)
            if not today_trades_pm.empty
            else "No trades logged today"
        )
        total_pnl_pm = (
            float(today_trades_pm["pnl"].sum())
            if not today_trades_pm.empty and "pnl" in today_trades_pm.columns
            else 0.0
        )

        prompt = f"""You are reviewing my trading day for a $150k TPT account.
Rules: 1% risk ($1,500), max 3-4 contracts, early-morning only, close by noon, daily limits -$1,500 soft / -$2,250 hard.

Date: {today_str_pm}
Total P&L today: ${total_pnl_pm:,.0f}
Trades taken:
{trades_text}

Provide a professional daily review:
1. Performance summary (win rate, avg RR, total risk taken)
2. What worked well (specific examples)
3. What went wrong / rule breaks
4. Key lessons & 1-2 specific improvements for tomorrow
5. Did I respect TPT rules and my personal P&L limits?

Bullet points only, honest, constructive, focused on consistency."""

        with st.spinner("Analyzing today's trades..."):
            try:
                result = _call_grok(prompt, max_tokens=1200, temperature=0.4)
                if result:
                    st.markdown(result)
            except Exception as e:
                st.error(f"API error: {e}")

    st.divider()

    # --- Historical Day Review ---
    st.subheader("3. Review Any Specific Past Day")
    review_date = st.date_input(
        "Select date to review", value=date.today(), max_value=date.today()
    )
    if st.button(f"Review {review_date}"):
        d_str = review_date.strftime("%Y-%m-%d")
        journal_df = get_journal()
        day_trades = (
            journal_df[journal_df["date"].str.startswith(d_str)]
            if not journal_df.empty
            else pd.DataFrame()
        )
        trades_text = (
            day_trades.to_string(index=False)
            if not day_trades.empty
            else "No trades on this day"
        )
        pnl = (
            float(day_trades["pnl"].sum())
            if not day_trades.empty and "pnl" in day_trades.columns
            else 0.0
        )

        prompt = f"""Review my trading on {d_str} for $150k TPT account.
P&L: ${pnl:,.0f}
Trades:
{trades_text}

Give detailed day-by-day review:
1. Performance summary
2. What worked well
3. What went wrong / rule breaks
4. Key lessons
5. TPT compliance check"""

        with st.spinner("Generating review..."):
            try:
                result = _call_grok(prompt, temperature=0.4)
                if result:
                    st.markdown(result)
            except Exception as e:
                st.error(f"API error: {e}")

# ===== TAB 6: Backtester ===================================================
with tab6:
    st.header("Backtester (last 5-10 days)")

    class EMACross(Strategy):
        n1 = 9
        n2 = 21

        def init(self):
            self.ema1 = self.I(ema, pd.Series(self.data.Close), self.n1)
            self.ema2 = self.I(ema, pd.Series(self.data.Close), self.n2)

        def next(self):
            if crossover(list(self.ema1), list(self.ema2)):
                self.buy()
            elif crossover(list(self.ema2), list(self.ema1)):
                self.sell()

    asset_bt = st.selectbox("Backtest Asset", selected_assets, key="bt")
    if asset_bt in data and not data[asset_bt].empty:
        df_bt = data[asset_bt].copy()
        bt = Backtest(
            df_bt,
            EMACross,
            cash=account_size,
            commission=0.0002,
            margin=1.0,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        stats_display = stats.apply(lambda x: str(x))
        st.dataframe(stats_display.to_frame(name="Value"), width="stretch")
    else:
        st.info("Select an asset with data to backtest")

# ===== TAB 7: Auto-Optimizer ===============================================
with tab7:
    st.header("Auto-Optimizer (All Assets)")
    st.write(
        "Runs Optuna 30-trial optimization for every selected asset. "
        "Results are cached for 1 hour in Redis so subsequent loads are instant."
    )

    # Show existing cached results
    st.subheader("Current Optimal Parameters")
    opt_rows = []
    for name in selected_assets:
        ticker = assets[name]
        cached = get_cached_optimization(ticker, interval, period)
        if cached:
            opt_rows.append(
                {
                    "Asset": name,
                    "Fast EMA (n1)": cached["n1"],
                    "Slow EMA (n2)": cached["n2"],
                    "Return %": cached["return_pct"],
                    "Last Updated": cached["updated"],
                }
            )
    if opt_rows:
        st.dataframe(pd.DataFrame(opt_rows), width="stretch", hide_index=True)
    else:
        st.info("No cached results yet. Run optimization below.")

    st.divider()

    # Single asset optimization
    col_single, col_all = st.columns(2)
    with col_single:
        opt_asset = st.selectbox(
            "Optimize Single Asset", selected_assets, key="opt_single"
        )
        if st.button("Optimize Selected Asset"):
            ticker = assets[opt_asset]
            with st.spinner(f"Optimizing {opt_asset}..."):
                result = run_optimization_for_asset(
                    ticker, interval, period, account_size
                )
                if result:
                    st.success(
                        f"{opt_asset}: Best n1={result['n1']}, "
                        f"n2={result['n2']} -> {result['return_pct']}%"
                    )

    # Full loop: optimize ALL selected assets
    with col_all:
        st.write("**Optimize All Assets**")
        if st.button("Run Full Optimization Loop", type="primary"):
            progress = st.progress(0)
            results = []
            for i, name in enumerate(selected_assets):
                ticker = assets[name]
                with st.spinner(
                    f"Optimizing {name} ({i + 1}/{len(selected_assets)})..."
                ):
                    result = run_optimization_for_asset(
                        ticker, interval, period, account_size
                    )
                    if result:
                        results.append(
                            {
                                "Asset": name,
                                "Fast EMA (n1)": result["n1"],
                                "Slow EMA (n2)": result["n2"],
                                "Return %": result["return_pct"],
                                "Last Updated": result["updated"],
                            }
                        )
                progress.progress((i + 1) / len(selected_assets))

            if results:
                st.subheader("Optimization Complete")
                st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "Local dashboard | Yahoo Finance data | VWAP resets daily | "
    "Pivots from prior session | TPT Playbook enforced | Close all by noon EST"
)
