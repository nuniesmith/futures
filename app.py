import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import optuna
import sqlite3
from datetime import datetime, date, timedelta
import requests


# ---------------------------------------------------------------------------
# Technical indicator helpers (replaces pandas_ta dependency)
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

st.set_page_config(page_title="Futures Morning Dashboard", layout="wide")
st.title("Futures Intraday Dashboard - $150k TakeProfit Trader")

# === TPT PLAYBOOK WARNING ===
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
# Data fetching (cached 60s)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_data(ticker, interval, period):
    return yf.download(ticker, interval=interval, period=period, prepost=True, auto_adjust=True)


data = {}
for name, ticker in assets.items():
    if name in selected_assets:
        data[name] = get_data(ticker, interval, period)

# ---------------------------------------------------------------------------
# SQLite journal
# ---------------------------------------------------------------------------
DB_PATH = "futures_journal.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            date TEXT,
            asset TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            contracts INTEGER,
            exit_price REAL,
            pnl REAL,
            rr REAL,
            notes TEXT,
            strategy TEXT
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
        "Optimizer",
    ]
)

# ===== TAB 1: Morning Scanner (with VWAP & Pivot) =========================
with tab1:
    st.header("3-5 AM Daily Scanner")
    if st.button("Refresh All Data (3-5 AM EST)"):
        st.cache_data.clear()

    rows = []
    for name, df in data.items():
        if df.empty:
            continue
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        pct_chg = (last - prev) / prev * 100

        # VWAP (last value for today)
        dfv = df.copy()
        dfv["date"] = dfv.index.date
        dfv["typical"] = (dfv["High"] + dfv["Low"] + dfv["Close"]) / 3
        dfv["tpv"] = dfv["typical"] * dfv["Volume"]
        dfv["cum_tpv"] = dfv.groupby("date")["tpv"].cumsum()
        dfv["cum_vol"] = dfv.groupby("date")["Volume"].cumsum()
        cum_vol_last = float(dfv["cum_vol"].iloc[-1])
        vwap = float(dfv["cum_tpv"].iloc[-1]) / cum_vol_last if cum_vol_last != 0 else last

        # Prior-day pivot
        daily = yf.download(assets[name], interval="1d", period="10d", auto_adjust=True)
        if len(daily) >= 2:
            prev_day = daily.iloc[-2]
            ph, pl, pc = float(prev_day["High"]), float(prev_day["Low"]), float(prev_day["Close"])
            pivot = (ph + pl + pc) / 3
            dist_pivot = last - pivot
        else:
            pivot = last
            dist_pivot = 0.0

        atr_val = float(atr(df["High"], df["Low"], df["Close"], length=14).iloc[-1])
        vol = int(df["Volume"].iloc[-1])

        rows.append(
            {
                "Asset": name,
                "% Overnight": round(pct_chg, 2),
                "Last": round(last, 2),
                "VWAP": round(vwap, 2),
                "% from VWAP": round((last - vwap) / vwap * 100, 2),
                "Pivot": round(pivot, 2),
                "Dist to Pivot": round(dist_pivot, 2),
                "ATR": round(atr_val, 2),
                "Volume": vol,
            }
        )

    df_scan = pd.DataFrame(rows)
    if not df_scan.empty:
        df_scan = df_scan.sort_values("% Overnight", key=abs, ascending=False)
        st.dataframe(
            df_scan.style.background_gradient(subset=["% from VWAP"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Today's Top Focus")
        st.write("**Focus today:** " + ", ".join(df_scan.head(5)["Asset"].tolist()))
    else:
        st.info("No data loaded yet. Select assets and refresh.")

# ===== TAB 2: Charts with VWAP + Pivots ===================================
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

            # EMAs
            df["EMA9"] = ema(df["Close"], length=9)
            df["EMA21"] = ema(df["Close"], length=21)
            fig.add_trace(
                go.Scatter(x=df.index, y=df["EMA9"], name="EMA9", line=dict(color="#ff9800"))
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["EMA21"], name="EMA21", line=dict(color="#2196f3"))
            )

            # VWAP
            if show_vwap:
                df["date"] = df.index.date
                df["typical"] = (df["High"] + df["Low"] + df["Close"]) / 3
                df["tpv"] = df["typical"] * df["Volume"]
                df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()
                df["cum_vol"] = df.groupby("date")["Volume"].cumsum()
                df["VWAP"] = df["cum_tpv"] / df["cum_vol"]
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["VWAP"],
                        name="VWAP",
                        line=dict(color="#9c27b0", width=2, dash="dash"),
                    )
                )

            # Pivots (from previous complete day)
            if show_pivots:
                daily = yf.download(assets[asset], interval="1d", period="10d", auto_adjust=True)
                if len(daily) >= 2:
                    prev = daily.iloc[-2]
                    ph, pl, pc = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
                    pp = (ph + pl + pc) / 3
                    r1 = 2 * pp - pl
                    s1 = 2 * pp - ph
                    r2 = pp + (ph - pl)
                    s2 = pp - (ph - pl)

                    levels = [
                        (ph, "Prior High", "red"),
                        (pl, "Prior Low", "green"),
                        (pc, "Prior Close", "white"),
                        (pp, "Pivot", "yellow"),
                        (r1, "R1", "orange"),
                        (s1, "S1", "lime"),
                        (r2, "R2", "darkorange"),
                        (s2, "S2", "lawngreen"),
                    ]
                    for y, txt, color in levels:
                        fig.add_hline(
                            y=y,
                            line_dash="dot",
                            line_color=color,
                            annotation_text=txt,
                            annotation_position="bottom right",
                            line_width=1.5,
                        )

            fig.update_layout(
                height=700, template="plotly_dark", xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation Matrix")
        if len(data) >= 2:
            closes = pd.DataFrame({name: d["Close"] for name, d in data.items()})
            returns = closes.pct_change().dropna()
            corr = returns.corr().round(2)
            st.dataframe(
                corr.style.background_gradient(cmap="RdYlGn"), use_container_width=True
            )
        else:
            st.info("Select 2+ assets for correlations")

# ===== TAB 3: Trade Planner (25% rule enforced) ===========================
with tab3:
    st.header("Trade Planner - 1% Risk + Levels")
    asset_plan = st.selectbox("Asset", selected_assets, key="plan")

    if asset_plan in data and not data[asset_plan].empty:
        dfp = data[asset_plan]
        last_price = float(dfp["Close"].iloc[-1])

        # Current VWAP
        dfv = dfp.copy()
        dfv["date"] = dfv.index.date
        dfv["typical"] = (dfv["High"] + dfv["Low"] + dfv["Close"]) / 3
        dfv["tpv"] = dfv["typical"] * dfv["Volume"]
        dfv["cum_tpv"] = dfv.groupby("date")["tpv"].cumsum()
        dfv["cum_vol"] = dfv.groupby("date")["Volume"].cumsum()
        cum_vol = float(dfv["cum_vol"].iloc[-1])
        current_vwap = float(dfv["cum_tpv"].iloc[-1]) / cum_vol if cum_vol != 0 else last_price

        # Pivot
        daily = yf.download(assets[asset_plan], interval="1d", period="10d", auto_adjust=True)
        if len(daily) >= 2:
            prev = daily.iloc[-2]
            pivot = (float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3
        else:
            pivot = last_price

        st.metric("Current Price", f"{last_price:.2f}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("VWAP", f"{current_vwap:.2f}", f"{last_price - current_vwap:+.2f}")
        with col_b:
            st.metric("Daily Pivot", f"{pivot:.2f}", f"{last_price - pivot:+.2f}")

        if last_price > max(current_vwap, pivot):
            st.subheader("Bias: BULLISH (above VWAP + Pivot)")
        else:
            st.subheader("Bias: BEARISH (below VWAP or Pivot)")

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

# ===== TAB 4: Journal + Daily P&L Tracker ==================================
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

    # ENFORCE DAILY RULES
    if total_today <= -2250:
        st.error("HARD STOP: -$2,250 MAX LOSS HIT - NO MORE TRADES TODAY")
    elif total_today <= -1500:
        st.warning("SOFT WARNING: -$1,500 reached - tighten up!")
    if total_today >= 2250:
        st.success("Excellent day! Consider locking profits and stopping.")

    st.divider()

    # All-time stats
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
        st.dataframe(dfj, use_container_width=True)
    else:
        st.info("No trades yet - log from Planner tab")

    # Export buttons
    st.divider()
    st.subheader("Export")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        if not dfj.empty:
            today_df = dfj[dfj["date"].str.startswith(today_str)]
            csv_today = today_df.to_csv(index=False).encode()
            st.download_button(
                "Export TODAY as CSV", csv_today, f"trades_{today_str}.csv", "text/csv"
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
            json_str = dfj.to_json(orient="records", date_format="iso")
            st.download_button(
                "Export ALL as JSON", json_str, "full_journal.json", "application/json"
            )

# ===== TAB 5: Grok AI Analyst (Pre & Post Market) =========================
with tab5:
    st.header("Grok AI Analyst - Pre & Post Market")
    api_key = st.text_input(
        "xAI Grok API Key (console.x.ai)", type="password", key="grok_key_input"
    )
    if api_key:
        st.session_state.grok_key = api_key

    # --- Pre-Market ---
    st.subheader("1. Pre-Market Morning Analysis (3-5 AM)")
    if st.button("Run Pre-Market Analysis (Full Game Plan)", type="primary"):
        if not st.session_state.get("grok_key"):
            st.error("Enter your Grok API key above")
        else:
            scan_text = (
                df_scan.to_string(index=False) if "df_scan" in dir() and not df_scan.empty else "No scanner data"
            )
            corr_text = corr.to_string() if "corr" in dir() else "No correlations"

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
                    resp = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {st.session_state.grok_key}"
                        },
                        json={
                            "model": "grok-4-1-fast-reasoning",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 1500,
                        },
                        timeout=30,
                    )
                    analysis = resp.json()["choices"][0]["message"]["content"]
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"API error: {e}")

    st.divider()

    # --- Post-Market ---
    st.subheader("2. Post-Market Daily Review (After Noon)")
    if st.button("Run Post-Market Daily Review (Today)"):
        if not st.session_state.get("grok_key"):
            st.error("Enter your Grok API key")
        else:
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
                    resp = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {st.session_state.grok_key}"
                        },
                        json={
                            "model": "grok-4-1-fast-reasoning",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.4,
                            "max_tokens": 1200,
                        },
                        timeout=30,
                    )
                    review = resp.json()["choices"][0]["message"]["content"]
                    st.markdown(review)
                except Exception as e:
                    st.error(f"API error: {e}")

    st.divider()

    # --- Historical Day Review ---
    st.subheader("3. Review Any Specific Past Day")
    review_date = st.date_input(
        "Select date to review", value=date.today(), max_value=date.today()
    )
    if st.button(f"Review {review_date}"):
        if not st.session_state.get("grok_key"):
            st.error("Enter your Grok API key")
        else:
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
                    resp = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {st.session_state.grok_key}"
                        },
                        json={
                            "model": "grok-4-1-fast-reasoning",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.4,
                        },
                        timeout=30,
                    )
                    review = resp.json()["choices"][0]["message"]["content"]
                    st.markdown(review)
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
            if crossover(self.ema1, self.ema2):
                self.buy()
            elif crossover(self.ema2, self.ema1):
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
        )
        stats = bt.run()
        st.write(stats)
    else:
        st.info("Select an asset with data to backtest")

# ===== TAB 7: Optimizer ====================================================
with tab7:
    st.header("Optuna Hyper-parameter Optimizer")
    st.write("Optimizes EMA periods for the selected backtest asset")

    if st.button("Run 30-trial Optimization"):
        if asset_bt not in data or data[asset_bt].empty:
            st.error("No data for selected asset")
        else:
            df_opt = data[asset_bt].copy()

            def objective(trial):
                n1 = trial.suggest_int("n1", 5, 20)
                n2 = trial.suggest_int("n2", 15, 50)

                class OptStrat(Strategy):
                    def init(self_strat):
                        self_strat.ema1 = self_strat.I(
                            ema, pd.Series(self_strat.data.Close), n1
                        )
                        self_strat.ema2 = self_strat.I(
                            ema, pd.Series(self_strat.data.Close), n2
                        )

                    def next(self_strat):
                        if crossover(self_strat.ema1, self_strat.ema2):
                            self_strat.buy()
                        elif crossover(self_strat.ema2, self_strat.ema1):
                            self_strat.sell()

                b = Backtest(df_opt, OptStrat, cash=account_size, commission=0.0002)
                return b.run()["Return [%]"]

            with st.spinner("Running optimization..."):
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=30, show_progress_bar=False)
                st.success(
                    f"Best: n1={study.best_params['n1']}, n2={study.best_params['n2']} "
                    f"-> Return {study.best_value:.1f}%"
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "Local dashboard | Yahoo Finance data | VWAP resets daily | "
    "Pivots from prior session | TPT Playbook enforced | Close all by noon EST"
)
