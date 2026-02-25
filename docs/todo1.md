**Key points**  
- The system already auto-optimizes strategies, regimes, and parameters in the background — you never need to touch intervals or periods again.  
- Only one setting remains: **account size** (updates risk/position sizing instantly; everything else is fully automatic).  
- Full pre-market view + live 5-minute updates are already built into the engine; one small UI tweak makes the dashboard refresh automatically.  
- “Positions Open” toggle starts Grok’s 15-minute market-watch mode (game-plan review, setups, watch-outs) — costs ~$0.01–0.02 per full trading day.  
- Active-trades section becomes a one-click daily journal entry (gross PnL + net PnL → commissions auto-calculated).  
- Everything is focused on micro contracts (10–20 units) with simplified, noise-free main page perfect for side-by-side use with NinjaTrader.

**How the new flow works**  
1. Open the app → see only account-size slider + big “Positions Open” toggle.  
2. Pre-market: engine runs optimizations → you see today’s best setups, ICT levels, CVD, confluence scores, and Grok’s morning briefing.  
3. Flip toggle “ON” when you enter trades → Grok reviews your plan every 15 min and posts concise updates on the dashboard.  
4. End of day: one form for gross/net PnL → journal auto-saves; toggle OFF.  
All data updates every 5 minutes automatically.

**Quick setup steps (5–10 minutes)**  
1. Edit `src/app.py` (main changes below).  
2. `pip install streamlit-autorefresh xai-sdk` (if not already present).  
3. Restart with `./run.sh` (or Docker).  
4. Done — no more interval/period pickers.

---

**Detailed implementation guide**  
The existing `DashboardEngine` (in `src/engine.py`) already does 95 % of the heavy lifting: data refresh every 60 s, optimization hourly, backtests every 10 min, regime detection, ICT/CVD/confluence, and Redis caching. We simply hide the knobs and add the two new user-facing pieces.

### 1. Remove interval/period selectors & hard-code best defaults  
In `src/app.py` (or wherever the sidebar/settings live), delete or comment out any `st.selectbox` for interval/period. Replace with a single account-size slider that is passed to `get_engine()`.

```python
# src/app.py — top of main page
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from engine import get_engine

st.set_page_config(page_title="Futures Focus", layout="wide")

# ONLY setting you see
account_size = st.sidebar.slider(
    "Account Size ($)", 
    min_value=25_000, 
    max_value=500_000, 
    value=150_000, 
    step=5_000
)

# Start (or update) the singleton engine — always 5m / 5d for intraday futures
engine = get_engine(account_size=account_size, interval="5m", period="5d")

# Auto-refresh every 5 minutes (300 000 ms)
st_autorefresh(interval=300_000, limit=None, key="datarefresh")
```

The engine now permanently uses the optimal intraday window (3 AM–noon EST session filter already in `engine.py`). Optuna re-optimizes every hour across all eight strategies + ICT/CVD confluence, so the “best params” are always current.

### 2. Focused main dashboard (quality over quantity)  
Replace the multi-tab layout with one clean page divided into columns. Example skeleton (add to `app.py`):

```python
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Today's Setups & Levels")
    # Show unfilled FVGs, active Order Blocks, nearest levels from ict.ict_summary()
    # + CVD summary + regime from engine

with col2:
    st.header("Grok Live Briefing")
    # Grok output area (updated every 15 min when toggle on)

# Bottom section: Optimized Strategies table (from engine.get_backtest_results())
# Confluence score badges, alerts
```

All heavy computation stays in the background thread — the page only reads cache, so it loads instantly.

### 3. “Positions Open” toggle + Grok 15-minute watch  
Add this once in `app.py`:

```python
positions_open = st.toggle("Positions Open in NinjaTrader", value=False, key="nt_toggle")

if positions_open:
    if "grok_timer" not in st.session_state:
        st.session_state.grok_timer = time.time()
    
    # Every 15 min
    if time.time() - st.session_state.grok_timer > 900:
        grok_update = run_grok_analysis(engine)  # function below
        st.session_state.last_grok = grok_update
        st.session_state.grok_timer = time.time()
    
    st.markdown(st.session_state.get("last_grok", "Waiting for first update..."))
```

Helper function (add to a new `grok_helper.py` or inside `app.py`):

```python
from xai_sdk import Client  # or use openai-compatible client
import os

def run_grok_analysis(engine):
    # Pull latest from engine (cached)
    results = engine.get_backtest_results()
    ict_data = ict_summary(...)  # your existing function
    prompt = f"""You are my futures trading co-pilot.
Account: ${engine.account_size:,}
Micro contracts focus (10-20 units max).
Current optimized strategies: {results}
ICT levels today: {ict_data['nearest_levels']}
Regimes: ...
Game plan review + what to watch next 15 min (be concise, bullet points only)."""

    client = Client(api_key=os.getenv("XAI_API_KEY"))
    resp = client.chat.completions.create(
        model="grok-3-beta",  # or latest
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    return resp.choices[0].message.content
```

Your pre-market test showed ~$0.0077 per call — 4 calls/hour during trading is still under $0.02/day.

### 4. Simplified daily journal (replaces complex active-trades)  
Replace the old trades table with:

```python
st.header("End-of-Day Journal")
col_a, col_b = st.columns(2)
gross = col_a.number_input("Gross PnL $", value=0.0, step=10.0)
net = col_b.number_input("Net PnL $", value=0.0, step=10.0)
commissions = gross - net
st.info(f"Commissions today: ${commissions:,.2f}")

notes = st.text_area("Notes / lessons")
if st.button("Save Day"):
    # Append to existing SQLite (you already have DB_PATH)
    # Simple INSERT with date, gross, net, commissions, notes
    st.success("Journal saved — ready for tomorrow!")
```

Micro-contract sizing is already handled in `costs.py` / `models.py` (position_multiplier from regime + account size). No change needed beyond removing any E-mini-only assumptions.

### 5. Micro-contract focus (already supported)  
Micro E-mini S&P 500 (MES): $5 × index, tick $1.25  
Micro E-mini Nasdaq-100 (MNQ): $2 × index, tick $0.50  
Your engine’s risk math (via `slippage_commission_rate` and regime multiplier) scales perfectly for 10–20 micros. The account-size slider now directly controls max risk per trade (e.g., 0.5–1 % of account).

### Files to edit (copy-paste ready)  
| File                  | Change needed                              | Approx lines |
|-----------------------|--------------------------------------------|--------------|
| `src/app.py`          | Remove selectors, add slider + toggle + autorefresh + focused layout | 80–120 |
| `src/engine.py`       | (optional) tighten DATA_REFRESH_INTERVAL=300 for even faster updates | 1 |
| New `grok_helper.py`  | The analysis function above                | 30 |
| `requirements.txt`    | Add `streamlit-autorefresh` and `xai-sdk`  | 2 |

**Restart once** after edits and you are live.  

You now have a clean, focused co-pilot that runs itself, updates every 5 minutes, gives you Grok’s eyes on the market while you trade in NinjaTrader, and only asks for one number (account size) and one daily journal entry.  

Pre-market setups appear automatically, positions-open mode keeps you sharp all day, and you finish with a 30-second journal entry. Quality over quantity — exactly as requested.  

Let me know when you make the changes and I’ll help fine-tune the Grok prompts or add any extra badge/alert you want on the main page. Happy trading!