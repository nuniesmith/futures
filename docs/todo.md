# Enhancing a CME futures dashboard: a complete technical blueprint

**The highest-impact upgrades for your Streamlit futures dashboard are a 3-state GaussianHMM regime filter, Volume Profile strategies, and a pre-market asset scorer—together these address the three biggest gaps in most discretionary setups: regime awareness, institutional-level support/resistance, and systematic focus.** This report synthesizes 2024–2026 best practices across HMM regime detection, complementary strategies, pre-market scanning, backtesting robustness, and dashboard design, with production-ready Python patterns for each. Every recommendation targets your existing stack (Streamlit, Optuna, Redis, SQLite) and your four instruments (GC/MGC, CL/MCL, ES/MES, NQ/MNQ).

---

## HMM regime detection: 3 states, multi-feature, filtered probabilities

**GaussianHMM from `hmmlearn` is the clear production winner.** It trains in under a second on 2,000 bars, provides native BIC/AIC for state selection (v0.3+), and has a convergence monitor built in. GMMHMM adds Gaussian mixtures per state—useful for fat-tailed distributions—but roughly doubles the parameter count and overfitting risk for intraday data. Reserve it for when GaussianHMM clearly underfits. Bayesian HMM via PyMC5 gives posterior uncertainty over all parameters, which Blanchard (2025, Wiley) showed improves regime persistence when combined with moving-average pre-smoothing, but MCMC sampling is too slow for session-start retraining. The alternative `pomegranate` (v1.0+, PyTorch backend) benchmarks **2× faster** than hmmlearn on large datasets and supports GPU acceleration, making it viable if you need sub-second fitting on 10,000+ bars.

**Multi-feature models consistently outperform univariate ones.** The recommended input vector for your dashboard is three features: log returns, normalized ATR (`ATR_14 / close`), and a volume ratio (`current_volume / 20-bar rolling mean`). StandardScaler each feature before fitting. All features must be stationary—use returns and ratios, never raw prices. PCA is optional but useful if you later add correlated features like credit spreads or DXY.

For state selection, **BIC reliably picks 3 states** on equity and commodity futures—corresponding to low-volatility trending, high-volatility trending, and choppy/ranging regimes. A practical pattern: fit models with 2–5 states across 5–10 random seeds each, select the configuration with minimum BIC. Embed this inside your existing Optuna loop to jointly optimize `n_states`, `covariance_type` (diagonal vs. full), lookback window (1,000–5,000 bars), and a regime-confidence threshold (0.5–0.9). Use walk-forward folds as the evaluation metric to avoid overfitting.

**The critical production rule is: never use Viterbi decoding for real-time decisions.** Viterbi optimizes the entire state path and changes retroactively as new data arrives. Instead, use the forward algorithm to compute filtered probabilities at the latest bar—these are causal and free of look-ahead bias. Map regimes to strategies with soft weighting: scale position size by `P(trending) × 1.0 + P(choppy) × 0.25`, and require a confidence threshold above **0.6** before activating any signal. Add a persistence filter: only switch strategy allocation when the regime persists for 3–5 consecutive bars.

Retrain the HMM at session open using a rolling 20–60 day window of intraday bars. Within the session, update filtered probabilities bar-by-bar without refitting. Check `model.monitor_.converged` after every fit; if `False`, fall back to the prior model and log a warning. Fit separate HMMs per instrument—GC, CL, ES, and NQ have materially different volatility structures.

Recent evidence supports this approach. The LSEG developer article applied GaussianHMM to **ES futures (ESc1)** over 2006–2023 and found the HMM strategy avoided the 2008 crash entirely and profited during COVID via short-selling. Shu, Yu, and Mulvey (2024, Princeton) showed that Statistical Jump Models outperform standard HMMs by adding an explicit jump penalty that enforces stickier regimes—worth evaluating as a complement. A regime-switching ETF strategy by PriceActionLab (2024) across SPY/QQQ/TLT/GLD achieved a **Sharpe ratio of 1.21** with 10.8% annualized return and only 16.5% max drawdown over 2003–2025.

---

## Ten complementary strategies spanning volume, order flow, events, and structure

Your existing six ATR-based strategies are all technically driven. The biggest diversification gains come from adding volume-structural, event-driven, and multi-timeframe approaches that use fundamentally different information.

**Volume Profile strategies provide institutional-grade support and resistance.** Calculate session volume profiles by distributing each bar's volume across price bins proportional to the bar's overlap with each bin. The Point of Control (POC) is the bin with highest volume—price reverts to it roughly **75% of the time** in ranging markets. Value Area High/Low (VAH/VAL, enclosing 70% of volume) act as the key boundaries. Three setups follow: (1) POC Mean Reversion—enter toward POC when price moves 30+ points away on ES, with a stop 10 points beyond the reversal candle; (2) Value Area Rejection—when price dips below VAL and a bullish engulfing forms, enter on close back above VAL, targeting POC then VAH; (3) Naked POC Magnet—track unfilled POCs from prior 5 sessions and trade toward them when price approaches within 15 points. For Python, distribute volume across `numpy.linspace` bins and use `scipy.signal.find_peaks` to identify high/low volume nodes. The `py-market-profile` library handles TPO charts.

**Pullback-to-EMA with candlestick confirmation** extends your TrendEMA strategy. Require EMAs stacked in order (10 > 20 > 50 for bullish), wait for price to retrace to the 10 or 21 EMA, then enter on the break of a confirmation candle (bullish engulfing or hammer) that closes above the EMA. Stop goes below the setup bar. Recommended periods by instrument: **9/21/50 on 5-min for ES and GC**, 10/20/50 for NQ, and 8/21/50 on 3-min for CL (which needs faster parameters due to volatility). Adding an RSI(14) < 40 filter at the pullback improves win rate by filtering out pullbacks-in-exhaustion.

**Cumulative Volume Delta (CVD) can be approximated from OHLCV data** without Level 2. The heuristic: `buy_volume = volume × (close − low) / (high − low)`, `sell_volume = volume − buy_volume`, `delta = buy − sell`. Accuracy is ±15–25% versus true bid/ask delta but adds useful confluence. Track CVD divergences (price makes lower low, CVD makes higher low → bullish accumulation) and volume absorption candles (high volume, small body near support → buyers absorbing selling). Reset CVD at market open for intraday anchoring.

**Event reaction strategies deserve instrument-specific playbooks.** For **EIA crude inventory** (Wednesday 10:30 AM ET): close CL positions 5 minutes before, wait for the first 5-minute candle to close post-report, enter in the direction of the move only if volume exceeds 2× the 20-period average. Typical CL move: **$0.50–$2.00/barrel** in the first 30 minutes. For **CPI/NFP** on ES/NQ: flatten 15 minutes before, skip the first 5 minutes of whipsaw, then enter with the 5-minute trend if confirmed by volume. Typical ES move on CPI surprise: **30–80 points**. For **FOMC**: the press conference at 2:30 PM often reverses the initial 2:00 PM reaction—wait until a 5-minute close after 2:35 PM shows clear direction. Build an event toggle so these playbooks can be activated or deactivated per session, critical for prop firms that ban news trading.

**Multi-timeframe confluence** acts as a meta-filter across all strategies. Use a three-layer system: higher timeframe for bias (1-hour EMA alignment), setup timeframe for pattern identification (15-minute), and entry timeframe for timing (5-minute). All three must agree on direction. Maintain a **4–6× factor between timeframes** (not 5/10/15, which are too close). Score confluence from 1 to 3; only trade on 3/3 alignment. Best combinations: 15m/5m/1m for ES and NQ, 1H/15m/5m for GC, 15m/5m/3m for CL.

**ICT/Smart Money Concepts have surged among prop firm traders in 2024–2026.** The most automatable are Fair Value Gaps (three-candle imbalance patterns where price tends to return) and Liquidity Sweep + Market Structure Shift (price hunts stops beyond obvious highs/lows, then reverses). The `smartmoneyconcepts` Python package detects FVGs, order blocks, and liquidity levels from OHLCV data. SMT Divergence—where ES makes a higher high but NQ fails to (or vice versa)—provides a high-probability reversal filter unique to correlated index futures.

---

## A composite pre-market scorer built on five metrics

Professional futures traders at Topstep, Apex, and similar firms consistently narrow to **2–3 instruments each morning** using a simple but systematic process: check overnight action, review the economic calendar, assess which instruments show elevated volatility relative to their norm, and identify the cleanest technical setups.

The recommended composite scoring formula uses five weighted metrics. **Normalized ATR** (30% weight): `NATR = ATR_14 / close × 100`, compared to its 20-day average; an instrument scoring 1.5× its norm gets a high volatility score. **Relative Volume** (25%): `RVOL = current_volume / 20-day_avg_volume`; values above 1.5 indicate meaningful participation. **Overnight gap magnitude** (15%): `|globex_open − prior_close| / prior_close × 100`; larger gaps create tradeable scenarios. **Economic catalyst score** (20%): binary/tiered—0 for no events, 33 for low-impact, 66 for medium, 100 for high-impact events directly affecting that instrument. **Momentum score** (10%): `|close − EMA_20| / ATR_14`, measuring displacement from equilibrium. The composite is a weighted sum normalized to 0–100. Present this in a traffic-light table sorted by score, with expandable detail cards per instrument.

For the economic calendar, **Trading Economics API** is the best option: it provides importance levels (1–3), actual/forecast/previous values, and date filters. The free tier returns sample data (`guest:guest`); full access starts at ~$50/month. **Finnhub** offers a usable free tier at 60 calls/minute with economic calendar events including impact ratings. **FRED API** (completely free, 120 requests/minute) is excellent for pulling actual macro data series (CPI, GDP, unemployment, Fed funds rate) but is not a real-time event calendar. Map events to instruments: EIA → CL, FOMC/CPI/NFP → ES+NQ+GC, OPEC → CL, USD/DXY → GC.

For free futures price data, **yfinance remains the most accessible** option in 2026: tickers `ES=F`, `NQ=F`, `GC=F`, `CL=F` (and `MES=F`, `MNQ=F`, `MGC=F`, `MCL=F` for micros) provide daily OHLCV and up to 30 days of minute-level intraday data. Data is delayed ~15 minutes and unofficial. For institutional-grade Globex overnight data with nanosecond timestamps, **Databento** (`GLBX.MDP3` dataset) is the best Python-native option at pay-per-use pricing (~$0.01–0.05/GB). Polygon.io (now Massive.com) has futures data listed as "Coming Soon" as of February 2026—not yet available.

Cache the pre-market scanner with `@st.cache_data(ttl=300)` (5-minute TTL). Calculate overnight ranges by fetching hourly data and filtering for Globex hours: 6 PM–9:30 AM ET for ES/NQ, 6 PM–8:20 AM for GC, 6 PM–9:00 AM for CL. Track Asian session (7 PM–2 AM ET) and European session (2 AM–8 AM ET) ranges separately for the AMD/Power-of-Three setup.

---

## Backtesting robustness: slippage, commissions, Monte Carlo, and PBO

**Realistic slippage for CME futures is 1 tick per side during RTH** for retail order sizes (1–20 micros, 1–2 minis). This translates to: **ES $12.50, MES $1.25, NQ $5.00, MNQ $0.50, CL $10.00, MCL $1.00, GC $10.00, MGC $1.00** per side. Apply a time-of-day multiplier: 1× during RTH, 2× during ETH/overnight, and 1.5× during high-volatility events (CPI, FOMC). For backtesting, fill market orders at next bar's open plus slippage. Fill limit orders only when price trades through the limit by at least 1 tick—never assume fills at exact limit price.

Commission rates in 2026 vary significantly by broker. On a conservative all-in round-turn basis (commission + exchange + NFA + clearing + slippage): **ES ~$28–30, MES ~$3.50, NQ ~$28, MNQ ~$3.50, CL ~$30, MCL ~$4, GC ~$30, MGC ~$4**. The break-even ticks per trade matter: MES needs ~2.8 ticks, MNQ needs ~7.0 ticks, which means MNQ strategies need wider targets. A critical cost insight: trading 10 MES costs roughly **$18 RT versus $5.76 for 1 ES**—switch to full contracts when consistently trading 10+ micros.

**Monte Carlo simulation via trade-level bootstrap is the gold standard** for robustness testing. Sample N trades with replacement from your backtest results, build a new equity curve, repeat 10,000 times. Extract percentile bands (5th, 25th, 50th, 75th, 95th) to create equity confidence cones. The 95th-percentile maximum drawdown becomes your risk planning figure. Track live performance against these bands—if it falls outside the 5th percentile, the strategy may be broken. Key caveat: Monte Carlo assumes trade independence; if your strategy produces correlated win/loss streaks (trending strategies), MC can be overly optimistic about drawdown distributions.

For walk-forward optimization with Optuna, use a **3:1 to 5:1 in-sample to out-of-sample ratio**. Anchored (expanding) windows work best for robust trend strategies; rolling (fixed-width) windows adapt faster for mean-reversion or regime-sensitive strategies. Detect overfitting using the **Probability of Backtest Overfitting (PBO)** methodology from Bailey and Lopez de Prado: partition the time series into S equal subsets, enumerate all C(S, S/2) train/test splits, and compute the fraction where the best in-sample configuration underperforms the median out-of-sample. PBO below 0.05 indicates low overfitting risk; above 0.50 means the strategy is likely curve-fit. Recent work by Arian, Norouzi, and Seco (2024) shows **Combinatorial Purged Cross-Validation (CPCV)** significantly outperforms standard walk-forward in reducing PBO.

For continuous contracts, use **backward ratio adjustment** on a first-of-delivery-month roll schedule—this preserves percentage returns, which is what matters for P&L calculations. Never backtest on raw spliced contracts. For the backtesting engine itself, **VectorBT** is the best match for your Optuna workflow due to its NumPy/Numba vectorization (1,000× faster than Backtrader for parameter sweeps), but it lacks native futures support. Build a custom layer on top that handles contract specs, tick-based slippage, commission schedules, session boundary filtering, and margin constraints.

---

## Dashboard design: fragments, dark mode, and prop firm guardrails

**Streamlit Fragments (GA since v1.37) are the single most important feature** for a trading dashboard. Decorate panels with `@st.fragment(run_every="5s")` to auto-refresh price tickers and signal panels independently without full-page reruns. This eliminates the stale-data problem that plagued earlier Streamlit trading apps. Use 5-second refresh for price panels and risk gauges, 10-second for signal panels and charts, and manual refresh for historical analysis pages.

For charting, **`streamlit-lightweight-charts-v5`** provides TradingView-quality candlestick rendering with multi-pane support for indicators—this is what traders are accustomed to seeing. Use Plotly for custom visualizations like volume profile heatmaps, Monte Carlo cone charts, and regime probability displays. Embed the chart in the center of a three-column layout: left sidebar for risk status and controls, center for the primary chart and signal panel, right for key levels and economic calendar.

The **highest-value dashboard feature for prop firm traders is real-time rule compliance tracking**: daily P&L versus daily loss limit (with warnings at 50% and 80% thresholds), trailing drawdown gauge, consistency checker (flag if single-day profit exceeds 30–40% of total), and a trade counter. Traders lose funded accounts over rule violations more often than bad trades. Build this as an always-visible sidebar fragment.

For alerts, implement a multi-channel dispatcher using simple `requests.post()` calls to Slack webhooks, Discord webhooks, and the Telegram Bot API. **Alert deduplication is essential**—without a 5-minute cooldown per unique signal key, volatile markets will generate hundreds of identical alerts. Store sent alert timestamps in `st.session_state` or Redis for persistence.

Structure the app as a multi-page Streamlit application: Dashboard (main trading view with live prices, signals, and risk gauges), Signals (detailed signal history and strategy performance), P&L (daily/weekly/monthly tracking with Monte Carlo projections), Journal (SQLite trade log with tagging for errors like FOMO and revenge trading), and Settings (alert configuration, risk profiles, strategy toggles). Dark mode is non-negotiable—configure via `.streamlit/config.toml` with a `#0E1117` background, `#1A1D23` panel background, and `#00D4AA` teal accent for buy signals. Use the color convention: bright green for strong buy, gold for neutral, coral red for sell, orange for warnings.

---

## Conclusion: implementation priority and architectural choices

The upgrades with the highest return on implementation effort, ordered by priority:

1. **HMM regime filter** (GaussianHMM, 3 states, multi-feature) directly improves all six existing strategies by suppressing signals in unfavorable regimes. The LSEG study on ES futures demonstrated this avoids catastrophic drawdowns. Implementation cost: ~2 days.

2. **Volume Profile with POC/VAH/VAL** adds a fundamentally different analytical lens (price-volume structure versus price-time indicators). It provides levels that institutional traders actually respect. Implementation cost: ~1 day for calculation, ~1 day for visualization.

3. **Pre-market composite scorer** systematizes instrument selection, eliminating the cognitive overhead of monitoring four markets simultaneously. Implementation cost: ~1 day.

4. **Monte Carlo + PBO robustness testing** prevents deploying overfit strategies to live capital. The PBO methodology is the most rigorous guard against Optuna overfitting. Implementation cost: ~1 day for Monte Carlo, ~2 days for full PBO.

5. **Streamlit Fragments migration** for real-time refresh solves the dashboard responsiveness problem at its root. Implementation cost: ~0.5 days.

The key architectural decision is to fit separate HMMs per instrument, use filtered (forward) probabilities for real-time regime estimation, and map regimes to both strategy selection and position sizing simultaneously. This creates a three-layer signal pipeline: regime filter → strategy signal → multi-timeframe confluence confirmation. Combined with realistic 1-tick slippage modeling and the PBO overfitting check, this pipeline produces strategies that are far more likely to survive the transition from backtest to live trading.