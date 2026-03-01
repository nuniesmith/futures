# Ruby v2 + Bridge — Backtest Guide

## Overview

This guide explains how to backtest the Ruby Opening Range Breakout (ORB) strategy
using NinjaTrader 8's Strategy Analyzer. Ruby (indicator) detects breakout signals
and pushes them through the **SignalBus** to Bridge (strategy), which executes
orders with risk sizing and bracket management.

```
Signal Flow (Backtest):

  Ruby.OnBarUpdate()
    → detects ORB breakout / ADD / EXIT
    → SignalBus.Enqueue(signal)

  Bridge.OnBarUpdate()
    → DrainSignalBus()
    → ExecuteEntryDirect() or ExecuteFlattenDirect()
    → SubmitOrderUnmanaged() with SL/TP brackets
    → Strategy Analyzer records trades
```

---

## Step 1: Compile All Three Files

In NinjaTrader 8:

1. Open **NinjaScript Editor** (Ctrl+Shift+E or Tools → NinjaScript Editor)
2. Ensure these three files exist in the correct locations:

| File            | NinjaTrader Location                                  |
|-----------------|-------------------------------------------------------|
| `SignalBus.cs`  | `NinjaScript/` (root — shared namespace)              |
| `Ruby.cs`       | `NinjaScript/Indicators/Ruby.cs`                      |
| `Bridge.cs`     | `NinjaScript/Strategies/Bridge.cs`                    |

3. Press **F5** to compile. Fix any errors before proceeding.

> **Important**: `SignalBus.cs` uses the `NinjaTrader.NinjaScript` namespace (no
> sub-namespace) so both the indicator and strategy can reference it. Place it in
> the NinjaScript root folder, not inside Indicators/ or Strategies/.

---

## Step 2: Open Strategy Analyzer

1. Go to **New** → **Strategy Analyzer** (or Ctrl+Shift+A)
2. In the strategy dropdown, select **Bridge**

---

## Step 3: Configure Bridge Properties

### 1. Account (Group 1)
| Property           | Recommended Value |
|--------------------|-------------------|
| Account to Monitor | `Sim101`          |

### 2. Web App (Group 2)
| Property             | Value                    |
|----------------------|--------------------------|
| Dashboard Base URL   | `http://localhost:8000`  |
| Signal Listener Port | `8080`                   |

> These don't matter for backtesting — HTTP is disabled during Historical state.

### 3. Options (Group 3)
| Property              | Value   | Notes                                      |
|-----------------------|---------|--------------------------------------------|
| Enable Position Push  | `true`  | No-op during backtest                      |
| Enable Auto Brackets  | `true`  | **Critical** — enables SL/TP bracket orders|
| Enable Risk Enforce   | `false` | No Python risk engine during backtest      |
| Default SL Ticks      | `20`    | Fallback if Ruby doesn't send SL price     |
| Default TP Ticks      | `40`    | Fallback if Ruby doesn't send TP price     |
| Enable SignalBus      | `true`  | **Critical** — this is the signal pathway  |
| Attach Ruby Indicator | `true`  | **Critical** — instantiates Ruby in backtest|

### 4. Risk Management (Group 4)
| Property            | Value   | Notes                                    |
|---------------------|---------|------------------------------------------|
| Account Size ($)    | `50000` | Starting simulated balance               |
| Risk % Per Trade    | `0.5`   | 0.5% = $250 risk per trade on $50k      |
| Max Micro Contracts | `5`     | Hard cap — Bridge won't exceed this      |

### 5. Ruby ORB (Group 5) — Forwarded to Ruby Indicator
| Property                   | Default | Tuning Notes                              |
|----------------------------|---------|-------------------------------------------|
| Ruby: Session Bias         | `Auto`  | Start with Auto, then test Long/Short     |
| Ruby: ORB Minutes          | `30`    | 15–45 min range. 30 is standard           |
| Ruby: Min Quality %        | `60`    | Higher = fewer but cleaner signals        |
| Ruby: Volume Gate (x avg)  | `1.2`   | Lower = more signals, higher = stricter   |
| Ruby: Require VWAP Cross   | `true`  | Key filter — keep on for first tests      |
| Ruby: Allow ADD Signal     | `true`  | Position building on pullback             |
| Ruby: SL ATR Mult          | `1.5`   | Stop loss distance in ATR multiples       |
| Ruby: TP1 ATR Mult         | `2.0`   | First target                              |
| Ruby: TP2 ATR Mult         | `3.5`   | Runner target                             |
| Ruby: Signal Cooldown (min)| `5`     | Min gap between signals                   |

---

## Step 4: Configure Data Settings

### Instrument
Pick the micro futures contract you trade:

| Symbol      | Name                  | Point Value | Tick Size |
|-------------|-----------------------|-------------|-----------|
| MGC         | Micro Gold            | $10         | 0.10      |
| MES         | Micro E-mini S&P 500  | $5          | 0.25      |
| MNQ         | Micro E-mini Nasdaq   | $2          | 0.25      |
| MCL         | Micro Crude Oil       | $10         | 0.01      |
| M6E         | Micro EUR/USD         | $12,500     | 0.0001    |

### Timeframe
**1 Minute** is the primary timeframe Ruby is designed for. It gives you
bar-by-bar resolution during the opening range period.

### Date Range
- **Start**: At least 20–60 trading days back
- **End**: Today or most recent trading day
- For initial testing, start with **5–10 days** to iterate quickly

### Session Template
Use the default session template for your instrument. Make sure it includes
the market open time:

| Market     | Open (ET)  | Best ORB Window         |
|------------|------------|-------------------------|
| CME Equity | 9:30 AM    | 9:30–10:00 AM           |
| CME Metals | 6:00 AM    | 6:00–6:30 AM (Globex)   |
| CME Energy | 6:00 AM    | 6:00–6:30 AM            |
| CME FX     | 6:00 PM    | 6:00–6:30 PM (Sun open) |

> **For MGC (your chart)**: The Globex session opens at 6:00 PM ET (Sunday)
> and the primary trading session starts at 8:20 AM ET. Your chart shows
> activity from 6:10 AM — this is likely the extended/Globex session.
> Adjust `ORB_Minutes` based on which session open you're targeting.

---

## Step 5: Run the Backtest

1. Click **Run** in the Strategy Analyzer
2. Watch the **Output Window** (Ctrl+Shift+O) for Bridge/Ruby log messages:

```
[Bridge] Ruby indicator attached for SignalBus integration
[Bridge] SignalBus consumer registered
[Ruby] ➤ BREAKOUT LONG → SignalBus | Q:72% W:1.8x SL:5195.20 TP1:5225.40 TP2:5248.00
[Bridge] SignalBus ENTRY: long strategy=Ruby:breakout Q=72% id=ruby-bl-20260228-080112-0001
[Bridge BT] LONG x3 (req=1, risk=3, cap=5) id=ruby-bl-20260228-080112-0001
[Bridge BT] Brackets: SL=5195.20 TP1=5225.40 TP2=5248.00
```

3. After completion, review:
   - **Summary** tab: Net profit, win rate, profit factor, max drawdown
   - **Trades** tab: Individual trade entries/exits with P&L
   - **Chart** tab: Visual display with Ruby's arrows and ORB zones

---

## Step 6: Analyze Results

### Key Metrics to Evaluate

| Metric              | Target        | What It Means                           |
|---------------------|---------------|-----------------------------------------|
| Profit Factor       | > 1.5         | Gross profit / gross loss               |
| Win Rate            | > 50%         | With 2:1 R:R, even 40% can be profitable|
| Max Drawdown        | < 10%         | Largest peak-to-trough equity drop      |
| Avg Winner / Loser  | > 1.5:1       | Your winners should be bigger than losers|
| Total Trades        | > 30          | Need enough samples for statistical sig |
| Sharpe Ratio        | > 1.0         | Risk-adjusted return                    |

### Common Issues

**No trades generated:**
- Check Output Window for errors
- Verify `EnableSignalBus = true` and `AttachRuby = true`
- Lower `ORB_MinQuality` to 40% temporarily to see if signals are being filtered
- Check that your date range covers market open hours

**Too many trades:**
- Increase `ORB_MinQuality` (try 70%)
- Increase `ORB_VolumeGate` (try 1.5)
- Increase `SignalCooldownMinutes`
- Set `ORB_AllowAdd = false` to disable position building

**Stops hit too often:**
- Increase `SL_ATR_Mult` (try 2.0 or 2.5)
- The ORB low/high is used as structural SL — if the range is too tight,
  try increasing `ORB_Minutes` to capture a wider range

**Targets never hit:**
- Decrease `TP1_ATR_Mult` (try 1.5 for quicker scalps)
- On quieter instruments, the ATR-based targets may be too ambitious

---

## Step 7: Optimization (Strategy Analyzer)

NinjaTrader's Strategy Analyzer supports parameter optimization:

1. Click the **Optimize** button instead of Run
2. Select parameters to optimize and their ranges:

### Recommended Optimization Parameters

| Parameter           | Min  | Max  | Step | Priority |
|---------------------|------|------|------|----------|
| RubyORB_Minutes     | 15   | 60   | 5    | High     |
| RubyORB_MinQuality  | 40   | 80   | 10   | High     |
| RubyORB_VolumeGate  | 0.8  | 2.0  | 0.2  | Medium   |
| RubySL_ATR_Mult     | 1.0  | 2.5  | 0.25 | High     |
| RubyTP1_ATR_Mult    | 1.5  | 3.0  | 0.5  | Medium   |
| RubyTP2_ATR_Mult    | 2.5  | 5.0  | 0.5  | Low      |

> **Warning**: Beware of overfitting. Use Walk-Forward optimization if available,
> or split your data into in-sample (70%) and out-of-sample (30%) periods.

---

## Research Topics for Improving the ORB Strategy

### 1. Timeframe Analysis

**Multi-timeframe confirmation** is the single biggest edge enhancer for ORB:

- **Higher timeframe trend**: Use a 15-min or 60-min EMA/VWAP slope to confirm
  the 1-min breakout direction. Don't take long breakouts if the 15-min trend
  is down.
- **Daily levels**: Prior day's high/low/close, weekly pivots. Breakouts that
  align with a break of the prior day's high have much higher follow-through.
- **Pre-market range**: The overnight/Globex range before RTH open. Breakouts
  of both the ORB *and* the pre-market range are the highest probability setups.

Research: *"Opening Range Breakout multi-timeframe filter"*, *Toby Crabel's work
on opening range patterns*.

### 2. Time-of-Day Filters

Not all breakouts are equal. Research shows:

- **First 30-60 minutes** after open have the highest volume and follow-through
- **10:00–10:30 AM ET** (for equities) often sees a reversal/retest
- **Lunch hour** (12:00–1:30 PM ET) breakouts have very low follow-through
- For **metals (MGC)**: The 8:20–9:00 AM ET window (US pre-market into equity
  open) often produces the day's biggest move

Consider adding a `MaxBreakoutHour` parameter to Ruby that stops looking for
breakouts after a certain time.

### 3. Volatility Regime Detection

- **VIX/GVZ correlation**: High implied volatility days have wider ORBs and
  bigger moves. Low vol days produce false breakouts.
- **ATR percentile**: If today's ATR(14) is in the bottom 20% of the last 60
  days, expect more false breakouts and tighten targets.
- **Bollinger Band squeeze**: A squeeze (BBs narrowing) before the open is a
  strong predictor that a breakout will have follow-through.

Research: *Bollinger Band squeeze breakout*, *TTM Squeeze indicator logic*,
*volatility contraction patterns (VCP)*.

### 4. Volume Profile & Market Structure

- **Prior day's POC**: If the breakout moves away from yesterday's POC, it has
  structural support. If it moves toward it, expect a pullback.
- **Value Area relationship**: Opening inside vs. outside the prior day's Value
  Area changes the expected move magnitude significantly.
- **Initial balance (IB)**: The high/low of the first 60 minutes. A break of the
  IB after an ORB breakout is a very high-conviction signal.

Research: *Market Profile (J. Peter Steidlmayer)*, *James Dalton's "Mind Over
Markets"*, *Volume Profile trading*.

### 5. Order Flow & Delta

- **Cumulative delta divergence**: If price breaks out up but delta is falling,
  the breakout is likely to fail. Ruby already tracks CVD — consider adding a
  delta confirmation gate to the breakout logic.
- **Absorption at ORB levels**: High volume with small body at the ORB high/low
  before breakout suggests institutional absorption (accumulation/distribution).
  Ruby's absorption detection could feed into the quality score.

Research: *Order flow trading*, *footprint charts*, *delta divergence*.

### 6. Mean Reversion Exits

Instead of fixed ATR targets, research:

- **VWAP bands**: Exit at the +1σ or +2σ VWAP band (Ruby already computes these)
- **Previous day's high/low**: Natural resistance/support levels for targets
- **Fibonacci extensions**: 1.272× and 1.618× extensions of the ORB range
- **Trailing stop**: After TP1 hits, trail the stop on the remaining position
  using the EMA9 or a chandelier stop

### 7. Session Bias Determination

Improving the `Auto` bias detection is high-value. Consider:

- **Overnight range direction**: Did price trend up or down during Globex?
- **Gap analysis**: Is price gapping up/down from prior close? Gap-and-go vs
  gap-fill have very different ORB characteristics.
- **Economic calendar**: Major news events (FOMC, NFP, CPI) change ORB behavior
  dramatically. Consider a "news mode" that widens the ORB and raises quality
  thresholds.
- **Correlation assets**: For MGC, check DXY (dollar index) direction. Gold and
  dollar are inversely correlated — a falling dollar supports long gold bias.

Research: *Gap trading strategies*, *news-based volatility filters*.

### 8. Position Sizing Refinement

- **Kelly Criterion**: Optimal position sizing based on win rate and avg win/loss
- **Volatility-adjusted sizing**: Smaller size on high-vol days, larger on low-vol
- **Scale-in logic**: The ADD signal is a start — research pyramiding strategies
  where each add is smaller than the previous entry

### 9. Walk-Forward Testing

Once you have promising parameters:

1. **In-sample**: Optimize on 60 days of data
2. **Out-of-sample**: Test on the next 20 days without changing parameters
3. **Roll forward**: Move the window and repeat
4. Only trust parameters that are profitable in both in-sample AND out-of-sample

### 10. Specific Instruments & Their Personalities

Each micro future has different ORB characteristics:

| Instrument | Typical ORB Range | Best ORB Period | Notes                        |
|------------|-------------------|-----------------|------------------------------|
| MGC        | $5–15             | 15–30 min       | Trends well, respects VWAP   |
| MES        | 5–15 pts          | 30 min          | Mean-reverts more than MGC   |
| MNQ        | 30–80 pts         | 15–30 min       | High momentum, wide ranges   |
| MCL        | $0.30–0.80        | 30–45 min       | News-driven, volatile        |
| M6E        | 15–40 pips        | 30–60 min       | Slower, more range-bound     |

---

## Recommended Workflow

1. **Start simple**: Backtest 5 days on MGC 1-min with all defaults
2. **Check Output Window**: Verify signals are flowing (Ruby → SignalBus → Bridge)
3. **Extend to 30 days**: Get a larger sample of trades
4. **Tune one parameter at a time**: Start with `ORB_Minutes` and `ORB_MinQuality`
5. **Test different biases**: Run Long-only, Short-only, and Auto separately
6. **Test different instruments**: Same parameters on MES, MNQ, MCL
7. **Optimize carefully**: Use Strategy Analyzer optimization on 2-3 key params
8. **Out-of-sample validate**: Never trust in-sample results alone
9. **Paper trade**: Run on Sim account with live data for 1-2 weeks
10. **Go live small**: Start with 1 micro contract, scale up only with proven edge

---

## File Reference

| File           | Type       | Purpose                                          |
|----------------|------------|--------------------------------------------------|
| `SignalBus.cs` | Shared     | In-memory ConcurrentQueue relay between Ruby → Bridge |
| `Ruby.cs`      | Indicator  | ORB detection, signal quality scoring, chart visuals  |
| `Bridge.cs`    | Strategy   | Order execution, risk sizing, bracket management      |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  Strategy Analyzer                    │
│                                                       │
│  ┌──────────┐   SignalBus    ┌──────────────────┐   │
│  │  Ruby    │──────────────→│     Bridge        │   │
│  │ (indic.) │  .Enqueue()   │   (strategy)      │   │
│  │          │               │                    │   │
│  │ ORB      │               │ DrainSignalBus()   │   │
│  │ detect   │               │ ExecuteEntryDirect │   │
│  │ quality  │               │ Risk sizing        │   │
│  │ scoring  │               │ SL/TP brackets     │   │
│  └──────────┘               └──────────────────┘   │
│       ↑                            ↓                 │
│   OnBarUpdate()            SubmitOrderUnmanaged()    │
│   (bar data)               (simulated fills)         │
│                                    ↓                 │
│                           Trade Log / Results        │
└─────────────────────────────────────────────────────┘
```
