# Ruby Futures — Backlog

> Deferred and post-launch work. Nothing here blocks the TPT $150K account go-live.
> Sequencing: CHARTS, POSINT, NEWS, and REDDIT can be built in parallel with the v8
> training run. TBRIDGE follows first live profits. Distillation and v9+ are long-horizon.
>
> **Extracted from**: `todo/` directory review — all prototype code, design blueprints,
> and rough notes consolidated into actionable phases below. See
> [`docs/todo_extracted_tasks.md`](todo_extracted_tasks.md) for full audit trail.

---

## Priority Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 | High value — build soon after launch |
| 🟡 | Medium value — nice to have within first month live |
| 🟢 | Low priority — scaling / future versions |

---

## 🔴 Phase POSINT — Position Intelligence Engine

> **The core "live trading co-pilot" feature.** Real-time per-position analysis using
> L1/L2 book data, DOM pressure, multi-TP zones, sweep-aware breakeven, and risk action
> recommendations. Builds with mock data first; swaps to real Rithmic feeds when creds arrive.
>
> **Source**: Extracted from `todo/position_engine.py` + `todo/live_page.html` prototypes.
> **Can be built in parallel with v8 training run.**

### Phase POSINT-A: Position Intelligence Module

- [ ] **`src/lib/services/engine/position_intelligence.py`** — new module
  - Port `compute_position_intelligence()` from prototype
  - Port helpers: `_compute_sweep_zones()`, `_compute_buffer_be()`,
    `_compute_tp_zones()`, `_compute_risk_actions()`, `_gen_live_analysis_signals()`
  - `SPECS` instrument dictionary (tick size, tick value, multiplier) — pull from
    `asset_registry.py` where possible, extend for micro contracts
  - Wire real analysis modules where prototype has `TODO` markers:
    - `ict.py` → liquidity levels for sweep zone cross-reference
    - `confluence.py` → scored confluence at each TP zone
    - `volume_profile.py` → VAH/VAL/POC as TP targets
    - `cvd.py` → real CVD divergence detection (replace mock)
    - `regime.py` → current regime classification
    - `signal_quality.py` → entry/exit quality score
    - `mtf_analyzer.py` → real-time bias signal
  - Input: position dict, L1 quote, L2 book, optional plan data
  - Output: `PositionIntelligence` dataclass with all computed fields:
    - `pts`, `unrealized`, `book_pressure` (ABSORBING / DISTRIBUTED / OPPOSED)
    - `bid_ask_ratio`, `spread_ticks`, `liquidity_grade` (TIGHT / NORMAL / WIDE)
    - `iceberg_alert`, `large_orders` (DOM orders ≥ 50 contracts)
    - `sweep_zones` — thin DOM spots where stops could be run
    - `breakeven` — `{hard, soft, buffer}` (buffer = past nearest DOM cluster)
    - `tp_zones` — 4-tier TP targets (plan-aware + Fibonacci + liquidity pools)
    - `risk_actions` — ordered list of recommendations (HOLD / SCALE / MOVE STOP / EXIT)
    - `live_signals` — CVD, VWAP, DOM pressure, 15m momentum pills
  - Mock mode: when Rithmic is not connected, generate realistic demo data
    (functions already written in prototype: `_mock_positions()`, `_mock_l1()`,
    `_mock_l2()`, `_mock_trades_tape()`)

### Phase POSINT-B: Rithmic Position Engine Wrapper

- [ ] **`src/lib/services/engine/rithmic_position_engine.py`** — new module
  - Port `RithmicPositionEngine` class from prototype
  - Methods: `connect()`, `get_positions()`, `get_l1(symbol)`,
    `get_l2(symbol, depth=10)`, `get_recent_trades(symbol, n=20)`
  - Each method has a clear real-Rithmic call path (documented in prototype TODOs):
    ```
    get_positions() → rithmic_client.get_open_positions()
    get_l1()        → rithmic_client.get_best_bid_ask(symbol)
    get_l2()        → rithmic_client.get_depth_of_market(symbol, levels=depth)
    get_trades()    → rithmic_client.get_time_and_sales(symbol, count=n)
    ```
  - Mock fallbacks for each method (already implemented in prototype)
  - Connection state tracking: `connected`, `last_sync`, `error_count`
  - Auto-reconnect logic with exponential backoff

### Phase POSINT-C: Position Intelligence API Routes

- [ ] **Add routes** to `src/lib/services/data/api/pipeline.py` (or new `position_api.py`)
  - `GET /api/live/positions` — SSE stream of position intelligence (1.5s interval)
    - Port `position_intelligence_stream()` generator from prototype
    - Event types: `position_update` (full intel payload), `no_positions`, `error`
    - Each update includes: book, TPs, risk actions, live signals
  - `GET /api/live/book?symbol=MES` — snapshot of L1 + L2 book
  - `GET /api/live/tape?symbol=MES&n=20` — recent time & sales
  - `GET /api/live/positions/snapshot` — non-SSE position snapshot
  - Wire corresponding web service proxy routes in `src/lib/services/web/main.py`

### Phase POSINT-D: Live Page UI Enhancement

- [ ] **Update `static/trading.html`** — enhance the Live Trading page
  - Replace current simulated price chart with position intelligence cards
  - Port CSS from prototype: `.pos-intel-card`, `.dom-grid`, `.tp-zone`,
    `.risk-action`, `.live-sig-pill`, `.tape-entry`, etc.
  - Per-position card layout:
    - **Header**: symbol, direction badge, entry, live price, unrealized P&L, qty
    - **Col 1 — Book**: L1 bid/ask with size, spread, time & sales tape
    - **Col 2 — DOM**: visual depth-of-market ladder, bid/ask bars, sweep warnings
    - **Col 3 — TP Zones**: 4-tier targets with price, pts, $, R:R, % to exit
    - **Col 4 — Actions**: BE panel (hard/soft/buffer), risk recommendations,
      live analysis signal pills (CVD, VWAP, DOM, 15m momentum)
  - Session stats header: P&L, win rate, max DD, trades today, risk used %
  - Rithmic banner: demo mode when not connected, symbol selector dropdown
  - No-position state: "Waiting for positions..." with instrument quick-select
  - Wire to `GET /api/live/positions` SSE stream for real-time updates

---

## 🔴 Phase NEWS — News Sentiment Pipeline

> Multi-source news sentiment scoring for the morning research workflow.
> Combines Finnhub (free, high-volume), Alpha Vantage (AI-scored), and Grok 4.1
> (context-aware) into a weighted hybrid sentiment score per asset.
>
> **Source**: Extracted from `todo/data_news.md` prototype scripts.
> **Complements**: Phase REDDIT (Reddit sentiment). Both feed the Research page.
> **Can be built in parallel with v8 training run.**

### Phase NEWS-A: News Data Collector

- [ ] **`src/lib/integrations/news_client.py`** — new module
  - `FinnhubClient` class:
    - `fetch_general_news(category='general')` → market-moving headlines
    - `fetch_company_news(ticker, days_back=7)` → per-ticker news
      (uses proxy tickers: `MES→SPY`, `MCL→USO`, `MGC→GLD`, `MNQ→QQQ`)
    - Rate limit: 60 calls/min (generous — no throttling needed)
    - Auth: `FINNHUB_API_KEY` env var
  - `AlphaVantageClient` class:
    - `fetch_news_sentiment(tickers='CL,GC,ES', topics='energy,commodities')`
      → articles with AI sentiment scores (0–1 scale)
    - `fetch_commodity_price(commodity='WTI', interval='daily')` → price data
      (direct futures proxies: WTI, BRENT, GOLD, NATURAL_GAS, COPPER)
    - Rate limit: 25 calls/day (tight — cache aggressively, fetch once/day)
    - Auth: `ALPHA_VANTAGE_API_KEY` env var
  - Add `finnhub-python` to `pyproject.toml` dependencies

### Phase NEWS-B: Hybrid Sentiment Scorer

- [ ] **`src/lib/analysis/news_sentiment.py`** — new module
  - **VADER with futures-specific lexicon**:
    - Custom additions: `surge: 3.0`, `rally: 2.8`, `short squeeze: 3.5`,
      `plunge: -3.0`, `crash: -3.5`, `glut: -2.5`, `rate hike: -2.0`,
      `inventory draw: 2.0`, `inventory build: -1.5`, `opec: 0.0` (full
      lexicon in `docs/todo_extracted_tasks.md`)
    - `vader_score(text)` → float [-1, +1]
    - Add `vaderSentiment` to `pyproject.toml` dependencies
  - **Grok sentiment scoring** (uses existing `grok_helper.py` infrastructure):
    - `grok_futures_sentiment(headline, summary, ticker)` → `(score, label, reason)`
    - Prompt: "You are a professional futures trader specializing in {ticker}…"
    - Model: `grok-4-1-fast-reasoning` (~$0.01 per 100 articles)
    - Cost optimization: only call Grok on ambiguous articles where
      `abs(vader_score) < 0.3` — VADER handles the obvious ones for free
  - **Hybrid score** (weighted combination):
    - `hybrid = 0.4 × vader + 0.4 × alpha_vantage_score + 0.2 × grok_score`
    - `compute_news_sentiment(symbol)` → `NewsSentiment` dataclass
    - `compute_all_news_sentiments()` → dict of symbol → `NewsSentiment`

### Phase NEWS-C: Scheduler Integration + Caching

- [ ] **Engine scheduler integration**:
  - `CHECK_NEWS_SENTIMENT` action — run at 07:00 ET (pre-market) and
    12:00 ET (midday refresh)
  - Cache in Redis: `engine:news_sentiment:<SYMBOL>` (2-hour TTL)
  - Store daily aggregates in Postgres `news_sentiment_history` table
  - Budget: Finnhub daily (high volume OK), Alpha Vantage once/day
    (25 call limit), Grok on top 50 ambiguous articles only
- [ ] **Spike detection**: if article count in last hour > 3× rolling average
  → publish `engine:news_spike` SSE event
  - Dashboard: "📰 News Spike: MCL — 12 articles/hr, sentiment -0.6 (bearish)"
- [ ] **Historical tracking**: daily aggregate rows in Postgres enable
  backtesting correlation between news sentiment and breakout outcomes

### Phase NEWS-D: Dashboard Integration

- [ ] **News panel on Research / Morning page**:
  - Top headlines with sentiment badges (🟢 bullish / 🔴 bearish / ⚪ neutral)
  - Hybrid sentiment score bar per focus asset
  - Grok AI reason tooltip on hover
  - "News Pulse" strip alongside Reddit sentiment and risk strip
- [ ] **API routes**:
  - `GET /api/news/sentiment?symbols=MES,MGC,MCL` → aggregated sentiment/symbol
  - `GET /api/news/headlines?symbol=MES&limit=10` → recent headlines with scores
- [ ] **Wire web service proxies** in `src/lib/services/web/main.py`

---

## 🔴 Phase UI-ENHANCE — Trading Dashboard Improvements

> Polish items from the UI design blueprint (`todo/trading_webui_review.md`) not yet
> implemented. The current `static/trading.html` has a 5-step flow; the full blueprint
> describes 6 pages with additional panels and UX features.
>
> **Source**: Extracted from `todo/trading_webui_review.md` + `todo/trading-dashboard.jsx`.

### Phase UI-A: Research Page Enhancements

- [ ] **Cross-asset context panel**:
  - ES/NQ/RTY correlation mini-heatmap
  - DXY, VIX, yields as leading indicator badges
  - Wire to existing `cross_asset.py` module
  - API: `GET /api/analysis/cross_asset` (may already exist)
- [ ] **Economic calendar integration**:
  - Free source: Forex Factory RSS or TradingEconomics free API
  - Show today's high-impact events: time, label, impact level, expected, previous
  - Warn on Plan page if trading during CPI/FOMC/NFP release
- [ ] **Combined sentiment gauges** (once Phase REDDIT + Phase NEWS are built):
  - Reddit sentiment bar + News sentiment bar
  - "Market Mood" gauge: bullish ◄──────► bearish

### Phase UI-B: Analysis Page Enhancements

- [ ] **Asset fingerprint display**:
  - Wire `asset_fingerprint.py` output to Analysis step in pipeline
  - Show: "This instrument tends to: run stops at open, respect VWAP…"
  - Optional: "Asset DNA" radar chart (6 fingerprint dimensions)
- [ ] **Wave structure panel**:
  - Wire `wave_analysis.py` + `swing_detector.py` output to pipeline
  - Show labeled swing highs/lows + current wave count
- [ ] **Asset selection output**:
  - After Analysis step, user picks 1–2 focus assets
  - Selection persists as session state, filters downstream pages

### Phase UI-C: Plan Page Enhancements

- [ ] **Range builders status panel**:
  - Wire `rb/detector.py` output to plan data
  - Show: current range boundaries, breakout direction, targets
- [ ] **Backtest validation button**:
  - "Backtest this level" on each entry zone → wire to `backtesting.py`
  - Show: "This setup type has hit T1 72% of the time over last 90 days"
- [ ] **CNN confidence badge on levels**:
  - CNN breakout probability next to each zone in the plan
  - Wire to `breakout_cnn.py` inference output from engine
- [ ] **ORB levels in plan**:
  - Surface ORB high/low from pipeline "orb" step into plan zones
  - Display as a level category alongside ICT, volume profile, etc.

### Phase UI-D: Journal Page Enhancements

- [ ] **Auto-populate from Rithmic fills** (when creds arrive):
  - Pre-fill trade log entries from `position_manager.py` fill data
  - User adds: was it planned? entry/exit reason, grade, notes
- [ ] **Plan adherence scoring**:
  - Compare each trade's entry price to locked plan zones
  - Auto-tag: "Trade 1: MES Long 5,784 ✅ Matched Zone A"
  - Session adherence score: % of trades that matched the plan
- [ ] **Session stats panel**:
  - Net P&L, win rate, avg R:R, best trade, worst trade
  - Equity curve mini-chart (small area chart of session P&L over time)

### Phase UI-E: UX Polish

- [ ] **Keyboard shortcuts**: `1-5` to jump between pages, `Space` to lock plan
- [ ] **One-click copy**: every level/price copies to clipboard on click
  (for pasting into MotiveWave)
- [ ] **Progress indicator in nav**: `Research ✅ → Analysis ✅ → Plan ✅ → Live ● → Journal`
- [ ] **Mobile-friendly live page**: responsive layout for phone during trades
- [ ] **Dark terminal theme refinements**:
  - Add `DM Sans` font import for labels alongside `JetBrains Mono` for prices
  - Level line color coding: ICT = purple, volume profile = blue, range = amber

---

## ✅ Phase CHARTS — Charting Service (ApexCharts, port 8003)

> **Completed.** The charting service is a standalone nginx + ApexCharts SPA in
> `docker/charting/` (port 8003). It is NOT a Lightweight Charts rewrite — it uses
> ApexCharts (already integrated and battle-tested). The service fetches bars directly
> from the data service (`/bars/{symbol}`) and receives live forming-candle updates via
> `/sse/dashboard`. All CHARTS-E volume indicators have been implemented.
>
> **Files changed (Phase CHARTS-E):**
> - `docker/charting/static/chart.js` — VWAP σ-bands, CVD sub-pane, Volume Profile,
>   Anchored VWAP (session + prev-day), localStorage persistence
> - `docker/charting/static/index.html` — CVD / VP / AVWAP-S / AVWAP-P toggle buttons,
>   `#chart-cvd` sub-pane div
> - `docker/charting/static/style.css` — `.chart-cvd` pane rules, per-indicator colours

### ~~Phase CHARTS-A: Bars SSE Endpoint~~ — N/A

> The standalone charting service connects directly to `/sse/dashboard` (not a
> per-symbol SSE endpoint). Live bar updates are received via the existing
> `bars_update` / `focus` event shapes in `handleSsePayload()`. No new SSE
> endpoint was needed.

### ~~Phase CHARTS-B: Chart Data Shaping Endpoint~~ — N/A

> The charting SPA calls `/bars/{symbol}?interval=X&days_back=Y` directly and
> reshapes the split-orient response in `splitToApex()` in `chart.js`. No Python
> shaping layer is needed.

### ~~Phase CHARTS-C: Charts Page Rewrite~~ — N/A

> `charts_page()` in `dashboard.py` already links to the standalone charting
> service at port 8003 via an iframe / direct link. The full SPA (topbar, symbol
> tabs, interval tabs, indicator toggles, live toggle, RSI + CVD sub-panes) lives
> in `docker/charting/static/`.

### ~~Phase CHARTS-D: Navigation Wire-Up~~ — ✅ Already wired

> `📈 Charts` nav link exists in `_SHARED_NAV_LINKS` → `/charts`.
> `charts_page_route()` is registered at `GET /charts` in `dashboard.py`.
> The charting service runs on port 8003 and is reachable directly by the browser.

### ✅ Phase CHARTS-E: Volume Indicators & UX Polish

> **Completed.** All items implemented in `docker/charting/static/chart.js`,
> `index.html`, and `style.css`.

#### ✅ VWAP Standard-Deviation Bands (±1σ / ±2σ)

- `calcVWAP()` now returns `{ vwap, upper1, lower1, upper2, lower2 }` — accumulates
  `cumTypSqV` alongside `cumTPV`/`cumVol` for running variance.
- Series slots `IDX.VWAP_U1/L1/U2/L2` (8–11) added to `buildSeries()`.
- `buildOptions()` extended: `stroke.dashArray` `[3,3,6,6]` for bands; `colors` array
  uses `C.vwapBand1` (±1σ) and `C.vwapBand2` (±2σ); 4 hidden `overlayYaxis` slots added.
- `updateIndicatorPoint()` recomputes session σ incrementally on every live tick.
- Tooltip `indLookup` shows `VWAP+1σ` / `VWAP-1σ` rows when VWAP is active.
- VWAP toggle handler calls `recalcSingleIndicator("vwap")` which updates all 5 series.

#### ✅ CVD — Cumulative Volume Delta

- `calcCVD(candles, volumes)` — bar approximation, daily reset, returns
  `[{x, y: Math.round(cvd), fillColor}]`.
- `state.cvdData`, `liveInd.cvdRunning/cvdLastDay` added.
- `recalcIndicators()` / `recalcSingleIndicator("cvd")` wired.
- `buildCvdOptions()`, `mountCvdChart()`, `unmountCvdChart()`, `syncCvdPane()` added —
  mirrors RSI pane lifecycle exactly.
- `applyLiveBar()` calls `syncCvdPane()` on every SSE tick.
- `<div id="chart-cvd" class="chart-cvd hidden">` added to `index.html`.
- `.chart-cvd` / `.chart-cvd.hidden` added to `style.css` (`flex: 0 0 120px`).
- `<button data-ind="cvd">CVD</button>` added to indicator-tabs; emerald active colour.

#### ✅ Volume Profile — POC / VAH / VAL

- `calcVolumeProfile(candles, volumes, bins=40, lookback=100)` — rolling 100-bar window,
  proportional bin distribution, 70% value-area expansion. Returns `{poc, vah, val}`.
- Series slots `IDX.POC/VAH/VAL` (12–14) added; POC=amber solid, VAH/VAL=indigo dashed.
- `state.pocData/vahData/valData` wired through `recalcIndicators()` and `recalcSingleIndicator("vp")`.
- VP recalc is skipped on forming-candle ticks (only runs on `isNewCandle`).
- `<button data-ind="vp">VP</button>` added; amber active colour.

#### ✅ Anchored VWAP

- `calcAnchoredVWAP(candles, volumes, anchorIndex)` — returns `y: null` before anchor.
- `findSessionAnchor(candles)` — first bar of current calendar day.
- `findPrevDayAnchor(candles)` — lowest-low bar of the previous calendar day.
- Series slots `IDX.AVWAP_S/AVWAP_P` (15–16); session=orange, prev-day=fuchsia.
- `state.avwapSessionData/avwapPrevDayData` wired; session anchor extended incrementally
  in `updateIndicatorPoint()`.
- `AVWAP-S` and `AVWAP-P` toggle buttons added; per-indicator active colours in CSS.

#### ✅ UX: Indicator Toggle Persistence

- `LS_KEY = "ruby_chart_indicators"` constant defined.
- `saveIndicatorPrefs()` — writes `state.indicators` to localStorage on every toggle.
- `loadIndicatorPrefs()` — restores saved flags (forward-compat: only known keys merged).
- `boot()` calls `loadIndicatorPrefs()` before `wireControls()` so button `active`
  classes reflect restored state from the first paint.

#### ⬜ Per-indicator configuration *(stretch — deferred)*

- Period inputs (EMA period, BB period, VP lookback/bins) via settings popover.
- Persist to localStorage alongside toggle flags.

---

## 🔴 Phase REDDIT — Reddit Sentiment Integration

> Monitor futures-relevant subreddits for crowd sentiment, activity spikes, and contrarian
> signals. Feeds into the dashboard as an additional metric layer. Can optionally influence
> CNN feature weighting in v9+.
>
> **Can be built in parallel with the v8 GPU training run (Week 2).**

### Phase REDDIT-A: Reddit Data Collector

- [ ] **`src/lib/integrations/reddit_client.py`** — Reddit API client using PRAW
  - Add `praw>=7.7.0` to `pyproject.toml` dependencies
  - Auth: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` env vars
  - Target subreddits:
    - `r/FuturesTrading` — direct futures discussion (gold, NQ, ES, 6E)
    - `r/Daytrading` — general day trading sentiment + setups
    - `r/wallstreetbets` — crowd euphoria/panic indicator (contrarian signal)
    - `r/InnerCircleTraders` — ICT/SMC methodology discussion
  - Methods:
    - `fetch_hot_posts(subreddit, limit=25)` → list of posts with title, score,
      num_comments, created_utc
    - `fetch_new_posts(subreddit, limit=50)` → newest posts for velocity tracking
    - `fetch_comments(post_id, limit=100)` → top comments for deeper sentiment
    - `search_posts(subreddit, query, time_filter="day")` → search by ticker/term

### Phase REDDIT-B: Sentiment Analysis Engine

- [ ] **`src/lib/analysis/reddit_sentiment.py`** — NLP sentiment scoring
  - Keyword extraction for futures tickers: `MGC`, `gold`, `GC`, `NQ`, `MNQ`, `ES`,
    `MES`, `6E`, `euro`, `BTC`, `ETH`, `SOL`
  - Sentiment classification per post/comment:
    - Rule-based first pass: bullish/bearish keyword lists + emoji patterns
      (🚀, 🐻, 💎🙌, etc.)
    - Optional: FinBERT or `distilbert-base-uncased-finetuned-sst-2-english` for
      nuanced scoring — gated behind `ENABLE_REDDIT_NLP=1` env var
  - Aggregate metrics per asset per subreddit:
    - `mention_count_1h` — ticker mentions in last hour
    - `mention_velocity` — rate of change (spike detection)
    - `avg_sentiment` — mean sentiment score [-1.0, +1.0]
    - `sentiment_skew` — bullish vs bearish ratio (extreme = contrarian signal)
    - `engagement_score` — weighted by upvotes + comment count
    - `wsb_euphoria_index` — WSB-specific: extreme bullishness = potential top (contrarian)
  - `compute_reddit_sentiment(symbol)` → `RedditSentiment` dataclass
  - `compute_all_reddit_sentiments()` → dict of symbol → `RedditSentiment`

### Phase REDDIT-C: Scheduler Integration + Caching

- [ ] **Engine scheduler**: poll Reddit every 15 min during ACTIVE + EVENING session modes
  - `CHECK_REDDIT_SENTIMENT` action in scheduler dispatch table
  - Results cached in Redis: `engine:reddit_sentiment:<SYMBOL>` (30-min TTL)
  - Rate limit: Reddit OAuth allows ~60 req/min — budget across 4 subreddits
- [ ] **Spike detection**: if `mention_velocity` > 3× rolling average
  → publish `engine:reddit_spike` SSE event
  - Dashboard shows: "🔥 Reddit Spike: MGC mentioned 47× in last hour (3.2× normal)"
- [ ] **Historical tracking**: store daily aggregates in Postgres
  `reddit_sentiment_history` table — enables backtesting correlation

### Phase REDDIT-D: Dashboard Integration

- [ ] **Reddit Sentiment Panel** on dashboard
  - Per-asset sentiment bar (green=bullish, red=bearish, grey=neutral) + mention count
  - "Reddit Pulse" strip alongside risk strip — top 4 focus assets' sentiment
  - Subreddit breakdown on hover
  - Spike alerts in market events feed (SSE `reddit-spike` listener)
  - Historical sentiment chart (7-day rolling) on focus cards
- [ ] **Focus card sentiment badge**
  - 🟢 "Reddit Bullish (0.72)" / 🔴 "Reddit Bearish (-0.45)" / ⚪ "Quiet"
  - Contrarian warning when WSB euphoria is extreme:
    "⚠️ WSB extremely bullish — consider fade"

### Phase REDDIT-E: CNN Feature Integration (v9+ — optional)

> Only add if backtesting shows >2% validation accuracy lift.

- [ ] **Correlation study**: backtest Reddit sentiment vs breakout outcomes over 90 days
  before adding to model
- [ ] **2 new tabular features** for v9 (if study passes):
  - `reddit_mention_velocity_norm` — normalized mention spike score [0, 1]
  - `reddit_sentiment_score` — aggregated sentiment across all tracked subs [0, 1]

---

## 🔴 Phase TV — TradingView Integration (Reference Overlay Only)

> TradingView is used as a **reference overlay for live price action only**.
> No position sendback. No order execution. All trading decisions happen in Tradovate,
> informed by the Python dashboard.

### Phase TV-B: Ruby Futures Indicator — Engine Signal Overlay

- [ ] Parse signals CSV: filter to current chart symbol + recent timestamps (last 5 bars)
- [ ] Draw on chart: entry line (dashed), stop line (red), TP1/TP2/TP3 (green dashes)
- [ ] Signal label box: breakout type name, CNN probability, contract sizing
  - "3× MGC ($330 risk) / 1× GC ($1,100 risk)"
- [ ] Colour-code by direction: green labels for LONG, red for SHORT
- [ ] Only show signals from last N hours (configurable input, default 4h)

### Phase TV-C: Ruby Futures Indicator — Core Futures Layer

- [ ] ORB box: session open high/low as shaded rectangle, extends until broken
- [ ] PDR: prior day high/low as dashed lines (extend right across chart)
- [ ] Initial Balance: first 60-min RTH high/low
- [ ] Asian range: 19:00–02:00 ET H/L as background shading
- [ ] VWAP: session VWAP line (standard Pine `ta.vwap`)
- [ ] EMA 9/21/50 on chart (toggleable)
- [ ] Session separators with session name labels (London, NY, etc.)
- [ ] Futures contract info panel (input): symbol, micro point value, tick size
  → shows ATR in ticks and dollars

### Phase TV-D: TradingView → Python Engine Webhook

- [ ] **Add `POST /api/tv/alert` endpoint** to data service
  - TV alert message format:
    `{"symbol": "MGC", "action": "LONG_ENTRY", "price": 2891.5, "note": "ORB breakout"}`
  - Engine logs alert, triggers fresh CNN inference on symbol, pushes result to dashboard
    via `dashboard:tv_alert` SSE channel
  - Auth: `TV_WEBHOOK_SECRET` env var as query param or header
  - Informational only — no order execution

### Phase TV-E: Dashboard + TradingView Side-by-Side Workflow

- [ ] Document and validate the full manual trading workflow:
  - Left monitor: TradingView with Ruby Futures indicator (reference only)
  - Right monitor: Python dashboard (focus mode, risk strip, swing signals, sentiment)
  - Pre-market: dashboard daily bias + Grok brief + Reddit sentiment → informs watchlist
  - All execution is manual via Tradovate, informed by dashboard
  - Zero dependency on NinjaTrader or Windows

---

## 🔴 Phase TBRIDGE — Tradovate JavaScript Bridge

> Replaces the TradingView position sendback approach entirely. Only one bridge
> connection needed (leader account) — PickMyTrade copies all trades to remaining accounts.

### Phase TBRIDGE-A: Tradovate API Client (JavaScript/Node.js)

- [ ] **Research Tradovate API docs** — REST endpoints for order placement, position query,
  account info
  - Auth flow: OAuth2 token → WebSocket session → order commands
  - Rate limits and order types (market, limit, stop-market, bracket)
- [ ] **`bridge/tradovate_client.js`** — Node.js client for Tradovate REST + WebSocket
  - `authenticate(credentials)` → access token
  - `placeOrder({symbol, action, qty, orderType, price?, stopPrice?})` → order ID
  - `getPositions()` → open positions array
  - `cancelOrder(orderId)` → confirmation
  - `flattenAll()` → close all positions
  - WebSocket: real-time fill notifications, position updates, P&L streaming
- [ ] **Environment config**: `TRADOVATE_USERNAME`, `TRADOVATE_PASSWORD`,
  `TRADOVATE_APP_ID`, `TRADOVATE_CID`, `TRADOVATE_SECRET` — all via env vars, never hardcoded

### Phase TBRIDGE-B: Python ↔ Node.js Bridge

- [ ] **`POST /api/bridge/order`** — Python engine sends order intent to Node.js bridge
  - Bridge runs as a sidecar container or subprocess
  - Communication via HTTP (localhost) or Redis pub/sub
  - Python publishes `OrderCommand` → bridge translates to Tradovate API call
  - Bridge publishes fill confirmations → Python `PositionManager` updates
- [ ] **Position sync**: bridge polls Tradovate positions every 5s
  → publishes to `engine:live_positions` Redis key
  (dashboard live position overlay already reads from this key)
- [ ] **Health monitoring**: bridge heartbeat → `broker_heartbeat` Redis key (30s TTL)
  (`positions.py` already reads this key for dashboard broker-connected indicator)

### Phase TBRIDGE-C: PickMyTrade Integration

- [ ] Wire bridge to 1st Tradovate account only (leader account)
- [ ] **PickMyTrade config**: connect all remaining TPT/Apex accounts as followers
  - Verify webhook latency (bridge fill → PickMyTrade copy → follower fill)
  - Test quantity multiplier config for different account sizes ($150K vs $300K)
  - Confirm simultaneous connection of all follower accounts
- [ ] **Failsafe**: if bridge disconnects, dashboard shows alert + blocks new signals
  - Manual trading via Tradovate UI remains available as fallback

---

## 🟡 Phase v8-F — Per-Asset Distillation (Optional)

> Only pursue if the unified v8 model fails the gate check (< 89% accuracy).
> If unified passes, skip this entirely.

```
Step 1: Train per-asset specialists
  MGC → train 80 epochs → best_mgc.pt  (gate: ≥75% acc)
  MNQ → train 80 epochs → best_mnq.pt
  MES → train 80 epochs → best_mes.pt
  ... 7 core assets ...

Step 2: Distill into champion
  All qualified teachers (≥75% acc) → DistillationTrainer → champ_v8.pt

Step 3: Compare
  champ_v8.pt vs best unified v8 → pick winner
```

- [ ] **`scripts/train_per_asset.py`** — loop over `['MGC', 'MNQ', 'MES', 'MYM', 'M2K', 'MBT', 'MET']`
  - Each: generate asset-specific dataset → train → gate (≥75% acc) → save
    `models/per_asset/best_{symbol}.pt`
  - Write `models/per_asset/asset_results.json` manifest with per-asset metrics
- [ ] **`scripts/distill_combined.py`** — knowledge distillation
  - Load all qualified teacher `.pt` files (frozen)
  - Student = same `HybridBreakoutCNN` v8 architecture
  - `temperature=4.0`, `alpha=0.7` (70% KL divergence + 30% hard cross-entropy)
  - Save best student to `models/champ_v8_distilled.pt`
- [ ] **`scripts/run_full_pipeline.py`** — master orchestrator
  - Single command: generate → train unified → train per-asset → distill → compare → promote winner
  - Write `models/pipeline_summary.json` with all metrics + comparison

---

## 🟡 Phase 9A — Correlation Anomaly Detection (Post-Live)

> Deferred until v8 champion is live and profitable. Nice-to-have market regime overlay —
> does not affect model quality.

- [ ] Rolling correlation matrix across all 10 core assets (updated every 5 min)
- [ ] Compare 30-bar vs 200-bar baseline → anomaly score per pair
- [ ] Publish `engine:correlation_anomalies` → dashboard heatmap panel

---

## 🟢 Scaling & Copy Trading

> Start after first consistent profits on TPT account 1.

### PickMyTrade + Account Scaling

- [ ] Sign up for PickMyTrade, test Tradovate bridge → PickMyTrade copy on a single Apex eval account
  - Verify webhook latency for intraday futures
  - Test quantity multiplier config ($150K vs $300K)
  - Confirm 20 Apex accounts can connect simultaneously
- [ ] Wire Phase TBRIDGE-B as the leader account signal source
- [ ] Scale TPT to 5 accounts (pass eval on each, connect as followers)
- [ ] Scale Apex to 20 accounts progressively

---

## 🟢 Phase 6 — Kraken Spot Portfolio Management

- [ ] `lib/strategies/crypto/portfolio_manager.py` — target % allocations, rebalance logic
- [ ] Add `add_order()` and `cancel_order()` to `KrakenDataProvider`
- [ ] `CryptoPortfolioConfig`: target allocations, 5% rebalance threshold, 4h cooldown, DCA mode
- [ ] Dashboard: Kraken portfolio card with allocations vs targets, P&L, rebalance status
- [ ] Gated behind `ENABLE_KRAKEN_TRADING=1` env var

---

## 🟢 Crypto Momentum Wiring

- [ ] Wire `crypto_momentum_score` into engine live scoring pipeline
  (currently computed but not fed into decisions)
- [ ] Show crypto momentum indicator on focused asset cards

---

## 🟢 CNN v9+ Ideas (Post-Launch Research)

> Ideas to evaluate once v8 is live and generating real signal data.

- [ ] **Cross-attention fusion** in `HybridBreakoutCNN`:
  instead of `cat(img_features, tab_features)`, use a single cross-attention layer
  where tabular queries attend to image feature map
  — only worth pursuing if v8 accuracy stalls below 91%
- [ ] **Reddit features** (Phase REDDIT-E): `reddit_mention_velocity_norm` [37],
  `reddit_sentiment_score` [38] — only if 90-day backtest shows >2% lift
- [ ] **News sentiment features** (Phase NEWS):
  `news_hybrid_sentiment` [39], `news_mention_velocity` [40]
  — only if 90-day backtest shows >2% lift (same gating as Reddit features)
- [ ] **"Asset DNA" radar chart** on focus cards — visualise the 6 fingerprint features
  (typical range, session concentration, breakout follow-through, hurst, gap tendency,
  vol profile shape) as a mini radar overlay
- [ ] **`ORBSession` → `RBSession` bulk rename** — alias works, purely cosmetic;
  do as a single find-and-replace PR when convenient
- [ ] **Deprecate `orb.py`** `detect_opening_range_breakout()` and `ORBResult` —
  safe to remove once v8 is validated in production (unified detector handles all cases)

---

## Appendix: `todo/` Directory Extraction Audit

> All 13 files in the former `todo/` directory have been reviewed and their actionable
> content extracted into the phases above. Full audit trail with file-by-file disposition
> is in [`docs/todo_extracted_tasks.md`](todo_extracted_tasks.md).

| File | Disposition |
|------|-------------|
| `README.md` | ✅ Superseded by `pipeline.py` integration |
| `notes.md` | ✅ Original vision — fully implemented |
| `trading_webui_review.md` | → Phase UI-ENHANCE (A–E) |
| `app.py`, `app1.py` | ✅ Integrated into `pipeline.py`; position routes → Phase POSINT-C |
| `index.html` | ✅ Copied to `static/trading.html` |
| `live_page.html`, `live_page1.html` | → Phase POSINT-D |
| `trading-dashboard.jsx` | React prototype — mock data patterns noted for demo mode |
| `position_engine.py`, `position_engine1.py` | → Phase POSINT-A/B |
| `data_news.md` | → Phase NEWS (A–D) |
| `requirements.txt` | ✅ Deps already in project |