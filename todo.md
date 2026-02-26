# Futures Trading Co-Pilot — Master Task List

> **Generated:** 2026-02-26 | **Codebase:** 72 files, 29k LOC Python, 2 C# files
> **Goal:** Ship a stable, simple trading dashboard for a full live trading week
> **Stack:** FastAPI + Redis + Postgres + HTMX/Tailwind (replacing Streamlit) + NinjaTrader 8
> **Session Rules:** Pre-market 00:00–05:00 ET | Active 05:00–12:00 ET | Off-hours 12:00–00:00 ET

---

## How to Use This File

Each task is self-contained with context, affected files, acceptance criteria, and dependencies.
AI agents should read the **Context** block first, then implement against the **Acceptance Criteria**.
Tasks are ordered by priority within each workstream. Cross-workstream dependencies are noted.

---

## WS-1: Critical Bug Fixes (Do First)

### TASK-101: Fix Gold Price $100 Discrepancy
- **Priority:** 🔴 P0 — Data accuracy is non-negotiable for live trading
- **Context:** Dashboard showed Gold at ~5300 when actual price was ~5200. Root cause is likely the symbol mapping pulling GC (full-size, $100/pt multiplier) instead of MGC (micro, $10/pt) or applying a contract multiplier to targets/levels that shouldn't be scaled.
- **Files to investigate:**
  - `src/services/data/api/analysis.py` — check symbol mapping and any multiplier logic
  - `src/massive_client.py` — verify which symbol is requested for Gold
  - `src/engine.py` — check if target/level calculations apply contract multipliers
  - `src/grok_helper.py` — check if Grok context formatting scales prices
- **Acceptance Criteria:**
  - [ ] Gold symbol hardcoded to `MGC` (micro) everywhere in the trading pipeline
  - [ ] Raw price from Massive API matches NinjaTrader chart price exactly
  - [ ] TP/SL/entry levels computed from raw price with no contract multiplier scaling
  - [ ] Add a unit test: `test_gold_price_no_scaling()` that asserts MGC price passthrough
  - [ ] Force refresh after fix confirms correct price on dashboard
- **Dependencies:** None

### TASK-102: Fix NinjaTrader Bridge Crash (NullReferenceException)
- **Priority:** 🔴 P0 — NT8 closing itself kills the entire trading workflow
- **Context:** `FuturesBridgeStrategy.cs` crashes during Session Break events (~8:05 AM ET) when NT8 resets internal objects. `Close[0]`, `myAccount`, and `Positions` become null. Async HTTP calls and `dynamic` JSON parsing compound the issue with cross-thread exceptions.
- **Files:**
  - `src/ninjatrader/FuturesBridgeStrategy.cs` — full rewrite of error handling
  - `src/ninjatrader/LivePositionBridgeIndicator.cs` — same null-guard pattern needed
- **Changes Required:**
  - [ ] Guard every access to `Close[0]`, `myAccount`, `Positions`, `Instrument` with null checks
  - [ ] Wrap `OnPositionUpdate`, `SendPositionUpdate`, `OnBarUpdate` in try/catch
  - [ ] Throttle error logging (max 1 log per 15 seconds) to prevent log spam
  - [ ] Queue order submissions via `Queue<Action>` processed on main thread in `OnBarUpdate`
  - [ ] Check `State == State.Realtime` before any position/order operations
  - [ ] Use fire-and-forget `httpClient.PostAsync()` (NT8 hates await in OnBarUpdate/OnTick)
  - [ ] Dispose `httpClient` in `State.Terminated`
- **Acceptance Criteria:**
  - [ ] NT8 survives session break at 8:05 AM without crash
  - [ ] Strategy stays loaded through full trading day (5:00 AM–12:00 PM ET)
  - [ ] Test on Sim101 account for full session before live use
- **Dependencies:** None

### TASK-103: Fix Dashboard Clock Not Updating Without Page Refresh
- **Priority:** 🟡 P1 — Annoyance during active trading, but addressed fully by WS-3 migration
- **Context:** Streamlit renders server-side so the clock at the top freezes until next rerun. Short-term fix for Streamlit; long-term fix is the HTMX migration (WS-3) which handles this natively.
- **Files:** `src/services/web/app.py`
- **Short-term fix (if still on Streamlit):**
  - [ ] Add `st_autorefresh(interval=5000)` import and call for 5-second page reruns during active hours
  - [ ] Or use `@st.fragment` with a 5-second rerun on just the clock component
- **Long-term fix:** Handled by TASK-301 (HTMX dashboard with JS clock)
- **Dependencies:** None (short-term) or WS-3 (long-term)

---

## WS-2: Architecture — 3-Container Split

### TASK-201: Create Engine Container (Separate from Data Service)
- **Priority:** 🟡 P1 — Needed before off-hours automation and daily focus features
- **Context:** Currently data-service handles both data collection AND heavy FKS computation. Split into: `data-service` (thin: Massive WS + REST API + Redis/Postgres writes), `engine` (heavy: FKS modules, optimization, backtesting, daily focus computation, pattern detection), `web` (UI only).
- **Files to create:**
  - `docker/engine/Dockerfile` — Python 3.13-slim, copy src/, PYTHONPATH=/app/src
  - `docker/engine/entrypoint.sh` — `python -m src.services.engine.main`
  - `src/services/engine/__init__.py`
  - `src/services/engine/main.py` — async main loop with session-aware scheduling
- **Files to modify:**
  - `docker-compose.yml` — add engine service with depends_on redis, postgres, data-service
  - `src/services/data/main.py` — remove engine/FKS computation, keep only data collection + API
- **Acceptance Criteria:**
  - [ ] `docker-compose.yml` defines 5 services: postgres, redis, data-service, engine, web
  - [ ] `data-service` only runs: Massive WS listener, REST API endpoints, Redis/Postgres I/O
  - [ ] `engine` runs: DashboardEngine, FKS modules, optimization, backtesting, daily focus
  - [ ] Engine reads market data from Redis (written by data-service), writes results back to Redis
  - [ ] `docker compose up --build -d` starts all 5 services with healthchecks passing
  - [ ] Engine logs show session-aware behavior (active vs off-hours)
- **Dependencies:** None

### TASK-202: Implement Session-Aware Engine Scheduling
- **Priority:** 🟡 P1 — Core to the 24/7 operation model
- **Context:** Engine should behave differently based on ET time:
  - **00:00–05:00 (Pre-market):** Compute daily focus, run Grok morning briefing, prep alerts
  - **05:00–12:00 (Active):** Live FKS recomputation every 5 min, publish focus updates to Redis, 15-min Grok updates
  - **12:00–00:00 (Off-hours):** Historical data backfill, full optimization runs, backtesting, next-day prep
- **Files:**
  - `src/services/engine/main.py` — main async loop with hour-based branching
  - `src/services/engine/scheduler.py` — ScheduleManager class
- **Acceptance Criteria:**
  - [ ] Engine main loop checks `datetime.now(pytz.timezone('US/Eastern')).hour` each iteration
  - [ ] Pre-market: runs `compute_daily_focus()` once, runs Grok morning brief
  - [ ] Active: recomputes FKS every 5 min, publishes to Redis Stream, runs Grok every 15 min
  - [ ] Off-hours: runs historical backfill, optimization, backtesting sequentially
  - [ ] Massive WS data collection continues 24/7 in data-service (not affected by engine schedule)
  - [ ] Logs clearly indicate which session mode is active
- **Dependencies:** TASK-201

### TASK-203: Implement Daily Focus Computation
- **Priority:** 🟡 P1 — The core "what should I trade today" feature
- **Context:** Engine should compute a daily focus payload for the top 3-4 assets (from MGC, MNQ, MES, MCL, SIL, HG). Each asset gets: bias (LONG/SHORT/NEUTRAL), entry zone, stop, TP1/TP2, wave ratio, signal quality %, position size in micros, risk in dollars. This feeds the dashboard's "TODAY'S TRADING PLAN" view.
- **Files:**
  - `src/services/engine/focus.py` — `compute_daily_focus()` function
  - Uses: `src/wave_analysis.py`, `src/signal_quality.py`, `src/volatility.py`, `src/ict.py`, `src/confluence.py`
- **Acceptance Criteria:**
  - [ ] `compute_daily_focus()` returns list of dicts with keys: `symbol`, `bias`, `last_price`, `entry_low`, `entry_high`, `stop`, `tp1`, `tp2`, `wave_ratio`, `quality`, `position_size`, `risk_dollars`, `vol_percentile`
  - [ ] Bias derived from: wave dominance direction + AO confirmation + confluence score
  - [ ] Risk per trade capped at 0.75% of $50k = $375 max
  - [ ] Position size calculated from stop distance × tick value for micro contracts
  - [ ] Assets with quality < 55% flagged as NEUTRAL with "skip today" note
  - [ ] `should_not_trade()` returns True if ALL assets < 55% quality or max vol_percentile > 88%
  - [ ] Result written to Redis key `engine:daily_focus` as JSON
  - [ ] Add API endpoint `GET /api/focus` in data-service that reads from Redis
- **Dependencies:** TASK-201, TASK-202

### TASK-204: Historical Data Backfill (Off-Hours)
- **Priority:** 🟢 P2 — Improves optimization quality but not blocking for live week
- **Context:** During off-hours (12:00–00:00 ET), engine should fetch and store up to 1 year of historical 1-minute bars for active contracts (MGC, MNQ, MES, MCL, SIL, HG) into Postgres. Use Massive API if available, or fallback sources.
- **Files:**
  - `src/services/engine/backfill.py` — `fetch_historical_bars(symbol, days_back)`
  - `src/models.py` — add `historical_bars` table if not exists
  - `src/massive_client.py` — add historical data fetch method
- **Acceptance Criteria:**
  - [ ] Stores 1-min OHLCV bars in Postgres with symbol, timestamp, open, high, low, close, volume
  - [ ] Idempotent: skips already-stored bars, only fetches gaps
  - [ ] Runs only during off-hours (12:00–00:00 ET)
  - [ ] Logs progress: "MGC: backfilled 45,000 bars (2025-03-01 to 2026-02-26)"
  - [ ] Engine optimization and backtesting jobs can query this table
- **Dependencies:** TASK-201

---

## WS-3: Dashboard Migration (Streamlit → HTMX)

### TASK-301: Create Base HTMX Dashboard Template
- **Priority:** 🟡 P1 — Replaces Streamlit for a cleaner, faster trading UI
- **Context:** Replace 1638-line Streamlit app with a single `index.html` using FastAPI + Jinja2 + HTMX + Tailwind CSS. Dark theme, card-based layout. The dashboard serves from data-service on port 8000 (root `/`). This eliminates the need for the Streamlit container entirely.
- **Files to create:**
  - `src/services/data/templates/index.html` — main dashboard template
  - `src/services/data/templates/partials/` — folder for HTMX partial HTML fragments
  - `src/services/data/templates/partials/asset_card.html` — single asset focus card
  - `src/services/data/templates/partials/positions.html` — live positions panel
  - `src/services/data/templates/partials/alerts.html` — alerts panel
  - `src/services/data/templates/partials/no_trade.html` — NO TRADE banner
  - `src/services/data/api/dashboard.py` — new router for HTML-serving endpoints
- **Files to modify:**
  - `src/services/data/main.py` — add Jinja2Templates, mount static, add dashboard router
  - `requirements.txt` — add `jinja2>=3.1.0`
- **Tech Stack:**
  - HTMX 2.x (CDN) — partial page updates, SSE extension
  - Hyperscript 0.9.x (CDN) — inline conditional logic
  - Tailwind CSS (CDN) — dark trading theme
  - htmx-ext-sse, htmx-ext-idiomorph (CDN) — smooth live updates
- **Acceptance Criteria:**
  - [ ] `GET /` returns full HTML dashboard page
  - [ ] Dark theme (`bg-zinc-950 text-white`), responsive grid layout
  - [ ] Live clock updates via JS `setInterval` (1-second updates, no page refresh)
  - [ ] Session banner shows current mode: 🌙 PRE-MARKET / 🟢 ACTIVE / ⚙️ OFF-HOURS
  - [ ] Focus grid with 2-column card layout for assets
  - [ ] Each card shows: symbol, bias emoji (🟢/🔴/⚪), wave ratio, quality %, entry, stop, targets
  - [ ] NO TRADE banner appears when all quality < 55%
  - [ ] Market status indicators (open/closed/overlap) in header
  - [ ] Page works on localhost:8000 without Streamlit running
- **Dependencies:** TASK-203 (needs daily focus data)

### TASK-302: Implement SSE Endpoint for Live Dashboard Updates
- **Priority:** 🟡 P1 — Real-time updates without polling
- **Context:** Engine publishes focus updates to Redis Stream. Data-service subscribes and streams HTML fragments to browser via SSE. Browser uses HTMX SSE extension to swap card content.
- **Files to create:**
  - `src/services/data/api/sse.py` — SSE router with `/sse/dashboard` endpoint
  - `src/services/engine/publisher.py` — Redis Stream publisher with throttling
- **Architecture:**
  - Engine → `XADD dashboard:stream:focus` (durable) + `PUBLISH dashboard:live` (trigger)
  - Data-service SSE → on connect: `XREVRANGE` last 8 messages (catch-up), then subscribe to pub/sub for live
  - Browser → `hx-ext="sse" sse-connect="/sse/dashboard"` with per-asset event names
- **Acceptance Criteria:**
  - [ ] `GET /sse/dashboard` returns `text/event-stream` with proper headers
  - [ ] New browser tab immediately receives last 8 updates (catch-up from Redis Stream)
  - [ ] Live updates arrive within 1 second of engine publishing
  - [ ] Per-asset events: `mgc-update`, `mnq-update`, `mes-update`, `mcl-update`
  - [ ] Global events: `no-trade-alert`, `session-change`, `heartbeat`
  - [ ] Throttling: max 1 update per asset per 7 seconds
  - [ ] Auto-reconnect on disconnect (HTMX handles this natively)
  - [ ] Heartbeat every 30 seconds to keep connection alive
- **Dependencies:** TASK-201, TASK-203, TASK-301

### TASK-303: Build HTML Fragment Endpoints for HTMX Partials
- **Priority:** 🟡 P1 — Powers the HTMX partial swap pattern
- **Context:** HTMX fetches HTML fragments (not JSON) from the server. Each endpoint returns a rendered Jinja2 partial.
- **Files:**
  - `src/services/data/api/dashboard.py` — add fragment endpoints
- **Endpoints to create:**
  - `GET /api/focus` → returns all asset cards as HTML (for full grid swap)
  - `GET /api/focus/{symbol}` → returns single asset card HTML
  - `GET /api/positions/html` → returns live positions panel HTML
  - `GET /api/alerts/html` → returns alerts panel HTML
  - `GET /api/time` → returns formatted time string with session indicator
- **Acceptance Criteria:**
  - [ ] Each endpoint returns `text/html` content type
  - [ ] Templates use Jinja2 with Tailwind classes
  - [ ] Asset cards include data attributes for Hyperscript conditionals (`data-quality`, `data-wave`)
  - [ ] Positions panel shows: symbol, side, quantity, avg price, unrealized P&L, total risk %
  - [ ] Over-risk warning (>5% total) rendered inline with red styling
- **Dependencies:** TASK-301, TASK-203

### TASK-304: Retire Streamlit Container
- **Priority:** 🟢 P2 — After HTMX dashboard is stable
- **Context:** Once the HTMX dashboard (served from data-service) is working, remove the Streamlit container.
- **Files to modify:**
  - `docker-compose.yml` — remove streamlit-app service
  - `docker/web/` — archive or delete Dockerfile + entrypoint.sh
- **Files to keep (reference only):**
  - `src/services/web/app.py` — keep in repo as reference, add deprecation note at top
- **Acceptance Criteria:**
  - [ ] `docker-compose.yml` has 4 services: postgres, redis, data-service, engine
  - [ ] Dashboard accessible at `localhost:8000`
  - [ ] All dashboard functionality verified working without Streamlit
  - [ ] `requirements.txt` cleaned: remove `streamlit`, `streamlit-autorefresh`
- **Dependencies:** TASK-301, TASK-302, TASK-303 all verified working

---

## WS-4: NinjaTrader Indicator & Bridge

### TASK-401: Build FKS_Core NinjaTrader Indicator
- **Priority:** 🟡 P1 — Needed for visual confirmation while trading
- **Context:** Create a single NT8 indicator that mirrors the Python FKS analysis on-chart. Combines: EMA9 (blue), Bollinger Bands (upper red, mid magenta, lower green), volume-colored bars (green bullish, red bearish, orange spike), adaptive S/R, wave dominance display, AO-based buy/sell arrows with quality labels.
- **Files to create:**
  - `src/ninjatrader/FKS_Core.cs` — full indicator code
- **Components:**
  1. EMA(9) plotted as blue line (overlay)
  2. Bollinger(2, 20) with custom colors: Upper=Red, Middle=Magenta, Lower=LimeGreen
  3. Dynamic trend EMA (alpha = 2/(20+1)) for wave tracking
  4. Wave dominance: track bull/bear wave strengths on trend crossovers, compute ratio
  5. Adaptive S/R: MAX(High, 20) / MIN(Low, 20) / midpoint
  6. AO(5, 34) for momentum confirmation
  7. Signal quality score (0-100%): wave ratio contribution + AO + price vs mid
  8. Buy arrow when: Low touches support + AO bullish + wave ratio > threshold × 0.7
  9. Sell arrow when: High touches resistance + AO bearish + wave ratio > threshold × 0.7
  10. Volume bar coloring via BarBrush: green/lime(spike) for bullish, red/orangeRed(spike) for bearish
  11. Top-right text box: live wave ratio, signal quality %, AO value
  12. Candle outline heatmap: lime if above dynEMA, red if below
- **Acceptance Criteria:**
  - [ ] Compiles in NT8 without errors
  - [ ] All plots render correctly on MGC, MNQ, MES charts
  - [ ] Volume bars colored correctly (verify against manual close[0] > close[1] check)
  - [ ] Wave ratio display matches Python dashboard within ±0.1x
  - [ ] Buy/Sell arrows appear at reasonable locations (not spamming — 5-min cooldown)
  - [ ] Parameters exposed: SR_Lookback, AO_Fast, AO_Slow, WaveLookback, MinWaveRatio, ShowLabels
- **Dependencies:** None (standalone C# file)

### TASK-402: Harden LivePositionBridgeIndicator
- **Priority:** 🟡 P1 — Same crash pattern as TASK-102
- **Context:** Apply identical null-guard and try/catch patterns from TASK-102 to the indicator version.
- **Files:** `src/ninjatrader/LivePositionBridgeIndicator.cs`
- **Acceptance Criteria:**
  - [ ] All position access wrapped in null checks and try/catch
  - [ ] Throttled error logging (15-second minimum between logs)
  - [ ] Survives session break without crash
  - [ ] Position data still pushes correctly to dashboard API
- **Dependencies:** TASK-102 (same pattern)

### TASK-403: Add Dynamic Volume Analysis to FKS_Core
- **Priority:** 🟢 P2 — Enhancement for TP/SL decisions
- **Context:** Beyond coloring, add logic that suggests actions based on volume patterns: volume spike at BB band → move SL to breakeven; volume spike with trend → add to position; volume drying up → take profit. Display as text annotations or a separate panel.
- **Files:** `src/ninjatrader/FKS_Core.cs` (extend)
- **Acceptance Criteria:**
  - [ ] When volume > 1.8× avg AND price at upper BB: show "TP/BE" label
  - [ ] When volume > 1.8× avg AND price trending (above dynEMA + AO bullish): show "ADD" label
  - [ ] When volume < 0.5× avg for 3+ bars: show "LOW VOL" warning
  - [ ] Labels positioned clearly, not overlapping price action
- **Dependencies:** TASK-401

---

## WS-5: Live Positions & Risk Management

### TASK-501: Build Live Positions Dashboard Panel
- **Priority:** 🟡 P1 — Must-have for active trading hours
- **Context:** During active hours, the dashboard needs a persistent, prominent panel showing all open positions from NinjaTrader, with real-time P&L and total risk calculation. Data flows: NT8 Bridge → POST /positions/update → Redis → SSE → Dashboard.
- **Files:**
  - `src/services/data/templates/partials/positions.html` — positions panel template
  - `src/services/data/api/positions.py` — ensure position data cached in Redis for SSE
  - `src/services/data/api/dashboard.py` — add `/api/positions/html` endpoint
- **Acceptance Criteria:**
  - [ ] Panel shows: each position's symbol, LONG/SHORT, quantity, avg price, unrealized P&L ($)
  - [ ] Total risk % displayed prominently (sum of position risks / account value)
  - [ ] Red warning banner if total risk > 5% of account
  - [ ] Panel updates via SSE when NT8 bridge pushes new position data
  - [ ] Empty state: "No open positions" with green checkmark
  - [ ] Styled as a bordered card at top of dashboard during active hours
- **Dependencies:** TASK-102, TASK-301, TASK-302

### TASK-502: Implement Risk Rules Engine
- **Priority:** 🟡 P1 — Prevents overtrading
- **Context:** Automated risk checks that feed into the dashboard and alerts:
  - Max 2 open trades at once
  - Max risk per trade: $375 (0.75% of $50k)
  - Max daily loss: $500
  - No new entries after 10:00 AM ET
  - No overnight positions (force warning at 11:30 AM)
  - Micro contract stacking: add only if +0.5R and wave > 1.8x
- **Files:**
  - `src/services/engine/risk.py` — RiskManager class
- **Acceptance Criteria:**
  - [ ] `RiskManager.can_enter_trade(symbol, side, size)` → returns (bool, reason_string)
  - [ ] `RiskManager.get_status()` → returns dict with all current risk metrics
  - [ ] Checks: open trade count, per-trade risk, daily P&L, time-of-day, overnight check
  - [ ] Status published to Redis for dashboard consumption
  - [ ] Dashboard shows risk status in positions panel
- **Dependencies:** TASK-201, TASK-501

---

## WS-6: Grok AI Integration Improvements

### TASK-601: Simplify Grok Live Update Output
- **Priority:** 🟡 P1 — Current output is too verbose for active trading
- **Context:** Grok 15-minute updates during active hours contain too much information. Simplify to 3 sections max: (1) Status line per focus asset (price, bias still valid?), (2) Key level to watch right now, (3) "DO NOW" action in 1 sentence.
- **Files:**
  - `src/grok_helper.py` — modify `format_market_context()` and the live update prompt
- **Acceptance Criteria:**
  - [ ] Live update output is ≤8 lines total
  - [ ] Each focus asset gets 1 status line: `GOLD 🟢 5212 (+4) | Bias VALID | Watch 5225`
  - [ ] Single "DO NOW" line at bottom: actionable, clear, 1 sentence
  - [ ] Full verbose analysis still available via separate endpoint/toggle for pre-market
  - [ ] Morning briefing (pre-market) remains detailed
- **Dependencies:** None

### TASK-602: Integrate Grok Updates into SSE Stream
- **Priority:** 🟢 P2 — Nice polish for live dashboard
- **Context:** When engine runs a Grok 15-minute update, publish the simplified result to Redis Stream so it appears on the HTMX dashboard via SSE without refresh.
- **Files:**
  - `src/services/engine/publisher.py` — add Grok update event
  - `src/services/data/templates/partials/grok_update.html` — Grok panel partial
- **Acceptance Criteria:**
  - [ ] SSE event `grok-update` fires every 15 minutes during active hours
  - [ ] Dashboard panel shows latest Grok summary with timestamp
  - [ ] Old updates collapse into expandable history
- **Dependencies:** TASK-302, TASK-601

---

## WS-7: Data & Infrastructure

### TASK-701: Docker First Boot — Verify Full Stack
- **Priority:** 🔴 P0 — Must pass before any live trading
- **Context:** Run through the full deployment checklist to verify everything works end-to-end.
- **Checklist:**
  - [ ] `docker compose up -d --build` — all services start cleanly (check `docker compose ps`)
  - [ ] Postgres healthcheck passes (pg_isready)
  - [ ] Redis healthcheck passes (redis-cli ping)
  - [ ] Data-service healthcheck passes (`GET /health` returns 200)
  - [ ] Engine healthcheck passes (custom check or log-based)
  - [ ] Massive WS connects (check data-service logs for connection message)
  - [ ] Dashboard loads at `localhost:8000` (or `localhost:8501` if still on Streamlit)
  - [ ] Focus cards render with real data
  - [ ] "Force Refresh" button triggers data update
  - [ ] NT8 bridge sends position update → appears on dashboard
  - [ ] SSE connection stays alive for >5 minutes
  - [ ] No error spam in any container logs
- **Dependencies:** All WS-1 tasks, TASK-201

### TASK-702: SQLite → Postgres Migration
- **Priority:** 🟢 P2 — Needed for production persistence
- **Context:** `scripts/migrate_to_postgres.py` and `src/models.py` `migrate_sqlite_to_postgres()` are already written. Need to actually run and verify.
- **Files:** `scripts/migrate_to_postgres.py`, `data/futures_journal.db`
- **Acceptance Criteria:**
  - [ ] Run migration script: `python scripts/migrate_to_postgres.py`
  - [ ] All journal entries transferred (count matches)
  - [ ] All trade records transferred
  - [ ] Data-service confirmed reading from Postgres (check `DATABASE_URL` env var)
  - [ ] SQLite file kept as backup but no longer used
- **Dependencies:** TASK-701

### TASK-703: Add Rate Limiting to Data Service
- **Priority:** 🟢 P2 — Security hardening
- **Files:** `src/services/data/main.py`
- **Acceptance Criteria:**
  - [ ] Add `slowapi` or similar rate limiter
  - [ ] Public endpoints (`/health`, `/docs`): 60 req/min
  - [ ] API endpoints: 30 req/min per client
  - [ ] SSE endpoint: 5 connections per client
- **Dependencies:** None

### TASK-704: Add Prometheus Metrics Endpoint
- **Priority:** 🟢 P3 — Nice-to-have for monitoring
- **Files:** `src/services/data/api/health.py`
- **Acceptance Criteria:**
  - [ ] `GET /metrics` returns Prometheus-format text
  - [ ] Metrics: request count, latency histogram, active SSE connections, engine last refresh time
- **Dependencies:** None

---

## WS-8: Pattern Detection & Trading Intelligence

### TASK-801: Opening Range Breakout Detection
- **Priority:** 🟢 P2 — Matches Jordan's observed daily patterns
- **Context:** Jordan sees breakout patterns daily from market opens. Detect the opening range (first 30-60 min after 9:30 ET) and flag breakouts.
- **Files:**
  - `src/services/engine/patterns.py` — `detect_opening_range_breakout(bars_1m, symbol)`
- **Acceptance Criteria:**
  - [ ] Computes OR high/low from first 30 minutes of 1-min bars after 9:30 ET
  - [ ] Flags breakout when close > OR_high + 0.5 × ATR(14) or close < OR_low - 0.5 × ATR(14)
  - [ ] Returns: `{type: "ORB", direction: "LONG"/"SHORT", trigger_price, or_high, or_low, timestamp}`
  - [ ] Published to Redis → appears as alert on dashboard
- **Dependencies:** TASK-201, TASK-204 (uses historical bars for ATR)

### TASK-802: "Should Not Trade" Detector
- **Priority:** 🟡 P1 — Directly addresses "hard for me to figure when I should not trade"
- **Context:** Rules-based filter that flags low-conviction days to prevent overtrading.
- **Files:**
  - `src/services/engine/patterns.py` — `should_not_trade(focus_data)` function
- **Conditions for NO TRADE:**
  - [ ] All focus assets have quality < 55%
  - [ ] Any focus asset has volatility percentile > 88% (extreme vol, stops get hit easily)
  - [ ] Daily loss already exceeds -$250
  - [ ] More than 2 consecutive losing trades today
  - [ ] It's after 10:00 AM ET and no setups have triggered
- **Acceptance Criteria:**
  - [ ] Returns `(bool, reason_string)` 
  - [ ] Reason displayed on dashboard NO TRADE banner
  - [ ] Checked every engine cycle during active hours
  - [ ] Published to Redis → SSE `no-trade-alert` event
- **Dependencies:** TASK-203, TASK-502

---

## WS-9: Testing & Quality

### TASK-901: Add Integration Tests for New Architecture
- **Priority:** 🟡 P1 — Confidence for live trading week
- **Files:** `tests/test_integration.py` (new)
- **Tests to add:**
  - [ ] Engine writes daily focus to Redis → data-service reads it correctly
  - [ ] SSE endpoint streams events when Redis Stream has data
  - [ ] Position update POST → appears in GET /positions
  - [ ] Focus card HTML endpoint returns valid HTML with correct data attributes
  - [ ] Risk manager blocks trade when over limit
- **Dependencies:** TASK-201, TASK-203, TASK-302, TASK-502

### TASK-902: End-to-End Smoke Test Script
- **Priority:** 🟡 P1 — Run before every trading day
- **Files:** `scripts/smoke_test.sh` (new)
- **Script should:**
  - [ ] Check all Docker containers are running and healthy
  - [ ] Curl `/health` on data-service
  - [ ] Curl `/api/focus` and verify non-empty response
  - [ ] Curl `/sse/dashboard` and verify SSE headers
  - [ ] Check Redis has `engine:daily_focus` key
  - [ ] Check Postgres connection
  - [ ] Print PASS/FAIL summary
- **Dependencies:** TASK-701

---

## Execution Order (Recommended Sprint Plan)

### Day 1: Critical Fixes + Architecture Foundation
1. TASK-101 — Gold price fix (30 min)
2. TASK-102 — NT8 bridge crash fix (1 hr)
3. TASK-201 — Create engine container (2 hr)
4. TASK-701 — Docker first boot verification (1 hr)

### Day 2: Engine Features + Dashboard Start
5. TASK-202 — Session-aware scheduling (1.5 hr)
6. TASK-203 — Daily focus computation (2 hr)
7. TASK-301 — Base HTMX dashboard template (3 hr)

### Day 3: Live Data + NT8 Indicator
8. TASK-302 — SSE endpoint (2 hr)
9. TASK-303 — HTML fragment endpoints (1.5 hr)
10. TASK-401 — FKS_Core NT8 indicator (2 hr)
11. TASK-402 — Harden LivePositionBridge (30 min)

### Day 4: Risk + Positions + Grok
12. TASK-501 — Live positions panel (1.5 hr)
13. TASK-502 — Risk rules engine (2 hr)
14. TASK-601 — Simplify Grok output (1 hr)
15. TASK-802 — Should-not-trade detector (1 hr)

### Day 5: Polish + Testing
16. TASK-901 — Integration tests (2 hr)
17. TASK-902 — Smoke test script (1 hr)
18. TASK-103 — Clock fix (if still needed) (30 min)
19. TASK-304 — Retire Streamlit (30 min)
20. Full end-to-end test with Sim101 account

### Backlog (Next Week+)
- TASK-204 — Historical data backfill
- TASK-403 — Dynamic volume analysis in NT8
- TASK-602 — Grok SSE integration
- TASK-702 — SQLite → Postgres migration
- TASK-703 — Rate limiting
- TASK-704 — Prometheus metrics
- TASK-801 — Opening range breakout detection
