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

### TASK-103: Fix Dashboard Clock Not Updating Without Page Refresh ✅ DONE
- **Priority:** 🟡 P1 — Annoyance during active trading, but addressed fully by WS-3 migration
- **Status:** ✅ Complete — resolved by HTMX dashboard (TASK-301) with client-side JS clock that updates every second via `setInterval(updateClock, 1000)`. Streamlit has been retired (TASK-304), so the short-term fix is no longer needed.
- **Context:** Streamlit renders server-side so the clock at the top freezes until next rerun. Short-term fix for Streamlit; long-term fix is the HTMX migration (WS-3) which handles this natively.
- **Files:** `src/services/data/api/dashboard.py` (JS clock in `_render_full_dashboard`)
- **Short-term fix (if still on Streamlit):**
  - [x] ~~Add `st_autorefresh(interval=5000)` import and call~~ — N/A, Streamlit retired
  - [x] ~~Or use `@st.fragment` with a 5-second rerun~~ — N/A, Streamlit retired
- **Long-term fix:** ✅ Handled by TASK-301 (HTMX dashboard with JS clock)
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

### TASK-204: Historical Data Backfill (Off-Hours) ✅ DONE
- **Priority:** 🟢 P2 — Improves optimization quality but not blocking for live week
- **Context:** During off-hours (12:00–00:00 ET), engine should fetch and store up to 1 year of historical 1-minute bars for active contracts (MGC, MNQ, MES, MCL, SIL, HG) into Postgres. Use Massive API if available, or fallback sources.
- **Files:**
  - `src/services/engine/backfill.py` — `run_backfill()`, `backfill_symbol()`, `get_stored_bars()`, `get_backfill_status()`, `get_gap_report()`, `init_backfill_table()`, chunked fetching with Massive/yfinance fallback
  - `src/services/engine/main.py` — `_handle_historical_backfill()` wired to call `run_backfill()` (replaces placeholder)
  - `src/services/data/api/health.py` — `GET /backfill/status`, `GET /backfill/gaps/{symbol}` API endpoints
  - `tests/test_backfill.py` — 72 tests covering table management, symbol resolution, SQL helpers, storage, date range computation, chunk generation, data fetching, single-symbol backfill, full orchestration, query interface, status, gap reports, publishing, engine handler integration, API endpoints, and edge cases
- **Acceptance Criteria:**
  - [x] Stores 1-min OHLCV bars in Postgres/SQLite with symbol, timestamp, open, high, low, close, volume (historical_bars table with UNIQUE constraint)
  - [x] Idempotent: uses INSERT OR IGNORE / ON CONFLICT DO NOTHING, skips already-stored bars, only fetches gaps from latest stored timestamp
  - [x] Runs only during off-hours (12:00–00:00 ET) via HISTORICAL_BACKFILL scheduler action
  - [x] Logs progress: per-symbol chunk-by-chunk logging with emoji indicators (📊 start, ✅ complete, ❌ error)
  - [x] Engine optimization and backtesting jobs can query via `get_stored_bars(symbol, days_back)` → DataFrame
  - [x] Gap analysis via `get_gap_report(symbol)` — coverage %, significant gaps, expected vs actual bar counts
  - [x] Backfill status published to Redis (`engine:backfill_status`) for dashboard visibility
  - [x] API endpoints: GET /backfill/status, GET /backfill/gaps/{symbol}
  - [x] Configurable via env vars: BACKFILL_DAYS_BACK, BACKFILL_CHUNK_DAYS, BACKFILL_SYMBOLS, BACKFILL_INSERT_BATCH
  - [x] Dual data source: Massive.com REST API (primary) with yfinance fallback (limited to ~7 days for 1-min)
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

### TASK-304: Retire Streamlit Container ✅ DONE
- **Priority:** 🟢 P2 — After HTMX dashboard is stable
- **Status:** ✅ Complete — Streamlit `app` service removed from `docker-compose.yml`, deprecation notice added to `src/services/web/app.py`, `streamlit` and `streamlit-autorefresh` removed from `requirements.txt`. HTMX dashboard at `localhost:8000` is the primary UI.
- **Context:** Once the HTMX dashboard (served from data-service) is working, remove the Streamlit container.
- **Files modified:**
  - `docker-compose.yml` — `app` service block removed, retirement comment added
  - `requirements.txt` — `streamlit>=1.37.0` and `streamlit-autorefresh>=1.0.1` removed
  - `src/services/web/app.py` — deprecation notice added at top
- **Acceptance Criteria:**
  - [x] `docker-compose.yml` has 4 services: postgres, redis, data-service, engine
  - [x] Dashboard accessible at `localhost:8000`
  - [x] All dashboard functionality verified working without Streamlit
  - [x] `requirements.txt` cleaned: remove `streamlit`, `streamlit-autorefresh`
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

### TASK-602: Integrate Grok Updates into SSE Stream ✅ DONE
- **Priority:** 🟢 P2 — Nice polish for live dashboard
- **Status:** ✅ Complete — Engine publishes Grok compact updates to `engine:grok_update` Redis key and `dashboard:grok` pub/sub channel. SSE generator handles `dashboard:grok` in pub/sub mode and polls `engine:grok_update` in polling fallback mode. Dashboard JS listens for `grok-update` SSE events and refreshes the Grok panel via `htmx.ajax('GET', '/api/grok/html')`. Risk updates also wired via `dashboard:risk` → `risk-update` SSE event.
- **Context:** When engine runs a Grok 15-minute update, publish the simplified result to Redis Stream so it appears on the HTMX dashboard via SSE without refresh.
- **Files modified:**
  - `src/services/data/api/sse.py` — added `_get_grok_from_cache()`, `_get_risk_from_cache()`, `dashboard:grok` and `dashboard:risk` pub/sub handlers, polling fallback, initial catch-up on connect
  - `src/services/data/api/dashboard.py` — added `grok-update` and `risk-update` JS event handlers
  - `src/services/engine/main.py` — `_publish_grok_update()` writes to Redis key + pub/sub (already in Day 4)
- **Acceptance Criteria:**
  - [x] SSE event `grok-update` fires every 15 minutes during active hours
  - [x] Dashboard panel shows latest Grok summary with timestamp
  - [x] Old updates collapse into expandable history
- **Dependencies:** TASK-302, TASK-601

---

## WS-7: Data & Infrastructure

### TASK-701: Docker First Boot — Verify Full Stack ✅ DONE
- **Priority:** 🔴 P0 — Must pass before any live trading
- **Context:** Run through the full deployment checklist to verify everything works end-to-end.
- **Files:**
  - `scripts/first_boot_verify.py` — Automated Python-based verification script (23 checks, severity levels, JSON report output, --quick/--verbose/--wait/--json flags)
  - `tests/test_first_boot_verify.py` — 100 tests covering data classes, HTTP/Docker helpers, individual checks, full run scenarios, and print summary
- **Checklist (all automated in `first_boot_verify.py`):**
  - [x] `docker compose up -d --build` — all 4 containers running (postgres, redis, data, engine)
  - [x] Postgres healthcheck passes (pg_isready)
  - [x] Redis healthcheck passes (redis-cli ping → PONG)
  - [x] Postgres tables exist (trades_v2, daily_journal; historical_bars optional)
  - [x] Redis engine keys present (engine:status, engine:daily_focus, engine:risk_status)
  - [x] Data-service healthcheck passes (`GET /health` returns 200)
  - [x] Dashboard loads at `localhost:8000` with expected HTML markers (title, SSE, HTMX)
  - [x] Engine healthcheck passes (/tmp/engine_health.json with healthy=true)
  - [x] SSE /sse/health returns status, SSE stream delivers events with correct Content-Type
  - [x] Risk API /risk/status returns 200 with source info
  - [x] Positions API, Prometheus metrics, no-trade, backfill status all return expected codes
  - [x] Postgres write round-trip (INSERT → SELECT → DELETE test row)
  - [x] Redis write round-trip (SET → GET → DEL test key)
  - [x] Cross-service pipeline verified (engine → Redis → data-service)
  - [x] Risk pre-flight check POST /risk/check returns result
  - [x] Engine and data-service container logs have low error count
  - [x] Streamlit container is NOT running (TASK-304 retirement confirmed)
  - [x] No error spam in any container logs
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

### TASK-703: Add Rate Limiting to Data Service ✅ DONE
- **Priority:** 🟢 P2 — Security hardening
- **Files:**
  - `src/services/data/api/rate_limit.py` — Rate limiting module (slowapi-based, per-client key derivation, path-based limits)
  - `src/services/data/main.py` — `setup_rate_limiting(app)` call wired in
  - `tests/test_metrics_and_ratelimit.py` — 121 tests covering rate limiting + metrics
- **Acceptance Criteria:**
  - [x] Add `slowapi` or similar rate limiter
  - [x] Public endpoints (`/health`, `/docs`): 60 req/min
  - [x] API endpoints: 30 req/min per client
  - [x] SSE endpoint: 5 connections per client
  - [x] Trades / position mutations: 20 req/min per client
  - [x] Heavy actions (force_refresh, optimize): 5 req/min per client
  - [x] Custom 429 JSON response with Retry-After header
  - [x] Configurable via environment variables (RATE_LIMIT_ENABLED, RATE_LIMIT_DEFAULT, etc.)
  - [x] Client key derivation: X-API-Key → X-Forwarded-For → remote address
- **Dependencies:** None

### TASK-704: Add Prometheus Metrics Endpoint ✅ DONE
- **Priority:** 🟢 P3 — Nice-to-have for monitoring
- **Files:**
  - `src/services/data/api/metrics.py` — Prometheus metrics module (registry, counters, gauges, histograms, middleware, endpoint)
  - `src/services/data/main.py` — PrometheusMiddleware + metrics router wired in
  - `src/services/data/api/auth.py` — `/metrics/prometheus` added to public paths
  - `tests/test_metrics_and_ratelimit.py` — 121 tests covering metrics + rate limiting
- **Acceptance Criteria:**
  - [x] `GET /metrics/prometheus` returns Prometheus text exposition format
  - [x] Metrics: `http_requests_total` counter (method, path, status labels)
  - [x] Metrics: `http_request_duration_seconds` histogram (method, path labels)
  - [x] Metrics: `sse_connections_active` gauge
  - [x] Metrics: `sse_events_total` counter (event_type label)
  - [x] Metrics: `engine_last_refresh_epoch` gauge
  - [x] Metrics: `engine_cycle_duration_seconds` histogram
  - [x] Metrics: `risk_checks_total` counter (result label: allowed/blocked/advisory)
  - [x] Metrics: `orb_detections_total` counter (direction label)
  - [x] Metrics: `no_trade_alerts_total` counter (condition label)
  - [x] Metrics: `focus_quality_gauge` gauge (per-symbol)
  - [x] Metrics: `positions_open_count` gauge
  - [x] Metrics: `redis_connected` gauge
  - [x] PrometheusMiddleware auto-instruments all requests
  - [x] Live gauges refreshed from cache on each scrape
  - [x] Path normalization to reduce metric cardinality
- **Dependencies:** None

---

### TASK-705: Persistent Audit Trail for Risk & ORB Events ✅ DONE
- **Priority:** 🟡 P1 — Needed for post-day review and compliance
- **Context:** Risk blocks and ORB detections were stored only in-memory and Redis (volatile). Now persisted to Postgres/SQLite for permanent audit trail.
- **Files:**
  - `src/models.py` — `risk_events` and `orb_events` table DDL (SQLite + Postgres), `_init_audit_tables()`, `record_risk_event()`, `get_risk_events()`, `record_orb_event()`, `get_orb_events()`, `get_audit_summary()`
  - `src/services/data/api/audit.py` — Audit API router: GET/POST /audit/risk, GET/POST /audit/orb, GET /audit/summary
  - `src/services/data/main.py` — Audit router registered at `/audit`
  - `src/services/engine/main.py` — `_persist_risk_event()` and `_persist_orb_event()` helpers; wired into `_handle_check_risk_rules()` and `_handle_check_orb()` handlers
  - `tests/test_audit.py` — 69 tests covering table creation, CRUD, API endpoints, engine persistence, edge cases
- **Acceptance Criteria:**
  - [x] `risk_events` table with timestamp, event_type, symbol, side, reason, daily_pnl, open_trades, account_size, risk_pct, session, metadata_json
  - [x] `orb_events` table with timestamp, symbol, or_high/low/range, atr_value, breakout_detected, direction, trigger_price, long/short_trigger, bar_count, session, metadata_json
  - [x] Tables created idempotently by `init_db()` (both SQLite and Postgres)
  - [x] Engine CHECK_RISK_RULES handler persists blocks and warnings
  - [x] Engine CHECK_ORB handler persists every ORB evaluation
  - [x] API endpoints support filtering (event_type, symbol, since, breakout_only)
  - [x] Audit summary endpoint aggregates counts by symbol and type
- **Dependencies:** TASK-502, TASK-801

### TASK-706: Prometheus + Grafana Monitoring Stack ✅ DONE
- **Priority:** 🟢 P2 — Production observability
- **Context:** Prometheus scrape config and pre-provisioned Grafana dashboard for monitoring the full stack.
- **Files:**
  - `docker/monitoring/prometheus.yml` — Scrape config (data-service at /metrics/prometheus every 10s, self-monitoring)
  - `docker/monitoring/grafana-dashboard.json` — 25+ panels across 6 rows: Service Health, HTTP Requests, SSE & Live Connections, Engine Performance, Risk & Trading, Focus Quality, Infrastructure
  - `docker/monitoring/grafana/provisioning/datasources/prometheus.yml` — Auto-provision Prometheus datasource
  - `docker/monitoring/grafana/provisioning/dashboards/dashboards.yml` — Auto-load dashboard from file
  - `docker-compose.yml` — Prometheus + Grafana services (under `monitoring` profile, optional)
- **Acceptance Criteria:**
  - [x] `docker compose --profile monitoring up -d` starts Prometheus (localhost:9090) + Grafana (localhost:3000)
  - [x] Prometheus scrapes data-service metrics every 10s
  - [x] Grafana auto-provisions Prometheus datasource and Futures Co-Pilot dashboard
  - [x] Dashboard panels: service up/down, request rate, latency percentiles (p50/p90/p99), error rate, SSE connections, engine action duration, risk blocks, ORB detections, focus quality, Redis connectivity, process memory/CPU
  - [x] 30-day retention for Prometheus TSDB
  - [x] Optional Redis and Postgres exporter configs (commented out, ready to enable)
- **Dependencies:** TASK-704

---

## WS-8: Pattern Detection & Trading Intelligence

### TASK-801: Opening Range Breakout Detection ✅ DONE
- **Priority:** 🟢 P2 — Matches Jordan's observed daily patterns
- **Context:** Jordan sees breakout patterns daily from market opens. Detect the opening range (first 30-60 min after 9:30 ET) and flag breakouts.
- **Files:**
  - `src/services/engine/orb.py` — `detect_opening_range_breakout(bars_1m, symbol)`, `compute_atr()`, `compute_opening_range()`, `scan_orb_all_assets()`
  - `src/services/engine/main.py` — `_handle_check_orb()` handler wired to scheduler
  - `src/services/engine/scheduler.py` — `CHECK_ORB` action type (every 2 min, 09:30–11:00 ET)
  - `src/services/data/api/sse.py` — `orb-update` SSE event (pub/sub + polling)
  - `src/services/data/api/dashboard.py` — `_render_orb_panel()`, `/api/orb/html` endpoint, JS handler
  - `tests/test_orb.py` — 80 tests covering ORB core, scanner, publishing, scheduler, dashboard, SSE
- **Acceptance Criteria:**
  - [x] Computes OR high/low from first 30 minutes of 1-min bars after 9:30 ET
  - [x] Flags breakout when close > OR_high + 0.5 × ATR(14) or close < OR_low - 0.5 × ATR(14)
  - [x] Returns: `{type: "ORB", direction: "LONG"/"SHORT", trigger_price, or_high, or_low, timestamp}`
  - [x] Published to Redis → appears as alert on dashboard
- **Dependencies:** TASK-201, TASK-204 (uses historical bars for ATR)

### TASK-802: "Should Not Trade" Detector ✅ DONE
- **Priority:** 🟡 P1 — Directly addresses "hard for me to figure when I should not trade"
- **Context:** Rules-based filter that flags low-conviction days to prevent overtrading.
- **Files:**
  - `src/services/engine/patterns.py` — `evaluate_no_trade()`, `publish_no_trade_alert()`, `clear_no_trade_alert()`, 7 condition checkers
  - `src/services/engine/main.py` — `_handle_check_no_trade()` handler wired to scheduler
  - `src/services/engine/scheduler.py` — `CHECK_NO_TRADE` action type (every 2 min during active hours)
  - `src/services/data/api/sse.py` — `no-trade-alert` SSE event (pub/sub + polling + catchup)
  - `src/services/data/api/dashboard.py` — `_render_no_trade_banner()`, `/api/no-trade` endpoint, `sse-swap="no-trade-alert"` JS handler
  - `tests/test_integration.py` — TestNoTradeIntegration (6 tests)
  - `tests/test_focus.py` — TestShouldNotTrade (6 tests)
  - `tests/test_metrics_and_ratelimit.py` — TestShouldNotTradePatterns (12 tests)
- **Conditions for NO TRADE:**
  - [x] No market data available (empty focus assets)
  - [x] All focus assets have quality < 55%
  - [x] Any focus asset has volatility percentile > 88% (extreme vol, stops get hit easily)
  - [x] Daily loss already exceeds -$250
  - [x] More than 2 consecutive losing trades today
  - [x] It's after 10:00 AM ET and no setups have triggered
  - [x] Trading session has ended (after 12:00 PM ET)
- **Acceptance Criteria:**
  - [x] Returns `NoTradeResult` with `should_skip`, `reasons`, `checks`, `severity`
  - [x] Reason displayed on dashboard NO TRADE banner (red pulsing banner with ⛔)
  - [x] Checked every engine cycle during active hours (every 2 min via scheduler)
  - [x] Published to Redis → SSE `no-trade-alert` event
  - [x] Clears automatically when conditions improve
- **Dependencies:** TASK-203, TASK-502

---

## WS-9: Testing & Quality

### TASK-901: Add Integration Tests for New Architecture ✅ DONE
- **Priority:** 🟡 P1 — Confidence for live trading week
- **Status:** ✅ Complete — 65 integration tests in `tests/test_integration.py` covering cross-module wiring, round-trip data flows, HTML rendering, docker-compose config verification, and requirements validation. Also fixed SSE test isolation issue (5 tests that failed when run with other test files now pass in any order).
- **Files:** `tests/test_integration.py` (new), `tests/test_sse.py` (fix: re-install mock cache in `_reset_cache_mock`)
- **Tests added (16 test classes, 65 tests):**
  - [x] Engine writes daily focus to Redis → data-service reads it correctly (TestFocusRoundTrip: 7 tests)
  - [x] SSE endpoint streams events when Redis Stream has data (TestSSEReadsEngineFocus: 3 tests)
  - [x] Position sync from NT8 bridge → RiskManager state (TestPositionsSyncIntegration: 2 tests)
  - [x] Focus card HTML endpoint returns valid HTML with correct data attributes (TestFocusHTMLRendering: 8 tests)
  - [x] Risk manager blocks trade when over limit (TestRiskManagerIntegration: 10 tests)
  - [x] No-trade detector integrates with RiskManager status (TestNoTradeIntegration: 8 tests)
  - [x] Grok compact formatter integration (TestGrokCompactIntegration: 4 tests)
  - [x] Engine status → dashboard time endpoint (TestEngineStatusIntegration: 1 test)
  - [x] Risk → SSE risk-update wiring (TestRiskSSEWiring: 1 test)
  - [x] Full pipeline: focus → publish → SSE → verify (TestFullPipeline: 3 tests)
  - [x] SSE format helpers with real payloads (TestSSEFormatIntegration: 5 tests)
  - [x] Scheduler action types match engine handlers (TestSchedulerEngineWiring: 3 tests)
  - [x] Grok SSE channel wiring (TestGrokSSEChannel: 2 tests)
  - [x] SafeJSONResponse NaN/inf handling (TestSafeJSONResponse: 2 tests)
  - [x] Docker-compose config validation (TestDockerComposeConfig: 3 tests)
  - [x] Requirements file validation (TestRequirements: 2 tests)
- **Dependencies:** TASK-201, TASK-203, TASK-302, TASK-502

### TASK-902: End-to-End Smoke Test Script ✅ DONE
- **Priority:** 🟡 P1 — Run before every trading day
- **Status:** ✅ Complete — `scripts/smoke_test.sh` runs 20 checks with colored PASS/FAIL/SKIP output, auto-detects container names, supports `--quick` and `--verbose` flags.
- **Files:** `scripts/smoke_test.sh` (new, 496 lines)
- **Script checks (20 total):**
  - [x] Check all Docker containers are running and healthy (postgres, redis, data, engine)
  - [x] Curl `/health` on data-service
  - [x] Curl `/api/focus` and verify non-empty JSON response
  - [x] Curl `/sse/dashboard` and verify SSE headers + events received
  - [x] Check Redis has `engine:daily_focus` key
  - [x] Check Postgres connection (pg_isready)
  - [x] Dashboard HTML loads with title and SSE connection
  - [x] `/api/info`, `/api/time`, `/api/positions/html`, `/api/risk/html`, `/api/grok/html`, `/api/alerts/html`, `/api/no-trade` all return 200
  - [x] SSE Content-Type is `text/event-stream`
  - [x] Engine health file (`/tmp/engine_health.json`) is healthy
  - [x] Engine and data-service logs have low error count
  - [x] Streamlit container is NOT running (TASK-304 retirement verification)
  - [x] Print PASS/FAIL summary with exit code
- **Dependencies:** TASK-701

---

## WS-10: Risk Enforcement & API

### TASK-1001: Risk API Router ✅ DONE
- **Priority:** 🟡 P1 — Enables pre-flight risk checks from NinjaTrader and other clients
- **Files:**
  - `src/services/data/api/risk.py` — `/risk/status`, `/risk/check`, `/risk/history` endpoints
  - `src/services/data/main.py` — Risk router mounted at `/risk`
- **Acceptance Criteria:**
  - [x] GET /risk/status reads from Redis (engine) or local RiskManager fallback
  - [x] POST /risk/check performs pre-flight risk check for proposed trades
  - [x] GET /risk/history returns in-memory audit trail of risk events
  - [x] Risk events published to Redis for persistence

### TASK-1002: Risk-Wired Trade Entry ✅ DONE
- **Priority:** 🟡 P1 — Prevents overtrading by checking risk before trade creation
- **Files:**
  - `src/services/data/api/trades.py` — Risk check in `api_create_trade()`, `enforce_risk` field
- **Acceptance Criteria:**
  - [x] POST /trades performs pre-flight risk check before creating trade
  - [x] Response includes `risk_checked`, `risk_blocked`, `risk_reason`, `risk_details` fields
  - [x] `enforce_risk=True` returns HTTP 403 when risk rules block the trade
  - [x] Default (`enforce_risk=False`) creates trade with advisory warning

### TASK-1003: Risk-Wired Position Updates ✅ DONE
- **Priority:** 🟡 P1 — Real-time risk evaluation when NinjaTrader pushes positions
- **Files:**
  - `src/services/data/api/positions.py` — Risk evaluation in `update_positions()`
- **Acceptance Criteria:**
  - [x] POST /positions/update syncs positions into local RiskManager
  - [x] Response includes `risk` dict with `can_trade`, `block_reason`, `warnings`
  - [x] Warnings logged when risk thresholds are hit

### TASK-1004: Test Isolation Fix ✅ DONE
- **Priority:** 🔴 P0 — Fixes flaky tests caused by cross-module cache mock pollution
- **Files:**
  - `tests/test_sse.py` — Save/restore `sys.modules["cache"]` around SSE import and via module-scoped fixture
  - `tests/test_focus.py` — Save/restore in `TestPublishFocusToRedis` setup/teardown
- **Acceptance Criteria:**
  - [x] 959 tests pass with 0 failures (excluding pre-existing missing-dep tests)
  - [x] test_positions.py and test_data_service.py no longer fail when run after test_sse.py or test_focus.py

---

## Execution Order (Recommended Sprint Plan)

### Day 1: Critical Fixes + Architecture Foundation ✅
1. ✅ TASK-101 — Gold price fix (30 min)
2. ✅ TASK-102 — NT8 bridge crash fix (1 hr)
3. ✅ TASK-201 — Create engine container (2 hr)
4. TASK-701 — Docker first boot verification (1 hr)

### Day 2: Engine Features + Dashboard Start ✅
5. ✅ TASK-202 — Session-aware scheduling (1.5 hr)
6. ✅ TASK-203 — Daily focus computation (2 hr)
7. ✅ TASK-301 — Base HTMX dashboard template (3 hr)

### Day 3: Live Data + NT8 Indicator ✅
8. ✅ TASK-302 — SSE endpoint (2 hr)
9. ✅ TASK-303 — HTML fragment endpoints (1.5 hr)
10. ✅ TASK-401 — FKS_Core NT8 indicator (2 hr)
11. ✅ TASK-402 — Harden LivePositionBridge (30 min)

### Day 4: Risk + Positions + Grok ✅
12. ✅ TASK-501 — Live positions panel (1.5 hr)
13. ✅ TASK-502 — Risk rules engine (2 hr)
14. ✅ TASK-601 — Simplify Grok output (1 hr)
15. ✅ TASK-802 — Should-not-trade detector (1 hr)

### Day 5: Polish + Testing ✅
16. ✅ TASK-901 — Integration tests (65 tests)
17. ✅ TASK-902 — Smoke test script (20 checks)
18. ✅ TASK-103 — Clock fix (resolved by HTMX JS clock)
19. ✅ TASK-304 — Retire Streamlit (removed from docker-compose, requirements cleaned)
20. ✅ TASK-602 — Grok SSE integration (grok-update + risk-update events wired)
21. Full end-to-end test with Sim101 account — ready to run

### Day 6: Risk Enforcement + ORB Detection ✅
22. ✅ TASK-1004 — Test isolation fix (test_sse.py + test_focus.py cache mock pollution)
23. ✅ TASK-801 — Opening Range Breakout detector (ORB module, scheduler, SSE, dashboard panel)
24. ✅ TASK-1001 — Risk API router (/risk/status, /risk/check, /risk/history)
25. ✅ TASK-1002 — Risk-wired trade entry (pre-flight check in POST /trades, enforce_risk flag)
26. ✅ TASK-1003 — Risk-wired position updates (evaluate risk on POST /positions/update)
27. 80 new tests (959 total passing, 0 failures)

### Day 7: Observability + Security Hardening + No-Trade Detector ✅
28. ✅ TASK-704 — Prometheus metrics endpoint (/metrics/prometheus, 12 metric families, auto-instrumented middleware)
29. ✅ TASK-703 — Rate limiting (slowapi, per-client keys, path-based limits, custom 429 handler)
30. ✅ TASK-802 — "Should Not Trade" detector (7 conditions, NO TRADE banner, SSE no-trade-alert, scheduler integration)
31. 121 new tests (1,080 total passing, 0 failures)

### Day 8: Historical Data Backfill ✅
32. ✅ TASK-204 — Historical data backfill (backfill module, chunked Massive/yfinance fetching, idempotent UPSERT storage, gap analysis, API endpoints, engine handler wired)
33. 72 new tests (1,152 total passing, 0 failures)

### Day 9: Production Readiness — First Boot, Audit Trail, Monitoring ✅
34. ✅ TASK-701 — Docker first boot verification (automated Python script, 23 checks, severity levels, JSON report)
35. ✅ Persistent audit tables — risk_events + orb_events tables in Postgres/SQLite (DDL, CRUD, API endpoints, engine wiring)
36. ✅ Audit API router — GET/POST /audit/risk, /audit/orb, /audit/summary (persistent event history)
37. ✅ Prometheus + Grafana monitoring stack — prometheus.yml scrape config, grafana-dashboard.json (6 panel rows, 25+ panels), provisioning configs, docker-compose services (monitoring profile)
38. 169 new tests (1,321 total passing, 0 failures)

### Backlog (Next Week+)
- TASK-403 — Dynamic volume analysis in NT8
- TASK-702 — SQLite → Postgres migration (run and verify)
