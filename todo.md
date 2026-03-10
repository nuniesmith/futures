# futures — TODO

> **Last updated**: Phase RA-CHAT — RustAssistant chat window, task capture, openai SDK standardisation

> **Repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:data` · `:engine` · `:web` · `:trainer`
> **Infrastructure**: Ubuntu Server `100.122.184.58` (data + engine + web + monitoring), Home GPU rig `100.113.72.63` (trainer)
>
> 📐 **Architecture reference**: [`docs/architecture.md`](docs/architecture.md)
> 📦 **Completed work**: [`docs/completed.md`](docs/completed.md)
> 🗂️ **Deferred backlog**: [`docs/backlog.md`](docs/backlog.md)

---

## 🎯 Goal

**Manual trading co-pilot with prop-firm compliant copy trading.** The system informs entries via CNN + Ruby signals — the trader pushes "SEND ALL" in the WebUI. All execution flows through Rithmic with `MANUAL` flag + humanized delays.

```
Python Engine  →  CNN signal + Ruby signal + daily bias + risk strip + Grok brief
Python Dashboard  →  Focus cards, swing signals, Reddit sentiment, one-click execution
Rithmic (async_rithmic)  →  Main account order + 1:1 copy to all slave accounts
TradingView  →  Reference overlay only (no position sendback)
```

**Two-stage scaling plan:**
- Stage 1 — TPT: 5 × $150K accounts = $750K buying power
- Stage 2 — Apex: 20 × $300K accounts = ~$6M buying power
- Copy layer: Rithmic `CopyTrader` (main → slaves) with `OrderPlacementMode.MANUAL` + 200–800 ms delay

**Prop-firm compliance:** Every order tagged `MANUAL` + humanized delay. Main account = human button push only. No autonomous entries. Server-side hard stops via `stop_ticks`. See Phase RITHMIC below + [`docs/rithmic_notes.md`](docs/rithmic_notes.md).

**EOD Safety (live now):** Rithmic EOD cancel-all + exit-position fires at 16:00 ET daily via the engine scheduler. 15:45 warning alert fires first. Manual trigger: `POST /api/rithmic/eod-close`. See [`docs/architecture.md`](docs/architecture.md) for full sequence.

---

## Current State

| Item | Status |
|------|--------|
| Champion model | v6 — 87.1% acc / 87.15% prec / 87.27% rec — 18 features, 25 epochs |
| Feature contract | v8 code complete — 37 tabular features + embeddings — **not yet trained** |
| v8 smoke test | ✅ 31/31 tests passing (`test_v8_smoke.py`) |
| Full test suite | ✅ 2657 passed, 1 skipped, 0 failed |
| Rithmic EOD close | ✅ wired into `DashboardEngine._loop()` — uses `OrderPlacement.MANUAL` |
| Rithmic copy trading | ✅ `CopyTrader` class built — 114 tests passing — see Phase RITHMIC |
| Prop-firm compliance | ✅ `MANUAL` flag + 200–800 ms delay enforced on all orders — see RITHMIC-B |
| PositionManager → Rithmic | ✅ `execute_order_commands()` fully wired — MODIFY_STOP/CANCEL/BUY/SELL all routed — see RITHMIC-C |
| Server-side brackets | ✅ `stop_price_to_stop_ticks()` + `TICK_SIZE` table for all 14 micro products — see RITHMIC-C |
| Copy trading engine gate | ✅ `RITHMIC_COPY_TRADING=1` env var gates Rithmic path; NT8 bridge preserved as fallback |
| Ruby signal engine | ❌ Pine Script not yet ported to Python — see RITHMIC-G |
| CI/CD secrets | ✅ verification script created (`scripts/verify_cicd.sh`) — run on each machine to confirm |
| TRAINER_SERVICE_URL | ✅ moved from hardcode to env var in `docker-compose.yml` |
| ENGINE_DATA_URL port | ✅ fixed — was `:8100` (wrong), now `:8050` (matches data service `8050:8000` mapping) |
| sync_models.sh | ✅ audited — platform-agnostic, works on Ubuntu Server (no Pi-specific paths) |
| Trading dashboard | ✅ integrated — pipeline API + trading.html wired into data + web services |
| Dataset smoke test | ✅ `scripts/smoke_test_dataset.py` — validates engine connectivity, bar loading, rendering before full run |
| Charts service | ✅ VWAP ±σ bands, CVD sub-pane, Volume Profile (POC/VAH/VAL), Anchored VWAP, localStorage persistence |
| News sentiment | ✅ `news_client.py` + `news_sentiment.py` + API router + scheduler wired (07:00 + 12:00 ET) |
| RustAssistant LLM integration | ✅ `openai` SDK — RA primary + Grok fallback — `grok_helper.py`, `chat.py`, `tasks.py` |
| Chat window API | ✅ `POST /api/chat`, `GET /sse/chat`, history, status — multi-turn, market context injected |
| Task/issue capture API | ✅ `POST /api/tasks` — bug/task/note with GitHub push via RA, HTMX feed, Redis pub/sub |
| v8 dataset | ❌ not yet generated |

---

## 🔴 Blocking — Must Do Before Training

### 1. Verify CI/CD secrets & fix port mismatch
- [x] `docker-compose.yml` `web` service — `TRAINER_SERVICE_URL` moved to env var: `${TRAINER_SERVICE_URL:-http://100.113.72.63:8200}`
- [x] Verify `scripts/sync_models.sh` still works for Ubuntu Server — audited, platform-agnostic (no Pi-specific code)
- [x] Created `scripts/verify_cicd.sh` — run on each machine to auto-check Tailscale IPs, SSH, Docker, secrets alignment
- [x] **Fixed ENGINE_DATA_URL port mismatch** — was `:8100`, corrected to `:8050` in:
  - `docker-compose.trainer.yml` (default fallback)
  - `.github/workflows/ci-cd.yml` (trainer pre-deploy `upsert_env`)
  - `scripts/verify_cicd.sh` (added port-8100 detection check)
- [ ] **Run `bash scripts/verify_cicd.sh` on Ubuntu Server** — confirm PROD_TAILSCALE_IP matches GitHub secret
- [ ] **Run `bash scripts/verify_cicd.sh --trainer` on GPU rig** — confirm TRAINER_TAILSCALE_IP matches GitHub secret
- [ ] Confirm GitHub secrets are set (script prints the checklist): `PROD_TAILSCALE_IP`, `TRAINER_TAILSCALE_IP`, `PROD_SSH_KEY`, `TRAINER_SSH_KEY`
- [ ] If `.env` on trainer already has `ENGINE_DATA_URL=...:8100`, fix it: `sed -i 's/:8100/:8050/' .env`

### 2. Smoke-test data pipeline, then generate v8 dataset
- [ ] **Run dataset smoke test** on trainer: `python scripts/smoke_test_dataset.py`
  - Validates: ENGINE_DATA_URL connectivity, bar loading (MES + BTC), chart rendering, mini CSV output
  - Quick mode (bar loading only): `python scripts/smoke_test_dataset.py --quick`
  - With explicit URL: `ENGINE_DATA_URL=http://<server-ip>:8050 python scripts/smoke_test_dataset.py`
- [ ] Run `generate_dataset(symbols=ALL_25, days_back=120)` with v8 `DatasetConfig` defaults
  - `max_samples_per_type_label=800` — prevent ORB from dominating
  - `max_samples_per_session_label=400` — balance overnight vs primary sessions
  - Expected output: ~50K–80K samples (vs ~20K in v6 run)
  - 25 symbols × 13 breakout types × 9 sessions

---

## 🔴 Active Sprint — CNN v8 Champion (3-Week Plan)

> One long GPU session. Best possible model before going live on TPT $150K account.

### Week 1 — Final pre-train verification
- [x] Smoke-test training loop (2 epochs, tiny dataset) — all v8 changes verified
- [x] Verify CI/CD secrets — verification script created, TRAINER_SERVICE_URL fixed
- [x] Fix ENGINE_DATA_URL port mismatch (8100 → 8050) — trainer couldn't reach data service
- [x] Dataset smoke test script created (`scripts/smoke_test_dataset.py`)
- [ ] **Run verify_cicd.sh on both machines** *(see Blocking above)*
- [ ] **Run smoke_test_dataset.py on trainer** *(see Blocking above)*
- [ ] Generate v8 dataset *(see Blocking above)*

### Week 2 — Train (GPU rig, mostly hands-off)
- [ ] Train unified v8 model — `epochs=80`, `patience=15`, ~6–10 hours on GPU
- [ ] Gate check: ≥89% acc, ≥87% prec, ≥84% rec
- [ ] If unified fails gate: proceed to per-asset distillation (see [`docs/backlog.md`](docs/backlog.md) — Phase v8-F)
- [ ] Promote winner → `breakout_cnn_best.pt` + regenerate `feature_contract.json` v8
- [ ] **Parallel track (Week 2)**: build position intelligence engine — Phase POSINT-A/B (see below + [`docs/backlog.md`](docs/backlog.md))
- [ ] **Parallel track (Week 2)**: build news sentiment pipeline — Phase NEWS-A/B (see below + [`docs/backlog.md`](docs/backlog.md))
- [ ] **Parallel track (Week 2)**: build Reddit sentiment integration — Phase REDDIT-A/B/C (see [`docs/backlog.md`](docs/backlog.md))

### Week 3 — Validate + go live on demo
- [ ] Phase v8-G: smoke test on 10 live breakouts — verify inference probabilities are sane
- [ ] Deploy to Ubuntu Server via `sync_models.sh` → engine hot-reload via `ModelWatcher`
- [ ] Update `_normalise_tabular_for_inference()` — add v8 backward-compat padding (28→37)
- [ ] Manual trading via Tradovate, dashboard tiled alongside TradingView
- [ ] Tune session thresholds based on live signal quality

---

## ✅ Recently Completed — Phase 1 & 2 (Dashboard Integration)

### Phase 1 — Blocking items resolved
- [x] `TRAINER_SERVICE_URL` hardcode → env var with default in `docker-compose.yml`
- [x] `scripts/verify_cicd.sh` — comprehensive verification script (Tailscale, SSH, Docker, secrets checklist)
- [x] `sync_models.sh` audit — confirmed platform-agnostic, no Pi-specific code

### Phase 1.5 — PORT fix + dataset smoke test
- [x] ENGINE_DATA_URL port corrected from `:8100` to `:8050` in `docker-compose.trainer.yml`, `ci-cd.yml`, `verify_cicd.sh`
- [x] `verify_cicd.sh` enhanced: now detects stale `:8100` in `.env` and trainer compose, validates data service port mapping
- [x] `scripts/smoke_test_dataset.py` — real-data pipeline smoke test (engine connectivity, bar loading, chart rendering, mini CSV)

### Phase 2 — Trading Dashboard integrated into production services
- [x] `src/lib/services/data/api/pipeline.py` — new API router (17 routes):
  - SSE morning pipeline (`/api/pipeline/run`) — 20-step analysis pipeline with real module calls + fallbacks
  - Plan management (`/api/plan`, `/api/plan/confirm`, `/api/plan/unlock`)
  - Live trading stream (`/api/live/stream`) — simulated ticks until Rithmic creds arrive
  - Market data (`/api/market/candles`, `/api/market/cvd`)
  - Journal (`/api/journal/trades`, grade updates)
  - Trading settings (`/api/trading/settings`, connection tests)
  - Trading dashboard page (`/trading`)
- [x] Pipeline router registered in data service (`src/lib/services/data/main.py`)
- [x] Web service proxy routes added (`src/lib/services/web/main.py`) — including SSE proxy helper
- [x] `static/trading.html` — full 5-page SPA (Morning Run → Confirm Plan → Live → Journal → Settings)
- [x] Docker builds updated — `static/` directory added to both `data` and `web` Dockerfiles
- [x] "🚀 Trading" nav tab added to main dashboard header
- [x] API endpoint paths updated in HTML (`/api/settings` → `/api/trading/settings` to avoid conflicts)
- [x] Test suite: 2543 passed, 0 failed (no regressions)

### Files changed
```
docker-compose.yml                          — TRAINER_SERVICE_URL → env var
docker-compose.trainer.yml                  — ENGINE_DATA_URL port fix 8100→8050
docker/data/Dockerfile                      — COPY static/ ./static/
docker/web/Dockerfile                       — COPY static/ ./static/
.github/workflows/ci-cd.yml                 — ENGINE_DATA_URL port fix 8100→8050 in trainer pre-deploy
scripts/verify_cicd.sh                      — NEW: CI/CD verification script (+ ENGINE_DATA_URL port check)
scripts/smoke_test_dataset.py               — NEW: real-data dataset smoke test
src/lib/services/data/api/dashboard.py      — "🚀 Trading" nav link added
src/lib/services/data/api/pipeline.py       — NEW: pipeline API router (17 routes)
src/lib/services/data/main.py               — pipeline_router registered
src/lib/services/web/main.py                — pipeline/trading proxy routes + SSE proxy helper
static/trading.html                         — NEW: full trading workflow SPA
```

---

## ✅ Phase RA-CHAT — RustAssistant Chat Window & Task Capture

> **Completed.** Multi-turn streaming chat backed by RustAssistant (Ollama + RAG + Redis) with
> automatic fallback to direct xAI/Grok. Standardised on the `openai` SDK for both backends.
> Quick-capture task/bug/note system wired to RustAssistant GitHub integration.

### What was built

#### RA-CHAT-A: openai SDK standardisation (`grok_helper.py`)
- Replaced hand-rolled `RustAssistantClient` (~200 lines of manual `requests` + SSE parsing) with two
  lazy factory functions: `_make_ra_client()` and `_make_grok_client()` using `openai.OpenAI`
- Both RA and xAI/Grok now use `client.chat.completions.create()` / `client.chat.completions.stream()`
- Stream events use `event.type == "content.delta"` / `event.delta` (openai 2.x API)
- `_RaClientShim` preserves backward-compat `.available` / `._endpoint` / `._headers()` interface
- Removed `import requests` from `grok_helper.py` entirely — httpx connection pool via openai SDK
- Built-in retries (`max_retries=1`), typed errors (`APIConnectionError`, `APIStatusError`), timeout control
- `_call_llm` / `_stream_llm`: RA primary → Grok fallback, seamless token stream to callers
- All prompt entry points (`run_morning_briefing`, `_run_live_compact`, `_run_live_verbose`, `run_daily_plan_grok_analysis`) now route through `_call_llm` / `_stream_llm`

#### RA-CHAT-B: Chat API router (`src/lib/services/data/api/chat.py`)
- `POST /api/chat` — non-streaming single-turn with multi-turn Redis history (`AsyncOpenAI`, no thread pool)
- `GET /sse/chat` — streaming SSE chat; native `async for event in stream` replaces `asyncio.Queue` + thread pool
- `GET /api/chat/history` / `DELETE /api/chat/history` — per-session history (Redis, 6h TTL, 20-pair window)
- `GET /api/chat/status` — live RA reachability probe + backend config (httpx async, no blocking)
- Market context auto-injected from Redis: scanner, ICT, CVD, AI analysis, open positions
- SSE protocol: `chat-start` → `chat-token` → `chat-heartbeat` → `chat-error` → `chat-done`
- Session ID auto-generated (UUID) if not supplied; `inject_context` and `clear_history` flags

#### RA-CHAT-C: Task / issue capture (`src/lib/services/data/api/tasks.py`)
- `POST /api/tasks` — create bug / task / note; optional market snapshot auto-captured from Redis
- `GET /api/tasks` / `GET /api/tasks/html` — list with status/type filters; HTMX-swappable panel
- `PUT /api/tasks/{id}` / `DELETE /api/tasks/{id}` — update status, priority, tags
- `POST /api/tasks/{id}/github` — push to GitHub via `POST {RA_BASE_URL}/api/github/issue`
- Background task: on `push_to_github=true`, RA creates GitHub issue + updates row with URL + number
- `tasks` table created idempotently in existing SQLite/Postgres DB (no models.py change needed)
- Redis pub/sub: publishes `dashboard:tasks` events on create/update/delete/github_linked
- HTML renderer: dark-theme cards with inline status select, priority badge, GitHub link, delete button

#### RA-CHAT-D: Service wiring
- `src/lib/services/data/main.py` — `chat_router` + `tasks_router` registered; `chat_set_engine()` called in lifespan
- `src/lib/services/web/main.py` — proxy routes for all chat + tasks endpoints including SSE via `_proxy_sse_request`
- `pyproject.toml` — `openai>=1.78.0` added to base dependencies

### Environment variables added
```
RA_BASE_URL      # RustAssistant base URL e.g. http://oryx:3500
RA_API_KEY       # Proxy key (must match RA_PROXY_API_KEYS on server)
RA_REPO_ID       # Optional RAG repo context e.g. futures-bot (sent as x-repo-id header)
GITHUB_REPO      # GitHub repo slug for task push e.g. jordan/futures
XAI_API_KEY      # xAI/Grok direct key — fallback only
CHAT_MAX_HISTORY # Rolling history window in pairs (default 20)
CHAT_MAX_TOKENS  # Max tokens per chat response (default 1024)
CHAT_HISTORY_TTL # Redis TTL for session history in seconds (default 21600 = 6h)
```

### New API surface
```
POST   /api/chat                   — Non-streaming chat
GET    /sse/chat                   — Streaming SSE chat
GET    /api/chat/history           — Fetch session history
DELETE /api/chat/history           — Clear session history
GET    /api/chat/status            — Backend health check

POST   /api/tasks                  — Create task/bug/note
GET    /api/tasks                  — List tasks (filterable)
GET    /api/tasks/html             — HTMX task feed panel
GET    /api/tasks/status           — Tasks subsystem status
GET    /api/tasks/{id}             — Single task JSON
GET    /api/tasks/{id}/html        — Single task card fragment
PUT    /api/tasks/{id}             — Update task
DELETE /api/tasks/{id}             — Delete task
POST   /api/tasks/{id}/github      — Push to GitHub via RA
```

### Linting results (pre-push)
- `ruff format --check`: 170 files already formatted ✅
- `ruff check` on changed files: all checks passed ✅ (19 pre-existing issues auto-fixed in unrelated files)
- `mypy` on changed files: 0 errors, 0 warnings ✅ (notes only — unannotated bodies per `mypy.ini` config)
- `openai 2.26.0` installed in `.venv` ✅

### Files changed
```
pyproject.toml                                      — openai>=1.78.0 added to dependencies
src/lib/integrations/grok_helper.py                 — RustAssistantClient replaced with openai.OpenAI factories;
                                                      _RaClientShim for backward compat; _call_llm/_stream_llm updated
src/lib/services/data/api/chat.py                   — NEW: multi-turn SSE chat router (AsyncOpenAI, no thread pools)
src/lib/services/data/api/tasks.py                  — NEW: task/bug/note capture with GitHub push via RA
src/lib/services/data/main.py                       — chat_router + tasks_router registered; chat_set_engine wired
src/lib/services/web/main.py                        — proxy routes for /api/chat/*, /sse/chat, /api/tasks/*
```

---

## 🔴 Phase RA-CHAT — Next Up

### RA-CHAT-E: Chat page HTML (`/chat`)
- [ ] Build `src/lib/services/data/api/chat_page.py` — serve full-page chat UI at `GET /chat`
- [ ] Dark theme matching `trading.html` design system (JetBrains Mono, same CSS variables)
- [ ] Left sidebar: session history list (click to restore), new chat button, backend indicator (RA/Grok pill)
- [ ] Main area: message bubbles (user right, assistant left), streaming token display, markdown rendering
- [ ] Input bar: textarea (Shift+Enter newline, Enter send), inject_context toggle, clear history button
- [ ] Quick-capture bar: 🐛 Bug / ✅ Task / 📝 Note buttons — pre-fill task modal with current page context
- [ ] Task capture modal: title, description, priority, repo, push-to-GitHub checkbox
- [ ] Task feed panel (right sidebar or bottom drawer): live HTMX-polled `/api/tasks/html`
- [ ] JS: `EventSource('/sse/chat?message=...')` consumer — assembles token stream, renders markdown
- [ ] JS: `updateTaskStatus(id, status)` / `deleteTask(id)` / `pushTaskToGitHub(id)` helpers
- [ ] Register `GET /chat` route in data service + web service proxy

### RA-CHAT-F: Dashboard integration
- [ ] Add "💬 Chat" nav button to main dashboard header (`api/dashboard.py`) — opens `/chat` in new tab or slide-over panel
- [ ] Add "⚡ Tasks" panel to dashboard sidebar — HTMX fragment polling `/api/tasks/html?status=open&limit=10`
- [ ] Quick-capture floating button on all pages — one click opens task modal pre-filled with current page/asset context
- [ ] Grok briefing panel "Ask about this" button — pre-fills chat with current briefing text as context
- [ ] Wire `source=chat` tasks: when assistant response contains `[TASK]`, `[BUG]`, or `[NOTE]` markers, auto-call `POST /api/tasks`

### RA-CHAT-G: Intent detection in chat
- [ ] Server-side intent parser in `chat.py`: scan assistant response for structured markers
  - `[BUG: <title>]`, `[TASK: <title>]`, `[NOTE: <title>]` → auto-create task row
  - `[PLAN: <content>]` → append to daily plan notes
- [ ] Return `tasks_created` list in `chat-done` SSE event so UI can refresh the task feed
- [ ] System prompt addition: teach assistant when/how to emit task markers

### RA-CHAT-H: RustAssistant GitHub actions (requires RA server config)
- [ ] Confirm RA server exposes `POST /api/github/issue` — test with `curl`
- [ ] Confirm RA server exposes `POST /api/github/pr` — for future code-change requests from chat
- [ ] Add `GET /api/tasks/{id}/github/status` — poll GitHub issue state (open/closed/merged)
- [ ] Add `POST /api/chat` intent: "create a PR for this" → RA generates diff + opens draft PR
- [ ] Set `GITHUB_REPO` env var in `docker-compose.yml` for the futures repo

---

## ✅ Phase NEWS — News Sentiment Pipeline

> **Completed.** See full entry in the Phase NEWS section below (now marked ✅).

### Files changed
```
pyproject.toml                                        — finnhub-python>=2.4.20 added
src/lib/integrations/news_client.py                  — NEW: FinnhubClient + AlphaVantageClient + fetch_all_news()
src/lib/analysis/news_sentiment.py                   — NEW: VADER+AV+Grok hybrid scorer + run_news_sentiment_pipeline()
src/lib/services/engine/scheduler.py                 — CHECK_NEWS_SENTIMENT + CHECK_NEWS_SENTIMENT_MIDDAY ActionTypes + schedule rules
src/lib/services/engine/main.py                      — _handle_check_news_sentiment() handler + action_handlers wiring
src/lib/services/data/api/news.py                    — NEW: news router (5 JSON + 2 HTMX routes)
src/lib/services/data/main.py                        — news_router registered
```

---

---

## ✅ Phase CHARTS — Charting Service Volume Indicators

> **Completed.** The standalone charting service (`docker/charting/`, port 8003) already
> existed with ApexCharts, EMA9/21, BB, VWAP, RSI sub-pane, and live SSE updates.
> All Phase CHARTS-E volume indicators have been implemented.

### Files changed
- `docker/charting/static/chart.js`
  - `calcVWAP()` rewritten to return `{ vwap, upper1, lower1, upper2, lower2 }` with
    running variance accumulation for ±1σ / ±2σ bands
  - `calcCVD()` — bar-approximation CVD with daily reset, per-bar `fillColor`
  - `calcVolumeProfile()` — rolling 100-bar POC / VAH / VAL (70% value area)
  - `calcAnchoredVWAP()` — cumulative VWAP from a given anchor bar index
  - `findSessionAnchor()` / `findPrevDayAnchor()` — anchor helpers
  - Series slots 8–16 added (`IDX.VWAP_U1/L1/U2/L2`, `POC`, `VAH`, `VAL`, `AVWAP_S/P`)
  - `buildCvdOptions()` + `mountCvdChart()` / `unmountCvdChart()` / `syncCvdPane()`
  - `destroyCvdChart()`, `state.chartCvd`, `dom.chartCvdEl` wired
  - `liveInd.cvdRunning` / `cvdLastDay` for incremental CVD on live ticks
  - `updateIndicatorPoint()` extended: CVD delta, VWAP σ-bands, session AVWAP
  - `recalcIndicators()` / `recalcSingleIndicator()` extended for all new indicators
  - `saveIndicatorPrefs()` / `loadIndicatorPrefs()` — localStorage key `ruby_chart_indicators`
  - `boot()` calls `loadIndicatorPrefs()` before `wireControls()` to restore state
  - Toggle handler dispatches to `syncCvdPane()` / VP / AVWAP branches correctly
- `docker/charting/static/index.html`
  - Added CVD, VP, AVWAP-S, AVWAP-P toggle buttons to indicator-tabs
  - Added `<div id="chart-cvd" class="chart-cvd hidden">` sub-pane below RSI pane
- `docker/charting/static/style.css`
  - Per-indicator active colours: CVD=emerald, VP=amber, AVWAP-S=orange, AVWAP-P=fuchsia
  - `.chart-cvd` / `.chart-cvd.hidden` rules (`flex: 0 0 120px`, matches RSI pane pattern)

---

## 🔴 Phase RITHMIC — Copy Trading & Prop-Firm Compliance

> **The #1 priority for going live on prop accounts.** All order execution must use
> `OrderPlacementMode.MANUAL` + randomized 200–800 ms delay between copies. Main
> account is always human-initiated (WebUI button); slaves mirror 1:1 via async_rithmic.
>
> **Source**: [`docs/rithmic_notes.md`](docs/rithmic_notes.md) — full API review, code
> skeletons, rate-limit analysis, and firm-by-firm compliance status (March 2026).
>
> **Existing code**: `rithmic_client.py` already has `RithmicAccountManager` +
> `eod_close_all_positions()` using `OrderPlacement.MANUAL`. `PositionManager` emits
> `OrderCommand` objects — needs wiring to Rithmic `submit_order` instead of NinjaTrader bridge.

### ✅ RITHMIC-A: CopyTrader Class (Core Multi-Account Engine)
- [x] `src/lib/services/engine/copy_trader.py` — new `CopyTrader` class
  - `add_account(config, is_main=False)` — spin up `RithmicClient` per credential
  - Main client: ORDER_PLANT for execution (fill listener deferred to Phase 2)
  - Slave clients: full ORDER_PLANT for execution
  - `send_order_and_copy()` — WebUI "SEND ALL" button handler (market/limit + bracket on main, then copies)
  - `send_order_from_ticker()` — convenience: resolve Yahoo ticker → Rithmic contract → send
  - `execute_order_commands()` — bridge from `PositionManager` `OrderCommand` → Rithmic path
  - `RollingRateCounter` — rolling 60-min action counter (warn at 3,000, hard stop at 4,500)
  - `_ConnectedAccount` wrapper with per-account order count + last-order timestamp
  - `TICKER_TO_RITHMIC` mapping — Yahoo tickers → product_code + exchange (core + extended + full-size)
  - Front-month contract cache with `invalidate_contract_cache()`
  - `_persist_batch_result()` → Redis log + pub/sub for real-time SSE
  - Tag every order: `RUBY_MANUAL_WEBUI` (main) / `COPY_FROM_MAIN_HUMAN_150K` (slaves)
  - Module-level singleton via `get_copy_trader()`
  - Engine `__init__.py` updated to export `CopyTrader` + `get_copy_trader`
  - **79 tests passing** (`tests/test_copy_trader.py`)

### ✅ RITHMIC-B: Compliance — MANUAL Flag + Humanized Delays
- [x] **Every** `submit_order` call includes `manual_or_auto=OrderPlacement.MANUAL`
  - Audit `eod_close_all_positions()` — ✅ already uses `OrderPlacement.MANUAL`
  - New copy-trade orders — enforced in `_submit_single_order()` (single code path for all orders)
  - New limit/market orders from WebUI — enforced in `send_order_and_copy()`
- [x] `asyncio.sleep(random.uniform(0.2, 0.8))` before every slave copy order
  - `set_high_impact_mode(True)` increases delay to `random.uniform(1.0, 2.0)` (NFP/FOMC)
  - Env vars: `CT_COPY_DELAY_MIN/MAX`, `CT_HIGH_IMPACT_DELAY_MIN/MAX`
- [x] Compliance log: `_build_compliance_checklist()` + `_log_compliance()` on every "SEND ALL"
  - Printed to logger + persisted to Redis (`engine:copy_trader:compliance_log`, 7-day TTL)
  - Warns on zero `stop_ticks` and approaching rate limit
  - Included in every `CopyBatchResult.compliance_log` for WebUI display

### ✅ RITHMIC-C: PositionManager → CopyTrader Wiring + Server-Side Brackets
- [x] `stop_price_to_stop_ticks()` — tick-size conversion helper
  - `TICK_SIZE` table: all 14 micro products (MGC, MCL, MES, MNQ, M6E, MBT, MET, SIL, MNG, MYM, M2K, M6A, M6B, M6J) + full-size fallbacks
  - `MIN_STOP_TICKS=2`, `DEFAULT_STOP_TICKS=20`; clamps to min; defaults on unknown product
  - Validates every entry in `TICKER_TO_RITHMIC` has a matching tick-size entry (tested)
- [x] `CopyTrader.modify_stop_on_all()` — move server-side bracket stop on all connected accounts
  - Converts absolute `stop_price` → `stop_ticks` via tick-size table (product_code auto-inferred from security_code prefix if omitted)
  - Enforces `OrderPlacement.MANUAL` on every `client.modify_order()` call
  - Per-account `RollingRateCounter.record(1)` on success; returns `accounts_modified` + `accounts_failed` audit dict
  - Full audit trail: `position_id`, `reason`, `new_stop_price`, `security_code` in result
- [x] `CopyTrader.cancel_on_all()` — cancel all working orders for a security on all accounts
  - Enforces `OrderPlacement.MANUAL` on every `client.cancel_all_orders()` call
  - Optional `security_code` filter; omit to cancel all open orders (use with caution)
  - Returns `accounts_cancelled` + `accounts_failed` audit dict
- [x] `CopyTrader.execute_order_commands()` — fully wired PositionManager → Rithmic bridge
  - `BUY`/`SELL` with `MARKET`/`LIMIT` → `send_order_from_ticker()` (main + slave copies, MANUAL flag)
  - `MODIFY_STOP` → `modify_stop_on_all()` (resolves contract, computes stop_ticks, MANUAL flag)
  - `CANCEL` → `cancel_on_all()` (MANUAL flag, resolves contract if available)
  - `STOP` companion order type → **silently skipped** (covered by server-side bracket on entry order)
  - `entry_prices` dict passed through for accurate stop_ticks on MODIFY_STOP commands
  - `OrderCommand.stop_price` captured from STOP companion and stored for subsequent entry's stop_ticks computation
- [x] `engine/main.py` — `_copy_trader` singleton + `_get_copy_trader()` lazy-init
  - Gated by `RITHMIC_COPY_TRADING=1` env var — degrades gracefully to NT8-bridge-only when unset
  - `_publish_pm_orders()` updated: NT8 Redis path preserved (backward compat) + new Rithmic path added
  - `_dispatch_orders_to_copy_trader()` — fire-and-forget async bridge from synchronous engine loop
    - Builds `entry_prices` dict from active `PositionManager` positions for MODIFY_STOP accuracy
    - Runs `ct.execute_order_commands()` in existing loop (or fresh one-shot loop as fallback)
    - Logs `ok` count per batch; non-fatal on any error
  - Logged on startup: "CopyTrader ready" (enabled) or "set RITHMIC_COPY_TRADING=1" (disabled)
- [x] **35 new tests** (`tests/test_copy_trader.py`) — total now **114 passing**
  - `TestStopPriceToStopTicks` (14 tests): MGC/MES/MCL/M6E/MNQ tick math, min clamp, zero/unknown inputs, table coverage assertions
  - `TestModifyStopOnAll` (7 tests): no-accounts, tick conversion, product_code inference, timeout, rate counter, audit fields
  - `TestCancelOnAll` (7 tests): no-accounts, cancel called, no-security-code variant, timeout, rate counter, audit fields
  - `TestExecuteOrderCommandsRouting` (8 tests): STOP companion skip/price-capture, MODIFY_STOP dispatch, CANCEL dispatch, unknown action skip, entry_prices forwarding, mixed batch end-to-end
  - Updated `TestExecuteOrderCommands`: `test_modify_stop_skipped` → `test_modify_stop_returns_result`, `test_cancel_skipped` → `test_cancel_returns_result`

### RITHMIC-D: Rate-Limit Monitoring & Safety
- [ ] Daily action counter (in-memory or Redis) — track orders per rolling 60 min
  - Alert threshold: warn at 3,000 actions/hour (hard limit ~5,000 per Rithmic)
  - For manual + copy setup this will never trigger, but monitor as safety net
- [ ] Enable `logging.getLogger("rithmic").setLevel(logging.DEBUG)` in production
- [ ] Detect "Consumer Slow" or rate-limit errors in event handlers → log + Slack/Discord alert

### RITHMIC-E: PositionManager Upgrades (One-Asset Focus + Pyramiding)
- [ ] Add focus lock: `open_asset` field — only one instrument at a time across all accounts
  - `can_trade(asset)` gate — reject signals for other assets while position open
- [ ] Quality-gated pyramiding: `get_next_pyramid_level(ruby_signal, current_price)`
  - Level 1 (+1R): add 1 micro, move SL to breakeven
  - Level 2 (+2R): add 1 micro, trail SL to entry + 0.5R
  - Level 3 (+3R): add 1 micro, trail SL to price − 1R
  - Gate: Ruby quality ≥ 65% + regime must be TRENDING ↑/↓ + wave_ratio > 1.5 for 3rd add
  - Max pyramid = 3 (quality ≥ 80) or 2 (quality 65–79)
- [ ] Max risk rule: never exceed 1.5% account risk on full scaled position (3 micros max)

### RITHMIC-F: WebUI Integration
- [ ] "SEND ALL" button on Live page → calls `CopyTrader.send_limit_order_and_copy()`
  - Inputs: asset, side (LONG/SHORT), limit price, qty, stop_ticks, optional target_ticks
  - Shows confirmation: "Main + N slaves, MANUAL flag, delay 200–800ms"
- [ ] "ADD PYRAMID" button — sends additional contract at pullback level via same copy loop
- [ ] Compliance checklist widget on Live page (daily pre-market, auto-checked from state)
- [ ] Account status cards: per-slave connection state, last order timestamp, P&L mirror
- [ ] Copy-trade log viewer: timestamped list of all copied orders with tags

### RITHMIC-G: Ruby Signal Engine (Pine → Python Port)
- [ ] `src/lib/services/engine/ruby_signal_engine.py` — `RubySignalEngine` class
  - Port all Pine Script v6 logic: Top G Channel, wave analysis, market regime, quality score
  - `update(new_bar)` → returns `{signal, quality, regime, wave_ratio, levels{entry, sl, tp1, tp2, tp3}}`
  - Feeds into `PositionManager.process_signal()` and WebUI signal cards
  - Uses `ta` + `talib` libraries (already available)
- [ ] `extract_features_for_cnn()` — Ruby features as additional CNN input channels
  - Top G position, wave ratio, regime enum, quality %, vol percentile
  - Wire into `RubyORB_CNN` hybrid model (Phase v9 — deferred unless >2% lift)

---

## 🟡 Post-Training Cleanup (non-blocking, do after v8 is live)

### Comment cleanup — NinjaTrader references
- [ ] `breakout_cnn.py` — references to "NinjaTrader BreakoutStrategy", "OrbCnnPredictor.NormaliseTabular() in C#", "NT8 inference" → update to "external consumers" / "TradingView"
- [ ] `chart_renderer.py` / `chart_renderer_parity.py` — "Ruby NinjaTrader indicator", "NT8 screen" → generic language
- [ ] `breakout_types.py` / `multi_session.py` — "C# NinjaTrader consumer" → generic language

### Dashboard naming — bridge → broker
- [ ] `_get_bridge_info()` → rename to `_get_broker_info()`; update `bridge_connected` / `bridge_age_seconds` / `bridge_account` param names
- [ ] SSE event name `bridge-status` — verify publisher matches; rename to `broker-status` when convenient
- [ ] `/api/nt8/health/html` endpoint path — low priority rename to `/api/health/html`
- [ ] `positions.py` — verify no FastAPI conflict between `get_bridge_status()` / `get_broker_status()` and `get_bridge_orders()` / `get_broker_orders()` duplicate route registrations

### Remaining refactor items
- [ ] `orb.py` — deprecate `detect_opening_range_breakout()` and `ORBResult` once v8 validates the unified detector path in production
- [ ] `ORBSession` → `RBSession` bulk rename in callers (alias works, non-breaking — do as a single find-and-replace PR)
- [ ] "Asset DNA" radar chart on focus cards (v8-C dashboard, low priority)

---

## 🟡 Next Up — Wire Real Modules into Trading Pipeline

> These are non-blocking improvements to replace simulated data with live module calls.
> Each step in `pipeline.py` already has try/except wiring — just needs cached data.

- [ ] Wire overnight step to `massive_client` real bars (needs Massive API key in settings)
- [ ] Wire regime step to `RegimeDetector` with cached 15m bars from engine
- [ ] Wire ICT step to `ict_summary()` with cached 5m bars
- [ ] Wire volume profile step to `compute_volume_profile()` with cached bars
- [ ] Wire ORB step to `engine:orb:{symbol}` Redis cache from engine scheduler
- [ ] Wire CNN step to live model inference probabilities from engine
- [ ] Wire Grok step to live `run_morning_briefing()` (needs `GROK_API_KEY` env var)
- [ ] Wire Kraken step to live `KrakenClient.get_ticker()` (already attempted, needs error handling)
- [ ] Replace simulated live stream with Rithmic tick data (when creds arrive)
- [ ] Persist journal trades to Postgres via existing journal API

---

## 🟡 Phase POSINT — Position Intelligence Engine

> **The core "live trading co-pilot."** Real-time per-position analysis: L1/L2 book,
> DOM pressure, multi-TP zones, sweep-aware breakeven, risk action recommendations.
> Builds with mock data first; swaps to real Rithmic when creds arrive.
>
> **Source**: Extracted from `todo/position_engine.py` + `todo/live_page.html` prototypes.
> Full spec: [`docs/backlog.md`](docs/backlog.md) — Phase POSINT.
> Detailed extraction audit: [`docs/todo_extracted_tasks.md`](docs/todo_extracted_tasks.md).

### POSINT-A: Position Intelligence Module
- [ ] `src/lib/services/engine/position_intelligence.py` — `compute_position_intelligence()`
  - Sweep zone detection, multi-TP calculation, book pressure, risk actions
  - Wire real modules: `ict.py`, `confluence.py`, `volume_profile.py`, `cvd.py`, `regime.py`
  - Mock fallbacks for demo mode (already prototyped)

### POSINT-B: Rithmic Position Engine Wrapper
- [ ] `src/lib/services/engine/rithmic_position_engine.py` — `RithmicPositionEngine` class
  - Methods: `connect()`, `get_positions()`, `get_l1()`, `get_l2()`, `get_recent_trades()`
  - Auto-reconnect with exponential backoff
  - Clear swap points documented for when Rithmic creds arrive

### POSINT-C: Position Intelligence API Routes
- [ ] `GET /api/live/positions` — SSE stream (1.5s interval) with full intel payload per position
- [ ] `GET /api/live/book?symbol=MES` — L1 + L2 depth-of-market snapshot
- [ ] `GET /api/live/tape?symbol=MES&n=20` — recent time & sales
- [ ] `GET /api/live/positions/snapshot` — non-SSE current positions
- [ ] Wire web service proxy routes

### POSINT-D: Live Page UI Enhancement
- [ ] Update `static/trading.html` Live page — per-position intelligence cards
  - Header: symbol, direction, entry, live price, unrealized P&L
  - Col 1 — Book: L1 bid/ask, spread, time & sales tape
  - Col 2 — DOM: visual depth ladder, bid/ask bars, sweep zone warnings
  - Col 3 — TP Zones: 4-tier targets (plan-aware + Fib + liquidity)
  - Col 4 — Actions: breakeven panel, risk recommendations, live signal pills
  - Session stats bar, Rithmic connection banner, no-position state

---

## ✅ Phase NEWS — News Sentiment Pipeline

> **Completed.** Multi-source hybrid sentiment pipeline: Finnhub + Alpha Vantage + VADER
> + Grok 4.1 (ambiguous articles only) → weighted hybrid score per asset.
> Engine scheduler fires at 07:00 ET (morning) and 12:00 ET (midday refresh).
> API routes live at `/api/news/*` + HTMX panel at `/htmx/news/panel`.

### ✅ NEWS-A: News Data Collector
- [x] `src/lib/integrations/news_client.py` — `FinnhubClient` + `AlphaVantageClient`
  - Finnhub: `fetch_general_news()` + `fetch_company_news()` (USO/GLD/SPY as futures proxies), 60 calls/min
  - Alpha Vantage: `fetch_news_sentiment()` with AI scores + `fetch_commodity_price()`, 25 calls/day
  - `fetch_all_news()` — single call fetches and merges both sources per symbol list
  - `finnhub-python>=2.4.20` added to `pyproject.toml`

### ✅ NEWS-B: Hybrid Sentiment Scorer
- [x] `src/lib/analysis/news_sentiment.py`
  - VADER with 60+ futures-specific lexicon terms (`surge: 3.0`, `crash: -3.5`, `rate hike: -2.0`, etc.)
  - Grok 4.1 batch scoring — only articles where `abs(vader) < 0.3` (ambiguous) — ~$0.01/100 articles
  - Hybrid: `0.4×vader + 0.4×alpha_vantage + 0.2×grok` (weights redistribute when a source is unavailable)
  - `run_news_sentiment_pipeline()` — full fetch → score → aggregate → cache entry point
  - `vaderSentiment>=3.3.2` already in `pyproject.toml`

### ✅ NEWS-C: Scheduler Integration + Caching
- [x] `ActionType.CHECK_NEWS_SENTIMENT` + `CHECK_NEWS_SENTIMENT_MIDDAY` added to `scheduler.py`
  - Morning run: fires at ≥07:00 ET within PRE_MARKET window (once per day)
  - Midday run: fires once per OFF_HOURS session (~12:00 ET)
- [x] Handler `_handle_check_news_sentiment()` in `engine/main.py` — reads API keys from env, resolves watchlist, calls pipeline, logs spikes
- [x] Redis cache: `engine:news_sentiment:<SYMBOL>` (2h TTL) via `cache_sentiments()` in `news_sentiment.py`
- [x] Spike detection: publishes to `dashboard:news_spike` Redis channel when article rate > 3× rolling avg
- [ ] Postgres `news_sentiment_history` table *(deferred — not blocking; spike/signal data is in Redis)*

### ✅ NEWS-D: Dashboard Integration
- [x] `src/lib/services/data/api/news.py` — new router registered in `data/main.py`
  - `GET /api/news/sentiment?symbols=MES,MGC,MCL` → aggregated sentiment per symbol (JSON)
  - `GET /api/news/sentiment/{symbol}` → single-symbol detail
  - `GET /api/news/headlines?symbol=MES&limit=10` → headlines with all scores
  - `GET /api/news/spike` → current spiking symbols from Redis
  - `GET /htmx/news/panel` → full panel HTML fragment (hx-trigger="every 120s")
  - `GET /htmx/news/asset/{symbol}` → single-asset card with headlines + Grok narrative
- [x] Web service proxy: existing blanket `/api/{path:path}` catch-all in `web/main.py` already covers `/api/news/*`
- [ ] "News Pulse" strip wired into main dashboard HTML *(next — Phase UI-ENHANCE UI-A)*
- [ ] Postgres `news_sentiment_history` table *(deferred)*

---

## 🟡 Phase UI-ENHANCE — Trading Dashboard Improvements

> Polish items from the original UI blueprint not yet implemented.
> Full spec: [`docs/backlog.md`](docs/backlog.md) — Phase UI-ENHANCE.

### UI-A: Research Page
- [ ] Cross-asset context panel (ES/NQ/RTY heatmap, DXY/VIX badges) — wire `cross_asset.py`
- [ ] Economic calendar integration (Forex Factory RSS or TradingEconomics free API)
- [ ] Combined sentiment gauges (Reddit + News → "Market Mood" gauge)

### UI-B: Analysis Page
- [ ] Asset fingerprint display — wire `asset_fingerprint.py` ("This instrument tends to…")
- [ ] Wave structure panel — wire `wave_analysis.py` + `swing_detector.py`
- [ ] Focus asset selection: user picks 1–2 assets, filters downstream pages

### UI-C: Plan Page
- [ ] Range builders status — wire `rb/detector.py` (current range, breakout direction)
- [ ] "Backtest this level" button — wire `backtesting.py` (historical hit rate)
- [ ] CNN confidence badge on each entry zone — wire `breakout_cnn.py` inference
- [ ] ORB levels surfaced in plan zones from pipeline "orb" step

### UI-D: Journal Page
- [ ] Auto-populate from Rithmic fills (when creds arrive)
- [ ] Plan adherence scoring: compare trades to locked plan zones
- [ ] Session stats panel: P&L, win rate, avg R:R, equity curve mini-chart

### UI-E: UX Polish
- [ ] Keyboard shortcuts (`1-5` for pages, `Space` to lock plan)
- [ ] One-click copy: every price → clipboard (for MotiveWave paste)
- [ ] Nav progress indicator: `Research ✅ → Plan ✅ → Live ● → Journal`
- [ ] Mobile-friendly Live page layout
- [ ] Add `DM Sans` font for labels alongside `JetBrains Mono` for prices

---

## 🟢 After First Live Profits

1. **Phase CHARTS** — replace placeholder `/charts` page with Lightweight Charts UI
2. **Phase REDDIT** — Reddit sentiment panel on dashboard
3. **Phase 9A** — correlation anomaly heatmap
4. **Phase 6** — Kraken spot portfolio management
5. **Phase v9** — cross-attention fusion, Ruby/Reddit/News CNN features (only if >2% accuracy lift)
6. **Phase COMPLIANCE-AUDIT** — one-page compliance log PDF exporter for prop-firm audits

Full specs for all of the above: [`docs/backlog.md`](docs/backlog.md)

---

## `todo/` Directory — Consolidated & Deleted

> All 13 files from the former `todo/` directory have been reviewed and their actionable
> content extracted into the phases above. Full audit trail with per-file disposition:
> [`docs/todo_extracted_tasks.md`](docs/todo_extracted_tasks.md).

| File | Disposition |
|------|-------------|
| `README.md`, `notes.md` | ✅ Original vision — fully implemented in `pipeline.py` |
| `app.py`, `app1.py` | ✅ Integrated into `pipeline.py`; position routes → Phase POSINT |
| `index.html` | ✅ Copied to `static/trading.html` |
| `trading_webui_review.md` | → Phase UI-ENHANCE (A–E) |
| `live_page.html`, `live_page1.html` | → Phase POSINT-D |
| `position_engine.py`, `position_engine1.py` | → Phase POSINT-A/B |
| `data_news.md` | → Phase NEWS (A–D) |
| `trading-dashboard.jsx` | React prototype — mock data patterns noted for demo mode |
| `requirements.txt` | ✅ Deps already in project |

---

## Pre-Retrain Readiness — Summary

> Full audit result: **v8 code is READY TO TRAIN.** All feature plumbing is wired end-to-end.

### ✅ Confirmed working
- `feature_contract.json` v8: 37 features, `asset_class_lookup` + `asset_id_lookup`, embedding dims (4+8=12), gate checks
- `HybridBreakoutCNN` v8: `nn.Embedding(5,4)` + `nn.Embedding(25,8)`, wider tabular head (37→256→128→64, GELU+BN)
- `_normalise_tabular_for_inference()`: v5→v4→v6→v7→v7.1→v8 backward-compat padding, all slots documented
- `_build_row()`: all 37 features computed with real data; v8-B uses `_bars_by_ticker`; v8-C uses `_daily_bars` + `_bars_1m`
- `train_model()`: grad accumulation (2×), mixup α=0.2, label smoothing 0.10, cosine warmup (5 epochs), separate LR groups, early stopping patience=15, NaN guard, grad clipping
- `BreakoutDataset.__getitem__()`: passes `asset_class_ids` and `asset_ids` as integer tensors
- `DatasetConfig` defaults: `breakout_type="all"`, `orb_session="all"`, caps 800/400
- `split_dataset()`: stratified by `(label, breakout_type, session)` triple
- Peer bar loading: `_resolve_peer_tickers()` → `bars_by_ticker` dict attached to each result
- Test suite: 2543 passed, 0 failed (smoke test: 31/31)

### ⚠️ Still open
- [x] CI/CD secrets — verification script created, TRAINER_SERVICE_URL fixed
- [x] ENGINE_DATA_URL port fix (8100→8050) — trainer compose + CI/CD workflow + verify script
- [x] Dataset smoke test — `scripts/smoke_test_dataset.py`
- [ ] **Run `verify_cicd.sh` on both machines** to confirm GitHub secrets match
- [ ] **Run `smoke_test_dataset.py` on trainer** to confirm bar loading works
- [ ] Generate v8 dataset

### 🚀 Deployment workflow (step by step)
1. `bash scripts/verify_cicd.sh --server` on cloud server — fix any failures
2. `bash scripts/verify_cicd.sh --trainer` on GPU rig — fix any failures (check ENGINE_DATA_URL port!)
3. `python scripts/smoke_test_dataset.py` on trainer — confirm bars load from engine over Tailscale
4. Use web UI on cloud server (`http://<server>:8180/trading`) to trigger dataset generation + training
5. Or trigger directly: `curl -X POST http://<trainer>:8200/train -H "Content-Type: application/json" -d '{"days_back": 120, "epochs": 80, "patience": 15}'`
