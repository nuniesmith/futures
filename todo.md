# futures — TODO

> **Last updated**: Phase MODEL-INT + PINE-INT — Model library & Pine generator integration, lint fixes, import path normalization

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

### 0. Async fill + trainer timeout mitigations (completed)
- [x] **`GET /bars/{symbol}` is now non-blocking** — fills fire in a background thread; response returns immediately with existing data + `filling: true` flag so the trainer never times out waiting for a long Massive backfill
  - Added `_SymbolFillJob` class + per-symbol fill registry (`_symbol_fills`)
  - Added `_get_or_start_symbol_fill()` helper — reuses a running job instead of spawning duplicates
  - `get_bars()` now checks `_bar_count()` and always fires async; returns stale-or-empty data immediately
  - Response payload gains `filling: bool` and `fill_status_url: str | null` fields
- [x] **`GET /bars/{symbol}/fill/status` endpoint added** — clients poll this until `status == "complete"` or `"failed"`, then re-fetch bars
- [x] **Trainer fill-poll loop added** (`_load_bars_from_engine`)
  - On `filling: true` response, polls `/bars/{symbol}/fill/status` every `ENGINE_FILL_POLL_INTERVAL` seconds (default 10)
  - Waits up to `ENGINE_FILL_POLL_MAX_WAIT` seconds (default 300 = 5 min) then re-fetches with `auto_fill=false`
  - Falls back gracefully to the partial data already returned if the fill takes too long or fails
- [x] **Engine bar-fetch timeout is now env-driven** — `ENGINE_BARS_TIMEOUT` (default 60s) replaces the hardcoded 60s in `_load_bars_from_engine`
- [x] **`BACKFILL_CHUNK_DAYS` documented** — expanded comment in `backfill.py` with recommended values (5 = safe default, 30 = Massive-optimised)
- [x] **New env vars wired into compose files** — `docker-compose.yml` (data + trainer services) and `docker-compose.trainer.yml`:
  - `data`: `BACKFILL_CHUNK_DAYS` (default 5; set to 30 when `MASSIVE_API_KEY` is in use)
  - `trainer`: `ENGINE_BARS_TIMEOUT`, `ENGINE_FILL_POLL_MAX_WAIT`, `ENGINE_FILL_POLL_INTERVAL`

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

### ✅ RITHMIC-D: Rate-Limit Monitoring & Safety
- [x] Daily action counter (`_daily_actions` + `_daily_reset_day`) — in-memory, resets at midnight, incremented per batch in `send_order_and_copy`
  - Rolling 60-min warn at 3,000 / hard stop at 4,500 (env: `CT_RATE_LIMIT_WARN` / `CT_RATE_LIMIT_HARD`)
  - `_check_rate_and_warn()` — called before every submission; logs throttled WARNING every 5 min in warn zone, logs ERROR + returns `False` at hard limit; replaces the old inline `is_hard_limit` check in `send_order_and_copy`
  - `get_rate_alert()` → `{level: "ok"|"warn"|"critical", message, actions_60min, daily_actions}` for WebUI polling
  - `get_rate_status()` and `status_summary["rate_limit"]` now include `daily_actions`
- [x] `RITHMIC_DEBUG_LOGGING=1` env var — sets `logging.getLogger("rithmic").setLevel(DEBUG)` when first account is added
- [x] `_detect_rate_limit_error(exc)` static method — scans exception message for "consumer slow", "rate limit", "throttl", "429", etc.
  - On match: logs ERROR + `asyncio.sleep(60s)` before returning the failed result from `_submit_single_order`

### ✅ RITHMIC-E: PositionManager Upgrades (One-Asset Focus + Pyramiding)
- [x] Focus lock: `_focus_asset: str | None` field on `PositionManager`
  - `can_trade(ticker)` — returns `False` for any ticker ≠ focused asset while a position is open (gated by `PM_FOCUS_LOCK=1` env var, default ON)
  - `set_focus(ticker)` — called in `_open_position()` after position is stored
  - `clear_focus()` — called in `_close_position()` (when last position closes), `close_all()`, `close_for_session_end()`
  - Focus asset persisted to Redis (`engine:position:_manager_state`) and restored on `load_state()`
  - `process_signal()` gates on `can_trade(ticker)` after core-watchlist check
- [x] Quality-gated pyramiding: `get_next_pyramid_level(pos, current_price, cnn_prob, *, regime, wave_ratio)`
  - 7 gates: pyramid enabled flag, CNN ≥ 65%, R-multiple threshold (L1=1R, L2=2R, L3=3R), level cap (CNN 65–79%→max L2, ≥80%→max L3), L3 extra gates (TRENDING regime + wave_ratio > 1.5), total contracts ≤ 3, 15-min cooldown
  - New stop per level: L1→breakeven, L2→entry+0.5R, L3→price−1R
- [x] `apply_pyramid(pos, action, current_price)` — mutates position pyramid fields, emits `[MARKET +1, MODIFY_STOP all_contracts]`
- [x] `process_signal()` — same-direction block replaced: calls `get_next_pyramid_level()` and `apply_pyramid()` when gates pass; falls back to hold-log when they don't
- [x] `MicroPosition` gains 4 new fields (default values — backward-compatible with existing Redis states): `pyramid_level`, `pyramid_contracts`, `pyramid_stop`, `last_pyramid_time`
- [x] `from_dict()` upgraded to use `dataclasses.fields()` filter — ignores unknown keys so old states load cleanly
- [x] Max risk: total contracts capped at 3 (≈1.5% account risk at typical micro sizing); env var `PM_PYRAMID_MAX_RISK_PCT=0.015`
- [x] `status_summary()` includes `pyramid_level`, `pyramid_contracts`, `total_contracts` per position + `focus_asset`, `focus_lock_enabled`, `pyramid_enabled` at top level

### RITHMIC-F: WebUI Integration
- [x] "SEND ALL" button on Live page → `CopyTrader.send_order_and_copy()` via `POST /api/copy-trade/send`
  - Inputs: asset, side (LONG/SHORT), limit price, qty, stop_ticks, optional target_ticks
  - Shows confirmation: "Main + N slaves, MANUAL flag, delay 200–800ms"
- [x] "ADD PYRAMID" button (`#ct-pyramid-btn`) — `ctAddPyramid()` → `POST /api/copy-trade/pyramid`
- [x] Compliance checklist widget in compliance modal (pre-market auto-checks from PM/CT state)
- [x] Account status cards (`#ct-accounts`) — HTMX-polled `GET /api/copy-trade/accounts/html`
- [x] Copy-trade log viewer (`#ct-log`) — `ctLoadHistory()` + `GET /api/copy-trade/history/html`
- [x] Rate-limit strip (`#ct-rate-strip`) — `GET /api/copy-trade/rate-alert` polled in `ctRefreshStatus()`
- [x] Focus lock status — `GET /api/copy-trade/focus` polled in `ctRefreshStatus()`, shown in `#ct-pyramid-info`
- [x] Ruby signal strip (`#ruby-signal-strip`) — HTMX `hx-get="/api/ruby/status/html" hx-trigger="every 30s"` above CT panel

### ✅ RITHMIC-G: Ruby Signal Engine (Pine → Python Port)
- [x] `src/lib/services/engine/ruby_signal_engine.py` — `RubySignalEngine` class
  - Full Pine Script v6 port: §1 settings, §2 core indicators (EMA9/VWAP/ATR14/AO/RSI), §3 Top G Channel
    (rolling lowest/highest + HMA mid + normalised ROC momentum), §4 Wave Analysis (EMA20 crossover
    tracking, bull/bear amplitude arrays, wave_ratio, cur_ratio), §5 Market Regime (SMA200 slope-norm +
    vol-norm → TRENDING ↑/↓ / VOLATILE / RANGING / NEUTRAL), §5b Market Phase (UPTREND/DOWNTREND/DISTRIB/
    ACCUM — sticky), §6 Volatility Percentile (rolling 200-bar ATR array → VERY HIGH/HIGH/MED/LOW/VERY LOW),
    §7 Session Bias + ORB (PD H/L, IB, ORB formation, bullBias 3-vote, ORB breakout detection, Squeeze BB/KC),
    §8 Quality Score (5 components → 0–100), §9 Main Signals (strong_bot/strong_top + 5-bar cooldown),
    §10 Entry/SL/TP levels (SL = ±1 ATR from wick, TP1/2/3 at configurable R multiples)
  - `update(bar)` → `RubySignal` dataclass — PositionManager-compatible (symbol, direction, trigger_price,
    breakout_detected, cnn_prob, cnn_signal, filter_passed, mtf_score, atr_value, range_high, range_low,
    regime, wave_ratio) plus full Ruby-specific fields (quality, phase, mkt_bias, bull_bias, vol_pct,
    vol_regime, ao, vwap, ema9, tg_hi/lo/mid/range, orb_high/low/ready, pd_high/low, ib_high/low/done,
    sqz_on/fired, entry, sl, tp1/2/3, risk, signal_class, is_orb_window, bar_time, computed_at)
  - `load_state()` / `save_state()` — Redis persistence for wave arrays, ORB/PD/IB levels, cooldown state
  - `status()` — summary dict for dashboard / API
  - `get_ruby_engine(symbol)` module-level singleton registry; `reset_ruby_engines()` for tests
  - Accepts both lower-case (`open/high/low/close/volume`) and Title-case (`Open/High/Low/Close/Volume`) bars,
    plus dict with `time` as ISO string / datetime / epoch float
  - ORB window detection: 30-min windows for NY RTH (09:30 ET), London (03:00 ET), Asia/CME (19:00 ET)
  - HMA helper using WMA(sqrt(n)) composition; EMA/RMA/SMA/ROC/ATR/RSI/AO/VWAP maths match Pine ta.*
  - Squeeze bands (BB inside KC) match Pine §7: SMA(20) ± 2σ vs SMA(20) ± ATR(20)×1.5
  - All internal state bounded: max_history bars (default 500), wave arrays ≤ 200 entries
  - Env vars: RUBY_TOP_G_LEN, RUBY_SIG_SENS, RUBY_ORB_MINUTES, RUBY_VOL_MULT, RUBY_MIN_QUALITY,
    RUBY_HTF_EMA_PERIOD, RUBY_TP1_R / RUBY_TP2_R / RUBY_TP3_R, RUBY_REQUIRE_VWAP, RUBY_IB_MINUTES,
    RUBY_BIAS_MODE, RUBY_STATE_TTL, RUBY_MAX_HISTORY
- [x] Wired into engine via `handle_ruby_recompute()` in `src/lib/services/engine/handlers.py`
  - Called by `_handle_fks_recompute()` in `main.py` (ActionType.RUBY_RECOMPUTE, every 5 min active session)
  - Incremental bar processing: tracks `engine:ruby_last_ts:{symbol}` in Redis so only new bars are fed
  - Per-symbol signals published to `engine:ruby_signal:{symbol}` (TTL 15 min)
  - Aggregate map published to `engine:ruby_signals` (TTL 15 min) for one-read dashboard refresh
  - Signals with `filter_passed=True` and `cnn_prob ≥ 0.45` forwarded to `dispatch_to_position_manager()`
- [x] `src/lib/services/data/api/ruby.py` — read-only API router (4 endpoints)
  - `GET /api/ruby/signals` — all symbols map `{symbol: RubySignal}`
  - `GET /api/ruby/signal/{symbol}` — single symbol; 404 with graceful body if not yet computed
  - `GET /api/ruby/status` — condensed summary strip (direction, quality, regime, wave_ratio, etc.)
  - `GET /api/ruby/status/html` — HTMX fragment of `.ruby-signal-card` divs; self-contained CSS
    (`.ruby-long/.ruby-short/.ruby-flat` borders; quality/bias/vol/SQZ/ORB badges; levels row on breakout)
  - Router registered in `src/lib/services/data/main.py`
  - All 4 routes proxied in `src/lib/services/web/main.py`
- [x] `static/trading.html` — Ruby signal strip card added above CT panel
  - `#ruby-panel` card with `#ruby-signal-strip` div; `hx-get="/api/ruby/status/html" hx-trigger="every 30s"`
  - `rubyRefresh()` JS function wired to manual ↺ button; falls back to `fetch()` when htmx not available
- [ ] `extract_features_for_cnn()` — Ruby features as additional CNN input channels (Phase v9 — deferred)
  - Top G position, wave ratio, regime enum, quality %, vol percentile
  - Wire into `RubyORB_CNN` hybrid model (only if >2% accuracy lift)

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

---

## ✅ Phase INDICATORS — Codebase Reorganization & Indicator Library Integration

> **Created**: Audit of `src/lib/analysis/` vs `reference/indicators/` — full code review completed.
> **Goal**: Copy the reference indicators library into the project, separate pure math/indicator logic from higher-level analysis/orchestration, eliminate duplication, and establish a clean architecture.

### Context & Audit Findings

The `reference/indicators/` directory has been copied into `src/lib/indicators/`. This is a well-structured indicator library with:
- **Base class** (`base.py`) — abstract `Indicator` with `calculate()` / `__call__()` / `get_value()` / `get_series()` / `reset()`
- **Registry** (`registry.py`) — singleton `IndicatorRegistry` with `@register_indicator` decorator
- **Factory** (`factory.py`) — static `IndicatorFactory` for creating indicators by name or config
- **Manager** (`manager.py`) — `IndicatorManager` for grouping, batch calculation, serialization
- **Categorized indicators**: `trend/`, `momentum/`, `volatility/`, `volume/`, `other/`
- **Pattern detection**: `candle_patterns.py`, `areas_of_interest.py`, `patterns.py`
- **Crypto-optimized**: `indicators.py` (BTC-tuned thresholds, Wyckoff/Stop-Hunt patterns)
- **Market timing**: `market_timing.py` (session analysis, should-trade-now logic)

**Key problems found:**
1. **Two indicator architectures coexist** — registry-based (`Indicator` ABC with lowercase columns) vs standalone classes (`update()`/`apply()` with capitalized `Close`/`High` columns)
2. **Duplicate files** — `market_cycle.py` and `parabolic_sar.py` each exist at top-level AND inside `other/`
3. **Column naming inconsistency** — registry indicators use `close`, standalone use `Close`
4. **`indicators.py` is a crypto fork** of `candle_patterns.py` + `areas_of_interest.py` with overlapping functions
5. **`src/lib/analysis/` mixes concerns** — pure math (volatility, wave, CVD, ICT, signal_quality) lives alongside orchestration (news_sentiment, reddit_sentiment, scorer, chart renderers)
6. **Inline indicator math in analysis files** — `confluence.py`, `breakout_filters.py`, `crypto_momentum.py`, `mtf_analyzer.py` all have their own `_ema()`, `_rsi()`, `_atr()` helper functions instead of using a shared indicator library
7. **`strategy_defs.py` also has inline indicators** — `_ema`, `_sma`, `_atr`, `_rsi`, `_macd_line`, `_macd_signal`, `_macd_histogram` duplicated there too
8. **Broken import paths** — indicators reference `core.component.base.Component`, `core.registry.base.Registry`, `core.validation.validation.validate_dataframe`, `app.trading.indicators.Indicator` — none of which exist in this project

---

### ✅ INDICATORS-A: Fix Import Paths & Establish Base Classes *(blocking — do first)*

The copied indicators have import paths from the old project (`core.component.base`, `core.registry.base`, `core.validation.validation`, `app.trading.indicators`). These must be adapted to work in this project.

- [x] **A1**: Created `src/lib/indicators/_shims.py` — minimal shims for `Component`, `Registry`, `validate_dataframe`, `process_data`
- [x] **A2**: Updated `base.py` imports to use local `_shims` module
- [x] **A3**: Updated `registry.py` imports to use local `_shims` module
- [x] **A4**: Updated `factory.py` + `manager.py` — fixed all `src.lib.indicators` absolute imports to `lib.indicators` relative
- [x] **A5**: `candle_patterns.py` and `areas_of_interest.py` — utils replaced with inline defaults
- [x] **A6**: `market_timing.py` — utils imports resolved
- [x] **A7**: `src/lib/indicators/__init__.py` loads cleanly — `from lib.indicators import indicator_categories` verified green
- [x] **A8**: (smoke test deferred — covered by B/C/E verification runs)

**Acceptance**: `python -m pytest tests/test_indicators_smoke.py` passes green.

---

### ✅ INDICATORS-B: Resolve Duplicates & Column Inconsistencies

- [x] **B1**: Confirmed no duplicate `indicators/market_cycle.py` at top-level — only exists in `other/`
- [x] **B2**: Confirmed no duplicate `indicators/parabolic_sar.py` at top-level — only exists in `other/`
- [x] **B3**: Column naming documented — registry-based indicators use lowercase, standalones use Title-case; `helpers.py` bridges both via case-insensitive col lookup. Full refactor deferred (low risk — `helpers.py` wraps both)
- [x] **B4**: `_ChoppinessIndexAdapter` created in `presets.py` as the adapter pattern for standalone → registry-compatible classes
- [x] **B5**: `indicators/__init__.py` fully rewritten — removed broken `from core import Indicator`, fixed `volatility` import path (`trend/volatility/`), removed missing `gold/` import, added `EMA`/`WMA`/`VWAP` from `moving_average.py`
- [x] **B6**: `indicators.py` crypto functions kept as-is (unique crypto logic); duplicate generic overlap documented — full extraction deferred to a future refactor

**Acceptance**: `python -c "from lib.indicators import indicator_categories; print(list(indicator_categories.keys()))"` → `['trend', 'volatility', 'momentum', 'volume', 'other']` ✅

---

### ✅ INDICATORS-C: Extract Pure Math from Analysis into Indicators

Several `src/lib/analysis/` modules contain inline indicator helper functions that should be replaced with calls to the indicator library.

- [x] **C1**: Created `src/lib/indicators/helpers.py` — `ema`, `ema_numpy`, `sma`, `rsi`, `rsi_scalar`, `atr`, `atr_scalar`, `macd`, `macd_numpy`, `bollinger`, `vwap`, `awesome_oscillator`
- [x] **C2–C4**: `analysis/confluence.py` already imports from `lib.core.utils` (done in prior session)
- [x] **C5**: `analysis/breakout_filters.py` — replaced local `_ema` with `from lib.indicators.helpers import ema_numpy as _ema`
- [x] **C6**: `analysis/crypto_momentum.py` — `compute_ema`/`compute_rsi`/`compute_atr` now delegate to `ema_numpy`/`rsi_scalar`/`atr_scalar`
- [x] **C7**: `analysis/mtf_analyzer.py` — replaced local `_ema_series`/`_macd` with `from lib.indicators.helpers import ema_numpy as _ema_series, macd_numpy as _macd_numpy`
- [x] **C8**: `analysis/volatility.py` — kept as-is (RMA/Wilder smoothing intentionally differs from EWM; not a duplicate)
- [x] **C9**: `analysis/signal_quality.py` — `_compute_rsi = rsi_scalar`, `_compute_ao = awesome_oscillator`
- [x] **C10**: `analysis/chart_renderer.py` (now at `rendering/`) — `_compute_ema` and `_compute_vwap` delegate to helpers
- [x] **C11**: `trading/strategies/strategy_defs.py` — already imports `_ema`/`_atr`/`_rsi` from `lib.core.utils` (done in prior session); `_macd_*` helpers kept local (backtesting.py compatibility)
- [x] **C12**: `trading/strategies/rb/range_builders.py` — kept as-is (DataFrame-based, returns Series; different interface from scalar helpers)
- [x] **C13**: Verified numerically via inline smoke tests

**Acceptance**: All inline duplicates removed or delegated. Helpers smoke test passes ✅

---

### ✅ INDICATORS-D: Reorganize `src/lib/analysis/` — Separate Concerns

Split the analysis directory into clear layers: pure computation vs orchestration vs rendering.

#### Current `analysis/` files → proposed new homes:

| File | Category | Proposed Location |
|------|----------|-------------------|
| `volatility.py` | Pure math (K-Means vol clusters) | **Stay** in `analysis/` — it's analysis-level (uses indicators as inputs) |
| `wave_analysis.py` | Pure math (wave dominance) | **Stay** in `analysis/` — asset-specific analysis |
| `cvd.py` | Pure math (CVD + divergence) | **Stay** in `analysis/` — composite analysis built on volume delta |
| `ict.py` | Pure math (ICT/SMC structures) | **Stay** in `analysis/` — structural analysis |
| `signal_quality.py` | Pure math (multi-factor scoring) | **Stay** in `analysis/` — scoring layer |
| `regime.py` | Pure math (HMM regime detection) | **Stay** in `analysis/` — regime classification |
| `confluence.py` | Pure math (MTF confluence) | **Stay** in `analysis/` — multi-indicator scoring |
| `mtf_analyzer.py` | Pure math (MTF EMA/MACD) | **Stay** in `analysis/` — multi-indicator scoring |
| `breakout_filters.py` | Pure math (quality gates) | **Stay** in `analysis/` — filter logic |
| `cross_asset.py` | Pure math (correlations) | **Stay** in `analysis/` — cross-asset analysis |
| `asset_fingerprint.py` | Pure math (asset profiling) | **Stay** in `analysis/` — profiling |
| `volume_profile.py` | Pure math + backtest strategy | **Stay** in `analysis/` — the strategy class could later be extracted |
| `crypto_momentum.py` | Hybrid (math + data fetching) | **Split** — math helpers → use `indicators/helpers.py`, orchestrator stays |
| `breakout_cnn.py` | ML model (training + inference) | Move to `analysis/ml/` or `analysis/cnn/` — it's large enough for its own subpackage |
| `chart_renderer.py` | Rendering (mplfinance PNGs) | Move to `analysis/rendering/` |
| `chart_renderer_parity.py` | Rendering (Pillow PNGs) | Move to `analysis/rendering/` |
| `news_sentiment.py` | Orchestration (API + Redis + NLP) | Move to `analysis/sentiment/` |
| `reddit_sentiment.py` | Orchestration (Redis aggregation) | Move to `analysis/sentiment/` |
| `scorer.py` | Orchestration (pre-market ranking) | **Stay** in `analysis/` — it's a scoring orchestrator |

- [x] **D1**: Created `src/lib/analysis/rendering/` with `__init__.py`
- [x] **D2**: Moved `chart_renderer.py` → `analysis/rendering/chart_renderer.py`
- [x] **D3**: Moved `chart_renderer_parity.py` → `analysis/rendering/chart_renderer_parity.py`
- [x] **D4**: `analysis/rendering/__init__.py` re-exports full public API with `try/except ImportError` guard
- [x] **D5**: Created `src/lib/analysis/sentiment/` with `__init__.py`
- [x] **D6**: Moved `news_sentiment.py` → `analysis/sentiment/news_sentiment.py`
- [x] **D7**: Moved `reddit_sentiment.py` → `analysis/sentiment/reddit_sentiment.py`
- [x] **D8**: `analysis/sentiment/__init__.py` re-exports full public API
- [x] **D9**: Created `src/lib/analysis/ml/` with `__init__.py`
- [x] **D10**: Moved `breakout_cnn.py` → `analysis/ml/breakout_cnn.py`
- [x] **D11**: `analysis/ml/__init__.py` re-exports full public API with `try/except ImportError` guard for torch
- [x] **D12**: `analysis/__init__.py` updated to pull from new sub-package paths; all old names still re-exported
- [x] **D13**: Updated 13 files with direct import paths: `services/data/api/reddit.py`, `services/data/api/news.py`, `services/data/api/cnn.py`, `services/data/main.py`, `services/engine/main.py`, `services/engine/handlers.py`, `services/engine/model_watcher.py`, `services/training/dataset_generator.py`, `services/training/trainer_server.py`, `tests/test_breakout_types.py`, `tests/test_kraken_training_pipeline.py`, `tests/test_v8_smoke.py`
- [x] **D14**: Zero new errors from reorganization (pre-existing torch type-check warnings unchanged)

**Acceptance**: Sub-packages confirmed present. `analysis/__init__.py` re-exports all symbols. All service imports updated ✅

---

### ✅ INDICATORS-E: Wire Indicators into the Analysis Pipeline (partial)

- [x] **E1**: `indicators/__init__.py` now imports `helpers` and `presets` sub-modules; `indicator_categories` is clean
- [ ] **E2**: `analysis/confluence.py` — optional `IndicatorManager` integration deferred (medium effort, no blocking need yet)
- [ ] **E3**: `analysis/mtf_analyzer.py` — pre-computed indicator injection deferred
- [ ] **E4**: `check_bollinger_squeeze` in `breakout_filters.py` deferred
- [x] **E5**: Created `src/lib/indicators/presets.py` — `SCALP_PRESET`, `SWING_PRESET`, `REGIME_PRESET`, `build_manager()`. Includes `_ChoppinessIndexAdapter` bridge for standalone→registry compatibility and `_make_indicator_name()` for unique naming of same-class instances
- [ ] **E6**: Integration tests deferred

**Acceptance**: `SCALP_PRESET`, `SWING_PRESET`, `REGIME_PRESET` importable; `build_manager(SCALP_PRESET)` verified ✅

---

### INDICATORS-F: Reference Code — Evaluate & Decide

The `reference/` directory (now deleted) contained additional code beyond indicators. Here's the triage of what was useful vs not needed:

#### ✅ Already integrated (copied):
- `reference/indicators/` → copied to `src/lib/indicators/`

#### 🟡 Potentially useful — evaluate later:
| Reference Module | What It Does | Project Equivalent | Recommendation |
|---|---|---|---|
| `reference/strategy/performance/tracker.py` | `PerformanceTracker` — equity curve, Sharpe, drawdown | Partial overlap with `services/engine/risk.py` | **Evaluate** — could enhance the existing risk module |
| `reference/strategy/positions/manager.py` | Signal-driven position manager with risk-based sizing | Already have `services/engine/position_manager.py` | **Skip** — existing is more mature (bracket phases, TP1-3, EMA trail) |
| `reference/strategy/optimization/parameter.py` | Optuna optimizer with Pine Script export | Already have `trading/engine.py` `run_optimization` | **Skip** — existing Optuna integration is sufficient |
| `reference/strategy/walk_forward/tester.py` | Walk-forward analysis with robust params | No direct equivalent | **Future** — valuable for v9+ CNN training validation |
| `reference/strategy/models/ensemble.py` | Weighted voting ensemble of multiple strategies | No direct equivalent | **Future** — useful when running multiple strategy variants |
| `reference/strategy/selector.py` | Dynamic strategy selection (ADX + vol) | Partial overlap with `analysis/regime.py` | **Skip** — regime detector serves this purpose |
| `reference/strategy/signals/generator.py` | Configurable signal generator wrapper | No direct equivalent | **Low priority** — simple wrapper, easy to build when needed |
| `reference/helpers/interpolation.py` | Gap-filling strategies (linear/quadratic/polynomial) | No equivalent | **Future** — useful for data quality pipeline |
| `reference/helpers/retry.py` | Async retry with exponential backoff | No equivalent | **Future** — useful for integration clients |
| `reference/helpers/validate_config.py` | YAML/JSON config validation against schemas | No equivalent | **Future** — nice-to-have for config safety |
| `reference/strategy/backtest/` | Full backtesting framework (broker, portfolio, engine) | Already have `trading/strategies/backtesting.py` + `backtesting.py` lib | **Skip** — existing framework is simpler and working |
| `reference/strategy/analyzers/` | Backtrader-style post-backtest analyzers | No direct equivalent | **Low priority** — would need Backtrader integration |
| `reference/strategy/confirmations/` | Signal confirmation logic (pullbacks, scoring) | Partial overlap with `analysis/signal_quality.py` | **Skip** — existing scorer is more comprehensive |
| `reference/strategy/core/` | Base strategy + enums + risk + execution | Covered by `trading/strategies/` | **Skip** — different architecture |
| `reference/strategy/trainer.py` | LogReg + XGBoost model training | Already have CNN pipeline in `breakout_cnn.py` | **Skip** — different ML approach |
| `reference/core/` | Full app infrastructure (lifecycle, middleware, websocket, telemetry) | Already have `services/` architecture | **Skip** — completely different framework |
| `reference/helpers/cache.py`, `staging.py`, `chunk.py` etc. | Data pipeline helpers for the old project | Already have `services/data/resolver.py` + `cache.py` | **Skip** — old project's data pipeline |

#### ❌ Not needed:
- `reference/core/` — entire old app infrastructure (lifecycle, middleware, websocket, telemetry, security, etc.) — incompatible architecture
- `reference/strategy/backtest/` — full Backtrader-style engine — we use the `backtesting.py` library instead
- `reference/strategy/assets/` — asset classification/aliasing — we have `core/asset_registry.py` which is more complete
- `reference/helpers/fetch_stage.py`, `process_data.py`, `staging.py` — old data pipeline, we have `services/data/resolver.py`

---

### ✅ INDICATORS-G: Cleanup & Documentation (partial)

- [x] **G1**: `src/lib/indicators/__init__.py` has full package-structure docstring
- [x] **G2**: Created `src/lib/indicators/README.md` — covers package layout, all 4 usage patterns (direct, factory, manager, helpers, presets), column naming conventions, and new-indicator skeleton
- [ ] **G3**: Verify `reference/` directory deleted — run `ls futures/reference/` to confirm (not code-changed)
- [ ] **G4**: `docs/architecture.md` update deferred
- [ ] **G5**: Full lint pass deferred (only style/deprecation warnings remain — no errors)
- [ ] **G6**: `__all__` exports in sub-package `__init__.py` files deferred

**Acceptance**: README and `__init__.py` docstring done ✅. Remaining items are polish/docs only.

---

### Task Execution Order

```
✅ INDICATORS-A (fix imports)      ← DONE
    ↓
✅ INDICATORS-B (deduplicate)      ← DONE
    ↓
✅ INDICATORS-C (extract math)     ← DONE
    ↓
✅ INDICATORS-D (reorg analysis)   ← DONE
    ↓
✅ INDICATORS-E (presets + wire)   ← DONE (E2/E3/E4/E6 deferred — low priority)
    ↓
   INDICATORS-F (evaluate ref)     ← advisory only, no code needed
    ↓
   INDICATORS-G (cleanup + docs)   ← G3/G4/G5/G6 remaining (polish only)
```

**Status**: Core work complete. Remaining items are polish (lint, __all__, architecture docs) and optional enhancements (IndicatorManager injection into confluence/mtf_analyzer).

---

## ✅ Phase TRAINER-UX — Trainer Page Redesign & Defaults Update

> **Completed**: Trainer page redesigned, defaults updated, charting proxy added.

### What was done

#### Trainer defaults updated
| Parameter | Old Default | New Default |
|-----------|------------|-------------|
| `CNN_RETRAIN_EPOCHS` | 25 | **60** |
| `CNN_RETRAIN_LR` | 0.0002 | **0.0001** |
| `CNN_RETRAIN_PATIENCE` | 8 | **12** |
| `CNN_RETRAIN_DAYS_BACK` | 90 | **180** |

Files changed: `docker-compose.yml`, `docker-compose.trainer.yml`, `trainer_server.py`

#### Trainer page redesign
- [x] Logs moved to full-width section at bottom (400px min-height, pre-wrap, user-select)
- [x] Added 📋 Copy All button for log output
- [x] Layout changed from 3-column to 2-column grid + full-width bottom sections
- [x] Added step buttons: 📥 Load Data, 🗃 Generate Dataset, 🧠 Train Model, 🚀 Full Pipeline
- [x] Form defaults updated to match new hyperparameters (60 epochs, 0.0001 LR, 12 patience, 180 days)
- [x] Model Archive moved to full-width section below logs

#### Charting container connectivity fixed
- [x] Created `src/lib/services/data/api/charting_proxy.py` — reverse proxy for `/charting-proxy/*`
- [x] Updated `dashboard.py` `charts_page()` — iframe now uses `/charting-proxy/` (relative URL)
- [x] Added `CHARTING_SERVICE_URL=http://charting:8003` to data service environment in `docker-compose.yml`
- [x] Registered proxy router in `src/lib/services/data/main.py`
- [x] Charts now work from any machine on the network (no more `localhost:8003` requirement)

#### Dead code removed
- [x] Deleted `src/lib/trading/strategies/backtesting.py` (1,340 lines — stale copy of `engine.py`, zero importers)

### Files changed
- `src/lib/services/data/api/trainer.py` — full page HTML redesign
- `src/lib/services/data/api/charting_proxy.py` — **new file** (reverse proxy)
- `src/lib/services/data/api/dashboard.py` — charts iframe → proxy URL
- `src/lib/services/data/main.py` — registered charting proxy router
- `docker-compose.yml` — trainer defaults + charting URL on data service
- `docker-compose.trainer.yml` — trainer defaults
- `src/lib/services/training/trainer_server.py` — trainer defaults
- `src/lib/trading/strategies/backtesting.py` — **deleted**

---

## 🟢 Phase CLEANUP — Codebase Audit Findings & Consolidation *(A, B, D complete — C remaining)*

> **Created**: Full audit of `src/lib/` completed. Findings below with prioritized task list.
> **Goal**: Eliminate duplication, consolidate shared utilities, split oversized files, remove dead code.

### Audit Findings Summary

| Category | Count | Est. Lines Affected |
|---|---|---|
| Near-duplicate files | 1 remaining (indicators) | ~260 |
| Duplicate utility functions | 6 patterns, ~20 copies | ~300 |
| Broken import paths (`indicators/`) | 14+ files | documented in Phase INDICATORS-A |
| Dead/unused files | `indicators/` package (zero importers) | ~5,700 |
| Misplaced code | 1 strategy class in `analysis/` | ~130 |
| Duplicated handler functions | 1 (`_run_mtf_on_result`) | ~50 |
| Files >1,000 lines needing splits | 6 files | N/A |

---

### CLEANUP-A: Consolidate Shared Utility Functions *(medium priority)*

#### A1: `_safe_float()` — 8 identical copies across 8 files

| File | Location |
|---|---|
| `analysis/ict.py` | ~L64 |
| `services/data/api/sse.py` | ~L967 |
| `services/engine/focus.py` | ~L79 |
| `services/engine/patterns.py` | ~L125 |
| `trading/strategies/daily/bias_analyzer.py` | ~L197 |
| `trading/strategies/daily/daily_plan.py` | ~L380 |
| `trading/strategies/daily/swing_detector.py` | ~L288 |
| `trading/strategies/strategy_defs.py` | ~L1518 |

- [x] **A1a**: Add `safe_float(value, default=0.0) -> float` to `lib/core/utils.py` (new file) — **done**
- [x] **A1b**: Replace 7/8 copies with `from lib.core.utils import safe_float as _safe_float` — **done** (`sse.py` nested closure left with TODO comment)

#### A2: `_ema()` — 4 copies across 4 files

| File | Notes |
|---|---|
| `analysis/breakout_filters.py` | numpy loop version |
| `analysis/confluence.py` | pandas ewm one-liner |
| `analysis/volume_profile.py` | pandas ewm one-liner |
| `trading/strategies/strategy_defs.py` | pandas ewm one-liner |

- [x] **A2**: Consolidated `ema()` + `ema_numpy()` into `lib/core/utils.py`; replaced copies in `confluence.py`, `volume_profile.py`, `strategy_defs.py` with `from lib.core.utils import ema as _ema` — **done** (`breakout_filters.py` numpy version left as-is, different algorithm)

#### A3: `_atr()` — 4 copies across 4 files

| File |
|---|
| `analysis/confluence.py` |
| `analysis/ict.py` |
| `analysis/volume_profile.py` |
| `trading/strategies/strategy_defs.py` |

- [x] **A3**: Consolidated `atr()` into `lib/core/utils.py`; replaced copies in `confluence.py`, `volume_profile.py`, `strategy_defs.py` with `from lib.core.utils import atr as _atr` — **done** (`ict.py` numpy version left as-is, different implementation)

#### A4: `_rsi()` — 2 copies

| File |
|---|
| `analysis/confluence.py` |
| `trading/strategies/strategy_defs.py` |

- [x] **A4**: Consolidated `rsi()` into `lib/core/utils.py`; replaced copies in `confluence.py` and `strategy_defs.py` with `from lib.core.utils import rsi as _rsi` — **done**

#### A5: `compute_atr()` — 3 copies (Wilder-smoothed)

| File | Notes |
|---|---|
| `analysis/crypto_momentum.py` | independent copy |
| `trading/strategies/rb/open/detector.py` | documented wrapper |
| `trading/strategies/rb/range_builders.py` | canonical version |

- [x] **A5**: `crypto_momentum.compute_atr` has incompatible signature (`np.ndarray → float`) vs `range_builders` (`DataFrame → Series`). Left in place with expanded docstring cross-referencing both canonical versions (`range_builders.compute_atr`, `core.utils.atr`) — **done**

#### A6: `_run_mtf_on_result()` — 2 copies

| File |
|---|
| `services/engine/handlers.py` (public) |
| `services/engine/main.py` (private copy) |

- [x] **A6**: `_run_mtf_on_result` in `main.py` was dead code (no callers) — deleted. Canonical `run_mtf_on_result` lives in `handlers.py` — **done**

---

### CLEANUP-B: Misplaced Code *(low priority)*

- [x] **B1**: Moved `VolumeProfileStrategy` from `analysis/volume_profile.py` → `trading/strategies/strategy_defs.py`; backwards-compatible re-export kept in `volume_profile.py` — **done**
- [x] **B2**: `suggest_volume_profile_params()` moved alongside the class to `strategy_defs.py`; re-exported from `volume_profile.py` — **done**

---

### CLEANUP-C: Split Oversized Files *(low priority, do incrementally)*

| File | Lines | Recommendation |
|---|---|---|
| `services/data/api/dashboard.py` | **~6,500** | Split into `dashboard/grid.py`, `dashboard/focus.py`, `dashboard/charts.py`, `dashboard/helpers.py` |
| `services/training/rb_simulator.py` | **~3,350** | Split per breakout type or into `sim_core.py` + `sim_batch_*.py` |
| `analysis/breakout_cnn.py` | **~3,150** | Split into `ml/model.py`, `ml/training.py`, `ml/inference.py`, `ml/features.py` |
| `services/training/dataset_generator.py` | **~2,930** | Extract `_build_row()` (~500 lines) into `feature_builder.py` |
| `services/engine/main.py` | **~2,670** | Extract scheduling handlers into `handlers_scheduled.py` |
| `core/models.py` | **~2,130** | Split into `models.py`, `constants.py`, `db.py` |

- [ ] **C1**: Split `dashboard.py` into sub-package (highest impact — most-edited file)
- [ ] **C2**: Extract `_build_row()` from `dataset_generator.py` into `feature_builder.py`
- [ ] **C3**: Split `breakout_cnn.py` into `analysis/ml/` sub-package (already planned in INDICATORS-D9–D11)
- [ ] **C4**: Split remaining files as time permits

---

### CLEANUP-D: Dead Code Removal *(low priority)*

- [x] ~~`trading/strategies/backtesting.py`~~ — **deleted** (Phase TRAINER-UX)
- [x] **D1**: `indicators/` package — all broken imports fixed (shims created, `core.component.base`, `core.registry.base`, `core.validation`, `utils.datetime_utils`, `utils.config_utils` all shimmed); package is now 0 errors — **done**
- [x] **D2**: `indicators/market_cycle.py` top-level duplicate — already absent (deleted previously) — **done**
- [x] **D3**: `indicators/parabolic_sar.py` top-level duplicate — already absent (deleted previously) — **done**

---

### Task Execution Order

```
✅ CLEANUP-A (shared utils)     ← DONE
    ↓
CLEANUP-B (misplaced code)      ← low priority
    ↓
CLEANUP-C (file splits)     ← do incrementally, dashboard.py first
    ↓
CLEANUP-D (dead code)       ← overlaps with Phase INDICATORS, coordinate
```

**Note**: CLEANUP-A overlaps significantly with INDICATORS-C (extract math from analysis). If doing both, create `lib/core/utils.py` first (CLEANUP-A1), then build `lib/indicators/helpers.py` on top of it (INDICATORS-C1). The indicator helpers would be the canonical source for `_ema`, `_atr`, `_rsi`, while `core/utils.py` holds generic helpers like `safe_float`.

---

## 🔴 Phase MODEL-INT — Model Library Integration & Lint Fixes

> **Source**: `src/lib/model/` (42 files copied from `fks_python/model/`)
> **Current state**: 2,713 ruff errors, broken import paths, syntax errors, no `__init__.py` exports, zero importers in the project.
> **Goal**: Make all files pass `ruff check` and `mypy`, fix import paths to use `lib.model.*` namespace, wire `__init__.py` exports, guard heavy deps behind `try/except ImportError`.

### Audit Summary

| Category | Count | Description |
|---|---|---|
| **Syntax errors** | 10 | Missing `from` keyword in imports (`prediction/generator.py`, `prediction/manager.py`, `prediction/multi.py`), concatenated statements (`manager.py` L14) |
| **Broken import paths** | 12 | `from src.lib.model.estimator` → `from lib.model.base.estimator`, `from core.exceptions.model` → doesn't exist, `from src.lib.core.utils.logging_utils` → doesn't exist, `from strategy.evaluator` → doesn't exist, `from core.constants.manager` → doesn't exist |
| **Runtime bugs** | 3 | `pl.training(...)` should be `pl.Trainer(...)` in `service.py`, `deep/lstm.py`, `deep/tft.py`; `nn.py` line-break splits `np.mean` call |
| **Whitespace/style (auto-fixable)** | ~2,048 | `W293` blank-line-whitespace (2,041), `W291` trailing-whitespace (158), `W292` missing-newline (27) |
| **Deprecated typing (auto-fixable)** | ~322 | `UP006` non-pep585 (131), `UP045` non-pep604-optional (91), `UP035` deprecated-import (53), `UP007` non-pep604-union (47) |
| **Unused imports** | 39 | Various `F401` across all files |
| **Missing `super().__init__()`** | 7 | `SimpleNN`, `EnhancedNN`, `ConcreteNN`, `LSTMModel`, `LogisticRegressionModel`, `HMMModel`, `ProphetModel` |
| **Missing external deps** | 3 | `loguru`, `arch`, `hmmlearn` not in `pyproject.toml` (others like `torch`, `sklearn`, `scipy` already present) |

### MODEL-INT-A: Fix Syntax Errors & Broken Imports *(blocking — do first)*

**Files with syntax errors (must fix before any other linting works):**

- [ ] **A1**: `src/lib/model/prediction/generator.py` — add missing `from` keyword on lines 12-13, fix `core.constants.manager` import
- [ ] **A2**: `src/lib/model/prediction/manager.py` — add missing `from` keyword on lines 10, 12, 13; split concatenated statements on line 14; fix `models.gaussian` → `lib.model.ml.gaussian`, `models.polynomial` → `lib.model.ml.polynomial`, `core.constants.manager` → stub or remove
- [ ] **A3**: `src/lib/model/prediction/multi.py` — add missing `from` keyword on lines 6-7; fix `data.manager` → stub or remove, `prediction.single` → `lib.model.prediction.single`

**Files with wrong import paths (all `from src.lib.model.X` → `from lib.model.X`):**

- [ ] **A4**: `src/lib/model/base/classifier.py` — `from src.lib.model.estimator` → `from lib.model.base.estimator`
- [ ] **A5**: `src/lib/model/base/regressor.py` — `from src.lib.model.estimator` → `from lib.model.base.estimator`
- [ ] **A6**: `src/lib/model/base/estimator.py` — remove broken `from core.models.base` and `from core.validation.dataframe` imports; have `Estimator` inherit from `lib.model.base.model.BaseModel` instead
- [ ] **A7**: `src/lib/model/ml/xgboost.py` — fix `from src.lib.model.regressor` → `from lib.model.base.regressor`, `from src.lib.model.classifier` → `from lib.model.base.classifier`, `from core.exceptions.model` → create shim or inline `ModelError`
- [ ] **A8**: `src/lib/model/statistical/arima.py` — fix `from src.lib.model.estimator` → `from lib.model.base.estimator`, `from core.exceptions` → inline `ModelError`, remove duplicate `StatsARIMAModel` import
- [ ] **A9**: `src/lib/model/statistical/garch.py` — same pattern: fix `model.estimator` and `core.exceptions.model`
- [ ] **A10**: `src/lib/model/factory.py` — fix all `from src.lib.model.*` → `from lib.model.*`
- [ ] **A11**: `src/lib/model/registry.py` — fix `from src.lib.model.base.model` → `from lib.model.base.model`, `from src.lib.model.utils.metadata` → `from lib.model.utils.metadata`
- [ ] **A12**: `src/lib/model/deep/lstm.py` — fix `from src.lib.model.base.model` → `from lib.model.base.model`, `from src.lib.model.utils.metadata` → `from lib.model.utils.metadata`
- [ ] **A13**: `src/lib/model/deep/nn.py` — fix `from src.lib.model.base.model` → `from lib.model.base.model`
- [ ] **A14**: `src/lib/model/deep/tft.py` — fix `from src.lib.model.base.model` → `from lib.model.base.model`, remove `from core.constants.manager`, remove `from utils.data_utils`
- [ ] **A15**: `src/lib/model/ensemble/ensemble.py` — fix `from src.lib.model.base.model` → `from lib.model.base.model`, `from src.lib.model.utils.metadata` → `from lib.model.utils.metadata`
- [ ] **A16**: `src/lib/model/evaluation/service.py` — guard `pytorch_forecasting` import with `try/except`
- [ ] **A17**: `src/lib/model/ml/logistic.py` — fix `from src.lib.model.base.model`, remove `from strategy.evaluator`
- [ ] **A18**: `src/lib/model/statistical/hmm.py` — fix `from src.lib.model.base.model`, remove `from strategy.evaluator`
- [ ] **A19**: `src/lib/model/statistical/prophet.py` — fix `from src.lib.model.base.model`, remove duplicate local `log_execution` def
- [ ] **A20**: `src/lib/model/service.py` — fix all internal `model.*` references
- [ ] **A21**: `src/lib/model/persistence.py` — `from loguru import logger` → guard or replace

**Agent prompt**: *"Fix all import paths in `src/lib/model/` to use the `lib.model.*` namespace. Fix syntax errors in `prediction/generator.py`, `prediction/manager.py`, `prediction/multi.py`. Replace all `from src.lib.model.X` with `from lib.model.X`. Replace all `from core.exceptions.model import ModelError` with an inline `class ModelError(Exception): pass` at the top of files that need it. Replace all `from src.lib.core.utils.logging_utils import log_execution` with a no-op shim: `def log_execution(func): return func`. Replace all `from strategy.evaluator import ModelEvaluator` with `ModelEvaluator = None`. Guard `loguru` with `try: from loguru import logger; except ImportError: import logging; logger = logging.getLogger(__name__)`. Do NOT change any business logic — only fix imports and add shims."*

**Acceptance**: `ruff check src/lib/model/ --select E,F --no-fix` reports zero `E999` (syntax) and zero `F821` (undefined name) errors.

---

### MODEL-INT-B: Create Shims for Missing External Dependencies

- [ ] **B1**: Create `src/lib/model/_shims.py` with:
  - `ModelError` exception class
  - `log_execution` no-op decorator
  - `logger` that falls back to stdlib `logging` when `loguru` is unavailable
  - `ModelEvaluator = None` stub
  - `DEFAULT_DEVICE` constant (default `"cpu"`)
- [ ] **B2**: Update all model files to import from `lib.model._shims` instead of `core.exceptions`, `utils.logging_utils`, `strategy.evaluator`, `core.constants.manager`
- [ ] **B3**: Guard ALL heavy external deps with `try/except ImportError`:
  - `torch` / `lightning` in `deep/lstm.py`, `deep/tft.py`
  - `pytorch_forecasting` in `deep/tft.py`, `evaluation/service.py`
  - `xgboost` in `ml/xgboost.py`
  - `statsmodels` in `statistical/arima.py`
  - `arch` in `statistical/garch.py`
  - `hmmlearn` in `statistical/hmm.py`
  - `prophet` in `statistical/prophet.py`
  - `pymc` / `arviz` in `statistical/bayesian.py`
  - `scipy` in `ml/gaussian.py`, `ml/polynomial.py`, `ensemble/ensemble.py`
  - `sklearn` in `base/model.py`, `deep/nn.py`, `ml/logistic.py`, `prediction/single.py`
  - `matplotlib` in `ml/gaussian.py`, `ml/polynomial.py`, `statistical/bayesian.py`, `evaluation/service.py`

**Agent prompt**: *"Create `src/lib/model/_shims.py` with compatibility shims (ModelError exception, log_execution no-op decorator, loguru-to-stdlib logger fallback, ModelEvaluator=None, DEFAULT_DEVICE='cpu'). Then update every file in `src/lib/model/` that imports from `utils.logging_utils`, `core.exceptions`, `strategy.evaluator`, or `core.constants.manager` to import from `lib.model._shims` instead. Also wrap every external optional dependency import (`torch`, `lightning`, `pytorch_forecasting`, `xgboost`, `statsmodels`, `arch`, `hmmlearn`, `prophet`, `pymc`, `arviz`, `scipy`, `sklearn`, `matplotlib`) in `try/except ImportError` blocks with a `HAS_X = True/False` flag pattern matching how `src/lib/analysis/__init__.py` does it. Do NOT change business logic."*

**Acceptance**: `python -c "from lib.model import factory, registry, service, persistence"` succeeds without any missing-import errors.

---

### MODEL-INT-C: Fix Runtime Bugs

- [ ] **C1**: `src/lib/model/deep/nn.py` line ~265 — fix the line-break bug: `loss = np.mean` / `((y - self.forward(X)) ** 2)` → `loss = np.mean((y - self.forward(X)) ** 2)`
- [ ] **C2**: `src/lib/model/service.py` — change `pl.training(...)` → `pl.Trainer(...)` (search for all occurrences)
- [ ] **C3**: `src/lib/model/deep/lstm.py` — change `pl.training(...)` → `pl.Trainer(...)`, fix `self.training.save_checkpoint()` → `self.trainer.save_checkpoint()`
- [ ] **C4**: `src/lib/model/deep/tft.py` — change `pl.training(...)` → `pl.Trainer(...)`
- [ ] **C5**: Add `super().__init__()` calls to all 7 subclasses that are missing them: `SimpleNN`, `EnhancedNN`, `ConcreteNN`, `LSTMModel`, `LogisticRegressionModel`, `HMMModel`, `ProphetModel`

**Agent prompt**: *"Fix runtime bugs in `src/lib/model/`: (1) In `deep/nn.py` around line 265, fix the broken `np.mean` call where a line break splits `loss = np.mean` from `((y - self.forward(X)) ** 2)` — join them into one line. (2) In `service.py`, `deep/lstm.py`, and `deep/tft.py`, replace all occurrences of `pl.training(` with `pl.Trainer(`. In `deep/lstm.py` also fix `self.training.save_checkpoint` → `self.trainer.save_checkpoint`. (3) Add `super().__init__()` calls to `SimpleNN.__init__`, `EnhancedNN.__init__`, `ConcreteNN.__init__`, `LSTMModel.__init__`, `LogisticRegressionModel.__init__`, `HMMModel.__init__`, `ProphetModel.__init__` — pass appropriate kwargs like `name=self.__class__.__name__` to `BaseModel.__init__`."*

**Acceptance**: `ruff check src/lib/model/ --select E,F` reports no syntax or undefined-name errors.

---

### MODEL-INT-D: Auto-Fix Whitespace, Deprecated Typing, Import Sorting

- [ ] **D1**: Run `ruff check src/lib/model/ --fix --unsafe-fixes` to auto-fix ~2,048 whitespace + style issues
- [ ] **D2**: Run `ruff format src/lib/model/` to normalize formatting
- [ ] **D3**: Manually review and fix remaining ~50 non-auto-fixable issues:
  - `B905` zip-without-explicit-strict (15) — add `strict=False` or `strict=True`
  - `SIM108` if-else-block-instead-of-if-exp (14) — use ternary where readable
  - `F841` unused-variable (6) — prefix with `_` or remove
  - `B006` mutable-argument-default (5) — change `def f(x=[])` → `def f(x=None)`
  - `B007` unused-loop-control-variable (3) — rename to `_`
  - `E722` bare-except (2) — change to `except Exception`
  - `E731` lambda-assignment (2) — convert lambda to `def`
  - `F811` redefined-while-unused (2) — remove duplicate imports in `statistical/arima.py`
  - `B904` raise-without-from-inside-except (2) — add `from e` or `from None`

**Agent prompt**: *"Run `ruff check src/lib/model/ --fix --unsafe-fixes` and then `ruff format src/lib/model/`. After auto-fixes, manually fix all remaining ruff errors. For `B905` (zip-without-strict), add `strict=False`. For `SIM108`, use ternary expressions where they improve readability (keep if/else for complex conditions). For `F841`, prefix unused variables with `_`. For `B006`, replace mutable defaults with `None` and set in function body. For `E722`, change bare `except:` to `except Exception:`. For `E731`, convert lambdas to named functions. For `F811`, remove the duplicate import. For `B904`, add `from e` to re-raises."*

**Acceptance**: `ruff check src/lib/model/ --no-fix` reports **0 errors**.

---

### MODEL-INT-E: Wire `__init__.py` Exports & Populate Empty Files

- [ ] **E1**: Populate `src/lib/model/__init__.py` with guarded re-exports of public API (matching the pattern in `src/lib/indicators/__init__.py`):
  - `BaseModel`, `Classifier`, `Regressor`, `Estimator` from `base/`
  - `ModelFactory` from `factory.py`
  - `ModelRegistry`, `model_registry`, `register_model` from `registry.py`
  - `ModelPersistence` from `persistence.py`
  - `ModelMetadata` from `utils/metadata.py`
  - All concrete model classes guarded by `try/except ImportError`
- [ ] **E2**: Populate empty `__init__.py` files in each sub-package (`base/`, `deep/`, `ensemble/`, `evaluation/`, `ml/`, `prediction/`, `statistical/`, `utils/`) with `__all__` lists
- [ ] **E3**: Populate empty stub files (`deep/cnn.py`, `deep/transformer.py`, `ml/catboost.py`, `ml/lightgbm.py`, `evaluation/cross_val.py`, `evaluation/metrics.py`) with placeholder classes that inherit from `BaseModel` and raise `NotImplementedError`
- [ ] **E4**: Add package-level docstring to `src/lib/model/__init__.py` describing the module hierarchy

**Agent prompt**: *"Populate all `__init__.py` files in `src/lib/model/` and its sub-packages with proper `__all__` exports and guarded imports following the exact same pattern as `src/lib/indicators/__init__.py` and `src/lib/analysis/__init__.py`. Every import should be wrapped in `try/except ImportError` that sets the name to `None`. Also populate empty stub files (`deep/cnn.py`, `deep/transformer.py`, `ml/catboost.py`, `ml/lightgbm.py`, `evaluation/cross_val.py`, `evaluation/metrics.py`) with skeleton classes that inherit from `BaseModel` and have `fit`/`predict`/`evaluate` methods that raise `NotImplementedError('Not yet implemented')`. Add a comprehensive docstring to the top-level `__init__.py` explaining the model package hierarchy."*

**Acceptance**: `python -c "from lib import model; print(model.__all__)"` succeeds and prints the exported names.

---

### MODEL-INT-F: Add Basic Tests

- [ ] **F1**: Create `src/tests/test_model_imports.py` — verify all model sub-packages import without error
- [ ] **F2**: Create `src/tests/test_model_registry.py` — verify `ModelRegistry`, `register_model`, `ModelFactory` work
- [ ] **F3**: Create `src/tests/test_model_base.py` — verify `BaseModel`, `Classifier`, `Regressor`, `Estimator` ABCs work
- [ ] **F4**: Create `src/tests/test_model_metadata.py` — verify `ModelMetadata` dataclass

**Agent prompt**: *"Create test files in `src/tests/` for the model library. `test_model_imports.py` should have one test per sub-package that does `from lib.model.X import Y` and asserts the import succeeded (or is None if optional deps missing). `test_model_registry.py` should test registering a stub model class, looking it up, creating instances via factory. `test_model_base.py` should verify the ABC hierarchy (BaseModel → Estimator → Classifier/Regressor). `test_model_metadata.py` should verify ModelMetadata creation and serialization. Follow existing test patterns in `src/tests/test_ruby_signal_engine.py`. Mark tests requiring optional deps with `@pytest.mark.skipif`."*

**Acceptance**: `pytest src/tests/test_model_imports.py src/tests/test_model_registry.py src/tests/test_model_base.py src/tests/test_model_metadata.py -v` — all pass.

---

### Task Execution Order (MODEL-INT)

```
MODEL-INT-A (fix syntax + import paths)     ← BLOCKING — nothing else works until this is done
    ↓
MODEL-INT-B (create shims for missing deps) ← BLOCKING — imports still fail without shims
    ↓
MODEL-INT-C (fix runtime bugs)              ← important but non-blocking for lint
    ↓
MODEL-INT-D (auto-fix whitespace + style)   ← bulk of the 2,713 errors, mostly automated
    ↓
MODEL-INT-E (wire __init__.py + stubs)      ← makes the package usable from other code
    ↓
MODEL-INT-F (basic tests)                   ← verification gate
```

**Estimated effort**: A+B = 1 agent session (~30 min). C = 1 agent session (~15 min). D = 1 agent session (~20 min, mostly automated). E = 1 agent session (~20 min). F = 1 agent session (~15 min). **Total: ~5 agent sessions.**

---

## 🔴 Phase PINE-INT — Pine Script Generator Integration & Lint Fixes

> **Source**: `src/lib/integrations/pine/` (3 Python files + `params.yaml` + 16 `.pine` modules + 1 generated output)
> **Current state**: 333 ruff errors, broken import paths (`from pine.generate`), `fks`/`ruby` key mismatch, Docker-specific hardcoded paths.
> **Goal**: Make all Python files pass `ruff check` and `mypy`, fix import paths to `lib.integrations.pine.*` namespace, resolve the `fks`/`ruby` naming inconsistency, guard NiceGUI deps.

### Audit Summary

| Category | Count | Description |
|---|---|---|
| **Broken imports** | 3 | `from pine.generate` → `from lib.integrations.pine.generate`, `from pine.app` → `from lib.integrations.pine.app`, broken `src.lib.core.lifecycle` import in `main.py` |
| **Key mismatch** | 1 | `generate.py` + `app.py` look for `fks` key everywhere; `params.yaml` defines `ruby` — all param lookups fall to hardcoded defaults |
| **Whitespace/style (auto-fixable)** | ~283 | `W293` blank-line-whitespace (262), trailing whitespace (4), missing newlines (3) |
| **Deprecated typing** | ~30 | `UP006`, `UP035`, `UP045` |
| **Unused imports** | 8 | `json`, `Path`, `Union`, `events` in `app.py`; `Union` in `generate.py`; `List`, `Union` in `main.py` |
| **Syntax errors** | 3 | Invalid syntax in `main.py` (likely from broken imports) |
| **Docker hardcoded paths** | 2 | `OUTPUT_DIR` defaults to `/app/outputs/pine`, cache to `/app/data/cache/pine` |

### PINE-INT-A: Fix Import Paths *(blocking — do first)*

- [ ] **A1**: `src/lib/integrations/pine/app.py` — change `from pine.generate import PineScriptGenerator` → `from lib.integrations.pine.generate import PineScriptGenerator` (or use relative: `from .generate import PineScriptGenerator`)
- [ ] **A2**: `src/lib/integrations/pine/main.py` — change `from pine.app import main as app_main` → `from lib.integrations.pine.app import main as app_main` (or relative: `from .app import main as app_main`)
- [ ] **A3**: `src/lib/integrations/pine/main.py` — fix broken `from src.lib.core.lifecycle import initialization, teardown, lifespan` — either import correctly from existing modules (`from lib.core.lifespan import ...`) or stub out the lifecycle imports behind `try/except` since `main.py`'s `PineService` class is not actively used by the project's service layer
- [ ] **A4**: `src/lib/integrations/pine/main.py` — fix `from src.lib.core.base import BaseService, ServiceRunner` — guard behind `try/except ImportError` with `BaseService = object` fallback since `core/base.py` itself has broken imports

**Agent prompt**: *"Fix import paths in `src/lib/integrations/pine/`. In `app.py`, change `from pine.generate import PineScriptGenerator` to `from .generate import PineScriptGenerator` (relative import). In `main.py`, change `from pine.app import main as app_main` to `from .app import main as app_main` and `from pine.app import app as ui_app` to `from .app import app as ui_app`. Wrap the `import_core_modules()` function's imports in `try/except ImportError` blocks — if `lib.core.base.BaseService` or lifecycle modules can't be imported, set them to `None` or `object` stubs. Do NOT change business logic."*

**Acceptance**: `python -c "import lib.integrations.pine"` succeeds; `ruff check src/lib/integrations/pine/ --select E999,F821` reports 0 errors.

---

### PINE-INT-B: Fix `fks`/`ruby` Key Mismatch

- [ ] **B1**: In `src/lib/integrations/pine/generate.py` — change the default indicator type from `'fks'` to `'ruby'` throughout:
  - `_get_file_orders()` fallback key
  - `_get_output_filenames()` fallback key
  - `_validate_params()` checks
  - `generate_full_script()` default type parameter
  - Any other hardcoded `'fks'` references
- [ ] **B2**: In `src/lib/integrations/pine/app.py` — change `get_fks_module_order()` → `get_ruby_module_order()` (or make it generic), update `params['file_orders']['fks']` → `params['file_orders']['ruby']`, same for `output_filenames`
- [ ] **B3**: Verify `params.yaml` `ruby:` keys now match the Python code expectations

**Agent prompt**: *"In `src/lib/integrations/pine/generate.py` and `src/lib/integrations/pine/app.py`, replace all hardcoded references to the indicator type `'fks'` with `'ruby'` to match what `params.yaml` defines. In `generate.py`: update `_get_file_orders()`, `_get_output_filenames()`, `_validate_params()`, and `generate_full_script()` defaults. In `app.py`: rename `get_fks_module_order()` to `get_ruby_module_order()` and update all `params['file_orders']['fks']` → `params['file_orders']['ruby']`. Verify that after changes, the generator reads the correct file order from `params.yaml` instead of falling back to hardcoded defaults."*

**Acceptance**: Running `python -c "from lib.integrations.pine.generate import PineScriptGenerator; g = PineScriptGenerator(); print(g._get_file_orders())"` returns the `ruby` file order from `params.yaml`.

---

### PINE-INT-C: Auto-Fix Whitespace, Deprecated Typing, Unused Imports

- [ ] **C1**: Run `ruff check src/lib/integrations/pine/ --fix --unsafe-fixes` to auto-fix ~283 whitespace + style issues
- [ ] **C2**: Run `ruff format src/lib/integrations/pine/` to normalize formatting
- [ ] **C3**: Remove unused imports: `json`, `Path`, `Union`, `events` from `app.py`; `Union` from `generate.py`; `List`, `Union` from `main.py`
- [ ] **C4**: Fix remaining non-auto-fixable issues:
  - `UP015` redundant-open-modes (8) — remove `'r'` from `open(f, 'r')`
  - `UP024` os-error-alias (4) — change `IOError` → `OSError`
  - `F841` unused-variable (1) — prefix with `_`
  - `SIM108` if-else-block (1) — use ternary
  - `SIM117` multiple-with-statements (1) — merge `with` blocks

**Agent prompt**: *"Run `ruff check src/lib/integrations/pine/ --fix --unsafe-fixes` and `ruff format src/lib/integrations/pine/`. Then manually fix all remaining ruff errors: remove unused imports (`json`, `Path`, `Union`, `events` from `app.py`; `Union` from `generate.py`; `List`, `Union` from `main.py`). Fix `UP015` by removing redundant `'r'` mode from `open()` calls. Fix `UP024` by changing `IOError` to `OSError`. Fix any remaining `F841`, `SIM108`, `SIM117` issues."*

**Acceptance**: `ruff check src/lib/integrations/pine/ --no-fix` reports **0 errors**.

---

### PINE-INT-D: Guard NiceGUI & Fix Docker Paths

- [ ] **D1**: In `app.py`, guard `from nicegui import ...` with `try/except ImportError` and set `HAS_NICEGUI = False` so the module can be imported even when NiceGUI is not installed
- [ ] **D2**: Guard `nicegui_app.native.window_args` access with `hasattr()` check
- [ ] **D3**: Change default `OUTPUT_DIR` from `/app/outputs/pine` to a relative path: `os.path.join(os.path.dirname(__file__), 'pine_output')` — still overridable via `PINE_OUTPUT_DIR` env var
- [ ] **D4**: Change default cache directory similarly to use a temp directory fallback
- [ ] **D5**: Fix the health check endpoint — change from `@ui.page('/health')` to `@nicegui_app.get('/health')` or a proper Starlette route

**Agent prompt**: *"In `src/lib/integrations/pine/app.py`: (1) Wrap `from nicegui import ui, app as nicegui_app, Client` and `from starlette.requests import Request` in `try/except ImportError` with `HAS_NICEGUI = False` flag — when NiceGUI is missing, make the `main()` function print a warning and return. (2) Guard `nicegui_app.native.window_args['host']` with `hasattr(nicegui_app, 'native')` check. (3) Change default `OUTPUT_DIR` to `os.path.join(os.path.dirname(__file__), 'pine_output')`. (4) Change default cache dir to `os.path.join(tempfile.gettempdir(), 'pine_cache')`. (5) Change `@ui.page('/health')` to use a proper Starlette JSON response route."*

**Acceptance**: `python -c "import lib.integrations.pine.app"` succeeds even without NiceGUI installed.

---

### PINE-INT-E: Fix Pine Script Module Issues *(low priority — Pine Script, not Python linting)*

- [ ] **E1**: Fix module order in `params.yaml` — move `ml_atr.pine` before `core_calculations.pine` (or refactor `core_calculations.pine` to not depend on `adaptive_atr` at top-level scope)
- [ ] **E2**: Remove duplicate `//@version=6` directives from `alerts.pine` and `main.pine` (keep only in `header.pine`)
- [ ] **E3**: Remove duplicate `indicator()` declaration from `main.pine` (keep only in `header.pine`)
- [ ] **E4**: Remove duplicate `calc_norm_roc()` function from `patterns.pine` (keep only in `core_calculations.pine`)
- [ ] **E5**: Remove duplicate alert message generator functions from `alert_conditions.pine` (keep only in `alerts.pine`)
- [ ] **E6**: Populate empty `visualization.pine` with a comment placeholder or actual visualization code

**Agent prompt**: *"Fix Pine Script module issues in `src/lib/integrations/pine/`: (1) In `params.yaml`, swap the order of `ml_atr.pine` and `core_calculations.pine` in the `ruby` file_orders list so that `ml_atr.pine` comes before `core_calculations.pine` (since core_calculations depends on `adaptive_atr` from ml_atr). (2) In `modules/alerts.pine`, remove the stray `//@version=6` directive. (3) In `modules/main.pine`, remove the duplicate `//@version=6` and `indicator()` declaration. (4) In `modules/patterns.pine`, remove the duplicate `calc_norm_roc()` function definition. (5) In `modules/alert_conditions.pine`, remove the duplicate alert message generator functions that are already defined in `alerts.pine`. (6) Add a comment block to empty `modules/visualization.pine`."*

**Acceptance**: Running `PineScriptGenerator().generate_full_script()` produces a valid concatenated `.pine` file with no duplicate declarations.

---

### PINE-INT-F: Add Basic Tests

- [ ] **F1**: Create `src/tests/test_pine_generator.py` — verify `PineScriptGenerator` instantiation, params loading, file order resolution
- [ ] **F2**: Test that `generate_full_script()` produces output containing expected sections (header, inputs, core_calculations, etc.)
- [ ] **F3**: Test that `params.yaml` loads and the `ruby` key resolves correctly
- [ ] **F4**: Test that all 16 `.pine` module files exist and are non-empty (except `visualization.pine`)

**Agent prompt**: *"Create `src/tests/test_pine_generator.py` with tests for the Pine Script generator. Test that `PineScriptGenerator` can be instantiated, that `params.yaml` loads correctly with the `ruby` key, that `_get_file_orders()` returns the ruby file order, that all 16 `.pine` module files in `src/lib/integrations/pine/modules/` exist, and that `generate_full_script()` produces output containing key markers like `//@version=6`, `indicator(`, and sections from each module. Mark NiceGUI-dependent tests with `@pytest.mark.skipif`. Follow test patterns from existing `src/tests/` files."*

**Acceptance**: `pytest src/tests/test_pine_generator.py -v` — all pass.

---

### Task Execution Order (PINE-INT)

```
PINE-INT-A (fix import paths)              ← BLOCKING — module can't be imported without this
    ↓
PINE-INT-B (fix fks/ruby key mismatch)     ← functional correctness
    ↓
PINE-INT-C (auto-fix whitespace + style)   ← bulk of 333 errors, mostly automated
    ↓
PINE-INT-D (guard NiceGUI + fix paths)     ← portability & graceful degradation
    ↓
PINE-INT-E (fix Pine Script modules)       ← low priority — Pine, not Python
    ↓
PINE-INT-F (basic tests)                   ← verification gate
```

**Estimated effort**: A = 1 agent session (~15 min). B = 1 agent session (~10 min). C = 1 agent session (~10 min, mostly automated). D = 1 agent session (~15 min). E = 1 agent session (~15 min). F = 1 agent session (~10 min). **Total: ~6 agent sessions.**

---

## 📊 Lint Status Dashboard

> After MODEL-INT + PINE-INT are complete, the project should be at **0 ruff errors** across all of `src/lib/`.

| Directory | Before | After (target) |
|---|---|---|
| `src/lib/model/` | 2,713 errors | 0 |
| `src/lib/integrations/pine/` | 333 errors | 0 |
| `src/lib/` (rest of project) | 833 errors | 833 (unchanged — separate cleanup) |
| **Total** | **3,879** | **833** |
