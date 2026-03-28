# Futures Multi-Asset Trader — Master TODO
## Repository: `nuniesmith/futures`

> Refactored from single-asset SOL scalper → multi-asset futures bot
> Assets: **50+ available** in asset registry, 10 enabled by default
> Exchange: KuCoin Perpetual Futures (USDTM)
> Modes: **sim** (paper trading on live data) | **live** (real orders)
> Timezone: America/New_York (EDT)
> AI Reports: Grok 4.1 via xAI API (daily / weekly / monthly performance reports)

---

## Table of Contents

1. [Architecture Decisions](#architecture-decisions)
2. [Phase 0 — Repo Rename & Cleanup](#phase-0--repo-rename--cleanup) ✅
3. [Phase 1 — Multi-Asset Config & Data Layer](#phase-1--multi-asset-config--data-layer) ✅
4. [Phase 2 — Redis Integration](#phase-2--redis-integration) ✅
5. [Phase 3 — Supervisor / Worker Architecture](#phase-3--supervisor--worker-architecture) ✅
6. [Phase 4 — Extract Trading Logic into Worker](#phase-4--extract-trading-logic-into-worker) ✅
7. [Phase 5 — Grok AI Reports](#phase-5--grok-ai-reports) ✅
8. [Phase 6 — Docker & Build Fixes](#phase-6--docker--build-fixes) ✅
9. [Phase 7 — Optuna as Background Service](#phase-7--optuna-as-background-service)
10. [Phase 8 — Discord & Monitoring](#phase-8--discord--monitoring)
11. [Phase 9 — Tests & Linting](#phase-9--tests--linting)
12. [Phase 10 — Data Retention & Historical Backfill](#phase-10--data-retention--historical-backfill)
13. [Phase 11 — Hardening & Performance](#phase-11--hardening--performance)
14. [Phase 12 — Web UI (HTMX Dashboard)](#phase-12--web-ui-htmx-dashboard)
15. [Completed Items (Legacy)](#completed-items-legacy)
15. [Current File Layout](#current-file-layout)
16. [Stale / Legacy Files to Clean Up](#stale--legacy-files-to-clean-up)
17. [Quick Reference: Asset Symbols](#quick-reference-asset-symbols)

---

## Architecture Decisions

### Margin Mode — Use Isolated ✅

**Isolated margin** is the correct choice for multi-asset trading. Each position
gets its own margin allocation, so a liquidation on FARTCOIN cannot cascade into
your BTC or ETH positions. This is critical when mixing blue-chip crypto (BTC,
ETH) with high-volatility assets (AVAX, FARTCOIN).

Cross margin shares your entire account balance across all open positions. A
single bad trade on a volatile meme coin could wipe your entire account. With
isolated margin, the worst case per position is losing only the margin allocated
to that specific trade.

**Per-asset margin allocation (starting point for $30 capital):**

| Asset    | Symbol          | Leverage | Max Lev | Margin % | Risk Character          |
|----------|-----------------|----------|---------|----------|-------------------------|
| BTC      | XBTUSDTM        | 20x      | 125x    | 15%      | Low vol, deepest books  |
| ETH      | ETHUSDTM        | 20x      | 100x    | 12%      | Medium vol, liquid      |
| SOL      | SOLUSDTM        | 20x      | 75x     | 10%      | High vol, fast moves    |
| DOGE     | DOGEUSDTM       | 20x      | 75x     | 10%      | Meme, very liquid       |
| SUI      | SUIUSDTM        | 20x      | 75x     | 10%      | Newer L1, good momentum |
| PEPE     | PEPEUSDTM       | 20x      | 75x     | 8%       | Meme, decent volume     |
| AVAX     | AVAXUSDTM       | 20x      | 75x     | 8%       | Solid L1                |
| WIF      | WIFUSDTM        | 20x      | 75x     | 8%       | SOL meme coin           |
| FARTCOIN | FARTCOINUSDTM   | 20x      | 50x     | 7%       | Extreme vol, thin books |
| KCS      | KCSUSDTM        | 4x       | 8x      | 12%      | KuCoin native, low lev  |

**Worst-case open margin:** 10 assets x 3 orders x $0.50 = $15.00 minimum.
Remaining $15 is buffer for P&L and margin fluctuations.

### Position Mode — Use One-Way ✅

**One-Way mode** remains correct. The bot uses `reduceOnly=True` on every close
order, which only works in One-Way mode. Hedge Mode requires `positionIdx`
(1=long, 2=short) on every order - a significant refactor for minimal benefit.

**Revisit hedge mode when:** Capital exceeds $500+ and you want to run a
market-neutral strategy with simultaneous long/short hedging on correlated pairs.

### Simulation Mode — Yes ✅

Before going live, run in **sim mode** to validate end-to-end:

- Live WebSocket feeds (real prices, real orderbook depth for all 10 assets)
- Paper trades executed at market price with simulated slippage
- PnL tracked in Redis - identical to live mode, just no real orders
- Signals recorded to Redis for analysis and report generation
- Grok 4.1 generates daily/weekly/monthly reports from Redis trade data
- Switch to live by changing `TRADING_MODE=live` in `.env` - zero code change

### Supervisor Architecture — Yes ✅

A supervisor `main()` spawns one `AssetWorker` async task per enabled asset.
Each worker is self-contained with its own state (tick deque, orderbook,
position stack, risk state, wave state, Optuna params). The supervisor:

- Spawns/restarts workers on crash
- Checks Redis heartbeats for stale workers
- Logs aggregate portfolio summaries every 15 minutes
- Handles SIGTERM/SIGINT graceful shutdown

### Redis — Yes, for Speed & Data Retention ✅

Redis serves multiple purposes:

1. **Signal tracking** - Every signal is recorded (sim mode: this IS the trade log)
2. **PnL tracking** - Per-asset and aggregate P&L in sorted sets
3. **Order history** - Last 1000 trades per asset
4. **Worker heartbeats** - Health monitoring with 120s TTL
5. **Report storage** - Generated Grok reports stored and retrievable
6. **Candle cache** - Workers can restart without cold-start delay

Automatic in-memory fallback when Redis is unavailable (dev/testing).

### Grok AI Reports — Yes ✅

Grok 4.1 via xAI API generates intelligent trading performance reports:

- **Daily** - End-of-day summary with per-asset breakdown, best/worst trade, highlights
- **Weekly** - 7-day rollup with trends, rankings, strategy effectiveness
- **Monthly** - 30-day comprehensive review with recommendations
- **Highlights** - Notable events, streaks, risk alerts

Reports are generated on schedule (daily 23:55 ET, weekly Sunday, monthly last day)
and stored in Redis for retrieval. Can also be generated on-demand via CLI.

---

## Phase 0 — Repo Rename & Cleanup ✅

> **Status: DONE**

### Completed

- [x] **0.1** Rename GitHub repo from `nuniesmith/sol` -> `nuniesmith/futures`
- [x] **0.2** Rename config file: `config/sol.yaml` -> `config/futures.yaml`
- [x] **0.3** Update `config/futures.yaml` with full multi-asset config
- [x] **0.4** Update `pyproject.toml` (name, version, dependencies, tooling)
- [x] **0.5** Update `run.sh` (lint, sim, redis-logs commands, banner, TZ)
- [x] **0.6** Update `docker/futures/entrypoint.sh` for multi-asset
- [x] **0.7** Update `docker-compose.yml` (redis as peer service, healthcheck, volumes)
- [x] **0.8** Update `README.md`
- [x] **0.9** Create `.dockerignore`
- [x] **0.10** Update `.gitignore`
- [x] **0.11** Update `.env.example` (REDIS_PASSWORD, TRADING_MODE, TZ, XAI_API_KEY)
- [x] **0.12** Write `scripts/setup-pi.sh` (Raspberry Pi 4 setup)

---

## Phase 1 — Multi-Asset Config & Data Layer ✅

> **Status: DONE** - `src/services/config_loader.py` created with full typed config system.

### Completed

- [x] **1.1** Create `src/services/__init__.py`
- [x] **1.2** Create `src/services/config_loader.py`:
  - `FuturesConfig` top-level dataclass with typed fields for every YAML section
  - `AssetConfig` per-asset dataclass (key, symbol, base, leverage, margin_pct, etc.)
  - `load_config()` - loads `config/futures.yaml`, expands `${ENV_VAR:-default}`, deep-merges
  - `get_config()` - thread-safe singleton with double-checked locking
  - `config.enabled_assets` property - filters to only enabled assets
  - `config.is_sim` / `config.is_live` convenience properties
  - 20+ typed sub-config dataclasses (ExchangeConfig, CapitalConfig, RiskConfig, etc.)
  - `_safe_int/_safe_float/_safe_bool/_safe_str` helpers for robust parsing
  - Graceful fallback to all defaults on missing file or parse error

### Remaining

- [ ] **1.3** Write tests for config loader (YAML loading, env expansion, deep merge, missing file fallback)
- [ ] **1.4** Write tests for asset parsing (defaults, per-asset overrides, disabled assets filtered out)

---

## Phase 2 — Redis Integration ✅

> **Status: DONE** - `src/services/redis_store.py` created with full async store + in-memory fallback.

### Completed

- [x] **2.1** Create `src/services/redis_store.py`:
  - `RedisStore` class with `async connect()` / `close()` / `connected` property
  - Automatic in-memory fallback when Redis package missing or connection fails
  - **Signals**: `record_signal()` / `get_signals()` - LPUSH + LTRIM to 1000
  - **PnL**: `record_pnl()` / `get_daily_pnl()` / `get_pnl_history()` - sorted sets by timestamp
  - **Aggregate stats**: `get_aggregate_stats()` - total_pnl, win_rate, best/worst asset, daily breakdown
  - **Orders**: `record_order()` / `get_orders()` / `get_all_orders()`
  - **Worker state**: `save_worker_state()` / `load_worker_state()` - JSON in hash fields
  - **Heartbeats**: `heartbeat()` / `get_heartbeats()` - string with 120s TTL
  - **Reports**: `store_report()` / `get_latest_report()` - with latest pointer
  - **Candle cache**: `cache_candles()` / `get_cached_candles()`
  - **Report data**: `get_report_data()` - gathers everything for Grok report generation
  - Daily PnL resets at midnight Eastern Time
  - All serialization via `json.dumps(default=str)` / `json.loads`

### Remaining

- [ ] **2.2** Write tests for `redis_store.py` with `fakeredis`:
  - PnL aggregation, daily reset at midnight ET
  - Order history LTRIM behavior
  - Heartbeat TTL expiry
  - Candle cache round-trip
  - In-memory fallback behavior
  - `get_report_data()` comprehensive output

---

## Phase 3 — Supervisor / Worker Architecture ✅

> **Status: DONE** - Integrated directly into the refactored `src/main.py`.

### Completed

- [x] **3.1** Supervisor `main()` function in `src/main.py`:
  - Loads config via `load_config()`
  - Connects to Redis via `RedisStore`
  - Creates shared `ccxt.kucoinfutures` exchange instance
  - Spawns one `AssetWorker` task per enabled asset (10 assets)
  - Spawns `ReportGenerator.run_scheduled()` background task
  - Monitor loop checks worker health via Redis heartbeats
  - Detects and restarts crashed workers automatically
  - Logs aggregate portfolio summary every 15 minutes
  - Graceful shutdown with `asyncio.Event` + `SIGTERM`/`SIGINT` handlers
  - 10-second timeout for task cancellation on shutdown

### Remaining

- [ ] **3.2** Extract supervisor into dedicated `src/supervisor/manager.py` module (optional refactor)
- [ ] **3.3** Write tests for supervisor (mock worker coroutine, crash detection, shutdown)

---

## Phase 4 — Extract Trading Logic into Worker ✅

> **Status: DONE** - `AssetWorker` class in `src/main.py` with full per-asset trading logic.

### Completed

- [x] **4.1** `AssetWorker` class with per-worker state:
  - Tick deque, orderbook, position stack, risk state, wave state, Optuna params
  - Per-asset Optuna interval (fast/slow/default based on asset volatility)
  - Stack helpers: `_stack_avg_price()`, `_stack_total_size()`, `_stack_clear()`
- [x] **4.2** Risk management (per-worker):
  - `_risk_check_daily_reset()` - midnight ET reset
  - `_risk_record_trade()` - updates daily PnL, trade count, consecutive losses
  - `_risk_can_trade()` - daily loss limit, consecutive loss breaker, cooldown, trade cap
- [x] **4.3** Order execution with sim/live split:
  - `_place_order()` - simulated fill at last price + slippage (sim) or `create_order()` (live)
  - `_close_position()` - calculates P&L, records to Redis, logs with full Ruby context
  - `_calc_size()` - position sizing based on capital * margin_pct * leverage, min notional enforcement
- [x] **4.4** WebSocket feeds with exponential backoff:
  - `_handle_trades()` - ccxt `watch_trades()` with reconnect backoff
  - `_handle_orderbook()` - ccxt `watch_order_book()` with reconnect backoff
  - Configurable `reconnect_delay_sec`, `reconnect_max_delay_sec`, `reconnect_backoff_factor`
- [x] **4.5** All indicator math preserved from original:
  - `build_candles()` - 5-second OHLCV from tick stream
  - `_compute_atr()` - Average True Range (Wilder EMA)
  - `compute_ao()` - Awesome Oscillator
  - `compute_vol_pct()` - Volatility percentile
  - `compute_regime()` - Market regime (TRENDING/VOLATILE/RANGING/NEUTRAL)
  - `update_wave_state()` - Wave analysis (Ruby Pine Script v6.3 section 4)
  - `compute_quality()` - Quality score 0-100
  - `adaptive_tp()` - Adaptive take profit by regime + vol
  - `adaptive_add_threshold()` - DCA pullback threshold
  - `wave_gate_ok()` / `regime_stack_ok()` - Stacking gates
- [x] **4.6** Optuna optimization in thread pool:
  - `_make_objective()` / `_run_optuna_sync()` - thread-safe objective function
  - `_maybe_optimize()` - runs in `ThreadPoolExecutor` (2 workers) via `run_in_executor()`
  - Per-asset Optuna intervals (fast 10min for meme coins, slow 30min for BTC/ETH)
- [x] **4.7** Signal routing to Redis (sim mode):
  - `record_signal()` on every generated signal
  - `record_pnl()` on every simulated close
  - `record_order()` on every simulated fill
  - Redis heartbeat with full worker status every iteration
- [x] **4.8** `main.py` is now a clean entry point:
  - `setup_logging()` -> `load_config()` -> `RedisStore` -> `ccxt` -> spawn workers
  - Old globals replaced with instance variables on `AssetWorker`
  - `WaveState` dataclass preserved at module level

### Remaining

- [ ] **4.9** Extract worker into `src/worker/trader.py`, signals into `src/worker/signals.py`,
  risk into `src/worker/risk.py`, orders into `src/worker/orders.py` (optional refactor -
  everything works in the single file, but splitting improves testability)
- [ ] **4.10** Move `WaveState` dataclass to `src/analysis/wave_analysis.py`
- [ ] **4.11** Create `src/services/discord_notify.py` - extract Discord logic from main.py

---

## Phase 5 — Grok AI Reports ✅

> **Status: DONE** - `src/services/report_generator.py` created with full Grok 4.1 integration.

### Completed

- [x] **5.1** Create `src/services/report_generator.py`:
  - `ReportGenerator` class using xAI API via `openai` SDK
  - Model: `grok-4-1`, temp 0.3, base URL `https://api.x.ai/v1`
  - `_call_grok()` runs sync SDK call via `asyncio.to_thread()` (non-blocking)
  - Lazy client initialization on first call
- [x] **5.2** Four report types:
  - `generate_daily_report()` - 3000 tokens - PnL summary, per-asset table, best/worst trade, signal quality, market conditions, highlights, tomorrow's outlook
  - `generate_weekly_report()` - 4000 tokens - week-over-week, asset ranking, daily breakdown, strategy trends, risk metrics, optimization suggestions
  - `generate_monthly_report()` - 5000 tokens - P&L trajectory, asset allocation, weekly buckets, strategy drift, risk analysis, capital growth, next-month recommendations
  - `generate_highlights()` - 2000 tokens - big wins/losses, streak detection, risk alerts, pattern changes
- [x] **5.3** `_detect_streaks()` - walks PnL history to find max win/loss streaks, current streak, alerts for extended losing (5+) and hot streaks (7+)
- [x] **5.4** `_read_recent_logs()` - scans log directory for `*.log` files, filters by timestamp, truncates to 6000 chars for prompt context
- [x] **5.5** `run_scheduled()` - async background loop:
  - Daily + highlights at 23:55+ ET
  - Weekly on Sundays
  - Monthly on last day of month
  - Checks every 30 seconds, safe for continuous background operation
- [x] **5.6** Reports stored in Redis via `store.store_report()` with latest pointer
- [x] **5.7** CLI entry point: `python -m src.services.report_generator [daily|weekly|monthly|highlights]`
- [x] **5.8** XAI_API_KEY loaded from environment (added to `.env`)

### Remaining

- [ ] **5.9** Test report generation with live Redis data after running sim for a while
- [ ] **5.10** Add Discord notification when reports are generated (post to channel)
- [ ] **5.11** Add web endpoint to serve latest reports (if dashboard is added later)

---

## Phase 6 — Docker & Build Fixes ✅

> **Status: DONE** - All Docker infrastructure fixed and buildable.

### Completed

- [x] **6.1** Create `requirements.txt` with all dependencies:
  - ccxt, optuna, pandas, aiohttp, python-dotenv, pyyaml, redis, openai
- [x] **6.2** Fix `docker/futures/Dockerfile`:
  - Fixed COPY path: `COPY docker/futures/entrypoint.sh /app/entrypoint.sh`
  - Added `COPY config/ ./config/` for default config baked into image
  - Python 3.12-slim-bookworm base (ARM64 compatible)
  - Multi-stage build with isolated venv
- [x] **6.3** Fix `docker/redis/Dockerfile`:
  - Fixed COPY path: `COPY docker/redis/entrypoint.sh /build/entrypoint.sh`
  - (was `infrastructure/docker/services/redis/entrypoint.sh` - old path)
  - Redis 7.4-alpine base, THP disable, healthcheck
- [x] **6.4** Fix `docker/futures/entrypoint.sh`:
  - Updated banner to "Multi-Asset Ruby Wave Edition"
  - Removed hardcoded SOLUSDTM reference
  - Shows TRADING_MODE (sim/live) from env
  - Removed hard exit on missing KuCoin keys (not needed in sim mode)
- [x] **6.5** `docker-compose.yml` validated:
  - Redis as peer service (not nested)
  - `depends_on: redis: condition: service_healthy`
  - Volume mounts for config and Redis data
  - TZ=America/New_York on both services
  - Redis 512MB memory limit, LRU eviction

### Remaining

- [ ] **6.6** Full build and run test: `docker compose build --no-cache && docker compose up -d`
- [ ] **6.7** Test on Raspberry Pi 4 (ARM64) - monitor memory/CPU usage
- [ ] **6.8** Verify `REDIS_PASSWORD` flows through correctly to healthcheck

---

## Phase 7 — Optuna as Background Service

> Optuna is already running per-worker in `_maybe_optimize()` via thread pool.
> This phase is about refinements.

### Tasks

- [ ] **7.1** Improve objective function:
  - Current one sums `signal.shift(1) * returns` - add fee deduction
  - Add KuCoin fee model: maker 0.02%, taker 0.06% (we use market orders = taker)
  - Add slippage model: 0.01% per trade
  - Return Sharpe-like ratio instead of raw PnL sum
  - Per-asset fee tiers if different
- [ ] **7.2** Store Optuna results in Redis:
  - Key: `futures:optuna:{asset}:latest` -> JSON of best params
  - Key: `futures:optuna:{asset}:history` -> sorted set of past results
  - Workers load latest params from Redis on restart (warm start)
- [ ] **7.3** Optuna log output format (keep concise):
  ```
  [SOL] Optuna complete (70 trials, 12.3s) -> fast=7 slow=24 sl=0.003 qual=55
  ```
- [ ] **7.4** Write tests for optimizer objective function with synthetic data

---

## Phase 8 — Discord & Monitoring

> Clean up Discord notifications for multi-asset. Add daily summary.

### Tasks

- [ ] **8.1** Create `src/services/discord_notify.py`:
  - `DiscordNotifier` class with rate limiting (25/min)
  - `[SIM]` prefix in simulation mode
  - Asset name in embed footer
  - Methods: `trade_opened()`, `trade_closed()`, `daily_summary()`, `risk_alert()`, `optuna_update()`, `supervisor_event()`
- [ ] **8.2** Add daily summary notification (midnight ET):
  ```
  | Asset    | Trades | Win% | PnL       |
  |----------|--------|------|-----------|
  | BTC      | 12     | 67%  | +$0.18    |
  | ...      | ...    | ...  | ...       |
  | TOTAL    | 44     | 61%  | +$0.33    |
  ```
- [ ] **8.3** Post Grok report summaries to Discord on generation
- [ ] **8.4** Rate-limit via async queue (batch if burst detected)
- [ ] **8.5** Risk alert notifications (per-asset and aggregate)

---

## Phase 9 — Tests & Linting

> Ensure all new code passes ruff, mypy, and pytest.

### Tasks

- [ ] **9.1** Fix existing tests for new import paths (main.py refactored)
- [ ] **9.2** Add tests for each new module:
  - `test_config_loader.py` - YAML loading, env expansion, defaults
  - `test_redis_store.py` - all Redis operations with fakeredis
  - `test_report_generator.py` - prompt building, streak detection (mock Grok API)
  - `test_asset_worker.py` - signal generation, risk gates, position management
- [ ] **9.3** Run ruff on all new code: `ruff check src/ --fix && ruff format src/`
- [ ] **9.4** Run mypy on all new code: `mypy src/ --ignore-missing-imports`
- [ ] **9.5** Fix the EN DASH warnings in main.py docstrings (use HYPHEN-MINUS)
- [ ] **9.6** Add CI-style check to `run.sh lint`

---

## Phase 10 — Data Retention & Historical Backfill

> Build up candle history in Redis for better Optuna optimization.

### Tasks

- [ ] **10.1** Add candle aggregation to worker (1m and 5m from 5s data)
- [ ] **10.2** On worker startup, load cached candles from Redis (no cold start)
- [ ] **10.3** REST API backfill on first run: `exchange.fetch_ohlcv(symbol, '1m', since=...)`
- [ ] **10.4** Candle cleanup job (daily at 00:05 UTC):
  - 5s candles: 1 week
  - 1m candles: 1 month
  - 5m candles: 3 months
- [ ] **10.5** Estimate Redis memory usage on Pi (~50 MB for all assets)

---

## Phase 11 — Hardening & Performance

> Production hardening for 24/7 Pi deployment.

### Tasks

- [ ] **11.1** Exchange session recycling (every 6 hours)
- [ ] **11.2** Memory monitoring (alert if > 400 MB)
- [ ] **11.3** Per-asset circuit breaker (disable after 5 WS failures in 10 min)
- [ ] **11.4** Handle KuCoin maintenance windows (check `exchange.fetch_status()`)
- [ ] **11.5** Trailing stop-loss option (configurable per-asset)
- [ ] **11.6** Position recovery on restart (check `exchange.fetch_positions()`)
- [ ] **11.7** Aggregate portfolio risk check (close worst position if unrealized > -3%)
- [ ] **11.8** Verify all asset symbols exist on KuCoin:
  `exchange.load_markets()` and filter for USDTM contracts
  (Now handled by `src/services/asset_registry.py` — all symbols verified against KuCoin API)

---

## Phase 12 — Web UI (HTMX Dashboard)

> Lightweight web dashboard for monitoring and managing the futures bot.
> Stack: **Python FastAPI + HTMX + Jinja2 templates**
> No heavy JS frameworks — HTMX handles dynamic updates.
> Asset registry: `src/services/asset_registry.py` (50+ contracts across crypto, metals, commodities, stocks)

### Tasks

- [x] **12.1** Create FastAPI app in `src/web/app.py`:
  - Mount Jinja2 templates in `src/web/templates/`
  - Mount static files (CSS, minimal JS) in `src/web/static/`
  - Auth middleware reused from `src/web/auth.py` (bcrypt + HMAC cookie, fixed imports)
  - Health check endpoint: `GET /api/health` → `{"status":"ok","redis":bool}`

- [x] **12.2** Dashboard page (`GET /`):
  - Per-worker status cards (asset, regime badge, quality bar, stack dir/count, daily PnL, last seen)
  - Stats bar: today PnL, win rate, total trades, best/worst asset
  - HTMX polling every 5s via `hx-get="/partials/workers"` + `hx-get="/partials/stats"`
  - Data sourced from `futures:heartbeat:*` via `RedisStore.get_heartbeats()`

- [x] **12.3** Asset management page (`GET /assets`):
  - All 62 contracts from `asset_registry.py` grouped by category with sticky tab nav
  - Toggle enable/disable per asset — `POST /assets/{key}/toggle` writes to `futures:ui:disabled_assets` Redis set
  - HTMX `outerHTML` swap on the toggle button (no full page reload)
  - Shows: symbol, base, KuCoin symbol, max leverage, tick size, multiplier, description, status
  - Disabled rows visually dimmed; toggles persist in Redis across dashboard restarts
  - Note: bot re-reads state on next restart

- [x] **12.4** Signals & trades page (`GET /signals`):
  - Recent signals table from `futures:signals:{asset}` (last 100 across all workers)
  - Filter by asset via dropdown — full-page HTMX swap with `?asset=` param
  - HTMX auto-refresh every 10s via `hx-get="/partials/signals"`
  - Side badges: LONG (green) / SHORT (red) / CLOSE (dim), quality coloured by threshold

- [x] **12.5** Reports page (`GET /reports`):
  - Day / Week / Month period tabs with HTMX full-page swap + `hx-push-url`
  - Report text rendered from `futures:reports:{type}:latest` via `get_latest_report()`
  - Historical date picker dropdown — populated via `list_report_dates()` scanning `futures:reports:{type}:*`
  - Specific date fetch via `get_report_by_date()` with `?date=` param

- [x] **12.6** PnL summary page (`GET /pnl`):
  - Per-asset PnL from Redis sorted sets via `get_aggregate_stats()` + `get_pnl_history()`
  - 1d / 7d / 30d period selector links
  - Summary cards: total PnL, win rate, total trades, best/worst asset
  - Daily breakdown table + per-asset table (sorted by PnL desc)
  - Inline SVG bar chart for daily PnL (green = positive, red = negative bars)

- [x] **12.7** Docker integration:
  - Web service added to `docker-compose.yml` (no `platform` pin — builds native on desktop x86_64 and Pi arm64)
  - Port `127.0.0.1:8080:8080` exposed (configurable via `WEB_PORT` env)
  - Shares Redis via `REDIS_URL=redis://redis:6379/0`, depends on `redis` healthcheck
  - `docker/web/Dockerfile` — Python 3.12 slim, multi-stage builder, uvicorn entrypoint

- [x] **12.8** Dependencies added to `requirements.txt`:
  - `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `jinja2>=3.1`, `python-multipart>=0.0.9`, `bcrypt>=4.1`

- [x] **12.9** Tailscale HTTPS integration (added beyond original spec):
  - `./run.sh tailscale-serve` — runs `tailscale serve --bg --https=443 http://127.0.0.1:8080`
  - `./run.sh tailscale-stop` — removes serve config
  - `./run.sh status` — shows green `●` when Tailscale is serving `:8080`, yellow `⚠` if mismatched port
  - Dashboard live at `https://desktop.tailfef10.ts.net` (tailnet only, accessible from `rasp`)

- [x] **12.10** `run.sh` new commands:
  - `web` — uvicorn local dev with `--reload` (auto-restarts on template/code changes)
  - `web-up` / `web-down` / `web-logs` — Docker service management
  - `web-hash-password` — interactive bcrypt hash generator for `WEB_PASSWORD_HASH`

> **All routes verified 200 OK** (Redis in-memory fallback mode, no bot running):
> `/ /assets /signals /reports /pnl /partials/workers /partials/stats`
> `/partials/signals /partials/report /partials/pnl /api/health`
> `POST /assets/{key}/toggle`

---

> **To test end-to-end:** start Redis + bot in sim mode, then `./run.sh web`.
> The dashboard will show live worker cards, signals, and PnL as soon as
> the bot connects and starts generating heartbeats.

---

## Completed Items (Legacy)

> Items completed during the initial SOL-only build.

### Position Mode — One-Way ✅
- Decided on One-Way mode (required for `reduceOnly` closes)
- Hedge Mode deferred until capital > $500

### Bugs Fixed ✅
1. `handle_ws` split into `handle_trades` + `handle_orderbook` (two independent async tasks)
2. `optimize_params` asyncio fix - removed deprecated `get_event_loop()` pattern
3. Missing `src/__init__.py` added
4. Docker build fixed - Poetry -> pip + requirements.txt
5. Dockerfile entrypoint COPY path fixed
6. Orderbook warm-up guard added

### Ruby Wave Analysis Ported ✅
- `update_wave_state()` - wave_ratio, cur_ratio, wr_pct, mom_pct
- `wave_gate_ok()` - gates position adds when wave energy flips
- `regime_stack_ok()` - blocks counter-trend adds
- `compute_quality()` - 0-100 quality score gate for first entries
- `adaptive_tp()` - TP scales with vol_pct and regime
- `adaptive_add_threshold()` - DCA threshold scales with volatility
- `compute_ao()` - Awesome Oscillator confirmation
- `compute_regime()` - SMA200 slope + vol ratio regime detection
- `compute_vol_pct()` - ATR-based volatility percentile

### Strategy Implementation ✅
- EMA crossover + order book imbalance combo
- 5-second candles from tick stream
- Optuna auto-optimization (70 trials, 14s timeout, per-asset intervals)
- Position stacking up to 3 adds with wave + regime gates
- Stop-and-reverse on signal flip
- Risk management: daily loss limit, consecutive loss breaker, cooldown, trade cap

### Config System ✅
- `config/futures.yaml` with all tunables for 10 assets
- `.env` overrides for credentials and core params
- Typed `FuturesConfig` dataclass hierarchy via `config_loader.py`

---

## Current File Layout

```
futures/
├── .dockerignore
├── .env                          # Credentials + XAI_API_KEY (gitignored)
├── .env.example                  # Template for .env
├── .gitignore
├── LICENSE
├── README.md
├── docker-compose.yml            # futures + redis services
├── pyproject.toml                # Project metadata, deps, tooling
├── requirements.txt              # pip deps (matches pyproject.toml + openai)
├── run.sh                        # Project management script
├── todo.md                       # This file
│
├── config/
│   └── futures.yaml              # Master config (10 assets, all tunables)
│
├── docker/
│   ├── futures/
│   │   ├── Dockerfile            # Python 3.12 multi-stage, ARM64 compatible
│   │   └── entrypoint.sh         # Container startup banner + exec
│   └── redis/
│       ├── Dockerfile            # Redis 7.4-alpine, THP disable
│       └── entrypoint.sh         # Kernel tuning + official entrypoint delegate
│
├── scripts/
│   └── setup-pi.sh               # Raspberry Pi 4 setup script
│
└── src/
    ├── __init__.py
    ├── logging_config.py          # setup_logging() + get_logger()
    ├── main.py                    # ★ ENTRY POINT — supervisor + AssetWorker
    │                              #   - load_config() -> RedisStore -> ccxt
    │                              #   - Spawns 10 AssetWorker tasks
    │                              #   - Spawns ReportGenerator scheduler
    │                              #   - Monitor loop + graceful shutdown
    │                              #   - All indicator math (EMA, AO, regime, wave, quality)
    │                              #   - Sim mode: signals -> Redis (no exchange orders)
    │
    ├── services/
    │   ├── __init__.py
    │   ├── asset_registry.py      # ★ ContractSpec + ASSET_REGISTRY (50+ KuCoin contracts)
    │   ├── config_loader.py       # ★ FuturesConfig dataclass + load_config()
    │   ├── redis_store.py         # ★ RedisStore (async, in-memory fallback)
    │   └── report_generator.py    # ★ ReportGenerator (Grok 4.1 daily/weekly/monthly)
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── cvd.py                 # Cumulative Volume Delta
    │   ├── signal_quality.py      # Signal quality scoring
    │   ├── volatility.py          # Volatility analysis
    │   └── wave_analysis.py       # Ruby wave analysis
    │
    ├── indicators/                # Full indicator library (ported from ruby)
    │   ├── __init__.py
    │   ├── base.py, factory.py, manager.py, registry.py, presets.py
    │   ├── helpers.py, _shims.py
    │   ├── indicators.py, patterns.py, candle_patterns.py
    │   ├── areas_of_interest.py, market_timing.py
    │   ├── momentum/              # RSI, Stochastic
    │   ├── trend/                 # EMA, MACD, MA, WMA, ADL, Bollinger, ATR
    │   ├── volume/                # VWAP, VZO
    │   └── other/                 # CMF, Choppiness, Keltner, SAR, etc.
    │
    ├── tests/
    │   ├── __init__.py, conftest.py
    │   ├── test_analysis_signal_quality.py
    │   ├── test_analysis_volatility.py
    │   ├── test_analysis_wave.py
    │   ├── test_indicators_helpers.py
    │   └── test_main_helpers.py
    │
    └── web/                       # (empty — placeholder for future dashboard)
        ├── __init__.py
        ├── auth.py
        └── main.py
```

**Files marked with ★ are the new/refactored files from this session.**

---

## Stale / Legacy Files to Clean Up

These files exist in the repo from the previous project (Ruby-based futures dashboard)
and are NOT used by the current multi-asset bot. They should be reviewed and cleaned up
after all phases are validated:

| File | Status | Action |
|------|--------|--------|
| `src/grok.py` | **STALE** | Old FastAPI router for Grok SSE streaming. References `lib.core.*` imports that don't exist. Delete or refactor if dashboard is added. |
| `src/grok_helper.py` | **STALE** | Old Grok helper with RustAssistant proxy support. Has useful prompt patterns but references `lib.core.*`. Superseded by `report_generator.py` for reports. Keep as reference for live analysis if dashboard is added. |
| `src/api/grok_api.py` | **STALE** | Duplicate of `src/grok.py`. Delete. |
| `src/core/__init__.py` | **STALE** | Empty `helpers/__init__.py` comment. Delete. |
| `src/core/grok_helper.py` | **STALE** | Another copy of grok_helper. Delete. |
| `src/web/auth.py` | **STALE** | Web auth module from dashboard project. Not used. |
| `src/web/main.py` | **STALE** | Web app entry point from dashboard. Not used. |
| `ruby/` directory | **REFERENCE** | Original Ruby project code. Delete after all ported code is validated. |

**Recommended cleanup (after sim validation):**
```bash
rm src/grok.py src/api/grok_api.py src/core/__init__.py src/core/grok_helper.py
# Keep src/grok_helper.py as reference for live analysis prompts
# Keep src/web/ as placeholder for future dashboard
```

---

## Task Priority / Execution Order

### Next Up: Validate & Test (High Priority)

1. [ ] **Build and start services**: `docker compose build --no-cache && docker compose up -d`
2. [ ] **Verify Redis healthy**: `docker compose logs redis` / `./run.sh redis-logs`
3. [ ] **Start sim locally**: `TRADING_MODE=sim python -m src.main`
4. [ ] **Verify WS feeds connect** for all 10 assets (check logs for tick data)
5. [ ] **Verify signals are being recorded to Redis**: `redis-cli -a PASSWORD LRANGE futures:signals:btc 0 5`
6. [ ] **Let sim run for 1+ hours** to accumulate data
7. [ ] **Generate a daily report**: `python -m src.services.report_generator daily`
8. [ ] **Verify asset symbols exist on KuCoin** (especially FARTCOIN):
   `exchange.load_markets()` filter for USDTM

### Sprint 1: Quality & Polish
- Phase 9 (Tests & Linting) — fix imports, add new tests
- Clean up stale files
- Phase 8 (Discord) — multi-asset notifications

### Sprint 2: Data & Optimization
- Phase 7 (Optuna refinements) — fee model, Redis persistence
- Phase 10 (Data retention) — candle caching, backfill

### Sprint 3: Production Hardening
- Phase 11 — circuit breakers, session recycling, position recovery
- Pi 4 deployment and monitoring

### Sprint 4: Live Trading
- After successful sim runs with validated data
- Change `TRADING_MODE=live` in `.env`
- Start with very small sizes, monitor closely
- Use Grok reports to analyze early live performance

---

## Quick Reference: Asset Symbols

> Full registry in `src/services/asset_registry.py` — 50+ contracts verified against KuCoin API.
> Default enabled: btc, eth, sol, doge, sui, pepe, avax, wif, fartcoin, kcs

### Crypto (Blue Chip & DeFi)

| Key      | KuCoin Symbol    | Max Lev | Category | Description        |
|----------|------------------|---------|----------|--------------------|
| btc      | XBTUSDTM         | 125x    | crypto   | Bitcoin            |
| eth      | ETHUSDTM         | 100x    | crypto   | Ethereum           |
| sol      | SOLUSDTM         | 75x     | crypto   | Solana             |
| xrp      | XRPUSDTM         | 75x     | crypto   | XRP/Ripple         |
| bnb      | BNBUSDTM         | 75x     | crypto   | BNB                |
| doge     | DOGEUSDTM        | 75x     | crypto   | Dogecoin           |
| avax     | AVAXUSDTM        | 75x     | crypto   | Avalanche          |
| sui      | SUIUSDTM         | 75x     | crypto   | Sui                |
| link     | LINKUSDTM        | 75x     | crypto   | Chainlink          |
| ltc      | LTCUSDTM         | 75x     | crypto   | Litecoin           |
| dot      | DOTUSDTM         | 75x     | crypto   | Polkadot           |
| near     | NEARUSDTM        | 75x     | crypto   | Near Protocol      |
| hbar     | HBARUSDTM        | 75x     | crypto   | Hedera             |
| ada      | ADAUSDTM         | 75x     | crypto   | Cardano            |
| ton      | TONUSDTM         | 75x     | crypto   | Toncoin            |
| trx      | TRXUSDTM         | 75x     | crypto   | Tron               |
| ena      | ENAUSDTM         | 75x     | crypto   | Ethena             |
| op       | OPUSDTM          | 75x     | crypto   | Optimism           |
| arb      | ARBUSDTM         | 75x     | crypto   | Arbitrum           |
| ip       | IPUSDTM          | 75x     | crypto   | Story Protocol     |
| hype     | HYPEUSDTM        | 75x     | crypto   | Hyperliquid        |
| pengu    | PENGUUSDTM       | 75x     | crypto   | Pudgy Penguins     |
| tao      | TAOUSDTM         | 75x     | crypto   | Bittensor TAO      |
| sei      | SEIUSDTM         | 75x     | crypto   | Sei                |
| render   | RENDERUSDTM      | 50x     | crypto   | Render             |
| atom     | ATOMUSDTM        | 50x     | crypto   | Cosmos             |
| inj      | INJUSDTM         | 50x     | crypto   | Injective          |
| ondo     | ONDOUSDTM        | 50x     | crypto   | Ondo Finance       |
| pendle   | PENDLEUSDTM      | 50x     | crypto   | Pendle             |
| jup      | JUPUSDTM         | 50x     | crypto   | Jupiter            |
| ray      | RAYUSDTM         | 50x     | crypto   | Raydium            |
| kas      | KASUSDTM         | 50x     | crypto   | Kaspa              |
| virtual  | VIRTUALUSDTM     | 50x     | crypto   | Virtuals Protocol  |
| river    | RIVERUSDTM       | 50x     | crypto   | River              |
| siren    | SIRENUSDTM       | 20x     | crypto   | Siren              |
| kcs      | KCSUSDTM         | 8x      | crypto   | KuCoin Token       |

### Meme Coins

| Key      | KuCoin Symbol    | Max Lev | Category | Description        |
|----------|------------------|---------|----------|--------------------|
| pepe     | PEPEUSDTM        | 75x     | meme     | Pepe               |
| wif      | WIFUSDTM         | 75x     | meme     | dogwifhat          |
| shib     | SHIBUSDTM        | 75x     | meme     | Shiba Inu          |
| floki    | FLOKIUSDTM       | 75x     | meme     | Floki              |
| bonk     | 1000BONKUSDTM    | 75x     | meme     | Bonk (1000x)       |
| fartcoin | FARTCOINUSDTM    | 50x     | meme     | Fartcoin           |
| trump    | TRUMPUSDTM       | 50x     | meme     | Trump Meme         |
| popcat   | POPCATUSDTM      | 50x     | meme     | Popcat             |
| moodeng  | MOODENGUSDTM     | 50x     | meme     | Moo Deng           |

### Precious Metals

| Key       | KuCoin Symbol   | Max Lev | Category | Description            |
|-----------|-----------------|---------|----------|------------------------|
| gold      | PAXGUSDTM       | 30x     | metals   | Gold (PAXG tokenized)  |
| xaut      | XAUTUSDTM       | 75x     | metals   | Gold (Tether Gold)     |
| silver    | XAGUSDTM        | 75x     | metals   | Silver                 |
| platinum  | XPTUSDTM        | 75x     | metals   | Platinum               |
| palladium | XPDUSDTM        | 75x     | metals   | Palladium              |

### Commodities

| Key    | KuCoin Symbol   | Max Lev | Category    | Description      |
|--------|-----------------|---------|-------------|------------------|
| oil    | CLUSDTM         | 50x     | commodities | Crude Oil (WTI)  |
| copper | COPPERUSDTM     | 50x     | commodities | Copper           |

### Stocks (Tokenized)

| Key   | KuCoin Symbol   | Max Lev | Category | Description    |
|-------|-----------------|---------|----------|----------------|
| tsla  | TSLAUSDTM       | 10x     | stocks   | Tesla          |
| nvda  | NVDAUSDTM       | 10x     | stocks   | NVIDIA         |
| amzn  | AMZNUSDTM       | 10x     | stocks   | Amazon         |
| googl | GOOGLUSDTM      | 10x     | stocks   | Google         |
| meta  | METAUSDTM       | 10x     | stocks   | Meta           |
| mstr  | MSTRUSDTM       | 10x     | stocks   | MicroStrategy  |
| coin  | COINUSDTM       | 10x     | stocks   | Coinbase       |
| pltr  | PLTRUSDTM       | 10x     | stocks   | Palantir       |
| hood  | HOODUSDTM       | 10x     | stocks   | Robinhood      |
| intc  | INTCUSDTM       | 10x     | stocks   | Intel          |

> **Notes:**
> - KuCoin uses `XBTUSDTM` for Bitcoin futures (not `BTCUSDTM`).
> - **All symbols verified** against KuCoin `GET /api/v1/contracts/active` API.
> - **KCS:** Only 8x max leverage. Using 4x (half max). Gets 12% margin
>   allocation because low leverage needs more margin for same position sizes.
> - **Stocks:** Max 10x leverage, market type `NASDAQ` on KuCoin.
> - **Metals/Commodities:** Trade 24/7 on KuCoin unlike traditional markets.
> - Asset registry location: `src/services/asset_registry.py`
> - Start in **sim mode** (`TRADING_MODE=sim`) to validate symbols before risking capital.

---

## Quick Reference: Redis Key Patterns

| Key Pattern | Type | Contents |
|-------------|------|----------|
| `futures:signals:{asset}` | List | Trading signals (LPUSH, LTRIM 1000) |
| `futures:pnl:{asset}` | Sorted Set | PnL entries (score=timestamp) |
| `futures:orders:{asset}` | List | Order history (LPUSH, LTRIM 1000) |
| `futures:worker_state` | Hash | Per-asset worker state (field=asset) |
| `futures:heartbeat:{asset}` | String | Worker heartbeat + status (TTL 120s) |
| `futures:reports:{type}:{date}` | String | Generated reports |
| `futures:reports:{type}:latest` | String | Latest report pointer |
| `futures:candles:{asset}:{tf}` | String | Cached candle data (TTL varies) |
| `futures:optuna:{asset}:latest` | String | Best Optuna params (Phase 7) |

---

## Quick Reference: Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KUCOIN_API_KEY` | Live only | - | KuCoin API key |
| `KUCOIN_API_SECRET` | Live only | - | KuCoin API secret |
| `KUCOIN_PASSPHRASE` | Live only | - | KuCoin passphrase |
| `REDIS_PASSWORD` | Yes | - | Redis auth password |
| `REDIS_URL` | No | `redis://futures-redis:6379/0` | Redis connection URL |
| `XAI_API_KEY` | For reports | - | xAI API key for Grok 4.1 |
| `TRADING_MODE` | No | `sim` | `sim` or `live` |
| `CAPITAL` | No | `30.0` | Starting balance USDT |
| `DISCORD_WEBHOOK_URL` | No | - | Discord notifications |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `TZ` | No | `America/New_York` | Timezone |
