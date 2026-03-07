# futures — TODO

> **Single repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:engine` · `:web` · `:trainer`
> **NinjaTrader deploy**: `scripts/deploy_nt8.bat` → `scripts/deploy_nt8.ps1` (pulls all C# source + ONNX from this repo)
> **Infrastructure**: Local (Tailscale mesh) — Pi (engine + web), GPU rig (trainer), Windows (NT8)

---

## Repo Layout

```
futures/
├── src/
│   ├── lib/
│   │   ├── core/          # BreakoutType, RangeConfig, models, multi_session, alerts, cache
│   │   ├── analysis/      # breakout_cnn, chart_renderer, mtf_analyzer, regime, scorer, …
│   │   ├── training/      # dataset_generator, orb_simulator, trainer_server
│   │   ├── trading/       # costs, engine, strategies
│   │   ├── services/
│   │   │   ├── engine/    # main, breakout, scheduler, position_manager, backfill, risk, focus, …
│   │   │   ├── web/       # HTMX dashboard, FastAPI reverse-proxy (port 8080)
│   │   │   └── data/      # FastAPI data API (positions, SSE, bridge, trades, journal, kraken, …)
│   │   └── integrations/  # kraken_client, massive_client, grok_helper
│   └── ninja/
│       ├── BreakoutStrategy.cs   # NT8 strategy (single-file, all deps inlined)
│       ├── RubyIndicator.cs      # NT8 chart indicator
│       └── addons/
│           ├── Bridge.cs         # NT8 HTTP bridge AddOn (port 5680)
│           └── DataPreloader.cs  # NT8 history seeder AddOn
├── models/                # champion .pt, .onnx, feature_contract.json (Git LFS)
├── scripts/
│   ├── deploy_nt8.bat     # Windows launcher (double-click to deploy NT8)
│   ├── deploy_nt8.ps1     # NT8 deploy: pulls C# + DLLs + ONNX from this repo
│   ├── sync_models.sh     # Pi-side: pull .pt + .onnx from this repo → restart engine
│   └── …
├── config/                # Prometheus, Grafana, Alertmanager
├── docker/                # Dockerfiles per service
└── docker-compose.yml
```

---

## Current State

- **Monorepo**: All source — engine, web, trainer, lib, C# strategies, deploy scripts — lives here. No separate `rb` or `ninjatrader` repos.
- **Models**: `models/breakout_cnn_best.pt` + `.onnx` + `feature_contract.json` committed (Git LFS). Engine pulls via `sync_models.sh`, NT8 pulls via `deploy_nt8.ps1`. Latest champion: **87.1% accuracy**, 87.15% precision, 87.27% recall, 25 epochs, v6 18-feature, 41 checkpoints saved.
- **Docker**: `:engine` (data API + CNN inference), `:web` (HTMX dashboard), `:trainer` (GPU training server). Runs on Pi (engine + web) and GPU rig (trainer).
- **Feature Contract**: v6, 18 tabular features. `models/feature_contract.json` is the canonical source for both Python and C#.
- **CNN Model**: EfficientNetV2-S + tabular head. ONNX export for NT8. `OrbCnnPredictor` auto-detects tabular dimension from ONNX metadata at load time. Training pipeline: generate dataset → train → evaluate → gate check (≥80% acc, ≥75% prec, ≥70% rec) → promote → export ONNX + feature_contract.json.
- **Breakout Types**: 13 — ORB, PrevDay, InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci. Fully wired in engine detection, training, dataset generator, C# enum, CNN tabular vector, chart renderer, `UpdateRangeWindow`, and `CheckBreakout`.
- **NT8 Strategy**: `BreakoutStrategy.cs` — 5 core instruments (`MGC, MES, MNQ, MYM, 6E`), v6 18-feature tabular vector, all 13 `BreakoutType` range builders fully implemented (no stubs remain), 13 chart renderer box styles, 3-phase bracket walk (TP1 → breakeven → EMA9 trail to TP3), per-type `tp3_atr_mult` loaded from `feature_contract.json` via `ParseTp3MultsFromContract` + `ApplyTp3MultsToStates`, crash-resilient. TPT Mode built-in ($50k/2, $100k/3, $150k/4 micros).
- **NT8 Bridge**: `Bridge.cs` AddOn — HTTP listener (port 5680), position push to dashboard (`POST /api/positions/update`), 15s heartbeat, `SignalBus.IsRiskBlocked` risk gate, Prometheus `/metrics`, all endpoints including `/execute_signal`, `/flatten`, `/cancel_orders`, `/status`, `/orders`, `/health`.
- **NT8 Deploy**: `deploy_nt8.bat` / `deploy_nt8.ps1` — downloads C# source, DLLs, ONNX model, `feature_contract.json` from this repo and installs into NT8.
- **TP3 + EMA9 Trailing**: 3-phase bracket in both Python `position_manager.py` and C# `BreakoutStrategy.cs`. Phase 1 (SL/TP1) → Phase 2 (breakeven) → Phase 3 (EMA9 trail to TP3).
- **Position Manager**: `position_manager.py` — always-in 1-lot micro positions, reversal gates (CNN ≥ 0.85, MTF ≥ 0.60, 30min cooldown), Redis persistence, `OrderCommand` emitter for Bridge.
- **Dashboard**: HTMX + FastAPI — live signals, 13 breakout type filter pills, 9 session tabs, MTF score column, trade journal, Kraken crypto chart + correlation panel, Grok AI analyst, CNN dataset preview, positions panel, Bridge status badge, flatten/cancel buttons, market regime (HMM), performance panel, volume profile, equity curve, asset focus cards with entry/stop/TP levels.
- **Kraken Integration**: `KrakenDataProvider` REST client (public + private endpoints), `KrakenFeedManager` WebSocket feed (OHLC + trades), 9 crypto pairs (ETH, SOL, LINK, AVAX, DOT, ADA + more), API key/secret from env vars, auto-start on engine boot when `ENABLE_KRAKEN_CRYPTO=1`. REST API connected, WS live with 5097+ bars and 7214+ trades streaming.
- **Massive Integration**: `MassiveDataProvider` REST client (futures OHLCV, snapshots, contracts), `MassiveFeedManager` WebSocket (bars, trades, quotes), front-month resolution, used as primary bars source for training.
- **Data Service**: Unified data layer inside engine service — checks Redis cache first, then Postgres, then fetches from external APIs (Massive for futures, Kraken for crypto). Startup cache warming from Postgres (7 days). Bar data persisted to both Redis (hot) and Postgres (durable).
- **Training**: `trainer_server.py` FastAPI HTTP server (port 8200). `dataset_generator.py` covers all 13 types + all 9 sessions + Kraken. Session-specific thresholds. Type embedding. Synthetic augmentation. Balanced sampling. Full pipeline: generate → split (85/15 stratified) → train → evaluate → gate → promote → ONNX export.
- **Monitoring**: Prometheus + Grafana. "Training Data Health" dashboard provisioned.
- **CI/CD**: Lint → Test → Build & push 3 Docker images → Deploy to Pi via Tailscale SSH → Health checks → Discord notifications. All on push to `main`.
- **Tailscale**: NT8 Windows at `100.127.182.112`, Pi (Docker) at `100.100.84.48`, all services communicate over Tailscale mesh. HTTP only (no domain/TLS needed for local mesh).

---

## 🔴 High Priority

### Training & Model Validation
- [x] **Validate ONNX ↔ PyTorch parity** — run same 18-feature v6 tabular batch through `.pt` and `.onnx` inference; assert max absolute difference < 1e-4
  - `scripts/check_onnx_parity.py` — runs 64+ synthetic samples through both models; asserts max abs diff < 1e-4; prints per-batch stats with `--verbose`; validates feature_contract.json name/count; exits 0 on pass, 1 on fail
- [x] **Verify `sync_models.sh`** pulls new `.pt` + `.onnx` + `feature_contract.json` from `nuniesmith/futures` and restarts engine container cleanly
  - Reviewed: LFS pointer detection, SHA256 verification, `--restart` flag calls `docker compose restart engine` — fully correct; no changes needed

### NT8 Validation
- [ ] **Test v6 ONNX auto-adapt** — deploy `BreakoutStrategy.cs` to NT8, compile, verify:
  - Startup log shows `CNN tabular dim: model expects 18, C# builds 18`
  - Per-type TP3 mults loaded from `feature_contract.json` (log each type's mult at startup)
  - Entry logs show `[positions: N/5]`
  - No `OCO ID cannot be reused` or `signal name longer than 50` errors
  - Run for a full session and review output logs
- [ ] **Parity-test Phase 3 EMA9 trailing** — run Python engine + C# strategy side-by-side on same OHLCV data, compare Phase 3 trail stop levels and exit prices. Target: ≤ 1 tick divergence per bar.
  - `test_phase3_ema9_parity.py` — 130 tests all green; warm-up sequences use trending bars

### NT8 Hard Stop (Take Profit Trader Safety)
- [x] **4:00 PM ET hard flatten** — added `CheckTptHardStop()` called from BIP0 path in `OnBarUpdate()` when `TptMode == true`
  - `CheckTptHardStop()` in `BreakoutStrategy.cs` — converts bar time to ET, calls `_engine.FlattenAll("TPT_HARD_STOP_16:00")` + sets `RiskBlocked=true` / `RiskBlockReason="TPT_SESSION_CLOSED"` at 16:00 ET
  - Re-enables at 18:00 ET when reason is `TPT_SESSION_CLOSED` — logs `[TPT] Risk gate LIFTED`
  - Crash-resilient: wrapped in try/catch so a timezone error never stops the strategy

---

## 🟡 Medium Priority

### Web UI — Trading Mode vs Review Mode
- [x] **Add UI mode toggle** — "Trading" vs "Review" mode switch added to dashboard header (right side, next to clock)
  - `⚡ Trading` / `🔍 Review` pill buttons in header; active state highlighted green (trading) or blue (review)
  - `body.mode-trading .review-only { display: none }` / `body.mode-review .trading-only { display: none }` CSS rules added
  - Panels marked `review-only`: Dataset Preview, Crypto Chart, Correlation, Volume Profile, Performance, Trade Journal, Market Regime
  - Mode stored in `localStorage['dashMode']`; auto-detected from ET hour (03:00–16:00 → Trading, otherwise → Review) when no saved preference
  - Pre-applied before first paint via inline `<script>` in `<head>` to prevent flash
  - In Review Mode, Grok container gets `hx-trigger="every 60s"` re-applied dynamically via htmx.process()
- [x] **Remove "Next Session" panel** — static schedule block removed from `_render_full_dashboard()` in `dashboard.py`; session strip at top already shows live open/closed state
- [x] **Grok AI → manual pull only** — `hx-trigger` changed from `every 60s` to `load` (single fetch on page load); the existing `📋 Brief` / `⚡ Update` buttons in the Grok panel header are the manual pull mechanism; Review Mode re-enables polling via JS
- [x] **Fix forex futures spread on asset cards** — added `_price_decimals(tick_size)` helper in `focus.py`; `_compute_entry_zone()` now accepts `tick_size` and rounds to `max(2, min(decimal_places_of_tick, 7))`; 6E (tick=0.00005) now shows 5 decimal places instead of collapsing to 4; `compute_asset_focus()` passes `tick_size` through; `price_decimals` field added to focus payload
- [x] **Estimated dollar value on asset cards** — `compute_asset_focus()` now computes `target1_dollars` and `target2_dollars` (position_size × ticks_to_tp × dollar_per_tick); displayed as inline `~$N` badges next to TP1/TP2 in the Levels grid; stop shows `-$risk` in red below price; asset card `_render_asset_card()` uses tick-aware `_fmt()` formatter for all price fields

### NT8 Bridge Trading Tests
- [ ] **Bridge `/flatten` from web UI** — ensure the Flatten All button in the dashboard triggers Bridge `FlattenAll` which closes every position across all instruments immediately (already wired, needs live test)
  - Python-side wiring fully tested offline: `src/tests/test_bridge_trading.py` — TestFlattenAll (6 tests) covers heartbeat liveness gate, port resolution from heartbeat, proxy forwarding to Bridge `/flatten`, default reason, connection error handling (503/504)
  - Dashboard HTML wiring verified: `_render_positions_panel()` in `dashboard.py` renders `hx-post="/api/positions/flatten"` button with `hx-confirm`, disabled when Bridge offline
  - Live test script ready: `scripts/test_bridge_live.py` — run with `--bridge HOST:PORT --data URL` when NT8 is up on Sim (also `--local` for same-machine testing)
  - **Remaining**: fire up NT8 on Sim, run `python scripts/test_bridge_live.py`, verify flatten closes all positions across all instruments
- [ ] **Manual trade from dashboard** — when the strategy is always running and I place a manual entry from the web UI via `/execute_signal`, it should coexist with automated entries. Verify:
  - Manual entry gets its own `PositionPhase` tracking
  - Automated entries continue alongside manual positions
  - Both respect `MaxConcurrentPositions = 5`
  - Python-side wiring fully tested offline: `src/tests/test_bridge_trading.py` — TestExecuteSignal (6 tests) covers market long, short with all fields, risk check enforcement, risk check bypass (enforce_risk=False), direction validation, connection errors
  - C# wiring verified by code review: `Bridge.cs` `ListenLoop` parses `/execute_signal` POST → creates `SignalBus.Signal` → `SignalBus.Enqueue()` → `BreakoutStrategy.OnBarUpdate` calls `_engine.DrainSignalBus()` → `ProcessSignal()` → `SubmitOrderUnmanaged()` with unique signal name `Signal-{dir}-{id}` (capped at 49 chars)
  - `FireEntry()` registers each entry in `_positionPhases[signalId]` with its own `PositionPhase` struct — manual Bridge entries use `signalId = "brg-{guid}"`, automated entries use `signalId = "brk-{dir}-{timestamp}-{instrument}-{type}"`
  - `MaxConcurrentPositions` gate at top of `FireEntry()`: `if (_activePositionCount >= MaxConcurrentPositions) return;` — applies to both automated and manual entries
  - **Remaining**: live test with NT8 on Sim — run `scripts/test_bridge_live.py --local` (or `--bridge HOST:PORT --data URL`), verify both entry types get PositionPhase in Output Window

---

## 🟢 Low Priority

### Web UI — Trainer Separation & New Pages
- [x] **Extract trainer UI into its own page** — trainer service is now API-only; full dashboard page lives in the data service
  - `trainer_server.py` — `trainer_ui` HTML endpoint removed; all `/train`, `/status`, `/logs`, `/models`, `/export_onnx`, `/metrics/prometheus` endpoints kept
  - `src/lib/services/data/api/trainer.py` — new router: `GET /trainer` (full HTML dashboard page), `GET|POST /trainer/config` (trainer URL config), `GET /trainer/service_status`, `GET|POST /trainer/api/*` (proxy to trainer service)
  - `src/lib/services/data/main.py` — `trainer_router` registered; `/trainer*` paths added to `api_info`
  - `src/lib/services/web/main.py` — `/trainer` and `/trainer/*` now proxy to the **data service** (not directly to trainer); trainer client removed from web service lifespan; `TRAINER_SERVICE_URL` env var no longer needed in web service
  - Dashboard page: training status card, start/cancel buttons, symbol/epoch/days_back params, model list, ONNX export, validation metrics, log stream, dataset stats — all backed by `/trainer/api/*` → trainer service
- [x] **Settings page** — new `/settings` page in the web dashboard
  - `src/lib/services/data/api/settings.py` — complete rewrite with 5 tabbed sections: Engine, Services, Features, Risk & Trading, API Keys
  - **Engine tab**: account size pills ($50K/$100K/$150K), primary interval selector, lookback period, force refresh / optimize buttons, live feed start/stop/upgrade/downgrade, engine status panel, quick links, about card
  - **Services tab**: editable service URLs (DATA_SERVICE_URL, TRAINER_SERVICE_URL, NT_BRIDGE_HOST, NT_BRIDGE_PORT), "Test Connectivity" button probes all services (data, trainer, Bridge, Redis, Postgres) with latency, NT8 Bridge heartbeat status card (connected/offline, account, version, port, risk blocked)
  - **Features tab**: 11 toggle switches — Kraken Crypto Feed, Massive Autostart, Grok AI, CNN Gate, ORB Filter Gate, MTF Alignment, SAR, TPT Mode, TP3 Trailing, Auto Brackets, Debug Logging; saved to Redis immediately via `POST /settings/features/update`
  - **Risk & Trading tab**: position sizing (risk %, max contracts, max concurrent positions), SL/TP ATR multiples (SL, TP1, TP2, TP3), entry cooldown, tick-based defaults, SAR parameters (min CNN prob, min MTF score, cooldown, chase ATR fraction, winning CNN prob, high winner R-mult), CNN filter settings (threshold override, session key, lookback bars), ORB quality filters (volume surge mult, volume avg period, min ATR ratio, ORB minutes, require VWAP toggle)
  - **API Keys tab**: shows configured/missing status for MASSIVE_API_KEY, KRAKEN_API_KEY, KRAKEN_API_SECRET, XAI_API_KEY, DISCORD_WEBHOOK_URL, POSTGRES_DSN, REDIS_URL — values never exposed, only boolean "SET" / "MISSING" badges; security note card explains env var approach
  - All settings persisted to Redis via `settings:overrides` key (JSON dict with `services`, `features`, `risk` sub-keys); `_load_persisted_settings()` / `_save_persisted_settings()` helpers
  - Backend API endpoints: `GET/POST /settings/services/config|update`, `GET /settings/services/probe`, `GET /settings/services/bridge_status`, `GET/POST /settings/features/config|update`, `GET/POST /settings/risk/config|update`, `GET /settings/keys/status`
  - `src/lib/services/web/main.py` — 9 new proxy routes added for all settings sub-endpoints
  - Section tabs with `localStorage['settingsTab']` persistence; auto-loads section-specific data on tab switch
  - Consistent dark/light theme support with existing dashboard styling

### Kraken — Full Data Integration for Training
- [x] **Kraken API key/secret via CI/CD** — `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` injected in both CI/CD and docker-compose
  - `.github/workflows/ci-cd.yml` pre-deploy step: `upsert_env "KRAKEN_API_KEY"` + `upsert_env "KRAKEN_API_SECRET"` added
  - `docker-compose.yml`: both vars passed to `engine` and `trainer` services via `${KRAKEN_API_KEY:-}` / `${KRAKEN_API_SECRET:-}`
  - `KrakenDataProvider.__init__` reads both from env vars at construction time
- [x] **Kraken data in training pipeline** — `dataset_generator.py` fully wired for Kraken OHLCV
  - `_is_kraken_symbol()` routes `KRAKEN:*` prefixed symbols to `_load_bars_from_kraken()`
  - `_load_bars_from_kraken()` calls `KrakenDataProvider.get_ohlcv_period()` and normalises to standard OHLCV DataFrame
  - `load_bars()` fallback chain: for Kraken symbols tries `kraken → db → csv`; for futures symbols tries configured source → `db → cache → massive → csv`
  - `_SYMBOL_TO_TICKER` maps short aliases (`BTC`, `ETH`, `SOL`, …) to `KRAKEN:*` internal tickers
  - `trainer_server.py` `DEFAULT_SYMBOLS` updated to include `BTC,ETH,SOL` — 25 symbols total (22 CME micros + 3 Kraken spot)
  - `breakout_cnn.py` `ASSET_CLASS_ORDINALS` + `ASSET_VOLATILITY_CLASS` include all Kraken internal tickers and short aliases
  - `models/feature_contract.json` regenerated: `asset_class_map` and `asset_volatility_classes` now include `BTC`, `ETH`, `SOL`, …, `KRAKEN:XBTUSD`, …, `KRAKEN:XRPUSD` (42 total asset_class entries)
- [x] **Unified data resolver for training** — `src/lib/services/data/resolver.py` created
  - `DataResolver` class: three-tier resolution Redis → Postgres → API (Massive for futures, Kraken for crypto)
  - `ResolveMetadata` dataclass tracks source, rows, cache_hit, backfilled_redis, backfilled_postgres, duration_ms
  - Auto-backfill: newly API-fetched data written back to Postgres + Redis for next-run cache hits
  - `resolve()`, `resolve_with_meta()`, `resolve_batch()`, `resolve_batch_with_meta()` public API
  - Module-level `get_resolver()` singleton + `resolve()` shortcut for simple callers
  - Import-cycle-safe: `_SYMBOL_TO_TICKER` / `_resolve_ticker` inlined (no import from `dataset_generator`)
  - Used by: training pipeline `load_bars()` cold path, future engine focus computation

### Multi-Source Breakout Detection (Futures + Crypto)
- [x] **Cross-asset breakout signals** — use Kraken crypto data alongside Massive futures data to find correlated breakouts
  - BTC/ETH breakout at Asian session → MES/MNQ follow at London/US open (known correlation)
  - Crypto 24/7 data provides overnight context for futures that only trade ~23h
  - Add crypto momentum as an additional CNN tabular feature (future v7 contract)
  - Start with correlation scoring (already in Kraken correlation panel on dashboard) → advance to signal generation
  - **DONE**: `src/lib/analysis/crypto_momentum.py` — full crypto momentum scorer module created
    - `CryptoMomentumScorer` class: orchestrates momentum computation across BTC/SOL anchors → scores 5 futures targets (MES, MNQ, MGC, MCL, MYM)
    - `compute_single_crypto_momentum()`: per-crypto metrics — EMA-9/21 spread & cross, RSI-14, ATR-14, session high/low breakout, volume surge ratio
    - `score_futures_from_crypto()`: weighted composite scoring — 40% momentum strength, 25% correlation reliability, 20% session timing, 15% volume confirmation
    - `CryptoMomentumSignal.to_tabular_feature()`: returns normalised [-1, +1] value ready for v7 feature contract (`crypto_momentum_score` as tabular feature #19)
    - Session-aware scoring: Asian session signals get 0.8 session_score (4h lead time), London 0.5 (2h), US pre-open 0.3, RTH 0.15 (contemporaneous)
    - Rolling Pearson correlation blended with base correlation config per futures instrument — trust increases with sample count
    - `CRYPTO_ANCHORS`: BTC (weight 0.50), SOL (weight 0.15) — weighted composite direction
    - `FUTURES_TARGETS`: MES (base_corr 0.55), MNQ (0.60), MYM (0.50), MGC (0.25), MCL (0.15)
    - `detect_session()`: auto-detects Asian/London/US pre-open/RTH from ET time for lead-time scoring
    - Pure computation functions (no I/O) for testability: `compute_ema`, `compute_rsi`, `compute_atr`, `compute_volume_ratio`, `pearson_correlation`, `log_returns`
    - `score_with_data()` accepts pre-built DataFrames for offline testing; `score_all()` pulls live Kraken + futures data
    - `crypto_momentum_to_tabular()` convenience: converts signal list → `{futures_symbol: float}` dict for CNN input
  - **Tests**: `src/tests/test_crypto_momentum.py` — 109 tests all green
    - `TestComputeEma`: empty, insufficient, exact period, trend tracking, length matching (5 tests)
    - `TestComputeRsi`: insufficient data, pure uptrend/downtrend, flat market, range bounds (5 tests)
    - `TestComputeAtr`: minimal data, two bars, positive ATR, volatility scaling (4 tests)
    - `TestComputeVolumeRatio`: uniform, surge, empty, single (4 tests)
    - `TestPearsonCorrelation`: perfect +/-, no correlation, insufficient, constant, mismatched (6 tests)
    - `TestLogReturns`: basic, empty, zero prices (3 tests)
    - `TestDetectSession`: all 4 sessions, boundaries, lead hours, gap period (10 tests)
    - `TestComputeSessionHighLow`: empty, normal, overnight, missing columns (4 tests)
    - `TestComputeSingleCryptoMomentum`: bullish/bearish/neutral, insufficient bars, EMA spread, RSI, ATR, volume surge, to_dict, lowercase cols, strength bounds (16 tests)
    - `TestScoreFuturesFromCrypto`: bullish/bearish/neutral, unknown symbol, rolling corr, weak corr, session timing, volume, multi-anchor, bounds, confidence, thresholds, to_dict (17 tests)
    - `TestTabularFeature`: bullish +, bearish -, neutral 0, not-actionable 0, clamped (5 tests)
    - `TestCryptoMomentumToTabular`: basic, empty (2 tests)
    - `TestCryptoMomentumScorer`: basic, all-bull, all-bear, no data, partial, with correlation, custom targets/anchors (8 tests)
    - `TestConfiguration`: anchor fields, weights, target fields, sessions, thresholds, min_bars (6 tests)
    - `TestEdgeCases`: single bar, exact min, large dataset, zero prices, constant prices, NaN, empty dicts/DFs, all targets scored, deterministic (10 tests)
    - `TestRealisticScenarios`: BTC Asian breakout→MES, crypto selloff→equity, flat neutral, mixed signals, gold lower corr (5 tests)
  - **Next**: wire `crypto_momentum_score` into engine scoring pipeline as an optional boost; add as v7 feature contract tabular feature #19 at next retrain
- [x] **Generalize model across asset classes** — the CNN is already trained on 22 symbols across 5 asset classes (indices, forex, metals, energy, crypto via MBT/MET)
  - Extend to include direct Kraken crypto pairs in training (BTC, ETH, SOL, etc.) — **DONE**: `trainer_server.py` `DEFAULT_SYMBOLS` includes `BTC,ETH,SOL` (25 symbols total)
  - The `feature_contract.json` already has `asset_class_map` entries for crypto — **VERIFIED**: 42 total entries including all 9 Kraken pairs (short + internal ticker forms)
  - `dataset_generator.py` can pull Kraken OHLCV and render chart images for crypto pairs — **VERIFIED**: full pipeline tested in `src/tests/test_kraken_training_pipeline.py` (67+ tests)
    - `_SYMBOL_TO_TICKER`: all 9 Kraken pairs mapped with short alias (BTC), internal ticker (KRAKEN:XBTUSD), and pair alias (XBTUSD) — 27 entries
    - `_is_kraken_symbol()`: correctly routes crypto to Kraken loader, CME futures (MBT/MET) to Massive
    - `_load_bars_from_kraken()`: REST pagination via Kraken public OHLC endpoint, deduplication, zero-bar filtering, graceful error handling
    - `load_bars()`: DataResolver three-tier fallback (Redis → Postgres → Kraken API) with auto-backfill for crypto
    - Chart rendering: `generate_dataset_for_symbol()` is symbol-agnostic — works on any OHLCV DataFrame; both mplfinance and parity renderers accept crypto bars
    - `_build_row()`: `get_asset_class_id()` returns 1.0 for all crypto, `get_asset_volatility_class()` returns 1.0 for all crypto
    - `feature_contract.json` ↔ `breakout_cnn.py` consistency verified: `ASSET_CLASS_ORDINALS` and `ASSET_VOLATILITY_CLASS` match contract for all 42 symbols
  - **Ready for retrain**: run trainer with `BTC,ETH,SOL` in symbol list — Kraken OHLCV will be fetched, chart images rendered, and tabular features built with correct asset_class_id=1.0

### Trade Copier (Future — Post First Funded Account)
- [ ] **Simple trade copier for multiple TPT accounts** — once the first $50k account is funded and profitable:
  - Mirror all fills from Account 1 → Accounts 2–5
  - Use Bridge AddOn's position push to detect fills on the primary account
  - Fire identical orders on secondary accounts via their own Bridge instances (or a shared copier service)
  - Respect per-account contract limits (each TPT tier has its own max)
  - Scale up to 5 accounts max

---

## Completed

### Trainer UI Separation (`src/lib/training/trainer_server.py`, `src/lib/services/data/api/trainer.py`, `src/lib/services/web/main.py`)
- [x] `trainer_server.py` HTML endpoint (`trainer_ui`) removed — trainer is now pure API server
- [x] `src/lib/services/data/api/trainer.py` created — full HTML trainer dashboard page at `GET /trainer`, config endpoints, `/trainer/api/*` proxy
- [x] `src/lib/services/data/main.py` — `trainer_router` imported and registered; trainer paths added to `api_info`
- [x] `src/lib/services/web/main.py` — `/trainer` and `/trainer/*` now proxy to data service (not directly to trainer:8200); trainer httpx client removed; `TRAINER_SERVICE_URL` env var removed from web service env block

### Unified Data Resolver (`src/lib/services/data/resolver.py`)
- [x] `DataResolver` class — Redis → Postgres → Massive/Kraken API three-tier resolution with automatic backfill
- [x] `ResolveMetadata` dataclass — source, rows, cache_hit, backfilled_redis/postgres, duration_ms, error
- [x] `resolve()`, `resolve_with_meta()`, `resolve_batch()`, `resolve_batch_with_meta()` public methods
- [x] `get_resolver()` module-level singleton + `resolve()` shortcut
- [x] No import cycle with `dataset_generator.py` — symbol map inlined in resolver

### Kraken Training Pipeline Integration (`src/lib/training/dataset_generator.py`, `src/lib/analysis/breakout_cnn.py`, `models/feature_contract.json`)
- [x] `dataset_generator.py` — `_is_kraken_symbol()`, `_load_bars_from_kraken()`, `_SYMBOL_TO_TICKER` short aliases (BTC→KRAKEN:XBTUSD etc.), `load_bars()` Kraken routing
- [x] `trainer_server.py` — `DEFAULT_SYMBOLS` updated: 22 CME micros + BTC, ETH, SOL (25 total)
- [x] `breakout_cnn.py` — `ASSET_CLASS_ORDINALS` + `ASSET_VOLATILITY_CLASS` include all Kraken tickers
- [x] `models/feature_contract.json` — regenerated with 42-entry `asset_class_map` + `asset_volatility_classes` including all Kraken internal tickers and short aliases
- [x] CI/CD — `KRAKEN_API_KEY` + `KRAKEN_API_SECRET` both injected in pre-deploy step and passed through docker-compose to engine + trainer


### ONNX ↔ PyTorch Parity Check (`scripts/check_onnx_parity.py`)
- [x] `scripts/check_onnx_parity.py` created — loads `.pt` via `_build_model_from_checkpoint` and `.onnx` via `onnxruntime`, runs 64 synthetic v6 18-feature batches, asserts max abs diff < 1e-4
- [x] Feature contract validation: checks version, feature count (18), and optionally name order against `TABULAR_FEATURES`
- [x] `--verbose` flag prints per-batch min/max/diff; `--n-samples`, `--threshold`, `--device` (auto/cpu/cuda/mps) args
- [x] Exit 0 = pass (safe to deploy to NT8), Exit 1 = fail (re-export ONNX)

### NT8 TPT Hard Stop — 4:00 PM ET session close (`src/ninja/BreakoutStrategy.cs`)
- [x] `CheckTptHardStop()` method added in new `#region TPT hard stop` block
- [x] Converts bar time to ET via `TimeZoneInfo.ConvertTimeFromUtc` with `"Eastern Standard Time"` zone
- [x] 16:00–17:59 ET: sets `RiskBlocked=true` + `RiskBlockReason="TPT_SESSION_CLOSED"`, calls `_engine.FlattenAll("TPT_HARD_STOP_16:00")` if `_activePositionCount > 0`; retries on next bar if FlattenAll throws
- [x] 18:00+ ET: clears `TPT_SESSION_CLOSED` block so new Globex session trading is allowed
- [x] Called from BIP0 path in `OnBarUpdate()` when `TptMode == true`; wrapped in try/catch for crash resilience

### Web UI — Trading / Review Mode + Dashboard Cleanup (`src/lib/services/data/api/dashboard.py`)
- [x] `⚡ Trading` / `🔍 Review` pill toggle added to dashboard header; persisted in `localStorage['dashMode']`; auto-detects from ET hour on first visit
- [x] CSS: `body.mode-trading .review-only { display: none }` / `body.mode-review .trading-only { display: none }` — zero JS overhead, pure CSS visibility
- [x] Review-only panels: Dataset Preview, Crypto Chart, Correlation, Volume Profile, Performance, Trade Journal, Market Regime
- [x] Grok `hx-trigger` changed from `every 60s` → `load`; Review Mode restores polling via `setDashboardMode()` JS
- [x] Static "Next Session / Schedule" panel removed from sidebar
- [x] `_price_decimals(tick_size)` helper + `tick_size` param to `_compute_entry_zone()` — forex (6E, 6B, 6J) now correctly uses 5–7 decimal places
- [x] `target1_dollars` / `target2_dollars` computed in `compute_asset_focus()` and displayed as `~$N` badges on TP1/TP2 in asset cards; stop shows `-$risk`

### CNN Model — Full Retrain (v6, 87.1% accuracy)
- [x] 22-symbol training: MGC, SIL, MHG, MCL, MNG, MES, MNQ, M2K, MYM, 6E, 6B, 6J, 6A, 6C, 6S, ZN, ZB, ZC, ZS, ZW, MBT, MET
- [x] All 13 breakout types, all 9 sessions, 90 days lookback, 25 epochs
- [x] 85/15 stratified train/val split
- [x] Validation gates: 87.1% acc (≥80%), 87.15% prec (≥75%), 87.27% rec (≥70%) — all passed
- [x] Champion promoted: `breakout_cnn_best.pt` + `breakout_cnn_best.onnx` (80.7 MB)
- [x] `feature_contract.json` v6 regenerated with all 13 type configs
- [x] `breakout_cnn_best_meta.json` written with full training config and metrics
- [x] 41 checkpoint `.pt` files saved during training (timestamped with accuracy)
- [x] ONNX export: opset 17, dynamic batch axes, validated with `onnx.checker`

### NT8 — Stop-and-Reverse (SAR) always-in micro position (`src/ninja/BreakoutStrategy.cs`)
- [x] `ReversalState` sealed class — direction, signalId, entryPrice, ATR, SL, lastReversalTime, reversalCount; `RMultiple(price)` + `IsWinning(price)` helpers; `Open()` / `Close()` lifecycle methods
- [x] `_sarStates[]` allocated at DataLoaded alongside `_states[]`; SAR constants mirror Python PM env vars: `CSarMinCnnProb=0.85`, `CSarWinningCnnProb=0.92`, `CSarHighWinnerCnnProb=0.95`, `CSarMinMtfScore=0.60`, `CSarCooldownMinutes=30`, `CSarChaseMaxAtrFraction=0.50`, `CSarChaseMinCnnProb=0.90`
- [x] `ShouldReverse()` — 5 gates matching Python `_should_reverse`: direction, CNN prob, cooldown, MTF, high-winner protection
- [x] `DecideEntryType()` — limit-at-range-edge / market-chase logic matching Python `_decide_entry_type`
- [x] `TryReversePosition()` — flatten → clean phase tracking → reset fired flags → pre-decrement active count → FireEntry → update cooldown
- [x] `CheckBreakout()` SAR path: when `rs.FiredLong`/`rs.FiredShort` already set, check `sar.IsShort`/`sar.IsLong` and evaluate reversal gates; fresh entries still use original path
- [x] `PassesCnnFilter()` overload with `out double cnnProbOut` for SAR reversal gate evaluation
- [x] `FireEntry()` → `sarRef.Open()` stamps direction + signalId on fill; `OnOrderUpdate` closes SAR state on flatten/SL/TP; per-bar `Positions[bip]` sync as belt-and-suspenders

### NT8 — MTF (15-minute) EMA/MACD alignment scoring (`src/ninja/BreakoutStrategy.cs`)
- [x] **`InstrumentState` MTF fields** — EMA-9/21/50 incremental state, MACD-12/26/9 incremental state, histogram ring-buffer (3 bars) for slope, EMA-50 5-bar ring-buffer for slope, `MtfScore` sentinel (-1 = ready, 1.0 = warm-up pass-through), `MtfBip` back-reference to the 15m BIP index
- [x] **15m `AddDataSeries`** — one 15m series added per tracked instrument in `Configure` immediately after each 1m series; primary instrument (BIP0) gets its 15m series separately; all use the same trading-hours template as the 1m series
- [x] **`_mtfBipBySymbol` map** — built at `DataLoaded` by scanning `BarsArray` for `BarsPeriod.Value == 15`; wires `st.MtfBip` on every matching `InstrumentState`
- [x] **`UpdateMtf(int mtfBip, InstrumentState st)`** — called from `OnBarUpdate` whenever a 15m BIP fires a new closed bar; incremental EMA-9/21/50, MACD-12/26/9, histogram ring-buffer; writes `-1` sentinel to `MtfScore` once both EMA-50 (≥50 bars) and MACD signal (≥35 bars) are warmed up; `1.0` pass-through during warm-up