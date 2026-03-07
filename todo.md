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

## Active

### 1. Training & Model — Final Validation

- [x] **Full retrain on GPU rig** — 22 symbols, all 13 breakout types, all 9 sessions, 90 days, 25 epochs
  - Champion: 87.1% accuracy, 87.15% precision, 87.27% recall ✅
  - ONNX exported (`breakout_cnn_best.onnx` — 80.7 MB) ✅
  - `feature_contract.json` v6 regenerated ✅

- [ ] **Validate ONNX ↔ PyTorch parity** — run same 18-feature v6 tabular batch through `.pt` and `.onnx` inference; assert max absolute difference < 1e-4
  - The `.pt` and `.onnx` are both present in `models/` — need to run the parity check script

- [ ] **Verify `sync_models.sh`** pulls new `.pt` + `.onnx` + `feature_contract.json` from `nuniesmith/futures` and restarts engine container cleanly

### 2. NT8 — Remaining Validation

- [ ] **Test v6 ONNX auto-adapt** — deploy `BreakoutStrategy.cs` to NT8, compile, verify:
  - Startup log shows `CNN tabular dim: model expects 18, C# builds 18`
  - Per-type TP3 mults loaded from `feature_contract.json` (log each type's mult at startup)
  - Entry logs show `[positions: N/5]`
  - No `OCO ID cannot be reused` or `signal name longer than 50` errors
  - Run for a full session and review output logs

- [ ] **Parity-test Phase 3 EMA9 trailing** — run Python engine + C# strategy side-by-side on same OHLCV data, compare Phase 3 trail stop levels and exit prices. Target: ≤ 1 tick divergence per bar.
  - `test_phase3_ema9_parity.py` — 130 tests all green; warm-up sequences use trending bars ✅

### 3. NT8 — Hard Stop (Take Profit Trader Safety)

- [ ] **4:00 PM ET hard flatten** — add a time-based safety check in `OnBarUpdate()` (BIP0 path) that calls `FlattenAll("TPT_HARD_STOP_16:00")` if current ET time ≥ 16:00 and any positions are open
  - **Critical for TPT accounts**: no overnight positions allowed — violating this = account termination
  - Guard: `if (TptMode && barTimeET.Hour >= 16 && _activePositionCount > 0)` → flatten everything
  - Log clearly: `[TPT] HARD STOP — flattening all positions at 16:00 ET (no overnight holds)`
  - Also set `RiskBlocked = true` with reason `"TPT_SESSION_CLOSED"` until 18:00 ET
  - Re-enable at 18:00 ET: `if (TptMode && barTimeET.Hour >= 18 && RiskBlockReason == "TPT_SESSION_CLOSED")` → unblock

- [ ] **Bridge `/flatten` from web UI** — ensure the Flatten All button in the dashboard triggers Bridge `FlattenAll` which closes every position across all instruments immediately (already wired, needs live test)

- [ ] **Manual trade from dashboard** — when the strategy is always running and I place a manual entry from the web UI via `/execute_signal`, it should coexist with automated entries. Verify:
  - Manual entry gets its own `PositionPhase` tracking
  - Automated entries continue alongside manual positions
  - Both respect `MaxConcurrentPositions = 5`

### 4. Web UI — Trading Mode vs Review Mode

- [ ] **Add UI mode toggle** — "Trading" vs "Review" mode switch in the dashboard header
  - **Trading Mode** (active session):
    - Shows: asset focus cards (entry/stop/TP), positions panel, flatten/cancel buttons, market events feed, alerts, live feed heartbeat, ORB signal history, engine status
    - Hides: next session schedule, Grok AI panel (moved to manual pull), performance stats, dataset preview, volume profile (collapsed by default)
    - Priority: low-latency, actionable info only, minimal clutter
  - **Review Mode** (off-hours / analysis):
    - Shows: everything — full dashboard with all panels expanded
    - Performance panel, trade journal, Grok AI auto-refresh, dataset preview, volume profile, correlation matrix, market regime
    - Useful for post-session review, model monitoring, system health checks
  - Store mode preference in `localStorage`, default to Trading Mode during active hours (03:00–16:00 ET), Review Mode otherwise
  - CSS class toggle: `.mode-trading .review-only { display: none }` / `.mode-review .trading-only { display: none }`

- [ ] **Remove "Next Session" panel** — it's static schedule info that doesn't change; remove from `_render_full_dashboard()` in `dashboard.py` (the session strip at the top already shows open/closed sessions dynamically)

- [ ] **Grok AI → manual pull only** — change Grok panel from auto-refresh (`hx-trigger="every 60s"`) to a manual button ("Ask Grok") that fetches on click
  - Useful during active trading when uncertain about a position
  - Remove the auto-polling to reduce noise and API calls
  - Add `hx-trigger="click"` on a "🤖 Ask Grok" button instead

- [ ] **Fix forex futures spread on asset cards** — `6E` and other forex pairs show identical entry/stop/TP values when ATR is very small relative to price
  - Root cause in `_compute_entry_zone()`: `entry_width = atr * 0.5` — for 6E (price ~1.08, ATR ~0.003) the rounding to 4 decimals collapses the zone
  - Fix: round to appropriate precision based on tick size (6E tick = 0.00005, needs 5 decimal places); or use tick-count display instead of raw price for forex pairs
  - Also: entry_low/entry_high/stop/tp1/tp2 all round to 4 decimals — forex needs at least 5

- [ ] **Estimated dollar value on asset cards** — add `$risk` and `$reward` estimates next to stop/TP levels
  - Already computing `position_size` and `risk_dollars` in `compute_asset_focus()` ✅
  - Add: `$target1` = `position_size × ticks_to_tp1 × dollar_per_tick`, same for TP2
  - Display in the Levels grid: "TP1: 1.0850 (~$45)" format
  - Helps quickly assess whether a trade is worth taking

### 5. Web UI — Trainer Separation & New Pages

- [ ] **Extract trainer UI into its own page** — currently the trainer server has its own HTML UI at `/trainer/` that's proxied through web service. Instead:
  - Create a dedicated `/train` page in the web service dashboard
  - Page shows: training status, start/cancel training, model list, checkpoint history, ONNX export button, validation metrics, dataset stats
  - Trainer service becomes **API-only** (`trainer_server.py` — remove the `trainer_ui` HTML endpoint, keep all API endpoints)
  - Web service proxies all `/api/trainer/*` requests to the trainer service (already mostly wired)
  - Training can be kicked off from the web UI with configurable params (symbols, epochs, days_back, etc.)

- [ ] **Settings page** — new `/settings` page in the web dashboard
  - Configure service URLs (DATA_SERVICE_URL, TRAINER_SERVICE_URL, NT_BRIDGE_HOST)
  - Toggle features: ENABLE_KRAKEN_CRYPTO, ORB_CNN_GATE, ORB_FILTER_GATE
  - View/edit environment overrides (stored in Redis or a config table)
  - Show current Tailscale IPs and connection status for all services
  - Manage Kraken API key status (show if configured, don't show the actual key)
  - Massive API key status
  - Account settings: account size, risk %, max contracts

### 6. Kraken — Full Data Integration for Training

- [ ] **Kraken API key/secret via CI/CD** — ensure `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` are injected as secrets in the CI/CD pipeline and `.env` on prod server
  - Already wired in `docker-compose.yml`: `KRAKEN_API_KEY=${KRAKEN_API_KEY:-}` ✅
  - Already wired in `KrakenDataProvider.__init__`: reads from env vars ✅
  - Need: add `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` to GitHub Actions secrets for deploy step
  - Need: ensure `.env` on Pi has the keys set

- [ ] **Kraken data in training pipeline** — when training is triggered from the web UI, the dataset generator should automatically:
  1. Check Redis cache for recent crypto bars (hot path)
  2. Check Postgres for historical crypto bars (warm path)
  3. Fetch missing data from Kraken REST API (`get_ohlcv_period`) for any gaps (cold path)
  4. Feed crypto data through the same 13 breakout-type detection pipeline
  - `dataset_generator.py` already supports Kraken symbols via `bars_source` config ✅
  - Need: wire the data service's cache-first logic into the training data fetch path
  - The data service already has this pattern in `lifespan()` with `startup_warm_caches()` — extend to training

- [ ] **Unified data resolver for training** — create a `DataResolver` class in the data service that:
  - Accepts a symbol + timeframe + date range
  - Checks: Redis → Postgres → Massive API (futures) / Kraken API (crypto)
  - Returns a unified DataFrame regardless of source
  - Tracks what was cache-hit vs API-fetched (for monitoring)
  - Backfills any newly fetched data into Redis + Postgres for next time
  - Used by: training pipeline, engine focus computation, backfill service

### 7. Multi-Source Breakout Detection (Futures + Crypto)

- [ ] **Cross-asset breakout signals** — use Kraken crypto data alongside Massive futures data to find correlated breakouts
  - BTC/ETH breakout at Asian session → MES/MNQ follow at London/US open (known correlation)
  - Crypto 24/7 data provides overnight context for futures that only trade ~23h
  - Add crypto momentum as an additional CNN tabular feature (future v7 contract)
  - Start with correlation scoring (already in Kraken correlation panel on dashboard) → advance to signal generation

- [ ] **Generalize model across asset classes** — the CNN is already trained on 22 symbols across 5 asset classes (indices, forex, metals, energy, crypto via MBT/MET)
  - Extend to include direct Kraken crypto pairs in training (BTC, ETH, SOL, etc.)
  - The `feature_contract.json` already has `asset_class_map` entries for crypto ✅
  - Need: ensure `dataset_generator.py` can pull Kraken OHLCV and render chart images for crypto pairs

### 8. Trade Copier (Future — Post First Funded Account)

- [ ] **Simple trade copier for multiple TPT accounts** — once the first $50k account is funded and profitable:
  - Mirror all fills from Account 1 → Accounts 2–5
  - Use Bridge AddOn's position push to detect fills on the primary account
  - Fire identical orders on secondary accounts via their own Bridge instances (or a shared copier service)
  - Respect per-account contract limits (each TPT tier has its own max)
  - Scale up to 5 accounts max

---

## Completed

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
- [x] **`ComputeMtfScore(InstrumentState st, string direction)`** — directional score 0.0–1.0 matching Python `MTFAnalyzer` weights: +0.30 EMA stacked, +0.15 EMA slope ≥0.02%/bar, +0.25 MACD histogram polarity, +0.15 histogram slope, +0.15 no opposing divergence (always granted)
- [x] **`GetMtfScore(InstrumentState st, string direction)`** — thin wrapper: calls `ComputeMtfScore` when sentinel is -1, returns `1.0` pass-through otherwise
- [x] **`ShouldReverse()` Gate 4 wired** — replaced `double mtfScore = 1.0` stand-in with `GetMtfScore(st, direction)` in both long and short SAR paths in `CheckBreakout()`
- [x] **Tailscale topology** — NT8 Windows server at `100.127.182.112`; Pi (Docker) at `100.100.84.48`; 15m bars subscribed directly from NT8's data feed — no Pi round-trip needed for MTF computation

### NT8 → Python SAR state sync (`src/ninja/BreakoutStrategy.cs` + `src/lib/services/data/api/sar.py`)
- [x] **`CSarSyncUrl` constant** — `http://100.100.84.48:8000/sar/sync`; `CEngineBaseUrl = http://100.100.84.48:8000`
- [x] **`_sarHttpClient`** — `System.Net.Http.HttpClient` initialised at `DataLoaded` with 3s timeout; disposed at `Terminated`
- [x] **`PushSarSyncAsync(instrName, newDirection, sar, barTime)`** — fire-and-forget `POST /sar/sync` with JSON body; `ContinueWith` logs HTTP status or throttled error on failure; never blocks the bar thread
- [x] **`SarEsc(string s)`** — minimal JSON string escape helper (backslash, quote, newline) used by `PushSarSyncAsync`
- [x] **`TryReversePosition()` Step 7** — calls `PushSarSyncAsync` after `FireEntry` returns for every reversal
- [x] **`FireEntry()` fresh-entry push** — calls `PushSarSyncAsync` when `sarRef.ReversalCount == 0` and `State != Historical` so Python PM learns about the very first position opening
- [x] **`src/lib/services/data/api/sar.py`** — FastAPI router: `POST /sar/sync`, `GET /sar/state`, `GET /sar/state/{asset}`, `DELETE /sar/state/{asset}`, `DELETE /sar/state`; Pydantic models; 24h Redis TTL
- [x] **`_notify_position_manager(payload)`** — lazy-imports `lib.services.engine.main._position_manager`; handles flat/reversal state sync
- [x] **Network** — NT8 → Pi over Tailscale; Pi never initiates to NT8 for SAR (push-only from NT8 side)

### Python PM — TP3 priority fix (`src/lib/services/engine/position_manager.py`)
- [x] **`update_all()` TP3 check order** — moved `_check_tp3_hit()` before `_check_stop_hit()` so TP3 hard exit always takes priority over EMA9 trailing stop on the same bar, matching C# `CheckPhase3Exits` behaviour

### C# `BreakoutStrategy.cs` — `PrepareCnnTabular` fix
- [x] **Feature [17] `tp3_atr_mult_norm`** — now reads `st.Ranges[breakoutType].Tp3AtrMult` (loaded from `feature_contract.json`) instead of hardcoding `5.0 / 5.0 = 1.0`; per-type values now flow correctly into CNN inference

### Parity test suite (`src/tests/test_phase3_ema9_parity.py`)
- [x] **`_warmed_up_state()` warm-up** — switched from flat-price warm-up to trending warm-up (drift=5, LONG rises, SHORT falls) so EMA9 lags the close throughout seeding
- [x] **`run_parity()` initialisation** — direction-aware initial `stop_loss` / `ema9_trail_price`: `0.0` for LONG, `float('inf')` for SHORT
- [x] **`PyPhase3State.process_bar()`** — stop-hit check guarded by `trail_established` flag
- [x] **`test_tp3_takes_priority_over_ema9_stop`** — rewritten with two-phase manual warm-up
- [x] **`TestEndToEndParity` warm-up sequences** — all five scenario tests updated to trending (not flat) warm-up bars
- [x] **`test_flat_market_no_premature_exit`** — replaced sinusoidal oscillation with slow linear rise

### NT8 — v6 Model Compatibility
- [x] **`BreakoutType` enum (4 → 13)** — renamed `Orb` → `ORB`, added 9 new values with explicit ordinals matching Python `IntEnum`
- [x] **`PrepareCnnTabular()` (14 → 18 features)** — added slots [14] `breakout_type_ord`, [15] `asset_volatility_class`, [16] `hour_of_day`, [17] `tp3_atr_mult_norm`
- [x] **`GetVolatilityClass()` helper** — matches Python `ASSET_VOLATILITY_CLASS` dict exactly
- [x] **`NormaliseTabular()` updated** — passthrough normalisation for features [14]–[17] with bounds clamping
- [x] **`CNumTabularFeatures` and `MaxTabular` updated** from 14 → 18
- [x] **Startup diagnostic log** — logs `feature_contract.json v6 (18 features)` and per-type TP3 mults
- [x] **`RangeConfig` entries for all 13 types** — `GetRangeConfig()` has cases for all 13 types
- [x] **All 13 types registered in `InstrumentState` constructor**

### NT8 — All 13 Range Builders (UpdateRangeWindow)
- [x] **ORB** — time-window accumulation (30 min from session open), mid-session startup guard
- [x] **PrevDay** — snapshots prior session H/L via `PrevOrbHigh`/`PrevOrbLow` on session boundary
- [x] **InitialBalance** — time-window accumulation (60 min), mid-session startup guard
- [x] **Consolidation** — ATR contraction squeeze (bar range < `SqueezeThreshold × ATR`), resets until `MinBarsRequired` consecutive squeeze bars
- [x] **Asian** — time-window 19:00–01:00 ET (360 min), hour filter for in-window bars
- [x] **BollingerSqueeze** — tighter squeeze threshold (0.50 × ATR), 10-bar minimum
- [x] **Weekly** — rolling prior-week H/L via bar scan (Mon–Fri before this week's Monday), capped at 2730 bars
- [x] **Monthly** — rolling prior-month H/L via bar scan (before 1st of current month), capped at 12000 bars
- [x] **ValueArea** — prior-session (close, volume) pairs sorted by price; POC outward expansion to 70% volume target → VAH/VAL
- [x] **InsideDay** — today H/L vs yesterday H/L; compression ratio guard (0.25–0.85); mother bar as range
- [x] **GapRejection** — overnight gap ≥ 0.25 × ATR; range = [min(yest_close, today_open), max(…)]
- [x] **PivotPoints** — classic floor pivot formula (P = (H+L+C)/3, R1 = 2P−L, S1 = 2P−H) from prior session H/L/C
- [x] **Fibonacci** — 50-bar swing H/L ≥ 1.5 × ATR; 38.2%–61.8% retracement zone; direction-aware

### NT8 — Per-Type TP3 Mults from feature_contract.json
- [x] **`ParseTp3MultsFromContract(json)`** — manual JSON parser extracts `breakout_types[type].tp3_atr_mult` for all 13 types; no `System.Text.Json` dependency
- [x] **`ApplyTp3MultsToStates()`** — writes loaded mults into every `InstrumentState`'s `RangeState.Tp3AtrMult` field
- [x] **`_tp3MultByType` dictionary** — populated at `OnStateChange(DataLoaded)` from loaded `feature_contract.json`

### NT8 — Chart Renderer (13 Box Styles)
- [x] **`OrbChartRenderer` updated** — `BoxStyle` helper (fill color, border color, solid/dashed), `GetBoxStyle(BreakoutType)` factory for all 13 types
- [x] **`BreakoutType` parameter** threaded through `Render()`, `RenderToTemp()`, `CheckBreakout()`, `PassesCnnFilter()`, `PrepareCnnTabular()`, `RenderCnnSnapshot()`
- [x] **PNG filenames** prefixed with `rng_{ordinal}_` for per-type disambiguation

### NT8 — TP3 + EMA9 Trailing
- [x] **`Tp3AtrMult` per-type** via `RangeState.Tp3AtrMult` (loaded from contract; fallback = 5.0)
- [x] **`EnableTp3Trailing` flag**
- [x] **`BreakoutPhase` enum** (`Phase1`, `Phase2`, `Phase3`, `Closed`) + `PositionPhase` class
- [x] **3-phase bracket walk** in `OnOrderUpdate` — Phase 1 → Phase 2 → Phase 3
- [x] **`UpdateEma9()` helper** — standard EMA(9) seeded from SMA, called per primary bar
- [x] **`CheckPhase3Exits()`** — per BIP0 bar, checks Phase 3 positions: TP3 limit exit or EMA9 adverse cross market exit

### NT8 — Bridge AddOn (`src/ninja/addons/Bridge.cs`)
- [x] **`Bridge.cs` as NT8 AddOn** (inherits `AddOnBase`) — HTTP listener on port 5680
- [x] **HTTP endpoints** — `GET /health`, `GET /status`, `GET /orders`, `GET /metrics`, `POST /execute_signal`, `POST /flatten`, `POST /cancel_orders`
- [x] **Position push** — POST snapshot to `{DashboardBaseUrl}/api/positions/update` on every fill + 15-second heartbeat
- [x] **Risk gate** — `SignalBus.IsRiskBlocked` / `SignalBus.RiskBlockReason` static volatile fields
- [x] **Prometheus metrics** — 14 gauge/counter metrics
- [x] **CORS headers** on all responses; graceful shutdown in `OnStateChange(Terminated)`

### NT8 — Deploy Scripts (`scripts/`)
- [x] **`deploy_nt8.ps1`** — pulls C# source + DLLs + ONNX + `feature_contract.json` from `nuniesmith/futures`; patches `NinjaTrader.Custom.csproj`; optional `-Launch`
- [x] **`deploy_nt8.bat`** — thin Windows launcher; forwards all args to PS1
- [x] **`scripts/pull_model.ps1`** — standalone model-only update: downloads `.onnx` + `feature_contract.json`, verifies SHA256
- [x] **`Bridge.cs` in deploy manifest** — deployed to `AddOns\Bridge.cs`

### NT8 — Crash Resilience
- [x] `OnOrderUpdate` handler — absorbs rejected orders instead of terminating strategy
- [x] OCO GUID uniqueness — 6-char GUID suffix on every OCO ID
- [x] SL price validation — corrects stop to correct side of market before submission
- [x] Signal name truncation — capped at 49 chars (NT8 limit = 50)
- [x] Try/catch on all `SubmitOrderUnmanaged` calls
- [x] `MaxConcurrentPositions = 5` with fill/flat tracking via `OnOrderUpdate`
- [x] Reduced to 5 core instruments (`MGC, MES, MNQ, MYM, 6E`)
- [x] Cooldown 10 min; `OrbCnnPredictor.NumTabular` dynamic from ONNX metadata
- [x] Startup diagnostics — logs CNN tabular dimension, per-type TP3 mults, and position count

### NT8 — TPT Mode
- [x] **`TptMode` property** — NinjaScript property, defaults to `true`
- [x] **`TptAccountTier` enum** — FiftyK (2 micros), HundredK (3 micros), HundredFiftyK (4 micros)
- [x] **`GetTptContracts()`** — returns fixed contract count based on tier
- [x] **`FireEntry()`** — uses `GetTptContracts()` when TptMode=true instead of dynamic ATR-based sizing

### Engine & Detection
- [x] 13 range builder functions + `DEFAULT_CONFIGS` for all types in Python engine
- [x] All 13 types wired into `detect_range_breakout()` dispatch and `detect_all_breakout_types()`
- [x] `BreakoutResult.extra` dict with type-specific metadata
- [x] `PositionManager` wired into engine main loop
- [x] `OrderCommand` emitter published to NT8 Bridge via `_publish_pm_orders()`; also writes `engine:pm:positions` to Redis (TTL 120s)
- [x] Session-level performance stats in daily report
- [x] Backfill gap detection — `_check_and_alert_gaps()` publishes `engine:gap_alerts`
- [x] Asset watchlists — `CORE_WATCHLIST`, `EXTENDED_WATCHLIST`, `ACTIVE_WATCHLIST`

### Dashboard & Bridge API
- [x] **`GET /positions/`** — reads `bridge:positions` cache key; returns `NTPositionsResponse`
- [x] **`POST /positions/update`** — written by Bridge AddOn on every fill + heartbeat
- [x] **`POST /positions/execute_signal`** — pre-flight risk check, proxies to Bridge
- [x] **`POST /positions/flatten`** — proxies to Bridge `/flatten`
- [x] **`POST /positions/cancel_orders`** — proxies to Bridge `/cancel_orders`
- [x] **`GET /positions/bridge_status`** — heartbeat age, account, bridge version, position count
- [x] **`GET /positions/bridge_orders`** — proxies to Bridge `/orders`
- [x] **SSE `bridge-status` event** — emitted whenever bridge online/offline state changes
- [x] **SSE `positions-update` event** — emitted via pub/sub channel and polled from cache fallback
- [x] 13 breakout type filter pills + 9 session tabs in signal history table
- [x] MTF score % column with MACD slope arrow and divergence icon
- [x] Trade journal — inline editing, tag filtering, quick-add, limit selector
- [x] Kraken crypto price chart — SVG candlestick, OHLCV REST, pair/interval/period selectors
- [x] Crypto/futures correlation panel — Pearson matrix (9 Kraken pairs + MES/MGC/MNQ)
- [x] CNN dataset preview — `GET /cnn/dataset/preview`, base64 PNG cards

### Training & Dataset
- [x] `evaluate_model()` in `breakout_cnn.py` — accuracy/precision/recall; used by trainer pipeline gate
- [x] Feature contract v6 — 18 tabular features; `generate_feature_contract()` + `contract` CLI subcommand
- [x] `BreakoutType` embedding in CNN — `Embedding(13, 8)` table, `use_type_embedding` flag
- [x] Session-specific inference thresholds — `SESSION_THRESHOLDS` dict (9 keys)
- [x] Balanced sampling — `max_samples_per_type_label` + `max_samples_per_session_label`
- [x] Synthetic augmentation — `RandomRotation(1.5°)` + `RandomErasing(p=0.05)`
- [x] `trainer_server.py` pipeline: generate → split → train → evaluate → gate → promote → ONNX export → feature contract write
- [x] `split_dataset()` — 85/15 stratified split by label, used by training pipeline

### Dashboard & Bridge UI
- [x] **Positions panel** — `#positions-container` with 10s polling; renders account, positions table, unrealized P&L, cash balance, pending orders count
- [x] **Bridge status badge** — green `● BRIDGE` (connected + age) or grey `○ BRIDGE` (offline)
- [x] **Flatten All button** — `hx-post="/api/positions/flatten"` with confirm dialog; disabled when bridge offline
- [x] **Cancel Orders button** — `hx-post="/api/positions/cancel_orders"` with confirm dialog
- [x] **SSE bridge-status event handler** — JS listener triggers HTMX refresh on connect/disconnect transition

### Kraken Integration
- [x] **`KrakenDataProvider`** — full REST client: public (OHLCV, ticker, asset pairs, server time) + private (balance, trade balance, open orders, trade history) endpoints
- [x] **`KrakenFeedManager`** — WebSocket feed: OHLC + trade subscriptions, bar aggregation, Redis cache push, reconnect logic
- [x] **9 crypto pairs** tracked with live prices on dashboard
- [x] **REST API connected** — health check shows green status
- [x] **WS live** — 5097+ bars and 7214+ trades streaming
- [x] **Kraken health endpoint** — `/api/kraken/health` with detailed status
- [x] **Kraken chart HTML** — SVG candlestick chart with pair/interval/period selectors
- [x] **Kraken account HTML** — balance display (when authenticated)
- [x] **Kraken correlation panel** — Pearson correlation matrix (crypto + futures)
- [x] **Rate limiting** — public (0.35s) and private (1.0s) rate limits enforced
- [x] **HMAC-SHA512 signing** for private endpoints

### Massive Integration
- [x] **`MassiveDataProvider`** — REST client: aggregates, daily, snapshots, recent trades, active contracts, products, schedules, quotes, market statuses
- [x] **`MassiveFeedManager`** — WebSocket: bars, trades, quotes, front-month resolution, second-agg upgrade/downgrade
- [x] **Front-month contract resolution** — automatic continuous contract mapping
- [x] **Primary data source for training** — `bars_source=massive` default in trainer config

### Infrastructure & Monitoring
- [x] Rate limiting — `rate_limit.py` Redis-backed with env-var overrides
- [x] `sync_models.sh` — Git LFS resolution + SHA256 verify; `--restart` flag
- [x] Trainer Prometheus metrics — `trainer_images_generated`, `trainer_label_balance`, `trainer_render_time_seconds`
- [x] "Training Data Health" Grafana dashboard provisioned
- [x] CI/CD — Lint → Test → Build & push 3 tagged images on push to `main`
- [x] **Pi deploy stage** (`deploy-pi` job in CI/CD) — Tailscale connect → SSH deploy → health checks → Discord notify
- [x] **Tailscale connect action** — OAuth-based Tailscale connection in CI/CD pipeline
- [x] **SSH deploy action** — git pull, docker-compose up, prune dangling images
- [x] **Health check action** — container health verification post-deploy
- [x] **Discord notifications** — build started, tests passed/failed, docker pushed, deploy success/failure

### Monorepo Consolidation
- [x] All training code merged from `orb` → `lib/training`
- [x] `rb` repo eliminated — models live in `futures/models/` (Git LFS)
- [x] `ninjatrader` repo eliminated — C# source lives in `src/ninja/`, deploy scripts in `scripts/`
- [x] All URLs updated from `nuniesmith/orb` → `nuniesmith/futures`

---

## Next Steps (Priority Order)

### Immediate — Validate & Ship (before TPT account)
1. **ONNX ↔ PyTorch parity check** — run validation script, confirm <1e-4 max diff
2. **NT8 deploy + compile** — `.\scripts\deploy_nt8.bat`, verify startup logs show correct tabular dim and TP3 mults
3. **NT8 full session test** — let it run a full trading day, review all logs for errors
4. **Hard stop at 4 PM ET** — implement in `BreakoutStrategy.cs` (critical for TPT safety)
5. **Fix forex spread display** — update `_compute_entry_zone()` rounding for forex tick sizes
6. **Verify `sync_models.sh`** — pull latest champion and restart engine cleanly

### Short-term — Web UI & UX (this week)
7. **Remove "Next Session" panel** from dashboard
8. **Grok AI → manual pull** — button instead of auto-refresh
9. **Trading/Review mode toggle** — CSS-based panel visibility switching
10. **Dollar estimates on asset cards** — show $risk and $target alongside price levels
11. **Extract trainer UI to `/train` page** — trainer becomes API-only
12. **Settings page** (`/settings`) — service URLs, feature toggles, API key status

### Medium-term — Data & Training Pipeline (next 2 weeks)
13. **Kraken API keys in CI/CD secrets** — add to GitHub Actions and `.env` on Pi
14. **Unified `DataResolver` class** — Redis → Postgres → API with auto-backfill
15. **Kraken data in training pipeline** — crypto chart images + breakout detection for training
16. **Cross-asset breakout signals** — crypto momentum as context for futures entries

### Launch — Take Profit Trader Account
17. **Get TPT $50k evaluation account** — real Rithmic data feed
18. **Verify strategy with real data** — Rithmic fills, slippage, commission costs
19. **Pass challenge** — demonstrate consistent profitability within TPT rules
20. **First funded account** — live trading with real capital

### Post-Launch — Scale
21. **Trade copier** — mirror fills from Account 1 → Accounts 2–5
22. **Cloud migration** — once profitable, move from local Tailscale to cloud infra with proper CI/CD
23. **Domain + TLS** — get a domain, set up HTTPS for the dashboard

### Ongoing
- Per-type/session win-rate Prometheus metrics and Grafana panels
- Feature contract drift detection — log warning in C# if loaded `feature_contract.json` version ≠ compiled `CFeatureContractVersion` constant
- Per-type backtest validation in NT8 Strategy Analyzer
- Automated label rebalancing + synthetic augmentation tuning as dataset grows
- SAR backtest validation — run NT8 Strategy Analyzer on a full quarter
- MTF divergence detection — add swing-point pivot detection to `UpdateMtf` for the +0.15 weight
- SAR dashboard panel — `GET /sar/state` visible in HTMX dashboard