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
- [ ] **Validate ONNX ↔ PyTorch parity** — run same 18-feature v6 tabular batch through `.pt` and `.onnx` inference; assert max absolute difference < 1e-4
  - The `.pt` and `.onnx` are both present in `models/` — need to run the parity check script
- [ ] **Verify `sync_models.sh`** pulls new `.pt` + `.onnx` + `feature_contract.json` from `nuniesmith/futures` and restarts engine container cleanly

### NT8 Validation
- [ ] **Test v6 ONNX auto-adapt** — deploy `BreakoutStrategy.cs` to NT8, compile, verify:
  - Startup log shows `CNN tabular dim: model expects 18, C# builds 18`
  - Per-type TP3 mults loaded from `feature_contract.json` (log each type's mult at startup)
  - Entry logs show `[positions: N/5]`
  - No `OCO ID cannot be reused` or `signal name longer than 50` errors
  - Run for a full session and review output logs
- [ ] **Parity-test Phase 3 EMA9 trailing** — run Python engine + C# strategy side-by-side on same OHLCV data, compare Phase 3 trail stop levels and exit prices. Target: ≤ 1 tick divergence per bar.
  - `test_phase3_ema9_parity.py` — 130 tests all green; warm-up sequences use trending bars ✅

### NT8 Hard Stop (Take Profit Trader Safety)
- [ ] **4:00 PM ET hard flatten** — add a time-based safety check in `OnBarUpdate()` (BIP0 path) that calls `FlattenAll("TPT_HARD_STOP_16:00")` if current ET time ≥ 16:00 and any positions are open
  - **Critical for TPT accounts**: no overnight positions allowed — violating this = account termination
  - Guard: `if (TptMode && barTimeET.Hour >= 16 && _activePositionCount > 0)` → flatten everything
  - Log clearly: `[TPT] HARD STOP — flattening all positions at 16:00 ET (no overnight holds)`
  - Also set `RiskBlocked = true` with reason `"TPT_SESSION_CLOSED"` until 18:00 ET
  - Re-enable at 18:00 ET: `if (TptMode && barTimeET.Hour >= 18 && RiskBlockReason == "TPT_SESSION_CLOSED")` → unblock

---

## 🟡 Medium Priority

### Web UI — Trading Mode vs Review Mode
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

### NT8 Bridge Trading Tests
- [ ] **Bridge `/flatten` from web UI** — ensure the Flatten All button in the dashboard triggers Bridge `FlattenAll` which closes every position across all instruments immediately (already wired, needs live test)
- [ ] **Manual trade from dashboard** — when the strategy is always running and I place a manual entry from the web UI via `/execute_signal`, it should coexist with automated entries. Verify:
  - Manual entry gets its own `PositionPhase` tracking
  - Automated entries continue alongside manual positions
  - Both respect `MaxConcurrentPositions = 5`

---

## 🟢 Low Priority

### Web UI — Trainer Separation & New Pages
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

### Kraken — Full Data Integration for Training
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

### Multi-Source Breakout Detection (Futures + Crypto)
- [ ] **Cross-asset breakout signals** — use Kraken crypto data alongside Massive futures data to find correlated breakouts
  - BTC/ETH breakout at Asian session → MES/MNQ follow at London/US open (known correlation)
  - Crypto 24/7 data provides overnight context for futures that only trade ~23h
  - Add crypto momentum as an additional CNN tabular feature (future v7 contract)
  - Start with correlation scoring (already in Kraken correlation panel on dashboard) → advance to signal generation
- [ ] **Generalize model across asset classes** — the CNN is already trained on 22 symbols across 5 asset classes (indices, forex, metals, energy, crypto via MBT/MET)
  - Extend to include direct Kraken crypto pairs in training (BTC, ETH, SOL, etc.)
  - The `feature_contract.json` already has `asset_class_map` entries for crypto ✅
  - Need: ensure `dataset_generator.py` can pull Kraken OHLCV and render chart images for crypto pairs

### Trade Copier (Future — Post First Funded Account)
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