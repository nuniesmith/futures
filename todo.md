# futures — TODO

> **Single repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:engine` · `:web` · `:trainer`
> **Infrastructure**: Local (Tailscale mesh) — Pi (engine + web), GPU rig (trainer)

---

## Repo Layout

```
futures/
├── src/
│   ├── lib/
│   │   ├── core/          # BreakoutType, RangeConfig, session, models, alerts, cache, asset_registry
│   │   ├── analysis/      # breakout_cnn, breakout_filters, chart_renderer, mtf_analyzer, regime, scorer, …
│   │   ├── strategies/
│   │   │   ├── rb/        # range breakout scalping (detector, range_builders, publisher)
│   │   │   ├── daily/     # bias_analyzer, daily_plan, swing_detector
│   │   │   └── costs.py   # slippage, commission modelling
│   │   ├── services/
│   │   │   ├── engine/    # main, handlers, scheduler, position_manager, backfill, risk, focus, live_risk, …
│   │   │   ├── web/       # HTMX dashboard, FastAPI reverse-proxy (port 8080)
│   │   │   └── data/      # FastAPI data API (positions, SSE, bridge, trades, journal, kraken, …)
│   │   └── integrations/  # kraken_client, massive_client, grok_helper
│   └── pine/
│       └── ruby_futures.pine   # TradingView Pine Script indicator (primary live trading UI)
├── models/                # champion .pt, feature_contract.json (Git LFS)
├── scripts/
│   ├── sync_models.sh     # Pi-side: pull .pt from this repo → restart engine
│   └── …
├── config/                # Prometheus, Grafana, Alertmanager
├── docker/                # Dockerfiles per service
└── docker-compose.yml
```

---

## 🎯 Goal — Python Co-Pilot + TradingView Live Trading

The system is a **manual trading co-pilot**. It informs entries, it doesn't execute them. The live trading workflow is entirely browser-native — no Windows, no NinjaTrader required.

```
Python Engine (Pi)
    ├── Computes: daily bias, ORB levels, PDR, IB, CNN signals, entry/stop/TP
    ├── Dashboard: focus mode cards, risk strip, swing actions, Grok brief
    └── Writes signals.csv → GitHub (nuniesmith/futures-signals)
               ↓ request.seed() reads (2-5 min lag)

TradingView (browser, Linux-native)
    ├── Ruby Futures indicator (Pine Script)
    │   ├── Draws engine levels: ORB box, PDR, IB, entry/stop/TP lines
    │   ├── CNN signal labels + futures contract sizing on every signal
    │   └── Tradovate broker connected → fills show natively in TV
    └── PickMyTrade webhook → copies fills to all accounts simultaneously

Python Dashboard (tiled alongside TV)
    └── Real-time CNN probabilities, risk strip, focus cards, swing signals
```

**Two-stage scaling plan:**
- **Stage 1 — TPT**: 5 × $150K accounts = $750K total buying power.
- **Stage 2 — Apex**: 20 × $300K accounts = ~$6M total buying power.
- **Copy layer**: TradingView manual entry → PickMyTrade webhook → all accounts via Tradovate simultaneously. Own-accounts-only copy trading is explicitly allowed by both Apex and TPT.

**Milestone before going live on demo funds:**
1. Codebase cleanup (Phase 1 refactors — reduce complexity before retraining)
2. CNN v7.1 retrain (28 features, targeting ≥89% accuracy)
3. TradingView `signals.csv` publisher + Ruby Futures indicator wired end-to-end
4. Full workflow test on Tradovate demo

---

## Current State

- **Monorepo**: All source — engine, web, trainer, lib, Pine Script, deploy scripts.
- **Models**: `models/breakout_cnn_best.pt` + `feature_contract.json` committed (Git LFS). Latest champion: **87.1% accuracy**, 87.15% precision, 87.27% recall, 25 epochs, v6 18-feature. Retrain on v7.1 (28-feature) pending — this is the next major milestone.
- **Docker**: `:engine` (data API + CNN inference), `:web` (HTMX dashboard), `:trainer` (GPU training server). Runs on Pi (engine + web) and GPU rig (trainer).
- **Feature Contract**: v7.1, 28 tabular features. `models/feature_contract.json` is the canonical source. Expanded from v6 (18) with 6 daily strategy features (bias direction/confidence, prior day pattern, weekly range position, monthly trend score, crypto momentum) and 4 sub-features (breakout type category, session overlap flag, ATR trend, volume trend).
- **CNN Model**: EfficientNetV2-S + tabular head. `OrbCnnPredictor` auto-detects tabular dimension from checkpoint metadata at load time. Training pipeline: generate dataset → train → evaluate → gate check (≥80% acc, ≥75% prec, ≥70% rec) → promote. Python `_normalise_tabular_for_inference()` handles v6→v7→v7.1 backward-compat padding so the old model works with the new code.
- **Breakout Types**: 13 — ORB, PrevDay, InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci. Fully wired in engine detection, training, dataset generator, CNN tabular vector, chart renderer.
- **Position Manager**: `position_manager.py` — always-in 1-lot micro positions, reversal gates (CNN ≥ 0.85, MTF ≥ 0.60, 30min cooldown), Redis persistence, `OrderCommand` emitter (informational — feeds dashboard, not live execution).
- **Dashboard**: HTMX + FastAPI — live signals, 13 breakout type filter pills, 9 session tabs, MTF score column, trade journal, Kraken crypto chart + correlation panel, Grok AI analyst, CNN dataset preview, positions panel, flatten/cancel buttons, market regime (HMM), performance panel, volume profile, equity curve, asset focus cards with entry/stop/TP levels, focus mode, risk strip, swing action buttons.
- **Kraken Integration**: `KrakenDataProvider` REST + `KrakenFeedManager` WebSocket feed. 9 crypto pairs streaming.
- **Massive Integration**: `MassiveDataProvider` REST + `MassiveFeedManager` WebSocket. Front-month resolution, primary bars source for training.
- **Data Service**: Unified data layer — Redis cache → Postgres → external APIs. Startup cache warming from Postgres (7 days).
- **Training**: `trainer_server.py` FastAPI (port 8200). `dataset_generator.py` covers all 13 types + 9 sessions + Kraken. Full pipeline: generate → split (85/15 stratified) → train → evaluate → gate → promote.
- **CI/CD**: Lint → Test → Build & push 3 Docker images → Deploy to Pi via Tailscale SSH → Health checks → Discord notifications.
- **Tailscale**: Pi (Docker) at `100.100.84.48`, GPU rig for training. All services communicate over Tailscale mesh.

---

## Architecture Issues Identified (Pre-Refactor)

### Triple duplication of breakout types & config
- `lib/core/breakout_types.py` — canonical `BreakoutType` (IntEnum) + `RangeConfig` (CNN/training source)
- `lib/services/engine/breakout.py` — **second** `BreakoutType` (StrEnum) + **second** `RangeConfig` (engine runtime)
- `lib/services/engine/orb.py` — **third** dataclass `ORBSession` with its own ATR params

Bridge mapping dicts (`_ENGINE_TO_TRAINING`, `_TRAINING_TO_ENGINE`) exist purely to convert between the two enums. These should not exist.

### `orb.py` is an isolated silo (1800+ lines)
Has its own `ORBResult`, `detect_opening_range_breakout()`, `compute_atr()`. `breakout.py` was built to generalise ORB but lives alongside it with parallel code paths. `main.py` has **10 separate `_handle_check_orb_*` functions** that all delegate to the same `_handle_check_orb`.

### `main.py` is a 3285-line god module
`_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` are 90% copy-paste. Each handler repeats: fetch bars → detect → get HTF bars → MTF enrich → persist → publish → dispatch to PM → send alert. ~400 lines of duplicate code that should be one generic function.

### `analysis/orb_filters.py` is misnamed
The filters (NR7, premarket range, session window, lunch filter, MTF bias, VWAP confluence) are NOT ORB-specific. They apply to any range breakout type.

### Web UI focus is too broad for live trading *(partially resolved)*
`focus.py` composite ranking and focus mode grid are implemented. Remaining: wire Grok optional macro brief cleanly into the pre-market plan.

### Risk system is not real-time position-aware *(resolved)*
`LiveRiskState` wired, dual micro/regular sizing on cards, live position overlays, risk strip all shipped.

---

## 🔴 Pre-Retrain Cleanup — RB System Refactor

Reduce complexity before the v7.1 retrain. These make the codebase easier to reason about during training experiments and reduce the chance of bugs introduced by the duplicate enum/config split.

### Phase 1A: Merge BreakoutType Enums → Single Source of Truth
- [ ] Eliminate the engine `StrEnum` in `services/engine/breakout.py` — use `core/breakout_types.BreakoutType` (IntEnum) everywhere
  - Remove `_ENGINE_TO_TRAINING` / `_TRAINING_TO_ENGINE` mapping dicts
  - Remove `to_training_type()` / `from_training_type()` / `breakout_type_ordinal()` bridge functions
  - Update all engine callers to import from `lib.core.breakout_types`
  - `BreakoutResult.to_dict()` uses `.name` for JSON serialisation and `.value` for ordinals

### Phase 1B: Merge `RangeConfig` → Single Dataclass
- [ ] Unify the two `RangeConfig` dataclasses into `core/breakout_types.py`
  - Merge detection-threshold fields (ATR mult, body ratio, range caps, squeeze params) INTO the core `RangeConfig`
  - All 13 `_*_CONFIG` registry entries get the detection fields added
  - Kill the engine-side `RangeConfig` entirely — `get_range_config(BreakoutType.ORB)` returns everything

### Phase 1C: Merge ORB Detection into Unified RB Detector
- [ ] `detect_opening_range_breakout()` becomes `detect_range_breakout(config=ORB_CONFIG)`
  - Extract all `_build_*_range()` functions into `strategies/rb/range_builders.py`
  - Single `detect_range_breakout(bars, symbol, config)` in `strategies/rb/detector.py`
  - `ORBResult` retired — `BreakoutResult` covers all types
  - Single `compute_atr()` in `strategies/rb/detector.py` (deduplicate 3 copies)

### Phase 1D: Extract Generic Handler Pipeline from `main.py`
- [ ] One handler function for all 13 breakout types — eliminate ~400 lines of copy-paste
  - Create `services/engine/handlers.py` with a single `handle_breakout_check(engine, breakout_type, session_key)`
  - `_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` become one-liners
  - Extract shared helpers: `fetch_bars_1m`, `get_htf_bars`, `run_mtf_on_result`, `persist_breakout_result`, `publish_breakout_result`, `send_breakout_alert`

### Phase 1E: Rename `orb_filters.py` → `breakout_filters.py` ✅
- [x] `ORBFilterResult` → `BreakoutFilterResult`. Backward-compat shim in place.

### Phase 1F: Rename `orb_simulator.py` → `rb_simulator.py` ✅
- [x] `simulate_orb_outcome` → `simulate_rb_outcome`. Shim in place.

### Phase 1G: Create `lib/strategies/` Package
- [ ] Clean separation of strategy code from infrastructure
  - `lib/strategies/rb/` — Range Breakout scalping system (detector, range_builders, publisher)
  - Move `lib/trading/costs.py` → `lib/strategies/costs.py`
  - Move `lib/trading/strategies.py` → `lib/strategies/strategy_defs.py`
  - Move `lib/trading/engine.py` → `lib/strategies/backtesting.py`
  - Rename `ORBSession` → `RBSession` (keep old name as alias)

---

## 🔴 CNN Retrain — v7.1 Feature Contract (THE Milestone)

This is the gate before going live. Everything above unblocks a cleaner training run. Everything below this can be tested once a retrained model is running.

### Phase 4A: New Features from Daily Strategy Layer ✅
- [x] 6 new v7 features (features [18]–[23]): `daily_bias_direction`, `daily_bias_confidence`, `prior_day_pattern`, `weekly_range_position`, `monthly_trend_score`, `crypto_momentum_score`
- [x] `feature_contract.json` updated to v7.1 (28 features)
- [x] `dataset_generator.py` `_build_row()` computes all 6 features with neutral fallbacks
- [x] `_normalise_tabular_for_inference()` handles v6→v7→v7.1 backward-compat padding

### Phase 4B: Sub-Features and Richer Encoding ✅
- [x] 4 sub-features (features [24]–[27]): `breakout_type_category`, `session_overlap_flag`, `atr_trend`, `volume_trend`
- [x] All 4 computed in `dataset_generator.py` `_build_row()` with neutral fallbacks

### Phase 4C: Retrain on v7.1 Contract
- [ ] **Generate new dataset** with all 28 features across all 25 symbols, 13 types, 9 sessions
  - Daily bias features computed from historical daily bars (look back 1 day per sample)
  - Weekly/monthly features from aligned historical weekly/monthly bars
  - Crypto momentum features from aligned Kraken data
- [ ] **Train** with same EfficientNetV2-S + tabular head architecture, larger tabular input (28 features)
- [ ] **Gate check**: ≥88% acc, ≥85% prec, ≥82% rec (higher bar than v6 — more features, more signal)
- [ ] **Promote** to `breakout_cnn_best.pt` + updated `feature_contract.json` v7.1
- [ ] Deploy to Pi via `sync_models.sh` → engine hot-reload via `ModelWatcher`

---

## 🔴 TradingView Integration — Live Trading UI

The Pine Script indicator is the primary live trading chart. The Python dashboard runs alongside it for CNN context and risk management.

### Phase TV-A: `signals.csv` GitHub Publisher
- [ ] **Add `signals_publisher.py`** to engine — writes engine signals to a GitHub-hosted CSV on every signal fire
  - On breakout signal + CNN gate pass: append row to `signals.csv`:
    `timestamp, symbol, direction, breakout_type, session, entry, stop, tp1, tp2, tp3, cnn_prob, atr`
  - Push to `nuniesmith/futures-signals` repo via GitHub API (or git push via deploy key)
  - Keep last 500 rows — prune older entries on each push
  - Rate-limit: max one push per 30 seconds (debounce for multiple simultaneous signals)
  - `GITHUB_SIGNALS_TOKEN` env var for auth, `ENABLE_SIGNALS_PUBLISHER=1` feature flag
  - TradingView `request.seed()` reads this with a 2-5 min natural lag — acceptable for signal levels

### Phase TV-B: Ruby Futures Indicator — Engine Signal Overlay
- [ ] **Add engine signal layer to `ruby_futures.pine`** using `request.seed()` to read `signals.csv`
  - Parse the CSV: filter to current chart symbol + recent timestamps (last 5 bars)
  - Draw on chart: entry line (dashed), stop line (red), TP1/TP2/TP3 levels (green dashes)
  - Signal label box: breakout type name, CNN probability, contract sizing
  - **Micro + regular sizing on every label**: "3× MGC ($330 risk) / 1× GC ($1,100 risk)"
  - Colour-code by direction: green labels for LONG, red for SHORT
  - Only show signals from last N hours (configurable input, default 4h) to avoid chart clutter

### Phase TV-C: Ruby Futures Indicator — Core Futures Layer
- [ ] **Pure price calculations** that run at zero delay (no `request.seed()` dependency)
  - ORB box: session open high/low as shaded rectangle, extends until broken
  - PDR: prior day high/low as dashed lines (extend right across chart)
  - Initial Balance: first 60-min RTH high/low
  - Asian range: 19:00–02:00 ET H/L as background shading
  - VWAP: session VWAP line (standard Pine `ta.vwap`)
  - EMA 9/21/50 on chart (toggleable)
  - Session separators with session name labels (London, NY, etc.)
  - Futures contract info panel (input): symbol, micro point value, tick size → shows ATR in ticks and dollars

### Phase TV-D: TradingView → Python Engine Webhook
- [ ] **Add `POST /api/tv/alert` endpoint** to data service
  - TV alert message format: `{"symbol": "MGC", "action": "LONG_ENTRY", "price": 2891.5, "note": "ORB breakout"}`
  - Engine logs the alert, triggers a fresh CNN inference on that symbol, pushes result to dashboard via `dashboard:tv_alert` SSE channel
  - Auth: `TV_WEBHOOK_SECRET` env var as query param or header
  - Informational only — no order execution

### Phase TV-E: Dashboard + TradingView Side-by-Side Workflow
- [ ] **Document and validate the full manual trading workflow**
  - Left monitor: TradingView with Ruby Futures indicator + Tradovate demo connected
  - Right monitor: Python dashboard (focus mode, risk strip, swing signals)
  - Pre-market: dashboard daily bias + Grok brief → informs TV watchlist
  - During session: TV draws engine levels (via `request.seed()`) → execute manually in Tradovate demo
  - Dashboard tracks signals + updates risk strip
  - Zero dependency on NinjaTrader or Windows for live trading

---

## 🟡 Daily Strategy Layer (Completed)

### Phase 2A: Daily Bias Analyzer ✅
### Phase 2B: Daily Trade Plan Generator ✅
### Phase 2C: Swing Detector ✅

---

## 🟡 Dashboard Focus Mode (Completed)

### Phase 3A: Top-4 Asset Selection ✅
### Phase 3B: Dashboard Focus Mode ✅
### Phase 3C: Grok Integration for Daily Selection ✅ *(structured JSON → DailyPlan)*
### Phase 3D: Swing Action Buttons ✅

---

## 🟡 Live Risk-Aware Position Sizing (Completed)

### Phase 5A: Generalized Asset Model ✅
### Phase 5B: Real-Time Risk Budget Integration ✅
### Phase 5C: Dynamic Position Sizing on Focus Cards ✅
### Phase 5D: Live Position Overlay on Focus Cards ✅
### Phase 5E: Risk Dashboard Strip ✅

---

## 🟡 Post-Live: CNN Asset-Class Intelligence (v8+)

Deferred until the v7.1 model is live and profitable. These add significant value but don't block demo trading.

### Phase 7A: Hierarchical Asset Embedding
- [ ] Replace flat `asset_class_id` ordinal with a 4-dim class embedding + 8-dim per-asset embedding, trained end-to-end
- [ ] Embedding lookup table stored in `feature_contract.json`

### Phase 7B: Cross-Asset Correlation Features
- [ ] Rolling Pearson correlations with peer assets as CNN features: `primary_peer_corr`, `cross_class_corr`, `correlation_regime`
- [ ] Peer asset mapping in `asset_registry.py`: `Asset.peers` → `["Silver", "Copper"]` for Gold, etc.
- [ ] Pure computation in `lib/analysis/cross_asset.py`

### Phase 7C: Asset Fingerprint Analysis
- [ ] `lib/analysis/asset_fingerprint.py` — per-asset daily profile vector: typical daily range, session concentration, breakout follow-through rate, mean-reversion tendency (Hurst), volume profile shape, overnight gap tendency
- [ ] Dashboard: "Asset DNA" radar chart per focused asset

### Phase 7D: Correlation Anomaly Detection
- [ ] Rolling correlation matrix across all 10 core assets (updated every 5 min)
- [ ] Compare 30-bar vs 200-bar baseline → anomaly score per pair
- [ ] Publish `engine:correlation_anomalies` → dashboard heatmap panel

---

## 🟡 Post-Live: Per-Asset Training + Knowledge Distillation (v8+ Champion Model)

Train one model per asset, distill into a single champion `.pt`. Gives ~95% of per-asset accuracy at normal inference speed with no multi-model complexity.

```
Asset 1 (MGC) → train → best_mgc.pt  ─┐
Asset 2 (MNQ) → train → best_mnq.pt  ─┤
Asset 3 (MES) → train → best_mes.pt  ─┼→ Distill → champ_combined.pt
Asset N (...)  → train → best_xxx.pt  ─┘
```

### Phase 8A: Per-Asset Training Loop (`train_per_asset.py`)
- [ ] Assets: `['MGC', 'MNQ', 'MES', 'MYM', 'M2K', 'MBT', 'MET']`
- [ ] `epochs=60`, `patience=12`, `min_accuracy=0.75` gate per asset
- [ ] Write `models/per_asset/asset_results.json` manifest

### Phase 8B: Knowledge Distillation (`distill_combined.py`)
- [ ] `DistillationTrainer`: load all teacher `.pt` files (frozen), student = same architecture
- [ ] `temperature=4.0`, `alpha=0.7` (70% KL divergence + 30% cross-entropy)
- [ ] Qualified teachers only: `min_teacher_accuracy=0.75`
- [ ] Save best student to `models/champ_combined.pt`

### Phase 8C: Master Orchestrator (`run_full_pipeline.py`)
- [ ] Single script: per-asset training → rank + filter teachers → distill → promote
- [ ] Write `models/pipeline_summary.json`

---

## 🟢 Low Priority — Scaling & Copy Trading

### PickMyTrade + Account Scaling
- [ ] Sign up for PickMyTrade, test TradingView → Tradovate copy on a single Apex eval account
  - Verify webhook latency (TV alert → fill) for intraday futures
  - Test quantity multiplier config for different account sizes
  - Confirm all 20 Apex accounts can be connected simultaneously
- [ ] Configure TV alerts/webhooks to trigger PickMyTrade on manual entries
- [ ] Scale TPT to 5 accounts (pass eval on each, connect via PickMyTrade)
- [ ] Scale Apex to 20 accounts progressively

### Kraken Spot Portfolio Management (Phase 6)
- [ ] `lib/strategies/crypto/portfolio_manager.py` — target % allocations, rebalance logic
- [ ] Add `add_order()` and `cancel_order()` to `KrakenDataProvider`
- [ ] `CryptoPortfolioConfig`: target allocations, 5% rebalance threshold, 4h cooldown, DCA mode
- [ ] Dashboard: Kraken portfolio card with allocations vs targets, P&L, rebalance status
- [ ] Gated behind `ENABLE_KRAKEN_TRADING=1` env var

### Crypto Momentum Wiring
- [ ] Wire `crypto_momentum_score` into engine live scoring pipeline (currently computed, not fed into decisions)
- [ ] Show crypto momentum indicator on focused asset cards

---

## Execution Order (Getting to Demo Live)

**Step 1 — Codebase cleanup (do before retraining to reduce noise):**
- Phase 1D — extract generic handler from `main.py` (~400 lines eliminated, lower bug surface)
- Phase 1A — merge BreakoutType enums (kills the mapping dict hell)
- Phase 1B — merge RangeConfig (one config per type, full stop)
- Phase 1C — unified RB detector (ORB is just another type)
- Phase 1G — create `lib/strategies/` package

**Step 2 — CNN v7.1 retrain (the milestone):**
- Phase 4C — generate dataset, train, gate, promote `breakout_cnn_best.pt`
- Deploy to Pi via `sync_models.sh`

**Step 3 — TradingView integration (live trading UI):**
- Phase TV-A — `signals.csv` GitHub publisher
- Phase TV-C — Ruby Futures core futures layer (pure price, zero delay)
- Phase TV-B — engine signal overlay via `request.seed()`
- Phase TV-D — TV → Python webhook endpoint
- Phase TV-E — end-to-end workflow test on Tradovate demo

**Step 4 — Go live on demo funds, observe, tune thresholds**

**Step 5 — Post-demo (when profitable/consistent):**
- Phase 7A–7D — CNN asset intelligence
- Phase 8A–8C — per-asset distillation
- Phase 6 — Kraken portfolio
- PickMyTrade → multi-account scaling

---

## 🗺️ System Logic Map — End-to-End Data & Signal Flow

### 1. Data Ingestion

```
External Sources
  ├─ Yahoo Finance (yfinance)    ← primary for CME futures (1m, 5m, 15m, daily)
  ├─ Kraken REST / WebSocket     ← crypto spot via kraken_client.py
  └─ MassiveAPI (massive_client) ← alternative / historical bars

         ↓
  lib/core/cache.py → get_data(ticker, interval, period)
         │  Fetches bars, caches in Redis
         │  Keys: engine:bars_1m:<TICKER>, engine:bars_15m:<TICKER>, engine:bars_daily:<TICKER>
         ↓
  lib/trading/engine.py → DashboardEngine
         │  _fetch_tf_safe() / _refresh_data() / _loop()
         ↓
  Redis (pub/sub + key-value) — central message bus
```

### 2. Engine Startup & Scheduler

```
src/lib/services/engine/main.py → main()
  ├─ Env: ACCOUNT_SIZE, ENGINE_INTERVAL, ENGINE_PERIOD
  ├─ Creates DashboardEngine, ScheduleManager, RiskManager, PositionManager, ModelWatcher
  └─ Main loop:
       ├─ scheduler.get_pending_actions()   ← time-of-day aware
       ├─ _check_redis_commands()           ← dashboard-triggered overrides
       ├─ Execute pending actions via dispatch table
       ├─ _handle_update_positions()        ← bracket/trailing stop updates
       ├─ _tick_live_risk_publisher()       ← publish LiveRiskState every loop
       └─ _publish_engine_status()          ← push state to Redis for web UI

Session Modes (Eastern Time):
  EVENING     18:00–00:00  →  CME, Sydney, Tokyo, Shanghai ORB sessions
  PRE_MARKET  00:00–03:00  →  Daily focus, Grok brief, generate_daily_plan()
  ACTIVE      03:00–12:00  →  Frankfurt, London, London-NY, US ORB + all 13 breakout types
  OFF_HOURS   12:00–18:00  →  Backfill, training, optimization, daily report
```

### 3. Daily Focus Computation

```
PRE_MARKET (00:00–03:00 ET) → generate_daily_plan()
  │
  ├─ compute_all_daily_biases()       ← 6-component scoring per asset
  ├─ Grok macro brief (optional)      ← if XAI_API_KEY set
  ├─ select_daily_focus_assets()      ← 5-factor composite ranking (0-100)
  │    signal quality 30%, ATR opportunity 25%, RB density 20%, session fit 15%, catalyst 10%
  ├─ _build_swing_candidate()         ← wider SL/TP (1.75×/2.5×/4×/5.5× ATR)
  └─ DailyPlan.publish_to_redis()     ← engine:daily_plan, engine:focus_assets (18h TTL)
```

### 4. Breakout Detection (13 Types, 10 Sessions)

```
CHECK_ORB_* / CHECK_PDR / CHECK_IB / CHECK_CONSOLIDATION / CHECK_BREAKOUT_MULTI
  │
  ├─ Fetch 1m bars from Redis cache
  ├─ detect_range_breakout(bars, symbol, config)
  │    └─ _build_*_range() → _scan_for_breakout() → BreakoutResult
  │
  ├─ apply_all_filters() ← NR7, premarket, session window, lunch, MTF bias, VWAP
  │
  │  IF passed:
  │    ├─ predict_breakout(image, tabular_28, session_key)  ← CNN inference
  │    │    threshold per session (us:0.82 → sydney:0.72)
  │    │
  │    │  IF cnn_signal:
  │    │    ├─ RiskManager.can_enter_trade()
  │    │    ├─ PositionManager.process_signal()  ← bracket, P&L tracking (informational)
  │    │    ├─ signals_publisher.write_signal()  ← append to signals.csv → GitHub push
  │    │    ├─ publish_breakout_result()          ← Redis pub/sub → dashboard SSE
  │    │    └─ alerts.send_signal()               ← push notification
```

### 5. CNN Inference (Python)

```
predict_breakout(image_path, tabular_28, session_key)
  │
  ├─ Image branch: chart_renderer_parity.py → 224×224 Ruby-style chart snapshot
  │    → ImageNet normalisation → (1, 3, 224, 224) tensor
  │
  ├─ Tabular branch (28 features, v7.1 contract):
  │    _normalise_tabular_for_inference(features) → (1, 28) float tensor
  │    [0-17]  v6 features (quality, volume, ATR, CVD, direction, session, etc.)
  │    [18-23] v7 daily features (bias direction/confidence, prior day pattern,
  │             weekly range position, monthly trend, crypto momentum)
  │    [24-27] v7.1 sub-features (breakout type category, session overlap,
  │             ATR trend, volume trend)
  │
  ├─ Forward pass:
  │    EfficientNetV2-S(image) → (1, 1280)
  │    tabular_head(tabular)   → (1, 32)
  │    classifier(combined)    → (1, 2) → softmax → P(clean breakout)
  │
  └─ Returns: { prob, signal, confidence, threshold }
       signal = True if prob ≥ session threshold
```

### 6. Live Risk State

```
LiveRiskPublisher (ticked every engine loop, force-publish on position change)
  │
  ├─ compute_live_risk(risk_manager, position_manager)
  │    → LiveRiskState: daily_pnl, open_positions, remaining_risk_budget,
  │                     total_unrealized_pnl, margin_used, can_trade, block_reason
  │
  ├─ Publish to Redis: engine:live_risk
  ├─ SSE channel: dashboard:live_risk → risk strip updates every 5s
  └─ Focus cards: dual micro/regular sizing reflects remaining_risk_budget
```

### 7. Dashboard → TradingView Signal Flow

```
Engine fires CNN-gated signal
  │
  ├─ signals_publisher.append_and_push(signal)
  │    → signals.csv committed to nuniesmith/futures-signals (GitHub API)
  │
  └─ TradingView ruby_futures.pine (request.seed(), 2-5 min lag)
       → Parses signals.csv, matches current chart symbol
       → Draws entry/stop/TP lines + CNN label + dual contract sizing
       → Trader sees the level, decides manually whether to execute in Tradovate
```

### 8. Training Pipeline

```
trainer_server.py → _run_training_pipeline(TrainRequest)
  │
  ├─ dataset_generator.py → generate_dataset(symbols, days_back, config)
  │    For each of 25 symbols × 13 types × 9 sessions:
  │      ├─ load_bars() ← DataResolver (Redis → Postgres → Massive/Kraken)
  │      ├─ rb_simulator.py → bracket replay → good/bad labels
  │      ├─ chart_renderer_parity.py → 224×224 PNG per sample
  │      └─ _build_row() → 28 tabular features
  │
  ├─ split_dataset(85/15 stratified)
  ├─ train_model(epochs, batch_size, lr)
  │    Phase 1: freeze CNN backbone, train tabular head + classifier
  │    Phase 2: unfreeze, fine-tune everything at lower LR
  ├─ evaluate_model() → acc / prec / rec
  ├─ Gate check → promote to breakout_cnn_best.pt
  └─ ModelWatcher detects new .pt → engine hot-reloads
```

### 9. Infrastructure

```
Docker Compose (3 containers):
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   :engine    │  │    :web      │  │   :trainer   │
  │  main.py     │  │  FastAPI     │  │  FastAPI     │
  │  scheduler   │  │  dashboard   │  │  dataset gen │
  │  risk mgr    │  │  focus cards │  │  CNN train   │
  │  position mgr│  │  settings    │  │  promote .pt │
  │  all handlers│  │  risk strip  │  │              │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └──────────────────┼─────────────────┘
                            ↓
                      ┌──────────┐
                      │  Redis   │
                      └──────────┘
                      ┌──────────┐
                      │ Postgres │  (audit trail: risk_events, orb_events, journal)
                      └──────────┘

Tailscale mesh:
  Pi       → engine + web (always on, 24/7 scheduling)
  GPU rig  → trainer (on-demand)
```

---

## Completed

### Daily Bias Analyzer (`src/lib/strategies/daily/bias_analyzer.py`)
- [x] `BiasDirection`, `CandlePattern`, `KeyLevels`, `DailyBias` dataclasses
- [x] `compute_daily_bias()` — 6-component weighted scoring (candle 25%, weekly 20%, monthly 25%, volume 10%, gap 10%, ATR 10%)
- [x] `compute_all_daily_biases()`, `rank_assets_by_conviction()`, CNN feature helpers

### Daily Plan Generator + Focus Asset Selection
- [x] `DailyPlan`, `SwingCandidate`, `ScalpFocusAsset` dataclasses with full serialisation
- [x] `generate_daily_plan()` orchestrator, `select_daily_focus_assets()` 5-factor composite ranking
- [x] `DailyPlan.publish_to_redis()` / `load_from_redis()` — 18h TTL
- [x] `get_daily_plan_focus_assets()` in `focus.py`, `compute_daily_focus(use_daily_plan=True)`
- [x] 66 tests passing

### Dashboard Focus Mode
- [x] `_render_focus_mode_grid()` — tiered layout: scalp focus (prominent), swing cards (amber), background collapse
- [x] `_render_daily_plan_header()` — Grok morning brief card + focus chip strip
- [x] `_render_why_these_assets()` — collapsible score breakdown table with mini score bars
- [x] `_render_swing_card()` — amber-bordered with TP1/TP2/TP3, entry style chips, confidence badge
- [x] SSE `daily-plan-update` listener, `GET /api/daily-plan/html` endpoint
- [x] 82 tests passing

### Swing Detector (`src/lib/strategies/daily/swing_detector.py`)
- [x] Three entry detectors: `detect_pullback_entry()`, `detect_breakout_entry()`, `detect_gap_continuation()`
- [x] Exit engine `evaluate_swing_exits()`: stop loss → TP1 scale 50% → TP2 close → EMA-21 trail → time stop
- [x] State machine: WATCHING → ENTRY_READY → ACTIVE → TP1_HIT → TRAILING → CLOSED
- [x] Redis publish/load, `engine:swing_signals`, `engine:swing_states`
- [x] 150 tests passing

### Swing Action Buttons
- [x] `swing_actions.py` router — 10 endpoints: accept, ignore, close, stop-to-BE, update-stop, status
- [x] HTMX fragments (success/error toasts + updated buttons) — no full page reload
- [x] Signal lifecycle: detect → pending → accept/ignore → active → manage → close → archive
- [x] SSE `swing-update` listener, structured action publishing via `_publish_swing_action()`
- [x] 88 tests passing

### Grok Structured Daily-Plan Analysis
- [x] `grok_helper.py` — parsed JSON → `DailyPlan.market_context`, dashboard rendering
- [x] 77 tests passing

### Generalized Asset Model (`src/lib/core/asset_registry.py`)
- [x] `Asset`, `ContractVariant`, `AssetClass`, `ASSET_REGISTRY`
- [x] `dual_sizing()`, `compute_position_size()`, `get_asset_by_ticker()`, `get_asset_group()`
- [x] Replaces split `MICRO_CONTRACT_SPECS` / `FULL_CONTRACT_SPECS`

### Live Risk Integration
- [x] `LiveRiskState`, `LiveRiskPublisher`, `compute_live_risk()` — `src/lib/services/engine/live_risk.py`
- [x] API endpoints: `/api/live-risk`, `/api/live-risk/html`, `/api/live-risk/summary`
- [x] Force-publish on position changes (1-2s latency)

### Dynamic Position Sizing + Live Position Overlays
- [x] `_compute_dual_sizing()` — micro + regular side-by-side on focus cards
- [x] `_render_live_position_overlay()` — LIVE badge, P&L, R-multiple, bracket progress bar
- [x] Risk strip (`get_live_risk_html()`) — health-coloured, HTMX polling + SSE

### CNN v7.1 Feature Contract
- [x] Features [18]–[23]: daily strategy features in `breakout_cnn.py` + `dataset_generator.py`
- [x] Features [24]–[27]: sub-features (breakout type category, session overlap, ATR trend, volume trend)
- [x] `feature_contract.json` updated to v7.1 (28 features)
- [x] `_normalise_tabular_for_inference()` v6→v7→v7.1 backward-compat padding

### CNN Model — v6 Champion
- [x] 22-symbol training, 13 types, 9 sessions, 25 epochs
- [x] **87.1% accuracy**, 87.15% precision, 87.27% recall — all gates passed
- [x] `breakout_cnn_best.pt` promoted, `feature_contract.json` v6 generated

### Unified Data Resolver (`src/lib/services/data/resolver.py`)
- [x] `DataResolver` — Redis → Postgres → Massive/Kraken API three-tier resolution
- [x] `resolve()`, `resolve_batch()`, `resolve_with_meta()`

### Kraken Training Pipeline Integration
- [x] `dataset_generator.py` — Kraken routing, `_is_kraken_symbol()`, `_load_bars_from_kraken()`
- [x] 25 total training symbols: 22 CME micros + BTC, ETH, SOL

### Web UI — Trading / Review Mode
- [x] `⚡ Trading` / `🔍 Review` pill toggle — auto-detects from ET hour
- [x] CSS visibility gates: review panels hidden in trading mode
- [x] Decimal precision fix for forex tickers (5–7dp)

### Trainer UI Separation
- [x] `trainer_server.py` HTML endpoint removed — pure API server
- [x] `src/lib/services/data/api/trainer.py` — full dashboard page at `GET /trainer`

### Web UI — Settings Page
- [x] `settings.py` — 5 tabbed sections: Engine, Services, Features, Risk & Trading, API Keys
- [x] All settings persisted to Redis via `settings:overrides`

### SSE Swing + TV Alert Wiring
- [x] `swing-update` SSE listener — auto-refreshes focus grid, parses action metadata for market events feed
- [x] `tv-alert` SSE listener — shows TradingView webhook alerts in market events feed
- [x] `dashboard:tv_alert` Redis PubSub channel handler in `sse.py`
