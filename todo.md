# futures — TODO

> **Single repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:data` · `:engine` · `:web` · `:trainer`
> **Infrastructure**: Ubuntu Server (data + engine + web + monitoring), Home GPU rig (trainer)

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
│   │   │   └── data/      # FastAPI REST + SSE API (port 8000) — bars, journal, positions, kraken, sse, …
│   │   └── integrations/  # kraken_client, massive_client, grok_helper
│   ├── entrypoints/
│   │   ├── data/main.py   # python -m entrypoints.data.main  → lib.services.data.main:app
│   │   ├── engine/main.py # python -m entrypoints.engine.main
│   │   ├── web/main.py    # python -m entrypoints.web.main
│   │   └── training/main.py
│   └── pine/
│       └── ruby_futures.pine   # TradingView Pine Script indicator
├── models/                # champion .pt, feature_contract.json (Git LFS)
├── scripts/
│   ├── sync_models.sh     # pull .pt from repo → restart engine
│   └── …
├── config/                # Prometheus, Grafana, Alertmanager
├── docker/
│   ├── data/              # Dockerfile + entrypoint.sh  (:data image)
│   ├── engine/            # Dockerfile + entrypoint.sh  (:engine image)
│   ├── web/               # Dockerfile + entrypoint.sh  (:web image)
│   ├── trainer/           # Dockerfile                  (:trainer image)
│   └── monitoring/        # prometheus/ + grafana/
└── docker-compose.yml
```

---

## 🎯 Goal — Python Co-Pilot + Manual Live Trading

The system is a **manual trading co-pilot**. It informs entries, it doesn't execute them. The live trading workflow is Python dashboard + TradingView side-by-side — no Windows, no NinjaTrader required.

```
Python Engine (Ubuntu Server)
    ├── Computes: daily bias, ORB levels, PDR, IB, CNN signals, entry/stop/TP
    ├── Dashboard: focus mode cards, risk strip, swing actions, Grok brief
    ├── Reddit Sentiment: live pulse from futures-relevant subreddits

Python Dashboard 
    └── Real-time CNN probabilities, risk strip, focus cards, swing signals, sentiment
```

> **TradingView Live Positions — DEFERRED**: Sending live positions back from TradingView is not straightforward and not worth the complexity right now. The Ruby Pine Script indicator is used as a **reference overlay for live market movements** only. All actual trading is manual via Tradovate, informed by the Python dashboard. A Tradovate JavaScript bridge is the preferred path for future automation (see Phase TBRIDGE below).

**Two-stage scaling plan:**
- **Stage 1 — TPT**: 5 × $150K accounts = $750K total buying power.
- **Stage 2 — Apex**: 20 × $300K accounts = ~$6M total buying power.
- **Copy layer**: Tradovate JS bridge (1st account) → PickMyTrade webhook → all remaining accounts simultaneously. Own-accounts-only copy trading is explicitly allowed by both Apex and TPT.

**Milestone before going live on demo funds:**
1. ~~Codebase cleanup (Phase 1 refactors — reduce complexity before retraining)~~ ✅
2. CNN v8 retrain (37 features + embeddings, targeting ≥89% accuracy)
3. Dashboard tiled alongside TradingView for manual trading
4. Full workflow test on Tradovate demo

---

## Current State

- **Monorepo**: All source — engine, web, data, trainer, lib, Pine Script, deploy scripts.
- **Models**: `models/breakout_cnn_best.pt` + `feature_contract.json` committed (Git LFS). Latest champion: **87.1% accuracy**, 87.15% precision, 87.27% recall, 25 epochs, v6 18-feature. Retrain to v8 (37-feature) is the next major milestone.
- **Docker**: `:data` (FastAPI REST + SSE, port 8050), `:engine` (background compute worker, no HTTP port), `:web` (HTMX dashboard reverse-proxy, port 8180), `:trainer` (GPU training server, port 8200). All run on Ubuntu Server; trainer also runs on home laptop GPU rig. Data service is the single HTTP entry point for all dashboard and API traffic — engine publishes computed state to Redis for data to serve.
- **Service split**: `data` and `engine` were previously a single `:engine` image running two processes (uvicorn + engine worker). They are now fully separate containers with clean responsibilities: data owns all REST/SSE/bar-cache/Kraken-feed, engine owns all computation/scheduling/risk.
- **Entrypoints**: `src/entrypoints/{data,engine,web,training}/main.py` — thin wrappers that import and call `main()` from the corresponding `lib.services.*` module. Dockerfiles use `python -m entrypoints.<service>.main`.
- **Feature Contract**: v8, 37 tabular features + hierarchical asset embeddings. `models/feature_contract.json` is the canonical source. Expanded from v6 (18) → v7.1 (28) → v8 (37) with cross-asset correlation features, asset fingerprint features, and learned embeddings.
- **CNN Model**: EfficientNetV2-S + wider tabular head + asset embeddings. Auto-detects tabular dimension from checkpoint metadata at load time. Training pipeline: generate dataset → train → evaluate → gate check (≥89% acc, ≥87% prec, ≥84% rec) → promote. Python `_normalise_tabular_for_inference()` handles v6→v7→v7.1→v8 backward-compat padding so older models work with the new code.
- **Breakout Types**: 13 — ORB, PrevDay, InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci. Fully wired in engine detection, training, dataset generator, CNN tabular vector, chart renderer.
- **Position Manager**: `position_manager.py` — always-in 1-lot micro positions, reversal gates (CNN ≥ 0.85, MTF ≥ 0.60, 30min cooldown), Redis persistence, `OrderCommand` emitter (informational — feeds dashboard, not live execution).
- **Dashboard**: HTMX + FastAPI — live signals, 13 breakout type filter pills, 9 session tabs, MTF score column, trade journal, Kraken crypto chart + correlation panel, Grok AI analyst, CNN dataset preview, positions panel, flatten/cancel buttons, market regime (HMM), performance panel, volume profile, equity curve, asset focus cards with entry/stop/TP levels, focus mode, risk strip, swing action buttons. Broker-agnostic position bridge (TradingView/Tradovate ready).
- **Kraken Integration**: `KrakenDataProvider` REST + `KrakenFeedManager` WebSocket feed. 9 crypto pairs streaming. Runs inside the data container (`ENABLE_KRAKEN_CRYPTO=1`).
- **Massive Integration**: `MassiveDataProvider` REST + `MassiveFeedManager` WebSocket. Front-month resolution, primary bars source for training.
- **Data Service**: Unified data layer — Redis cache → Postgres → external APIs. Startup cache warming from Postgres (7 days). Exposes `/bars/{symbol}` (auto-fill), `/bars/bulk`, `/sse/dashboard`, `/kraken/*`, and all dashboard HTMX endpoints.
- **Training**: `trainer_server.py` FastAPI (port 8200). `dataset_generator.py` covers all 13 types + 9 sessions + Kraken. Full pipeline: generate → split (85/15 stratified) → train → evaluate → gate → promote. Fetches bars from `ENGINE_DATA_URL=http://data:8000`.
- **CI/CD**: Lint → Test → Build & push **6 Docker images** (data, engine, web, trainer, prometheus, grafana) → Deploy data + engine + web to Ubuntu Server via Tailscale SSH → Deploy trainer to home laptop via Tailscale SSH → Health checks → Discord notifications. `:data` is the new `:latest` alias.
- **Tailscale**: Ubuntu Server (Docker) at `100.122.184.58`, all services communicate over Tailscale mesh. HTTP only (no domain/TLS needed for local mesh). Trainer laptop (CUDA GPU) at `100.113.72.63:8200`.
- **NinjaTrader removed**: All NT8 Bridge code, deploy scripts, and C# patchers removed from Python codebase. Position management is now broker-agnostic (`positions.py`). The `src/ninja/` and `src/pine/` directories contain C#/Pine source for reference but are not part of the Python runtime. TradingView integration (`tradingview.py`) is the intended live trading path.

---

## Architecture Issues Identified (Pre-Refactor)

### Triple duplication of breakout types & config
- `lib/core/breakout_types.py` — canonical `BreakoutType` (IntEnum) + `RangeConfig` (CNN/training source)
- `lib/services/engine/breakout.py` — **second** `BreakoutType` (StrEnum) + **second** `RangeConfig` (engine runtime)
- `lib/services/engine/orb.py` — **third** dataclass `ORBSession` with its own ATR params

Mapping dicts (`_ENGINE_TO_TRAINING`, `_TRAINING_TO_ENGINE`) existed purely to convert between the two enums. These have been removed.

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

### Phase 1A: Merge BreakoutType Enums → Single Source of Truth ✅
- [x] Eliminate the engine `StrEnum` in `services/engine/breakout.py` — use `core/breakout_types.BreakoutType` (IntEnum) everywhere
  - Removed `_ENGINE_TO_TRAINING` / `_TRAINING_TO_ENGINE` mapping dicts
  - Removed `to_training_type()` / `from_training_type()` / `breakout_type_ordinal()` bridge functions
  - All engine callers import from `lib.core.breakout_types`
  - `BreakoutResult.to_dict()` uses `.name` for JSON serialisation and `.value` for ordinals
  - Short-name aliases (`"PDR"` → `PrevDay`, etc.) retained in `breakout.py` for backward compat

### Phase 1B: Merge `RangeConfig` → Single Dataclass ✅
- [x] Unify the two `RangeConfig` dataclasses into `core/breakout_types.py`
  - Detection-threshold fields (ATR mult, body ratio, range caps, squeeze params) merged INTO the core `RangeConfig`
  - All 13 `_*_CONFIG` registry entries have detection fields
  - Engine-side `RangeConfig` eliminated — `get_range_config(BreakoutType.ORB)` returns everything
  - `DEFAULT_CONFIGS` in `breakout.py` delegates to `get_range_config()`

### Phase 1C: Merge ORB Detection into Unified RB Detector ✅
- [x] `detect_range_breakout(config=ORB_CONFIG)` handles all 13 types including ORB
  - All `_build_*_range()` functions extracted into `strategies/rb/range_builders.py`
  - Single `detect_range_breakout(bars, symbol, config)` in `strategies/rb/detector.py`
  - `BreakoutResult` covers all types (ORB fields mapped: `range_high`↔`or_high`, etc.)
  - Single `compute_atr()` in `strategies/rb/range_builders.py` (canonical implementation)
- [x] `_handle_check_orb()` (~800 lines) replaced by `handle_orb_check()` delegation
  - Quality filters pipeline extracted to `handlers.run_quality_filters()`
  - CNN inference pipeline extracted to `handlers.run_cnn_inference()`
  - CNN tabular feature construction extracted to `handlers.build_cnn_tabular_features()`
  - Session-aware filter windows extracted to `handlers.get_filter_windows_for_session()`
  - All 11 ORB session handlers (`_handle_check_orb_london`, etc.) are now one-liners
  - ORB-specific Redis publishing handled via `_publish_orb_result()` shim for backward compat
- [ ] **Remaining (post-retrain):** `orb.py` still has `detect_opening_range_breakout()` and `ORBResult` — can be deprecated once v7.1 retrain validates the unified path

### Phase 1D: Extract Generic Handler Pipeline from `main.py` ✅
- [x] One handler function for all 13 breakout types — `handle_breakout_check()` in `handlers.py`
  - `_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` are one-liners
  - Shared helpers extracted: `fetch_bars_1m`, `get_htf_bars`, `run_mtf_on_result`, `persist_breakout_result`, `publish_breakout_result`, `send_breakout_alert`
  - `handle_breakout_multi()` runs multiple types in parallel via ThreadPoolExecutor
  - `enable_filters=True` / `enable_cnn=True` flags bring filter+CNN support to any type

### Phase 1E: Rename `orb_filters.py` → `breakout_filters.py` ✅
- [x] `ORBFilterResult` → `BreakoutFilterResult`. Backward-compat shim in place.

### Phase 1F: Rename `orb_simulator.py` → `rb_simulator.py` ✅
- [x] `simulate_orb_outcome` → `simulate_rb_outcome`. Shim in place.

### Phase 1G: Create `lib/strategies/` Package ✅
- [x] Clean separation of strategy code from infrastructure
  - `lib/strategies/rb/` — Range Breakout scalping system (detector, range_builders, publisher)
  - `lib/trading/costs.py` → `lib/strategies/costs.py` (shim in place)
  - `lib/trading/strategies.py` → `lib/strategies/strategy_defs.py` (shim in place)
  - `lib/trading/engine.py` → `lib/strategies/backtesting.py` (shim in place)
  - `RBSession` alias added in `multi_session.py`, exported from `lib.core`
- [ ] **Remaining (non-blocking):** Bulk-rename `ORBSession` → `RBSession` in callers (non-breaking — alias works)

### Phase 1H: Pre-Retrain Test Cleanup ✅
- [x] All unit tests passing (2552 passed, 0 failed, 1 skipped)
- [x] Fixed all `BreakoutType.UPPER_CASE` → `BreakoutType.PascalCase` enum name mismatches in `test_breakout_types.py` (76 occurrences)
- [x] Updated feature contract test expectations from v6 (18 features) to v7.1 (28 features):
  - `test_breakout_types.py::TestFeatureContractGeneration` — version 7, 28 tabular features, v7/v7.1 feature spot-checks
  - `test_kraken_training_pipeline.py::TestFeatureContract` — version 7.1 (on-disk JSON), 28 features
  - `test_kraken_training_pipeline.py::TestTabularFeatureShape` — 28 features with all v7/v7.1 entries
- [x] Fixed `lib/trading/strategies.py` backward-compat shim — added missing re-exports:
  - `_atr`, `_ema` (used by `test_strategies_ict.py`)
  - `_ict_confluence_array`, `_compute_ict_confluence` (used by ICT confluence tests)
- [x] Fixed `positions.py` `clear_positions()` cache isolation bug — `REDIS_AVAILABLE` was a stale name binding from import time; now reads from `lib.core.cache` module at call time so test fixtures that toggle Redis mode are respected

---

## 🔴 CNN v8 Champion Retrain (THE Milestone — One Big Train)

> **Strategy change**: Skip standalone v7.1 retrain. Roll all v7/v7.1 features (already coded) + v8 architecture upgrades into a single champion training run. One long GPU session, best possible model before going live. 3-week runway until TPT $150K account.

### Phase 4A: New Features from Daily Strategy Layer ✅
- [x] 6 new v7 features (features [18]–[23]): `daily_bias_direction`, `daily_bias_confidence`, `prior_day_pattern`, `weekly_range_position`, `monthly_trend_score`, `crypto_momentum_score`
- [x] `feature_contract.json` updated to v7.1 (28 features)
- [x] `dataset_generator.py` `_build_row()` computes all 6 features with neutral fallbacks
- [x] `_normalise_tabular_for_inference()` handles v6→v7→v7.1 backward-compat padding

### Phase 4B: Sub-Features and Richer Encoding ✅
- [x] 4 sub-features (features [24]–[27]): `breakout_type_category`, `session_overlap_flag`, `atr_trend`, `volume_trend`
- [x] All 4 computed in `dataset_generator.py` `_build_row()` with neutral fallbacks

---

### Phase v8-A: Hierarchical Asset Embedding (replaces flat `asset_class_id`) ✅
> Replaces features [13] `asset_class_id` and [15] `asset_volatility_class` with learned embeddings. The model learns *what each asset behaves like* rather than us hand-coding ordinals.

- [x] **Add embedding layers to `HybridBreakoutCNN`**
  - `nn.Embedding(num_classes=5, embedding_dim=4)` — 5 asset classes (equity index, FX, metals/energy, bonds/ags, crypto)
  - `nn.Embedding(num_assets=25, embedding_dim=8)` — one per tradeable symbol
  - Concatenate 4+8=12-dim embedding with existing tabular head output (replaces 2 flat features, net +10 dims)
  - Train end-to-end — embeddings learn asset personality from breakout outcomes
- [x] **Update `feature_contract.json`** — add `asset_class_lookup` and `asset_id_lookup` tables mapping symbol→index
- [x] **Update `_build_row()` and `BreakoutDataset.__getitem__()`** — pass `asset_class_idx` and `asset_idx` as integer IDs (not floats)
- [x] **Update `_normalise_tabular_for_inference()`** — route embedding IDs separately from the float tabular vector
- [x] **Backward compat**: if checkpoint lacks embedding weights, fall back to flat `asset_class_id` + `asset_volatility_class` (v7.1 mode)

### Phase v8-B: Cross-Asset Correlation Features (+3 tabular features) ✅
> Adds market-regime context: is this asset moving with or against its peers? Correlation breakdowns are strong breakout-quality signals.

- [x] **`lib/analysis/cross_asset.py`** — pure computation module (already existed)
  - `rolling_peer_correlation(symbol, bars_1m, peer_bars, window=60)` → Pearson r with primary peer
  - `cross_class_correlation(symbol, bars_1m, class_index_bars, window=60)` → r with asset-class index
  - `correlation_regime(r_short=30, r_long=200)` → 0.0=decorrelated, 0.5=normal, 1.0=high-corr
- [x] **Peer mapping in `asset_registry.py`** — `Asset.peers` field: Gold→[Silver, Copper], MNQ→[MES, M2K], etc. (already existed)
- [x] **3 new features in `feature_contract.json` v8**: `primary_peer_corr` [28], `cross_class_corr` [29], `correlation_regime` [30]
- [x] **`_build_row()`** — compute from attached peer bars (neutral 0.5 fallback when peer data unavailable)
- [x] **`dataset_generator.py`** — load peer bars alongside primary bars during generation (implemented: `generate_dataset()` pre-loads peer bars via `_resolve_peer_tickers()`, builds `bars_by_ticker` dict, passes to `generate_dataset_for_symbol()` which attaches `_bars_by_ticker` to each `ORBSimResult`)

### Phase v8-C: Asset Fingerprint Features (+6 tabular features) ✅
> Per-asset "DNA" — the model learns that MGC mean-reverts but MNQ trends, without us hard-coding it.

- [x] **`lib/analysis/asset_fingerprint.py`** — compute per-asset daily profile (already existed)
  - `typical_daily_range_norm` — median daily range / ATR (is this asset wide or tight today?)
  - `session_concentration` — % of daily volume in primary session (concentrated vs distributed)
  - `breakout_follow_through` — historical win rate of breakouts for this asset (trailing 20-day)
  - `hurst_exponent` — mean-reversion tendency (H<0.5=mean-revert, H>0.5=trending)
  - `overnight_gap_tendency` — median overnight gap / ATR (gap-prone vs smooth)
  - `volume_profile_shape` — kurtosis of intraday volume distribution (spiky vs flat)
- [x] **6 new features in `feature_contract.json` v8**: features [31]–[36]
- [x] **`_build_row()`** — compute from daily bars + 1m bars with neutral fallbacks
- [ ] **Dashboard (low priority, post-train)**: "Asset DNA" radar chart on focus cards

### Phase v8-D: Architecture Upgrades to `HybridBreakoutCNN` ✅
> Better tabular head, attention fusion, and training recipe. These are code-only changes before the big train.

- [x] **Wider tabular head**: Linear(N→256) → BN → GELU → Dropout(0.3) → Linear(256→128) → BN → GELU → Linear(128→64)
  - Justification: tabular input grows from 18→37 features (+embeddings), needs more capacity
  - GELU instead of ReLU (smoother gradients, standard in modern architectures)
- [ ] **Cross-attention fusion** (optional, test on small dataset first):
  - Instead of `cat(img_features, tab_features)`, use a single cross-attention layer where tabular queries attend to image feature map
  - Fallback: keep concatenation if cross-attention doesn't lift accuracy
- [x] **Mixup augmentation**: α=0.2 mixup on tabular features during training (proven regulariser for tabular+image models)
- [x] **Label smoothing**: increase from 0.05 → 0.10 (more features = more confident model = more benefit from smoothing)
- [x] **Cosine warmup**: 5-epoch linear warmup before cosine decay (stabilises early training with embeddings)
- [x] **Gradient accumulation**: effective batch size 128 (2× accumulation steps with batch_size=64) for more stable gradients

### Phase v8-E: Training Recipe & Hyperparameters ✅
> Dial in the training config for maximum accuracy on the single long run.

- [ ] **Dataset generation** — all 25 symbols × 13 breakout types × 9 sessions × 120 days back
  - `max_samples_per_type_label=800` — prevent ORB from dominating (now default in `DatasetConfig`)
  - `max_samples_per_session_label=400` — balance overnight vs primary sessions (now default in `DatasetConfig`)
  - `DatasetConfig` defaults updated: `breakout_type="all"`, `orb_session="all"`, caps 800/400
  - Expected: ~50K–80K samples (vs ~20K in v6 run)
  - **Ready to run** — just needs `generate_dataset(symbols=ALL_25, days_back=120)`
- [x] **Training config** (defaults wired into `train_model()`):
  - `epochs=80`, `patience=15` (longer run, more patience — embeddings need time to converge)
  - `freeze_epochs=5` (freeze backbone longer since tabular head is now bigger + embeddings)
  - `batch_size=64`, gradient accumulation → effective 128
  - `lr=2e-4` (backbone), `lr=1e-3` (tabular head + embeddings) — separate param groups
  - `weight_decay=1e-4` (slightly higher for larger model)
- [x] **Stratified 85/15 split** — stratify by `(label, breakout_type, session)` triple for balanced eval
  - `split_dataset()` now builds `_strat_key = label + "__" + breakout_type + "__" + session`
  - Audit logging shows per-split breakdown of labels, breakout types, and sessions
- [x] **Gate check**: ≥89% acc, ≥87% prec, ≥84% rec — documented in `feature_contract.json` `v8_training_recipe.gate_check`

### Phase v8-F: Per-Asset Distillation → Single Champion `.pt` *(optional — try unified first)*
> Train per-asset specialist models, distill into one master. This is the final accuracy squeeze.

```
Step 1: Train per-asset specialists
  MGC → train (80 epochs) → best_mgc.pt  (gate: ≥75% acc)
  MNQ → train (80 epochs) → best_mnq.pt
  MES → train (80 epochs) → best_mes.pt
  ...7 core assets...

Step 2: Distill into champion
  All qualified teachers (≥75% acc) → DistillationTrainer → champ_v8.pt

Step 3: Compare
  champ_v8.pt vs best single v8 model → pick the winner
```

- [ ] **`scripts/train_per_asset.py`** — loop over `['MGC', 'MNQ', 'MES', 'MYM', 'M2K', 'MBT', 'MET']`
  - Each: generate asset-specific dataset → train → gate (≥75% acc) → save `models/per_asset/best_{symbol}.pt`
  - Write `models/per_asset/asset_results.json` manifest with per-asset metrics
- [ ] **`scripts/distill_combined.py`** — knowledge distillation
  - Load all qualified teacher `.pt` files (frozen)
  - Student = same `HybridBreakoutCNN` v8 architecture
  - `temperature=4.0`, `alpha=0.7` (70% KL divergence + 30% hard cross-entropy)
  - Save best student to `models/champ_v8_distilled.pt`
- [ ] **`scripts/run_full_pipeline.py`** — master orchestrator
  - Single command: generate → train unified → train per-asset → distill → compare → promote winner
  - Write `models/pipeline_summary.json` with all metrics + comparison

### Phase v8-G: Promote & Deploy
- [ ] **Promote** winner to `breakout_cnn_best.pt` + regenerate `feature_contract.json` v8
- [ ] **ONNX export** — `export_onnx_model()` with updated tabular + embedding inputs
- [ ] **Deploy to Ubuntu Server** via `sync_models.sh` → engine hot-reload via `ModelWatcher`
- [ ] **Smoke test** — run inference on 10 live breakouts, verify probabilities are sane
- [ ] **Update `_normalise_tabular_for_inference()`** — add v8 backward-compat padding (28→37 features + embedding IDs)

---

## 🔴 Immediate Fixes — Post-Cleanup (before training)

> Items discovered during NinjaTrader bridge removal and v8 code review.
> Must be fixed before starting the big training run.

### Python Model & Training Fixes
- [x] **Run full test suite** — `pytest src/tests/` → 2552 passed, 1 skipped. Fixed `test_bridge_trading.py` heartbeat cache key (`bridge_heartbeat` → `broker_heartbeat`) and `test_kraken_training_pipeline.py` feature count (28 → 37 for v8). Also fixed `positions.py` `_get_broker_url()` to derive localhost from heartbeat `listenerPort` when no explicit broker host is configured.
- [x] **`test_bridge_trading.py`** — updated 4 cache key references from `bridge_heartbeat` to `broker_heartbeat` to match renamed positions module. All 37 bridge trading tests pass.
- [x] **`_build_row()` peer bars** — implemented in `dataset_generator.py`: `generate_dataset()` now pre-loads peer bars via `_resolve_peer_tickers()` (reads `cross_asset.PEER_MAP`), builds a `bars_by_ticker` dict per symbol, and `generate_dataset_for_symbol()` attaches `_daily_bars`, `_bars_1m` (window-sliced), and `_bars_by_ticker` to each `ORBSimResult`. v8-B cross-asset and v8-C fingerprint features now compute real values instead of neutral 0.5 fallbacks.
- [ ] **Smoke-test training loop** — run a short 2-epoch training on a tiny synthetic dataset to verify v8 changes (mixup, grad accumulation, cosine warmup, separate LR groups, embedding layers) don't crash at runtime
- [ ] **`breakout_cnn.py` comment cleanup** — several comments still reference "NinjaTrader BreakoutStrategy", "OrbCnnPredictor.NormaliseTabular() in C#", "NT8 inference". Update to say "external consumers" or "TradingView" where appropriate (cosmetic, not blocking)
- [ ] **`chart_renderer.py` / `chart_renderer_parity.py` comment cleanup** — references to "Ruby NinjaTrader indicator" and "NT8 screen". Update to generic language (cosmetic, not blocking)
- [ ] **`breakout_types.py` / `multi_session.py` comment cleanup** — references to "C# NinjaTrader consumer". Update (cosmetic, not blocking)

### Dashboard & API Fixes
- [ ] **Dashboard `_get_bridge_info()` callers** — function still named `_get_bridge_info()` and passes data as `bridge_connected`/`bridge_age_seconds`/`bridge_account` params to `_render_positions_panel()`. Consider renaming to `_get_broker_info()` and updating param names (functional — backward compat aliases keep it working, but confusing)
- [ ] **Dashboard SSE event name** — `bridge-status` event listener in JS is still named `bridge-status`. The SSE publisher must match this name. Verify the engine/SSE publisher uses the same event name, or update both together
- [ ] **Dashboard health bar** — `/api/nt8/health/html` endpoint path still says "nt8". Works fine (route is stable), but a future rename to `/api/health/html` would be cleaner. Low priority.
- [ ] **`positions.py` duplicate route** — both `get_bridge_status()` and `get_broker_status()` are registered as GET endpoints. FastAPI may warn about duplicate routes. Verify no conflict (the legacy `/bridge_status` alias just calls `get_broker_status()`)
- [ ] **`positions.py` duplicate route** — same for `get_bridge_orders()` / `get_broker_orders()`. Check FastAPI doesn't reject duplicate path registrations.

### CI/CD & Infrastructure
- [ ] **CI/CD trainer target IP** — verify `TRAINER_TAILSCALE_IP` secret is set to `100.113.72.63` (home laptop with CUDA)
- [ ] **CI/CD cloud server** — verify `PROD_TAILSCALE_IP` secret points to the Ubuntu Server (was previously Pi)
- [ ] **Docker Compose `web` service** — `TRAINER_SERVICE_URL` is hardcoded to `http://100.113.72.63:8200`. This is correct but fragile — consider using a Tailscale hostname or env var
- [ ] **Remove `scripts/sync_models.sh`** reference check — ensure the script still works for Ubuntu Server (originally written for Pi)

### Dataset Generation (blocking for training)
- [ ] **Generate v8 dataset** — all 25 symbols × 13 breakout types × 9 sessions × 120 days back
  - `max_samples_per_type_label=800` — prevent ORB from dominating
  - `max_samples_per_session_label=400` — balance overnight vs primary sessions
  - Expected: ~50K–80K samples (vs ~20K in v6 run)
- [ ] **Stratified 85/15 split** — stratify by `(label, breakout_type, session)` triple for balanced eval

---

## 🔴 Lightweight Charts — Dedicated Chart UI *(Phase CHARTS)*

> Replace the placeholder `/charts` page (currently 4 static HTMX divs) with a full interactive candlestick chart UI using TradingView's [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) JS library directly in the browser.
>
> **Why not the `lightweight-charts-python` PyPI package?** Reviewed — it is a desktop/notebook wrapper that spawns a `pywebview` native OS window. It requires a display, has no HTTP endpoint, and cannot run in Docker. The underlying JS library it wraps is what we want, and we already have everything needed to drive it ourselves: `/bars/{symbol}` (auto-fill OHLCV), `/sse/dashboard` (live Redis pub/sub), and `/kraken/ohlcv/{pair}`.

### Phase CHARTS-A: Bars SSE Endpoint
- [ ] **`src/lib/services/data/api/sse.py`** — add `GET /sse/bars/{symbol}` endpoint
  - Subscribe to Redis pub/sub channel `engine:bars_1m:{symbol}`
  - Stream `text/event-stream` events: `data: {"time": <unix>, "open": …, "high": …, "low": …, "close": …, "volume": …}`
  - Engine already publishes 1m bars to Redis on each bar close — no engine changes needed
  - Fallback: if no Redis event within 60s, re-fetch latest bar from `/bars/{symbol}?days_back=1` and stream it

### Phase CHARTS-B: Chart Data Shaping Endpoint
- [ ] **`src/lib/services/data/api/charts.py`** — new router, registered at `/api/charts`
  - `GET /api/charts/bars/{symbol}` — wraps `/bars/{symbol}`, reshapes split-orient DataFrame into
    `[{"time": <unix_seconds>, "open": …, "high": …, "low": …, "close": …, "volume": …}]`
    array that Lightweight Charts `candleSeries.setData()` accepts directly
  - `GET /api/charts/symbols` — returns grouped symbol list (CME micros + Kraken crypto) for the symbol switcher
  - `GET /api/charts/orb-markers/{symbol}` — returns ORB session open/high/low as marker objects
    `[{"time": …, "position": "aboveBar", "color": "…", "shape": "circle", "text": "ORB London"}]`
    sourced from `engine:orb_results` Redis key

### Phase CHARTS-C: Charts Page Rewrite
- [ ] **`src/lib/services/data/api/dashboard.py`** — rewrite `charts_page()` (currently L5945)
  - Full-page HTML (not HTMX fragment) with Lightweight Charts loaded from CDN:
    `unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js`
  - **Layout**: topbar (symbol switcher + timeframe switcher) + main candle pane + volume histogram subchart
  - **Symbol switcher**: dropdown from `/api/charts/symbols` — CME micros grouped separately from Kraken crypto
  - **Timeframe switcher**: `1m | 5m | 15m | 1h | 1D` pills — re-fetches `/api/charts/bars/{symbol}?interval=<tf>`
  - **Styling**: dark background (`#0c0d0f`), matches existing dashboard palette, Ruby green up-candles / red down-candles
  - **ORB markers**: fetched from `/api/charts/orb-markers/{symbol}` on symbol load, drawn via `candleSeries.setMarkers()`
  - **Price lines**: entry / stop / TP1 / TP2 from current focus card data (`/api/focus`) drawn as `candleSeries.createPriceLine()`
  - **Live updates**: `EventSource('/sse/bars/{symbol}')` → `candleSeries.update(bar)` on each event
  - **Multi-pane**: volume histogram in a subchart synced to main chart timescale (`createChart` + `chart.timeScale()` sync)
  - **No new Python dependencies** — Lightweight Charts is a CDN JS include only

### Phase CHARTS-D: Navigation Wire-Up
- [ ] Link `📈 Charts` nav item (already exists in `_build_page_shell`) to `/charts`
- [ ] Add `charts_page_route` FastAPI GET handler at `/charts` (already scaffolded at L6145 — just needs the new `charts_page()` wired in)
- [ ] Web service proxy: ensure `/charts` and `/api/charts/*` and `/sse/bars/*` are proxied through `:web → :data` (currently all `/api/*` and `/sse/*` paths are already proxied — no web service changes needed)

### Phase CHARTS-E: Charting Service — Volume Indicators & UX Polish *(charting service, `docker/charting/static/`)*

> Extends the existing ApexCharts charting service (port 8003) with the three remaining volume-based indicators that give full context for ORB/breakout trading, plus VWAP standard-deviation bands and UX persistence. All work is pure front-end — no Python or data service changes needed.

#### VWAP Standard-Deviation Bands (±1σ / ±2σ)

Currently VWAP is implemented as a single line. The bands require accumulating a running variance term alongside the existing `cumTPV` / `cumVol` state:

- [ ] **`calcVWAP()`** — extend to also accumulate `cumTypicalVolSq += typical * typical * vol`
  - Derive `variance = (cumTypicalVolSq / cumVol) − vwap²`, then `σ = √max(variance, 0)`
  - Return `{ vwap, upper1, lower1, upper2, lower2 }` per bar
- [ ] **Add 4 new series slots** (after VWAP at index 7): `VWAP+1σ` (8), `VWAP−1σ` (9), `VWAP+2σ` (10), `VWAP−2σ` (11)
  - Colors: `+1σ/−1σ` soft cyan dashed, `+2σ/−2σ` faint cyan dotted
  - All 4 share `overlayYaxis` (hidden, price-scale linked) — same pattern as existing overlays
- [ ] **`buildSeries()`** — add the 4 band series; empty array when VWAP toggle is off
- [ ] **`buildOptions()`** — extend `stroke.width`, `stroke.dashArray`, `colors`, `yaxis` arrays to cover all 12 slots
- [ ] **Incremental live update** (`updateIndicatorPoint`) — extend `push_or_replace` calls to cover the 4 band series using the same running-variance approach
- [ ] **Indicator toggle** — VWAP button already exists; turning it off should hide all 5 series (line + 4 bands) simultaneously
- [ ] **Tooltip** — add `VWAP+1σ` / `VWAP−1σ` rows to the OHLC tooltip `indLookup` block when VWAP is active

#### CVD — Cumulative Volume Delta

CVD shows net buy/sell pressure (buy volume − sell volume) as a running total. Since we only have OHLCV bars (no tape), use the standard bar-approximation: `delta = volume × (2 × (close − low) / (high − low) − 1)`. Reset per session (daily) the same way VWAP resets.

- [ ] **`calcCVD(candles, volumes)`** — new function
  - For each bar: `range = high − low`, guard `range > 0`; `delta = volume × (2 × (close − low) / range − 1)` if range > 0 else `0`
  - Daily reset: when `new Date(x).toDateString()` changes, reset `cvd = 0`
  - Returns `[{x, y: Math.round(cvd)}]`
- [ ] **Add `cvdData` to `state`** and `recalcIndicators()` / `recalcSingleIndicator("cvd")`
- [ ] **`liveInd`** — track `cvdRunning` and `cvdLastDay` for incremental `updateIndicatorPoint` extension
- [ ] **CVD sub-pane** — separate ApexCharts instance in a new `#chart-cvd` div, same pattern as the existing RSI pane
  - `buildCvdOptions()` — `type: "bar"`, height `120px`, green bars for positive delta / red for negative (per-point `fillColor`)
  - Zero-line annotation (y=0 dashed white)
  - `mountCvdChart()` / `unmountCvdChart()` / `syncCvdPane()` — mirror the RSI pane lifecycle functions
  - `state.chartCvd` instance reference
- [ ] **`index.html`** — add `<div id="chart-cvd" class="chart-cvd hidden"></div>` below `#chart-rsi`
- [ ] **`style.css`** — add `.chart-cvd` / `.chart-cvd.hidden` rules (same `flex: 0 0 120px` pattern as `.chart-rsi`)
- [ ] **Indicator toggle button** — add `<button class="ind-btn" data-ind="cvd" title="Cumulative Volume Delta">CVD</button>` to `index.html` indicator-tabs; add `cvd: false` to `state.indicators`; add `.ind-btn.active[data-ind="cvd"]` colour in `style.css`
- [ ] **Live update** — on each SSE tick, call `syncCvdPane()` after `updateIndicatorPoint`

#### Volume Profile — POC / VAH / VAL

Shows how much volume traded at each price level over a rolling lookback window. Draws the Point of Control (highest-volume price), Value Area High, and Value Area Low as horizontal overlay lines on the price pane — the same three levels your Ruby NinjaTrader indicator tracks.

- [ ] **`calcVolumeProfile(candles, volumes, bins=40, lookback=100)`** — new function
  - For each bar `i`, take a rolling slice of `min(lookback, i+1)` bars
  - Find `priceMin = min(slice lows)`, `priceMax = max(slice highs)`, `binSize = (priceMax − priceMin) / bins`
  - Distribute each bar's volume across the bins it spans (proportional to overlap with `[low, high]`)
  - **POC**: bin index with highest total volume → `priceMin + (pocBin + 0.5) × binSize`
  - **Value Area**: expand outward from POC bin until cumulative volume ≥ 70% of total → `VAH` = top of upper bin, `VAL` = bottom of lower bin
  - Returns `{ poc: [{x, y}], vah: [{x, y}], val: [{x, y}] }`
  - Note: `O(n × lookback × bins)` — cap `lookback ≤ 100` and only recalc on new candle (not forming-candle updates) to keep it fast
- [ ] **Add `pocData`, `vahData`, `valData` to `state`** and wire into `recalcIndicators()` / `recalcSingleIndicator("vp")`
- [ ] **Series slots** (append after band series, e.g. indices 12/13/14): `POC` line, `VAH` line, `VAL` line
  - Colors: POC bright cyan solid, VAH/VAL muted blue dashed — distinct from VWAP cyan
  - All three use `overlayYaxis` (hidden, price-scale linked)
- [ ] **`buildSeries()`** / **`buildOptions()`** — extend arrays to cover all slots
- [ ] **Live update** — skip VP recalc on forming-candle ticks (only recalc on `isNewCandle === true` to avoid O(n) recalc every tick)
- [ ] **Indicator toggle button** — add `<button class="ind-btn" data-ind="vp">VP</button>` to `index.html`; add `vp: false` to `state.indicators`; add colour rule to `style.css`

#### Anchored VWAP

Instead of resetting at midnight, anchor VWAP to a specific bar index. Most useful anchored to the ORB low/high (session open) or previous-day high/low — levels your engine already tracks.

- [ ] **`calcAnchoredVWAP(candles, volumes, anchorIndex)`** — new function
  - Same cumulative `(typicalPrice × vol) / cumVol` formula as `calcVWAP`, but starts accumulating from `anchorIndex` (returns `null` for bars before the anchor)
  - Returns `[{x, y}]` with `y: null` for bars before the anchor
- [ ] **Two default anchors** (user-selectable via dropdown or toggle):
  - **Session open** (`anchorIndex` = index of first bar of the current day) — useful for ORB setups
  - **Previous-day low/high** — index of the bar with the lowest low / highest high in the previous calendar day's slice
  - Helper `findAnchorIndex(candles, mode)` → `number` that resolves the correct index for each mode
- [ ] **`state`** additions: `avwapSessionData`, `avwapPrevDayData`; `indicators.avwap_session`, `indicators.avwap_prevday` (both default `false`)
- [ ] **Series slots** for the two anchors; `overlayYaxis` linked; distinct colors (session=orange, prev-day=magenta)
- [ ] **`buildSeries()`** / **`buildOptions()`** / **`recalcSingleIndicator`** — extend for both anchors
- [ ] **Indicator toggle buttons** — `AVWAP-S` (session) and `AVWAP-P` (prev-day) in `index.html` indicator-tabs
- [ ] **Live update** — both anchored VWAPs extend incrementally on each tick using the same running-accumulator pattern as `calcVWAP`; anchor index never changes within a session

#### UX: Indicator Toggle Persistence

- [ ] **localStorage persistence** — save `state.indicators` to `localStorage` key `ruby_chart_indicators` on every toggle
  - On `boot()`, read saved preferences and apply before the first `renderBars()` call so the chart opens with the user's last indicator configuration
  - Sync indicator button `active` classes from restored state in `wireControls()` (already done for initial state; just needs the read-from-localStorage step at the top of `boot()`)
- [ ] **Per-indicator configuration** (stretch) — add period inputs (e.g. EMA period, BB period, VP lookback/bins) accessible via a small settings popover on each indicator button (long-press or right-click); persist to localStorage alongside the toggle flags

---

## 🔴 TradingView Integration — Reference Overlay Only *(deferred — after v8 champion)*

> **Decision**: TradingView is used as a **reference overlay for live price action** only. Position sendback from TV is not practical. The Ruby Pine Script indicator shows ORB boxes, PDR levels, session separators, and engine signals — but all trading decisions and execution happen manually via Tradovate, informed by the Python dashboard. Future automation will use the Tradovate JavaScript bridge (Phase TBRIDGE), not TradingView webhooks.

### Phase TV-B: Ruby Futures Indicator — Engine Signal Overlay
  - Parse the CSV: filter to current chart symbol + recent timestamps (last 5 bars)
  - Draw on chart: entry line (dashed), stop line (red), TP1/TP2/TP3 levels (green dashes)
  - Signal label box: breakout type name, CNN probability, contract sizing
  - **Micro + regular sizing on every label**: "3× MGC ($330 risk) / 1× GC ($1,100 risk)"
  - Colour-code by direction: green labels for LONG, red for SHORT
  - Only show signals from last N hours (configurable input, default 4h) to avoid chart clutter

### Phase TV-C: Ruby Futures Indicator — Core Futures Layer
- [ ] **Pure price calculations** that run at zero delay
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
  - Left monitor: TradingView with Ruby Futures indicator (reference only — no position sendback)
  - Right monitor: Python dashboard (focus mode, risk strip, swing signals, sentiment)
  - Pre-market: dashboard daily bias + Grok brief + Reddit sentiment → informs watchlist
  - Dashboard tracks signals + updates risk strip
  - All execution is manual via Tradovate, informed by dashboard
  - Zero dependency on NinjaTrader or Windows for live trading

---

## 🔴 Tradovate JavaScript Bridge — Direct API Execution *(Phase TBRIDGE)*

> Tradovate exposes a REST + WebSocket API that accepts JavaScript. We only need one bridge connection to our primary Tradovate account — PickMyTrade copies all trades to remaining accounts. This replaces the TradingView position sendback approach entirely.

### Phase TBRIDGE-A: Tradovate API Client (JavaScript/Node.js)
- [ ] **Research Tradovate API docs** — REST endpoints for order placement, position query, account info
  - Auth flow: OAuth2 token → WebSocket session → order commands
  - Rate limits and order types supported (market, limit, stop-market, bracket)
- [ ] **`bridge/tradovate_client.js`** — Node.js client for Tradovate REST + WebSocket
  - `authenticate(credentials)` → access token
  - `placeOrder({symbol, action, qty, orderType, price?, stopPrice?})` → order ID
  - `getPositions()` → open positions array
  - `cancelOrder(orderId)` → confirmation
  - `flattenAll()` → close all positions
  - WebSocket: real-time fill notifications, position updates, P&L streaming
- [ ] **Environment config**: `TRADOVATE_USERNAME`, `TRADOVATE_PASSWORD`, `TRADOVATE_APP_ID`, `TRADOVATE_CID`, `TRADOVATE_SECRET` — all via env vars, never hardcoded

### Phase TBRIDGE-B: Python ↔ Node.js Bridge
- [ ] **`POST /api/bridge/order`** — Python engine sends order intent to Node.js bridge
  - Bridge runs as a sidecar container or subprocess
  - Communication via HTTP (localhost) or Redis pub/sub
  - Python publishes `OrderCommand` → bridge translates to Tradovate API call
  - Bridge publishes fill confirmations back → Python `PositionManager` updates
- [ ] **Position sync**: bridge polls Tradovate positions every 5s → publishes to `engine:live_positions` Redis key
  - Dashboard live position overlay reads from this key (already wired for broker-agnostic positions)
- [ ] **Health monitoring**: bridge heartbeat → `broker_heartbeat` Redis key (existing positions.py already reads this)

### Phase TBRIDGE-C: PickMyTrade Integration
- [ ] **Wire bridge to 1st Tradovate account only** — this is the "leader" account
- [ ] **PickMyTrade config**: connect all remaining TPT/Apex accounts as "followers"
  - Verify webhook latency (bridge fill → PickMyTrade copy → follower fill)
  - Test quantity multiplier config for different account sizes ($150K vs $300K)
  - Confirm simultaneous connection of all follower accounts
- [ ] **Failsafe**: if bridge disconnects, dashboard shows alert + blocks new signals
  - Manual trading via Tradovate UI remains available as fallback

---

## 🔴 Reddit Sentiment Integration — Futures Market Pulse *(Phase REDDIT)*

> Monitor futures-relevant subreddits for crowd sentiment, unusual activity spikes, and contrarian signals. This feeds into the dashboard as an additional metric layer and can optionally influence CNN feature weighting in v9+.

### Phase REDDIT-A: Reddit Data Collector
- [ ] **`src/lib/integrations/reddit_client.py`** — Reddit API client using PRAW (Python Reddit API Wrapper)
  - Add `praw>=7.7.0` to `pyproject.toml` dependencies
  - Auth: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` env vars
  - Target subreddits:
    - `r/FuturesTrading` — direct futures discussion (gold, NQ, ES, 6E)
    - `r/Daytrading` — general day trading sentiment + setups
    - `r/wallstreetbets` — crowd euphoria/panic indicator (contrarian signal)
    - `r/InnerCircleTraders` — ICT/SMC methodology discussion (confluence with our ICT analysis)
  - Methods:
    - `fetch_hot_posts(subreddit, limit=25)` → list of posts with title, score, num_comments, created_utc
    - `fetch_new_posts(subreddit, limit=50)` → newest posts for velocity tracking
    - `fetch_comments(post_id, limit=100)` → top comments for deeper sentiment analysis
    - `search_posts(subreddit, query, time_filter="day")` → search for specific tickers/terms

### Phase REDDIT-B: Sentiment Analysis Engine
- [ ] **`src/lib/analysis/reddit_sentiment.py`** — NLP sentiment scoring
  - Keyword extraction for futures tickers: `MGC`, `gold`, `GC`, `NQ`, `MNQ`, `ES`, `MES`, `6E`, `euro`, `BTC`, `ETH`, `SOL`
  - Sentiment classification per post/comment:
    - Rule-based first pass: bullish/bearish keyword lists + emoji patterns (🚀, 🐻, 💎🙌, etc.)
    - Optional: lightweight transformer (FinBERT or distilbert-base-uncased-finetuned-sst-2-english) for more nuanced scoring — gated behind `ENABLE_REDDIT_NLP=1` env var
  - Aggregate metrics per asset per subreddit:
    - `mention_count_1h` — how many times ticker mentioned in last hour
    - `mention_velocity` — rate of change in mentions (spike detection)
    - `avg_sentiment` — mean sentiment score [-1.0, +1.0]
    - `sentiment_skew` — bullish vs bearish ratio (extreme = contrarian signal)
    - `engagement_score` — weighted by upvotes + comment count (high engagement = conviction)
    - `wsb_euphoria_index` — WSB-specific: extreme bullishness = potential top signal (contrarian)
  - `compute_reddit_sentiment(symbol)` → `RedditSentiment` dataclass with all metrics
  - `compute_all_reddit_sentiments()` → dict of symbol → RedditSentiment for focus assets

### Phase REDDIT-C: Scheduler Integration + Caching
- [ ] **Engine scheduler**: poll Reddit every 15 minutes during ACTIVE + EVENING session modes
  - `CHECK_REDDIT_SENTIMENT` action in scheduler dispatch table
  - Results cached in Redis: `engine:reddit_sentiment:<SYMBOL>` (30-min TTL)
  - Rate limit aware: Reddit API allows ~60 requests/min with OAuth — budget across 4 subreddits
- [ ] **Spike detection**: if `mention_velocity` exceeds 3× rolling average → publish `engine:reddit_spike` SSE event
  - Dashboard shows "🔥 Reddit Spike: MGC mentioned 47 times in last hour (3.2× normal)" alert
- [ ] **Historical tracking**: store daily aggregates in Postgres `reddit_sentiment_history` table
  - Enables backtesting correlation between Reddit spikes and breakout outcomes

### Phase REDDIT-D: Dashboard Integration
- [ ] **Reddit Sentiment Panel** on dashboard
  - Per-asset sentiment bar (green=bullish, red=bearish, grey=neutral) with mention count
  - "Reddit Pulse" strip alongside risk strip — shows top 4 focus assets' sentiment
  - Subreddit breakdown: hover to see which sub is driving sentiment
  - Spike alerts in market events feed (SSE `reddit-spike` listener)
  - Historical sentiment chart (7-day rolling) on focus cards
- [ ] **Focus card integration**: sentiment badge on each focus card
  - 🟢 "Reddit Bullish (0.72)" / 🔴 "Reddit Bearish (-0.45)" / ⚪ "Quiet"
  - Contrarian warning when WSB euphoria is extreme: "⚠️ WSB extremely bullish — consider fade"

### Phase REDDIT-E: CNN Feature Integration (v9+ — optional)
- [ ] **2 new tabular features** for future v9 model:
  - `reddit_mention_velocity_norm` — normalized mention spike score [0, 1]
  - `reddit_sentiment_score` — aggregated sentiment across all tracked subs [0, 1]
- [ ] **Correlation study**: backtest Reddit sentiment vs breakout outcomes over 90 days before adding to model
  - Only add to CNN if sentiment features show >2% lift in validation accuracy
  - Otherwise keep as dashboard-only informational metric

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

## 🟡 Post-Live: Correlation Anomaly Detection (v9+)

Deferred until v8 champion is live and profitable. Nice-to-have market regime overlay, doesn't affect model quality.

### Phase 9A: Live Correlation Anomaly Dashboard
- [ ] Rolling correlation matrix across all 10 core assets (updated every 5 min)
- [ ] Compare 30-bar vs 200-bar baseline → anomaly score per pair
- [ ] Publish `engine:correlation_anomalies` → dashboard heatmap panel

---

## 🟢 Low Priority — Scaling & Copy Trading

### PickMyTrade + Account Scaling
- [ ] Sign up for PickMyTrade, test Tradovate bridge → PickMyTrade copy on a single Apex eval account
  - Verify webhook latency (bridge fill → PickMyTrade copy → follower fill) for intraday futures
  - Test quantity multiplier config for different account sizes ($150K vs $300K)
  - Confirm all 20 Apex accounts can be connected simultaneously
- [ ] Wire Tradovate JS bridge (Phase TBRIDGE-B) as the leader account signal source
- [ ] Scale TPT to 5 accounts (pass eval on each, connect via PickMyTrade as followers)
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

## 🔍 Pre-Retrain Readiness Review

> **Reviewed: full codebase audit of model, training pipeline, dataset generator, feature contract, and engine.**
> **Verdict: v8 code is READY TO TRAIN. All feature plumbing is wired end-to-end.**

### ✅ What's Confirmed Working
- **Feature contract** (`models/feature_contract.json`): v8, 37 tabular features, `asset_class_lookup` + `asset_id_lookup` tables, embedding dims (4+8=12), v8 training recipe with gate checks — all present and correct
- **`HybridBreakoutCNN`** (`breakout_cnn.py` L1456–1579): v8 architecture with `nn.Embedding(5,4)` + `nn.Embedding(25,8)`, wider tabular head (37→256→128→64 with GELU+BN), classifier (1280+64+12→512→128→2) — matches contract exactly
- **`_normalise_tabular_for_inference()`** (`breakout_cnn.py` L2280–2519): handles v5(8)→v4(14)→v6(18)→v7(24)→v7.1(28)→v8(37) backward compat padding — all slots documented with correct neutral defaults
- **`_build_row()`** (`dataset_generator.py` L1801–2263): computes all 37 features with real data + neutral fallbacks — v8-B cross-asset features use `_bars_by_ticker`, v8-C fingerprint features use `_daily_bars` + `_bars_1m`
- **`train_model()`** (`breakout_cnn.py` L1666–1970): v8 recipe wired — gradient accumulation (2×), mixup α=0.2, label smoothing 0.10, cosine warmup (5 epochs), separate LR groups (backbone 2e-4, head+embeddings 1e-3), early stopping patience=15, NaN guard, gradient clipping
- **`BreakoutDataset.__getitem__()`**: passes `asset_class_ids` and `asset_ids` as integer tensors alongside float tabular vector
- **`_run_training_pipeline()`** (`trainer_server.py` L338–640): generates dataset → stratified split → trains → evaluates → gate check → promotes → exports feature contract + ONNX — full pipeline intact
- **`DatasetConfig`**: defaults already set for v8 — `breakout_type="all"`, `orb_session="all"`, `max_samples_per_type_label=800`, `max_samples_per_session_label=400`
- **`split_dataset()`**: stratifies by `(label, breakout_type, session)` triple — correct for balanced eval
- **Peer bar loading**: `generate_dataset()` → `_resolve_peer_tickers()` → `bars_by_ticker` dict attached to each result — v8-B cross-asset features will compute real correlations
- **Tests**: 2552 passed, 0 failed, 1 skipped — clean baseline

### ⚠️ Blocking Before Training (must fix)
- [x] **Smoke-test training loop** — run 2-epoch training on tiny synthetic dataset to verify v8 changes don't crash at runtime (mixup, grad accumulation, cosine warmup, separate LR groups, embedding layers all interacting). Added `tests/test_v8_smoke.py` (31 tests, all passing: architecture, dataset loading, full 2-epoch train, evaluate_model, predict_breakout, predict_breakout_batch, grad accumulation, mixup, separate LR groups, cosine warmup, label smoothing). Also fixed 119 stale `@patch("lib.strategies.…")` paths across `test_daily_plan.py`, `test_swing_engine_grok.py`, `test_swing_detector.py`, `test_swing_actions.py` → `lib.trading.strategies.…`, and fixed 44 `lib.trading.strategies.rb.orb` import paths in `test_orb.py` → correct sub-modules (`open.detector`, `open.sessions`, `open.publisher`, `open.models`). Full suite: **2543 passed, 0 failed**.
- [ ] **Generate v8 dataset** — `generate_dataset(symbols=ALL_25, days_back=120)` with v8 `DatasetConfig` defaults
- [ ] **Verify CI/CD secrets** — `TRAINER_TAILSCALE_IP` = `100.113.72.63`, `PROD_TAILSCALE_IP` = Ubuntu Server IP

### ℹ️ Non-Blocking (cosmetic / post-train)
- [ ] `breakout_cnn.py` comments still reference "NinjaTrader", "C#", "OrbCnnPredictor" — update to generic language
- [ ] `chart_renderer.py` / `chart_renderer_parity.py` comments reference "Ruby NinjaTrader indicator" — update
- [ ] Dashboard still has `_get_bridge_info()` / `bridge-status` naming — rename to `_get_broker_info()` / `broker-status`
- [ ] `orb.py` still has `detect_opening_range_breakout()` and `ORBResult` — deprecate after v8 validates unified path
- [ ] `ORBSession` → `RBSession` bulk rename in callers (alias works, non-breaking)
- [ ] Cross-attention fusion (Phase v8-D optional) — test on small dataset after initial v8 train
- [ ] "Asset DNA" radar chart on focus cards (Phase v8-C dashboard, low priority)
- [ ] Phase CHARTS — replace placeholder `/charts` page with Lightweight Charts UI (see Phase CHARTS above)

---

## Execution Order (3-Week Sprint → TPT $150K Account)

> Focus: Docker + Python services only. TradingView is reference-only for live price action.
> Strategy: 1 asset per day with a good breakout setup, done trading for the day.
> After TPT account 1, get account 2 and wire PickMyTrade to copy trades from Tradovate.
> Reddit sentiment is a parallel track — can be built during Week 2 while GPU trains.

**✅ Step 1 — Codebase cleanup (DONE):**
- Phase 1A–1H — all complete, unified RB system, 2552 tests passing
- NinjaTrader bridge code removed from Python codebase — positions API is now broker-agnostic
- v8 feature code all shipped: embeddings, cross-asset correlation, asset fingerprints, architecture upgrades, training recipe

**Step 2 — CNN v8 Champion (THE milestone, do once, do it right):**

  *Week 1: Final pre-train verification* 🔜
  - ✅ Smoke-test training loop (2 epochs, tiny dataset) — all v8 changes verified working together
  - Verify CI/CD secrets (trainer IP, prod IP)
  - Generate v8 dataset (~50K–80K samples, all 25 symbols × 13 types × 9 sessions)

  *Week 2: Train (GPU rig at 100.113.72.63:8200, mostly hands-off)*
  - Train unified v8 model (80 epochs, ~6–10 hours on GPU)
  - **Parallel track**: build Reddit sentiment integration (Phase REDDIT-A/B/C) while GPU trains
  - Optionally: train 7 per-asset specialists + distill → compare vs unified
  - Gate check: ≥89% acc, ≥87% prec, ≥84% rec
  - Promote winner → `breakout_cnn_best.pt` + deploy to Ubuntu Server

  *Week 3: Validate + Go Live on Demo*
  - Phase v8-G — smoke test on live breakouts, verify inference
  - Manual trading via Tradovate, dashboard tiled alongside TradingView (reference only)
  - Dashboard: CNN probabilities, risk strip, focus cards, Reddit sentiment panel
  - Tune session thresholds based on live signal quality
  - Document the workflow end-to-end

**Step 3 — TPT $150K account goes live:**
- Trade 1 asset/day with best breakout setup
- CNN co-pilot for entry confirmation (≥0.85 probability gate)
- Risk strip + position sizing + Reddit sentiment on dashboard

**Step 4 — Scale (when consistent):**
- Phase TBRIDGE — Tradovate JavaScript bridge for 1st account automation
- PickMyTrade copies from 1st Tradovate account to all followers
- Scale to 5 TPT accounts → then Apex 20 accounts
- Phase 9A — correlation anomalies (nice-to-have overlay)
- Phase 6 — Kraken portfolio (separate from futures)
- Phase REDDIT-E — Reddit features in CNN v9 (if backtesting shows >2% lift)

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

### 7. Dashboard → Manual Trading Signal Flow

```
Engine fires CNN-gated signal
  │
  ├─ signals_publisher.append_and_push(signal)
  │    → signals.csv committed to nuniesmith/futures-signals (GitHub API)
  │    → SSE push to dashboard
  │
  ├─ Dashboard (primary decision surface)
  │    → Focus cards with CNN probability, entry/stop/TP, dual sizing, risk strip
  │    → Reddit sentiment badge + spike alerts
  │    → Trader decides whether to execute manually in Tradovate
  │
  ├─ TradingView (reference overlay — no position sendback)
  │    → Ruby Futures indicator shows levels on chart for visual confirmation
  │    → NOT used for order execution or position management
  │
  └─ Tradovate JS Bridge (future — Phase TBRIDGE)
       → Direct API execution on leader account
       → PickMyTrade copies to follower accounts
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
Docker Compose:
  Ubuntu Server (100.122.184.58)                          Home Laptop (100.113.72.63)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
  │    :data     │  │   :engine    │  │    :web      │   │   :trainer   │
  │  FastAPI     │  │  main.py     │  │  FastAPI     │   │  FastAPI     │
  │  REST + SSE  │  │  scheduler   │  │  reverse-    │   │  dataset gen │
  │  bar cache   │  │  risk mgr    │  │  proxy only  │   │  CNN train   │
  │  Kraken feed │  │  position mgr│  │  port 8180   │   │  promote .pt │
  │  Reddit poll │  │  all handlers│  │              │   │  CUDA GPU    │
  │  port 8050   │  │  (no HTTP)   │  │              │   │  port 8200   │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   └──────┬───────┘
         │    publishes     │   reads          │ proxies          │
         │    Redis state ←─┘   Redis          │ → :data          │
         ↓                                     ↓                  │
  ┌──────────┐  ┌──────────┐           Browser (8180)            │
  │  Redis   │  │ Postgres │                                      │
  └──────────┘  └──────────┘                                      │
  ┌─────────────┐  ┌──────────┐                                   │
  │ Prometheus  │  │ Grafana  │                                   │
  └──────┬──────┘  └──────────┘                                   │
         └──────────────────────── Tailscale mesh ────────────────┘

  Service responsibilities:
    :data    — all REST/SSE endpoints, bar cache (Postgres+Redis), Kraken WS feed,
               Reddit sentiment polling, /bars/{symbol} auto-fill, /api/charts/*
    :engine  — DashboardEngine, ScheduleManager, RiskManager, PositionManager,
               breakout detection, CNN inference, Grok briefs, Redis publish
               writes /tmp/engine_health.json as heartbeat (no HTTP port)
    :web     — stateless reverse-proxy; proxies all /api/* and /sse/* to :data

  Port map:
    8050  → :data    (internal 8000)   REST + SSE API
    8180  → :web     (internal 8080)   HTMX dashboard
    8200  → :trainer (internal 8200)   GPU training server
    9095  → Prometheus
    3010  → Grafana

  (Future) Tradovate JS Bridge:
  ┌────────────────────┐
  │  Node.js sidecar   │ ← runs alongside engine on Ubuntu Server
  │  Tradovate REST+WS │ ← leader account execution
  │  → PickMyTrade     │ ← follower accounts via webhook
  └────────────────────┘

Tailscale mesh:
  Ubuntu Server → data + engine + web + postgres + redis + monitoring (always on, 24/7)
  Home Laptop   → trainer (on-demand, CUDA GPU, port 8200)

CI/CD (6-image matrix — nuniesmith/futures):
  :data       amd64 + arm64   ← new, `:latest` alias
  :engine     amd64 + arm64
  :web        amd64 + arm64
  :trainer    amd64 only
  :prometheus amd64 + arm64
  :grafana    amd64 + arm64
```

### 10. Reddit Sentiment Pipeline (Phase REDDIT)

```
Engine Scheduler (every 15 min during ACTIVE + EVENING)
  │
  ├─ reddit_client.py → PRAW OAuth → fetch hot/new posts from 4 subreddits
  │    r/FuturesTrading, r/Daytrading, r/wallstreetbets, r/InnerCircleTraders
  │
  ├─ reddit_sentiment.py → per-asset scoring
  │    mention_count, velocity, avg_sentiment, engagement, wsb_euphoria
  │
  ├─ Cache: Redis engine:reddit_sentiment:<SYMBOL> (30-min TTL)
  ├─ History: Postgres reddit_sentiment_history (daily aggregates)
  │
  ├─ Spike detection: mention_velocity > 3× rolling avg
  │    → SSE engine:reddit_spike → dashboard alert
  │
  └─ Dashboard: sentiment badges on focus cards + Reddit Pulse strip
```

---

## Completed

### Data Service Split (`:data` + `:engine` separation)
- [x] `docker/data/Dockerfile` — standalone image: `python:3.13-slim`, copies `src/`, runs via `entrypoints/data/main.py`
- [x] `docker/data/entrypoint.sh` — `exec python -m entrypoints.data.main`
- [x] `docker/engine/entrypoint.sh` — stripped to `exec python -m lib.services.engine.main` (uvicorn removed)
- [x] `docker/engine/Dockerfile` — removed `EXPOSE 8000`, removed HTTP healthcheck, uses `test -f /tmp/engine_health.json`
- [x] `lib.services.data.main` — added `main()` function (was `if __name__ == "__main__"` only); `LOG_LEVEL` env var wired
- [x] `docker-compose.yml` — `data` service (port 8050), `engine` (no ports, depends on data), `web`/`trainer`/`prometheus` all point at `http://data:8000`; `ENABLE_KRAKEN_CRYPTO=1` on data, `=0` on engine
- [x] CI/CD — `data` added to docker matrix (amd64+arm64, `is_default: true` → `:latest`); engine set `is_default: false`; deploy pulls + starts `data engine web prometheus grafana`; summary table updated


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
