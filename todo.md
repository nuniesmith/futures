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
│       └── ruby_futures.pine   # TradingView Pine Script indicator
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
- **Models**: `models/breakout_cnn_best.pt` + `feature_contract.json` committed (Git LFS). Latest champion: **87.1% accuracy**, 87.15% precision, 87.27% recall, 25 epochs, v6 18-feature. Retrain to v8 (37-feature) is the next major milestone.
- **Docker**: `:engine` (data API + CNN inference), `:web` (HTMX dashboard), `:trainer` (GPU training server). Runs on Ubuntu Server (engine + web + postgres + redis + monitoring) and home laptop GPU rig (trainer only).
- **Feature Contract**: v8, 37 tabular features + hierarchical asset embeddings. `models/feature_contract.json` is the canonical source. Expanded from v6 (18) → v7.1 (28) → v8 (37) with cross-asset correlation features, asset fingerprint features, and learned embeddings.
- **CNN Model**: EfficientNetV2-S + wider tabular head + asset embeddings. Auto-detects tabular dimension from checkpoint metadata at load time. Training pipeline: generate dataset → train → evaluate → gate check (≥89% acc, ≥87% prec, ≥84% rec) → promote. Python `_normalise_tabular_for_inference()` handles v6→v7→v7.1→v8 backward-compat padding so older models work with the new code.
- **Breakout Types**: 13 — ORB, PrevDay, InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci. Fully wired in engine detection, training, dataset generator, CNN tabular vector, chart renderer.
- **Position Manager**: `position_manager.py` — always-in 1-lot micro positions, reversal gates (CNN ≥ 0.85, MTF ≥ 0.60, 30min cooldown), Redis persistence, `OrderCommand` emitter (informational — feeds dashboard, not live execution).
- **Dashboard**: HTMX + FastAPI — live signals, 13 breakout type filter pills, 9 session tabs, MTF score column, trade journal, Kraken crypto chart + correlation panel, Grok AI analyst, CNN dataset preview, positions panel, flatten/cancel buttons, market regime (HMM), performance panel, volume profile, equity curve, asset focus cards with entry/stop/TP levels, focus mode, risk strip, swing action buttons. Broker-agnostic position bridge (TradingView/Tradovate ready).
- **Kraken Integration**: `KrakenDataProvider` REST + `KrakenFeedManager` WebSocket feed. 9 crypto pairs streaming.
- **Massive Integration**: `MassiveDataProvider` REST + `MassiveFeedManager` WebSocket. Front-month resolution, primary bars source for training.
- **Data Service**: Unified data layer — Redis cache → Postgres → external APIs. Startup cache warming from Postgres (7 days).
- **Training**: `trainer_server.py` FastAPI (port 8200). `dataset_generator.py` covers all 13 types + 9 sessions + Kraken. Full pipeline: generate → split (85/15 stratified) → train → evaluate → gate → promote.
- **CI/CD**: Lint → Test → Build & push 5 Docker images (engine, web, trainer, prometheus, grafana) → Deploy to Ubuntu Server via Tailscale SSH → Deploy trainer to home laptop via Tailscale SSH → Health checks → Discord notifications.
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

## 🔴 TradingView Integration — Live Trading UI *(deferred — after v8 champion)*

The Pine Script indicator is the primary live trading chart. The Python dashboard runs alongside it for CNN context and risk management.

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
  - Left monitor: TradingView with Ruby Futures indicator + Tradovate demo connected
  - Right monitor: Python dashboard (focus mode, risk strip, swing signals)
  - Pre-market: dashboard daily bias + Grok brief → informs TV watchlist
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

## 🟡 Post-Live: Correlation Anomaly Detection (v9+)

Deferred until v8 champion is live and profitable. Nice-to-have market regime overlay, doesn't affect model quality.

### Phase 9A: Live Correlation Anomaly Dashboard
- [ ] Rolling correlation matrix across all 10 core assets (updated every 5 min)
- [ ] Compare 30-bar vs 200-bar baseline → anomaly score per pair
- [ ] Publish `engine:correlation_anomalies` → dashboard heatmap panel

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

## Execution Order (3-Week Sprint → TPT $150K Account)

> Focus: Docker + Python services only. TradingView is good enough for manual trading right now.
> Strategy: 1 asset per day with a good breakout setup, done trading for the day.
> After TPT account 1, get account 2 and wire PickMyTrade to copy trades from Tradovate.

**✅ Step 1 — Codebase cleanup (DONE):**
- Phase 1A–1H — all complete, unified RB system, 2552 tests passing
- NinjaTrader bridge code removed from Python codebase — positions API is now broker-agnostic

**Step 2 — CNN v8 Champion (THE milestone, do once, do it right):**

  *Week 1: Code the upgrades (no GPU needed yet)* ✅
  - Phase v8-A — hierarchical asset embeddings (replace flat ordinals) ✅
  - Phase v8-B — cross-asset correlation features (+3 features) ✅
  - Phase v8-C — asset fingerprint features (+6 features) ✅
  - Phase v8-D — architecture upgrades (wider tabular head, GELU, mixup, warmup) ✅
  - Phase v8-E — lock down training recipe & hyperparameters ✅
  - **Remaining**: fix immediate issues (see "Immediate Fixes" section above), smoke-test training loop

  *Week 2: Train (GPU rig at 100.113.72.63:8200, mostly hands-off)*
  - Generate v8 dataset (~50K–80K samples, all 25 symbols × 13 types × 9 sessions)
  - Train unified v8 model (80 epochs, ~6–10 hours on GPU)
  - Optionally: train 7 per-asset specialists + distill → compare vs unified
  - Gate check: ≥89% acc, ≥87% prec, ≥84% rec
  - Promote winner → `breakout_cnn_best.pt` + deploy to Ubuntu Server

  *Week 3: Validate + Go Live on Demo*
  - Phase v8-G — smoke test on live breakouts, verify inference
  - Manual trading on TradingView demo (Tradovate connected)
  - Dashboard tiled alongside TV — CNN probabilities, risk strip, focus cards
  - Tune session thresholds based on live signal quality
  - Document the workflow end-to-end

**Step 3 — TPT $150K account goes live:**
- Trade 1 asset/day with best breakout setup
- CNN co-pilot for entry confirmation (≥0.85 probability gate)
- Risk strip + position sizing on dashboard

**Step 4 — Scale (when consistent):**
- Get 2nd TPT account → PickMyTrade copies from 1st Tradovate account
- Phase TV-D — TV → Python webhook (if needed for automation)
- Scale to 5 TPT accounts → then Apex 20 accounts
- Phase 9A — correlation anomalies (nice-to-have overlay)
- Phase 6 — Kraken portfolio (separate from futures)

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
  │    → SSE push to dashboard
  │
  └─ TradingView (ruby_futures.pine)
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
Docker Compose:
  Ubuntu Server (100.122.184.58)          Home Laptop (100.113.72.63)
  ┌──────────────┐  ┌──────────────┐      ┌──────────────┐
  │   :engine    │  │    :web      │      │   :trainer   │
  │  main.py     │  │  FastAPI     │      │  FastAPI     │
  │  scheduler   │  │  dashboard   │      │  dataset gen │
  │  risk mgr    │  │  focus cards │      │  CNN train   │
  │  position mgr│  │  settings    │      │  promote .pt │
  │  all handlers│  │  risk strip  │      │  CUDA GPU    │
  └──────┬───────┘  └──────┬───────┘      └──────┬───────┘
         └──────────────────┤                     │
                            ↓                     │
  ┌──────────┐  ┌──────────┐                      │
  │  Redis   │  │ Postgres │                      │
  └──────────┘  └──────────┘                      │
  ┌─────────────┐  ┌──────────┐                   │
  │ Prometheus  │  │ Grafana  │                   │
  └─────────────┘  └──────────┘                   │
         └────────── Tailscale mesh ──────────────┘

Tailscale mesh:
  Ubuntu Server → engine + web + postgres + redis + monitoring (always on, 24/7)
  Home Laptop   → trainer (on-demand, CUDA GPU, port 8200)
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
