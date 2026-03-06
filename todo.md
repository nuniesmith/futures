Docker Hub repo (`nuniesmith/futures`):
- `:engine` — engine + data API (runs on Raspberry Pi)
- `:web` — HTMX dashboard (runs on Raspberry Pi)
- `:trainer` — GPU training server (runs on dedicated rig)
- `:latest` — alias for `:engine` (default pull)
- `:engine-<sha>` / `:web-<sha>` / `:trainer-<sha>` — pinned to commit

Repos:
| Repo | Path | Purpose |
|---|---|---|
| **futures** | `~/github/futures` | Live dashboard, web UI, engine, shared `lib` (this repo) |
| **rb** | `~/github/rb` | Service-only trainer (compose pulls `nuniesmith/futures:trainer`), hosts trained models (.pt, .onnx) |
| **ninjatrader** | `~/github/ninjatrader` | NinjaTrader 8 C# strategies, indicators, Bridge — pulls best ONNX from `rb` repo |

---
## Current State
- **Dashboard**: HTMX + FastAPI serving live market stats, multi-type breakout signals, risk status, Grok AI analyst.
- **Engine**: Session-aware scheduler covering full 24h Globex day (9 sessions, 18:00 ET start).
- **Breakout Pipeline**: Detection → 6 deterministic filters (majority gate) → optional CNN inference → Redis publish.
- **Breakout Types**: 13 types — ORB, PrevDay, InitialBalance, Consolidation + 9 researched (Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci). All implemented in engine detection + training simulators.
- **TP3 + EMA9 Trailing**: 3-phase bracket walk-forward (SL/TP1 → TP2 → EMA9 trail toward TP3) on all 13 types.
- **CNN Model**: EfficientNetV2-S + tabular head, 15 features (v5 contract), ONNX export for NT8.
- **Feature Contract**: v5 → v6 ready — 15 tabular features, 13 breakout types (ordinals 0–12), TP3 + EMA9 trailing, sessions, Kraken pairs.
- **Multi-Session**: All 9 sessions with bracket params matching futures + NT8.
- **Parity Renderer**: Default for training — pixel-perfect match with NT8 `OrbChartRenderer`.
- **CNN Inference**: CPU-only fallback in engine, watchdog-based hot-reload.
- **Kraken Crypto**: REST + WebSocket v2 integration for 9 spot pairs (BTC, ETH, SOL, LINK, AVAX, DOT, ADA, MATIC, XRP).
- **NT8 Deploy**: Dashboard panel generates installer .bat that pulls C# from ninjatrader repo.
- **Monitoring**: Prometheus + Grafana dashboards (optional profile).
- **CI/CD**: Lint → Test → Build & push 3 tagged images to `nuniesmith/futures` on DockerHub.
- **rb Repo**: Renamed from `orb` → `rb`. Service-only compose pulling trainer image. Hosts trained champion models (.pt for engine, .onnx for NT8). NinjaTrader pulls best ONNX via PowerShell.

---
## Active — `futures` repo (`~/github/futures`)

### Engine & Detection
- [x] **Add engine detection for new 9 breakout types** — Wired WEEKLY, MONTHLY, ASIAN, BBSQUEEZE, VA, INSIDE, GAP, PIVOT, FIB into `detect_all_breakout_types()` with proper range builders, DEFAULT_CONFIGS for all 13 types, scheduler action types, and 81 tests
- [x] **Position Manager — Stop-and-Reverse strategy** — New `position_manager.py` module: persistent 1-lot micro positions, 3-phase bracket walk (SL/TP1 → breakeven → EMA9 trail to TP3), reversal gates (CNN, MTF, cooldown, winning-position protection), limit/market entry decision, session-end closure for intraday types, Redis state persistence, 105 tests
- [x] **Asset watchlists** — Added `CORE_WATCHLIST` (5 assets: MGC, MCL, MES, MNQ, M6E), `EXTENDED_WATCHLIST` (5: SIL, M2K, M6B, MBT, ZN), `ACTIVE_WATCHLIST`, and ticker frozensets to `models.py` + exported from `core/__init__.py`
- [x] **Strategy plan document** — Comprehensive `docs/STRATEGY_PLAN.md` covering asset review (22→10 active, 12 dropped), model versatility architecture, stop-and-reverse design, order management, EMA9 trailing rules, MTF integration, and phased implementation plan
- [ ] **Add session-level performance stats** to daily report
- [ ] **Backfill gap detection** — alert when historical bars have gaps > N minutes

### Dashboard & Web UI
- [ ] **Add "Breakout Type" filter + MTF score column** to signal history table
- [ ] Trade journal UI improvements — inline editing, tag filtering
- [ ] Kraken crypto price chart — live candlestick chart in dashboard for tracked pairs
- [ ] Crypto/futures correlation panel — show BTC vs MES/MGC correlation in sidebar

### Training & Dataset (shared `lib`)
- [x] **Fix trainer_server.py ↔ breakout_cnn.py signature mismatch** — Corrected `_run_training_pipeline` to call `train_model(data_csv=..., val_csv=..., model_dir=..., image_root=...)` with correct kwargs, added new `evaluate_model()` function for post-training metrics (accuracy/precision/recall via sklearn), fixed ONNX export to use `importlib.getattr` for missing function, smoke test passes end-to-end on GPU
- [x] **Add `evaluate_model()` to breakout_cnn.py** — Loads checkpoint, runs inference on validation CSV, returns `{val_accuracy, val_precision, val_recall}` dict; used by trainer pipeline for gate checks
- [ ] **Expand tabular features v5→v6** (8→12 features) — Add `breakout_type_ord`, `asset_volatility_class`, `range_atr_ratio`, `hour_of_day` to CNN input; update `TABULAR_FEATURES`, `NUM_TABULAR`, `BreakoutDataset.__getitem__`, `_normalise_tabular_for_inference`, and engine inference callsite
- [ ] **Per-type model heads or type-embedding** in CNN
- [ ] **Session-specific training thresholds** (per `SESSION_THRESHOLDS`)
- [ ] **Automated good/bad balancing** across all 13 types + 9 sessions
- [ ] Synthetic data augmentation (noise, time shifts)
- [ ] CLI command: `python -m lib.training.dataset_generator generate --symbols MGC MES --session all --breakout-type all`
- [ ] Dashboard preview: view random good/bad snapshots per type/session

### Infrastructure
- [ ] Rate limiting tuning — review slowapi config for SSE vs REST endpoints
- [ ] Deployment pipeline — add Pi deploy stage back to CI/CD when ready
- [x] **Fix `sync_models.sh` repo reference** — updated `nuniesmith/orb` → `nuniesmith/rb`
- [ ] Auto-sync trained models from `rb` repo to engine (rsync/scp post-train hook)

### Monitoring
- [ ] Prometheus metrics: `images_generated`, `label_balance`, `render_time` (trainer)
- [ ] Grafana panel: "Training Data Health" (win-rate per type/session)

---
## Active — `rb` repo (`~/github/rb`)

### Trainer Service
- [x] **End-to-end trainer smoke test** — Smoke test passes: 257 images generated, 2-epoch training on RTX 2070 SUPER, 63.8% accuracy (expected for quick test), model promoted, champion .pt (83.1 MB) written to disk, full pipeline in 20s
- [ ] **Full retrain smoke test** — `docker compose up` on GPU rig, `POST /train` with `--symbols MGC MCL MES MNQ M6E SIL M2K M6B MBT ZN --session all --breakout-type all --days 90 --epochs 25`
- [ ] **Verify compose pulls `nuniesmith/futures:trainer`** correctly and trainer server starts

### Training & Model Export
- [ ] **Retrain with all 13 breakout types + TP3**
  - Generate dataset for ALL 13 breakout types + ALL 9 sessions + Kraken
  - Validate ONNX export matches PyTorch predictions
  - Export `feature_contract.json` v6 with 13 types + TP3 fields
- [ ] **Commit champion models to `rb` repo**
  - `models/breakout_cnn_best.pt` — PyTorch checkpoint (engine pulls this)
  - `models/breakout_cnn_best.onnx` — ONNX export (NT8 pulls this via PowerShell)
  - `models/breakout_cnn_best_meta.json` — metadata
  - `models/feature_contract.json` — v6 contract

### Model Hosting
- [ ] **Verify `futures` engine can pull best `.pt`** from `rb` repo via `scripts/sync_models.sh`
- [ ] **Verify `ninjatrader` PowerShell can pull best `.onnx`** from `rb` repo
- [ ] Document model promotion workflow (train → evaluate → commit champion → consumers pull)

---
## Active — `ninjatrader` repo (`~/github/ninjatrader`)

### C# BreakoutType Expansion
- [ ] **Update C# `BreakoutType` enum** to match 13-value IntEnum:
  ```
  ORB=0, PrevDay=1, InitialBalance=2, Consolidation=3,
  Weekly=4, Monthly=5, Asian=6, BollingerSqueeze=7,
  ValueArea=8, InsideDay=9, GapRejection=10, PivotPoints=11, Fibonacci=12
  ```
- [ ] **Update C# `OrbCnnPredictor`** to pass `breakout_type_ord` for all 13 types (ordinals 0–12, normalised /12)

### Chart Rendering
- [ ] **Update C# `OrbChartRenderer`** to draw 9 new box styles:
  - `teal_solid` (Weekly), `orange_solid` (Monthly), `red_dashed` (Asian)
  - `magenta_dashed` (BollingerSqueeze), `olive_solid` (ValueArea), `lime_dashed` (InsideDay)
  - `coral_solid` (GapRejection), `steel_dashed` (PivotPoints), `amber_solid` (Fibonacci)

### TP3 + EMA9 Trailing
- [ ] **Implement C# TP3 + EMA9 trailing** in `BreakoutStrategy.cs`:
  - Add `tp3_atr_mult` from `RangeConfig` / `feature_contract.json`
  - After TP2 hit: trail remaining contracts with EMA9 crossover
  - Exit at TP3 or EMA9 stop
  - Match Python 3-phase bracket logic in `position_manager.py` exactly

### Stop-and-Reverse Integration
- [ ] **Implement C# stop-and-reverse** in `BreakoutStrategy.cs`:
  - Mirror `PositionManager` logic: always-in 1-lot micro for core 5 assets
  - Reversal gates: CNN ≥ 0.85 (0.92 for winning positions), MTF ≥ 0.60, 30min cooldown
  - Limit entry at range edge, market chase only with CNN ≥ 0.90 and < 0.5×ATR overshoot
  - Sync position state with Python engine via Bridge WebSocket

### Model Pull
- [ ] **Update PowerShell model pull script** to fetch `.onnx` + `feature_contract.json` v6 from `rb` repo

---
## Completed

### BreakoutStrategy Crash Fix (2026-03-05)
- [x] **Root cause analysis**: Strategy terminated with 34 open positions across 15 instruments
  - OCO ID reuse: NinjaTrader rejected a bracket order because the OCO group ID was already consumed by a prior rejected SL in the same session
  - SL price validation: BuyToCover StopMarket for a SHORT position was placed at/below market price ("Buy stop orders can't be placed below the market")
  - NinjaTrader's default unmanaged error handling (`ErrorHandling=Stop strategy, cancel orders, close positions`) killed the entire strategy
- [x] **CNN dimension mismatch**: C# sent 14 tabular features (v4 contract) but deployed ONNX model expects 8 (v3) — every CNN inference failed, meaning zero AI filtering was active
- [x] **Signal name > 50 chars**: NinjaTrader silently ignores orders with signal names exceeding 50 characters
- [x] **Fix 1**: Reduced tracked instruments from 15 → 5 core assets (`MGC,MES,MNQ,MYM,6E`)
- [x] **Fix 2**: Added `MaxConcurrentPositions` (default 5) with active position tracking via `OnOrderUpdate` fill/flat detection
- [x] **Fix 3**: Added `OnOrderUpdate` override to absorb rejected orders gracefully instead of letting NT8 terminate the strategy
- [x] **Fix 4**: Added max concurrent positions gate before entry submission
- [x] **Fix 5**: Made OCO IDs globally unique by appending a 6-char GUID suffix (`OCO-{signalId}-{guid6}`)
- [x] **Fix 6**: Truncated signal names to 49 chars (NT8 limit is 50)
- [x] **Fix 7**: Added SL price validation — corrects stop price to correct side of market before submission
- [x] **Fix 8**: Wrapped `SubmitOrderUnmanaged` calls in try/catch to prevent unhandled exceptions
- [x] **Fix 9**: Made `OrbCnnPredictor.NumTabular` dynamic — auto-detects expected dimension from ONNX model metadata and adapts the tabular vector (truncate/zero-pad)
- [x] **Fix 10**: Added position count to entry log lines for monitoring
- [x] **Fix 11**: Increased entry cooldown from 5 → 10 minutes to reduce over-trading
- [x] **Fix 12**: Added CNN tabular dimension logging at startup for diagnosis
- [x] Created `scripts/patch_breakout_strategy.py` — repeatable patch script with 25 targeted text replacements, dry-run mode, and verification

### Trainer Pipeline Fix + Evaluate Model
- [x] Fixed `trainer_server.py` `_run_training_pipeline()` — replaced incorrect `train_model(csv_path=..., save_path=..., patience=...)` with correct `train_model(data_csv=..., val_csv=..., model_dir=..., image_root=...)`
- [x] Added `evaluate_model()` to `breakout_cnn.py` — loads checkpoint, computes accuracy/precision/recall via sklearn on validation set, returns metrics dict consumed by trainer pipeline gate checks
- [x] Fixed ONNX export step — uses `importlib.getattr` to gracefully handle missing `export_onnx_model` function
- [x] Trainer smoke test passes end-to-end on RTX 2070 SUPER: dataset generation (257 images) → CNN training (2 epochs, GPU) → evaluation (63.8% acc) → champion promotion → .pt on disk (83.1 MB)

### Position Manager & Strategy
- [x] Created `lib/services/engine/position_manager.py` — stop-and-reverse micro contract strategy module
  - `MicroPosition` dataclass: full position state with bracket levels, phase tracking, EMA9 trail, excursion metrics, serialisable to/from Redis
  - `PositionManager` class: manages persistent 1-lot positions for core watchlist assets
  - 3-phase bracket walk: Phase 1 (SL/TP1) → Phase 2 (breakeven after TP1) → Phase 3 (EMA9 trailing after TP2, hard cap at TP3)
  - Reversal gate: CNN prob (0.85 min, 0.92 for winners, 0.95 for +1R winners), filter pass, 30min cooldown, MTF ≥ 0.60
  - Entry type decision: limit at range edge, market chase only with CNN ≥ 0.90 within 0.5×ATR
  - Session-end closure: closes intraday types (ORB, IB, etc.) but keeps swing types (Weekly, Monthly, Asian)
  - `OrderCommand` emitter: BUY/SELL/MODIFY_STOP/CANCEL for NinjaTrader Bridge consumption
  - Redis persistence: save/load state for engine restart survival
  - 105 tests covering all paths
- [x] Added `CORE_WATCHLIST` (5 assets), `EXTENDED_WATCHLIST` (5 assets), `ACTIVE_WATCHLIST` (union), ticker frozensets to `models.py`
- [x] Exported new watchlist constants from `lib/core/__init__.py`
- [x] Created `docs/STRATEGY_PLAN.md` — comprehensive strategy document

### Breakout Types Expansion
- [x] Extended `BreakoutType` IntEnum from 4 → 13 types (ORB through Fibonacci)
- [x] Added 9 new `RangeConfig` entries with unique box styles, RGBA colours, and tuned bracket params
- [x] Added engine `BreakoutType` StrEnum: WEEKLY, MONTHLY, ASIAN, BBSQUEEZE, VA, INSIDE, GAP, PIVOT, FIB
- [x] Added bidirectional engine↔training type mapping for all 13 types
- [x] Implemented 9 new simulators: Weekly, Monthly, Asian, BollingerSqueeze (BB inside KC), ValueArea (volume profile VAH/VAL), InsideDay, GapRejection, PivotPoints (classic/Woodie/Camarilla), Fibonacci (38.2%–61.8% zone)
- [x] Wired all 13 types into dataset generator dispatcher
- [x] Added grouping constants: `EXCHANGE_BREAKOUT_TYPES`, `RESEARCHED_BREAKOUT_TYPES`, `HTF_BREAKOUT_TYPES`, `DETECTED_BREAKOUT_TYPES`
- [x] Added helper functions: `types_with_ema_trailing()`, `types_with_tp3()`

### TP3 + EMA9 Trailing
- [x] Added `tp3_atr_mult`, `enable_ema_trail_after_tp2`, `ema_trail_period` to `RangeConfig` and `BracketConfig`
- [x] Added `tp3`, `hit_tp2`, `hit_tp3`, `ema_trail_exit`, `trail_exit_price` to `ORBSimResult`
- [x] Implemented 3-phase bracket walk-forward: Phase 1 SL/TP1 → Phase 2 TP2 → Phase 3 EMA9 trail toward TP3
- [x] Applied to both `simulate_orb_outcome` (ORB) and `_simulate_range_outcome` (all other types)
- [x] New outcome types: `tp2_hit`, `tp3_hit`, `ema_trail_exit` with correct R-multiple PnL

### Monorepo Merge (orb → lib)
- [x] Merged `orb/breakout_types.py` → `lib/core/breakout_types.py` (canonical `BreakoutType(IntEnum)` + `RangeConfig`)
- [x] Merged `orb/multi_session.py` → `lib/core/multi_session.py` (9 sessions, bracket params, ordinals)
- [x] Merged `orb/chart_renderer.py` → `lib/analysis/chart_renderer.py` (mplfinance Ruby-style renderer)
- [x] Merged `orb/chart_renderer_parity.py` → `lib/analysis/chart_renderer_parity.py` (pixel-perfect C# match)
- [x] Merged `orb/dataset_generator.py` → `lib/training/dataset_generator.py`
- [x] Merged `orb/orb_simulator.py` → `lib/training/orb_simulator.py`
- [x] Created `lib/training/trainer_server.py` — FastAPI HTTP training server
- [x] Updated all bare sibling imports to `lib.*` paths
- [x] Updated `lib/core/__init__.py` — exports `BreakoutType`, `RangeConfig`, etc.

### CI/CD & Docker
- [x] Rewrote CI/CD pipeline — Lint → Test → Build & push 3 Docker images
- [x] `nuniesmith/futures:engine`, `nuniesmith/futures:web`, `nuniesmith/futures:trainer`
- [x] Added `profiles: [training]` to trainer service

### Breakout Detection (Engine)
- [x] Generalized breakout detection — `lib/services/engine/breakout.py` with `BreakoutType` + `RangeConfig` covering ORB, PrevDay, IB, Consolidation
- [x] Integrated full MTF analyzer as hard filter + CNN features for all types
- [x] Updated scheduler to run multiple BreakoutType checks in parallel per session
- [x] Extended `orb_events` table with `breakout_type`, `mtf_score`, `macd_slope`, `divergence`

### Training Pipeline
- [x] Core generalization: `BreakoutType` enum + `RangeConfig`
- [x] All 9 sessions as `ORBSession` frozen dataclasses
- [x] Updated `dataset_generator.py` — stores `breakout_type` + `breakout_type_ord`, Kraken support, parity renderer default
- [x] Added Kraken crypto support in backfill
- [x] Extended tabular features to 15 (CNN & ONNX) — `breakout_type_ord` at [14]
- [x] Updated `feature_contract.json` to v5 (v6 ready with 13 types + TP3)
- [x] Parity renderer is now default
- [x] Normalised `breakout_type_ord` to [0.0, 1.0] across 13 types (was /3, now /12)

### Dashboard & Web UI
- [x] Volume profile chart visualization
- [x] Historical performance charts
- [x] Kraken crypto ORB tuning
- [x] Dark/light theme toggle
- [x] Per-session ORB signal history view with multi-type filter tabs
- [x] Grok AI analyst — streaming response display with SSE

### Engine & Infrastructure
- [x] Full 24h Globex coverage (9 sessions)
- [x] Per-session CNN gate via Redis
- [x] Prometheus + Grafana monitoring stack
- [x] HTMX dashboard with SSE live updates
- [x] Session-aware scheduler
- [x] Risk engine integrated
- [x] Grok AI morning briefing + live updates
- [x] Daily report generation + email
- [x] Docker consolidation
- [x] CNN model watcher
- [x] CNN sync endpoint
- [x] Alert rules in Prometheus
- [x] Regime detection display

### Kraken Crypto
- [x] REST + WebSocket v2 integration for 9 spot pairs
- [x] Contract specs + data routing
- [x] WebSocket feed + dashboard panel
- [x] API endpoints
- [x] ORB injection + backfill

### Repo Split & Cleanup
- [x] Split into three repos: futures, rb (training service), ninjatrader
- [x] Renamed `orb` → `rb` — service-only compose, hosts trained models
- [x] Removed training code from futures (moved to `lib/training`)
- [x] Added `scripts/sync_models.sh` (pulls from `rb` repo)
- [x] Fixed `sync_models.sh` GitHub repo URL from `nuniesmith/orb` → `nuniesmith/rb`
- [x] Cleaned up `pyproject.toml`

### Engine Detection — 9 New Breakout Types
- [x] Added 9 range builder functions: `_build_weekly_range`, `_build_monthly_range`, `_build_asian_range`, `_build_bbsqueeze_range`, `_build_va_range`, `_build_inside_day_range`, `_build_gap_rejection_range`, `_build_pivot_range`, `_build_fibonacci_range`
- [x] Added `DEFAULT_CONFIGS` entries for all 9 new types with tuned thresholds
- [x] Extended engine `RangeConfig` with type-specific fields (weekly lookback, Asian window, BB/KC params, VA bins, fib levels, pivot formula, etc.)
- [x] Wired all 9 new types into `detect_range_breakout()` dispatch (range build + scan + result population)
- [x] Added `BreakoutResult.extra` dict population with type-specific metadata (pivots, fib levels, gap direction, POC/VAH/VAL, etc.)
- [x] Fixed `to_dict()` JSON serialization for BBSQUEEZE squeeze fields and numpy type safety in `extra`
- [x] Added 9 new `ActionType` entries and interval constants in scheduler
- [x] Updated scheduler active-session windows to include all 13 types in multi-type sweeps
- [x] Added 81 tests: range builders, detect_range_breakout integration, detect_all_breakout_types (13 types), config completeness, enum mapping, scheduler wiring, RangeConfig fields

---
## Next Steps (Priority Order)
1. **`ninjatrader`**: Deploy patched `BreakoutStrategy.cs` to NT8 machine, compile, and verify 5-instrument startup with CNN dimension logging
2. **`futures`**: Expand CNN tabular features v5→v6 (8→12) — add `breakout_type_ord`, `asset_volatility_class`, `range_atr_ratio`, `hour_of_day`
3. **`rb`**: Full retrain with all 13 types + 5 core assets + all 9 sessions + 90 days → export ONNX v6 → commit champion
4. **`futures`**: Wire `PositionManager` into engine main loop — call `process_signal()` on breakout detections, `update_all()` on every 1m bar
5. **`futures`**: Verify `sync_models.sh` pulls new .pt from `rb` repo
6. **`ninjatrader`**: Update C# `BreakoutType` enum (13 values), `OrbCnnPredictor` (12 features), `OrbChartRenderer` (9 new box styles)
7. **`ninjatrader`**: Implement C# TP3 + EMA9 trailing + stop-and-reverse in `BreakoutStrategy.cs` (match `position_manager.py`)
8. **`ninjatrader`**: Update PowerShell to pull `.onnx` + `feature_contract.json` v6 from `rb` repo
9. **`futures`**: Add Pi deployment stage back to CI/CD when ready