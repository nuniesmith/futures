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
- [ ] **Add engine detection for new 9 breakout types** — Wire WEEKLY, MONTHLY, ASIAN, BBSQUEEZE, VA, INSIDE, GAP, PIVOT, FIB into `detect_all_breakout_types()` with proper range builders
- [ ] **Add session-level performance stats** to daily report
- [ ] **Backfill gap detection** — alert when historical bars have gaps > N minutes

### Dashboard & Web UI
- [ ] **Add "Breakout Type" filter + MTF score column** to signal history table
- [ ] Trade journal UI improvements — inline editing, tag filtering
- [ ] Kraken crypto price chart — live candlestick chart in dashboard for tracked pairs
- [ ] Crypto/futures correlation panel — show BTC vs MES/MGC correlation in sidebar

### Training & Dataset (shared `lib`)
- [ ] **Per-type model heads or type-embedding** in CNN
- [ ] **Session-specific training thresholds** (per `SESSION_THRESHOLDS`)
- [ ] **Automated good/bad balancing** across all 13 types + 9 sessions
- [ ] Synthetic data augmentation (noise, time shifts)
- [ ] CLI command: `python -m lib.training.dataset_generator generate --symbols MGC MES --session all --breakout-type all`
- [ ] Dashboard preview: view random good/bad snapshots per type/session

### Infrastructure
- [ ] Rate limiting tuning — review slowapi config for SSE vs REST endpoints
- [ ] Deployment pipeline — add Pi deploy stage back to CI/CD when ready
- [ ] Auto-sync trained models from `rb` repo to engine (rsync/scp post-train hook)

### Monitoring
- [ ] Prometheus metrics: `images_generated`, `label_balance`, `render_time` (trainer)
- [ ] Grafana panel: "Training Data Health" (win-rate per type/session)

---
## Active — `rb` repo (`~/github/rb`)

### Trainer Service
- [ ] **End-to-end trainer smoke test** — `docker compose up` on GPU rig, `POST /train` with `--symbols MGC MES MNQ KRAKEN:XBTUSD --session all --breakout-type all --days 90`
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
  - Match Python 3-phase bracket logic exactly

### Model Pull
- [ ] **Update PowerShell model pull script** to fetch `.onnx` + `feature_contract.json` v6 from `rb` repo

---
## Completed

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
- [x] Cleaned up `pyproject.toml`

---
## Next Steps (Priority Order)
1. **`rb`**: Smoke-test trainer on GPU rig — `docker compose up`, `POST /train --breakout-type all --session all`
2. **`rb`**: Retrain with all 13 types + TP3 → export ONNX v6 → commit champion models
3. **`futures`**: Verify `sync_models.sh` pulls new .pt from `rb` repo
4. **`ninjatrader`**: Update C# `BreakoutType` enum (13 values), `OrbCnnPredictor`, `OrbChartRenderer` (9 new box styles)
5. **`ninjatrader`**: Implement C# TP3 + EMA9 trailing in `BreakoutStrategy.cs`
6. **`ninjatrader`**: Update PowerShell to pull `.onnx` + `feature_contract.json` v6 from `rb` repo
7. **`futures`**: Add Pi deployment stage back to CI/CD when ready