# Ruby + Bridge ORB System — Project Status & Next Steps

> **Goal:** Evolve the Ruby + Bridge ORB system into a quality-first pipeline
> with deterministic filters and an image-based CNN. Target: 1–3 high-conviction
> trades/day on micro contracts, scaling to normal size once proven.

---

## Architecture Overview

```
Python Engine (Data Service + Scheduler)
  │
  ├─ Grok + Focus → 1–3 assets
  ├─ Live 1m bars (Massive WS)
  ├─ ORB detection (src/lib/services/engine/orb.py)
  ├─ Deterministic filter gate (NR7, pre-market, session, lunch, multi-TF, VWAP)
  └─ CNN inference (optional — EfficientNetV2-S + tabular)
        │
        ▼
  POST /execute_signal  →  Bridge.cs  →  risk sizing + brackets  →  order
        │
  Ruby.cs draws arrows/zones on chart (visual confirmation)
```

**Backtests are unchanged:** Ruby → SignalBus → Bridge.
**Live path adds:** Python Engine filters + optional CNN before calling Bridge HTTP.

---

## ✅ Phase 1 — Completed

### Deterministic ORB Filters
- [x] **NR7 (Narrow Range 7)** — Crabel filter; narrowest daily range of prior 7 days → quality boost
- [x] **Pre-Market Range Break** — Globex high/low confluence check
- [x] **Session Windows** — only allow signals inside configurable ET windows
- [x] **Lunch / Dead-Zone Filter** — reject 10:30–13:30 ET chop
- [x] **Multi-TF EMA Bias** — 15m/HTF trend must agree with ORB direction
- [x] **VWAP Confluence** — price vs session VWAP alignment
- [x] **Composite `apply_all_filters()`** — runs all filters, returns pass/fail + summary
- [x] **Unit tests** — comprehensive coverage in `src/tests/test_orb_filters.py`

| File | Description |
|------|-------------|
| `src/lib/analysis/orb_filters.py` | All filter functions + `ORBFilterResult` dataclass |
| `src/tests/test_orb_filters.py` | Filter unit tests |

### Engine Integration (Quality Gate)
- [x] `_handle_check_orb()` runs filter gate before publishing ORB alerts
- [x] Breakouts that fail hard filters are logged but **not** published
- [x] Filter results included in alert messages and audit trail
- [x] Graceful fallback — if filter module missing, breakouts pass unfiltered

| File | Key function |
|------|-------------|
| `src/lib/services/engine/main.py` | `_handle_check_orb()` (L379–618) |

### ML Pipeline Scaffold
- [x] **Chart Renderer** — Ruby-style snapshots (ORB box, EMA9, VWAP, quality badge, nightclouds theme)
- [x] **ORB Simulator / Auto-Labeler** — replays ORB logic + Bridge brackets → ground-truth labels
- [x] **Dataset Generator** — sliding-window batch job, writes `dataset/labels.csv` + images
- [x] **Hybrid CNN** — EfficientNetV2-S (1280-dim) + tabular head (6 features → 32-dim) → classifier
- [x] **Training pipeline** — AdamW, freeze/unfreeze backbone, model saved to `models/`
- [x] **Inference API** — `predict_breakout()`, `predict_breakout_batch()`, thread-safe model caching
- [x] **CLI** — `python -m src.lib.analysis.breakout_cnn train|predict|info`

| File | Description |
|------|-------------|
| `src/lib/analysis/chart_renderer.py` | `render_ruby_snapshot()`, `render_snapshot_for_inference()`, `RenderConfig` |
| `src/lib/analysis/orb_simulator.py` | `simulate_orb_outcome()`, `simulate_batch()`, `BracketConfig` |
| `src/lib/analysis/dataset_generator.py` | `generate_dataset()`, `split_dataset()`, `validate_dataset()`, `DatasetConfig` |
| `src/lib/analysis/breakout_cnn.py` | `HybridBreakoutCNN`, `train_model()`, `predict_breakout()`, `model_info()` |

### Scheduler & Exports
- [x] `GENERATE_CHART_DATASET` action — runs dataset generation during off-hours
- [x] `TRAIN_BREAKOUT_CNN` action — trains CNN after dataset generation completes
- [x] `__init__.py` exports all new modules with guarded optional imports

| File | Description |
|------|-------------|
| `src/lib/services/engine/scheduler.py` | `ActionType` enum + `_get_off_hours_actions()` |
| `src/lib/services/engine/main.py` | `_handle_generate_chart_dataset()`, `_handle_train_breakout_cnn()` |
| `src/lib/analysis/__init__.py` | Lazy/guarded re-exports for all analysis modules |

### Dependencies
- [x] `pyproject.toml` updated — `mplfinance`, `Pillow` in main deps; `torch`, `torchvision`, `torchaudio` in `[gpu]` optional group

---

## ✅ Phase 2 — Validate Filters (Complete)

> **Objective:** Prove the deterministic filters improve metrics before adding ML.
> This is the highest-ROI next step — no GPU, no new deps, just backtesting.

### Infrastructure & Fixes

- [x] **Docker stack fully operational** — all 4 services healthy (`postgres`, `redis`, `engine`, `data`)
- [x] **Added `httpx` to main deps** in `pyproject.toml` (was only in `dev`; data-service positions API requires it)
- [x] **Fixed `dataset_generator.py`** — `_load_bars_from_massive()` referenced non-existent `MassiveClient`; now uses `get_massive_provider()` + `MassiveDataProvider.get_aggs()` with proper Yahoo-style ticker resolution
- [x] **Fixed `_load_bars_from_cache()`** — now uses `get_data()` (hashed `futures:*` keys) instead of non-existent `engine:bars_1m:*` keys
- [x] **Added symbol→ticker mapping** — `_resolve_ticker()` converts short symbols (`MGC`) to Yahoo tickers (`MGC=F`) for cache and Massive API compatibility
- [x] **Massive API + xAI keys** configured in `.env`; engine pulling live data for 6 assets
- [x] **Massive WebSocket** connected — streaming `AM.*` (minute aggs) and `T.*` (trades)
- [x] **Grok/xAI morning brief** running successfully via scheduler

### Backtest Tooling

- [x] **Backtest comparison script** — `scripts/backtest_filters.py`
  - Runs ORB simulation per day, evaluates all filters, prints side-by-side BASELINE vs FILTERED table
  - Per-filter rejection breakdown (shows which filters remove the most trades)
  - Supports CSV, Redis cache, or Massive API as data sources
  - Exports per-trade detail to CSV (`--export results.csv`) with individual filter pass/fail columns
  - Gate modes: `--gate-mode all` (strict) or `--gate-mode majority` (permissive)
  - Configurable brackets: `--sl-mult`, `--tp1-mult`
  - Verbose mode (`-v`) shows every trade as it's processed
- [x] **Synthetic bar generator** — `scripts/generate_sample_bars.py`
  - Generates realistic 1m OHLCV + daily bars for any symbol (MGC, MES, MNQ, MCL, or custom)
  - Simulates pre-market, opening range, breakouts, lunch chop, NR7 days, volume spikes
  - Reproducible with `--seed`; writes to `data/bars/{SYMBOL}_1m.csv` + `{SYMBOL}_daily.csv`

| File | Description |
|------|-------------|
| `scripts/backtest_filters.py` | Filter backtest comparison — CLI + `run_backtest()` API |
| `scripts/generate_sample_bars.py` | Synthetic bar data generator for testing |

### Real-Data Backtest Results (5 trading days, 2026-02-23 → 2026-02-27)

- [x] **2.1** Run backtests on **real** bar data via Massive API (`--source massive --days 5`)
- [x] **2.2** Run on MGC, MES, and MNQ with real data
- [x] **2.3** Document results — see tables below

#### Gate mode: `all` (strict — trade must pass ALL hard filters)

| Symbol | Baseline Trades | Filtered Trades | Removed | Baseline WR | Filtered WR | Baseline PF | Filtered PF | Baseline R | Filtered R |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **MGC** | 5 | 1 | 80% | 80.0% | 100.0% | 5.33 | ∞ | +4.33 | +1.33 |
| **MES** | 5 | 1 | 80% | 80.0% | 100.0% | 5.33 | ∞ | +4.33 | +1.33 |
| **MNQ** | 5 | 1 | 80% | 40.0% | 100.0% | 0.89 | ∞ | −0.33 | +1.33 |
| **ALL** | **15** | **3** | **80%** | **66.7%** | **100.0%** | **2.67** | **∞** | **+8.33** | **+4.00** |

Per-filter rejection rates (`all` mode):
| Filter | Passed | Rejected | Reject % |
|--------|:-:|:-:|:-:|
| NR7 | 15 | 0 | 0.0% |
| Session Window | 13 | 2 | 13.3% |
| Lunch Filter | 13 | 2 | 13.3% |
| Pre-Market Range | 9 | 6 | **40.0%** |
| Multi-TF Bias | 9 | 6 | **40.0%** |
| VWAP Confluence | 8 | 7 | **46.7%** |

#### Gate mode: `majority` (permissive — trade passes if >50% of hard filters pass)

| Symbol | Baseline Trades | Filtered Trades | Removed | Baseline WR | Filtered WR | Baseline PF | Filtered PF | Baseline R | Filtered R |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **MGC** | 5 | 5 | 0% | 80.0% | 80.0% | 5.33 | 5.33 | +4.33 | +4.33 |
| **MES** | 5 | 3 | 40% | 80.0% | 66.7% | 5.33 | 2.67 | +4.33 | +1.67 |
| **MNQ** | 5 | 3 | 40% | 40.0% | 66.7% | 0.89 | 2.67 | −0.33 | +1.67 |
| **ALL** | **15** | **11** | **27%** | **66.7%** | **72.7%** | **2.67** | **3.56** | **+8.33** | **+7.67** |

#### Key Findings

1. **`majority` mode is the better default** — removes 27% of trades while improving WR (+6%), PF (+0.89), and avg R/trade (+0.14R). Retains 73% of trade flow.
2. **`all` mode is too aggressive** — removes 80% of trades. Perfect 100% WR but only 3 trades in 5 days across 3 symbols. Leaves R on the table.
3. **MNQ benefits most from filtering** — baseline was 40% WR / 0.89 PF / −0.33R. Both gate modes flip MNQ to profitable.
4. **Top rejection filters** — Pre-Market Range (40%), Multi-TF Bias (40%), VWAP Confluence (47%) are the three most aggressive. NR7 never rejects (it's a soft quality boost only).
5. **Avg R/trade improves under both modes** — strict: +0.56 → +1.33; majority: +0.56 → +0.70.

### Remaining steps

- [x] **2.4** Tune filter parameters — **recommendation: switch default gate mode from `all` to `majority`**
  - `majority` mode delivers the best balance: meaningful WR/PF improvement while keeping trade volume viable
  - VWAP Confluence and Pre-Market Range are the most aggressive — consider relaxing tolerance thresholds if more trades are needed
- [ ] **2.5** Run live paper trades on Sim100 for 2–3 sessions with filters on (`majority` mode)
- [ ] **2.6** Accumulate 20–30 trading days of real data to strengthen statistical significance
  - Re-run: `export $(grep -v '^#' .env | xargs) && PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py --symbols MGC MES MNQ --source massive --days 30 --gate-mode majority -v --export data/backtest_results_30d.csv`

### Run commands

```bash
# Source env vars for API access
export $(grep -v '^#' .env | xargs)

# Real-data backtest via Massive API (strict mode)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
  --symbols MGC MES MNQ --source massive --days 5 -v --export data/backtest_results.csv

# Real-data backtest (majority mode — recommended)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
  --symbols MGC MES MNQ --source massive --days 5 --gate-mode majority -v --export data/backtest_results_majority.csv

# Longer lookback (30 days, when available)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
  --symbols MGC MES MNQ --source massive --days 30 --gate-mode majority -v --export data/backtest_results_30d.csv

# Generate synthetic data for testing (no API needed)
.venv/bin/python scripts/generate_sample_bars.py --symbols MGC MES MNQ --days 60 --seed 42
```

---

## ✅ Phase 3 — Generate First Dataset (Complete)

> **Objective:** Produce labeled chart images for CNN training.

### Completed

- [x] **3.1** `mplfinance` (0.12.10b0) and `Pillow` (12.1.1) installed and verified in `.venv`
- [x] **3.2** Generated dataset from 30 days of real Massive API data (MGC, MES, MNQ)
- [x] **3.3** Visual QA — inspected images across all 3 symbols, both directions, all 4 labels
  - ✅ ORB box (gold shaded region with dashed high/low lines)
  - ✅ EMA9 overlay (DodgerBlue)
  - ✅ VWAP overlay (Gold)
  - ✅ Quality badge (top-left, color-coded)
  - ✅ Direction indicator (top-right, ▲ LONG green / ▼ SHORT pink)
  - ✅ Volume panel (colored by candle direction)
  - ✅ Nightclouds dark theme throughout
- [x] **3.4** Label distribution verified — well-balanced across symbols and labels
- [x] **3.5** Train/val split created (85/15)

### Dataset Stats (30 days, 2026-01-29 → 2026-02-27)

| Metric | Value |
|--------|-------|
| **Total images** | 477 |
| **Total windows simulated** | 2,965 |
| **Trade rate** | 16.1% of windows |
| **Render failures** | 0 |
| **Missing images** | 0 |
| **Train / Val split** | 408 / 69 (85% / 15%) |
| **Disk usage** | 80.3 MB (172 KB avg/image) |
| **Generation time** | 196 seconds |
| **Source** | Massive API, 1m bars, ~30k bars/symbol |

#### Label Distribution

| Label | Count | % |
|-------|:-----:|:-:|
| good_long | 152 | 31.9% |
| good_short | 133 | 27.9% |
| bad_long | 112 | 23.5% |
| bad_short | 80 | 16.8% |
| **GOOD (total)** | **285** | **59.7%** |
| **BAD (total)** | **192** | **40.3%** |

Binary balance ratio: 0.67 (acceptable for training without heavy class weighting)

#### Per-Symbol Breakdown

| Symbol | Total | Good | Bad | Sim WR |
|--------|:-----:|:----:|:---:|:------:|
| MGC | 158 | 89 | 69 | 56.3% |
| MES | 160 | 100 | 60 | 62.5% |
| MNQ | 159 | 96 | 63 | 60.4% |

#### CSV Columns (22 features)

`image_path`, `label`, `symbol`, `direction`, `quality_pct`, `volume_ratio`, `atr_pct`, `cvd_delta`, `nr7_flag`, `entry`, `sl`, `tp1`, `or_high`, `or_low`, `or_range`, `atr`, `pnl_r`, `hold_bars`, `outcome`, `breakout_time`, `pm_high`, `pm_low`

### Run Commands

```bash
# Source env vars for Massive API
export $(grep -v '^#' .env | xargs)

# Generate dataset (30 days, 3 symbols)
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator generate \
  --symbols MGC MES MNQ --days 30 --source massive \
  --window-size 240 --step-size 30 --dpi 150

# Split into train/val
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator split \
  --csv dataset/labels.csv --val-frac 0.15 --seed 42

# Validate dataset integrity
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator validate \
  --csv dataset/labels.csv
```

### Scaling Notes

- Current: 477 images from 30 days × 3 symbols (step=30, window=240)
- To scale: decrease `--step-size` to 15 → ~950 images; add more symbols (MCL, MYM)
- Massive API supports 3mo history (50k bars) — could yield 1,500+ images per symbol
- For 10k+ images needed for serious CNN training: accumulate 90 days × 6 symbols, or run nightly incremental builds via the scheduler

---

## ✅ Phase 4 — Train First CNN Model (Complete)

> **Objective:** Train the EfficientNetV2-S hybrid model and evaluate.

- [x] **4.1** Install PyTorch (CPU — no GPU available on this host)
  ```
  pip install torch torchvision tqdm --index-url https://download.pytorch.org/whl/cpu
  ```
  Installed: PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu
- [x] **4.2** Expand dataset from 477 → 812 images
  - Added 3 new symbols: MYM, M2K (MCL unavailable on Yahoo fallback)
  - Step-size 15 for higher density
  - Final split: 693 train / 119 val
  - Label distribution: good_long=264, good_short=259, bad_long=143, bad_short=146
  - Binary: 523 good / 289 bad (~64/36)
- [x] **4.3** Train first model (8 epochs, 2-phase: frozen backbone → fine-tune)
  ```
  python -m src.lib.analysis.breakout_cnn train --csv dataset/train.csv --val-csv dataset/val.csv --epochs 8 --batch-size 32 --freeze-epochs 2
  ```
  - Best val accuracy: **68.9%** (epoch 7) — saved as `models/breakout_cnn_best.pt`
  - Model size: 79.2 MB (20.5M params — EfficientNetV2-S + tabular head)
  - Training time: ~11 min on CPU
- [x] **4.4** Inspect model info
  ```
  python -m src.lib.analysis.breakout_cnn info
  ```
- [x] **4.5** Test inference on sample images
  - Single prediction and 20-sample batch both work correctly
  - Probability separation visible: "good" trades → 0.6–0.77, "bad" trades → 0.47–0.59
  - 65% accuracy on random 20-sample spot-check (threshold=0.5)

**Result:** Model functional but accuracy limited by dataset size (812 images). Probability
distribution already shows separation — more data will sharpen the decision boundary.
Accuracy target of 81–87% requires scaling to 5k–10k+ images.

### Training Progression
| Epoch | Phase     | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|-----------|----------|---------|
| 1     | frozen    | 0.6781    | 58.6%     | 0.6441   | 64.7%   |
| 2     | frozen    | 0.6673    | 63.8%     | 0.6521   | 64.7%   |
| 3     | fine-tune | 0.6488    | 64.9%     | 0.6326   | 63.9%   |
| 4     | fine-tune | 0.6370    | 63.7%     | 0.6204   | 65.5%   |
| 5     | fine-tune | 0.6153    | 65.6%     | 0.6059   | 65.5%   |
| 6     | fine-tune | 0.6060    | 66.1%     | 0.6005   | 65.5%   |
| 7     | fine-tune | 0.5967    | 68.3%     | 0.5916   | **68.9%** ← best |

---

## ✅ Phase 5 — Live Integration (Complete) / 🔲 Paper Trading (Pending)

> **Objective:** Wire CNN into the live ORB path and validate on paper.

- [x] **5.1** Switch engine filter gate to `majority` mode (recommended default)
  - Added `ORB_FILTER_GATE` env var (default: `"majority"`) to engine and docker-compose
  - Engine now passes `gate_mode` to `apply_all_filters()` instead of hardcoded `"all"`
- [x] **5.2** Add CNN inference to `_handle_check_orb()` after filter gate passes
  - Renders chart snapshot via `render_snapshot_for_inference()`
  - Runs `predict_breakout()` with tabular features (quality, ATR, direction)
  - Includes `cnn_prob` and `cnn_confidence` in log output and alert payload
  - CNN is **advisory by default** — enriches alerts but does not gate publishing
  - Optional hard gate: set `ORB_CNN_GATE=1` env var to block low-probability signals
  - Graceful degradation: CNN failures are non-fatal, system continues without inference
  - Automatic cleanup of inference images (30-minute TTL)
- [x] **5.3** Added `ORB_CNN_GATE` env var to docker-compose (default: `0` / disabled)
- [x] **5.4** Added volume mounts for `./models` and `./dataset` in docker-compose
- [ ] **5.5** Paper trade on Sim100 with CNN enrichment for 1 week (1 micro contract)
- [ ] **5.6** Compare paper results: majority gate + CNN vs. filter-only baseline (Phase 2)
- [ ] **5.7** Once accuracy > 80%, enable `ORB_CNN_GATE=1` for hard gating

---

## 🔲 Phase 6 — Dashboard & Monitoring

> **Objective:** Visibility into what the system is doing.

- [ ] **6.1** Add dashboard panel showing per-signal: `cnn_prob`, `quality_pct`, chart image
- [ ] **6.2** Show model version and training stats (`model_info()`)
- [ ] **6.3** Track filter vs. published counts (already logged — surface in dashboard)
- [ ] **6.4** Monitor `dataset_stats.json` for dataset health

---

## 🔲 Phase 7 — Docker & Production Hardening

- [ ] **7.1** Update `Dockerfile` to include `mplfinance`, `Pillow`, and optional GPU deps
- [x] **7.2** Add volume mounts: `./dataset:/app/dataset`, `./models:/app/models` — done in docker-compose
- [x] **7.3** Add env vars: `ORB_FILTER_GATE`, `ORB_CNN_GATE` — done in docker-compose
- [ ] **7.4** Add health check for model file existence
- [ ] **7.5** Validate CUDA availability inside container (`torch.cuda.is_available()`)
- [ ] **7.6** Rebuild engine container with PyTorch wheels for in-container CNN inference

---

## 🔲 Phase 8 — Iterate & Scale (Ongoing)

- [ ] **8.1** Manual QA workflow — review a sample of auto-labeler labels before large-scale training
- [ ] **8.2** Tune retraining cadence — start weekly, move to daily once stable
- [ ] **8.3** Expand to more symbols as confidence grows
- [ ] **8.4** Consider hybrid CNN+ViT architecture (MobileViT, EfficientFormer) once dataset exceeds 100k images
- [ ] **8.5** Add `predict_breakout_batch()` to the live path for low-latency multi-asset inference

---

## Key Configuration & Defaults

| Setting | Value | Location |
|---------|-------|----------|
| CNN inference threshold | `0.82` | `breakout_cnn.py` → `DEFAULT_THRESHOLD` |
| Bracket SL | 1.5 × ATR | `orb_simulator.py` → `BracketConfig` |
| Bracket TP1 | 2.0 × ATR | `orb_simulator.py` → `BracketConfig` |
| Bracket TP2 | 3.0 × ATR | `orb_simulator.py` → `BracketConfig` |
| Max hold bars | 120 (2 hours) | `dataset_generator.py` → `DatasetConfig` |
| Chart DPI | 150 (dataset), 180 (live) | `chart_renderer.py` → `RenderConfig` |
| Image size | 224 × 224 | `breakout_cnn.py` → `IMAGE_SIZE` |
| Model backbone | EfficientNetV2-S (1280-dim) | `breakout_cnn.py` → `HybridBreakoutCNN` |
| Tabular features | quality_pct, volume_ratio, atr_pct, cvd_delta, nr7_flag, direction_flag | `breakout_cnn.py` → `TABULAR_FEATURES` |
| Dataset output | `dataset/labels.csv` + `dataset/images/` | `dataset_generator.py` → `DatasetConfig` |
| Model output | `models/breakout_cnn_*.pt` | `breakout_cnn.py` → `DEFAULT_MODEL_DIR` |

---

## Quick Commands

```bash
# Run filter tests
.venv/bin/python -m pytest src/tests/test_orb_filters.py -v

# Generate dataset (off-hours, needs mplfinance + historical bars)
python -m src.lib.analysis.dataset_generator generate --symbols MGC MES MNQ --days 90 --source cache

# Train model (needs PyTorch + GPU)
python -m src.lib.analysis.breakout_cnn train --csv dataset/labels.csv --epochs 8 --batch-size 32

# Check model info
python -m src.lib.analysis.breakout_cnn info

# Predict single chart
python -m src.lib.analysis.breakout_cnn predict --image path/to/chart.png --features 0.87 1.2 0.001 0.0 1 1

# Install GPU deps (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install charting deps
pip install mplfinance Pillow
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Trade frequency | 1–3 / day on focus assets |
| Win rate | 58–65% (with 1:2+ R:R) |
| Max drawdown | < 8% on $50k micro sizing |
| CNN hold-out accuracy | > 82% |
| CNN inference latency | < 50 ms per chart (CUDA) |
| Daily retrain | Complete before 04:30 ET |
