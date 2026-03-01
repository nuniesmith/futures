# Futures Trading Co-Pilot

> Quality-first algorithmic futures trading system with GPU-accelerated CNN pattern
> recognition, deterministic ORB filters, and session-aware automation.

A Python + NinjaTrader system that detects Opening Range Breakouts on CME micro
futures, gates them through six research-backed filters and a hybrid CNN, sizes
risk via ATR-adaptive brackets, and routes execution through NinjaTrader's Bridge
strategy — targeting 1–3 high-conviction trades per day.

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Local Development](#local-development)
- [CNN Breakout Model](#cnn-breakout-model)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Scripts & Tools](#scripts--tools)
- [Success Metrics](#success-metrics)
- [Technologies](#technologies)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Futures Trading Co-Pilot                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Postgres    │  │    Redis     │  │ Data Service │  │   Engine   │ │
│  │  (journal,    │  │  (hot cache, │  │  (FastAPI +  │  │ (scheduler,│ │
│  │   history,    │  │   live bars, │  │   HTMX dash, │  │  analysis, │ │
│  │   risk)       │  │   focus,     │  │   REST API)  │  │  training) │ │
│  │              │  │   positions) │  │              │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         └─────────────────┴─────────────────┴────────────────┘        │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐ │
│  │                    Analysis & ML Pipeline                          │ │
│  │                                                                    │ │
│  │  Wave Analysis ─ Volatility Clustering ─ Regime Detection (HMM)   │ │
│  │  ICT/SMC (FVGs, OBs, Sweeps) ─ Volume Profile ─ CVD              │ │
│  │  Monte Carlo ─ Multi-TF Confluence ─ Signal Quality               │ │
│  │  ORB Detection ─ 6 Deterministic Filters ─ CNN Inference          │ │
│  │                                                                    │ │
│  └─────────────────────────────────┬─────────────────────────────────┘ │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐ │
│  │                    Execution Layer                                  │ │
│  │                                                                    │ │
│  │  Grok AI Analyst ─ Pre-Market Scorer ─ Risk Manager               │ │
│  │  POST /execute_signal → Bridge.cs → ATR brackets → CME order      │ │
│  │  Ruby.cs draws zones/arrows on NinjaTrader chart                  │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Four-service Docker stack:**

| Service | Role | Port |
|---|---|---|
| **Postgres** | Durable storage — trade journal, historical optimizations, risk events | 5432 |
| **Redis** | Hot cache — live bars, FKS metrics, positions, focus updates, SSE pub/sub | 6379 |
| **Data Service** | FastAPI API + HTMX dashboard — thin read layer over Redis | 8000 |
| **Engine** | Background worker — all heavy computation, analysis, training, scheduling | — |

Optional monitoring stack (Prometheus + Grafana) available via `--profile monitoring`.

---

## How It Works

### Session-Aware Scheduling

The engine operates on three Eastern Time sessions, each with different responsibilities:

| Session | Hours (ET) | What Happens |
|---|---|---|
| 🌙 **Pre-Market** | 00:00–05:00 | Compute daily focus, Grok morning briefing, prep alerts |
| 🟢 **Active** | 05:00–12:00 | Live FKS recompute (5 min), ORB detection (2 min), risk checks (1 min), Grok updates (15 min) |
| ⚙️ **Off-Hours** | 12:00–00:00 | Historical backfill, strategy optimization, backtesting, CNN dataset generation + retraining |

### The Trade Pipeline

```
1. Grok AI Morning Brief → Focus on 1–3 instruments (MGC, MES, MNQ)
                │
2. Live 1-min bars stream in via Massive WebSocket
                │
3. ORB Detection — identify the 09:30–10:00 Opening Range
                │
4. Breakout triggered — price crosses OR high/low
                │
5. Deterministic Filter Gate (majority mode):
   ├── NR7 (Narrow Range 7)
   ├── Pre-Market Range Break
   ├── Session Window
   ├── Lunch / Dead-Zone Filter
   ├── Multi-TF EMA Bias
   └── VWAP Confluence
                │
6. CNN Inference — EfficientNetV2-S scores chart image (0.0–1.0)
   ├── Advisory mode (ORB_CNN_GATE=0): enriches alert with probability
   └── Hard gate (ORB_CNN_GATE=1): blocks signals below threshold
                │
7. Risk Manager — position limits, daily loss, time rules
                │
8. POST /execute_signal → Bridge.cs → ATR-based brackets → CME order
```

### Backtest Results (Real Data, 5 Days × 3 Symbols)

| Mode | Trades | Win Rate | Profit Factor | Avg R |
|---|---|---|---|---|
| Baseline (no filters) | 15 | 66.7% | 2.67 | +0.56 |
| Majority filter gate | 11 | 72.7% | 3.56 | +0.70 |
| Strict (all filters) | 3 | 100.0% | ∞ | +1.33 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git LFS (`git lfs install`)
- (For GPU training) NVIDIA GPU with CUDA 12.x drivers

### 1. Clone & Setup

```bash
git clone https://github.com/nuniesmith/futures.git
cd futures
git lfs pull          # downloads the champion CNN model (~85 MB)
```

### 2. One-Command Docker Start

```bash
./run.sh              # creates .env, runs tests, builds & starts all services
```

This will:
- Create a Python virtualenv and install dependencies
- Generate a `.env` file with secure random secrets (you'll need to add API keys)
- Run the test suite and linter
- Build and start all four Docker services

### 3. Add Your API Keys

Edit `.env` and set:

```
MASSIVE_API_KEY=your_key_here    # https://massive.com/dashboard (real-time futures data)
XAI_API_KEY=your_key_here        # https://console.x.ai (Grok AI analyst)
```

Without `MASSIVE_API_KEY`, the system falls back to yfinance (delayed data).
Without `XAI_API_KEY`, the Grok AI analyst tab is disabled (everything else works).

### 4. Verify

```bash
docker compose ps                 # all services should be "healthy"
docker compose logs -f engine     # watch the engine schedule actions
open http://localhost:8000        # dashboard
```

---

## Docker Deployment

### Standard (4 services)

```bash
docker compose up -d --build
```

### With Monitoring (+ Prometheus & Grafana)

```bash
docker compose --profile monitoring up -d --build
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

### GPU-Enabled Engine

The engine container needs NVIDIA runtime for GPU-accelerated CNN training:

```bash
# Install NVIDIA Container Toolkit first:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Then add to docker-compose.yml engine service:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]
```

### Useful Docker Commands

```bash
docker compose logs -f engine           # follow engine logs
docker compose logs -f data             # follow API logs
docker compose exec engine bash         # shell into engine container
docker compose down                     # stop everything
docker compose down -v                  # stop + remove volumes (⚠️ deletes data)
```

---

## Local Development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                 # install project + dev dependencies
```

### GPU Dependencies (Optional)

```bash
# CUDA 12.8 (RTX 20/30/40/50 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Run Locally

```bash
# Start Postgres + Redis via Docker, run Python services locally
docker compose up -d postgres redis

# Data service (API + dashboard)
PYTHONPATH=src uvicorn lib.services.data.main:app --host 0.0.0.0 --port 8000 --reload

# Engine (background worker) — in another terminal
PYTHONPATH=src python -m lib.services.engine.main
```

### Run Tests

```bash
pytest src/tests/ -x -q --tb=short      # all tests
pytest src/tests/test_orb_filters.py -v  # specific module
ruff check src/                          # linting
```

---

## CNN Breakout Model

### Overview

A hybrid **EfficientNetV2-S + tabular** model that classifies Ruby-style chart
snapshots as "good breakout" (high follow-through) or "bad breakout" (likely chop).

| Component | Details |
|---|---|
| **Image backbone** | EfficientNetV2-S (1280-dim features), pre-trained on ImageNet |
| **Tabular head** | 6 features → 64 → 32-dim (quality %, volume ratio, ATR %, CVD delta, NR7, direction) |
| **Classifier** | Merged 1312-dim → 256 → 2 classes (bad/good) |
| **Parameters** | 20.5M total |
| **Training hardware** | NVIDIA RTX 2070 SUPER (8 GB VRAM) |
| **Best accuracy** | 84.7% on validation set |
| **Inference latency** | < 50ms per chart on GPU |

### Image Philosophy

**Chart images are generated artifacts, not static assets.** They are rendered
on-demand from historical bar data and should never be committed to git.

- **Training images** — generated off-hours from 90 days of historical bars
- **Validation images** — same pipeline, stratified 15% holdout
- **Live images** — rendered in-memory from current bars during ORB detection
- **All use the same renderer** — `render_snapshot_for_inference()` ensures zero train-serve skew

### Overnight Retraining Pipeline

The engine automatically retrains the CNN every evening:

```
Off-Hours (12:00–00:00 ET)
  │
  ├── Stage 1: Dataset Refresh — render charts from latest bars
  ├── Stage 2: Train/Val Split — stratified 85/15
  ├── Stage 3: GPU Training — mixed precision, class weighting, early stopping
  ├── Stage 4: Validation Gate — must beat champion on accuracy/precision/recall
  ├── Stage 5: Model Promotion — atomic swap of breakout_cnn_best.pt
  └── Stage 6: Cleanup — archive old checkpoints, prune stale artifacts
```

The validation gate prevents regressions — a new model is only promoted if it
meets absolute thresholds (accuracy ≥ 80%, precision ≥ 75%, recall ≥ 70%) AND
beats the current champion.

### Manual Training

```bash
# Quick retrain on existing data
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --skip-dataset --immediate

# Full pipeline (generate + train + validate + promote)
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate

# Dry run (train + validate, no promotion)
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate --dry-run

# Custom hyperparameters
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate \
    --epochs 30 --batch-size 64 --lr 2e-4 --min-accuracy 82.0
```

### Git & Model Versioning

Only the validated champion model is committed (via Git LFS). Everything else
is `.gitignored` and regenerated locally:

| Path | In Git? | Why |
|---|---|---|
| `models/breakout_cnn_best.pt` | ✅ (LFS) | Live champion — needed after clone |
| `models/breakout_cnn_best_meta.json` | ✅ | Promotion metadata (accuracy, date) |
| `models/*.pt` (others) | ❌ | Training checkpoints — ephemeral |
| `models/archive/` | ❌ | Archived previous champions — local only |
| `dataset/images/` | ❌ | Generated PNGs — regenerated from bars |
| `dataset/*.csv` | ❌ | Labels + splits — regenerated each cycle |

After cloning, run `git lfs pull` to download the champion model.

See [docs/cnn-pipeline.md](docs/cnn-pipeline.md) for the full CNN pipeline reference.

---

## Project Structure

```
futures/
├── src/
│   ├── lib/
│   │   ├── analysis/                 # Market analysis & ML modules
│   │   │   ├── breakout_cnn.py       #   Hybrid CNN: model, training, inference
│   │   │   ├── chart_renderer.py     #   Ruby-style chart image rendering
│   │   │   ├── confluence.py         #   Multi-timeframe confluence filter
│   │   │   ├── cvd.py                #   Cumulative Volume Delta + divergences
│   │   │   ├── dataset_generator.py  #   Chart dataset generation from bars
│   │   │   ├── ict.py                #   ICT/SMC: FVGs, order blocks, sweeps
│   │   │   ├── monte_carlo.py        #   Bootstrap Monte Carlo + PBO
│   │   │   ├── orb_filters.py        #   6 deterministic ORB quality filters
│   │   │   ├── orb_simulator.py      #   ORB trade simulation + auto-labeling
│   │   │   ├── regime.py             #   HMM market regime detection
│   │   │   ├── scorer.py             #   Pre-market instrument scoring
│   │   │   ├── signal_quality.py     #   FKS signal quality score (Pine port)
│   │   │   ├── volatility.py         #   K-Means adaptive vol clustering
│   │   │   ├── volume_profile.py     #   POC, VAH/VAL, naked POCs
│   │   │   └── wave_analysis.py      #   Wave dominance tracking (Pine port)
│   │   │
│   │   ├── core/                     # Infrastructure
│   │   │   ├── alerts.py             #   Alert dispatch (email, webhook)
│   │   │   ├── cache.py              #   Redis cache + data source abstraction
│   │   │   ├── logging_config.py     #   Structured logging (structlog)
│   │   │   └── models.py             #   Database models + Postgres/SQLite ORM
│   │   │
│   │   ├── integrations/             # External services
│   │   │   ├── grok_helper.py        #   xAI Grok AI analyst (briefing + live)
│   │   │   └── massive_client.py     #   Massive.com REST + WebSocket client
│   │   │
│   │   ├── trading/                  # Trading engine
│   │   │   ├── engine.py             #   DashboardEngine: FKS, optimization, backtest
│   │   │   ├── strategies.py         #   10 backtesting strategies (Optuna-tunable)
│   │   │   └── costs.py              #   CME slippage + commission model
│   │   │
│   │   └── services/                 # Deployable services
│   │       ├── data/                 #   FastAPI data service
│   │       └── engine/               #   Background engine service
│   │           ├── main.py           #     Action handlers + main loop
│   │           ├── scheduler.py      #     Session-aware scheduling
│   │           ├── orb.py            #     ORB detection logic
│   │           ├── risk.py           #     Risk manager
│   │           ├── focus.py          #     Daily focus computation
│   │           ├── backfill.py       #     Historical bar backfill
│   │           └── patterns.py       #     Pattern detection
│   │
│   ├── ninjatrader/                  # NinjaTrader C# components
│   │   ├── Ruby.cs                   #   Chart overlay (ORB box, zones, arrows)
│   │   ├── Bridge.cs                 #   Execution strategy (brackets, risk)
│   │   ├── SignalBus.cs              #   Signal routing between indicators
│   │   └── BACKTEST_GUIDE.md         #   NinjaTrader backtesting guide
│   │
│   ├── pinescript/
│   │   └── ruby.pine                 # TradingView Pine Script (original FKS)
│   │
│   └── tests/                        # Pytest test suite (25 test modules)
│
├── scripts/
│   ├── retrain_overnight.py          # CNN retraining pipeline orchestrator
│   ├── train_gpu.py                  # GPU-optimized standalone trainer
│   ├── backtest_filters.py           # ORB filter backtest comparison tool
│   ├── generate_sample_bars.py       # Synthetic bar data generator
│   ├── migrate_git_lfs.sh            # One-time Git LFS migration
│   └── analyze.sh                    # Analysis helper
│
├── config/
│   ├── grafana/                      # Grafana provisioning + dashboards
│   └── prometheus/                   # Prometheus scrape config
│
├── docker/
│   ├── data/Dockerfile               # Data service container
│   ├── engine/Dockerfile             # Engine container
│   └── monitoring/                   # Prometheus + Grafana Dockerfiles
│
├── models/                           # CNN model artifacts
│   ├── breakout_cnn_best.pt          #   Champion model (Git LFS)
│   └── breakout_cnn_best_meta.json   #   Promotion metadata
│
├── dataset/                          # Generated training data (gitignored)
│   └── images/                       #   Chart PNGs (regenerated from bars)
│
├── docs/
│   └── cnn-pipeline.md               # CNN pipeline deep-dive documentation
│
├── docker-compose.yml                # 4-service + monitoring stack
├── pyproject.toml                    # Python project config (hatch + deps)
├── run.sh                            # One-command build + deploy script
├── todo.md                           # Project status & phase tracking
└── .env.example                      # Environment variable template
```

---

## Configuration

### Environment Variables

#### Required

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL password (auto-generated by `run.sh`) |
| `REDIS_PASSWORD` | Redis password (auto-generated by `run.sh`) |

#### API Keys

| Variable | Description | Fallback |
|---|---|---|
| `MASSIVE_API_KEY` | [Massive.com](https://massive.com) real-time futures data | yfinance (delayed) |
| `XAI_API_KEY` | [xAI](https://console.x.ai) Grok AI analyst | AI features disabled |

#### Trading

| Variable | Default | Description |
|---|---|---|
| `ACCOUNT_SIZE` | `150000` | Account size for risk calculations |
| `ORB_FILTER_GATE` | `majority` | Filter mode: `all`, `majority`, or `none` |
| `ORB_CNN_GATE` | `0` | `0` = CNN advisory, `1` = CNN hard gate |

#### CNN Retraining

| Variable | Default | Description |
|---|---|---|
| `CNN_RETRAIN_SYMBOLS` | `MGC,MES,MNQ` | Symbols for dataset generation |
| `CNN_RETRAIN_DAYS_BACK` | `90` | Days of history to process |
| `CNN_RETRAIN_EPOCHS` | `25` | Training epochs |
| `CNN_RETRAIN_BATCH_SIZE` | `64` | Batch size (64 fits 8 GB VRAM) |
| `CNN_RETRAIN_LR` | `2e-4` | Peak learning rate |
| `CNN_RETRAIN_PATIENCE` | `8` | Early stopping patience |
| `CNN_RETRAIN_MIN_ACC` | `80.0` | Minimum val accuracy to promote |
| `CNN_RETRAIN_MIN_PRECISION` | `75.0` | Minimum precision to promote |
| `CNN_RETRAIN_MIN_RECALL` | `70.0` | Minimum recall to promote |
| `CNN_RETRAIN_IMPROVEMENT` | `0.0` | Required accuracy gain over champion |

### Key Defaults

| Setting | Value | Location |
|---|---|---|
| CNN inference threshold | 0.82 | `breakout_cnn.py` |
| Bracket stop-loss | 1.5 × ATR | `orb_simulator.py` |
| Bracket take-profit 1 | 2.0 × ATR | `orb_simulator.py` |
| Bracket take-profit 2 | 3.0 × ATR | `orb_simulator.py` |
| Max hold time | 120 bars (2 hours) | `dataset_generator.py` |
| Chart image size | 224 × 224 px | `breakout_cnn.py` |
| FKS recompute interval | 5 minutes | `scheduler.py` |
| Grok update interval | 15 minutes | `scheduler.py` |
| Risk check interval | 1 minute | `scheduler.py` |

---

## Testing

The project has 25 test modules covering analysis, services, and integrations:

```bash
# Full test suite
pytest src/tests/ -x -q --tb=short

# Specific modules
pytest src/tests/test_orb_filters.py -v       # ORB filter logic
pytest src/tests/test_scheduler.py -v          # session scheduling
pytest src/tests/test_risk.py -v               # risk manager
pytest src/tests/test_ict.py -v                # ICT/SMC concepts
pytest src/tests/test_cvd.py -v                # cumulative volume delta
pytest src/tests/test_volume_profile.py -v     # volume profile analysis
pytest src/tests/test_data_service.py -v       # FastAPI endpoints

# With coverage
pytest src/tests/ --cov=lib --cov-report=html

# Linting
ruff check src/
```

---

## Scripts & Tools

### Overnight CNN Retraining

```bash
# Full pipeline (respects trading session windows)
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py

# Immediate mode (run all stages now, ignore time windows)
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate

# Retrain on existing data only
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --skip-dataset --immediate

# Validate a candidate without promoting
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --dry-run --immediate
```

### GPU Training (Standalone)

```bash
PYTHONPATH=src .venv/bin/python scripts/train_gpu.py \
    --epochs 25 --batch-size 64 --lr 2e-4 --freeze-epochs 3
```

### Filter Backtesting

```bash
# Source API keys
export $(grep -v '^#' .env | xargs)

# Real data via Massive API (majority gate mode)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
    --symbols MGC MES MNQ --source massive --days 30 \
    --gate-mode majority -v --export data/backtest_results.csv
```

### Dataset Generation

```bash
# Generate chart images from historical bars
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator generate \
    --symbols MGC MES MNQ --days 90 --source cache

# Split into train/val
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator split \
    --csv dataset/labels.csv --val-frac 0.15

# Validate dataset integrity
PYTHONPATH=src .venv/bin/python -m lib.analysis.dataset_generator validate \
    --csv dataset/labels.csv
```

### Synthetic Data (No API Needed)

```bash
.venv/bin/python scripts/generate_sample_bars.py \
    --symbols MGC MES MNQ --days 60 --seed 42
```

### CNN Model Info

```bash
PYTHONPATH=src .venv/bin/python -m lib.analysis.breakout_cnn info
```

### Git LFS Migration

If your repo has images/checkpoints committed directly (pre-LFS setup):

```bash
bash scripts/migrate_git_lfs.sh
```

---

## Success Metrics

| Metric | Target | Current |
|---|---|---|
| Trade frequency | 1–3 per day on focus assets | — |
| Win rate | 58–65% (with 1:2+ R:R) | 72.7% (backtest, majority gate) |
| Max drawdown | < 8% on micro sizing | — |
| CNN validation accuracy | > 82% | 84.7% ✅ |
| CNN inference latency | < 50ms per chart (GPU) | ~30ms |
| Daily retrain | Complete before 05:00 ET | ~23 min |

---

## Technologies

| Layer | Stack |
|---|---|
| **Language** | Python 3.11+, C# (NinjaTrader), Pine Script |
| **ML** | PyTorch, EfficientNetV2-S, mixed precision (AMP) |
| **Web** | FastAPI, HTMX, Jinja2 |
| **Data** | Massive.com (real-time CME), yfinance (fallback), pandas |
| **Storage** | PostgreSQL 16, Redis 7 |
| **AI** | xAI Grok (morning briefing + live updates) |
| **Backtesting** | backtesting.py, Optuna (Bayesian optimization) |
| **Observability** | structlog, Prometheus, Grafana |
| **Deployment** | Docker Compose, Git LFS |
| **Execution** | NinjaTrader 8 (Ruby.cs + Bridge.cs) |

---

## License

[MIT](LICENSE) — Copyright (c) 2026 nuniesmith