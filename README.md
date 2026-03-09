# Ruby Futures

> Live dashboard, market stats & web UI for futures trading — real-time range breakout detection,
> session-aware scheduling, and a full HTMX dashboard powered by FastAPI.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Local Development](#local-development)
- [CNN Model Training](#cnn-model-training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Scripts & Tools](#scripts--tools)
- [Technologies](#technologies)
- [License](#license)

---

## Architecture

Everything lives in this single repo. There is no external training repo — models are trained
by the built-in trainer service and stored in `models/`.


```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Ruby Futures                                 │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Postgres   │  │    Redis     │  │ Data Service │  │   Engine   │  │
│  │  (journal,   │  │  (hot cache, │  │  (FastAPI +  │  │ (scheduler,│  │
│  │   history,   │  │   live bars, │  │   HTMX dash, │  │  analysis, │  │
│  │   risk)      │  │   positions) │  │   REST API,  │  │  ORB, risk │  │
│  │              │  │              │  │   SSE)       │  │  scoring)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         └─────────────────┴─────────────────┴────────────────┘         │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐  │
│  │                    Analysis Pipeline                               │  │
│  │                                                                   │  │
│  │  Wave Analysis · Volatility Clustering · Regime Detection (HMM)   │  │
│  │  ICT/SMC (FVGs, OBs, Sweeps) · Volume Profile · CVD              │  │
│  │  Multi-TF Confluence · Signal Quality · Pre-Market Scoring        │  │
│  │  Range Breakout Detection · 6 Deterministic Filters · CNN Inference│  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │  Trainer Service     │  │  Monitoring (optional profile)          │  │
│  │  GPU CNN training    │  │  Prometheus · Grafana                   │  │
│  │  port 8200           │  │                                         │  │
│  └──────────────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Docker services:**

| Service | Role | Port |
|---|---|---|
| **Postgres** | Durable storage — trade journal, historical bars, risk events | 5433 |
| **Redis** | Hot cache — live bars, analysis metrics, positions, focus, SSE pub/sub | 6380 |
| **Engine** | FastAPI REST API + HTMX dashboard + SSE stream + background analysis worker | 8100 |
| **Web** | HTMX dashboard frontend (reverse-proxies to engine) | 8180 |
| **Trainer** | GPU CNN training server — triggered via the 🧠 Trainer UI *(training profile)* | 8200 |
| **Prometheus** | Metrics collection *(monitoring profile)* | 9095 |
| **Grafana** | Dashboards & visualization *(monitoring profile)* | 3010 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 1. Clone

```bash
git clone https://github.com/nuniesmith/futures.git
cd futures
```

### 2. One-Command Start

```bash
./run.sh
```

This will:
1. Create a Python virtualenv and install all dependencies
2. Generate a `.env` file with secure random secrets
3. Run the test suite and linter
4. Build and start all Docker services

### 3. Add Your API Keys

Edit `.env` and set:

```
MASSIVE_API_KEY=your_key_here    # https://massive.com/dashboard  (real-time CME futures data)
XAI_API_KEY=your_key_here        # https://console.x.ai           (Grok AI analyst)
```

Without `MASSIVE_API_KEY` the system falls back to yfinance (delayed data).
Without `XAI_API_KEY` the Grok AI analyst tab is disabled — everything else works normally.

### 4. Verify

```bash
docker compose ps                 # all services should be "healthy"
docker compose logs -f engine     # watch the engine schedule actions
```

Open the dashboard at **http://localhost:8180**.

---

## Docker Deployment

### Standard (engine + web + postgres + redis)

```bash
docker compose up -d --build
```

### With CNN Trainer (+ GPU training server)

```bash
docker compose --profile training up -d --build
```

Requires an NVIDIA GPU with `nvidia-container-toolkit` installed on the host.
Once running, open the **🧠 Trainer** tab in the dashboard to configure and launch training runs.

### With Monitoring (+ Prometheus & Grafana)

```bash
docker compose --profile monitoring up -d --build
```

### Combine Profiles

```bash
docker compose --profile training --profile monitoring up -d --build
```

### Useful Commands

```bash
docker compose logs -f engine           # follow engine + API logs
docker compose logs -f web              # follow web frontend logs
docker compose logs -f trainer          # follow trainer logs
docker compose exec engine bash         # shell into engine container
docker compose restart engine           # restart engine (picks up new model)
docker compose down                     # stop all services
docker compose down -v                  # stop + remove volumes (⚠️ deletes all data)
```

---

## Local Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the Web Frontend Locally

```bash
./run.sh --local
```

This starts only the web service (port 8180) and expects the engine + Redis + Postgres
to already be running (e.g. via Docker Compose).

### Run Tests

```bash
pytest src/tests/ -x -q --tb=short     # full suite
ruff check src/                         # linting
./run.sh --test                         # tests + lint together
```




## CNN Model Training

Training runs entirely within this repo — no external training service is needed.

### Via the Dashboard UI (recommended)

1. Start the trainer service: `docker compose --profile training up -d trainer`
2. Open the dashboard → **🧠 Trainer** tab
3. Configure epochs, batch size, learning rate, symbols, validation gates
4. Click **▶ Start Training** — live logs stream into the UI
5. On successful promotion the champion model is written to `models/` automatically
   and the engine hot-reloads it within one refresh cycle

### Via CLI (one-shot)

```bash
docker compose --profile training run --rm trainer \
    python -m lib.services.training.trainer_server
```

### Model Files

| File | Description |
|---|---|
| `models/breakout_cnn_best.pt` | PyTorch checkpoint — used by the Python engine for live inference |
| `models/breakout_cnn_best_meta.json` | Metadata: accuracy, precision, recall, training date |
| `models/feature_contract.json` | Feature names + normalization constants for inference |

### Syncing Models Locally

```bash
bash scripts/sync_models.sh              # download all model files from this repo's main branch
bash scripts/sync_models.sh --check      # check whether local models are current
bash scripts/sync_models.sh --pt-only    # download only the .pt checkpoint
bash scripts/sync_models.sh --restart    # download + restart the engine container
```

`run.sh` automatically calls `sync_models.sh` if `models/breakout_cnn_best.pt` is missing.

---

## Project Structure

```
futures/
├── src/
│   ├── lib/
│   │   ├── analysis/                   # Market analysis modules
│   │   │   ├── breakout_cnn.py         #   CNN inference (PyTorch)
│   │   │   ├── confluence.py           #   Multi-timeframe confluence filter
│   │   │   ├── cvd.py                  #   Cumulative Volume Delta + divergences
│   │   │   ├── ict.py                  #   ICT/SMC: FVGs, order blocks, sweeps
│   │   │   ├── orb_filters.py          #   6 deterministic ORB quality filters
│   │   │   ├── regime.py               #   HMM market regime detection
│   │   │   ├── scorer.py               #   Pre-market instrument scoring
│   │   │   ├── signal_quality.py       #   Signal quality score
│   │   │   ├── volatility.py           #   K-Means adaptive vol clustering
│   │   │   ├── volume_profile.py       #   POC, VAH/VAL, naked POCs
│   │   │   └── wave_analysis.py        #   Wave dominance tracking
│   │   │
│   │   ├── core/                       # Infrastructure
│   │   │   ├── alerts.py               #   Alert dispatch (email, webhook)
│   │   │   ├── cache.py                #   Redis cache + data source abstraction
│   │   │   ├── logging_config.py       #   Structured logging (structlog)
│   │   │   ├── models.py               #   Database models + Postgres ORM
│   │   │   └── redis_helpers.py        #   Redis utility functions
│   │   │
│   │   ├── integrations/               # External services
│   │   │   ├── grok_helper.py          #   xAI Grok AI analyst
│   │   │   └── massive_client.py       #   Massive.com REST + WebSocket client
│   │   │
│   │   ├── trading/                    # Trading engine
│   │   │   ├── engine.py               #   DashboardEngine: optimization, backtest
│   │   │   ├── strategies.py           #   Backtesting strategies (Optuna-tunable)
│   │   │   └── costs.py                #   CME slippage + commission model
│   │   │
│   │   ├── training/                   # CNN training pipeline
│   │   │   ├── trainer_server.py       #   FastAPI training server (port 8200)
│   │   │   ├── train.py                #   Full training pipeline
│   │   │   ├── dataset.py              #   Dataset generation from bars
│   │   │   └── model.py                #   CNN model architecture
│   │   │
│   │   └── services/                   # Deployable services
│   │       ├── data/                   #   FastAPI data service + API routers
│   │       │   └── api/
│   │       │       ├── dashboard.py    #     Main HTMX dashboard page
│   │       │       ├── trainer.py      #     🧠 Trainer page + proxy to trainer:8200
│   │       │       ├── settings.py     #     ⚙️ Settings page

│   │       │       ├── sse.py          #     Server-sent events stream
│   │       │       └── ...
│   │       ├── engine/                 #   Background engine service
│   │       └── web/                    #   HTMX dashboard frontend (reverse proxy)
│   │
│   ├── tradovate/                      # Tradovate broker bridge (JS)
│   │   ├── Bridge.js                   #   Broker bridge connector
│   │   └── Copier.js                   #   Trade copier
│   │
│   └── tests/                          # Pytest test suite
│
├── scripts/
│   ├── daily_report.py                 # End-of-day breakout session summary
│   ├── monitor_signals.py              # Live breakout signal terminal monitor
│   ├── session_signal_audit.py         # Per-session signal quality audit
│   ├── check_onnx_parity.py            # Verify PT and ONNX model outputs match
│   ├── smoke_test_trainer.py           # Quick end-to-end trainer smoke test
│   ├── patch_breakout_strategy.py      # Patch BreakoutStrategy.cs with latest params
│   └── patch_datapreloader.py          # Patch DataPreloader.cs with symbol list
│
├── models/                             # CNN model files (git-tracked via LFS)
│   ├── breakout_cnn_best.pt            #   Champion PyTorch checkpoint
│   ├── breakout_cnn_best_meta.json     #   Champion metadata (acc, prec, recall, date)
│   └── feature_contract.json           #   Feature names + normalization constants
│
├── config/
│   ├── grafana/                        # Grafana provisioning + dashboards
│   └── prometheus/                     # Prometheus scrape config
│
├── docker/
│   ├── data/Dockerfile                 # Engine + data API container
│   ├── engine/Dockerfile               # Background engine container
│   ├── web/Dockerfile                  # Web frontend container
│   ├── trainer/Dockerfile              # GPU trainer container
│   └── monitoring/                     # Prometheus + Grafana Dockerfiles
│
├── dataset/                            # Generated training datasets (git-ignored)
├── data/                               # Persistent app data (git-ignored)
├── docs/                               # Design docs and audit reports
├── docker-compose.yml                  # Full service stack
├── pyproject.toml                      # Python project config (hatch + deps)
├── run.sh                              # One-command build + deploy script
└── todo.md                             # Project status & phase tracking
```

---

## Configuration

### Environment Variables

#### Required (auto-generated by `run.sh`)

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `REDIS_PASSWORD` | Redis password |

#### API Keys

| Variable | Description | Fallback |
|---|---|---|
| `MASSIVE_API_KEY` | [Massive.com](https://massive.com) real-time CME futures data | yfinance (delayed) |
| `XAI_API_KEY` | [xAI](https://console.x.ai) Grok AI analyst | AI features disabled |
| `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` | [Kraken](https://www.kraken.com) crypto spot data | Crypto panels hidden |

#### Trading

| Variable | Default | Description |
|---|---|---|
| `ACCOUNT_SIZE` | `150000` | Account size for risk calculations ($50K, $100K, or $150K) |
| `ORB_FILTER_GATE` | `majority` | Filter strictness: `all`, `majority`, or `none` |
| `ORB_CNN_GATE` | `0` | `0` = CNN advisory only, `1` = CNN hard gate (blocks trade signal)

#### Trainer

| Variable | Default | Description |
|---|---|---|
| `CNN_RETRAIN_EPOCHS` | `25` | Default training epochs |
| `CNN_RETRAIN_BATCH_SIZE` | `64` | Default batch size |
| `CNN_RETRAIN_LR` | `0.0002` | Default learning rate |
| `CNN_RETRAIN_PATIENCE` | `8` | Early stopping patience |
| `CNN_RETRAIN_MIN_ACC` | `80.0` | Minimum validation accuracy to promote a model (%) |
| `CNN_RETRAIN_MIN_PRECISION` | `75.0` | Minimum precision gate (%) |
| `CNN_RETRAIN_MIN_RECALL` | `70.0` | Minimum recall gate (%) |
| `CNN_RETRAIN_DAYS_BACK` | `90` | Days of history for dataset generation |
| `CNN_RETRAIN_SYMBOLS` | *(22 CME futures + BTC/ETH/SOL)* | Comma-separated symbol list |
| `TRAINER_API_KEY` | *(unset)* | Optional bearer token to protect the `/train` endpoint |

### Key Runtime Defaults

| Setting | Value | Location |
|---|---|---|
| CNN inference threshold | 0.82 | `breakout_cnn.py` |
| Engine refresh interval | 5 minutes | `scheduler.py` |
| Grok update interval | 15 minutes | `scheduler.py` |
| Risk check interval | 1 minute | `scheduler.py` |

---

## Testing

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

# Verify ONNX and PyTorch model outputs match
PYTHONPATH=src python scripts/check_onnx_parity.py

# Quick trainer smoke test (requires trainer service running)
PYTHONPATH=src python scripts/smoke_test_trainer.py
```

---

## Scripts & Tools

### Model Sync

```bash
bash scripts/sync_models.sh              # pull latest model files
bash scripts/sync_models.sh --check      # check if models are current
bash scripts/sync_models.sh --pt-only    # pull only the .pt checkpoint
bash scripts/sync_models.sh --restart    # pull + restart engine container
```

### Daily Report

```bash
PYTHONPATH=src python scripts/daily_report.py              # today's report
PYTHONPATH=src python scripts/daily_report.py --days 5     # last 5 days
PYTHONPATH=src python scripts/daily_report.py --json       # JSON output
```

### Live Signal Monitor

```bash
PYTHONPATH=src python scripts/monitor_signals.py              # watch live signals
PYTHONPATH=src python scripts/monitor_signals.py --interval 2 # 2s polling
PYTHONPATH=src python scripts/monitor_signals.py --json       # JSON output
```

### Session Signal Audit

```bash
PYTHONPATH=src python scripts/session_signal_audit.py                      # all sessions, 30 days
PYTHONPATH=src python scripts/session_signal_audit.py --days 14            # last 14 days
PYTHONPATH=src python scripts/session_signal_audit.py --export-json out.json
```


# Patch BreakoutStrategy.cs with latest model params from feature_contract.json
PYTHONPATH=src python scripts/patch_breakout_strategy.py

# Patch DataPreloader.cs with the current tracked symbol list
PYTHONPATH=src python scripts/patch_datapreloader.py
```

---

## Related Repos



---

## Technologies

| Layer | Stack |
|---|---|
| **Language** | Python 3.11+ |
| **Web** | FastAPI, HTMX, SSE |
| **Data** | Massive.com (real-time CME), yfinance (fallback), Kraken (crypto), pandas |
| **Storage** | PostgreSQL 16, Redis 7 |
| **AI / ML** | PyTorch (CNN training), ONNX (inference), xAI Grok (AI analyst) |
| **Analysis** | scikit-learn, hmmlearn (HMM regime), backtesting.py, Optuna |
| **Execution** | NinjaTrader 8 (C# strategies + Bridge add-on) |
| **Observability** | structlog, Prometheus, Grafana |
| **Deployment** | Docker Compose |

---

## License

[MIT](LICENSE) — Copyright (c) 2026 nuniesmith