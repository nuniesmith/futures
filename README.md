# Futures Trading Co-Pilot

> Live dashboard, market stats & web UI for futures trading — real-time range breakout detection,
> session-aware scheduling, and a full HTMX dashboard powered by FastAPI.

This repo is the **web UI and live engine** layer of a three-repo system:

| Repo | Purpose |
|---|---|
| **[futures](https://github.com/nuniesmith/futures)** (this) | Live dashboard, web UI, market stats, engine, shared `lib` |
| **[rb](https://github.com/nuniesmith/rb)** | Service-only trainer (Docker Compose pulls `nuniesmith/futures:trainer`), hosts trained models (.pt, .onnx) |
| **[ninjatrader](https://github.com/nuniesmith/ninjatrader)** | NinjaTrader 8 C# strategies, indicators, Bridge — pulls best ONNX from `rb` repo |

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Local Development](#local-development)
- [Model Sync](#model-sync)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Scripts & Tools](#scripts--tools)
- [Technologies](#technologies)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Futures Trading Co-Pilot                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Postgres   │  │    Redis     │  │ Data Service │  │   Engine   │  │
│  │  (journal,   │  │  (hot cache, │  │  (FastAPI +  │  │ (scheduler,│  │
│  │   history,   │  │   live bars, │  │   HTMX dash, │  │  analysis, │  │
│  │   risk)      │  │   focus,     │  │   REST API,  │  │  ORB, risk │  │
│  │              │  │   positions) │  │   SSE)       │  │  scoring)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         └─────────────────┴─────────────────┴────────────────┘         │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐  │
│  │                    Analysis Pipeline                               │  │
│  │                                                                   │  │
│  │  Wave Analysis · Volatility Clustering · Regime Detection (HMM)   │  │
│  │  ICT/SMC (FVGs, OBs, Sweeps) · Volume Profile · CVD              │  │
│  │  Multi-TF Confluence · Signal Quality · Pre-Market Scoring        │  │
│  │  Range Breakout Detection · 6 Deterministic Filters · CNN Inference │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Monitoring: Prometheus · Grafana          (optional profile)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                                           │
         ▼                                           ▼
   rb repo (training service + models)     ninjatrader repo (execution)
   CNN .pt/.onnx → models/                Bridge.cs → CME orders
```

**Docker services:**

| Service | Role | Port |
|---|---|---|
| **Postgres** | Durable storage — trade journal, historical bars, risk events | 5433 |
| **Redis** | Hot cache — live bars, Ruby metrics, positions, focus, SSE pub/sub | 6380 |
| **Data Service** | FastAPI REST API + HTMX dashboard + SSE live stream | 8100 |
| **Web** | HTMX dashboard frontend (reverse-proxies to data service) | 8180 |
| **Engine** | Background worker — analysis, range breakout detection, scheduling | — |
| **Prometheus** | Metrics collection *(monitoring profile)* | 9095 |
| **Grafana** | Dashboards & visualization *(monitoring profile)* | 3010 |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 1. Clone & Setup

```bash
git clone https://github.com/nuniesmith/futures.git
cd futures
```

### 2. One-Command Docker Start

```bash
./run.sh
```

This will:
1. Create a Python virtualenv and install dependencies
2. Generate a `.env` file with secure random secrets
3. Pull the CNN model from the [rb repo](https://github.com/nuniesmith/rb) (if not present)
4. Run the test suite and linter
5. Build and start all Docker services

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
```

Open the dashboard at `http://localhost:8180`.

---

## Docker Deployment

### Standard (5 services)

```bash
docker compose up -d --build
```

### With Monitoring (+ Prometheus & Grafana)

```bash
docker compose --profile monitoring up -d --build
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
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Locally

```bash
./run.sh --local      # starts web service (requires Redis + Postgres running)
```

### Run Tests

```bash
pytest src/tests/ -x -q --tb=short     # full suite
ruff check src/                        # linting
./run.sh --test                        # tests + lint together
```

---

## Model Sync

The CNN model is trained via the [rb repo](https://github.com/nuniesmith/rb) (which
pulls the `nuniesmith/futures:trainer` Docker image) and the resulting champion models
are committed back to `rb`. This repo pulls those models for live inference.
The model files live in `models/` and are downloaded from GitHub:

```bash
# Download latest champion model from rb repo
bash scripts/sync_models.sh

# Check if local models are current
bash scripts/sync_models.sh --check

# Download + restart engine to pick up new model
bash scripts/sync_models.sh --restart

# Download only the .pt checkpoint (skip ONNX)
bash scripts/sync_models.sh --pt-only
```

`run.sh` automatically pulls the model if `models/breakout_cnn_best.pt` is missing.

---

## Project Structure

```
futures/
├── src/
│   ├── lib/
│   │   ├── analysis/                 # Market analysis modules
│   │   │   ├── breakout_cnn.py       #   CNN inference (model from rb repo)
│   │   │   ├── confluence.py         #   Multi-timeframe confluence filter
│   │   │   ├── cvd.py                #   Cumulative Volume Delta + divergences
│   │   │   ├── ict.py                #   ICT/SMC: FVGs, order blocks, sweeps
│   │   │   ├── orb_filters.py        #   6 deterministic ORB quality filters
│   │   │   ├── regime.py             #   HMM market regime detection
│   │   │   ├── scorer.py             #   Pre-market instrument scoring
│   │   │   ├── signal_quality.py     #   Ruby signal quality score
│   │   │   ├── volatility.py         #   K-Means adaptive vol clustering
│   │   │   ├── volume_profile.py     #   POC, VAH/VAL, naked POCs
│   │   │   └── wave_analysis.py      #   Wave dominance tracking
│   │   │
│   │   ├── core/                     # Infrastructure
│   │   │   ├── alerts.py             #   Alert dispatch (email, webhook)
│   │   │   ├── cache.py              #   Redis cache + data source abstraction
│   │   │   ├── logging_config.py     #   Structured logging (structlog)
│   │   │   ├── models.py             #   Database models + Postgres/SQLite ORM
│   │   │   └── redis_helpers.py      #   Redis utility functions
│   │   │
│   │   ├── integrations/             # External services
│   │   │   ├── grok_helper.py        #   xAI Grok AI analyst
│   │   │   └── massive_client.py     #   Massive.com REST + WebSocket client
│   │   │
│   │   ├── trading/                  # Trading engine
│   │   │   ├── engine.py             #   DashboardEngine: optimization, backtest
│   │   │   ├── strategies.py         #   Backtesting strategies (Optuna-tunable)
│   │   │   └── costs.py              #   CME slippage + commission model
│   │   │
│   │   └── services/                 # Deployable services
│   │       ├── data/                 #   FastAPI data service + API routers
│   │       ├── engine/               #   Background engine service
│   │       └── web/                  #   HTMX dashboard frontend
│   │
│   └── tests/                        # Pytest test suite
│
├── scripts/
│   ├── sync_models.sh                # Pull CNN model from rb repo
│   ├── daily_report.py               # End-of-day breakout session summary
│   ├── monitor_signals.py            # Live breakout signal terminal monitor
│   └── session_signal_audit.py       # Per-session signal quality audit
│
├── config/
│   ├── grafana/                      # Grafana provisioning + dashboards
│   └── prometheus/                   # Prometheus scrape config
│
├── docker/
│   ├── data/Dockerfile               # Data service container
│   ├── engine/Dockerfile             # Engine container
│   ├── web/Dockerfile                # Web frontend container
│   └── monitoring/                   # Prometheus + Grafana Dockerfiles
│
├── models/                           # CNN model files (pulled from rb repo)
│   └── .gitkeep
│
├── docker-compose.yml                # Full service stack
├── pyproject.toml                    # Python project config (hatch + deps)
├── run.sh                            # One-command build + deploy script
└── todo.md                           # Project status & phase tracking
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

### Key Defaults

| Setting | Value | Location |
|---|---|---|
| CNN inference threshold | 0.82 | `breakout_cnn.py` |
| Ruby recompute interval | 5 minutes | `scheduler.py` |
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
```

---

## Scripts & Tools

### Model Sync

```bash
bash scripts/sync_models.sh              # pull latest model from rb repo
bash scripts/sync_models.sh --check      # check if models are current
bash scripts/sync_models.sh --restart    # pull + restart engine
```

### Daily Report

```bash
PYTHONPATH=src python scripts/daily_report.py              # today's report
PYTHONPATH=src python scripts/daily_report.py --days 5     # last 5 days
PYTHONPATH=src python scripts/daily_report.py --json       # JSON output
```

### Live Signal Monitor

```bash
PYTHONPATH=src python scripts/monitor_signals.py              # watch signals
PYTHONPATH=src python scripts/monitor_signals.py --interval 2 # 2s polling
PYTHONPATH=src python scripts/monitor_signals.py --json       # JSON output
```

### Session Signal Audit

```bash
PYTHONPATH=src python scripts/session_signal_audit.py                    # all sessions, 30 days
PYTHONPATH=src python scripts/session_signal_audit.py --days 14          # last 14 days
PYTHONPATH=src python scripts/session_signal_audit.py --export-json out.json
```

---

## Related Repos

- **[rb](https://github.com/nuniesmith/rb)** — Service-only training repo. Uses Docker Compose to pull the `nuniesmith/futures:trainer` image from Docker Hub. Hosts trained champion models (.pt for Python engine, .onnx for NinjaTrader). Models are pulled from here via `scripts/sync_models.sh`. NinjaTrader uses PowerShell to pull the best ONNX model from this repo.
- **[ninjatrader](https://github.com/nuniesmith/ninjatrader)** — NinjaTrader 8 C# code: Ruby indicator, Bridge execution strategy, SignalBus, OrbCnnPredictor. The dashboard's NT8 Deploy panel pulls installer scripts from this repo. Pulls best `.onnx` model from the `rb` repo.

---

## Technologies

| Layer | Stack |
|---|---|
| **Language** | Python 3.11+ |
| **Web** | FastAPI, HTMX, Jinja2, SSE |
| **Data** | Massive.com (real-time CME), yfinance (fallback), pandas |
| **Storage** | PostgreSQL 16, Redis 7 |
| **AI** | xAI Grok (morning briefing + live updates) |
| **Analysis** | scikit-learn, HMM (hmmlearn), backtesting.py, Optuna |
| **Observability** | structlog, Prometheus, Grafana |
| **Deployment** | Docker Compose |

---

## License

[MIT](LICENSE) — Copyright (c) 2026 nuniesmith