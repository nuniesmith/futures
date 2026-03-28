# Futures — Multi-Asset Crypto Scalper

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

KuCoin Perpetual Futures scalper with **10 assets**, Optuna auto-optimization,
Ruby wave analysis, and a supervisor/worker architecture — designed to run 24/7
on a Raspberry Pi 4.

---

## Assets

| Asset    | Symbol          | Max Leverage | Default Leverage | Character          |
|----------|-----------------|--------------|------------------|--------------------|
| BTC      | BTCUSDTM        | 125x         | 20x              | Low vol, deep book |
| ETH      | ETHUSDTM        | 100x         | 20x              | Medium vol         |
| SOL      | SOLUSDTM        | 75x          | 20x              | High vol, fast     |
| AVAX     | AVAXUSDTM       | 75x          | 20x              | High vol           |
| FARTCOIN | FARTCOINUSDTM   | 50x          | 20x              | Extreme vol, thin  |
| WIF      | WIFUSDTM        | 75x          | 20x              | Meme, high vol     |
| KCS      | KCSUSDTM        | 8x           | 4x               | Low vol, KuCoin    |
| DOGE     | DOGEUSDTM       | 75x          | 20x              | Meme, liquid       |
| PEPE     | PEPEUSDTM       | 75x          | 20x              | Meme, volatile     |
| SUI      | SUIUSDTM        | 75x          | 20x              | High vol           |

**Margin mode:** Isolated (each position's risk is capped independently).
**Position mode:** One-Way (required for `reduceOnly` closes).

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    SUPERVISOR                           │
│  Spawns/monitors workers • Aggregates PnL • Heartbeats │
├──────┬──────┬──────┬──────┬──────┬──────┬───────────────┤
│ BTC  │ ETH  │ SOL  │ AVAX │ FART │ WIF  │ ... (workers) │
│Worker│Worker│Worker│Worker│Worker│Worker│               │
└──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴───────────────┘
   │      │      │      │      │      │
   └──────┴──────┴──────┴──────┴──────┘
                  │
          ┌───────┴───────┐
          │    Redis      │
          │ PnL • Orders  │
          │ Candles • State│
          └───────────────┘
```

Each **worker** is a self-contained trading bot for one asset:
- WebSocket feeds (trades + orderbook)
- Candle builder (5s OHLCV from tick stream)
- Signal engine (EMA crossover + book imbalance + Ruby wave gates)
- Risk manager (per-asset daily limits, consecutive loss breaker)
- Optuna optimizer (background, per-asset parameter tuning)

The **supervisor** manages all workers:
- Spawns one async task per enabled asset
- Monitors health via Redis heartbeats
- Restarts crashed workers with backoff
- Logs aggregate portfolio stats
- Handles SIGTERM/SIGINT for clean shutdown

---

## Modes

### Live Mode (default)
Connects to KuCoin API, places real orders, manages real positions.

### Simulation Mode
Watches live WebSocket data but **does not execute any orders**. Simulated
trades are tracked in Redis so you can evaluate strategy performance on
real market data before risking capital.

```bash
# Enable sim mode via environment
SIM_MODE=true ./run.sh up

# Or set in .env
SIM_MODE=true
```

In sim mode:
- All WS data feeds are live (real prices, real orderbook)
- Trades are simulated at the current market price
- PnL is calculated and stored in Redis
- Discord alerts are tagged with `[SIM]`
- No API calls are made for order placement
- Useful for validating strategy on a new asset before going live

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/nuniesmith/futures.git
cd futures
cp .env.example .env
# Edit .env with your KuCoin API credentials
```

### 2. Run locally (development)

```bash
./run.sh setup    # Create venv, install deps
./run.sh test     # Run tests
./run.sh sim      # Start in simulation mode (no real orders)
```

### 3. Run with Docker (production / Pi)

```bash
./run.sh build    # Build Docker images
./run.sh up       # Start futures + redis containers
./run.sh logs     # Follow logs
./run.sh down     # Stop everything
```

### 4. Deploy on Raspberry Pi 4

```bash
# On the Pi:
git clone https://github.com/nuniesmith/futures.git
cd futures
chmod +x scripts/setup-pi.sh
./scripts/setup-pi.sh   # Installs Docker, configures system
cp .env.example .env && nano .env
./run.sh up
```

---

## Configuration

### Environment Variables (`.env`)

| Variable             | Required | Default      | Description                          |
|----------------------|----------|--------------|--------------------------------------|
| `KUCOIN_API_KEY`     | Yes      | —            | KuCoin Futures API key               |
| `KUCOIN_API_SECRET`  | Yes      | —            | KuCoin Futures API secret            |
| `KUCOIN_PASSPHRASE`  | Yes      | —            | KuCoin API passphrase                |
| `DISCORD_WEBHOOK_URL`| No       | —            | Discord webhook for trade alerts     |
| `REDIS_PASSWORD`     | Yes      | —            | Redis auth password                  |
| `CAPITAL`            | No       | `30.0`       | Total USDT available for trading     |
| `MARGIN_MODE`        | No       | `isolated`   | `isolated` or `cross`                |
| `SIM_MODE`           | No       | `false`      | `true` to run without real orders    |
| `LOG_LEVEL`          | No       | `INFO`       | `DEBUG`, `INFO`, `WARNING`, `ERROR`  |
| `TZ`                 | No       | `America/New_York` | Timezone for logs and daily resets |

### YAML Config (`config/futures.yaml`)

All trading parameters, per-asset overrides, strategy defaults, Optuna search
ranges, risk limits, and monitoring settings live in one file. Environment
variables are expanded with `${VAR_NAME}` syntax.

---

## Risk Management

- **Isolated margin** — each position's loss is capped to its allocated margin
- **1% risk per add** — each stack entry risks 1% of capital
- **Max 3 adds per asset** — maximum 3% of capital at risk per asset
- **Min $0.50 per order** — ensures orders are large enough to fill
- **Worst case exposure** — 10 assets × 3 orders = 30 orders × $0.50 = $15.00
- **Daily loss limit** — halt trading after 5% daily drawdown
- **Consecutive loss breaker** — pause after 4 losses in a row
- **Post-SL cooldown** — 30s wait after stop-loss before re-entry
- **Daily trade cap** — max 50 round-trips per day per asset

---

## Strategy

**EMA Crossover + Order Book Imbalance + Ruby Wave Gates**

1. **Signal generation** — Fast/Slow EMA crossover for direction, confirmed by
   top-10 orderbook bid/ask volume imbalance
2. **Quality gate** — 0–100 multi-factor score (AO + EMA alignment + imbalance +
   volume + regime) must exceed threshold for first entry
3. **Wave gate** — Ruby wave analysis (`wave_ratio`, `cur_ratio`) must confirm
   momentum before adding positions #2 and #3
4. **Regime filter** — SMA-200 slope detects trending/volatile/ranging — adapts
   TP targets and blocks counter-trend stacking
5. **Adaptive TP/SL** — Take-profit scales with volatility percentile and regime;
   wider in trends, tighter in ranges
6. **Optuna** — Auto-optimizes EMA periods, imbalance threshold, SL%, quality
   minimum, and wave gate every 20 minutes on live data

---

## Project Structure

```text
futures/
├── config/
│   └── futures.yaml          # Master config (all tunables)
├── docker/
│   ├── futures/
│   │   ├── Dockerfile        # Two-stage Python build
│   │   └── entrypoint.sh     # Container startup
│   └── redis/
│       ├── Dockerfile        # Redis with THP tuning
│       └── entrypoint.sh     # Kernel parameter setup
├── scripts/
│   └── setup-pi.sh           # Raspberry Pi setup script
├── src/
│   ├── analysis/             # Wave analysis, CVD, signal quality, volatility
│   ├── config/               # Config loader, asset registry
│   ├── indicators/           # Technical indicators (EMA, ATR, AO, etc.)
│   ├── services/             # Redis store, Discord notifier, Optuna optimizer
│   ├── supervisor/           # Master process, worker management
│   ├── tests/                # Pytest suite
│   ├── worker/               # Per-asset trading bot (trader, signals, risk, orders)
│   ├── logging_config.py     # Structured logging setup
│   └── main.py               # Entry point
├── .env.example              # Environment template
├── docker-compose.yml        # Futures bot + Redis
├── pyproject.toml            # Dependencies, ruff, mypy, pytest config
├── run.sh                    # Project management CLI
└── todo.md                   # Detailed task tracker
```

---

## Commands

```bash
./run.sh              # Full pipeline: setup → test → build → up
./run.sh setup        # Create .venv, install dependencies
./run.sh test         # Run pytest
./run.sh lint         # Run ruff check + ruff format --check + mypy
./run.sh sim          # Start in simulation mode (local, no Docker)
./run.sh build        # Build Docker images
./run.sh up           # Start Docker Compose (futures + redis)
./run.sh down         # Stop all containers
./run.sh logs         # Follow futures container logs
./run.sh status       # Show container status
./run.sh clean        # Remove .venv and Docker artifacts
```

---

## Redis Data

Redis stores all persistent state (survives container restarts):

| Key Pattern                      | Type       | TTL       | Description                     |
|----------------------------------|------------|-----------|----------------------------------|
| `futures:pnl:{asset}`           | Sorted Set | Never     | PnL entries (score=timestamp)    |
| `futures:orders:{asset}`        | List       | Never     | Last 1000 orders per asset       |
| `futures:candles:{asset}:{tf}`  | Sorted Set | 1w–3mo    | OHLCV candles by timeframe       |
| `futures:state:{asset}`        | Hash       | Never     | Worker state (stack, params)     |
| `futures:heartbeat:{asset}`    | String     | 60s       | Worker health heartbeat          |
| `futures:optuna:{asset}`       | Hash       | Never     | Latest optimized params          |
| `futures:sim:pnl:{asset}`      | Sorted Set | Never     | Simulated PnL (sim mode only)   |

---

## Timezone

All logs, daily resets, and Discord summaries use **America/New_York** (EDT/EST)
by default. Change via the `TZ` environment variable:

```bash
TZ=America/New_York  # Eastern (default)
TZ=UTC               # UTC
TZ=America/Toronto   # Same as New York for EDT/EST
```

---

## Development

```bash
# Lint
ruff check src/ --fix
ruff format src/

# Type check
mypy src/ --ignore-missing-imports

# Test
pytest src/tests/ -v

# All at once
./run.sh lint && ./run.sh test
```

---

## License

MIT — see [LICENSE](LICENSE) for details.