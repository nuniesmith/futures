**✅ Refactored & Updated TODO.md (March 2026 — Full 24h Coverage + Per-Session CNN Gate)**

**Project: Ruby + Bridge ORB System**  
**Status: Live Paper Trading + Full 24h Globex Coverage + Per-Session CNN Gate (9.8/10)**  
**Goal:** Fully automated ORB system covering the entire Globex trading day (18:00 ET start) across all major CME exchanges and sessions — CME Globex open, Sydney/ASX, Tokyo/TSE, Shanghai/HK, Frankfurt/Xetra, London, London-NY crossover, US Equity, and CME Settlement — on CME micro contracts + FX futures with deterministic filters + hybrid CNN scoring → 1–5 high-conviction trades/day.

---

### Executive Summary (Current State)
- **Dataset**: 3,691 labeled PNG charts (growing — full 24h + all-session regeneration now unblocked).
- **Models**: 36+ trained checkpoints (champion: 83.73% val accuracy, 91.71% precision).
- **Live Pipeline**: Complete + **running** — ORB detection → filters (majority gate) → CNN inference → Redis publish → NT8 Bridge execution on Sim100.
- **Monitoring Stack**: Redis + Prometheus + Grafana **live** — full ORB/CNN/filter dashboards.
- **Scheduler**: Now covers **full Globex day from 18:00 ET** with 9 sessions across all major exchanges.
- **Risk Engine**: Integrated, publishing to Redis, tracking Sim100 trades in real-time.
- **NinjaTrader**: Strategy + indicator running live on Sim100, receiving and executing Python signals.

**Latest Additions**:
- **Per-session CNN gate via Redis** — `engine:config:cnn_gate:{session_key}` keys allow enabling/disabling the hard CNN filter per session without a restart. API: `GET/PUT/DELETE /cnn/gate` + `/cnn/gate/{session_key}` + dashboard HTML fragment at `/cnn/gate/html`.
- **Session signal quality audit** — `scripts/session_signal_audit.py`: cross-references ORB audit trail with paper-trade outcomes, outputs per-session win-rate / PF / CNN P-distribution, and recommends which sessions are ready for CNN gate enablement.

**Biggest Wins Achieved**:
- Realistic auto-labeling matching live Bridge brackets.
- Hybrid EfficientNetV2-S + tabular CNN (8 features including real CVD delta + session encoding).
- Overnight retraining pipeline with strict validation gate.
- Full audit trail for every ORB evaluation (even filtered ones).
- **Full 24h exchange coverage**: CME open (18:00), Sydney/ASX (18:30), Tokyo/TSE (19:00), Shanghai/HK (21:00), Frankfurt/Xetra (03:00), London (03:00), London-NY (08:00), US Equity (09:30), CME Settlement (14:00).
- **DST-safe**: all session times stored as ET wall-clock (ZoneInfo), auto-adjusts EST↔EDT.
- **Expanded symbol universe**: MGC, SIL, MHG, MCL, MES, MNQ, M2K, MYM, MBT, 6E, 6B, 6J, 6A, 6C.

**Current Status**: Paper trading active on Sim100. Dataset expansion to 10k+ images in progress via nightly incremental build across all 9 sessions and 15 symbols.

---

### ✅ Completed (Phases 1–6)

**Phase 1 – Core ORB + Filters**  
- Deterministic filter gate with NR7, pre-market range, session windows, lunch filter, multi-TF EMA, VWAP.  
- `majority` gate mode (recommended default) via `ORB_FILTER_GATE` env var.  
- Full audit trail (`record_orb_event` + enrichment).

**Phase 2 – Validation**  
- Real-data backtests via Massive API on MGC/MES/MNQ showing strong improvements.  
- Backtest tool (`scripts/backtest_filters.py`) with per-filter breakdown + CSV export.

**Phase 3 – Dataset Generation**  
- 3,691 labeled images generated (MGC/MES/MNQ + others) — expanding to 10k+ via all-session rebuild.  
- Ruby-style renderer with ORB box, EMA9, VWAP, quality badge, nightclouds theme.  
- Incremental generation + stratified train/val split.
- Dataset generator now supports `orb_session="all"` — covers all 9 Globex-day sessions.

**Phase 4 – CNN Training**  
- Champion model: 83.73% val accuracy, 91.71% precision, 25 epochs.  
- Overnight retraining pipeline with validation gate + atomic promotion.  
- Training history CSV + retrain audit JSONL.
- Tabular feature 7 updated: session ordinal encoding (0.0–1.0) replaces binary `session_flag`.

**Phase 5 – Live Integration**  
- Full ORB → Filter → Optional CNN → Redis publish → NT8 Bridge.  
- `ORB_CNN_GATE` env var (advisory by default).  
- RiskManager fully wired.  
- Scheduler handles all sessions + Sim100 trade tracking.

**Phase 6 – Full 24h Globex Coverage** ✅ NEW  
- **9 sessions** from 18:00 ET start: CME open, Sydney/ASX, Tokyo/TSE, Shanghai/HK, Frankfurt/Xetra, London, London-NY, US Equity, CME Settlement.  
- **DST-safe**: all times stored as ET wall-clock via `ZoneInfo("America/New_York")` — auto EST↔EDT.  
- **15 symbols**: MGC, SIL, MHG, MCL, MES, MNQ, M2K, MYM, MBT, 6E, 6B, 6J, 6A, 6C (+ 6C).  
- **Per-session asset lists**: each session scans only its relevant instruments (e.g. 6J/6A for Tokyo).  
- **`get_session_for_utc()`** + **`get_active_session_keys()`** helpers for DST-safe lookups.  
- **`DATASET_SESSIONS`** list for CNN dataset scoping.  
- **New contract specs**: M2K (Micro Russell), MYM (Micro Dow), 6B/6J/6A/6C (FX), MBT (Micro Bitcoin).  
- **`POINT_VALUE` / `TICK_SIZE` / `FX_TICKERS` / `OVERNIGHT_TICKERS`** convenience dicts in `models.py`.  
- **Session filter windows** in `main.py` extended for all 9 sessions (CME overnight, Shanghai, Frankfurt, etc.).  
- **Scheduler `EVENING` mode** (18:00–00:00 ET) fires overnight ORB checks; off-hours (12:00–18:00 ET) runs CNN/backfill.

---

### ✅ Priority 1 – Paper Trading & Monitoring (COMPLETE)

- [x] **Paper trading running** on Sim100 (1 micro contract) — NT8 strategy + indicator live, receiving Python signals.
- [x] **Build Grafana dashboards** (leverage existing Prometheus + Redis):
  - `config/grafana/orb-trading-dashboard.json` — full dashboard provisioned automatically
  - Live ORB signals with CNN probability ✓
  - Risk metrics (daily P&L, exposure, consecutive losses) ✓
  - Filter rejection rates + CNN confidence distribution ✓
  - Model performance (last retrain accuracy, `model_stale`, `model_last_retrain_epoch` gauges) ✓
- [x] **Daily health report** — `scripts/daily_report.py` + `_handle_daily_report` in engine publishes to `engine:daily_report` Redis key; email via `_send_daily_report_email`; REST at `GET /audit/daily-report`.

---

### 🔲 Priority 2 – Dataset Expansion to 10k+ (This Week)

- [x] **Expand asset universe to 22 symbols** — added to `MICRO_CONTRACT_SPECS`, `FULL_CONTRACT_SPECS`,
  `YAHOO_TO_MASSIVE_PRODUCT`, `_SYMBOL_TO_TICKER`, `BuildConfig.symbols`, and `_handle_generate_chart_dataset`:
  - **New energy**: `MNG=F` Micro Natural Gas (data via `NG=F`)
  - **New FX**: `6S=F` Swiss Franc, `M6E=F` Micro Euro FX (upgraded from 6E), `M6B=F` Micro GBP (upgraded from 6B)
  - **New rates (CBOT)**: `ZN=F` 10-Year T-Note, `ZB=F` 30-Year T-Bond
  - **New agri (CBOT)**: `ZC=F` Corn, `ZS=F` Soybeans, `ZW=F` Wheat
  - **New crypto**: `MET=F` Micro Ether
  - All new product codes added to `YAHOO_TO_MASSIVE_PRODUCT` in `massive_client.py` (also ZF, ZT, ZL, ZM for future use)
  - `RATES_TICKERS` and `AG_TICKERS` frozensets added to `models.py`
  - `OVERNIGHT_TICKERS` expanded to include NG, MET, 6C, 6S (13 overnight tickers total)
  - `BuildConfig.orb_session` default changed from `"both"` → `"all"` (full 24h Globex coverage)
  - Total: 22 active assets (14 original + 8 new: MNG, 6S, M6E↑, M6B↑, ZN, ZB, ZC, ZS, ZW, MET)
- [ ] **Trigger full dataset regeneration** with `--session all` and 22-symbol universe:
  ```bash
  # Manual trigger (runs all 9 sessions × 22 symbols × 90 days → target 10k+ images):
  # Both env var aliases are now supported: CNN_ORB_SESSION and DATASET_ORB_SESSION.
  PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
    --symbols MGC SIL MHG MCL MNG MES MNQ M2K MYM MBT MET \
              6E 6B 6J 6A 6C 6S ZN ZB ZC ZS ZW \
    --days-back 90 --session all
  ```
- [ ] **Retrain CNN** after dataset reaches 10k+ images (run on GPU machine via trainer stack):
  ```bash
  # Option A — GPU machine via docker-compose.train.yml (recommended):
  docker compose -f docker-compose.train.yml up --build trainer
  bash scripts/sync_models.sh <gpu-machine-host>
  docker compose restart engine

  # Option B — bare-metal on GPU machine:
  PYTHONPATH=src .venv/bin/python scripts/train_gpu.py --epochs 30 --batch-size 64
  ```
  Target: push val accuracy from 83.73% → 86%+ with richer multi-session data.
- [x] **Validate overnight session signals** — `scripts/session_signal_audit.py` cross-references ORB audit events with paper-trade outcomes and prints a per-session quality table (win rate, PF, CNN P-distribution) with CNN gate recommendations. Run weekly to decide when to flip the gate on.
- [x] **Backfill all new symbols** via Massive — all 23 symbols are now registered in `ASSETS`/`MICRO_CONTRACT_SPECS`
  and will be included in the next scheduled `run_backfill()` call. Trigger manually:
  ```bash
  # Backfill all new symbols (rates, agri, FX, energy, crypto):
  BACKFILL_SYMBOLS=NG=F,ZN=F,ZB=F,ZC=F,ZS=F,ZW=F,6S=F,MET=F \
  PYTHONPATH=src .venv/bin/python -c "
  from lib.services.engine.backfill import run_backfill
  r = run_backfill(
      symbols=['NG=F','ZN=F','ZB=F','ZC=F','ZS=F','ZW=F','6S=F','MET=F',
               '6B=F','6J=F','6A=F','6C=F'],
      days_back=90
  )
  print('Status:', r['status'], '| Bars added:', r['total_bars_added'])
  "
  ```

---

### ✅ Priority 3 – Production Hardening (Completed)

- [x] **GPU support in Docker** — Separated into dedicated `docker/trainer/Dockerfile` + `docker-compose.train.yml`. Main stack (`docker-compose.yml`) is now CPU-only; `engine` uses Ubuntu 24.04 base with no CUDA dependency.  
- [x] **Model selection logic** — `_find_best_model()` in `breakout_cnn.py`: prefers champion → highest val_accuracy from meta JSON → newest mtime fallback.  
- [x] **Incremental dataset build** — `scripts/incremental_dataset_build.py` (nightly, adds new bars without full regeneration).  
- [x] **Centralized Redis helpers** — `src/lib/core/redis_helpers.py`: typed wrappers for pub/sub, key-value, streams; `publish_and_cache()` combined helper.  
- [x] **Health checks** for model existence + last retrain timestamp — Prometheus gauges + Redis `engine:model_health` key.

---

### 🔲 Priority 4 – Scale & Optimization (Next 30 Days)

- [x] **Full walk-forward backtest** — `scripts/walk_forward_backtest.py`: time-series CV with expanding or rolling windows, all 9 sessions, per-fold CNN threshold tuning, aggregate summary table + CSV export.
- [ ] **Experiment with hybrid CNN+ViT** once dataset reaches 15k+ images.  
- [x] **Per-session CNN thresholds** — `SESSION_THRESHOLDS` dict added to `breakout_cnn.py`. Overnight sessions use lower thresholds (cme=0.75, sydney=0.72, tokyo=0.74, shanghai=0.74); daytime sessions keep 0.82. `get_session_threshold(session_key)` is the single authoritative lookup used by `predict_breakout`, `predict_breakout_batch`, and both backtest scripts.
- [x] **`backtest_filters.py` updated** — now accepts `--session all` (or any named session key), `--cnn-gate 1`, `--cnn-threshold`, prints per-session breakdown table with CNN thresholds. Default source changed to `db`, default gate-mode to `majority`.
- [x] **Per-session CNN gate via Redis** — `get_cnn_gate(session_key)` in `redis_helpers.py` checks `engine:config:cnn_gate:{session_key}` first, falls back to `ORB_CNN_GATE` env var. Engine `_handle_check_orb` updated to use this. REST API at `GET/PUT/DELETE /cnn/gate` + `/cnn/gate/{session_key}`. Dashboard HTML panel at `/cnn/gate/html` with per-row toggles + "🌙 Enable overnight" bulk button. No engine restart needed.
- [ ] **Run walk-forward** after dataset hits 10k+ to get statistically meaningful per-session PF numbers and confirm overnight threshold values.

---

### ✅ Priority 7 – Docker Stack Split (Completed)

- [x] **CPU-only main stack** — Removed `deploy.resources.reservations.devices` (NVIDIA) block from `engine` service in `docker-compose.yml`. Engine now runs on any machine without CUDA drivers.
- [x] **CPU-only engine Dockerfile** — `docker/engine/Dockerfile` now uses `ubuntu:24.04` base (was `nvidia/cuda:12.8.0-runtime-ubuntu24.04`). Installs base deps only — no `[gpu]` extras, no torch.
- [x] **Dedicated trainer Dockerfile** — `docker/trainer/Dockerfile`: CUDA 12.8 runtime + Python 3.13 + `.[gpu]` wheels. Mirrors the old GPU engine image.
- [x] **`docker-compose.train.yml`** — Standalone GPU training stack. Single `trainer` service with `restart: "no"`, `shm_size: 4gb`, NVIDIA device reservation, named volumes (`futures_trainer_models`, `futures_trainer_dataset`, `futures_trainer_data`). Default CMD: `retrain_overnight.py --immediate`.
- [x] **`scripts/sync_models.sh`** — rsync helper to pull trained models from GPU machine back to the main stack. Supports `--export-volume` mode for named Docker volumes. Prints a formatted summary of synced files + val metrics.
- [x] **`RETRAIN_IMMEDIATE` env var** — `retrain_overnight.py` now reads `RETRAIN_IMMEDIATE=1` as an alias for `--immediate`, so `docker-compose.train.yml` can trigger immediate mode without overriding the CMD.
- [x] **`--session all` support** — Both `incremental_dataset_build.py` and `retrain_overnight.py` now accept `all` (and all 9 named session keys) in their `--session` / `--orb-session` CLI args. Previously only `us`, `london`, `both` were accepted.
- [x] **`CNN_ORB_SESSION` env var alias** — Both scripts now read `CNN_ORB_SESSION` (used in todo Quick Commands) as well as their primary env var (`DATASET_ORB_SESSION` / `CNN_RETRAIN_ORB_SESSION`).
- [x] **Monitoring profiles fixed** — `prometheus` and `grafana` now have `profiles: [monitoring]` in `docker-compose.yml` (was documented but the YAML key was missing — they were starting unconditionally).

---

### 🔲 Long-Term Nice-to-Haves

- [ ] **Web dashboard page** for manual review of borderline CNN decisions.  
- [ ] **Hybrid CNN + rule-based ensemble** for even higher precision.  
- [ ] **Multi-timeframe ORB** (combine London + US signals).  
- [ ] **Automated position sizing** based on CNN confidence + volatility regime.

---

### Key Configuration & Defaults (Updated March 2026 — Full 24h + Per-Session Gate)

| Setting                     | Value            | Location                                  |
|-----------------------------|------------------|-------------------------------------------|
| CNN inference threshold     | 0.82             | `breakout_cnn.py`                         |
| Filter gate mode            | `majority`       | `ORB_FILTER_GATE` env var                 |
| CNN hard gate (global)      | Disabled         | `ORB_CNN_GATE=1` to enable globally       |
| CNN hard gate (per-session) | Off by default   | Redis `engine:config:cnn_gate:{key}` or `PUT /cnn/gate/{key}` |
| Account size                | $150,000         | `ACCOUNT_SIZE` env var                    |
| Retrain cadence             | Nightly 02:00 ET | Scheduler (`TRAIN_BREAKOUT_CNN`)          |
| Chart DPI (dataset)         | 150              | `DatasetConfig`                           |
| Chart DPI (live)            | 180              | `RenderConfig`                            |
| Dataset ORB sessions        | `all`            | `CNN_ORB_SESSION` env var (default: all)  |
| Dataset days back           | 90               | `CNN_RETRAIN_DAYS_BACK` env var           |
| Globex day start            | 18:00 ET         | `SessionMode.EVENING` in scheduler        |
| DST handling                | Auto (ZoneInfo)  | `ZoneInfo("America/New_York")` everywhere |
| Active session symbols      | 15 instruments   | `SESSION_ASSETS` in `orb.py`              |

---

### Quick Commands (Copy-Paste Ready)

```bash
# 1. Paper trading already running — check status
bash scripts/paper_trade_start.sh --check
docker compose ps
docker compose logs -f engine 2>&1 | grep -iE 'ORB|CNN|FILTER|BREAKOUT|SESSION'

# 2. Full dataset regeneration — all 9 sessions, 22 symbols, 90 days
#    (run once; subsequent nightly runs are incremental via --skip-existing default)
#    --session all  is now fully supported (was broken before — only us/london/both worked)
#    New universe: metals, energy, equity index, FX, rates (CBOT), agri (CBOT), crypto
PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
  --symbols MGC SIL MHG MCL MNG MES MNQ M2K MYM MBT MET \
            6E 6B 6J 6A 6C 6S ZN ZB ZC ZS ZW \
  --days-back 90 --session all

# 3. Nightly incremental update (run automatically by engine at 02:30 ET)
#    --force-regen flag replaces old --skip-existing=False invocation
PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
  --symbols MGC SIL MHG MCL MNG MES MNQ M2K MYM MBT MET \
            6E 6B 6J 6A 6C 6S ZN ZB ZC ZS ZW \
  --days-back 7 --session all

# 4a. Retrain CNN on dedicated GPU machine (recommended)
docker compose -f docker-compose.train.yml up --build trainer
#     after training completes:
bash scripts/sync_models.sh <gpu-machine-host>
docker compose restart engine

# 4b. Retrain CNN bare-metal on GPU machine (manual override)
PYTHONPATH=src .venv/bin/python scripts/train_gpu.py --epochs 30 --batch-size 64

# 4c. Full overnight pipeline on GPU machine (dataset refresh → train → gate → promote)
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py \
  --immediate --session all \
  --symbols MGC,SIL,MHG,MCL,MNG,MES,MNQ,M2K,MYM,MBT,MET,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW \
  --epochs 30

# 5. Backtest with live path (all sessions)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
  --symbols MGC MES MNQ 6E --source massive --days 30 --gate-mode majority --cnn-gate 1 -v

# 6. Check which session is active right now (DST-safe)
PYTHONPATH=src python -c "
from datetime import datetime, timezone
from lib.services.engine.orb import get_session_for_utc, get_active_session_keys
now = datetime.now(timezone.utc)
s = get_session_for_utc(now)
print('Active session:', s.name if s else 'None')
print('All active:', get_active_session_keys(now))
"

# 7. Daily report
curl -s http://localhost:8000/audit/daily-report | python -m json.tool
docker compose exec engine python scripts/daily_report.py

# 8. Walk-forward backtest — expanding window, all 9 sessions, 6 folds
PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \
  --symbols MGC MES MNQ 6E \
  --source db --days 180 \
  --session all \
  --folds 6 --test-days 30 \
  --gate-mode majority \
  --export wf_results.csv

# 8a. Walk-forward with CNN gate + per-fold threshold tuning
PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \
  --symbols MGC MES MNQ \
  --source db --days 180 \
  --session all \
  --cnn-gate --tune-threshold \
  --folds 6 --export wf_cnn_results.csv

# 8b. Rolling-window walk-forward, London + US only
PYTHONPATH=src .venv/bin/python scripts/walk_forward_backtest.py \
  --symbols MGC 6E \
  --source db --days 180 \
  --session london us \
  --mode rolling --train-days 60 --test-days 30

# 9. Backtest a single session with all filters + CNN gate (per-session thresholds)
PYTHONPATH=src .venv/bin/python scripts/backtest_filters.py \
  --symbols MGC MES MNQ 6E --source db --days 90 \
  --session all --gate-mode majority --cnn-gate 1 \
  --export backtest_all_sessions.csv

# 10. Backfill all new symbols (rates/agri/FX/energy/crypto — no bar history yet)
PYTHONPATH=src .venv/bin/python -c "
from lib.services.engine.backfill import run_backfill
NEW_SYMS = [
    'NG=F',   # Natural Gas
    'ZN=F',   # 10Y T-Note
    'ZB=F',   # 30Y T-Bond
    'ZC=F',   # Corn
    'ZS=F',   # Soybeans
    'ZW=F',   # Wheat
    '6S=F',   # Swiss Franc
    'MET=F',  # Micro Ether
    '6B=F', '6J=F', '6A=F', '6C=F',  # FX (may already have some history)
]
r = run_backfill(symbols=NEW_SYMS, days_back=90)
print('Status:', r['status'], '| Bars added:', r['total_bars_added'])
for s in r['symbols']:
    print(f\"  {s['symbol']:12s}  +{s['bars_added']:6d} bars  ({s['error'] or 'ok'})\")
"

# 10b. Force backfill ALL 23 symbols from scratch (e.g. after DB reset):
PYTHONPATH=src .venv/bin/python -c "
from lib.services.engine.backfill import run_backfill
from lib.core.models import ASSETS
r = run_backfill(symbols=list(set(ASSETS.values())), days_back=90)
print('Status:', r['status'], '| Total bars:', r['total_bars_added'], '| Errors:', len(r['errors']))
"

# 11. GPU machine trainer — one-shot run with custom args (no dataset refresh)
docker compose -f docker-compose.train.yml run --rm trainer \
  python scripts/retrain_overnight.py --skip-dataset --immediate --epochs 30

# 12. Verify monitoring stack is up (requires --profile monitoring)
docker compose --profile monitoring up -d --build
curl -s http://localhost:9095/-/healthy   # Prometheus
curl -s http://localhost:3010/api/health  # Grafana

# 13. Session signal quality audit — cross-reference ORB signals with paper trades
#     Prints per-session win rate, profit factor, CNN P-distribution
PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \
  --days 30 --recommend --min-win-rate 0.58 --min-signals 5 -v

# 13a. Overnight sessions only audit (14 days)
PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \
  --sessions cme sydney tokyo shanghai --days 14 --recommend -v

# 13b. Export audit results + push to Redis for Grafana
PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \
  --days 30 --export-json audit_session_quality.json \
  --export-csv audit_session_quality.csv --push-redis

# 14. Manage per-session CNN gate (no restart needed)
# View all gate states:
curl -s http://localhost:8000/cnn/gate | python -m json.tool

# Enable CNN gate for overnight sessions (after audit confirms signal quality):
curl -X PUT 'http://localhost:8000/cnn/gate/cme?enabled=true'
curl -X PUT 'http://localhost:8000/cnn/gate/sydney?enabled=true'
curl -X PUT 'http://localhost:8000/cnn/gate/tokyo?enabled=true'
curl -X PUT 'http://localhost:8000/cnn/gate/shanghai?enabled=true'

# Disable gate for a single session:
curl -X PUT 'http://localhost:8000/cnn/gate/cme?enabled=false'

# Remove Redis override (revert session to ORB_CNN_GATE env var):
curl -X DELETE 'http://localhost:8000/cnn/gate/cme'

# Reset ALL overrides:
curl -X DELETE 'http://localhost:8000/cnn/gate'

# Via Python (e.g. after running session_signal_audit.py):
PYTHONPATH=src .venv/bin/python -c "
from lib.core.redis_helpers import set_cnn_gate, get_all_cnn_gates
set_cnn_gate('cme', True)
set_cnn_gate('sydney', True)
print(get_all_cnn_gates())
"

---

### Success Metrics to Track Weekly
- Trade frequency: 1–5/day across all sessions (overnight + daytime)
- Win rate: 58–65% (1:2+ R:R) per session — track separately for overnight vs daytime
- Max drawdown: <8% on micro sizing  
- CNN hold-out accuracy: >82% (current champion: 83.73%); target >86% after 10k dataset
- Inference latency: <50 ms (GPU)  
- Daily retrain complete before 04:30 ET
- Overnight session signal count: track CME/Sydney/Tokyo/Shanghai separately in Grafana
- DST transition: next EDT→EST transition (first Sunday of November) — verify session times shift correctly
- Per-session CNN gate state: check `GET /cnn/gate` weekly after reviewing session_signal_audit.py output
- Session signal quality: run `session_signal_audit.py --days 14 --recommend` every Sunday to track overnight session maturity

---

**This is now your single source of truth.**  
System is **live on Sim100** with full 24h Globex coverage across 9 sessions and **22 symbols** (metals, energy, equity index, FX, rates, agri, crypto).

**Immediate next steps:**
1. **Run the full dataset regeneration command** (see Quick Commands #2) — 22-symbol universe × 9 sessions × 90 days → target 10k+ images.
2. **Backfill new symbols first** (see Quick Commands #10) — NG, ZN, ZB, ZC, ZS, ZW, 6S, MET have no bar history yet; run before step 1.
3. **Retrain the CNN** after dataset hits 10k — target 86%+ accuracy with the richer multi-asset, multi-session data.
4. **Run session_signal_audit.py weekly** (see Quick Commands #13) — once overnight sessions (CME, Tokyo, Shanghai) hit ≥5 matched trades and ≥58% win rate, use `PUT /cnn/gate/{session}` to flip the hard gate on.
5. **Run walk-forward backtest** (see Quick Commands #8) after dataset hits 10k — get statistically meaningful per-session PF numbers to confirm overnight threshold values.

**DST reminder**: Next transition is first Sunday of November (EDT→EST). All times auto-adjust via `ZoneInfo("America/New_York")` — no manual intervention needed. ✅
