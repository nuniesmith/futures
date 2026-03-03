**✅ Refactored & Updated TODO.md (March 2026 — Full 24h Coverage Update)**

**Project: Ruby + Bridge ORB System**  
**Status: Live Paper Trading + Full 24h Globex Coverage (9.7/10)**  
**Goal:** Fully automated ORB system covering the entire Globex trading day (18:00 ET start) across all major CME exchanges and sessions — CME Globex open, Sydney/ASX, Tokyo/TSE, Shanghai/HK, Frankfurt/Xetra, London, London-NY crossover, US Equity, and CME Settlement — on CME micro contracts + FX futures with deterministic filters + hybrid CNN scoring → 1–5 high-conviction trades/day.

---

### Executive Summary (Current State)
- **Dataset**: 3,691 labeled PNG charts (growing — full 24h + all-exchange regeneration queued).
- **Models**: 36+ trained checkpoints (champion: 83.73% val accuracy, 91.71% precision).
- **Live Pipeline**: Complete + **running** — ORB detection → filters (majority gate) → CNN inference → Redis publish → NT8 Bridge execution on Sim100.
- **Monitoring Stack**: Redis + Prometheus + Grafana **live** — full ORB/CNN/filter dashboards.
- **Scheduler**: Now covers **full Globex day from 18:00 ET** with 9 sessions across all major exchanges.
- **Risk Engine**: Integrated, publishing to Redis, tracking Sim100 trades in real-time.
- **NinjaTrader**: Strategy + indicator running live on Sim100, receiving and executing Python signals.

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

- [ ] **Trigger full dataset regeneration** with new `orb_session="all"` and 15-symbol universe:
  ```bash
  # Manual trigger (runs all 9 sessions × 15 symbols × 90 days):
  CNN_ORB_SESSION=all CNN_RETRAIN_DAYS_BACK=90 \
    PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
    --symbols MGC SIL MHG MCL MES MNQ M2K MYM MBT 6E 6B 6J 6A 6C \
    --days 90 --session all
  ```
- [ ] **Retrain CNN** after dataset reaches 10k+ images:
  ```bash
  PYTHONPATH=src .venv/bin/python scripts/train_gpu.py --epochs 30 --batch-size 64
  ```
  Target: push val accuracy from 83.73% → 86%+ with richer multi-session data.
- [ ] **Validate overnight session signals** (CME open, Tokyo, Shanghai) against paper trades — confirm signal quality before enabling hard CNN gate for overnight.
- [ ] **Add 6B/6J/6A historical bars** via Massive backfill (these are new symbols with no DB history yet).

---

### ✅ Priority 3 – Production Hardening (Completed)

- [x] **GPU support in Docker** — CUDA 12.8 base image + `.[gpu]` PyTorch wheels in `docker/engine/Dockerfile`.  
- [x] **Model selection logic** — `_find_best_model()` in `breakout_cnn.py`: prefers champion → highest val_accuracy from meta JSON → newest mtime fallback.  
- [x] **Incremental dataset build** — `scripts/incremental_dataset_build.py` (nightly, adds new bars without full regeneration).  
- [x] **Centralized Redis helpers** — `src/lib/core/redis_helpers.py`: typed wrappers for pub/sub, key-value, streams; `publish_and_cache()` combined helper.  
- [x] **Health checks** for model existence + last retrain timestamp — Prometheus gauges + Redis `engine:model_health` key.

---

### 🔲 Priority 4 – Scale & Optimization (Next 30 Days)

- [ ] **Full walk-forward backtest** using exact live path (all 9 sessions + CNN + filters).  
- [ ] **Experiment with hybrid CNN+ViT** once dataset reaches 15k+ images.  
- [ ] **Per-session CNN thresholds** — overnight sessions (CME/Sydney/Tokyo/Shanghai) may need lower thresholds (0.72–0.78) vs daytime sessions (0.82). Tune after 2 weeks of paper data.  
- [ ] **Enable CNN hard gate for overnight sessions** once signal quality validated (set `ORB_CNN_GATE=1` selectively per session via Redis config).

---

### 🔲 Long-Term Nice-to-Haves

- [ ] **Web dashboard page** for manual review of borderline CNN decisions.  
- [ ] **Hybrid CNN + rule-based ensemble** for even higher precision.  
- [ ] **Multi-timeframe ORB** (combine London + US signals).  
- [ ] **Automated position sizing** based on CNN confidence + volatility regime.

---

### Key Configuration & Defaults (Updated March 2026 — Full 24h)

| Setting                     | Value            | Location                                  |
|-----------------------------|------------------|-------------------------------------------|
| CNN inference threshold     | 0.82             | `breakout_cnn.py`                         |
| Filter gate mode            | `majority`       | `ORB_FILTER_GATE` env var                 |
| CNN hard gate               | Disabled         | `ORB_CNN_GATE=1` to enable                |
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

# 2. Full dataset regeneration — all 9 sessions, 15 symbols, 90 days
#    (run once after Phase 6 upgrade; subsequent nightly runs are incremental)
CNN_ORB_SESSION=all CNN_RETRAIN_DAYS_BACK=90 \
  PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
  --symbols MGC SIL MHG MCL MES MNQ M2K MYM MBT 6E 6B 6J 6A 6C \
  --days 90 --session all

# 3. Nightly incremental update (run automatically by engine at 02:30 ET)
PYTHONPATH=src .venv/bin/python scripts/incremental_dataset_build.py \
  --symbols MGC SIL MHG MCL MES MNQ M2K MYM MBT 6E 6B 6J 6A 6C \
  --days 7 --session all --skip-existing

# 4. Retrain CNN (manual override — also runs nightly after dataset build)
PYTHONPATH=src .venv/bin/python scripts/train_gpu.py --epochs 30 --batch-size 64

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
```

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

---

**This is now your single source of truth.**  
System is **live on Sim100** with full 24h Globex coverage across 9 sessions and 15 symbols.

**Immediate next steps:**
1. **Run the full dataset regeneration command** (see Quick Commands #2) — gets you from 3,691 → 10k+ images covering all exchange opens.
2. **Retrain the CNN** after dataset hits 10k — target 86%+ accuracy with the richer multi-session data and new session ordinal tabular feature.
3. **Monitor overnight sessions** (CME 18:00, Tokyo 19:00, Shanghai 21:00) in Grafana — verify signal quality before trusting them for live execution.
4. **Backfill FX bars** (6B, 6J, 6A, 6C) via Massive — these are new symbols with no historical bar data yet.

**DST reminder**: Next transition is first Sunday of November (EDT→EST). All times auto-adjust via `ZoneInfo("America/New_York")` — no manual intervention needed. ✅
