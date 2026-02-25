**✅ Perfect Plan — This Split Will Fix Your Loading Issues Completely**

Splitting the Streamlit app into its own lightweight container + creating a dedicated **data-service** (FastAPI + background workers) is the exact right move.  
The current bottleneck (heavy Massive WS, 5m/1m refreshes, K-Means/wave/signal-quality calcs, Optuna/VectorBT optimizations, CVD/ICT processing) runs in the same process as the UI → slow page loads and reloads.

**New Architecture (after split)**

```
docker-compose services:
├── postgres          ← durable storage (journal, historical opts, alerts)
├── redis             ← hot cache (latest FKS metrics, live positions, 1m bars)
├── data-service      ← FastAPI + APScheduler/Celery
│     ├── Massive WS listener (background)
│     ├── Engine + all FKS modules (vol, wave, signal_quality, regime, cvd, ict)
│     ├── Periodic jobs (5m refresh, optimization, etc.)
│     └── REST API: /latest/{ticker}, /status, /force_refresh, /positions, /journal, etc.
└── streamlit-app     ← Pure UI only (fast reloads)
      └── Calls data-service API + reads Redis directly for ultra-low latency
```

**Benefits you’ll see immediately**
- Streamlit loads in <2 seconds (no more data crunching on every refresh)
- Data-service runs 24/7 in background, even when you close the browser
- Easy scaling later (add more workers, separate WS process, etc.)
- Clean separation: UI = presentation, data-service = business logic

---

### Detailed Task Plan (Phased, Realistic for Solo Dev)

**Phase 0: Preparation (30–45 min)**
- [ ] Create new folder structure (run these commands):
  ```bash
  mkdir -p data_service
  mkdir -p data_service/api data_service/tasks data_service/models
  cp -r src/{engine,volatility,wave_analysis,signal_quality,regime,cvd,ict,costs,massive_client,cache,grok_helper,db} data_service/
  ```
- [ ] Update `.env` (add Postgres if not already there — we discussed this earlier)
  ```
  POSTGRES_PASSWORD=your_secure_pass
  DATABASE_URL=postgresql+psycopg://futures_user:${POSTGRES_PASSWORD}@postgres:5432/futures_db
  DATA_SERVICE_URL=http://data-service:8000
  ```
- [ ] Backup current `docker-compose.yml` as `docker-compose.yml.bak`

**Phase 1: Infrastructure & Docker (1–1.5 hours)**
1. Update `docker-compose.yml` (I’ll give you the exact file when you say “phase 1 ready”)
2. Create `Dockerfile.data` (copy of your current Dockerfile but with `data_service/main.py` as entrypoint)
3. Create `data_service/main.py` (FastAPI skeleton with lifespan for startup/shutdown)
4. Add healthchecks for all services
5. `docker compose up -d --build` → verify all 4 services start cleanly

**Phase 2: Build the Data Service (4–6 hours)**
1. Move heavy logic into `data_service/tasks/`:
   - `background_refresh.py` (APScheduler for 5m/1m cycles)
   - `massive_ws.py` (keep your existing WS manager)
   - `optimization_runner.py` (VectorBT or current Optuna)
2. Create FastAPI routers in `data_service/api/`:
   - `analysis.py` → `/latest/{ticker}` returns full FKS dict (wave + vol + sq + regime + ict + cvd)
   - `actions.py` → `/force_refresh`, `/optimize_now`
   - `positions.py`, `journal.py`
3. Use existing `db.py` + `cache.py` (already Redis + Postgres ready)
4. Add lifespan events:
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       await start_massive_ws()
       scheduler.start()
       yield
       scheduler.shutdown()
   ```
5. Expose Redis pub/sub or SSE for real-time updates (optional but nice)

**Phase 3: Refactor Streamlit into Pure Client (3–4 hours)**
1. Create `streamlit_app/app.py` (new home for your current Streamlit code)
2. Replace every direct call like:
   ```python
   # OLD
   result = engine.get_latest_analysis(ticker)
   ```
   with:
   ```python
   # NEW
   async with httpx.AsyncClient() as client:
       resp = await client.get(f"{DATA_SERVICE_URL}/latest/{ticker}")
       data = resp.json()
   ```
3. For ultra-fast metrics (live positions, 1m SQ) → read directly from Redis (you already have `cache_get`)
4. Keep dashboard tabs exactly the same — only the data source changes

**Phase 4: Database & Persistence (1–2 hours)**
1. Use the `db.py` we discussed earlier (JournalEntry, OptimizationHistory, etc.)
2. Run `init_db()` on data-service startup
3. One-time migration script (SQLite → Postgres) — I can give you the exact script
4. Update journal and alert logging to use Postgres

**Phase 5: Testing & Polish (2–3 hours)**
1. Test flow:
   - Start all containers
   - Verify Massive WS connects in data-service logs
   - Open Streamlit → should load instantly
   - Click “Force Refresh” → see data update
   - Approve signal → goes through bridge
2. Add basic auth or API keys between streamlit ↔ data-service (localhost only = safe)
3. Logging: centralise with structlog or keep separate for now
4. Monitoring: add `/health` and `/metrics` endpoints

**Total Estimated Time**: 12–18 hours spread over 2–3 days (very doable)

---

**Immediate Next Step**

Reply with one of these to get the exact files:

- **“phase 1 ready”** → I’ll give you the full updated `docker-compose.yml` + `Dockerfile.data` + folder structure commands
- **“give me full architecture diagram in mermaid”** → visual overview
- **“start with data_service skeleton”** → I’ll paste the complete `data_service/main.py` + routers

This refactor will make your co-pilot feel **snappy and professional** while keeping everything 100% automatic (still just the account-size slider).

You’re making the right move — this is how production trading dashboards are built in 2026.

Let’s do this! What do you want first?