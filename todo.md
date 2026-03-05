# Futures Trading Co-Pilot — TODO

**Repo:** `futures` (Web UI, Dashboard, Live Engine)
**Scope:** Live market stats, HTMX dashboard, ORB detection, session-aware scheduling, real-time SSE.
**Related:** [orb](https://github.com/nuniesmith/orb) (CNN training) · [ninjatrader](https://github.com/nuniesmith/ninjatrader) (NT8 C# execution)

---

## Current State

- **Dashboard**: HTMX + FastAPI serving live market stats, ORB signals, risk status, Grok AI analyst.
- **Engine**: Session-aware scheduler covering full 24h Globex day (9 sessions, 18:00 ET start).
- **ORB Pipeline**: Detection → 6 deterministic filters (majority gate) → optional CNN inference → Redis publish.
- **Monitoring**: Prometheus + Grafana dashboards (optional profile).
- **CNN Inference**: Model pulled from orb repo (`scripts/sync_models.sh`), CPU-only fallback in engine.
- **NT8 Deploy**: Dashboard panel generates installer .bat that pulls C# from ninjatrader repo.

---

## Active

- [ ] Dashboard polish — improve mobile responsiveness of HTMX panels
- [ ] SSE reconnect reliability — handle Redis connection drops gracefully in SSE stream
- [ ] Health endpoint improvements — surface model staleness, last sync time
- [ ] Add model sync status to dashboard CNN panel (last pull time, version from meta.json)

## Backlog

### Dashboard & Web UI
- [ ] Dark/light theme toggle (currently dark only)
- [ ] Per-session ORB signal history view (table + chart)
- [ ] Trade journal UI improvements — inline editing, tag filtering
- [ ] Grok AI analyst — streaming response display (SSE from Grok API)
- [ ] Volume profile chart visualization in dashboard
- [ ] Historical performance charts (win rate over time, equity curve from journal)

### Engine & Analysis
- [ ] Improve regime detection display — show current HMM state on dashboard
- [ ] Add session-level performance stats to daily report
- [ ] Engine hot-reload of CNN model when `models/breakout_cnn_best.pt` changes (inotify/polling)
- [ ] Backfill gap detection — alert when historical bars have gaps > N minutes

### Infrastructure
- [ ] Add `--sync-models` flag to `run.sh` (force re-pull even if present)
- [ ] Healthcheck improvements — engine should report per-module health (Redis, Postgres, Massive WS)
- [ ] Structured error responses across all API endpoints (consistent JSON shape)
- [ ] Rate limiting tuning — review slowapi config for SSE vs REST endpoints
- [ ] CI/CD — GitHub Actions for test + lint on PR

### Monitoring
- [ ] Alert rules in Prometheus for: model stale > 7 days, engine crash loop, Redis disconnect
- [ ] Grafana dashboard for API latency percentiles (p50, p95, p99)
- [ ] Add ORB filter rejection breakdown to Grafana (which filters reject most)

---

## Completed

- [x] Split repo into three: futures (this), orb (training), ninjatrader (C# execution)
- [x] Remove training code, scripts, dataset generation, NinjaTrader C# source
- [x] Add `scripts/sync_models.sh` to pull CNN model from orb repo
- [x] Update `nt8_deploy.py` to reference ninjatrader repo instead of futures
- [x] Update `run.sh` to auto-pull model if missing
- [x] Clean up `pyproject.toml` — remove GPU extras, training-only deps
- [x] Full 24h Globex coverage (9 sessions from 18:00 ET)
- [x] Per-session CNN gate via Redis (enable/disable per session without restart)
- [x] Prometheus + Grafana monitoring stack (optional profile)
- [x] HTMX dashboard with SSE live updates
- [x] Session-aware scheduler (pre-market, active, off-hours)
- [x] Risk engine integrated with ORB pipeline
- [x] Grok AI morning briefing + live updates
- [x] Daily report generation + email
- [x] NT8 Bridge health monitoring in dashboard