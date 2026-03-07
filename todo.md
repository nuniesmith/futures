# futures — TODO

> **Single repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:engine` · `:web` · `:trainer`
> **NinjaTrader deploy**: `scripts/deploy_nt8.bat` → `scripts/deploy_nt8.ps1` (pulls all C# source + ONNX from this repo)
> **Infrastructure**: Local (Tailscale mesh) — Pi (engine + web), GPU rig (trainer), Windows (NT8)

---

## Repo Layout

```
futures/
├── src/
│   ├── lib/
│   │   ├── core/          # BreakoutType, RangeConfig, session, models, alerts, cache
│   │   ├── analysis/      # breakout_cnn, breakout_filters, chart_renderer, mtf_analyzer, regime, scorer, …
│   │   ├── strategies/    # rb/ (range breakout scalping), daily/ (swing/intraday), backtesting, costs
│   │   ├── services/
│   │   │   ├── engine/    # main, handlers, scheduler, position_manager, backfill, risk, focus, …
│   │   │   ├── web/       # HTMX dashboard, FastAPI reverse-proxy (port 8080)
│   │   │   └── data/      # FastAPI data API (positions, SSE, bridge, trades, journal, kraken, …)
│   │   └── integrations/  # kraken_client, massive_client, grok_helper
│   └── ninja/
│       ├── BreakoutStrategy.cs   # NT8 strategy (single-file, all deps inlined)
│       ├── RubyIndicator.cs      # NT8 chart indicator
│       └── addons/
│           ├── Bridge.cs         # NT8 HTTP bridge AddOn (port 5680)
│           └── DataPreloader.cs  # NT8 history seeder AddOn
├── models/                # champion .pt, .onnx, feature_contract.json (Git LFS)
├── scripts/
│   ├── deploy_nt8.bat     # Windows launcher (double-click to deploy NT8)
│   ├── deploy_nt8.ps1     # NT8 deploy: pulls C# + DLLs + ONNX from this repo
│   ├── sync_models.sh     # Pi-side: pull .pt + .onnx from this repo → restart engine
│   └── …
├── config/                # Prometheus, Grafana, Alertmanager
├── docker/                # Dockerfiles per service
└── docker-compose.yml
```

---

## Current State

- **Monorepo**: All source — engine, web, trainer, lib, C# strategies, deploy scripts — lives here. No separate repos.
- **Models**: `models/breakout_cnn_best.pt` + `.onnx` + `feature_contract.json` committed (Git LFS). Engine pulls via `sync_models.sh`, NT8 pulls via `deploy_nt8.ps1`. Latest champion: **87.1% accuracy**, 87.15% precision, 87.27% recall, 25 epochs, v6 18-feature, 41 checkpoints saved.
- **Docker**: `:engine` (data API + CNN inference), `:web` (HTMX dashboard), `:trainer` (GPU training server). Runs on Pi (engine + web) and GPU rig (trainer).
- **Feature Contract**: v6, 18 tabular features. `models/feature_contract.json` is the canonical source for both Python and C#.
- **CNN Model**: EfficientNetV2-S + tabular head. ONNX export for NT8. `OrbCnnPredictor` auto-detects tabular dimension from ONNX metadata at load time. Training pipeline: generate dataset → train → evaluate → gate check (≥80% acc, ≥75% prec, ≥70% rec) → promote → export ONNX + feature_contract.json.
- **Breakout Types**: 13 — ORB, PrevDay, InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci. Fully wired in engine detection, training, dataset generator, C# enum, CNN tabular vector, chart renderer, `UpdateRangeWindow`, and `CheckBreakout`.
- **NT8 Strategy**: `BreakoutStrategy.cs` — 5 core instruments (`MGC, MES, MNQ, MYM, 6E`), v6 18-feature tabular vector, all 13 `BreakoutType` range builders fully implemented (no stubs remain), 13 chart renderer box styles, 3-phase bracket walk (TP1 → breakeven → EMA9 trail to TP3), per-type `tp3_atr_mult` loaded from `feature_contract.json` via `ParseTp3MultsFromContract` + `ApplyTp3MultsToStates`, crash-resilient. TPT Mode built-in ($50k/2, $100k/3, $150k/4 micros).
- **NT8 Bridge**: `Bridge.cs` AddOn — HTTP listener (port 5680), position push to dashboard (`POST /api/positions/update`), 15s heartbeat, `SignalBus.IsRiskBlocked` risk gate, Prometheus `/metrics`, all endpoints including `/execute_signal`, `/flatten`, `/cancel_orders`, `/status`, `/orders`, `/health`.
- **NT8 Deploy**: `deploy_nt8.bat` / `deploy_nt8.ps1` — downloads C# source, DLLs, ONNX model, `feature_contract.json` from this repo and installs into NT8.
- **TP3 + EMA9 Trailing**: 3-phase bracket in both Python `position_manager.py` and C# `BreakoutStrategy.cs`. Phase 1 (SL/TP1) → Phase 2 (breakeven) → Phase 3 (EMA9 trail to TP3).
- **Position Manager**: `position_manager.py` — always-in 1-lot micro positions, reversal gates (CNN ≥ 0.85, MTF ≥ 0.60, 30min cooldown), Redis persistence, `OrderCommand` emitter for Bridge.
- **Dashboard**: HTMX + FastAPI — live signals, 13 breakout type filter pills, 9 session tabs, MTF score column, trade journal, Kraken crypto chart + correlation panel, Grok AI analyst, CNN dataset preview, positions panel, Bridge status badge, flatten/cancel buttons, market regime (HMM), performance panel, volume profile, equity curve, asset focus cards with entry/stop/TP levels.
- **Kraken Integration**: `KrakenDataProvider` REST client (public + private endpoints), `KrakenFeedManager` WebSocket feed (OHLC + trades), 9 crypto pairs (ETH, SOL, LINK, AVAX, DOT, ADA + more), API key/secret from env vars, auto-start on engine boot when `ENABLE_KRAKEN_CRYPTO=1`. REST API connected, WS live with 5097+ bars and 7214+ trades streaming.
- **Massive Integration**: `MassiveDataProvider` REST client (futures OHLCV, snapshots, contracts), `MassiveFeedManager` WebSocket (bars, trades, quotes), front-month resolution, used as primary bars source for training.
- **Data Service**: Unified data layer inside engine service — checks Redis cache first, then Postgres, then fetches from external APIs (Massive for futures, Kraken for crypto). Startup cache warming from Postgres (7 days). Bar data persisted to both Redis (hot) and Postgres (durable).
- **Training**: `trainer_server.py` FastAPI HTTP server (port 8200). `dataset_generator.py` covers all 13 types + all 9 sessions + Kraken. Session-specific thresholds. Type embedding. Synthetic augmentation. Balanced sampling. Full pipeline: generate → split (85/15 stratified) → train → evaluate → gate → promote → ONNX export.
- **Monitoring**: Prometheus + Grafana. "Training Data Health" dashboard provisioned.
- **CI/CD**: Lint → Test → Build & push 3 Docker images → Deploy to Pi via Tailscale SSH → Health checks → Discord notifications. All on push to `main`.
- **Tailscale**: NT8 Windows at `100.127.182.112`, Pi (Docker) at `100.100.84.48`, all services communicate over Tailscale mesh. HTTP only (no domain/TLS needed for local mesh).

---

## Architecture Issues Identified (Pre-Refactor)

### Triple duplication of breakout types & config
Three separate files define breakout type enums, range configs, ATR computation, and quality gates independently:
- `lib/core/breakout_types.py` — canonical `BreakoutType` (IntEnum) + `RangeConfig` with TP/SL/box styling (CNN/training canonical source)
- `lib/services/engine/breakout.py` — **second** `BreakoutType` (StrEnum) + **second** `RangeConfig` with detection thresholds (engine runtime)
- `lib/services/engine/orb.py` — **third** dataclass `ORBSession` with its own ATR params, quality gates, breakout detection

Bridge mapping dicts (`_ENGINE_TO_TRAINING`, `_TRAINING_TO_ENGINE`) exist purely to convert between the two BreakoutType enums. These should not exist.

### `orb.py` is an isolated silo (1800+ lines)
- Has its own `ORBResult`, `detect_opening_range_breakout()`, `compute_atr()`, quality gates, Redis publishing
- `breakout.py` was built to "generalize" ORB but actually lives alongside it with parallel code paths
- `main.py` has **10 separate `_handle_check_orb_*` functions** (one per session) that all delegate to the same `_handle_check_orb`

### `main.py` is a 3285-line god module
- `_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` are 90% copy-paste
- Each handler repeats the same: fetch bars → detect → get HTF bars → MTF enrich → persist → publish → dispatch to PM → send alert
- ~400 lines of duplicate handler code that should be one generic function

### `analysis/orb_filters.py` is misnamed
- The filters (NR7, premarket range, session window, lunch filter, MTF bias, VWAP confluence) are NOT ORB-specific
- They apply to any range breakout type — the name creates false coupling

### No daily/swing strategy layer
- Everything is built for intraday scalping breakouts
- No structure for the "daily bias trade" concept you want (historical analysis → directional conviction → big move, small position)

### Web UI focus is too broad for live trading
- `focus.py` computes focus for ALL assets (10+ CME + crypto)
- No mechanism to narrow to 3-4 focused assets per day for live trading simplicity
- Too much information on screen when you're trying to execute quick decisions

### Risk system is not real-time position-aware
- `RiskManager` tracks open positions but doesn't feed back into focus cards or position sizing in real time
- When you have a live position, the dashboard still shows static entry/stop/TP levels from the morning focus computation — not dynamically adjusted for current risk exposure
- No concept of "remaining risk budget" — if you're already in 2 trades, the focus cards should show reduced position sizes for the next entry
- Position sizing only shows micro contracts — no side-by-side micro vs regular contract comparison for quick decision-making on NinjaTrader
- `RiskManager` and `PositionManager` are separate systems that don't talk to each other in real time

### Asset model is ticker-centric, not generalized
- The system thinks in tickers (`MGC=F`, `MES=F`) not generalized assets ("Gold", "S&P")
- Both `MICRO_CONTRACT_SPECS` and `FULL_CONTRACT_SPECS` exist but are selected by a global `CONTRACT_MODE` env var — you can't see both at once
- When live trading on NinjaTrader, you know which symbol to use (micro vs regular) based on your account — the dashboard should show you both options with dollar values
- No unified "Gold" asset that links MGC (micro), GC (full), and KRAKEN:XAUUSD (spot) together

### CNN treats all assets as flat feature vectors
- `asset_class_id` is a single ordinal (0=equity, 1=fx, 2=metals, 3=treasuries, 4=crypto) — too coarse
- No cross-asset learning: the CNN can't see that gold and silver are correlated metals, or that MES and MNQ move together
- No mechanism to discover hidden correlations or regime-dependent relationships between asset classes
- The model doesn't learn "what makes gold gold" — it just knows gold is class 2 (metals/energy)

### No Kraken spot portfolio management
- Kraken integration is read-only (OHLCV + ticker data for analysis)
- `get_balance()`, `get_trade_balance()` private endpoints exist but aren't used
- No concept of maintaining target crypto allocations or rebalancing
- Futures hard-stop at 4 PM ET is set up, but no equivalent portfolio rules for 24/7 crypto spot holdings

---

## 🔴 High Priority — RB System Refactor

The core architectural change: **ORB becomes a sub-type of RB (Range Breakout)**, not the other way around. All 13 breakout types are peers under the RB system. On top of this fast-scalping RB system, we add a separate Daily Strategy layer for slower intraday swing trades.

### Phase 1A: Merge BreakoutType Enums → Single Source of Truth
- [ ] **Eliminate the engine StrEnum** in `services/engine/breakout.py` — use `core/breakout_types.BreakoutType` (IntEnum) everywhere
  - Remove `class BreakoutType(StrEnum)` from `breakout.py`
  - Remove `_ENGINE_TO_TRAINING` / `_TRAINING_TO_ENGINE` mapping dicts
  - Remove `to_training_type()` / `from_training_type()` / `breakout_type_ordinal()` bridge functions
  - Update all engine callers to import from `lib.core.breakout_types`
  - `BreakoutResult.breakout_type` changes from StrEnum → IntEnum
  - `BreakoutResult.to_dict()` uses `.name` for JSON serialization (human-readable) and `.value` for ordinals
  - Keep backward compat in Redis/SSE payloads: `"type": "ORB"` still works (use `.name`)

### Phase 1B: Merge `RangeConfig` → Single Dataclass
- [ ] **Unify the two RangeConfig dataclasses** — one in `core/breakout_types.py`, one in `engine/breakout.py`
  - The core `RangeConfig` has: TP/SL mults, box styling, CNN ordinals, EMA trail config
  - The engine `RangeConfig` has: detection thresholds (ATR mult, body ratio, range caps, squeeze params)
  - Merge detection-threshold fields INTO the core `RangeConfig` — it becomes the single config for everything
  - All 13 `_*_CONFIG` registry entries in `breakout_types.py` get the detection fields added
  - Engine `breakout.py` imports from `core/breakout_types` and reads from the unified config
  - `get_range_config(BreakoutType.ORB)` returns everything: thresholds, TP/SL, box style, CNN ordinal
  - Kill the engine-side `RangeConfig` entirely

### Phase 1C: Merge ORB Detection into Unified RB Detector
- [ ] **Make ORB just another RB type** — `detect_opening_range_breakout()` becomes `detect_range_breakout(config=ORB_CONFIG)`
  - Extract range-building functions from `orb.py` and `breakout.py` into `strategies/rb/range_builders.py`:
    - `_build_orb_range()` — from orb.py's `compute_opening_range()`, session-parameterized
    - `_build_pdr_range()` — from breakout.py
    - `_build_ib_range()` — from breakout.py
    - `_build_consolidation_range()` — from breakout.py
    - `_build_weekly_range()`, `_build_monthly_range()`, `_build_asian_range()` — from breakout.py
    - `_build_bbsqueeze_range()`, `_build_va_range()`, `_build_inside_day_range()` — from breakout.py
    - `_build_gap_rejection_range()`, `_build_pivot_range()`, `_build_fibonacci_range()` — from breakout.py
  - Single `detect_range_breakout(bars, symbol, config)` in `strategies/rb/detector.py`:
    - Dispatches to the correct range builder based on `config.breakout_type`
    - Applies quality gates (depth, body ratio, range size) uniformly
    - Returns a unified `BreakoutResult` regardless of type
  - ORB session logic (wraps_midnight, session windows, `ORBSession` instances) stays but moves into the ORB range builder
  - `ORBResult` is retired — `BreakoutResult` covers all types including ORB
  - `MultiSessionORBResult` renamed to `MultiSessionResult` — works for any type scanned across sessions
  - ATR computation: single `compute_atr()` in `strategies/rb/detector.py` (deduplicate the 3 copies)

### Phase 1D: Extract Generic Handler Pipeline from `main.py`
- [ ] **One handler function for all 13 breakout types** — eliminate ~400 lines of copy-paste
  - Create `services/engine/handlers.py` with a single `handle_breakout_check()`:
    ```
    def handle_breakout_check(engine, breakout_type: BreakoutType, session_key: str):
        assets = get_assets_for_session_key(session_key)
        for asset in assets:
            bars_1m = fetch_bars_1m(engine, ticker, symbol)
            config = get_range_config(breakout_type)
            result = detect_range_breakout(bars_1m, symbol, config)
            bars_htf = get_htf_bars(bars_1m, ticker)
            result = run_mtf_on_result(result, bars_htf)
            persist_breakout_result(result, session_key)
            if result.breakout_detected:
                publish_breakout_result(result, session_key)
                dispatch_to_position_manager(result, bars_1m, session_key)
                send_breakout_alert(result, breakout_type)
    ```
  - `_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` in `main.py` become one-liners:
    - `handle_breakout_check(engine, BreakoutType.PrevDay, "london_ny")`
    - `handle_breakout_check(engine, BreakoutType.InitialBalance, "us")`
    - `handle_breakout_check(engine, BreakoutType.Consolidation, "london_ny")`
  - `_handle_check_orb` calls `handle_breakout_check(engine, BreakoutType.ORB, session.key)` with the ORB session config attached
  - 10 `_handle_check_orb_*` thin wrappers stay (they just pass the session object) but become 2-liners
  - Extract shared helpers to `handlers.py`: `fetch_bars_1m`, `get_htf_bars`, `run_mtf_on_result`, `persist_breakout_result`, `publish_breakout_result`, `send_breakout_alert`
  - `_handle_check_breakout_multi` delegates to the generic handler for each type in the list

### Phase 1E: Rename `orb_filters.py` → `breakout_filters.py`
- [ ] **Rename and update imports** — these filters apply to ALL range breakout types
  - `lib/analysis/orb_filters.py` → `lib/analysis/breakout_filters.py`
  - `ORBFilterResult` → `BreakoutFilterResult`
  - `apply_all_filters()` signature stays the same — it already accepts a generic result dict
  - Update all imports in: `main.py`, `__init__.py` files, tests
  - Add backward-compat re-export in `lib/analysis/__init__.py` if needed during transition

### Phase 1F: Rename `orb_simulator.py` → `rb_simulator.py`
- [ ] **Rename training simulator** — it already handles PDR, IB, Consolidation batch sims
  - `lib/services/training/orb_simulator.py` → `lib/services/training/rb_simulator.py`
  - `simulate_orb_outcome` → `simulate_rb_outcome` (keep old name as alias during transition)
  - `ORBSimResult` → `RBSimResult`
  - Update imports in: `dataset_generator.py`, `trainer_server.py`, tests

### Phase 1G: Create `lib/strategies/` Package
- [ ] **Move trading logic into strategies package** — clean separation of strategy code from infrastructure
  - Create `lib/strategies/__init__.py`
  - Create `lib/strategies/rb/__init__.py` — the Range Breakout scalping system
  - Create `lib/strategies/rb/detector.py` — unified `detect_range_breakout()`
  - Create `lib/strategies/rb/range_builders.py` — all `_build_*_range()` functions
  - Create `lib/strategies/rb/publisher.py` — Redis pub + alerting (extracted from main.py)
  - Move `lib/trading/costs.py` → `lib/strategies/costs.py`
  - Move `lib/trading/strategies.py` → `lib/strategies/strategy_defs.py` (backtesting strategy classes)
  - Move `lib/trading/engine.py` → `lib/strategies/backtesting.py` (DashboardEngine, run_backtest, etc.)
  - Keep `lib/trading/` as a deprecated redirect (thin `__init__.py` that re-exports from `strategies/`) until all imports are updated
  - Rename `multi_session.py` → `session.py` and `ORBSession` → `RBSession` (keep old names as aliases)

---

## 🔴 High Priority — Daily Strategy Layer

### Phase 2A: Daily Bias Analyzer
- [ ] **Create `lib/strategies/daily/bias_analyzer.py`** — "what direction for today?" per asset
  - Inputs: prior day's OHLCV, prior week's OHLCV, monthly trend, ATR regime
  - Prior day candle classification: inside day, outside day, doji, bullish engulfing, bearish engulfing, hammer, shooting star, strong close (upper/lower 25% of range)
  - Weekly range position: where price closed relative to the prior week's high/low (0.0 = at low, 1.0 = at high)
  - Monthly trend score: slope of 20-day EMA on daily bars, normalized [-1, +1]
  - Volume confirmation: was yesterday's volume above/below the 20-day average?
  - Overnight gap context: gap direction and size relative to ATR (from Globex open vs prior close)
  - Output: `DailyBias` dataclass per asset — direction (LONG/SHORT/NEUTRAL), confidence (0-1), reasoning string, key levels (support/resistance derived from prior day H/L, weekly H/L)
  - Pure computation — no side effects, fully testable

### Phase 2B: Daily Trade Plan Generator
- [ ] **Create `lib/strategies/daily/daily_plan.py`** — orchestrates daily trade selection
  - Morning routine (runs at pre-market, ~05:00-06:00 ET):
    1. Run `bias_analyzer` on all 10+ tracked assets
    2. Optionally call Grok for macro context (economic calendar, overnight news, sector rotation)
    3. Score each asset for daily swing potential: bias confidence × ATR opportunity × volume regime × catalyst presence
    4. Select 1-2 daily swing candidates (biggest expected move, highest conviction direction)
    5. Compute entry zone, stop, TP for daily swing (wider than scalp: SL at 1.5-2× ATR, TP at 3-5× ATR)
    6. Position size: small (1 micro contract) — these are "big move, small risk" trades
  - Output: `DailyPlan` dataclass — swing_candidates (1-2 assets), scalp_focus (3-4 assets for RB system), market_context from Grok, no_trade_flags
  - Persist to Redis key `engine:daily_plan` for dashboard consumption
  - Separate from the RB scalping system — daily trades run on different timeframe and risk profile

### Phase 2C: Swing Detector
- [ ] **Create `lib/strategies/daily/swing_detector.py`** — entry/exit logic for daily trades
  - Uses the daily bias + key levels from `bias_analyzer` to define trade parameters
  - Entry styles:
    - **Pullback entry**: wait for price to pull back to a key level (prior day H/L, VWAP, EMA) in the direction of the daily bias, then enter on confirmation bar
    - **Breakout entry**: enter when price breaks the prior day high (for long bias) or low (for short bias) with volume confirmation
    - **Gap continuation**: if overnight gap aligns with daily bias and doesn't fill in first 30min, enter on first pullback
  - Exit logic:
    - TP1 at 2× ATR (scale 50%), TP2 at 3.5× ATR (scale remaining), or trail with EMA-21 on 15m bars
    - SL at 1.5× ATR from entry — wider than scalp trades
    - Time stop: close by 15:30 ET if neither TP nor SL hit (no overnight holds)
  - These trades coexist with the always-running RB scalping system — different position tracking, different risk budget
  - Daily trades use a separate risk allocation (e.g., 0.5% of account vs 0.75% for scalps)

---

## 🔴 High Priority — Web UI Focus Narrowing & Live Risk

### Phase 3A: Top-4 Asset Selection for Live Trading
- [ ] **Add `select_daily_focus_assets()` to `focus.py`** — narrows the full asset list to 3-4 per day
  - Composite ranking score (0-100) per asset:
    - Signal quality weight (30%): from existing `compute_signal_quality()`
    - ATR opportunity (25%): normalized ATR as % of price — higher = more tradeable
    - RB setup density (20%): how many breakout types are forming ranges near current price
    - Session fit (15%): is this asset's best session (London for gold/FX, US for indices) currently active?
    - Catalyst presence (10%): from `scorer.py` economic event calendar
  - Select top 3-4 by composite score — these become the "focused assets" for the trading day
  - The daily swing candidates (from Phase 2B) may be different from the scalp focus assets
  - Persist to Redis: `engine:focus_assets` (list of 3-4 tickers for scalping) + `engine:swing_assets` (1-2 for daily)
  - The full watchlist still runs in the background (signals fire, CNN infers, data flows) — but the UI only shows the focused set

### Phase 3B: Dashboard Focus Mode
- [ ] **Update web UI to show focused assets prominently** — simplify live trading view
  - Top section: 3-4 focused asset cards (large, prominent, with live price + RB signals + bias)
  - Each focused card shows:
    - Current price + direction bias (from daily plan)
    - Active RB signals (any of the 13 types that are firing or forming)
    - Key levels: prior day H/L, session VWAP, ORB range edges
    - CNN probability for the latest signal
    - Position status from Bridge (if in a trade)
  - Below: 1-2 daily swing candidate cards (different styling — labeled "DAILY SWING", wider TP levels)
  - Collapsed/minimized section: remaining assets from the full watchlist (expandable if needed)
  - "Why these assets?" tooltip/section explaining the composite score ranking
  - Live trading mode auto-hides review panels (already implemented) — now also auto-focuses on the selected assets

### Phase 3C: Grok Integration for Daily Selection
- [ ] **Add Grok analysis call during daily plan generation** — optional but valuable
  - During pre-market `daily_plan.py` run, if `XAI_API_KEY` is set:
    - Send Grok a prompt with: overnight price action summary, economic calendar for the day, sector/asset correlation snapshot, prior day's performance per asset
    - Ask for: macro bias (risk-on/risk-off), top 2-3 assets to watch, key levels to monitor, events that could cause big moves
    - Grok response gets parsed and folded into the `DailyPlan.market_context` field
  - Dashboard shows Grok's morning brief in a dedicated card above the focused assets
  - During live trading, the existing "⚡ Update" button in Grok panel can be used for ad-hoc analysis of specific setups
  - This is supplementary — the system works fine without Grok, it just adds macro context

---

## 🔴 High Priority — Live Risk-Aware Position Sizing

The goal: when you're live trading for a few hours in the morning, everything is real-time, fast, and always keeping you up to date. Position sizes adjust dynamically based on current risk exposure. You see micro AND regular contract values side by side so you know exactly what to type into NinjaTrader. The strategy runs itself — you poke in when you see something good. Helps manage emotions.

### Phase 5A: Generalized Asset Model
- [ ] **Create `lib/core/asset_registry.py`** — unified asset abstraction that links micro, regular, and spot variants
  - `Asset` dataclass: generalized name ("Gold", "S&P", "Bitcoin"), asset_class (metals, equity_index, fx, energy, treasuries, ags, crypto)
  - Each `Asset` holds a dict of `ContractVariant` objects:
    - `micro`: ticker="MGC=F", point_value=10, tick=0.10, margin=1100
    - `full`: ticker="GC=F", point_value=100, tick=0.10, margin=11000
    - `spot`: ticker="KRAKEN:XAUUSD" (for crypto assets, or None for pure futures)
  - `ASSET_REGISTRY: dict[str, Asset]` — single lookup: `ASSET_REGISTRY["Gold"].micro.ticker` → `"MGC=F"`
  - Replaces the split between `MICRO_CONTRACT_SPECS`, `FULL_CONTRACT_SPECS`, `KRAKEN_CONTRACT_SPECS`
  - `get_asset_by_ticker("MGC=F")` → returns the "Gold" `Asset` regardless of which variant was passed
  - `get_variants("Gold")` → `{"micro": ContractVariant(...), "full": ContractVariant(...)}` — for dashboard display
  - Backward-compat: `CONTRACT_SPECS`, `ASSETS`, `TICKER_TO_NAME` still work but delegate to the registry
  - Asset class grouping: `get_asset_group("metals")` → `["Gold", "Silver", "Copper"]` — for cross-referencing

### Phase 5B: Real-Time Risk Budget Integration
- [ ] **Wire `RiskManager` ↔ `PositionManager` into a unified live risk state** — published to Redis every tick
  - New `LiveRiskState` dataclass that merges:
    - From `RiskManager`: account_size, daily_pnl, max_daily_loss, can_trade, block_reason, consecutive_losses
    - From `PositionManager`: all active `MicroPosition` objects with current P&L, bracket phase, R-multiple
    - Computed fields: `remaining_risk_budget` = max_risk_per_trade × (max_open_trades − current_open), `total_unrealized_pnl`, `total_margin_used`, `margin_remaining`
  - Published to Redis key `engine:live_risk` every 5 seconds (or on every Bridge position update push)
  - SSE channel `dashboard:live_risk` for real-time push to web UI
  - `RiskManager.sync_positions()` already receives Bridge position updates — enhance to recompute `LiveRiskState` on every sync
  - When a position is opened/closed, immediately recompute and publish — don't wait for the next 5s interval

### Phase 5C: Dynamic Position Sizing on Focus Cards
- [ ] **Focus cards update in real time based on live risk state** — not just the morning pre-market computation
  - `compute_asset_focus()` gets a new optional param: `live_risk: LiveRiskState | None`
  - When `live_risk` is provided:
    - `remaining_risk_budget` replaces static `max_risk_per_trade` for position sizing
    - If already in a position on this asset: card shows LIVE position info (direction, entry, current P&L, bracket phase, R-multiple) instead of entry zone
    - If at max open trades: position_size shows 0 with "MAX POSITIONS" badge
    - If daily loss limit hit: all cards show "RISK BLOCKED" overlay
  - **Show both micro and regular contract sizing side by side:**
    - "📏 Micro: 3× MGC @ $330 risk" / "📏 Full: 1× GC @ $1,100 risk" — computed from the same stop distance
    - Use `Asset.micro` and `Asset.full` from the registry to compute both simultaneously
    - Trader knows which to use based on their account tier — just reads the number and types it into NT8
  - Dollar P&L estimates for TP1/TP2 shown for BOTH contract sizes:
    - "TP1: +$660 (micro 3×) / +$2,200 (full 1×)"
  - Card refreshes via SSE `dashboard:live_risk` — no page reload, no polling, instant updates
  - When Bridge pushes a position update, the relevant asset card flips from "setup" mode to "live position" mode within 1-2 seconds

### Phase 5D: Live Position Overlay on Focus Cards
- [ ] **When in a trade, the focus card becomes a position management card** — real-time P&L and bracket status
  - Header changes from "🟢 LONG setup" to "🟢 LONG LIVE — Phase 2 (Breakeven)" with green/red pulse animation
  - Shows: entry price, current price, unrealized P&L ($), R-multiple, hold duration, bracket phase
  - Bracket progress bar: `[ENTRY]---[TP1 ✓]---[TP2]---[TP3]` with current price marker
  - Stop loss level shown with distance in ticks and dollars
  - "Close Position" button (fires Bridge `/flatten` for that instrument)
  - "Move to Breakeven" manual override button (fires Bridge stop modification)
  - When position closes (TP hit, SL hit, or manual close): card flips back to "setup" mode with a brief P&L summary flash (+$X or -$X)
  - All updates driven by Bridge position push → Redis → SSE — no polling

### Phase 5E: Risk Dashboard Strip
- [ ] **Add a persistent risk strip at the top of the trading dashboard** — always visible, always current
  - Horizontal bar showing: Daily P&L ($), Open Positions (N/max), Risk Exposure (%), Margin Used/Available, Consecutive Losses, Session Time Remaining
  - Color-coded: green (healthy) → yellow (approaching limits) → red (blocked)
  - Flashes/pulses when a risk state changes (new position opened, loss taken, limit approaching)
  - "RISK BLOCKED" full-width red banner when `can_trade` is false — hard to miss
  - Updates via same `dashboard:live_risk` SSE channel — 1-2 second latency from NT8 to screen

---

## 🟡 Medium Priority — CNN Expansion (v7 Feature Contract)

### Phase 4A: New Features from Daily Strategy Layer
- [ ] **Expand CNN tabular features from 18 → 24** — leverage the new daily/historical analysis
  - Feature #19: `daily_bias_direction` — from `bias_analyzer.py`, encoded as -1 (short), 0 (neutral), +1 (long), normalized to [0, 1]
  - Feature #20: `daily_bias_confidence` — 0.0 to 1.0 scalar from bias analyzer
  - Feature #21: `prior_day_pattern` — ordinal encoding of yesterday's candle pattern (inside=0, doji=1, engulfing_bull=2, engulfing_bear=3, hammer=4, shooting_star=5, strong_close_up=6, strong_close_down=7), normalized to [0, 1]
  - Feature #22: `weekly_range_position` — where price sits within prior week's high/low range, 0.0 (at low) to 1.0 (at high)
  - Feature #23: `monthly_trend_score` — normalized slope of 20-day EMA on daily bars, [-1, +1] mapped to [0, 1]
  - Feature #24: `crypto_momentum_score` — from `crypto_momentum.py` (already built, needs wiring into feature contract)
  - Update `feature_contract.json` to v7 with 24 features
  - Update `breakout_cnn.py` `TABULAR_FEATURES` list
  - Update `dataset_generator.py` `_build_row()` to compute and include new features
  - Update C# `BreakoutStrategy.cs` to build 24-element tabular vector (add daily bias fields)
  - ONNX auto-adapt already handles dimension changes — just needs new feature_contract.json

### Phase 4B: Sub-Features and Richer Encoding
- [ ] **Add sub-feature decomposition for existing features** — make the CNN see more nuance
  - `breakout_type_ord` → split into `breakout_type_category` (time-based=0, range-based=0.5, squeeze-based=1.0) + existing ordinal
  - `session_ordinal` → add `session_overlap_flag` (1.0 if London+NY overlap, 0.0 otherwise) — captures the highest-volume window
  - `atr_regime` → add `atr_trend` (is ATR expanding or contracting over last 10 bars? 1.0 = expanding, 0.0 = contracting)
  - `volume_surge_ratio` → add `volume_trend` (5-bar volume slope — rising volume into breakout is bullish for continuation)
  - These sub-features don't replace existing ones — they add alongside for richer representation
  - Target: v7 contract with ~28-30 total features (24 base + 4-6 sub-features)

### Phase 4C: Retrain on v7 Contract
- [ ] **Full retrain with expanded feature set** — target ≥89% accuracy
  - Generate new dataset with all 24+ features across all 25 symbols, 13 types, 9 sessions
  - Daily bias features computed from historical daily bars (look back 1 day for each sample's date)
  - Weekly/monthly features computed from historical weekly/monthly bars
  - Crypto momentum features computed from aligned Kraken data
  - Train with same architecture (EfficientNetV2-S + tabular head) but larger tabular input
  - Gate check: ≥88% acc, ≥85% prec, ≥82% rec (higher bar than v6 since we have more features)
  - Export ONNX + feature_contract.json v7
  - Deploy to NT8 via `deploy_nt8.ps1` — C# auto-adapts to new tabular dimension

---

## 🟡 Medium Priority — CNN Asset-Class Intelligence (v8+)

The CNN currently treats `asset_class_id` as a single flat ordinal — it knows gold is "2" but doesn't know *why* gold is gold. Phase 7 adds hierarchical asset understanding so the model can learn what makes each asset class unique, how assets within a class relate, and discover hidden cross-asset correlations and regime-dependent states.

### Phase 7A: Hierarchical Asset Embedding
- [ ] **Replace flat `asset_class_id` with a learned embedding** — let the CNN discover asset relationships
  - Instead of a single ordinal (0-4), give the CNN a richer asset identity:
    - `asset_class_embedding` — 4-dim learned vector per asset class (metals, equity_index, fx, energy, crypto, treasuries, ags → 7 classes)
    - `asset_id_embedding` — 8-dim learned vector per individual asset (Gold, Silver, Copper, S&P, Nasdaq, etc.)
    - These embeddings are trained end-to-end with the CNN — the model discovers what makes gold similar to silver but different from crude oil
  - Replace the tabular head's flat `asset_class_id` + `asset_volatility_class` with the embedding vectors
  - Embedding lookup table stored in `feature_contract.json` so C# can reconstruct the same vectors
  - Net feature count change: remove 2 flat features, add 12 embedding dims → net +10 features
  - **Why this matters**: the model currently can't distinguish between "Gold breakout during London" and "S&P breakout during London" at the asset-identity level — it only sees class=2 vs class=0. With embeddings, it learns Gold's unique volatility structure, session preferences, and correlation patterns

### Phase 7B: Cross-Asset Correlation Features
- [ ] **Add real-time cross-asset correlation signals as CNN features** — discover hidden states
  - For each breakout signal, compute rolling correlations with related assets:
    - Gold signal → include: Silver correlation (30-bar rolling Pearson), Copper correlation, DXY proxy (6E inverse), S&P correlation
    - S&P signal → include: Nasdaq correlation, Russell correlation, VIX proxy (from options-derived vol), Gold inverse correlation
    - Crude signal → include: Natural Gas correlation, S&P correlation (risk-on/off proxy)
  - New tabular features (per signal):
    - `primary_peer_corr` — correlation with the most-related peer asset (Gold↔Silver, S&P↔Nasdaq, etc.), [-1, 1] → [0, 1]
    - `cross_class_corr` — correlation with the strongest cross-class mover (e.g., Gold↔S&P when they diverge = risk-off signal), [-1, 1] → [0, 1]
    - `correlation_regime` — is the correlation structure normal (0.5), elevated (1.0), or broken/inverted (0.0)? Detected by comparing current 30-bar corr to 200-bar baseline
  - These features let the CNN see regime shifts: when Gold and S&P suddenly correlate strongly, that's a risk-off flight-to-safety regime. When they decorrelate, it's normal. When they invert, something is breaking.
  - Peer asset mapping defined in `asset_registry.py`: `Asset.peers` → `["Silver", "Copper"]` for Gold, etc.
  - Pure computation in `lib/analysis/cross_asset.py` — no side effects

### Phase 7C: Asset Fingerprint Analysis
- [ ] **Create `lib/analysis/asset_fingerprint.py`** — profile what makes each asset unique for the CNN
  - Per-asset fingerprint vector (computed daily, cached):
    - `typical_daily_range_atr` — how many ATR does this asset typically move in a day? (Gold ~1.2, Nasdaq ~1.8, 6E ~0.7)
    - `session_concentration` — what fraction of the daily range happens in London vs US vs overnight? (Gold: 40% London, S&P: 70% US)
    - `breakout_follow_through_rate` — historically, what % of breakouts on this asset continue vs fade? (per breakout type)
    - `mean_reversion_tendency` — does this asset tend to revert (choppy) or trend (momentum)? Rolling Hurst exponent, normalized [0, 1]
    - `volume_profile_shape` — is volume U-shaped (equity open/close), L-shaped (London open), or flat (crypto 24/7)?
    - `overnight_gap_tendency` — how often does this asset gap overnight, and do gaps fill or continue?
  - These are NOT tabular features directly — they're used to create the asset embedding training labels
  - The fingerprint analysis runs during off-hours and is persisted to Redis/Postgres
  - Dashboard: "Asset DNA" panel showing the fingerprint radar chart for each focused asset
  - **Key insight**: if we can quantify "what makes gold gold", we can detect when gold is acting like something else (regime anomaly) and flag it

### Phase 7D: Correlation Anomaly Detection
- [ ] **Detect when cross-asset correlations break from historical norms** — hidden state discovery
  - Maintain a rolling correlation matrix across all 10 core assets (updated every 5 min during active session)
  - Compare current 30-bar correlation matrix to the 200-bar baseline → compute anomaly score per pair
  - When a correlation pair deviates by >2σ from baseline, flag as "correlation break":
    - Gold↔S&P suddenly +0.8 (normally ~0.0) → "flight to safety" regime
    - Crude↔Nasdaq suddenly −0.6 (normally +0.3) → "energy divergence" regime
    - BTC↔MES suddenly +0.9 (normally +0.5) → "risk-on euphoria" regime
  - Publish anomalies to Redis `engine:correlation_anomalies` for dashboard display
  - Dashboard: correlation heatmap panel showing current vs baseline, with anomalous cells highlighted
  - Feed anomaly flags into CNN as additional context features at v8 retrain
  - This is where you find the "hidden states" — regime shifts that aren't visible from any single asset's price action alone

---

## 🟡 Medium Priority — Existing Tasks

### NT8 Validation
- [ ] **Test v6 ONNX auto-adapt** — deploy `BreakoutStrategy.cs` to NT8, compile, verify:
  - Startup log shows `CNN tabular dim: model expects 18, C# builds 18`
  - Per-type TP3 mults loaded from `feature_contract.json` (log each type's mult at startup)
  - Entry logs show `[positions: N/5]`
  - No `OCO ID cannot be reused` or `signal name longer than 50` errors
  - Run for a full session and review output logs
- [ ] **Parity-test Phase 3 EMA9 trailing** — run Python engine + C# strategy side-by-side on same OHLCV data, compare Phase 3 trail stop levels and exit prices. Target: ≤ 1 tick divergence per bar.
  - `test_phase3_ema9_parity.py` — 130 tests all green; warm-up sequences use trending bars

### NT8 Bridge Trading Tests
- [ ] **Bridge `/flatten` from web UI** — ensure the Flatten All button in the dashboard triggers Bridge `FlattenAll` which closes every position across all instruments immediately (already wired, needs live test)
  - Python-side wiring fully tested offline: `src/tests/test_bridge_trading.py` — TestFlattenAll (6 tests) covers heartbeat liveness gate, port resolution from heartbeat, proxy forwarding to Bridge `/flatten`, default reason, connection error handling (503/504)
  - Dashboard HTML wiring verified: `_render_positions_panel()` in `dashboard.py` renders `hx-post="/api/positions/flatten"` button with `hx-confirm`, disabled when Bridge offline
  - Live test script ready: `scripts/test_bridge_live.py` — run with `--bridge HOST:PORT --data URL` when NT8 is up on Sim (also `--local` for same-machine testing)
  - **Remaining**: fire up NT8 on Sim, run `python scripts/test_bridge_live.py`, verify flatten closes all positions across all instruments
- [ ] **Manual trade from dashboard** — when the strategy is always running and I place a manual entry from the web UI via `/execute_signal`, it should coexist with automated entries. Verify:
  - Manual entry gets its own `PositionPhase` tracking
  - Automated entries continue alongside manual positions
  - Both respect `MaxConcurrentPositions = 5`
  - Python-side wiring fully tested offline: `src/tests/test_bridge_trading.py` — TestExecuteSignal (6 tests) covers market long, short with all fields, risk check enforcement, risk check bypass (enforce_risk=False), direction validation, connection errors
  - C# wiring verified by code review: `Bridge.cs` `ListenLoop` parses `/execute_signal` POST → creates `SignalBus.Signal` → `SignalBus.Enqueue()` → `BreakoutStrategy.OnBarUpdate` calls `_engine.DrainSignalBus()` → `ProcessSignal()` → `SubmitOrderUnmanaged()` with unique signal name `Signal-{dir}-{id}` (capped at 49 chars)
  - `FireEntry()` registers each entry in `_positionPhases[signalId]` with its own `PositionPhase` struct — manual Bridge entries use `signalId = "brg-{guid}"`, automated entries use `signalId = "brk-{dir}-{timestamp}-{instrument}-{type}"`
  - `MaxConcurrentPositions` gate at top of `FireEntry()`: `if (_activePositionCount >= MaxConcurrentPositions) return;` — applies to both automated and manual entries
  - **Remaining**: live test with NT8 on Sim — run `scripts/test_bridge_live.py --local` (or `--bridge HOST:PORT --data URL`), verify both entry types get PositionPhase in Output Window

---

## 🟢 Low Priority

### Trade Copier (Future — Post First Funded Account)
- [ ] **Simple trade copier for multiple TPT accounts** — once the first $50k account is funded and profitable:
  - Mirror all fills from Account 1 → Accounts 2–5
  - Use Bridge AddOn's position push to detect fills on the primary account
  - Fire identical orders on secondary accounts via their own Bridge instances (or a shared copier service)
  - Respect per-account contract limits (each TPT tier has its own max)
  - Scale up to 5 accounts max

### Multi-Source Breakout Detection Enhancements
- [ ] **Wire `crypto_momentum_score` into engine scoring pipeline** — currently computed but not fed into live decisions
  - Add as optional boost to breakout signal quality scoring (engine-side, before CNN v7 adds it as a feature)
  - Dashboard: show crypto momentum indicator on focused asset cards when crypto data is available
  - Strongest value: Asian session crypto breakout → London/US equity open prediction

### Kraken Spot Portfolio Management (Phase 6)
- [ ] **Create `lib/strategies/crypto/portfolio_manager.py`** — maintain target % allocations for spot crypto holdings
  - Kraken private API already has `get_balance()`, `get_trade_balance()`, `get_open_orders()` — need to add `add_order()` and `cancel_order()` to `KrakenDataProvider`
  - `CryptoPortfolioConfig` dataclass:
    - Target allocations: `{"BTC": 0.50, "ETH": 0.30, "SOL": 0.10, "LINK": 0.05, "AVAX": 0.05}` (% of total crypto portfolio value)
    - Rebalance threshold: 5% deviation from target triggers rebalance consideration
    - Max trade size per rebalance: 10% of total portfolio (don't dump everything at once)
    - Rebalance cooldown: minimum 4 hours between rebalances
    - DCA mode: option to buy fixed USD amount on schedule (daily/weekly) into target allocations
  - `check_rebalance()` — compare current holdings to targets, return list of needed trades
  - `execute_rebalance()` — place limit orders on Kraken to bring allocations back to target
  - **No hard stop equivalent** — crypto runs 24/7, but:
    - Risk rules: max drawdown alert (if total crypto portfolio drops >10% from peak, alert + pause rebalancing)
    - Volatility filter: don't rebalance during extreme vol (BTC ATR > 2σ from 20-day mean)
    - Integration with futures strategy: when the futures system detects a strong crypto momentum signal (from `crypto_momentum.py`), optionally overweight that asset temporarily
  - Dashboard: Kraken portfolio card showing current allocations vs targets, P&L, rebalance status
  - Separate from futures — this is a "set it and forget it" spot portfolio that runs alongside the active trading
  - All Kraken trading gated behind `ENABLE_KRAKEN_TRADING=1` env var (separate from `ENABLE_KRAKEN_CRYPTO` which is read-only data)

---

## Completed

### Trainer UI Separation (`src/lib/training/trainer_server.py`, `src/lib/services/data/api/trainer.py`, `src/lib/services/web/main.py`)
- [x] `trainer_server.py` HTML endpoint (`trainer_ui`) removed — trainer is now pure API server
- [x] `src/lib/services/data/api/trainer.py` created — full HTML trainer dashboard page at `GET /trainer`, config endpoints, `/trainer/api/*` proxy
- [x] `src/lib/services/data/main.py` — `trainer_router` imported and registered; trainer paths added to `api_info`
- [x] `src/lib/services/web/main.py` — `/trainer` and `/trainer/*` now proxy to data service (not directly to trainer:8200); trainer httpx client removed; `TRAINER_SERVICE_URL` env var removed from web service env block

### Unified Data Resolver (`src/lib/services/data/resolver.py`)
- [x] `DataResolver` class — Redis → Postgres → Massive/Kraken API three-tier resolution with automatic backfill
- [x] `ResolveMetadata` dataclass — source, rows, cache_hit, backfilled_redis/postgres, duration_ms, error
- [x] `resolve()`, `resolve_with_meta()`, `resolve_batch()`, `resolve_batch_with_meta()` public methods
- [x] `get_resolver()` module-level singleton + `resolve()` shortcut
- [x] No import cycle with `dataset_generator.py` — symbol map inlined in resolver

### Kraken Training Pipeline Integration (`src/lib/training/dataset_generator.py`, `src/lib/analysis/breakout_cnn.py`, `models/feature_contract.json`)
- [x] `dataset_generator.py` — `_is_kraken_symbol()`, `_load_bars_from_kraken()`, `_SYMBOL_TO_TICKER` short aliases (BTC→KRAKEN:XBTUSD etc.), `load_bars()` Kraken routing
- [x] `trainer_server.py` — `DEFAULT_SYMBOLS` updated: 22 CME micros + BTC, ETH, SOL (25 total)
- [x] `breakout_cnn.py` — `ASSET_CLASS_ORDINALS` + `ASSET_VOLATILITY_CLASS` include all Kraken tickers
- [x] `models/feature_contract.json` — regenerated with 42-entry `asset_class_map` + `asset_volatility_classes` including all Kraken internal tickers and short aliases
- [x] CI/CD — `KRAKEN_API_KEY` + `KRAKEN_API_SECRET` both injected in pre-deploy step and passed through docker-compose to engine + trainer

### ONNX ↔ PyTorch Parity Check (`scripts/check_onnx_parity.py`)
- [x] `scripts/check_onnx_parity.py` created — loads `.pt` via `_build_model_from_checkpoint` and `.onnx` via `onnxruntime`, runs 64 synthetic v6 18-feature batches, asserts max abs diff < 1e-4
- [x] Feature contract validation: checks version, feature count (18), and optionally name order against `TABULAR_FEATURES`
- [x] `--verbose` flag prints per-batch min/max/diff; `--n-samples`, `--threshold`, `--device` (auto/cpu/cuda/mps) args
- [x] Exit 0 = pass (safe to deploy to NT8), Exit 1 = fail (re-export ONNX)

### NT8 TPT Hard Stop — 4:00 PM ET session close (`src/ninja/BreakoutStrategy.cs`)
- [x] `CheckTptHardStop()` method added in new `#region TPT hard stop` block
- [x] Converts bar time to ET via `TimeZoneInfo.ConvertTimeFromUtc` with `"Eastern Standard Time"` zone
- [x] 16:00–17:59 ET: sets `RiskBlocked=true` + `RiskBlockReason="TPT_SESSION_CLOSED"`, calls `_engine.FlattenAll("TPT_HARD_STOP_16:00")` if `_activePositionCount > 0`; retries on next bar if FlattenAll throws
- [x] 18:00+ ET: clears `TPT_SESSION_CLOSED` block so new Globex session trading is allowed
- [x] Called from BIP0 path in `OnBarUpdate()` when `TptMode == true`; wrapped in try/catch for crash resilience

### Web UI — Trading / Review Mode + Dashboard Cleanup (`src/lib/services/data/api/dashboard.py`)
- [x] `⚡ Trading` / `🔍 Review` pill toggle added to dashboard header; persisted in `localStorage['dashMode']`; auto-detects from ET hour on first visit
- [x] CSS: `body.mode-trading .review-only { display: none }` / `body.mode-review .trading-only { display: none }` — zero JS overhead, pure CSS visibility
- [x] Review-only panels: Dataset Preview, Crypto Chart, Correlation, Volume Profile, Performance, Trade Journal, Market Regime
- [x] Grok `hx-trigger` changed from `every 60s` → `load`; Review Mode restores polling via `setDashboardMode()` JS
- [x] Static "Next Session / Schedule" panel removed from sidebar
- [x] `_price_decimals(tick_size)` helper + `tick_size` param to `_compute_entry_zone()` — forex (6E, 6B, 6J) now correctly uses 5–7 decimal places
- [x] `target1_dollars` / `target2_dollars` computed in `compute_asset_focus()` and displayed as `~$N` badges on TP1/TP2 in asset cards; stop shows `-$risk`

### CNN Model — Full Retrain (v6, 87.1% accuracy)
- [x] 22-symbol training: MGC, SIL, MHG, MCL, MNG, MES, MNQ, M2K, MYM, 6E, 6B, 6J, 6A, 6C, 6S, ZN, ZB, ZC, ZS, ZW, MBT, MET
- [x] All 13 breakout types, all 9 sessions, 90 days lookback, 25 epochs
- [x] 85/15 stratified train/val split
- [x] Validation gates: 87.1% acc (≥80%), 87.15% prec (≥75%), 87.27% rec (≥70%) — all passed
- [x] Champion promoted: `breakout_cnn_best.pt` + `breakout_cnn_best.onnx` (80.7 MB)
- [x] `feature_contract.json` v6 regenerated with all 13 type configs
- [x] `breakout_cnn_best_meta.json` written with full training config and metrics
- [x] 41 checkpoint `.pt` files saved during training (timestamped with accuracy)
- [x] ONNX export: opset 17, dynamic batch axes, validated with `onnx.checker`

### NT8 — Stop-and-Reverse (SAR) always-in micro position (`src/ninja/BreakoutStrategy.cs`)
- [x] `ReversalState` sealed class — direction, signalId, entryPrice, ATR, SL, lastReversalTime, reversalCount; `RMultiple(price)` + `IsWinning(price)` helpers; `Open()` / `Close()` lifecycle methods
- [x] `_sarStates[]` allocated at DataLoaded alongside `_states[]`; SAR constants mirror Python PM env vars: `CSarMinCnnProb=0.85`, `CSarWinningCnnProb=0.92`, `CSarHighWinnerCnnProb=0.95`, `CSarMinMtfScore=0.60`, `CSarCooldownMinutes=30`, `CSarChaseMaxAtrFraction=0.50`, `CSarChaseMinCnnProb=0.90`
- [x] `ShouldReverse()` — 5 gates matching Python `_should_reverse`: direction, CNN prob, cooldown, MTF, high-winner protection
- [x] `DecideEntryType()` — limit-at-range-edge / market-chase logic matching Python `_decide_entry_type`
- [x] `TryReversePosition()` — flatten → clean phase tracking → reset fired flags → pre-decrement active count → FireEntry → update cooldown
- [x] `CheckBreakout()` SAR path: when `rs.FiredLong`/`rs.FiredShort` already set, check `sar.IsShort`/`sar.IsLong` and evaluate reversal gates; fresh entries still use original path
- [x] `PassesCnnFilter()` overload with `out double cnnProbOut` for SAR reversal gate evaluation
- [x] `FireEntry()` → `sarRef.Open()` stamps direction + signalId on fill; `OnOrderUpdate` closes SAR state on flatten/SL/TP; per-bar `Positions[bip]` sync as belt-and-suspenders

### NT8 — MTF (15-minute) EMA/MACD alignment scoring (`src/ninja/BreakoutStrategy.cs`)
- [x] **`InstrumentState` MTF fields** — EMA-9/21/50 incremental state, MACD-12/26/9 incremental state, histogram ring-buffer (3 bars) for slope, EMA-50 5-bar ring-buffer for slope, `MtfScore` sentinel (-1 = ready, 1.0 = warm-up pass-through), `MtfBip` back-reference to the 15m BIP index
- [x] **15m `AddDataSeries`** — one 15m series added per tracked instrument in `Configure` immediately after each 1m series; primary instrument (BIP0) gets its 15m series separately; all use the same trading-hours template as the 1m series
- [x] **`_mtfBipBySymbol` map** — built at `DataLoaded` by scanning `BarsArray` for `BarsPeriod.Value == 15`; wires `st.MtfBip` on every matching `InstrumentState`
- [x] **`UpdateMtf(int mtfBip, InstrumentState st)`** — called from `OnBarUpdate` whenever a 15m BIP fires a new closed bar; incremental EMA-9/21/50, MACD-12/26/9, histogram ring-buffer; writes `-1` sentinel to `MtfScore` once both EMA-50 (≥50 bars) and MACD signal (≥35 bars) are warmed up; `1.0` pass-through during warm-up

### ONNX Parity Check (`scripts/check_onnx_parity.py`)
- [x] Validated: 64 synthetic 18-feature batches → max abs diff < 1e-4 between .pt and .onnx

### Kraken — Full Data Integration for Training
- [x] Kraken API key/secret via CI/CD
- [x] Kraken data in training pipeline — `dataset_generator.py` fully wired for Kraken OHLCV
- [x] Unified data resolver for training — `src/lib/services/data/resolver.py`

### Multi-Source Breakout Detection (Futures + Crypto)
- [x] `src/lib/analysis/crypto_momentum.py` — full crypto momentum scorer module
  - `CryptoMomentumScorer`, `compute_single_crypto_momentum()`, `score_futures_from_crypto()`
  - Session-aware scoring, rolling Pearson correlation, weighted composite
  - `CryptoMomentumSignal.to_tabular_feature()` returns normalized [-1, +1] for v7 feature
- [x] `src/tests/test_crypto_momentum.py` — 109 tests all green
- [x] Generalized model across asset classes — CNN trained on 25 symbols across 5 asset classes

### Web UI — Settings Page
- [x] `src/lib/services/data/api/settings.py` — 5 tabbed sections: Engine, Services, Features, Risk & Trading, API Keys
- [x] All settings persisted to Redis via `settings:overrides` key
- [x] `src/lib/services/web/main.py` — 9 new proxy routes for settings endpoints

### Web UI — Trainer Separation & New Pages
- [x] Trainer UI extracted into its own data-service page
- [x] Settings page with full 5-tab configuration interface

---

## Execution Order

The refactor phases are ordered by dependency and risk:

**Immediate (safe renames, no logic changes):**
1. **Phase 1E** — Rename `orb_filters.py` → `breakout_filters.py`
2. **Phase 1F** — Rename `orb_simulator.py` → `rb_simulator.py`

**RB System Merge (core refactor, sequential):**
3. **Phase 1D** — Extract generic handler pipeline from `main.py` (biggest immediate LOC reduction, ~400 lines eliminated)
4. **Phase 1A** — Merge BreakoutType enums (foundational for everything else)
5. **Phase 1B** — Merge RangeConfig dataclasses (depends on 1A)
6. **Phase 1C** — Merge ORB detection into unified RB detector (depends on 1A + 1B)
7. **Phase 1G** — Create `lib/strategies/` package (depends on 1C, moves files into new structure)

**Daily Strategy + Focus (can start in parallel with RB merge):**
8. **Phase 2A** — Daily bias analyzer (independent of Phase 1)
9. **Phase 5A** — Generalized asset model / asset registry (independent, enables 5C)
10. **Phase 2B** — Daily plan generator (depends on 2A)
11. **Phase 2C** — Swing detector (depends on 2B)
12. **Phase 3A** — Top-4 asset selection (depends on 2B for swing vs scalp split)

**Live Risk & Dashboard (depends on asset registry + focus selection):**
13. **Phase 5B** — Real-time risk budget integration (depends on 5A)
14. **Phase 5C** — Dynamic position sizing on focus cards with micro/regular dual display (depends on 5A + 5B)
15. **Phase 5D** — Live position overlay on focus cards (depends on 5C)
16. **Phase 5E** — Risk dashboard strip (depends on 5B)
17. **Phase 3B** — Dashboard focus mode (depends on 3A + 5C + 5D)
18. **Phase 3C** — Grok integration for daily selection (depends on 2B)

**CNN Intelligence (depends on daily strategy layer being complete):**
19. **Phase 4A** — CNN v7 features from daily strategy layer (depends on 2A)
20. **Phase 4B** — Sub-features and richer encoding (depends on 4A)
21. **Phase 4C** — Retrain on v7 contract (depends on 4A + 4B)
22. **Phase 7A** — Hierarchical asset embedding (depends on 5A for asset registry)
23. **Phase 7B** — Cross-asset correlation features (depends on 7A)
24. **Phase 7C** — Asset fingerprint analysis (independent, can start with 7A)
25. **Phase 7D** — Correlation anomaly detection (depends on 7B + 7C)

**Low Priority / When Profitable:**
26. **Phase 6** — Kraken spot portfolio management (independent, needs `add_order` in Kraken client)
27. **Trade Copier** — post first funded account

Phases 1E/1F, 2A, 5A, and 7C can start immediately and in parallel. Phase 1D is the highest-value single change. Phase 5B-5E is the highest-value *user experience* change — making live trading feel real-time and risk-aware.

---

## 🗺️ System Logic Map — End-to-End Data & Signal Flow

> **Purpose**: Reference map of how data enters the system, flows through
> analysis / risk / breakout detection / CNN inference / position management,
> and ultimately reaches the NinjaTrader execution layer. Use this to
> research each subsystem in isolation.

---

### 1. Data Ingestion

```
External Sources
  ├─ Yahoo Finance (yfinance)  ← primary for CME futures (1m, 5m, 15m, daily)
  ├─ Kraken REST / WebSocket   ← crypto spot (BTC, ETH, SOL, etc.) via kraken_client.py
  └─ MassiveAPI (massive_client.py) ← alternative / historical bars

         │
         ▼

  lib/core/cache.py  →  get_data(ticker, interval, period)
         │                 Fetches bars, caches in Redis as JSON
         │                 Keys: engine:bars_1m:<TICKER>
         │                        engine:bars_15m:<TICKER>
         │                        engine:bars_daily:<TICKER>
         ▼

  lib/trading/engine.py  →  DashboardEngine
         │  _fetch_tf_safe()  — safe wrapper around get_data with retry
         │  _refresh_data()   — periodic bar refresh into Redis cache
         │  _loop()           — main engine refresh cycle
         ▼

  Redis (pub/sub + key-value)
         │  Central message bus for all services
         │  Bars, focus, signals, risk state, position state
         └─ engine:daily_focus, engine:risk:*, engine:positions:*
```

**Key files to research:**
- `src/lib/core/cache.py` — data fetch & Redis caching
- `src/lib/integrations/kraken_client.py` — Kraken OHLCV + WebSocket
- `src/lib/integrations/massive_client.py` — MassiveAPI client
- `src/lib/trading/engine.py` → `DashboardEngine._refresh_data()` — refresh loop
- `src/lib/core/models.py` — `ASSETS`, `CORE_WATCHLIST`, `ACTIVE_WATCHLIST`, `MICRO_CONTRACT_SPECS`, ticker mappings

---

### 2. Engine Startup & Scheduler

```
src/lib/services/engine/main.py  →  main()
  │
  ├─ Reads env: ACCOUNT_SIZE, ENGINE_INTERVAL, ENGINE_PERIOD
  ├─ Creates DashboardEngine via get_engine()
  ├─ Creates ScheduleManager (session-aware action scheduler)
  ├─ Initialises RiskManager (risk rules engine)
  ├─ Initialises PositionManager (micro stop-and-reverse positions)
  ├─ Starts ModelWatcher (filesystem watcher for CNN hot-reload)
  │
  └─ Main loop:
       while not shutdown:
         ├─ scheduler.get_pending_actions()   ← time-of-day aware
         ├─ _check_redis_commands()           ← dashboard-triggered overrides
         ├─ Execute each pending action via action_handlers dispatch table
         ├─ _handle_update_positions()        ← bracket / trailing stop updates
         ├─ _publish_engine_status()          ← push state to Redis for web UI
         └─ time.sleep(scheduler.sleep_interval)

Session Modes (Eastern Time):
  EVENING     18:00–00:00  →  CME, Sydney, Tokyo, Shanghai ORB sessions
  PRE_MARKET  00:00–03:00  →  Daily focus computation, Grok morning brief
  ACTIVE      03:00–12:00  →  Frankfurt, London, London-NY, US ORB + all breakout types
  OFF_HOURS   12:00–18:00  →  Backfill, optimization, CNN training, daily report
```

**Key files to research:**
- `src/lib/services/engine/main.py` → `main()` — the god loop & action dispatch
- `src/lib/services/engine/scheduler.py` → `ScheduleManager`, `ActionType`, `SessionMode`

---

### 3. Daily Focus Computation

```
ActionType.COMPUTE_DAILY_FOCUS  (runs once, pre-market 00:00–03:00 ET)
  │
  ▼
focus.py → compute_daily_focus(account_size, symbols)
  │
  │  For each asset in ASSETS:
  │    ├─ get_data(ticker, "5m", "5d")          ← 5-min bars, 5 days
  │    ├─ wave_analysis.calculate_wave_analysis() ← wave ratio, bias, dominance
  │    ├─ volatility.kmeans_volatility_clusters() ← ATR percentile, vol cluster
  │    ├─ signal_quality.compute_signal_quality() ← composite quality score (0–100%)
  │    ├─ _derive_bias() → LONG / SHORT / NEUTRAL
  │    ├─ _compute_entry_zone() → entry_low, entry_high, stop, tp1, tp2
  │    └─ _compute_position_size() → contracts, risk_dollars
  │
  │  Sort by quality (best first), then wave_ratio
  │  should_not_trade() check (all assets skip → no-trade day)
  │
  ▼
publish_focus_to_redis()
  │  Writes JSON to engine:daily_focus
  │  Contains: per-asset bias, levels, position sizes, quality scores
  └─ Web UI reads this for dashboard focus cards
```

**Key files to research:**
- `src/lib/services/engine/focus.py` — `compute_asset_focus()`, `compute_daily_focus()`
- `src/lib/analysis/wave_analysis.py` — wave ratio & trend detection
- `src/lib/analysis/volatility.py` — K-means ATR clustering
- `src/lib/analysis/signal_quality.py` — composite quality scorer

---

### 4. Breakout Detection System

The system detects **13 breakout types** across **10 global sessions**.

#### 4A. Opening Range Breakout (ORB) — Intraday Core

```
ActionType.CHECK_ORB_*  (every 2 min within each session's scan window)
  │
  ▼
main.py → _handle_check_orb(engine, orb_session)
  │
  │  For each asset in engine:daily_focus:
  │    ├─ Fetch 1m bars from Redis cache (engine:bars_1m:<TICKER>)
  │    │
  │    ├─ orb.py → detect_opening_range_breakout(bars_1m, symbol, session)
  │    │    ├─ compute_opening_range()    ← H/L of first N minutes of session
  │    │    ├─ _check_or_size()           ← range vs ATR quality gate
  │    │    ├─ _check_breakout_bar_quality() ← body ratio, volume, wick
  │    │    └─ Returns ORBResult (breakout_detected, direction, trigger, etc.)
  │    │
  │    ├─ _persist_orb_event()  ← audit trail to Postgres/SQLite
  │    │
  │    │  IF breakout_detected:
  │    │    │
  │    │    ▼
  │    ├─ orb_filters.py → apply_all_filters()  ← Quality Filter Gate
  │    │    ├─ check_nr7()              ← NR7 (narrowest range of 7 days) flag
  │    │    ├─ check_premarket_range()  ← premarket range vs OR size
  │    │    ├─ check_session_window()   ← time-of-day allowed window
  │    │    ├─ check_lunch_filter()     ← avoid 11:30–13:00 ET chop
  │    │    ├─ check_multi_tf_bias()    ← 15m EMA alignment with direction
  │    │    ├─ check_mtf_analyzer()     ← MACD slope + divergence on HTF
  │    │    └─ check_vwap_confluence()  ← price vs session VWAP alignment
  │    │    Gate mode: "majority" (>50% pass) or "all" (every filter passes)
  │    │
  │    │  IF filter_passed:
  │    │    │
  │    │    ▼
  │    ├─ CNN Inference (see §6 below)
  │    │    breakout_cnn.py → predict_breakout(image, tabular_18, session_key)
  │    │    Uses per-session probability threshold from feature_contract.json
  │    │
  │    │  IF cnn_signal (or CNN disabled):
  │    │    │
  │    │    ▼
  │    ├─ publish_orb_alert()           ← Redis pub/sub → web UI alert
  │    ├─ _dispatch_to_position_manager() ← PositionManager.process_signal()
  │    └─ alerts.send_signal()          ← push notification / email
  │
  ▼
10 ORB Sessions (all follow same pipeline):
  CME Open          18:00–20:00 ET
  Sydney/ASX Open   18:30–20:30 ET
  Tokyo/TSE Open    19:00–21:00 ET
  Shanghai/HK Open  21:00–23:00 ET
  Frankfurt/Xetra   03:00–04:30 ET
  London Open       03:00–05:00 ET
  London–NY Cross   08:00–10:00 ET
  US Equity Open    09:30–11:00 ET  (primary session)
  CME Settlement    14:00–15:30 ET
  Crypto UTC0/UTC12 (Kraken-only sessions)
```

#### 4B. Range Breakout Types (PDR, IB, Consolidation, + 9 More)

```
ActionType.CHECK_PDR / CHECK_IB / CHECK_CONSOLIDATION / CHECK_BREAKOUT_MULTI
  │
  ▼
main.py → _handle_check_pdr() / _handle_check_ib() / _handle_check_consolidation()
  │        _handle_check_breakout_multi()  ← runs multiple types in one sweep
  │
  │  For each asset in session's asset list:
  │    ├─ _fetch_bars_1m()  ← Redis cache or engine fallback
  │    ├─ (PDR) Fetch daily bars for prev_day_high / prev_day_low
  │    │
  │    ├─ breakout.py → detect_range_breakout(bars, symbol, config)
  │    │    ├─ _compute_atr()               ← 14-bar ATR for thresholds
  │    │    ├─ _build_*_range()             ← range builder per type:
  │    │    │    _build_orb_range()          (ORB)
  │    │    │    _build_pdr_range()          (Previous Day)
  │    │    │    _build_ib_range()           (Initial Balance, 60 min RTH)
  │    │    │    _build_consolidation_range() (BB squeeze contraction)
  │    │    │    _build_weekly_range()       (Prior week H/L)
  │    │    │    _build_monthly_range()      (Prior month H/L)
  │    │    │    _build_asian_range()        (19:00–02:00 ET H/L)
  │    │    │    _build_bbsqueeze_range()    (BB inside Keltner Channel)
  │    │    │    _build_va_range()           (Value Area VAH/VAL)
  │    │    │    _build_inside_day_range()   (Today inside yesterday)
  │    │    │    _build_gap_rejection_range() (Overnight gap fill/reject)
  │    │    │    _build_pivot_range()        (Floor pivot R1/S1)
  │    │    │    _build_fibonacci_range()    (38.2–61.8% retracement)
  │    │    │
  │    │    ├─ _scan_for_breakout()         ← close beyond range ± ATR depth
  │    │    └─ Returns BreakoutResult
  │    │
  │    ├─ _run_mtf_on_result()  ← enrich with MTF score, MACD slope, divergence
  │    ├─ _persist_breakout_result()  ← audit trail
  │    │
  │    │  IF breakout_detected:
  │    ├─ _publish_breakout_result()         ← Redis pub/sub
  │    ├─ _dispatch_to_position_manager()    ← stop-and-reverse
  │    └─ alerts.send_signal()               ← notification
  │
  ▼
RangeConfig (per-type defaults in breakout.py):
  Each BreakoutType has its own:
    atr_period, atr_multiplier, min_depth_atr_pct, min_body_ratio,
    max_range_atr_ratio, min_range_atr_ratio, plus type-specific params
    (e.g. ib_duration_minutes=60, asian_start_time=19:00, fib_upper=0.618)
```

**Key files to research:**
- `src/lib/services/engine/orb.py` — ORB detection, session definitions, `detect_opening_range_breakout()`
- `src/lib/services/engine/breakout.py` — `BreakoutType` enum, `RangeConfig`, `detect_range_breakout()`, all `_build_*_range()` functions
- `src/lib/analysis/orb_filters.py` — quality filter gate: NR7, premarket, session window, lunch, MTF bias, VWAP
- `src/lib/analysis/mtf_analyzer.py` — multi-timeframe EMA/MACD scoring
- `src/lib/core/breakout_types.py` — canonical IntEnum for CNN training ordinals

---

### 5. Risk Management

```
RiskManager  (src/lib/services/engine/risk.py)
  │
  │  Initialised at engine startup with:
  │    account_size, risk_pct_per_trade (1%), max_daily_loss,
  │    max_open_trades, no_entry_after (cutoff time), session_end
  │
  │  can_enter_trade(symbol, side, size, risk_per_contract, ...)
  │    ├─ Rule 1: Daily P&L ≤ max_daily_loss  → BLOCKED
  │    ├─ Rule 2: Open positions ≥ max_open_trades  → BLOCKED
  │    ├─ Rule 3: Per-trade risk > account × risk_pct  → BLOCKED
  │    ├─ Rule 4: Past no_entry_after cutoff time  → BLOCKED
  │    ├─ Rule 5: Session has ended  → BLOCKED
  │    ├─ Rule 6: Stacking rules (min R-multiple, min wave ratio)  → BLOCKED
  │    └─ Rule 7: 3 consecutive losses circuit breaker  → BLOCKED
  │    Returns: (allowed: bool, reason: str)
  │
  │  register_open(symbol, side, size, entry_price, ...)
  │  register_close(symbol, exit_price, pnl, ...)
  │  update_unrealized(pnl)
  │  sync_positions(positions_dict)
  │
  │  publish_to_redis()  → engine:risk:status
  │    Exposes: daily_pnl, open_positions, consecutive_losses,
  │             open_trade_count, risk budget remaining
  │
  ▼
_handle_check_risk_rules()  (main.py, runs every loop iteration)
  │  Checks all risk rules, publishes risk state
  │  If daily loss hit → sets no-trade flag, sends alert
  │
_handle_check_no_trade()
  │  Evaluates should_not_trade() from focus data
  │  If all assets have quality < threshold → no-trade day
```

**Key files to research:**
- `src/lib/services/engine/risk.py` — `RiskManager`, all 7 risk rules, P&L tracking
- `src/lib/services/engine/main.py` → `_handle_check_risk_rules()`, `_handle_check_no_trade()`

---

### 6. CNN Model — Inference (Live)

```
Breakout detected + filters passed
  │
  ▼
breakout_cnn.py → predict_breakout(image_path, tabular_18, session_key)
  │
  ├─ _load_model()  ← loads breakout_cnn_best.pt (cached, hot-reloaded by ModelWatcher)
  │    Model: HybridBreakoutCNN (EfficientNetV2-S backbone + tabular branch)
  │
  ├─ Image branch:
  │    chart_renderer_parity.py renders a 224×224 Ruby-style chart snapshot
  │    showing the breakout bar, range box, VWAP, EMA lines
  │    → get_inference_transform() → ImageNet normalisation → (1, 3, 224, 224) tensor
  │
  ├─ Tabular branch (18 features, v6 contract):
  │    _normalise_tabular_for_inference(features)
  │    ┌─────────────────────────────────────────────────────────┐
  │    │  [0]  quality_pct_norm      quality_pct / 100           │
  │    │  [1]  volume_ratio          breakout bar vol / 20-avg   │
  │    │  [2]  atr_pct               ATR as fraction of price    │
  │    │  [3]  cvd_delta             normalised CVD delta [-1,1] │
  │    │  [4]  nr7_flag              1.0 if NR7 day              │
  │    │  [5]  direction_flag        1.0=LONG, 0.0=SHORT         │
  │    │  [6]  session_ordinal       Globex day position [0,1]   │
  │    │  [7]  london_overlap_flag   1.0 if 08:00–09:00 ET      │
  │    │  [8]  or_range_atr_ratio    OR range / ATR              │
  │    │  [9]  premarket_range_ratio premarket range / OR range  │
  │    │  [10] bar_of_day            minutes since open / 1380   │
  │    │  [11] day_of_week           Mon=0..Fri=4 / 4            │
  │    │  [12] vwap_distance         (price-VWAP) / ATR          │
  │    │  [13] asset_class_id        asset class ordinal / 4     │
  │    │  [14] breakout_type_ord     BreakoutType ordinal / 12   │
  │    │  [15] asset_volatility_class low=0 / med=0.5 / high=1   │
  │    │  [16] hour_of_day           ET hour / 23                │
  │    │  [17] tp3_atr_mult_norm     TP3 ATR mult / 5.0          │
  │    └─────────────────────────────────────────────────────────┘
  │    → (1, 18) float tensor
  │
  ├─ Forward pass:
  │    img_features = EfficientNetV2-S(image)           → (1, 1280)
  │    tab_features = tabular_head(tabular)             → (1, 32)
  │    combined     = cat(img_features, tab_features)   → (1, 1312)
  │    logits       = classifier(combined)              → (1, 2)
  │    prob_good    = softmax(logits)[0, 1]             → P(clean breakout)
  │
  ├─ Per-session thresholds (from feature_contract.json):
  │    us: 0.82  london: 0.82  london_ny: 0.82  frankfurt: 0.80
  │    cme_settle: 0.78  cme: 0.75  tokyo: 0.74  shanghai: 0.74  sydney: 0.72
  │
  └─ Returns: { prob, signal, confidence ("high"/"medium"/"low"), threshold }
       signal = True if prob_good ≥ session threshold

NT8 Side (C#):
  BreakoutStrategy.cs loads breakout_cnn_best.onnx via OnnxRuntime
  OrbCnnPredictor inlines the same 18-feature normalisation (PrepareCnnTabular)
  OrbChartRenderer renders a matching 224×224 chart bitmap
  CnnSessionThresholds mirrors the same per-session threshold table
  → Same model, same features, same thresholds — Python trains, C# infers
```

**Key files to research:**
- `src/lib/analysis/breakout_cnn.py` → `predict_breakout()`, `HybridBreakoutCNN`, `_normalise_tabular_for_inference()`
- `src/lib/analysis/chart_renderer.py` / `chart_renderer_parity.py` — chart image rendering
- `models/feature_contract.json` — the v6 contract (18 features, thresholds, ordinals)
- `src/ninja/BreakoutStrategy.cs` — NT8 C# side: `OrbCnnPredictor`, `OrbChartRenderer`, `CnnSessionThresholds`

---

### 7. CNN Model — Training Pipeline

```
ActionType.TRAIN_BREAKOUT_CNN  (off-hours, or triggered from trainer web UI)
  │
  ▼
trainer_server.py → _run_training_pipeline(TrainRequest)
  │
  │  ─── Step 1: Dataset Generation ───
  │    dataset_generator.py → generate_dataset(symbols, days_back, config)
  │      │
  │      │  For each symbol:
  │      │    ├─ load_bars()  ← multi-source resolver:
  │      │    │    engine cache → Postgres DB → CSV files → MassiveAPI → Kraken
  │      │    │
  │      │    ├─ generate_dataset_for_symbol(symbol, bars_1m, bars_daily, config)
  │      │    │    │
  │      │    │    ├─ _run_simulators_for_breakout_type()
  │      │    │    │    For each breakout type (ORB/PDR/IB/CONS/all 13):
  │      │    │    │      orb_simulator.py runs historical simulation
  │      │    │    │      walks forward through bars, detects ranges,
  │      │    │    │      classifies outcome as "good" (clean follow-through)
  │      │    │    │      or "bad" (fail / chop) using TP/SL bracket replay
  │      │    │    │    For ORB with session="all": simulates all 9 Globex sessions
  │      │    │    │
  │      │    │    ├─ For each simulated result:
  │      │    │    │    chart_renderer_parity.py → render 224×224 PNG snapshot
  │      │    │    │    _build_row() → CSV row with:
  │      │    │    │      image_path, label (good/bad), 18 tabular features,
  │      │    │    │      breakout_type, session, symbol metadata
  │      │    │    │
  │      │    │    └─ Caps: max_samples_per_label, per_type_label, per_session_label
  │      │    │
  │      │    └─ Writes: <output_dir>/labels.csv + <output_dir>/images/*.png
  │      │
  │      └─ Returns DatasetStats (total_images, label_distribution, duration)
  │
  │  ─── Step 1b: Train/Val Split ───
  │    split_dataset(labels.csv, val_fraction=0.15, stratify=True)
  │    → train.csv (85%) + val.csv (15%), stratified by label
  │
  │  ─── Step 2: Model Training ───
  │    breakout_cnn.py → train_model(train.csv, val.csv, epochs, batch_size, lr)
  │      │
  │      │  BreakoutDataset (PyTorch Dataset):
  │      │    __getitem__: load image → transform, parse 18 tabular features,
  │      │    read label → (image_tensor, tabular_tensor, label)
  │      │
  │      │  HybridBreakoutCNN:
  │      │    EfficientNetV2-S (ImageNet pre-trained) + tabular MLP + classifier
  │      │
  │      │  Two-phase training:
  │      │    Phase 1 (freeze_epochs=2): CNN backbone frozen, train tabular head + classifier
  │      │    Phase 2 (remaining epochs): unfreeze backbone, fine-tune everything at lower LR
  │      │
  │      │  Optimizer: AdamW (lr=3e-4, weight_decay=1e-5)
  │      │  Scheduler: CosineAnnealingLR
  │      │  Loss
: CrossEntropyLoss (label_smoothing=0.05)
  │      │  Saves checkpoint every epoch: breakout_cnn_<timestamp>_acc<N>.pt
  │      │
  │      └─ Returns TrainResult (model_path, best_epoch, epochs_trained)
  │
  │  ─── Step 3: Evaluation ───
  │    breakout_cnn.py → evaluate_model(candidate.pt, val.csv)
  │    → val_accuracy, val_precision, val_recall
  │
  │  ─── Step 4: Promotion Gates ───
  │    ├─ accuracy  ≥ min_acc (default ~80%)
  │    ├─ precision ≥ min_prec
  │    └─ recall    ≥ min_rec
  │    All