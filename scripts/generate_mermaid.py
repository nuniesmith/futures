from pathlib import Path

import requests  # type: ignore[import-untyped]

# =============================================================================
# FUTURES TRADING SYSTEM — FULL LOGIC FLOW (v6)
# Accurately reflects the real project structure under src/
#
# Run: python scripts/generate_mermaid.py
# Outputs:
#   - docs/futures_logic_flow.mmd     (open in mermaid.live or VSCode)
#   - docs/futures_logic_flow.png     (rendered image)
# =============================================================================

MERMAID = """flowchart TD
    %% ==================== EXTERNAL DATA SOURCES ====================
    subgraph External["🌐 External Data Sources"]
        A1["MassiveAPI\nCME futures OHLCV\n(1m/5m/15m/daily)"]
        A2["Kraken REST + WebSocket\nCrypto spot pairs 24/7\n(KrakenDataProvider + KrakenFeedManager)"]
        A3["Reddit via PRAW + VADER\n(RedditWatcher + reddit_sentiment)"]
        A4["News: Finnhub + Alpha Vantage\n(news_client + news_sentiment)"]
        A5["Rithmic\nLive account data + positions\n(RithmicAccountManager)"]
    end

    %% ==================== DATA LAYER ====================
    subgraph Data["📥 Data Service  [Docker: data]"]
        B["DataResolver\nRedis hot → Postgres durable → External fallback\n(services/data/resolver.py)"]
        B -->|"bars_1m / bars_15m / daily"| C["Redis :6379\nBar caches, focus assets,\ndaily plan, risk state,\nmodel events"]
        B -->|"audit, trades, journal,\nbars, logs"| D["Postgres :5432\nDurable store"]
        E["FastAPI data service\n(services/data/main.py)\nEmbedded engine proxy\nor standalone REST + SSE"]
        E --> B
        F["Kraken WebSocket feed\nstart_kraken_feed()\nPushes bars → Redis on tick"]
        F --> C
        G["Reddit aggregation job\n5-min polling loop\nget_full_snapshot() → Redis"]
        G --> C
    end

    %% ==================== ASSET REGISTRY ====================
    subgraph Registry["📋 Asset Registry  (core/asset_registry.py)"]
        H["AssetRegistry\nFutures: Metals, Energy,\nEquity Index, FX, Treasuries,\nAgriculture\nCrypto: BTC, ETH, SOL…\nMicro + Full + Spot variants"]
    end

    %% ==================== SCHEDULER & SESSION MODES ====================
    subgraph Scheduler["⏰ ScheduleManager  (engine/scheduler.py)"]
        I1["EVENING  18:00–00:00 ET\nCME Globex, Sydney, Tokyo,\nShanghai ORB sessions"]
        I2["PRE-MARKET  00:00–03:00 ET\nCompute daily focus\nGrok morning brief\nNews sentiment (07:00)"]
        I3["ACTIVE  03:00–12:00 ET\nLondon, Frankfurt, LN-NY,\nUS Open ORB + multi-type\nbreakout scans every 2 min"]
        I4["OFF-HOURS  12:00–18:00 ET\nBackfill, CNN training,\nbacktest, optimisation,\ndaily report, EOD close"]
    end

    %% ==================== PRE-MARKET ROUTINE ====================
    subgraph PreMarket["🌅 Pre-Market Pipeline  (00:00–03:00 ET)"]
        J["COMPUTE_DAILY_FOCUS\nPreMarketScorer: NATR, RVOL,\ngap, momentum, catalyst scores\n(analysis/scorer.py)"]
        J --> K["DailyBiasAnalyzer + DailyPlanGenerator\nbias_analyzer.py + daily_plan.py\nSwingDetector candidates"]
        K --> L["compute_daily_focus()\nConvictionStack multipliers:\nnews sentiment, crypto momentum,\nGrok brief\n(engine/focus.py)"]
        L --> M["publish_focus_to_redis()\nengine:focus_assets\nengine:daily_plan → Redis"]
        M --> N["GROK_MORNING_BRIEF\ngrok_helper.py\nMacro context + daily plan\n→ dashboard chat cards"]
        N --> O["CHECK_NEWS_SENTIMENT  07:00 ET\nFinnhub + Alpha Vantage + VADER\ncached engine:news_sentiment:<SYM>  2h TTL"]
    end

    %% ==================== BREAKOUT DETECTION CORE ====================
    subgraph BreakoutCore["🔍 Breakout Detection  (trading/strategies/rb/)"]
        P["13 BreakoutTypes\nORB · PrevDay · InitialBalance\nConsolidation · Weekly · Monthly\nAsian · BollingerSqueeze · ValueArea\nInsideDay · GapRejection · PivotPoints · Fibonacci"]
        P --> Q["detect_range_breakout()\nrange_builders.py + breakout.py\nRangeConfig per type"]
        Q --> R["apply_all_filters()\nNR7, pre-market range,\nsession window, lunch filter,\nVWAP confluence, MTF bias\n(analysis/breakout_filters.py)"]
        R --> S["build_cnn_tabular_features()\n25+ features: ATR trend, RVOL,\ndaily bias, session overlap,\ncrypto momentum, news score\n(engine/handlers.py)"]
        S --> T["HybridBreakoutCNN inference\nPyTorch tabular + type/asset\nembeddings → LONG/SHORT/SKIP\n(analysis/breakout_cnn.py)"]
    end

    %% ==================== ORB SESSION HANDLERS ====================
    subgraph ORBSessions["📡 ORB Session Checks  (engine/main.py)"]
        U1["CHECK_ORB_CME  18:00–20:00 ET"]
        U2["CHECK_ORB_SYDNEY  18:30–20:30 ET"]
        U3["CHECK_ORB_TOKYO  19:00–21:00 ET"]
        U4["CHECK_ORB_SHANGHAI  21:00–23:00 ET"]
        U5["CHECK_ORB_FRANKFURT  03:00–04:30 ET"]
        U6["CHECK_ORB_LONDON  03:00–05:00 ET"]
        U7["CHECK_ORB_LONDON_NY  08:00–10:00 ET"]
        U8["CHECK_ORB  09:30–11:00 ET  US open"]
        U9["CHECK_ORB_CRYPTO_UTC0 / UTC12\nBTC/ETH/SOL crypto windows"]
        U10["CHECK_BREAKOUT_MULTI  (every 2 min)\nPDR · IB · Consolidation +\n9 additional types in parallel\n(handle_breakout_multi)"]
    end

    %% ==================== SIGNAL QUALITY & CONFLUENCE ====================
    subgraph Quality["✅ Signal Quality & Confluence  (analysis/)"]
        V1["MultiTimeframeFilter\nHTF bias + EMA alignment\n(confluence.py)"]
        V2["RegimeFilter\nVolatility regime, trend\n(regime.py)"]
        V3["CryptoMomentumScore\ncross_asset.py + crypto_momentum.py"]
        V4["CVD + VolumeProfile\ncvd.py + volume_profile.py"]
        V5["ICT concepts\nict.py  FVG, OB, liquidity"]
        V6["WaveAnalysis\nwave_analysis.py  EW structure"]
    end

    %% ==================== RISK MANAGEMENT ====================
    subgraph Risk["🛡️ Risk Management  (engine/risk.py + live_risk.py)"]
        W["RiskManager.can_enter_trade()\nDaily P&L gate\nConsecutive loss limit\nOpen position cap\nOvernight risk check"]
        W --> X["LiveRiskState\nPer-asset snapshots\nDynamic sizing\nHealth score\n(engine/live_risk.py)"]
        X --> Y["LiveRiskPublisher\nTicks every 5s\nPublishes → Redis\nengine:live_risk"]
    end

    %% ==================== POSITION & ORDER MANAGEMENT ====================
    subgraph Positions["📈 Position & Order Management"]
        Z["PositionManager.process_signal()\nMicroPosition state machine\nBracketPhase: ENTRY→TP1→BE→TRAIL→TP3\n(engine/position_manager.py)"]
        Z --> AA["3-Phase Bracket\nTP1 → move stop to BE\nEMA9 trailing stop → TP3\nSAR always-in reversal logic"]
        AA --> AB["CopyTrader.execute_order_commands()\nRithmic multi-account copy\nCompliance checklist + rate limiter\n(engine/copy_trader.py)"]
        AB --> AC["RithmicAccountManager\nOrder placement on prop accounts\nEOD hard close 16:00 ET\n(integrations/rithmic_client.py)"]
    end

    %% ==================== SWING DETECTOR ====================
    subgraph Swing["📊 Swing Detector  (engine/swing.py)"]
        AD["CHECK_SWING  every 2 min  03:00–15:30 ET\nScans daily-plan swing candidates\nSwingState per asset: PENDING→ACTIVE→CLOSED"]
        AD --> AE["Pullback / Breakout / Gap entries\n15m + 5m bar fetch\nManual accept/ignore/close\nvia dashboard actions"]
    end

    %% ==================== TRAINING PIPELINE ====================
    subgraph Training["🧠 Training Pipeline  [Docker: trainer]"]
        AF["OFF-HOURS triggers\nGENERATE_CHART_DATASET\n→ DatasetGenerator\nRBSimulator + ORBSimulator\n(services/training/)"]
        AF --> AG["dataset_generator.py\n180-day lookback\n13 breakout types × all assets\n_build_row() 25+ features"]
        AG --> AH["TRAIN_BREAKOUT_CNN\nHybridBreakoutCNN training\nPyTorch + Optuna walk-forward\ntrainer_server.py  FastAPI"]
        AH --> AI["Champion promotion\nbreakout_cnn_best.pt\nfeature_contract.json\n→ models/ directory"]
        AI --> AJ["ModelWatcher hot-reload\nwatchdog inotify / polling fallback\ninvalidate_model_cache()\n→ Redis model_reloaded event\n(engine/model_watcher.py)"]
    end

    %% ==================== BACKFILL & OPTIMISATION (OFF-HOURS) ====================
    subgraph OffHours["⚙️ Off-Hours  (12:00–18:00 ET)"]
        AK["HISTORICAL_BACKFILL\nengine/backfill.py\nWarm Redis + Postgres\nfrom MassiveAPI / Kraken"]
        AL["RUN_OPTIMIZATION\nOptuna nightly study\nwalk-forward 30–90 days\n(strategies/backtesting.py)"]
        AM["RUN_BACKTEST\nstrategies/backtesting.py\nP&L + win-rate stats"]
        AN["DAILY_REPORT\n_handle_daily_report()\nP&L, trades, signals,\nGrok review → email + Discord\n(engine/main.py)"]
        AO["POSITION_CLOSE_WARNING 15:45 ET\nEOD_POSITION_CLOSE 16:00 ET\nRithmic cancel_all + exit_all"]
    end

    %% ==================== KRAKEN PORTFOLIO ====================
    subgraph Kraken["💰 Kraken Crypto Portfolio  24/7"]
        AP["KrakenDataProvider\nREST OHLCV + ticker\nPortfolio balance queries"]
        AP --> AQ["KrakenFeedManager WebSocket\nReal-time OHLC + trades\nBars pushed → Redis on close"]
        AQ --> AR["Crypto ORB sessions\nCHECK_ORB_CRYPTO_UTC0\nCHECK_ORB_CRYPTO_UTC12\nSame 13-type detection pipeline"]
    end

    %% ==================== DASHBOARD & WEB ====================
    subgraph Web["📊 Web / Dashboard  [Docker: web + data]"]
        AS["FastAPI web proxy\n(services/web/main.py)\nHTMX + SSE dashboard"]
        AS --> AT["Live Risk strip\nengine:live_risk → Redis → SSE\nPer-asset P&L + health score"]
        AS --> AU["Focus cards\nDaily plan + conviction scores\nManual swing accept/ignore"]
        AS --> AV["Grok chat + review cards\nGrok live update every N min\n(grok_helper.py)"]
        AS --> AW["Rithmic account panel\nSettings: services, features,\nrisk config, API keys"]
        AS --> AX["CNN panel + model info\nTrainer redirect + log stream\nChart dataset viewer"]
        AS --> AY["Journal history\nTrade grading + audit log\nORB / RB history pages"]
    end

    %% ==================== ALERTS ====================
    subgraph Alerts["🔔 Alerts  (core/alerts.py)"]
        AZ["Discord smart gate\nMaster toggle\nFocus-only filter\nLive breakout events\nGap alerts + daily report"]
    end

    %% ==================== MONITORING ====================
    subgraph Monitoring["📈 Monitoring  [Docker: prometheus + grafana]"]
        BA["Prometheus :9090\nScrapes /metrics from\ndata, engine, trainer"]
        BA --> BB["Grafana :3000\nDashboards: P&L, signal\nquality, CNN accuracy,\nrisk utilisation"]
    end

    %% ==================== FLOW CONNECTIONS ====================

    %% External → Data
    A1 -->|"OHLCV bars"| B
    A2 -->|"REST bars"| B
    A2 -->|"WebSocket ticks"| F
    A3 -->|"sentiment snapshots"| G
    A4 -->|"news scores"| O
    A5 -->|"positions / fills"| Y

    %% Data → Registry
    B --> H
    H -->|"asset specs, tickers"| B

    %% Data → Scheduler
    C -->|"Redis commands\nforce_retrain, etc."| Scheduler

    %% Scheduler → Session flows
    Scheduler --> PreMarket
    Scheduler --> ORBSessions
    Scheduler --> Swing
    Scheduler --> OffHours

    %% Pre-Market feeds into live detection
    M -->|"focus assets + daily plan"| BreakoutCore

    %% ORB session handlers → Breakout core
    ORBSessions -->|"fetch_bars_1m + symbol list"| BreakoutCore

    %% Breakout core → Quality filters
    T -->|"CNN verdict + features"| Quality

    %% Quality → Risk gate
    Quality -->|"filtered signal"| W

    %% Risk → Position
    W -->|"can_enter=True"| Z

    %% Position → Orders
    Z --> AB

    %% Swing → Risk
    AE -->|"swing signal"| W

    %% Training pipeline
    OffHours --> AF
    AF --> Training
    AJ -->|"hot-reload new model"| T

    %% Kraken integrated into detection
    Kraken --> BreakoutCore

    %% Publish results → Redis → Dashboard
    Y -->|"live_risk state"| C
    Z -->|"positions + PnL"| C
    BreakoutCore -->|"breakout results\npersisted to Postgres"| D
    C -->|"SSE stream"| Web

    %% Alerts wired to events
    BreakoutCore -->|"signal alerts"| AZ
    AN -->|"daily report"| AZ
    Y -->|"risk events"| AZ
    AZ -->|"Discord webhooks"| External

    %% Monitoring scrapes
    E -->|"/metrics"| BA
    AH -->|"/metrics"| BA
"""

# =============================================================================
# SCRIPT LOGIC
# =============================================================================


def generate_mermaid_files() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "docs"
    output_dir.mkdir(exist_ok=True)

    # 1. Save raw Mermaid markdown
    mmd_path = output_dir / "futures_logic_flow.mmd"
    mmd_path.write_text(MERMAID, encoding="utf-8")
    print(f"✅ Saved Mermaid source: {mmd_path}")

    # 2. Render SVG via kroki.io POST (handles large diagrams that exceed URL limits).
    #    SVG is the primary output — it scales perfectly and opens in any browser.
    #    A PNG render is also attempted; it may fail on the free kroki.io tier for
    #    very large diagrams due to puppeteer memory limits, which is non-fatal.
    svg_path = output_dir / "futures_logic_flow.svg"
    png_path = output_dir / "futures_logic_flow.png"

    # --- SVG (primary) ---
    try:
        response = requests.post(
            "https://kroki.io/mermaid/svg",
            json={"diagram_source": MERMAID},
            timeout=60,
        )
        response.raise_for_status()
        svg_path.write_bytes(response.content)
        print(f"✅ Rendered SVG image (kroki.io): {svg_path}")
        print("   Open in any browser, or paste the .mmd into https://mermaid.live")
    except Exception as svg_err:
        print(f"⚠️  Could not render SVG (no internet or API issue): {svg_err}")
        print("   Just open the .mmd file in https://mermaid.live instead")

    # --- PNG (best-effort, may time out for large diagrams on free tier) ---
    try:
        response = requests.post(
            "https://kroki.io/mermaid/png",
            json={"diagram_source": MERMAID},
            timeout=90,
        )
        response.raise_for_status()
        png_path.write_bytes(response.content)
        print(f"✅ Rendered PNG image (kroki.io): {png_path}")
    except Exception as png_err:
        print(f"ℹ️  PNG render skipped (diagram too large for free kroki.io tier): {png_err}")
        print(f"   Use the SVG instead: {svg_path}")


if __name__ == "__main__":
    print("🚀 Generating Futures Trading System Logic Flow (v6)...\n")
    generate_mermaid_files()
    print("\nDone! Check the 'docs/' folder.")
