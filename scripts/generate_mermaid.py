import base64
from pathlib import Path

import requests  # type: ignore[import-untyped]

# =============================================================================
# FUTURES TRADING SYSTEM — FULL LOGIC FLOW (v5)
# Copy-paste this entire script into your repo root as:
#     generate_mermaid.py
# Then run: python generate_mermaid.py
# It will create:
#   - futures_logic_flow.mmd     (open in mermaid.live or VSCode)
#   - futures_logic_flow.png     (beautiful rendered image)
# =============================================================================

MERMAID = """flowchart TD
    %% ==================== EXTERNAL SOURCES ====================
    subgraph External["🌐 External Sources"]
        A1[MassiveAPI + yfinance\nCME futures 1m/5m/15m/daily]
        A2[Kraken REST + WebSocket\n9+ spot pairs 24/7]
        A3[Reddit WSB via PRAW + VADER]
    end

    %% ==================== DATA LAYER ====================
    subgraph Data["📥 Unified Data Layer"]
        B[DataResolver\nRedis hot → Postgres durable → External fallback]
        B -->|cache warm| C[Redis\nbars_1m, bars_15m, daily, focus, risk]
        B -->|persist| D[Postgres\naudit, trades, journal, logs]
    end

    %% ==================== SCHEDULER & TIMING ====================
    subgraph Scheduler["⏰ ScheduleManager (EST aware)"]
        E[18:00–04:00 → FUTURES ACTIVE\nRB detection every 2min]
        F[03:00–12:00 → MANUAL LONDON MODE]
        G[04:00–06:00 → POST-MARKET\nOptuna + Journal + Grok]
        H[Weekend 48h → FULL RETRAIN]
        I[17:59 → PRE-MARKET LOG\n04:00 → POST-MARKET LOG\nFriday → WEEKLY REPORT]
    end

    %% ==================== PRE-MARKET ROUTINE ====================
    subgraph PreMarket["🌅 Pre-Market (04:00–06:00)"]
        J[Optuna nightly study\nwalk-forward on 30-90 days]
        J --> K[DailyBiasAnalyzer + DailyPlanGenerator\n+ Grok macro brief]
        K --> L[ConvictionStack.compute\nWSB + CryptoMomentum + Grok multipliers]
        L --> M[select_daily_focus_assets\n3-4 scalps + 1-2 swings]
        M --> N[Persist to Redis\nengine:focus_assets + engine:daily_plan]
        N --> O[PRE-MARKET LOG → JSON + Postgres]
    end

    %% ==================== MANUAL LONDON MODE ====================
    subgraph Manual["🛠️ Manual London Mode (03:00–12:00 ET)"]
        P[Dashboard auto-switches to Manual View]
        Q[Focus cards ranked by Conviction Stack Score 0-100]
        Q -->|high stack| R[Manual entry buttons\nTradingView / dashboard\nquick LONG/SHORT + Take Profit Now]
        R --> S[You take early profit & done by noon]
        S --> T[Auto switches back to full automation at 12:00]
    end

    %% ==================== LIVE TRADING CORE ====================
    subgraph Live["🔴 LIVE TRADING (18:00–04:00)"]
        U[handle_breakout_check\ngeneric for all 13 RB types]
        U --> V[detect_range_breakout\nrange_builders.py + unified filters]
        V --> W[apply_all_filters + CNN inference\nv7 25-feature with WSB social]
        W --> X[ConvictionStack score\nmultipliers: WSB hype, Crypto momentum, Grok]
        X --> Y[RiskManager.can_enter\nLiveRiskState + Discord smart gate]
        Y --> Z[PositionManager\nSAR always-in scalps OR DailySwing]
        Z --> AA[3-phase bracket\nTP1 → BE → EMA9 trail to TP3]
        AA --> BB[TradingView signal → Tradovate\nexecutes on prop accounts]
    end

    %% ==================== JOURNALING & LLM ====================
    subgraph Journal["📓 Journaling & Grok Reviews"]
        CC[POST-MARKET LOG 04:00\nP&L + trades + signals + stack stats]
        CC --> DD[Grok daily review\ninsights → new metrics/features]
        DD --> EE[WEEKLY REPORT Friday\nGrok weekly + rolling suggestions]
        EE --> FF[Weekend full retrain\nuses journal data as labeling hints]
    end

    %% ==================== KRAKEN 24/7 ====================
    subgraph Kraken["💰 Kraken Spot Portfolio 24/7"]
        GG[PortfolioManager\nrebalance every 30min if >5% deviation]
        GG --> HH[DCA + volatility filter + cooldown]
        HH --> II[Futures profits sweep → Kraken growth]
    end

    %% ==================== DASHBOARD & ALERTS ====================
    subgraph UI["📊 HTMX Dashboard"]
        JJ[LiveRisk strip + Focus cards + Manual London toggle]
        JJ --> KK[Smart Discord gate\nMaster toggle + focus-only + live events]
        KK --> LL[Grok review cards + Journal history]
    end

    %% ==================== WEEKEND RETRAIN ====================
    subgraph Weekend["🧠 Weekend 48h Training"]
        MM[Full dataset regen\n180 days + WSB + Conviction features]
        MM --> NN[Long CNN retrain + deep Optuna]
        NN --> OO[Export ONNX + feature_contract vX\nsync_models → engine hot-reload]
    end

    %% ==================== CONNECTIONS ====================
    External --> Data
    Data --> Scheduler
    Scheduler --> PreMarket
    PreMarket --> Manual
    Manual --> Live
    Live --> Journal
    Live --> Kraken
    Journal --> Weekend
    Weekend --> Data
    Live --> UI
    Journal --> UI
    Kraken --> UI
"""

# =============================================================================
# SCRIPT LOGIC
# =============================================================================


def generate_mermaid_files():
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    # 1. Save raw Mermaid markdown
    mmd_path = output_dir / "futures_logic_flow.mmd"
    mmd_path.write_text(MERMAID, encoding="utf-8")
    print(f"✅ Saved Mermaid source: {mmd_path}")

    # 2. Render PNG via public mermaid.ink API (no dependencies beyond requests)
    try:
        # Encode Mermaid to base64 (mermaid.ink format)
        graph_bytes = MERMAID.encode("utf-8")
        b64 = base64.urlsafe_b64encode(graph_bytes).decode("ascii").rstrip("=")
        url = f"https://mermaid.ink/img/{b64}"

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        png_path = output_dir / "futures_logic_flow.png"
        png_path.write_bytes(response.content)
        print(f"✅ Rendered PNG image: {png_path}")
        print("   Open the PNG or paste the .mmd into https://mermaid.live for editing")

    except Exception as e:
        print(f"⚠️  Could not render PNG (no internet or API issue): {e}")
        print("   Just open the .mmd file in mermaid.live instead")


if __name__ == "__main__":
    print("🚀 Generating Futures Trading System Logic Flow (v5)...\n")
    generate_mermaid_files()
    print("\nDone! Check the 'docs/' folder.")
