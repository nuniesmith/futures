"""
Grok AI helper for live trading analysis.

Provides two main functions:
  1. run_morning_briefing()  — comprehensive pre-market game plan
  2. run_live_analysis()     — concise 15-minute update during active trading

Uses the xAI API (OpenAI-compatible) with grok-4-1-fast-reasoning model.
Cost is extremely low: ~$0.007-0.01 per call, so a full trading day
(pre-market + ~16 live calls over 4 hours) costs well under $0.20.

Usage:
    from grok_helper import run_live_analysis, run_morning_briefing, format_market_context

    # Build context from engine + scanner data
    context = format_market_context(engine, scanner_df, account_size, ...)

    # Pre-market
    briefing = run_morning_briefing(context, api_key)

    # Every 15 min during live trading
    update = run_live_analysis(context, api_key, previous_briefing=briefing)
"""

import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("grok_helper")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_MAX_TOKENS_BRIEFING = 3000
DEFAULT_MAX_TOKENS_LIVE = 800
DEFAULT_TEMPERATURE = 0.3


def _escape_dollars(text: str) -> str:
    """Escape bare $ signs so Streamlit doesn't render them as LaTeX."""
    text = text.replace("$$", "\x00DBL\x00")
    text = re.sub(r"\$([0-9,.\-+])", r"\\$\1", text)
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    text = text.replace("\x00DBL\x00", "$$")
    return text


def _call_grok(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
) -> str | None:
    """Call the Grok API and return the response text.

    Returns None on error (logged, not raised).
    """
    if not api_key:
        logger.warning("No Grok API key provided")
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            GROK_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROK_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content
    except requests.exceptions.Timeout:
        logger.error("Grok API timeout after 90s")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error("Grok API HTTP error: %s", e)
        return None
    except Exception as e:
        logger.error("Grok API unexpected error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Market context builder
# ---------------------------------------------------------------------------


def format_market_context(
    engine,
    scanner_df,
    account_size: int,
    risk_dollars: int,
    max_contracts: int,
    contract_specs: dict,
    selected_assets: list[str],
    ict_summaries: dict | None = None,
    confluence_results: dict | None = None,
    cvd_summaries: dict | None = None,
    scorer_results: list[dict] | None = None,
    live_positions: dict | None = None,
) -> dict:
    """Build a structured context dict from all available market data.

    This context is passed to both morning briefing and live analysis
    so Grok has full visibility into the current market state.
    """
    now_est = datetime.now(tz=_EST)

    # Scanner text
    scanner_text = "No scanner data available"
    if scanner_df is not None and not scanner_df.empty:
        scanner_text = scanner_df.to_string(index=False)

    # Contract specs text
    specs_parts = []
    for asset_name, spec in contract_specs.items():
        data_ticker = spec.get("data_ticker", spec["ticker"])
        scan_price = "N/A"
        if scanner_df is not None and not scanner_df.empty:
            match = scanner_df.loc[scanner_df["Asset"] == asset_name, "Last"]
            if not match.empty:
                scan_price = str(match.iloc[0])
        specs_parts.append(
            f"  {asset_name} ({data_ticker}): "
            f"current_price={scan_price}, "
            f"tick_size={spec['tick']}, "
            f"point_value=USD {spec['point']}/point, "
            f"margin=USD {spec['margin']:,}"
        )
    specs_text = "\n".join(specs_parts)

    # Optimization results from engine cache
    from cache import get_cached_optimization

    opt_parts = []
    from models import ASSETS

    for name in selected_assets:
        ticker = ASSETS.get(name)
        if ticker:
            opt = get_cached_optimization(ticker, "5m", "5d")
            if opt:
                strat_label = opt.get("strategy_label", opt.get("strategy", "?"))
                confidence = opt.get("confidence", "?")
                regime = opt.get("regime", "?")
                opt_parts.append(
                    f"  {name}: strategy={strat_label}, "
                    f"return={opt.get('return_pct', '?')}%, "
                    f"sharpe={opt.get('sharpe', '?')}, "
                    f"win_rate={opt.get('win_rate', '?')}%, "
                    f"confidence={confidence}, regime={regime}"
                )
    opt_text = "\n".join(opt_parts) if opt_parts else "Not yet run"

    # Backtest results
    bt_results = engine.get_backtest_results() if engine else []
    bt_parts = []
    for r in bt_results:
        bt_parts.append(
            f"  {r['Asset']}: return={r['Return %']}%, "
            f"win_rate={r['Win Rate %']}%, "
            f"sharpe={r['Sharpe']}, trades={r['# Trades']}"
        )
    bt_text = "\n".join(bt_parts) if bt_parts else "Not yet run"

    # ICT summary text
    ict_text = "Not available"
    if ict_summaries:
        ict_parts = []
        for asset_name, summary in ict_summaries.items():
            stats = summary.get("stats", {})
            nearest = summary.get("nearest_levels", {})
            above = nearest.get("above", {})
            below = nearest.get("below", {})
            ict_parts.append(
                f"  {asset_name}: "
                f"unfilled_FVGs={stats.get('unfilled_fvgs', 0)}, "
                f"active_OBs={stats.get('active_obs', 0)}, "
                f"sweeps={stats.get('recent_sweeps', 0)}, "
                f"nearest_above={above.get('label', '—')} @ {above.get('price', '—')}, "
                f"nearest_below={below.get('label', '—')} @ {below.get('price', '—')}"
            )
        ict_text = "\n".join(ict_parts)

    # Confluence summary text
    conf_text = "Not available"
    if confluence_results:
        conf_parts = []
        for asset_name, conf in confluence_results.items():
            score = conf.get("score", 0)
            direction = conf.get("direction", "neutral")
            emoji = "🟢" if score >= 3 else "🟡" if score >= 2 else "🔴"
            conf_parts.append(
                f"  {asset_name}: {emoji} {score}/3 — bias={direction.upper()}"
            )
        conf_text = "\n".join(conf_parts)

    # CVD summary text
    cvd_text = "Not available"
    if cvd_summaries:
        cvd_parts = []
        for asset_name, summary in cvd_summaries.items():
            bias = summary.get("bias", "neutral")
            slope = summary.get("cvd_slope", 0)
            cvd_parts.append(
                f"  {asset_name}: bias={bias}, "
                f"slope={slope:+.3f}, "
                f"delta={summary.get('delta_current', 0):,.0f}"
            )
        cvd_text = "\n".join(cvd_parts)

    # Scorer summary text
    scorer_text = "Not available"
    if scorer_results:
        scorer_parts = []
        for r in scorer_results:
            scorer_parts.append(
                f"  {r['asset']}: score={r['composite_score']:.0f}/100, "
                f"signal={r['signal']}"
            )
        scorer_text = "\n".join(scorer_parts)

    # Session status
    current_hour = now_est.hour
    if 3 <= current_hour < 10:
        session_status = "ACTIVE — primary entry window"
    elif 10 <= current_hour < 12:
        session_status = "WIND DOWN — manage only, no new entries"
    else:
        session_status = "CLOSED — no trading"

    # Live positions from NinjaTrader
    positions_text = "No live positions"
    has_positions = False
    if live_positions and live_positions.get("has_positions"):
        has_positions = True
        pos_parts = []
        for p in live_positions.get("positions", []):
            symbol = p.get("symbol", "?")
            side = p.get("side", "?")
            qty = p.get("quantity", 0)
            avg = p.get("avgPrice", 0)
            upnl = p.get("unrealizedPnL", 0)
            pnl_emoji = "🟢" if upnl >= 0 else "🔴"
            pos_parts.append(
                f"  {symbol}: {side} x{qty} @ {avg:.2f} — "
                f"{pnl_emoji} unrealized USD {upnl:+,.2f}"
            )
        total_pnl = live_positions.get("total_unrealized_pnl", 0)
        acct_name = live_positions.get("account", "")
        positions_text = (
            f"Account: {acct_name} | Total unrealized: USD {total_pnl:+,.2f}\n"
            + "\n".join(pos_parts)
        )

    return {
        "time": now_est.strftime("%Y-%m-%d %H:%M EST"),
        "account_size": account_size,
        "risk_dollars": risk_dollars,
        "max_contracts": max_contracts,
        "session_status": session_status,
        "scanner_text": scanner_text,
        "specs_text": specs_text,
        "opt_text": opt_text,
        "bt_text": bt_text,
        "ict_text": ict_text,
        "conf_text": conf_text,
        "cvd_text": cvd_text,
        "scorer_text": scorer_text,
        "positions_text": positions_text,
        "has_positions": has_positions,
    }


# ---------------------------------------------------------------------------
# Morning briefing (pre-market)
# ---------------------------------------------------------------------------

_MORNING_SYSTEM = (
    "You are a disciplined futures trading co-pilot focused on micro contracts "
    "(MES, MNQ, MCL, MGC, MHG, SIL). You prioritize capital preservation and "
    "quality setups over quantity. Be concise, use bullet points, and always "
    "anchor prices to the scanner's 'Last' column. NEVER use bare $ signs — "
    "write 'USD' instead. Do not use LaTeX or math notation."
)


def run_morning_briefing(context: dict, api_key: str) -> str | None:
    """Generate a comprehensive pre-market game plan.

    Returns the formatted analysis text, or None on error.
    """
    prompt = f"""Pre-market briefing for {context["time"]}.

Account: USD {context["account_size"]:,} | Risk/trade: USD {context["risk_dollars"]:,} | Max contracts: {context["max_contracts"]}
Session: {context["session_status"]}

CONTRACT SPECS:
{context["specs_text"]}

MARKET SCANNER (Last = current price):
{context["scanner_text"]}

OPTIMIZED STRATEGIES (auto-selected by engine):
{context["opt_text"]}

BACKTESTS (session-hours only):
{context["bt_text"]}

ICT LEVELS (FVGs, Order Blocks, Sweeps):
{context["ict_text"]}

CONFLUENCE (Multi-Timeframe):
{context["conf_text"]}

CVD (Volume Delta):
{context["cvd_text"]}

PRE-MARKET SCORES:
{context["scorer_text"]}

Give me today's game plan:
1. **Market Bias** — overall read on the session (1-2 sentences)
2. **Top 3 Focus Assets** — rank by setup quality, explain why each
3. **Key Levels to Watch** — entry zones, SL, TP for each focus asset (use scanner prices + ICT levels)
4. **Correlations** — what pairs to monitor together
5. **Risk Warnings** — anything that could trip us up today
6. **Session Plan** — when to be aggressive vs. patient

Keep it actionable. No fluff. This is my reference card for the trading session."""

    result = _call_grok(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_BRIEFING,
        system_prompt=_MORNING_SYSTEM,
    )
    return result


# ---------------------------------------------------------------------------
# Live trading analysis (every 15 minutes)
# ---------------------------------------------------------------------------

_LIVE_SYSTEM = (
    "You are a real-time futures trading co-pilot. Give concise, actionable "
    "15-minute updates. Use bullet points. Focus on what changed since last "
    "update and what to do about it. NEVER use bare $ signs — write 'USD' instead."
)


def run_live_analysis(
    context: dict,
    api_key: str,
    previous_briefing: str | None = None,
    previous_update: str | None = None,
    update_number: int = 1,
) -> str | None:
    """Generate a concise 15-minute market update during active trading.

    This is designed to be cheap (~$0.007 per call) and fast.
    Returns formatted update text, or None on error.
    """
    # Include a summary of the morning plan for continuity
    plan_ref = ""
    if previous_briefing:
        # Take first 500 chars of the morning briefing as context
        plan_ref = (
            f"\nMORNING PLAN SUMMARY (reference):\n{previous_briefing[:500]}...\n"
        )

    prev_ref = ""
    if previous_update:
        prev_ref = f"\nLAST UPDATE:\n{previous_update[:300]}...\n"

    # Include live positions if available
    positions_block = ""
    if context.get("has_positions"):
        positions_block = f"""
LIVE POSITIONS (from NinjaTrader):
{context["positions_text"]}
"""

    prompt = f"""15-minute market update #{update_number} — {context["time"]}

Account: USD {context["account_size"]:,} | Session: {context["session_status"]}
{plan_ref}{prev_ref}{positions_block}
CURRENT SCANNER:
{context["scanner_text"]}

ICT LEVELS:
{context["ict_text"]}

CVD:
{context["cvd_text"]}

CONFLUENCE:
{context["conf_text"]}

Give me a quick update (5-8 bullet points max):
- What moved since last check? Any significant price action?
- Are our focus assets still in play? Any setups triggered or invalidated?
- CVD/volume delta shifts — is buying or selling pressure changing?
- Any new ICT levels hit (FVGs filled, OBs tested, liquidity swept)?"""

    # Add position-specific prompts when positions are open
    if context.get("has_positions"):
        prompt += """
- POSITION CHECK — how are our open positions doing relative to key levels?
- Should we hold, scale, trail stops, or exit any positions?
- Risk check — unrealized P&L vs daily drawdown limit, approaching session wind-down?"""
    else:
        prompt += """
- Risk check — are we approaching session wind-down or any danger zones?"""

    prompt += """
- One-line summary: what to do RIGHT NOW

Be extremely concise. This is a quick check-in, not a full analysis."""

    result = _call_grok(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_LIVE,
        system_prompt=_LIVE_SYSTEM,
    )
    return result


# ---------------------------------------------------------------------------
# Grok session manager (tracks state across 15-min intervals)
# ---------------------------------------------------------------------------


class GrokSession:
    """Manages Grok analysis state across a trading session.

    Tracks timing, stores briefings/updates, and handles the 15-minute
    interval logic so the Streamlit app stays simple.
    """

    LIVE_INTERVAL_SEC = 900  # 15 minutes

    def __init__(self):
        self.morning_briefing: str | None = None
        self.updates: list[dict] = []  # [{time, text, number}]
        self.last_update_time: float = 0
        self.is_active: bool = False
        self.total_calls: int = 0
        self.estimated_cost: float = 0.0

    def activate(self) -> None:
        """Start the live analysis session."""
        self.is_active = True
        # Force first update on next check
        self.last_update_time = 0
        logger.info("Grok live session activated")

    def deactivate(self) -> None:
        """Stop the live analysis session."""
        self.is_active = False
        logger.info(
            "Grok live session deactivated after %d updates (est. cost: $%.4f)",
            len(self.updates),
            self.estimated_cost,
        )

    def set_morning_briefing(self, text: str) -> None:
        """Store the morning briefing for reference during live updates."""
        self.morning_briefing = text
        self.total_calls += 1
        self.estimated_cost += 0.008  # ~$0.008 per briefing call

    def needs_update(self) -> bool:
        """Check if enough time has passed for the next 15-min update."""
        if not self.is_active:
            return False
        return (time.time() - self.last_update_time) >= self.LIVE_INTERVAL_SEC

    def run_update(self, context: dict, api_key: str) -> str | None:
        """Run a live analysis update if the interval has elapsed.

        Returns the update text, or None if not yet time or on error.
        """
        if not self.needs_update():
            return None

        update_number = len(self.updates) + 1
        previous_update = self.updates[-1]["text"] if self.updates else None

        result = run_live_analysis(
            context=context,
            api_key=api_key,
            previous_briefing=self.morning_briefing,
            previous_update=previous_update,
            update_number=update_number,
        )

        if result:
            self.updates.append(
                {
                    "time": datetime.now(tz=_EST).strftime("%H:%M EST"),
                    "text": result,
                    "number": update_number,
                }
            )
            self.last_update_time = time.time()
            self.total_calls += 1
            self.estimated_cost += 0.007  # ~$0.007 per live update
            logger.info("Grok live update #%d completed", update_number)

        return result

    def get_latest_update(self) -> dict | None:
        """Return the most recent update dict, or None."""
        return self.updates[-1] if self.updates else None

    def get_session_summary(self) -> dict:
        """Return a summary of this Grok session."""
        return {
            "is_active": self.is_active,
            "has_briefing": self.morning_briefing is not None,
            "total_updates": len(self.updates),
            "total_calls": self.total_calls,
            "estimated_cost": round(self.estimated_cost, 4),
            "last_update": self.updates[-1]["time"] if self.updates else None,
        }
