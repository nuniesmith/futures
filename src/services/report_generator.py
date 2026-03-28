"""
Grok AI Report Generator — Daily / Weekly / Monthly Performance Reports
=======================================================================
Uses xAI API (Grok 4.1) to generate intelligent trading performance
reports from Redis trade data and local log files.

Reports:
  - Daily:   End-of-day summary with per-asset breakdown, highlights
  - Weekly:  7-day rollup with trends, best/worst performers, strategy insights
  - Monthly: 30-day comprehensive review with recommendations
  - Highlights: Notable events (big wins, losses, streaks, pattern changes)

Usage::

    from src.services.report_generator import ReportGenerator

    gen = ReportGenerator(redis_store=store)
    daily = await gen.generate_daily_report()
    weekly = await gen.generate_weekly_report()
    monthly = await gen.generate_monthly_report()
"""

from __future__ import annotations

import asyncio
import calendar
import glob
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.logging_config import get_logger

logger = get_logger(__name__)

_ET = ZoneInfo("America/New_York")
GROK_MODEL = "grok-4-1"
XAI_API_URL = "https://api.x.ai/v1"

ASSETS = ["BTC", "ETH", "SOL", "DOGE", "SUI", "PEPE", "AVAX", "WIF", "FARTCOIN", "KCS"]

_SYSTEM_BASE = (
    "You are a senior quantitative crypto-futures trading performance analyst. "
    "You work for a systematic trading desk that runs a multi-asset scalping bot on "
    "KuCoin USDTM Perpetual Futures across 10 assets: "
    "BTC, ETH, SOL, DOGE, SUI, PEPE, AVAX, WIF, FARTCOIN, and KCS.\n\n"
    "Your job is to produce clear, actionable performance reports from raw trade data. "
    "Be direct, data-driven, and insightful. Use tables where appropriate. "
    "Highlight anomalies, streaks, and anything the trader should pay attention to. "
    "All PnL figures are in USDT. Times are Eastern Time (ET).\n\n"
    "Format your reports in clean Markdown."
)


class ReportGenerator:
    """Generate AI-powered trading reports using Grok 4.1."""

    def __init__(
        self,
        redis_store: Any,
        api_key: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        """
        Args:
            redis_store: RedisStore instance for reading trade data.
            api_key: xAI API key. Reads from ``XAI_API_KEY`` env if not provided.
            log_dir: Directory containing local log files to include in analysis.
        """
        self.store = redis_store
        self.api_key = api_key or os.environ.get("XAI_API_KEY", "")
        self.log_dir = log_dir
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    #  Public — report generators
    # ------------------------------------------------------------------ #

    async def generate_daily_report(self) -> str | None:
        """Generate end-of-day performance report."""
        logger.info("Generating daily report …")
        try:
            data = await self.store.get_report_data(period="day")
            log_context = self._read_recent_logs(hours=24)
            system_prompt, user_prompt = self._build_daily_prompt(data, log_context)
            report = await self._call_grok(system_prompt, user_prompt, max_tokens=3000)
            if report:
                await self.store.store_report("daily", report)
                logger.info("Daily report stored (%d chars)", len(report))
            return report
        except Exception:
            logger.exception("Failed to generate daily report")
            return None

    async def generate_weekly_report(self) -> str | None:
        """Generate 7-day performance rollup."""
        logger.info("Generating weekly report …")
        try:
            data = await self.store.get_report_data(period="week")
            agg = await self.store.get_aggregate_stats(days=7)
            data["aggregate"] = agg
            log_context = self._read_recent_logs(hours=168)
            system_prompt, user_prompt = self._build_weekly_prompt(data, log_context)
            report = await self._call_grok(system_prompt, user_prompt, max_tokens=4000)
            if report:
                await self.store.store_report("weekly", report)
                logger.info("Weekly report stored (%d chars)", len(report))
            return report
        except Exception:
            logger.exception("Failed to generate weekly report")
            return None

    async def generate_monthly_report(self) -> str | None:
        """Generate 30-day comprehensive review."""
        logger.info("Generating monthly report …")
        try:
            data = await self.store.get_report_data(period="month")
            agg = await self.store.get_aggregate_stats(days=30)
            data["aggregate"] = agg
            log_context = self._read_recent_logs(hours=720)
            system_prompt, user_prompt = self._build_monthly_prompt(data, log_context)
            report = await self._call_grok(system_prompt, user_prompt, max_tokens=5000)
            if report:
                await self.store.store_report("monthly", report)
                logger.info("Monthly report stored (%d chars)", len(report))
            return report
        except Exception:
            logger.exception("Failed to generate monthly report")
            return None

    async def generate_highlights(self) -> str | None:
        """Generate notable events / highlights report."""
        logger.info("Generating highlights report …")
        try:
            data = await self.store.get_report_data(period="day")
            agg = await self.store.get_aggregate_stats(days=3)

            pnl_history = await self.store.get_pnl_history(asset=None, days=3)
            data["pnl_history_3d"] = pnl_history

            system_prompt, user_prompt = self._build_highlights_prompt(data, agg)
            report = await self._call_grok(system_prompt, user_prompt, max_tokens=2000)
            if report:
                await self.store.store_report("highlights", report)
                logger.info("Highlights report stored (%d chars)", len(report))
            return report
        except Exception:
            logger.exception("Failed to generate highlights report")
            return None

    # ------------------------------------------------------------------ #
    #  Scheduler
    # ------------------------------------------------------------------ #

    async def run_scheduled(self) -> None:
        """Run as a background task — generates reports on schedule.

        Daily:   every day at 23:59 ET
        Weekly:  every Sunday at 23:59 ET
        Monthly: last day of month at 23:59 ET

        Safe to run continuously; sleeps in 30-second intervals.
        """
        logger.info("Report scheduler started")
        last_daily: str | None = None
        last_weekly: str | None = None
        last_monthly: str | None = None

        while True:
            try:
                now_et = datetime.now(tz=_ET)
                today_str = now_et.strftime("%Y-%m-%d")
                is_report_time = now_et.hour == 23 and now_et.minute >= 55

                if is_report_time and last_daily != today_str:
                    logger.info("Scheduler: triggering daily report")
                    await self.generate_daily_report()
                    await self.generate_highlights()
                    last_daily = today_str

                    # Sunday = 6
                    if now_et.weekday() == 6 and last_weekly != today_str:
                        logger.info("Scheduler: triggering weekly report")
                        await self.generate_weekly_report()
                        last_weekly = today_str

                    # Last day of month
                    _, last_day = calendar.monthrange(now_et.year, now_et.month)
                    if now_et.day == last_day and last_monthly != today_str:
                        logger.info("Scheduler: triggering monthly report")
                        await self.generate_monthly_report()
                        last_monthly = today_str

            except Exception:
                logger.exception("Scheduler tick failed")

            await asyncio.sleep(30)

    # ------------------------------------------------------------------ #
    #  Log reader
    # ------------------------------------------------------------------ #

    def _read_recent_logs(self, hours: int = 24) -> str:
        """Read recent log file entries for context.

        Scans ``self.log_dir`` for ``*.log`` files, reads lines whose
        embedded timestamps fall within the last *hours*.  Returns a
        truncated string suitable for inclusion in an LLM prompt.
        """
        if not self.log_dir:
            return ""

        log_path = Path(self.log_dir)
        if not log_path.is_dir():
            return ""

        cutoff = datetime.now(tz=_ET) - timedelta(hours=hours)
        lines: list[str] = []
        max_chars = 6000  # keep prompt size reasonable

        log_files = sorted(glob.glob(str(log_path / "*.log")), key=os.path.getmtime, reverse=True)

        for fpath in log_files[:5]:  # only look at 5 most recent files
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        # Best-effort timestamp parse from standard log format
                        # Expected: "2025-01-15 14:30:00 [INFO] ..."
                        try:
                            ts_str = line[:19]
                            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                            ts = ts.replace(tzinfo=_ET)
                            if ts < cutoff:
                                continue
                        except (ValueError, IndexError):
                            pass  # include lines we can't parse timestamps for
                        lines.append(line.rstrip())
            except OSError:
                continue

        combined = "\n".join(lines)
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
            # trim to nearest newline
            nl = combined.find("\n")
            if nl != -1:
                combined = combined[nl + 1 :]
        return combined

    # ------------------------------------------------------------------ #
    #  Grok API call
    # ------------------------------------------------------------------ #

    async def _call_grok(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 3000,
    ) -> str | None:
        """Call Grok 4.1 via xAI API using the ``openai`` SDK.

        Returns the response text or ``None`` on failure.
        """
        if not self.api_key:
            logger.warning("No XAI_API_KEY configured — skipping Grok call")
            return None

        try:
            from openai import OpenAI

            if self._client is None:
                self._client = OpenAI(
                    base_url=XAI_API_URL,
                    api_key=self.api_key,
                    timeout=120.0,
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Run the synchronous SDK call in a thread so we don't block
            # the async event loop.
            resp = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=GROK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if content:
                logger.info(
                    "Grok response received (%d chars, model=%s)",
                    len(content),
                    resp.model,
                )
            return content

        except Exception:
            logger.exception("Grok API call failed")
            return None

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_daily_prompt(self, data: dict, log_context: str) -> tuple[str, str]:
        """Build system + user prompts for the daily report."""
        system_prompt = (
            _SYSTEM_BASE + "\n\n"
            "You are generating an **End-of-Day Daily Report**.\n\n"
            "Structure your report with these sections:\n"
            "1. **📊 Performance Summary** — total PnL, win rate, total trades, "
            "net result verdict (good / bad / flat day)\n"
            "2. **📈 Per-Asset Breakdown** — Markdown table: Asset | PnL | Trades | "
            "Win Rate | Notes\n"
            "3. **🏆 Best Trade / Worst Trade** — highlight the single best and worst "
            "trades with entry/exit details if available\n"
            "4. **📡 Signal Quality** — signals generated vs acted upon, any patterns "
            "in rejected signals\n"
            "5. **🌊 Market Conditions** — inferred volatility regime, trending vs "
            "ranging, notable price moves\n"
            "6. **⚡ Highlights** — notable events, streaks, unusual activity, "
            "large position sizes\n"
            "7. **🔮 Tomorrow's Outlook** — based on today's patterns, what to watch\n\n"
            "Keep it concise but thorough. Use emoji sparingly for section headers only."
        )

        stats = data.get("stats", {})
        signals = data.get("signals_by_asset", {})
        orders = data.get("recent_orders", [])

        total_signals = sum(len(sigs) for sigs in signals.values())
        signal_summary = {asset: len(sigs) for asset, sigs in signals.items() if sigs}

        user_prompt = (
            f"**Date:** {data.get('date', 'N/A')} (Eastern Time)\n\n"
            f"## Aggregate Stats\n"
            f"```json\n{json.dumps(stats, indent=2, default=str)}\n```\n\n"
            f"## Signal Counts by Asset\n"
            f"Total signals: {total_signals}\n"
            f"```json\n{json.dumps(signal_summary, indent=2)}\n```\n\n"
            f"## Recent Orders ({len(orders)} total)\n"
            f"```json\n{json.dumps(orders[:30], indent=2, default=str)}\n```\n\n"
        )

        # Add per-asset signal details (most recent 3 per asset)
        if signals:
            user_prompt += "## Recent Signal Details (last 3 per asset)\n"
            for asset, sigs in signals.items():
                if sigs:
                    user_prompt += (
                        f"\n### {asset}\n"
                        f"```json\n"
                        f"{json.dumps(sigs[:3], indent=2, default=str)}\n"
                        f"```\n"
                    )

        if log_context:
            user_prompt += f"\n## Bot Log Excerpts (last 24h)\n```\n{log_context}\n```\n"

        return system_prompt, user_prompt

    def _build_weekly_prompt(self, data: dict, log_context: str) -> tuple[str, str]:
        """Build system + user prompts for the weekly report."""
        system_prompt = (
            _SYSTEM_BASE + "\n\n"
            "You are generating a **Weekly Performance Report** (7-day rollup).\n\n"
            "Structure your report with these sections:\n"
            "1. **📊 Week in Review** — total PnL, total trades, win rate, "
            "comparison to prior week if data suggests it\n"
            "2. **🏅 Asset Performance Ranking** — table ranking all 10 assets by PnL, "
            "with trades and win rate columns\n"
            "3. **📅 Daily Breakdown** — table with each day's PnL, trades, win rate\n"
            "4. **📈 Strategy Effectiveness Trends** — are signals improving? "
            "Deteriorating? Changes in hit rate over the week\n"
            "5. **⚠️ Risk Metrics** — max drawdown during the week, largest single loss, "
            "worst day, estimate Sharpe-like ratio if possible\n"
            "6. **🏆 Best & Worst** — best performing asset, worst performing, "
            "best single trade, worst single trade\n"
            "7. **🔧 Optimization Suggestions** — specific, actionable recommendations "
            "for the coming week based on the data\n\n"
            "Be analytical and quantitative. Compare across assets and days."
        )

        stats = data.get("stats", {})
        aggregate = data.get("aggregate", stats)
        signals = data.get("signals_by_asset", {})
        orders = data.get("recent_orders", [])

        daily_breakdown = aggregate.get("daily_breakdown", [])
        per_asset = aggregate.get("per_asset", {})

        user_prompt = (
            f"**Week ending:** {data.get('date', 'N/A')} (Eastern Time)\n\n"
            f"## 7-Day Aggregate Stats\n"
            f"```json\n{json.dumps(aggregate, indent=2, default=str)}\n```\n\n"
            f"## Daily Breakdown\n"
            f"```json\n{json.dumps(daily_breakdown, indent=2, default=str)}\n```\n\n"
            f"## Per-Asset Performance\n"
            f"```json\n{json.dumps(per_asset, indent=2, default=str)}\n```\n\n"
            f"## Order History ({len(orders)} orders this week)\n"
            f"```json\n{json.dumps(orders[:50], indent=2, default=str)}\n```\n\n"
        )

        total_signals = sum(len(sigs) for sigs in signals.values())
        signal_summary = {asset: len(sigs) for asset, sigs in signals.items() if sigs}
        user_prompt += (
            f"## Signal Volume\n"
            f"Total signals: {total_signals}\n"
            f"```json\n{json.dumps(signal_summary, indent=2)}\n```\n\n"
        )

        if log_context:
            # For weekly, just include a tail of logs — heavily truncated
            trimmed = log_context[-3000:] if len(log_context) > 3000 else log_context
            user_prompt += f"## Bot Log Excerpts (tail)\n```\n{trimmed}\n```\n"

        return system_prompt, user_prompt

    def _build_monthly_prompt(self, data: dict, log_context: str) -> tuple[str, str]:
        """Build system + user prompts for the monthly report."""
        system_prompt = (
            _SYSTEM_BASE + "\n\n"
            "You are generating a **Monthly Comprehensive Review** (30-day period).\n\n"
            "This is the most important report. Be thorough and strategic.\n\n"
            "Structure your report with these sections:\n"
            "1. **📊 Monthly Summary** — total PnL, total trades, win rate, average "
            "daily PnL, best day, worst day\n"
            "2. **📈 P&L Trajectory** — describe the cumulative PnL curve shape: "
            "steady growth, volatile, drawdown-and-recovery, etc.\n"
            "3. **🏅 Asset Allocation Effectiveness** — rank all 10 assets by PnL. "
            "Identify which assets are worth keeping and which should be reconsidered. "
            "Table format: Asset | PnL | Trades | Win Rate | Avg PnL/Trade | Verdict\n"
            "4. **📅 Weekly Breakdown** — group the 30 days into ~4 weekly buckets, "
            "show trend across weeks\n"
            "5. **⚙️ Strategy & Parameter Notes** — any evidence of strategy drift, "
            "Optuna re-optimization effects, regime changes\n"
            "6. **📉 Risk Analysis** — max drawdown, worst losing streak, largest "
            "single loss, risk-adjusted returns (if computable)\n"
            "7. **💰 Capital Growth** — estimated account growth trajectory, "
            "compound growth rate if positive\n"
            "8. **🎯 Recommendations for Next Month** — specific, prioritized actions: "
            "which assets to add/drop, position sizing changes, strategy tweaks\n\n"
            "Think like a portfolio manager reviewing a trader's book."
        )

        stats = data.get("stats", {})
        aggregate = data.get("aggregate", stats)
        orders = data.get("recent_orders", [])

        daily_breakdown = aggregate.get("daily_breakdown", [])
        per_asset = aggregate.get("per_asset", {})

        user_prompt = (
            f"**Month ending:** {data.get('date', 'N/A')} (Eastern Time)\n"
            f"**Period:** 30 days\n\n"
            f"## 30-Day Aggregate Stats\n"
            f"```json\n{json.dumps(aggregate, indent=2, default=str)}\n```\n\n"
            f"## Daily Breakdown (all 30 days)\n"
            f"```json\n{json.dumps(daily_breakdown, indent=2, default=str)}\n```\n\n"
            f"## Per-Asset Performance\n"
            f"```json\n{json.dumps(per_asset, indent=2, default=str)}\n```\n\n"
            f"## Order History ({len(orders)} orders sampled)\n"
            f"```json\n{json.dumps(orders[:80], indent=2, default=str)}\n```\n\n"
        )

        # Compute some derived metrics for the LLM
        if daily_breakdown:
            pnls = [d["pnl"] for d in daily_breakdown]
            cumulative = []
            running = 0.0
            for p in pnls:
                running += p
                cumulative.append(round(running, 4))

            max_dd = 0.0
            peak = 0.0
            for c in cumulative:
                if c > peak:
                    peak = c
                dd = peak - c
                if dd > max_dd:
                    max_dd = dd

            winning_days = sum(1 for p in pnls if p > 0)
            losing_days = sum(1 for p in pnls if p < 0)
            flat_days = sum(1 for p in pnls if p == 0)

            user_prompt += (
                f"## Derived Metrics\n"
                f"- Cumulative PnL curve: {cumulative}\n"
                f"- Max drawdown: {round(max_dd, 4)} USDT\n"
                f"- Winning days: {winning_days}, Losing days: {losing_days}, "
                f"Flat days: {flat_days}\n"
                f"- Best day PnL: {round(max(pnls), 4)} USDT\n"
                f"- Worst day PnL: {round(min(pnls), 4)} USDT\n"
                f"- Avg daily PnL: {round(sum(pnls) / len(pnls), 4)} USDT\n\n"
            )

        if log_context:
            trimmed = log_context[-2000:] if len(log_context) > 2000 else log_context
            user_prompt += f"## Bot Log Excerpts (tail)\n```\n{trimmed}\n```\n"

        return system_prompt, user_prompt

    def _build_highlights_prompt(self, data: dict, aggregate: dict) -> tuple[str, str]:
        """Build system + user prompts for the highlights / notable-events report."""
        system_prompt = (
            _SYSTEM_BASE + "\n\n"
            "You are generating a **Highlights & Notable Events** report.\n\n"
            "Focus on things that stand out — things the trader MUST see:\n"
            "- 🔥 Big wins (top 3 trades by PnL)\n"
            "- 💀 Big losses (worst 3 trades by PnL)\n"
            "- 🔁 Streaks (consecutive wins or losses)\n"
            "- 📊 Unusual volume or activity spikes on any asset\n"
            "- ⚠️ Risk alerts (high drawdown, over-concentration, etc.)\n"
            "- 🆕 Pattern changes (shift in which assets are winning)\n\n"
            "Keep it punchy — bullet points and short paragraphs. "
            "This should be scannable in under 60 seconds."
        )

        pnl_history = data.get("pnl_history_3d", [])
        stats = data.get("stats", {})

        # Sort trades by PnL to find extremes
        trades_by_pnl = sorted(
            pnl_history,
            key=lambda t: t.get("pnl_usdt", 0.0),
        )

        top_wins = trades_by_pnl[-3:] if len(trades_by_pnl) >= 3 else trades_by_pnl
        top_losses = trades_by_pnl[:3] if len(trades_by_pnl) >= 3 else trades_by_pnl

        # Detect streaks
        streak_info = self._detect_streaks(pnl_history)

        user_prompt = (
            f"**Date:** {data.get('date', 'N/A')}\n\n"
            f"## 3-Day Aggregate Stats\n"
            f"```json\n{json.dumps(aggregate, indent=2, default=str)}\n```\n\n"
            f"## Top Winning Trades (last 3 days)\n"
            f"```json\n{json.dumps(top_wins, indent=2, default=str)}\n```\n\n"
            f"## Top Losing Trades (last 3 days)\n"
            f"```json\n{json.dumps(top_losses, indent=2, default=str)}\n```\n\n"
            f"## Streak Analysis\n{streak_info}\n\n"
            f"## Today's Stats\n"
            f"```json\n{json.dumps(stats, indent=2, default=str)}\n```\n"
        )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_streaks(pnl_history: list[dict]) -> str:
        """Analyse PnL history for consecutive win/loss streaks."""
        if not pnl_history:
            return "No trade history available for streak analysis."

        current_streak = 0
        current_dir = 0  # 1 = win, -1 = loss
        max_win_streak = 0
        max_loss_streak = 0
        lines: list[str] = []

        for entry in pnl_history:
            pnl = entry.get("pnl_usdt", 0.0)
            direction = 1 if pnl > 0 else (-1 if pnl < 0 else 0)

            if direction == 0:
                continue

            if direction == current_dir:
                current_streak += 1
            else:
                current_streak = 1
                current_dir = direction

            if current_dir == 1:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)

        lines.append(f"- Longest winning streak: {max_win_streak} trades")
        lines.append(f"- Longest losing streak: {max_loss_streak} trades")
        lines.append(
            f"- Current streak: {current_streak} "
            f"{'win' if current_dir == 1 else 'loss' if current_dir == -1 else 'flat'}"
            f"{'s' if current_streak != 1 else ''}"
        )

        if max_loss_streak >= 5:
            lines.append("- ⚠️ ALERT: Extended losing streak detected (5+)")
        if max_win_streak >= 7:
            lines.append("- 🔥 Notable hot streak detected (7+ consecutive wins)")

        return "\n".join(lines)


# ====================================================================== #
#  CLI entry point
# ====================================================================== #

if __name__ == "__main__":
    import asyncio
    import sys

    from src.services.redis_store import RedisStore

    async def main() -> None:
        store = RedisStore()
        await store.connect()

        gen = ReportGenerator(redis_store=store)

        report_type = sys.argv[1] if len(sys.argv) > 1 else "daily"
        generators = {
            "daily": gen.generate_daily_report,
            "weekly": gen.generate_weekly_report,
            "monthly": gen.generate_monthly_report,
            "highlights": gen.generate_highlights,
        }

        func = generators.get(report_type)
        if func is None:
            print(f"Unknown report type: {report_type}")
            print(f"Available: {', '.join(generators)}")
            sys.exit(1)

        print(f"Generating {report_type} report …")
        report = await func()
        if report:
            print(report)
        else:
            print("No report generated — check XAI_API_KEY and Redis connectivity.")

        await store.close()

    asyncio.run(main())
