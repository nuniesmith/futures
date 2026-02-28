"""
Session-Aware Engine Scheduler (TASK-202)
==========================================
Manages the engine's behavior based on Eastern Time trading sessions:

  - **Pre-market (00:00–05:00 ET):** Compute daily focus once, run Grok
    morning briefing, prepare alerts for the trading day.
  - **Active (05:00–12:00 ET):** Live FKS recomputation every 5 min,
    publish focus updates to Redis, run Grok updates every 15 min.
  - **Off-hours (12:00–00:00 ET):** Historical data backfill, full
    optimization runs, backtesting, next-day prep.

The ScheduleManager is consumed by the engine main loop. It tracks what
has already run within each session to avoid redundant work (e.g. daily
focus is computed once per pre-market window, not every loop iteration).

Usage:
    from lib.services.engine.scheduler import ScheduleManager

    mgr = ScheduleManager()
    while running:
        actions = mgr.get_pending_actions()
        for action in actions:
            run(action)
        mgr.mark_done(action)
        sleep(mgr.sleep_interval)
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("engine.scheduler")

_EST = ZoneInfo("America/New_York")


class SessionMode(str, Enum):
    PRE_MARKET = "pre-market"
    ACTIVE = "active"
    OFF_HOURS = "off-hours"


class ActionType(str, Enum):
    """All schedulable engine actions."""

    # Pre-market actions (run once per day)
    COMPUTE_DAILY_FOCUS = "compute_daily_focus"
    GROK_MORNING_BRIEF = "grok_morning_brief"
    PREP_ALERTS = "prep_alerts"

    # Active session actions (recurring)
    FKS_RECOMPUTE = "fks_recompute"
    PUBLISH_FOCUS_UPDATE = "publish_focus_update"
    GROK_LIVE_UPDATE = "grok_live_update"
    CHECK_RISK_RULES = "check_risk_rules"
    CHECK_NO_TRADE = "check_no_trade"
    CHECK_ORB = "check_orb"

    # Off-hours actions (run once per session)
    HISTORICAL_BACKFILL = "historical_backfill"
    RUN_OPTIMIZATION = "run_optimization"
    RUN_BACKTEST = "run_backtest"
    NEXT_DAY_PREP = "next_day_prep"


@dataclass
class ScheduledAction:
    """A single action the engine should execute."""

    action: ActionType
    priority: int = 0  # lower = higher priority
    description: str = ""


@dataclass
class _ActionTracker:
    """Tracks when an action was last executed and whether it's been
    completed for the current session/day."""

    last_run: Optional[float] = None  # timestamp
    last_run_date: Optional[date] = None  # for once-per-day actions
    last_run_session: Optional[str] = None  # for once-per-session actions
    run_count_today: int = 0


class ScheduleManager:
    """Session-aware scheduler for engine actions.

    Determines which actions need to run based on the current ET time,
    what has already been completed, and configured intervals.

    Thread-safe: all state is read/written from a single engine thread.
    """

    # Recurring interval configuration (seconds)
    FKS_INTERVAL = 5 * 60  # 5 minutes during active
    GROK_INTERVAL = 15 * 60  # 15 minutes during active
    RISK_CHECK_INTERVAL = 60  # 1 minute during active
    NO_TRADE_INTERVAL = 2 * 60  # 2 minutes during active
    ORB_CHECK_INTERVAL = 2 * 60  # 2 minutes during active (09:30–11:00 window)
    FOCUS_PUBLISH_INTERVAL = 30  # 30 seconds during active (throttled downstream)
    STATUS_PUBLISH_INTERVAL = 10  # 10 seconds always

    # Sleep intervals per session (how long the main loop sleeps between cycles)
    SLEEP_PRE_MARKET = 30.0  # check every 30s during pre-market
    SLEEP_ACTIVE = 10.0  # check every 10s during active hours
    SLEEP_OFF_HOURS = 60.0  # check every 60s during off-hours

    def __init__(self) -> None:
        self._trackers: dict[ActionType, _ActionTracker] = {
            action: _ActionTracker() for action in ActionType
        }
        self._current_session: Optional[SessionMode] = None
        self._session_started_at: Optional[float] = None
        self._today: Optional[date] = None
        self._session_transition_logged: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_session_mode(now: Optional[datetime] = None) -> SessionMode:
        """Determine current trading session based on ET time."""
        if now is None:
            now = datetime.now(tz=_EST)
        hour = now.hour
        if 0 <= hour < 5:
            return SessionMode.PRE_MARKET
        elif 5 <= hour < 12:
            return SessionMode.ACTIVE
        else:
            return SessionMode.OFF_HOURS

    @property
    def current_session(self) -> SessionMode:
        """Return the current session mode (cached, updated each cycle)."""
        return self._current_session or self.get_session_mode()

    @property
    def sleep_interval(self) -> float:
        """How long the main loop should sleep between scheduler cycles."""
        session = self.current_session
        if session == SessionMode.PRE_MARKET:
            return self.SLEEP_PRE_MARKET
        elif session == SessionMode.ACTIVE:
            return self.SLEEP_ACTIVE
        else:
            return self.SLEEP_OFF_HOURS

    def get_pending_actions(
        self,
        now: Optional[datetime] = None,
    ) -> list[ScheduledAction]:
        """Return ordered list of actions that should run right now.

        Call this each engine loop iteration. It handles:
          - Session transitions (resets per-session trackers)
          - Day transitions (resets per-day trackers)
          - Interval-based recurring actions
          - Once-per-session and once-per-day actions

        Returns actions sorted by priority (lowest number = highest priority).
        """
        if now is None:
            now = datetime.now(tz=_EST)
        ts = time.monotonic()

        session = self.get_session_mode(now)
        today = now.date()

        # Handle day transition
        if self._today != today:
            self._on_day_change(today)

        # Handle session transition
        if self._current_session != session:
            self._on_session_change(session, now)

        self._current_session = session

        # Gather pending actions based on session
        pending: list[ScheduledAction] = []

        if session == SessionMode.PRE_MARKET:
            pending.extend(self._get_pre_market_actions(ts, today))
        elif session == SessionMode.ACTIVE:
            pending.extend(self._get_active_actions(ts, now))
        else:
            pending.extend(self._get_off_hours_actions(ts))

        # Sort by priority
        pending.sort(key=lambda a: a.priority)
        return pending

    def mark_done(self, action: ActionType, now: Optional[datetime] = None) -> None:
        """Mark an action as completed. Called after successful execution.

        Parameters
        ----------
        action : ActionType
            The action that completed.
        now : datetime, optional
            Override the current time (used by tests).  When *None* the
            real wall-clock time is used.
        """
        if now is None:
            now = datetime.now(tz=_EST)
        tracker = self._trackers[action]
        tracker.last_run = time.monotonic()
        tracker.last_run_date = now.date()
        tracker.last_run_session = (
            self._current_session.value if self._current_session else None
        )
        tracker.run_count_today += 1
        logger.debug(
            "Action completed: %s (run #%d today)",
            action.value,
            tracker.run_count_today,
        )

    def mark_failed(self, action: ActionType, error: str) -> None:
        """Mark an action as failed. It will be retried on the next cycle."""
        logger.warning("Action failed: %s — %s", action.value, error)
        # Don't update last_run so it gets retried

    def get_status(self, now: Optional[datetime] = None) -> dict:
        """Return scheduler status for health/monitoring.

        Parameters
        ----------
        now : datetime, optional
            Override the current time (used by tests).
        """
        if now is None:
            now = datetime.now(tz=_EST)
        session = self.get_session_mode(now)

        action_statuses = {}
        for action, tracker in self._trackers.items():
            action_statuses[action.value] = {
                "last_run": tracker.last_run_date.isoformat()
                if tracker.last_run_date
                else None,
                "run_count_today": tracker.run_count_today,
                "last_session": tracker.last_run_session,
            }

        return {
            "session_mode": session.value,
            "session_emoji": self._session_emoji(session),
            "current_time_et": now.strftime("%H:%M:%S"),
            "sleep_interval": self.sleep_interval,
            "actions": action_statuses,
        }

    def time_until_next_session(
        self, now: Optional[datetime] = None
    ) -> tuple[SessionMode, float]:
        """Return the next session and seconds until it starts."""
        if now is None:
            now = datetime.now(tz=_EST)
        hour = now.hour

        if 0 <= hour < 5:
            # In pre-market, next is active at 05:00
            next_session = SessionMode.ACTIVE
            target_hour = 5
        elif 5 <= hour < 12:
            # In active, next is off-hours at 12:00
            next_session = SessionMode.OFF_HOURS
            target_hour = 12
        else:
            # In off-hours, next is pre-market at 00:00 (next day)
            next_session = SessionMode.PRE_MARKET
            target_hour = 24  # midnight

        # Calculate seconds until target hour
        seconds_remaining = (target_hour - hour - 1) * 3600
        seconds_remaining += (60 - now.minute - 1) * 60
        seconds_remaining += 60 - now.second
        # Clamp to non-negative
        seconds_remaining = max(0, seconds_remaining)

        return next_session, seconds_remaining

    # ------------------------------------------------------------------
    # Internal: per-session action generators
    # ------------------------------------------------------------------

    def _get_pre_market_actions(
        self,
        ts: float,
        today: date,
    ) -> list[ScheduledAction]:
        """Pre-market (00:00–05:00 ET): focus computation + morning prep."""
        actions: list[ScheduledAction] = []

        # Daily focus — run once per day
        if not self._ran_today(ActionType.COMPUTE_DAILY_FOCUS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.COMPUTE_DAILY_FOCUS,
                    priority=0,
                    description="Compute daily focus for today's trading plan",
                )
            )

        # Grok morning briefing — run once per day
        if not self._ran_today(ActionType.GROK_MORNING_BRIEF, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.GROK_MORNING_BRIEF,
                    priority=1,
                    description="Generate Grok AI morning market briefing",
                )
            )

        # Prep alerts — run once per day
        if not self._ran_today(ActionType.PREP_ALERTS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.PREP_ALERTS,
                    priority=2,
                    description="Prepare alert thresholds for active session",
                )
            )

        return actions

    def _get_active_actions(
        self,
        ts: float,
        now: datetime,
    ) -> list[ScheduledAction]:
        """Active (05:00–12:00 ET): live recomputation + updates."""
        actions: list[ScheduledAction] = []
        today = now.date()

        # Daily focus — also compute at start of active if missed in pre-market
        if not self._ran_today(ActionType.COMPUTE_DAILY_FOCUS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.COMPUTE_DAILY_FOCUS,
                    priority=0,
                    description="Compute daily focus (catch-up — missed pre-market)",
                )
            )

        # FKS recomputation — every 5 minutes
        if self._interval_elapsed(ActionType.FKS_RECOMPUTE, ts, self.FKS_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.FKS_RECOMPUTE,
                    priority=1,
                    description="Recompute FKS wave/vol/quality for all assets",
                )
            )

        # Publish focus update to Redis — every 30 seconds
        if self._interval_elapsed(
            ActionType.PUBLISH_FOCUS_UPDATE, ts, self.FOCUS_PUBLISH_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.PUBLISH_FOCUS_UPDATE,
                    priority=2,
                    description="Publish focus update to Redis for SSE",
                )
            )

        # Risk rules check — every 1 minute
        if self._interval_elapsed(
            ActionType.CHECK_RISK_RULES, ts, self.RISK_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_RISK_RULES,
                    priority=3,
                    description="Check risk rules (position limits, daily loss, time)",
                )
            )

        # No-trade detector — every 2 minutes
        if self._interval_elapsed(
            ActionType.CHECK_NO_TRADE, ts, self.NO_TRADE_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_NO_TRADE,
                    priority=4,
                    description="Check should-not-trade conditions",
                )
            )

        # Grok live update — every 15 minutes
        if self._interval_elapsed(ActionType.GROK_LIVE_UPDATE, ts, self.GROK_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.GROK_LIVE_UPDATE,
                    priority=5,
                    description="Run Grok 15-minute live market update",
                )
            )

        # Opening Range Breakout check — every 2 minutes during 09:30–11:00 ET
        now_time = now.time()
        from datetime import time as _dt_time

        _orb_start = _dt_time(9, 30)
        _orb_end = _dt_time(11, 0)
        if _orb_start <= now_time <= _orb_end:
            if self._interval_elapsed(
                ActionType.CHECK_ORB, ts, self.ORB_CHECK_INTERVAL
            ):
                actions.append(
                    ScheduledAction(
                        action=ActionType.CHECK_ORB,
                        priority=6,
                        description="Check Opening Range Breakout (09:30–10:00 OR)",
                    )
                )

        return actions

    def _get_off_hours_actions(self, ts: float) -> list[ScheduledAction]:
        """Off-hours (12:00–00:00 ET): backfill, optimize, backtest."""
        actions: list[ScheduledAction] = []
        session = SessionMode.OFF_HOURS.value

        # Historical backfill — once per off-hours session
        if not self._ran_this_session(ActionType.HISTORICAL_BACKFILL, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.HISTORICAL_BACKFILL,
                    priority=0,
                    description="Backfill historical 1-min bars to Postgres",
                )
            )

        # Optimization — once per off-hours session
        if not self._ran_this_session(ActionType.RUN_OPTIMIZATION, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.RUN_OPTIMIZATION,
                    priority=1,
                    description="Run Optuna strategy optimization",
                )
            )

        # Backtesting — once per off-hours session
        if not self._ran_this_session(ActionType.RUN_BACKTEST, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.RUN_BACKTEST,
                    priority=2,
                    description="Run walk-forward backtesting",
                )
            )

        # Next-day prep — once per off-hours session
        if not self._ran_this_session(ActionType.NEXT_DAY_PREP, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.NEXT_DAY_PREP,
                    priority=3,
                    description="Prepare next trading day parameters",
                )
            )

        return actions

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _ran_today(self, action: ActionType, today: date) -> bool:
        """Check if action has already run today."""
        return self._trackers[action].last_run_date == today

    def _ran_this_session(
        self, action: ActionType, session_value: str, today: Optional[date] = None
    ) -> bool:
        """Check if action has already run during this session instance."""
        tracker = self._trackers[action]
        if tracker.last_run_session != session_value:
            return False
        # Also ensure it ran today (not a stale session marker from yesterday)
        if today is None:
            today = datetime.now(tz=_EST).date()
        return tracker.last_run_date == today

    def _interval_elapsed(
        self,
        action: ActionType,
        now_ts: float,
        interval_seconds: float,
    ) -> bool:
        """Check if enough time has passed since the last run."""
        tracker = self._trackers[action]
        if tracker.last_run is None:
            return True  # never run → due immediately
        return (now_ts - tracker.last_run) >= interval_seconds

    def _on_day_change(self, today: date) -> None:
        """Reset daily counters on day transition."""
        logger.info("=" * 50)
        logger.info("  Day change detected: %s", today.isoformat())
        logger.info("=" * 50)
        self._today = today
        for tracker in self._trackers.values():
            tracker.run_count_today = 0

    def _on_session_change(self, new_session: SessionMode, now: datetime) -> None:
        """Handle session transition."""
        old = self._current_session
        logger.info("=" * 50)
        logger.info(
            "  Session transition: %s → %s %s at %s ET",
            old.value if old else "INIT",
            new_session.value,
            self._session_emoji(new_session),
            now.strftime("%H:%M:%S"),
        )
        logger.info("=" * 50)
        self._session_started_at = time.monotonic()
        self._session_transition_logged = True

    @staticmethod
    def _session_emoji(session: SessionMode) -> str:
        return {
            SessionMode.PRE_MARKET: "🌙",
            SessionMode.ACTIVE: "🟢",
            SessionMode.OFF_HOURS: "⚙️",
        }.get(session, "❓")
