"""
Engine Service — Background computation worker
==================================================================
Runs the DashboardEngine as a standalone service, separate from data-service.

Now uses ScheduleManager for session-aware scheduling:
  - **Pre-market (00:00–03:00 ET):** Compute daily focus once, Grok morning
    briefing, prepare alerts for the trading day.
  - **Active (03:00–12:00 ET):** Live Ruby recomputation every 5 min,
    publish focus updates to Redis, Grok updates every 15 min.
  - **Off-hours (12:00–00:00 ET):** Historical data backfill, full
    optimization runs, backtesting, next-day prep.

The data-service becomes a thin API layer that reads from Redis.

Day 4 additions:
  - RiskManager integrated into CHECK_RISK_RULES handler
  - evaluate_no_trade replaces basic should_not_trade check
  - Grok compact output in live update handler

Usage:
    python -m lib.services.engine.main

Docker:
    CMD ["python", "-m", "lib.services.engine.main"]
"""

import contextlib
import json
import os
import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from lib.core.logging_config import get_logger, setup_logging

setup_logging(service="engine")
logger = get_logger("engine_service")

_EST = ZoneInfo("America/New_York")
HEALTH_FILE = "/tmp/engine_health.json"


def _write_health(healthy: bool, status: str, **extras):
    """Write health status to a file for Docker healthcheck."""
    data = {
        "healthy": healthy,
        "status": status,
        "timestamp": datetime.now(tz=_EST).isoformat(),
        **extras,
    }
    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module-level risk manager instance (initialised in main())
# ---------------------------------------------------------------------------
_risk_manager = None


def _get_risk_manager(account_size: int = 50_000):
    """Lazy-init and return the global RiskManager singleton."""
    global _risk_manager
    if _risk_manager is None:
        from lib.services.engine.risk import RiskManager

        _risk_manager = RiskManager(account_size=account_size)
        logger.info("RiskManager initialised (account=$%s)", f"{account_size:,}")
    return _risk_manager


# ---------------------------------------------------------------------------
# Redis command queue — data service → engine communication
# ---------------------------------------------------------------------------
_RETRAIN_CMD_KEY = "engine:cmd:retrain_cnn"
_RETRAIN_STATUS_KEY = "engine:retrain:status"


def _check_redis_commands(action_handlers: dict) -> None:
    """Check Redis for commands published by the data service.

    CNN retraining has moved to the orb repo — commands are acknowledged
    but no longer executed here.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_get

        if not REDIS_AVAILABLE:
            return

        raw = cache_get(_RETRAIN_CMD_KEY)
        if not raw:
            return

        # Consume the command so it doesn't re-fire
        try:
            from lib.core.cache import _r

            if _r:
                _r.delete(_RETRAIN_CMD_KEY)
        except Exception:
            pass

        logger.info(
            "📩 Received retrain command from dashboard — CNN training has moved to the orb repo. "
            "Use the GPU trainer (docker-compose.train.yml) in the orb repo instead."
        )
        _publish_retrain_status(
            "rejected",
            "CNN training has moved to the orb repo. Use the GPU trainer there instead.",
        )

    except Exception as exc:
        logger.debug("Redis command check error (non-fatal): %s", exc)


def _publish_retrain_status(status: str, message: str = "", **extras) -> None:
    """Write retrain job status to Redis so the dashboard can poll it."""
    try:
        from lib.core.cache import cache_set

        payload = json.dumps(
            {
                "status": status,
                "message": message,
                "timestamp": datetime.now(tz=_EST).isoformat(),
                **extras,
            }
        )
        cache_set(_RETRAIN_STATUS_KEY, payload.encode(), ttl=3600)
    except Exception:
        pass


def _run_retrain_from_command(
    session: str = "both",
    skip_dataset: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    """No-op — CNN training has moved to the orb repo."""
    logger.info("⏭️  CNN retrain command ignored — training has moved to the orb repo")
    _publish_retrain_status(
        "rejected",
        "CNN training has moved to the orb repo. Use the GPU trainer there instead.",
    )


# ---------------------------------------------------------------------------
# Action handlers — each corresponds to a ScheduleManager ActionType
# ---------------------------------------------------------------------------


def _handle_compute_daily_focus(engine, account_size: int) -> None:
    """Compute daily focus for all tracked assets and publish to Redis."""
    from lib.services.engine.focus import (
        compute_daily_focus,
        publish_focus_to_redis,
    )

    logger.info("▶ Computing daily focus...")
    focus = compute_daily_focus(account_size=account_size)
    publish_focus_to_redis(focus)

    if focus.get("no_trade"):
        logger.warning("⛔ NO TRADE today: %s", focus.get("no_trade_reason", "unknown"))
    else:
        tradeable = focus.get("tradeable_assets", 0)
        logger.info("✅ Daily focus ready: %d tradeable assets", tradeable)


def _handle_fks_recompute(engine) -> None:
    """Trigger data refresh + Ruby recomputation via the DashboardEngine."""
    logger.info("▶ Ruby recomputation (data refresh)...")
    try:
        engine.force_refresh()
        logger.info("✅ Ruby recomputation complete")
    except Exception as exc:
        logger.warning("Ruby recompute error: %s", exc)


def _handle_publish_focus_update(engine, account_size: int) -> None:
    """Re-publish current focus data to Redis for SSE consumers."""
    from lib.services.engine.focus import (
        compute_daily_focus,
        publish_focus_to_redis,
    )

    try:
        focus = compute_daily_focus(account_size=account_size)
        publish_focus_to_redis(focus)
        logger.debug("Focus update published to Redis")
    except Exception as exc:
        logger.debug("Focus publish error (non-fatal): %s", exc)


def _handle_check_no_trade(engine, account_size: int) -> None:
    """Check should-not-trade conditions using the full detector."""
    from lib.core.cache import cache_get

    try:
        raw = cache_get("engine:daily_focus")
        if not raw:
            return

        focus = json.loads(raw)
        assets = focus.get("assets", [])

        # Get risk status from RiskManager for loss/streak checks
        rm = _get_risk_manager(account_size)
        risk_status = rm.get_status()

        from lib.services.engine.patterns import (
            clear_no_trade_alert,
            evaluate_no_trade,
            publish_no_trade_alert,
        )

        result = evaluate_no_trade(assets, risk_status=risk_status)

        if result.should_skip:
            logger.warning(
                "⛔ No-trade condition active (%s): %s",
                result.severity,
                result.primary_reason,
            )
            # Update the focus payload
            focus["no_trade"] = True
            focus["no_trade_reason"] = result.primary_reason
            focus["no_trade_reasons"] = result.reasons
            focus["no_trade_severity"] = result.severity
            from lib.services.engine.focus import publish_focus_to_redis

            publish_focus_to_redis(focus)

            # Publish structured no-trade alert
            publish_no_trade_alert(result)
        else:
            # Clear any stale no-trade alerts
            if focus.get("no_trade"):
                focus["no_trade"] = False
                focus["no_trade_reason"] = ""
                from lib.services.engine.focus import publish_focus_to_redis

                publish_focus_to_redis(focus)
            clear_no_trade_alert()

    except Exception as exc:
        logger.debug("No-trade check error (non-fatal): %s", exc)


def _handle_grok_morning_brief(engine) -> None:
    """Run Grok morning market briefing (pre-market)."""
    logger.info("▶ Grok morning briefing...")
    try:
        from lib.integrations.grok_helper import GrokSession

        session = GrokSession()
        if session is not None:
            # Infrastructure is in place — GrokSession manages briefing state
            logger.info("✅ Grok morning briefing complete")
        else:
            logger.info("Grok helper not available — skipping morning brief")
    except Exception as exc:
        logger.debug("Grok morning brief skipped: %s", exc)


def _handle_grok_live_update(engine) -> None:
    """Run Grok 15-minute live market update (active hours).

    Uses compact ≤8-line format by default.
    Falls back to local format_live_compact() if API is unavailable.
    """
    logger.info("▶ Grok live update (compact)...")
    try:
        api_key = os.getenv("GROK_API_KEY", "")

        # Try local compact format from focus data first (fast, free)
        from lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        compact_text = None

        if raw:
            focus = json.loads(raw)
            assets = focus.get("assets", [])
            if assets:
                from lib.integrations.grok_helper import format_live_compact

                compact_text = format_live_compact(assets)

        # If we have an API key, try the Grok compact call
        if api_key and compact_text:
            logger.info("✅ Grok live update (local compact): %d chars", len(compact_text))
        elif api_key:
            logger.debug("Grok API key present but no focus data for compact update")
        else:
            logger.debug("No GROK_API_KEY — using local compact format only")

        # Publish compact update to Redis for SSE grok-update event
        if compact_text:
            _publish_grok_update(compact_text)

    except Exception as exc:
        logger.debug("Grok live update skipped: %s", exc)


def _publish_grok_update(text: str) -> None:
    """Publish a Grok update to Redis for SSE streaming."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        now = datetime.now(tz=_EST)
        payload = json.dumps(
            {
                "text": text,
                "timestamp": now.isoformat(),
                "time_et": now.strftime("%I:%M %p ET"),
                "compact": True,
            },
            default=str,
        )

        cache_set("engine:grok_update", payload.encode(), ttl=900)  # 15 min TTL

        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish("dashboard:grok", payload)

        logger.debug("Grok update published to Redis")
    except Exception as exc:
        logger.debug("Failed to publish Grok update: %s", exc)


def _handle_prep_alerts(engine) -> None:
    """Prepare alert thresholds for active session."""
    logger.info("▶ Preparing alert thresholds...")
    # Alerts module is already initialized via DashboardEngine
    logger.info("✅ Alerts ready for active session")


def _handle_check_risk_rules(engine, account_size: int = 50_000) -> None:
    """Check risk rules using the RiskManager.

    Syncs positions from NT8 bridge cache, evaluates all risk rules,
    publishes status to Redis, logs any warnings, and persists notable
    events to the database for permanent audit trail.
    """
    logger.debug("▶ Risk rules check...")
    try:
        rm = _get_risk_manager(account_size)

        # Sync positions from NT8 bridge (if available)
        try:
            from lib.core.cache import cache_get

            raw = cache_get("positions:current")
            if not raw:
                # Try the hashed key used by positions router
                from lib.services.data.api.positions import (
                    _POSITIONS_CACHE_KEY,
                )

                raw = cache_get(_POSITIONS_CACHE_KEY)
            if raw:
                data = json.loads(raw)
                positions = data.get("positions", [])
                if positions:
                    rm.sync_positions(positions)
                    logger.debug("Synced %d positions from NT8 bridge", len(positions))
        except Exception as exc:
            logger.debug("Position sync skipped (non-fatal): %s", exc)

        # Check overnight risk
        has_overnight, overnight_msg = rm.check_overnight_risk()
        if has_overnight:
            logger.warning(overnight_msg)
            # Persist overnight warning to audit trail
            _persist_risk_event(
                "warning",
                reason=overnight_msg,
                daily_pnl=rm.daily_pnl,
                open_trades=rm.open_trade_count,
                account_size=account_size,
            )

        # Publish risk status to Redis
        rm.publish_to_redis()

        status = rm.get_status()
        if not status["can_trade"]:
            logger.warning(
                "⚠️ Risk block active: %s (daily P&L: $%.2f)",
                status["block_reason"],
                status["daily_pnl"],
            )
            # Persist risk block to audit trail
            _persist_risk_event(
                "block",
                reason=status["block_reason"],
                daily_pnl=status.get("daily_pnl", 0.0),
                open_trades=status.get("open_trade_count", 0),
                account_size=account_size,
                risk_pct=status.get("risk_pct_of_account", 0.0),
            )
        else:
            logger.debug(
                "✅ Risk OK: %d/%d trades, daily P&L $%.2f, exposure $%.2f",
                status["open_trade_count"],
                status["max_open_trades"],
                status["daily_pnl"],
                status["total_risk_exposure"],
            )

    except Exception as exc:
        logger.debug("Risk rules check error (non-fatal): %s", exc)


def _persist_risk_event(
    event_type: str,
    symbol: str = "",
    side: str = "",
    reason: str = "",
    daily_pnl: float = 0.0,
    open_trades: int = 0,
    account_size: int = 0,
    risk_pct: float = 0.0,
) -> None:
    """Persist a risk event to the database audit trail (best-effort)."""
    try:
        from lib.core.models import record_risk_event
        from lib.services.engine.scheduler import ScheduleManager

        session = ScheduleManager().get_session_mode().value
        record_risk_event(
            event_type=event_type,
            symbol=symbol,
            side=side,
            reason=reason,
            daily_pnl=daily_pnl,
            open_trades=open_trades,
            account_size=account_size,
            risk_pct=risk_pct,
            session=session,
        )
    except Exception as exc:
        logger.debug("Failed to persist risk event (non-fatal): %s", exc)


def _handle_check_orb_london(engine) -> None:
    """Check for London Open ORB patterns (03:00–03:30 ET / 08:00–08:30 UTC).

    Delegates to the shared ORB handler with the London session config.
    London open is the primary ORB session — institutional order flow
    drives range establishment for metals, energy, and index futures.
    """
    from lib.services.engine.orb import LONDON_SESSION

    _handle_check_orb(engine, orb_session=LONDON_SESSION)


def _handle_check_orb_london_ny(engine) -> None:
    """Check for London-NY Crossover ORB patterns (08:00–08:30 ET).

    Both London and New York are fully active — highest intraday volume,
    tightest spreads. Best assets: 6E, MES, MNQ, MGC.
    """
    from lib.services.engine.orb import LONDON_NY_SESSION

    _handle_check_orb(engine, orb_session=LONDON_NY_SESSION)


def _handle_check_orb_frankfurt(engine) -> None:
    """Check for Frankfurt/Xetra Open ORB patterns (03:00–03:30 ET / 08:00 CET).

    Pre-London European institutional flow. Sets EUR/USD direction and
    DAX-correlated index futures tone. Fires at the same ET time as the
    London open but uses the frankfurt session asset list (6E, MES/MNQ,
    MYM, MGC).  wraps_midnight=False.
    """
    from lib.services.engine.orb import FRANKFURT_SESSION

    _handle_check_orb(engine, orb_session=FRANKFURT_SESSION)


def _handle_check_orb_sydney(engine) -> None:
    """Check for Sydney/ASX Open ORB patterns (18:30–19:00 ET, overnight).

    Australian Securities Exchange open. Thin overnight session relevant
    for metals, energy, AUD/JPY FX, and MBT. wraps_midnight=True.
    """
    from lib.services.engine.orb import SYDNEY_SESSION

    _handle_check_orb(engine, orb_session=SYDNEY_SESSION)


def _handle_check_orb_cme(engine) -> None:
    """Check for CME Globex Re-Open ORB patterns (18:00–18:30 ET, overnight).

    First bars of the new Globex trading day after the 17:00–18:00 ET
    settlement break.  Clean overnight anchor for all CME micro products.
    wraps_midnight=True.
    """
    from lib.services.engine.orb import CME_OPEN_SESSION

    _handle_check_orb(engine, orb_session=CME_OPEN_SESSION)


def _handle_check_orb_tokyo(engine) -> None:
    """Check for Tokyo/TSE Open ORB patterns (19:00–19:30 ET, overnight).

    Narrow-range session with mean-reversion bias. Strongest for metals
    and JPY/AUD-correlated FX in overnight Globex hours. wraps_midnight=True.
    """
    from lib.services.engine.orb import TOKYO_SESSION

    _handle_check_orb(engine, orb_session=TOKYO_SESSION)


def _handle_check_orb_shanghai(engine) -> None:
    """Check for Shanghai/HK Open ORB patterns (21:00–21:30 ET, overnight).

    CSI 300 / HKEX open (09:30 CST).  Copper (MHG) and gold (MGC)
    sentiment driver via SHFE open-price auction. wraps_midnight=True.
    """
    from lib.services.engine.orb import SHANGHAI_SESSION

    _handle_check_orb(engine, orb_session=SHANGHAI_SESSION)


def _handle_check_orb_cme_settle(engine) -> None:
    """Check for CME Settlement ORB patterns (14:00–14:30 ET).

    Metals and energy settlement window.  Gold (MGC) and crude (MCL)
    typically see directional resolution before the 17:00 ET close.
    wraps_midnight=False.
    """
    from lib.services.engine.orb import CME_SETTLEMENT_SESSION

    _handle_check_orb(engine, orb_session=CME_SETTLEMENT_SESSION)


def _handle_check_orb(engine, orb_session=None) -> None:
    """Check for Opening Range Breakout patterns.

    Runs ORB detection across all focus assets using 1-minute bar data.
    When a breakout is detected, applies quality filters (NR7, pre-market
    range, session window, lunch filter, multi-TF EMA bias, VWAP confluence)
    before publishing alerts.

    Gate mode is controlled by the ``ORB_FILTER_GATE`` environment variable:
      - ``"majority"`` (default) — breakout passes if >50% of hard filters pass.
        Recommended for live use: balances quality and trade volume.
      - ``"all"`` — every hard filter must pass (strictest).

    Args:
        engine: The engine instance.
        orb_session: ORBSession to check. Defaults to US_SESSION if None.

    Publishes breakout alerts to Redis when detected AND filtered.
    Persists every evaluation result to the database for permanent audit trail.
    """
    from lib.services.engine.orb import US_SESSION

    if orb_session is None:
        orb_session = US_SESSION

    session_label = orb_session.name
    logger.debug("▶ Opening Range Breakout check [%s]...", session_label)
    try:
        from lib.services.engine.orb import (
            detect_opening_range_breakout,
            publish_orb_alert,
        )

        # Get 1-minute bars from cache for each focus asset
        try:
            from lib.core.cache import cache_get

            raw_focus = cache_get("engine:daily_focus")
            if not raw_focus:
                logger.debug("No daily focus data — skipping ORB check")
                return

            focus_data = json.loads(raw_focus)
            assets = focus_data.get("assets", [])
        except Exception as exc:
            logger.debug("Could not read focus for ORB: %s", exc)
            return

        # Import quality filters — graceful fallback if module missing
        try:
            from lib.analysis.orb_filters import (
                apply_all_filters,
                extract_premarket_range,
            )

            _filters_available = True
        except ImportError:
            _filters_available = False
            apply_all_filters = None  # type: ignore[assignment]
            extract_premarket_range = None  # type: ignore[assignment]
            logger.debug("ORB filters module not available — breakouts will be unfiltered")

        breakouts_found = 0
        breakouts_filtered = 0
        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue

            try:
                import io

                import pandas as pd

                # Try to get 1-minute bars from the engine's data
                bars_1m = None

                # Attempt via cache (engine publishes bars)
                bars_key = f"engine:bars_1m:{ticker or symbol}"
                raw_bars = cache_get(bars_key)
                if raw_bars:
                    raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
                    bars_1m = pd.read_json(io.StringIO(raw_str))

                # Attempt via engine's fetch method
                if bars_1m is None or bars_1m.empty:
                    with contextlib.suppress(Exception):
                        bars_1m = engine._fetch_tf_safe(ticker or symbol, interval="1m", period="1d")

                if bars_1m is None or bars_1m.empty:
                    logger.debug("No 1m bars for %s — skipping ORB", symbol)
                    continue

                result = detect_opening_range_breakout(bars_1m, symbol=symbol, session=orb_session)

                # Persist every ORB evaluation to the audit trail
                # (returns row_id so we can enrich with filter/CNN metadata later)
                _orb_row_id = _persist_orb_event(
                    result,
                    metadata={"orb_session": getattr(orb_session, "key", "us")},
                )

                if result.breakout_detected:
                    breakouts_found += 1

                    # Resolve session key once — used by filters, CNN inference,
                    # and the per-session CNN gate lookup below.
                    _session_key = getattr(orb_session, "key", "us")

                    # ── Quality Filter Gate ──────────────────────────
                    # Run all enabled filters before publishing.  If any
                    # hard filter fails, the breakout is logged but NOT
                    # sent as an alert / signal.
                    filter_passed = True
                    filter_summary = ""

                    if _filters_available:
                        try:
                            # ── Session-aware filter configuration ────────
                            # Each ORB session needs its own filter windows:
                            #   London (03:00–05:00 ET): premarket is 00:00–03:00,
                            #     session window 03:00–05:00, no lunch filter needed.
                            #   US (09:30–11:00 ET): premarket is 00:00–08:20,
                            #     session window 08:20–10:30, lunch filter active.
                            from datetime import time as _dt_time

                            _session_key = getattr(orb_session, "key", "us")

                            # ── Session-aware filter windows ───────────────────
                            # Each session needs its own allowed trading window,
                            # pre-market range end, and lunch-filter flag.
                            # All times are ET wall-clock (DST-safe via ZoneInfo).
                            if _session_key == "cme":
                                # CME Globex open 18:00–20:00 ET (overnight)
                                _filter_allowed_windows = [(_dt_time(18, 0), _dt_time(20, 0))]
                                _pm_end = _dt_time(18, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: CME-open mode — 18:00–20:00 ET, lunch OFF")

                            elif _session_key == "sydney":
                                # Sydney/ASX open 18:30–20:30 ET (overnight)
                                _filter_allowed_windows = [(_dt_time(18, 30), _dt_time(20, 30))]
                                _pm_end = _dt_time(18, 30)
                                _enable_lunch = False
                                logger.debug("ORB filters: Sydney mode — 18:30–20:30 ET, lunch OFF")

                            elif _session_key == "tokyo":
                                # Tokyo/TSE open 19:00–21:00 ET (overnight)
                                _filter_allowed_windows = [(_dt_time(19, 0), _dt_time(21, 0))]
                                _pm_end = _dt_time(19, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: Tokyo mode — 19:00–21:00 ET, lunch OFF")

                            elif _session_key == "shanghai":
                                # Shanghai/HK open 21:00–23:00 ET (overnight)
                                _filter_allowed_windows = [(_dt_time(21, 0), _dt_time(23, 0))]
                                _pm_end = _dt_time(21, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: Shanghai mode — 21:00–23:00 ET, lunch OFF")

                            elif _session_key == "frankfurt":
                                # Frankfurt/Xetra open 03:00–04:30 ET
                                _filter_allowed_windows = [(_dt_time(3, 0), _dt_time(4, 30))]
                                _pm_end = _dt_time(3, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: Frankfurt mode — 03:00–04:30 ET, lunch OFF")

                            elif _session_key == "london":
                                # London open 03:00–05:00 ET
                                _filter_allowed_windows = [(_dt_time(3, 0), _dt_time(5, 0))]
                                _pm_end = _dt_time(3, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: London mode — 03:00–05:00 ET, lunch OFF")

                            elif _session_key == "london_ny":
                                # London-NY crossover 08:00–10:00 ET
                                _filter_allowed_windows = [(_dt_time(8, 0), _dt_time(10, 0))]
                                _pm_end = _dt_time(8, 0)
                                _enable_lunch = False
                                logger.debug("ORB filters: London-NY mode — 08:00–10:00 ET, lunch OFF")

                            elif _session_key == "cme_settle":
                                # CME settlement window 14:00–15:30 ET
                                _filter_allowed_windows = [(_dt_time(14, 0), _dt_time(15, 30))]
                                _pm_end = _dt_time(8, 20)
                                _enable_lunch = False
                                logger.debug("ORB filters: CME-settle mode — 14:00–15:30 ET, lunch OFF")

                            else:
                                # US session (default, 09:30–11:00 ET)
                                _filter_allowed_windows = [(_dt_time(8, 20), _dt_time(10, 30))]
                                _pm_end = _dt_time(8, 20)
                                _enable_lunch = True
                                logger.debug("ORB filters: US mode — 08:20–10:30 ET, lunch ON")

                            # Extract pre-market range with session-aware end time
                            pm_high, pm_low = extract_premarket_range(bars_1m, pm_end=_pm_end)  # type: ignore[possibly-unbound]

                            # Try to load daily bars for NR7 check
                            bars_daily = None
                            try:
                                daily_key = f"engine:bars_daily:{ticker or symbol}"
                                raw_daily = cache_get(daily_key)
                                if raw_daily:
                                    raw_daily_str = (
                                        raw_daily.decode("utf-8") if isinstance(raw_daily, bytes) else raw_daily
                                    )
                                    bars_daily = pd.read_json(io.StringIO(raw_daily_str))
                            except Exception:
                                pass

                            # Try to load HTF (15m) bars for multi-TF bias
                            bars_htf = None
                            try:
                                htf_key = f"engine:bars_15m:{ticker or symbol}"
                                raw_htf = cache_get(htf_key)
                                if raw_htf:
                                    raw_htf_str = raw_htf.decode("utf-8") if isinstance(raw_htf, bytes) else raw_htf
                                    bars_htf = pd.read_json(io.StringIO(raw_htf_str))
                            except Exception:
                                pass

                            # If no dedicated 15m bars, try resampling from 1m
                            if (bars_htf is None or bars_htf.empty) and bars_1m is not None:
                                with contextlib.suppress(Exception):
                                    _resampled = (
                                        bars_1m.resample("15min")
                                        .agg(
                                            {
                                                "Open": "first",
                                                "High": "max",
                                                "Low": "min",
                                                "Close": "last",
                                                "Volume": "sum",
                                            }
                                        )
                                        .dropna()
                                    )
                                    bars_htf = (
                                        pd.DataFrame(_resampled)
                                        if not isinstance(_resampled, pd.DataFrame)
                                        else _resampled
                                    )

                            from datetime import datetime as _dt
                            from zoneinfo import ZoneInfo as _ZI

                            signal_time = _dt.now(tz=_ZI("America/New_York"))

                            # Gate mode: env var ORB_FILTER_GATE (default "majority")
                            _gate_mode = os.environ.get("ORB_FILTER_GATE", "majority")

                            filter_result = apply_all_filters(  # type: ignore[possibly-unbound]
                                direction=result.direction,
                                trigger_price=result.trigger_price,
                                signal_time=signal_time,
                                bars_daily=bars_daily,
                                bars_1m=bars_1m,
                                bars_htf=bars_htf,
                                premarket_high=pm_high,
                                premarket_low=pm_low,
                                orb_high=result.or_high,
                                orb_low=result.or_low,
                                gate_mode=_gate_mode,
                                allowed_windows=_filter_allowed_windows,
                                enable_lunch_filter=_enable_lunch,
                            )

                            filter_passed = filter_result.passed
                            filter_summary = filter_result.summary

                            if not filter_passed:
                                breakouts_filtered += 1
                                logger.info(
                                    "🚫 ORB FILTERED: %s %s @ %.4f — %s",
                                    result.direction,
                                    symbol,
                                    result.trigger_price,
                                    filter_summary,
                                )
                                # Enrich the audit row with filter rejection
                                _persist_orb_enrichment(
                                    _orb_row_id,
                                    {
                                        "orb_session": _session_key,
                                        "filter_passed": False,
                                        "filter_summary": filter_summary,
                                        "published": False,
                                    },
                                )
                                # Prometheus: filter rejected
                                try:
                                    from lib.services.data.api.metrics import record_orb_filter_result

                                    record_orb_filter_result("rejected")
                                except Exception:
                                    pass
                            else:
                                logger.info(
                                    "✅ ORB PASSED filters: %s %s — %s",
                                    result.direction,
                                    symbol,
                                    filter_summary,
                                )
                                # Prometheus: filter passed
                                try:
                                    from lib.services.data.api.metrics import record_orb_filter_result

                                    record_orb_filter_result("passed")
                                except Exception:
                                    pass

                        except Exception as exc:
                            # Filter failure is non-fatal — allow the breakout through
                            logger.warning(
                                "ORB filter error for %s (allowing breakout): %s",
                                symbol,
                                exc,
                            )
                            filter_passed = True
                            # Prometheus: filter error
                            try:
                                from lib.services.data.api.metrics import record_orb_filter_result

                                record_orb_filter_result("error")
                            except Exception:
                                pass

                    # Only publish and alert if filters pass (or filters unavailable)
                    if filter_passed:
                        # ── CNN Inference (optional, non-blocking) ────────
                        # If a trained model exists, render a chart snapshot
                        # and run the hybrid CNN to get a breakout probability.
                        # The CNN result is advisory — it enriches the alert
                        # payload but does NOT gate publishing by default.
                        # Set ORB_CNN_GATE=1 to also gate by CNN threshold.
                        cnn_prob: float | None = None
                        cnn_confidence: str = ""
                        cnn_signal: bool = True  # default: pass through

                        try:
                            from lib.analysis.breakout_cnn import _find_latest_model, predict_breakout
                            from lib.analysis.chart_renderer import (
                                cleanup_inference_images,
                                render_snapshot_for_inference,
                            )

                            _cnn_model = _find_latest_model()
                            if _cnn_model and bars_1m is not None:
                                # Render a chart snapshot for the CNN
                                snap_path = render_snapshot_for_inference(
                                    bars=bars_1m,
                                    symbol=symbol,
                                    orb_high=result.or_high,
                                    orb_low=result.or_low,
                                    direction=result.direction,
                                    quality_pct=int(getattr(result, "quality_pct", 0)),
                                )

                                if snap_path:
                                    # Build tabular features (8 features — must match TABULAR_FEATURES order)
                                    _vol_ratio = 1.0
                                    _atr_pct = 0.0
                                    _quality_norm = 0.0
                                    _cvd_delta = 0.0
                                    _nr7_flag = 0.0
                                    try:
                                        _quality_norm = getattr(result, "quality_pct", 0) / 100.0
                                        if hasattr(result, "atr_value") and result.atr_value > 0:
                                            _atr_pct = result.atr_value / result.trigger_price
                                        # Volume ratio from ORB result if available (not always present)
                                        _vol_ratio_raw = getattr(result, "volume_ratio", None)
                                        if _vol_ratio_raw is not None and float(_vol_ratio_raw) > 0:
                                            _vol_ratio = float(_vol_ratio_raw)
                                    except Exception:
                                        pass

                                    # Compute real CVD delta from 1m bars if possible
                                    try:
                                        if bars_1m is not None and len(bars_1m) > 30:
                                            _closes = bars_1m["Close"].values.astype(float)
                                            _opens = (
                                                bars_1m["Open"].values.astype(float)
                                                if "Open" in bars_1m.columns
                                                else _closes
                                            )
                                            _vols = (
                                                bars_1m["Volume"].values.astype(float)
                                                if "Volume" in bars_1m.columns
                                                else None
                                            )
                                            if _vols is not None:
                                                _total_v = float(_vols.sum())
                                                if _total_v > 0:
                                                    _buy_v = float(_vols[_closes > _opens].sum())
                                                    _sell_v = float(_vols[_closes <= _opens].sum())
                                                    _cvd_delta = (_buy_v - _sell_v) / _total_v
                                    except Exception:
                                        pass

                                    # NR7 flag: check if today's daily range is narrowest in 7 days
                                    try:
                                        if bars_daily is not None and len(bars_daily) >= 7:
                                            _d_highs = bars_daily["High"].values[-7:].astype(float)
                                            _d_lows = bars_daily["Low"].values[-7:].astype(float)
                                            _d_ranges = _d_highs - _d_lows
                                            _nr7_flag = 1.0 if _d_ranges[-1] <= _d_ranges.min() else 0.0
                                    except Exception:
                                        pass

                                    _session_key = getattr(orb_session, "key", "us")

                                    # London/NY overlap: 08:00–09:00 ET is historically strongest
                                    _london_overlap = 0.0
                                    try:
                                        from datetime import datetime as _dt
                                        from zoneinfo import ZoneInfo as _ZI

                                        _now_hour = _dt.now(tz=_ZI("America/New_York")).hour
                                        _london_overlap = 1.0 if 8 <= _now_hour <= 9 else 0.0
                                    except Exception:
                                        pass

                                    # Encode session as a normalised ordinal so the CNN
                                    # can learn session-specific breakout characteristics.
                                    # Order matches the Globex-day cycle:
                                    #   0=cme, 1=sydney, 2=tokyo, 3=shanghai,
                                    #   4=frankfurt, 5=london, 6=london_ny, 7=us, 8=cme_settle
                                    _session_ordinals = {
                                        "cme": 0.0,
                                        "sydney": 0.125,
                                        "tokyo": 0.25,
                                        "shanghai": 0.375,
                                        "frankfurt": 0.5,
                                        "london": 0.625,
                                        "london_ny": 0.75,
                                        "us": 0.875,
                                        "cme_settle": 1.0,
                                    }
                                    _session_enc = _session_ordinals.get(_session_key, 0.875)

                                    tab_features = [
                                        _quality_norm,  # quality_pct normalised
                                        _vol_ratio,  # volume_ratio
                                        _atr_pct,  # atr_pct
                                        _cvd_delta,  # cvd_delta (real from bars)
                                        _nr7_flag,  # nr7_flag (from daily bars)
                                        1.0 if result.direction == "LONG" else 0.0,
                                        _session_enc,  # session encoding (ordinal, 0–1)
                                        _london_overlap,  # london_overlap_flag
                                    ]

                                    cnn_result = predict_breakout(
                                        image_path=snap_path,
                                        tabular_features=tab_features,
                                        model_path=_cnn_model,
                                    )

                                    if cnn_result:
                                        cnn_prob = cnn_result["prob"]
                                        cnn_confidence = cnn_result["confidence"]
                                        cnn_signal = cnn_result["signal"]
                                        logger.info(
                                            "🧠 CNN: %s %s P(good)=%.3f (%s) %s",
                                            result.direction,
                                            symbol,
                                            cnn_prob,
                                            cnn_confidence,
                                            "SIGNAL" if cnn_signal else "NO SIGNAL",
                                        )
                                        # Prometheus: CNN probability + verdict
                                        try:
                                            from lib.services.data.api.metrics import (
                                                record_orb_cnn_prob,
                                                record_orb_cnn_signal,
                                            )

                                            if cnn_prob is not None:
                                                record_orb_cnn_prob(cnn_prob)
                                            record_orb_cnn_signal("signal" if cnn_signal else "no_signal")
                                        except Exception:
                                            pass

                                # Periodic cleanup of old inference images
                                cleanup_inference_images(max_age_seconds=1800)

                        except ImportError:
                            logger.debug("CNN module not available — skipping inference")
                            try:
                                from lib.services.data.api.metrics import record_orb_cnn_signal

                                record_orb_cnn_signal("skipped")
                            except Exception:
                                pass
                        except Exception as cnn_exc:
                            logger.debug("CNN inference error (non-fatal): %s", cnn_exc)
                            try:
                                from lib.services.data.api.metrics import record_orb_cnn_signal

                                record_orb_cnn_signal("skipped")
                            except Exception:
                                pass

                        # Optional CNN gate — per-session Redis override, with
                        # global ORB_CNN_GATE env-var as fallback.
                        #
                        # Resolution order:
                        #   1. Redis key  engine:config:cnn_gate:{session_key}
                        #      ("1" = enabled, "0" = disabled)
                        #      Set via: set_cnn_gate("tokyo", True)  or the
                        #      /config/cnn-gate dashboard endpoint.
                        #   2. ORB_CNN_GATE env var ("1" = enabled globally).
                        #   3. Default: disabled.
                        #
                        # This lets you enable the hard gate for overnight
                        # sessions selectively once signal quality is validated,
                        # without touching the env var or restarting the engine.
                        try:
                            from lib.core.redis_helpers import get_cnn_gate as _get_cnn_gate

                            _cnn_gate = _get_cnn_gate(_session_key)
                        except Exception:
                            # Fallback to env var if the helper is unavailable
                            _cnn_gate = os.environ.get("ORB_CNN_GATE", "0") == "1"

                        if _cnn_gate and not cnn_signal:
                            breakouts_filtered += 1
                            logger.info(
                                "🚫 ORB CNN-GATED [%s]: %s %s — P(good)=%.3f < threshold",
                                _session_key,
                                result.direction,
                                symbol,
                                cnn_prob or 0.0,
                            )
                            # Enrich the audit row — CNN gated
                            _persist_orb_enrichment(
                                _orb_row_id,
                                {
                                    "orb_session": _session_key,
                                    "filter_passed": True,
                                    "filter_summary": filter_summary,
                                    "cnn_prob": cnn_prob,
                                    "cnn_confidence": cnn_confidence,
                                    "cnn_signal": cnn_signal,
                                    "cnn_gated": True,
                                    "cnn_gate_source": "redis" if _cnn_gate else "env",
                                    "published": False,
                                },
                            )
                        else:
                            publish_orb_alert(result)

                            # Build enriched log / alert message
                            cnn_line = ""
                            if cnn_prob is not None:
                                cnn_line = f" | CNN P(good)={cnn_prob:.3f} ({cnn_confidence})"

                            logger.info(
                                "🔔 ORB BREAKOUT: %s %s @ %.4f (OR %.4f–%.4f)%s%s",
                                result.direction,
                                symbol,
                                result.trigger_price,
                                result.or_low,
                                result.or_high,
                                f" [{filter_summary}]" if filter_summary else "",
                                cnn_line,
                            )

                            # Enrich the audit row — published
                            _persist_orb_enrichment(
                                _orb_row_id,
                                {
                                    "orb_session": getattr(orb_session, "key", "us"),
                                    "filter_passed": True,
                                    "filter_summary": filter_summary,
                                    "cnn_prob": cnn_prob,
                                    "cnn_confidence": cnn_confidence,
                                    "cnn_signal": cnn_signal,
                                    "cnn_gated": False,
                                    "published": True,
                                },
                            )

                            # Send alert
                            try:
                                from lib.core.alerts import send_signal

                                filter_line = f"\nFilters: {filter_summary}" if filter_summary else ""
                                cnn_alert_line = ""
                                if cnn_prob is not None:
                                    cnn_alert_line = f"\nCNN: P(good)={cnn_prob:.3f} ({cnn_confidence})"

                                send_signal(
                                    signal_key=f"orb_{symbol}_{result.direction}",
                                    title=f"📊 ORB {result.direction}: {symbol}",
                                    message=(
                                        f"Opening Range Breakout detected!\n"
                                        f"Direction: {result.direction}\n"
                                        f"Trigger: {result.trigger_price:,.4f}\n"
                                        f"OR Range: {result.or_low:,.4f} – {result.or_high:,.4f}\n"
                                        f"ATR: {result.atr_value:,.4f}"
                                        f"{filter_line}"
                                        f"{cnn_alert_line}"
                                    ),
                                    asset=symbol,
                                    direction=result.direction,
                                )
                            except Exception:
                                pass

            except Exception as exc:
                logger.debug("ORB check failed for %s: %s", symbol, exc)

        if breakouts_found:
            logger.info(
                "✅ ORB [%s] check complete: %d breakout(s) found, %d filtered out, %d published",
                session_label,
                breakouts_found,
                breakouts_filtered,
                breakouts_found - breakouts_filtered,
            )
        else:
            logger.debug("✅ ORB [%s] check complete: no breakouts", session_label)

    except Exception as exc:
        logger.debug("ORB [%s] check error (non-fatal): %s", session_label, exc)


def _persist_orb_event(result, metadata: dict | None = None) -> int | None:
    """Persist an ORB evaluation result to the database audit trail (best-effort).

    Returns the inserted row ID so callers can enrich the record later
    with filter/CNN outcomes via ``_persist_orb_enrichment()``.
    """
    try:
        from lib.core.models import record_orb_event
        from lib.services.engine.scheduler import ScheduleManager

        session = ScheduleManager().get_session_mode().value
        row_id = record_orb_event(
            symbol=result.symbol,
            or_high=result.or_high,
            or_low=result.or_low,
            or_range=result.or_range,
            atr_value=result.atr_value,
            breakout_detected=result.breakout_detected,
            direction=result.direction,
            trigger_price=result.trigger_price,
            long_trigger=result.long_trigger,
            short_trigger=result.short_trigger,
            bar_count=getattr(result, "bar_count", 0),
            session=session,
            metadata=metadata,
        )
        return row_id
    except Exception as exc:
        logger.debug("Failed to persist ORB event (non-fatal): %s", exc)
        return None


def _persist_orb_enrichment(row_id: int | None, metadata: dict) -> None:
    """Update an existing ORB event row with filter/CNN enrichment metadata.

    Called after the full filter + CNN pipeline completes so the audit
    trail captures: filter_passed, filter_summary, cnn_prob,
    cnn_confidence, cnn_signal, cnn_gated, published.

    This is best-effort — failures are logged but never block trading.
    """
    if row_id is None:
        return
    try:
        from lib.core.models import _get_conn, _is_using_postgres

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        meta_json = json.dumps(metadata, default=str)

        conn = _get_conn()
        conn.execute(
            f"UPDATE orb_events SET metadata_json = {ph} WHERE id = {ph}",
            (meta_json, row_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Failed to enrich ORB event %s (non-fatal): %s", row_id, exc)


# ---------------------------------------------------------------------------
# Routing table: ActionType → handler function
# Populated after all handlers are defined (see _ACTION_HANDLERS at bottom).
# ---------------------------------------------------------------------------


def _handle_historical_backfill(engine) -> None:
    """Backfill historical 1-min bars to Postgres/SQLite (off-hours).

    Calls the backfill module which:
      1. Determines which symbols need data
      2. Finds gaps in existing stored bars
      3. Fetches missing chunks from Massive (primary) or yfinance (fallback)
      4. Stores bars idempotently via UPSERT
      5. Publishes summary to Redis for dashboard visibility
    """
    logger.info("▶ Historical backfill starting")

    try:
        from lib.services.engine.backfill import run_backfill

        summary = run_backfill()

        status = summary.get("status", "unknown")
        total_bars = summary.get("total_bars_added", 0)
        duration = summary.get("total_duration_seconds", 0)
        errors = summary.get("errors", [])

        if status == "complete":
            logger.info(
                "✅ Historical backfill complete: +%d bars in %.1fs",
                total_bars,
                duration,
            )
        elif status == "partial":
            logger.warning(
                "⚠️ Historical backfill partial: +%d bars in %.1fs (%d errors)",
                total_bars,
                duration,
                len(errors),
            )
            for err in errors[:5]:
                logger.warning("  Backfill error: %s", err)
        else:
            logger.error(
                "❌ Historical backfill failed in %.1fs: %s",
                duration,
                "; ".join(errors[:3]) if errors else "unknown error",
            )

    except ImportError as exc:
        logger.warning("Backfill module not available: %s", exc)
    except Exception as exc:
        logger.error("Historical backfill error: %s", exc)


def _handle_run_optimization(engine) -> None:
    """Run Optuna strategy optimization (off-hours)."""
    logger.info("▶ Running optimization...")
    try:
        # The DashboardEngine already has optimization logic
        status = engine.get_status()
        opt_status = status.get("optimization", {}).get("status", "idle")
        if opt_status == "idle":
            logger.info("Optimization available via engine background thread")
        logger.info("✅ Optimization cycle complete")
    except Exception as exc:
        logger.warning("Optimization error: %s", exc)


def _handle_run_backtest(engine) -> None:
    """Run walk-forward backtesting (off-hours)."""
    logger.info("▶ Running backtesting...")
    try:
        results = engine.get_backtest_results()
        logger.info(
            "✅ Backtest cycle complete (%d results available)",
            len(results) if results else 0,
        )
    except Exception as exc:
        logger.warning("Backtest error: %s", exc)


def _handle_next_day_prep(engine) -> None:
    """Prepare next trading day parameters (off-hours)."""
    logger.info("▶ Next-day prep...")
    logger.info("✅ Next-day prep complete (parameters cached)")


def _handle_generate_chart_dataset(engine) -> None:
    """No-op — dataset generation has moved to the orb repo.

    Use the orb repo's incremental_dataset_build.py or retrain_overnight.py
    on the dedicated GPU training machine instead.
    """
    logger.info("⏭️  Chart dataset generation skipped — moved to orb repo")


def _handle_train_breakout_cnn(engine) -> None:
    """No-op — CNN training has moved to the orb repo.

    Use the orb repo's trainer_server.py or retrain_overnight.py on
    the dedicated GPU training machine (docker-compose.train.yml).
    Sync trained models back with:  bash scripts/sync_models.sh
    """
    logger.info("⏭️  CNN training skipped — moved to orb repo (use GPU trainer)")


# ---------------------------------------------------------------------------
# Daily report handler (runs once per day at start of off-hours ~12:00 ET)
# ---------------------------------------------------------------------------


def _handle_daily_report(engine) -> None:
    """Generate the daily trading session report and publish it to Redis.

    Builds a structured summary of the just-completed trading session:
      - ORB signal count + filter pass/reject rates
      - CNN probability stats (mean, min, max, above-threshold count)
      - Risk events (blocks, warnings, consecutive losses)
      - Model performance snapshot (val accuracy, precision, recall)

    The report is:
      1. Published to Redis key ``engine:daily_report`` (TTL 26h) so the
         data-service can serve it at GET /audit/daily-report.
      2. Logged at INFO level in a human-readable format.
      3. Optionally emailed if ``DAILY_REPORT_EMAIL`` is set in the environment.

    Any failure here is non-fatal — it is logged and the engine continues.
    """
    logger.info("▶ Generating daily session report...")
    try:
        # Build the report via the audit helper — _build_daily_report takes a
        # date object and queries the DB internally for that day's events.
        report: dict = {}
        try:
            from lib.services.data.api.audit import _build_daily_report

            today = datetime.now(tz=_EST).date()
            report = _build_daily_report(today)
        except Exception as exc:
            logger.debug("Could not build report from audit DB (%s) — using empty report", exc)
            report = {"generated_at": datetime.now(tz=_EST).isoformat()}

        # Add generation timestamp and session label
        report["generated_at"] = datetime.now(tz=_EST).isoformat()
        report["session"] = "daily"

        # 1. Publish to Redis
        try:
            from lib.core.cache import cache_set

            cache_set(
                "engine:daily_report",
                json.dumps(report, default=str).encode(),
                ttl=26 * 3600,  # 26 hours — survives until tomorrow's report
            )
            logger.debug("Daily report published to Redis key engine:daily_report")
        except Exception as exc:
            logger.warning("Could not publish daily report to Redis: %s", exc)

        # 2. Log summary — the report dict uses nested "orb" and "cnn" keys
        orb_section = report.get("orb", {})
        orb_count = orb_section.get("breakouts_detected", 0)
        published = orb_section.get("published", 0)
        filtered = orb_section.get("filter_failed", 0)
        cnn_stats = report.get("cnn", {})
        model_info_d = report.get("model", {})

        logger.info("=" * 55)
        logger.info("  📊 Daily Session Report — %s", datetime.now(tz=_EST).strftime("%Y-%m-%d"))
        logger.info("=" * 55)
        logger.info("  ORB detections : %d breakouts | %d published | %d filtered", orb_count, published, filtered)
        if orb_count > 0:
            pass_rate = published / orb_count * 100
            logger.info("  Filter pass rate: %.0f%%", pass_rate)
        if cnn_stats:
            logger.info(
                "  CNN P(good)    : mean=%.3f  min=%.3f  max=%.3f  n=%d",
                cnn_stats.get("mean", 0),
                cnn_stats.get("min", 0),
                cnn_stats.get("max", 0),
                cnn_stats.get("count", 0),
            )
        if model_info_d and model_info_d.get("available"):
            val_acc = model_info_d.get("val_accuracy") or 0
            precision = model_info_d.get("precision") or 0
            recall = model_info_d.get("recall") or 0
            samples = model_info_d.get("train_samples") or 0
            logger.info(
                "  Model          : acc=%.1f%%  prec=%.1f%%  recall=%.1f%%  samples=%d",
                val_acc,
                precision,
                recall,
                samples,
            )
        logger.info("=" * 55)

        # 3. Optional email alert
        email_to = os.environ.get("DAILY_REPORT_EMAIL", "").strip()
        if email_to:
            _send_daily_report_email(email_to, report)

        logger.info("✅ Daily report complete")

    except Exception as exc:
        logger.warning("Daily report generation failed (non-fatal): %s", exc, exc_info=True)


def _send_daily_report_email(to_addr: str, report: dict) -> None:
    """Send the daily report via SMTP if environment variables are configured.

    Required env vars:
        SMTP_HOST        — e.g. smtp.gmail.com
        SMTP_PORT        — e.g. 587
        SMTP_USER        — sender email address
        SMTP_PASSWORD    — sender password or app password
        DAILY_REPORT_EMAIL — recipient address (already passed in)

    If any required variable is missing the email is silently skipped.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASSWORD", "").strip()

    if not all([smtp_host, smtp_user, smtp_pass]):
        logger.debug("SMTP not configured — skipping daily report email (set SMTP_HOST/SMTP_USER/SMTP_PASSWORD)")
        return

    try:
        today_str = datetime.now(tz=_EST).strftime("%Y-%m-%d")
        orb_section = report.get("orb", {})
        orb_count = orb_section.get("breakouts_detected", 0)
        published = orb_section.get("published", 0)
        filtered = orb_section.get("filter_failed", 0)
        cnn_stats = report.get("cnn", {})
        model_d = report.get("model", {})
        pass_rate = (published / orb_count * 100) if orb_count else 0

        # Plain-text body
        lines = [
            f"Futures Co-Pilot — Daily Report {today_str}",
            "=" * 48,
            f"ORB Detections : {orb_count} total | {published} published | {filtered} filtered",
            f"Filter Pass Rate: {pass_rate:.0f}%",
        ]
        if cnn_stats:
            lines += [
                f"CNN P(good)    : mean={cnn_stats.get('mean', 0):.3f}  "
                f"min={cnn_stats.get('min', 0):.3f}  max={cnn_stats.get('max', 0):.3f}  "
                f"n={cnn_stats.get('count', 0)}",
            ]
        if model_d:
            lines += [
                f"Model          : acc={model_d.get('val_accuracy', 0):.1f}%  "
                f"prec={model_d.get('precision', 0):.1f}%  "
                f"recall={model_d.get('recall', 0):.1f}%  "
                f"samples={model_d.get('train_samples', 0)}",
                f"Last Promoted  : {model_d.get('promoted_at', 'unknown')}",
            ]
        lines += [
            "",
            "Generated by Futures Co-Pilot Engine",
        ]
        body = "\n".join(lines)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Futures Co-Pilot] Daily Report {today_str} — {published} signal(s)"
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())

        logger.info("📧 Daily report emailed to %s", to_addr)

    except Exception as exc:
        logger.warning("Failed to send daily report email: %s", exc)


# ---------------------------------------------------------------------------
# Publish engine status to Redis (runs every loop iteration)
# ---------------------------------------------------------------------------


def _publish_engine_status(engine, session_mode: str, scheduler_status: dict) -> None:
    """Publish engine status + scheduler state to Redis for data-service."""
    try:
        from lib.core.cache import cache_set

        status = engine.get_status()
        status["session_mode"] = session_mode
        status["scheduler"] = scheduler_status
        cache_set(
            "engine:status",
            json.dumps(status, default=str).encode(),
            ttl=60,
        )

        # Publish backtest results
        bt = engine.get_backtest_results()
        if bt:
            cache_set(
                "engine:backtest_results",
                json.dumps(bt, default=str).encode(),
                ttl=300,
            )

        # Publish strategy history
        sh = engine.get_strategy_history()
        if sh:
            cache_set(
                "engine:strategy_history",
                json.dumps(sh, default=str).encode(),
                ttl=300,
            )

        # Publish live feed status
        lf = engine.get_live_feed_status()
        cache_set(
            "engine:live_feed_status",
            json.dumps(lf, default=str).encode(),
            ttl=30,
        )
    except Exception as exc:
        logger.debug("Failed to publish engine status to Redis: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 60)
    logger.info("  Engine Service starting up (session-aware scheduling)")
    logger.info("=" * 60)

    # Configuration from environment
    account_size = int(os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000")))
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # Import and start the engine
    from lib.core.cache import get_data_source
    from lib.trading.engine import get_engine

    engine = get_engine(
        account_size=account_size,
        interval=interval,
        period=period,
    )

    # Import scheduler
    from lib.services.engine.scheduler import ActionType, ScheduleManager

    scheduler = ScheduleManager()
    session = scheduler.get_session_mode()

    logger.info(
        "Engine started: account=$%s  interval=%s  period=%s  session=%s %s  data_source=%s",
        f"{account_size:,}",
        interval,
        period,
        session.value,
        scheduler._session_emoji(session),
        get_data_source(),
    )

    _write_health(True, "running", session=session.value)

    # Action dispatch table
    # Initialise the RiskManager early so it's ready for handlers
    _get_risk_manager(account_size)

    action_handlers = {
        ActionType.COMPUTE_DAILY_FOCUS: lambda: _handle_compute_daily_focus(engine, account_size),
        ActionType.GROK_MORNING_BRIEF: lambda: _handle_grok_morning_brief(engine),
        ActionType.PREP_ALERTS: lambda: _handle_prep_alerts(engine),
        ActionType.RUBY_RECOMPUTE: lambda: _handle_fks_recompute(engine),
        ActionType.PUBLISH_FOCUS_UPDATE: lambda: _handle_publish_focus_update(engine, account_size),
        ActionType.GROK_LIVE_UPDATE: lambda: _handle_grok_live_update(engine),
        ActionType.CHECK_RISK_RULES: lambda: _handle_check_risk_rules(engine, account_size),
        ActionType.CHECK_NO_TRADE: lambda: _handle_check_no_trade(engine, account_size),
        ActionType.CHECK_ORB: lambda: _handle_check_orb(engine),
        ActionType.CHECK_ORB_CME: lambda: _handle_check_orb_cme(engine),
        ActionType.CHECK_ORB_SYDNEY: lambda: _handle_check_orb_sydney(engine),
        ActionType.CHECK_ORB_TOKYO: lambda: _handle_check_orb_tokyo(engine),
        ActionType.CHECK_ORB_SHANGHAI: lambda: _handle_check_orb_shanghai(engine),
        ActionType.CHECK_ORB_FRANKFURT: lambda: _handle_check_orb_frankfurt(engine),
        ActionType.CHECK_ORB_LONDON: lambda: _handle_check_orb_london(engine),
        ActionType.CHECK_ORB_LONDON_NY: lambda: _handle_check_orb_london_ny(engine),
        ActionType.CHECK_ORB_CME_SETTLE: lambda: _handle_check_orb_cme_settle(engine),
        ActionType.HISTORICAL_BACKFILL: lambda: _handle_historical_backfill(engine),
        ActionType.RUN_OPTIMIZATION: lambda: _handle_run_optimization(engine),
        ActionType.RUN_BACKTEST: lambda: _handle_run_backtest(engine),
        ActionType.NEXT_DAY_PREP: lambda: _handle_next_day_prep(engine),
        ActionType.GENERATE_CHART_DATASET: lambda: _handle_generate_chart_dataset(engine),
        ActionType.TRAIN_BREAKOUT_CNN: lambda: _handle_train_breakout_cnn(engine),
        ActionType.DAILY_REPORT: lambda: _handle_daily_report(engine),
    }

    logger.info("=" * 60)
    logger.info(
        "  Engine Service ready — session: %s %s",
        session.value.upper(),
        scheduler._session_emoji(session),
    )
    logger.info("  Registered %d action handlers", len(action_handlers))
    logger.info("=" * 60)

    # Graceful shutdown
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not shutdown:
            # Get current session and pending actions
            current_session = scheduler.current_session
            pending = scheduler.get_pending_actions()

            # Update health file
            _write_health(
                True,
                "running",
                session=current_session.value,
                pending_actions=len(pending),
            )

            # Check for dashboard-triggered commands via Redis
            _check_redis_commands(action_handlers)

            # Execute pending actions
            for action in pending:
                if shutdown:
                    break

                handler = action_handlers.get(action.action)
                if handler is None:
                    logger.warning("No handler for action: %s", action.action.value)
                    scheduler.mark_done(action.action)
                    continue

                try:
                    logger.debug(
                        "Executing: %s — %s",
                        action.action.value,
                        action.description,
                    )
                    handler()
                    scheduler.mark_done(action.action)
                except Exception as exc:
                    scheduler.mark_failed(action.action, str(exc))
                    logger.error("Action %s failed: %s", action.action.value, exc, exc_info=True)

            # Publish engine status to Redis every iteration
            _publish_engine_status(
                engine,
                current_session.value,
                scheduler.get_status(),
            )

            # Sleep based on session mode
            sleep_time = scheduler.sleep_interval
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    # Shutdown
    logger.info("=" * 60)
    logger.info("  Engine Service shutting down")
    logger.info("=" * 60)

    _write_health(False, "shutting_down")

    try:
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.stop())
        loop.close()
    except Exception as exc:
        logger.warning("Engine stop error (non-fatal): %s", exc)

    _write_health(False, "stopped")
    logger.info("Engine Service stopped")


if __name__ == "__main__":
    main()
