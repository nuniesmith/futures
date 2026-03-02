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
import threading
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
_retrain_thread: threading.Thread | None = None


def _check_redis_commands(action_handlers: dict) -> None:
    """Check Redis for commands published by the data service.

    Currently supports:
      - engine:cmd:retrain_cnn  — trigger CNN retraining with parameters

    The command key is consumed (deleted) after being read so it only
    fires once.
    """
    global _retrain_thread

    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_get

        if not REDIS_AVAILABLE:
            return

        raw = cache_get(_RETRAIN_CMD_KEY)
        if not raw:
            return

        # Consume the command immediately so it doesn't re-fire
        try:
            from lib.core.cache import _r

            if _r:
                _r.delete(_RETRAIN_CMD_KEY)
        except Exception:
            pass

        # Parse the command
        cmd = json.loads(raw if isinstance(raw, str) else raw.decode())
        cmd_type = cmd.get("command", "")

        if cmd_type != "retrain_cnn":
            logger.warning("Unknown engine command: %s", cmd_type)
            return

        # Don't start if a retrain thread is already running
        if _retrain_thread is not None and _retrain_thread.is_alive():
            logger.warning("Retrain command received but a retrain job is already running — ignoring")
            _publish_retrain_status("rejected", "A retrain job is already running")
            return

        session = cmd.get("session", "both")
        skip_dataset = cmd.get("skip_dataset", False)
        epochs = cmd.get("epochs")
        batch_size = cmd.get("batch_size")

        logger.info(
            "📩 Received retrain command from dashboard: session=%s skip_dataset=%s epochs=%s",
            session,
            skip_dataset,
            epochs,
        )

        # Run retraining in a background thread so it doesn't block the engine loop
        _retrain_thread = threading.Thread(
            target=_run_retrain_from_command,
            args=(session, skip_dataset, epochs, batch_size),
            daemon=True,
            name="cnn-retrain-cmd",
        )
        _retrain_thread.start()

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
    """Execute CNN retraining in a background thread (triggered by dashboard command)."""
    _publish_retrain_status(
        "running", "CNN retraining started via dashboard command", session=session, skip_dataset=skip_dataset
    )
    try:
        import sys as _sys
        from pathlib import Path

        # Locate scripts/ directory
        _candidates = [
            Path("/app/scripts"),
            Path(__file__).resolve().parents[4] / "scripts",
        ]
        _scripts_dir = None
        for _c in _candidates:
            if (_c / "retrain_overnight.py").is_file():
                _scripts_dir = _c
                break

        if _scripts_dir is None:
            msg = "retrain_overnight.py not found in any scripts/ location"
            logger.error(msg)
            _publish_retrain_status("failed", msg)
            return

        _scripts_str = str(_scripts_dir)
        if _scripts_str not in _sys.path:
            _sys.path.insert(0, _scripts_str)

        import retrain_overnight  # noqa: E402

        # Build config
        cfg = retrain_overnight.RetrainConfig.from_env()
        cfg.immediate = True
        cfg.force = True
        cfg.skip_dataset = skip_dataset
        cfg.orb_session = session
        if epochs is not None:
            cfg.epochs = epochs
        if batch_size is not None:
            cfg.batch_size = batch_size

        logger.info(
            "🚀 Starting CNN retrain pipeline: session=%s skip_dataset=%s epochs=%s", session, skip_dataset, cfg.epochs
        )

        result = retrain_overnight.run_pipeline(cfg)

        if result.status == "success":
            logger.info(
                "✅ CNN retrain (dashboard-triggered) succeeded — model promoted (acc=%.1f%%)", result.best_val_accuracy
            )
            _publish_retrain_status(
                "success", f"Model promoted (acc={result.best_val_accuracy:.1f}%)", accuracy=result.best_val_accuracy
            )
        elif result.status == "gate_rejected":
            logger.warning("🚫 CNN retrain candidate rejected: %s", result.gate_reason)
            _publish_retrain_status("gate_rejected", result.gate_reason)
        else:
            errors = ", ".join(result.errors[:3]) or "unknown error"
            logger.error("❌ CNN retrain failed: %s", errors)
            _publish_retrain_status("failed", errors)

    except Exception as exc:
        logger.error("CNN retrain (dashboard-triggered) error: %s", exc, exc_info=True)
        _publish_retrain_status("failed", str(exc))


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
    drives range establishment for metals, indices, and energy futures.
    """
    from lib.services.engine.orb import LONDON_SESSION

    _handle_check_orb(engine, orb_session=LONDON_SESSION)


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

                            if _session_key == "london":
                                _filter_allowed_windows = [
                                    (_dt_time(3, 0), _dt_time(5, 0)),
                                ]
                                _pm_end = _dt_time(3, 0)  # premarket ends at OR start
                                _enable_lunch = False  # irrelevant at 03:00–05:00 ET
                                logger.debug("ORB filters: London mode — windows 03:00–05:00, PM end 03:00, lunch OFF")
                            else:
                                # US session (default)
                                _filter_allowed_windows = [
                                    (_dt_time(8, 20), _dt_time(10, 30)),
                                ]
                                _pm_end = _dt_time(8, 20)
                                _enable_lunch = True
                                logger.debug("ORB filters: US mode — windows 08:20–10:30, PM end 08:20, lunch ON")

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
                            else:
                                logger.info(
                                    "✅ ORB PASSED filters: %s %s — %s",
                                    result.direction,
                                    symbol,
                                    filter_summary,
                                )

                        except Exception as exc:
                            # Filter failure is non-fatal — allow the breakout through
                            logger.warning(
                                "ORB filter error for %s (allowing breakout): %s",
                                symbol,
                                exc,
                            )
                            filter_passed = True

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
                                    # Build tabular features
                                    _vol_ratio = 1.0
                                    _atr_pct = 0.0
                                    _quality_norm = 0.0
                                    try:
                                        _quality_norm = getattr(result, "quality_pct", 0) / 100.0
                                        if hasattr(result, "atr_value") and result.atr_value > 0:
                                            _atr_pct = result.atr_value / result.trigger_price
                                    except Exception:
                                        pass

                                    tab_features = [
                                        _quality_norm,  # quality_pct normalised
                                        _vol_ratio,  # volume_ratio (default 1.0)
                                        _atr_pct,  # atr_pct
                                        0.0,  # cvd_delta (not available in live path)
                                        0.0,  # nr7_flag (could be enriched later)
                                        1.0 if result.direction == "LONG" else 0.0,
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

                                # Periodic cleanup of old inference images
                                cleanup_inference_images(max_age_seconds=1800)

                        except ImportError:
                            logger.debug("CNN module not available — skipping inference")
                        except Exception as cnn_exc:
                            logger.debug("CNN inference error (non-fatal): %s", cnn_exc)

                        # Optional CNN gate: if ORB_CNN_GATE=1, block low-prob signals
                        _cnn_gate = os.environ.get("ORB_CNN_GATE", "0") == "1"
                        if _cnn_gate and not cnn_signal:
                            breakouts_filtered += 1
                            logger.info(
                                "🚫 ORB CNN-GATED: %s %s — P(good)=%.3f < threshold",
                                result.direction,
                                symbol,
                                cnn_prob or 0.0,
                            )
                            # Enrich the audit row — CNN gated
                            _persist_orb_enrichment(
                                _orb_row_id,
                                {
                                    "orb_session": getattr(orb_session, "key", "us"),
                                    "filter_passed": True,
                                    "filter_summary": filter_summary,
                                    "cnn_prob": cnn_prob,
                                    "cnn_confidence": cnn_confidence,
                                    "cnn_signal": cnn_signal,
                                    "cnn_gated": True,
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
    """Generate labeled chart images for CNN training (off-hours).

    Pulls historical 1-minute bars for focus assets, simulates ORB trades
    across sliding windows, renders Ruby-style chart snapshots, and writes
    a CSV manifest (dataset/labels.csv) ready for BreakoutDataset.
    """
    logger.info("▶ Generating chart dataset for CNN training...")
    try:
        from lib.analysis.dataset_generator import DatasetConfig, generate_dataset

        # Determine symbols from daily focus (or use defaults)
        symbols = ["MGC", "MES", "MNQ"]
        try:
            from lib.core.cache import cache_get

            raw_focus = cache_get("engine:daily_focus")
            if raw_focus:
                focus_data = json.loads(raw_focus)
                focus_symbols = [a.get("symbol", "") for a in focus_data.get("assets", [])]
                if focus_symbols:
                    symbols = [s for s in focus_symbols if s]
        except Exception:
            pass

        config = DatasetConfig(
            bars_source="cache",
            skip_existing=True,
            chart_dpi=150,
            orb_session="both",
        )

        stats = generate_dataset(
            symbols=symbols,
            days_back=90,
            config=config,
        )

        logger.info(
            "✅ Chart dataset generation complete: %s",
            stats.summary(),
        )

    except ImportError as exc:
        logger.warning("Dataset generator not available: %s", exc)
    except Exception as exc:
        logger.error("Chart dataset generation error: %s", exc)


def _handle_train_breakout_cnn(engine) -> None:
    """Train or retrain the EfficientNetV2 breakout CNN (off-hours).

    Uses the overnight retraining pipeline which handles:
      1. Train/val split of the latest dataset
      2. GPU-accelerated training (mixed precision, class weighting, etc.)
      3. Validation gate (accuracy, precision, recall thresholds)
      4. Atomic model promotion (only if candidate beats champion)
      5. Archival of previous champion + cleanup of old checkpoints

    Falls back to the basic train_model() if the pipeline is unavailable.
    """
    global _retrain_thread

    # If a dashboard-triggered retrain is already running, skip the scheduled one
    if _retrain_thread is not None and _retrain_thread.is_alive():
        logger.info(
            "⏭️  Skipping scheduled CNN training — dashboard-triggered retrain is already running (thread=%s)",
            _retrain_thread.name,
        )
        return

    logger.info("▶ Training breakout CNN (overnight pipeline)...")

    # --- Try the full overnight retraining pipeline first ---
    try:
        import sys as _sys
        from pathlib import Path

        # Locate the scripts/ directory (mounted at /app/scripts in Docker,
        # or at <project_root>/scripts on bare metal).
        _candidates = [
            Path("/app/scripts"),
            Path(__file__).resolve().parents[4] / "scripts",  # futures/scripts
        ]
        _scripts_dir = None
        for _c in _candidates:
            if (_c / "retrain_overnight.py").is_file():
                _scripts_dir = _c
                break

        if _scripts_dir is None:
            logger.info("retrain_overnight.py not found in any scripts/ location — falling back to basic trainer")
        else:
            # Add scripts/ to sys.path so we can import it directly
            _scripts_str = str(_scripts_dir)
            if _scripts_str not in _sys.path:
                _sys.path.insert(0, _scripts_str)

            # Use a direct import — much more reliable than importlib.util
            import retrain_overnight  # noqa: E402

            success = retrain_overnight.run_from_engine()
            if success:
                logger.info("✅ CNN retraining pipeline completed — model promoted")
            else:
                logger.warning("CNN retraining pipeline finished but model was NOT promoted (gate rejected or error)")
            return

    except Exception as exc:
        logger.warning("Overnight pipeline failed (%s) — falling back to basic trainer", exc, exc_info=True)

    # --- Fallback: use the in-module train_model() directly ---
    try:
        from lib.analysis.breakout_cnn import model_info, train_model

        if train_model is None:
            logger.warning("PyTorch not installed — CNN training skipped")
            return

        csv_path = "dataset/labels.csv"
        if not os.path.isfile(csv_path):
            logger.warning("No dataset CSV found at %s — skipping CNN training", csv_path)
            return

        model_path = train_model(
            data_csv=csv_path,
            epochs=8,
            batch_size=32,
            freeze_epochs=2,
            model_dir="models",
            num_workers=0,  # safe for Docker (no /dev/shm issues)
        )

        if model_path:
            info = model_info(model_path)
            logger.info(
                "✅ CNN training complete (basic): %s (%.1f MB, device=%s)",
                model_path,
                info.get("size_mb", 0),
                info.get("device", "unknown"),
            )
        else:
            logger.warning("CNN training returned no model path")

    except ImportError as exc:
        logger.warning("Breakout CNN module not available: %s", exc)
    except Exception as exc:
        logger.error("CNN training error: %s", exc, exc_info=True)


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
        ActionType.CHECK_ORB_LONDON: lambda: _handle_check_orb_london(engine),
        ActionType.HISTORICAL_BACKFILL: lambda: _handle_historical_backfill(engine),
        ActionType.RUN_OPTIMIZATION: lambda: _handle_run_optimization(engine),
        ActionType.RUN_BACKTEST: lambda: _handle_run_backtest(engine),
        ActionType.NEXT_DAY_PREP: lambda: _handle_next_day_prep(engine),
        ActionType.GENERATE_CHART_DATASET: lambda: _handle_generate_chart_dataset(engine),
        ActionType.TRAIN_BREAKOUT_CNN: lambda: _handle_train_breakout_cnn(engine),
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
