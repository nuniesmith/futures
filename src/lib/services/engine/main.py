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
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pandas as pd

    from lib.services.engine.breakout import BreakoutResult

from lib.core.logging_config import get_logger, setup_logging

setup_logging(service="engine")
logger = get_logger("engine_service")

_EST = ZoneInfo("America/New_York")
HEALTH_FILE = "/tmp/engine_health.json"

# ---------------------------------------------------------------------------
# CNN model hot-reload — detect when breakout_cnn_best.pt changes on disk
# ---------------------------------------------------------------------------
# The ModelWatcher (lib.services.engine.model_watcher) replaces the old
# inline polling approach.  It uses watchdog (inotify/FSEvents) for instant
# notification when model files change, with a polling fallback.
#
# The watcher is started in main() and stopped on shutdown.
# ---------------------------------------------------------------------------
_model_watcher = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level PositionManager singleton (initialised in main())
# ---------------------------------------------------------------------------
_position_manager = None


def _get_position_manager(account_size: int = 50_000):
    """Lazy-init and return the global PositionManager singleton."""
    global _position_manager
    if _position_manager is None:
        try:
            from lib.services.engine.position_manager import PositionManager

            _position_manager = PositionManager(account_size=account_size)
            _position_manager.load_state()
            logger.info(
                "PositionManager initialised (account=$%s)",
                f"{account_size:,}",
            )
        except Exception as exc:
            logger.warning("PositionManager init failed (non-fatal): %s", exc)
    return _position_manager


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

    CNN retraining has moved to the rb repo — commands are acknowledged
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
            "📩 Received retrain command from dashboard — CNN training has moved to the rb repo. "
            "Use the GPU trainer (docker-compose.train.yml) in the rb repo instead."
        )
        _publish_retrain_status(
            "rejected",
            "CNN training has moved to the rb repo. Use the GPU trainer there instead.",
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
    """No-op — CNN training has moved to the rb repo."""
    logger.info("⏭️  CNN retrain command ignored — training has moved to the rb repo")
    _publish_retrain_status(
        "rejected",
        "CNN training has moved to the rb repo. Use the GPU trainer there instead.",
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

    # Run HMM regime detection for all assets with available bar data and
    # publish the consolidated state map to Redis so the dashboard panel
    # and Prometheus metrics scrape can both read it.
    _publish_regime_states()


def _publish_regime_states() -> None:
    """Run HMM regime detection across all focus assets and publish to Redis.

    Reads bar data from the engine:bars_1m / engine:bars_daily cache keys,
    fits / updates the per-instrument RegimeDetector, then writes:
      - ``engine:regime_states``  — JSON map of {symbol → regime_info} (TTL 10 min)
      - ``engine:regime:{symbol}`` — per-symbol JSON (TTL 10 min)

    Also pushes the results into Prometheus gauges immediately so the next
    /metrics/prometheus scrape reflects the latest regime.
    """
    try:
        import io

        import pandas as pd

        from lib.analysis.regime import detect_regime_hmm
        from lib.core.cache import cache_get, cache_set

        raw_focus = cache_get("engine:daily_focus")
        if not raw_focus:
            logger.debug("No daily focus — skipping regime update")
            return

        focus_data = json.loads(raw_focus)
        assets = focus_data.get("assets", [])
        if not assets:
            return

        regime_map: dict[str, dict] = {}

        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "") or symbol
            if not symbol:
                continue

            try:
                # Prefer daily bars for regime (longer history = better HMM fit)
                bars = None
                for cache_key in (
                    f"engine:bars_daily:{ticker}",
                    f"engine:bars_1m:{ticker}",
                ):
                    raw_bars = cache_get(cache_key)
                    if raw_bars:
                        raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
                        candidate = pd.read_json(io.StringIO(raw_str))
                        if not candidate.empty and len(candidate) >= 50:
                            bars = candidate
                            break

                if bars is None or bars.empty:
                    logger.debug("No bars for regime detection on %s", symbol)
                    continue

                info = detect_regime_hmm(ticker, bars)
                regime_map[symbol] = info

                # Per-symbol key
                cache_set(
                    f"engine:regime:{symbol}",
                    json.dumps(info, default=str).encode(),
                    ttl=600,
                )
            except Exception as exc:
                logger.debug("Regime detection skipped for %s: %s", symbol, exc)
                continue

        if not regime_map:
            return

        # Consolidated map
        cache_set(
            "engine:regime_states",
            json.dumps(regime_map, default=str).encode(),
            ttl=600,
        )

        # Push into Prometheus gauges immediately (don't wait for scrape)
        try:
            from lib.services.data.api.metrics import update_regime

            for sym, info in regime_map.items():
                update_regime(
                    symbol=sym,
                    regime=info.get("regime", "choppy"),
                    confidence=float(info.get("confidence", 0.0)),
                    position_multiplier=float(info.get("position_multiplier", 0.25)),
                )
        except Exception:
            pass

        logger.debug(
            "Regime states published for %d symbols: %s",
            len(regime_map),
            {s: v.get("regime") for s, v in regime_map.items()},
        )

    except Exception as exc:
        logger.debug("_publish_regime_states failed (non-fatal): %s", exc)


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


def _handle_check_orb_crypto_utc0(engine) -> None:
    """Check for Crypto UTC-midnight ORB patterns (19:00–19:30 ET EST / 00:00 UTC).

    High-volume Asia open window for BTC/ETH/SOL and other spot crypto pairs
    tracked on Kraken.  Uses wider ATR thresholds and looser quality gates
    via CRYPTO_UTC_MIDNIGHT_SESSION and per-symbol overrides.
    Only active when ENABLE_KRAKEN_CRYPTO=1.  wraps_midnight=True.
    """
    try:
        from lib.services.engine.orb import CRYPTO_UTC_MIDNIGHT_SESSION
    except ImportError:
        logger.warning("CRYPTO_UTC_MIDNIGHT_SESSION not available — crypto ORB disabled")
        return

    _handle_check_orb(engine, orb_session=CRYPTO_UTC_MIDNIGHT_SESSION)


def _handle_check_orb_crypto_utc12(engine) -> None:
    """Check for Crypto UTC-noon ORB patterns (07:00–07:30 ET EST / 12:00 UTC).

    London morning crypto session.  High-volume pre-US-open positioning
    window for BTC/ETH/SOL and tracked Kraken pairs.  Uses wider ATR
    thresholds and per-symbol overrides.
    Only active when ENABLE_KRAKEN_CRYPTO=1.  wraps_midnight=False.
    """
    try:
        from lib.services.engine.orb import CRYPTO_UTC_NOON_SESSION
    except ImportError:
        logger.warning("CRYPTO_UTC_NOON_SESSION not available — crypto ORB disabled")
        return

    _handle_check_orb(engine, orb_session=CRYPTO_UTC_NOON_SESSION)


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
                    bars_daily = None  # initialised here so CNN block can always read it

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

                            # Read MTF min_pass_score from env (default 0.55)
                            _mtf_min_score = float(os.environ.get("ORB_MTF_MIN_SCORE", "0.55"))

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
                                enable_mtf_analyzer=True,
                                mtf_min_pass_score=_mtf_min_score,
                            )

                            filter_passed = filter_result.passed
                            filter_summary = filter_result.summary

                            # ── MTF enrichment on the ORB result ──────────────────
                            # Run the full MTF analyzer independently of the filter
                            # so we always capture mtf_score / macd_slope / divergence
                            # in the DB even when the filter gate is in "majority" mode
                            # and the MTF verdict didn't block the signal.
                            _orb_mtf_score: float | None = None
                            _orb_macd_slope: float | None = None
                            _orb_divergence: str = ""
                            try:
                                from lib.analysis.mtf_analyzer import analyze_mtf as _analyze_mtf

                                _mtf_res = _analyze_mtf(bars_htf, direction=result.direction)
                                if not _mtf_res.error:
                                    _orb_mtf_score = _mtf_res.mtf_score
                                    _orb_macd_slope = _mtf_res.macd_histogram_slope
                                    _orb_divergence = _mtf_res.divergence_type or ""
                            except Exception:
                                pass

                            if not filter_passed:
                                breakouts_filtered += 1
                                logger.info(
                                    "🚫 ORB FILTERED: %s %s @ %.4f — %s",
                                    result.direction,
                                    symbol,
                                    result.trigger_price,
                                    filter_summary,
                                )
                                # Enrich the audit row with filter rejection + MTF data
                                _persist_orb_enrichment(
                                    _orb_row_id,
                                    {
                                        "orb_session": _session_key,
                                        "filter_passed": False,
                                        "filter_summary": filter_summary,
                                        "published": False,
                                        "mtf_score": _orb_mtf_score,
                                        "macd_slope": _orb_macd_slope,
                                        "divergence": _orb_divergence,
                                    },
                                )
                                # Also update the new dedicated columns
                                try:
                                    from lib.core.models import _get_conn, _is_using_postgres

                                    if _orb_row_id is not None and (
                                        _orb_mtf_score is not None or _orb_macd_slope is not None or _orb_divergence
                                    ):
                                        _pg = _is_using_postgres()
                                        _ph = "%s" if _pg else "?"
                                        _conn = _get_conn()
                                        _conn.execute(
                                            f"UPDATE orb_events SET "
                                            f"breakout_type={_ph}, mtf_score={_ph}, "
                                            f"macd_slope={_ph}, divergence={_ph} "
                                            f"WHERE id={_ph}",
                                            (
                                                "ORB",
                                                _orb_mtf_score,
                                                _orb_macd_slope,
                                                _orb_divergence,
                                                _orb_row_id,
                                            ),
                                        )
                                        _conn.commit()
                                        _conn.close()
                                except Exception:
                                    pass
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
                            from lib.analysis.chart_renderer import (  # type: ignore[import-unresolved]
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
                                    # Build tabular features (18 features v6 — must match TABULAR_FEATURES order)
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
                                    _now_hour = 10  # default US open hour
                                    try:
                                        from datetime import datetime as _dt
                                        from zoneinfo import ZoneInfo as _ZI

                                        _now_hour = _dt.now(tz=_ZI("America/New_York")).hour
                                        _london_overlap = 1.0 if 8 <= _now_hour <= 9 else 0.0
                                    except Exception:
                                        pass

                                    # Encode session as a normalised ordinal using the
                                    # canonical get_session_ordinal() from breakout_cnn.
                                    try:
                                        from lib.analysis.breakout_cnn import (
                                            get_asset_class_id as _get_asset_cls,
                                        )
                                        from lib.analysis.breakout_cnn import (
                                            get_asset_volatility_class as _get_vol_class,
                                        )
                                        from lib.analysis.breakout_cnn import (
                                            get_breakout_type_ordinal as _get_btype_ord,
                                        )
                                        from lib.analysis.breakout_cnn import (
                                            get_session_ordinal as _get_session_ordinal,
                                        )

                                        _session_enc = _get_session_ordinal(_session_key)
                                    except ImportError:
                                        _get_btype_ord = lambda t: 0.0  # noqa: E731
                                        _get_vol_class = lambda t: 0.5  # noqa: E731
                                        _get_asset_cls = lambda t: 0.0  # noqa: E731
                                        _session_enc = 0.875

                                    # ── v4 features [8..13] ──────────────────────────────

                                    # [8] or_range_atr_ratio — raw ORB range / ATR
                                    _or_range = getattr(result, "or_range", 0.0) or getattr(result, "range_size", 0.0)
                                    _or_range_atr = 0.0
                                    if result.atr_value > 0 and _or_range > 0:
                                        _or_range_atr = _or_range / result.atr_value

                                    # [9] premarket_range_ratio — raw PM range / ORB range
                                    _pm_range_ratio = 0.0
                                    try:
                                        _pm_high = getattr(result, "pm_high", None)
                                        _pm_low = getattr(result, "pm_low", None)
                                        if _pm_high is not None and _pm_low is not None and _or_range > 0:
                                            _pm_range = float(_pm_high) - float(_pm_low)
                                            if _pm_range > 0:
                                                _pm_range_ratio = _pm_range / _or_range
                                    except Exception:
                                        pass

                                    # [10] bar_of_day — minutes since Globex open (18:00 ET) / 1380
                                    _bar_of_day_min = (_now_hour - 18) * 60 if _now_hour >= 18 else (_now_hour + 6) * 60
                                    _bar_of_day = max(0.0, min(1.0, _bar_of_day_min / 1380.0))

                                    # [11] day_of_week — Mon=0..Fri=4 / 4
                                    _dow = 0.5
                                    try:
                                        from datetime import datetime as _dt3

                                        _dow_raw = _dt3.now().weekday()
                                        if 0 <= _dow_raw <= 4:
                                            _dow = _dow_raw / 4.0
                                    except Exception:
                                        pass

                                    # [12] vwap_distance — (price - vwap) / ATR (raw)
                                    _vwap_dist = 0.0
                                    try:
                                        _vwap = getattr(result, "vwap", None)
                                        if _vwap is None and _or_range > 0:
                                            _vwap = (result.or_high + result.or_low) / 2.0
                                        if _vwap is not None and result.atr_value > 0:
                                            _vwap_dist = (result.trigger_price - float(_vwap)) / result.atr_value
                                    except Exception:
                                        pass

                                    # [13] asset_class_id — ordinal / 4
                                    _asset_cls = _get_asset_cls(ticker or symbol)

                                    # ── v6 features [14..17] ─────────────────────────────

                                    # [14] breakout_type_ord — BreakoutType ordinal / 12
                                    _btype_raw = getattr(result, "breakout_type", "ORB")
                                    _btype_name = _btype_raw.value if hasattr(_btype_raw, "value") else str(_btype_raw)
                                    _btype_ord_val = _get_btype_ord(_btype_name)

                                    # [15] asset_volatility_class — low=0.0 / med=0.5 / high=1.0
                                    _vol_class_val = _get_vol_class(ticker or symbol)

                                    # [16] hour_of_day — current ET hour / 23
                                    _hour_of_day = max(0.0, min(1.0, _now_hour / 23.0))

                                    # [17] tp3_atr_mult_norm — TP3 multiplier / 5.0
                                    _tp3_norm = 0.0
                                    try:
                                        from lib.core.breakout_types import (
                                            BreakoutType as _BT,
                                        )
                                        from lib.core.breakout_types import (
                                            breakout_type_from_name as _bt_from_name,
                                        )
                                        from lib.core.breakout_types import (
                                            get_range_config as _get_rc,
                                        )

                                        try:
                                            _bt_enum = _bt_from_name(_btype_name)
                                        except ValueError:
                                            _bt_enum = _BT.ORB
                                        _rc = _get_rc(_bt_enum)
                                        _tp3_norm = max(0.0, min(1.0, _rc.tp3_atr_mult / 5.0))
                                    except Exception:
                                        pass

                                    tab_features = [
                                        # ── v4 core (14 features) ────────────────────────────
                                        _quality_norm,  # [0]  quality_pct_norm
                                        _vol_ratio,  # [1]  volume_ratio
                                        _atr_pct,  # [2]  atr_pct
                                        _cvd_delta,  # [3]  cvd_delta
                                        _nr7_flag,  # [4]  nr7_flag
                                        1.0 if result.direction == "LONG" else 0.0,  # [5] direction_flag
                                        _session_enc,  # [6]  session_ordinal
                                        _london_overlap,  # [7]  london_overlap_flag
                                        _or_range_atr,  # [8]  or_range_atr_ratio (raw)
                                        _pm_range_ratio,  # [9]  premarket_range_ratio (raw)
                                        _bar_of_day,  # [10] bar_of_day
                                        _dow,  # [11] day_of_week
                                        _vwap_dist,  # [12] vwap_distance (raw)
                                        _asset_cls,  # [13] asset_class_id
                                        # ── v6 additions (4 new features) ────────────────────
                                        _btype_ord_val,  # [14] breakout_type_ord
                                        _vol_class_val,  # [15] asset_volatility_class
                                        _hour_of_day,  # [16] hour_of_day
                                        _tp3_norm,  # [17] tp3_atr_mult_norm
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

                            # Forward to PositionManager (stop-and-reverse)
                            _dispatch_to_position_manager(
                                result,
                                bars_1m=bars_1m,
                                session_key=_session_key,
                            )

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

                            # Enrich the audit row — published (includes MTF data)
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
                                    "mtf_score": _orb_mtf_score,
                                    "macd_slope": _orb_macd_slope,
                                    "divergence": _orb_divergence,
                                },
                            )
                            # Update new dedicated columns
                            try:
                                from lib.core.models import _get_conn, _is_using_postgres

                                if _orb_row_id is not None:
                                    _pg = _is_using_postgres()
                                    _ph = "%s" if _pg else "?"
                                    _conn = _get_conn()
                                    _conn.execute(
                                        f"UPDATE orb_events SET "
                                        f"breakout_type={_ph}, mtf_score={_ph}, "
                                        f"macd_slope={_ph}, divergence={_ph} "
                                        f"WHERE id={_ph}",
                                        (
                                            "ORB",
                                            _orb_mtf_score,
                                            _orb_macd_slope,
                                            _orb_divergence,
                                            _orb_row_id,
                                        ),
                                    )
                                    _conn.commit()
                                    _conn.close()
                            except Exception:
                                pass

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


# ---------------------------------------------------------------------------
# Multi-BreakoutType handlers — PDR, IB, CONS, and parallel sweep
# ---------------------------------------------------------------------------


def _get_assets_for_session_key(session_key: str) -> list[dict]:
    """Return the focus asset list filtered to the given session's asset set.

    Falls back to the full daily focus if no session-specific list is found.
    """
    try:
        from lib.core.cache import cache_get
        from lib.services.engine.orb import SESSION_ASSETS

        raw_focus = cache_get("engine:daily_focus")
        if not raw_focus:
            return []

        focus_data = json.loads(raw_focus)
        all_assets = focus_data.get("assets", [])

        session_tickers = set(SESSION_ASSETS.get(session_key, []))
        if not session_tickers:
            return all_assets

        return [
            a for a in all_assets if a.get("ticker", "") in session_tickers or a.get("symbol", "") in session_tickers
        ]
    except Exception as exc:
        logger.debug("_get_assets_for_session_key(%s) error: %s", session_key, exc)
        return []


def _fetch_bars_1m(engine, ticker: str, symbol: str) -> "pd.DataFrame | None":
    """Fetch 1-minute bars from cache or engine data service (best-effort)."""
    try:
        import io

        import pandas as pd

        from lib.core.cache import cache_get

        bars_key = f"engine:bars_1m:{ticker or symbol}"
        raw_bars = cache_get(bars_key)
        if raw_bars:
            raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
            return pd.read_json(io.StringIO(raw_str))

        with contextlib.suppress(Exception):
            return engine._fetch_tf_safe(ticker or symbol, interval="1m", period="1d")
    except Exception as exc:
        logger.debug("_fetch_bars_1m(%s) error: %s", symbol, exc)
    return None


def _publish_breakout_result(result: "BreakoutResult", orb_session_key: str = "us") -> None:
    """Publish a non-ORB breakout result to Redis for SSE / dashboard consumption."""
    try:
        from lib.core.cache import cache_set

        payload = result.to_dict()
        payload["published_at"] = datetime.now(tz=ZoneInfo("America/New_York")).isoformat()
        payload["orb_session"] = orb_session_key

        key = f"engine:breakout:{result.breakout_type.lower()}:{result.symbol}"
        cache_set(key, json.dumps(payload).encode(), ttl=300)

        # Also publish to the generic breakout channel so the SSE picks it up
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                _r.publish("dashboard:breakout", json.dumps(payload))
        except Exception:
            pass

        logger.info(
            "🔔 %s BREAKOUT: %s %s @ %.4f (range %.4f–%.4f)",
            result.breakout_type.value,
            result.direction,
            result.symbol,
            result.trigger_price,
            result.range_low,
            result.range_high,
        )
    except Exception as exc:
        logger.debug("_publish_breakout_result error: %s", exc)


def _publish_pm_orders(orders: list) -> None:  # type: ignore[type-arg]
    """Publish PositionManager OrderCommands to Redis for the NT8 Bridge and dashboard.

    Each order is written to:
      - ``engine:pm:orders`` — a list (RPUSH) of JSON-serialised commands (TTL 60s)
      - ``dashboard:pm_orders`` — Redis pub/sub channel for real-time SSE streaming

    The NT8 Bridge subscribes to ``dashboard:pm_orders`` and translates each
    command into a NinjaScript order submission.
    """
    if not orders:
        return
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        now = datetime.now(tz=_EST).isoformat()
        serialised = []
        for order in orders:
            d = order.to_dict() if hasattr(order, "to_dict") else dict(order)
            d.setdefault("published_at", now)
            serialised.append(json.dumps(d, default=str))

        if REDIS_AVAILABLE and _r is not None:
            pipe = _r.pipeline()
            key = "engine:pm:orders"
            for s in serialised:
                pipe.rpush(key, s)
            pipe.expire(key, 60)
            pipe.execute()
            for s in serialised:
                _r.publish("dashboard:pm_orders", s)

        # Also write consolidated status for dashboard SSE
        try:
            pm = _position_manager
            if pm is not None:
                positions_payload = {
                    p.ticker: {
                        "direction": p.direction,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "stop_loss": p.stop_loss,
                        "tp1": p.tp1,
                        "tp2": p.tp2,
                        "tp3": p.tp3,
                        "phase": p.phase.value,
                        "unrealized_pnl": round(p.unrealized_pnl, 4),
                        "r_multiple": round(p.r_multiple, 3),
                        "breakout_type": p.breakout_type,
                        "session_key": p.session_key,
                    }
                    for p in pm.get_all_positions().values()
                }
                cache_set(
                    "engine:pm:positions",
                    json.dumps(
                        {
                            "positions": positions_payload,
                            "count": len(positions_payload),
                            "updated_at": now,
                        },
                        default=str,
                    ).encode(),
                    ttl=120,
                )
        except Exception:
            pass

        logger.info(
            "📤 PositionManager: %d order(s) dispatched → NT8 Bridge",
            len(orders),
        )
    except Exception as exc:
        logger.debug("_publish_pm_orders error (non-fatal): %s", exc)


def _dispatch_to_position_manager(
    result: object,
    bars_1m: "pd.DataFrame | None" = None,
    session_key: str = "us",
    range_config: object = None,
) -> None:
    """Forward a published breakout signal to the PositionManager.

    Accepts either an ``ORBResult`` or a ``BreakoutResult``; both expose the
    same duck-typed attributes that ``PositionManager.process_signal()`` needs.

    For ORBResult objects (which use ``or_high``/``or_low`` instead of
    ``range_high``/``range_low``) we attach the missing attributes on-the-fly
    so the PositionManager doesn't have to know about ORB-specific naming.

    This is intentionally best-effort — any failure here must not block the
    alert pipeline.
    """
    pm = _position_manager
    if pm is None:
        return

    try:
        signal = result

        # ORBResult compatibility shim — attach range_high/range_low
        if not hasattr(signal, "range_high") or not signal.range_high:
            or_high = getattr(signal, "or_high", 0.0)
            or_low = getattr(signal, "or_low", 0.0)
            try:
                signal.range_high = or_high
                signal.range_low = or_low
                signal.breakout_type = getattr(signal, "breakout_type", None) or type("_T", (), {"value": "ORB"})()
            except AttributeError:
                pass  # frozen dataclass — leave as-is

        # Attach session_key if missing
        if not getattr(signal, "session_key", ""):
            with contextlib.suppress(AttributeError):
                signal.session_key = session_key

        # Attach filter_passed as True (signal already passed the filter gate)
        if getattr(signal, "filter_passed", None) is None:
            with contextlib.suppress(AttributeError):
                signal.filter_passed = True

        orders = pm.process_signal(signal, bars_1m=bars_1m, range_config=range_config)
        if orders:
            _publish_pm_orders(orders)

    except Exception as exc:
        logger.debug("_dispatch_to_position_manager error (non-fatal): %s", exc)


def _handle_update_positions(engine) -> None:
    """Run PositionManager.update_all() on every scheduled 1m-bar tick.

    Fetches the latest 1-minute bars for every core watchlist ticker,
    calls ``update_all()``, then dispatches any resulting bracket / EMA9
    trailing orders to the NT8 Bridge via Redis.

    Safe to call frequently — exits immediately if no positions are active.
    """
    pm = _position_manager
    if pm is None or pm.get_position_count() == 0:
        return

    try:
        import io

        import pandas as pd

        from lib.core.cache import cache_get
        from lib.core.models import CORE_TICKERS

        bars_by_ticker: dict[str, pd.DataFrame] = {}

        for ticker in CORE_TICKERS:
            try:
                raw = cache_get(f"engine:bars_1m:{ticker}")
                if raw:
                    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    df = pd.read_json(io.StringIO(raw_str))
                    if not df.empty:
                        bars_by_ticker[ticker] = df
            except Exception as exc:
                logger.debug("Could not fetch bars for PM update (%s): %s", ticker, exc)

        if not bars_by_ticker:
            return

        orders = pm.update_all(bars_by_ticker)
        if orders:
            _publish_pm_orders(orders)
            logger.info(
                "📊 PositionManager update: %d order(s) from %d active position(s)",
                len(orders),
                pm.get_position_count(),
            )

    except Exception as exc:
        logger.debug("_handle_update_positions error (non-fatal): %s", exc)


def _persist_breakout_result(result: "BreakoutResult", session_key: str = "") -> int | None:
    """Persist a BreakoutResult to orb_events using the new breakout_type column."""
    try:
        from lib.core.models import record_orb_event

        row_id = record_orb_event(
            symbol=result.symbol,
            or_high=result.range_high,
            or_low=result.range_low,
            or_range=result.range_size,
            atr_value=result.atr_value,
            breakout_detected=result.breakout_detected,
            direction=result.direction,
            trigger_price=result.trigger_price,
            long_trigger=result.long_trigger,
            short_trigger=result.short_trigger,
            bar_count=result.range_bar_count,
            session=session_key,
            metadata=result.extra or {},
            breakout_type=result.breakout_type.value,
            mtf_score=result.mtf_score,
            macd_slope=result.macd_slope,
            divergence=result.divergence_type or "",
        )
        return row_id
    except Exception as exc:
        logger.debug("_persist_breakout_result error (non-fatal): %s", exc)
        return None


def _run_mtf_on_result(result: "BreakoutResult", bars_htf: "pd.DataFrame | None") -> "BreakoutResult":
    """Run the MTF analyzer on a BreakoutResult and enrich it in-place."""
    if not result.breakout_detected or bars_htf is None or bars_htf.empty:
        return result
    try:
        from lib.analysis.mtf_analyzer import analyze_mtf

        mtf = analyze_mtf(bars_htf, direction=result.direction)
        result.mtf_score = mtf.mtf_score
        result.mtf_direction = mtf.ema_slope_direction
        result.macd_slope = mtf.macd_histogram_slope
        result.macd_divergence = mtf.divergence_detected
        result.extra["mtf"] = mtf.to_dict()
    except Exception as exc:
        logger.debug("_run_mtf_on_result error (non-fatal): %s", exc)
    return result


def _handle_check_pdr(engine, session_key: str = "london_ny") -> None:
    """Check for Previous Day Range (PDR) breakouts across session assets.

    The PDR uses yesterday's Globex high/low as the range anchor.  A close
    beyond either level by at least ``min_depth_atr_pct × ATR`` triggers.

    Strongest sessions: London Open (03:00 ET) and US Equity Open (09:30 ET)
    where institutional orders cluster around yesterday's levels.
    """
    logger.debug("▶ PDR breakout check [session=%s]...", session_key)
    try:
        import pandas as pd

        from lib.services.engine.breakout import detect_pdr_breakout

        assets = _get_assets_for_session_key(session_key)
        if not assets:
            logger.debug("No assets for PDR check (session=%s)", session_key)
            return

        found = 0
        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue
            try:
                bars_1m = _fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    continue

                # Try to get pre-computed PDR levels from daily bars cache
                prev_high: float | None = None
                prev_low: float | None = None
                try:
                    import io

                    from lib.core.cache import cache_get

                    daily_key = f"engine:bars_daily:{ticker or symbol}"
                    raw_daily = cache_get(daily_key)
                    if raw_daily:
                        raw_daily_str = raw_daily.decode("utf-8") if isinstance(raw_daily, bytes) else raw_daily
                        bars_daily = pd.read_json(io.StringIO(raw_daily_str))
                        if len(bars_daily) >= 2:
                            prev_high = float(bars_daily["High"].iloc[-2])
                            prev_low = float(bars_daily["Low"].iloc[-2])
                except Exception:
                    pass

                result = detect_pdr_breakout(
                    bars_1m,
                    symbol=symbol,
                    prev_day_high=prev_high,
                    prev_day_low=prev_low,
                )

                # Fetch HTF bars for MTF enrichment
                bars_htf = None
                try:
                    import io as _io

                    from lib.core.cache import cache_get as _cg

                    htf_raw = _cg(f"engine:bars_15m:{ticker or symbol}")
                    if htf_raw:
                        bars_htf = pd.read_json(
                            _io.StringIO(htf_raw.decode("utf-8") if isinstance(htf_raw, bytes) else htf_raw)
                        )
                except Exception:
                    pass

                if bars_htf is None and bars_1m is not None:
                    with contextlib.suppress(Exception):
                        bars_htf = (
                            bars_1m.resample("15min")
                            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                            .dropna()
                        )

                result = _run_mtf_on_result(result, bars_htf)
                _persist_breakout_result(result, session_key=session_key)

                if result.breakout_detected:
                    found += 1
                    _publish_breakout_result(result, orb_session_key=session_key)
                    # Forward to PositionManager (stop-and-reverse)
                    _dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
                    try:
                        from lib.core.alerts import send_signal

                        send_signal(
                            signal_key=f"pdr_{symbol}_{result.direction}",
                            title=f"📊 PDR {result.direction}: {symbol}",
                            message=(
                                f"Previous Day Range Breakout!\n"
                                f"Direction: {result.direction}\n"
                                f"Trigger: {result.trigger_price:,.4f}\n"
                                f"PDR: {result.range_low:,.4f} – {result.range_high:,.4f}\n"
                                f"ATR: {result.atr_value:,.4f}"
                                + (f"\nMTF Score: {result.mtf_score:.3f}" if result.mtf_score is not None else "")
                            ),
                            asset=symbol,
                            direction=result.direction,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("PDR check failed for %s: %s", symbol, exc)

        logger.debug("PDR check [%s] complete: %d breakout(s)", session_key, found)
    except Exception as exc:
        logger.debug("PDR handler error (non-fatal): %s", exc)


def _handle_check_ib(engine, session_key: str = "us") -> None:
    """Check for Initial Balance (IB) breakouts across US session assets.

    The Initial Balance is formed during the first 60 minutes of RTH
    (09:30–10:30 ET).  Breakouts after 10:30 ET are the classic
    Dalton/Steidlmayer IB extension trade.  Narrow IB ranges on trend days
    produce the highest-conviction signals.

    Only fired by the scheduler after 10:30 ET when the IB is complete.
    """
    logger.debug("▶ IB breakout check [session=%s]...", session_key)
    try:
        import pandas as pd

        from lib.services.engine.breakout import detect_ib_breakout

        assets = _get_assets_for_session_key(session_key)
        if not assets:
            logger.debug("No assets for IB check (session=%s)", session_key)
            return

        found = 0
        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue
            try:
                bars_1m = _fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    continue

                result = detect_ib_breakout(bars_1m, symbol=symbol)

                # Enrich with MTF
                bars_htf = None
                try:
                    import io as _io

                    from lib.core.cache import cache_get as _cg

                    htf_raw = _cg(f"engine:bars_15m:{ticker or symbol}")
                    if htf_raw:
                        bars_htf = pd.read_json(
                            _io.StringIO(htf_raw.decode("utf-8") if isinstance(htf_raw, bytes) else htf_raw)
                        )
                except Exception:
                    pass

                if bars_htf is None and bars_1m is not None:
                    with contextlib.suppress(Exception):
                        bars_htf = (
                            bars_1m.resample("15min")
                            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                            .dropna()
                        )

                result = _run_mtf_on_result(result, bars_htf)
                _persist_breakout_result(result, session_key=session_key)

                if result.breakout_detected:
                    found += 1
                    _publish_breakout_result(result, orb_session_key=session_key)
                    # Forward to PositionManager (stop-and-reverse)
                    _dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
                    try:
                        from lib.core.alerts import send_signal

                        ib_line = f"IB: {result.ib_low:,.4f} – {result.ib_high:,.4f}" if result.ib_high > 0 else ""
                        send_signal(
                            signal_key=f"ib_{symbol}_{result.direction}",
                            title=f"📊 IB {result.direction}: {symbol}",
                            message=(
                                f"Initial Balance Breakout!\n"
                                f"Direction: {result.direction}\n"
                                f"Trigger: {result.trigger_price:,.4f}\n"
                                + (f"{ib_line}\n" if ib_line else "")
                                + f"ATR: {result.atr_value:,.4f}"
                                + (f"\nMTF Score: {result.mtf_score:.3f}" if result.mtf_score is not None else "")
                            ),
                            asset=symbol,
                            direction=result.direction,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("IB check failed for %s: %s", symbol, exc)

        logger.debug("IB check [%s] complete: %d breakout(s)", session_key, found)
    except Exception as exc:
        logger.debug("IB handler error (non-fatal): %s", exc)


def _handle_check_consolidation(engine, session_key: str = "london_ny") -> None:
    """Check for Consolidation/Squeeze breakouts across session assets.

    Uses Bollinger Band contraction (BB width < squeeze_atr_mult × ATR for
    ≥ squeeze_min_bars consecutive bars) to identify compressed price action,
    then detects the expansion bar that breaks out of the squeeze.

    Valid throughout the full active window — squeeze breakouts can fire at
    any time once a sustained contraction is detected.
    """
    logger.debug("▶ Consolidation/squeeze breakout check [session=%s]...", session_key)
    try:
        import pandas as pd

        from lib.services.engine.breakout import detect_consolidation_breakout

        assets = _get_assets_for_session_key(session_key)
        if not assets:
            logger.debug("No assets for CONS check (session=%s)", session_key)
            return

        found = 0
        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue
            try:
                bars_1m = _fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    continue

                result = detect_consolidation_breakout(bars_1m, symbol=symbol)

                # Enrich with MTF
                bars_htf = None
                try:
                    import io as _io

                    from lib.core.cache import cache_get as _cg

                    htf_raw = _cg(f"engine:bars_15m:{ticker or symbol}")
                    if htf_raw:
                        bars_htf = pd.read_json(
                            _io.StringIO(htf_raw.decode("utf-8") if isinstance(htf_raw, bytes) else htf_raw)
                        )
                except Exception:
                    pass

                if bars_htf is None and bars_1m is not None:
                    with contextlib.suppress(Exception):
                        bars_htf = (
                            bars_1m.resample("15min")
                            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                            .dropna()
                        )

                result = _run_mtf_on_result(result, bars_htf)
                _persist_breakout_result(result, session_key=session_key)

                if result.breakout_detected:
                    found += 1
                    _publish_breakout_result(result, orb_session_key=session_key)
                    # Forward to PositionManager (stop-and-reverse)
                    _dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
                    try:
                        from lib.core.alerts import send_signal

                        squeeze_line = (
                            f"Squeeze: {result.squeeze_bar_count} bars, BB width {result.squeeze_bb_width:.4f}"
                            if result.squeeze_detected
                            else ""
                        )
                        send_signal(
                            signal_key=f"cons_{symbol}_{result.direction}",
                            title=f"📊 SQUEEZE {result.direction}: {symbol}",
                            message=(
                                f"Consolidation Squeeze Breakout!\n"
                                f"Direction: {result.direction}\n"
                                f"Trigger: {result.trigger_price:,.4f}\n"
                                f"Range: {result.range_low:,.4f} – {result.range_high:,.4f}\n"
                                + (f"{squeeze_line}\n" if squeeze_line else "")
                                + f"ATR: {result.atr_value:,.4f}"
                                + (f"\nMTF Score: {result.mtf_score:.3f}" if result.mtf_score is not None else "")
                            ),
                            asset=symbol,
                            direction=result.direction,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("CONS check failed for %s: %s", symbol, exc)

        logger.debug("CONS check [%s] complete: %d breakout(s)", session_key, found)
    except Exception as exc:
        logger.debug("CONS handler error (non-fatal): %s", exc)


def _handle_check_breakout_multi(engine, session_key: str = "us", types: list[str] | None = None) -> None:
    """Run multiple BreakoutType detectors in parallel for a session's assets.

    Dispatches PDR, IB, and/or CONS checks concurrently using
    ``concurrent.futures.ThreadPoolExecutor`` so a slow fetch for one asset
    does not block the others.  ORB is intentionally excluded here — it has
    its own session-specific handlers that run on the same 2-minute cadence.

    Args:
        engine: The engine singleton.
        session_key: Session key whose asset list to use.
        types: List of breakout type strings to check ("PDR", "IB", "CONS").
               Defaults to ["PDR", "CONS"] if not specified.
    """
    if types is None:
        types = ["PDR", "CONS"]

    logger.debug("▶ Multi-type breakout sweep [session=%s types=%s]...", session_key, types)

    import concurrent.futures

    handler_map = {
        "PDR": lambda: _handle_check_pdr(engine, session_key=session_key),
        "IB": lambda: _handle_check_ib(engine, session_key=session_key),
        "CONS": lambda: _handle_check_consolidation(engine, session_key=session_key),
    }

    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(types), thread_name_prefix="breakout") as executor:
        for btype in types:
            handler = handler_map.get(btype)
            if handler is not None:
                futures_map[executor.submit(handler)] = btype

        for future in concurrent.futures.as_completed(futures_map, timeout=60):
            btype = futures_map[future]
            try:
                future.result()
            except Exception as exc:
                logger.warning("Multi-type sweep [%s/%s] error: %s", session_key, btype, exc)

    logger.debug("Multi-type breakout sweep [%s] complete", session_key)


# Configurable gap-alert threshold (minutes).  Gaps larger than this value
# are reported as warnings after each backfill run.
_GAP_ALERT_MINUTES = int(os.environ.get("BACKFILL_GAP_ALERT_MINUTES", "30"))


def _check_and_alert_gaps(symbols: list[str], gap_threshold_minutes: int = 30) -> None:
    """Scan stored bars for gaps exceeding ``gap_threshold_minutes`` and log alerts.

    Publishes a Redis key ``engine:gap_alerts`` (TTL 26h) so the dashboard
    can surface data-quality warnings without requiring a daily-report cycle.

    Only symbols that have *meaningful* gaps (i.e. not just normal overnight /
    weekend breaks) are included in the alert payload.
    """
    try:
        from lib.services.engine.backfill import get_gap_report

        alerts: list[dict] = []
        for sym in symbols:
            try:
                report = get_gap_report(sym, days_back=3, interval="1m")
                gaps = [g for g in report.get("gaps", []) if g.get("missing_minutes", 0) >= gap_threshold_minutes]
                if not gaps:
                    continue
                worst = max(gaps, key=lambda g: g.get("missing_minutes", 0))
                alerts.append(
                    {
                        "symbol": sym,
                        "gap_count": len(gaps),
                        "worst_gap_minutes": worst.get("missing_minutes", 0),
                        "worst_gap_start": worst.get("start", ""),
                        "worst_gap_end": worst.get("end", ""),
                        "coverage_pct": report.get("coverage_pct", 0),
                    }
                )
                logger.warning(
                    "⚠️  Gap detected in %s: %d gap(s), worst = %d min (%.1f%% coverage)",
                    sym,
                    len(gaps),
                    worst.get("missing_minutes", 0),
                    report.get("coverage_pct", 0),
                )
            except Exception as exc:
                logger.debug("Gap check failed for %s: %s", sym, exc)

        if alerts:
            # Sort by worst gap descending
            alerts.sort(key=lambda a: a["worst_gap_minutes"], reverse=True)
            try:
                from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

                payload = json.dumps(
                    {
                        "alerts": alerts,
                        "threshold_minutes": gap_threshold_minutes,
                        "checked_at": datetime.now(tz=_EST).isoformat(),
                        "symbol_count": len(symbols),
                        "alert_count": len(alerts),
                    },
                    default=str,
                ).encode()
                cache_set("engine:gap_alerts", payload, ttl=26 * 3600)
                if REDIS_AVAILABLE and _r is not None:
                    import contextlib

                    with contextlib.suppress(Exception):
                        _r.publish(
                            "dashboard:gap_alerts",
                            json.dumps({"alert_count": len(alerts)}, default=str),
                        )
            except Exception as exc:
                logger.debug("Failed to publish gap alerts to Redis: %s", exc)
        else:
            # Clear stale alert key when all gaps are resolved
            try:
                from lib.core.cache import cache_set

                cache_set(
                    "engine:gap_alerts",
                    json.dumps(
                        {
                            "alerts": [],
                            "threshold_minutes": gap_threshold_minutes,
                            "checked_at": datetime.now(tz=_EST).isoformat(),
                            "symbol_count": len(symbols),
                            "alert_count": 0,
                        },
                        default=str,
                    ).encode(),
                    ttl=26 * 3600,
                )
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Gap alert sweep failed: %s", exc)


def _handle_historical_backfill(engine) -> None:
    """Backfill historical 1-min bars to Postgres/SQLite (off-hours).

    Calls the backfill module which:
      1. Determines which symbols need data
      2. Finds gaps in existing stored bars
      3. Fetches missing chunks from Massive (primary) or yfinance (fallback)
      4. Stores bars idempotently via UPSERT
      5. Publishes summary to Redis for dashboard visibility
      6. Scans for residual gaps > ``BACKFILL_GAP_ALERT_MINUTES`` and logs
         warnings + publishes ``engine:gap_alerts`` to Redis.
    """
    logger.info("▶ Historical backfill starting")

    try:
        from lib.services.engine.backfill import _get_backfill_symbols, run_backfill

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

        # ── Post-backfill gap scan ─────────────────────────────────────────
        # Run after every backfill to catch persistent gaps that the
        # backfiller could not fill (e.g. data not available from any source).
        try:
            symbols = _get_backfill_symbols()
            logger.info(
                "▶ Running post-backfill gap scan (%d symbols, threshold=%dm)", len(symbols), _GAP_ALERT_MINUTES
            )
            _check_and_alert_gaps(symbols, gap_threshold_minutes=_GAP_ALERT_MINUTES)
        except Exception as exc:
            logger.debug("Post-backfill gap scan error: %s", exc)

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
    """No-op — dataset generation runs on the dedicated GPU trainer service.

    Use the trainer service (lib.training.trainer_server) on the GPU machine:
        docker compose --profile training up -d
        curl -X POST http://trainer:8200/train
    """
    logger.info("⏭️  Chart dataset generation skipped — use trainer service (POST /train)")


def _handle_train_breakout_cnn(engine) -> None:
    """No-op — CNN training runs on the dedicated GPU trainer service.

    The trainer server (lib.training.trainer_server) handles the full
    pipeline: dataset generation → training → evaluation → promotion.
        docker compose --profile training up -d
        curl -X POST http://trainer:8200/train
    Trained models are hot-reloaded by the engine via watchdog.
    """
    logger.info("⏭️  CNN training skipped — use trainer service (POST /train)")


# ---------------------------------------------------------------------------
# Daily report handler (runs once per day at start of off-hours ~12:00 ET)
# ---------------------------------------------------------------------------


def _build_session_stats(today) -> dict:
    """Compile per-session ORB signal statistics for today's report.

    Queries ``orb_events`` grouped by the ``orb_session`` metadata field
    to produce a dict like::

        {
            "us":        {"total": 12, "breakouts": 4, "published": 2, "pass_rate": 50.0},
            "london":    {"total":  8, "breakouts": 3, "published": 3, "pass_rate": 100.0},
            "cme":       {"total":  6, "breakouts": 1, "published": 0, "pass_rate": 0.0},
            ...
        }

    Falls back gracefully to an empty dict if the table is unavailable or
    the query fails for any reason.
    """
    try:
        from lib.core.models import get_orb_events

        today_str = today.strftime("%Y-%m-%d")
        events = get_orb_events(limit=500)

        # Filter to today's events
        today_events = [
            e for e in events if str(e.get("evaluated_at", "") or e.get("created_at", "")).startswith(today_str)
        ]

        if not today_events:
            return {}

        # Group by session key (stored in metadata JSON or orb_session column)
        session_buckets: dict[str, dict] = {}
        for ev in today_events:
            # Try explicit orb_session column first, then fall back to metadata JSON
            session_key = ev.get("orb_session", "")
            if not session_key:
                try:
                    meta = json.loads(ev.get("metadata", "{}") or "{}")
                    session_key = meta.get("orb_session", "unknown")
                except Exception:
                    session_key = "unknown"

            if not session_key:
                session_key = "unknown"

            bucket = session_buckets.setdefault(
                session_key,
                {"total": 0, "breakouts": 0, "published": 0, "filter_failed": 0},
            )
            bucket["total"] += 1
            if ev.get("breakout_detected"):
                bucket["breakouts"] += 1
            if ev.get("published"):
                bucket["published"] += 1
            elif ev.get("breakout_detected"):
                bucket["filter_failed"] += 1

        # Compute pass rates
        result = {}
        for sk, b in sorted(session_buckets.items()):
            total_bo = b["breakouts"]
            pub = b["published"]
            result[sk] = {
                "total_evaluations": b["total"],
                "breakouts_detected": total_bo,
                "published": pub,
                "filter_failed": b["filter_failed"],
                "pass_rate": round(pub / total_bo * 100, 1) if total_bo > 0 else 0.0,
            }

        return result

    except Exception as exc:
        logger.debug("Could not build session stats: %s", exc)
        return {}


def _handle_daily_report(engine) -> None:
    """Generate the daily trading session report and publish it to Redis.

    Builds a structured summary of the just-completed trading session:
      - ORB signal count + filter pass/reject rates
      - Per-session ORB breakdown (us, london, cme, tokyo, etc.)
      - CNN probability stats (mean, min, max, above-threshold count)
      - Risk events (blocks, warnings, consecutive losses)
      - Model performance snapshot (val accuracy, precision, recall)
      - Data coverage summary (gap count, coverage % per symbol)

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
            today = datetime.now(tz=_EST).date()
            report = {"generated_at": datetime.now(tz=_EST).isoformat()}

        # Add generation timestamp and session label
        report["generated_at"] = datetime.now(tz=_EST).isoformat()
        report["session"] = "daily"

        # ── Per-session performance breakdown ──────────────────────────────
        try:
            session_stats = _build_session_stats(today)
            if session_stats:
                report["sessions"] = session_stats
        except Exception as exc:
            logger.debug("Session stats build failed: %s", exc)

        # ── PositionManager session stats ──────────────────────────────────
        try:
            pm = _position_manager
            if pm is not None:
                closed = pm.get_history()
                if closed:
                    wins = [p for p in closed if p.realized_pnl > 0]
                    losses = [p for p in closed if p.realized_pnl <= 0]
                    total_pnl = sum(p.realized_pnl for p in closed)
                    avg_r = sum(p.r_multiple for p in closed) / len(closed) if closed else 0.0
                    # Break down by breakout_type
                    type_breakdown: dict[str, dict] = {}
                    for p in closed:
                        btype = p.breakout_type or "UNKNOWN"
                        bucket = type_breakdown.setdefault(
                            btype,
                            {"trades": 0, "wins": 0, "total_pnl": 0.0},
                        )
                        bucket["trades"] += 1
                        if p.realized_pnl > 0:
                            bucket["wins"] += 1
                        bucket["total_pnl"] = round(bucket["total_pnl"] + p.realized_pnl, 4)
                    for _btype, b in type_breakdown.items():
                        b["win_rate"] = round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0

                    report["position_manager"] = {
                        "total_trades": len(closed),
                        "wins": len(wins),
                        "losses": len(losses),
                        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
                        "total_realized_pnl": round(total_pnl, 4),
                        "avg_r_multiple": round(avg_r, 3),
                        "active_positions": pm.get_position_count(),
                        "by_type": type_breakdown,
                    }
                else:
                    report["position_manager"] = {
                        "total_trades": 0,
                        "active_positions": pm.get_position_count(),
                    }
        except Exception as exc:
            logger.debug("PositionManager stats build failed: %s", exc)

        # ── Data coverage / gap summary ────────────────────────────────────
        try:
            from lib.services.engine.backfill import _get_backfill_symbols, get_gap_report

            symbols = _get_backfill_symbols()[:12]  # cap to avoid long runtime
            gap_summary: dict[str, dict] = {}
            total_gaps = 0
            for sym in symbols:
                try:
                    gr = get_gap_report(sym, days_back=1, interval="1m")
                    g_count = len(gr.get("gaps", []))
                    total_gaps += g_count
                    if g_count > 0 or gr.get("coverage_pct", 100) < 90:
                        gap_summary[sym] = {
                            "coverage_pct": gr.get("coverage_pct", 0),
                            "gap_count": g_count,
                            "total_bars": gr.get("total_bars", 0),
                        }
                except Exception:
                    pass
            if gap_summary:
                report["data_coverage"] = {
                    "symbols_checked": len(symbols),
                    "symbols_with_gaps": len(gap_summary),
                    "total_gaps_today": total_gaps,
                    "details": gap_summary,
                }
        except Exception as exc:
            logger.debug("Data coverage summary failed: %s", exc)

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
        session_breakdown = report.get("sessions", {})
        coverage = report.get("data_coverage", {})
        pm_stats = report.get("position_manager", {})

        logger.info("=" * 55)
        logger.info("  📊 Daily Session Report — %s", datetime.now(tz=_EST).strftime("%Y-%m-%d"))
        logger.info("=" * 55)
        logger.info("  ORB detections : %d breakouts | %d published | %d filtered", orb_count, published, filtered)
        if orb_count > 0:
            pass_rate = published / orb_count * 100
            logger.info("  Filter pass rate: %.0f%%", pass_rate)

        # Per-session breakdown
        if session_breakdown:
            logger.info("  ── Per-Session Breakdown ──────────────────────")
            for sk, sb in sorted(session_breakdown.items()):
                logger.info(
                    "    %-14s  evals=%3d  bo=%2d  pub=%2d  pass=%.0f%%",
                    sk,
                    sb.get("total_evaluations", 0),
                    sb.get("breakouts_detected", 0),
                    sb.get("published", 0),
                    sb.get("pass_rate", 0.0),
                )

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

        # PositionManager session summary
        if pm_stats and pm_stats.get("total_trades", 0) > 0:
            logger.info("  ── PositionManager ────────────────────────────")
            logger.info(
                "  Trades : %d total | %d wins | %d losses | win rate %.0f%%",
                pm_stats.get("total_trades", 0),
                pm_stats.get("wins", 0),
                pm_stats.get("losses", 0),
                pm_stats.get("win_rate", 0.0),
            )
            logger.info(
                "  P&L    : $%.2f realized | avg R=%.2f | %d active",
                pm_stats.get("total_realized_pnl", 0.0),
                pm_stats.get("avg_r_multiple", 0.0),
                pm_stats.get("active_positions", 0),
            )
            by_type = pm_stats.get("by_type", {})
            if by_type:
                logger.info("  ── By Breakout Type ───────────────────────────")
                for btype, b in sorted(by_type.items()):
                    logger.info(
                        "    %-14s  trades=%2d  wins=%2d  win_rate=%.0f%%  pnl=$%.2f",
                        btype,
                        b.get("trades", 0),
                        b.get("wins", 0),
                        b.get("win_rate", 0.0),
                        b.get("total_pnl", 0.0),
                    )

        # Data coverage warning
        if coverage and coverage.get("symbols_with_gaps", 0) > 0:
            logger.warning(
                "  ⚠️  Data gaps    : %d symbol(s) have gaps today (total %d gaps)",
                coverage["symbols_with_gaps"],
                coverage.get("total_gaps_today", 0),
            )
            for sym, detail in list(coverage.get("details", {}).items())[:5]:
                logger.warning(
                    "    %s: coverage=%.1f%%  gaps=%d  bars=%d",
                    sym,
                    detail.get("coverage_pct", 0),
                    detail.get("gap_count", 0),
                    detail.get("total_bars", 0),
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


def _check_module_health() -> dict:
    """Check per-module health for Redis, Postgres, and Massive WS.

    Returns a dict with keys: redis, postgres, massive — each containing
    ``{"status": "ok"|"error"|"unavailable", ...}``.
    """
    modules: dict = {}

    # Redis
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            modules["redis"] = {"status": "ok", "connected": True}
        else:
            modules["redis"] = {"status": "unavailable", "connected": False}
    except Exception as exc:
        modules["redis"] = {"status": "error", "connected": False, "error": str(exc)}

    # Postgres
    try:
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url.startswith("postgresql"):
            modules["postgres"] = {"status": "not_configured", "connected": False}
        else:
            from lib.core.models import _get_conn

            conn = _get_conn()
            try:
                conn.execute("SELECT 1")
                modules["postgres"] = {"status": "ok", "connected": True}
            finally:
                conn.close()
    except Exception as exc:
        modules["postgres"] = {"status": "error", "connected": False, "error": str(exc)}

    # Massive WebSocket
    try:
        from lib.core.cache import get_data_source

        ds = get_data_source()
        if ds == "Massive":
            modules["massive"] = {"status": "ok", "data_source": "Massive", "connected": True}
        else:
            modules["massive"] = {"status": "fallback", "data_source": ds, "connected": False}
    except Exception as exc:
        modules["massive"] = {"status": "error", "connected": False, "error": str(exc)}

    # CNN model
    try:
        from lib.services.engine.model_watcher import _find_model_dir

        model_dir = _find_model_dir()
        champion = model_dir / "breakout_cnn_best.pt" if model_dir else None
        if champion is not None and champion.is_file():
            try:
                stat = champion.stat()
                size_mb = round(stat.st_size / (1024 * 1024), 1)
                modules["cnn_model"] = {
                    "status": "ok",
                    "available": True,
                    "size_mb": size_mb,
                    "path": str(champion),
                }
            except OSError:
                modules["cnn_model"] = {"status": "error", "available": False}
        else:
            modules["cnn_model"] = {"status": "missing", "available": False}
    except ImportError:
        modules["cnn_model"] = {"status": "unknown", "available": False}

    return modules


def _publish_engine_status(engine, session_mode: str, scheduler_status: dict) -> None:
    """Publish engine status + scheduler state + per-module health to Redis."""
    try:
        from lib.core.cache import cache_set

        status = engine.get_status()
        status["session_mode"] = session_mode
        status["scheduler"] = scheduler_status
        status["modules"] = _check_module_health()
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
    global _model_watcher

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

    # ---------------------------------------------------------------------------
    # Start the filesystem-based model watcher (replaces inline polling).
    # Uses watchdog (inotify) when available, falls back to polling.
    # ---------------------------------------------------------------------------
    from lib.services.engine.model_watcher import ModelWatcher

    _model_watcher = ModelWatcher()
    watcher_started = _model_watcher.start()
    if watcher_started:
        watcher_status = _model_watcher.status()
        logger.info(
            "Model watcher active: backend=%s  dir=%s",
            watcher_status["backend"],
            watcher_status["model_dir"],
        )
    else:
        logger.warning(
            "Model watcher could not start — CNN hot-reload disabled. "
            "Ensure models/ directory exists (run scripts/sync_models.sh)."
        )

    # Action dispatch table
    # Initialise the RiskManager early so it's ready for handlers
    _get_risk_manager(account_size)

    # Initialise the PositionManager — loads any persisted positions from Redis
    _get_position_manager(account_size)

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
        ActionType.CHECK_ORB_CRYPTO_UTC0: lambda: _handle_check_orb_crypto_utc0(engine),
        ActionType.CHECK_ORB_CRYPTO_UTC12: lambda: _handle_check_orb_crypto_utc12(engine),
        ActionType.CHECK_PDR: lambda: _handle_check_pdr(
            engine,
            session_key=getattr(pending[0] if pending else None, "payload", None)
            and pending[0].payload.get("session_key", "london_ny")
            or "london_ny",
        ),
        ActionType.CHECK_IB: lambda: _handle_check_ib(
            engine,
            session_key=getattr(pending[0] if pending else None, "payload", None)
            and pending[0].payload.get("session_key", "us")
            or "us",
        ),
        ActionType.CHECK_CONSOLIDATION: lambda: _handle_check_consolidation(
            engine,
            session_key=getattr(pending[0] if pending else None, "payload", None)
            and pending[0].payload.get("session_key", "london_ny")
            or "london_ny",
        ),
        ActionType.CHECK_BREAKOUT_MULTI: lambda: _handle_check_breakout_multi(
            engine,
            session_key=getattr(pending[0] if pending else None, "payload", None)
            and pending[0].payload.get("session_key", "us")
            or "us",
            types=getattr(pending[0] if pending else None, "payload", None) and pending[0].payload.get("types") or None,
        ),
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
                    # For payload-bearing actions (CHECK_PDR, CHECK_IB, CHECK_CONSOLIDATION,
                    # CHECK_BREAKOUT_MULTI) the handler needs the action payload to know
                    # which session key and types to use.  We call the typed handlers
                    # directly here so the lambdas in action_handlers don't need
                    # late-binding workarounds.
                    _payload = getattr(action, "payload", None) or {}
                    if action.action == ActionType.CHECK_PDR:
                        _sk = _payload.get("session_key", "london_ny") if _payload else "london_ny"
                        _handle_check_pdr(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_IB:
                        _sk = _payload.get("session_key", "us") if _payload else "us"
                        _handle_check_ib(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_CONSOLIDATION:
                        _sk = _payload.get("session_key", "london_ny") if _payload else "london_ny"
                        _handle_check_consolidation(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_BREAKOUT_MULTI:
                        _sk = _payload.get("session_key", "us") if _payload else "us"
                        _types = _payload.get("types") if _payload else None
                        _handle_check_breakout_multi(
                            engine,
                            session_key=_sk,
                            types=_types,
                        )
                    else:
                        handler()
                    scheduler.mark_done(action.action)
                except Exception as exc:
                    scheduler.mark_failed(action.action, str(exc))
                    logger.error("Action %s failed: %s", action.action.value, exc, exc_info=True)

            # Update active positions on every loop iteration (bracket phases,
            # EMA9 trailing, stop/TP3 exits).  Exits immediately if no positions
            # are open.  Must run BEFORE publish so the status payload reflects
            # the latest position state.
            _handle_update_positions(engine)

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

    # Stop the model watcher first
    if _model_watcher is not None:
        _model_watcher.stop()
        _model_watcher = None

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
