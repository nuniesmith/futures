"""
Generic Breakout Handler Pipeline — Phase 1D
=============================================
One handler function for all 13 breakout types, eliminating ~400 lines of
copy-paste from ``main.py``.

Public API::

    from lib.services.engine.handlers import handle_breakout_check

    # In scheduler / main.py:
    handle_breakout_check(engine, BreakoutType.PrevDay, session_key="london_ny")
    handle_breakout_check(engine, BreakoutType.InitialBalance, session_key="us")
    handle_breakout_check(engine, BreakoutType.Consolidation, session_key="london_ny")

Design:
  - Pure orchestration — no detection logic.  Delegates to
    ``detect_range_breakout()`` / ``detect_pdr_breakout()`` etc.
  - Shared helpers (``fetch_bars_1m``, ``get_htf_bars``, ``run_mtf_on_result``,
    ``persist_breakout_result``, ``publish_breakout_result``,
    ``send_breakout_alert``) extracted from ``main.py``.
  - Each call handles one breakout type for one session's asset list.
  - Thread-safe, no shared mutable state.
  - All errors are caught and logged — never raises into the scheduler.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pandas as pd

from lib.core.breakout_types import BreakoutType, get_range_config

logger = logging.getLogger("engine.handlers")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Alert message templates per breakout type
# ---------------------------------------------------------------------------

_ALERT_TEMPLATES: dict[BreakoutType, dict[str, str]] = {
    BreakoutType.ORB: {
        "emoji": "📊",
        "short": "ORB",
        "title": "Opening Range Breakout",
    },
    BreakoutType.PrevDay: {
        "emoji": "📊",
        "short": "PDR",
        "title": "Previous Day Range Breakout",
    },
    BreakoutType.InitialBalance: {
        "emoji": "📊",
        "short": "IB",
        "title": "Initial Balance Breakout",
    },
    BreakoutType.Consolidation: {
        "emoji": "📊",
        "short": "SQUEEZE",
        "title": "Consolidation Squeeze Breakout",
    },
    BreakoutType.Weekly: {
        "emoji": "📈",
        "short": "WEEKLY",
        "title": "Weekly Range Breakout",
    },
    BreakoutType.Monthly: {
        "emoji": "📈",
        "short": "MONTHLY",
        "title": "Monthly Range Breakout",
    },
    BreakoutType.Asian: {
        "emoji": "🌏",
        "short": "ASIAN",
        "title": "Asian Session Range Breakout",
    },
    BreakoutType.BollingerSqueeze: {
        "emoji": "💥",
        "short": "BBSQUEEZE",
        "title": "Bollinger Squeeze Breakout",
    },
    BreakoutType.ValueArea: {
        "emoji": "📊",
        "short": "VA",
        "title": "Value Area Breakout",
    },
    BreakoutType.InsideDay: {
        "emoji": "📦",
        "short": "INSIDE",
        "title": "Inside Day Breakout",
    },
    BreakoutType.GapRejection: {
        "emoji": "🔲",
        "short": "GAP",
        "title": "Gap Rejection Breakout",
    },
    BreakoutType.PivotPoints: {
        "emoji": "📍",
        "short": "PIVOT",
        "title": "Pivot Points Breakout",
    },
    BreakoutType.Fibonacci: {
        "emoji": "🔢",
        "short": "FIB",
        "title": "Fibonacci Retracement Breakout",
    },
}


def _get_alert_template(bt: BreakoutType) -> dict[str, str]:
    """Return alert template for a breakout type, with sensible fallback."""
    return _ALERT_TEMPLATES.get(
        bt,
        {
            "emoji": "📊",
            "short": bt.name.upper(),
            "title": f"{bt.name} Breakout",
        },
    )


# ===========================================================================
# Shared helper functions (extracted from main.py)
# ===========================================================================


def get_assets_for_session_key(session_key: str) -> list[dict[str, Any]]:
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
        all_assets: list[dict[str, Any]] = focus_data.get("assets", [])

        session_tickers = set(SESSION_ASSETS.get(session_key, []))
        if not session_tickers:
            return all_assets

        return [
            a for a in all_assets if a.get("ticker", "") in session_tickers or a.get("symbol", "") in session_tickers
        ]
    except Exception as exc:
        logger.debug("get_assets_for_session_key(%s) error: %s", session_key, exc)
        return []


def fetch_bars_1m(engine: Any, ticker: str, symbol: str) -> "pd.DataFrame | None":
    """Fetch 1-minute bars from cache or engine data service (best-effort)."""
    try:
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
        logger.debug("fetch_bars_1m(%s) error: %s", symbol, exc)
    return None


def get_htf_bars(
    bars_1m: "pd.DataFrame | None",
    ticker: str,
) -> "pd.DataFrame | None":
    """Get 15-minute bars for MTF enrichment — from cache or resampled from 1m.

    Returns ``None`` if no usable data is available.
    """
    import pandas as pd

    # Try cached 15-min bars first
    try:
        from lib.core.cache import cache_get

        htf_raw = cache_get(f"engine:bars_15m:{ticker}")
        if htf_raw:
            raw_str = htf_raw.decode("utf-8") if isinstance(htf_raw, bytes) else htf_raw
            return pd.read_json(io.StringIO(raw_str))
    except Exception:
        pass

    # Fall back to resampling 1-minute bars
    if bars_1m is not None and not bars_1m.empty:
        with contextlib.suppress(Exception):
            return (
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
    return None


def get_prev_day_levels(ticker: str, symbol: str) -> tuple[float | None, float | None]:
    """Try to get pre-computed previous-day high/low from the daily bars cache.

    Returns (prev_high, prev_low) or (None, None) if unavailable.
    """
    try:
        import pandas as pd

        from lib.core.cache import cache_get

        daily_key = f"engine:bars_daily:{ticker or symbol}"
        raw_daily = cache_get(daily_key)
        if raw_daily:
            raw_str = raw_daily.decode("utf-8") if isinstance(raw_daily, bytes) else raw_daily
            bars_daily = pd.read_json(io.StringIO(raw_str))
            if len(bars_daily) >= 2:
                return (
                    float(bars_daily["High"].iloc[-2]),
                    float(bars_daily["Low"].iloc[-2]),
                )
    except Exception:
        pass
    return None, None


def run_mtf_on_result(
    result: Any,
    bars_htf: "pd.DataFrame | None",
) -> Any:
    """Run the MTF analyzer on a BreakoutResult and enrich it in-place.

    Returns the (possibly enriched) result.
    """
    if not getattr(result, "breakout_detected", False):
        return result
    if bars_htf is None or bars_htf.empty:
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
        logger.debug("run_mtf_on_result error (non-fatal): %s", exc)
    return result


def persist_breakout_result(result: Any, session_key: str = "") -> int | None:
    """Persist a BreakoutResult to orb_events table."""
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
            breakout_type=result.breakout_type.name,
            mtf_score=result.mtf_score,
            macd_slope=result.macd_slope,
            divergence=getattr(result, "divergence_type", "") or "",
        )
        return row_id
    except Exception as exc:
        logger.debug("persist_breakout_result error (non-fatal): %s", exc)
        return None


def publish_breakout_result(result: Any, session_key: str = "us") -> None:
    """Publish a breakout result to Redis for SSE / dashboard consumption.

    Delegates to ``main._publish_breakout_result()`` for TradingView signal
    integration.  If that function is unavailable, does a minimal Redis publish.
    """
    try:
        # Prefer the main.py publisher which handles TV signals + full pipeline
        from lib.services.engine.main import _publish_breakout_result

        _publish_breakout_result(result, orb_session_key=session_key)
    except ImportError:
        # Fallback: minimal Redis publish
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

            payload = result.to_dict()
            payload["published_at"] = datetime.now(tz=_EST).isoformat()
            payload["orb_session"] = session_key

            key = f"engine:breakout:{result.breakout_type.name.lower()}:{result.symbol}"
            cache_set(key, json.dumps(payload).encode(), ttl=300)

            if REDIS_AVAILABLE and _r is not None:
                _r.publish("dashboard:breakout", json.dumps(payload))
        except Exception:
            pass
    except Exception as exc:
        logger.debug("publish_breakout_result error (non-fatal): %s", exc)


def dispatch_to_position_manager(
    result: Any,
    bars_1m: "pd.DataFrame | None" = None,
    session_key: str = "us",
) -> None:
    """Forward a breakout result to the PositionManager for order execution.

    Delegates to ``main._dispatch_to_position_manager()`` so we don't
    duplicate the ORBResult compatibility shim logic.
    """
    try:
        from lib.services.engine.main import _dispatch_to_position_manager

        _dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
    except ImportError:
        logger.debug("dispatch_to_position_manager: main module not available")
    except Exception as exc:
        logger.debug("dispatch_to_position_manager error (non-fatal): %s", exc)


def send_breakout_alert(
    result: Any,
    breakout_type: BreakoutType,
    session_key: str = "",
) -> None:
    """Send a push notification / alert for a detected breakout.

    Uses the alert template for the given breakout type to format a
    human-readable message.
    """
    try:
        from lib.core.alerts import send_signal

        tmpl = _get_alert_template(breakout_type)
        symbol = result.symbol

        # Build message body
        lines = [
            f"{tmpl['title']}!",
            f"Direction: {result.direction}",
            f"Trigger: {result.trigger_price:,.4f}",
            f"Range: {result.range_low:,.4f} – {result.range_high:,.4f}",
        ]

        # Type-specific extra lines
        if breakout_type == BreakoutType.PrevDay and getattr(result, "prev_day_high", 0) > 0:
            lines.append(f"PDR: {result.prev_day_low:,.4f} – {result.prev_day_high:,.4f}")
        elif breakout_type == BreakoutType.InitialBalance and getattr(result, "ib_high", 0) > 0:
            lines.append(f"IB: {getattr(result, 'ib_low', 0):,.4f} – {result.ib_high:,.4f}")
        elif breakout_type in (BreakoutType.Consolidation, BreakoutType.BollingerSqueeze):
            if getattr(result, "squeeze_detected", False):
                lines.append(f"Squeeze: {result.squeeze_bar_count} bars, BB width {result.squeeze_bb_width:.4f}")

        lines.append(f"ATR: {result.atr_value:,.4f}")

        mtf_score = getattr(result, "mtf_score", None)
        if mtf_score is not None:
            lines.append(f"MTF Score: {mtf_score:.3f}")

        signal_key = f"{tmpl['short'].lower()}_{symbol}_{result.direction}"
        if session_key:
            signal_key = f"{signal_key}_{session_key}"

        send_signal(
            signal_key=signal_key,
            title=f"{tmpl['emoji']} {tmpl['short']} {result.direction}: {symbol}",
            message="\n".join(lines),
            asset=symbol,
            direction=result.direction,
        )
    except Exception as exc:
        logger.debug("send_breakout_alert error (non-fatal): %s", exc)


# ===========================================================================
# Main generic handler
# ===========================================================================


def handle_breakout_check(
    engine: Any,
    breakout_type: BreakoutType,
    session_key: str = "us",
    *,
    # Optional overrides for specific types
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: Any | None = None,
    orb_session_end: Any | None = None,
    orb_scan_start: Any | None = None,
    config_override: Any | None = None,
) -> None:
    """Universal handler for any of the 13 breakout types.

    Replaces ``_handle_check_pdr``, ``_handle_check_ib``,
    ``_handle_check_consolidation``, and the future handlers for the
    9 researched types.

    Pipeline:
        1. Get asset list for the session
        2. For each asset: fetch 1m bars → detect breakout → MTF enrich →
           persist → publish → dispatch to PositionManager → send alert
        3. Log summary

    All errors are caught per-asset so one failure never blocks the rest.

    Args:
        engine: The engine singleton (for bar fetching fallback).
        breakout_type: Which ``BreakoutType`` to detect.
        session_key: Session key whose asset list to use.
        prev_day_high: Override PDR high (PrevDay type only).
        prev_day_low: Override PDR low (PrevDay type only).
        ib_high: Override IB high (InitialBalance type only).
        ib_low: Override IB low (InitialBalance type only).
        orb_session_start: ORB session start time (ORB type only).
        orb_session_end: ORB session end time (ORB type only).
        orb_scan_start: ORB scan start time (ORB type only).
        config_override: Optional ``RangeConfig`` override.
    """
    from lib.services.engine.breakout import (
        DEFAULT_CONFIGS,
        detect_range_breakout,
    )

    short_name = _get_alert_template(breakout_type)["short"]
    logger.debug("▶ %s breakout check [session=%s]...", short_name, session_key)

    try:
        assets = get_assets_for_session_key(session_key)
        if not assets:
            logger.debug("No assets for %s check (session=%s)", short_name, session_key)
            return

        # Resolve config
        config = config_override or DEFAULT_CONFIGS.get(breakout_type)
        if config is None:
            logger.warning("No config for breakout type %s", breakout_type.name)
            return

        found = 0

        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue

            try:
                # 1. Fetch 1-minute bars
                bars_1m = fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    continue

                # 2. For PDR: try to get pre-computed daily levels
                _pdh, _pdl = prev_day_high, prev_day_low
                if breakout_type == BreakoutType.PrevDay and _pdh is None:
                    _pdh, _pdl = get_prev_day_levels(ticker, symbol)

                # 3. Detect breakout
                result = detect_range_breakout(
                    bars_1m,
                    symbol=symbol,
                    config=config,
                    prev_day_high=_pdh,
                    prev_day_low=_pdl,
                    ib_high=ib_high,
                    ib_low=ib_low,
                    orb_session_start=orb_session_start,
                    orb_session_end=orb_session_end,
                    orb_scan_start=orb_scan_start,
                )

                # 4. MTF enrichment
                bars_htf = get_htf_bars(bars_1m, ticker or symbol)
                result = run_mtf_on_result(result, bars_htf)

                # 5. Persist
                persist_breakout_result(result, session_key=session_key)

                # 6. On detection: publish → dispatch → alert
                if result.breakout_detected:
                    found += 1
                    publish_breakout_result(result, session_key=session_key)
                    dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
                    send_breakout_alert(result, breakout_type, session_key)

            except Exception as exc:
                logger.debug("%s check failed for %s: %s", short_name, symbol, exc)

        logger.debug("%s check [%s] complete: %d breakout(s)", short_name, session_key, found)

    except Exception as exc:
        logger.debug("%s handler error (non-fatal): %s", short_name, exc)


def handle_breakout_multi(
    engine: Any,
    session_key: str = "us",
    types: list[BreakoutType] | None = None,
) -> None:
    """Run multiple breakout type detectors for a session's assets.

    Dispatches each type sequentially via ``handle_breakout_check()``.
    For parallel execution, wrap in a ThreadPoolExecutor at the call site.

    Args:
        engine: The engine singleton.
        session_key: Session key whose asset list to use.
        types: List of ``BreakoutType`` to check.
               Defaults to ``[PrevDay, Consolidation]``.
    """
    if types is None:
        types = [BreakoutType.PrevDay, BreakoutType.Consolidation]

    logger.debug(
        "▶ Multi-type breakout sweep [session=%s types=%s]...",
        session_key,
        [bt.name for bt in types],
    )

    import concurrent.futures

    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(types), 4),
        thread_name_prefix="breakout",
    ) as executor:
        for btype in types:
            fut = executor.submit(
                handle_breakout_check,
                engine,
                btype,
                session_key=session_key,
            )
            futures_map[fut] = btype

        for future in concurrent.futures.as_completed(futures_map, timeout=60):
            btype = futures_map[future]
            try:
                future.result()
            except Exception as exc:
                logger.warning(
                    "Multi-type sweep [%s/%s] error: %s",
                    session_key,
                    btype.name,
                    exc,
                )

    logger.debug("Multi-type breakout sweep [%s] complete", session_key)
