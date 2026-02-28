"""
Daily Focus Computation (TASK-203)
====================================
Computes the daily trading focus — the core "what should I trade today" payload.

For each tracked asset (MGC, MNQ, MES, MCL, SIL, HG), computes:
  - Bias: LONG / SHORT / NEUTRAL (from wave dominance + AO + confluence)
  - Entry zone (low–high), stop loss, TP1, TP2
  - Wave ratio, signal quality %, volatility percentile
  - Position size in micro contracts, risk in dollars
  - Should-not-trade flag with reason

The result is written to Redis key `engine:daily_focus` as JSON and served
by data-service via `GET /api/focus`.

Risk rules (from todo.md TASK-203):
  - Risk per trade capped at 0.75% of account size (default $50k = $375)
  - Assets with quality < 55% flagged as NEUTRAL with "skip today" note
  - should_not_trade() returns True if ALL assets < 55% quality or
    max vol_percentile > 88%

Usage:
    from src.lib.services.engine.focus import compute_daily_focus, should_not_trade

    focus = compute_daily_focus(account_size=50_000)
    if should_not_trade(focus):
        print("NO TRADE today")
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("engine.focus")

_EST = ZoneInfo("America/New_York")

# Default risk parameters
DEFAULT_ACCOUNT_SIZE = 50_000
DEFAULT_RISK_PCT = 0.0075  # 0.75% per trade
MIN_QUALITY_THRESHOLD = 0.55  # 55% — below this, flag as NEUTRAL/skip
EXTREME_VOL_THRESHOLD = 0.88  # 88th percentile — too volatile


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _compute_entry_zone(
    last_price: float,
    bias: str,
    atr: float,
    wave_ratio: float,
) -> dict[str, float]:
    """Compute entry zone, stop, TP1, TP2 based on bias and ATR.

    Entry zone is a range around the current price adjusted by ATR.
    Stop is placed at 1.5× ATR from entry midpoint.
    TP1 at 2× ATR, TP2 at 3.5× ATR (for scaling out).
    """
    if atr <= 0:
        atr = last_price * 0.005  # fallback: 0.5% of price

    # Tighter entries when wave is strong
    entry_width = atr * 0.5 if wave_ratio > 1.5 else atr * 0.75

    if bias == "LONG":
        entry_low = last_price - entry_width
        entry_high = last_price + entry_width * 0.3
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint - atr * 1.5
        tp1 = midpoint + atr * 2.0
        tp2 = midpoint + atr * 3.5
    elif bias == "SHORT":
        entry_high = last_price + entry_width
        entry_low = last_price - entry_width * 0.3
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint + atr * 1.5
        tp1 = midpoint - atr * 2.0
        tp2 = midpoint - atr * 3.5
    else:
        # NEUTRAL — no real entry, just show reference levels
        entry_low = last_price - atr
        entry_high = last_price + atr
        midpoint = last_price
        stop = last_price - atr * 2.0
        tp1 = last_price + atr * 2.0
        tp2 = last_price + atr * 3.5

    return {
        "entry_low": round(entry_low, 4),
        "entry_high": round(entry_high, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
    }


def _compute_position_size(
    last_price: float,
    stop_price: float,
    tick_size: float,
    point_value: float,
    max_risk_dollars: float,
) -> tuple[int, float]:
    """Calculate position size in micro contracts from stop distance.

    Returns:
        (position_size, risk_dollars) — contracts and actual risk in $
    """
    stop_distance = abs(last_price - stop_price)
    if stop_distance <= 0 or tick_size <= 0 or point_value <= 0:
        return 1, max_risk_dollars

    # Number of ticks in the stop distance
    ticks = stop_distance / tick_size
    # Dollar risk per contract = ticks × (tick_size × point_value)
    # For micro contracts, point_value already reflects micro sizing
    dollar_per_tick = tick_size * point_value
    risk_per_contract = ticks * dollar_per_tick

    if risk_per_contract <= 0:
        return 1, max_risk_dollars

    # Max contracts within risk budget
    contracts = int(max_risk_dollars / risk_per_contract)
    contracts = max(1, contracts)

    actual_risk = contracts * risk_per_contract

    return contracts, round(actual_risk, 2)


def _derive_bias(
    wave_result: dict[str, Any],
    sq_result: dict[str, Any],
    quality: float,
) -> str:
    """Derive trading bias from wave analysis + signal quality context.

    Bias = wave dominance direction, confirmed by AO and quality threshold.
    """
    if quality < MIN_QUALITY_THRESHOLD:
        return "NEUTRAL"

    wave_bias = wave_result.get("bias", "NEUTRAL")
    dominance = _safe_float(wave_result.get("dominance", 0.0))
    ao = _safe_float(sq_result.get("ao", 0.0))
    market_context = sq_result.get("market_context", "RANGING")

    # Strong directional bias from waves
    if wave_bias == "BULLISH" and dominance > 0.05:
        # Confirm with AO or context
        if ao > 0 or market_context == "UPTREND":
            return "LONG"
        # Weaker confirmation — still LONG but less confident
        if dominance > 0.15:
            return "LONG"
    elif wave_bias == "BEARISH" and dominance < -0.05:
        if ao < 0 or market_context == "DOWNTREND":
            return "SHORT"
        if dominance < -0.15:
            return "SHORT"

    return "NEUTRAL"


def compute_asset_focus(
    name: str,
    account_size: int = DEFAULT_ACCOUNT_SIZE,
) -> Optional[dict[str, Any]]:
    """Compute focus data for a single asset.

    Returns dict with all focus fields, or None on failure.
    """
    try:
        from src.lib.analysis.signal_quality import compute_signal_quality
        from src.lib.analysis.volatility import kmeans_volatility_clusters
        from src.lib.analysis.wave_analysis import calculate_wave_analysis
        from src.lib.core.cache import get_data
        from src.lib.core.models import (  # noqa: F401
            ASSETS,
            CONTRACT_SPECS,
            MICRO_CONTRACT_SPECS,
        )
    except ImportError as exc:
        logger.error("Failed to import required modules: %s", exc)
        return None

    ticker = ASSETS.get(name)
    if not ticker:
        logger.warning("Unknown asset: %s", name)
        return None

    # Get micro contract specs for position sizing
    spec = MICRO_CONTRACT_SPECS.get(name, {})
    tick_size = _safe_float(spec.get("tick", 0.01))
    point_value = _safe_float(spec.get("point", 1.0))

    # Fetch data
    try:
        df = get_data(ticker, "5m", "5d")
        if df is None or df.empty or len(df) < 30:
            logger.warning(
                "Insufficient data for %s (%s): %d bars",
                name,
                ticker,
                len(df) if df is not None else 0,
            )
            return None
    except Exception as exc:
        logger.warning("Data fetch failed for %s: %s", name, exc)
        return None

    last_price = _safe_float(df["Close"].iloc[-1])
    if last_price <= 0:
        logger.warning("Invalid last price for %s: %s", name, last_price)
        return None

    # Run analysis modules
    try:
        wave_result = calculate_wave_analysis(df, asset_name=name)
    except Exception as exc:
        logger.warning("Wave analysis failed for %s: %s", name, exc)
        wave_result = {"wave_ratio": 1.0, "bias": "NEUTRAL", "dominance": 0.0}

    try:
        vol_result = kmeans_volatility_clusters(df)
    except Exception as exc:
        logger.warning("Volatility analysis failed for %s: %s", name, exc)
        vol_result = {
            "percentile": 0.5,
            "raw_atr": last_price * 0.005,
            "adaptive_atr": last_price * 0.005,
            "cluster": "MEDIUM",
        }

    try:
        sq_result = compute_signal_quality(
            df,
            wave_result=wave_result,
            vol_result=vol_result,
        )
    except Exception as exc:
        logger.warning("Signal quality failed for %s: %s", name, exc)
        sq_result = {
            "score": 0.0,
            "quality_pct": 0.0,
            "ao": 0.0,
            "market_context": "RANGING",
        }

    # Extract key metrics
    wave_ratio = _safe_float(wave_result.get("wave_ratio", 1.0))
    quality = _safe_float(sq_result.get("score", 0.0))
    quality_pct = _safe_float(sq_result.get("quality_pct", 0.0))
    vol_percentile = _safe_float(vol_result.get("percentile", 0.5))
    atr = _safe_float(vol_result.get("raw_atr", 0.0))

    # Derive bias
    bias = _derive_bias(wave_result, sq_result, quality)

    # Compute levels
    levels = _compute_entry_zone(last_price, bias, atr, wave_ratio)

    # Max risk per trade
    max_risk = account_size * DEFAULT_RISK_PCT

    # Position sizing
    position_size, risk_dollars = _compute_position_size(
        last_price=last_price,
        stop_price=levels["stop"],
        tick_size=tick_size,
        point_value=point_value,
        max_risk_dollars=max_risk,
    )

    # Build skip note
    notes = []
    if quality < MIN_QUALITY_THRESHOLD:
        notes.append(f"Quality too low ({quality_pct:.0f}%) — skip today")
    if vol_percentile > EXTREME_VOL_THRESHOLD:
        notes.append(f"Extreme volatility ({vol_percentile:.0%}) — dangerous")

    return {
        "symbol": name,
        "ticker": ticker,
        "bias": bias,
        "bias_emoji": {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(bias, "⚪"),
        "last_price": round(last_price, 4),
        "entry_low": levels["entry_low"],
        "entry_high": levels["entry_high"],
        "stop": levels["stop"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "wave_ratio": round(wave_ratio, 2),
        "wave_ratio_text": wave_result.get("wave_ratio_text", f"{wave_ratio:.2f}x"),
        "quality": round(quality, 3),
        "quality_pct": round(quality_pct, 1),
        "high_quality": quality >= MIN_QUALITY_THRESHOLD,
        "vol_percentile": round(vol_percentile, 4),
        "vol_cluster": vol_result.get("cluster", "MEDIUM"),
        "market_phase": wave_result.get("market_phase", "UNKNOWN"),
        "trend_direction": wave_result.get("trend_direction", "NEUTRAL ↔️"),
        "momentum_state": wave_result.get("momentum_state", "NEUTRAL"),
        "dominance_text": wave_result.get("dominance_text", "Neutral"),
        "position_size": position_size,
        "risk_dollars": risk_dollars,
        "max_risk_allowed": round(max_risk, 2),
        "atr": round(atr, 6),
        "notes": "; ".join(notes) if notes else "",
        "skip": quality < MIN_QUALITY_THRESHOLD,
    }


def compute_daily_focus(
    account_size: int = DEFAULT_ACCOUNT_SIZE,
    symbols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compute full daily focus payload for all tracked assets.

    Returns a dict with:
      - assets: list of per-asset focus dicts
      - no_trade: bool — True if should_not_trade()
      - no_trade_reason: str
      - computed_at: ISO timestamp
      - account_size: int
      - session_mode: current session
    """
    try:
        from src.lib.core.models import ASSETS
    except ImportError:
        ASSETS = {}

    if symbols is None:
        symbols = list(ASSETS.keys())

    now = datetime.now(tz=_EST)

    logger.info(
        "Computing daily focus for %d assets (account=$%s)",
        len(symbols),
        f"{account_size:,}",
    )

    asset_results = []
    for name in symbols:
        try:
            result = compute_asset_focus(name, account_size=account_size)
            if result is not None:
                asset_results.append(result)
                logger.info(
                    "  %s: %s %s | quality=%.0f%% | wave=%.2fx | price=%.2f",
                    name,
                    result["bias_emoji"],
                    result["bias"],
                    result["quality_pct"],
                    result["wave_ratio"],
                    result["last_price"],
                )
            else:
                logger.warning("  %s: no data available", name)
        except Exception as exc:
            logger.error("  %s: focus computation failed: %s", name, exc)

    # Sort by quality (best first), then by wave ratio
    asset_results.sort(
        key=lambda x: (x.get("quality", 0), x.get("wave_ratio", 0)),
        reverse=True,
    )

    # Check no-trade conditions
    no_trade, no_trade_reason = should_not_trade(asset_results)

    # Determine session
    hour = now.hour
    if 0 <= hour < 5:
        session = "pre-market"
    elif 5 <= hour < 12:
        session = "active"
    else:
        session = "off-hours"

    payload = {
        "assets": asset_results,
        "no_trade": no_trade,
        "no_trade_reason": no_trade_reason,
        "computed_at": now.isoformat(),
        "account_size": account_size,
        "session_mode": session,
        "total_assets": len(asset_results),
        "tradeable_assets": sum(1 for a in asset_results if not a.get("skip")),
    }

    logger.info(
        "Daily focus computed: %d assets, %d tradeable, no_trade=%s%s",
        len(asset_results),
        payload["tradeable_assets"],
        no_trade,
        f" ({no_trade_reason})" if no_trade else "",
    )

    return payload


def should_not_trade(
    focus_assets: list[dict[str, Any]],
    max_daily_loss: float = -250.0,
    max_consecutive_losses: int = 2,
) -> tuple[bool, str]:
    """Determine if today is a no-trade day.

    Conditions (from TASK-802):
      1. ALL focus assets have quality < 55%
      2. Any focus asset has volatility percentile > 88%
      3. Daily loss already exceeds -$250 (placeholder — needs trade log)
      4. More than 2 consecutive losing trades today (placeholder)
      5. After 10:00 AM ET and no setups triggered (placeholder)

    Returns:
        (should_skip, reason) — True if should not trade.
    """
    if not focus_assets:
        return True, "No market data available"

    # Condition 1: All assets below quality threshold
    qualities = [_safe_float(a.get("quality", 0)) for a in focus_assets]
    if all(q < MIN_QUALITY_THRESHOLD for q in qualities):
        best_q = max(qualities) * 100
        return (
            True,
            f"All assets below {MIN_QUALITY_THRESHOLD * 100:.0f}% quality (best: {best_q:.0f}%)",
        )

    # Condition 2: Any asset has extreme volatility
    vol_percentiles = [_safe_float(a.get("vol_percentile", 0)) for a in focus_assets]
    extreme_vols = [
        a.get("symbol", "?")
        for a in focus_assets
        if _safe_float(a.get("vol_percentile", 0)) > EXTREME_VOL_THRESHOLD
    ]
    if extreme_vols:
        max_vol = max(vol_percentiles)
        return True, (
            f"Extreme volatility on {', '.join(extreme_vols)} "
            f"({max_vol:.0%} percentile) — high risk of stop hunts"
        )

    # Condition 5: Time-based (after 10 AM and no high-quality setups)
    now = datetime.now(tz=_EST)
    if now.hour >= 10 and now.hour < 12:
        tradeable = [a for a in focus_assets if not a.get("skip")]
        if not tradeable:
            return (
                True,
                "After 10:00 AM ET with no quality setups — session winding down",
            )

    return False, ""


def publish_focus_to_redis(focus_data: dict[str, Any]) -> bool:
    """Write focus payload to Redis for data-service to serve.

    Writes to:
      - `engine:daily_focus` — full JSON payload (TTL 5 min)
      - `engine:daily_focus:ts` — last update timestamp
      - Redis Stream `dashboard:stream:focus` — for SSE catch-up
      - Redis PubSub `dashboard:live` — trigger for SSE push

    Returns True on success.
    """
    try:
        from src.lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module")
        return False

    try:
        # Serialize with safe float handling
        payload_json = json.dumps(focus_data, default=str, allow_nan=False)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize focus data: %s", exc)
        return False

    try:
        # Write main focus key
        cache_set("engine:daily_focus", payload_json.encode(), ttl=300)

        # Write timestamp
        ts = datetime.now(tz=_EST).isoformat()
        cache_set("engine:daily_focus:ts", ts.encode(), ttl=300)

        # Write to Redis Stream for SSE catch-up (if Redis is available)
        if REDIS_AVAILABLE and _r is not None:
            try:
                # Add to stream (keep last 100 entries, auto-trim)
                _r.xadd(
                    "dashboard:stream:focus",
                    {"data": payload_json, "ts": ts},
                    maxlen=100,
                    approximate=True,
                )
                # Publish trigger for SSE subscribers
                _r.publish("dashboard:live", payload_json)

                # Also publish per-asset events for granular SSE
                for asset in focus_data.get("assets", []):
                    symbol = asset.get("symbol", "").lower().replace(" ", "_")
                    if symbol:
                        asset_json = json.dumps(asset, default=str, allow_nan=False)
                        _r.publish(f"dashboard:asset:{symbol}", asset_json)

                # Publish no-trade event if applicable
                if focus_data.get("no_trade"):
                    _r.publish(
                        "dashboard:no_trade",
                        json.dumps(
                            {
                                "no_trade": True,
                                "reason": focus_data.get("no_trade_reason", ""),
                                "ts": ts,
                            }
                        ),
                    )

            except Exception as exc:
                logger.debug("Redis Stream/PubSub publish failed (non-fatal): %s", exc)

        logger.debug("Focus data published to Redis (key=engine:daily_focus)")
        return True

    except Exception as exc:
        logger.error("Failed to publish focus to Redis: %s", exc)
        return False
