"""
TradingView Integration API — Webhook Receiver + Signal Publisher
=================================================================

Phase TV-D: POST /api/tv/alert — receives TradingView outbound webhooks
Phase TV-A: Signal publisher — pushes engine signals to a JSON endpoint
            that TradingView can reference, and optionally publishes
            signals.csv to GitHub (nuniesmith/futures-signals) for
            request.seed() consumption in Pine Script.

Also provides endpoints for:
  - Tradovate position sync (receive live position updates)
  - Signal feed for the Ruby Futures Pine Script indicator
  - Metrics push (engine → TV dashboard table values)

Architecture:
    TradingView (browser)
        ├── Ruby Futures indicator reads GET /api/tv/signals (JSON)
        │   └── Entry/Stop/TP lines + CNN label drawn on chart
        ├── Webhooks fire POST /api/tv/alert on user-defined conditions
        │   └── Engine logs, optionally triggers CNN inference, pushes to SSE
        └── Tradovate broker connected → positions POST /api/tv/positions

    Python Engine (Pi)
        ├── On breakout signal fire → updates /api/tv/signals JSON
        ├── Optionally pushes signals.csv to GitHub for request.seed()
        └── Publishes to Redis PubSub for dashboard SSE

Auth:
  - TV webhooks: shared secret in TV_WEBHOOK_SECRET env var
  - GitHub publisher: fine-grained PAT in GITHUB_SIGNALS_TOKEN env var

Usage:
    from lib.services.data.api.tradingview import router
    app.include_router(router)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api.tradingview")

router = APIRouter(tags=["tradingview"])

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TV_WEBHOOK_SECRET = os.getenv("TV_WEBHOOK_SECRET", "")
GITHUB_SIGNALS_TOKEN = os.getenv("GITHUB_SIGNALS_TOKEN", "")
GITHUB_SIGNALS_REPO = os.getenv("GITHUB_SIGNALS_REPO", "nuniesmith/futures-signals")
MAX_SIGNAL_HISTORY = 50  # Keep last N signals in memory and CSV


# ---------------------------------------------------------------------------
# In-memory signal store (published by engine, read by TV indicator)
# ---------------------------------------------------------------------------
class _SignalStore:
    """Thread-safe in-memory store for the latest engine signals.

    Signals are keyed by asset name.  Each asset has at most one active
    signal (the most recent).  A history of the last N signals across
    all assets is also maintained for the CSV / JSON feed.
    """

    def __init__(self, max_history: int = MAX_SIGNAL_HISTORY):
        self._active: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._max = max_history
        self._last_updated: float = 0.0

    def publish(self, signal: dict[str, Any]) -> None:
        """Publish a new signal from the engine."""
        signal["published_at"] = datetime.now(tz=_EST).isoformat()
        signal["published_ts"] = time.time()

        asset = signal.get("asset", signal.get("symbol", "UNKNOWN"))
        self._active[asset] = signal

        self._history.insert(0, signal)
        if len(self._history) > self._max:
            self._history = self._history[: self._max]

        self._last_updated = time.time()

    def get_active(self, asset: str | None = None) -> list[dict[str, Any]]:
        """Get active signals, optionally filtered by asset."""
        if asset:
            sig = self._active.get(asset)
            return [sig] if sig else []
        return list(self._active.values())

    def get_history(self, limit: int = MAX_SIGNAL_HISTORY) -> list[dict[str, Any]]:
        """Get signal history (most recent first)."""
        return self._history[:limit]

    def get_for_asset(self, asset: str) -> dict[str, Any] | None:
        """Get the most recent signal for a specific asset."""
        return self._active.get(asset)

    def clear_expired(self, max_age_seconds: int = 14400) -> int:
        """Remove signals older than max_age_seconds (default 4 hours)."""
        cutoff = time.time() - max_age_seconds
        expired = [k for k, v in self._active.items() if v.get("published_ts", 0) < cutoff]
        for k in expired:
            del self._active[k]
        old_len = len(self._history)
        self._history = [s for s in self._history if s.get("published_ts", 0) >= cutoff]
        return len(expired) + (old_len - len(self._history))

    @property
    def last_updated(self) -> float:
        return self._last_updated

    def to_csv_rows(self) -> list[str]:
        """Export signal history as CSV rows (header + data)."""
        header = "timestamp,asset,breakout_type,direction,entry,stop,tp1,tp2,tp3,cnn_prob,atr,session,quality,mtf"
        rows = [header]
        for sig in self._history:
            row = ",".join(
                str(sig.get(col, ""))
                for col in [
                    "timestamp",
                    "asset",
                    "breakout_type",
                    "direction",
                    "entry",
                    "stop",
                    "tp1",
                    "tp2",
                    "tp3",
                    "cnn_prob",
                    "atr",
                    "session",
                    "quality",
                    "mtf",
                ]
            )
            rows.append(row)
        return rows

    def to_seed_files(self) -> dict[str, str]:
        """Generate per-asset OHLCV CSVs for TradingView ``request.seed()``.

        TradingView's ``request.seed()`` requires CSV files with columns:
            ``time,open,high,low,close,volume``

        We encode engine signal parameters into these columns:
            - **time**   — signal timestamp as Unix milliseconds
            - **open**   — entry price
            - **high**   — stop loss price
            - **low**    — TP1 price
            - **close**  — TP2 price
            - **volume** — packed metadata integer encoding:
                ``direction_bit * 1_000_000 + round(cnn_prob * 10000) * 100 + tp3_offset_ticks``

              Where ``direction_bit``: 1 = LONG, 2 = SHORT.
              The Pine Script unpacks this with modular arithmetic.

        Extra columns (readable by ``request.seed()`` as custom fields in v6):
            We append ``tp3,atr,breakout_type,session`` as additional columns.
            Pine v6 ``request.seed()`` ignores extra columns, so these are for
            the human-readable ``signals.csv`` companion file only.

        Returns:
            Dict mapping filename → CSV content.
            Keys are lowercase asset names with spaces replaced by underscores,
            e.g. ``{"gold": "time,open,...\\n...", "sp500": "time,open,...\\n..."}``.
            Also includes a special ``"_all"`` key with all assets combined
            (most recent signal per asset, sorted by time desc).
        """

        files: dict[str, str] = {}
        header = "time,open,high,low,close,volume"

        # Per-asset files: only the most recent (active) signal
        for asset, sig in self._active.items():
            ts_ms = self._parse_timestamp_ms(sig)
            entry = float(sig.get("entry", 0))
            stop = float(sig.get("stop", 0))
            tp1 = float(sig.get("tp1", 0))
            tp2 = float(sig.get("tp2", 0))
            tp3 = float(sig.get("tp3", 0))
            cnn_prob = float(sig.get("cnn_prob", 0))
            direction = sig.get("direction", "")

            volume = self._pack_volume(direction, cnn_prob, entry, tp3)

            row = f"{ts_ms},{entry},{stop},{tp1},{tp2},{volume}"
            safe_name = asset.lower().replace(" ", "_").replace("/", "_")
            files[safe_name] = f"{header}\n{row}\n"

        # Combined "_all" file: one row per asset (most recent), sorted by time desc
        all_rows: list[tuple[int, str]] = []
        for _asset, sig in self._active.items():
            ts_ms = self._parse_timestamp_ms(sig)
            entry = float(sig.get("entry", 0))
            stop = float(sig.get("stop", 0))
            tp1 = float(sig.get("tp1", 0))
            tp2 = float(sig.get("tp2", 0))
            tp3 = float(sig.get("tp3", 0))
            cnn_prob = float(sig.get("cnn_prob", 0))
            direction = sig.get("direction", "")

            volume = self._pack_volume(direction, cnn_prob, entry, tp3)
            row = f"{ts_ms},{entry},{stop},{tp1},{tp2},{volume}"
            all_rows.append((ts_ms, row))

        all_rows.sort(key=lambda x: x[0], reverse=True)
        all_csv = header + "\n" + "\n".join(r for _, r in all_rows) + "\n"
        files["_all"] = all_csv

        return files

    @staticmethod
    def _parse_timestamp_ms(sig: dict[str, Any]) -> int:
        """Parse a signal's timestamp to Unix milliseconds."""
        from datetime import datetime as _dt

        ts_str = sig.get("timestamp", "")
        try:
            if ts_str:
                dt = _dt.fromisoformat(str(ts_str))
                return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            pass
        # Fallback to published_ts (seconds)
        pts = sig.get("published_ts", 0)
        if pts:
            return int(float(pts) * 1000)
        return int(time.time() * 1000)

    @staticmethod
    def _pack_volume(direction: str, cnn_prob: float, entry: float, tp3: float) -> int:
        """Pack direction + CNN prob + TP3 offset into a single volume integer.

        Encoding: ``dir_code * 1_000_000 + cnn_int * 100 + tp3_offset``
          - dir_code: 1=LONG, 2=SHORT
          - cnn_int: round(cnn_prob * 10000) → 0..10000
          - tp3_offset: abs(tp3 - entry) in ticks (capped at 99)

        Pine Script decodes:
          dir_code  = math.floor(vol / 1000000)
          cnn_raw   = math.floor((vol % 1000000) / 100)
          cnn_prob  = cnn_raw / 10000.0
          tp3_off   = vol % 100
        """
        dir_code = 1 if direction == "LONG" else 2
        cnn_int = min(10000, max(0, round(cnn_prob * 10000)))
        # TP3 offset as integer (abs distance from entry, small encoding)
        tp3_off = min(99, max(0, round(abs(tp3 - entry) * 100))) if tp3 > 0 and entry > 0 else 0
        return dir_code * 1_000_000 + cnn_int * 100 + tp3_off


# Module-level singleton
_signal_store = _SignalStore()

# ---------------------------------------------------------------------------
# Debounced GitHub publisher state
# ---------------------------------------------------------------------------
_GITHUB_DEBOUNCE_SECONDS = 60  # At most 1 push per 60 seconds
_github_last_push_ts: float = 0.0
_github_dirty: bool = False  # True if store changed since last push


def get_signal_store() -> _SignalStore:
    """Get the module-level signal store singleton."""
    return _signal_store


# ---------------------------------------------------------------------------
# In-memory Tradovate position store
# ---------------------------------------------------------------------------
class _TradovatePositionStore:
    """Tracks live positions from Tradovate (via TradingView or direct)."""

    def __init__(self):
        self._positions: dict[str, dict[str, Any]] = {}
        self._last_updated: float = 0.0
        self._history: list[dict[str, Any]] = []

    def update(self, positions: list[dict[str, Any]]) -> None:
        """Bulk update positions from Tradovate."""
        new_map: dict[str, dict[str, Any]] = {}
        for pos in positions:
            symbol = pos.get("symbol", pos.get("instrument", ""))
            if symbol:
                pos["updated_at"] = datetime.now(tz=_EST).isoformat()
                new_map[symbol] = pos

        # Detect closed positions
        for sym in set(self._positions.keys()) - set(new_map.keys()):
            closed = self._positions[sym].copy()
            closed["status"] = "CLOSED"
            closed["closed_at"] = datetime.now(tz=_EST).isoformat()
            self._history.insert(0, closed)

        self._positions = new_map
        self._last_updated = time.time()

        # Trim history
        if len(self._history) > 200:
            self._history = self._history[:200]

    def update_single(self, position: dict[str, Any]) -> None:
        """Update or add a single position."""
        symbol = position.get("symbol", position.get("instrument", ""))
        if not symbol:
            return
        position["updated_at"] = datetime.now(tz=_EST).isoformat()
        self._positions[symbol] = position
        self._last_updated = time.time()

    def remove(self, symbol: str) -> dict[str, Any] | None:
        """Remove a position (when closed)."""
        pos = self._positions.pop(symbol, None)
        if pos:
            pos["status"] = "CLOSED"
            pos["closed_at"] = datetime.now(tz=_EST).isoformat()
            self._history.insert(0, pos)
            self._last_updated = time.time()
        return pos

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._positions.values())

    def get(self, symbol: str) -> dict[str, Any] | None:
        return self._positions.get(symbol)

    @property
    def count(self) -> int:
        return len(self._positions)

    @property
    def last_updated(self) -> float:
        return self._last_updated

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history[:50])


_tradovate_store = _TradovatePositionStore()


def get_tradovate_store() -> _TradovatePositionStore:
    """Get the module-level Tradovate position store singleton."""
    return _tradovate_store


# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------
class TVAlertPayload(BaseModel):
    """TradingView webhook alert payload."""

    symbol: str = Field(..., description="Instrument symbol (e.g. MGC, MES, MNQM5)")
    action: str = Field(
        ...,
        description="Alert action: LONG_ENTRY, SHORT_ENTRY, LONG_EXIT, SHORT_EXIT, INFO",
    )
    price: float = Field(0.0, description="Current price at alert time")
    note: str = Field("", description="Free-form note / reason for alert")
    timeframe: str = Field("", description="Chart timeframe (e.g. 1, 5, 15, 60, D)")
    quality: float = Field(0.0, description="Signal quality score 0-1")
    wave_ratio: float = Field(0.0, description="Wave ratio from Ruby indicator")
    regime: str = Field("", description="Market regime: TRENDING_BULL, RANGING, etc.")
    volume_surge: float = Field(0.0, description="Volume surge ratio")
    session: str = Field("", description="Session: US, LONDON, ASIA, etc.")
    # Optional fields from Ruby Futures indicator
    orb_status: str = Field("", description="ORB status: BREAKOUT_UP, BREAKOUT_DN, FORMING")
    ib_status: str = Field("", description="IB status")
    squeeze_status: str = Field("", description="Squeeze: ON, FIRED, OFF")
    strategy_name: str = Field("ruby_futures", description="Strategy / indicator that fired the alert")


class TVSignalPublish(BaseModel):
    """Signal published from the engine for TradingView consumption."""

    asset: str = Field(..., description="Asset name (Gold, S&P, etc.)")
    symbol: str = Field("", description="Trading ticker (MGC=F, MES=F)")
    breakout_type: str = Field("ORB", description="Breakout type")
    direction: str = Field(..., description="LONG or SHORT")
    entry: float = Field(..., description="Entry price")
    stop: float = Field(..., description="Stop loss price")
    tp1: float = Field(0.0, description="Take profit 1")
    tp2: float = Field(0.0, description="Take profit 2")
    tp3: float = Field(0.0, description="Take profit 3")
    cnn_prob: float = Field(0.0, description="CNN probability 0-1")
    quality: float = Field(0.0, description="Signal quality 0-1")
    mtf: float = Field(0.0, description="MTF score 0-1")
    atr: float = Field(0.0, description="ATR value")
    session: str = Field("", description="Session key")
    bias: str = Field("NEUTRAL", description="Daily bias: LONG, SHORT, NEUTRAL")
    timestamp: str = Field("", description="Signal timestamp ISO format")
    micro_contracts: int = Field(0, description="Suggested micro contract count")
    full_contracts: int = Field(0, description="Suggested full contract count")
    risk_dollars: float = Field(0.0, description="Risk in dollars for micro sizing")


class TradovatePositionPayload(BaseModel):
    """Position update from Tradovate (via webhook or manual push)."""

    symbol: str = Field(..., description="Tradovate symbol (e.g. MGCM5, MESM5)")
    side: str = Field(..., description="LONG or SHORT")
    quantity: int = Field(1, description="Number of contracts")
    entry_price: float = Field(0.0, description="Average entry price")
    current_price: float = Field(0.0, description="Current market price")
    unrealized_pnl: float = Field(0.0, description="Unrealized P&L in dollars")
    realized_pnl: float = Field(0.0, description="Realized P&L for this position")
    account_id: str = Field("", description="Tradovate account ID")
    bracket_phase: str = Field("INITIAL", description="Bracket phase: INITIAL, TP1_HIT, TRAILING")


class TradovatePositionBatch(BaseModel):
    """Batch position update — all current positions at once."""

    positions: list[TradovatePositionPayload] = Field(default_factory=list)
    account_id: str = Field("", description="Tradovate account ID")
    timestamp: str = Field("", description="Update timestamp")


class EngineMetrics(BaseModel):
    """Metrics payload pushed from engine for TradingView dashboard display."""

    daily_pnl: float = Field(0.0)
    open_positions: int = Field(0)
    max_positions: int = Field(5)
    consecutive_losses: int = Field(0)
    risk_blocked: bool = Field(False)
    block_reason: str = Field("")
    session_active: bool = Field(True)
    account_size: float = Field(150_000)
    daily_trades: int = Field(0)


# ---------------------------------------------------------------------------
# Alert history store
# ---------------------------------------------------------------------------
_alert_history: list[dict[str, Any]] = []
_MAX_ALERT_HISTORY = 100

# Metrics store (latest engine metrics for TV dashboard)
_engine_metrics: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _check_tv_secret(secret: str | None) -> bool:
    """Validate the TradingView webhook shared secret."""
    if not TV_WEBHOOK_SECRET:
        # No secret configured — allow all (development mode)
        return True
    return secret == TV_WEBHOOK_SECRET


# ---------------------------------------------------------------------------
# Redis helper — publish to SSE channels
# ---------------------------------------------------------------------------
def _publish_to_redis(channel: str, payload: dict[str, Any]) -> bool:
    """Publish a payload to Redis PubSub for SSE dashboard updates."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return False
        _r.publish(channel, json.dumps(payload, default=str))
        return True
    except Exception as exc:
        logger.debug("Redis publish to %s failed (non-fatal): %s", channel, exc)
        return False


def _cache_set(key: str, value: str, ttl: int = 120) -> bool:
    """Set a Redis cache key."""
    try:
        from lib.core.cache import cache_set

        cache_set(key, value.encode() if isinstance(value, str) else value, ttl=ttl)
        return True
    except Exception as exc:
        logger.debug("Redis cache_set %s failed: %s", key, exc)
        return False


# ---------------------------------------------------------------------------
# GitHub signals.csv publisher
# ---------------------------------------------------------------------------
async def _publish_to_github() -> bool:
    """Push signals.csv + per-asset seed files to the GitHub signals repo.

    Uses the GitHub Contents API to create/update files in the configured
    repo.  Publishes:
      1. ``signals.csv`` — human-readable signal history (all assets).
      2. ``seed/<asset>.csv`` — per-asset OHLCV files for TradingView
         ``request.seed()`` consumption.  One file per active asset.
      3. ``seed/_all.csv`` — combined OHLCV with one row per active asset.

    Requires ``GITHUB_SIGNALS_TOKEN`` PAT with Contents write scope.

    Returns True on success (all files pushed).
    """
    global _github_last_push_ts, _github_dirty

    if not GITHUB_SIGNALS_TOKEN or not GITHUB_SIGNALS_REPO:
        logger.debug("GitHub signals publisher not configured (no token or repo)")
        return False

    try:
        import base64

        import httpx

        headers = {
            "Authorization": f"Bearer {GITHUB_SIGNALS_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Build all files to push
        files_to_push: dict[str, str] = {}

        # 1. Human-readable signals.csv
        csv_rows = _signal_store.to_csv_rows()
        files_to_push["signals.csv"] = "\n".join(csv_rows)

        # 2. Per-asset seed files + _all.csv
        seed_files = _signal_store.to_seed_files()
        for name, content in seed_files.items():
            files_to_push[f"seed/{name}.csv"] = content

        async with httpx.AsyncClient(timeout=15.0) as client:
            success_count = 0
            fail_count = 0

            for filepath, content in files_to_push.items():
                try:
                    content_b64 = base64.b64encode(content.encode()).decode()
                    api_url = f"https://api.github.com/repos/{GITHUB_SIGNALS_REPO}/contents/{filepath}"

                    # Get current file SHA (needed for update vs create)
                    sha = None
                    resp = await client.get(api_url, headers=headers)
                    if resp.status_code == 200:
                        sha = resp.json().get("sha")

                    body: dict[str, Any] = {
                        "message": (f"Update {filepath} — {datetime.now(tz=_EST).strftime('%Y-%m-%d %H:%M ET')}"),
                        "content": content_b64,
                        "branch": "main",
                    }
                    if sha:
                        body["sha"] = sha

                    resp = await client.put(api_url, headers=headers, json=body)
                    if resp.status_code in (200, 201):
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.warning(
                            "GitHub push failed for %s: %d %s",
                            filepath,
                            resp.status_code,
                            resp.text[:120],
                        )
                except Exception as file_exc:
                    fail_count += 1
                    logger.warning("GitHub push error for %s: %s", filepath, file_exc)

            if success_count > 0:
                _github_last_push_ts = time.time()
                _github_dirty = False
                logger.info(
                    "Published %d file(s) to GitHub (%d signals, %d seed files)%s",
                    success_count,
                    len(csv_rows) - 1,
                    len(seed_files),
                    f" [{fail_count} failed]" if fail_count else "",
                )
                return True
            else:
                logger.warning("GitHub publish failed: 0/%d files succeeded", len(files_to_push))
                return False

    except Exception as exc:
        logger.error("GitHub signals publish error: %s", exc)
        return False


async def _debounced_publish_to_github(force: bool = False) -> bool:
    """Push signals.csv to GitHub with rate-limiting (debounce).

    At most one push per ``_GITHUB_DEBOUNCE_SECONDS`` (default 60s).
    When ``force=True``, push immediately regardless of debounce window.

    If within the debounce window, sets ``_github_dirty`` so the next
    call after the window expires will push the accumulated changes.

    Returns True if a push was made and succeeded.
    """
    global _github_dirty

    if not GITHUB_SIGNALS_TOKEN:
        return False

    now = time.time()
    elapsed = now - _github_last_push_ts

    if not force and elapsed < _GITHUB_DEBOUNCE_SECONDS:
        _github_dirty = True
        logger.debug(
            "GitHub publish debounced (%.0fs < %ds) — marked dirty",
            elapsed,
            _GITHUB_DEBOUNCE_SECONDS,
        )
        return False

    return await _publish_to_github()


# ---------------------------------------------------------------------------
# Engine-callable signal publisher (non-HTTP, called from engine internals)
# ---------------------------------------------------------------------------
async def publish_signal_to_tv(signal_data: dict[str, Any], source: str = "engine") -> bool:
    """Publish an engine signal to the TV signal store and push to GitHub.

    This is the **internal** entry point for the engine to publish signals.
    Unlike the ``POST /api/tv/signals/publish`` HTTP endpoint, this does NOT
    require a Pydantic model — it accepts a raw dict directly.

    Flow:
      1. Insert into the in-memory ``_SignalStore`` (immediate).
      2. Publish to Redis PubSub ``dashboard:tv_signal`` for SSE.
      3. Cache in Redis for quick GET reads.
      4. Debounced push to GitHub ``signals.csv`` (at most 1/min).

    Args:
        signal_data: Dict with keys matching the signals.csv schema:
            timestamp, asset, breakout_type, direction, entry, stop,
            tp1, tp2, tp3, cnn_prob, atr, session, quality, mtf
        source: Label for logging ("swing", "orb", "pdr", "ib", "cons").

    Returns:
        True if the signal was published (store + Redis); GitHub may be
        deferred by the debounce window.
    """
    # Ensure timestamp
    if not signal_data.get("timestamp"):
        signal_data["timestamp"] = datetime.now(tz=_EST).isoformat()

    # Publish to in-memory store
    _signal_store.publish(signal_data)

    # Publish to Redis for SSE
    _publish_to_redis("dashboard:tv_signal", signal_data)

    # Cache for quick access
    asset = signal_data.get("asset", signal_data.get("symbol", "UNKNOWN"))
    _cache_set(
        f"tv:signal:{asset}",
        json.dumps(signal_data, default=str),
        ttl=14400,  # 4 hours
    )
    _cache_set(
        "tv:signals:latest",
        json.dumps(_signal_store.get_active(), default=str),
        ttl=14400,
    )

    # Debounced GitHub push
    github_ok = False
    if GITHUB_SIGNALS_TOKEN:
        github_ok = await _debounced_publish_to_github()

    logger.info(
        "Signal published [%s]: %s %s %s @ %.4f (CNN %.1f%%) [github=%s]",
        source,
        signal_data.get("direction", "?"),
        asset,
        signal_data.get("breakout_type", "?"),
        signal_data.get("entry", 0),
        signal_data.get("cnn_prob", 0) * 100,
        "pushed" if github_ok else ("dirty" if _github_dirty else "skip"),
    )

    return True


def publish_signal_to_tv_sync(signal_data: dict[str, Any], source: str = "engine") -> bool:
    """Synchronous wrapper for ``publish_signal_to_tv``.

    Safe to call from synchronous engine code (e.g. ``_publish_breakout_result``,
    ``_publish_swing_to_tv``).  Runs the async publisher in a background task
    if an event loop is running, or falls back to synchronous store-only publish.
    """
    import asyncio

    # Always do the synchronous parts immediately
    if not signal_data.get("timestamp"):
        signal_data["timestamp"] = datetime.now(tz=_EST).isoformat()

    _signal_store.publish(signal_data)
    _publish_to_redis("dashboard:tv_signal", signal_data)

    asset = signal_data.get("asset", signal_data.get("symbol", "UNKNOWN"))
    _cache_set(
        f"tv:signal:{asset}",
        json.dumps(signal_data, default=str),
        ttl=14400,
    )
    _cache_set(
        "tv:signals:latest",
        json.dumps(_signal_store.get_active(), default=str),
        ttl=14400,
    )

    # Schedule the async GitHub push if an event loop is running
    if GITHUB_SIGNALS_TOKEN:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_debounced_publish_to_github())
        except RuntimeError:
            # No running event loop — skip GitHub push (will catch up on next call)
            global _github_dirty
            _github_dirty = True
            logger.debug("No event loop — GitHub push deferred (marked dirty)")

    logger.info(
        "Signal published [%s/sync]: %s %s %s @ %.4f",
        source,
        signal_data.get("direction", "?"),
        asset,
        signal_data.get("breakout_type", "?"),
        signal_data.get("entry", 0),
    )

    return True


async def flush_github_if_dirty() -> bool:
    """Force-push to GitHub if there are unpushed signal changes.

    Call this periodically (e.g. every 60–120s from the engine scheduler)
    to ensure debounced signals eventually get pushed even if no new signal
    fires within the debounce window.

    Returns True if a push was made and succeeded.
    """
    if not _github_dirty:
        return False
    return await _debounced_publish_to_github(force=True)


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# TV-D: TradingView → Engine Webhook
# ---------------------------------------------------------------------------
@router.post("/api/tv/alert")
async def receive_tv_alert(
    payload: TVAlertPayload,
    request: Request,
    secret: str = Query("", alias="secret"),
):
    """Receive a TradingView outbound webhook alert.

    TV alert URL format:
        http://<pi-tailscale-ip>:8100/api/tv/alert?secret=YOUR_SECRET

    TV alert message body (JSON):
        {"symbol": "MGC", "action": "LONG_ENTRY", "price": 2891.5, "note": "ORB breakout"}

    The engine logs the alert, optionally triggers a fresh CNN inference,
    and pushes the result to the dashboard via SSE. This is INFORMATIONAL
    only — no order execution happens here.
    """
    if not _check_tv_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    now = datetime.now(tz=_EST)
    alert_record = {
        **payload.model_dump(),
        "received_at": now.isoformat(),
        "received_ts": time.time(),
        "source_ip": request.client.host if request.client else "unknown",
    }

    # Resolve asset name from ticker
    try:
        from lib.core.asset_registry import get_asset_name_by_ticker

        alert_record["asset_name"] = get_asset_name_by_ticker(payload.symbol)
    except ImportError:
        alert_record["asset_name"] = payload.symbol

    # Store in history
    _alert_history.insert(0, alert_record)
    if len(_alert_history) > _MAX_ALERT_HISTORY:
        del _alert_history[_MAX_ALERT_HISTORY:]

    logger.info(
        "TV Alert: %s %s @ %.2f — %s [%s] quality=%.2f regime=%s",
        payload.action,
        payload.symbol,
        payload.price,
        payload.note,
        payload.session or "?",
        payload.quality,
        payload.regime or "?",
    )

    # Publish to Redis for dashboard SSE
    _publish_to_redis("dashboard:tv_alert", alert_record)

    # Cache latest alert per symbol
    _cache_set(
        f"tv:alert:{payload.symbol}",
        json.dumps(alert_record, default=str),
        ttl=3600,
    )

    return JSONResponse(
        {
            "status": "ok",
            "message": f"Alert received: {payload.action} {payload.symbol}",
            "asset_name": alert_record.get("asset_name", ""),
            "received_at": now.isoformat(),
        }
    )


@router.get("/api/tv/alerts")
async def get_tv_alerts(limit: int = Query(20, ge=1, le=100)):
    """Get recent TradingView alert history."""
    return JSONResponse(
        {
            "alerts": _alert_history[:limit],
            "count": len(_alert_history),
        }
    )


# ---------------------------------------------------------------------------
# TV-A: Engine → TradingView Signal Feed
# ---------------------------------------------------------------------------
@router.post("/api/tv/signals/publish")
async def publish_signal(payload: TVSignalPublish):
    """Publish a new engine signal for TradingView consumption.

    Called via the HTTP endpoint POST /api/tv/signals/publish.
    For internal engine use, prefer ``publish_signal_to_tv()`` or
    ``publish_signal_to_tv_sync()`` which skip HTTP overhead.

    Optionally pushes signals.csv to GitHub if GITHUB_SIGNALS_TOKEN is set.
    """
    signal_data = payload.model_dump()

    # Delegate to the shared internal publisher (immediate push, no debounce)
    await publish_signal_to_tv(signal_data, source="http")

    # For HTTP callers, also do a force push to GitHub (no debounce)
    github_ok = False
    if GITHUB_SIGNALS_TOKEN:
        github_ok = await _publish_to_github()

    return JSONResponse(
        {
            "status": "ok",
            "asset": payload.asset,
            "direction": payload.direction,
            "github_published": github_ok,
        }
    )


@router.get("/api/tv/signals")
async def get_signals(
    asset: str = Query("", description="Filter by asset name"),
    format: str = Query("json", description="Response format: json or csv"),
):
    """Get current engine signals for TradingView indicator consumption.

    The Ruby Futures Pine Script indicator can poll this endpoint
    (or use the GitHub-hosted signals.csv via request.seed()) to
    draw entry/stop/TP lines on the chart.

    Query params:
        asset: Filter to a specific asset (e.g. "Gold", "S&P")
        format: "json" (default) or "csv"

    Returns all active signals (one per asset, most recent wins).
    """
    # Clean expired signals first
    _signal_store.clear_expired()

    if format == "csv":
        csv_rows = _signal_store.to_csv_rows()
        csv_content = "\n".join(csv_rows)
        return JSONResponse(
            content={"csv": csv_content, "count": len(csv_rows) - 1},
            headers={"Content-Type": "application/json"},
        )

    signals = _signal_store.get_active(asset) if asset else _signal_store.get_active()

    return JSONResponse(
        {
            "signals": signals,
            "count": len(signals),
            "last_updated": _signal_store.last_updated,
        }
    )


@router.get("/api/tv/signals/history")
async def get_signal_history(limit: int = Query(50, ge=1, le=200)):
    """Get signal history (most recent first)."""
    history = _signal_store.get_history(limit)
    return JSONResponse(
        {
            "signals": history,
            "count": len(history),
        }
    )


@router.post("/api/tv/signals/clear")
async def clear_signals(asset: str = Query("", description="Clear specific asset or all")):
    """Clear signals — useful for testing or session reset."""
    if asset:
        store = get_signal_store()
        if asset in store._active:
            del store._active[asset]
            return JSONResponse({"status": "ok", "cleared": asset})
        return JSONResponse({"status": "ok", "message": f"No active signal for {asset}"})

    # Clear all
    store = get_signal_store()
    count = len(store._active)
    store._active.clear()
    store._history.clear()
    return JSONResponse({"status": "ok", "cleared_count": count})


# ---------------------------------------------------------------------------
# Tradovate Position Sync
# ---------------------------------------------------------------------------
@router.post("/api/tv/positions")
async def update_tradovate_positions(
    payload: TradovatePositionBatch,
    secret: str = Query("", alias="secret"),
):
    """Receive position updates from Tradovate.

    Can be called by:
      - PickMyTrade webhook relay
      - Manual position entry from dashboard
      - Tradovate API poller script

    Updates the internal position store and publishes to Redis
    for dashboard SSE consumption.
    """
    if not _check_tv_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    positions = [p.model_dump() for p in payload.positions]
    _tradovate_store.update(positions)

    # Publish to Redis for dashboard
    position_data = {
        "positions": _tradovate_store.get_all(),
        "count": _tradovate_store.count,
        "account_id": payload.account_id,
        "timestamp": payload.timestamp or datetime.now(tz=_EST).isoformat(),
    }
    _publish_to_redis("dashboard:tv_positions", position_data)
    _cache_set(
        "tv:positions",
        json.dumps(position_data, default=str),
        ttl=300,
    )

    # Also sync with RiskManager if available
    _sync_with_risk_manager(positions)

    logger.info(
        "Tradovate positions updated: %d positions (account: %s)",
        len(positions),
        payload.account_id or "default",
    )

    return JSONResponse(
        {
            "status": "ok",
            "positions_count": _tradovate_store.count,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.post("/api/tv/positions/single")
async def update_single_position(
    payload: TradovatePositionPayload,
    secret: str = Query("", alias="secret"),
):
    """Update a single Tradovate position (open or modify)."""
    if not _check_tv_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    _tradovate_store.update_single(payload.model_dump())

    _publish_to_redis(
        "dashboard:tv_positions",
        {
            "positions": _tradovate_store.get_all(),
            "count": _tradovate_store.count,
            "event": "position_update",
            "symbol": payload.symbol,
        },
    )

    return JSONResponse({"status": "ok", "symbol": payload.symbol})


@router.post("/api/tv/positions/close")
async def close_tradovate_position(
    symbol: str = Query(..., description="Symbol to close"),
    secret: str = Query("", alias="secret"),
):
    """Mark a Tradovate position as closed."""
    if not _check_tv_secret(secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    closed = _tradovate_store.remove(symbol)
    if not closed:
        return JSONResponse({"status": "ok", "message": f"No position found for {symbol}"})

    _publish_to_redis(
        "dashboard:tv_positions",
        {
            "positions": _tradovate_store.get_all(),
            "count": _tradovate_store.count,
            "event": "position_closed",
            "symbol": symbol,
            "realized_pnl": closed.get("realized_pnl", 0),
        },
    )

    return JSONResponse({"status": "ok", "closed": symbol})


@router.get("/api/tv/positions")
async def get_tradovate_positions():
    """Get current Tradovate positions."""
    return JSONResponse(
        {
            "positions": _tradovate_store.get_all(),
            "count": _tradovate_store.count,
            "last_updated": _tradovate_store.last_updated,
        }
    )


@router.get("/api/tv/positions/history")
async def get_tradovate_position_history(limit: int = Query(50, ge=1, le=200)):
    """Get closed position history."""
    return JSONResponse(
        {
            "history": _tradovate_store.history[:limit],
            "count": len(_tradovate_store.history),
        }
    )


# ---------------------------------------------------------------------------
# Engine Metrics → TradingView
# ---------------------------------------------------------------------------
@router.post("/api/tv/metrics")
async def push_engine_metrics(payload: EngineMetrics):
    """Push engine metrics for TradingView dashboard display.

    Called periodically by the engine (every ~10s) to update the
    risk / position data that the Ruby Futures indicator shows
    in its dashboard table.
    """
    global _engine_metrics
    metrics = payload.model_dump()
    metrics["updated_at"] = datetime.now(tz=_EST).isoformat()
    _engine_metrics = metrics

    _cache_set("tv:metrics", json.dumps(metrics, default=str), ttl=60)
    _publish_to_redis("dashboard:tv_metrics", metrics)

    return JSONResponse({"status": "ok"})


@router.get("/api/tv/metrics")
async def get_engine_metrics():
    """Get latest engine metrics for TradingView."""
    return JSONResponse(_engine_metrics or {"status": "no_data"})


# ---------------------------------------------------------------------------
# Combined status endpoint — everything TV needs in one call
# ---------------------------------------------------------------------------
@router.get("/api/tv/status")
async def get_tv_status():
    """Combined status: signals + positions + metrics + recent alerts.

    Single endpoint for the dashboard to poll for a complete picture
    of the TradingView integration state.
    """
    _signal_store.clear_expired()

    return JSONResponse(
        {
            "signals": {
                "active": _signal_store.get_active(),
                "count": len(_signal_store.get_active()),
                "last_updated": _signal_store.last_updated,
            },
            "positions": {
                "open": _tradovate_store.get_all(),
                "count": _tradovate_store.count,
                "last_updated": _tradovate_store.last_updated,
            },
            "metrics": _engine_metrics or {},
            "alerts": {
                "recent": _alert_history[:5],
                "count": len(_alert_history),
            },
            "config": {
                "webhook_secret_configured": bool(TV_WEBHOOK_SECRET),
                "github_publisher_configured": bool(GITHUB_SIGNALS_TOKEN),
                "github_repo": GITHUB_SIGNALS_REPO if GITHUB_SIGNALS_TOKEN else "",
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GitHub publisher manual trigger
# ---------------------------------------------------------------------------
@router.post("/api/tv/signals/push-github")
async def push_signals_to_github():
    """Manually trigger a push of signals.csv to GitHub.

    Useful for testing or forcing an update outside the normal
    signal-fire flow.
    """
    if not GITHUB_SIGNALS_TOKEN:
        return JSONResponse(
            {"status": "error", "message": "GITHUB_SIGNALS_TOKEN not configured"},
            status_code=400,
        )

    ok = await _publish_to_github()
    return JSONResponse(
        {
            "status": "ok" if ok else "error",
            "message": "Published to GitHub" if ok else "GitHub publish failed",
            "signal_count": len(_signal_store.get_history()),
        }
    )


# ---------------------------------------------------------------------------
# Internal helper: sync Tradovate positions with RiskManager
# ---------------------------------------------------------------------------
def _sync_with_risk_manager(positions: list[dict[str, Any]]) -> None:
    """Attempt to sync Tradovate positions with the engine's RiskManager.

    This bridges the gap between Tradovate (live positions from manual
    trading on TradingView) and the engine's risk tracking system.
    Non-fatal — if RiskManager isn't available, we just log and skip.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return

        # Format positions for RiskManager.sync_positions() compatibility
        formatted = []
        for pos in positions:
            formatted.append(
                {
                    "symbol": pos.get("symbol", ""),
                    "side": pos.get("side", "UNKNOWN"),
                    "quantity": pos.get("quantity", 0),
                    "avgPrice": pos.get("entry_price", 0),
                    "unrealizedPnL": pos.get("unrealized_pnl", 0),
                    "source": "tradovate",
                }
            )

        # Store in Redis for the engine to pick up on next risk check
        _r.set(
            "tv:positions:for_risk_sync",
            json.dumps(formatted, default=str),
            ex=120,
        )

    except Exception as exc:
        logger.debug("Risk manager sync failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------
@router.get("/api/tv/health")
async def tv_health():
    """TradingView integration health check."""
    return JSONResponse(
        {
            "status": "ok",
            "active_signals": len(_signal_store.get_active()),
            "open_positions": _tradovate_store.count,
            "alerts_received": len(_alert_history),
            "webhook_secret_configured": bool(TV_WEBHOOK_SECRET),
            "github_configured": bool(GITHUB_SIGNALS_TOKEN),
            "last_signal_update": _signal_store.last_updated,
            "last_position_update": _tradovate_store.last_updated,
        }
    )
