"""
Dataset Generator — Orchestrates Chart Rendering + Auto-Labeling for CNN Training
==================================================================================
Combines the ORB simulator (auto-labeler) and chart renderer to produce a
complete labeled dataset of Ruby-style chart images suitable for training
the HybridBreakoutCNN model.

Pipeline:
  1. Load 1-minute bar data for target symbols (via Massive client or cache).
  2. Slide windows across the data, simulating ORB trades per window.
  3. Render a Ruby-style chart snapshot for each window.
  4. Write a CSV manifest (labels.csv) with image paths, labels, and tabular
     features ready for BreakoutDataset.

The generator is designed to run as an off-hours scheduled job (e.g. 02:30 ET)
via the engine scheduler, but can also be invoked manually from the CLI.

Public API:
    from lib.analysis.dataset_generator import (
        generate_dataset,
        generate_dataset_for_symbol,
        DatasetConfig,
        DatasetStats,
    )

    stats = generate_dataset(
        symbols=["MGC", "MES", "MNQ"],
        days_back=90,
    )
    # stats.total_images → 18432
    # stats.csv_path → "dataset/labels.csv"

Dependencies:
  - lib.analysis.orb_simulator (auto-labeling)
  - lib.analysis.chart_renderer (image generation)
  - pandas, numpy (already in project)
  - Massive client or cached bar data (for historical bars)

Design:
  - Pure orchestration — delegates to orb_simulator and chart_renderer.
  - Resumable: skips images that already exist on disk (by filename).
  - Thread-safe: each symbol can be processed independently.
  - Produces balanced datasets by capping over-represented labels.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
import pandas as pd

logger = logging.getLogger("analysis.dataset_generator")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for the dataset generation pipeline."""

    # Output paths
    output_dir: str = "dataset"
    image_dir: str = "dataset/images"
    csv_filename: str = "labels.csv"

    # Window parameters (passed to orb_simulator.simulate_batch)
    window_size: int = 240  # 4 hours of 1-min bars
    step_size: int = 30  # 30-minute steps between windows
    min_window_bars: int = 60  # minimum bars to attempt simulation

    # ORB simulation bracket config
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 2.0
    tp2_atr_mult: float = 3.0
    max_hold_bars: int = 120
    atr_period: int = 14

    # ORB session: "us" (default, OR 09:30–10:00) or "london" (OR 03:00–03:30).
    # When "london", the bracket config is automatically adjusted to use
    # London OR times and a narrower pre-market window (00:00–03:00).
    # When "both", dataset generation runs both sessions per day (combined).
    orb_session: str = "us"

    # Chart rendering
    chart_dpi: int = 150  # lower than live for disk space savings
    chart_figsize: tuple[float, float] = (12, 8)

    # Dataset balancing
    max_samples_per_label: int = 0  # 0 = no cap
    include_no_trade: bool = False  # include no_trade samples (usually not useful)

    # Resumability
    skip_existing: bool = True  # skip images that already exist on disk

    # Data source
    bars_source: str = "cache"  # "cache" (Redis) or "massive" (API) or "csv"
    csv_bars_dir: str = "data/bars"  # only used if bars_source == "csv"

    # Parallelism
    max_workers: int = 1  # symbols processed in parallel (1 = sequential)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Statistics from a dataset generation run."""

    total_windows: int = 0
    total_trades: int = 0
    total_images: int = 0
    skipped_existing: int = 0
    render_failures: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    symbols_processed: list[str] = field(default_factory=list)
    csv_path: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "total_trades": self.total_trades,
            "total_images": self.total_images,
            "skipped_existing": self.skipped_existing,
            "render_failures": self.render_failures,
            "label_distribution": self.label_distribution,
            "symbols_processed": self.symbols_processed,
            "csv_path": self.csv_path,
            "duration_seconds": round(self.duration_seconds, 1),
            "errors": self.errors[:20],  # cap error list
        }

    def summary(self) -> str:
        ld = ", ".join(f"{k}={v}" for k, v in sorted(self.label_distribution.items()))
        return (
            f"Dataset: {self.total_images} images from {len(self.symbols_processed)} symbols | "
            f"Trades: {self.total_trades}/{self.total_windows} windows | "
            f"Labels: [{ld}] | "
            f"Skipped: {self.skipped_existing}, Failures: {self.render_failures} | "
            f"Time: {self.duration_seconds:.0f}s | CSV: {self.csv_path}"
        )


# ---------------------------------------------------------------------------
# Bar data loading helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Symbol → Yahoo-style ticker mapping
# ---------------------------------------------------------------------------
# The backtest/dataset scripts use short names like "MGC", "MES", "MNQ"
# but the cache layer and Massive API expect Yahoo-style tickers like
# "MGC=F", "ES=F", etc.  This mapping bridges the two.

_SYMBOL_TO_TICKER: dict[str, str] = {
    # Micro contracts
    "MGC": "MGC=F",
    "MES": "MES=F",
    "MNQ": "MNQ=F",
    "MCL": "MCL=F",
    "MYM": "MYM=F",
    "M2K": "M2K=F",
    "SIL": "SIL=F",
    "MHG": "MHG=F",
    # Full-size contracts
    "GC": "GC=F",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "CL": "CL=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
}


def _resolve_ticker(symbol: str) -> str:
    """Convert a short symbol like 'MGC' to a Yahoo-style ticker like 'MGC=F'.

    If the symbol already looks like a Yahoo ticker (contains '='), returns as-is.
    Falls back to appending '=F' if not found in the explicit map.
    """
    if "=" in symbol:
        return symbol
    return _SYMBOL_TO_TICKER.get(symbol.upper(), f"{symbol.upper()}=F")


def _load_bars_from_cache(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Attempt to load 1-minute bars from Redis cache.

    Uses the standard ``get_data()`` cache layer (which stores bars under
    hashed ``futures:*`` keys) with proper Yahoo-style ticker resolution.
    Also checks legacy ``engine:bars_1m:*`` keys as a fallback.
    """
    # --- Primary path: use get_data() which handles hashed cache keys ---
    try:
        from lib.core.cache import get_data

        ticker = _resolve_ticker(symbol)
        # Map days to a period string for get_data()
        if days <= 5:
            period = f"{days}d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        else:
            period = "6mo"

        df = get_data(ticker, interval="1m", period=period)
        if df is not None and not df.empty and len(df) > 50:
            logger.debug(
                "Loaded %d bars for %s (ticker=%s) from cache via get_data()",
                len(df),
                symbol,
                ticker,
            )
            return df
    except ImportError:
        logger.debug("Cache module not available")
    except Exception as exc:
        logger.debug("get_data() failed for %s: %s", symbol, exc)

    # --- Fallback: check legacy engine:bars_1m:* keys ---
    try:
        from lib.core.cache import cache_get
    except ImportError:
        return None

    import io

    ticker = _resolve_ticker(symbol)
    for key_pattern in [
        f"engine:bars_1m_hist:{ticker}",
        f"engine:bars_1m:{ticker}",
        f"engine:bars_1m_hist:{symbol}",
        f"engine:bars_1m:{symbol}",
    ]:
        try:
            raw = cache_get(key_pattern)
            if raw:
                raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                df = pd.read_json(io.StringIO(raw_str))
                if not df.empty and len(df) > 100:
                    logger.debug("Loaded %d bars for %s from cache key %s", len(df), symbol, key_pattern)
                    return df
        except Exception as exc:
            logger.debug("Cache load failed for %s key %s: %s", symbol, key_pattern, exc)

    return None


def _load_bars_from_csv(symbol: str, csv_dir: str = "data/bars") -> pd.DataFrame | None:
    """Load 1-minute bars from a local CSV file.

    Expected filename pattern: ``{csv_dir}/{symbol}_1m.csv``
    Expected columns: Date/Datetime, Open, High, Low, Close, Volume
    """
    csv_path = os.path.join(csv_dir, f"{symbol}_1m.csv")
    if not os.path.isfile(csv_path):
        # Also try lowercase
        csv_path = os.path.join(csv_dir, f"{symbol.lower()}_1m.csv")
    if not os.path.isfile(csv_path):
        logger.debug("No CSV file found for %s in %s", symbol, csv_dir)
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        # Ensure proper column names
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("open", "o"):
                col_map[col] = "Open"
            elif cl in ("high", "h"):
                col_map[col] = "High"
            elif cl in ("low", "l"):
                col_map[col] = "Low"
            elif cl in ("close", "c"):
                col_map[col] = "Close"
            elif cl in ("volume", "vol", "v"):
                col_map[col] = "Volume"
        if col_map:
            df = df.rename(columns=col_map)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        logger.debug("Loaded %d bars for %s from %s", len(df), symbol, csv_path)
        return df
    except Exception as exc:
        logger.warning("Failed to load CSV for %s: %s", symbol, exc)
        return None


def _load_bars_from_massive(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Load 1-minute bars via the Massive REST API.

    Uses ``MassiveDataProvider.get_aggs()`` with Yahoo-style ticker resolution.
    Massive only stores a limited window of 1-minute data, so for longer
    histories the effective day count may be clamped by the API.
    """
    try:
        from lib.integrations.massive_client import get_massive_provider
    except ImportError:
        logger.debug("Massive client not available")
        return None

    try:
        provider = get_massive_provider()
        if not provider.is_available:
            logger.debug("Massive provider not available (no API key?)")
            return None

        ticker = _resolve_ticker(symbol)

        # Map days to a period string that get_aggs() understands
        if days <= 1:
            period = "1d"
        elif days <= 5:
            period = "5d"
        elif days <= 10:
            period = "10d"
        elif days <= 15:
            period = "15d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        else:
            period = "6mo"

        df = provider.get_aggs(ticker, interval="1m", period=period)
        if df is not None and not df.empty:
            logger.debug("Loaded %d bars for %s (ticker=%s) from Massive", len(df), symbol, ticker)
            return df
    except Exception as exc:
        logger.debug("Massive load failed for %s: %s", symbol, exc)

    return None


def load_bars(
    symbol: str,
    source: str = "cache",
    days: int = 90,
    csv_dir: str = "data/bars",
) -> pd.DataFrame | None:
    """Load 1-minute bars from the configured source, with fallback chain.

    Tries the specified source first, then falls back through:
      cache → csv → massive

    Args:
        symbol: Instrument symbol (e.g. "MGC").
        source: Primary source — "cache", "csv", or "massive".
        days: Number of days of history to request.
        csv_dir: Directory for CSV bar files.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None.
    """
    loaders = {
        "cache": lambda: _load_bars_from_cache(symbol, days),
        "csv": lambda: _load_bars_from_csv(symbol, csv_dir),
        "massive": lambda: _load_bars_from_massive(symbol, days),
    }

    # Try primary source first
    if source in loaders:
        df = loaders[source]()
        if df is not None and not df.empty:
            return df

    # Fallback chain
    for name, loader in loaders.items():
        if name == source:
            continue
        try:
            df = loader()
            if df is not None and not df.empty:
                logger.info("Loaded bars for %s via fallback source: %s", symbol, name)
                return df
        except Exception:
            continue

    logger.warning("Could not load bars for %s from any source", symbol)
    return None


def load_daily_bars(
    symbol: str,
    source: str = "cache",
    csv_dir: str = "data/bars",
) -> pd.DataFrame | None:
    """Load daily bars for NR7 detection.

    Tries to derive daily bars from 1-minute data by resampling,
    or loads a dedicated daily CSV if available.
    """
    # Try dedicated daily CSV first
    daily_csv = os.path.join(csv_dir, f"{symbol}_daily.csv")
    if os.path.isfile(daily_csv):
        try:
            df = pd.read_csv(daily_csv, parse_dates=True, index_col=0)
            if not df.empty:
                return df
        except Exception:
            pass

    # Resample from 1-minute bars
    bars_1m = load_bars(symbol, source=source, csv_dir=csv_dir)
    if bars_1m is not None and len(bars_1m) > 100:
        try:
            _resampled = (
                bars_1m.resample("1D")
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
            daily = pd.DataFrame(_resampled) if not isinstance(_resampled, pd.DataFrame) else _resampled
            if len(daily) >= 7:
                return daily
        except Exception as exc:
            logger.debug("Daily resample failed for %s: %s", symbol, exc)

    return None


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def _bracket_configs_for_session(
    cfg: DatasetConfig,
) -> list[tuple[str, Any]]:
    """Return (session_key, BracketConfig) pairs based on ``cfg.orb_session``.

    - ``"us"``     → single US config (OR 09:30–10:00, PM end 08:20)
    - ``"london"`` → single London config (OR 03:00–03:30, PM end 03:00)
    - ``"both"``   → both configs so the generator runs each session per day
    """
    from datetime import time as dt_time

    from lib.analysis.orb_simulator import BracketConfig

    us_cfg = BracketConfig(
        sl_atr_mult=cfg.sl_atr_mult,
        tp1_atr_mult=cfg.tp1_atr_mult,
        tp2_atr_mult=cfg.tp2_atr_mult,
        max_hold_bars=cfg.max_hold_bars,
        atr_period=cfg.atr_period,
        or_start=dt_time(9, 30),
        or_end=dt_time(10, 0),
        pm_end=dt_time(8, 20),
    )

    london_cfg = BracketConfig(
        sl_atr_mult=cfg.sl_atr_mult,
        tp1_atr_mult=cfg.tp1_atr_mult,
        tp2_atr_mult=cfg.tp2_atr_mult,
        max_hold_bars=cfg.max_hold_bars,
        atr_period=cfg.atr_period,
        or_start=dt_time(3, 0),
        or_end=dt_time(3, 30),
        pm_end=dt_time(3, 0),
    )

    session = cfg.orb_session.lower().strip()
    if session == "london":
        return [("london", london_cfg)]
    elif session == "both":
        return [("london", london_cfg), ("us", us_cfg)]
    else:
        return [("us", us_cfg)]


def generate_dataset_for_symbol(
    symbol: str,
    bars_1m: pd.DataFrame,
    bars_daily: pd.DataFrame | None = None,
    config: DatasetConfig | None = None,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Generate labeled chart images for a single symbol.

    This is the workhorse function.  It:
      1. Runs simulate_batch on the bars to get ORBSimResults.
      2. For each result that is a trade (or no_trade if enabled), renders
         a Ruby-style chart snapshot.
      3. Collects rows for the CSV manifest.

    When ``config.orb_session`` is ``"both"``, the function runs simulation
    for both London and US sessions, producing training data from each.

    Args:
        symbol: Instrument symbol.
        bars_1m: 1-minute OHLCV bars.
        bars_daily: Daily bars for NR7 (optional).
        config: DatasetConfig.

    Returns:
        (rows, stats) where rows is a list of dicts for the CSV, and
        stats is a DatasetStats for this symbol.
    """
    from lib.analysis.orb_simulator import simulate_batch

    cfg = config or DatasetConfig()
    stats = DatasetStats()
    stats.symbols_processed.append(symbol)
    rows: list[dict[str, Any]] = []

    # Build session-aware bracket configs
    session_configs = _bracket_configs_for_session(cfg)

    all_sim_results = []
    for session_key, bracket_cfg in session_configs:
        session_label = f"{symbol}/{session_key}"
        logger.info(
            "Simulating ORB trades for %s (OR %s–%s, %d bars)...",
            session_label,
            bracket_cfg.or_start.strftime("%H:%M"),
            bracket_cfg.or_end.strftime("%H:%M"),
            len(bars_1m),
        )
        sim_results = simulate_batch(
            bars_1m=bars_1m,
            symbol=symbol,
            config=bracket_cfg,
            bars_daily=bars_daily,
            window_size=cfg.window_size,
            step_size=cfg.step_size,
            min_window_bars=cfg.min_window_bars,
        )
        # Tag each result with session_key for downstream traceability
        for r in sim_results:
            r._session_key = session_key  # type: ignore[attr-defined]
        all_sim_results.extend(sim_results)
        logger.info(
            "%s: %d windows → %d trades",
            session_label,
            len(sim_results),
            sum(1 for r in sim_results if r.is_trade),
        )

    stats.total_windows = len(all_sim_results)
    stats.total_trades = sum(1 for r in all_sim_results if r.is_trade)

    sim_results = all_sim_results

    # Try to import chart renderer (optional — dataset can be generated
    # without images for tabular-only models)
    try:
        from lib.analysis.chart_renderer import RenderConfig, render_ruby_snapshot

        _can_render = True
    except ImportError:
        logger.warning("Chart renderer not available — generating tabular-only dataset")
        _can_render = False
        RenderConfig = None  # type: ignore[assignment,misc]  # noqa: N806
        render_ruby_snapshot = None  # type: ignore[assignment]  # noqa: N806

    render_cfg = None
    if _can_render and RenderConfig is not None:
        render_cfg = RenderConfig(
            dpi=cfg.chart_dpi,
            figsize=cfg.chart_figsize,
            output_dir=cfg.image_dir,
        )

    for sim_idx, result in enumerate(sim_results):
        # Skip no_trade unless configured to include them
        if not result.is_trade and not cfg.include_no_trade:
            continue

        label = result.label
        stats.label_distribution[label] = stats.label_distribution.get(label, 0) + 1

        # Check max samples per label
        if cfg.max_samples_per_label > 0 and stats.label_distribution[label] > cfg.max_samples_per_label:
            continue

        # Determine image path
        ts_str = result.breakout_time or result.or_start_time or datetime.now(_EST).isoformat()
        # Create a safe filename component from the timestamp
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_filename = f"{symbol}_{safe_ts}_{label}_{sim_idx}.png"
        image_path = os.path.join(cfg.image_dir, image_filename)

        # Skip if already exists (resumability)
        if cfg.skip_existing and os.path.isfile(image_path):
            stats.skipped_existing += 1
            # Still add the row to CSV
            rows.append(_build_row(result, image_path))
            stats.total_images += 1
            continue

        # Render chart image
        rendered_path = None
        if _can_render:
            # Extract the window of bars for this simulation.
            # Use _window_offset stored by simulate_batch() when available —
            # this is the authoritative start index into bars_1m.  The old
            # ``sim_idx * step_size`` calculation breaks when multiple
            # sessions are concatenated (e.g. orb_session="both") because
            # sim_idx keeps incrementing across session boundaries.
            try:
                _stored_offset = getattr(result, "_window_offset", -1)
                _stored_wsize = getattr(result, "_window_size", 0)

                if _stored_offset >= 0:
                    window_start = _stored_offset
                    window_end = min(
                        window_start + (_stored_wsize or cfg.window_size),
                        len(bars_1m),
                    )
                else:
                    # Fallback for older ORBSimResult objects without provenance
                    window_start = sim_idx * cfg.step_size
                    window_end = min(window_start + cfg.window_size, len(bars_1m))

                window_bars = bars_1m.iloc[window_start:window_end].copy()

                if len(window_bars) < 10:
                    logger.warning(
                        "Skipping render for %s window %d: only %d bars (offset=%d, end=%d, total_bars=%d)",
                        symbol,
                        sim_idx,
                        len(window_bars),
                        window_start,
                        window_end,
                        len(bars_1m),
                    )
                else:
                    rendered_path = render_ruby_snapshot(  # type: ignore[misc]
                        bars=window_bars,
                        symbol=symbol,
                        orb_high=result.or_high if result.or_high > 0 else None,
                        orb_low=result.or_low if result.or_low > 0 else None,
                        direction=result.direction or None,
                        quality_pct=result.quality_pct,
                        label=label,
                        save_path=image_path,
                        config=render_cfg,
                    )
                    if rendered_path is None:
                        logger.warning(
                            "render_ruby_snapshot returned None for %s window %d (bars=%d, orb_h=%s, orb_l=%s, dir=%s)",
                            symbol,
                            sim_idx,
                            len(window_bars),
                            result.or_high,
                            result.or_low,
                            result.direction,
                        )
            except Exception as exc:
                logger.warning(
                    "Render exception for %s window %d: %s",
                    symbol,
                    sim_idx,
                    exc,
                )

        if rendered_path is None and _can_render:
            stats.render_failures += 1
            # Still create a row with empty image path for tabular-only use
            image_path = ""
        elif rendered_path:
            image_path = rendered_path

        if image_path or not _can_render:
            rows.append(_build_row(result, image_path))
            stats.total_images += 1

    logger.info(
        "%s dataset: %d images (%d skipped, %d failures)",
        symbol,
        stats.total_images,
        stats.skipped_existing,
        stats.render_failures,
    )

    return rows, stats


def _build_row(result, image_path: str) -> dict[str, Any]:
    """Build a single CSV row from an ORBSimResult."""
    # Compute atr_pct (ATR as fraction of entry price)
    atr_pct = 0.0
    if result.entry > 0 and result.atr > 0:
        atr_pct = result.atr / result.entry

    return {
        "image_path": image_path,
        "label": result.label,
        "symbol": result.symbol,
        "direction": result.direction,
        "quality_pct": result.quality_pct,
        "volume_ratio": round(result.breakout_volume_ratio, 4),
        "atr_pct": round(atr_pct, 6),
        "cvd_delta": round(getattr(result, "cvd_delta", 0.0), 4),
        "nr7_flag": 1 if result.nr7 else 0,
        "london_overlap_flag": getattr(result, "london_overlap_flag", 0.0),
        "entry": round(result.entry, 6),
        "sl": round(result.sl, 6),
        "tp1": round(result.tp1, 6),
        "or_high": round(result.or_high, 6),
        "or_low": round(result.or_low, 6),
        "or_range": round(result.or_range, 6),
        "atr": round(result.atr, 6),
        "pnl_r": round(result.pnl_r, 4),
        "hold_bars": result.hold_bars,
        "outcome": result.outcome,
        "breakout_time": result.breakout_time,
        "pm_high": round(result.pm_high, 6),
        "pm_low": round(result.pm_low, 6),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_dataset(
    symbols: Sequence[str],
    days_back: int = 90,
    config: DatasetConfig | None = None,
    bars_override: dict[str, pd.DataFrame] | None = None,
) -> DatasetStats:
    """Generate a complete labeled dataset for CNN training.

    This is the main entry point called by the scheduler or CLI.

    Args:
        symbols: List of instrument symbols (e.g. ["MGC", "MES", "MNQ"]).
        days_back: Number of days of historical data to process.
        config: DatasetConfig (uses defaults if None).
        bars_override: Optional pre-loaded bars dict (symbol → DataFrame).
                       If provided, skips the data loading step.

    Returns:
        DatasetStats with aggregate statistics.
    """
    cfg = config or DatasetConfig()
    start_time = time.monotonic()

    # Ensure output directories exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.image_dir, exist_ok=True)

    aggregate_stats = DatasetStats()
    all_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        logger.info("Processing %s...", symbol)

        # Load bar data
        if bars_override and symbol in bars_override:
            bars_1m = bars_override[symbol]
        else:
            bars_1m = load_bars(
                symbol,
                source=cfg.bars_source,
                days=days_back,
                csv_dir=cfg.csv_bars_dir,
            )

        if bars_1m is None or bars_1m.empty:
            msg = f"No bar data available for {symbol} — skipping"
            logger.warning(msg)
            aggregate_stats.errors.append(msg)
            continue

        # Load daily bars for NR7
        bars_daily = load_daily_bars(symbol, source=cfg.bars_source, csv_dir=cfg.csv_bars_dir)

        try:
            rows, symbol_stats = generate_dataset_for_symbol(
                symbol=symbol,
                bars_1m=bars_1m,
                bars_daily=bars_daily,
                config=cfg,
            )

            all_rows.extend(rows)

            # Merge stats
            aggregate_stats.total_windows += symbol_stats.total_windows
            aggregate_stats.total_trades += symbol_stats.total_trades
            aggregate_stats.total_images += symbol_stats.total_images
            aggregate_stats.skipped_existing += symbol_stats.skipped_existing
            aggregate_stats.render_failures += symbol_stats.render_failures
            aggregate_stats.symbols_processed.append(symbol)
            for label, count in symbol_stats.label_distribution.items():
                aggregate_stats.label_distribution[label] = aggregate_stats.label_distribution.get(label, 0) + count

        except Exception as exc:
            msg = f"Dataset generation failed for {symbol}: {exc}"
            logger.error(msg, exc_info=True)
            aggregate_stats.errors.append(msg)

    # Write CSV manifest
    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)
    if all_rows:
        df = pd.DataFrame(all_rows)

        # If CSV already exists, append (for incremental builds)
        if os.path.isfile(csv_path) and cfg.skip_existing:
            try:
                existing_df = pd.read_csv(csv_path)
                # Deduplicate by image_path
                existing_paths = list(existing_df["image_path"].tolist())
                new_rows = df[~df["image_path"].isin(existing_paths)]
                if not new_rows.empty:
                    df = pd.concat([existing_df, new_rows], ignore_index=True)
                    logger.info("Appended %d new rows to existing CSV (%d total)", len(new_rows), len(df))
                else:
                    df = existing_df
                    logger.info("No new rows to append — CSV unchanged (%d rows)", len(df))
            except Exception as exc:
                logger.warning("Could not append to existing CSV, overwriting: %s", exc)

        df.to_csv(csv_path, index=False)
        aggregate_stats.csv_path = csv_path
        logger.info("Dataset CSV written: %s (%d rows)", csv_path, len(df))
    else:
        logger.warning("No data generated — CSV not written")

    aggregate_stats.duration_seconds = time.monotonic() - start_time

    # Write stats JSON alongside CSV for audit
    stats_path = os.path.join(cfg.output_dir, "dataset_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump(aggregate_stats.to_dict(), f, indent=2)
    except Exception:
        pass

    logger.info(aggregate_stats.summary())
    return aggregate_stats


# ---------------------------------------------------------------------------
# Train/Val split helper
# ---------------------------------------------------------------------------


def split_dataset(
    csv_path: str,
    val_fraction: float = 0.15,
    output_dir: str | None = None,
    stratify: bool = True,
    random_seed: int = 42,
) -> tuple[str, str]:
    """Split a dataset CSV into train and validation sets.

    Args:
        csv_path: Path to the full dataset CSV.
        val_fraction: Fraction of data for validation (default 0.15).
        output_dir: Where to write the split CSVs (default: same dir as input).
        stratify: If True, maintain label distribution in both splits.
        random_seed: Random seed for reproducibility.

    Returns:
        (train_csv_path, val_csv_path)
    """
    df = pd.read_csv(csv_path)
    out_dir = output_dir or os.path.dirname(csv_path)

    rng = np.random.RandomState(random_seed)

    # --- Infer session from breakout_time for stratification ---
    # London session: breakout hour < 8 ET;  US session: hour >= 8 ET.
    def _infer_session(bt: Any) -> str:
        try:
            bt_str = str(bt).strip()
            if not bt_str or bt_str.lower() == "nan":
                return "unknown"
            # Parse hour from timestamp like "2026-01-29 03:30:00-05:00"
            if " " in bt_str:
                hour = int(bt_str.split(" ")[1].split(":")[0])
            else:
                hour = 10  # fallback to US if can't parse
            return "london" if hour < 8 else "us"
        except Exception:
            return "unknown"

    if stratify and "label" in df.columns:
        # Build a composite stratification key from label + session so that
        # both London and US images are proportionally represented in the
        # train and val sets for every label class.
        if "breakout_time" in df.columns:
            df["_session"] = df["breakout_time"].apply(_infer_session)
            df["_strat_key"] = df["label"].astype(str) + "__" + df["_session"]
        else:
            df["_strat_key"] = df["label"].astype(str)

        train_parts = []
        val_parts = []
        for _key, group in df.groupby("_strat_key"):
            n_val = max(1, int(len(group) * val_fraction))
            shuffled = group.sample(frac=1, random_state=rng)
            val_parts.append(shuffled.iloc[:n_val])
            train_parts.append(shuffled.iloc[n_val:])

        train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=rng)
        val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=rng)

        # Log stratification breakdown for audit
        if "_session" in df.columns:
            for split_name, split_df in [("train", train_df), ("val", val_df)]:
                session_counts = split_df["_session"].value_counts().to_dict()
                label_counts = split_df["label"].value_counts().to_dict()
                logger.info(
                    "  %s split — sessions: %s | labels: %s",
                    split_name,
                    session_counts,
                    label_counts,
                )

        # Drop helper columns before saving
        for col in ("_session", "_strat_key"):
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in val_df.columns:
                val_df = val_df.drop(columns=[col])
    else:
        shuffled = df.sample(frac=1, random_state=rng)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_df = shuffled.iloc[:n_val]
        train_df = shuffled.iloc[n_val:]

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(
        "Split dataset: %d train / %d val (%.1f%%) — %s, %s",
        len(train_df),
        len(val_df),
        val_fraction * 100,
        train_path,
        val_path,
    )

    return train_path, val_path


# ---------------------------------------------------------------------------
# Dataset validation helper
# ---------------------------------------------------------------------------


def validate_dataset(csv_path: str, check_images: bool = True) -> dict[str, Any]:
    """Validate a dataset CSV and optionally check that all images exist.

    Returns a report dict with counts, missing images, label distribution, etc.
    """
    if not os.path.isfile(csv_path):
        return {"valid": False, "error": f"CSV not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    report: dict[str, Any] = {
        "valid": True,
        "total_rows": len(df),
        "columns": list(df.columns),
        "label_distribution": {},
        "symbols": [],
        "missing_images": 0,
        "empty_image_paths": 0,
    }

    if "label" in df.columns:
        report["label_distribution"] = df["label"].value_counts().to_dict()

    if "symbol" in df.columns:
        report["symbols"] = sorted(df["symbol"].unique().tolist())

    if check_images and "image_path" in df.columns:
        missing = 0
        empty = 0
        for path in df["image_path"]:
            if not path or (isinstance(path, float) and np.isnan(path)):
                empty += 1
                continue
            if not os.path.isfile(str(path)):
                missing += 1

        report["missing_images"] = missing
        report["empty_image_paths"] = empty
        if missing > 0:
            report["valid"] = False
            report["error"] = f"{missing} images not found on disk"

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    """Command-line interface for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate labeled chart dataset for CNN training",
    )
    sub = parser.add_subparsers(dest="command")

    # Generate
    gen_parser = sub.add_parser("generate", help="Generate dataset")
    gen_parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to process (e.g. MGC MES MNQ)",
    )
    gen_parser.add_argument("--days", type=int, default=90, help="Days of history")
    gen_parser.add_argument("--output-dir", default="dataset", help="Output directory")
    gen_parser.add_argument("--image-dir", default="dataset/images", help="Image output directory")
    gen_parser.add_argument("--source", default="cache", choices=["cache", "csv", "massive"])
    gen_parser.add_argument("--csv-bars-dir", default="data/bars", help="Directory for CSV bar files")
    gen_parser.add_argument("--window-size", type=int, default=240)
    gen_parser.add_argument("--step-size", type=int, default=30)
    gen_parser.add_argument("--max-per-label", type=int, default=0, help="Max samples per label (0=unlimited)")
    gen_parser.add_argument("--dpi", type=int, default=150)
    gen_parser.add_argument("--no-skip", action="store_true", help="Re-render existing images")
    gen_parser.add_argument(
        "--session",
        default="us",
        choices=["us", "london", "both"],
        help="ORB session: 'us' (OR 09:30–10:00), 'london' (OR 03:00–03:30), or 'both'",
    )

    # Split
    split_parser = sub.add_parser("split", help="Split dataset into train/val")
    split_parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    split_parser.add_argument("--val-frac", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=42)

    # Validate
    val_parser = sub.add_parser("validate", help="Validate dataset")
    val_parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    val_parser.add_argument("--no-check-images", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command == "generate":
        cfg = DatasetConfig(
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            max_samples_per_label=args.max_per_label,
            chart_dpi=args.dpi,
            skip_existing=not args.no_skip,
            bars_source=args.source,
            csv_bars_dir=args.csv_bars_dir,
            orb_session=args.session,
        )
        stats = generate_dataset(
            symbols=args.symbols,
            days_back=args.days,
            config=cfg,
        )
        print(f"\n{stats.summary()}")

    elif args.command == "split":
        train_path, val_path = split_dataset(
            csv_path=args.csv,
            val_fraction=args.val_frac,
            random_seed=args.seed,
        )
        print(f"\nTrain: {train_path}")
        print(f"Val:   {val_path}")

    elif args.command == "validate":
        report = validate_dataset(
            csv_path=args.csv,
            check_images=not args.no_check_images,
        )
        print("\nDataset validation report:")
        for k, v in report.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
