#!/usr/bin/env python
"""
Incremental Dataset Build
=========================
Nightly script that:
  1. Calls the data service to ensure all enabled assets have fresh,
     gap-free 1-minute bar data in Postgres (via POST /bars/fill/all).
  2. Waits for the fill to complete (polling /bars/fill/status).
  3. Runs the dataset generator in incremental mode — only generating
     chart images for trading sessions that don't already have images,
     so the full labels.csv grows over time without regenerating history.
  4. Re-splits train.csv / val.csv with the updated dataset.
  5. Writes a build summary to models/dataset_build_log.jsonl.

This replaces the old "full regeneration" approach and implements:
  - Priority 2: Incremental dataset build (nightly script, no full regen)
  - Priority 3: Expand dataset to 10k+ images (90 days × 6+ symbols)

Usage
-----
From project root (standalone):
    PYTHONPATH=src python scripts/incremental_dataset_build.py

With explicit data service URL:
    DATA_SERVICE_URL=http://localhost:8000 \\
    PYTHONPATH=src python scripts/incremental_dataset_build.py

Skip the fill step (data already fresh):
    SKIP_FILL=1 PYTHONPATH=src python scripts/incremental_dataset_build.py

Force full regeneration (ignore existing images):
    FORCE_REGEN=1 PYTHONPATH=src python scripts/incremental_dataset_build.py

Environment Variables
---------------------
DATA_SERVICE_URL        Base URL of the data service (default: http://localhost:8000)
DATA_SERVICE_API_KEY    API key for the data service (if AUTH_REQUIRED=1)
FILL_DAYS_BACK          Days of history to ensure are filled (default: 90)
FILL_POLL_INTERVAL_SEC  Seconds between fill-status polls (default: 10)
FILL_TIMEOUT_SEC        Max seconds to wait for fill to complete (default: 600)
DATASET_SYMBOLS         Comma-separated symbol list (default: 23-symbol universe covering
                        metals, energy, equity index, FX, rates, agri, and crypto — see
                        BuildConfig.symbols for the full list)
DATASET_DAYS_BACK       Days of bar history to use for image generation (default: 90)
DATASET_CHART_DPI       Chart image DPI (default: 150)
DATASET_ORB_SESSION     ORB session: us, london, both, or all (default: all — full 24h
                        Globex coverage recommended for 10k+ dataset)
SKIP_FILL               Set to "1" to skip the data fill step
SKIP_SPLIT              Set to "1" to skip the train/val split step
FORCE_REGEN             Set to "1" to regenerate all images (ignore existing)
PYTHONPATH              Must include the project src/ directory
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("incremental_dataset_build")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
LABELS_CSV = DATASET_DIR / "labels.csv"
TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
BUILD_LOG = MODEL_DIR / "dataset_build_log.jsonl"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    """All tunables for the incremental build pipeline."""

    # Data fill settings
    data_service_url: str = "http://localhost:8000"
    api_key: str = ""
    fill_days_back: int = 90
    fill_poll_interval: int = 10  # seconds
    fill_timeout: int = 600  # seconds

    # Dataset generation settings
    symbols: list[str] = field(
        default_factory=lambda: [
            # ── Micro metals ──────────────────────────────────────────────
            "MGC",  # Micro Gold
            "SIL",  # Micro Silver
            "MHG",  # Micro Copper
            # ── Micro energy ──────────────────────────────────────────────
            "MCL",  # Micro Crude Oil
            "MNG",  # Micro Natural Gas (data via NG=F)
            # ── Micro equity index ────────────────────────────────────────
            "MES",  # Micro S&P 500
            "MNQ",  # Micro Nasdaq-100
            "M2K",  # Micro Russell 2000
            "MYM",  # Micro Dow Jones
            # ── FX (CME standard + micro) ─────────────────────────────────
            "6E",  # Euro FX
            "6B",  # British Pound
            "6J",  # Japanese Yen
            "6A",  # Australian Dollar
            "6C",  # Canadian Dollar
            "6S",  # Swiss Franc
            # ── Interest rates (CBOT) ─────────────────────────────────────
            "ZN",  # 10-Year T-Note
            "ZB",  # 30-Year T-Bond
            # ── Agricultural (CBOT) ───────────────────────────────────────
            "ZC",  # Corn
            "ZS",  # Soybeans
            "ZW",  # Wheat
            # ── Crypto ────────────────────────────────────────────────────
            "MBT",  # Micro Bitcoin
            "MET",  # Micro Ether
        ]
    )
    days_back: int = 90
    chart_dpi: int = 150
    orb_session: str = "all"  # "us" | "london" | "both" | "all" | any named session key
    skip_existing: bool = True  # True = incremental, False = full regen

    # Train/val split settings
    val_fraction: float = 0.15
    stratify: bool = True
    random_seed: int = 42

    # Behavior flags
    skip_fill: bool = False
    skip_split: bool = False

    @classmethod
    def from_env(cls) -> "BuildConfig":
        cfg = cls()
        cfg.data_service_url = os.getenv("DATA_SERVICE_URL", cfg.data_service_url).rstrip("/")
        cfg.api_key = os.getenv("DATA_SERVICE_API_KEY", cfg.api_key)
        cfg.fill_days_back = int(os.getenv("FILL_DAYS_BACK", str(cfg.fill_days_back)))
        cfg.fill_poll_interval = int(os.getenv("FILL_POLL_INTERVAL_SEC", str(cfg.fill_poll_interval)))
        cfg.fill_timeout = int(os.getenv("FILL_TIMEOUT_SEC", str(cfg.fill_timeout)))

        sym_str = os.getenv("DATASET_SYMBOLS", "")
        if sym_str.strip():
            cfg.symbols = [s.strip() for s in sym_str.split(",") if s.strip()]

        cfg.days_back = int(os.getenv("DATASET_DAYS_BACK", str(cfg.days_back)))
        cfg.chart_dpi = int(os.getenv("DATASET_CHART_DPI", str(cfg.chart_dpi)))
        # Support both DATASET_ORB_SESSION and CNN_ORB_SESSION (todo Quick Command alias).
        # CNN_ORB_SESSION takes precedence if both are set.
        cfg.orb_session = os.getenv("DATASET_ORB_SESSION", cfg.orb_session)
        cfg.orb_session = os.getenv("CNN_ORB_SESSION", cfg.orb_session)

        cfg.skip_fill = os.getenv("SKIP_FILL", "0").strip() in ("1", "true", "yes")
        cfg.skip_split = os.getenv("SKIP_SPLIT", "0").strip() in ("1", "true", "yes")

        # FORCE_REGEN overrides skip_existing
        if os.getenv("FORCE_REGEN", "0").strip() in ("1", "true", "yes"):
            cfg.skip_existing = False

        return cfg


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    """Accumulated result for the full incremental build run."""

    started_at: str = ""
    finished_at: str = ""
    status: str = "unknown"  # success | partial | failed

    # Stage 1 — fill
    fill_skipped: bool = False
    fill_job_id: str = ""
    fill_status: str = ""
    fill_bars_added: int = 0
    fill_duration_seconds: float = 0.0
    fill_errors: list[str] = field(default_factory=list)

    # Stage 2 — dataset generation
    dataset_new_images: int = 0
    dataset_total_images: int = 0
    dataset_skipped_existing: int = 0
    dataset_total_rows: int = 0
    dataset_label_distribution: dict[str, int] = field(default_factory=dict)
    dataset_duration_seconds: float = 0.0
    dataset_errors: list[str] = field(default_factory=list)

    # Stage 3 — split
    split_skipped: bool = False
    split_train_rows: int = 0
    split_val_rows: int = 0
    split_duration_seconds: float = 0.0

    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests dependency)
# ---------------------------------------------------------------------------


def _http_get(url: str, api_key: str = "", timeout: int = 30) -> dict[str, Any]:
    """Make a GET request and return the parsed JSON body."""
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("X-API-Key", api_key)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:200]}") from exc


def _http_post(url: str, payload: dict, api_key: str = "", timeout: int = 30) -> dict[str, Any]:
    """Make a POST request with a JSON body and return the parsed JSON body."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:200]}") from exc


def _data_service_available(base_url: str, api_key: str = "") -> bool:
    """Return True if the data service health endpoint responds OK."""
    try:
        resp = _http_get(f"{base_url}/health", api_key=api_key, timeout=5)
        return isinstance(resp, dict)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage 1 — Fill gaps via data service
# ---------------------------------------------------------------------------


def stage_fill(cfg: BuildConfig, result: BuildResult) -> bool:
    """Trigger POST /bars/fill/all and wait for completion.

    Returns True on success (or if fill was skipped), False on error.
    """
    if cfg.skip_fill:
        logger.info("Stage 1: Fill — SKIPPED (SKIP_FILL=1)")
        result.fill_skipped = True
        return True

    logger.info("=" * 60)
    logger.info("Stage 1: Ensure fresh bar data via data service")
    logger.info("=" * 60)
    logger.info("Data service: %s", cfg.data_service_url)
    logger.info("Symbols:      %s", ", ".join(cfg.symbols))
    logger.info("Days back:    %d", cfg.fill_days_back)

    t0 = time.monotonic()

    # ── Check data service availability ─────────────────────────────────
    if not _data_service_available(cfg.data_service_url, cfg.api_key):
        logger.warning(
            "Data service not reachable at %s — falling back to direct fill",
            cfg.data_service_url,
        )
        return _stage_fill_direct(cfg, result, t0)

    # ── Convert symbol short names to Yahoo tickers ───────────────────────
    try:
        from lib.analysis.dataset_generator import _resolve_ticker

        tickers = [_resolve_ticker(s) for s in cfg.symbols]
    except ImportError:
        tickers = [f"{s}=F" if "=" not in s else s for s in cfg.symbols]

    # ── POST /bars/fill/all ───────────────────────────────────────────────
    try:
        resp = _http_post(
            f"{cfg.data_service_url}/bars/fill/all",
            payload={"symbols": tickers, "days_back": cfg.fill_days_back, "interval": "1m"},
            api_key=cfg.api_key,
            timeout=30,
        )
        job_id = resp.get("job_id", "")
        result.fill_job_id = job_id
        logger.info("Fill job submitted: %s (%d symbols)", job_id, len(tickers))
    except Exception as exc:
        logger.warning("Failed to submit fill job: %s — falling back to direct fill", exc)
        return _stage_fill_direct(cfg, result, t0)

    # ── Poll /bars/fill/status ────────────────────────────────────────────
    deadline = time.monotonic() + cfg.fill_timeout
    last_progress = -1

    while time.monotonic() < deadline:
        try:
            status_resp = _http_get(
                f"{cfg.data_service_url}/bars/fill/status?job_id={job_id}",
                api_key=cfg.api_key,
                timeout=10,
            )
            job_status = status_resp.get("status", "running")
            progress = status_resp.get("progress", 0)

            if progress != last_progress:
                logger.info(
                    "  Fill progress: %d%% (%d/%d symbols) — status: %s",
                    progress,
                    status_resp.get("completed_count", 0),
                    status_resp.get("symbol_count", len(tickers)),
                    job_status,
                )
                last_progress = progress

            if job_status in ("complete", "partial", "failed"):
                result.fill_status = job_status
                result.fill_bars_added = status_resp.get("total_bars_added", 0)
                result.fill_errors = status_resp.get("errors", [])[:10]
                result.fill_duration_seconds = round(time.monotonic() - t0, 2)

                if job_status == "failed":
                    logger.error("Fill job failed: %s", result.fill_errors)
                    result.errors.append(f"Fill job failed: {result.fill_errors[:3]}")
                    return False

                if job_status == "partial":
                    logger.warning(
                        "Fill job partially completed: %d errors, +%d bars",
                        len(result.fill_errors),
                        result.fill_bars_added,
                    )
                else:
                    logger.info(
                        "✅ Fill complete: +%d bars across %d symbols (%.1f min)",
                        result.fill_bars_added,
                        len(tickers),
                        result.fill_duration_seconds / 60,
                    )
                return True

        except Exception as exc:
            logger.debug("Fill status poll error: %s", exc)

        time.sleep(cfg.fill_poll_interval)

    # Timeout
    result.fill_status = "timeout"
    result.fill_duration_seconds = round(time.monotonic() - t0, 2)
    msg = f"Fill job timed out after {cfg.fill_timeout}s"
    logger.warning(msg)
    result.errors.append(msg)
    # Not fatal — we may have partial data; continue to dataset generation
    return True


def _stage_fill_direct(cfg: BuildConfig, result: BuildResult, t0: float) -> bool:
    """Fallback: run backfill directly (in-process) without data service.

    Used when the data service is not reachable (e.g. standalone script run
    outside Docker).
    """
    logger.info("Running direct in-process backfill (data service unavailable)")
    try:
        from lib.analysis.dataset_generator import _resolve_ticker
        from lib.services.engine.backfill import run_backfill

        tickers = [_resolve_ticker(s) for s in cfg.symbols]
        summary = run_backfill(
            symbols=tickers,
            days_back=cfg.fill_days_back,
            chunk_days=5,
            interval="1m",
        )
        result.fill_status = summary.get("status", "unknown")
        result.fill_bars_added = summary.get("total_bars_added", 0)
        result.fill_errors = summary.get("errors", [])[:10]
        result.fill_duration_seconds = round(time.monotonic() - t0, 2)

        logger.info(
            "✅ Direct fill %s: +%d bars (%.1f min)",
            result.fill_status,
            result.fill_bars_added,
            result.fill_duration_seconds / 60,
        )
        return result.fill_status != "failed"

    except Exception as exc:
        result.fill_status = "error"
        result.fill_duration_seconds = round(time.monotonic() - t0, 2)
        msg = f"Direct fill failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Stage 2 — Incremental dataset generation
# ---------------------------------------------------------------------------


def stage_generate(cfg: BuildConfig, result: BuildResult) -> bool:
    """Generate new chart images for any trading sessions not yet in the dataset.

    With ``skip_existing=True`` (default / incremental mode), images that
    already exist on disk are skipped, so only newly-available sessions
    produce new images.  The labels.csv is appended to, not regenerated.

    Returns True on success (including partial success), False on hard failure.
    """
    logger.info("=" * 60)
    logger.info("Stage 2: Incremental dataset generation")
    logger.info("=" * 60)
    logger.info("Symbols:       %s", ", ".join(cfg.symbols))
    logger.info("Days back:     %d", cfg.days_back)
    logger.info("ORB session:   %s", cfg.orb_session)
    logger.info("Skip existing: %s", cfg.skip_existing)
    logger.info("Chart DPI:     %d", cfg.chart_dpi)

    t0 = time.monotonic()

    try:
        from lib.analysis.dataset_generator import DatasetConfig, generate_dataset

        ds_config = DatasetConfig(
            bars_source="db",  # always use the DB (canonical store)
            skip_existing=cfg.skip_existing,
            chart_dpi=cfg.chart_dpi,
            orb_session=cfg.orb_session,
        )

        stats = generate_dataset(
            symbols=cfg.symbols,
            days_back=cfg.days_back,
            config=ds_config,
        )

        result.dataset_new_images = stats.total_images - stats.skipped_existing
        result.dataset_total_images = stats.total_images
        result.dataset_skipped_existing = stats.skipped_existing
        result.dataset_label_distribution = dict(stats.label_distribution)
        result.dataset_duration_seconds = round(time.monotonic() - t0, 2)
        result.dataset_errors = stats.errors[:20]

        # Count rows in labels.csv
        if LABELS_CSV.exists():
            with contextlib.suppress(Exception):
                import pandas as pd

                result.dataset_total_rows = len(pd.read_csv(LABELS_CSV))

        logger.info(
            "✅ Dataset generation complete:\n"
            "   Total images:    %d\n"
            "   New images:      %d\n"
            "   Skipped:         %d\n"
            "   Labels CSV rows: %d\n"
            "   Label dist:      %s\n"
            "   Duration:        %.1f min",
            result.dataset_total_images,
            result.dataset_new_images,
            result.dataset_skipped_existing,
            result.dataset_total_rows,
            result.dataset_label_distribution,
            result.dataset_duration_seconds / 60,
        )

        if stats.errors:
            for err in stats.errors[:5]:
                logger.warning("  ⚠ %s", err)

        return True

    except ImportError as exc:
        msg = f"Dataset generator not available: {exc}"
        logger.error(msg)
        result.errors.append(msg)
        result.dataset_duration_seconds = round(time.monotonic() - t0, 2)
        return False
    except Exception as exc:
        msg = f"Dataset generation failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        result.dataset_duration_seconds = round(time.monotonic() - t0, 2)
        return False


# ---------------------------------------------------------------------------
# Stage 3 — Train/val split
# ---------------------------------------------------------------------------


def stage_split(cfg: BuildConfig, result: BuildResult) -> bool:
    """Re-split labels.csv into train.csv and val.csv.

    With stratified sampling enabled, the class distribution is preserved
    across splits.  This stage is fast (~seconds) and always runs unless
    explicitly skipped.

    Returns True on success, False on failure.
    """
    if cfg.skip_split:
        logger.info("Stage 3: Split — SKIPPED (SKIP_SPLIT=1)")
        result.split_skipped = True
        return True

    if not LABELS_CSV.exists():
        logger.warning("Stage 3: labels.csv not found — skipping split")
        result.split_skipped = True
        return True

    logger.info("=" * 60)
    logger.info("Stage 3: Train/val split")
    logger.info("=" * 60)

    t0 = time.monotonic()

    try:
        import pandas as pd

        from lib.analysis.dataset_generator import split_dataset

        labels_df = pd.read_csv(LABELS_CSV)
        total_rows = len(labels_df)

        if total_rows < 10:
            logger.warning("Too few rows in labels.csv (%d) — skipping split", total_rows)
            result.split_skipped = True
            return True

        logger.info("Splitting %d rows (val=%.0f%%)", total_rows, cfg.val_fraction * 100)

        train_path, val_path = split_dataset(
            str(LABELS_CSV),
            val_fraction=cfg.val_fraction,
            output_dir=str(DATASET_DIR),
            stratify=cfg.stratify,
            random_seed=cfg.random_seed,
        )

        result.split_train_rows = len(pd.read_csv(train_path)) if Path(train_path).exists() else 0
        result.split_val_rows = len(pd.read_csv(val_path)) if Path(val_path).exists() else 0
        result.split_duration_seconds = round(time.monotonic() - t0, 2)

        logger.info(
            "✅ Split complete: %d train / %d val rows (%.1fs)",
            result.split_train_rows,
            result.split_val_rows,
            result.split_duration_seconds,
        )
        return True

    except ImportError as exc:
        msg = f"split_dataset not available: {exc}"
        logger.error(msg)
        result.errors.append(msg)
        result.split_duration_seconds = round(time.monotonic() - t0, 2)
        return False
    except Exception as exc:
        msg = f"Split failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        result.split_duration_seconds = round(time.monotonic() - t0, 2)
        return False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _append_build_log(result: BuildResult) -> None:
    """Append the build result to models/dataset_build_log.jsonl."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        entry = result.to_dict()
        entry["_ts"] = datetime.now(tz=_EST).isoformat()
        with BUILD_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.debug("Build log written to %s", BUILD_LOG)
    except Exception as exc:
        logger.warning("Failed to write build log: %s", exc)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(cfg: BuildConfig) -> BuildResult:
    """Execute the full incremental dataset build pipeline."""
    result = BuildResult(started_at=datetime.now(tz=_EST).isoformat())

    logger.info("=" * 60)
    logger.info("  Incremental Dataset Build")
    logger.info("  %s", result.started_at)
    logger.info("=" * 60)

    # Ensure output directories exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images").mkdir(parents=True, exist_ok=True)

    overall_ok = True

    # ── Stage 1: Fill ─────────────────────────────────────────────────────
    fill_ok = stage_fill(cfg, result)
    if not fill_ok:
        logger.warning("Fill stage failed — dataset generation may use stale data")
        overall_ok = False

    # ── Stage 2: Generate ─────────────────────────────────────────────────
    gen_ok = stage_generate(cfg, result)
    if not gen_ok:
        logger.error("Dataset generation failed — aborting")
        result.status = "failed"
        result.finished_at = datetime.now(tz=_EST).isoformat()
        _append_build_log(result)
        return result

    # ── Stage 3: Split ────────────────────────────────────────────────────
    split_ok = stage_split(cfg, result)
    if not split_ok:
        logger.warning("Split stage failed — train/val CSVs may be stale")
        overall_ok = False

    # ── Finalise ──────────────────────────────────────────────────────────
    result.finished_at = datetime.now(tz=_EST).isoformat()

    if not gen_ok:
        result.status = "failed"
    elif not overall_ok:
        result.status = "partial"
    else:
        result.status = "success"

    total_seconds = result.fill_duration_seconds + result.dataset_duration_seconds + result.split_duration_seconds

    logger.info("=" * 60)
    logger.info("  Build %s", result.status.upper())
    logger.info("  New images:   %d", result.dataset_new_images)
    logger.info("  Total images: %d", result.dataset_total_images)
    logger.info("  Total rows:   %d", result.dataset_total_rows)
    logger.info("  Train/val:    %d / %d", result.split_train_rows, result.split_val_rows)
    logger.info("  Bars added:   %d", result.fill_bars_added)
    logger.info("  Duration:     %.1f min", total_seconds / 60)
    if result.errors:
        logger.warning("  Errors:       %d", len(result.errors))
        for e in result.errors[:5]:
            logger.warning("    - %s", e)
    logger.info("=" * 60)

    _append_build_log(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental dataset build: fill gaps → generate images → split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Symbols to process (e.g. MGC MES MNQ). Overrides DATASET_SYMBOLS env var.",
    )
    p.add_argument(
        "--days-back",
        type=int,
        default=None,
        metavar="N",
        help="Days of history to use for image generation (default: 90).",
    )
    p.add_argument(
        "--fill-days-back",
        type=int,
        default=None,
        metavar="N",
        help="Days of bar history to ensure are filled (default: same as --days-back).",
    )
    p.add_argument(
        "--data-service-url",
        default=None,
        metavar="URL",
        help="Base URL of the data service (default: http://localhost:8000).",
    )
    # All valid session keys (mirrors _ALL_SESSION_KEYS in dataset_generator.py)
    _SESSION_CHOICES = [
        "all",  # All 9 sessions across the full Globex day (recommended)
        "both",  # London + US (backward-compatible two-session alias)
        "us",  # US Equity Open  09:30–10:00 ET
        "london",  # London Open  03:00–03:30 ET
        "london_ny",  # London-NY Crossover  08:00–08:30 ET
        "frankfurt",  # Frankfurt/Xetra  03:00–03:30 ET
        "cme",  # CME Globex re-open  18:00–18:30 ET
        "sydney",  # Sydney/ASX  18:30–19:00 ET
        "tokyo",  # Tokyo/TSE  19:00–19:30 ET
        "shanghai",  # Shanghai/HK  21:00–21:30 ET
        "cme_settle",  # CME Settlement  14:00–14:30 ET
    ]
    p.add_argument(
        "--orb-session",
        "--session",  # alias used in todo Quick Commands
        choices=_SESSION_CHOICES,
        dest="orb_session",
        default=None,
        metavar="SESSION",
        help=(
            "ORB session(s) to generate images for. "
            "Choices: %(choices)s. "
            "Use 'all' for full 24-hour Globex coverage (recommended for 10k+ dataset). "
            "Default: both (London + US)."
        ),
    )
    p.add_argument(
        "--skip-fill",
        action="store_true",
        help="Skip the data fill stage (assume data is already fresh).",
    )
    p.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip the train/val split stage.",
    )
    p.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate all images even if they already exist on disk.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="DPI",
        help="Chart image DPI (default: 150).",
    )
    return p.parse_args()


def main() -> int:
    """Entry point — returns 0 on success, 1 on failure."""
    args = _parse_args()

    # Build config from env, then apply CLI overrides
    cfg = BuildConfig.from_env()

    if args.symbols:
        cfg.symbols = args.symbols
    if args.days_back is not None:
        cfg.days_back = args.days_back
    if args.fill_days_back is not None:
        cfg.fill_days_back = args.fill_days_back
    elif args.days_back is not None:
        cfg.fill_days_back = args.days_back  # keep fill and gen in sync
    if args.data_service_url:
        cfg.data_service_url = args.data_service_url.rstrip("/")
    if args.orb_session:
        cfg.orb_session = args.orb_session
        # Keep fill scope in sync: when generating all sessions we need the
        # full 90-day bar history regardless of which session is requested.
        # fill_days_back was already sync'd with days_back above; no extra
        # action needed here — just a note for clarity.
    if args.skip_fill:
        cfg.skip_fill = True
    if args.skip_split:
        cfg.skip_split = True
    if args.force_regen:
        cfg.skip_existing = False
    if args.dpi is not None:
        cfg.chart_dpi = args.dpi

    result = run(cfg)

    return 0 if result.status in ("success", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
