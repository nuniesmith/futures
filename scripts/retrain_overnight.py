#!/usr/bin/env python
"""
Overnight CNN Retraining Orchestrator
======================================
End-to-end pipeline that runs during off-hours (12:00–00:00 ET) and
overnight to continuously improve the breakout CNN model.

Pipeline stages:
  1. **Dataset Refresh** — Pull latest historical bars, generate new
     chart images + labels, append to the existing dataset.
  2. **Train/Val Split** — Re-split the full dataset with stratified
     sampling so the model always trains on the freshest data.
  3. **GPU Training** — Run the EfficientNetV2-S hybrid trainer with
     mixed precision, class weighting, cosine schedule, early stopping.
  4. **Validation Gate** — Evaluate the new model against the current
     champion on the held-out val set.  The new model must beat the
     champion on accuracy AND maintain acceptable precision/recall
     before it gets promoted.
  5. **Model Promotion** — Atomically swap ``breakout_cnn_best.pt`` so
     the live inference pipeline picks up the new model on the next
     prediction call.
  6. **Cleanup** — Archive old model checkpoints, prune stale training
     artifacts, write an audit log.

Scheduling:
  - Designed to be called by the engine scheduler during OFF_HOURS
    (``ActionType.TRAIN_BREAKOUT_CNN``), or run standalone via cron /
    Task Scheduler / systemd timer.
  - The script is idempotent: if it detects a run already completed
    today it will skip unless ``--force`` is passed.
  - Safe to kill at any point — the champion model is only replaced on
    successful validation.

Session-aware time windows (all times Eastern):
  ┌──────────────────────────────────────────────────────────┐
  │ 12:00–17:00 ET  Dataset generation (CPU-bound I/O)      │
  │ 17:00–00:00 ET  GPU training window (GPU-bound)         │
  │ 00:00–03:00 ET  Validation + promotion + cleanup        │
  │ 03:00–12:00 ET  ACTIVE SESSION — NO TRAINING            │
  └──────────────────────────────────────────────────────────┘
  With ``--immediate`` the pipeline runs all stages back-to-back
  regardless of clock time (useful for manual kicks).

Usage:
    cd futures

    # Full pipeline (respects time windows):
    PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py

    # Immediate run (ignore time windows):
    PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate

    # Skip dataset generation (just retrain on existing data):
    PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --skip-dataset --immediate

    # Dry run (validate only, no promotion):
    PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --dry-run --immediate

Environment variables:
    CNN_RETRAIN_MIN_ACC       Minimum val accuracy to promote (default: 80.0)
    CNN_RETRAIN_MIN_PRECISION Minimum precision to promote (default: 75.0)
    CNN_RETRAIN_MIN_RECALL    Minimum recall to promote (default: 70.0)
    CNN_RETRAIN_IMPROVEMENT   Required accuracy improvement over champion (default: 0.0)
    CNN_RETRAIN_EPOCHS        Training epochs (default: 25)
    CNN_RETRAIN_BATCH_SIZE    Batch size (default: 64)
    CNN_RETRAIN_LR            Learning rate (default: 2e-4)
    CNN_RETRAIN_PATIENCE      Early stopping patience (default: 8)
    CNN_RETRAIN_SYMBOLS       Comma-separated symbols (default: MGC,MES,MNQ)
    CNN_RETRAIN_DAYS_BACK     Days of history for dataset gen (default: 90)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

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
_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_LOG_DATEFMT)
logger = logging.getLogger("retrain_overnight")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = str(PROJECT_ROOT / "dataset")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
MODEL_DIR = str(PROJECT_ROOT / "models")
CHAMPION_PATH = os.path.join(MODEL_DIR, "breakout_cnn_best.pt")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
AUDIT_LOG_PATH = os.path.join(MODEL_DIR, "retrain_audit.jsonl")
LOCKFILE_PATH = os.path.join(MODEL_DIR, ".retrain_lock")

import threading  # needed for lock heartbeat

# Session boundaries (Eastern Time hours)
ACTIVE_SESSION_START = 3  # 03:00 ET
ACTIVE_SESSION_END = 12  # 12:00 ET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RetrainConfig:
    """All tunables for the overnight retraining pipeline."""

    # Dataset generation
    symbols: list[str] = field(default_factory=lambda: ["MGC", "MES", "MNQ"])
    days_back: int = 90
    skip_existing_images: bool = True
    chart_dpi: int = 150
    bars_source: str = "cache"
    orb_session: str = "both"  # "us", "london", or "both"

    # Train/val split
    val_fraction: float = 0.15
    stratify: bool = True
    random_seed: int = 42

    # Training hyperparameters
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    freeze_epochs: int = 3
    warmup_epochs: int = 1
    patience: int = 8
    label_smoothing: float = 0.05
    dropout: float = 0.4
    use_amp: bool = True
    num_workers: int = 4

    # Validation gate thresholds
    min_val_accuracy: float = 80.0
    min_precision: float = 75.0
    min_recall: float = 70.0
    min_improvement: float = 0.0  # must beat champion by this much

    # Model management
    max_archived_models: int = 10
    max_checkpoints_per_run: int = 3  # keep best-acc, best-loss, final

    # Behavior
    immediate: bool = False  # ignore time windows
    skip_dataset: bool = False  # skip dataset generation stage
    dry_run: bool = False  # validate only, no promotion
    force: bool = False  # run even if already ran today

    @classmethod
    def from_env(cls) -> RetrainConfig:
        """Build config from environment variables with sensible defaults."""
        cfg = cls()

        symbols_str = os.getenv("CNN_RETRAIN_SYMBOLS", "")
        if symbols_str.strip():
            cfg.symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

        cfg.days_back = int(os.getenv("CNN_RETRAIN_DAYS_BACK", str(cfg.days_back)))
        cfg.epochs = int(os.getenv("CNN_RETRAIN_EPOCHS", str(cfg.epochs)))
        cfg.batch_size = int(os.getenv("CNN_RETRAIN_BATCH_SIZE", str(cfg.batch_size)))
        cfg.lr = float(os.getenv("CNN_RETRAIN_LR", str(cfg.lr)))
        cfg.patience = int(os.getenv("CNN_RETRAIN_PATIENCE", str(cfg.patience)))
        cfg.min_val_accuracy = float(os.getenv("CNN_RETRAIN_MIN_ACC", str(cfg.min_val_accuracy)))
        cfg.min_precision = float(os.getenv("CNN_RETRAIN_MIN_PRECISION", str(cfg.min_precision)))
        cfg.min_recall = float(os.getenv("CNN_RETRAIN_MIN_RECALL", str(cfg.min_recall)))
        cfg.min_improvement = float(os.getenv("CNN_RETRAIN_IMPROVEMENT", str(cfg.min_improvement)))
        cfg.orb_session = os.getenv("CNN_RETRAIN_ORB_SESSION", cfg.orb_session)

        return cfg


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class RetrainResult:
    """Captures the outcome of a full retraining run for audit."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    status: str = "pending"  # pending | success | failed | skipped | gate_rejected

    # Dataset stage
    dataset_rows: int = 0
    dataset_new_images: int = 0
    dataset_duration_seconds: float = 0.0

    # Split stage
    train_samples: int = 0
    val_samples: int = 0

    # Training stage
    training_epochs_run: int = 0
    training_duration_seconds: float = 0.0
    best_val_accuracy: float = 0.0
    best_val_loss: float = 0.0
    best_precision: float = 0.0
    best_recall: float = 0.0
    candidate_model_path: str = ""

    # Validation gate
    champion_accuracy: float = 0.0
    gate_passed: bool = False
    gate_reason: str = ""

    # Promotion
    promoted: bool = False
    promoted_model_path: str = ""
    archived_champion_path: str = ""

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _now_et() -> datetime:
    """Current time in Eastern."""
    return datetime.now(tz=_EST)


def _is_active_session(now: datetime | None = None) -> bool:
    """Return True if we're in the active trading session (03:00–12:00 ET)."""
    now = now or _now_et()
    return ACTIVE_SESSION_START <= now.hour < ACTIVE_SESSION_END


def _already_ran_today() -> bool:
    """Check if a successful retrain already completed today."""
    if not os.path.isfile(AUDIT_LOG_PATH):
        return False
    today_str = _now_et().strftime("%Y-%m-%d")
    try:
        with open(AUDIT_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("status") == "success" and entry.get("started_at", "").startswith(today_str):
                    return True
    except Exception:
        pass
    return False


# Heartbeat: the lock holder touches this file periodically to prove it's alive.
_LOCK_HEARTBEAT_PATH = LOCKFILE_PATH + ".heartbeat"
_LOCK_HEARTBEAT_INTERVAL = 30  # seconds between heartbeat touches
_lock_heartbeat_stop: threading.Event | None = None
_lock_heartbeat_thread: threading.Thread | None = None

# Stale thresholds
_LOCK_STALE_AGE = timedelta(hours=2)  # absolute age of lock
_LOCK_HEARTBEAT_STALE = timedelta(minutes=5)  # no heartbeat for this long → dead


def _is_lock_holder_alive(lock_data: dict) -> bool:
    """Determine whether the process that holds the lock is still actively training.

    Checks in order:
      1. Heartbeat file: if present and recently touched, holder is alive.
      2. PID liveness: only meaningful if we're on the same host (not in
         Docker where PID 1 is always the main entrypoint, not the trainer).
      3. Lock age: if older than _LOCK_STALE_AGE, assume dead.
    """
    # --- Check heartbeat file (most reliable) ---
    if os.path.isfile(_LOCK_HEARTBEAT_PATH):
        try:
            hb_mtime = datetime.fromtimestamp(os.path.getmtime(_LOCK_HEARTBEAT_PATH), tz=_EST)
            age = datetime.now(tz=_EST) - hb_mtime
            if age < _LOCK_HEARTBEAT_STALE:
                return True  # heartbeat is fresh → holder is alive
            else:
                logger.warning(
                    "Lock heartbeat stale (last beat %.0fs ago) — holder likely dead",
                    age.total_seconds(),
                )
                return False
        except Exception:
            pass  # fall through to other checks

    # --- Check PID (skip if PID 1 — Docker entrypoint, not the trainer) ---
    lock_pid = lock_data.get("pid")
    if lock_pid and lock_pid != 1:
        try:
            os.kill(lock_pid, 0)  # signal 0 = existence check
            return True
        except ProcessLookupError:
            logger.warning("Lock holder PID %d no longer exists", lock_pid)
            return False
        except PermissionError:
            return True  # process exists but we can't signal it

    # --- Fall back to lock age ---
    try:
        lock_time = datetime.fromisoformat(lock_data.get("acquired_at", ""))
        if datetime.now(tz=_EST) - lock_time > _LOCK_STALE_AGE:
            return False
    except Exception:
        return False  # can't parse → treat as dead

    # Lock is recent and we can't prove it's dead — assume alive
    return True


def _heartbeat_loop(stop_event: threading.Event) -> None:
    """Background thread that periodically touches the heartbeat file."""
    while not stop_event.is_set():
        try:
            with open(_LOCK_HEARTBEAT_PATH, "w") as f:
                f.write(_now_et().isoformat())
        except Exception:
            pass
        stop_event.wait(_LOCK_HEARTBEAT_INTERVAL)


def _start_heartbeat() -> None:
    """Start the lock heartbeat background thread."""
    global _lock_heartbeat_stop, _lock_heartbeat_thread
    _lock_heartbeat_stop = threading.Event()
    _lock_heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(_lock_heartbeat_stop,),
        daemon=True,
        name="retrain-lock-heartbeat",
    )
    _lock_heartbeat_thread.start()


def _stop_heartbeat() -> None:
    """Stop the lock heartbeat background thread and clean up heartbeat file."""
    global _lock_heartbeat_stop, _lock_heartbeat_thread
    if _lock_heartbeat_stop is not None:
        _lock_heartbeat_stop.set()
    if _lock_heartbeat_thread is not None:
        _lock_heartbeat_thread.join(timeout=5)
        _lock_heartbeat_thread = None
    _lock_heartbeat_stop = None
    try:
        if os.path.isfile(_LOCK_HEARTBEAT_PATH):
            os.remove(_LOCK_HEARTBEAT_PATH)
    except Exception:
        pass


def _acquire_lock() -> bool:
    """File-based lock with heartbeat to prevent concurrent retraining runs.

    Improvements over simple age-based staleness:
      - A background heartbeat thread proves the holder is alive.
      - If the heartbeat goes stale (>5 min), the lock is broken.
      - PID check is skipped for PID 1 (Docker entrypoint ≠ trainer).
      - Absolute stale timeout reduced from 4h to 2h.
    """
    if os.path.isfile(LOCKFILE_PATH):
        try:
            with open(LOCKFILE_PATH) as f:
                lock_data = json.load(f)

            if _is_lock_holder_alive(lock_data):
                logger.error(
                    "Retrain lock held since %s by PID %s — aborting",
                    lock_data.get("acquired_at"),
                    lock_data.get("pid"),
                )
                return False
            else:
                logger.warning(
                    "Breaking stale/dead lock (acquired %s, PID %s)",
                    lock_data.get("acquired_at"),
                    lock_data.get("pid"),
                )
                os.remove(LOCKFILE_PATH)
                try:
                    os.remove(_LOCK_HEARTBEAT_PATH)
                except FileNotFoundError:
                    pass
        except Exception:
            # Lock file is corrupt — remove it
            logger.warning("Corrupt lock file — removing")
            try:
                os.remove(LOCKFILE_PATH)
            except FileNotFoundError:
                pass

    os.makedirs(os.path.dirname(LOCKFILE_PATH), exist_ok=True)
    with open(LOCKFILE_PATH, "w") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_at": _now_et().isoformat(),
            },
            f,
        )

    # Start the heartbeat so other processes can tell we're alive
    _start_heartbeat()
    return True


def _release_lock() -> None:
    """Stop heartbeat and remove the lock file."""
    _stop_heartbeat()
    try:
        if os.path.isfile(LOCKFILE_PATH):
            os.remove(LOCKFILE_PATH)
    except Exception:
        pass


def _append_audit(result: RetrainResult) -> None:
    """Append a run result to the JSONL audit log."""
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    try:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
    except Exception as exc:
        logger.error("Failed to write audit log: %s", exc)


def _wait_for_window(window_name: str, target_hour: int, cfg: RetrainConfig) -> bool:
    """Wait until the target ET hour (or return immediately if --immediate).

    Returns False if we enter the active session before reaching the
    target hour (safety bail-out).
    """
    if cfg.immediate:
        return True

    now = _now_et()
    if now.hour >= target_hour:
        return True

    wait_seconds = (target_hour - now.hour - 1) * 3600 + (60 - now.minute) * 60
    wait_seconds = max(0, wait_seconds)

    logger.info(
        "⏳ Waiting for %s window (target: %02d:00 ET, ~%.0f min)...",
        window_name,
        target_hour,
        wait_seconds / 60,
    )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _is_active_session():
            logger.warning(
                "Active session started before %s window — aborting wait",
                window_name,
            )
            return False
        time.sleep(min(60, deadline - time.monotonic()))

    return True


# ---------------------------------------------------------------------------
# Stage 1: Dataset Refresh
# ---------------------------------------------------------------------------


def stage_dataset_refresh(cfg: RetrainConfig, result: RetrainResult) -> bool:
    """Generate new chart images from recent historical bars.

    Returns True on success, False on failure.
    """
    if cfg.skip_dataset:
        logger.info("📂 Stage 1: Dataset refresh — SKIPPED (--skip-dataset)")
        # Still count existing rows
        if os.path.isfile(LABELS_CSV):
            with contextlib.suppress(Exception):
                result.dataset_rows = len(pd.read_csv(LABELS_CSV))
        return True

    logger.info("=" * 60)
    logger.info("📂 Stage 1: Dataset Refresh")
    logger.info("=" * 60)
    logger.info("Symbols:    %s", ", ".join(cfg.symbols))
    logger.info("Days back:  %d", cfg.days_back)
    logger.info("Source:     %s", cfg.bars_source)

    t0 = time.monotonic()

    try:
        from lib.analysis.dataset_generator import DatasetConfig, generate_dataset

        ds_config = DatasetConfig(
            bars_source=cfg.bars_source,
            skip_existing=cfg.skip_existing_images,
            chart_dpi=cfg.chart_dpi,
            orb_session=cfg.orb_session,
        )

        stats = generate_dataset(
            symbols=cfg.symbols,
            days_back=cfg.days_back,
            config=ds_config,
        )

        result.dataset_new_images = stats.total_images - stats.skipped_existing
        result.dataset_duration_seconds = time.monotonic() - t0

        # Count total rows in the CSV
        if os.path.isfile(LABELS_CSV):
            result.dataset_rows = len(pd.read_csv(LABELS_CSV))

        logger.info(
            "✅ Dataset refresh complete: %d total rows, %d new images (%.1f min)",
            result.dataset_rows,
            result.dataset_new_images,
            result.dataset_duration_seconds / 60,
        )

        if stats.errors:
            for err in stats.errors[:5]:
                logger.warning("  ⚠ %s", err)
                result.errors.append(f"dataset: {err}")

        return True

    except ImportError as exc:
        msg = f"Dataset generator not available: {exc}"
        logger.error(msg)
        result.errors.append(msg)
        return False
    except Exception as exc:
        msg = f"Dataset refresh failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Stage 2: Train/Val Split
# ---------------------------------------------------------------------------


def stage_split_dataset(cfg: RetrainConfig, result: RetrainResult) -> bool:
    """Re-split labels.csv into train.csv and val.csv.

    Returns True on success, False on failure.
    """
    logger.info("=" * 60)
    logger.info("✂️  Stage 2: Train/Val Split")
    logger.info("=" * 60)

    if not os.path.isfile(LABELS_CSV):
        msg = f"Labels CSV not found: {LABELS_CSV}"
        logger.error(msg)
        result.errors.append(msg)
        return False

    try:
        from lib.analysis.dataset_generator import split_dataset

        train_path, val_path = split_dataset(
            csv_path=LABELS_CSV,
            val_fraction=cfg.val_fraction,
            output_dir=DATASET_DIR,
            stratify=cfg.stratify,
            random_seed=cfg.random_seed,
        )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        result.train_samples = len(train_df)
        result.val_samples = len(val_df)

        # Log label distribution
        if "label" in train_df.columns:
            dist = train_df["label"].value_counts().to_dict()
            logger.info("Train label distribution: %s", dist)
        if "label" in val_df.columns:
            dist = val_df["label"].value_counts().to_dict()
            logger.info("Val label distribution:   %s", dist)

        logger.info(
            "✅ Split complete: %d train / %d val (%.1f%%)",
            result.train_samples,
            result.val_samples,
            cfg.val_fraction * 100,
        )
        return True

    except ImportError as exc:
        msg = f"Dataset split module not available: {exc}"
        logger.error(msg)
        result.errors.append(msg)
        return False
    except Exception as exc:
        msg = f"Dataset split failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Stage 3: GPU Training
# ---------------------------------------------------------------------------


def stage_train(cfg: RetrainConfig, result: RetrainResult) -> bool:
    """Run GPU-accelerated training and return the best model path.

    Returns True on success (result.candidate_model_path is populated).
    """
    logger.info("=" * 60)
    logger.info("🚀 Stage 3: GPU Training")
    logger.info("=" * 60)

    if not os.path.isfile(TRAIN_CSV):
        msg = f"Training CSV not found: {TRAIN_CSV}"
        logger.error(msg)
        result.errors.append(msg)
        return False

    if not os.path.isfile(VAL_CSV):
        msg = f"Validation CSV not found: {VAL_CSV}"
        logger.error(msg)
        result.errors.append(msg)
        return False

    t0 = time.monotonic()

    # Prefer the GPU-optimised trainer in scripts/train_gpu.py if available,
    # otherwise fall back to the in-module train_model().
    try:
        # Import the GPU trainer (it lives in this same scripts/ directory)
        # We dynamically import to avoid hard coupling.
        import importlib.util

        train_gpu_path = SCRIPT_DIR / "train_gpu.py"
        if train_gpu_path.is_file():
            spec = importlib.util.spec_from_file_location("train_gpu", str(train_gpu_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec from {train_gpu_path}")
            train_gpu_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_gpu_mod)

            logger.info("Using GPU-optimised trainer: %s", train_gpu_path)

            best_path = train_gpu_mod.train(
                train_csv=TRAIN_CSV,
                val_csv=VAL_CSV,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                freeze_epochs=cfg.freeze_epochs,
                warmup_epochs=cfg.warmup_epochs,
                model_dir=MODEL_DIR,
                num_workers=cfg.num_workers,
                patience=cfg.patience,
                label_smoothing=cfg.label_smoothing,
                use_amp=cfg.use_amp,
                dropout=cfg.dropout,
            )
        else:
            logger.info("GPU trainer script not found — falling back to breakout_cnn.train_model()")
            from lib.analysis.breakout_cnn import train_model

            best_path = train_model(
                data_csv=TRAIN_CSV,
                val_csv=VAL_CSV,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                freeze_epochs=cfg.freeze_epochs,
                model_dir=MODEL_DIR,
                num_workers=cfg.num_workers,
            )

        result.training_duration_seconds = time.monotonic() - t0

        if best_path and os.path.isfile(best_path):
            result.candidate_model_path = best_path
            logger.info(
                "✅ Training complete in %.1f min — candidate: %s",
                result.training_duration_seconds / 60,
                best_path,
            )

            # Parse training history for metrics
            history_path = os.path.join(MODEL_DIR, "training_history.csv")
            if os.path.isfile(history_path):
                try:
                    hist = pd.read_csv(history_path)
                    if not hist.empty:
                        result.training_epochs_run = len(hist)
                        best_row = hist.loc[hist["val_acc"].idxmax()]
                        result.best_val_accuracy = float(best_row["val_acc"])
                        result.best_val_loss = float(best_row["val_loss"])
                        result.best_precision = float(best_row.get("precision", 0))
                        result.best_recall = float(best_row.get("recall", 0))
                except Exception as exc:
                    logger.warning("Could not parse training history: %s", exc)

            return True
        else:
            msg = "Training returned no valid model path"
            logger.error(msg)
            result.errors.append(msg)
            return False

    except Exception as exc:
        result.training_duration_seconds = time.monotonic() - t0
        msg = f"Training failed after {result.training_duration_seconds:.0f}s: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Stage 4: Validation Gate
# ---------------------------------------------------------------------------


def _evaluate_model_on_val(model_path: str, val_csv: str) -> dict[str, float]:
    """Load a model and evaluate it on the validation set.

    Returns dict with keys: accuracy, precision, recall, avg_loss.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from lib.analysis.breakout_cnn import (
        BreakoutDataset,
        HybridBreakoutCNN,
        get_device,
        get_inference_transform,
    )

    device = torch.device(get_device())

    model = HybridBreakoutCNN(pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    val_dataset = BreakoutDataset(val_csv, transform=get_inference_transform())
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = 0  # true positives (good predicted as good)
    fp = 0  # false positives (bad predicted as good)
    fn = 0  # false negatives (good predicted as bad)

    with torch.no_grad():
        for imgs, tabs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs, tabs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100.0 * correct / max(total, 1)
    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / max(tp + fn, 1)
    avg_loss = total_loss / max(total, 1)

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "avg_loss": round(avg_loss, 4),
        "total_samples": total,
    }


def stage_validation_gate(cfg: RetrainConfig, result: RetrainResult) -> bool:
    """Compare candidate model against the champion on the val set.

    The candidate must:
      1. Meet absolute accuracy threshold (min_val_accuracy)
      2. Meet precision/recall thresholds
      3. Beat the champion accuracy by min_improvement (if champion exists)

    Returns True if the gate passes (candidate should be promoted).
    """
    logger.info("=" * 60)
    logger.info("🔍 Stage 4: Validation Gate")
    logger.info("=" * 60)

    if not result.candidate_model_path or not os.path.isfile(result.candidate_model_path):
        msg = "No candidate model to validate"
        logger.error(msg)
        result.errors.append(msg)
        result.gate_reason = msg
        return False

    # Evaluate candidate
    logger.info("Evaluating candidate: %s", result.candidate_model_path)
    try:
        candidate_metrics = _evaluate_model_on_val(result.candidate_model_path, VAL_CSV)
    except Exception as exc:
        msg = f"Candidate evaluation failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        result.gate_reason = msg
        return False

    cand_acc = candidate_metrics["accuracy"]
    cand_prec = candidate_metrics["precision"]
    cand_rec = candidate_metrics["recall"]

    logger.info(
        "Candidate metrics: acc=%.1f%% prec=%.1f%% rec=%.1f%% loss=%.4f (n=%d)",
        cand_acc,
        cand_prec,
        cand_rec,
        candidate_metrics["avg_loss"],
        candidate_metrics["total_samples"],
    )

    # Update result with fresh evaluation numbers
    result.best_val_accuracy = cand_acc
    result.best_precision = cand_prec
    result.best_recall = cand_rec

    # Check absolute thresholds
    if cand_acc < cfg.min_val_accuracy:
        reason = f"REJECT: accuracy {cand_acc:.1f}% < minimum {cfg.min_val_accuracy:.1f}%"
        logger.warning("🚫 %s", reason)
        result.gate_reason = reason
        result.gate_passed = False
        return False

    if cand_prec < cfg.min_precision:
        reason = f"REJECT: precision {cand_prec:.1f}% < minimum {cfg.min_precision:.1f}%"
        logger.warning("🚫 %s", reason)
        result.gate_reason = reason
        result.gate_passed = False
        return False

    if cand_rec < cfg.min_recall:
        reason = f"REJECT: recall {cand_rec:.1f}% < minimum {cfg.min_recall:.1f}%"
        logger.warning("🚫 %s", reason)
        result.gate_reason = reason
        result.gate_passed = False
        return False

    # Evaluate champion (if it exists)
    champion_acc = 0.0
    if os.path.isfile(CHAMPION_PATH):
        logger.info("Evaluating champion: %s", CHAMPION_PATH)
        try:
            champion_metrics = _evaluate_model_on_val(CHAMPION_PATH, VAL_CSV)
            champion_acc = champion_metrics["accuracy"]
            result.champion_accuracy = champion_acc
            logger.info(
                "Champion metrics:  acc=%.1f%% prec=%.1f%% rec=%.1f%% loss=%.4f",
                champion_acc,
                champion_metrics["precision"],
                champion_metrics["recall"],
                champion_metrics["avg_loss"],
            )
        except Exception as exc:
            logger.warning("Could not evaluate champion (will proceed with promotion): %s", exc)
            champion_acc = 0.0
    else:
        logger.info("No existing champion — candidate will be promoted if thresholds pass")

    # Check improvement over champion
    improvement = cand_acc - champion_acc
    if champion_acc > 0 and improvement < cfg.min_improvement:
        reason = (
            f"REJECT: improvement {improvement:+.1f}% "
            f"(candidate {cand_acc:.1f}% vs champion {champion_acc:.1f}%) "
            f"< required {cfg.min_improvement:.1f}%"
        )
        logger.warning("🚫 %s", reason)
        result.gate_reason = reason
        result.gate_passed = False
        return False

    # All checks passed
    reason = (
        f"PASS: acc={cand_acc:.1f}% prec={cand_prec:.1f}% rec={cand_rec:.1f}% "
        f"(improvement: {improvement:+.1f}% over champion {champion_acc:.1f}%)"
    )
    logger.info("✅ %s", reason)
    result.gate_reason = reason
    result.gate_passed = True
    return True


# ---------------------------------------------------------------------------
# Stage 5: Model Promotion
# ---------------------------------------------------------------------------


def stage_promote(cfg: RetrainConfig, result: RetrainResult) -> bool:
    """Atomically promote the candidate model to champion.

    1. Archive the current champion (if exists)
    2. Copy candidate → breakout_cnn_best.pt
    3. Write promotion metadata

    Returns True on success.
    """
    logger.info("=" * 60)
    logger.info("🏆 Stage 5: Model Promotion")
    logger.info("=" * 60)

    if cfg.dry_run:
        logger.info("DRY RUN — would promote %s but skipping", result.candidate_model_path)
        result.promoted = False
        return True

    if not result.candidate_model_path or not os.path.isfile(result.candidate_model_path):
        msg = "No candidate model to promote"
        logger.error(msg)
        result.errors.append(msg)
        return False

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Archive current champion
        if os.path.isfile(CHAMPION_PATH):
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            archive_name = "breakout_cnn_best_{}.pt".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            archive_path = os.path.join(ARCHIVE_DIR, archive_name)
            shutil.copy2(CHAMPION_PATH, archive_path)
            result.archived_champion_path = archive_path
            logger.info("Archived champion → %s", archive_path)

        # Promote: copy candidate to champion path
        shutil.copy2(result.candidate_model_path, CHAMPION_PATH)
        result.promoted = True
        result.promoted_model_path = CHAMPION_PATH

        # Write promotion metadata alongside the model
        meta_path = os.path.join(MODEL_DIR, "breakout_cnn_best_meta.json")
        meta = {
            "promoted_at": _now_et().isoformat(),
            "run_id": result.run_id,
            "source_model": result.candidate_model_path,
            "val_accuracy": result.best_val_accuracy,
            "precision": result.best_precision,
            "recall": result.best_recall,
            "train_samples": result.train_samples,
            "val_samples": result.val_samples,
            "epochs_trained": result.training_epochs_run,
            "champion_accuracy_before": result.champion_accuracy,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Invalidate any cached model in the inference module
        try:
            from lib.analysis import breakout_cnn

            if hasattr(breakout_cnn, "_cached_model"):
                import threading

                lock = getattr(breakout_cnn, "_model_lock", threading.Lock())
                with lock:
                    breakout_cnn._cached_model = None
                    breakout_cnn._cached_model_path = None
                logger.info("Invalidated cached inference model")
        except Exception:
            pass  # non-critical

        logger.info(
            "✅ Model promoted: %s → %s (acc=%.1f%%)",
            os.path.basename(result.candidate_model_path),
            CHAMPION_PATH,
            result.best_val_accuracy,
        )
        return True

    except Exception as exc:
        msg = f"Model promotion failed: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Stage 6: Cleanup
# ---------------------------------------------------------------------------


def stage_cleanup(cfg: RetrainConfig, result: RetrainResult) -> None:
    """Clean up old checkpoints and archived models."""
    logger.info("=" * 60)
    logger.info("🧹 Stage 6: Cleanup")
    logger.info("=" * 60)

    # Prune archived models (keep only max_archived_models most recent)
    if os.path.isdir(ARCHIVE_DIR):
        archived = sorted(
            Path(ARCHIVE_DIR).glob("breakout_cnn_best_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(archived) > cfg.max_archived_models:
            for old_model in archived[cfg.max_archived_models :]:
                try:
                    old_model.unlink()
                    logger.info("Pruned archive: %s", old_model.name)
                except Exception as exc:
                    logger.warning("Could not prune %s: %s", old_model.name, exc)

    # Prune old training checkpoints from models/ (keep recent runs)
    # We keep: breakout_cnn_best.pt, breakout_cnn_best_meta.json,
    # training_history.csv, and the 3 most recent dated checkpoints
    model_dir_path = Path(MODEL_DIR)
    protected_names = {
        "breakout_cnn_best.pt",
        "breakout_cnn_best_meta.json",
        "training_history.csv",
        "retrain_audit.jsonl",
    }

    checkpoint_files = sorted(
        [f for f in model_dir_path.glob("breakout_cnn_*.pt") if f.name not in protected_names],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Keep the most recent N checkpoints (from today's run + last run)
    max_keep = cfg.max_checkpoints_per_run * 2  # current run + previous
    if len(checkpoint_files) > max_keep:
        for old_ckpt in checkpoint_files[max_keep:]:
            try:
                old_ckpt.unlink()
                logger.info("Pruned checkpoint: %s", old_ckpt.name)
            except Exception as exc:
                logger.warning("Could not prune %s: %s", old_ckpt.name, exc)

    logger.info("✅ Cleanup complete")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(cfg: RetrainConfig) -> RetrainResult:
    """Execute the full overnight retraining pipeline.

    This is the main entry point.  It can be called by:
      - The CLI (``__main__``)
      - The engine scheduler (``_handle_train_breakout_cnn``)
      - A cron job or systemd timer
    """
    result = RetrainResult(
        run_id=f"retrain_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}",
        started_at=_now_et().isoformat(),
    )

    t0 = time.monotonic()

    logger.info("=" * 70)
    logger.info("  🌙 OVERNIGHT CNN RETRAINING PIPELINE")
    logger.info("  Run ID:    %s", result.run_id)
    logger.info("  Started:   %s ET", result.started_at)
    logger.info("  Mode:      %s", "IMMEDIATE" if cfg.immediate else "SESSION-AWARE")
    logger.info("  Dry run:   %s", cfg.dry_run)
    logger.info("=" * 70)

    # Pre-flight checks
    if not cfg.force and _already_ran_today():
        logger.info("✓ Retraining already completed today — skipping (use --force to override)")
        result.status = "skipped"
        result.finished_at = _now_et().isoformat()
        result.duration_seconds = time.monotonic() - t0
        _append_audit(result)
        return result

    if not cfg.immediate and _is_active_session():
        logger.warning(
            "🚫 Cannot start retraining during active session (03:00–12:00 ET). Use --immediate to override."
        )
        result.status = "skipped"
        result.finished_at = _now_et().isoformat()
        result.duration_seconds = time.monotonic() - t0
        _append_audit(result)
        return result

    if not _acquire_lock():
        result.status = "failed"
        result.errors.append("Could not acquire retrain lock")
        result.finished_at = _now_et().isoformat()
        result.duration_seconds = time.monotonic() - t0
        _append_audit(result)
        return result

    try:
        # ── Stage 1: Dataset Refresh ──────────────────────────────────
        if not stage_dataset_refresh(cfg, result):
            # Dataset refresh failed — if we have existing data, continue;
            # otherwise abort.
            if not os.path.isfile(LABELS_CSV):
                logger.error("No dataset available — aborting pipeline")
                result.status = "failed"
                return result
            logger.warning("Dataset refresh failed but existing data found — continuing")

        if result.dataset_rows < 100:
            logger.error(
                "Dataset too small (%d rows) — need at least 100 for meaningful training",
                result.dataset_rows,
            )
            result.status = "failed"
            result.errors.append(f"Dataset too small: {result.dataset_rows} rows")
            return result

        # ── Stage 2: Train/Val Split ──────────────────────────────────
        if not stage_split_dataset(cfg, result):
            result.status = "failed"
            return result

        # ── Wait for GPU training window (if session-aware) ──────────
        # In session-aware mode, we wait until 17:00 ET for training so
        # dataset generation can use the GPU-free afternoon hours.
        if not _wait_for_window("training", 17, cfg):
            logger.warning("Could not enter training window — aborting")
            result.status = "failed"
            result.errors.append("Training window not reached before active session")
            return result

        # Bail out if active session started while we were generating data
        if not cfg.immediate and _is_active_session():
            logger.warning("Active session started — aborting before training")
            result.status = "failed"
            result.errors.append("Active session started before training could begin")
            return result

        # ── Stage 3: GPU Training ─────────────────────────────────────
        if not stage_train(cfg, result):
            result.status = "failed"
            return result

        # ── Stage 4: Validation Gate ──────────────────────────────────
        gate_ok = stage_validation_gate(cfg, result)

        if not gate_ok:
            logger.warning("🚫 Validation gate REJECTED candidate — champion unchanged")
            result.status = "gate_rejected"
            return result

        # ── Stage 5: Model Promotion ──────────────────────────────────
        if not stage_promote(cfg, result):
            result.status = "failed"
            return result

        # ── Stage 6: Cleanup ──────────────────────────────────────────
        stage_cleanup(cfg, result)

        result.status = "success"
        return result

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        result.status = "failed"
        result.errors.append("KeyboardInterrupt")
        return result

    except Exception as exc:
        logger.error("Pipeline crashed: %s", exc, exc_info=True)
        result.status = "failed"
        result.errors.append(f"Unhandled exception: {exc}")
        return result

    finally:
        result.finished_at = _now_et().isoformat()
        result.duration_seconds = time.monotonic() - t0

        _release_lock()
        _append_audit(result)

        # ── Final Summary ─────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE RESULT: %s", result.status.upper())
        logger.info("=" * 70)
        logger.info("  Run ID:        %s", result.run_id)
        logger.info("  Duration:      %.1f min", result.duration_seconds / 60)
        logger.info("  Dataset rows:  %d", result.dataset_rows)
        logger.info("  Train/Val:     %d / %d", result.train_samples, result.val_samples)
        logger.info("  Epochs:        %d", result.training_epochs_run)
        logger.info("  Best accuracy: %.1f%%", result.best_val_accuracy)
        logger.info("  Precision:     %.1f%%", result.best_precision)
        logger.info("  Recall:        %.1f%%", result.best_recall)
        logger.info("  Gate:          %s", "PASS ✅" if result.gate_passed else "FAIL 🚫")
        logger.info("  Promoted:      %s", "YES 🏆" if result.promoted else "NO")
        if result.champion_accuracy > 0:
            logger.info("  Champion was:  %.1f%%", result.champion_accuracy)
        if result.errors:
            logger.info("  Errors:        %d", len(result.errors))
            for err in result.errors[:5]:
                logger.info("    • %s", err)
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Engine integration helper
# ---------------------------------------------------------------------------


def run_from_engine() -> bool:
    """Entry point for the engine scheduler.

    Called by ``_handle_train_breakout_cnn()`` in
    ``lib/services/engine/main.py``.  Runs in immediate mode with
    dataset generation skipped (the engine runs dataset gen as a
    separate scheduled action beforehand).

    Returns True if training + promotion succeeded.
    """
    cfg = RetrainConfig.from_env()
    cfg.immediate = True
    cfg.skip_dataset = True  # engine handles dataset gen separately

    result = run_pipeline(cfg)
    return result.status == "success"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Overnight CNN Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Session-aware time windows (all times Eastern):
  12:00–17:00 ET   Dataset generation (CPU-bound)
  17:00–00:00 ET   GPU training window
  00:00–03:00 ET   Validation + promotion + cleanup
  03:00–12:00 ET   ACTIVE SESSION — NO TRAINING

Use --immediate to run all stages back-to-back regardless of time.

Examples:
  # Full pipeline (respects time windows):
  PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py

  # Immediate full run:
  PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate

  # Retrain on existing data (skip dataset generation):
  PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --skip-dataset --immediate

  # Validate only, no promotion:
  PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --dry-run --immediate
        """,
    )

    # Behavior flags
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Run all stages immediately (ignore time windows)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation (train on existing data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training + validation but don't promote the model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if retraining already completed today",
    )

    # Dataset options
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (default: MGC,MES,MNQ)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Days of historical data for dataset generation (default: 90)",
    )
    parser.add_argument(
        "--bars-source",
        choices=["cache", "massive", "csv"],
        default=None,
        help="Data source for bars (default: cache)",
    )
    parser.add_argument(
        "--session",
        choices=["us", "london", "both"],
        default=None,
        help="ORB session for dataset generation: 'us', 'london', or 'both' (default: both)",
    )

    # Training options
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-4)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 8)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")

    # Validation gate
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Minimum val accuracy to promote (default: 80.0)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Minimum precision to promote (default: 75.0)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Minimum recall to promote (default: 70.0)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=None,
        help="Required accuracy improvement over champion (default: 0.0)",
    )

    args = parser.parse_args()

    # Build config from env, then overlay CLI args
    cfg = RetrainConfig.from_env()

    # Behavior flags
    cfg.immediate = args.immediate
    cfg.skip_dataset = args.skip_dataset
    cfg.dry_run = args.dry_run
    cfg.force = args.force

    # Dataset options
    if args.symbols:
        cfg.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.days_back is not None:
        cfg.days_back = args.days_back
    if args.bars_source is not None:
        cfg.bars_source = args.bars_source
    if args.session is not None:
        cfg.orb_session = args.session

    # Training options
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.patience is not None:
        cfg.patience = args.patience
    if args.no_amp:
        cfg.use_amp = False

    # Validation gate
    if args.min_accuracy is not None:
        cfg.min_val_accuracy = args.min_accuracy
    if args.min_precision is not None:
        cfg.min_precision = args.min_precision
    if args.min_recall is not None:
        cfg.min_recall = args.min_recall
    if args.min_improvement is not None:
        cfg.min_improvement = args.min_improvement

    # Run the pipeline
    result = run_pipeline(cfg)

    # Exit code
    if result.status == "success":
        print(f"\n✅ Retraining succeeded — model promoted (acc={result.best_val_accuracy:.1f}%)")
        sys.exit(0)
    elif result.status == "skipped":
        print("\n⏭️  Retraining skipped (already ran today or active session)")
        sys.exit(0)
    elif result.status == "gate_rejected":
        print(f"\n🚫 Candidate rejected by validation gate: {result.gate_reason}")
        sys.exit(1)
    else:
        print(f"\n❌ Retraining failed: {', '.join(result.errors[:3]) or 'unknown error'}")
        sys.exit(2)


if __name__ == "__main__":
    main()
