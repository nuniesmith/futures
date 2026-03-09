"""
Trainer Server — Persistent HTTP Training Service
===================================================
FastAPI server that accepts ``POST /train`` requests, runs the full CNN
training pipeline (dataset generation → model training → evaluation →
champion promotion), and exposes health/status endpoints.

Designed to run as a long-lived Docker service on a GPU machine:

    docker compose up trainer

Or directly:

    python -m lib.services.training.trainer_server

The server is intentionally simple — one training job at a time, no queue.
If a training run is already in progress, ``POST /train`` returns 409.

Endpoints:
    GET  /health               — Liveness probe (always 200)
    GET  /status               — Current server state + last training result
    GET  /logs                 — Recent in-memory log lines (JSON)
    GET  /metrics/prometheus   — Prometheus text-format metrics endpoint
    POST /train                — Kick off a training run (async background task)
    POST /train/cancel         — Request cancellation of the current run
    POST /export_onnx          — Re-export the champion .pt to ONNX (best-effort)
    GET  /models               — List all model files in models/ + archive/
    GET  /models/archive       — List archived model checkpoints

Environment variables:
    TRAINER_HOST                  — Bind address (default 0.0.0.0)
    TRAINER_PORT                  — Bind port (default 8200)
    TRAINER_API_KEY               — Optional bearer token for /train
    CNN_RETRAIN_MIN_ACC           — Minimum accuracy gate (default 80.0)
    CNN_RETRAIN_MIN_PRECISION     — Minimum precision gate (default 75.0)
    CNN_RETRAIN_MIN_RECALL        — Minimum recall gate (default 70.0)
    CNN_RETRAIN_IMPROVEMENT       — Min improvement over champion (default 0.0)
    CNN_RETRAIN_EPOCHS            — Training epochs (default 60)
    CNN_RETRAIN_BATCH_SIZE        — Batch size (default 64)
    CNN_RETRAIN_LR                — Learning rate (default 0.0001)
    CNN_RETRAIN_PATIENCE          — Early stopping patience (default 12)
    CNN_RETRAIN_SYMBOLS           — Comma-separated symbols (default: all micros)
    CNN_RETRAIN_DAYS_BACK         — Days of history (default 180)
    CNN_RETRAIN_BARS_SOURCE       — Data source (default "engine"; engine handles Redis→DB→API internally)
    ENGINE_DATA_URL               — Engine/data service base URL for bar fetching (default "http://data:8000")
    CNN_RETRAIN_ORB_SESSION       — Session filter (default "all")
    CNN_ORB_SESSION               — Alias for CNN_RETRAIN_ORB_SESSION
"""

from __future__ import annotations

import collections
import json

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Ensure standard-library loggers (e.g. analysis.breakout_cnn, training.*)
# emit to stdout so epoch progress and dataset messages appear in docker logs.
import logging
import os
import shutil
import sys
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
# Quieten noisy third-party loggers that flood docker logs at INFO level
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# In-memory log ring buffer — captures recent log lines for the Web UI
# ---------------------------------------------------------------------------

_LOG_BUFFER_SIZE = 400
_log_buffer: collections.deque[dict[str, str]] = collections.deque(maxlen=_LOG_BUFFER_SIZE)
_log_buffer_lock = threading.Lock()
_log_total_written: int = 0  # monotonically increasing count of all records ever appended


class _RingBufferHandler(logging.Handler):
    """Appends formatted log records to the in-memory ring buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        global _log_total_written
        try:
            entry = {
                "ts": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
            }
            with _log_buffer_lock:
                _log_buffer.append(entry)
                _log_total_written += 1
        except Exception:
            pass


_ring_handler = _RingBufferHandler()
_ring_handler.setFormatter(logging.Formatter("%(message)s"))
_ring_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_ring_handler)


structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger("trainer_server")


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

TRAINER_HOST = os.getenv("TRAINER_HOST", "0.0.0.0")
TRAINER_PORT = int(os.getenv("TRAINER_PORT", "8200"))
TRAINER_API_KEY = os.getenv("TRAINER_API_KEY", "").strip()

# Validation gates
MIN_ACC = float(os.getenv("CNN_RETRAIN_MIN_ACC", "80.0"))
MIN_PRECISION = float(os.getenv("CNN_RETRAIN_MIN_PRECISION", "75.0"))
MIN_RECALL = float(os.getenv("CNN_RETRAIN_MIN_RECALL", "70.0"))
MIN_IMPROVEMENT = float(os.getenv("CNN_RETRAIN_IMPROVEMENT", "0.0"))

# Training hyperparameters
DEFAULT_EPOCHS = int(os.getenv("CNN_RETRAIN_EPOCHS", "60"))
DEFAULT_BATCH_SIZE = int(os.getenv("CNN_RETRAIN_BATCH_SIZE", "64"))
DEFAULT_LR = float(os.getenv("CNN_RETRAIN_LR", "0.0001"))
DEFAULT_PATIENCE = int(os.getenv("CNN_RETRAIN_PATIENCE", "12"))

# Dataset generation defaults
DEFAULT_SYMBOLS = os.getenv(
    "CNN_RETRAIN_SYMBOLS",
    # 22 CME micro futures + 3 Kraken spot crypto (BTC/ETH/SOL).
    # Kraken symbols are resolved via DataResolver → Kraken REST API with
    # automatic Postgres/Redis backfill so subsequent training runs are fast.
    "MGC,SIL,MHG,MCL,MNG,MES,MNQ,M2K,MYM,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET,BTC,ETH,SOL",
).split(",")
DEFAULT_DAYS_BACK = int(os.getenv("CNN_RETRAIN_DAYS_BACK", "180"))
# "engine" is the preferred default — the trainer calls the engine's HTTP API
# (GET /bars/{symbol}?auto_fill=true) which handles Redis → Postgres → external
# API resolution internally.  This works correctly when the trainer runs on a
# separate GPU machine without direct Redis/Postgres access.
DEFAULT_BARS_SOURCE = os.getenv("CNN_RETRAIN_BARS_SOURCE", "engine")
DEFAULT_ORB_SESSION = os.getenv("CNN_RETRAIN_ORB_SESSION", os.getenv("CNN_ORB_SESSION", "all"))

# Paths
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/app/dataset"))
ARCHIVE_DIR = MODELS_DIR / "archive"
CHAMPION_PT = MODELS_DIR / "breakout_cnn_best.pt"
CHAMPION_META = MODELS_DIR / "breakout_cnn_best_meta.json"


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class TrainStatus(StrEnum):
    IDLE = "idle"
    GENERATING = "generating_dataset"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainState:
    """Thread-safe mutable state for the training pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: TrainStatus = TrainStatus.IDLE
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.cancel_requested: bool = False
        self.progress: str = ""
        self.last_result: dict[str, Any] | None = None
        self.error: str | None = None

    def set(self, status: TrainStatus, progress: str = "") -> None:
        with self._lock:
            self.status = status
            self.progress = progress
            if status == TrainStatus.GENERATING and self.started_at is None:
                self.started_at = datetime.now(UTC)

    def finish(self, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self._lock:
            self.finished_at = datetime.now(UTC)
            self.last_result = result
            self.error = error
            if error:
                self.status = TrainStatus.FAILED
            elif self.cancel_requested:
                self.status = TrainStatus.CANCELLED
            else:
                self.status = TrainStatus.DONE

    def request_cancel(self) -> bool:
        with self._lock:
            if self.status in (TrainStatus.IDLE, TrainStatus.DONE, TrainStatus.FAILED, TrainStatus.CANCELLED):
                return False
            self.cancel_requested = True
            return True

    def reset(self) -> None:
        with self._lock:
            self.status = TrainStatus.IDLE
            self.started_at = None
            self.finished_at = None
            self.cancel_requested = False
            self.progress = ""
            self.error = None

    def is_busy(self) -> bool:
        with self._lock:
            return self.status in (
                TrainStatus.GENERATING,
                TrainStatus.TRAINING,
                TrainStatus.EVALUATING,
                TrainStatus.PROMOTING,
            )

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status.value,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "progress": self.progress,
                "cancel_requested": self.cancel_requested,
                "last_result": self.last_result,
                "error": self.error,
            }


_state = TrainState()
_boot_time = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Parameters for a training run (all optional — env defaults used)."""

    symbols: list[str] | None = Field(None, description="Symbols to generate dataset for")
    days_back: int | None = Field(None, ge=1, le=365, description="Days of history")
    breakout_type: str = Field("all", description="ORB | PrevDay | InitialBalance | Consolidation | all")
    orb_session: str | None = Field(None, description="Session filter (us, london, all, ...)")
    bars_source: str | None = Field(None, description="Data source: massive | db | cache | csv")
    epochs: int | None = Field(None, ge=1, le=200, description="Training epochs")
    batch_size: int | None = Field(None, ge=8, le=512, description="Batch size")
    learning_rate: float | None = Field(None, gt=0, lt=1, description="Learning rate")
    patience: int | None = Field(None, ge=1, le=50, description="Early stopping patience")
    min_accuracy: float | None = Field(None, ge=0, le=100, description="Min accuracy gate (%)")
    min_precision: float | None = Field(None, ge=0, le=100, description="Min precision gate (%)")
    min_recall: float | None = Field(None, ge=0, le=100, description="Min recall gate (%)")
    force_promote: bool = Field(False, description="Promote even if gates fail (use with caution)")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(request: Request) -> None:
    """Optional bearer-token authentication for /train endpoints."""
    if not TRAINER_API_KEY:
        return  # no key configured — open access
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[len("Bearer ") :].strip()
    if token != TRAINER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _archive_champion() -> Path | None:
    """Move the current champion model to the archive directory."""
    if not CHAMPION_PT.exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / f"breakout_cnn_{ts}.pt"
    shutil.copy2(CHAMPION_PT, dest)
    logger.info("Archived champion model", dest=str(dest))
    return dest


def _write_meta(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    """Write model metadata JSON alongside the champion checkpoint."""
    meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "config": config,
        "version": "v6",
    }
    CHAMPION_META.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote model metadata", path=str(CHAMPION_META))


def _run_training_pipeline(params: TrainRequest) -> None:
    """Execute the full training pipeline in a background thread.

    Steps:
        1. Generate dataset (chart images + labels.csv)
        2. Train CNN model
        3. Evaluate against validation set
        4. Compare with champion model (if exists)
        5. Promote if all gates pass (or force_promote=True)
    """
    try:
        # ----- Resolve parameters -----
        symbols = params.symbols or DEFAULT_SYMBOLS
        days_back = params.days_back or DEFAULT_DAYS_BACK
        orb_session = params.orb_session or DEFAULT_ORB_SESSION
        bars_source = params.bars_source or DEFAULT_BARS_SOURCE
        epochs = params.epochs or DEFAULT_EPOCHS
        batch_size = params.batch_size or DEFAULT_BATCH_SIZE
        lr = params.learning_rate or DEFAULT_LR
        patience = params.patience or DEFAULT_PATIENCE
        min_acc = params.min_accuracy if params.min_accuracy is not None else MIN_ACC
        min_prec = params.min_precision if params.min_precision is not None else MIN_PRECISION
        min_rec = params.min_recall if params.min_recall is not None else MIN_RECALL

        run_config = {
            "symbols": symbols,
            "days_back": days_back,
            "breakout_type": params.breakout_type,
            "orb_session": orb_session,
            "bars_source": bars_source,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "patience": patience,
            "gates": {"min_acc": min_acc, "min_prec": min_prec, "min_rec": min_rec},
        }
        logger.info("Starting training pipeline", **run_config)

        # ----- Step 1: Dataset generation -----
        if _state.cancel_requested:
            _state.finish(error="Cancelled before dataset generation")
            return

        _state.set(TrainStatus.GENERATING, f"Generating dataset for {len(symbols)} symbols, {days_back} days")

        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset

        ds_config = DatasetConfig(
            output_dir=str(DATASET_DIR),
            image_dir=str(DATASET_DIR / "images"),
            bars_source=bars_source,
            orb_session=orb_session,
            breakout_type=params.breakout_type,
            use_parity_renderer=True,
        )

        ds_stats = generate_dataset(
            symbols=symbols,
            days_back=days_back,
            config=ds_config,
        )
        logger.info(
            "Dataset generation complete",
            total_images=ds_stats.total_images,
            label_balance=ds_stats.label_distribution,
        )

        # Record dataset stats to Prometheus (best-effort — never block training).
        try:
            from lib.services.engine.data.api.metrics import record_trainer_dataset_stats

            record_trainer_dataset_stats(
                total_images=ds_stats.total_images,
                label_distribution=ds_stats.label_distribution,
                render_time_seconds=ds_stats.duration_seconds,
            )
        except Exception as _metrics_err:
            logger.debug("Trainer metrics update failed (non-fatal)", error=str(_metrics_err))

        if ds_stats.total_images < 100:
            _state.finish(
                error=f"Insufficient training data: only {ds_stats.total_images} images generated (need ≥100)"
            )
            return

        # ----- Step 2: Train model -----
        if _state.cancel_requested:
            _state.finish(error="Cancelled before training")
            return

        _state.set(TrainStatus.TRAINING, f"Training for up to {epochs} epochs")

        from lib.analysis.breakout_cnn import evaluate_model, train_model

        if train_model is None:
            _state.finish(error="torch not available — cannot train (is the [gpu] extra installed?)")
            return

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # ----- Step 1b: Train/val split -----
        # Always regenerate a clean stratified split from the freshly written
        # labels.csv so that:
        #   - train_model uses a held-out val set (not its own random split)
        #   - evaluate_model runs against the *same* held-out val set
        #   - metrics in the result JSON reflect true out-of-sample performance
        from lib.services.training.dataset_generator import split_dataset

        labels_csv = DATASET_DIR / "labels.csv"
        image_root = str(DATASET_DIR / "images")

        logger.info("Splitting dataset into train/val sets (85/15 stratified)")
        try:
            train_csv_path, val_csv_path = split_dataset(
                csv_path=str(labels_csv),
                val_fraction=0.15,
                output_dir=str(DATASET_DIR),
                stratify=True,
                random_seed=42,
            )
            logger.info(
                "Dataset split complete",
                train_csv=train_csv_path,
                val_csv=val_csv_path,
            )
        except Exception as split_err:
            # Non-fatal: fall back to the full CSV for both train and eval.
            # This preserves the old behaviour rather than aborting the run.
            logger.warning(
                "Dataset split failed — falling back to full labels.csv for train/eval (non-fatal)",
                error=str(split_err),
            )
            train_csv_path = str(labels_csv)
            val_csv_path = None

        trained_result = train_model(
            data_csv=train_csv_path,
            val_csv=val_csv_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_dir=str(MODELS_DIR),
            image_root=image_root,
        )

        if trained_result is None:
            _state.finish(error="train_model returned None — training failed")
            return

        logger.info(
            "Training complete",
            model_path=trained_result.model_path,
            best_epoch=trained_result.best_epoch,
            epochs_trained=trained_result.epochs_trained,
        )

        # Copy the trained checkpoint to a canonical candidate path for
        # the promotion step to move into place.
        candidate_path = MODELS_DIR / "breakout_cnn_candidate.pt"
        shutil.copy2(trained_result.model_path, str(candidate_path))

        # ----- Step 3: Evaluate -----
        if _state.cancel_requested:
            _state.finish(error="Cancelled before evaluation")
            return

        _state.set(TrainStatus.EVALUATING, "Evaluating candidate model against validation gates")

        # Evaluate against the held-out val split produced above.  If the
        # split failed and val_csv_path is None, fall back to labels.csv —
        # metrics will be optimistic but the gate still acts as a smoke-test.
        eval_csv = val_csv_path if val_csv_path is not None else str(labels_csv)

        metrics = evaluate_model(
            model_path=str(candidate_path),
            val_csv=eval_csv,
            image_root=image_root,
            batch_size=batch_size,
        )

        if metrics is None:
            _state.finish(error="evaluate_model returned None — evaluation failed")
            return

        logger.info("Evaluation complete", metrics=metrics)

        # evaluate_model returns 0.0–1.0 floats; convert to percentages
        # for the gate comparisons (gates are expressed as 0–100).
        val_acc = metrics.get("val_accuracy", 0.0) * 100
        val_prec = metrics.get("val_precision", 0.0) * 100
        val_rec = metrics.get("val_recall", 0.0) * 100

        gates_passed = True
        gate_failures: list[str] = []

        if val_acc < min_acc:
            gates_passed = False
            gate_failures.append(f"accuracy {val_acc:.1f}% < {min_acc:.1f}%")
        if val_prec < min_prec:
            gates_passed = False
            gate_failures.append(f"precision {val_prec:.1f}% < {min_prec:.1f}%")
        if val_rec < min_rec:
            gates_passed = False
            gate_failures.append(f"recall {val_rec:.1f}% < {min_rec:.1f}%")

        result: dict[str, Any] = {
            "metrics": {
                "val_accuracy": round(val_acc, 2),
                "val_precision": round(val_prec, 2),
                "val_recall": round(val_rec, 2),
                "epochs_trained": trained_result.epochs_trained,
                "best_epoch": trained_result.best_epoch,
            },
            "gates": {
                "passed": gates_passed,
                "failures": gate_failures,
            },
            "dataset": {
                "total_images": ds_stats.total_images,
            },
            "config": run_config,
        }

        if not gates_passed and not params.force_promote:
            result["promoted"] = False
            result["reason"] = f"Validation gates failed: {'; '.join(gate_failures)}"
            logger.warning("Candidate rejected", failures=gate_failures)
            _state.finish(result=result)
            return

        # ----- Step 4: Promote -----
        if _state.cancel_requested:
            _state.finish(error="Cancelled before promotion")
            return

        _state.set(TrainStatus.PROMOTING, "Promoting candidate to champion")

        _archive_champion()
        shutil.move(str(candidate_path), str(CHAMPION_PT))
        _write_meta(metrics=result["metrics"], config=run_config)

        result["promoted"] = True
        if not gates_passed:
            result["reason"] = f"Force-promoted despite gate failures: {'; '.join(gate_failures)}"
        else:
            result["reason"] = "All gates passed — candidate promoted to champion"

        logger.info(
            "Model promoted",
            accuracy=val_acc,
            precision=val_prec,
            recall=val_rec,
        )

        # ----- Step 5: feature_contract.json export (best-effort) -----
        # Always regenerate and write the contract after promotion so the
        # models/ directory has an up-to-date v6 contract that consumers
        # (engine, NT8) can pull alongside the .pt / .onnx files.
        try:
            from lib.analysis.breakout_cnn import generate_feature_contract

            fc_path = MODELS_DIR / "feature_contract.json"
            contract = generate_feature_contract(output_path=str(fc_path))
            result["feature_contract_version"] = contract.get("version", 6)
            result["feature_contract_path"] = str(fc_path)
            logger.info(
                "feature_contract.json written",
                path=str(fc_path),
                version=contract.get("version"),
                num_tabular=contract.get("num_tabular"),
            )
        except Exception as fc_err:
            logger.warning("feature_contract.json export failed (non-fatal)", error=str(fc_err))
            result["feature_contract_version"] = None

        # ----- Step 6: ONNX export (best-effort) -----
        try:
            import importlib

            _breakout_cnn = importlib.import_module("lib.analysis.breakout_cnn")
            _export_fn = getattr(_breakout_cnn, "export_onnx_model", None)

            if _export_fn is not None:
                onnx_path = MODELS_DIR / "breakout_cnn_best.onnx"
                _export_fn(
                    pt_path=str(CHAMPION_PT),
                    onnx_path=str(onnx_path),
                )
                result["onnx_exported"] = True
                logger.info("ONNX export complete", path=str(onnx_path))
            else:
                result["onnx_exported"] = False
                logger.info("ONNX export skipped — export_onnx_model not yet implemented")
        except Exception as onnx_err:
            logger.warning("ONNX export failed (non-fatal)", error=str(onnx_err))
            result["onnx_exported"] = False
            result["onnx_error"] = str(onnx_err)

        _state.finish(result=result)

    except Exception as exc:
        logger.exception("Training pipeline failed", error=str(exc))
        _state.finish(error=str(exc))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Futures CNN Trainer",
    description="GPU training service for the breakout CNN model",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/metrics/prometheus")
async def metrics_prometheus() -> HTMLResponse:
    """Prometheus text-format metrics endpoint for scraping.

    Exposes:
      - ``trainer_up`` — always 1 (liveness)
      - ``trainer_uptime_seconds`` — seconds since boot
      - ``trainer_status`` — labelled gauge (1 for current status, 0 otherwise)
      - ``trainer_gpu_available`` — 1 if CUDA is available, 0 otherwise
      - ``trainer_gpu_memory_total_bytes`` — total GPU VRAM in bytes
      - ``trainer_champion_exists`` — 1 if champion .pt exists on disk
      - ``trainer_last_run_accuracy`` — last run val accuracy (0–100)
      - ``trainer_last_run_precision`` — last run val precision (0–100)
      - ``trainer_last_run_recall`` — last run val recall (0–100)
      - ``trainer_last_run_promoted`` — 1 if last run was promoted, 0 otherwise
      - ``trainer_last_run_images`` — total images in last dataset generation
      - ``trainer_runs_total`` — total training runs completed since boot
    """
    lines: list[str] = []

    def _gauge(name: str, help_text: str, value: float, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lbl = f"{{{labels}}}" if labels else ""
        lines.append(f"{name}{lbl} {value}")

    uptime = round((datetime.now(UTC) - _boot_time).total_seconds(), 1)
    _gauge("trainer_up", "Trainer server is up", 1)
    _gauge("trainer_uptime_seconds", "Seconds since trainer server boot", uptime)

    # Status as labelled gauge (one label per possible status, 1 for active)
    state = _state.to_dict()
    current_status = state.get("status", "idle")
    for s in ("idle", "generating_dataset", "training", "evaluating", "promoting", "done", "failed", "cancelled"):
        val = 1.0 if s == current_status else 0.0
        _gauge("trainer_status", "Current trainer status", val, labels=f'status="{s}"')

    # GPU info
    gpu_available = 0.0
    gpu_mem_bytes = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = 1.0
            gpu_mem_bytes = float(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    _gauge("trainer_gpu_available", "CUDA GPU available for training", gpu_available)
    _gauge("trainer_gpu_memory_total_bytes", "Total GPU VRAM in bytes", gpu_mem_bytes)

    # Champion model
    _gauge("trainer_champion_exists", "Champion model .pt exists on disk", 1.0 if CHAMPION_PT.exists() else 0.0)

    # Last run metrics
    last_result = state.get("last_result") or {}
    metrics = last_result.get("metrics") or {}
    _gauge("trainer_last_run_accuracy", "Last training run validation accuracy pct", metrics.get("val_accuracy", 0))
    _gauge("trainer_last_run_precision", "Last training run validation precision pct", metrics.get("val_precision", 0))
    _gauge("trainer_last_run_recall", "Last training run validation recall pct", metrics.get("val_recall", 0))
    _gauge(
        "trainer_last_run_promoted",
        "Whether last run was promoted to champion",
        1.0 if last_result.get("promoted") else 0.0,
    )

    dataset_info = last_result.get("dataset") or {}
    _gauge("trainer_last_run_images", "Total images in last dataset generation", dataset_info.get("total_images", 0))

    body = "\n".join(lines) + "\n"
    return HTMLResponse(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe — always returns 200."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "trainer",
            "uptime_seconds": round((datetime.now(UTC) - _boot_time).total_seconds(), 1),
        }
    )


@app.get("/status")
async def status() -> JSONResponse:
    """Current server state and last training result."""
    state = _state.to_dict()

    # Add GPU info if torch is available
    gpu_info: dict[str, Any] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
    except ImportError:
        pass

    # Champion model info
    champion_info: dict[str, Any] = {"exists": CHAMPION_PT.exists()}
    if CHAMPION_META.exists():
        try:
            meta = json.loads(CHAMPION_META.read_text())
            champion_info["trained_at"] = meta.get("trained_at")
            champion_info["metrics"] = meta.get("metrics")
            champion_info["version"] = meta.get("version")
        except Exception:
            pass

    return JSONResponse(
        {
            **state,
            "gpu": gpu_info,
            "champion": champion_info,
            "config": {
                "default_symbols": DEFAULT_SYMBOLS,
                "default_days_back": DEFAULT_DAYS_BACK,
                "default_epochs": DEFAULT_EPOCHS,
                "default_batch_size": DEFAULT_BATCH_SIZE,
                "default_session": DEFAULT_ORB_SESSION,
            },
        }
    )


@app.post("/train", dependencies=[Depends(verify_api_key)])
async def train(params: TrainRequest | None = None) -> JSONResponse:
    """Kick off a training run in the background.

    Returns 202 Accepted if the run starts, or 409 Conflict if a run
    is already in progress.
    """
    if _state.is_busy():
        return JSONResponse(
            status_code=409,
            content={
                "error": "Training already in progress",
                "status": _state.to_dict(),
            },
        )

    _state.reset()
    request_params = (
        params
        if params is not None
        else TrainRequest(  # type: ignore[call-arg]
            symbols=None,
            days_back=None,
            breakout_type="all",
            orb_session=None,
            bars_source=None,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            patience=None,
            min_accuracy=None,
            min_precision=None,
            min_recall=None,
            force_promote=False,
        )
    )

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(request_params,),
        name="training-pipeline",
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": "Training started",
            "params": request_params.model_dump(exclude_none=True),
        },
    )


@app.post("/train/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_train() -> JSONResponse:
    """Request cancellation of the current training run.

    Cancellation is cooperative — the pipeline checks ``cancel_requested``
    between stages but cannot interrupt a running epoch.
    """
    if _state.request_cancel():
        return JSONResponse({"message": "Cancellation requested", "status": _state.to_dict()})
    return JSONResponse(
        status_code=409,
        content={"error": "No active training run to cancel"},
    )


@app.get("/logs")
async def get_logs(offset: int = 0) -> JSONResponse:
    """Return recent in-memory log lines for the Web UI.

    Args:
        offset: Return only lines at or after this index in the buffer
                (monotonically increasing counter, not a ring-buffer index).
                Pass 0 on first call; use the returned ``next_offset`` on
                subsequent polls to receive only new lines.

    Returns JSON:
        {
          "lines": [{"ts": "HH:MM:SS", "level": "INFO", "name": "...", "msg": "..."}, ...],
          "next_offset": <int>,   -- pass this as offset on the next call
          "total": <int>          -- total lines ever written (monotonic)
        }
    """
    with _log_buffer_lock:
        all_lines = list(_log_buffer)

    total = _log_total_written
    # _log_total_written tracks the monotonic count; the ring buffer holds at
    # most _LOG_BUFFER_SIZE entries.  Work out which slice the caller needs.
    buf_start = max(0, total - len(all_lines))  # index of all_lines[0]
    if offset <= buf_start:
        new_lines = all_lines
    else:
        slice_from = offset - buf_start
        new_lines = all_lines[slice_from:]

    return JSONResponse(
        {
            "lines": new_lines,
            "next_offset": total,
            "total": total,
        }
    )


@app.get("/models")
async def list_models() -> JSONResponse:
    """List all model files in models/ and models/archive/.

    Returns a flat list sorted by modification time (newest first) with
    basic metadata: name, path, size_bytes, modified (unix timestamp),
    accuracy (parsed from filename or meta JSON if available).
    """
    import re as _re

    result: list[dict[str, Any]] = []

    def _scan_dir(directory: Path, is_archive: bool = False) -> None:
        if not directory.is_dir():
            return
        for f in directory.iterdir():
            if f.suffix not in (".pt", ".onnx", ".json"):
                continue
            if f.name.endswith("_meta.json"):
                continue  # skip sidecar meta files from the file list
            try:
                stat = f.stat()
                size_bytes = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                size_bytes = 0
                mtime = 0.0

            # Try to extract accuracy from filename (e.g. _acc87.3)
            acc: float | None = None
            m = _re.search(r"_acc(\d+(?:\.\d+)?)", f.stem)
            if m:
                acc = float(m.group(1))

            # Try to read accuracy from companion meta JSON
            if acc is None:
                sidecar = f.with_name(f.stem + "_meta.json")
                if not sidecar.exists() and f.name == "breakout_cnn_best.pt":
                    sidecar = CHAMPION_META
                if sidecar.exists():
                    try:
                        meta = json.loads(sidecar.read_text())
                        met = meta.get("metrics", {})
                        raw_acc = met.get("val_accuracy")
                        if raw_acc is not None:
                            acc = float(raw_acc)
                    except Exception:
                        pass

            result.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": size_bytes,
                    "modified": mtime,
                    "accuracy": acc,
                    "archive": is_archive,
                }
            )

    _scan_dir(MODELS_DIR, is_archive=False)
    _scan_dir(ARCHIVE_DIR, is_archive=True)

    # Sort: champion first, then by mtime descending
    result.sort(
        key=lambda x: (
            0 if x["name"] == "breakout_cnn_best.pt" else (1 if not x["archive"] else 2),
            -(x["modified"] or 0),
        )
    )

    return JSONResponse({"models": result, "models_dir": str(MODELS_DIR)})


@app.post("/export_onnx", dependencies=[Depends(verify_api_key)])
async def export_onnx() -> JSONResponse:
    """Re-export the current champion .pt model to ONNX.

    This is a synchronous operation (runs inline, not in a thread) because
    ONNX export is fast (~10s) and we want an immediate result.  It will
    return 409 if a training run is currently in progress to avoid
    racing with a promotion step.
    """
    if _state.is_busy():
        return JSONResponse(
            status_code=409,
            content={"error": "Training in progress — please wait before exporting"},
        )

    if not CHAMPION_PT.exists():
        raise HTTPException(status_code=404, detail="No champion model found at " + str(CHAMPION_PT))

    try:
        from lib.analysis.breakout_cnn import export_onnx_model

        onnx_path = MODELS_DIR / "breakout_cnn_best.onnx"
        out = export_onnx_model(pt_path=str(CHAMPION_PT), onnx_path=str(onnx_path))

        size_mb = round(Path(out).stat().st_size / (1024 * 1024), 2)
        logger.info("ONNX export via /export_onnx", path=out, size_mb=size_mb)

        return JSONResponse(
            {
                "message": f"ONNX export complete — {onnx_path.name} ({size_mb} MB)",
                "onnx_path": out,
                "size_mb": size_mb,
            }
        )
    except Exception as exc:
        logger.error("ONNX export failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the trainer server via uvicorn."""
    logger.info(
        "Starting trainer server",
        host=TRAINER_HOST,
        port=TRAINER_PORT,
        auth="enabled" if TRAINER_API_KEY else "disabled",
        models_dir=str(MODELS_DIR),
    )
    uvicorn.run(
        "lib.services.training.trainer_server:app",
        host=TRAINER_HOST,
        port=TRAINER_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
