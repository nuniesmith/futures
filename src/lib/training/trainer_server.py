"""
Trainer Server — Persistent HTTP Training Service
===================================================
FastAPI server that accepts ``POST /train`` requests, runs the full CNN
training pipeline (dataset generation → model training → evaluation →
champion promotion), and exposes health/status endpoints.

Designed to run as a long-lived Docker service on a GPU machine:

    docker compose up trainer

Or directly:

    python -m lib.training.trainer_server

The server is intentionally simple — one training job at a time, no queue.
If a training run is already in progress, ``POST /train`` returns 409.

Endpoints:
    GET  /                     — Full-featured HTML trainer dashboard (Web UI)
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
    CNN_RETRAIN_EPOCHS            — Training epochs (default 25)
    CNN_RETRAIN_BATCH_SIZE        — Batch size (default 64)
    CNN_RETRAIN_LR                — Learning rate (default 0.0002)
    CNN_RETRAIN_PATIENCE          — Early stopping patience (default 8)
    CNN_RETRAIN_SYMBOLS           — Comma-separated symbols (default: all micros)
    CNN_RETRAIN_DAYS_BACK         — Days of history (default 90)
    CNN_RETRAIN_BARS_SOURCE       — Data source (default "massive")
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
DEFAULT_EPOCHS = int(os.getenv("CNN_RETRAIN_EPOCHS", "25"))
DEFAULT_BATCH_SIZE = int(os.getenv("CNN_RETRAIN_BATCH_SIZE", "64"))
DEFAULT_LR = float(os.getenv("CNN_RETRAIN_LR", "0.0002"))
DEFAULT_PATIENCE = int(os.getenv("CNN_RETRAIN_PATIENCE", "8"))

# Dataset generation defaults
DEFAULT_SYMBOLS = os.getenv(
    "CNN_RETRAIN_SYMBOLS",
    "MGC,SIL,MHG,MCL,MNG,MES,MNQ,M2K,MYM,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET",
).split(",")
DEFAULT_DAYS_BACK = int(os.getenv("CNN_RETRAIN_DAYS_BACK", "90"))
DEFAULT_BARS_SOURCE = os.getenv("CNN_RETRAIN_BARS_SOURCE", "massive")
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

        from lib.training.dataset_generator import DatasetConfig, generate_dataset

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
            from lib.services.data.api.metrics import record_trainer_dataset_stats

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

        # Determine CSV paths — dataset_generator writes labels.csv (train)
        # and optionally val.csv (validation split).
        train_csv = DATASET_DIR / "labels.csv"
        val_csv = DATASET_DIR / "val.csv"
        image_root = str(DATASET_DIR / "images")

        trained_model_path = train_model(
            data_csv=str(train_csv),
            val_csv=str(val_csv) if val_csv.exists() else None,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_dir=str(MODELS_DIR),
            image_root=image_root,
        )

        if trained_model_path is None:
            _state.finish(error="train_model returned None — training failed")
            return

        logger.info("Training complete", model_path=trained_model_path)

        # Copy the trained checkpoint to a canonical candidate path for
        # the promotion step to move into place.
        candidate_path = MODELS_DIR / "breakout_cnn_candidate.pt"
        shutil.copy2(trained_model_path, str(candidate_path))

        # ----- Step 3: Evaluate -----
        if _state.cancel_requested:
            _state.finish(error="Cancelled before evaluation")
            return

        _state.set(TrainStatus.EVALUATING, "Evaluating candidate model against validation gates")

        # Determine which CSV to evaluate against.  Prefer an explicit
        # val.csv; fall back to the training CSV (metrics will be
        # optimistic but still useful as a smoke-test gate).
        eval_csv = str(val_csv) if val_csv.exists() else str(train_csv)

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
                "epochs_trained": metrics.get("epochs_trained", epochs),
                "best_epoch": metrics.get("best_epoch"),
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
# Web UI — HTML dashboard served at GET /
# ---------------------------------------------------------------------------

_TRAINER_UI_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Futures CNN Trainer</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background:#0f172a; color:#e2e8f0; font-family:ui-monospace,monospace; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:0.75rem; padding:1.25rem; }
  .badge-idle     { background:#1e3a5f; color:#60a5fa; }
  .badge-running  { background:#14532d; color:#4ade80; }
  .badge-done     { background:#14532d; color:#86efac; }
  .badge-failed   { background:#450a0a; color:#f87171; }
  .badge-cancelled{ background:#3b2700; color:#fb923c; }
  .badge          { display:inline-block; padding:2px 10px; border-radius:9999px; font-size:0.78rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
  .metric-val     { font-size:1.5rem; font-weight:700; }
  .metric-lbl     { font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }
  #log-box        { background:#0f172a; border:1px solid #334155; border-radius:0.5rem; height:340px; overflow-y:auto; padding:0.75rem; font-size:0.72rem; line-height:1.5; }
  .log-INFO  { color:#93c5fd; }
  .log-WARNING { color:#fbbf24; }
  .log-ERROR   { color:#f87171; }
  .log-DEBUG   { color:#64748b; }
  .log-ts      { color:#475569; margin-right:6px; }
  .log-name    { color:#818cf8; margin-right:6px; }
  .progress-bar { height:6px; background:#1e293b; border-radius:3px; overflow:hidden; margin-top:6px; }
  .progress-fill{ height:100%; background:linear-gradient(90deg,#3b82f6,#06b6d4); transition:width .4s; }
  input[type=text],input[type=number],select {
    background:#0f172a; border:1px solid #334155; border-radius:0.375rem;
    color:#e2e8f0; padding:4px 8px; width:100%; font-size:0.82rem; outline:none;
  }
  input:focus,select:focus { border-color:#3b82f6; }
  label { font-size:0.75rem; color:#94a3b8; display:block; margin-bottom:3px; }
  .btn { border-radius:0.5rem; padding:8px 20px; font-size:0.85rem; font-weight:600; cursor:pointer; border:none; transition:opacity .15s; }
  .btn:hover { opacity:0.85; }
  .btn-primary { background:#2563eb; color:#fff; }
  .btn-danger  { background:#dc2626; color:#fff; }
  .btn-secondary { background:#334155; color:#e2e8f0; }
  .btn-success { background:#16a34a; color:#fff; }
  .btn:disabled { opacity:0.4; cursor:not-allowed; }
  .section-title { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:#64748b; margin-bottom:0.75rem; }
  table { width:100%; border-collapse:collapse; font-size:0.78rem; }
  th { color:#64748b; text-align:left; padding:4px 8px; border-bottom:1px solid #334155; font-weight:600; text-transform:uppercase; font-size:0.68rem; letter-spacing:.05em; }
  td { padding:5px 8px; border-bottom:1px solid #1e293b; }
  tr:last-child td { border-bottom:none; }
  .symbol-pill { display:inline-block; background:#1e3a5f; color:#7dd3fc; border-radius:4px; padding:1px 6px; margin:2px; font-size:0.68rem; }
  .spinner { display:inline-block; width:14px; height:14px; border:2px solid #334155; border-top-color:#3b82f6; border-radius:50%; animation:spin .7s linear infinite; vertical-align:middle; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .gate-pass { color:#4ade80; }
  .gate-fail { color:#f87171; }
  .tooltip { position:relative; cursor:help; }
  .tooltip:hover::after { content:attr(data-tip); position:absolute; left:0; top:120%; background:#1e293b; border:1px solid #334155; color:#e2e8f0; font-size:0.72rem; padding:4px 8px; border-radius:4px; white-space:nowrap; z-index:10; }
</style>
</head>
<body class="min-h-screen p-4">

<!-- Header -->
<div class="flex items-center justify-between mb-6">
  <div class="flex items-center gap-3">
    <div class="text-2xl">🧠</div>
    <div>
      <div class="text-lg font-bold tracking-tight">Futures CNN Trainer</div>
      <div class="text-xs text-slate-400">EfficientNetV2-S Hybrid Breakout Model v6 (18 features) — GPU Retraining Dashboard</div>
    </div>
  </div>
  <div class="flex items-center gap-3">
    <div id="status-badge" class="badge badge-idle">idle</div>
    <div id="gpu-info" class="text-xs text-slate-400"></div>
    <div class="text-xs text-slate-500" id="uptime-display"></div>
  </div>
</div>

<!-- Main grid -->
<div class="grid grid-cols-12 gap-4">

  <!-- LEFT COLUMN: Status + Train Form (col-span-4) -->
  <div class="col-span-12 lg:col-span-4 flex flex-col gap-4">

    <!-- Current Status Card -->
    <div class="card">
      <div class="section-title">Training Status</div>
      <div id="status-progress" class="text-sm text-slate-300 mb-2 min-h-[1.5rem]">Idle — ready to train</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div class="grid grid-cols-3 gap-3 mt-4">
        <div>
          <div class="metric-val text-blue-400" id="metric-acc">—</div>
          <div class="metric-lbl tooltip" data-tip="Validation accuracy (%)">Val Acc %</div>
        </div>
        <div>
          <div class="metric-val text-purple-400" id="metric-prec">—</div>
          <div class="metric-lbl tooltip" data-tip="Validation precision (%)">Precision %</div>
        </div>
        <div>
          <div class="metric-val text-cyan-400" id="metric-rec">—</div>
          <div class="metric-lbl tooltip" data-tip="Validation recall (%)">Recall %</div>
        </div>
      </div>
      <div class="mt-3 text-xs text-slate-400" id="last-result-info"></div>
      <div class="mt-3 flex gap-2" id="action-buttons">
        <button class="btn btn-primary" id="btn-train" onclick="startTrain()">▶ Start Training</button>
        <button class="btn btn-danger" id="btn-cancel" onclick="cancelTrain()" disabled>✕ Cancel</button>
      </div>
    </div>

    <!-- Champion Model Card -->
    <div class="card">
      <div class="section-title">Champion Model</div>
      <div id="champion-info">
        <div class="text-slate-500 text-sm">No champion model found</div>
      </div>
      <div class="mt-3 flex gap-2 flex-wrap">
        <button class="btn btn-secondary text-xs" onclick="exportOnnx()" id="btn-onnx">⬇ Export ONNX</button>
        <button class="btn btn-secondary text-xs" onclick="loadModels()">📂 Refresh Models</button>
      </div>
      <div class="mt-2 text-xs" id="onnx-status"></div>
    </div>

    <!-- GPU Card -->
    <div class="card">
      <div class="section-title">Hardware</div>
      <div id="gpu-detail" class="text-sm text-slate-300">Loading...</div>
    </div>

  </div>

  <!-- MIDDLE COLUMN: Train Config (col-span-4) -->
  <div class="col-span-12 lg:col-span-4 flex flex-col gap-4">

    <div class="card flex-1">
      <div class="section-title">Training Configuration</div>

      <div class="grid grid-cols-2 gap-3 mb-3">
        <div>
          <label>Epochs</label>
          <input type="number" id="cfg-epochs" value="25" min="1" max="200"/>
        </div>
        <div>
          <label>Batch Size</label>
          <input type="number" id="cfg-batch" value="64" min="8" max="512"/>
        </div>
        <div>
          <label>Learning Rate</label>
          <input type="number" id="cfg-lr" value="0.0002" step="0.00001" min="0.00001" max="0.1"/>
        </div>
        <div>
          <label>Patience (early stop)</label>
          <input type="number" id="cfg-patience" value="8" min="1" max="50"/>
        </div>
        <div>
          <label>Days Back</label>
          <input type="number" id="cfg-days" value="90" min="7" max="365"/>
        </div>
        <div>
          <label>Bars Source</label>
          <select id="cfg-source">
            <option value="massive" selected>massive</option>
            <option value="db">db</option>
            <option value="cache">cache</option>
            <option value="csv">csv</option>
          </select>
        </div>
        <div>
          <label>ORB Session</label>
          <select id="cfg-session">
            <option value="all" selected>all</option>
            <option value="us">us</option>
            <option value="london">london</option>
            <option value="london_ny">london_ny</option>
            <option value="frankfurt">frankfurt</option>
            <option value="tokyo">tokyo</option>
            <option value="shanghai">shanghai</option>
            <option value="sydney">sydney</option>
            <option value="cme">cme</option>
          </select>
        </div>
        <div>
          <label>Breakout Type</label>
          <select id="cfg-btype">
            <option value="all" selected>all</option>
            <option value="ORB">ORB</option>
            <option value="PDR">PDR (Prev Day)</option>
            <option value="IB">IB (Init Balance)</option>
            <option value="CONS">CONS (Consolidation)</option>
          </select>
        </div>
      </div>

      <div class="section-title mt-4">Validation Gates</div>
      <div class="grid grid-cols-3 gap-3 mb-3">
        <div>
          <label>Min Accuracy %</label>
          <input type="number" id="gate-acc" value="80" min="0" max="100"/>
        </div>
        <div>
          <label>Min Precision %</label>
          <input type="number" id="gate-prec" value="75" min="0" max="100"/>
        </div>
        <div>
          <label>Min Recall %</label>
          <input type="number" id="gate-rec" value="70" min="0" max="100"/>
        </div>
      </div>

      <div class="flex items-center gap-2 mb-3">
        <input type="checkbox" id="cfg-force" class="w-4 h-4 accent-orange-500"/>
        <label class="text-orange-400 cursor-pointer mb-0" for="cfg-force" style="display:inline">Force promote (bypass gates)</label>
      </div>

      <div class="section-title mt-4">Symbols</div>
      <div class="mb-2">
        <label>Custom symbols (comma-separated, blank = all defaults)</label>
        <input type="text" id="cfg-symbols" placeholder="e.g. MES,MNQ,MGC  (blank = all 22 defaults)"/>
      </div>
      <div class="text-xs text-slate-500" id="symbol-default-hint">Defaults: MGC SIL MHG MCL MNG MES MNQ M2K MYM 6E 6B 6J 6A 6C 6S ZN ZB ZC ZS ZW MBT MET</div>
    </div>

  </div>

  <!-- RIGHT COLUMN: Logs + Model Archive (col-span-4) -->
  <div class="col-span-12 lg:col-span-4 flex flex-col gap-4">

    <!-- Live Logs -->
    <div class="card flex-1">
      <div class="flex items-center justify-between mb-2">
        <div class="section-title mb-0">Live Logs</div>
        <div class="flex gap-2 items-center">
          <label class="text-xs text-slate-500 mb-0" style="display:inline">
            <input type="checkbox" id="log-autoscroll" checked class="w-3 h-3 accent-blue-500"/> auto-scroll
          </label>
          <button class="btn btn-secondary text-xs py-1 px-2" onclick="clearLogs()">clear</button>
        </div>
      </div>
      <div id="log-box"></div>
    </div>

    <!-- Model Archive -->
    <div class="card">
      <div class="section-title">Model Archive</div>
      <div id="archive-table">
        <div class="text-slate-500 text-xs">Loading...</div>
      </div>
    </div>

  </div>

</div>

<!-- Last Run Results Panel (shown after training) -->
<div id="results-panel" class="card mt-4" style="display:none">
  <div class="section-title">Last Training Run Results</div>
  <div id="results-content" class="grid grid-cols-2 lg:grid-cols-4 gap-4"></div>
</div>

<script>
// ============================================================
// State
// ============================================================
let _lastStatus = null;
let _logOffset = 0;
let _pollTimer = null;
let _logTimer = null;

// ============================================================
// Polling
// ============================================================
function startPolling() {
  clearInterval(_pollTimer);
  clearInterval(_logTimer);
  _pollTimer = setInterval(pollStatus, 2500);
  _logTimer  = setInterval(pollLogs,   1500);
  pollStatus();
  pollLogs();
  loadModels();
}

async function pollStatus() {
  try {
    const r = await fetch('/status');
    if (!r.ok) return;
    const s = await r.json();
    renderStatus(s);
  } catch(e) {}
}

async function pollLogs() {
  try {
    const r = await fetch('/logs?offset=' + _logOffset);
    if (!r.ok) return;
    const d = await r.json();
    if (d.lines && d.lines.length > 0) {
      appendLogs(d.lines);
      _logOffset = d.next_offset;
    }
  } catch(e) {}
}

async function loadModels() {
  try {
    const r = await fetch('/models');
    if (!r.ok) return;
    const d = await r.json();
    renderArchive(d);
  } catch(e) {}
}

// ============================================================
// Render Status
// ============================================================
const STATUS_PROGRESS_MAP = {
  'idle':                 [0,   'Idle — ready to train'],
  'generating_dataset':   [20,  'Step 1/4 — Generating dataset...'],
  'training':             [50,  'Step 2/4 — Training model...'],
  'evaluating':           [80,  'Step 3/4 — Evaluating candidate...'],
  'promoting':            [95,  'Step 4/4 — Promoting champion...'],
  'done':                 [100, 'Complete ✓'],
  'failed':               [100, 'Failed ✗'],
  'cancelled':            [100, 'Cancelled'],
};

const STATUS_BADGE_CLASS = {
  'idle':               'badge-idle',
  'generating_dataset': 'badge-running',
  'training':           'badge-running',
  'evaluating':         'badge-running',
  'promoting':          'badge-running',
  'done':               'badge-done',
  'failed':             'badge-failed',
  'cancelled':          'badge-cancelled',
};

function renderStatus(s) {
  _lastStatus = s;

  // Badge
  const badge = document.getElementById('status-badge');
  badge.className = 'badge ' + (STATUS_BADGE_CLASS[s.status] || 'badge-idle');
  badge.textContent = s.status.replace('_', ' ');

  // Progress bar
  const [pct, label] = STATUS_PROGRESS_MAP[s.status] || [0, s.status];
  document.getElementById('progress-fill').style.width = pct + '%';

  let progressTxt = s.progress || label;
  if (s.status === 'failed' && s.error) progressTxt = '✗ ' + s.error;
  document.getElementById('status-progress').textContent = progressTxt;

  // Buttons
  const busy = ['generating_dataset','training','evaluating','promoting'].includes(s.status);
  document.getElementById('btn-train').disabled  = busy;
  document.getElementById('btn-cancel').disabled = !busy;

  // GPU info header
  if (s.gpu && s.gpu.available) {
    document.getElementById('gpu-info').textContent =
      s.gpu.device_name + '  ' + s.gpu.memory_total_gb + ' GB';
  }

  // GPU detail card
  if (s.gpu) {
    const g = s.gpu;
    document.getElementById('gpu-detail').innerHTML = g.available
      ? `<div class="font-semibold text-green-400 mb-1">✓ CUDA Available</div>
         <div class="text-slate-300">${g.device_name}</div>
         <div class="text-slate-400 text-xs mt-1">${g.device_count} device(s) · ${g.memory_total_gb} GB VRAM</div>`
      : `<div class="text-orange-400">⚠ No GPU detected — training will run on CPU</div>`;
  }

  // Champion card
  renderChampion(s.champion);

  // Metrics from last result
  if (s.last_result && s.last_result.metrics) {
    const m = s.last_result.metrics;
    document.getElementById('metric-acc').textContent  = m.val_accuracy  != null ? m.val_accuracy.toFixed(1)  : '—';
    document.getElementById('metric-prec').textContent = m.val_precision != null ? m.val_precision.toFixed(1) : '—';
    document.getElementById('metric-rec').textContent  = m.val_recall    != null ? m.val_recall.toFixed(1)    : '—';
  }

  // Last run details
  if (s.last_result) {
    const r = s.last_result;
    let info = '';
    if (r.promoted) info += '<span class="text-green-400">✓ Promoted to champion</span>';
    else if (r.reason) info += '<span class="text-orange-400">⚠ ' + escHtml(r.reason) + '</span>';
    if (r.dataset) info += ` · ${r.dataset.total_images} images`;
    if (s.finished_at) {
      const d = new Date(s.finished_at);
      info += ` · ${d.toLocaleTimeString()}`;
    }
    document.getElementById('last-result-info').innerHTML = info;
    renderResults(s.last_result);
  }

  // Uptime
  if (s.started_at) {
    const age = Math.floor((Date.now() - new Date(s.started_at)) / 1000);
    document.getElementById('uptime-display').textContent = 'Run time: ' + fmtSecs(age);
  }

  // Config defaults on first load
  if (s.config) {
    const c = s.config;
    if (!document.getElementById('cfg-epochs').dataset.userEdited) {
      document.getElementById('cfg-epochs').value  = c.default_epochs    || 25;
      document.getElementById('cfg-days').value    = c.default_days_back || 90;
      const srcEl = document.getElementById('cfg-source');
      if (c.default_session) document.getElementById('cfg-session').value = c.default_session;
    }
  }
}

function renderChampion(ch) {
  if (!ch || !ch.exists) {
    document.getElementById('champion-info').innerHTML =
      '<div class="text-slate-500 text-sm">No champion model on disk</div>';
    document.getElementById('btn-onnx').disabled = true;
    return;
  }
  document.getElementById('btn-onnx').disabled = false;
  let html = '<div class="text-green-400 font-semibold text-sm mb-2">✓ breakout_cnn_best.pt</div>';
  if (ch.metrics) {
    const m = ch.metrics;
    html += `<div class="grid grid-cols-3 gap-2 mb-2">
      <div><div class="metric-val text-sm text-blue-400">${fmtPct(m.val_accuracy)}</div><div class="metric-lbl">Acc</div></div>
      <div><div class="metric-val text-sm text-purple-400">${fmtPct(m.val_precision)}</div><div class="metric-lbl">Prec</div></div>
      <div><div class="metric-val text-sm text-cyan-400">${fmtPct(m.val_recall)}</div><div class="metric-lbl">Recall</div></div>
    </div>`;
  }
  if (ch.trained_at) {
    const d = new Date(ch.trained_at);
    html += `<div class="text-slate-400 text-xs">Trained: ${d.toLocaleString()}</div>`;
  }
  if (ch.version) html += `<div class="text-slate-500 text-xs">Version: ${ch.version}</div>`;
  document.getElementById('champion-info').innerHTML = html;
}

function renderResults(r) {
  if (!r) { document.getElementById('results-panel').style.display='none'; return; }
  document.getElementById('results-panel').style.display='';
  let html = '';

  // Metrics block
  if (r.metrics) {
    const m = r.metrics;
    html += `
      <div class="card bg-slate-900">
        <div class="section-title">Metrics</div>
        <div class="space-y-1 text-sm">
          <div>Accuracy:  <span class="text-blue-400 font-bold">${fmtPct(m.val_accuracy)}%</span></div>
          <div>Precision: <span class="text-purple-400 font-bold">${fmtPct(m.val_precision)}%</span></div>
          <div>Recall:    <span class="text-cyan-400 font-bold">${fmtPct(m.val_recall)}%</span></div>
          <div class="text-slate-400 text-xs">Epochs: ${m.epochs_trained || '—'} · Best epoch: ${m.best_epoch || '—'}</div>
        </div>
      </div>`;
  }

  // Gates block
  if (r.gates) {
    const g = r.gates;
    html += `
      <div class="card bg-slate-900">
        <div class="section-title">Gates</div>
        <div class="text-sm">
          <div class="${g.passed ? 'gate-pass' : 'gate-fail'} font-bold mb-1">${g.passed ? '✓ All gates passed' : '✗ Gates failed'}</div>
          ${(g.failures || []).map(f => `<div class="gate-fail text-xs">• ${escHtml(f)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Dataset block
  if (r.dataset) {
    html += `
      <div class="card bg-slate-900">
        <div class="section-title">Dataset</div>
        <div class="text-sm space-y-1">
          <div>Images: <span class="text-slate-200 font-bold">${r.dataset.total_images}</span></div>
          ${r.promoted ? '<div class="text-green-400 font-bold">✓ Promoted</div>' : ''}
          ${r.onnx_exported ? '<div class="text-green-400">✓ ONNX exported</div>' : ''}
          ${r.feature_contract_version ? '<div class="text-slate-400 text-xs">feature_contract v' + r.feature_contract_version + '</div>' : ''}
        </div>
      </div>`;
  }

  // Config block
  if (r.config) {
    const c = r.config;
    html += `
      <div class="card bg-slate-900">
        <div class="section-title">Config</div>
        <div class="text-xs text-slate-400 space-y-1">
          <div>Epochs: ${c.epochs} · LR: ${c.learning_rate} · Batch: ${c.batch_size}</div>
          <div>Session: ${c.orb_session} · Source: ${c.bars_source}</div>
          <div>Days: ${c.days_back} · Type: ${c.breakout_type}</div>
          <div class="flex flex-wrap mt-1">${(c.symbols||[]).map(s=>`<span class="symbol-pill">${s}</span>`).join('')}</div>
        </div>
      </div>`;
  }

  document.getElementById('results-content').innerHTML = html;
}

function renderArchive(data) {
  const models = data.models || [];
  if (!models.length) {
    document.getElementById('archive-table').innerHTML = '<div class="text-slate-500 text-xs">No model files found</div>';
    return;
  }
  let html = '<table><thead><tr><th>File</th><th>Size</th><th>Modified</th><th>Acc</th></tr></thead><tbody>';
  for (const m of models) {
    const name = m.name;
    const isChamp = name === 'breakout_cnn_best.pt';
    const isOnnx  = name.endsWith('.onnx');
    const nameClass = isChamp ? 'text-green-400 font-bold' : (isOnnx ? 'text-yellow-400' : 'text-slate-300');
    const prefix = isChamp ? '★ ' : (isOnnx ? '⬡ ' : '');
    html += `<tr>
      <td class="${nameClass}" title="${escHtml(m.path)}">${prefix}${escHtml(name)}</td>
      <td class="text-slate-400">${fmtSize(m.size_bytes)}</td>
      <td class="text-slate-500">${m.modified ? new Date(m.modified*1000).toLocaleDateString() : '—'}</td>
      <td class="text-blue-400">${m.accuracy ? m.accuracy.toFixed(1)+'%' : '—'}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('archive-table').innerHTML = html;
}

// ============================================================
// Actions
// ============================================================
async function startTrain() {
  const symbols_raw = document.getElementById('cfg-symbols').value.trim();
  const symbols = symbols_raw ? symbols_raw.split(',').map(s=>s.trim()).filter(Boolean) : null;

  const payload = {
    epochs:        parseInt(document.getElementById('cfg-epochs').value),
    batch_size:    parseInt(document.getElementById('cfg-batch').value),
    learning_rate: parseFloat(document.getElementById('cfg-lr').value),
    patience:      parseInt(document.getElementById('cfg-patience').value),
    days_back:     parseInt(document.getElementById('cfg-days').value),
    bars_source:   document.getElementById('cfg-source').value,
    orb_session:   document.getElementById('cfg-session').value,
    breakout_type: document.getElementById('cfg-btype').value,
    min_accuracy:  parseFloat(document.getElementById('gate-acc').value),
    min_precision: parseFloat(document.getElementById('gate-prec').value),
    min_recall:    parseFloat(document.getElementById('gate-rec').value),
    force_promote: document.getElementById('cfg-force').checked,
  };
  if (symbols) payload.symbols = symbols;

  document.getElementById('btn-train').disabled = true;
  document.getElementById('results-panel').style.display = 'none';

  try {
    const r = await fetch('/train', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (r.status === 202) {
      appendLog({level:'INFO', ts: nowTs(), name:'ui', msg:'Training started ✓'});
    } else if (r.status === 409) {
      appendLog({level:'WARNING', ts: nowTs(), name:'ui', msg:'Training already in progress'});
    } else {
      appendLog({level:'ERROR', ts: nowTs(), name:'ui', msg: JSON.stringify(d)});
    }
  } catch(e) {
    appendLog({level:'ERROR', ts: nowTs(), name:'ui', msg: String(e)});
  }
  pollStatus();
}

async function cancelTrain() {
  if (!confirm('Cancel the current training run?')) return;
  try {
    const r = await fetch('/train/cancel', {method:'POST'});
    const d = await r.json();
    appendLog({level:'WARNING', ts: nowTs(), name:'ui', msg: d.message || 'Cancellation requested'});
  } catch(e) {}
  pollStatus();
}

async function exportOnnx() {
  document.getElementById('onnx-status').innerHTML = '<span class="spinner"></span> Exporting ONNX...';
  document.getElementById('btn-onnx').disabled = true;
  try {
    const r = await fetch('/export_onnx', {method:'POST'});
    const d = await r.json();
    if (r.ok) {
      document.getElementById('onnx-status').innerHTML =
        `<span class="text-green-400">✓ ${escHtml(d.message || 'ONNX exported')}</span>` +
        (d.size_mb ? ` <span class="text-slate-500 text-xs">(${d.size_mb} MB)</span>` : '');
      appendLog({level:'INFO', ts:nowTs(), name:'ui', msg:'ONNX export complete: ' + (d.onnx_path||'')});
    } else {
      document.getElementById('onnx-status').innerHTML =
        `<span class="text-red-400">✗ ${escHtml(d.detail || JSON.stringify(d))}</span>`;
    }
  } catch(e) {
    document.getElementById('onnx-status').innerHTML = `<span class="text-red-400">✗ ${e}</span>`;
  }
  document.getElementById('btn-onnx').disabled = false;
  loadModels();
}

function clearLogs() {
  document.getElementById('log-box').innerHTML = '';
}

// ============================================================
// Logs
// ============================================================
function appendLog(entry) {
  const box = document.getElementById('log-box');
  const div = document.createElement('div');
  const cls = 'log-' + (entry.level || 'INFO');
  div.innerHTML =
    `<span class="log-ts">${escHtml(entry.ts)}</span>` +
    `<span class="log-name">[${escHtml(entry.name||'')}]</span>` +
    `<span class="${cls}">${escHtml(entry.msg)}</span>`;
  box.appendChild(div);
  if (document.getElementById('log-autoscroll').checked) {
    box.scrollTop = box.scrollHeight;
  }
}

function appendLogs(lines) {
  for (const l of lines) appendLog(l);
}

// ============================================================
// Helpers
// ============================================================
function fmtPct(v) {
  if (v == null || v === '') return '—';
  return parseFloat(v).toFixed(1);
}
function fmtSize(b) {
  if (!b) return '—';
  if (b > 1048576) return (b/1048576).toFixed(1)+' MB';
  if (b > 1024)    return (b/1024).toFixed(0)+' KB';
  return b+' B';
}
function fmtSecs(s) {
  const m = Math.floor(s/60), sec = s%60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function escHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function nowTs() {
  return new Date().toTimeString().slice(0,8);
}

// ============================================================
// Boot
// ============================================================
document.addEventListener('DOMContentLoaded', startPolling);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def trainer_ui() -> HTMLResponse:
    """Serve the full-featured trainer web dashboard."""
    return HTMLResponse(content=_TRAINER_UI_HTML)


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
    _gauge("trainer_last_run_promoted", "Whether last run was promoted to champion", 1.0 if last_result.get("promoted") else 0.0)

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


@app.get("/metrics/prometheus")
async def metrics_prometheus() -> HTMLResponse:
    """Prometheus text-format metrics endpoint.

    Exposes the following gauges:

    * ``trainer_up``                      — 1 if healthy
    * ``trainer_uptime_seconds``          — seconds since boot
    * ``trainer_gpu_available``           — 1 if CUDA is available
    * ``trainer_gpu_memory_total_bytes``  — total VRAM in bytes (0 if no GPU)
    * ``trainer_status``                  — labelled gauge: status=idle|training|...
    * ``trainer_champion_exists``         — 1 if breakout_cnn_best.pt is on disk
    * ``trainer_champion_accuracy``       — last champion val accuracy (0–100)
    * ``trainer_champion_precision``      — last champion val precision (0–100)
    * ``trainer_champion_recall``         — last champion val recall (0–100)
    * ``trainer_runs_total``              — monotonic count of completed training runs
    * ``trainer_last_run_duration_seconds`` — wall-clock seconds of the last run
    * ``trainer_last_run_promoted``       — 1 if the last run was promoted
    * ``trainer_images_generated``        — images generated in last dataset build
    """
    lines: list[str] = []

    def _gauge(name: str, help_text: str, value: float, labels: str = "") -> None:
        lines.append(f"# HELP


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
    request_params = params or TrainRequest()

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
        raise HTTPException(status_code=500, detail=str(exc))


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
        "lib.training.trainer_server:app",
        host=TRAINER_HOST,
        port=TRAINER_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
