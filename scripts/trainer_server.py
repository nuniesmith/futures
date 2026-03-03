#!/usr/bin/env python
"""
Trainer Server — Long-Running GPU Training Service
====================================================
FastAPI HTTP server that wraps the overnight retraining pipeline and
exposes it as a persistent service.  Instead of exiting after one run,
the server stays alive, accepts training requests via HTTP, and runs the
pipeline in a background thread.

This allows the main stack (docker-compose.yml, running on CPU on any
machine including a Raspberry Pi) to trigger GPU retraining on this
dedicated machine via a simple HTTP POST, and then rsync the resulting
model back over Tailscale.

Architecture:
    Main stack  ──HTTP POST──→  Trainer Server (port 8200)
    (any CPU)                     │
                                  ├─ POST /train          trigger pipeline
                                  ├─ GET  /status         job + health info
                                  ├─ GET  /health         liveness check
                                  ├─ GET  /model          current champion info
                                  └─ POST /train/cancel   interrupt running job

Sync workflow (after training completes):
    bash scripts/sync_models.sh 100.113.72.63
    docker compose restart engine   # pick up new model

Usage:
    # Inside Docker (default CMD):
    python scripts/trainer_server.py

    # Standalone:
    PYTHONPATH=src python scripts/trainer_server.py --host 0.0.0.0 --port 8200

Environment variables:
    TRAINER_HOST        Bind host (default: 0.0.0.0)
    TRAINER_PORT        Bind port (default: 8200)
    TRAINER_API_KEY     Optional bearer token for auth (leave unset = no auth)
    PYTHONPATH          Must include /app/src (or project src/)
    CNN_RETRAIN_*       All the usual pipeline tunables from RetrainConfig
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# ---------------------------------------------------------------------------
# Path bootstrap — allows running directly without installing the package
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
logger = logging.getLogger("trainer_server")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAINER_HOST = os.getenv("TRAINER_HOST", "0.0.0.0")
TRAINER_PORT = int(os.getenv("TRAINER_PORT", "8200"))
TRAINER_API_KEY = os.getenv("TRAINER_API_KEY", "").strip()  # empty = no auth

MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
HEALTH_FILE = Path("/tmp/trainer_health.json")

# ---------------------------------------------------------------------------
# Global training-job state (single-job concurrency)
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()

_job: dict[str, Any] = {
    "status": "idle",  # idle | running | success | failed | gate_rejected | cancelled
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "duration_seconds": None,
    "message": "Trainer ready — waiting for a training request.",
    "config": {},
    "result": None,  # RetrainResult.to_dict() on completion
    "progress": "",  # last progress line from the pipeline logger
    "cancel_requested": False,
    "total_runs": 0,
    "last_success_at": None,
    "last_failure_at": None,
}

_cancel_event = threading.Event()
_pipeline_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Health file writer (Docker HEALTHCHECK reads this)
# ---------------------------------------------------------------------------


def _write_health(healthy: bool = True) -> None:
    """Write /tmp/trainer_health.json for Docker HEALTHCHECK."""
    data = {
        "healthy": healthy,
        "status": _job["status"],
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }
    try:
        HEALTH_FILE.write_text(json.dumps(data))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional API-key auth (no-op when TRAINER_API_KEY is unset)
# ---------------------------------------------------------------------------


def _check_api_key(request: Request) -> None:
    """Dependency: validate Bearer token if TRAINER_API_KEY is configured."""
    if not TRAINER_API_KEY:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != TRAINER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Pipeline runner (executes in a background thread)
# ---------------------------------------------------------------------------


def _run_pipeline_thread(cfg_overrides: dict[str, Any]) -> None:
    """Target function for the background training thread.

    Imports retrain_overnight locally so that the server starts up fast
    even if the heavy dependencies (torch, torchvision) are not yet warm.
    """
    global _job

    run_id = f"retrain_{datetime.now():%Y%m%d_%H%M%S}"

    with _job_lock:
        _job.update(
            status="running",
            run_id=run_id,
            started_at=datetime.now(tz=_EST).isoformat(),
            finished_at=None,
            duration_seconds=None,
            result=None,
            progress="Importing pipeline modules…",
            cancel_requested=False,
            message="Training pipeline is running.",
        )

    _write_health(True)
    t0 = time.monotonic()

    try:
        # Lazy import — load retrain_overnight from the scripts/ directory.
        # We use importlib so the import works whether scripts/ is on sys.path
        # or not (mirrors how retrain_overnight.py loads train_gpu.py itself).
        import importlib.util

        _rto_path = SCRIPT_DIR / "retrain_overnight.py"
        _spec = importlib.util.spec_from_file_location("retrain_overnight", str(_rto_path))
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not load module spec from {_rto_path}")
        _rto_mod = importlib.util.module_from_spec(_spec)
        # Register in sys.modules BEFORE exec_module so that @dataclass
        # decorators can resolve cls.__module__ via sys.modules correctly.
        sys.modules["retrain_overnight"] = _rto_mod
        _spec.loader.exec_module(_rto_mod)  # type: ignore[union-attr]

        RetrainConfig = _rto_mod.RetrainConfig  # noqa: N806
        run_pipeline = _rto_mod.run_pipeline

        # Build config from environment, then apply per-request overrides
        cfg = RetrainConfig.from_env()
        cfg.immediate = cfg_overrides.get("immediate", True)
        cfg.skip_dataset = cfg_overrides.get("skip_dataset", False)
        cfg.dry_run = cfg_overrides.get("dry_run", False)
        cfg.force = cfg_overrides.get("force", True)  # server always forces (no skip-today guard)

        if "session" in cfg_overrides and cfg_overrides["session"]:
            cfg.orb_session = cfg_overrides["session"]
        if "epochs" in cfg_overrides and cfg_overrides["epochs"] is not None:
            cfg.epochs = int(cfg_overrides["epochs"])
        if "batch_size" in cfg_overrides and cfg_overrides["batch_size"] is not None:
            cfg.batch_size = int(cfg_overrides["batch_size"])
        if "days_back" in cfg_overrides and cfg_overrides["days_back"] is not None:
            cfg.days_back = int(cfg_overrides["days_back"])

        with _job_lock:
            _job["config"] = {
                "session": cfg.orb_session,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "days_back": cfg.days_back,
                "skip_dataset": cfg.skip_dataset,
                "dry_run": cfg.dry_run,
                "immediate": cfg.immediate,
            }
            _job["progress"] = "Pipeline started — stage 1: dataset refresh"

        logger.info("Pipeline thread started: run_id=%s  cfg=%s", run_id, _job["config"])

        # run_pipeline is blocking — it handles its own logging
        result = run_pipeline(cfg)

        duration = time.monotonic() - t0

        with _job_lock:
            _job["finished_at"] = datetime.now(tz=_EST).isoformat()
            _job["duration_seconds"] = round(duration, 1)
            _job["result"] = result.to_dict() if hasattr(result, "to_dict") else {"status": result.status}
            _job["total_runs"] += 1

            if result.status == "success":
                _job["status"] = "success"
                _job["message"] = (
                    f"Training succeeded — model promoted "
                    f"(acc={result.best_val_accuracy:.1f}%, "
                    f"prec={result.best_precision:.1f}%, "
                    f"recall={result.best_recall:.1f}%)"
                )
                _job["last_success_at"] = _job["finished_at"]
                _job["progress"] = "✅ Done — model promoted"
                logger.info("Pipeline SUCCESS: %s", _job["message"])

            elif result.status == "gate_rejected":
                _job["status"] = "gate_rejected"
                _job["message"] = f"Candidate rejected by validation gate: {result.gate_reason}"
                _job["last_failure_at"] = _job["finished_at"]
                _job["progress"] = f"🚫 Gate rejected: {result.gate_reason}"
                logger.warning("Pipeline GATE_REJECTED: %s", result.gate_reason)

            elif result.status == "skipped":
                _job["status"] = "idle"
                _job["message"] = "Pipeline skipped (already ran today or active session)."
                _job["progress"] = "⏭️ Skipped"
                logger.info("Pipeline SKIPPED")

            else:
                _job["status"] = "failed"
                errors_preview = "; ".join((result.errors or [])[:3])
                _job["message"] = f"Pipeline failed: {errors_preview or 'unknown error'}"
                _job["last_failure_at"] = _job["finished_at"]
                _job["progress"] = f"❌ Failed: {errors_preview}"
                logger.error("Pipeline FAILED: %s", _job["message"])

    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("Pipeline thread crashed: %s", exc, exc_info=True)
        with _job_lock:
            _job["status"] = "failed"
            _job["finished_at"] = datetime.now(tz=_EST).isoformat()
            _job["duration_seconds"] = round(duration, 1)
            _job["message"] = f"Pipeline crashed: {exc}"
            _job["last_failure_at"] = _job["finished_at"]
            _job["progress"] = f"💥 Crashed: {exc}"
            _job["total_runs"] += 1

    finally:
        _write_health(True)


# ---------------------------------------------------------------------------
# Model info helper (torch-free)
# ---------------------------------------------------------------------------


def _model_info() -> dict[str, Any]:
    """Return champion model metadata without importing torch."""
    champion = MODEL_DIR / "breakout_cnn_best.pt"
    meta_path = MODEL_DIR / "breakout_cnn_best_meta.json"
    audit_path = MODEL_DIR / "retrain_audit.jsonl"

    info: dict[str, Any] = {
        "champion_available": champion.is_file(),
        "champion_path": str(champion) if champion.is_file() else None,
        "champion_size_mb": None,
        "champion_modified": None,
        "meta": None,
        "total_pt_files": len(list(MODEL_DIR.glob("breakout_cnn_*.pt"))),
        "audit_entries": 0,
    }

    if champion.is_file():
        st = champion.stat()
        info["champion_size_mb"] = round(st.st_size / (1024 * 1024), 1)
        info["champion_modified"] = datetime.fromtimestamp(st.st_mtime, tz=_EST).isoformat()

    if meta_path.is_file():
        try:
            info["meta"] = json.loads(meta_path.read_text())
        except Exception:
            pass

    if audit_path.is_file():
        try:
            info["audit_entries"] = sum(1 for line in audit_path.read_text().splitlines() if line.strip())
        except Exception:
            pass

    return info


def _dataset_info() -> dict[str, Any]:
    """Return dataset stats without importing pandas."""
    labels = DATASET_DIR / "labels.csv"
    train = DATASET_DIR / "train.csv"
    val = DATASET_DIR / "val.csv"

    def _count(p: Path) -> int:
        if not p.is_file():
            return 0
        try:
            return max(0, sum(1 for _ in p.open()) - 1)  # minus header
        except Exception:
            return 0

    return {
        "labels_csv": labels.is_file(),
        "train_csv": train.is_file(),
        "val_csv": val.is_file(),
        "total_samples": _count(labels),
        "train_samples": _count(train),
        "val_samples": _count(val),
    }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Futures Trainer Server",
    description="GPU training service — triggers CNN retraining pipeline via HTTP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    """Liveness + readiness check.

    Returns 200 when the server is running regardless of training state.
    The 'status' field tells you what the trainer is currently doing.
    """
    with _job_lock:
        status = _job["status"]
        message = _job["message"]
        run_id = _job["run_id"]

    return JSONResponse(
        {
            "healthy": True,
            "status": status,
            "run_id": run_id,
            "message": message,
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "service": "trainer-server",
        }
    )


@app.get("/status", response_class=JSONResponse)
def status() -> JSONResponse:
    """Full job status + model info + dataset info."""
    with _job_lock:
        job_snapshot = dict(_job)

    return JSONResponse(
        {
            "job": job_snapshot,
            "model": _model_info(),
            "dataset": _dataset_info(),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@app.post("/train", response_class=JSONResponse)
async def trigger_train(
    request: Request,
    _auth: None = Depends(_check_api_key),
) -> JSONResponse:
    """Trigger a training run.

    Body (all fields optional, JSON):
    {
        "session":       "all",     // ORB session key or "all"
        "skip_dataset":  false,     // skip dataset generation
        "dry_run":       false,     // validate but don't promote
        "immediate":     true,      // ignore time-window guards
        "force":         true,      // run even if ran today
        "epochs":        null,      // override epoch count
        "batch_size":    null,      // override batch size
        "days_back":     null       // override days of history
    }

    Returns 409 if a training job is already running.
    Returns 202 (Accepted) when the job is queued.
    """
    global _pipeline_thread

    with _job_lock:
        current_status = _job["status"]

    if current_status == "running":
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Check /status or POST /train/cancel first.",
        )

    # Parse optional body
    cfg_overrides: dict[str, Any] = {}
    try:
        body_bytes = await request.body()
        if body_bytes:
            cfg_overrides = json.loads(body_bytes) or {}
    except Exception:
        pass  # malformed or empty body — use defaults

    # Defaults: always run immediately + force (server use-case)
    cfg_overrides.setdefault("immediate", True)
    cfg_overrides.setdefault("force", True)
    cfg_overrides.setdefault("skip_dataset", False)
    cfg_overrides.setdefault("dry_run", False)

    # Launch background thread
    _pipeline_thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(cfg_overrides,),
        name="trainer-pipeline",
        daemon=True,
    )
    _pipeline_thread.start()

    logger.info("Training job launched: overrides=%s", cfg_overrides)

    return JSONResponse(
        {
            "status": "accepted",
            "message": "Training pipeline started in background.",
            "config": cfg_overrides,
            "poll": "/status",
        },
        status_code=202,
    )


@app.post("/train/cancel", response_class=JSONResponse)
def cancel_train(
    _auth: None = Depends(_check_api_key),
) -> JSONResponse:
    """Request cancellation of the currently running training job.

    The pipeline checks _cancel_event periodically in long-running loops.
    The response is immediate; the job may take a moment to notice.
    """
    with _job_lock:
        current_status = _job["status"]
        if current_status != "running":
            raise HTTPException(
                status_code=409,
                detail=f"No job is currently running (status={current_status}).",
            )
        _job["cancel_requested"] = True
        _job["progress"] = "Cancellation requested — stopping after current stage…"

    _cancel_event.set()
    logger.info("Cancellation requested")

    return JSONResponse(
        {"status": "cancellation_requested", "message": "The running job will stop after its current stage."}
    )


@app.get("/model", response_class=JSONResponse)
def model_info() -> JSONResponse:
    """Return champion model metadata."""
    return JSONResponse(_model_info())


@app.get("/dataset", response_class=JSONResponse)
def dataset_info() -> JSONResponse:
    """Return dataset statistics."""
    return JSONResponse(_dataset_info())


@app.get("/logs", response_class=PlainTextResponse)
def recent_logs(lines: int = 100) -> PlainTextResponse:
    """Return the last N lines of the retrain audit log."""
    audit_path = MODEL_DIR / "retrain_audit.jsonl"
    if not audit_path.is_file():
        return PlainTextResponse("No audit log found.\n")
    try:
        all_lines = audit_path.read_text().splitlines()
        recent = all_lines[-lines:]
        # Pretty-print each JSON line
        pretty: list[str] = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                pretty.append(json.dumps(json.loads(line), indent=2))
            except Exception:
                pretty.append(line)
        return PlainTextResponse("\n---\n".join(pretty) + "\n")
    except Exception as exc:
        return PlainTextResponse(f"Error reading audit log: {exc}\n")


@app.get("/", response_class=JSONResponse)
def root() -> JSONResponse:
    """API index."""
    return JSONResponse(
        {
            "service": "futures-trainer-server",
            "version": "1.0.0",
            "endpoints": {
                "GET  /health": "Liveness check",
                "GET  /status": "Full job + model + dataset status",
                "POST /train": "Trigger training pipeline",
                "POST /train/cancel": "Cancel running job",
                "GET  /model": "Champion model info",
                "GET  /dataset": "Dataset statistics",
                "GET  /logs": "Recent audit log entries",
            },
        }
    )


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    logger.info("=" * 60)
    logger.info("  🖥️  Futures Trainer Server")
    logger.info("  Listening on %s:%d", TRAINER_HOST, TRAINER_PORT)
    logger.info("  Project root: %s", PROJECT_ROOT)
    logger.info("  Models dir:   %s", MODEL_DIR)
    logger.info("  Dataset dir:  %s", DATASET_DIR)
    logger.info("  Auth:         %s", "enabled" if TRAINER_API_KEY else "disabled (set TRAINER_API_KEY)")
    logger.info("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "archive").mkdir(parents=True, exist_ok=True)

    _write_health(True)
    logger.info("Trainer server ready — waiting for training requests.")


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("Trainer server shutting down")
    _write_health(False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Futures Trainer Server — GPU training service",
    )
    parser.add_argument("--host", default=TRAINER_HOST, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=TRAINER_PORT, help="Bind port (default: 8200)")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev only)")
    args = parser.parse_args()

    # Graceful shutdown on SIGTERM (Docker stop)
    def _handle_signal(signum: int, frame: Any) -> None:
        logger.info("Received signal %d — shutting down", signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Starting trainer server on %s:%d", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        access_log=True,
    )


if __name__ == "__main__":
    main()
