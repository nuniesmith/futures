"""
CNN Model Management API Router
=================================
Provides endpoints for CNN model status, retraining triggers, and
training log access from the web dashboard.

Endpoints:
    GET  /cnn/status          — Current model info + last training results
    POST /cnn/retrain         — Trigger CNN retraining pipeline (async)
    GET  /cnn/retrain/status  — Poll status of a running retrain job
    POST /cnn/retrain/cancel  — Cancel a running retrain job
    GET  /cnn/history         — Recent retrain audit log entries
    GET  /cnn/status/html     — HTML fragment for dashboard panel
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.cnn")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["CNN"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # futures/
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIT_LOG_PATH = MODEL_DIR / "retrain_audit.jsonl"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.csv"
LABELS_CSV = DATASET_DIR / "labels.csv"
TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
CHAMPION_PATH = MODEL_DIR / "breakout_cnn_best.pt"

# ---------------------------------------------------------------------------
# Retrain job state — module-level singleton (only one retrain at a time)
# ---------------------------------------------------------------------------

_retrain_lock = threading.Lock()
_retrain_job: dict[str, Any] | None = None
_retrain_process: subprocess.Popen | None = None  # type: ignore[type-arg]
_retrain_log: deque[str] = deque(maxlen=500)


def _now_et() -> datetime:
    return datetime.now(tz=_EST)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_model_info() -> dict[str, Any]:
    """Gather CNN model status without importing torch (safe for any env)."""
    info: dict[str, Any] = {
        "available": False,
        "model_path": None,
        "size_mb": 0.0,
        "modified": None,
        "modified_ago": None,
    }

    # Find the champion / latest model
    champion = CHAMPION_PATH
    if champion.is_file():
        info["available"] = True
        info["model_path"] = str(champion)
        info["is_champion"] = True
    else:
        # Fallback: find newest .pt file
        pt_files = sorted(MODEL_DIR.glob("breakout_cnn_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pt_files:
            champion = pt_files[0]
            info["available"] = True
            info["model_path"] = str(champion)
            info["is_champion"] = False

    if info["available"] and champion.is_file():
        stat = champion.stat()
        info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
        mod_dt = datetime.fromtimestamp(stat.st_mtime, tz=_EST)
        info["modified"] = mod_dt.isoformat()
        delta = _now_et() - mod_dt
        hours = delta.total_seconds() / 3600
        if hours < 1:
            info["modified_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
        elif hours < 24:
            info["modified_ago"] = f"{hours:.1f}h ago"
        else:
            info["modified_ago"] = f"{delta.days}d ago"

    # Count models in directory
    all_pt = list(MODEL_DIR.glob("breakout_cnn_*.pt"))
    info["total_models"] = len(all_pt)

    return info


def _get_dataset_info() -> dict[str, Any]:
    """Gather dataset stats."""
    info: dict[str, Any] = {
        "labels_csv_exists": LABELS_CSV.is_file(),
        "train_csv_exists": TRAIN_CSV.is_file(),
        "val_csv_exists": VAL_CSV.is_file(),
        "total_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
    }

    try:
        if LABELS_CSV.is_file():
            # Count lines (minus header)
            with open(LABELS_CSV) as f:
                info["total_samples"] = max(0, sum(1 for _ in f) - 1)
        if TRAIN_CSV.is_file():
            with open(TRAIN_CSV) as f:
                info["train_samples"] = max(0, sum(1 for _ in f) - 1)
        if VAL_CSV.is_file():
            with open(VAL_CSV) as f:
                info["val_samples"] = max(0, sum(1 for _ in f) - 1)
    except Exception as exc:
        logger.warning("Error reading dataset CSVs: %s", exc)

    # Dataset stats JSON
    stats_path = DATASET_DIR / "dataset_stats.json"
    if stats_path.is_file():
        try:
            with open(stats_path) as f:
                info["stats"] = json.load(f)
        except Exception:
            pass

    return info


def _get_training_history_summary() -> dict[str, Any]:
    """Read the last few rows of training_history.csv for a quick summary."""
    summary: dict[str, Any] = {"available": False}
    if not TRAINING_HISTORY_PATH.is_file():
        return summary

    try:
        import csv

        rows: list[dict[str, str]] = []
        with open(TRAINING_HISTORY_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return summary

        summary["available"] = True
        summary["total_epochs"] = len(rows)

        # Best epoch by val_acc
        best_row = max(rows, key=lambda r: float(r.get("val_acc", "0") or "0"))
        summary["best_epoch"] = int(best_row.get("epoch", 0))
        summary["best_val_acc"] = float(best_row.get("val_acc", 0) or 0)
        summary["best_val_loss"] = float(best_row.get("val_loss", 0) or 0)

        # Last epoch
        last = rows[-1]
        summary["last_epoch"] = int(last.get("epoch", 0))
        summary["last_val_acc"] = float(last.get("val_acc", 0) or 0)
        summary["last_train_acc"] = float(last.get("train_acc", 0) or 0)

    except Exception as exc:
        logger.warning("Error reading training history: %s", exc)

    return summary


def _get_recent_audit_entries(limit: int = 10) -> list[dict[str, Any]]:
    """Read recent entries from retrain_audit.jsonl."""
    entries: list[dict[str, Any]] = []
    if not AUDIT_LOG_PATH.is_file():
        return entries

    try:
        with open(AUDIT_LOG_PATH) as f:
            lines = f.readlines()

        for line in lines[-limit:]:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    entries.append(json.loads(line))
    except Exception as exc:
        logger.warning("Error reading audit log: %s", exc)

    # Reverse so newest first
    entries.reverse()
    return entries


def _get_retrain_job_status() -> dict[str, Any] | None:
    """Return current retrain job status, or None if no job running."""
    global _retrain_job, _retrain_process

    if _retrain_job is None:
        return None

    # Check if process is still alive
    if _retrain_process is not None:
        poll = _retrain_process.poll()
        if poll is not None:
            # Process finished
            _retrain_job["status"] = "success" if poll == 0 else "failed"
            _retrain_job["exit_code"] = poll
            _retrain_job["finished_at"] = _now_et().isoformat()
            elapsed = time.monotonic() - _retrain_job.get("_start_mono", time.monotonic())
            _retrain_job["duration_seconds"] = round(elapsed, 1)
            _retrain_process = None

    return {k: v for k, v in _retrain_job.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Background retrain runner
# ---------------------------------------------------------------------------


def _stream_output(proc: subprocess.Popen, job: dict[str, Any]) -> None:  # type: ignore[type-arg]
    """Read stdout/stderr from the retrain subprocess and buffer it."""
    global _retrain_log
    try:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            _retrain_log.append(line)
            # Update progress hints from log lines
            if "Stage" in line:
                job["current_stage"] = line.strip()
            elif "epoch" in line.lower() or "Epoch" in line:
                job["last_epoch_line"] = line.strip()
    except Exception:
        pass


def _start_retrain(
    session: str = "both",
    skip_dataset: bool = False,
    immediate: bool = True,
    force: bool = True,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Launch the retrain subprocess and return the initial job status."""
    global _retrain_job, _retrain_process, _retrain_log

    with _retrain_lock:
        # Check if already running
        existing = _get_retrain_job_status()
        if existing and existing.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail="A retrain job is already running. Cancel it first or wait for it to finish.",
            )

        # Build command
        retrain_script = SCRIPT_DIR / "retrain_overnight.py"
        if not retrain_script.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Retrain script not found: {retrain_script}",
            )

        python_exe = sys.executable
        cmd = [python_exe, str(retrain_script)]

        if immediate:
            cmd.append("--immediate")
        if skip_dataset:
            cmd.append("--skip-dataset")
        if force:
            cmd.append("--force")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR)
        if session:
            env["CNN_RETRAIN_ORB_SESSION"] = session
        if epochs is not None:
            env["CNN_RETRAIN_EPOCHS"] = str(epochs)
        if batch_size is not None:
            env["CNN_RETRAIN_BATCH_SIZE"] = str(batch_size)

        # Clear log buffer
        _retrain_log.clear()

        # Launch subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start retrain process: {exc}",
            ) from exc

        _retrain_process = proc

        job: dict[str, Any] = {
            "status": "running",
            "pid": proc.pid,
            "started_at": _now_et().isoformat(),
            "finished_at": None,
            "exit_code": None,
            "duration_seconds": 0,
            "session": session,
            "skip_dataset": skip_dataset,
            "immediate": immediate,
            "force": force,
            "epochs": epochs,
            "batch_size": batch_size,
            "current_stage": "Starting...",
            "last_epoch_line": "",
            "_start_mono": time.monotonic(),
        }
        _retrain_job = job

        # Start output streaming thread
        reader = threading.Thread(target=_stream_output, args=(proc, job), daemon=True)
        reader.start()

        return {k: v for k, v in job.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.get("/cnn/status")
def cnn_status():
    """Return CNN model status, dataset info, and training history summary."""
    model = _get_model_info()
    dataset = _get_dataset_info()
    history = _get_training_history_summary()
    job = _get_retrain_job_status()

    return {
        "model": model,
        "dataset": dataset,
        "training_history": history,
        "retrain_job": job,
        "timestamp": _now_et().isoformat(),
    }


@router.post("/cnn/retrain")
def trigger_retrain(
    session: str = Query("both", description="ORB session: us, london, or both"),
    skip_dataset: bool = Query(False, description="Skip dataset generation stage"),
    epochs: int | None = Query(None, description="Override training epochs (default from env/config)"),
    batch_size: int | None = Query(None, description="Override batch size"),
):
    """Trigger a CNN retraining pipeline run.

    Launches the retrain_overnight.py script as a background subprocess
    with --immediate --force flags. Only one retrain job can run at a time.
    """
    logger.info("CNN retrain triggered: session=%s skip_dataset=%s epochs=%s", session, skip_dataset, epochs)

    job = _start_retrain(
        session=session,
        skip_dataset=skip_dataset,
        immediate=True,
        force=True,
        epochs=epochs,
        batch_size=batch_size,
    )

    return {
        "status": "started",
        "message": "CNN retraining pipeline started in background.",
        "job": job,
    }


@router.get("/cnn/retrain/status")
def retrain_status():
    """Poll the status of the current/last retrain job."""
    job = _get_retrain_job_status()
    if job is None:
        return {"status": "idle", "message": "No retrain job has been started."}

    # Include tail of log
    log_lines = list(_retrain_log)[-50:]

    return {
        "job": job,
        "log_tail": log_lines,
    }


@router.post("/cnn/retrain/cancel")
def cancel_retrain():
    """Cancel a running retrain job."""
    global _retrain_job, _retrain_process

    with _retrain_lock:
        if _retrain_process is None or _retrain_process.poll() is not None:
            return {"status": "no_job", "message": "No running retrain job to cancel."}

        try:
            pid = _retrain_process.pid
            # Try graceful SIGTERM first, then SIGKILL
            if sys.platform == "win32":
                _retrain_process.terminate()
            else:
                os.kill(pid, signal.SIGTERM)

            # Give it a moment to terminate
            try:
                _retrain_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _retrain_process.kill()

            if _retrain_job is not None:
                _retrain_job["status"] = "cancelled"
                _retrain_job["finished_at"] = _now_et().isoformat()

            _retrain_process = None

            return {
                "status": "cancelled",
                "message": f"Retrain job (PID {pid}) cancelled.",
            }

        except Exception as exc:
            logger.error("Error cancelling retrain: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to cancel: {exc}") from exc


@router.get("/cnn/history")
def retrain_history(limit: int = Query(10, ge=1, le=100)):
    """Return recent retrain audit log entries."""
    entries = _get_recent_audit_entries(limit=limit)
    return {"entries": entries, "total": len(entries)}


@router.get("/cnn/retrain/log")
def retrain_log(lines: int = Query(100, ge=1, le=500)):
    """Return the last N lines of the current retrain job's output."""
    log_lines = list(_retrain_log)[-lines:]
    job = _get_retrain_job_status()
    return {
        "log": log_lines,
        "job_status": job.get("status") if job else "idle",
        "total_lines": len(_retrain_log),
    }


# ---------------------------------------------------------------------------
# HTML fragment for dashboard panel
# ---------------------------------------------------------------------------


@router.get("/cnn/status/html", response_class=HTMLResponse)
def cnn_status_html():
    """Return a dashboard-ready HTML fragment for the CNN model panel."""
    model = _get_model_info()
    dataset = _get_dataset_info()
    history = _get_training_history_summary()
    job = _get_retrain_job_status()

    # Model status indicator
    if model["available"]:
        model_dot = '<span class="text-green-500">●</span>'
        model_text = f"Model ready ({model['size_mb']} MB)"
        model_age = model.get("modified_ago", "")
        if model_age:
            model_text += f" · {model_age}"
    else:
        model_dot = '<span class="text-red-500">●</span>'
        model_text = "No model found"

    # Training history
    history_html = ""
    if history.get("available"):
        best_acc = history.get("best_val_acc", 0)
        acc_color = "text-green-400" if best_acc >= 80 else "text-yellow-400" if best_acc >= 65 else "text-red-400"
        history_html = f"""
            <div class="text-[10px] text-zinc-500 mt-1 space-y-0.5">
                <div>Best val acc: <span class="{acc_color} font-mono">{best_acc:.1f}%</span>
                     (epoch {history.get("best_epoch", "?")})</div>
                <div>Epochs trained: <span class="text-zinc-300 font-mono">{history.get("total_epochs", 0)}</span></div>
            </div>
        """

    # Dataset info
    dataset_html = ""
    total = dataset.get("total_samples", 0)
    train = dataset.get("train_samples", 0)
    val = dataset.get("val_samples", 0)
    if total > 0:
        dataset_html = f"""
            <div class="text-[10px] text-zinc-500">
                Dataset: <span class="text-zinc-300 font-mono">{total:,}</span> samples
                (train {train:,} / val {val:,})
            </div>
        """

    # Retrain job status
    retrain_html = ""
    if job and job.get("status") == "running":
        stage = job.get("current_stage", "Running...")
        started = job.get("started_at", "")
        retrain_html = f"""
            <div class="bg-blue-900/30 border border-blue-600/40 rounded px-2 py-1.5 mt-2">
                <div class="flex items-center justify-between">
                    <span class="text-blue-400 text-[10px] font-bold animate-pulse">⚡ TRAINING</span>
                    <button hx-post="/cnn/retrain/cancel"
                            hx-target="#cnn-panel"
                            hx-swap="innerHTML"
                            hx-confirm="Cancel the running CNN retrain job?"
                            class="text-red-400 hover:text-red-300 text-[10px] px-1.5 py-0.5
                                   bg-red-900/30 rounded border border-red-700/40
                                   transition-colors duration-150">
                        ✕ Cancel
                    </button>
                </div>
                <div class="text-zinc-400 text-[10px] mt-1 truncate" title="{stage}">{stage}</div>
                <div class="text-zinc-600 text-[10px]">Started: {started[:19] if started else "—"}</div>
            </div>
        """
    elif job and job.get("status") in ("success", "failed", "cancelled", "gate_rejected"):
        status = job["status"]
        s_colors = {
            "success": ("text-green-400", "✅"),
            "failed": ("text-red-400", "❌"),
            "cancelled": ("text-yellow-400", "⚠️"),
            "gate_rejected": ("text-yellow-400", "🚫"),
        }
        s_color, s_icon = s_colors.get(status, ("text-zinc-400", "?"))
        duration = job.get("duration_seconds", 0)
        dur_str = f"{duration:.0f}s" if duration < 120 else f"{duration / 60:.1f}m"
        retrain_html = f"""
            <div class="text-[10px] text-zinc-500 mt-1">
                Last run: <span class="{s_color}">{s_icon} {status}</span>
                ({dur_str})
            </div>
        """

    # Retrain button (disabled if already running)
    if job and job.get("status") == "running":
        btn_class = "opacity-40 cursor-not-allowed bg-zinc-800 text-zinc-600"
        btn_disabled = "disabled"
        btn_text = "Training..."
    else:
        btn_class = (
            "bg-purple-900/50 hover:bg-purple-800/60 text-purple-300 "
            "border-purple-600/40 hover:border-purple-500/60 cursor-pointer"
        )
        btn_disabled = ""
        btn_text = "🧠 Retrain CNN"

    # Build the retrain button with dropdown for options
    retrain_btn = f"""
        <div class="mt-2 space-y-1.5">
            <div class="flex items-center gap-1.5">
                <button id="cnn-retrain-btn"
                        hx-post="/cnn/retrain?session=both&skip_dataset=false"
                        hx-target="#cnn-panel"
                        hx-swap="innerHTML"
                        hx-confirm="Start CNN retraining? This will run dataset generation + GPU training in the background."
                        hx-indicator="#cnn-retrain-spinner"
                        class="flex-1 px-2 py-1.5 rounded text-[11px] font-semibold
                               border transition-all duration-200 {btn_class}"
                        {btn_disabled}>
                    {btn_text}
                </button>
                <button hx-post="/cnn/retrain?session=both&skip_dataset=true"
                        hx-target="#cnn-panel"
                        hx-swap="innerHTML"
                        hx-confirm="Retrain using existing dataset (skip generation)?"
                        hx-indicator="#cnn-retrain-spinner"
                        class="px-2 py-1.5 rounded text-[10px]
                               bg-zinc-800 hover:bg-zinc-700 text-zinc-400
                               border border-zinc-700 transition-colors duration-150"
                        title="Quick retrain — skip dataset generation, use existing images"
                        {btn_disabled}>
                    ⚡ Quick
                </button>
            </div>
            <span id="cnn-retrain-spinner" class="htmx-indicator text-zinc-500 text-[10px]">
                Starting retrain job...
            </span>
        </div>
    """

    return HTMLResponse(
        content=f"""
        <div class="flex items-center justify-between mb-1.5">
            <h3 class="text-sm font-semibold text-zinc-400">🧠 CNN MODEL</h3>
            <span class="text-zinc-600 text-[10px]">{model.get("total_models", 0)} checkpoints</span>
        </div>
        <div class="text-xs">
            <div class="flex items-center gap-1.5">
                {model_dot}
                <span class="text-zinc-300">{model_text}</span>
            </div>
            {history_html}
            {dataset_html}
        </div>
        {retrain_html}
        {retrain_btn}
    """
    )
