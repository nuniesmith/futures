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
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
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
# Redis keys for engine ↔ data-service communication
# ---------------------------------------------------------------------------
_RETRAIN_CMD_KEY = "engine:cmd:retrain_cnn"
_RETRAIN_STATUS_KEY = "engine:retrain:status"


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
    """Return current retrain job status from Redis (engine writes it).

    Returns None if no retrain job has been triggered.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_get

        if not REDIS_AVAILABLE:
            return None

        raw = cache_get(_RETRAIN_STATUS_KEY)
        if not raw:
            return None

        data = raw if isinstance(raw, str) else raw.decode()
        return json.loads(data)

    except Exception as exc:
        logger.debug("Error reading retrain status from Redis: %s", exc)
        return None


def _start_retrain(
    session: str = "both",
    skip_dataset: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Publish a retrain command to Redis for the engine to pick up.

    The engine service has PyTorch installed and checks for this command
    key each scheduler loop iteration.  It runs the retrain pipeline in
    a background thread and publishes status back to Redis.
    """
    # Check if already running
    existing = _get_retrain_job_status()
    if existing and existing.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="A retrain job is already running. Wait for it to finish.",
        )

    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_set

        if not REDIS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Redis unavailable — cannot send retrain command to engine.",
            )

        cmd_payload = json.dumps(
            {
                "command": "retrain_cnn",
                "session": session,
                "skip_dataset": skip_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "requested_at": _now_et().isoformat(),
                "requested_by": "dashboard",
            }
        ).encode()

        cache_set(_RETRAIN_CMD_KEY, cmd_payload, ttl=300)

        # Write an initial "queued" status so the dashboard shows feedback
        # immediately (before the engine picks it up on its next loop tick)
        status_payload = json.dumps(
            {
                "status": "queued",
                "message": "Retrain command sent to engine — waiting for pickup...",
                "session": session,
                "skip_dataset": skip_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "timestamp": _now_et().isoformat(),
            }
        ).encode()
        cache_set(_RETRAIN_STATUS_KEY, status_payload, ttl=3600)

        logger.info(
            "CNN retrain command published to Redis: session=%s skip_dataset=%s epochs=%s",
            session,
            skip_dataset,
            epochs,
        )

        return {
            "status": "queued",
            "message": "Retrain command sent to engine.",
            "session": session,
            "skip_dataset": skip_dataset,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish retrain command: {exc}",
        ) from exc


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
    request: Request,
    session: str = Query("both", description="ORB session: us, london, or both"),
    skip_dataset: bool = Query(False, description="Skip dataset generation stage"),
    epochs: int | None = Query(None, description="Override training epochs (default from env/config)"),
    batch_size: int | None = Query(None, description="Override batch size"),
):
    """Trigger a CNN retraining pipeline run via Redis command to the engine.

    The engine service (which has PyTorch) picks up the command and runs
    the retrain_overnight.py pipeline in a background thread.
    """
    logger.info("CNN retrain triggered: session=%s skip_dataset=%s epochs=%s", session, skip_dataset, epochs)

    job = _start_retrain(
        session=session,
        skip_dataset=skip_dataset,
        epochs=epochs,
        batch_size=batch_size,
    )

    # If called from HTMX, return the updated CNN panel HTML
    if request.headers.get("HX-Request"):
        return cnn_status_html()

    return {
        "status": "queued",
        "message": "CNN retrain command sent to engine.",
        "job": job,
    }


@router.get("/cnn/retrain/status")
def retrain_status():
    """Poll the status of the current/last retrain job."""
    job = _get_retrain_job_status()
    if job is None:
        return {"status": "idle", "message": "No retrain job has been started."}

    return {
        "job": job,
    }


@router.post("/cnn/retrain/cancel")
def cancel_retrain(request: Request):
    """Cancel / clear a running retrain job.

    Removes the command key (in case the engine hasn't picked it up yet)
    and marks the status as cancelled.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_set

        if REDIS_AVAILABLE:
            # Remove pending command so engine doesn't pick it up
            try:
                from lib.core.cache import _r

                if _r:
                    _r.delete(_RETRAIN_CMD_KEY)
            except Exception:
                pass

            # Mark status as cancelled
            cancel_payload = json.dumps(
                {
                    "status": "cancelled",
                    "message": "Cancelled by user from dashboard",
                    "timestamp": _now_et().isoformat(),
                }
            ).encode()
            cache_set(_RETRAIN_STATUS_KEY, cancel_payload, ttl=3600)

    except Exception as exc:
        logger.error("Error cancelling retrain: %s", exc)
        if not request.headers.get("HX-Request"):
            raise HTTPException(status_code=500, detail=f"Failed to cancel: {exc}") from exc

    if request.headers.get("HX-Request"):
        return cnn_status_html()

    return {"status": "cancelled", "message": "Retrain job cancelled."}


@router.get("/cnn/history")
def retrain_history(limit: int = Query(10, ge=1, le=100)):
    """Return recent retrain audit log entries."""
    entries = _get_recent_audit_entries(limit=limit)
    return {"entries": entries, "total": len(entries)}


@router.get("/cnn/retrain/log")
def retrain_log():
    """Return the current retrain job status (logs are on the engine side)."""
    job = _get_retrain_job_status()
    return {
        "job_status": job.get("status") if job else "idle",
        "job": job,
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
        model_text = "No model found — retrain to create one"

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
    job_status: str | None = job.get("status") if job else None

    if job_status == "queued":
        retrain_html = """
            <div class="bg-yellow-900/30 border border-yellow-600/40 rounded px-2 py-1.5 mt-2">
                <div class="flex items-center justify-between">
                    <span class="text-yellow-400 text-[10px] font-bold animate-pulse">📩 QUEUED</span>
                    <button hx-post="/cnn/retrain/cancel"
                            hx-target="#cnn-panel"
                            hx-swap="innerHTML"
                            class="text-red-400 hover:text-red-300 text-[10px] px-1.5 py-0.5
                                   bg-red-900/30 rounded border border-red-700/40
                                   transition-colors duration-150">
                        ✕ Cancel
                    </button>
                </div>
                <div class="text-zinc-400 text-[10px] mt-1">Waiting for engine to pick up command...</div>
            </div>
        """
    elif job_status == "running" and job is not None:
        msg = job.get("message", "Running...")
        started = job.get("timestamp", "")
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
                <div class="text-zinc-400 text-[10px] mt-1 truncate" title="{msg}">{msg}</div>
                <div class="text-zinc-600 text-[10px]">Started: {started[:19] if started else "—"}</div>
            </div>
        """
    elif job_status in ("success", "failed", "cancelled", "gate_rejected", "rejected") and job is not None:
        status = job["status"]
        s_colors = {
            "success": ("text-green-400", "✅"),
            "failed": ("text-red-400", "❌"),
            "cancelled": ("text-yellow-400", "⚠️"),
            "gate_rejected": ("text-yellow-400", "🚫"),
            "rejected": ("text-yellow-400", "⏭️"),
        }
        s_color, s_icon = s_colors.get(status, ("text-zinc-400", "?"))
        msg = job.get("message", "") if job else ""
        msg_html = f' — <span class="text-zinc-400">{msg[:80]}</span>' if msg else ""
        retrain_html = f"""
            <div class="text-[10px] text-zinc-500 mt-1">
                Last run: <span class="{s_color}">{s_icon} {status}</span>{msg_html}
            </div>
        """

    # Retrain button (disabled if queued or running)
    if job_status in ("queued", "running"):
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

    # Build the retrain button with session selector
    retrain_btn = f"""
        <div class="mt-2 space-y-1.5">
            <!-- Session selector -->
            <div class="flex items-center gap-1 text-[10px]">
                <span class="text-zinc-500 mr-0.5">Session:</span>
                <button onclick="document.querySelectorAll('.cnn-sess-btn').forEach(b=>b.classList.remove('ring-1','ring-purple-500','text-purple-300'));this.classList.add('ring-1','ring-purple-500','text-purple-300');document.getElementById('cnn-session-val').value='london'"
                        class="cnn-sess-btn px-1.5 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400
                               border border-zinc-700 transition-all duration-150"
                        title="London Open session only (03:00–05:00 ET)">
                    🌙 London
                </button>
                <button onclick="document.querySelectorAll('.cnn-sess-btn').forEach(b=>b.classList.remove('ring-1','ring-purple-500','text-purple-300'));this.classList.add('ring-1','ring-purple-500','text-purple-300');document.getElementById('cnn-session-val').value='us'"
                        class="cnn-sess-btn px-1.5 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400
                               border border-zinc-700 transition-all duration-150"
                        title="US Open session only (08:20–10:30 ET)">
                    🇺🇸 US
                </button>
                <button onclick="document.querySelectorAll('.cnn-sess-btn').forEach(b=>b.classList.remove('ring-1','ring-purple-500','text-purple-300'));this.classList.add('ring-1','ring-purple-500','text-purple-300');document.getElementById('cnn-session-val').value='both'"
                        class="cnn-sess-btn ring-1 ring-purple-500 text-purple-300 px-1.5 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700
                               border border-zinc-700 transition-all duration-150"
                        title="Both London + US sessions">
                    Both
                </button>
                <input type="hidden" id="cnn-session-val" value="both">
            </div>
            <!-- Retrain actions -->
            <div class="flex items-center gap-1.5">
                <button id="cnn-retrain-btn"
                        hx-post="/cnn/retrain?skip_dataset=false"
                        hx-target="#cnn-panel"
                        hx-swap="innerHTML"
                        hx-confirm="Start CNN retraining? This will run dataset generation + GPU training in the background."
                        hx-indicator="#cnn-retrain-spinner"
                        hx-vals="js:{{session: document.getElementById('cnn-session-val').value}}"
                        class="flex-1 px-2 py-1.5 rounded text-[11px] font-semibold
                               border transition-all duration-200 {btn_class}"
                        {btn_disabled}>
                    {btn_text}
                </button>
                <button hx-post="/cnn/retrain?skip_dataset=true"
                        hx-target="#cnn-panel"
                        hx-swap="innerHTML"
                        hx-confirm="Retrain using existing dataset (skip generation)?"
                        hx-indicator="#cnn-retrain-spinner"
                        hx-vals="js:{{session: document.getElementById('cnn-session-val').value}}"
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
