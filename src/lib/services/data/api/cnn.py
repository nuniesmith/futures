"""
CNN Model Management API Router
=================================
Provides endpoints for CNN model status, per-session gate management,
model sync from the orb repo, and a dashboard HTML panel for the web frontend.

CNN training lives in the orb repo (github.com/nuniesmith/orb).
Models are pulled into this repo via scripts/sync_models.sh.
This service only runs inference on CPU from the synced .pt checkpoint.

Endpoints:
    GET  /cnn/status              — Current model info + metadata (JSON)
    GET  /cnn/status/html         — HTML fragment for dashboard panel
    GET  /cnn/history             — Recent ORB audit log entries with CNN data
    POST /cnn/sync                — Trigger model sync from orb repo
    GET  /cnn/watcher/status      — Model file-watcher health

Per-session CNN gate endpoints:
    GET    /cnn/gate              — Return gate state for all 9 sessions
    PUT    /cnn/gate/{session_key} — Enable or disable the gate for one session
    DELETE /cnn/gate/{session_key} — Remove per-session override (revert to env-var)
    DELETE /cnn/gate              — Remove all overrides
    GET    /cnn/gate/html         — Dashboard HTML fragment for gate panel
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.cnn")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["CNN"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # futures/
MODEL_DIR = PROJECT_ROOT / "models"
CHAMPION_PATH = MODEL_DIR / "breakout_cnn_best.pt"
META_PATH = MODEL_DIR / "breakout_cnn_best_meta.json"
ONNX_PATH = MODEL_DIR / "breakout_cnn_best.onnx"
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_models.sh"

# Docker paths (when running inside the engine container)
_DOCKER_SYNC_SCRIPT = Path("/app/scripts/sync_models.sh")

# Sync state — shared across requests (single-process)
_sync_lock = asyncio.Lock()
_sync_status: dict[str, Any] = {
    "running": False,
    "last_run": None,
    "last_result": None,
    "last_exit_code": None,
    "last_error": None,
}


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
        "is_champion": False,
    }

    champion = CHAMPION_PATH
    if champion.is_file():
        info["available"] = True
        info["model_path"] = str(champion)
        info["is_champion"] = True
    else:
        # Fallback: find newest .pt file
        pt_files = sorted(
            MODEL_DIR.glob("breakout_cnn_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pt_files:
            champion = pt_files[0]
            info["available"] = True
            info["model_path"] = str(champion)

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

    # Count checkpoints in directory
    all_pt = list(MODEL_DIR.glob("breakout_cnn_*.pt"))
    info["total_models"] = len(all_pt)

    return info


def _get_meta_info() -> dict[str, Any] | None:
    """Read breakout_cnn_best_meta.json if it exists."""
    if not META_PATH.is_file():
        return None
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return None


def _get_sync_status() -> dict[str, Any]:
    """Compute sync / staleness status from the model + meta files."""
    result: dict[str, Any] = {
        "meta_available": False,
        "onnx_available": ONNX_PATH.is_file(),
        "stale": False,
        "val_accuracy": None,
        "precision": None,
        "recall": None,
        "epochs": None,
        "promoted_ago": None,
        "last_sync_ago": None,
        "version": None,
    }

    meta = _get_meta_info()
    if not meta:
        return result

    result["meta_available"] = True
    result["val_accuracy"] = meta.get("val_accuracy")
    result["precision"] = meta.get("precision")
    result["recall"] = meta.get("recall")
    result["epochs"] = meta.get("epochs_trained")
    result["version"] = meta.get("run_id", "")

    # Promoted age
    promoted_at = meta.get("promoted_at")
    if promoted_at:
        try:
            dt = datetime.fromisoformat(promoted_at)
            delta = _now_et() - dt
            hours = delta.total_seconds() / 3600
            if hours < 1:
                result["promoted_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
            elif hours < 24:
                result["promoted_ago"] = f"{hours:.1f}h ago"
            else:
                result["promoted_ago"] = f"{delta.days}d ago"
        except Exception:
            pass

    # Last sync age (mtime of the .pt file)
    if CHAMPION_PATH.is_file():
        try:
            mod_dt = datetime.fromtimestamp(CHAMPION_PATH.stat().st_mtime, tz=_EST)
            delta = _now_et() - mod_dt
            hours = delta.total_seconds() / 3600
            if hours < 1:
                result["last_sync_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
            elif hours < 24:
                result["last_sync_ago"] = f"{hours:.1f}h ago"
            else:
                result["last_sync_ago"] = f"{delta.days}d ago"
        except Exception:
            pass

    # Consider stale if promoted > 7 days ago
    if promoted_at:
        try:
            dt = datetime.fromisoformat(promoted_at)
            if (_now_et() - dt).days > 7:
                result["stale"] = True
        except Exception:
            pass

    return result


def _get_recent_audit_entries(limit: int = 10) -> list[dict[str, Any]]:
    """Read recent ORB audit entries that include CNN data."""
    try:
        from lib.core.models import get_orb_events

        events = get_orb_events(limit=limit, breakout_only=False)
        # Filter to events that have CNN metadata
        cnn_events = []
        for ev in events:
            meta_raw = ev.get("metadata_json", "")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                    if meta.get("cnn_prob") is not None:
                        ev["cnn_meta"] = meta
                        cnn_events.append(ev)
                except Exception:
                    pass
        return cnn_events[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# JSON endpoints
# ---------------------------------------------------------------------------


@router.get("/cnn/status")
def cnn_status():
    """Return CNN model status and metadata."""
    model = _get_model_info()
    meta = _get_meta_info()
    sync = _get_sync_status()

    return {
        "model": model,
        "meta": meta,
        "sync": {
            "onnx_available": sync["onnx_available"],
            "stale": sync["stale"],
            "meta_available": sync["meta_available"],
        },
        "note": "CNN training lives in the orb repo (github.com/nuniesmith/orb). Sync model with: bash scripts/sync_models.sh",
        "timestamp": _now_et().isoformat(),
    }


@router.get("/cnn/history")
def cnn_history(limit: int = Query(10, ge=1, le=100)):
    """Return recent ORB events that include CNN inference data."""
    entries = _get_recent_audit_entries(limit=limit)
    return {"entries": entries, "total": len(entries)}


# ---------------------------------------------------------------------------
# Model sync endpoint — runs scripts/sync_models.sh in the background
# ---------------------------------------------------------------------------


def _find_sync_script() -> Path | None:
    """Locate the sync_models.sh script (Docker or bare-metal)."""
    for candidate in (_DOCKER_SYNC_SCRIPT, SYNC_SCRIPT):
        if candidate.is_file():
            return candidate
    return None


def _run_sync(pt_only: bool = False) -> dict[str, Any]:
    """Execute sync_models.sh synchronously and return the result.

    Called from a background task so it doesn't block the event loop.
    """
    global _sync_status

    script = _find_sync_script()
    if script is None:
        result = {
            "success": False,
            "exit_code": -1,
            "error": "sync_models.sh not found",
            "stdout": "",
            "stderr": "",
        }
        _sync_status.update(
            running=False,
            last_run=_now_et().isoformat(),
            last_result="error",
            last_exit_code=-1,
            last_error="sync_models.sh not found",
        )
        return result

    cmd = ["bash", str(script)]
    if pt_only:
        cmd.append("--pt-only")

    logger.info("🔄 Starting model sync: %s", " ".join(cmd))
    _sync_status["running"] = True

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(script.parent.parent),  # repo root
        )

        success = proc.returncode == 0
        result = {
            "success": success,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "error": None if success else f"Exit code {proc.returncode}",
        }

        _sync_status.update(
            running=False,
            last_run=_now_et().isoformat(),
            last_result="ok" if success else "failed",
            last_exit_code=proc.returncode,
            last_error=None if success else proc.stderr[-500:] if proc.stderr else f"exit {proc.returncode}",
        )

        if success:
            logger.info("✅ Model sync completed successfully")
            # Invalidate the model cache so next inference picks up the new model
            try:
                from lib.analysis.breakout_cnn import invalidate_model_cache

                invalidate_model_cache()
                logger.info("CNN model cache invalidated after sync")
            except Exception as exc:
                logger.debug("Cache invalidation after sync failed (non-fatal): %s", exc)
        else:
            logger.warning(
                "⚠️  Model sync failed (exit %d): %s",
                proc.returncode,
                proc.stderr[:200] if proc.stderr else "no stderr",
            )

        return result

    except subprocess.TimeoutExpired:
        result = {
            "success": False,
            "exit_code": -1,
            "error": "Sync script timed out after 120 seconds",
            "stdout": "",
            "stderr": "",
        }
        _sync_status.update(
            running=False,
            last_run=_now_et().isoformat(),
            last_result="timeout",
            last_exit_code=-1,
            last_error="Script timed out after 120s",
        )
        logger.error("Model sync timed out")
        return result

    except Exception as exc:
        result = {
            "success": False,
            "exit_code": -1,
            "error": str(exc),
            "stdout": "",
            "stderr": "",
        }
        _sync_status.update(
            running=False,
            last_run=_now_et().isoformat(),
            last_result="error",
            last_exit_code=-1,
            last_error=str(exc),
        )
        logger.error("Model sync error: %s", exc)
        return result


@router.post("/cnn/sync")
async def cnn_sync(
    background_tasks: BackgroundTasks,
    pt_only: bool = Query(False, description="Only sync the .pt checkpoint (skip ONNX/meta)"),
):
    """Trigger a model sync from the orb GitHub repo.

    Runs ``scripts/sync_models.sh`` in a background task.  The script
    downloads the latest champion model files from the orb repo via
    the Git LFS batch API.

    If a sync is already running, returns 409 Conflict.

    Query the sync status with GET /cnn/sync/status.
    """
    if _sync_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="A model sync is already in progress. Check /cnn/sync/status for updates.",
        )

    script = _find_sync_script()
    if script is None:
        raise HTTPException(
            status_code=404,
            detail="sync_models.sh not found. Expected at scripts/sync_models.sh",
        )

    # Run sync in background so the API returns immediately
    _sync_status["running"] = True

    def _bg_sync():
        _run_sync(pt_only=pt_only)

    background_tasks.add_task(_bg_sync)

    return {
        "status": "started",
        "message": "Model sync started in background. Check /cnn/sync/status for progress.",
        "script": str(script),
        "pt_only": pt_only,
        "timestamp": _now_et().isoformat(),
    }


@router.get("/cnn/sync/status")
def cnn_sync_status():
    """Return the current model sync status."""
    return {
        **_sync_status,
        "script_found": _find_sync_script() is not None,
        "timestamp": _now_et().isoformat(),
    }


# ---------------------------------------------------------------------------
# Model watcher status endpoint
# ---------------------------------------------------------------------------


@router.get("/cnn/watcher/status")
def cnn_watcher_status():
    """Return the status of the engine's model file-watcher.

    The watcher monitors the models/ directory for changes to
    ``breakout_cnn_best.pt`` and related files, and automatically
    invalidates the CNN inference cache when changes are detected.

    Returns watcher backend (watchdog/polling/none), watched directory,
    and whether the watcher thread is alive.
    """
    try:
        from lib.services.engine.model_watcher import _WATCHDOG_AVAILABLE, _find_model_dir

        # Try to access the engine's running watcher instance
        try:
            from lib.services.engine.main import _model_watcher

            if _model_watcher is not None:
                return {
                    "status": "ok",
                    **_model_watcher.status(),
                    "timestamp": _now_et().isoformat(),
                }
        except (ImportError, AttributeError):
            pass

        # Watcher not running — return capability info
        model_dir = _find_model_dir()
        return {
            "status": "not_running",
            "running": False,
            "backend": "none",
            "model_dir": str(model_dir) if model_dir else None,
            "watchdog_available": _WATCHDOG_AVAILABLE,
            "note": "The model watcher runs inside the engine process. "
            "If engine is not running in this process, status is unavailable.",
            "timestamp": _now_et().isoformat(),
        }
    except ImportError:
        return {
            "status": "unavailable",
            "running": False,
            "backend": "none",
            "error": "model_watcher module not available",
            "timestamp": _now_et().isoformat(),
        }


# ---------------------------------------------------------------------------
# Per-session CNN gate endpoints
# ---------------------------------------------------------------------------

_OVERNIGHT_SESSIONS = {"cme", "sydney", "tokyo", "shanghai"}
_SESSION_LABELS: dict[str, str] = {
    "cme": "CME Open 18:00 ET",
    "sydney": "Sydney/ASX 18:30 ET",
    "tokyo": "Tokyo/TSE 19:00 ET",
    "shanghai": "Shanghai/HK 21:00 ET",
    "frankfurt": "Frankfurt 03:00 ET",
    "london": "London 03:00 ET",
    "london_ny": "London-NY 08:00 ET",
    "us": "US Equity 09:30 ET",
    "cme_settle": "CME Settle 14:00 ET",
}
_SESSION_ORDER = list(_SESSION_LABELS.keys())


def _get_gates_payload() -> dict[str, Any]:
    """Return the full CNN gate state dict, annotated with effective values."""
    try:
        from lib.core.redis_helpers import get_all_cnn_gates

        raw = get_all_cnn_gates()
        global_env = raw.pop("_global_env", False)

        sessions = []
        for sk in _SESSION_ORDER:
            override = raw.get(sk)  # True / False / None
            effective = override if override is not None else global_env
            sessions.append(
                {
                    "key": sk,
                    "label": _SESSION_LABELS.get(sk, sk),
                    "is_overnight": sk in _OVERNIGHT_SESSIONS,
                    "override": override,
                    "effective": effective,
                    "source": "redis" if override is not None else "env",
                }
            )

        return {
            "sessions": sessions,
            "global_env": global_env,
            "redis_available": True,
        }
    except Exception as exc:
        logger.warning("get_gates_payload failed: %s", exc)
        return {
            "sessions": [],
            "global_env": False,
            "redis_available": False,
            "error": str(exc),
        }


@router.get("/cnn/gate")
def get_cnn_gates():
    """Return the CNN hard-gate state for all 9 ORB sessions.

    Each session entry reports:
    - ``override``: ``true``/``false`` if a Redis key is set, ``null`` otherwise.
    - ``effective``: the value the engine will actually use.
    - ``source``: ``"redis"`` or ``"env"``.
    """
    return _get_gates_payload()


@router.put("/cnn/gate/{session_key}")
def set_cnn_gate_endpoint(session_key: str, enabled: bool = True):
    """Enable or disable the CNN hard gate for a single session."""
    sk = session_key.lower().strip()
    if sk not in _SESSION_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown session key '{sk}'. Valid keys: {', '.join(_SESSION_ORDER)}",
        )

    try:
        from lib.core.redis_helpers import set_cnn_gate

        ok = set_cnn_gate(sk, enabled)
        if not ok:
            raise HTTPException(status_code=503, detail="Redis unavailable")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "session_key": sk,
        "label": _SESSION_LABELS[sk],
        "enabled": enabled,
        "message": f"CNN gate {'ENABLED' if enabled else 'DISABLED'} for '{sk}'",
    }


@router.delete("/cnn/gate/{session_key}")
def reset_cnn_gate_endpoint(session_key: str):
    """Remove the Redis override for *session_key*."""
    sk = session_key.lower().strip()
    if sk not in _SESSION_LABELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown session key '{sk}'. Valid keys: {', '.join(_SESSION_ORDER)}",
        )

    try:
        from lib.core.redis_helpers import reset_cnn_gate

        reset_cnn_gate(sk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "session_key": sk,
        "removed": True,
        "message": f"Override removed for '{sk}' — now using ORB_CNN_GATE env var",
    }


@router.delete("/cnn/gate")
def reset_all_cnn_gates_endpoint():
    """Remove all per-session Redis overrides."""
    try:
        from lib.core.redis_helpers import reset_all_cnn_gates

        count = reset_all_cnn_gates()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "removed_count": count,
        "message": f"Removed {count} override(s) — all sessions now use ORB_CNN_GATE env var",
    }


@router.get("/cnn/gate/html", response_class=HTMLResponse)
def cnn_gate_html():
    """Return a dashboard HTML fragment for the per-session CNN gate panel."""
    data = _get_gates_payload()
    sessions: list[dict[str, Any]] = data.get("sessions", [])
    global_env: bool = data.get("global_env", False)
    redis_ok: bool = data.get("redis_available", False)

    if not redis_ok:
        return HTMLResponse(
            content='<div class="text-[10px] text-red-400">⚠ Redis unavailable — gate state unknown</div>'
        )

    rows_html = ""
    for s in sessions:
        sk: str = s["key"]
        label: str = s["label"]
        effective: bool = s["effective"]
        source: str = s["source"]
        is_overnight: bool = s["is_overnight"]

        if effective:
            dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e"></span>'
            gate_text = "ON"
            gate_color = "color:#4ade80"
            toggle_action = (
                f"hx-delete='/cnn/gate/{sk}'" if source == "redis" else f"hx-put='/cnn/gate/{sk}?enabled=false'"
            )
        else:
            dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#52525b"></span>'
            gate_text = "off"
            gate_color = "color:#71717a"
            toggle_action = f"hx-put='/cnn/gate/{sk}?enabled=true'"

        source_badge = (
            '<span style="font-size:9px;color:#60a5fa;background:rgba(30,58,138,0.3);border:1px solid rgba(37,99,235,0.4);border-radius:3px;padding:0 3px">redis</span>'
            if source == "redis"
            else '<span style="font-size:9px;color:#52525b">env</span>'
        )
        overnight_badge = '<span style="font-size:9px">🌙</span>' if is_overnight else ""

        rows_html += f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:2px 0">
            <div style="display:flex;align-items:center;gap:6px;min-width:0">
                {dot} {overnight_badge}
                <span style="font-size:10px;color:#d4d4d8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{label}">{label}</span>
                {source_badge}
            </div>
            <button {toggle_action}
                    hx-target="#cnn-gate-panel" hx-swap="innerHTML"
                    style="font-size:10px;{gate_color};padding:2px 6px;border-radius:4px;
                           background:var(--bg-input);border:1px solid var(--border-panel);
                           cursor:pointer;font-family:monospace;width:32px;text-align:center">
                {gate_text}
            </button>
        </div>
        """

    env_badge = '<span style="color:#4ade80">ON</span>' if global_env else '<span style="color:#71717a">off</span>'

    bulk_html = """
        <div style="display:flex;gap:4px;margin-top:8px;padding-top:6px;border-top:1px solid var(--border-subtle)">
            <button onclick="['cme','sydney','tokyo','shanghai'].forEach(function(s){htmx.ajax('PUT','/cnn/gate/'+s+'?enabled=true',{target:'#cnn-gate-panel',swap:'innerHTML'})});return false;"
                    style="flex:1;font-size:10px;padding:4px 6px;border-radius:4px;background:var(--bg-input);
                           color:var(--text-muted);border:1px solid var(--border-panel);cursor:pointer">
                🌙 Enable overnight
            </button>
            <button hx-delete="/cnn/gate" hx-target="#cnn-gate-panel" hx-swap="innerHTML"
                    hx-confirm="Remove all Redis overrides?"
                    style="font-size:10px;padding:4px 6px;border-radius:4px;background:var(--bg-input);
                           color:var(--text-muted);border:1px solid var(--border-panel);cursor:pointer">
                ↺ Reset all
            </button>
        </div>
    """

    return HTMLResponse(
        content=f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <h3 class="text-sm font-semibold t-text-muted">🔒 CNN GATE</h3>
            <span class="text-[10px] t-text-faint">env default: {env_badge}</span>
        </div>
        <div class="space-y-0 text-xs">{rows_html}</div>
        {bulk_html}
        """
    )


# ---------------------------------------------------------------------------
# Dashboard HTML panel
# ---------------------------------------------------------------------------


@router.get("/cnn/status/html", response_class=HTMLResponse)
def cnn_status_html():
    """Return a dashboard-ready HTML fragment for the CNN model panel.

    Shows model availability, accuracy metrics from meta.json, sync info,
    and a button to sync the latest model from the orb repo.
    """
    model = _get_model_info()
    sync = _get_sync_status()

    # ── Model availability indicator ──────────────────────────────────
    if model["available"]:
        if sync["stale"]:
            model_dot = '<span style="color:#eab308">●</span>'
            stale_badge = '<span style="font-size:9px;color:#eab308;background:rgba(113,63,18,0.3);border:1px solid rgba(161,98,7,0.4);border-radius:3px;padding:0 4px;margin-left:4px">stale</span>'
        else:
            model_dot = '<span style="color:#22c55e">●</span>'
            stale_badge = ""
        model_text = f"Model ready ({model['size_mb']} MB)"
        model_age = model.get("modified_ago", "")
        if model_age:
            model_text += f" · {model_age}"
    else:
        model_dot = '<span style="color:#ef4444">●</span>'
        stale_badge = ""
        model_text = 'No model found — run: <span style="font-family:monospace">bash scripts/sync_models.sh</span>'

    # ── Accuracy metrics (from meta.json) ─────────────────────────────
    metrics_html = ""
    if sync["meta_available"]:
        acc = sync.get("val_accuracy")
        prec = sync.get("precision")
        rec = sync.get("recall")
        epochs = sync.get("epochs")

        def _fmt_pct(val: Any) -> tuple[str, str]:
            if val is None:
                return "—", "color:#71717a"
            try:
                f = float(val)
                if f <= 1.0:
                    f *= 100
                color = "#4ade80" if f >= 75 else "#facc15" if f >= 60 else "#f87171"
                return f"{f:.1f}%", f"color:{color}"
            except (TypeError, ValueError):
                return str(val), "color:#71717a"

        acc_str, acc_style = _fmt_pct(acc)
        prec_str, prec_style = _fmt_pct(prec)
        rec_str, rec_style = _fmt_pct(rec)

        epoch_line = (
            f'<div style="font-size:9px;color:#52525b;margin-top:4px;text-align:right">Epochs trained: {epochs}</div>'
            if epochs
            else ""
        )

        metrics_html = f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:6px;text-align:center">
                <div style="background:var(--bg-panel-inner);border-radius:4px;padding:4px">
                    <div style="font-size:9px;color:#52525b;text-transform:uppercase;letter-spacing:0.05em">Acc</div>
                    <div style="font-size:11px;font-family:monospace;font-weight:600;{acc_style}">{acc_str}</div>
                </div>
                <div style="background:var(--bg-panel-inner);border-radius:4px;padding:4px">
                    <div style="font-size:9px;color:#52525b;text-transform:uppercase;letter-spacing:0.05em">Prec</div>
                    <div style="font-size:11px;font-family:monospace;font-weight:600;{prec_style}">{prec_str}</div>
                </div>
                <div style="background:var(--bg-panel-inner);border-radius:4px;padding:4px">
                    <div style="font-size:9px;color:#52525b;text-transform:uppercase;letter-spacing:0.05em">Recall</div>
                    <div style="font-size:11px;font-family:monospace;font-weight:600;{rec_style}">{rec_str}</div>
                </div>
            </div>
            {epoch_line}
        """

    # ── Sync / version info ───────────────────────────────────────────
    sync_html = ""
    sync_parts: list[str] = []

    promoted_ago = sync.get("promoted_ago")
    last_sync_ago = sync.get("last_sync_ago")
    version = sync.get("version")

    if promoted_ago:
        sync_parts.append(
            f'<span class="t-text-muted">Trained:</span> <span class="t-text-secondary">{promoted_ago}</span>'
        )
    if last_sync_ago:
        sync_parts.append(
            f'<span class="t-text-muted">Synced:</span> <span class="t-text-secondary">{last_sync_ago}</span>'
        )
    if version:
        sync_parts.append(
            f'<span class="t-text-muted">Run:</span> <span class="font-mono t-text-faint">{str(version)[:16]}</span>'
        )

    onnx_badge = (
        '<span style="font-size:9px;color:#22d3ee;background:rgba(8,51,68,0.3);border:1px solid rgba(6,95,70,0.4);border-radius:3px;padding:0 4px">ONNX ✓</span>'
        if sync["onnx_available"]
        else '<span style="font-size:9px;color:#52525b;border:1px solid #27272a;border-radius:3px;padding:0 4px">ONNX —</span>'
    )

    if sync_parts:
        sync_html = f"""
            <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;font-size:10px;flex-wrap:wrap;gap:2px 8px">
                <div>{" · ".join(sync_parts)}</div>
                {onnx_badge}
            </div>
        """
    elif model["available"] and not sync["meta_available"]:
        sync_html = """
            <div style="font-size:10px;color:#52525b;margin-top:4px">
                No meta.json — run <span style="font-family:monospace;color:#71717a">sync_models.sh</span> to pull metadata
            </div>
        """

    # ── Sync model button (replaces old retrain button) ───────────────
    sync_btn = """
        <div style="margin-top:8px">
            <div style="font-size:9px;color:#52525b;margin-bottom:4px">
                Training runs on the <a href="https://github.com/nuniesmith/orb" target="_blank"
                style="color:#60a5fa;text-decoration:underline">orb repo</a> (GPU).
                Sync the latest champion model here for CPU inference.
            </div>
        </div>
    """

    return HTMLResponse(
        content=f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <h3 class="text-sm font-semibold t-text-muted">🧠 CNN MODEL</h3>
            <span class="text-[10px] t-text-faint">{model.get("total_models", 0)} checkpoints</span>
        </div>
        <div class="text-xs">
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
                {model_dot}
                <span class="t-text-secondary">{model_text}</span>
                {stale_badge}
            </div>
            {metrics_html}
            {sync_html}
        </div>
        {sync_btn}
    """
    )
