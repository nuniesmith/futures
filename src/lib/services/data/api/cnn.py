"""
CNN Model Management API Router
=================================
Provides endpoints for CNN model status and per-session gate management
for the web dashboard.

CNN training has moved to the orb repo (github.com/nuniesmith/orb).
Models are pulled into this repo via scripts/sync_models.sh.

Endpoints:
    GET  /cnn/status              — Current model info + metadata
    GET  /cnn/status/html         — HTML fragment for dashboard panel
    POST /cnn/retrain             — (deprecated) Returns redirect to orb repo
    GET  /cnn/retrain/status      — Poll status of last retrain attempt
    GET  /cnn/history             — Recent retrain audit log entries

Per-session CNN gate endpoints:
    GET  /cnn/gate                — Return gate state for all 9 sessions
    PUT  /cnn/gate/{session_key}  — Enable or disable the gate for one session
    DELETE /cnn/gate/{session_key}— Remove per-session override (revert to env-var)
    DELETE /cnn/gate              — Remove all overrides
    GET  /cnn/gate/html           — Dashboard HTML fragment for gate panel
"""

from __future__ import annotations

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
CHAMPION_PATH = MODEL_DIR / "breakout_cnn_best.pt"
META_PATH = MODEL_DIR / "breakout_cnn_best_meta.json"

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
    """Dataset generation has moved to the orb repo."""
    return {
        "note": "Dataset generation has moved to the orb repo (github.com/nuniesmith/orb)",
        "total_samples": 0,
    }


def _get_sync_status() -> dict[str, Any]:
    """Return sync metadata derived from the meta.json sidecar and model files.

    Combines:
      - promoted_at / val_accuracy / precision / recall from meta.json
      - last_sync_ago from meta.json file mtime (when sync_models.sh last ran)
      - onnx_available flag
      - stale flag (> 26 h since promotion)
    """
    result: dict[str, Any] = {
        "meta_available": False,
        "promoted_at": None,
        "promoted_ago": None,
        "last_sync_ago": None,
        "val_accuracy": None,
        "precision": None,
        "recall": None,
        "epochs": None,
        "stale": False,
        "onnx_available": False,
        "version": None,
    }

    result["onnx_available"] = (MODEL_DIR / "breakout_cnn_best.onnx").is_file()

    if not META_PATH.is_file():
        return result

    # last_sync_ago — file mtime reflects when sync_models.sh last wrote it
    try:
        sync_dt = datetime.fromtimestamp(META_PATH.stat().st_mtime, tz=_EST)
        now_et = _now_et()
        sync_delta = now_et - sync_dt
        sync_hours = sync_delta.total_seconds() / 3600
        if sync_hours < 1:
            result["last_sync_ago"] = f"{int(sync_delta.total_seconds() / 60)}m ago"
        elif sync_hours < 24:
            result["last_sync_ago"] = f"{sync_hours:.1f}h ago"
        else:
            result["last_sync_ago"] = f"{sync_delta.days}d ago"
    except Exception:
        pass

    try:
        meta = _get_meta_info()
        if not meta:
            return result

        result["meta_available"] = True

        # Version string — try common field names from the orb trainer
        for vkey in ("version", "model_version", "run_id", "run"):
            if vkey in meta:
                result["version"] = str(meta[vkey])
                break

        # Accuracy metrics
        for src, dst in (
            ("val_accuracy", "val_accuracy"),
            ("accuracy", "val_accuracy"),
            ("val_precision", "precision"),
            ("precision", "precision"),
            ("val_recall", "recall"),
            ("recall", "recall"),
            ("epochs", "epochs"),
        ):
            if src in meta and result[dst] is None:
                result[dst] = meta[src]

        # promoted_at — when the model was promoted in the orb repo
        promoted_str = meta.get("promoted_at") or meta.get("trained_at")
        if promoted_str:
            try:
                promoted_at = datetime.fromisoformat(str(promoted_str))
                if promoted_at.tzinfo is None:
                    promoted_at = promoted_at.replace(tzinfo=_EST)
                result["promoted_at"] = promoted_at.isoformat()
                now_et = _now_et()
                delta = now_et - promoted_at
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    result["promoted_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
                elif hours < 24:
                    result["promoted_ago"] = f"{hours:.1f}h ago"
                else:
                    result["promoted_ago"] = f"{delta.days}d ago"
                result["stale"] = hours > 26
            except Exception:
                pass

    except Exception as exc:
        logger.debug("_get_sync_status: %s", exc)

    return result


def _get_meta_info() -> dict[str, Any]:
    """Read champion model metadata from the sidecar JSON (synced from orb repo)."""
    if META_PATH.is_file():
        try:
            with open(META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_training_history_summary() -> dict[str, Any]:
    """Training history lives in the orb repo — return metadata from sidecar JSON if available."""
    summary: dict[str, Any] = {"available": False}
    meta = _get_meta_info()
    if meta:
        summary["available"] = True
        summary["note"] = "Training history lives in the orb repo. Metadata from last sync shown."
        for key in ("val_accuracy", "val_precision", "val_recall", "epochs", "trained_at"):
            if key in meta:
                summary[key] = meta[key]
    return summary


def _get_recent_audit_entries(limit: int = 10) -> list[dict[str, Any]]:
    """Audit log lives in the orb repo — return empty list."""
    return []


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
    """CNN training has moved to the orb repo.

    Returns an informational response directing users to the orb repo's
    GPU trainer instead.
    """
    return {
        "status": "moved",
        "message": (
            "CNN training has moved to the orb repo (github.com/nuniesmith/orb). "
            "Use the GPU trainer there, then run: bash scripts/sync_models.sh"
        ),
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.get("/cnn/status")
def cnn_status():
    """Return CNN model status and metadata."""
    model = _get_model_info()
    meta = _get_meta_info()

    return {
        "model": model,
        "meta": meta,
        "training_note": "CNN training lives in the orb repo (github.com/nuniesmith/orb). Pull latest model with: bash scripts/sync_models.sh",
        "timestamp": _now_et().isoformat(),
    }


@router.post("/cnn/retrain")
def trigger_retrain(
    request: Request,
    session: str = Query("both", description="(deprecated)"),
    skip_dataset: bool = Query(False, description="(deprecated)"),
    epochs: int | None = Query(None, description="(deprecated)"),
    batch_size: int | None = Query(None, description="(deprecated)"),
):
    """(Deprecated) CNN training has moved to the orb repo.

    Use the GPU trainer in github.com/nuniesmith/orb instead, then pull
    the trained model with: bash scripts/sync_models.sh
    """
    logger.info("CNN retrain request received — training has moved to orb repo")

    result = _start_retrain()

    # If called from HTMX, return the updated CNN panel HTML
    if request.headers.get("HX-Request"):
        return cnn_status_html()

    return result


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
                    "override": override,  # None = no Redis key set
                    "effective": effective,  # what the engine actually uses
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
    - ``effective``: the value the engine will actually use (override if set,
      else ``ORB_CNN_GATE`` env var).
    - ``source``: ``"redis"`` or ``"env"``.
    """
    return _get_gates_payload()


@router.put("/cnn/gate/{session_key}")
def set_cnn_gate_endpoint(session_key: str, enabled: bool = True):
    """Enable or disable the CNN hard gate for a single session.

    Args:
        session_key: One of: cme, sydney, tokyo, shanghai, frankfurt, london,
                     london_ny, us, cme_settle.
        enabled: ``true`` (default) to enable; ``false`` to disable.

    The value is stored in Redis at ``engine:config:cnn_gate:{session_key}``
    and persists for 30 days.  The engine picks it up on the next ORB check
    with no restart required.
    """
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
            raise HTTPException(status_code=503, detail="Redis unavailable — could not persist gate setting")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "session_key": sk,
        "label": _SESSION_LABELS[sk],
        "enabled": enabled,
        "stored": True,
        "message": (
            f"CNN gate {'ENABLED' if enabled else 'DISABLED'} for session '{sk}' "
            f"— takes effect on next ORB check (no restart needed)"
        ),
    }


@router.delete("/cnn/gate/{session_key}")
def reset_cnn_gate_endpoint(session_key: str):
    """Remove the Redis override for *session_key*.

    After this call the session reverts to the ``ORB_CNN_GATE`` env-var default.
    """
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
        "message": f"Override removed for session '{sk}' — now using ORB_CNN_GATE env var",
    }


@router.delete("/cnn/gate")
def reset_all_cnn_gates_endpoint():
    """Remove all per-session Redis overrides.

    All sessions revert to the ``ORB_CNN_GATE`` env-var default.
    """
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
    """Return a dashboard HTML fragment for the per-session CNN gate panel.

    Designed to be loaded into ``#cnn-gate-panel`` via HTMX polling every
    30 seconds so the dashboard stays in sync with Redis state changes.
    """
    data = _get_gates_payload()
    sessions: list[dict[str, Any]] = data.get("sessions", [])
    global_env: bool = data.get("global_env", False)
    redis_ok: bool = data.get("redis_available", False)

    if not redis_ok:
        return HTMLResponse(
            content="""
            <div class="text-[10px] text-red-400 flex items-center gap-1">
                <span>⚠</span>
                <span>Redis unavailable — gate state unknown</span>
            </div>
            """
        )

    rows_html = ""
    for s in sessions:
        sk: str = s["key"]
        label: str = s["label"]
        effective: bool = s["effective"]
        source: str = s["source"]
        is_overnight: bool = s["is_overnight"]

        # Gate toggle button
        if effective:
            dot = '<span class="w-2 h-2 rounded-full bg-green-500 inline-block"></span>'
            gate_text = "ON"
            gate_color = "text-green-400"
            toggle_title = f"Disable CNN gate for {sk}"
            toggle_action = (
                f"hx-delete='/cnn/gate/{sk}'" if source == "redis" else (f"hx-put='/cnn/gate/{sk}?enabled=false'")
            )
        else:
            dot = '<span class="w-2 h-2 rounded-full bg-zinc-600 inline-block"></span>'
            gate_text = "off"
            gate_color = "text-zinc-500"
            toggle_title = f"Enable CNN gate for {sk}"
            toggle_action = f"hx-put='/cnn/gate/{sk}?enabled=true'"

        source_badge = (
            '<span class="text-[9px] text-blue-400 bg-blue-900/30 border border-blue-700/40 rounded px-1">redis</span>'
            if source == "redis"
            else '<span class="text-[9px] text-zinc-600">env</span>'
        )
        overnight_badge = '<span class="text-[9px] text-yellow-600">🌙</span>' if is_overnight else ""

        rows_html += f"""
        <div class="flex items-center justify-between gap-2 py-0.5">
            <div class="flex items-center gap-1.5 min-w-0">
                {dot}
                {overnight_badge}
                <span class="text-[10px] text-zinc-300 truncate" title="{label}">{label}</span>
                {source_badge}
            </div>
            <button
                {toggle_action}
                hx-target="#cnn-gate-panel"
                hx-swap="innerHTML"
                title="{toggle_title}"
                class="text-[10px] {gate_color} hover:text-white px-1.5 py-0.5 rounded
                       bg-zinc-800 hover:bg-zinc-700 border border-zinc-700/60
                       transition-colors duration-150 shrink-0 font-mono w-8 text-center">
                {gate_text}
            </button>
        </div>
        """

    env_badge = '<span class="text-green-400">ON</span>' if global_env else '<span class="text-zinc-500">off</span>'

    # Bulk action buttons
    bulk_html = """
        <div class="flex gap-1 mt-2 pt-1.5 border-t border-zinc-800">
            <button hx-put="/cnn/gate/cme?enabled=true"
                    hx-include="[name=_dummy]"
                    hx-target="#cnn-gate-panel"
                    hx-swap="innerHTML"
                    hx-trigger="click"
                    onclick="['cme','sydney','tokyo','shanghai'].forEach(s=>{
                        htmx.ajax('PUT','/cnn/gate/'+s+'?enabled=true',{target:'#cnn-gate-panel',swap:'innerHTML'})
                    });return false;"
                    class="flex-1 text-[10px] px-1.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700
                           text-zinc-400 hover:text-zinc-200 border border-zinc-700/60
                           transition-colors duration-150"
                    title="Enable CNN gate for all 4 overnight sessions (CME/Sydney/Tokyo/Shanghai)">
                🌙 Enable overnight
            </button>
            <button hx-delete="/cnn/gate"
                    hx-target="#cnn-gate-panel"
                    hx-swap="innerHTML"
                    hx-confirm="Remove all Redis overrides? All sessions will revert to the ORB_CNN_GATE env var."
                    class="text-[10px] px-1.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700
                           text-zinc-400 hover:text-red-300 border border-zinc-700/60
                           transition-colors duration-150"
                    title="Remove all Redis overrides">
                ↺ Reset all
            </button>
        </div>
    """

    return HTMLResponse(
        content=f"""
        <div class="flex items-center justify-between mb-1.5">
            <h3 class="text-sm font-semibold text-zinc-400">🔒 CNN GATE</h3>
            <span class="text-zinc-600 text-[10px]">env default: {env_badge}</span>
        </div>
        <div class="space-y-0 text-xs">
            {rows_html}
        </div>
        {bulk_html}
        """
    )


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
    sync = _get_sync_status()
    job = _get_retrain_job_status()

    # ── Model availability dot + headline ────────────────────────────
    if model["available"]:
        if sync["stale"]:
            model_dot = '<span class="text-yellow-500">●</span>'
            stale_badge = '<span class="text-[9px] text-yellow-500 bg-yellow-900/30 border border-yellow-700/40 rounded px-1 ml-1">stale</span>'
        else:
            model_dot = '<span class="text-green-500">●</span>'
            stale_badge = ""
        model_text = f"Model ready ({model['size_mb']} MB)"
        model_age = model.get("modified_ago", "")
        if model_age:
            model_text += f" · {model_age}"
    else:
        model_dot = '<span class="text-red-500">●</span>'
        stale_badge = ""
        model_text = "No model found — run: bash scripts/sync_models.sh"

    # ── Accuracy / metrics row (from meta.json) ───────────────────────
    metrics_html = ""
    if sync["meta_available"]:
        acc = sync.get("val_accuracy")
        prec = sync.get("precision")
        rec = sync.get("recall")
        epochs = sync.get("epochs")

        acc_str = "—"
        acc_color = "text-zinc-500"
        if acc is not None:
            try:
                acc_f = float(acc)
                # orb trainer stores accuracy as a fraction (0–1) or percentage (0–100)
                if acc_f <= 1.0:
                    acc_f *= 100
                acc_str = f"{acc_f:.1f}%"
                acc_color = "text-green-400" if acc_f >= 75 else "text-yellow-400" if acc_f >= 60 else "text-red-400"
            except (TypeError, ValueError):
                acc_str = str(acc)

        prec_str = "—"
        if prec is not None:
            try:
                p = float(prec)
                prec_str = f"{p * 100:.1f}%" if p <= 1.0 else f"{p:.1f}%"
            except (TypeError, ValueError):
                prec_str = str(prec)

        rec_str = "—"
        if rec is not None:
            try:
                r = float(rec)
                rec_str = f"{r * 100:.1f}%" if r <= 1.0 else f"{r:.1f}%"
            except (TypeError, ValueError):
                rec_str = str(rec)

        epoch_line = f'<div class="text-[9px] text-zinc-600 mt-1 text-right">{epochs} epochs</div>' if epochs else ""

        metrics_html = f"""
            <div class="grid grid-cols-3 gap-1 mt-1.5 text-center">
                <div class="bg-zinc-800/60 rounded px-1 py-1">
                    <div class="text-[9px] text-zinc-600 uppercase tracking-wide">Acc</div>
                    <div class="text-[11px] font-mono font-semibold {acc_color}">{acc_str}</div>
                </div>
                <div class="bg-zinc-800/60 rounded px-1 py-1">
                    <div class="text-[9px] text-zinc-600 uppercase tracking-wide">Prec</div>
                    <div class="text-[11px] font-mono font-semibold text-zinc-300">{prec_str}</div>
                </div>
                <div class="bg-zinc-800/60 rounded px-1 py-1">
                    <div class="text-[9px] text-zinc-600 uppercase tracking-wide">Recall</div>
                    <div class="text-[11px] font-mono font-semibold text-zinc-300">{rec_str}</div>
                </div>
            </div>
            {epoch_line}
        """

    # ── Sync / version row ────────────────────────────────────────────
    sync_html = ""
    sync_parts: list[str] = []

    promoted_ago = sync.get("promoted_ago")
    last_sync_ago = sync.get("last_sync_ago")
    version = sync.get("version")
    onnx_available = sync.get("onnx_available", False)

    if promoted_ago:
        sync_parts.append(
            f'<span class="text-zinc-500">Trained:</span> <span class="text-zinc-300">{promoted_ago}</span>'
        )
    if last_sync_ago:
        sync_parts.append(
            f'<span class="text-zinc-500">Synced:</span> <span class="text-zinc-300">{last_sync_ago}</span>'
        )
    if version:
        sync_parts.append(
            f'<span class="text-zinc-500">Ver:</span> <span class="font-mono text-zinc-400">{str(version)[:12]}</span>'
        )

    onnx_badge = (
        '<span class="text-[9px] text-cyan-500 bg-cyan-900/30 border border-cyan-700/40 rounded px-1">ONNX ✓</span>'
        if onnx_available
        else '<span class="text-[9px] text-zinc-700 border border-zinc-800 rounded px-1">ONNX —</span>'
    )

    if sync_parts or onnx_available:
        sync_html = f"""
            <div class="flex items-center justify-between mt-1.5 text-[10px] flex-wrap gap-x-2 gap-y-0.5">
                <div class="space-x-2">{" · ".join(sync_parts)}</div>
                {onnx_badge}
            </div>
        """
    elif model["available"] and not sync["meta_available"]:
        sync_html = """
            <div class="text-[10px] text-zinc-600 mt-1">
                No meta.json — run <span class="font-mono text-zinc-500">sync_models.sh</span> to pull metadata
            </div>
        """

    # ── Retrain job status ────────────────────────────────────────────
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
            <div class="flex items-center gap-1.5 flex-wrap">
                {model_dot}
                <span class="text-zinc-300">{model_text}</span>
                {stale_badge}
            </div>
            {metrics_html}
            {sync_html}
        </div>
        {retrain_html}
        {retrain_btn}
    """
    )
