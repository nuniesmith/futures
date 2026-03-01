"""
NT8 Deploy & Health API Router
================================
Serves a downloadable .bat installer that pulls the latest NinjaTrader 8
C# source files from the GitHub repo and deploys them to the correct
NT8 Custom directories on Windows.

Also provides live health indicators for the Bridge strategy and Ruby
indicator connections, plus a compact toolbar/dropdown for the dashboard
header area.

Endpoints:
    GET  /api/nt8/installer      — Download the deploy-nt8.bat installer
    GET  /api/nt8/panel/html     — Compact header toolbar HTML fragment
    GET  /api/nt8/health/html    — Health status bar HTML fragment (polled)
    GET  /api/nt8/health         — Health status JSON
    POST /api/nt8/deploy         — Trigger local deployment (WSL/server-side)
"""

import json
import logging
import textwrap
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

logger = logging.getLogger("api.nt8_deploy")

router = APIRouter(tags=["NT8 Deploy"])

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_REPO = "nuniesmith/futures"
_GITHUB_BRANCH = "main"
_GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{_GITHUB_REPO}/{_GITHUB_BRANCH}"

# Mapping: source path in repo → destination relative to NT8 Custom dir
_FILE_MAP = {
    "src/ninjatrader/Bridge.cs": "Strategies\\Bridge.cs",
    "src/ninjatrader/Ruby.cs": "Indicators\\Ruby.cs",
    "src/ninjatrader/SignalBus.cs": "SignalBus.cs",
}

# Default NT8 Custom directory on Windows
_DEFAULT_NT8_CUSTOM = r"C:\Users\jordan\Documents\NinjaTrader 8\bin\Custom"


# ---------------------------------------------------------------------------
# Health helpers — read Bridge heartbeat + status from cache
# ---------------------------------------------------------------------------


def _get_heartbeat() -> dict[str, Any] | None:
    """Read the latest Bridge heartbeat from cache."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("futures:bridge_heartbeat:current")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _get_bridge_status_from_cache() -> dict[str, Any] | None:
    """Read cached Bridge /status response (set by positions router probes)."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("futures:bridge_status:latest")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _get_positions_data() -> dict[str, Any] | None:
    """Read cached positions payload (contains bridge_version, rubyAttached etc.)."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("futures:live_positions:current")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _compute_health() -> dict[str, Any]:
    """Compute NT8 health status from all available cache sources.

    Returns a dict with:
        bridge_connected: bool   — heartbeat received within TTL
        bridge_state: str        — "Realtime", "Historical", "disconnected", etc.
        bridge_version: str      — e.g. "2.0"
        bridge_account: str      — e.g. "Sim101"
        bridge_age_seconds: float — seconds since last heartbeat
        ruby_attached: bool      — whether Ruby indicator is loaded on Bridge
        signalbus_active: bool   — whether SignalBus consumer is registered
        signalbus_pending: int   — signals waiting in queue
        positions_count: int     — open positions
        risk_blocked: bool       — risk enforcement blocking
        last_heartbeat: str      — ISO timestamp of last heartbeat
    """
    heartbeat = _get_heartbeat()
    bridge_status = _get_bridge_status_from_cache()
    positions = _get_positions_data()

    result: dict[str, Any] = {
        "bridge_connected": False,
        "bridge_state": "disconnected",
        "bridge_version": "",
        "bridge_account": "",
        "bridge_age_seconds": -1,
        "ruby_attached": False,
        "signalbus_active": False,
        "signalbus_pending": 0,
        "positions_count": 0,
        "risk_blocked": False,
        "last_heartbeat": None,
    }

    # --- From heartbeat (primary freshness signal) ---
    if heartbeat:
        received_at = heartbeat.get("received_at", "")
        result["bridge_account"] = heartbeat.get("account", "")
        result["bridge_state"] = heartbeat.get("state", "unknown")
        result["bridge_version"] = heartbeat.get("bridge_version", "")
        result["positions_count"] = heartbeat.get("positions", 0)
        result["risk_blocked"] = heartbeat.get("riskBlocked", False)
        result["last_heartbeat"] = received_at

        if received_at:
            try:
                dt = datetime.fromisoformat(received_at)
                age = (datetime.now(tz=_EST) - dt).total_seconds()
                result["bridge_age_seconds"] = round(age, 1)
                # Bridge sends heartbeat every ~15s; consider alive if < 60s
                result["bridge_connected"] = age < 60
            except Exception:
                pass

    # --- From Bridge /status probe (richer data: rubyAttached, signalBus) ---
    if bridge_status:
        result["ruby_attached"] = bridge_status.get("rubyAttached", False)
        result["signalbus_active"] = bridge_status.get("signalBusActive", False)
        result["signalbus_pending"] = bridge_status.get("signalBusPending", 0)
        # Fill in fields the heartbeat may not carry
        if not result["bridge_version"]:
            result["bridge_version"] = bridge_status.get("bridge_version", "")
        if not result["bridge_account"]:
            result["bridge_account"] = bridge_status.get("account", "")

    # --- From positions data (fallback for bridge_version, account) ---
    if positions:
        if not result["bridge_version"]:
            result["bridge_version"] = positions.get("bridge_version", "")
        if not result["bridge_account"]:
            result["bridge_account"] = positions.get("account", "")

    return result


# ---------------------------------------------------------------------------
# .bat installer generator
# ---------------------------------------------------------------------------


def _generate_bat_installer() -> str:
    """Generate a Windows .bat file that downloads CS files and deploys them."""

    download_cmds = ""
    file_list_display = ""
    success_checks = ""
    file_count = len(_FILE_MAP)

    for i, (repo_path, nt8_rel_path) in enumerate(_FILE_MAP.items(), 1):
        raw_url = f"{_GITHUB_RAW_BASE}/{repo_path}"
        filename = repo_path.split("/")[-1]
        dest_subdir = nt8_rel_path.rsplit("\\", 1)[0] if "\\" in nt8_rel_path else ""

        file_list_display += f"echo     {filename:20s} -^> {nt8_rel_path}\n"

        download_cmds += textwrap.dedent(f"""\
            echo.
            echo   [{i}/{file_count}] Downloading {filename}...
            curl -sS -L -f -o "%TEMP_DIR%\\{filename}" "{raw_url}"
            if errorlevel 1 (
                echo   [FAIL] Could not download {filename}
                set /a ERRORS+=1
            ) else (
                echo   [OK]   Downloaded {filename}
            )
        """)

        if dest_subdir:
            download_cmds += textwrap.dedent(f"""\
                if not exist "%NT8_CUSTOM%\\{dest_subdir}" mkdir "%NT8_CUSTOM%\\{dest_subdir}"
            """)

        download_cmds += textwrap.dedent(f"""\
            if exist "%TEMP_DIR%\\{filename}" (
                copy /Y "%TEMP_DIR%\\{filename}" "%NT8_CUSTOM%\\{nt8_rel_path}" >nul 2>&1
                if errorlevel 1 (
                    echo   [FAIL] Could not copy {filename} to {nt8_rel_path}
                    set /a ERRORS+=1
                ) else (
                    echo   [OK]   {filename} -^> {nt8_rel_path}
                )
            )
        """)

        success_checks += f'if exist "%NT8_CUSTOM%\\{nt8_rel_path}" set /a DEPLOYED+=1\n'

    bat_content = textwrap.dedent(f"""\
        @echo off
        setlocal EnableDelayedExpansion

        :: =====================================================================
        :: NinjaTrader 8 — Deploy CS Files from GitHub
        :: =====================================================================
        :: Generated by Futures Trading Co-Pilot
        ::
        :: This script downloads the latest NinjaTrader C# source files from
        :: the GitHub repository and copies them to the correct NT8 Custom
        :: directories. After running, open NinjaTrader 8 and compile.
        ::
        :: Requirements: curl (included in Windows 10+)
        :: =====================================================================

        title NinjaTrader 8 — Deploy CS Files

        echo.
        echo  ================================================================
        echo   NinjaTrader 8 — Deploy CS Files from GitHub
        echo  ================================================================
        echo.
        echo  Repository: {_GITHUB_REPO} (branch: {_GITHUB_BRANCH})
        echo.
        echo  Files to deploy:
        {file_list_display}
        echo.

        :: ── Configuration ────────────────────────────────────────────────────

        set "NT8_CUSTOM={_DEFAULT_NT8_CUSTOM}"
        set "TEMP_DIR=%TEMP%\\nt8_deploy_%RANDOM%"
        set /a ERRORS=0
        set /a DEPLOYED=0

        :: Allow override via command-line argument
        if not "%~1"=="" set "NT8_CUSTOM=%~1"

        echo  Target: %NT8_CUSTOM%
        echo.

        :: ── Validate NT8 directory ───────────────────────────────────────────

        if not exist "%NT8_CUSTOM%" (
            echo  [ERROR] NT8 Custom directory not found:
            echo          %NT8_CUSTOM%
            echo.
            echo  Make sure NinjaTrader 8 is installed, or pass the path as an argument:
            echo      deploy-nt8.bat "C:\\path\\to\\NinjaTrader 8\\bin\\Custom"
            echo.
            goto :error_exit
        )

        :: ── Check for curl ───────────────────────────────────────────────────

        where curl >nul 2>&1
        if errorlevel 1 (
            echo  [ERROR] curl not found. Please install curl or use Windows 10+.
            goto :error_exit
        )

        :: ── Create temp directory ────────────────────────────────────────────

        mkdir "%TEMP_DIR%" >nul 2>&1

        :: ── Download and deploy files ────────────────────────────────────────

        echo  Downloading from GitHub...
        {download_cmds}

        :: ── Verify deployment ────────────────────────────────────────────────

        {success_checks}

        :: ── Clean up temp directory ──────────────────────────────────────────

        rmdir /S /Q "%TEMP_DIR%" >nul 2>&1

        :: ── Summary ──────────────────────────────────────────────────────────

        echo.
        echo  ================================================================
        if !ERRORS! EQU 0 (
            echo   SUCCESS: All {file_count} files deployed to NT8 Custom directory.
        ) else (
            echo   COMPLETED with !ERRORS! error(s). !DEPLOYED!/{file_count} files deployed.
        )
        echo  ================================================================
        echo.
        echo  Next steps:
        echo    1. Open NinjaTrader 8
        echo    2. Tools -^> NinjaScript Editor
        echo    3. Right-click -^> Compile
        echo.

        if !ERRORS! GTR 0 goto :error_exit

        echo  Press any key to exit...
        pause >nul
        exit /b 0

        :error_exit
        echo.
        echo  Press any key to exit...
        pause >nul
        exit /b 1
    """)

    return bat_content


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------


def _render_health_bar(health: dict[str, Any]) -> str:
    """Render the NT8 health indicators as a compact HTML fragment.

    Shows colored dots for Bridge and Ruby connection status, plus
    a tooltip with details. Designed to sit in the header toolbar.
    """
    bridge_ok = health.get("bridge_connected", False)
    ruby_ok = health.get("ruby_attached", False)
    signalbus_ok = health.get("signalbus_active", False)
    risk_blocked = health.get("risk_blocked", False)
    age = health.get("bridge_age_seconds", -1)
    state = health.get("bridge_state", "disconnected")
    account = health.get("bridge_account", "")
    version = health.get("bridge_version", "")
    positions = health.get("positions_count", 0)
    pending = health.get("signalbus_pending", 0)

    # Bridge dot
    if bridge_ok:
        bridge_color = "bg-green-500"
        bridge_ring = "ring-green-500/30"
        bridge_title = f"Bridge: Connected ({state})"
        if account:
            bridge_title += f" — {account}"
        if age >= 0:
            bridge_title += f" — {age:.0f}s ago"
    else:
        bridge_color = "bg-red-500"
        bridge_ring = "ring-red-500/30"
        bridge_title = "Bridge: Disconnected"
        if age >= 0:
            bridge_title += f" — last seen {age:.0f}s ago"

    # Ruby dot
    if ruby_ok:
        ruby_color = "bg-green-500"
        ruby_ring = "ring-green-500/30"
        ruby_title = "Ruby: Attached to Bridge"
    elif bridge_ok:
        ruby_color = "bg-yellow-500"
        ruby_ring = "ring-yellow-500/30"
        ruby_title = "Ruby: Not attached (Bridge running without Ruby indicator)"
    else:
        ruby_color = "bg-zinc-600"
        ruby_ring = "ring-zinc-600/30"
        ruby_title = "Ruby: Unknown (Bridge not connected)"

    # SignalBus dot
    if signalbus_ok:
        sb_color = "bg-green-500"
        sb_ring = "ring-green-500/30"
        sb_title = f"SignalBus: Active ({pending} pending)"
    elif bridge_ok:
        sb_color = "bg-yellow-500"
        sb_ring = "ring-yellow-500/30"
        sb_title = "SignalBus: Inactive (no consumer)"
    else:
        sb_color = "bg-zinc-600"
        sb_ring = "ring-zinc-600/30"
        sb_title = "SignalBus: Unknown"

    # Risk badge
    risk_html = ""
    if risk_blocked:
        risk_html = """
            <span class="ml-1 px-1.5 py-0.5 bg-red-900/60 border border-red-700 rounded
                         text-[10px] text-red-400 font-semibold uppercase tracking-wide"
                  title="Risk enforcement is blocking new trades">
                RISK BLOCK
            </span>
        """

    # Positions badge (only if > 0)
    pos_html = ""
    if positions > 0 and bridge_ok:
        pos_html = f"""
            <span class="ml-1 px-1.5 py-0.5 bg-zinc-800 border border-zinc-700 rounded
                         text-[10px] text-zinc-300 font-mono"
                  title="{positions} open position(s)">
                {positions} pos
            </span>
        """

    # Version tag
    ver_html = ""
    if version and bridge_ok:
        ver_html = f"""
            <span class="text-[10px] text-zinc-600 ml-1" title="Bridge version {version}">v{version}</span>
        """

    return f"""
    <div class="flex items-center gap-3">
        <!-- Bridge Status -->
        <div class="flex items-center gap-1.5 cursor-default" title="{bridge_title}">
            <span class="relative flex h-2.5 w-2.5">
                {"<span class='animate-ping absolute inline-flex h-full w-full rounded-full " + bridge_color + " opacity-40'></span>" if bridge_ok else ""}
                <span class="relative inline-flex rounded-full h-2.5 w-2.5 {bridge_color} ring-2 {bridge_ring}"></span>
            </span>
            <span class="text-[11px] {"text-zinc-300" if bridge_ok else "text-zinc-500"}">Bridge</span>
            {ver_html}
        </div>

        <!-- Ruby Status -->
        <div class="flex items-center gap-1.5 cursor-default" title="{ruby_title}">
            <span class="relative flex h-2.5 w-2.5">
                {"<span class='animate-ping absolute inline-flex h-full w-full rounded-full " + ruby_color + " opacity-40'></span>" if ruby_ok else ""}
                <span class="relative inline-flex rounded-full h-2.5 w-2.5 {ruby_color} ring-2 {ruby_ring}"></span>
            </span>
            <span class="text-[11px] {"text-zinc-300" if ruby_ok else "text-zinc-500"}">Ruby</span>
        </div>

        <!-- SignalBus Status -->
        <div class="flex items-center gap-1.5 cursor-default" title="{sb_title}">
            <span class="relative flex h-2.5 w-2.5">
                <span class="relative inline-flex rounded-full h-2.5 w-2.5 {sb_color} ring-2 {sb_ring}"></span>
            </span>
            <span class="text-[11px] {"text-zinc-300" if signalbus_ok else "text-zinc-500"}">Bus</span>
        </div>

        {pos_html}
        {risk_html}
    </div>
    """


def _render_toolbar_dropdown() -> str:
    """Render the NT8 tools dropdown for the header area.

    A small icon button that reveals deploy options on click.
    """
    return """
    <div class="relative" id="nt8-toolbar">
        <!-- Trigger Button -->
        <button onclick="document.getElementById('nt8-dropdown').classList.toggle('hidden')"
                class="flex items-center gap-1.5 px-2.5 py-1.5 bg-zinc-800/80 hover:bg-zinc-700
                       rounded-lg text-xs text-zinc-400 hover:text-zinc-200
                       transition-all duration-200 border border-zinc-700/50 hover:border-zinc-600"
                title="NinjaTrader 8 Tools">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none"
                 viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round"
                      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span>NT8</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 opacity-50" fill="none"
                 viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
        </button>

        <!-- Dropdown Panel -->
        <div id="nt8-dropdown"
             class="hidden absolute right-0 top-full mt-2 w-72 bg-zinc-900 border border-zinc-700
                    rounded-lg shadow-2xl shadow-black/50 z-50 p-4">

            <!-- Header -->
            <div class="flex items-center justify-between mb-3">
                <h3 class="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                    NT8 Deploy Tools
                </h3>
                <button onclick="document.getElementById('nt8-dropdown').classList.add('hidden')"
                        class="text-zinc-500 hover:text-zinc-300 transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none"
                         viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            <!-- File mapping -->
            <div class="mb-3 text-[10px] text-zinc-600 space-y-0.5 bg-zinc-950/50 rounded p-2 border border-zinc-800">
                <div class="flex justify-between">
                    <span>📄 Bridge.cs</span><span class="text-zinc-500">→ Strategies/</span>
                </div>
                <div class="flex justify-between">
                    <span>📄 Ruby.cs</span><span class="text-zinc-500">→ Indicators/</span>
                </div>
                <div class="flex justify-between">
                    <span>📄 SignalBus.cs</span><span class="text-zinc-500">→ Custom/</span>
                </div>
            </div>

            <!-- Actions -->
            <div class="space-y-2">
                <a href="/api/nt8/installer"
                   download="deploy-nt8.bat"
                   class="flex items-center justify-center gap-2 w-full px-3 py-2
                          bg-indigo-600 hover:bg-indigo-500 rounded-md text-xs text-white
                          font-semibold transition-colors duration-200 border border-indigo-500/50">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none"
                         viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round"
                              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Download deploy-nt8.bat
                </a>

                <div id="nt8-deploy-status"></div>

                <button hx-post="/api/nt8/deploy"
                        hx-target="#nt8-deploy-status"
                        hx-swap="innerHTML"
                        hx-indicator="#nt8-deploy-spinner"
                        class="flex items-center justify-center gap-2 w-full px-3 py-2
                               bg-zinc-800 hover:bg-zinc-700 rounded-md text-xs text-zinc-300
                               transition-colors duration-200 border border-zinc-700">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none"
                         viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round"
                              d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Deploy via WSL
                    <span id="nt8-deploy-spinner" class="htmx-indicator text-zinc-500">⏳</span>
                </button>
            </div>

            <!-- Source info -->
            <div class="mt-3 pt-2 border-t border-zinc-800 text-[10px] text-zinc-600 text-center">
                Source: github.com/nuniesmith/futures (main)
            </div>
        </div>
    </div>

    <!-- Close dropdown on outside click -->
    <script>
        document.addEventListener('click', function(e) {
            var toolbar = document.getElementById('nt8-toolbar');
            var dropdown = document.getElementById('nt8-dropdown');
            if (toolbar && dropdown && !toolbar.contains(e.target)) {
                dropdown.classList.add('hidden');
            }
        });
    </script>
    """


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/nt8/installer")
def download_nt8_installer():
    """Download the deploy-nt8.bat installer file."""
    bat_content = _generate_bat_installer()
    return PlainTextResponse(
        content=bat_content,
        media_type="application/x-bat",
        headers={
            "Content-Disposition": 'attachment; filename="deploy-nt8.bat"',
        },
    )


@router.get("/api/nt8/panel/html", response_class=HTMLResponse)
def get_nt8_panel():
    """Return the NT8 toolbar dropdown as an HTML fragment."""
    return HTMLResponse(content=_render_toolbar_dropdown())


@router.get("/api/nt8/health/html", response_class=HTMLResponse)
def get_nt8_health_html():
    """Return NT8 health indicators as an HTML fragment (polled by HTMX)."""
    health = _compute_health()
    return HTMLResponse(content=_render_health_bar(health))


@router.get("/api/nt8/health")
def get_nt8_health():
    """Return NT8 health status as JSON."""
    health = _compute_health()
    return JSONResponse(content=health)


@router.post("/api/nt8/deploy", response_class=HTMLResponse)
def trigger_nt8_deploy():
    """Trigger server-side NT8 deployment via the deploy_nt8.sh script.

    This works when the data service is running under WSL or Linux with
    access to the NT8 Custom directory via /mnt/c/... path.
    """
    import os
    import subprocess

    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "deploy_nt8.sh")
    script_path = os.path.normpath(script_path)

    if not os.path.isfile(script_path):
        logger.warning("deploy_nt8.sh not found at %s", script_path)
        return HTMLResponse(
            content="""
            <div class="text-red-400 text-xs mt-1 p-2 bg-red-900/20 rounded border border-red-800">
                ✗ deploy_nt8.sh not found. Use the .bat download instead.
            </div>
            """
        )

    try:
        result = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(script_path),
        )

        if result.returncode == 0:
            logger.info("NT8 deploy succeeded via WSL script")
            return HTMLResponse(
                content="""
                <div class="text-green-400 text-xs mt-1 p-2 bg-green-900/20 rounded border border-green-800">
                    ✓ All CS files deployed to NT8 Custom directory.
                    <br><span class="text-zinc-500">Open NT8 → Tools → NinjaScript Editor → Compile</span>
                </div>
                """
            )
        else:
            stderr_short = (result.stderr or "unknown error")[-200:]
            logger.error("NT8 deploy script failed: %s", stderr_short)
            return HTMLResponse(
                content=f"""
                <div class="text-red-400 text-xs mt-1 p-2 bg-red-900/20 rounded border border-red-800">
                    ✗ Deploy failed: {stderr_short}
                </div>
                """
            )
    except subprocess.TimeoutExpired:
        logger.error("NT8 deploy script timed out")
        return HTMLResponse(
            content="""
            <div class="text-yellow-400 text-xs mt-1 p-2 bg-yellow-900/20 rounded border border-yellow-800">
                ⚠ Deploy script timed out after 30s.
            </div>
            """
        )
    except Exception as exc:
        logger.error("NT8 deploy error: %s", exc)
        return HTMLResponse(
            content=f"""
            <div class="text-red-400 text-xs mt-1 p-2 bg-red-900/20 rounded border border-red-800">
                ✗ Error: {exc}
            </div>
            """
        )
