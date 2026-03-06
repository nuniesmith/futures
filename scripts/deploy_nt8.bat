@echo off
:: ==============================================================================
:: deploy_nt8.bat  —  NinjaTrader 8 Deploy Launcher
:: ==============================================================================
::
:: Thin wrapper that locates deploy_nt8.ps1 relative to this script and
:: invokes it under PowerShell.  If PowerShell execution policy blocks the
:: script, it temporarily bypasses the policy for this process only.
::
:: Usage (double-click or run from cmd):
::   deploy_nt8.bat
::   deploy_nt8.bat --dry-run
::   deploy_nt8.bat --no-dlls
::   deploy_nt8.bat --no-model
::   deploy_nt8.bat --launch
::
:: Any arguments passed to this .bat are forwarded verbatim to the PS1.
::
:: Elevation:
::   The script does NOT require Administrator rights — NT8 files live under
::   the user's Documents folder.  However if your Documents is on a network
::   share or you have redirected it to a location that needs elevation, the
::   script will tell you.
::
:: ==============================================================================

setlocal EnableDelayedExpansion

:: Resolve the directory this .bat lives in (handles UNC paths and spaces)
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PS1=%SCRIPT_DIR%\deploy_nt8.ps1"

:: ── Sanity check ──────────────────────────────────────────────────────────────
if not exist "%PS1%" (
    echo.
    echo   [x] Cannot find deploy_nt8.ps1
    echo       Expected: %PS1%
    echo.
    echo   Make sure you are running this .bat from the repo's scripts\ folder,
    echo   or from the repo root.
    echo.
    pause
    exit /b 1
)

:: ── Translate any --long-flags to PS1 -ShortFlags ─────────────────────────────
:: PowerShell switch params use -Flag style; pass remaining args straight through
:: since PS1 is declared with [CmdletBinding()] and understands -- separator.
::
:: We do a simple translation for the most common flags so users can type
:: natural --dry-run style at the command line.  Unknown flags are forwarded
:: as-is so the PS1 can report the error with full context.

set "PS_ARGS="
set "EXTRA_ARGS="

:parse_args
if "%~1"=="" goto done_args

set "ARG=%~1"

if /i "%ARG%"=="--dry-run"    ( set "PS_ARGS=!PS_ARGS! -DryRun"   & shift & goto parse_args )
if /i "%ARG%"=="--no-dlls"    ( set "PS_ARGS=!PS_ARGS! -NoDlls"   & shift & goto parse_args )
if /i "%ARG%"=="--no-model"   ( set "PS_ARGS=!PS_ARGS! -NoModel"  & shift & goto parse_args )
if /i "%ARG%"=="--no-source"  ( set "PS_ARGS=!PS_ARGS! -NoSource" & shift & goto parse_args )
if /i "%ARG%"=="--no-patch"   ( set "PS_ARGS=!PS_ARGS! -NoPatch"  & shift & goto parse_args )
if /i "%ARG%"=="--launch"     ( set "PS_ARGS=!PS_ARGS! -Launch"   & shift & goto parse_args )

:: Unknown / pass-through arg (e.g. -NtCustomDir "path" -Branch dev -LocalRepo "path")
set "EXTRA_ARGS=!EXTRA_ARGS! %ARG%"
shift
goto parse_args

:done_args

:: ── Run the PS1 ───────────────────────────────────────────────────────────────
echo.
echo   Running: powershell.exe -ExecutionPolicy Bypass -File "%PS1%"%PS_ARGS%%EXTRA_ARGS%
echo.

powershell.exe -NoLogo -ExecutionPolicy Bypass -File "%PS1%"%PS_ARGS%%EXTRA_ARGS%

set "EXIT_CODE=%ERRORLEVEL%"

:: ── Keep window open on failure so the user can read the error ────────────────
if %EXIT_CODE% neq 0 (
    echo.
    echo   [x] Deploy failed with exit code %EXIT_CODE%
    echo.
    echo   Common fixes:
    echo     - Close NinjaTrader 8 before running this script
    echo     - Check your internet connection
    echo     - Set GITHUB_TOKEN if the repo is private or you are rate-limited:
    echo         set GITHUB_TOKEN=ghp_your_token_here
    echo         deploy_nt8.bat
    echo     - Use a local clone to deploy offline:
    echo         powershell -File scripts\deploy_nt8.ps1 -LocalRepo C:\code\futures
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo   Done.  Press any key to close.
pause > nul
exit /b 0
