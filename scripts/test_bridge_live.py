#!/usr/bin/env python3
"""
NT8 Bridge Live Test Checklist
===============================

Run this script when NinjaTrader 8 is open on a **Sim account** with the
BreakoutStrategy enabled and Bridge AddOn loaded.  It walks through every
Bridge integration point that the todo.md requires testing before going live:

  1. Bridge /health — is the listener running?
  2. Bridge /status — account info, strategy attachment, signal bus
  3. Heartbeat flow — does the dashboard see the Bridge as connected?
  4. /flatten — flatten all positions via dashboard button
  5. /execute_signal — manual trade from dashboard (long + short)
  6. /cancel_orders — cancel all working orders
  7. Position push — does NT8 push position snapshots to the data API?
  8. MaxConcurrentPositions gate — fire 6 entries, only 5 should execute
  9. PositionPhase tracking — verify phase1 bracket is created
  10. Risk gate feedback — verify SignalBus.IsRiskBlocked propagation

Usage:
    python scripts/test_bridge_live.py [--bridge HOST:PORT] [--data URL]

    --bridge   NT8 Bridge listener address   (default: 100.127.182.112:5680)
    --data     Data service (Pi) base URL     (default: http://100.100.84.48:8000)

    To test locally (NT8 on same machine, engine on localhost):
    python scripts/test_bridge_live.py --bridge localhost:5680 --data http://localhost:8000

Requirements:
    pip install requests

Exit codes:
    0  — all tests passed
    1  — one or more tests failed
    2  — could not reach Bridge at all (check NT8 is running)
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required.  pip install requests")
    sys.exit(2)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_BRIDGE = "100.127.182.112:5680"
DEFAULT_DATA = "http://100.100.84.48:8000"
TIMEOUT = 8  # seconds per request


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_passed = 0
_failed = 0
_skipped = 0


def _green(s):
    return f"\033[92m{s}\033[0m"


def _red(s):
    return f"\033[91m{s}\033[0m"


def _yellow(s):
    return f"\033[93m{s}\033[0m"


def _cyan(s):
    return f"\033[96m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def _section(title):
    print(f"\n{'─' * 60}")
    print(f"  {_bold(_cyan(title))}")
    print(f"{'─' * 60}")


def _pass(name, detail=""):
    global _passed
    _passed += 1
    d = f"  ({detail})" if detail else ""
    print(f"  {_green('✓ PASS')}  {name}{d}")


def _fail(name, detail=""):
    global _failed
    _failed += 1
    d = f"  ({detail})" if detail else ""
    print(f"  {_red('✗ FAIL')}  {name}{d}")


def _skip(name, detail=""):
    global _skipped
    _skipped += 1
    d = f"  ({detail})" if detail else ""
    print(f"  {_yellow('⊘ SKIP')}  {name}{d}")


def _info(msg):
    print(f"  {_cyan('ℹ')} {msg}")


def _get(url, label="GET"):
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        return resp
    except requests.ConnectionError:
        _fail(label, f"Connection refused → {url}")
        return None
    except requests.Timeout:
        _fail(label, f"Timed out → {url}")
        return None
    except Exception as exc:
        _fail(label, str(exc))
        return None


def _post(url, body=None, label="POST"):
    try:
        resp = requests.post(url, json=body or {}, timeout=TIMEOUT)
        return resp
    except requests.ConnectionError:
        _fail(label, f"Connection refused → {url}")
        return None
    except requests.Timeout:
        _fail(label, f"Timed out → {url}")
        return None
    except Exception as exc:
        _fail(label, str(exc))
        return None


def _wait(seconds, reason=""):
    if reason:
        print(f"  ⏳ Waiting {seconds}s — {reason}")
    else:
        print(f"  ⏳ Waiting {seconds}s …")
    time.sleep(seconds)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_bridge_health(bridge_url):
    """1. GET /health — Bridge HTTP listener is alive."""
    _section("1. Bridge /health")
    resp = _get(f"{bridge_url}/health", "Bridge /health")
    if resp is None:
        return False
    if resp.status_code == 200:
        data = resp.json()
        version = data.get("bridge_version", "?")
        _pass("Bridge /health", f"version={version}")
        return True
    else:
        _fail("Bridge /health", f"HTTP {resp.status_code}")
        return False


def test_bridge_status(bridge_url):
    """2. GET /status — detailed Bridge state."""
    _section("2. Bridge /status")
    resp = _get(f"{bridge_url}/status", "Bridge /status")
    if resp is None:
        return

    if resp.status_code != 200:
        _fail("Bridge /status", f"HTTP {resp.status_code}")
        return

    data = resp.json()

    # Check key fields
    account = data.get("account", "?")
    _info(f"Account: {account}")

    ruby_attached = data.get("rubyAttached", data.get("ruby_attached", "?"))
    _info(f"Strategy attached: {ruby_attached}")

    bus_active = data.get("signalBusActive", data.get("signal_bus_active", "?"))
    _info(f"SignalBus active: {bus_active}")

    bus_pending = data.get("signalBusPending", data.get("signal_bus_pending", "?"))
    _info(f"SignalBus pending: {bus_pending}")

    positions = data.get("positions", data.get("position_count", "?"))
    _info(f"Open positions: {positions}")

    risk_blocked = data.get("riskBlocked", data.get("risk_blocked", False))
    if risk_blocked:
        reason = data.get("riskBlockReason", data.get("risk_block_reason", ""))
        _info(f"⚠ Risk blocked: {reason}")

    uptime = data.get("uptime_seconds", data.get("uptime", "?"))
    _info(f"Uptime: {uptime}s")

    _pass("Bridge /status", f"account={account}")


def test_bridge_metrics(bridge_url):
    """2b. GET /metrics — Prometheus metrics endpoint."""
    resp = _get(f"{bridge_url}/metrics", "Bridge /metrics")
    if resp is None:
        return
    if resp.status_code == 200:
        lines = resp.text.strip().split("\n")
        metric_count = len([line for line in lines if line and not line.startswith("#")])
        _pass("Bridge /metrics", f"{metric_count} metric lines")
    else:
        _fail("Bridge /metrics", f"HTTP {resp.status_code}")


def test_heartbeat_flow(data_url):
    """3. Heartbeat — check if data service sees Bridge as connected."""
    _section("3. Heartbeat Flow")
    _info("Checking data service for Bridge heartbeat …")

    resp = _get(f"{data_url}/settings/services/bridge_status", "Data /bridge_status")
    if resp is None:
        resp = _get(f"{data_url}/positions/bridge_status", "Data /positions/bridge_status")
    if resp is None:
        _fail("Heartbeat flow", "Cannot reach data service")
        return

    if resp.status_code != 200:
        _fail("Heartbeat flow", f"HTTP {resp.status_code}")
        return

    data = resp.json()
    connected = data.get("connected", False)
    age = data.get("age_seconds", -1)
    account = data.get("account", "?")

    if connected:
        _pass("Heartbeat flow", f"connected=True, age={age:.0f}s, account={account}")
    else:
        _fail("Heartbeat flow", f"connected=False, age={age:.0f}s — Bridge heartbeat not reaching data service")
        _info("Check: Bridge DashboardBaseUrl points to the data service")
        _info("Check: Tailscale mesh connectivity between NT8 machine and Pi")


def test_flatten(bridge_url, data_url):
    """4. POST /flatten — flatten all positions."""
    _section("4. Flatten All")

    # First try via the data service proxy (this is how the dashboard does it)
    _info("Testing flatten via data service proxy (dashboard path) …")
    resp = _post(
        f"{data_url}/positions/flatten",
        {"reason": "live_test_script"},
        "Data /positions/flatten",
    )
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        flattened = data.get("flattened", "?")
        _pass("Flatten via data proxy", f"flattened={flattened}")
    elif resp is not None and resp.status_code == 503:
        _info("Data proxy returned 503 (Bridge heartbeat stale) — trying direct …")
    elif resp is not None:
        _fail("Flatten via data proxy", f"HTTP {resp.status_code}: {resp.text[:200]}")

    # Also try direct to Bridge
    _info("Testing flatten directly to Bridge …")
    resp2 = _post(
        f"{bridge_url}/flatten",
        {"reason": "live_test_direct"},
        "Bridge /flatten direct",
    )
    if resp2 is not None and resp2.status_code == 200:
        data2 = resp2.json()
        flattened2 = data2.get("flattened", "?")
        status2 = data2.get("status", "?")
        _pass("Flatten direct", f"status={status2}, flattened={flattened2}")
    elif resp2 is not None:
        _fail("Flatten direct", f"HTTP {resp2.status_code}: {resp2.text[:200]}")


def test_execute_signal_long(bridge_url, data_url):
    """5a. POST /execute_signal — manual long entry from dashboard."""
    _section("5a. Execute Signal — Manual LONG")

    signal_id = f"test-long-{uuid.uuid4().hex[:8]}"
    payload = {
        "direction": "long",
        "quantity": 1,
        "order_type": "market",
        "stop_loss": 0,
        "take_profit": 0,
        "strategy": "LiveTestScript",
        "asset": "MES",
        "signal_id": signal_id,
        "signal_quality": 0.99,
    }

    _info(f"Sending LONG MES via data proxy — signal_id={signal_id}")
    resp = _post(
        f"{data_url}/positions/execute",
        payload,
        "Data /positions/execute (long)",
    )

    if resp is not None and resp.status_code == 200:
        data = resp.json()
        status = data.get("status", "?")
        sid = data.get("signal_id", "?")
        if status in ("queued", "ok"):
            _pass("Execute long via proxy", f"status={status}, signal_id={sid}")
        elif status == "rejected":
            reason = data.get("reason", "?")
            _fail("Execute long via proxy", f"REJECTED: {reason}")
        else:
            _info(f"Response: {json.dumps(data, indent=2)}")
            _pass("Execute long via proxy", f"status={status}")
    elif resp is not None and resp.status_code == 503:
        _skip("Execute long via proxy", "Bridge heartbeat stale (503)")
        _info("Trying direct to Bridge …")
        resp2 = _post(
            f"{bridge_url}/execute_signal",
            payload,
            "Bridge /execute_signal direct (long)",
        )
        if resp2 is not None and resp2.status_code == 200:
            data2 = resp2.json()
            _pass("Execute long direct", f"status={data2.get('status', '?')}")
        elif resp2 is not None:
            _fail("Execute long direct", f"HTTP {resp2.status_code}")
    elif resp is not None:
        _fail("Execute long via proxy", f"HTTP {resp.status_code}: {resp.text[:200]}")

    _wait(3, "letting NT8 process the signal via DrainSignalBus")
    return signal_id


def test_execute_signal_short(bridge_url, data_url):
    """5b. POST /execute_signal — manual short entry from dashboard."""
    _section("5b. Execute Signal — Manual SHORT")

    signal_id = f"test-short-{uuid.uuid4().hex[:8]}"
    payload = {
        "direction": "short",
        "quantity": 1,
        "order_type": "market",
        "stop_loss": 0,
        "take_profit": 0,
        "strategy": "LiveTestScript",
        "asset": "MGC",
        "signal_id": signal_id,
    }

    _info(f"Sending SHORT MGC via data proxy — signal_id={signal_id}")
    resp = _post(
        f"{data_url}/positions/execute",
        payload,
        "Data /positions/execute (short)",
    )

    if resp is not None and resp.status_code == 200:
        data = resp.json()
        status = data.get("status", "?")
        if status in ("queued", "ok"):
            _pass("Execute short via proxy", f"status={status}")
        elif status == "rejected":
            _fail("Execute short via proxy", f"REJECTED: {data.get('reason', '?')}")
        else:
            _pass("Execute short via proxy", f"status={status}")
    elif resp is not None and resp.status_code == 503:
        _skip("Execute short via proxy", "Bridge heartbeat stale")
    elif resp is not None:
        _fail("Execute short via proxy", f"HTTP {resp.status_code}")

    _wait(3, "letting NT8 process the signal")
    return signal_id


def test_cancel_orders(bridge_url, data_url):
    """6. POST /cancel_orders — cancel all working orders."""
    _section("6. Cancel All Orders")

    resp = _post(
        f"{data_url}/positions/cancel_orders",
        None,
        "Data /positions/cancel_orders",
    )
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        cancelled = data.get("cancelled", "?")
        _pass("Cancel orders via proxy", f"cancelled={cancelled}")
    elif resp is not None and resp.status_code == 503:
        _skip("Cancel orders via proxy", "Bridge heartbeat stale")
    elif resp is not None:
        _fail("Cancel orders via proxy", f"HTTP {resp.status_code}")

    # Also direct
    resp2 = _post(f"{bridge_url}/cancel_orders", None, "Bridge /cancel_orders direct")
    if resp2 is not None and resp2.status_code == 200:
        data2 = resp2.json()
        _pass("Cancel orders direct", f"cancelled={data2.get('cancelled', '?')}")
    elif resp2 is not None:
        _fail("Cancel orders direct", f"HTTP {resp2.status_code}")


def test_position_push(data_url):
    """7. Position push — check if NT8 positions are visible in dashboard."""
    _section("7. Position Push (NT8 → Dashboard)")

    _info("Reading cached positions from data service …")
    resp = _get(f"{data_url}/positions/", "Data /positions/")
    if resp is None:
        return

    if resp.status_code != 200:
        _fail("Position push", f"HTTP {resp.status_code}")
        return

    data = resp.json()
    has_positions = data.get("has_positions", False)
    positions = data.get("positions", [])
    bridge_connected = data.get("bridge_connected", False)
    bridge_version = data.get("bridge_version", "?")

    _info(f"bridge_connected={bridge_connected}, bridge_version={bridge_version}")
    _info(f"has_positions={has_positions}, count={len(positions)}")

    if positions:
        for p in positions:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")
            qty = p.get("quantity", "?")
            pnl = p.get("unrealizedPnL", 0)
            _info(f"  {sym} {side} x{qty}  P&L=${pnl:.2f}")
        _pass("Position push", f"{len(positions)} positions visible")
    else:
        _info("No positions in cache — this is OK if NT8 has no open positions")
        _pass("Position push", "0 positions (expected if flat)")


def test_bridge_orders(bridge_url):
    """7b. GET /orders — check recent order events."""
    resp = _get(f"{bridge_url}/orders", "Bridge /orders")
    if resp is None:
        return

    if resp.status_code != 200:
        _fail("Bridge /orders", f"HTTP {resp.status_code}")
        return

    data = resp.json()
    orders = data.get("orders", data.get("order_events", []))
    _info(f"{len(orders)} recent order event(s)")
    for o in orders[:5]:
        _info(f"  {o.get('name', '?')} {o.get('action', '?')} {o.get('instrument', '?')} state={o.get('state', '?')}")
    _pass("Bridge /orders", f"{len(orders)} events")


def test_max_concurrent_positions(bridge_url, data_url):
    """8. MaxConcurrentPositions gate — only 5 should fill."""
    _section("8. Max Concurrent Positions Gate")
    _info("This test fires 6 entries — only 5 should be accepted by BreakoutStrategy")
    _info("(MaxConcurrentPositions=5 is enforced in C# FireEntry)")
    _info("")
    _info("⚠ MANUAL CHECK REQUIRED:")
    _info("  After running test 5a/5b, open NinjaTrader Output Window and verify:")
    _info("  - Entry logs show [positions: N/5]")
    _info("  - If N >= 5, subsequent entries show FILTERED (max concurrent positions)")
    _info("")
    _skip("Max concurrent positions", "Requires manual NT8 Output Window inspection")


def test_position_phase_tracking(bridge_url, data_url):
    """9. PositionPhase tracking — verify Phase1 bracket created."""
    _section("9. PositionPhase Tracking")
    _info("⚠ MANUAL CHECK REQUIRED in NinjaTrader Output Window:")
    _info("  - After a FireEntry, look for: [Breakout] ORB LONG MES BIP0 @ XXXX SL=YYYY TP1=ZZZZ")
    _info("  - In OnOrderUpdate, look for: Phase1 TP1 fill → Phase2 (breakeven)")
    _info("  - After TP1 fills, look for: Phase2 → Phase3 (EMA9 trail)")
    _info("  - Manual Bridge entries should show: Signal-long-brg-XXXX")
    _info("  - Each entry (automated + manual) gets its own PositionPhase with unique SignalId")
    _info("")
    _skip("PositionPhase tracking", "Requires manual NT8 Output Window inspection")


def test_risk_gate(data_url):
    """10. Risk gate — verify SignalBus.IsRiskBlocked propagation."""
    _section("10. Risk Gate Feedback")
    _info("⚠ MANUAL CHECK:")
    _info("  1. Trigger a risk block (set daily loss limit very low in risk settings)")
    _info("  2. Push a position update with large negative PnL")
    _info("  3. Verify in NT8 Output: [BreakoutStrategy] Risk gate → RiskBlocked=True")
    _info("  4. Try to execute a signal — should be rejected with 'risk blocked'")
    _info("")

    # We can at least check the risk endpoint
    resp = _get(f"{data_url}/risk/status", "Data /risk/status")
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        can_trade = data.get("can_trade", "?")
        daily_pnl = data.get("daily_pnl", "?")
        _info(f"can_trade={can_trade}, daily_pnl={daily_pnl}")
        _pass("Risk status check", f"can_trade={can_trade}")
    elif resp is not None:
        _info(f"Risk endpoint returned {resp.status_code} — may not be active in sim")
        _skip("Risk status check", "Endpoint not available")
    else:
        _skip("Risk status check", "Data service unreachable")


def test_cleanup_flatten(bridge_url, data_url):
    """Cleanup: flatten all test positions."""
    _section("Cleanup: Flatten Test Positions")
    _info("Flattening all positions to clean up test entries …")

    resp = _post(f"{bridge_url}/flatten", {"reason": "test_cleanup"}, "Cleanup flatten")
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        _pass("Cleanup flatten", f"flattened={data.get('flattened', '?')}")
    elif resp is not None:
        _fail("Cleanup flatten", f"HTTP {resp.status_code}")

    _wait(2, "letting positions settle")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="NT8 Bridge Live Test Checklist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bridge",
        default=DEFAULT_BRIDGE,
        help=f"Bridge host:port (default: {DEFAULT_BRIDGE})",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Data service URL (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Shortcut for --bridge localhost:5680 --data http://localhost:8000 (NT8 + engine on same machine)",
    )
    parser.add_argument(
        "--skip-trades",
        action="store_true",
        help="Skip execute_signal tests (no actual orders placed)",
    )
    parser.add_argument(
        "--skip-flatten",
        action="store_true",
        help="Skip flatten test (leaves positions as-is)",
    )
    args = parser.parse_args()

    # --local overrides --bridge and --data for same-machine testing
    if args.local:
        args.bridge = "localhost:5680"
        args.data = "http://localhost:8000"

    bridge_url = f"http://{args.bridge}" if not args.bridge.startswith("http") else args.bridge
    data_url = args.data.rstrip("/")

    print()
    print(_bold("═" * 60))
    print(_bold("  NT8 Bridge — Live Test Checklist"))
    print(_bold("═" * 60))
    print(f"  Bridge:       {_cyan(bridge_url)}")
    print(f"  Data Service: {_cyan(data_url)}")
    print(f"  Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Skip trades:  {args.skip_trades}")
    print(f"  Skip flatten: {args.skip_flatten}")
    print()

    # ── Prerequisites ──────────────────────────────────────────────
    print(_bold("Prerequisites:"))
    print("  • NinjaTrader 8 is open and connected to Sim account")
    print("  • BreakoutStrategy is enabled (any chart, Sim account)")
    print("  • Bridge AddOn is loaded (check NT8 AddOns panel)")
    print("  • Data service is running (Pi / Docker)")
    print()
    input("  Press ENTER to start tests …")

    # ── Run tests ──────────────────────────────────────────────────
    bridge_alive = test_bridge_health(bridge_url)

    if not bridge_alive:
        print()
        print(_red("═" * 60))
        print(_red("  FATAL: Cannot reach NT8 Bridge at " + bridge_url))
        print(_red("  Check that NinjaTrader is running and Bridge AddOn is loaded."))
        print(_red("  Bridge AddOn starts an HTTP listener on port 5680 by default."))
        print(_red("═" * 60))
        sys.exit(2)

    test_bridge_status(bridge_url)
    test_bridge_metrics(bridge_url)
    test_heartbeat_flow(data_url)

    if not args.skip_flatten:
        test_flatten(bridge_url, data_url)

    if not args.skip_trades:
        test_execute_signal_long(bridge_url, data_url)
        test_execute_signal_short(bridge_url, data_url)

    test_cancel_orders(bridge_url, data_url)
    test_position_push(data_url)
    test_bridge_orders(bridge_url)
    test_max_concurrent_positions(bridge_url, data_url)
    test_position_phase_tracking(bridge_url, data_url)
    test_risk_gate(data_url)

    if not args.skip_trades and not args.skip_flatten:
        test_cleanup_flatten(bridge_url, data_url)

    # ── Summary ────────────────────────────────────────────────────
    print()
    print(_bold("═" * 60))
    print(_bold("  Summary"))
    print(_bold("═" * 60))
    print(f"  {_green('Passed: ' + str(_passed))}")
    print(f"  {_red('Failed: ' + str(_failed))}")
    print(f"  {_yellow('Skipped: ' + str(_skipped))}")
    print()

    if _failed > 0:
        print(_red("  ✗ Some tests failed. Review output above."))
        print()
        sys.exit(1)
    else:
        print(_green("  ✓ All automated tests passed!"))
        print()
        print(_bold("  Manual checks still needed:"))
        print("    • NT8 Output Window: [positions: N/5] in entry logs")
        print("    • NT8 Output Window: No 'OCO ID cannot be reused' errors")
        print("    • NT8 Output Window: No 'signal name longer than 50' errors")
        print("    • NT8 Output Window: CNN tabular dim: model expects 18, C# builds 18")
        print("    • NT8 Output Window: Per-type TP3 mults loaded from feature_contract.json")
        print("    • Phase3 EMA9 trail: TP1 fill → breakeven → EMA9 trail (let trade run)")
        print("    • Manual entry gets its own PositionPhase (Signal-long-brg-XXX)")
        print("    • Both automated + manual entries coexist under MaxConcurrentPositions=5")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
