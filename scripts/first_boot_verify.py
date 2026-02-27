#!/usr/bin/env python3
"""
Docker First Boot Verification — TASK-701
============================================
Automated checklist that validates the full Futures Trading Co-Pilot stack
is healthy after a fresh `docker compose up -d --build`.

This is a P0 gate: **no live trading until every critical check passes**.

Checks:
  1.  Docker containers running (postgres, redis, data, engine)
  2.  Postgres healthcheck (pg_isready)
  3.  Redis healthcheck (PING → PONG)
  4.  Postgres tables exist (trades_v2, daily_journal, historical_bars)
  5.  Redis engine keys present (engine:status, engine:daily_focus)
  6.  Data-service /health returns 200
  7.  Dashboard (GET /) returns 200 with expected markers
  8.  API /api/info returns service info
  9.  API /api/focus returns JSON (or null if engine hasn't run)
  10. SSE /sse/health returns status
  11. SSE /sse/dashboard streams events (Content-Type check)
  12. Risk API /risk/status returns 200
  13. Positions API /positions/ returns 200 or 404
  14. Prometheus metrics /metrics/prometheus returns 200
  15. Engine health file exists with healthy=true
  16. Engine container logs have low error count
  17. Data-service container logs have low error count
  18. No-trade endpoint /api/no-trade returns 200
  19. Backfill status /backfill/status returns 200
  20. Streamlit is NOT running (TASK-304 retirement confirmed)
  21. Database write round-trip (insert + read + delete a test row)
  22. Redis write round-trip (SET + GET + DEL a test key)
  23. Cross-service: engine writes to Redis → data-service reads it

Usage:
    # Run after docker compose up -d --build
    python scripts/first_boot_verify.py

    # Options:
    python scripts/first_boot_verify.py --url http://localhost:8000
    python scripts/first_boot_verify.py --quick          # skip slow checks
    python scripts/first_boot_verify.py --verbose        # show response bodies
    python scripts/first_boot_verify.py --json           # output JSON report
    python scripts/first_boot_verify.py --wait 120       # wait up to 120s for services

Exit codes:
    0 — all critical checks passed (warnings/skips are OK)
    1 — one or more critical checks failed

Prerequisites:
    - Docker Compose stack running
    - Python 3.10+ (uses the host Python, not the container's)
    - docker CLI available on PATH
    - requests library (pip install requests) — or falls back to urllib
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class CheckSeverity(str, Enum):
    CRITICAL = "critical"  # Blocks trading — must pass
    IMPORTANT = "important"  # Should pass — review if fails
    ADVISORY = "advisory"  # Nice to have — skip OK


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    severity: CheckSeverity
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class VerificationReport:
    results: list[CheckResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.WARN)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == CheckStatus.SKIP)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def critical_failures(self) -> int:
        return sum(
            1
            for r in self.results
            if r.status == CheckStatus.FAIL and r.severity == CheckSeverity.CRITICAL
        )

    @property
    def all_critical_pass(self) -> bool:
        return self.critical_failures == 0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "warned": self.warned,
                "skipped": self.skipped,
                "critical_failures": self.critical_failures,
                "ready_for_trading": self.all_critical_pass,
            },
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "severity": r.severity.value,
                    "detail": r.detail,
                    "duration_ms": round(r.duration_ms, 1),
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# HTTP helper — uses requests if available, falls back to urllib
# ---------------------------------------------------------------------------

_HAS_REQUESTS = False
try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    import urllib.error
    import urllib.request


def http_get(
    url: str,
    timeout: float = 10.0,
    headers: Optional[dict] = None,
    stream: bool = False,
) -> tuple[int, str, dict]:
    """GET request returning (status_code, body, response_headers).

    Returns (0, error_message, {}) on connection failure.
    When *stream* is True (used for SSE), reads with a short timeout and
    returns whatever data was received before the read timed out.
    """
    if _HAS_REQUESTS:
        try:
            if stream:
                # For SSE: open a streaming connection, read whatever arrives
                # within the timeout window, then return it.
                resp = _requests.get(
                    url,
                    timeout=(5.0, timeout),
                    headers=headers or {},
                    stream=True,
                )
                resp_headers = dict(resp.headers)
                chunks: list[str] = []
                try:
                    for chunk in resp.iter_content(
                        chunk_size=None, decode_unicode=True
                    ):
                        if chunk:
                            chunks.append(chunk)
                except Exception:
                    pass  # read timeout is expected for SSE
                finally:
                    resp.close()
                return resp.status_code, "".join(chunks), resp_headers

            resp = _requests.get(url, timeout=timeout, headers=headers or {})
            resp_headers = dict(resp.headers)
            return resp.status_code, resp.text, resp_headers
        except _requests.ConnectionError as e:
            return 0, f"Connection refused: {e}", {}
        except _requests.Timeout:
            return 0, f"Timeout after {timeout}s", {}
        except Exception as e:
            return 0, str(e), {}
    else:
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                resp_headers = dict(resp.headers)
                return resp.status, body, resp_headers
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return e.code, body, {}
        except urllib.error.URLError as e:
            return 0, f"Connection error: {e.reason}", {}
        except Exception as e:
            return 0, str(e), {}


def http_post(
    url: str, json_body: dict, timeout: float = 10.0
) -> tuple[int, str, dict]:
    """POST JSON request returning (status_code, body, response_headers)."""
    if _HAS_REQUESTS:
        try:
            resp = _requests.post(url, json=json_body, timeout=timeout)
            return resp.status_code, resp.text, dict(resp.headers)
        except Exception as e:
            return 0, str(e), {}
    else:
        try:
            data = json.dumps(json_body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, body, dict(resp.headers)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return e.code, body, {}
        except Exception as e:
            return 0, str(e), {}


# ---------------------------------------------------------------------------
# Docker helper
# ---------------------------------------------------------------------------


def docker_exec(
    container: str, command: list[str], timeout: float = 15.0
) -> tuple[int, str]:
    """Run a command inside a Docker container.

    Returns (return_code, stdout+stderr).
    """
    try:
        result = subprocess.run(
            ["docker", "exec", container] + command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"
    except FileNotFoundError:
        return -1, "docker CLI not found on PATH"
    except Exception as e:
        return -1, str(e)


def docker_inspect_running(container: str) -> tuple[bool, str]:
    """Check if a Docker container is running.

    Returns (is_running, status_string).
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = result.stdout.strip()
        return status == "running", status
    except Exception:
        return False, "unknown"


def docker_logs_tail(container: str, lines: int = 50) -> str:
    """Get the last N lines of a container's logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (result.stdout + result.stderr).strip()
    except Exception:
        return ""


def detect_container_name(service_name: str, default: str) -> str:
    """Auto-detect the actual container name for a compose service.

    Tries common docker compose naming conventions.
    """
    # Try the default first
    running, _ = docker_inspect_running(default)
    if running:
        return default

    # Try common patterns
    for pattern in [
        f"futures-{service_name}-1",
        f"futures_{service_name}_1",
        f"futures-{service_name}1",
    ]:
        running, _ = docker_inspect_running(pattern)
        if running:
            return pattern

    # Try docker compose ps
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            if service_name in line.lower():
                return line.strip()
    except Exception:
        pass

    return default


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class FirstBootVerifier:
    """Runs all first-boot verification checks."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        quick: bool = False,
        verbose: bool = False,
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.quick = quick
        self.verbose = verbose
        self.api_key = api_key or os.getenv("API_KEY", "")

        # Container names (auto-detected)
        self.containers = {}

        # Report
        self.report = VerificationReport()

    def _headers(self) -> dict:
        """Build request headers (include API key if set)."""
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _timed(self, fn) -> tuple[Any, float]:
        """Run fn() and return (result, elapsed_ms)."""
        t0 = time.monotonic()
        result = fn()
        elapsed = (time.monotonic() - t0) * 1000
        return result, elapsed

    def _add(
        self,
        name: str,
        status: CheckStatus,
        severity: CheckSeverity,
        detail: str = "",
        duration_ms: float = 0.0,
    ):
        """Record a check result."""
        self.report.results.append(
            CheckResult(
                name=name,
                status=status,
                severity=severity,
                detail=detail,
                duration_ms=duration_ms,
            )
        )

    def _log(self, result: CheckResult):
        """Print a check result to stdout."""
        icons = {
            CheckStatus.PASS: "\033[0;32m✅ PASS\033[0m",
            CheckStatus.FAIL: "\033[0;31m❌ FAIL\033[0m",
            CheckStatus.WARN: "\033[1;33m⚠️  WARN\033[0m",
            CheckStatus.SKIP: "\033[1;33m⏭️  SKIP\033[0m",
        }
        sev_tag = ""
        if result.severity == CheckSeverity.CRITICAL:
            sev_tag = " \033[0;31m[CRITICAL]\033[0m"
        elif result.severity == CheckSeverity.IMPORTANT:
            sev_tag = " \033[0;36m[IMPORTANT]\033[0m"

        icon = icons.get(result.status, "  ???")
        line = f"  {icon}  {result.name}{sev_tag}"
        if result.detail:
            line += f" — {result.detail}"
        if self.verbose and result.duration_ms > 0:
            line += f"  ({result.duration_ms:.0f}ms)"
        print(line)

    def _check(self, name: str, severity: CheckSeverity, fn) -> CheckResult:
        """Run a check function and record + log the result.

        fn() should return (CheckStatus, detail_string).
        """
        try:
            (status, detail), elapsed = self._timed(fn)
        except Exception as exc:
            status = CheckStatus.FAIL
            detail = f"Exception: {exc}"
            elapsed = 0.0

        self._add(name, status, severity, detail, elapsed)
        result = self.report.results[-1]
        self._log(result)
        return result

    # -----------------------------------------------------------------------
    # Container detection
    # -----------------------------------------------------------------------

    def detect_containers(self):
        """Auto-detect container names for all services."""
        print("\n\033[0;36mℹ️  Detecting container names...\033[0m")
        service_defaults = {
            "postgres": "futures-postgres-1",
            "redis": "futures-redis-1",
            "data": "futures-data-1",
            "engine": "futures-engine-1",
        }
        for svc, default in service_defaults.items():
            name = detect_container_name(svc, default)
            self.containers[svc] = name
            print(f"    {svc:10s} → {name}")

    # -----------------------------------------------------------------------
    # Check implementations
    # -----------------------------------------------------------------------

    def check_container_running(
        self, service: str, severity: CheckSeverity = CheckSeverity.CRITICAL
    ):
        """Check that a Docker container is running."""
        container = self.containers.get(service, f"futures-{service}-1")

        def _check_fn():
            running, status = docker_inspect_running(container)
            if running:
                return CheckStatus.PASS, f"container={container}, status={status}"
            return (
                CheckStatus.FAIL,
                f"container={container} not running (status={status})",
            )

        return self._check(
            f"{service.capitalize()} container running", severity, _check_fn
        )

    def check_postgres_ready(self):
        """Check pg_isready inside the Postgres container."""
        container = self.containers.get("postgres", "futures-postgres-1")

        def _check_fn():
            rc, output = docker_exec(
                container,
                [
                    "pg_isready",
                    "-U",
                    "futures_user",
                    "-d",
                    "futures_db",
                ],
            )
            if rc == 0:
                return CheckStatus.PASS, "pg_isready OK"
            return CheckStatus.FAIL, f"pg_isready failed: {output}"

        return self._check("Postgres pg_isready", CheckSeverity.CRITICAL, _check_fn)

    def check_redis_ping(self):
        """Check redis-cli ping inside the Redis container."""
        container = self.containers.get("redis", "futures-redis-1")

        def _check_fn():
            rc, output = docker_exec(container, ["redis-cli", "ping"])
            if "PONG" in output:
                return CheckStatus.PASS, "PONG received"
            return CheckStatus.FAIL, f"Expected PONG, got: {output}"

        return self._check("Redis PING → PONG", CheckSeverity.CRITICAL, _check_fn)

    def check_postgres_tables(self):
        """Verify expected tables exist in Postgres."""
        container = self.containers.get("postgres", "futures-postgres-1")
        expected_tables = {"trades_v2", "daily_journal"}
        optional_tables = {"historical_bars"}

        def _check_fn():
            rc, output = docker_exec(
                container,
                [
                    "psql",
                    "-U",
                    "futures_user",
                    "-d",
                    "futures_db",
                    "-t",
                    "-A",
                    "-c",
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;",
                ],
            )
            if rc != 0:
                return CheckStatus.FAIL, f"psql failed: {output}"

            found_tables = {
                line.strip() for line in output.splitlines() if line.strip()
            }
            missing_critical = expected_tables - found_tables
            missing_optional = optional_tables - found_tables
            found_optional = optional_tables & found_tables

            detail_parts = [f"found: {', '.join(sorted(found_tables))}"]
            if missing_critical:
                detail_parts.append(
                    f"MISSING critical: {', '.join(sorted(missing_critical))}"
                )
                return CheckStatus.FAIL, "; ".join(detail_parts)

            if missing_optional:
                detail_parts.append(
                    f"optional not yet created: {', '.join(sorted(missing_optional))} "
                    f"(created on first backfill run)"
                )
                return CheckStatus.WARN, "; ".join(detail_parts)

            return CheckStatus.PASS, "; ".join(detail_parts)

        return self._check("Postgres tables exist", CheckSeverity.CRITICAL, _check_fn)

    def check_redis_engine_keys(self):
        """Check for engine-published keys in Redis."""
        container = self.containers.get("redis", "futures-redis-1")

        def _check_fn():
            keys_to_check = [
                "engine:status",
                "engine:daily_focus",
                "engine:risk_status",
            ]
            found = []
            missing = []
            for key in keys_to_check:
                rc, output = docker_exec(container, ["redis-cli", "exists", key])
                if "1" in output.strip():
                    found.append(key)
                else:
                    missing.append(key)

            if found and not missing:
                return CheckStatus.PASS, f"all keys present: {', '.join(found)}"
            elif found:
                return (
                    CheckStatus.WARN,
                    f"found: {', '.join(found)}; missing: {', '.join(missing)} "
                    f"(engine may still be starting)",
                )
            else:
                return (
                    CheckStatus.WARN,
                    f"no engine keys found yet — engine may still be initialising",
                )

        return self._check("Redis engine keys", CheckSeverity.IMPORTANT, _check_fn)

    def check_health_endpoint(self):
        """Check GET /health returns 200."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/health", headers=self._headers()
            )
            if status == 200:
                if self.verbose:
                    return CheckStatus.PASS, f"200 OK — {body[:200]}"
                return CheckStatus.PASS, "200 OK"
            return CheckStatus.FAIL, f"HTTP {status}: {body[:200]}"

        return self._check("GET /health returns 200", CheckSeverity.CRITICAL, _check_fn)

    def check_dashboard(self):
        """Check GET / returns 200 with expected HTML markers."""

        def _check_fn():
            status, body, _ = http_get(f"{self.base_url}/", headers=self._headers())
            if status != 200:
                return CheckStatus.FAIL, f"HTTP {status}"

            markers = {
                "title": "Futures Trading Co-Pilot" in body
                or "futures" in body.lower(),
                "sse": "sse-connect" in body or "EventSource" in body,
                "htmx": "htmx" in body.lower(),
            }
            found = [k for k, v in markers.items() if v]
            missing = [k for k, v in markers.items() if not v]

            if len(found) >= 2:
                return CheckStatus.PASS, f"200 OK, markers found: {', '.join(found)}"
            elif found:
                return (
                    CheckStatus.WARN,
                    f"200 OK but only found: {', '.join(found)}; "
                    f"missing: {', '.join(missing)}",
                )
            return CheckStatus.FAIL, f"200 OK but no expected markers in HTML"

        return self._check("Dashboard (GET /) loads", CheckSeverity.CRITICAL, _check_fn)

    def check_api_info(self):
        """Check GET /api/info returns service info."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/api/info", headers=self._headers()
            )
            if status != 200:
                return CheckStatus.FAIL, f"HTTP {status}"
            if "futures-data-service" in body:
                return CheckStatus.PASS, "200 OK, service info present"
            return CheckStatus.WARN, f"200 OK but unexpected body: {body[:150]}"

        return self._check("GET /api/info", CheckSeverity.IMPORTANT, _check_fn)

    def check_api_focus(self):
        """Check GET /api/focus returns JSON."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/api/focus", headers=self._headers()
            )
            if status != 200:
                if status == 404:
                    return CheckStatus.WARN, "404 — endpoint not found"
                return CheckStatus.FAIL, f"HTTP {status}"
            if body.strip() in ("null", ""):
                return (
                    CheckStatus.WARN,
                    "empty/null — engine may not have computed focus yet",
                )
            try:
                json.loads(body)
                return CheckStatus.PASS, f"200 OK, valid JSON ({len(body)} bytes)"
            except json.JSONDecodeError:
                return CheckStatus.WARN, "200 OK but body is not valid JSON"

        return self._check("GET /api/focus", CheckSeverity.IMPORTANT, _check_fn)

    def check_sse_health(self):
        """Check GET /sse/health returns status."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/sse/health", headers=self._headers()
            )
            if status != 200:
                return CheckStatus.FAIL, f"HTTP {status}"
            if '"status"' in body:
                return CheckStatus.PASS, "200 OK, status field present"
            return CheckStatus.WARN, f"200 OK but no status field: {body[:150]}"

        return self._check("GET /sse/health", CheckSeverity.CRITICAL, _check_fn)

    def check_sse_stream(self):
        """Check SSE stream delivers events with correct Content-Type."""
        if self.quick:
            self._add(
                "SSE stream test",
                CheckStatus.SKIP,
                CheckSeverity.IMPORTANT,
                "skipped in --quick mode",
            )
            self._log(self.report.results[-1])
            return

        def _check_fn():
            # Open a streaming connection — SSE never closes, so we read
            # whatever arrives within the timeout window (8 s) and inspect it.
            status, body, headers = http_get(
                f"{self.base_url}/sse/dashboard",
                timeout=8.0,
                headers=self._headers(),
                stream=True,
            )
            content_type = headers.get("Content-Type", headers.get("content-type", ""))

            if status == 0:
                # True connection failure (not a read timeout)
                if "event:" in body:
                    # Unlikely path — got data despite status 0
                    return (
                        CheckStatus.PASS,
                        "SSE stream delivers events (connection closed early)",
                    )
                return CheckStatus.WARN, f"Could not connect to SSE: {body[:120]}"

            if "text/event-stream" in content_type.lower():
                has_events = "event:" in body
                detail = "correct Content-Type"
                if has_events:
                    # Count distinct event types received
                    import re as _re

                    event_types = _re.findall(r"^event:\s*(.+)$", body, _re.MULTILINE)
                    unique = sorted(set(event_types))
                    detail += f", events received ({', '.join(unique[:5])})"
                else:
                    detail += " (no events in window — engine may still be starting)"
                return CheckStatus.PASS, detail
            elif status == 200:
                # Got 200 but wrong Content-Type — still mostly OK
                has_events = "event:" in body
                if has_events:
                    return (
                        CheckStatus.PASS,
                        f"200 OK, events received (Content-Type={content_type})",
                    )
                return (
                    CheckStatus.WARN,
                    f"200 OK but Content-Type={content_type}, no events",
                )
            return CheckStatus.FAIL, f"HTTP {status}"

        return self._check(
            "SSE stream (Content-Type + events)", CheckSeverity.IMPORTANT, _check_fn
        )

    def check_risk_status(self):
        """Check GET /risk/status returns 200."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/risk/status", headers=self._headers()
            )
            if status == 200:
                try:
                    data = json.loads(body)
                    source = data.get("source", "unknown")
                    can_trade = data.get("can_trade", "?")
                    return (
                        CheckStatus.PASS,
                        f"200 OK, source={source}, can_trade={can_trade}",
                    )
                except Exception:
                    return CheckStatus.PASS, "200 OK"
            return CheckStatus.FAIL, f"HTTP {status}: {body[:150]}"

        return self._check("GET /risk/status", CheckSeverity.IMPORTANT, _check_fn)

    def check_positions_api(self):
        """Check GET /positions/ returns 200 or 404."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/positions/", headers=self._headers()
            )
            if status == 200:
                return CheckStatus.PASS, "200 OK"
            if status == 404:
                return (
                    CheckStatus.PASS,
                    "404 — no position data yet (expected on fresh boot)",
                )
            return CheckStatus.FAIL, f"HTTP {status}"

        return self._check("GET /positions/", CheckSeverity.IMPORTANT, _check_fn)

    def check_prometheus_metrics(self):
        """Check GET /metrics/prometheus returns 200."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/metrics/prometheus", headers=self._headers()
            )
            if status == 200:
                # Look for prometheus metric lines
                has_metrics = "# HELP" in body or "# TYPE" in body or "_total" in body
                if has_metrics:
                    line_count = len(body.strip().splitlines())
                    return CheckStatus.PASS, f"200 OK, {line_count} metric lines"
                return CheckStatus.PASS, f"200 OK ({len(body)} bytes)"
            return CheckStatus.FAIL, f"HTTP {status}"

        return self._check("GET /metrics/prometheus", CheckSeverity.ADVISORY, _check_fn)

    def check_engine_health(self):
        """Check engine health file inside the engine container."""
        container = self.containers.get("engine", "futures-engine-1")

        def _check_fn():
            rc, output = docker_exec(container, ["cat", "/tmp/engine_health.json"])
            if rc != 0:
                return CheckStatus.FAIL, f"health file not found: {output}"
            try:
                data = json.loads(output)
                healthy = data.get("healthy", False)
                status = data.get("status", "unknown")
                session = data.get("session", "?")
                if healthy:
                    return (
                        CheckStatus.PASS,
                        f"healthy=true, status={status}, session={session}",
                    )
                return CheckStatus.FAIL, f"healthy=false, status={status}"
            except json.JSONDecodeError:
                return CheckStatus.FAIL, f"invalid JSON: {output[:100]}"

        return self._check("Engine health file", CheckSeverity.CRITICAL, _check_fn)

    def check_container_logs(
        self, service: str, severity: CheckSeverity = CheckSeverity.IMPORTANT
    ):
        """Check a container's recent logs for error count."""
        container = self.containers.get(service, f"futures-{service}-1")

        def _check_fn():
            logs = docker_logs_tail(container, lines=50)
            if not logs:
                return CheckStatus.WARN, "no logs available"

            error_patterns = re.compile(
                r"(?:error|exception|traceback|critical|fatal)",
                re.IGNORECASE,
            )
            # Patterns that look like errors but are actually benign
            # e.g. "Errors: 0", "errors: 0", "0 errors", summary lines
            false_positive = re.compile(
                r"(?:"
                r"errors:\s*0"  # "Errors: 0"
                r"|\b0\s+errors?\b"  # "0 errors" / "0 error"
                r"|error.?like"  # "error-like lines" (from this script)
                r"|no.?error"  # "no errors"
                r"|exc_info="  # structured log field, not an actual error
                r"|log.?level.*error"  # log config mentioning error level
                r")",
                re.IGNORECASE,
            )
            error_lines = [
                line
                for line in logs.splitlines()
                if error_patterns.search(line) and not false_positive.search(line)
            ]
            error_count = len(error_lines)

            if error_count == 0:
                return CheckStatus.PASS, "0 error-like lines in last 50 log lines"
            elif error_count < 3:
                return (
                    CheckStatus.WARN,
                    f"{error_count} error-like line(s) in last 50 — review recommended",
                )
            return (
                CheckStatus.FAIL,
                f"{error_count} error-like lines in last 50 log lines",
            )

        return self._check(
            f"{service.capitalize()} container logs", severity, _check_fn
        )

    def check_no_trade_endpoint(self):
        """Check GET /api/no-trade returns 200."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/api/no-trade", headers=self._headers()
            )
            if status == 200:
                return CheckStatus.PASS, "200 OK"
            if status == 404:
                return CheckStatus.WARN, "404 — endpoint may not be registered"
            return CheckStatus.FAIL, f"HTTP {status}"

        return self._check("GET /api/no-trade", CheckSeverity.ADVISORY, _check_fn)

    def check_backfill_status(self):
        """Check GET /backfill/status returns 200."""

        def _check_fn():
            status, body, _ = http_get(
                f"{self.base_url}/backfill/status", headers=self._headers()
            )
            if status == 200:
                return CheckStatus.PASS, "200 OK"
            if status == 404:
                return (
                    CheckStatus.WARN,
                    "404 — backfill endpoints may not be registered yet",
                )
            return CheckStatus.FAIL, f"HTTP {status}"

        return self._check("GET /backfill/status", CheckSeverity.ADVISORY, _check_fn)

    def check_streamlit_retired(self):
        """Confirm Streamlit container is NOT running (TASK-304)."""

        def _check_fn():
            try:
                result = subprocess.run(
                    ["docker", "compose", "ps", "--format", "{{.Name}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                names = result.stdout.strip().lower()
                if "streamlit" in names or "app" in names:
                    # Check more carefully — "app" might be a substring of other names
                    for line in result.stdout.strip().splitlines():
                        name = line.strip().lower()
                        if name.endswith("-app-1") or "streamlit" in name:
                            return (
                                CheckStatus.FAIL,
                                f"Streamlit container still running: {line.strip()}",
                            )
                return CheckStatus.PASS, "no Streamlit/app container found (retired)"
            except Exception as e:
                return CheckStatus.WARN, f"could not check: {e}"

        return self._check(
            "Streamlit retired (TASK-304)", CheckSeverity.ADVISORY, _check_fn
        )

    def check_db_write_roundtrip(self):
        """Verify database write round-trip via the data-service API."""
        container = self.containers.get("postgres", "futures-postgres-1")

        def _check_fn():
            # Insert a test row into a known table, read it back, delete it
            test_value = f"__first_boot_verify_{int(time.time())}__"

            # INSERT
            rc, output = docker_exec(
                container,
                [
                    "psql",
                    "-U",
                    "futures_user",
                    "-d",
                    "futures_db",
                    "-t",
                    "-A",
                    "-c",
                    f"INSERT INTO daily_journal (trade_date, account_size, gross_pnl, net_pnl, created_at) "
                    f"VALUES ('{test_value}', 0, 0, 0, '{test_value}') RETURNING id;",
                ],
            )
            if rc != 0:
                return CheckStatus.FAIL, f"INSERT failed: {output}"

            # psql output for RETURNING is e.g. "3\nINSERT 0 1" — take first non-empty line
            lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
            row_id = lines[0] if lines else ""
            if not row_id:
                return CheckStatus.FAIL, f"INSERT returned no id: {output}"

            # SELECT
            rc, output = docker_exec(
                container,
                [
                    "psql",
                    "-U",
                    "futures_user",
                    "-d",
                    "futures_db",
                    "-t",
                    "-A",
                    "-c",
                    f"SELECT trade_date FROM daily_journal WHERE id = {row_id};",
                ],
            )
            if rc != 0 or test_value not in output:
                return CheckStatus.FAIL, f"SELECT failed or wrong value: {output}"

            # DELETE
            rc, output = docker_exec(
                container,
                [
                    "psql",
                    "-U",
                    "futures_user",
                    "-d",
                    "futures_db",
                    "-t",
                    "-A",
                    "-c",
                    f"DELETE FROM daily_journal WHERE id = {row_id};",
                ],
            )
            if rc != 0:
                return CheckStatus.WARN, f"DELETE cleanup failed: {output}"

            return CheckStatus.PASS, f"INSERT→SELECT→DELETE round-trip OK (id={row_id})"

        return self._check(
            "Postgres write round-trip", CheckSeverity.CRITICAL, _check_fn
        )

    def check_redis_write_roundtrip(self):
        """Verify Redis write round-trip."""
        container = self.containers.get("redis", "futures-redis-1")
        test_key = f"__first_boot_verify:{int(time.time())}"
        test_value = "hello_from_first_boot"

        def _check_fn():
            # SET
            rc, output = docker_exec(
                container, ["redis-cli", "SET", test_key, test_value, "EX", "30"]
            )
            if rc != 0 or "OK" not in output:
                return CheckStatus.FAIL, f"SET failed: {output}"

            # GET
            rc, output = docker_exec(container, ["redis-cli", "GET", test_key])
            if rc != 0 or test_value not in output:
                return CheckStatus.FAIL, f"GET failed or wrong value: {output}"

            # DEL
            rc, output = docker_exec(container, ["redis-cli", "DEL", test_key])
            if rc != 0:
                return CheckStatus.WARN, f"DEL cleanup failed: {output}"

            return CheckStatus.PASS, "SET→GET→DEL round-trip OK"

        return self._check("Redis write round-trip", CheckSeverity.CRITICAL, _check_fn)

    def check_cross_service_redis(self):
        """Check that engine has published data readable by data-service.

        Verifies the engine → Redis → data-service pipeline.
        """

        def _check_fn():
            # The data-service /health endpoint includes engine info from Redis
            status, body, _ = http_get(
                f"{self.base_url}/health", headers=self._headers()
            )
            if status != 200:
                return CheckStatus.FAIL, f"/health returned HTTP {status}"

            try:
                data = json.loads(body)
            except Exception:
                return CheckStatus.WARN, "/health returned non-JSON"

            # Also check /risk/status which reads from Redis
            rstatus, rbody, _ = http_get(
                f"{self.base_url}/risk/status", headers=self._headers()
            )
            if rstatus == 200:
                try:
                    rdata = json.loads(rbody)
                    source = rdata.get("source", "unknown")
                    if source == "redis":
                        return (
                            CheckStatus.PASS,
                            "engine → Redis → data-service pipeline confirmed (risk source=redis)",
                        )
                    elif source == "local":
                        return (
                            CheckStatus.WARN,
                            "risk reads from local RiskManager, not Redis — engine may still be starting",
                        )
                except Exception:
                    pass

            # Check if any engine keys exist
            container = self.containers.get("redis", "futures-redis-1")
            rc, output = docker_exec(
                container, ["redis-cli", "exists", "engine:status"]
            )
            if "1" in output.strip():
                return CheckStatus.PASS, "engine:status key exists in Redis"

            return (
                CheckStatus.WARN,
                "engine data not yet in Redis — engine may still be starting up",
            )

        return self._check(
            "Cross-service Redis pipeline", CheckSeverity.IMPORTANT, _check_fn
        )

    def check_risk_preflight(self):
        """Run a risk pre-flight check via POST /risk/check."""

        def _check_fn():
            payload = {
                "symbol": "MGC",
                "side": "LONG",
                "size": 1,
                "risk_per_contract": 100.0,
            }
            status, body, _ = http_post(
                f"{self.base_url}/risk/check",
                json_body=payload,
                timeout=10.0,
            )
            if status == 200:
                try:
                    data = json.loads(body)
                    allowed = data.get("allowed", "?")
                    return (
                        CheckStatus.PASS,
                        f"200 OK, allowed={allowed}",
                    )
                except Exception:
                    return CheckStatus.PASS, "200 OK"
            if status == 503:
                return CheckStatus.WARN, "503 — risk engine not available yet"
            if status == 422:
                return CheckStatus.WARN, f"422 validation error: {body[:150]}"
            return CheckStatus.FAIL, f"HTTP {status}: {body[:150]}"

        return self._check(
            "POST /risk/check pre-flight", CheckSeverity.ADVISORY, _check_fn
        )

    # -----------------------------------------------------------------------
    # Run all checks
    # -----------------------------------------------------------------------

    def run(self) -> VerificationReport:
        """Execute all verification checks and return the report."""
        self.report = VerificationReport()
        self.report.started_at = datetime.now(UTC).isoformat()
        t0 = time.monotonic()

        # Header
        print()
        print("\033[1m" + "═" * 65 + "\033[0m")
        print(
            "\033[1m  Futures Trading Co-Pilot — First Boot Verification (TASK-701)\033[0m"
        )
        print("\033[1m" + "═" * 65 + "\033[0m")
        print(f"  Target:    {self.base_url}")
        print(f"  Mode:      {'quick' if self.quick else 'full'}")
        print(f"  Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  API Key:   {'set' if self.api_key else 'not set'}")
        print("\033[1m" + "═" * 65 + "\033[0m")

        # Phase 1: Docker containers
        self.detect_containers()

        print("\n\033[1m--- Docker Containers ---\033[0m")
        self.check_container_running("postgres")
        self.check_container_running("redis")
        self.check_container_running("data")
        self.check_container_running("engine")

        # Phase 2: Infrastructure health
        print("\n\033[1m--- Infrastructure Health ---\033[0m")
        self.check_postgres_ready()
        self.check_redis_ping()
        self.check_postgres_tables()
        self.check_redis_engine_keys()

        # Phase 3: Data round-trips
        print("\n\033[1m--- Data Round-Trip Tests ---\033[0m")
        self.check_db_write_roundtrip()
        self.check_redis_write_roundtrip()

        # Phase 4: API endpoints
        print("\n\033[1m--- Data Service Endpoints ---\033[0m")
        self.check_health_endpoint()
        self.check_dashboard()
        self.check_api_info()
        self.check_api_focus()
        self.check_risk_status()
        self.check_positions_api()
        self.check_prometheus_metrics()
        self.check_no_trade_endpoint()
        self.check_backfill_status()

        # Phase 5: SSE
        print("\n\033[1m--- SSE Subsystem ---\033[0m")
        self.check_sse_health()
        self.check_sse_stream()

        # Phase 6: Engine health & logs
        print("\n\033[1m--- Engine & Logs ---\033[0m")
        self.check_engine_health()
        self.check_container_logs("engine")
        self.check_container_logs("data")

        # Phase 7: Cross-service & advanced
        print("\n\033[1m--- Cross-Service Integration ---\033[0m")
        self.check_cross_service_redis()
        self.check_risk_preflight()

        # Phase 8: Retirement checks
        print("\n\033[1m--- Retirement Checks ---\033[0m")
        self.check_streamlit_retired()

        # Finish
        self.report.finished_at = datetime.now(UTC).isoformat()
        self.report.duration_s = time.monotonic() - t0

        return self.report


# ---------------------------------------------------------------------------
# Wait-for-services helper
# ---------------------------------------------------------------------------


def wait_for_services(
    base_url: str, timeout_s: int = 60, headers: Optional[dict] = None
):
    """Block until the data-service responds to /health, or timeout."""
    print(f"\n\033[0;36mℹ️  Waiting for services (up to {timeout_s}s)...\033[0m")
    deadline = time.monotonic() + timeout_s
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        status, body, _ = http_get(
            f"{base_url}/health", timeout=3.0, headers=headers or {}
        )
        if status == 200:
            print(
                f"  \033[0;32m✅ Data-service ready after {attempt} attempt(s)\033[0m\n"
            )
            return True
        remaining = int(deadline - time.monotonic())
        print(
            f"  Attempt {attempt}: HTTP {status} — retrying ({remaining}s remaining)..."
        )
        time.sleep(min(5, remaining))

    print(
        f"  \033[0;31m❌ Timed out after {timeout_s}s waiting for data-service\033[0m\n"
    )
    return False


# ---------------------------------------------------------------------------
# Print report summary
# ---------------------------------------------------------------------------


def print_summary(report: VerificationReport):
    """Print a formatted summary of the verification report."""
    print()
    print("\033[1m" + "═" * 65 + "\033[0m")
    print("\033[1m  FIRST BOOT VERIFICATION SUMMARY\033[0m")
    print("\033[1m" + "═" * 65 + "\033[0m")
    print()

    # Recap of results (compact)
    for r in report.results:
        icon_map = {
            CheckStatus.PASS: "\033[0;32m✅\033[0m",
            CheckStatus.FAIL: "\033[0;31m❌\033[0m",
            CheckStatus.WARN: "\033[1;33m⚠️\033[0m ",
            CheckStatus.SKIP: "\033[1;33m⏭️\033[0m ",
        }
        sev_marker = ""
        if r.severity == CheckSeverity.CRITICAL and r.status == CheckStatus.FAIL:
            sev_marker = " \033[0;31m← BLOCKER\033[0m"
        icon = icon_map.get(r.status, "  ")
        print(f"  {icon} {r.name}{sev_marker}")

    print()
    print("\033[1m" + "─" * 65 + "\033[0m")
    print(f"  Total:     {report.total}")
    print(f"  \033[0;32mPassed:    {report.passed}\033[0m")
    if report.failed:
        print(
            f"  \033[0;31mFailed:    {report.failed} ({report.critical_failures} critical)\033[0m"
        )
    else:
        print(f"  Failed:    0")
    if report.warned:
        print(f"  \033[1;33mWarnings:  {report.warned}\033[0m")
    if report.skipped:
        print(f"  \033[1;33mSkipped:   {report.skipped}\033[0m")
    print(f"  Duration:  {report.duration_s:.1f}s")
    print("\033[1m" + "─" * 65 + "\033[0m")

    if report.all_critical_pass:
        print()
        print(
            "  \033[0;32m\033[1m🎉 ALL CRITICAL CHECKS PASSED — Ready for trading!\033[0m"
        )
        if report.warned:
            print(
                f"  \033[1;33m   ({report.warned} non-critical warning(s) — review recommended)\033[0m"
            )
        print()
    else:
        print()
        print(
            f"  \033[0;31m\033[1m🚫 {report.critical_failures} CRITICAL CHECK(S) FAILED — DO NOT TRADE\033[0m"
        )
        print()
        print("  Critical failures:")
        for r in report.results:
            if r.status == CheckStatus.FAIL and r.severity == CheckSeverity.CRITICAL:
                print(f"    ❌ {r.name}: {r.detail}")
        print()
        print("  Fix the above issues and re-run this script.")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Docker First Boot Verification (TASK-701) — "
            "Automated checklist for the Futures Trading Co-Pilot stack"
        ),
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DATA_SERVICE_URL", "http://localhost:8000"),
        help="Data-service base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow checks (SSE stream hold test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print response bodies and timing details",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output a JSON report instead of formatted text",
    )
    parser.add_argument(
        "--json-file",
        default="",
        help="Write JSON report to this file (in addition to stdout)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Wait up to N seconds for services to become healthy before running checks",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="API key for authenticated endpoints (default: from API_KEY env var)",
    )
    args = parser.parse_args()

    # Optionally wait for services
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    if args.wait > 0:
        ready = wait_for_services(args.url, timeout_s=args.wait, headers=headers)
        if not ready and not args.json_output:
            print("\033[1;33m⚠️  Proceeding with checks despite timeout...\033[0m")

    # Run verification
    verifier = FirstBootVerifier(
        base_url=args.url,
        quick=args.quick,
        verbose=args.verbose,
        api_key=args.api_key,
    )
    report = verifier.run()

    # Output
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_summary(report)

    # Optionally write JSON file
    if args.json_file:
        with open(args.json_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if not args.json_output:
            print(f"  📄 JSON report written to: {args.json_file}\n")

    # Exit code
    sys.exit(0 if report.all_critical_pass else 1)


if __name__ == "__main__":
    main()
