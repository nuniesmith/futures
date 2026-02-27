"""
Tests for the first-boot verification script (TASK-701).

Covers:
  - CheckResult and VerificationReport data classes
  - HTTP helper functions (http_get, http_post)
  - Docker helper functions (docker_exec, docker_inspect_running, detect_container_name)
  - FirstBootVerifier check methods (with mocked Docker/HTTP)
  - Report generation and summary logic
  - CLI argument parsing
"""

import json
import os
import subprocess
import sys
import time  # noqa: F401 — used by @patch("time.time") and @patch("time.monotonic")
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the scripts directory is importable
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_scripts_dir = os.path.join(_project_root, "scripts")

if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Import the module under test (it's a script, so we import by manipulating
# the path rather than using importlib)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "first_boot_verify", os.path.join(_scripts_dir, "first_boot_verify.py")
)
assert _spec is not None, "Could not find first_boot_verify.py"
assert _spec.loader is not None, "Spec has no loader"
fbv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fbv)


# ===========================================================================
# Data class tests
# ===========================================================================


class TestCheckStatus:
    """Test CheckStatus enum."""

    def test_values(self):
        assert fbv.CheckStatus.PASS == "pass"
        assert fbv.CheckStatus.FAIL == "fail"
        assert fbv.CheckStatus.WARN == "warn"
        assert fbv.CheckStatus.SKIP == "skip"

    def test_all_values_are_strings(self):
        for status in fbv.CheckStatus:
            assert isinstance(status.value, str)


class TestCheckSeverity:
    """Test CheckSeverity enum."""

    def test_values(self):
        assert fbv.CheckSeverity.CRITICAL == "critical"
        assert fbv.CheckSeverity.IMPORTANT == "important"
        assert fbv.CheckSeverity.ADVISORY == "advisory"


class TestCheckResult:
    """Test CheckResult data class."""

    def test_creation(self):
        r = fbv.CheckResult(
            name="test check",
            status=fbv.CheckStatus.PASS,
            severity=fbv.CheckSeverity.CRITICAL,
            detail="all good",
            duration_ms=42.5,
        )
        assert r.name == "test check"
        assert r.status == fbv.CheckStatus.PASS
        assert r.severity == fbv.CheckSeverity.CRITICAL
        assert r.detail == "all good"
        assert r.duration_ms == pytest.approx(42.5)

    def test_defaults(self):
        r = fbv.CheckResult(
            name="minimal",
            status=fbv.CheckStatus.FAIL,
            severity=fbv.CheckSeverity.ADVISORY,
        )
        assert r.detail == ""
        assert r.duration_ms == 0.0


class TestVerificationReport:
    """Test VerificationReport data class."""

    def _make_result(self, status, severity):
        return fbv.CheckResult(
            name="test", status=status, severity=severity, detail="", duration_ms=0
        )

    def test_empty_report(self):
        report = fbv.VerificationReport()
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.warned == 0
        assert report.skipped == 0
        assert report.critical_failures == 0
        assert report.all_critical_pass is True

    def test_counts(self):
        report = fbv.VerificationReport()
        report.results = [
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL),
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.IMPORTANT),
            self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.ADVISORY),
            self._make_result(fbv.CheckStatus.WARN, fbv.CheckSeverity.IMPORTANT),
            self._make_result(fbv.CheckStatus.SKIP, fbv.CheckSeverity.ADVISORY),
        ]
        assert report.total == 5
        assert report.passed == 2
        assert report.failed == 1
        assert report.warned == 1
        assert report.skipped == 1

    def test_critical_failures(self):
        report = fbv.VerificationReport()
        report.results = [
            self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.CRITICAL),
            self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.IMPORTANT),
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL),
        ]
        assert report.critical_failures == 1
        assert report.all_critical_pass is False

    def test_all_critical_pass_true(self):
        report = fbv.VerificationReport()
        report.results = [
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL),
            self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.ADVISORY),
            self._make_result(fbv.CheckStatus.WARN, fbv.CheckSeverity.IMPORTANT),
        ]
        assert report.all_critical_pass is True

    def test_to_dict(self):
        report = fbv.VerificationReport()
        report.started_at = "2024-01-01T00:00:00Z"
        report.finished_at = "2024-01-01T00:01:00Z"
        report.duration_s = 60.123
        report.results = [
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL),
        ]

        d = report.to_dict()
        assert d["started_at"] == "2024-01-01T00:00:00Z"
        assert d["finished_at"] == "2024-01-01T00:01:00Z"
        assert d["duration_s"] == pytest.approx(60.12)
        assert d["summary"]["total"] == 1
        assert d["summary"]["passed"] == 1
        assert d["summary"]["failed"] == 0
        assert d["summary"]["ready_for_trading"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0]["status"] == "pass"
        assert d["checks"][0]["severity"] == "critical"

    def test_to_dict_is_json_serializable(self):
        report = fbv.VerificationReport()
        report.started_at = "2024-01-01T00:00:00Z"
        report.finished_at = "2024-01-01T00:01:00Z"
        report.duration_s = 1.5
        report.results = [
            self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL),
            self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.IMPORTANT),
        ]
        # Should not raise
        serialized = json.dumps(report.to_dict())
        parsed = json.loads(serialized)
        assert parsed["summary"]["total"] == 2


# ===========================================================================
# HTTP helper tests (mocked)
# ===========================================================================


class TestHTTPGet:
    """Test http_get helper with mocked backends."""

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_successful_get_requests(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "ok"}'
        mock_resp.headers = {"Content-Type": "application/json"}

        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.get.return_value = mock_resp
            mock_requests.ConnectionError = ConnectionError
            mock_requests.Timeout = TimeoutError

            status, body, headers = fbv.http_get("http://localhost:8000/health")

        assert status == 200
        assert body == '{"status": "ok"}'
        assert "Content-Type" in headers

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_connection_refused_requests(self):
        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.ConnectionError = ConnectionError
            mock_requests.Timeout = TimeoutError
            mock_requests.get.side_effect = ConnectionError("refused")

            status, body, headers = fbv.http_get("http://localhost:9999/bad")

        assert status == 0
        assert "refused" in body.lower() or "Connection" in body
        assert headers == {}

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_timeout_requests(self):
        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.ConnectionError = ConnectionError
            mock_requests.Timeout = TimeoutError
            mock_requests.get.side_effect = TimeoutError("timed out")

            status, body, headers = fbv.http_get(
                "http://localhost:8000/slow", timeout=1.0
            )

        assert status == 0
        assert headers == {}

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_headers_passed_through(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"
        mock_resp.headers = {}

        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.get.return_value = mock_resp
            mock_requests.ConnectionError = ConnectionError
            mock_requests.Timeout = TimeoutError

            fbv.http_get(
                "http://localhost:8000/test",
                headers={"X-API-Key": "secret"},
            )

        call_kwargs = mock_requests.get.call_args
        assert call_kwargs[1]["headers"] == {"X-API-Key": "secret"}


class TestHTTPPost:
    """Test http_post helper with mocked backend."""

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_successful_post(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.text = '{"id": 1}'
        mock_resp.headers = {"Content-Type": "application/json"}

        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.post.return_value = mock_resp

            status, body, headers = fbv.http_post(
                "http://localhost:8000/audit/risk",
                json_body={"event_type": "block"},
            )

        assert status == 201
        assert '"id"' in body

    @patch.object(fbv, "_HAS_REQUESTS", True)
    def test_post_failure(self):
        with patch.object(fbv, "_requests", create=True) as mock_requests:
            mock_requests.post.side_effect = Exception("connection failed")

            status, body, headers = fbv.http_post(
                "http://localhost:8000/fail",
                json_body={},
            )

        assert status == 0
        assert "connection failed" in body


# ===========================================================================
# Docker helper tests (mocked)
# ===========================================================================


class TestDockerExec:
    """Test docker_exec helper."""

    @patch("subprocess.run")
    def test_successful_exec(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="PONG\n", stderr="")

        rc, output = fbv.docker_exec("futures-redis-1", ["redis-cli", "ping"])

        assert rc == 0
        assert "PONG" in output
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["docker", "exec", "futures-redis-1", "redis-cli", "ping"]

    @patch("subprocess.run")
    def test_failed_exec(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error: container not found"
        )

        rc, output = fbv.docker_exec("bad-container", ["echo", "hi"])

        assert rc == 1
        assert "not found" in output

    @patch("subprocess.run")
    def test_timeout_exec(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=15)

        rc, output = fbv.docker_exec("slow-container", ["sleep", "999"])

        assert rc == -1
        assert "timed out" in output.lower()

    @patch("subprocess.run")
    def test_docker_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        rc, output = fbv.docker_exec("any", ["any"])

        assert rc == -1
        assert "not found" in output.lower()


class TestDockerInspectRunning:
    """Test docker_inspect_running helper."""

    @patch("subprocess.run")
    def test_running_container(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="running\n", stderr="")

        is_running, status = fbv.docker_inspect_running("futures-data-1")

        assert is_running is True
        assert status == "running"

    @patch("subprocess.run")
    def test_stopped_container(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="exited\n", stderr="")

        is_running, status = fbv.docker_inspect_running("futures-data-1")

        assert is_running is False
        assert status == "exited"

    @patch("subprocess.run")
    def test_no_such_container(self, mock_run):
        mock_run.side_effect = Exception("No such container")

        is_running, status = fbv.docker_inspect_running("nonexistent")

        assert is_running is False
        assert status == "unknown"


class TestDockerLogsTail:
    """Test docker_logs_tail helper."""

    @patch("subprocess.run")
    def test_returns_logs(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="line1\nline2\nline3\n",
            stderr="",
        )

        logs = fbv.docker_logs_tail("futures-engine-1", lines=3)
        assert "line1" in logs
        assert "line3" in logs

    @patch("subprocess.run")
    def test_returns_empty_on_error(self, mock_run):
        mock_run.side_effect = Exception("container not found")
        logs = fbv.docker_logs_tail("bad", lines=10)
        assert logs == ""


class TestDetectContainerName:
    """Test detect_container_name helper."""

    @patch.object(fbv, "docker_inspect_running")
    def test_uses_default_if_running(self, mock_inspect):
        mock_inspect.return_value = (True, "running")

        name = fbv.detect_container_name("redis", "futures-redis-1")

        assert name == "futures-redis-1"
        mock_inspect.assert_called_once_with("futures-redis-1")

    @patch("subprocess.run")
    @patch.object(fbv, "docker_inspect_running")
    def test_tries_alternate_patterns(self, mock_inspect, mock_run):
        # Default not running, first alternate running
        def side_effect(container):
            if container == "futures-redis-1":
                return (True, "running")
            return (False, "not found")

        mock_inspect.side_effect = side_effect

        name = fbv.detect_container_name("redis", "my-custom-redis")

        # Should have tried default first, then patterns
        assert name == "futures-redis-1"

    @patch("subprocess.run")
    @patch.object(fbv, "docker_inspect_running")
    def test_falls_back_to_default(self, mock_inspect, mock_run):
        mock_inspect.return_value = (False, "not found")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        name = fbv.detect_container_name("redis", "my-default")

        assert name == "my-default"


# ===========================================================================
# FirstBootVerifier tests
# ===========================================================================


class TestFirstBootVerifierInit:
    """Test verifier initialisation."""

    def test_default_init(self):
        v = fbv.FirstBootVerifier()
        assert v.base_url == "http://localhost:8000"
        assert v.quick is False
        assert v.verbose is False
        assert isinstance(v.report, fbv.VerificationReport)

    def test_custom_init(self):
        v = fbv.FirstBootVerifier(
            base_url="http://example.com:9000",
            quick=True,
            verbose=True,
            api_key="test-key",
        )
        assert v.base_url == "http://example.com:9000"
        assert v.quick is True
        assert v.verbose is True
        assert v.api_key == "test-key"

    def test_trailing_slash_stripped(self):
        v = fbv.FirstBootVerifier(base_url="http://localhost:8000/")
        assert v.base_url == "http://localhost:8000"

    def test_headers_include_api_key(self):
        v = fbv.FirstBootVerifier(api_key="my-secret")
        h = v._headers()
        assert h == {"X-API-Key": "my-secret"}

    def test_headers_empty_without_api_key(self):
        v = fbv.FirstBootVerifier(api_key="")
        h = v._headers()
        assert h == {}


class TestFirstBootVerifierChecks:
    """Test individual check methods with mocked Docker/HTTP."""

    def _make_verifier(self, **kwargs):
        v = fbv.FirstBootVerifier(**kwargs)
        v.containers = {
            "postgres": "futures-postgres-1",
            "redis": "futures-redis-1",
            "data": "futures-data-1",
            "engine": "futures-engine-1",
        }
        return v

    # --- Container running checks ---

    @patch.object(fbv, "docker_inspect_running", return_value=(True, "running"))
    def test_check_container_running_pass(self, mock_inspect):
        v = self._make_verifier()
        v.check_container_running("postgres")

        assert len(v.report.results) == 1
        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "docker_inspect_running", return_value=(False, "exited"))
    def test_check_container_running_fail(self, mock_inspect):
        v = self._make_verifier()
        v.check_container_running("postgres")

        assert len(v.report.results) == 1
        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Postgres ready ---

    @patch.object(fbv, "docker_exec", return_value=(0, "accepting connections"))
    def test_check_postgres_ready_pass(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_ready()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "docker_exec", return_value=(1, "not ready"))
    def test_check_postgres_ready_fail(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_ready()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Redis ping ---

    @patch.object(fbv, "docker_exec", return_value=(0, "PONG"))
    def test_check_redis_ping_pass(self, mock_exec):
        v = self._make_verifier()
        v.check_redis_ping()

        assert v.report.results[0].status == fbv.CheckStatus.PASS
        assert "PONG" in v.report.results[0].detail

    @patch.object(fbv, "docker_exec", return_value=(1, "error"))
    def test_check_redis_ping_fail(self, mock_exec):
        v = self._make_verifier()
        v.check_redis_ping()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Postgres tables ---

    @patch.object(
        fbv,
        "docker_exec",
        return_value=(0, "daily_journal\nhistorical_bars\ntrades_v2\n"),
    )
    def test_check_postgres_tables_all_present(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_tables()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(
        fbv,
        "docker_exec",
        return_value=(0, "daily_journal\ntrades_v2\n"),
    )
    def test_check_postgres_tables_optional_missing(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_tables()

        # Optional tables missing → WARN not FAIL
        assert v.report.results[0].status == fbv.CheckStatus.WARN

    @patch.object(
        fbv,
        "docker_exec",
        return_value=(0, "daily_journal\n"),
    )
    def test_check_postgres_tables_critical_missing(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_tables()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL
        assert "trades_v2" in v.report.results[0].detail

    @patch.object(fbv, "docker_exec", return_value=(1, "psql error"))
    def test_check_postgres_tables_psql_error(self, mock_exec):
        v = self._make_verifier()
        v.check_postgres_tables()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Redis engine keys ---

    @patch.object(fbv, "docker_exec")
    def test_check_redis_engine_keys_all_present(self, mock_exec):
        mock_exec.return_value = (0, "(integer) 1")
        v = self._make_verifier()
        v.check_redis_engine_keys()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "docker_exec")
    def test_check_redis_engine_keys_none_present(self, mock_exec):
        mock_exec.return_value = (0, "(integer) 0")
        v = self._make_verifier()
        v.check_redis_engine_keys()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- Health endpoint ---

    @patch.object(fbv, "http_get", return_value=(200, '{"status": "ok"}', {}))
    def test_check_health_endpoint_pass(self, mock_get):
        v = self._make_verifier()
        v.check_health_endpoint()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(500, "error", {}))
    def test_check_health_endpoint_fail(self, mock_get):
        v = self._make_verifier()
        v.check_health_endpoint()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    @patch.object(fbv, "http_get", return_value=(0, "Connection refused", {}))
    def test_check_health_endpoint_connection_refused(self, mock_get):
        v = self._make_verifier()
        v.check_health_endpoint()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Dashboard ---

    @patch.object(
        fbv,
        "http_get",
        return_value=(
            200,
            '<html><title>Futures Trading Co-Pilot</title><div sse-connect="/sse/dashboard" hx-get="/api/focus"></div></html>',
            {},
        ),
    )
    def test_check_dashboard_pass(self, mock_get):
        v = self._make_verifier()
        v.check_dashboard()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(200, "<html>empty</html>", {}))
    def test_check_dashboard_no_markers(self, mock_get):
        v = self._make_verifier()
        v.check_dashboard()

        # 200 OK but no markers → FAIL
        result = v.report.results[0]
        assert result.status in (fbv.CheckStatus.FAIL, fbv.CheckStatus.WARN)

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_dashboard_404(self, mock_get):
        v = self._make_verifier()
        v.check_dashboard()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- API Info ---

    @patch.object(
        fbv,
        "http_get",
        return_value=(200, '{"service": "futures-data-service"}', {}),
    )
    def test_check_api_info_pass(self, mock_get):
        v = self._make_verifier()
        v.check_api_info()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(200, '{"other": true}', {}))
    def test_check_api_info_unexpected(self, mock_get):
        v = self._make_verifier()
        v.check_api_info()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- API Focus ---

    @patch.object(
        fbv,
        "http_get",
        return_value=(200, '{"assets": []}', {}),
    )
    def test_check_api_focus_pass(self, mock_get):
        v = self._make_verifier()
        v.check_api_focus()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(200, "null", {}))
    def test_check_api_focus_null(self, mock_get):
        v = self._make_verifier()
        v.check_api_focus()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_api_focus_404(self, mock_get):
        v = self._make_verifier()
        v.check_api_focus()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- SSE health ---

    @patch.object(fbv, "http_get", return_value=(200, '{"status": "ok"}', {}))
    def test_check_sse_health_pass(self, mock_get):
        v = self._make_verifier()
        v.check_sse_health()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(500, "error", {}))
    def test_check_sse_health_fail(self, mock_get):
        v = self._make_verifier()
        v.check_sse_health()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- SSE stream ---

    def test_check_sse_stream_skipped_in_quick_mode(self):
        v = self._make_verifier(quick=True)
        v.check_sse_stream()

        assert v.report.results[0].status == fbv.CheckStatus.SKIP

    @patch.object(
        fbv,
        "http_get",
        return_value=(
            200,
            "event: focus-update\ndata: {}\n\n",
            {"Content-Type": "text/event-stream"},
        ),
    )
    def test_check_sse_stream_pass(self, mock_get):
        v = self._make_verifier(quick=False)
        v.check_sse_stream()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    # --- Risk status ---

    @patch.object(
        fbv,
        "http_get",
        return_value=(200, '{"source": "redis", "can_trade": true}', {}),
    )
    def test_check_risk_status_pass(self, mock_get):
        v = self._make_verifier()
        v.check_risk_status()

        result = v.report.results[0]
        assert result.status == fbv.CheckStatus.PASS
        assert "redis" in result.detail

    @patch.object(fbv, "http_get", return_value=(500, "error", {}))
    def test_check_risk_status_fail(self, mock_get):
        v = self._make_verifier()
        v.check_risk_status()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Positions ---

    @patch.object(fbv, "http_get", return_value=(200, "[]", {}))
    def test_check_positions_200(self, mock_get):
        v = self._make_verifier()
        v.check_positions_api()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_positions_404(self, mock_get):
        v = self._make_verifier()
        v.check_positions_api()

        # 404 is acceptable on fresh boot
        assert v.report.results[0].status == fbv.CheckStatus.PASS

    # --- Prometheus metrics ---

    @patch.object(
        fbv,
        "http_get",
        return_value=(
            200,
            "# HELP http_requests_total\n# TYPE http_requests_total counter\nhttp_requests_total 42\n",
            {},
        ),
    )
    def test_check_prometheus_metrics_pass(self, mock_get):
        v = self._make_verifier()
        v.check_prometheus_metrics()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_prometheus_metrics_fail(self, mock_get):
        v = self._make_verifier()
        v.check_prometheus_metrics()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Engine health ---

    @patch.object(
        fbv,
        "docker_exec",
        return_value=(0, '{"healthy": true, "status": "running", "session": "active"}'),
    )
    def test_check_engine_health_pass(self, mock_exec):
        v = self._make_verifier()
        v.check_engine_health()

        result = v.report.results[0]
        assert result.status == fbv.CheckStatus.PASS
        assert "active" in result.detail

    @patch.object(
        fbv,
        "docker_exec",
        return_value=(0, '{"healthy": false, "status": "error"}'),
    )
    def test_check_engine_health_unhealthy(self, mock_exec):
        v = self._make_verifier()
        v.check_engine_health()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    @patch.object(fbv, "docker_exec", return_value=(1, "file not found"))
    def test_check_engine_health_no_file(self, mock_exec):
        v = self._make_verifier()
        v.check_engine_health()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    @patch.object(fbv, "docker_exec", return_value=(0, "not json"))
    def test_check_engine_health_invalid_json(self, mock_exec):
        v = self._make_verifier()
        v.check_engine_health()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Container logs ---

    @patch.object(
        fbv,
        "docker_logs_tail",
        return_value="INFO Starting up\nINFO Ready\nINFO Processing",
    )
    def test_check_container_logs_clean(self, mock_logs):
        v = self._make_verifier()
        v.check_container_logs("engine")

        result = v.report.results[0]
        assert result.status == fbv.CheckStatus.PASS
        assert "0 error" in result.detail

    @patch.object(
        fbv,
        "docker_logs_tail",
        return_value="INFO ok\nERROR something went wrong\nWARN check",
    )
    def test_check_container_logs_few_errors(self, mock_logs):
        v = self._make_verifier()
        v.check_container_logs("engine")

        # 1 error → WARN
        assert v.report.results[0].status == fbv.CheckStatus.WARN

    @patch.object(
        fbv,
        "docker_logs_tail",
        return_value="ERROR a\nERROR b\nERROR c\nException d\nTraceback e",
    )
    def test_check_container_logs_many_errors(self, mock_logs):
        v = self._make_verifier()
        v.check_container_logs("engine")

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    @patch.object(fbv, "docker_logs_tail", return_value="")
    def test_check_container_logs_empty(self, mock_logs):
        v = self._make_verifier()
        v.check_container_logs("engine")

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- No-trade endpoint ---

    @patch.object(fbv, "http_get", return_value=(200, "<div>no trade</div>", {}))
    def test_check_no_trade_pass(self, mock_get):
        v = self._make_verifier()
        v.check_no_trade_endpoint()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_no_trade_404(self, mock_get):
        v = self._make_verifier()
        v.check_no_trade_endpoint()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- Backfill status ---

    @patch.object(fbv, "http_get", return_value=(200, '{"symbols": []}', {}))
    def test_check_backfill_status_pass(self, mock_get):
        v = self._make_verifier()
        v.check_backfill_status()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_get", return_value=(404, "not found", {}))
    def test_check_backfill_status_404(self, mock_get):
        v = self._make_verifier()
        v.check_backfill_status()

        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- Streamlit retired ---

    @patch("subprocess.run")
    def test_check_streamlit_retired_pass(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="futures-postgres-1\nfutures-redis-1\nfutures-data-1\nfutures-engine-1\n",
            stderr="",
        )
        v = self._make_verifier()
        v.check_streamlit_retired()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch("subprocess.run")
    def test_check_streamlit_retired_still_running(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="futures-postgres-1\nfutures-redis-1\nfutures-data-1\nfutures-engine-1\nfutures-streamlit-1\n",
            stderr="",
        )
        v = self._make_verifier()
        v.check_streamlit_retired()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- DB write round-trip ---

    @patch("time.time", return_value=1700000000)
    @patch.object(fbv, "docker_exec")
    def test_check_db_write_roundtrip_pass(self, mock_exec, mock_time):
        test_value = "__first_boot_verify_1700000000__"
        mock_exec.side_effect = [
            (0, "42"),  # INSERT RETURNING id
            (0, test_value),  # SELECT
            (0, "DELETE 1"),  # DELETE
        ]
        v = self._make_verifier()
        v.check_db_write_roundtrip()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "docker_exec")
    def test_check_db_write_roundtrip_insert_fail(self, mock_exec):
        mock_exec.return_value = (1, "INSERT error")
        v = self._make_verifier()
        v.check_db_write_roundtrip()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Redis write round-trip ---

    @patch.object(fbv, "docker_exec")
    def test_check_redis_write_roundtrip_pass(self, mock_exec):
        mock_exec.side_effect = [
            (0, "OK"),  # SET
            (0, "hello_from_first_boot"),  # GET
            (0, "(integer) 1"),  # DEL
        ]
        v = self._make_verifier()
        v.check_redis_write_roundtrip()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "docker_exec")
    def test_check_redis_write_roundtrip_set_fail(self, mock_exec):
        mock_exec.return_value = (1, "error")
        v = self._make_verifier()
        v.check_redis_write_roundtrip()

        assert v.report.results[0].status == fbv.CheckStatus.FAIL

    # --- Cross-service Redis ---

    @patch.object(
        fbv,
        "http_get",
        side_effect=[
            (200, '{"status": "ok"}', {}),  # /health
            (200, '{"source": "redis", "can_trade": true}', {}),  # /risk/status
        ],
    )
    def test_check_cross_service_redis_pass(self, mock_get):
        v = self._make_verifier()
        v.check_cross_service_redis()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(
        fbv,
        "http_get",
        side_effect=[
            (200, '{"status": "ok"}', {}),  # /health
            (200, '{"source": "local"}', {}),  # /risk/status — local fallback
        ],
    )
    @patch.object(fbv, "docker_exec", return_value=(0, "(integer) 0"))
    def test_check_cross_service_redis_local_fallback(self, mock_exec, mock_get):
        v = self._make_verifier()
        v.check_cross_service_redis()

        # Local source → WARN
        assert v.report.results[0].status == fbv.CheckStatus.WARN

    # --- Risk pre-flight ---

    @patch.object(
        fbv,
        "http_post",
        return_value=(200, '{"allowed": true}', {}),
    )
    def test_check_risk_preflight_pass(self, mock_post):
        v = self._make_verifier()
        v.check_risk_preflight()

        assert v.report.results[0].status == fbv.CheckStatus.PASS

    @patch.object(fbv, "http_post", return_value=(503, "not available", {}))
    def test_check_risk_preflight_503(self, mock_post):
        v = self._make_verifier()
        v.check_risk_preflight()

        assert v.report.results[0].status == fbv.CheckStatus.WARN


# ===========================================================================
# Full run test (all mocked)
# ===========================================================================


class TestFullRun:
    """Test the full verification run with everything mocked."""

    @patch("time.time", return_value=1700000000)
    @patch.object(
        fbv, "detect_container_name", side_effect=lambda svc, default: default
    )
    @patch.object(fbv, "docker_inspect_running", return_value=(True, "running"))
    @patch.object(fbv, "docker_exec")
    @patch.object(fbv, "docker_logs_tail", return_value="INFO ok")
    @patch.object(fbv, "http_get")
    @patch.object(fbv, "http_post")
    @patch("subprocess.run")
    def test_all_pass_scenario(
        self,
        mock_subprocess,
        mock_http_post,
        mock_http_get,
        mock_logs,
        mock_docker_exec,
        mock_inspect,
        mock_detect,
        mock_time,
    ):
        """Simulate a scenario where everything is healthy."""
        test_value = "__first_boot_verify_1700000000__"
        # Docker exec responses in call order:
        #  1. check_postgres_ready
        #  2. check_redis_ping
        #  3. check_postgres_tables
        #  4-6. check_redis_engine_keys (3 keys)
        #  7-9. check_db_write_roundtrip (INSERT, SELECT, DELETE)
        #  10-12. check_redis_write_roundtrip (SET, GET, DEL)
        #  13. check_engine_health
        mock_docker_exec.side_effect = [
            # 1. postgres ready
            (0, "accepting connections"),
            # 2. redis ping
            (0, "PONG"),
            # 3. postgres tables
            (0, "daily_journal\nhistorical_bars\ntrades_v2\n"),
            # 4-6. redis engine keys (3 calls)
            (0, "(integer) 1"),
            (0, "(integer) 1"),
            (0, "(integer) 1"),
            # 7-9. db write roundtrip (3 calls)
            (0, "42"),
            (0, test_value),
            (0, "DELETE 1"),
            # 10-12. redis write roundtrip (3 calls)
            (0, "OK"),
            (0, "hello_from_first_boot"),
            (0, "(integer) 1"),
            # 13. engine health
            (0, '{"healthy": true, "status": "running", "session": "active"}'),
        ]

        # HTTP GET responses
        mock_http_get.side_effect = [
            # health
            (200, '{"status": "ok"}', {}),
            # dashboard
            (
                200,
                "<html><title>Futures Trading Co-Pilot</title>"
                '<div sse-connect="/sse" hx-get="/api" htmx></div></html>',
                {},
            ),
            # api info
            (200, '{"service": "futures-data-service"}', {}),
            # api focus
            (200, '{"assets": []}', {}),
            # risk status
            (200, '{"source": "redis", "can_trade": true}', {}),
            # positions
            (200, "[]", {}),
            # prometheus metrics
            (200, "# HELP http_requests_total\nhttp_requests_total 1\n", {}),
            # no-trade
            (200, "<div></div>", {}),
            # backfill status
            (200, '{"symbols": []}', {}),
            # sse health
            (200, '{"status": "ok"}', {}),
            # sse stream (skipped in quick mode — but we're not in quick)
            (
                200,
                "event: focus-update\ndata: {}\n",
                {"Content-Type": "text/event-stream"},
            ),
            # cross-service: /health
            (200, '{"status": "ok"}', {}),
            # cross-service: /risk/status
            (200, '{"source": "redis"}', {}),
        ]

        # HTTP POST
        mock_http_post.return_value = (200, '{"allowed": true}', {})

        # subprocess for streamlit check
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="futures-postgres-1\nfutures-redis-1\nfutures-data-1\nfutures-engine-1\n",
            stderr="",
        )

        v = fbv.FirstBootVerifier(quick=False)
        report = v.run()

        assert report.total > 0
        assert report.all_critical_pass is True
        # There should be no critical failures
        assert report.critical_failures == 0

    @patch.object(
        fbv, "detect_container_name", side_effect=lambda svc, default: default
    )
    @patch.object(fbv, "docker_inspect_running", return_value=(False, "exited"))
    @patch.object(fbv, "docker_exec", return_value=(1, "error"))
    @patch.object(fbv, "docker_logs_tail", return_value="")
    @patch.object(fbv, "http_get", return_value=(0, "Connection refused", {}))
    @patch.object(fbv, "http_post", return_value=(0, "Connection refused", {}))
    @patch("subprocess.run")
    def test_all_fail_scenario(
        self,
        mock_subprocess,
        mock_http_post,
        mock_http_get,
        mock_logs,
        mock_docker_exec,
        mock_inspect,
        mock_detect,
    ):
        """Simulate a scenario where everything is down."""
        mock_subprocess.return_value = MagicMock(
            returncode=1, stdout="", stderr="error"
        )

        v = fbv.FirstBootVerifier(quick=True)
        report = v.run()

        assert report.total > 0
        assert report.failed > 0
        assert report.critical_failures > 0
        assert report.all_critical_pass is False


# ===========================================================================
# Wait for services
# ===========================================================================


class TestWaitForServices:
    """Test the wait_for_services helper."""

    @patch.object(fbv, "http_get", return_value=(200, "ok", {}))
    def test_immediate_ready(self, mock_get):
        result = fbv.wait_for_services("http://localhost:8000", timeout_s=5)
        assert result is True

    @patch.object(fbv, "http_get")
    def test_eventual_ready(self, mock_get):
        # Fail twice, then succeed
        mock_get.side_effect = [
            (0, "refused", {}),
            (0, "refused", {}),
            (200, "ok", {}),
        ]
        result = fbv.wait_for_services("http://localhost:8000", timeout_s=30)
        assert result is True

    @patch.object(fbv, "http_get", return_value=(0, "refused", {}))
    @patch("time.monotonic")
    @patch("time.sleep")
    def test_timeout(self, mock_sleep, mock_monotonic, mock_get):
        # Simulate time passing beyond deadline
        mock_monotonic.side_effect = [
            0,
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ]
        result = fbv.wait_for_services("http://localhost:8000", timeout_s=5)
        assert result is False


# ===========================================================================
# Print summary (smoke test — just ensure it doesn't crash)
# ===========================================================================


class TestPrintSummary:
    """Test print_summary doesn't crash for various report states."""

    def _make_report(self, results):
        r = fbv.VerificationReport()
        r.results = results
        r.started_at = "2024-01-01T00:00:00Z"
        r.finished_at = "2024-01-01T00:01:00Z"
        r.duration_s = 60.0
        return r

    def _make_result(self, status, severity, name="test"):
        return fbv.CheckResult(
            name=name, status=status, severity=severity, detail="detail"
        )

    def test_all_pass(self, capsys):
        report = self._make_report(
            [self._make_result(fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL)]
        )
        fbv.print_summary(report)
        captured = capsys.readouterr()
        assert "PASSED" in captured.out or "Ready" in captured.out

    def test_critical_failure(self, capsys):
        report = self._make_report(
            [self._make_result(fbv.CheckStatus.FAIL, fbv.CheckSeverity.CRITICAL)]
        )
        fbv.print_summary(report)
        captured = capsys.readouterr()
        assert "FAILED" in captured.out or "DO NOT TRADE" in captured.out

    def test_mixed_results(self, capsys):
        report = self._make_report(
            [
                self._make_result(
                    fbv.CheckStatus.PASS, fbv.CheckSeverity.CRITICAL, "check1"
                ),
                self._make_result(
                    fbv.CheckStatus.WARN, fbv.CheckSeverity.IMPORTANT, "check2"
                ),
                self._make_result(
                    fbv.CheckStatus.SKIP, fbv.CheckSeverity.ADVISORY, "check3"
                ),
                self._make_result(
                    fbv.CheckStatus.FAIL, fbv.CheckSeverity.ADVISORY, "check4"
                ),
            ]
        )
        fbv.print_summary(report)
        captured = capsys.readouterr()
        # Should still say ready because no critical failures
        assert "Ready" in captured.out or "PASSED" in captured.out

    def test_empty_report(self, capsys):
        report = self._make_report([])
        fbv.print_summary(report)
        # Should not crash
        captured = capsys.readouterr()
        assert "Total" in captured.out or "0" in captured.out
