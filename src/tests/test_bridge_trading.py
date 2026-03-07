"""
Bridge Trading Tests — NT8 Bridge ``/flatten`` and ``/execute_signal`` flows.

Tests the full Python-side wiring for the Bridge proxy endpoints that the
dashboard uses to send commands *back* to NinjaTrader:

  POST /positions/flatten       — Flatten All button in dashboard
  POST /positions/execute       — Manual trade from dashboard
  POST /positions/cancel_orders — Cancel all working orders

These tests run **offline** (no NT8 or Bridge required) by mocking the
outbound ``httpx`` call to the Bridge listener and exercising the full
FastAPI endpoint logic — request validation, risk checks, heartbeat
liveness, and response shape.

Run with:
    cd futures
    python -m pytest src/tests/test_bridge_trading.py -v
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Stub out heavy dependencies that are not needed for these tests
# ---------------------------------------------------------------------------

# Redis / cache stubs
_fake_cache: dict[str, Any] = {}


def _fake_cache_get(key):
    return _fake_cache.get(key)


def _fake_cache_set(key, value, ttl=0):
    _fake_cache[key] = value


def _fake_cache_key(*parts):
    return ":".join(str(p) for p in parts)


def _fake_flush_all():
    _fake_cache.clear()


# Patch cache module before importing the router
_cache_mod = MagicMock()
_cache_mod.cache_get = _fake_cache_get
_cache_mod.cache_set = _fake_cache_set
_cache_mod._cache_key = _fake_cache_key
_cache_mod.flush_all = _fake_flush_all
_cache_mod.clear_cached_optimization = MagicMock()
_cache_mod.REDIS_AVAILABLE = True
sys.modules.setdefault("lib.core.cache", _cache_mod)

# Stub out models (ASSETS dict)
_models_mod = MagicMock()
_models_mod.ASSETS = {"MGC": "GCZ5", "MES": "MESZ5", "MNQ": "MNQZ5"}
sys.modules.setdefault("lib.core.models", _models_mod)

# Stub out risk module
_risk_mod = MagicMock()


def _fake_evaluate_risk(positions):
    """Basic risk evaluator stub."""
    total_pnl = sum(p.get("unrealizedPnL", 0) for p in positions)
    return {
        "can_trade": total_pnl > -2000,
        "daily_pnl": total_pnl,
        "block_reason": "daily_loss_limit" if total_pnl <= -2000 else "",
        "warnings": [],
    }


def _fake_check_entry_risk(symbol="", side="", size=1):
    return (True, "", {})


_risk_mod.evaluate_position_risk = _fake_evaluate_risk
_risk_mod.check_trade_entry_risk = _fake_check_entry_risk
sys.modules.setdefault("lib.services.data.api.risk", _risk_mod)

# Now import FastAPI test machinery
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the positions router
from lib.services.data.api.positions import router as positions_router

# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(positions_router, prefix="/positions")
    return app


def _clear():
    _fake_cache.clear()


def _inject_heartbeat(
    account: str = "Sim101",
    port: int = 5680,
    age_seconds: float = 5.0,
    risk_blocked: bool = False,
):
    """Plant a fresh heartbeat in the fake cache so Bridge appears alive."""
    received = (datetime.now(tz=_EST) - timedelta(seconds=age_seconds)).isoformat()
    hb = {
        "account": account,
        "state": "Realtime",
        "connected": True,
        "positions": 0,
        "cashBalance": 50000.0,
        "riskBlocked": risk_blocked,
        "bridge_version": "4.0",
        "listenerPort": port,
        "received_at": received,
        "timestamp": received,
    }
    key = _fake_cache_key("bridge_heartbeat", "current")
    _fake_cache[key] = json.dumps(hb)


def _inject_positions(positions: list[dict]):
    """Plant a position snapshot in the fake cache."""
    key = _fake_cache_key("live_positions", "current")
    data = {
        "account": "Sim101",
        "positions": positions,
        "pendingOrders": [],
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "received_at": datetime.now(tz=_EST).isoformat(),
        "cashBalance": 50000.0,
        "realizedPnL": 0.0,
        "totalUnrealizedPnL": 0.0,
        "riskBlocked": False,
        "riskBlockReason": "",
        "bridge_version": "4.0",
    }
    _fake_cache[key] = json.dumps(data).encode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    _clear()
    yield
    _clear()


@pytest.fixture
def client():
    app = _make_app()
    return TestClient(app)


@pytest.fixture
def live_client():
    """Client with a fresh heartbeat so Bridge appears alive."""
    app = _make_app()
    _inject_heartbeat()
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Flatten All — via POST /positions/flatten
# ═══════════════════════════════════════════════════════════════════════════


class TestFlattenAll:
    """Verify the Flatten All button flow: dashboard → data API → Bridge."""

    def test_flatten_rejected_when_bridge_offline(self, client):
        """When no heartbeat exists, flatten should return 503."""
        resp = client.post(
            "/positions/flatten",
            json={"reason": "dashboard"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "not connected" in data["detail"].lower() or "heartbeat" in data["detail"].lower()

    def test_flatten_rejected_when_heartbeat_stale(self, client):
        """When heartbeat is older than 60s, Bridge is considered dead."""
        _inject_heartbeat(age_seconds=120)
        resp = client.post(
            "/positions/flatten",
            json={"reason": "dashboard"},
        )
        assert resp.status_code == 503

    @patch("lib.services.data.api.positions.httpx")
    def test_flatten_forwards_to_bridge(self, mock_httpx, live_client):
        """When Bridge is alive, flatten should proxy POST to Bridge /flatten."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "flattened": 3,
            "reason": "dashboard",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = live_client.post(
            "/positions/flatten",
            json={"reason": "dashboard_panic"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["flattened"] == 3

        # Verify the proxy call went to the right URL
        call_args = mock_client_instance.post.call_args
        assert "/flatten" in call_args[0][0]
        body = call_args[1].get("json", {})
        assert body.get("reason") == "dashboard_panic"

    @patch("lib.services.data.api.positions.httpx")
    def test_flatten_uses_port_from_heartbeat(self, mock_httpx, client):
        """Bridge listener port comes from the heartbeat, not a hardcoded default."""
        _inject_heartbeat(port=9999)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "flattened": 0, "reason": "test"}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = client.post(
            "/positions/flatten",
            json={"reason": "port_test"},
        )
        assert resp.status_code == 200

        call_url = mock_client_instance.post.call_args[0][0]
        assert ":9999" in call_url

    @patch("lib.services.data.api.positions.httpx")
    def test_flatten_default_reason(self, mock_httpx, live_client):
        """Omitting `reason` defaults to 'dashboard'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "flattened": 1, "reason": "dashboard"}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        # Send with default reason
        resp = live_client.post("/positions/flatten", json={})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Execute Signal — via POST /positions/execute
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteSignal:
    """Verify manual trade from dashboard → data API → Bridge /execute_signal."""

    def test_execute_rejected_when_bridge_offline(self, client):
        """When no heartbeat, execute should return 503."""
        resp = client.post(
            "/positions/execute",
            json={"direction": "long", "asset": "MES"},
        )
        assert resp.status_code == 503

    @patch("lib.services.data.api.positions.httpx")
    def test_execute_basic_market_long(self, mock_httpx, live_client):
        """A simple long market order should be forwarded to Bridge."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "queued",
            "signal_id": "brg-abc12345",
            "bus_pending": 1,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = live_client.post(
            "/positions/execute",
            json={
                "direction": "long",
                "quantity": 2,
                "order_type": "market",
                "asset": "MES",
                "stop_loss": 5900.0,
                "take_profit": 5950.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "signal_id" in data

        # Verify what was sent to Bridge
        call_args = mock_client_instance.post.call_args
        assert "/execute_signal" in call_args[0][0]
        payload = call_args[1].get("json", {})
        assert payload["direction"] == "long"
        assert payload["quantity"] == 2
        assert payload["asset"] == "MES"

    @patch("lib.services.data.api.positions.httpx")
    def test_execute_short_with_all_fields(self, mock_httpx, live_client):
        """A short entry with SL, TP, TP2, limit price."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "queued",
            "signal_id": "brg-short1",
            "bus_pending": 1,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = live_client.post(
            "/positions/execute",
            json={
                "direction": "short",
                "quantity": 1,
                "order_type": "limit",
                "limit_price": 2750.0,
                "stop_loss": 2770.0,
                "take_profit": 2720.0,
                "tp2": 2700.0,
                "asset": "MGC",
                "strategy": "ManualDashboard",
                "signal_id": "manual-short-1",
                "enforce_risk": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

    def test_execute_requires_direction(self, live_client):
        """Direction is required — omitting it should fail validation."""
        resp = live_client.post(
            "/positions/execute",
            json={"quantity": 1},
        )
        assert resp.status_code == 422  # Pydantic validation error

    @patch("lib.services.data.api.positions.httpx")
    def test_execute_with_risk_check_failure(self, mock_httpx, live_client):
        """When pre-flight risk check denies entry, signal should be rejected."""
        # Override the risk check to deny
        original_check = _risk_mod.check_trade_entry_risk

        def _deny_entry(symbol="", side="", size=1):
            return (False, "daily_loss_limit_reached", {"daily_pnl": -2100})

        _risk_mod.check_trade_entry_risk = _deny_entry

        try:
            resp = live_client.post(
                "/positions/execute",
                json={
                    "direction": "long",
                    "asset": "MES",
                    "enforce_risk": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "rejected"
            assert "risk" in data.get("reason", "").lower()
        finally:
            _risk_mod.check_trade_entry_risk = original_check

    @patch("lib.services.data.api.positions.httpx")
    def test_execute_skip_risk_check(self, mock_httpx, live_client):
        """When enforce_risk=False, risk check should be skipped."""
        original_check = _risk_mod.check_trade_entry_risk

        def _deny_entry(symbol="", side="", size=1):
            return (False, "should_not_reach", {})

        _risk_mod.check_trade_entry_risk = _deny_entry

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "queued",
            "signal_id": "brg-noskip",
            "bus_pending": 1,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        try:
            resp = live_client.post(
                "/positions/execute",
                json={
                    "direction": "long",
                    "asset": "MES",
                    "enforce_risk": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            # Should succeed because risk check was skipped
            assert data["status"] == "queued"
        finally:
            _risk_mod.check_trade_entry_risk = original_check


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Cancel Orders — via POST /positions/cancel_orders
# ═══════════════════════════════════════════════════════════════════════════


class TestCancelOrders:
    """Verify the Cancel Orders button flow."""

    def test_cancel_rejected_when_bridge_offline(self, client):
        resp = client.post("/positions/cancel_orders")
        assert resp.status_code == 503

    @patch("lib.services.data.api.positions.httpx")
    def test_cancel_forwards_to_bridge(self, mock_httpx, live_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "cancelled": 2}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = live_client.post("/positions/cancel_orders")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["cancelled"] == 2

        call_url = mock_client_instance.post.call_args[0][0]
        assert "/cancel_orders" in call_url


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Position Updates — via POST /positions/update
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionUpdate:
    """Verify position snapshot ingestion and risk evaluation."""

    def _make_payload(
        self,
        positions: list[dict] | None = None,
        account: str = "Sim101",
        cash: float = 50000.0,
        risk_blocked: bool = False,
    ) -> dict:
        return {
            "account": account,
            "positions": positions or [],
            "pendingOrders": [],
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "cashBalance": cash,
            "realizedPnL": 0.0,
            "totalUnrealizedPnL": sum(p.get("unrealizedPnL", 0) for p in (positions or [])),
            "riskBlocked": risk_blocked,
            "riskBlockReason": "",
            "bridge_version": "4.0",
        }

    def test_update_empty_positions(self, client):
        """Push with no positions should succeed."""
        resp = client.post(
            "/positions/update",
            json=self._make_payload(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"
        assert data["positions_count"] == 0
        assert data["open_positions"] == 0

    def test_update_with_positions(self, client):
        """Push with active positions should return correct counts."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 2,
                "avgPrice": 5925.50,
                "unrealizedPnL": 125.0,
            },
            {
                "symbol": "MGCZ5",
                "side": "Short",
                "quantity": 1,
                "avgPrice": 2745.0,
                "unrealizedPnL": -30.0,
            },
        ]
        resp = client.post(
            "/positions/update",
            json=self._make_payload(positions=positions),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["positions_count"] == 2
        assert data["open_positions"] == 2
        assert data["total_unrealized_pnl"] == 95.0  # 125 + (-30)

    def test_update_caches_positions(self, client):
        """Position update should be readable from the cache."""
        positions = [
            {
                "symbol": "MNQZ5",
                "side": "Long",
                "quantity": 3,
                "avgPrice": 21000.0,
                "unrealizedPnL": 200.0,
            }
        ]
        client.post(
            "/positions/update",
            json=self._make_payload(positions=positions),
        )

        # Read back from cache
        key = _fake_cache_key("live_positions", "current")
        raw = _fake_cache.get(key)
        assert raw is not None
        cached = json.loads(raw if isinstance(raw, str) else raw.decode())
        assert cached["account"] == "Sim101"
        assert len(cached["positions"]) == 1
        assert cached["positions"][0]["symbol"] == "MNQZ5"

    def test_update_includes_risk_evaluation(self, client):
        """The response should include risk status."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 1,
                "avgPrice": 5900.0,
                "unrealizedPnL": -100.0,
            }
        ]
        resp = client.post(
            "/positions/update",
            json=self._make_payload(positions=positions),
        )
        data = resp.json()
        risk = data.get("risk", {})
        assert "can_trade" in risk
        assert risk["can_trade"] is True  # -100 is within limits

    def test_update_risk_blocks_on_large_loss(self, client):
        """If daily loss exceeds limit, risk should block trading."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 5,
                "avgPrice": 5900.0,
                "unrealizedPnL": -2500.0,
            }
        ]
        resp = client.post(
            "/positions/update",
            json=self._make_payload(positions=positions),
        )
        data = resp.json()
        risk = data.get("risk", {})
        assert risk["can_trade"] is False
        assert "loss" in risk.get("block_reason", "").lower()

    def test_update_requires_account(self, client):
        """Missing account field should fail validation."""
        resp = client.post(
            "/positions/update",
            json={
                "positions": [],
                "pendingOrders": [],
            },
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Heartbeat — via POST /positions/heartbeat
# ═══════════════════════════════════════════════════════════════════════════


class TestHeartbeat:
    """Verify Bridge heartbeat reception and caching."""

    def _make_heartbeat(
        self,
        account: str = "Sim101",
        port: int = 5680,
        positions: int = 0,
    ) -> dict:
        return {
            "account": account,
            "state": "Realtime",
            "connected": True,
            "positions": positions,
            "cashBalance": 50000.0,
            "riskBlocked": False,
            "bridge_version": "4.0",
            "listenerPort": port,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }

    def test_heartbeat_stores_in_cache(self, client):
        resp = client.post(
            "/positions/heartbeat",
            json=self._make_heartbeat(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

        # Verify cached
        key = _fake_cache_key("bridge_heartbeat", "current")
        raw = _fake_cache.get(key)
        assert raw is not None
        hb = json.loads(raw)
        assert hb["account"] == "Sim101"
        assert hb["listenerPort"] == 5680

    def test_heartbeat_with_custom_port(self, client):
        """Bridge port from heartbeat should be usable by proxy endpoints."""
        client.post(
            "/positions/heartbeat",
            json=self._make_heartbeat(port=9876),
        )

        key = _fake_cache_key("bridge_heartbeat", "current")
        raw = _fake_cache.get(key)
        assert raw is not None
        hb = json.loads(raw)
        assert hb["listenerPort"] == 9876

    def test_heartbeat_updates_liveness(self, client):
        """After a heartbeat, the bridge should be considered alive."""
        # First, no heartbeat — bridge is dead
        _clear()
        _inject_heartbeat(age_seconds=120)  # stale

        # Send fresh heartbeat
        client.post(
            "/positions/heartbeat",
            json=self._make_heartbeat(),
        )

        # Now the heartbeat key should be fresh
        key = _fake_cache_key("bridge_heartbeat", "current")
        raw = _fake_cache.get(key)
        assert raw is not None
        hb = json.loads(raw)
        assert "received_at" in hb


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Bridge Status — via GET /positions/bridge_status
# ═══════════════════════════════════════════════════════════════════════════


class TestBridgeStatus:
    """Verify the GET /positions/bridge_status endpoint."""

    @patch("lib.services.data.api.positions.httpx")
    def test_status_when_bridge_alive(self, mock_httpx, live_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bridge_version": "4.0",
            "rubyAttached": True,
            "signalBusActive": True,
            "signalBusPending": 0,
            "account": "Sim101",
            "positions": 1,
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance

        resp = live_client.get("/positions/bridge_status")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Get Positions — via GET /positions/
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPositions:
    """Verify reading cached positions back."""

    def test_get_no_positions(self, client):
        resp = client.get("/positions/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_positions"] is False
        assert data["positions"] == []

    def test_get_after_push(self, client):
        """After pushing positions, GET should return them."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 1,
                "avgPrice": 5920.0,
                "unrealizedPnL": 50.0,
            }
        ]
        client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": positions,
                "pendingOrders": [],
                "cashBalance": 50000.0,
                "realizedPnL": 0.0,
                "totalUnrealizedPnL": 50.0,
                "riskBlocked": False,
                "riskBlockReason": "",
                "bridge_version": "4.0",
            },
        )

        resp = client.get("/positions/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_positions"] is True
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "MESZ5"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Connection Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestBridgeConnectionErrors:
    """Verify graceful handling of Bridge communication failures."""

    @patch("lib.services.data.api.positions.httpx")
    def test_flatten_bridge_unreachable(self, mock_httpx, live_client):
        """When Bridge is unreachable, should return 503."""
        import httpx as real_httpx

        mock_client_instance = MagicMock()
        mock_client_instance.post.side_effect = real_httpx.ConnectError("Connection refused")
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance
        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException

        resp = live_client.post(
            "/positions/flatten",
            json={"reason": "test"},
        )
        assert resp.status_code == 503

    @patch("lib.services.data.api.positions.httpx")
    def test_execute_bridge_timeout(self, mock_httpx, live_client):
        """When Bridge times out, should return 504."""
        import httpx as real_httpx

        mock_client_instance = MagicMock()
        mock_client_instance.post.side_effect = real_httpx.TimeoutException("Read timed out")
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx.Client.return_value = mock_client_instance
        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException

        resp = live_client.post(
            "/positions/execute",
            json={"direction": "long", "asset": "MES"},
        )
        assert resp.status_code == 504


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Multiple Concurrent Positions
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrentPositions:
    """
    Verify that position tracking handles multiple concurrent entries.

    While the MaxConcurrentPositions=5 gate is enforced in BreakoutStrategy.cs
    (C# side), the Python-side position ingestion and display must correctly
    track all N positions simultaneously — including both automated and
    manual (Bridge) entries.
    """

    def test_five_concurrent_positions(self, client):
        """Push 5 open positions and verify all are tracked."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long" if i % 2 == 0 else "Short",
                "quantity": i + 1,
                "avgPrice": 5900.0 + i * 10,
                "unrealizedPnL": (i - 2) * 50.0,
                "instrument": "MES",
            }
            for i in range(5)
        ]
        total_pnl = sum(float(p["unrealizedPnL"]) for p in positions)
        resp = client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": positions,
                "pendingOrders": [],
                "cashBalance": 50000.0,
                "realizedPnL": 0.0,
                "totalUnrealizedPnL": total_pnl,
                "riskBlocked": False,
                "riskBlockReason": "",
                "bridge_version": "4.0",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["open_positions"] == 5
        assert data["positions_count"] == 5

    def test_mixed_automated_and_manual_positions(self, client):
        """
        Both automated (BreakoutStrategy) and manual (Bridge) entries
        should appear in the same position snapshot and be tracked together.
        """
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 2,
                "avgPrice": 5920.0,
                "unrealizedPnL": 80.0,
                "instrument": "MES",
            },
            {
                "symbol": "MGCZ5",
                "side": "Long",
                "quantity": 1,
                "avgPrice": 2750.0,
                "unrealizedPnL": 15.0,
                "instrument": "MGC",
            },
            {
                "symbol": "MNQZ5",
                "side": "Short",
                "quantity": 3,
                "avgPrice": 21100.0,
                "unrealizedPnL": -45.0,
                "instrument": "MNQ",
            },
        ]
        resp = client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": positions,
                "pendingOrders": [],
                "cashBalance": 50000.0,
                "realizedPnL": 150.0,
                "totalUnrealizedPnL": 50.0,
                "riskBlocked": False,
                "riskBlockReason": "",
                "bridge_version": "4.0",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["open_positions"] == 3
        total_pnl = data["total_unrealized_pnl"]
        assert abs(total_pnl - 50.0) < 0.01

    def test_position_with_pending_orders(self, client):
        """Push positions WITH pending bracket orders (SL/TP)."""
        positions = [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 2,
                "avgPrice": 5920.0,
                "unrealizedPnL": 60.0,
            }
        ]
        pending_orders = [
            {
                "orderId": "ORD-001",
                "name": "SL-brg-abc123",
                "instrument": "MESZ5",
                "action": "Sell",
                "type": "StopMarket",
                "quantity": 2,
                "limitPrice": 0,
                "stopPrice": 5900.0,
                "state": "Working",
            },
            {
                "orderId": "ORD-002",
                "name": "TP1-brg-abc123",
                "instrument": "MESZ5",
                "action": "Sell",
                "type": "Limit",
                "quantity": 1,
                "limitPrice": 5950.0,
                "stopPrice": 0,
                "state": "Working",
            },
        ]
        resp = client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": positions,
                "pendingOrders": pending_orders,
                "cashBalance": 50000.0,
                "realizedPnL": 0.0,
                "totalUnrealizedPnL": 60.0,
                "riskBlocked": False,
                "riskBlockReason": "",
                "bridge_version": "4.0",
            },
        )
        assert resp.status_code == 200

        # Verify orders are cached alongside positions
        key = _fake_cache_key("live_positions", "current")
        raw = _fake_cache.get(key)
        assert raw is not None
        cached = json.loads(raw)
        assert len(cached["pendingOrders"]) == 2
        assert cached["pendingOrders"][0]["name"] == "SL-brg-abc123"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Dashboard Render Context (position panel data for flatten button)
# ═══════════════════════════════════════════════════════════════════════════


class TestDashboardBridgeContext:
    """
    Verify that the data used to render the dashboard positions panel
    (including flatten/cancel buttons and Bridge status dot) is correctly
    assembled from cache.
    """

    def test_bridge_info_when_connected(self):
        """_get_bridge_info should return connected=True with fresh heartbeat."""
        _inject_heartbeat(age_seconds=5)

        # Import the dashboard helper
        from lib.services.data.api.dashboard import _get_bridge_info

        info = _get_bridge_info()
        assert info["connected"] is True
        assert info["age_seconds"] < 60
        assert info["account"] == "Sim101"

    def test_bridge_info_when_disconnected(self):
        """_get_bridge_info should return connected=False with no heartbeat."""
        _clear()

        from lib.services.data.api.dashboard import _get_bridge_info

        info = _get_bridge_info()
        assert info["connected"] is False

    def test_bridge_info_stale_heartbeat(self):
        """_get_bridge_info should return connected=False when heartbeat > 60s old."""
        _inject_heartbeat(age_seconds=90)

        from lib.services.data.api.dashboard import _get_bridge_info

        info = _get_bridge_info()
        assert info["connected"] is False
        assert info["age_seconds"] > 60


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Request Model Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRequestValidation:
    """Ensure Pydantic models enforce constraints."""

    def test_execute_quantity_must_be_positive(self, live_client):
        resp = live_client.post(
            "/positions/execute",
            json={"direction": "long", "quantity": 0, "asset": "MES"},
        )
        assert resp.status_code == 422

    def test_execute_direction_required(self, live_client):
        resp = live_client.post(
            "/positions/execute",
            json={"asset": "MES"},
        )
        assert resp.status_code == 422

    def test_update_position_missing_symbol(self, client):
        resp = client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "side": "Long",
                        "quantity": 1,
                        "avgPrice": 5900.0,
                    }
                ],
                "pendingOrders": [],
            },
        )
        assert resp.status_code == 422

    def test_heartbeat_requires_account(self, client):
        resp = client.post(
            "/positions/heartbeat",
            json={
                "state": "Realtime",
                "connected": True,
                "positions": 0,
            },
        )
        assert resp.status_code == 422
