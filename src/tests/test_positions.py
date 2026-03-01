"""
Tests for the NinjaTrader Live Position Bridge API.

Covers:
  - POST /update_positions — push positions from NinjaTrader
  - GET  /positions        — read current positions
  - DELETE /positions      — clear stale positions
  - get_live_positions()   — direct cache read helper
  - Edge cases: empty positions, malformed payloads, cache expiry, etc.
  - Integration with Grok context builder (positions in market context)
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_position_cache():
    """Clear the position cache before and after each test.

    Also temporarily disables Redis so that cache_get/cache_set use the
    in-memory ``_mem_cache`` dict, which the test controls.
    """
    import src.lib.core.cache as _cache_mod

    # Save original state
    original_mem = dict(_cache_mod._mem_cache)
    original_redis_available = _cache_mod.REDIS_AVAILABLE
    original_r = _cache_mod._r

    # Force in-memory mode
    _cache_mod.REDIS_AVAILABLE = False
    _cache_mod._r = None
    _cache_mod._mem_cache.clear()

    yield

    # Restore
    _cache_mod._mem_cache.clear()
    _cache_mod._mem_cache.update(original_mem)
    _cache_mod.REDIS_AVAILABLE = original_redis_available
    _cache_mod._r = original_r


@pytest.fixture()
def client():
    """FastAPI test client for the trade API."""
    from lib.api_server import app

    return TestClient(app)


@pytest.fixture()
def sample_payload():
    """A realistic NinjaTrader position payload."""
    return {
        "account": "Sim101",
        "positions": [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 5,
                "avgPrice": 6045.25,
                "unrealizedPnL": 125.00,
                "lastUpdate": "2025-01-15T14:30:00Z",
            },
            {
                "symbol": "MNQZ5",
                "side": "Short",
                "quantity": 3,
                "avgPrice": 21580.00,
                "unrealizedPnL": -42.00,
                "lastUpdate": "2025-01-15T14:30:00Z",
            },
        ],
        "timestamp": "2025-01-15T14:30:00Z",
    }


@pytest.fixture()
def single_position_payload():
    """Payload with a single position."""
    return {
        "account": "Live001",
        "positions": [
            {
                "symbol": "MCLZ5",
                "side": "Long",
                "quantity": 2,
                "avgPrice": 71.45,
                "unrealizedPnL": 30.00,
            },
        ],
        "timestamp": "2025-01-15T15:00:00Z",
    }


@pytest.fixture()
def empty_payload():
    """Payload with no open positions (all closed)."""
    return {
        "account": "Sim101",
        "positions": [],
        "timestamp": "2025-01-15T16:00:00Z",
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST /update_positions
# ═══════════════════════════════════════════════════════════════════════════


class TestUpdatePositions:
    """Tests for the POST /update_positions endpoint."""

    def test_post_positions_success(self, client, sample_payload):
        """Basic successful position push."""
        resp = client.post("/update_positions", json=sample_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["positions_received"] == 2
        assert body["open_positions"] == 2
        assert body["total_unrealized_pnl"] == 83.00  # 125 + (-42)
        assert "received_at" in body

    def test_post_single_position(self, client, single_position_payload):
        """Push a single position."""
        resp = client.post("/update_positions", json=single_position_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_received"] == 1
        assert body["open_positions"] == 1
        assert body["total_unrealized_pnl"] == 30.00

    def test_post_empty_positions(self, client, empty_payload):
        """Push an empty positions list (all positions closed)."""
        resp = client.post("/update_positions", json=empty_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_received"] == 0
        assert body["open_positions"] == 0
        assert body["total_unrealized_pnl"] == 0.0

    def test_post_updates_cache(self, client, sample_payload):
        """Verify that POST actually writes to the cache."""
        from lib.api_server import get_live_positions

        # Before: no positions
        before = get_live_positions()
        assert before["has_positions"] is False

        # Push
        client.post("/update_positions", json=sample_payload)

        # After: positions present
        after = get_live_positions()
        assert after["has_positions"] is True
        assert after["account"] == "Sim101"
        assert len(after["positions"]) == 2

    def test_post_overwrites_previous(
        self, client, sample_payload, single_position_payload
    ):
        """A new POST replaces the previous positions snapshot."""
        from lib.api_server import get_live_positions

        # First push: 2 positions
        client.post("/update_positions", json=sample_payload)
        assert len(get_live_positions()["positions"]) == 2

        # Second push: 1 position (replaces, not appends)
        client.post("/update_positions", json=single_position_payload)
        result = get_live_positions()
        assert len(result["positions"]) == 1
        assert result["account"] == "Live001"

    def test_post_negative_pnl(self, client):
        """Positions with negative unrealized PnL."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 10,
                    "avgPrice": 6100.00,
                    "unrealizedPnL": -500.00,
                },
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 200
        assert resp.json()["total_unrealized_pnl"] == -500.00

    def test_post_zero_quantity_excluded(self, client):
        """Positions with quantity=0 are still accepted by API but counted correctly."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 0,
                    "avgPrice": 6045.25,
                    "unrealizedPnL": 0.0,
                },
                {
                    "symbol": "MNQZ5",
                    "side": "Short",
                    "quantity": 3,
                    "avgPrice": 21580.00,
                    "unrealizedPnL": -20.0,
                },
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        # Total received includes both, but open count only counts qty > 0
        assert body["positions_received"] == 2
        assert body["open_positions"] == 1

    def test_post_missing_account_fails(self, client):
        """Missing required 'account' field should fail validation."""
        payload = {
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 6000.00,
                    "unrealizedPnL": 0,
                }
            ],
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_post_missing_position_fields_fails(self, client):
        """Missing required position fields should fail validation."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    # missing side, quantity, avgPrice
                }
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 422

    def test_post_optional_fields_default(self, client):
        """Optional fields should default gracefully."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 6000.0,
                    # unrealizedPnL and lastUpdate are optional
                },
            ],
            # timestamp is optional
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_received"] == 1
        assert body["total_unrealized_pnl"] == 0.0  # default

    def test_post_large_position_count(self, client):
        """Handle a large number of positions (e.g., 20 micro contracts)."""
        positions = []
        for i in range(20):
            positions.append(
                {
                    "symbol": f"MES{chr(65 + i % 6)}5",
                    "side": "Long" if i % 2 == 0 else "Short",
                    "quantity": i + 1,
                    "avgPrice": 6000.0 + i * 10,
                    "unrealizedPnL": (i - 10) * 25.0,
                }
            )
        payload = {
            "account": "Sim101",
            "positions": positions,
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/update_positions", json=payload)
        assert resp.status_code == 200
        assert resp.json()["positions_received"] == 20


# ═══════════════════════════════════════════════════════════════════════════
# GET /positions
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPositions:
    """Tests for the GET /positions endpoint."""

    def test_get_no_positions(self, client):
        """GET with no cached positions returns empty response."""
        resp = client.get("/positions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_positions"] is False
        assert body["positions"] == []
        assert body["account"] == ""
        assert body["total_unrealized_pnl"] == 0.0

    def test_get_after_push(self, client, sample_payload):
        """GET returns the positions pushed by POST."""
        client.post("/update_positions", json=sample_payload)

        resp = client.get("/positions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_positions"] is True
        assert body["account"] == "Sim101"
        assert len(body["positions"]) == 2
        assert body["total_unrealized_pnl"] == 83.00
        assert body["received_at"] != ""

    def test_get_position_details(self, client, sample_payload):
        """Verify individual position fields are preserved."""
        client.post("/update_positions", json=sample_payload)

        resp = client.get("/positions")
        positions = resp.json()["positions"]

        mes = next(p for p in positions if p["symbol"] == "MESZ5")
        assert mes["side"] == "Long"
        assert mes["quantity"] == 5
        assert mes["avgPrice"] == 6045.25
        assert mes["unrealizedPnL"] == 125.00

        mnq = next(p for p in positions if p["symbol"] == "MNQZ5")
        assert mnq["side"] == "Short"
        assert mnq["quantity"] == 3
        assert mnq["avgPrice"] == 21580.00
        assert mnq["unrealizedPnL"] == -42.00

    def test_get_after_clear(self, client, sample_payload):
        """GET returns empty after DELETE /positions."""
        client.post("/update_positions", json=sample_payload)
        client.delete("/positions")

        resp = client.get("/positions")
        body = resp.json()
        assert body["has_positions"] is False
        assert body["positions"] == []


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /positions
# ═══════════════════════════════════════════════════════════════════════════


class TestClearPositions:
    """Tests for the DELETE /positions endpoint."""

    def test_clear_when_empty(self, client):
        """DELETE on empty cache succeeds without error."""
        resp = client.delete("/positions")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

    def test_clear_removes_data(self, client, sample_payload):
        """DELETE removes cached positions."""
        from lib.api_server import get_live_positions

        client.post("/update_positions", json=sample_payload)
        assert get_live_positions()["has_positions"] is True

        client.delete("/positions")
        assert get_live_positions()["has_positions"] is False

    def test_clear_idempotent(self, client, sample_payload):
        """Multiple DELETEs don't cause errors."""
        client.post("/update_positions", json=sample_payload)
        client.delete("/positions")
        client.delete("/positions")
        client.delete("/positions")

        resp = client.get("/positions")
        assert resp.json()["has_positions"] is False


# ═══════════════════════════════════════════════════════════════════════════
# get_live_positions() helper
# ═══════════════════════════════════════════════════════════════════════════


class TestGetLivePositionsHelper:
    """Tests for the get_live_positions() direct cache reader."""

    def test_no_data(self):
        """Returns empty dict structure when cache is empty."""
        from lib.api_server import get_live_positions

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []
        assert result["account"] == ""
        assert result["total_unrealized_pnl"] == 0.0
        assert result["timestamp"] == ""
        assert result["received_at"] == ""

    def test_after_push(self, client, sample_payload):
        """Returns correct data after a push."""
        from lib.api_server import get_live_positions

        client.post("/update_positions", json=sample_payload)

        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["account"] == "Sim101"
        assert len(result["positions"]) == 2
        assert result["total_unrealized_pnl"] == 83.00

    def test_pnl_calculation(self, client):
        """Total unrealized PnL is computed correctly across positions."""
        from lib.api_server import get_live_positions

        payload = {
            "account": "Test",
            "positions": [
                {
                    "symbol": "A",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 100,
                    "unrealizedPnL": 50.0,
                },
                {
                    "symbol": "B",
                    "side": "Short",
                    "quantity": 2,
                    "avgPrice": 200,
                    "unrealizedPnL": -30.0,
                },
                {
                    "symbol": "C",
                    "side": "Long",
                    "quantity": 3,
                    "avgPrice": 300,
                    "unrealizedPnL": 100.0,
                },
            ],
        }
        client.post("/update_positions", json=payload)

        result = get_live_positions()
        assert result["total_unrealized_pnl"] == 120.0  # 50 + (-30) + 100

    def test_corrupt_cache_data(self):
        """Handles corrupt cache data gracefully."""
        from lib.api_server import _POSITIONS_CACHE_KEY, get_live_positions
        from lib.core.cache import cache_set

        cache_set(_POSITIONS_CACHE_KEY, b"not valid json{{{", 60)

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []

    def test_partial_cache_data(self):
        """Handles cache data with missing fields gracefully."""
        from lib.api_server import _POSITIONS_CACHE_KEY, get_live_positions
        from lib.core.cache import cache_set

        data = json.dumps({"account": "Test"}).encode()  # missing positions key
        cache_set(_POSITIONS_CACHE_KEY, data, 60)

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []
        assert result["account"] == "Test"


# ═══════════════════════════════════════════════════════════════════════════
# Health endpoint with bridge status
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthWithBridge:
    """Tests for the /health endpoint's NT bridge status."""

    def test_health_no_positions(self, client):
        """Health shows bridge as disconnected when no positions."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "nt_bridge" in body
        assert body["nt_bridge"]["connected"] is False
        assert body["nt_bridge"]["open_positions"] == 0

    def test_health_with_positions(self, client, sample_payload):
        """Health shows bridge as connected when positions are present."""
        client.post("/update_positions", json=sample_payload)

        resp = client.get("/health")
        body = resp.json()
        assert body["nt_bridge"]["connected"] is True
        assert body["nt_bridge"]["account"] == "Sim101"
        assert body["nt_bridge"]["open_positions"] == 2
        assert body["nt_bridge"]["last_update"] != ""


# ═══════════════════════════════════════════════════════════════════════════
# Grok context integration
# ═══════════════════════════════════════════════════════════════════════════


class TestGrokContextIntegration:
    """Tests for live positions in Grok market context."""

    def test_context_without_positions(self):
        """Market context without positions shows 'No live positions'."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
        )
        assert ctx["positions_text"] == "No live positions"
        assert ctx["has_positions"] is False

    def test_context_with_positions(self):
        """Market context includes position details when available."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": True,
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 5,
                    "avgPrice": 6045.25,
                    "unrealizedPnL": 125.00,
                },
            ],
            "total_unrealized_pnl": 125.00,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is True
        assert "MESZ5" in ctx["positions_text"]
        assert "Long" in ctx["positions_text"]
        assert "Sim101" in ctx["positions_text"]
        assert "125" in ctx["positions_text"]

    def test_context_with_negative_pnl(self):
        """Market context formats negative PnL correctly."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": True,
            "account": "Live001",
            "positions": [
                {
                    "symbol": "MCLZ5",
                    "side": "Short",
                    "quantity": 2,
                    "avgPrice": 72.00,
                    "unrealizedPnL": -150.00,
                },
            ],
            "total_unrealized_pnl": -150.00,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is True
        assert "🔴" in ctx["positions_text"]
        assert "-150" in ctx["positions_text"]

    def test_context_with_empty_positions(self):
        """Empty positions list means no positions flag."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": False,
            "account": "Sim101",
            "positions": [],
            "total_unrealized_pnl": 0.0,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is False
        assert ctx["positions_text"] == "No live positions"

    def test_context_positions_none(self):
        """Passing None for live_positions is handled gracefully."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=None,
        )
        assert ctx["has_positions"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic model validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModelValidation:
    """Tests for Pydantic request/response models."""

    def test_nt_position_model(self):
        """NTPosition model accepts valid data."""
        from lib.api_server import NTPosition

        pos = NTPosition(
            symbol="MESZ5",
            side="Long",
            quantity=5,
            avgPrice=6045.25,
            unrealizedPnL=125.0,
            lastUpdate="2025-01-15T14:30:00Z",
        )
        assert pos.symbol == "MESZ5"
        assert pos.side == "Long"
        assert pos.quantity == 5
        assert pos.avgPrice == 6045.25

    def test_nt_position_defaults(self):
        """NTPosition model defaults for optional fields."""
        from lib.api_server import NTPosition

        pos = NTPosition(
            symbol="MESZ5",
            side="Long",
            quantity=1,
            avgPrice=6000.0,
        )
        assert pos.unrealizedPnL == 0.0
        assert pos.lastUpdate is None

    def test_nt_payload_model(self):
        """NTPositionsPayload model validates correctly."""
        from lib.api_server import NTPosition, NTPositionsPayload

        payload = NTPositionsPayload(
            account="Sim101",
            positions=[
                NTPosition(
                    symbol="MESZ5",
                    side="Long",
                    quantity=5,
                    avgPrice=6045.25,
                ),
            ],
            timestamp="2025-01-15T14:30:00Z",
        )
        assert payload.account == "Sim101"
        assert len(payload.positions) == 1

    def test_nt_payload_empty_positions(self):
        """Payload with empty positions list is valid."""
        from lib.api_server import NTPositionsPayload

        payload = NTPositionsPayload(
            account="Sim101",
            positions=[],
        )
        assert payload.account == "Sim101"
        assert len(payload.positions) == 0

    def test_nt_response_model(self):
        """NTPositionsResponse model works with defaults."""
        from lib.api_server import NTPositionsResponse

        resp = NTPositionsResponse()
        assert resp.account == ""
        assert resp.positions == []
        assert resp.has_positions is False
        assert resp.total_unrealized_pnl == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Existing trade endpoints still work (regression)
# ═══════════════════════════════════════════════════════════════════════════


class TestExistingEndpointsRegression:
    """Verify that existing trade endpoints still work after position bridge addition."""

    def test_health_still_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_accounts_still_works(self, client):
        resp = client.get("/accounts")
        assert resp.status_code == 200
        data = resp.json()
        assert "50k" in data
        assert "100k" in data
        assert "150k" in data

    def test_assets_still_works(self, client):
        resp = client.get("/assets")
        assert resp.status_code == 200
        data = resp.json()
        assert "Gold" in data or "S&P" in data

    def test_trades_open_still_works(self, client):
        resp = client.get("/trades/open")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════════════════════
# Rapid-fire updates (simulating NinjaTrader behavior)
# ═══════════════════════════════════════════════════════════════════════════


class TestRapidUpdates:
    """Simulate rapid position updates like NinjaTrader would send."""

    def test_rapid_position_updates(self, client):
        """Multiple rapid POSTs don't corrupt data."""
        from lib.api_server import get_live_positions

        for i in range(10):
            payload = {
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": float(i * 10),
                    },
                ],
                "timestamp": f"2025-01-15T14:30:{i:02d}Z",
            }
            resp = client.post("/update_positions", json=payload)
            assert resp.status_code == 200

        # Final state should reflect the last update
        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["total_unrealized_pnl"] == 90.0  # last: 9 * 10

    def test_position_lifecycle(self, client):
        """Simulate: no positions → open → update PnL → close → clear."""
        from lib.api_server import get_live_positions

        # 1. No positions initially
        assert get_live_positions()["has_positions"] is False

        # 2. Open a position
        client.post(
            "/update_positions",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": 0.0,
                    },
                ],
            },
        )
        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["total_unrealized_pnl"] == 0.0

        # 3. PnL updates
        client.post(
            "/update_positions",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": 250.0,
                    },
                ],
            },
        )
        assert get_live_positions()["total_unrealized_pnl"] == 250.0

        # 4. Position closed (empty list)
        client.post(
            "/update_positions",
            json={
                "account": "Sim101",
                "positions": [],
            },
        )
        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["total_unrealized_pnl"] == 0.0

        # 5. Clean up
        client.delete("/positions")
        assert get_live_positions()["has_positions"] is False
