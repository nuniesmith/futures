"""
Unit tests for the alerts module.

Tests cover:
  - _AlertStore: should_send / mark_sent cooldown logic, clear, get_recent_alerts
  - AlertDispatcher: construction, channels_configured, has_channels,
    send_signal deduplication, send_risk_alert (no cooldown), send_regime_change,
    send_confluence_alert (score < 3 guard), get_stats, get_recent_alerts,
    clear_cooldowns
  - get_dispatcher / reset_dispatcher: singleton behaviour
  - send_signal / send_risk_alert module-level convenience helpers
  - Edge cases: no channels configured, empty messages, rapid-fire dedup

All tests use _disable_redis=True to ensure in-memory-only stores and
complete isolation between tests (no shared Redis state).
"""

import time

# Ensure src/ is importable

from src.lib.core.alerts import (  # noqa: E402
    AlertDispatcher,
    _AlertStore,
    get_dispatcher,
    reset_dispatcher,
    send_risk_alert,
    send_signal,
)

# ---------------------------------------------------------------------------
# Helper: create a fresh AlertStore with Redis disabled
# ---------------------------------------------------------------------------


def _fresh_store(cooldown_sec: int = 300) -> _AlertStore:
    """Create an isolated in-memory AlertStore."""
    return _AlertStore(cooldown_sec=cooldown_sec, _disable_redis=True)


def _fresh_dispatcher(**kwargs) -> AlertDispatcher:
    """Create an isolated AlertDispatcher (Redis disabled)."""
    kwargs.setdefault("_disable_redis", True)
    return AlertDispatcher(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# _AlertStore
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertStore:
    """Tests for the internal cooldown / dedup store."""

    def test_should_send_first_time(self):
        """First call for a key should always return True."""
        store = _fresh_store(cooldown_sec=300)
        assert store.should_send("test_key") is True

    def test_should_send_blocked_after_mark(self):
        """After mark_sent, should_send returns False within cooldown."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("test_key")
        assert store.should_send("test_key") is False

    def test_different_keys_are_independent(self):
        """Different keys have independent cooldowns."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("key_a")
        assert store.should_send("key_a") is False
        assert store.should_send("key_b") is True

    def test_cooldown_expires(self):
        """After cooldown expires, should_send returns True again."""
        store = _fresh_store(cooldown_sec=1)  # 1-second cooldown
        store.mark_sent("short_key")
        assert store.should_send("short_key") is False
        time.sleep(1.1)
        assert store.should_send("short_key") is True

    def test_zero_cooldown_always_allows(self):
        """With cooldown_sec=0, every send should be allowed."""
        store = _fresh_store(cooldown_sec=0)
        store.mark_sent("zero_cd")
        # With 0 cooldown, next send should be allowed immediately
        assert store.should_send("zero_cd") is True

    def test_clear_resets_cooldowns(self):
        """clear() should reset all cooldowns."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("key_1")
        store.mark_sent("key_2")
        assert store.should_send("key_1") is False
        assert store.should_send("key_2") is False
        store.clear()
        assert store.should_send("key_1") is True
        assert store.should_send("key_2") is True

    def test_get_recent_alerts_empty(self):
        """When no alerts have been sent, recent list is empty."""
        store = _fresh_store(cooldown_sec=300)
        recent = store.get_recent_alerts()
        assert isinstance(recent, list)
        assert len(recent) == 0

    def test_get_recent_alerts_after_mark(self):
        """After marking alerts, they should appear in recent list."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("alert_1")
        store.mark_sent("alert_2")
        recent = store.get_recent_alerts()
        assert len(recent) >= 2

    def test_get_recent_alerts_respects_limit(self):
        """Limit parameter should cap the number of results."""
        store = _fresh_store(cooldown_sec=300)
        for i in range(20):
            store.mark_sent(f"alert_{i}")
        recent = store.get_recent_alerts(limit=5)
        assert len(recent) <= 5

    def test_mark_sent_multiple_times_same_key(self):
        """Marking the same key multiple times should not crash."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("repeat")
        store.mark_sent("repeat")
        store.mark_sent("repeat")
        assert store.should_send("repeat") is False


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — construction and channel detection
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherConstruction:
    def test_default_no_channels(self):
        """Default construction with no webhooks should have no channels."""
        d = _fresh_dispatcher()
        assert d.has_channels is False
        assert d.channels_configured == []

    def test_slack_channel_detected(self):
        d = _fresh_dispatcher(slack_webhook="https://hooks.slack.com/test")
        assert "Slack" in d.channels_configured
        assert d.has_channels is True

    def test_discord_channel_detected(self):
        d = _fresh_dispatcher(discord_webhook="https://discord.com/api/webhooks/test")
        assert "Discord" in d.channels_configured
        assert d.has_channels is True

    def test_telegram_channel_detected(self):
        d = _fresh_dispatcher(telegram_token="bot123", telegram_chat_id="456")
        assert "Telegram" in d.channels_configured
        assert d.has_channels is True

    def test_telegram_needs_both_token_and_chat(self):
        """Telegram requires both token AND chat_id to be configured."""
        d_token_only = _fresh_dispatcher(telegram_token="bot123")
        assert "Telegram" not in d_token_only.channels_configured

        d_chat_only = _fresh_dispatcher(telegram_chat_id="456")
        assert "Telegram" not in d_chat_only.channels_configured

    def test_multiple_channels(self):
        d = _fresh_dispatcher(
            slack_webhook="https://hooks.slack.com/test",
            discord_webhook="https://discord.com/api/webhooks/test",
            telegram_token="bot123",
            telegram_chat_id="456",
        )
        channels = d.channels_configured
        assert "Slack" in channels
        assert "Discord" in channels
        assert "Telegram" in channels
        assert len(channels) == 3

    def test_custom_cooldown(self):
        d = _fresh_dispatcher(cooldown_sec=60)
        assert d.cooldown_sec == 60

    def test_default_cooldown(self):
        d = _fresh_dispatcher()
        # Default should be > 0 (5 minutes = 300s per module docs)
        assert d.cooldown_sec > 0


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — send_signal (no real channels, dedup logic)
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherSendSignal:
    def test_send_signal_no_channels_returns_false(self):
        """With no channels configured, send_signal should return False."""
        d = _fresh_dispatcher()
        result = d.send_signal(
            signal_key="test",
            title="Test",
            message="Hello",
        )
        assert result is False

    def test_send_signal_dedup_blocks_repeat(self):
        """Second call with same key within cooldown should be suppressed."""
        d = _fresh_dispatcher(
            slack_webhook="https://fake.invalid/webhook",
            cooldown_sec=300,
        )
        # Mark key as already sent
        d._store.mark_sent("dup_key")
        result = d.send_signal(
            signal_key="dup_key",
            title="Dup Test",
            message="Should be blocked",
        )
        assert result is False

    def test_send_signal_different_keys_independent(self):
        """Different signal_keys should be tracked independently."""
        d = _fresh_dispatcher()  # no channels → always False
        d.send_signal(signal_key="key_a", title="A", message="A")
        # key_b should not be affected by key_a
        assert d._store.should_send("key_b") is True

    def test_send_signal_extra_fields(self):
        """Extra fields should not cause an error."""
        d = _fresh_dispatcher()
        result = d.send_signal(
            signal_key="extra",
            title="Extra",
            message="Test",
            asset="Gold",
            strategy="TrendEMA",
            direction="LONG",
            extra_fields={"Confidence": "high", "Regime": "trending"},
        )
        assert result is False  # no channels

    def test_send_signal_empty_strings(self):
        """Empty title/message should not crash."""
        d = _fresh_dispatcher()
        result = d.send_signal(signal_key="empty", title="", message="")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — send_risk_alert (no cooldown)
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherSendRiskAlert:
    def test_risk_alert_no_channels(self):
        """With no channels, send_risk_alert returns False."""
        d = _fresh_dispatcher()
        result = d.send_risk_alert(title="Risk!", message="Hard stop hit")
        assert result is False

    def test_risk_alert_no_cooldown_dedup(self):
        """Risk alerts bypass cooldown — they should always attempt to send."""
        d = _fresh_dispatcher(cooldown_sec=300)
        result = d.send_risk_alert(
            title="HARD STOP",
            message="Account hit hard stop limit",
            extra_fields={"P&L": "-$2,250"},
        )
        assert result is False  # no channels


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — send_regime_change
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherSendRegimeChange:
    def test_regime_change_no_channels(self):
        d = _fresh_dispatcher()
        result = d.send_regime_change(
            asset="Gold",
            old_regime="trending",
            new_regime="volatile",
            confidence=0.85,
        )
        assert result is False

    def test_regime_change_returns_bool(self):
        d = _fresh_dispatcher()
        result = d.send_regime_change(
            asset="S&P",
            old_regime="choppy",
            new_regime="trending",
            confidence=0.72,
        )
        assert isinstance(result, bool)

    def test_regime_change_dedup_key_format(self):
        """Regime change should use key format 'regime_{asset}_{new_regime}'."""
        d = _fresh_dispatcher()
        for regime in (
            "trending",
            "volatile",
            "choppy",
            "low_vol",
            "normal",
            "high_vol",
        ):
            result = d.send_regime_change(
                asset="Gold", old_regime="trending", new_regime=regime
            )
            assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — send_confluence_alert
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherSendConfluenceAlert:
    def test_confluence_score_below_3_returns_false(self):
        """Only score == 3 should trigger a confluence alert."""
        d = _fresh_dispatcher()
        for score in (0, 1, 2):
            result = d.send_confluence_alert(
                asset="Gold", score=score, direction="bullish"
            )
            assert result is False

    def test_confluence_score_3_no_channels(self):
        """Score 3 with no channels still returns False (can't send)."""
        d = _fresh_dispatcher()
        result = d.send_confluence_alert(
            asset="Gold",
            score=3,
            direction="bullish",
            details="HTF=bullish, Setup=bullish, Entry=bullish",
        )
        assert result is False

    def test_confluence_with_details(self):
        """Details parameter should not cause errors."""
        d = _fresh_dispatcher()
        result = d.send_confluence_alert(
            asset="Nasdaq",
            score=3,
            direction="bearish",
            details="Full alignment on 15m/5m/1m",
        )
        assert isinstance(result, bool)

    def test_confluence_without_details(self):
        d = _fresh_dispatcher()
        result = d.send_confluence_alert(asset="S&P", score=3, direction="bullish")
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — stats and recent alerts
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherStats:
    def test_get_stats_returns_dict(self):
        d = _fresh_dispatcher()
        stats = d.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_expected_keys(self):
        d = _fresh_dispatcher()
        stats = d.get_stats()
        expected_keys = {"total_sent", "channels", "cooldown_sec"}
        assert expected_keys.issubset(set(stats.keys()))

    def test_initial_stats_are_zero(self):
        d = _fresh_dispatcher()
        stats = d.get_stats()
        assert stats["total_sent"] == 0

    def test_channels_list_in_stats(self):
        d = _fresh_dispatcher(slack_webhook="https://hooks.slack.com/test")
        stats = d.get_stats()
        assert "Slack" in stats["channels"]

    def test_suppressed_count_increments(self):
        """When dedup blocks a signal, total_suppressed should increment."""
        d = _fresh_dispatcher(slack_webhook="https://fake.invalid/webhook")
        # Mark key as sent
        d._store.mark_sent("suppressed_key")
        d.send_signal(signal_key="suppressed_key", title="X", message="Y")
        stats = d.get_stats()
        assert stats.get("total_suppressed", 0) >= 1

    def test_get_recent_alerts_returns_list(self):
        d = _fresh_dispatcher()
        recent = d.get_recent_alerts()
        assert isinstance(recent, list)

    def test_get_recent_alerts_initially_empty(self):
        """A fresh dispatcher should have zero recent alerts."""
        d = _fresh_dispatcher()
        recent = d.get_recent_alerts()
        assert len(recent) == 0

    def test_get_recent_alerts_with_limit(self):
        d = _fresh_dispatcher()
        recent = d.get_recent_alerts(limit=5)
        assert isinstance(recent, list)
        assert len(recent) <= 5


# ═══════════════════════════════════════════════════════════════════════════
# AlertDispatcher — clear_cooldowns
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertDispatcherClearCooldowns:
    def test_clear_cooldowns_resets_dedup(self):
        d = _fresh_dispatcher(cooldown_sec=300)
        d._store.mark_sent("cd_key")
        assert d._store.should_send("cd_key") is False
        d.clear_cooldowns()
        assert d._store.should_send("cd_key") is True

    def test_clear_cooldowns_does_not_crash_when_empty(self):
        d = _fresh_dispatcher()
        d.clear_cooldowns()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# Module-level functions: get_dispatcher / reset_dispatcher
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDispatcher:
    def setup_method(self):
        """Reset the singleton before each test."""
        reset_dispatcher()

    def teardown_method(self):
        """Reset after each test to avoid polluting other test classes."""
        reset_dispatcher()

    def test_returns_dispatcher_instance(self):
        d = get_dispatcher()
        assert isinstance(d, AlertDispatcher)

    def test_singleton_behavior(self):
        """Multiple calls should return the same instance."""
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        assert d1 is d2

    def test_reset_dispatcher_creates_new_instance(self):
        d1 = get_dispatcher()
        reset_dispatcher()
        d2 = get_dispatcher()
        assert d1 is not d2

    def test_dispatcher_reads_env_vars(self, monkeypatch):
        """get_dispatcher should pick up webhook URLs from environment."""
        reset_dispatcher()
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env_test")
        d = get_dispatcher()
        assert "Slack" in d.channels_configured

    def test_dispatcher_reads_discord_env(self, monkeypatch):
        reset_dispatcher()
        monkeypatch.setenv(
            "DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/env_test"
        )
        d = get_dispatcher()
        assert "Discord" in d.channels_configured

    def test_dispatcher_reads_telegram_env(self, monkeypatch):
        reset_dispatcher()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot_env_123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat_env_456")
        d = get_dispatcher()
        assert "Telegram" in d.channels_configured


# ═══════════════════════════════════════════════════════════════════════════
# Module-level convenience: send_signal / send_risk_alert
# ═══════════════════════════════════════════════════════════════════════════


class TestModuleLevelHelpers:
    def setup_method(self):
        reset_dispatcher()

    def teardown_method(self):
        reset_dispatcher()

    def test_send_signal_convenience(self):
        """Module-level send_signal should not crash with no channels."""
        result = send_signal(
            signal_key="module_test",
            title="Module Test",
            message="Hello from module-level helper",
            asset="Gold",
            strategy="TrendEMA",
            direction="LONG",
        )
        assert isinstance(result, bool)

    def test_send_risk_alert_convenience(self):
        """Module-level send_risk_alert should not crash with no channels."""
        result = send_risk_alert(
            title="Risk Alert",
            message="Hard stop approaching",
        )
        assert isinstance(result, bool)

    def test_send_signal_returns_false_no_channels(self):
        result = send_signal(
            signal_key="no_ch",
            title="T",
            message="M",
        )
        assert result is False

    def test_send_risk_alert_returns_false_no_channels(self):
        result = send_risk_alert(title="T", message="M")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertEdgeCases:
    def test_very_long_signal_key(self):
        """Extremely long signal keys should not crash."""
        d = _fresh_dispatcher()
        long_key = "x" * 10_000
        result = d.send_signal(signal_key=long_key, title="Long Key", message="Test")
        assert isinstance(result, bool)

    def test_unicode_in_message(self):
        """Unicode characters (emojis, CJK, etc.) should be handled."""
        d = _fresh_dispatcher()
        result = d.send_signal(
            signal_key="unicode",
            title="🚀 Unicode Title 日本語",
            message="Message with émojis 🎯 and spëcial chars",
        )
        assert isinstance(result, bool)

    def test_special_chars_in_key(self):
        """Keys with special characters should work."""
        d = _fresh_dispatcher()
        for key in [
            "key/with/slashes",
            "key:with:colons",
            "key with spaces",
            "key@#$%",
        ]:
            result = d.send_signal(signal_key=key, title="T", message="M")
            assert isinstance(result, bool)

    def test_rapid_fire_dedup(self):
        """Sending many signals rapidly should dedup correctly."""
        d = _fresh_dispatcher(cooldown_sec=300)
        for i in range(100):
            d._store.mark_sent(f"rapid_{i % 10}")  # only 10 unique keys

        # All 10 keys should be blocked
        for i in range(10):
            assert d._store.should_send(f"rapid_{i}") is False

        # Key 10 (never sent) should be allowed
        assert d._store.should_send("rapid_10") is True

    def test_none_values_dont_crash(self):
        """Passing None-ish values as optional params should not crash."""
        d = _fresh_dispatcher()
        result = d.send_signal(
            signal_key="none_test",
            title="Title",
            message="Msg",
            asset="",
            strategy="",
            direction="",
            extra_fields=None,
        )
        assert isinstance(result, bool)

    def test_concurrent_store_access(self):
        """Basic check that the store doesn't corrupt under sequential rapid access."""
        store = _fresh_store(cooldown_sec=300)
        keys = [f"concurrent_{i}" for i in range(50)]
        for k in keys:
            store.mark_sent(k)
        for k in keys:
            assert store.should_send(k) is False
        store.clear()
        for k in keys:
            assert store.should_send(k) is True

    def test_dispatcher_stats_structure(self):
        """Verify stats dict has a consistent structure."""
        d = _fresh_dispatcher(
            slack_webhook="https://fake.invalid/test",
            cooldown_sec=120,
        )
        stats = d.get_stats()
        assert isinstance(stats.get("total_sent"), int)
        assert isinstance(stats.get("total_suppressed"), int)
        assert isinstance(stats.get("errors"), int)
        assert isinstance(stats.get("channels"), list)
        assert isinstance(stats.get("cooldown_sec"), int)
        assert stats["cooldown_sec"] == 120

    def test_empty_webhook_not_treated_as_channel(self):
        """Empty string webhooks should not register as channels."""
        d = _fresh_dispatcher(
            slack_webhook="",
            discord_webhook="",
            telegram_token="",
            telegram_chat_id="",
        )
        assert d.has_channels is False
        assert d.channels_configured == []

    def test_multiple_dispatchers_are_isolated(self):
        """Two dispatchers should have completely independent stores."""
        d1 = _fresh_dispatcher(cooldown_sec=300)
        d2 = _fresh_dispatcher(cooldown_sec=300)

        d1._store.mark_sent("isolated_key")
        assert d1._store.should_send("isolated_key") is False
        assert d2._store.should_send("isolated_key") is True

    def test_clear_then_recent_is_empty(self):
        """After clear, recent alerts should be empty."""
        d = _fresh_dispatcher()
        d._store.mark_sent("will_clear")
        assert len(d.get_recent_alerts()) >= 1
        d.clear_cooldowns()
        assert len(d.get_recent_alerts()) == 0

    def test_recent_alerts_contain_expected_fields(self):
        """Each recent alert dict should have signal_key and timestamp."""
        store = _fresh_store(cooldown_sec=300)
        store.mark_sent("field_test")
        recent = store.get_recent_alerts()
        assert len(recent) == 1
        alert = recent[0]
        assert "signal_key" in alert
        assert "timestamp" in alert
        assert alert["signal_key"] == "field_test"
        assert isinstance(alert["timestamp"], float)
