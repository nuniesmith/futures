"""
Multi-channel alert dispatcher for futures trading signals.

Sends alerts to Slack, Discord, and Telegram via webhooks/bot API.
Includes mandatory 5-minute deduplication cooldown per unique signal key
to prevent alert spam during volatile markets.

Per the notes.md blueprint:
  - Simple requests.post() calls to webhook URLs
  - Alert deduplication is essential — 5-minute cooldown per unique signal key
  - Store sent alert timestamps in Redis (with fallback to in-memory dict)
  - Channels: Slack webhooks, Discord webhooks, Telegram Bot API

Configuration via environment variables:
  SLACK_WEBHOOK_URL    — Slack incoming webhook URL
  DISCORD_WEBHOOK_URL  — Discord webhook URL
  TELEGRAM_BOT_TOKEN   — Telegram bot token
  TELEGRAM_CHAT_ID     — Telegram chat ID to send messages to
  ALERT_COOLDOWN_SEC   — Cooldown period in seconds (default 300 = 5 min)

Usage:
    from alerts import AlertDispatcher, get_dispatcher

    dispatcher = get_dispatcher()

    # Send a signal alert (respects cooldown)
    sent = dispatcher.send_signal(
        signal_key="Gold_TrendEMA_long",
        title="🟢 Gold — Long Signal",
        message="TrendEMA crossover on Gold (MGC). Entry: $2,705. SL: $2,698. TP: $2,720.",
        asset="Gold",
        strategy="TrendEMA",
        direction="long",
    )
    # sent == True if dispatched, False if within cooldown

    # Send a risk alert (always sent, no cooldown)
    dispatcher.send_risk_alert(
        title="⚠️ Hard Stop Approaching",
        message="Daily P&L: -$1,200 / -$1,500 hard stop. Consider stopping.",
    )
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("alerts")

_EST = ZoneInfo("America/New_York")

# Default cooldown: 5 minutes (300 seconds)
DEFAULT_COOLDOWN_SEC = 300


# ---------------------------------------------------------------------------
# Channel configuration from environment
# ---------------------------------------------------------------------------


def _get_config() -> dict[str, Any]:
    """Load alert channel configuration from environment variables."""
    return {
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
        "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL", ""),
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        "cooldown_sec": int(os.getenv("ALERT_COOLDOWN_SEC", str(DEFAULT_COOLDOWN_SEC))),
    }


# ---------------------------------------------------------------------------
# Redis-backed deduplication store (with in-memory fallback)
# ---------------------------------------------------------------------------


class _AlertStore:
    """Stores sent alert timestamps for deduplication.

    Uses Redis if available (persistent across restarts), otherwise
    falls back to an in-memory dict (cleared on restart).
    """

    def __init__(
        self,
        cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
        _disable_redis: bool = False,
    ):
        self.cooldown_sec = cooldown_sec
        self._memory_store: dict[str, float] = {}
        self._redis = None

        if _disable_redis:
            logger.debug("Alert store: Redis explicitly disabled (test mode)")
            return

        # Try to connect to Redis
        try:
            import redis as redis_lib

            r = redis_lib.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
                socket_connect_timeout=2,
            )
            r.ping()
            self._redis = r
            logger.info("Alert store using Redis for deduplication persistence")
        except Exception:
            logger.info("Alert store using in-memory fallback (Redis unavailable)")

    def should_send(self, signal_key: str) -> bool:
        """Check if an alert with this key should be sent (not in cooldown).

        Returns True if the alert should be sent, False if within cooldown.
        """
        now = time.time()

        if self._redis is not None:
            try:
                redis_key = f"alert:sent:{signal_key}"
                last_sent_raw = self._redis.get(redis_key)
                if last_sent_raw is not None:
                    last_sent = float(str(last_sent_raw))
                    if now - last_sent < self.cooldown_sec:
                        return False
                return True
            except Exception as exc:
                logger.debug("Redis check failed, using memory: %s", exc)

        # In-memory fallback
        last_sent = self._memory_store.get(signal_key, 0.0)
        return now - last_sent >= self.cooldown_sec

    def mark_sent(self, signal_key: str) -> None:
        """Record that an alert was just sent for this signal key."""
        now = time.time()

        if self._redis is not None:
            try:
                redis_key = f"alert:sent:{signal_key}"
                self._redis.setex(
                    redis_key,
                    self.cooldown_sec + 10,  # TTL slightly longer than cooldown
                    str(now),
                )
                return
            except Exception as exc:
                logger.debug("Redis mark_sent failed, using memory: %s", exc)

        self._memory_store[signal_key] = now

    def get_recent_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recently sent alert keys and timestamps (for dashboard display).

        Returns a list of dicts with signal_key and timestamp.
        """
        if self._redis is not None:
            try:
                _result = self._redis.keys("alert:sent:*")
                raw_keys: list[str] = list(_result) if _result else []  # type: ignore[arg-type]
                key_list: list[str] = [str(k) for k in raw_keys]
                alerts = []
                for key in sorted(key_list)[-limit:]:
                    ts_raw = self._redis.get(key)
                    if ts_raw is not None:
                        ts_val = float(str(ts_raw))
                        signal_key = key.replace("alert:sent:", "")
                        alerts.append(
                            {
                                "signal_key": signal_key,
                                "timestamp": ts_val,
                                "datetime": datetime.fromtimestamp(
                                    ts_val, tz=_EST
                                ).strftime("%H:%M:%S"),
                            }
                        )
                return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[
                    :limit
                ]
            except Exception:
                pass

        # In-memory fallback
        alerts = [
            {
                "signal_key": k,
                "timestamp": v,
                "datetime": datetime.fromtimestamp(v, tz=_EST).strftime("%H:%M:%S"),
            }
            for k, v in self._memory_store.items()
        ]
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def clear(self) -> None:
        """Clear all stored alert timestamps."""
        self._memory_store.clear()
        if self._redis is not None:
            try:
                _result = self._redis.keys("alert:sent:*")
                raw_keys = list(_result) if _result else []  # type: ignore[arg-type]
                key_list = [str(k) for k in raw_keys]
                if key_list:
                    self._redis.delete(*key_list)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Channel senders
# ---------------------------------------------------------------------------


def _send_slack(
    webhook_url: str, title: str, message: str, fields: Optional[dict] = None
) -> bool:
    """Send an alert to Slack via incoming webhook.

    Uses Slack Block Kit for rich formatting.
    """
    if not webhook_url:
        return False

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": title[:150], "emoji": True},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": message},
        },
    ]

    if fields:
        field_blocks = []
        for k, v in fields.items():
            field_blocks.append({"type": "mrkdwn", "text": f"*{k}*\n{v}"})
        if field_blocks:
            blocks.append({"type": "section", "fields": field_blocks[:10]})

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Futures Dashboard • {datetime.now(_EST).strftime('%I:%M %p ET')}",
                }
            ],
        }
    )

    payload = {"blocks": blocks, "text": title}

    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.debug("Slack alert sent: %s", title)
            return True
        else:
            logger.warning(
                "Slack alert failed: %s %s", resp.status_code, resp.text[:200]
            )
            return False
    except Exception as exc:
        logger.warning("Slack alert error: %s", exc)
        return False


def _send_discord(
    webhook_url: str, title: str, message: str, fields: Optional[dict] = None
) -> bool:
    """Send an alert to Discord via webhook.

    Uses Discord embed for rich formatting.
    """
    if not webhook_url:
        return False

    embed = {
        "title": title[:256],
        "description": message[:4096],
        "color": 43210,  # teal (#00D4AA approximate)
        "timestamp": datetime.now(_EST).isoformat(),
        "footer": {"text": "Futures Dashboard"},
    }

    if fields:
        embed["fields"] = [
            {"name": str(k), "value": str(v), "inline": True}
            for k, v in list(fields.items())[:25]
        ]

    payload = {"embeds": [embed]}

    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code in (200, 204):
            logger.debug("Discord alert sent: %s", title)
            return True
        else:
            logger.warning(
                "Discord alert failed: %s %s", resp.status_code, resp.text[:200]
            )
            return False
    except Exception as exc:
        logger.warning("Discord alert error: %s", exc)
        return False


def _send_telegram(
    bot_token: str,
    chat_id: str,
    title: str,
    message: str,
    fields: Optional[dict] = None,
) -> bool:
    """Send an alert to Telegram via Bot API.

    Uses HTML parse mode for formatting.
    """
    if not bot_token or not chat_id:
        return False

    # Build HTML-formatted message
    parts = [f"<b>{_escape_html(title)}</b>", "", _escape_html(message)]

    if fields:
        parts.append("")
        for k, v in fields.items():
            parts.append(f"<b>{_escape_html(str(k))}:</b> {_escape_html(str(v))}")

    parts.append("")
    parts.append(
        f"<i>Futures Dashboard • {datetime.now(_EST).strftime('%I:%M %p ET')}</i>"
    )

    text = "\n".join(parts)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("ok"):
                logger.debug("Telegram alert sent: %s", title)
                return True
            else:
                logger.warning("Telegram API error: %s", result.get("description", ""))
                return False
        else:
            logger.warning(
                "Telegram alert failed: %s %s", resp.status_code, resp.text[:200]
            )
            return False
    except Exception as exc:
        logger.warning("Telegram alert error: %s", exc)
        return False


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


class AlertDispatcher:
    """Multi-channel alert dispatcher with deduplication.

    Sends alerts to all configured channels (Slack, Discord, Telegram).
    Signal alerts respect the cooldown period; risk alerts are always sent.

    Usage:
        dispatcher = AlertDispatcher()
        dispatcher.send_signal(
            signal_key="Gold_TrendEMA_long",
            title="Gold Long Signal",
            message="Details...",
        )
    """

    def __init__(
        self,
        slack_webhook: str = "",
        discord_webhook: str = "",
        telegram_token: str = "",
        telegram_chat_id: str = "",
        cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
        _disable_redis: bool = False,
    ):
        self.slack_webhook = slack_webhook
        self.discord_webhook = discord_webhook
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.cooldown_sec = cooldown_sec

        self._store = _AlertStore(cooldown_sec, _disable_redis=_disable_redis)

        # Track statistics
        self._stats = {
            "total_sent": 0,
            "total_suppressed": 0,
            "slack_sent": 0,
            "discord_sent": 0,
            "telegram_sent": 0,
            "errors": 0,
        }

    @property
    def channels_configured(self) -> list[str]:
        """List of configured (non-empty) alert channels."""
        channels = []
        if self.slack_webhook:
            channels.append("Slack")
        if self.discord_webhook:
            channels.append("Discord")
        if self.telegram_token and self.telegram_chat_id:
            channels.append("Telegram")
        return channels

    @property
    def has_channels(self) -> bool:
        """True if at least one alert channel is configured."""
        return len(self.channels_configured) > 0

    def send_signal(
        self,
        signal_key: str,
        title: str,
        message: str,
        asset: str = "",
        strategy: str = "",
        direction: str = "",
        extra_fields: Optional[dict[str, str]] = None,
    ) -> bool:
        """Send a trading signal alert (subject to deduplication cooldown).

        Args:
            signal_key: Unique key for deduplication (e.g. "Gold_TrendEMA_long").
            title: Alert title/headline.
            message: Alert body text.
            asset: Asset name (for fields display).
            strategy: Strategy name (for fields display).
            direction: Trade direction (for fields display).
            extra_fields: Additional key-value pairs to display.

        Returns:
            True if the alert was sent, False if suppressed by cooldown
            or no channels configured.
        """
        if not self.has_channels:
            return False

        # Check cooldown
        if not self._store.should_send(signal_key):
            self._stats["total_suppressed"] += 1
            logger.debug("Alert suppressed (cooldown): %s", signal_key)
            return False

        # Build fields
        fields: dict[str, str] = {}
        if asset:
            fields["Asset"] = asset
        if strategy:
            fields["Strategy"] = strategy
        if direction:
            fields["Direction"] = direction.upper()
        if extra_fields:
            fields.update(extra_fields)

        # Send to all configured channels
        success = self._dispatch_all(title, message, fields)

        if success:
            self._store.mark_sent(signal_key)
            self._stats["total_sent"] += 1
        else:
            self._stats["errors"] += 1

        return success

    def send_risk_alert(
        self,
        title: str,
        message: str,
        extra_fields: Optional[dict[str, str]] = None,
    ) -> bool:
        """Send a risk/compliance alert (always sent, no cooldown).

        Use for critical alerts like hard stop approaching, drawdown warnings,
        or compliance violations.

        Returns:
            True if sent to at least one channel.
        """
        if not self.has_channels:
            return False

        fields = extra_fields or {}
        success = self._dispatch_all(title, message, fields)

        if success:
            self._stats["total_sent"] += 1
        else:
            self._stats["errors"] += 1

        return success

    def send_regime_change(
        self,
        asset: str,
        old_regime: str,
        new_regime: str,
        confidence: float = 0.0,
    ) -> bool:
        """Send an alert when a regime change is detected.

        Subject to cooldown using key "regime_{asset}_{new_regime}".
        """
        signal_key = f"regime_{asset}_{new_regime}"
        emoji_map = {
            "trending": "📈",
            "volatile": "⚡",
            "choppy": "〰️",
            "low_vol": "🟢",
            "normal": "🟡",
            "high_vol": "🔴",
        }
        emoji = emoji_map.get(new_regime, "🔄")

        title = f"{emoji} Regime Change — {asset}"
        message = (
            f"{asset} regime shifted from {old_regime.upper()} to {new_regime.upper()} "
            f"(confidence: {confidence:.0%})"
        )
        fields = {
            "Asset": asset,
            "Previous": old_regime.upper(),
            "Current": new_regime.upper(),
            "Confidence": f"{confidence:.0%}",
        }

        return self.send_signal(
            signal_key=signal_key,
            title=title,
            message=message,
            extra_fields=fields,
        )

    def send_confluence_alert(
        self,
        asset: str,
        score: int,
        direction: str,
        details: str = "",
    ) -> bool:
        """Send an alert when full (3/3) multi-timeframe confluence is detected.

        Only sends for score == 3 (full confluence).
        """
        if score < 3:
            return False

        signal_key = f"confluence_{asset}_{direction}"
        emoji = "🟢" if direction == "bullish" else "🔴"

        title = f"{emoji} Full Confluence — {asset} {direction.upper()}"
        message = f"{asset}: 3/3 multi-timeframe confluence detected for {direction.upper()} trade."
        if details:
            message += f"\n{details}"

        return self.send_signal(
            signal_key=signal_key,
            title=title,
            message=message,
            asset=asset,
            direction=direction,
            extra_fields={"Confluence": f"{score}/3"},
        )

    def _dispatch_all(
        self,
        title: str,
        message: str,
        fields: Optional[dict[str, str]] = None,
    ) -> bool:
        """Send to all configured channels. Returns True if any succeeded."""
        any_success = False

        if self.slack_webhook:
            if _send_slack(self.slack_webhook, title, message, fields):
                self._stats["slack_sent"] += 1
                any_success = True

        if self.discord_webhook:
            if _send_discord(self.discord_webhook, title, message, fields):
                self._stats["discord_sent"] += 1
                any_success = True

        if self.telegram_token and self.telegram_chat_id:
            if _send_telegram(
                self.telegram_token,
                self.telegram_chat_id,
                title,
                message,
                fields,
            ):
                self._stats["telegram_sent"] += 1
                any_success = True

        return any_success

    def get_stats(self) -> dict[str, Any]:
        """Return alert dispatch statistics for dashboard display."""
        return {
            **self._stats,
            "channels": self.channels_configured,
            "cooldown_sec": self.cooldown_sec,
            "recent_alerts": self._store.get_recent_alerts(10),
        }

    def get_recent_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recently sent alerts for dashboard display."""
        return self._store.get_recent_alerts(limit)

    def clear_cooldowns(self) -> None:
        """Clear all cooldown timers (useful for testing)."""
        self._store.clear()
        logger.info("All alert cooldowns cleared")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_dispatcher: Optional[AlertDispatcher] = None


def get_dispatcher() -> AlertDispatcher:
    """Return (or create) the singleton AlertDispatcher.

    Reads configuration from environment variables on first call.
    """
    global _dispatcher
    if _dispatcher is None:
        config = _get_config()
        _dispatcher = AlertDispatcher(
            slack_webhook=config["slack_webhook"],
            discord_webhook=config["discord_webhook"],
            telegram_token=config["telegram_token"],
            telegram_chat_id=config["telegram_chat_id"],
            cooldown_sec=config["cooldown_sec"],
        )
        if _dispatcher.has_channels:
            logger.info(
                "Alert dispatcher initialized: channels=%s, cooldown=%ds",
                _dispatcher.channels_configured,
                _dispatcher.cooldown_sec,
            )
        else:
            logger.info(
                "Alert dispatcher initialized with no channels configured. "
                "Set SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL, or "
                "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable alerts."
            )
    return _dispatcher


def reset_dispatcher() -> None:
    """Reset the singleton (forces re-read of environment on next get_dispatcher())."""
    global _dispatcher
    _dispatcher = None


# ---------------------------------------------------------------------------
# Quick-send functions (use the singleton dispatcher)
# ---------------------------------------------------------------------------


def send_signal(
    signal_key: str,
    title: str,
    message: str,
    asset: str = "",
    strategy: str = "",
    direction: str = "",
) -> bool:
    """Quick-send a signal alert using the default dispatcher."""
    return get_dispatcher().send_signal(
        signal_key=signal_key,
        title=title,
        message=message,
        asset=asset,
        strategy=strategy,
        direction=direction,
    )


def send_risk_alert(title: str, message: str) -> bool:
    """Quick-send a risk alert using the default dispatcher."""
    return get_dispatcher().send_risk_alert(title=title, message=message)
