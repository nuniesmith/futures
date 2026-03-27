"""
lib.integrations — External service integrations.

Re-exports the public API from each sub-module so callers can do:

    from lib.integrations import get_massive_provider, GrokSession
    from lib.integrations import get_janus_client, JanusSignal

Janus transport selection
--------------------------
Two Janus client implementations are available:

- :class:`JanusHTTPClient` — REST/HTTP transport (Session 2b, always available)
- :class:`JanusGRPCClient` — gRPC transport (Session 4; requires
  ``pip install grpcio`` and running ``scripts/gen-proto.sh``)

Use :func:`get_janus_transport` to get whichever transport is configured::

    from lib.integrations import get_janus_transport

    client = get_janus_transport()          # auto-selects based on env
    signals = await client.get_latest_signals()

Set ``JANUS_TRANSPORT=grpc`` to force gRPC; any other value (or unset) uses HTTP.
"""

import os as _os

from lib.integrations.grok_helper import (
    GrokSession,
    format_market_context,
    run_live_analysis,
    run_morning_briefing,
)
from lib.integrations.janus_client import (
    JanusClientError,
    JanusConnectionError,
    JanusHTTPClient,
    JanusResponseError,
    JanusSignal,
    JanusSignalPoller,
    JanusSignalSummary,
    get_janus_client,
    get_latest_signals_sync,
    is_janus_healthy_sync,
    publish_signal_sync,
    reset_janus_client,
)
from lib.integrations.janus_grpc_client import (
    JanusGRPCClient,
    JanusGRPCConnectionError,
    JanusGRPCError,
    JanusGRPCPoller,
    JanusGRPCResponseError,
    get_janus_grpc_client,
    get_latest_signals_grpc_sync,
    is_janus_grpc_healthy_sync,
    publish_signal_grpc_sync,
    reset_janus_grpc_client,
)
from lib.integrations.massive_client import (
    MassiveFeedManager,
    get_massive_provider,
    is_massive_available,
)
from lib.integrations.tradingview_client import (
    TradingViewConnectionError,
    TradingViewDataClient,
    TradingViewError,
    TradingViewSubscriptionError,
    get_tv_client,
    reset_tv_client,
)
from lib.integrations.tradingview_search import (
    ASSET_TYPES as TV_ASSET_TYPES,
)
from lib.integrations.tradingview_search import (
    classify_asset_type as tv_classify_asset_type,
)
from lib.integrations.tradingview_search import (
    search_symbols as tv_search_symbols,
)


def get_janus_transport(
    *,
    force_http: bool = False,
    force_grpc: bool = False,
) -> "JanusHTTPClient | JanusGRPCClient":
    """Return the appropriate Janus client based on runtime configuration.

    Selection priority:
    1. ``force_http=True`` → always return :class:`JanusHTTPClient`
    2. ``force_grpc=True`` → always return :class:`JanusGRPCClient`
    3. ``JANUS_TRANSPORT=grpc`` env var → :class:`JanusGRPCClient`
    4. Otherwise → :class:`JanusHTTPClient` (safe default)

    Both clients expose an **identical public interface** so you can swap
    transports without changing any call sites.

    Args:
        force_http: If ``True``, always return the HTTP client regardless of env.
        force_grpc: If ``True``, always return the gRPC client regardless of env.

    Returns:
        A ready-to-use client instance (singleton per transport type).

    Raises:
        ValueError: If both ``force_http`` and ``force_grpc`` are ``True``.
    """
    if force_http and force_grpc:
        raise ValueError("Cannot force both HTTP and gRPC transports simultaneously.")

    if force_grpc:
        return get_janus_grpc_client()

    if force_http:
        return get_janus_client()

    transport = _os.environ.get("JANUS_TRANSPORT", "http").strip().lower()
    if transport == "grpc":
        return get_janus_grpc_client()

    return get_janus_client()


__all__ = [
    # grok_helper
    "GrokSession",
    "format_market_context",
    "run_live_analysis",
    "run_morning_briefing",
    # janus_client (HTTP transport)
    "JanusClientError",
    "JanusConnectionError",
    "JanusHTTPClient",
    "JanusResponseError",
    "JanusSignal",
    "JanusSignalPoller",
    "JanusSignalSummary",
    "get_janus_client",
    "get_latest_signals_sync",
    "is_janus_healthy_sync",
    "publish_signal_sync",
    "reset_janus_client",
    # janus_grpc_client (gRPC transport — Session 4)
    "JanusGRPCClient",
    "JanusGRPCConnectionError",
    "JanusGRPCError",
    "JanusGRPCPoller",
    "JanusGRPCResponseError",
    "get_janus_grpc_client",
    "get_latest_signals_grpc_sync",
    "is_janus_grpc_healthy_sync",
    "publish_signal_grpc_sync",
    "reset_janus_grpc_client",
    # transport factory
    "get_janus_transport",
    # massive_client
    "MassiveFeedManager",
    "get_massive_provider",
    "is_massive_available",
    # tradingview_client
    "TradingViewDataClient",
    "TradingViewConnectionError",
    "TradingViewError",
    "TradingViewSubscriptionError",
    "get_tv_client",
    "reset_tv_client",
    # tradingview_search
    "TV_ASSET_TYPES",
    "tv_classify_asset_type",
    "tv_search_symbols",
]
