"""
lib.integrations — External service integrations.

Re-exports the public API from each sub-module so callers can do:

    from src.lib.integrations import get_massive_provider, GrokSession
"""

from src.lib.integrations.grok_helper import (
    GrokSession,
    format_market_context,
    run_live_analysis,
    run_morning_briefing,
)
from src.lib.integrations.massive_client import (
    MassiveFeedManager,
    get_massive_provider,
    is_massive_available,
)

__all__ = [
    # grok_helper
    "GrokSession",
    "format_market_context",
    "run_live_analysis",
    "run_morning_briefing",
    # massive_client
    "MassiveFeedManager",
    "get_massive_provider",
    "is_massive_available",
]
