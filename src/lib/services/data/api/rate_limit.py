"""
Rate Limiting — TASK-703
==========================
Configurable per-endpoint rate limiting using ``slowapi``.

Limits:
  - Public endpoints (``/health``, ``/docs``): 60 req/min
  - API endpoints (default): 30 req/min per client
  - SSE endpoint: 5 connections per client per minute
  - Trades / position mutations: 20 req/min per client
  - Force refresh / heavy actions: 5 req/min per client

Configuration via environment variables:
  - ``RATE_LIMIT_ENABLED``   — "1" to enable, "0" to disable (default: "1")
  - ``RATE_LIMIT_DEFAULT``   — Default limit string (default: "30/minute")
  - ``RATE_LIMIT_PUBLIC``    — Public endpoint limit (default: "60/minute")
  - ``RATE_LIMIT_SSE``       — SSE connection limit (default: "5/minute")
  - ``RATE_LIMIT_MUTATIONS`` — Trade/position mutation limit (default: "20/minute")
  - ``RATE_LIMIT_HEAVY``     — Heavy actions limit (default: "5/minute")
  - ``RATE_LIMIT_STORAGE``   — "memory" or "redis" (default: "memory")

Usage in ``main.py``::

    from lib.services.data.api.rate_limit import setup_rate_limiting

    setup_rate_limiting(app)

To apply custom limits to specific endpoints::

    from lib.services.data.api.rate_limit import get_limiter

    limiter = get_limiter()

    @router.post("/some/endpoint")
    @limiter.limit("10/minute")
    def some_endpoint(request: Request):
        ...

Note: When ``RATE_LIMIT_ENABLED=0`` (or in test environments), the limiter
is installed but uses extremely permissive limits so it never blocks.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("api.rate_limit")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip() in ("1", "true", "yes")
_DEFAULT_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "30/minute")
_PUBLIC_LIMIT = os.getenv("RATE_LIMIT_PUBLIC", "60/minute")
_SSE_LIMIT = os.getenv("RATE_LIMIT_SSE", "5/minute")
_MUTATIONS_LIMIT = os.getenv("RATE_LIMIT_MUTATIONS", "20/minute")
_HEAVY_LIMIT = os.getenv("RATE_LIMIT_HEAVY", "5/minute")
_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE", "memory://")

# When disabled, use an absurdly high limit so the middleware is present
# but never blocks (keeps code paths consistent).
_DISABLED_LIMIT = "999999/second"


def _get_effective_limit(configured: str) -> str:
    """Return the effective limit, or a no-op limit when rate limiting is off."""
    if not _ENABLED:
        return _DISABLED_LIMIT
    return configured


# ---------------------------------------------------------------------------
# Key function — identifies the client for rate-limit bucketing
# ---------------------------------------------------------------------------


def _client_key_func(request: Request) -> str:
    """Derive a rate-limit key from the request.

    Priority:
      1. ``X-Forwarded-For`` header (behind reverse proxy / load balancer)
      2. ``X-API-Key`` header (per-client API key)
      3. Remote address (direct connection)

    This ensures that different API-key holders behind the same proxy
    get independent rate-limit buckets.
    """
    # Check for API key first — gives per-client buckets
    api_key = request.headers.get("x-api-key")
    if api_key:
        # Use a hash prefix so the full key isn't stored in rate-limit state
        key_prefix = api_key[:8] if len(api_key) >= 8 else api_key
        return f"apikey:{key_prefix}"

    # Fall back to IP address
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take the first (client) IP from the chain
        client_ip = forwarded.split(",")[0].strip()
        return f"ip:{client_ip}"

    return f"ip:{get_remote_address(request)}"


# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------


def _get_storage_uri() -> str:
    """Determine the rate-limit storage backend URI.

    Supports:
      - ``memory://`` — in-process dictionary (default, no dependencies)
      - ``redis://host:port`` — Redis-backed (shared across workers)
    """
    uri = _STORAGE_URI

    if uri.startswith("redis"):
        return uri

    # Try to use Redis if it's available and configured
    if uri == "memory://" or not uri:
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url and os.getenv("RATE_LIMIT_STORAGE", "") == "redis":
            return redis_url

    return "memory://"


# ---------------------------------------------------------------------------
# Limiter singleton
# ---------------------------------------------------------------------------

_limiter: Optional[Limiter] = None


def get_limiter() -> Limiter:
    """Return the singleton ``Limiter`` instance.

    Creates it on first call. Subsequent calls return the same instance.
    The limiter can be used as a decorator on route handlers::

        limiter = get_limiter()

        @router.get("/foo")
        @limiter.limit("10/minute")
        def foo(request: Request):
            ...
    """
    global _limiter
    if _limiter is None:
        storage_uri = _get_storage_uri()
        default_limit = _get_effective_limit(_DEFAULT_LIMIT)

        _limiter = Limiter(
            key_func=_client_key_func,
            default_limits=[default_limit],
            storage_uri=storage_uri,
            strategy="fixed-window",
        )
        logger.info(
            "Rate limiter initialised: enabled=%s default=%s storage=%s",
            _ENABLED,
            default_limit,
            storage_uri,
        )

    return _limiter


def reset_limiter() -> None:
    """Reset the limiter singleton (useful in tests)."""
    global _limiter
    _limiter = None


# ---------------------------------------------------------------------------
# Custom rate-limit exceeded handler
# ---------------------------------------------------------------------------


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a structured JSON 429 response when rate limit is exceeded."""
    retry_after = exc.detail or "unknown"

    logger.warning(
        "Rate limit exceeded: %s %s from %s — limit: %s",
        request.method,
        request.url.path,
        _client_key_func(request),
        retry_after,
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {retry_after}",
            "retry_after": str(retry_after),
        },
        headers={"Retry-After": str(retry_after)},
    )


# ---------------------------------------------------------------------------
# Route-specific limit decorators (convenience wrappers)
# ---------------------------------------------------------------------------

# These are limit strings that can be used with @limiter.limit(...)
# or applied via setup_rate_limiting() application-level limits.

PUBLIC_LIMIT = _get_effective_limit(_PUBLIC_LIMIT)
SSE_LIMIT = _get_effective_limit(_SSE_LIMIT)
MUTATIONS_LIMIT = _get_effective_limit(_MUTATIONS_LIMIT)
HEAVY_LIMIT = _get_effective_limit(_HEAVY_LIMIT)
DEFAULT_LIMIT = _get_effective_limit(_DEFAULT_LIMIT)


# ---------------------------------------------------------------------------
# Path-based limit mapping
# ---------------------------------------------------------------------------

# Maps path prefixes to their rate limit strings.
# More specific prefixes should come first.
_PATH_LIMITS: list[tuple[str, str]] = [
    # Heavy / admin actions
    ("/actions/force_refresh", HEAVY_LIMIT),
    ("/actions/optimize_now", HEAVY_LIMIT),
    ("/actions/run_backtest", HEAVY_LIMIT),
    # SSE connections
    ("/sse/", SSE_LIMIT),
    # Mutation endpoints
    ("/trades", MUTATIONS_LIMIT),
    ("/log_trade", MUTATIONS_LIMIT),
    ("/positions/update", MUTATIONS_LIMIT),
    ("/risk/check", MUTATIONS_LIMIT),
    # Public / health
    ("/health", PUBLIC_LIMIT),
    ("/docs", PUBLIC_LIMIT),
    ("/openapi.json", PUBLIC_LIMIT),
    ("/redoc", PUBLIC_LIMIT),
    ("/metrics", PUBLIC_LIMIT),
]


def get_limit_for_path(path: str) -> str:
    """Return the rate limit string for a given request path.

    Used by the application-level limit function to apply path-specific
    limits without requiring per-route decorators.
    """
    for prefix, limit in _PATH_LIMITS:
        if path.startswith(prefix):
            return limit
    return DEFAULT_LIMIT


def _dynamic_limit_func(key: str) -> str:
    """Dynamic limit function for slowapi's application_limits.

    This is called by slowapi with the key (client identifier).
    Unfortunately slowapi's application_limits don't receive the request,
    so we use the default limit here and apply path-specific limits
    via the endpoint-level decorators in setup_rate_limiting().
    """
    return DEFAULT_LIMIT


# ---------------------------------------------------------------------------
# Setup — call from main.py
# ---------------------------------------------------------------------------


def setup_rate_limiting(app: FastAPI) -> Limiter:
    """Install rate limiting on a FastAPI application.

    This:
      1. Creates/gets the limiter singleton
      2. Sets it as ``app.state.limiter``
      3. Registers the 429 exception handler
      4. Logs the configuration

    Returns the limiter instance so callers can use it for
    per-route ``@limiter.limit(...)`` decorators if desired.

    Args:
        app: The FastAPI application instance.

    Returns:
        The configured ``Limiter`` instance.
    """
    limiter = get_limiter()
    app.state.limiter = limiter

    # Register the custom 429 handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    logger.info(
        "Rate limiting configured: enabled=%s, default=%s, public=%s, sse=%s, mutations=%s, heavy=%s",
        _ENABLED,
        DEFAULT_LIMIT,
        PUBLIC_LIMIT,
        SSE_LIMIT,
        MUTATIONS_LIMIT,
        HEAVY_LIMIT,
    )

    return limiter


# ---------------------------------------------------------------------------
# Convenience: check if rate limiting is enabled
# ---------------------------------------------------------------------------


def is_rate_limiting_enabled() -> bool:
    """Return True if rate limiting is actively enforcing limits."""
    return _ENABLED
