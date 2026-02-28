"""
Prometheus Metrics Endpoint — TASK-704
========================================
Exposes ``GET /metrics/prometheus`` returning Prometheus text-format metrics.

Tracked metrics:
  - ``http_requests_total``           — Counter: total HTTP requests by method, path, status
  - ``http_request_duration_seconds`` — Histogram: request latency by method and path
  - ``sse_connections_active``        — Gauge: currently active SSE connections
  - ``sse_events_total``              — Counter: total SSE events emitted by event type
  - ``engine_last_refresh_epoch``     — Gauge: epoch timestamp of last engine data refresh
  - ``engine_cycle_duration_seconds`` — Histogram: engine cycle duration
  - ``risk_checks_total``             — Counter: risk checks by result (allowed/blocked/advisory)
  - ``orb_detections_total``          — Counter: ORB breakout detections by direction (LONG/SHORT/none)
  - ``no_trade_alerts_total``         — Counter: no-trade alerts by condition
  - ``focus_quality_gauge``           — Gauge: latest focus quality per asset symbol
  - ``positions_open_count``          — Gauge: number of currently open positions
  - ``redis_connected``               — Gauge: 1 if Redis is connected, 0 otherwise

All metrics are collected in-process via ``prometheus_client`` and the ASGI
middleware automatically instruments request count + latency.

Usage:
    from src.lib.services.data.api.metrics import router as metrics_router, PrometheusMiddleware
    app.include_router(metrics_router)
    app.add_middleware(PrometheusMiddleware)

The ``/metrics/prometheus`` path is public (no API key required).
"""

import logging
import time

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger("api.metrics")

# ---------------------------------------------------------------------------
# Prometheus client setup
# ---------------------------------------------------------------------------
# We use a custom CollectorRegistry so tests can create isolated instances
# without polluting the global default.
# ---------------------------------------------------------------------------
from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Singleton registry for the application
_registry = CollectorRegistry()

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# -- HTTP request metrics (populated by middleware) --
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    labelnames=["method", "path", "status"],
    registry=_registry,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_registry,
)

# -- SSE metrics --
SSE_CONNECTIONS_ACTIVE = Gauge(
    "sse_connections_active",
    "Number of currently active SSE connections",
    registry=_registry,
)

SSE_EVENTS_TOTAL = Counter(
    "sse_events_total",
    "Total SSE events emitted",
    labelnames=["event_type"],
    registry=_registry,
)

# -- Engine metrics --
ENGINE_LAST_REFRESH_EPOCH = Gauge(
    "engine_last_refresh_epoch",
    "Unix epoch timestamp of the last engine data refresh",
    registry=_registry,
)

ENGINE_CYCLE_DURATION = Histogram(
    "engine_cycle_duration_seconds",
    "Duration of engine scheduler cycles",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=_registry,
)

# -- Risk metrics --
RISK_CHECKS_TOTAL = Counter(
    "risk_checks_total",
    "Total risk checks performed",
    labelnames=["result"],  # allowed, blocked, advisory
    registry=_registry,
)

# -- ORB metrics --
ORB_DETECTIONS_TOTAL = Counter(
    "orb_detections_total",
    "Opening Range Breakout detections",
    labelnames=["direction"],  # LONG, SHORT, none
    registry=_registry,
)

# -- No-trade metrics --
NO_TRADE_ALERTS_TOTAL = Counter(
    "no_trade_alerts_total",
    "No-trade alerts emitted",
    labelnames=["condition"],
    registry=_registry,
)

# -- Focus quality per asset --
FOCUS_QUALITY_GAUGE = Gauge(
    "focus_quality_gauge",
    "Latest focus quality score per asset",
    labelnames=["symbol"],
    registry=_registry,
)

# -- Positions --
POSITIONS_OPEN_COUNT = Gauge(
    "positions_open_count",
    "Number of currently open positions",
    registry=_registry,
)

# -- Redis connectivity --
REDIS_CONNECTED = Gauge(
    "redis_connected",
    "Whether Redis is currently connected (1=yes, 0=no)",
    registry=_registry,
)


# ---------------------------------------------------------------------------
# Helpers for recording metrics from other modules
# ---------------------------------------------------------------------------


def record_sse_connect() -> None:
    """Call when a new SSE client connects."""
    SSE_CONNECTIONS_ACTIVE.inc()


def record_sse_disconnect() -> None:
    """Call when an SSE client disconnects."""
    SSE_CONNECTIONS_ACTIVE.dec()


def record_sse_event(event_type: str) -> None:
    """Record an SSE event emission."""
    SSE_EVENTS_TOTAL.labels(event_type=event_type).inc()


def record_risk_check(result: str) -> None:
    """Record a risk check result ('allowed', 'blocked', or 'advisory')."""
    RISK_CHECKS_TOTAL.labels(result=result).inc()


def record_orb_detection(direction: str) -> None:
    """Record an ORB detection ('LONG', 'SHORT', or 'none')."""
    ORB_DETECTIONS_TOTAL.labels(direction=direction).inc()


def record_no_trade_alert(condition: str) -> None:
    """Record a no-trade alert by condition name."""
    NO_TRADE_ALERTS_TOTAL.labels(condition=condition).inc()


def record_engine_refresh() -> None:
    """Record that the engine just refreshed data (set timestamp to now)."""
    ENGINE_LAST_REFRESH_EPOCH.set(time.time())


def record_engine_cycle(duration_seconds: float) -> None:
    """Record the duration of an engine scheduler cycle."""
    ENGINE_CYCLE_DURATION.observe(duration_seconds)


def update_focus_quality(symbol: str, quality: float) -> None:
    """Update the focus quality gauge for a specific asset."""
    FOCUS_QUALITY_GAUGE.labels(symbol=symbol).set(quality)


def update_positions_count(count: int) -> None:
    """Update the open positions count gauge."""
    POSITIONS_OPEN_COUNT.set(count)


def update_redis_status(connected: bool) -> None:
    """Update the Redis connectivity gauge."""
    REDIS_CONNECTED.set(1 if connected else 0)


# ---------------------------------------------------------------------------
# Path normalization for HTTP metrics
# ---------------------------------------------------------------------------

# Paths that should be collapsed to reduce cardinality
_PATH_PREFIXES_TO_NORMALIZE = [
    "/api/focus/",
    "/trades/",
    "/positions/",
    "/journal/",
    "/analysis/latest/",
    "/data/ohlcv/",
    "/data/daily/",
]


def _normalize_path(path: str) -> str:
    """Normalize request paths to reduce metric cardinality.

    For example:
        /api/focus/mgc     → /api/focus/{symbol}
        /trades/123/close  → /trades/{id}/close
        /sse/dashboard     → /sse/dashboard  (unchanged)
    """
    if not path:
        return "/"

    for prefix in _PATH_PREFIXES_TO_NORMALIZE:
        if path.startswith(prefix) and len(path) > len(prefix):
            # Collapse the next path segment to {id}
            rest = path[len(prefix) :]
            slash_pos = rest.find("/")
            if slash_pos == -1:
                return prefix + "{id}"
            else:
                return prefix + "{id}" + rest[slash_pos:]

    return path


# ---------------------------------------------------------------------------
# ASGI Middleware for automatic HTTP metrics
# ---------------------------------------------------------------------------


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records HTTP request count and latency.

    Automatically instruments every request. SSE endpoints (streaming
    responses) are tracked for connection start; their duration reflects
    the full connection lifetime.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        method = request.method
        path = _normalize_path(request.url.path)

        start = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status=status_code
            ).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)

        return response


# ---------------------------------------------------------------------------
# Collect live state from Redis/cache when /metrics/prometheus is hit
# ---------------------------------------------------------------------------


def _collect_live_gauges() -> None:
    """Read current state from Redis/cache and update gauges.

    Called each time the metrics endpoint is scraped so that gauges
    reflect the latest known state.
    """
    # Redis connectivity
    try:
        from src.lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            update_redis_status(True)
        else:
            update_redis_status(False)
    except Exception:
        update_redis_status(False)

    # Focus quality from cache
    try:
        import json as _json

        from src.lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            focus = _json.loads(raw)
            for asset in focus.get("assets", []):
                sym = asset.get("symbol", "")
                quality = asset.get("quality", 0)
                if sym:
                    update_focus_quality(sym, float(quality))
    except Exception:
        pass

    # Open positions count
    try:
        import json as _json

        from src.lib.core.cache import cache_get

        raw = cache_get("engine:positions")
        if raw:
            positions = _json.loads(raw)
            if isinstance(positions, list):
                update_positions_count(len(positions))
            elif isinstance(positions, dict):
                pos_list = positions.get("positions", [])
                update_positions_count(len(pos_list))
        else:
            update_positions_count(0)
    except Exception:
        pass

    # Engine last refresh
    try:
        import json as _json

        from src.lib.core.cache import cache_get

        raw = cache_get("engine:status")
        if raw:
            status = _json.loads(raw)
            last_refresh = status.get("last_refresh_epoch")
            if last_refresh:
                ENGINE_LAST_REFRESH_EPOCH.set(float(last_refresh))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics/prometheus",
    response_class=Response,
    summary="Prometheus metrics",
    description="Returns all application metrics in Prometheus text exposition format.",
)
def prometheus_metrics():
    """Serve metrics in Prometheus text exposition format.

    Scrape target configuration for ``prometheus.yml``::

        scrape_configs:
          - job_name: 'futures-data-service'
            scrape_interval: 15s
            static_configs:
              - targets: ['data-service:8000']
            metrics_path: '/metrics/prometheus'
    """
    # Refresh gauges from live state before generating output
    _collect_live_gauges()

    # Generate Prometheus text format
    output = generate_latest(_registry)

    return Response(
        content=output,
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Convenience: get the registry (for tests or custom collectors)
# ---------------------------------------------------------------------------


def get_registry() -> CollectorRegistry:
    """Return the application's Prometheus CollectorRegistry."""
    return _registry
