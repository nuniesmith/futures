"""
Web Service — HTMX Dashboard Frontend
========================================
Thin FastAPI frontend that serves the HTMX dashboard HTML page and
reverse-proxies all API + SSE requests to the data service backend.

Architecture:
    Browser  ──→  Web Service (port 8080)  ──→  Data Service (port 8000)
                   │                              │
                   ├─ GET /  (dashboard HTML)      ├─ GET /api/focus
                   ├─ GET /sse/dashboard ──proxy──→├─ GET /sse/dashboard
                   ├─ GET /api/* ──────proxy──────→├─ GET /api/*
                   └─ static assets (Tailwind)     └─ all API endpoints

The web service is fully stateless — it never touches Redis or Postgres
directly. All data flows through the data service API layer.

Benefits of splitting:
  - Dashboard can be scaled/restarted independently of the API.
  - CDN/caching for static assets without affecting API latency.
  - Cleaner security boundary: web faces users, data faces engine.
  - Frontend can be replaced (e.g. React/Next.js) without touching API.

Environment variables:
    DATA_SERVICE_URL   — URL of the data service (default: http://data:8000)
    WEB_HOST           — Bind host (default: 0.0.0.0)
    WEB_PORT           — Bind port (default: 8080)
    LOG_LEVEL          — Logging level (default: info)

Usage:
    PYTHONPATH=src uvicorn lib.services.web.main:app --host 0.0.0.0 --port 8080

Docker:
    ENV PYTHONPATH="/app/src"
    CMD ["uvicorn", "lib.services.web.main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data:8000").rstrip("/")
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Timeout configuration for proxied requests
PROXY_TIMEOUT_DEFAULT = 15.0  # seconds for normal API calls
PROXY_CONNECT_TIMEOUT = 5.0  # seconds to establish connection

# SSE-specific: heartbeat interval on the data service is 30s, so we
# need to tolerate gaps of at least that long without any data arriving.
# 90s gives us 3x headroom before considering the upstream dead.
SSE_READ_TIMEOUT = 90.0

logger = logging.getLogger("web_service")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="[WEB] %(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# HTTP client — shared async httpx client for proxying to data service
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


# Dedicated SSE client — separate from the regular API proxy client so
# that connection-pool keepalive/expiry settings for short-lived API
# requests don't accidentally kill the long-lived SSE stream.
_sse_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client for regular API proxying."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=DATA_SERVICE_URL,
            timeout=httpx.Timeout(
                PROXY_TIMEOUT_DEFAULT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
    return _http_client


def _get_sse_client() -> httpx.AsyncClient:
    """Return a dedicated async HTTP client for SSE streaming.

    This client has NO keepalive expiry and a generous read timeout so
    that the long-lived EventSource connection isn't reaped by the
    connection pool or timed out between heartbeats.
    """
    global _sse_client
    if _sse_client is None or _sse_client.is_closed:
        _sse_client = httpx.AsyncClient(
            base_url=DATA_SERVICE_URL,
            timeout=httpx.Timeout(
                timeout=SSE_READ_TIMEOUT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=10,
                # No expiry — SSE streams are indefinitely long-lived
                keepalive_expiry=None,
            ),
        )
    return _sse_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the web service."""
    logger.info("=" * 60)
    logger.info("  Web Service starting up")
    logger.info("  Data service backend: %s", DATA_SERVICE_URL)
    logger.info("=" * 60)

    # Pre-warm the HTTP client
    _get_client()

    yield

    # Shutdown
    logger.info("Web Service shutting down...")
    global _http_client, _sse_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
    if _sse_client is not None and not _sse_client.is_closed:
        await _sse_client.aclose()
        _sse_client = None
    logger.info("Web Service stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Futures Trading Co-Pilot — Web Dashboard",
    description=(
        "HTMX dashboard frontend for the Futures Trading Co-Pilot. "
        "Serves the dashboard HTML and proxies API/SSE requests to "
        "the data service backend."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for the dashboard (browser-facing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: build proxied headers
# ---------------------------------------------------------------------------


def _proxy_headers(request: Request) -> dict[str, str]:
    """Build headers to forward to the data service.

    Strips hop-by-hop headers and adds X-Forwarded-* headers.
    """
    # Headers that should NOT be forwarded (hop-by-hop)
    skip = {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
    }

    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in skip:
            headers[key] = value

    # Add forwarding headers
    client_host = request.client.host if request.client else "unknown"
    headers["X-Forwarded-For"] = client_host
    headers["X-Forwarded-Proto"] = request.url.scheme
    headers["X-Forwarded-Host"] = request.headers.get("host", "")

    return headers


# ---------------------------------------------------------------------------
# Health check — local (no proxy needed)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Web service health check.

    Also checks connectivity to the data service backend.
    """
    backend_ok = False
    backend_status = "unknown"

    try:
        client = _get_client()
        resp = await client.get("/health", timeout=5.0)
        backend_ok = resp.status_code == 200
        backend_status = "healthy" if backend_ok else f"unhealthy (HTTP {resp.status_code})"
    except httpx.ConnectError:
        backend_status = "unreachable"
    except httpx.TimeoutException:
        backend_status = "timeout"
    except Exception as exc:
        backend_status = f"error: {exc}"

    return JSONResponse(
        status_code=200 if backend_ok else 503,
        content={
            "status": "ok" if backend_ok else "degraded",
            "service": "web",
            "backend": {
                "url": DATA_SERVICE_URL,
                "status": backend_status,
                "healthy": backend_ok,
            },
        },
    )


# ---------------------------------------------------------------------------
# Dashboard — GET / serves the full HTML page from data service
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the HTMX dashboard page.

    Fetches the full dashboard HTML from the data service and returns it.
    The dashboard's SSE and HTMX polling requests will come back to this
    web service, which proxies them to the data service.
    """
    try:
        client = _get_client()
        resp = await client.get(
            "/",
            headers=_proxy_headers(request),
            timeout=PROXY_TIMEOUT_DEFAULT,
        )
        return HTMLResponse(
            content=resp.text,
            status_code=resp.status_code,
        )
    except httpx.ConnectError:
        return HTMLResponse(
            content=_render_error_page(
                "Data Service Unavailable",
                f"Cannot connect to data service at {DATA_SERVICE_URL}. "
                "The service may be starting up — try refreshing in a few seconds.",
            ),
            status_code=503,
        )
    except httpx.TimeoutException:
        return HTMLResponse(
            content=_render_error_page(
                "Data Service Timeout",
                "The data service took too long to respond. Please try again.",
            ),
            status_code=504,
        )
    except Exception as exc:
        logger.error("Dashboard proxy error: %s", exc, exc_info=True)
        return HTMLResponse(
            content=_render_error_page("Internal Error", str(exc)),
            status_code=500,
        )


# ---------------------------------------------------------------------------
# SSE proxy — streams events from data service to browser
# ---------------------------------------------------------------------------


@app.get("/sse/dashboard")
async def sse_dashboard_proxy(request: Request):
    """Proxy the SSE dashboard stream from the data service.

    This is a long-lived streaming connection. We open a streaming
    request to the data service and forward each chunk to the browser.

    Uses a dedicated httpx client (_get_sse_client) so that the
    connection-pool keepalive settings for short-lived API requests
    don't prematurely close the SSE stream.

    If the upstream connection drops or times out (SSE_READ_TIMEOUT
    without receiving any data — including heartbeats), we send the
    browser a retry hint and close gracefully. The browser-side
    EventSource (or HTMX sse.js) will auto-reconnect.
    """

    async def _stream():
        try:
            client = _get_sse_client()
            async with client.stream(
                "GET",
                "/sse/dashboard",
                headers={
                    **_proxy_headers(request),
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                timeout=httpx.Timeout(
                    # Read timeout: if no data (not even a heartbeat)
                    # arrives within this window, assume upstream is dead.
                    timeout=SSE_READ_TIMEOUT,
                    connect=PROXY_CONNECT_TIMEOUT,
                ),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        except httpx.ConnectError:
            logger.warning("SSE proxy: cannot connect to data service")
            yield _sse_error_event("Cannot connect to data service")
            # Send a retry hint so EventSource reconnects quickly
            yield b"retry: 3000\n\n"
        except (httpx.ReadTimeout, httpx.TimeoutException):
            logger.warning("SSE proxy: upstream read timed out (no data for %ss)", SSE_READ_TIMEOUT)
            yield _sse_error_event("Data service stream timed out — reconnecting")
            yield b"retry: 2000\n\n"
        except asyncio.CancelledError:
            logger.debug("SSE proxy cancelled (client navigated away)")
        except Exception as exc:
            logger.error("SSE proxy error: %s", exc)
            yield _sse_error_event(f"Proxy error: {exc}")
            yield b"retry: 3000\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            # Prevent any intermediate proxy from buffering chunks
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/sse/health")
async def sse_health_proxy(request: Request):
    """Proxy the SSE health endpoint."""
    return await _proxy_request(request, "/sse/health")


# ---------------------------------------------------------------------------
# Favicon — return 204 (same as data service)
# ---------------------------------------------------------------------------


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# API proxy — catch-all for /api/*, /analysis/*, /actions/*, etc.
# ---------------------------------------------------------------------------

# Explicit route patterns that should be proxied to the data service.
# We list them explicitly rather than using a blanket catch-all so that
# typos/invalid routes get a clean 404 from the web service.

_PROXY_PREFIXES = (
    "/api/",
    "/analysis/",
    "/actions/",
    "/positions/",
    "/trades",
    "/log_trade",
    "/risk/",
    "/audit/",
    "/journal/",
    "/cnn/",
    "/data/",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
)


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_api(request: Request, path: str):
    """Proxy all /api/* requests to the data service."""
    return await _proxy_request(request, f"/api/{path}")


@app.api_route(
    "/analysis/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_analysis(request: Request, path: str):
    """Proxy all /analysis/* requests to the data service."""
    return await _proxy_request(request, f"/analysis/{path}")


@app.api_route(
    "/actions/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_actions(request: Request, path: str):
    """Proxy all /actions/* requests to the data service."""
    return await _proxy_request(request, f"/actions/{path}")


@app.api_route(
    "/positions/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_positions(request: Request, path: str):
    """Proxy all /positions/* requests to the data service."""
    return await _proxy_request(request, f"/positions/{path}")


@app.api_route(
    "/trades",
    methods=["GET", "POST"],
)
async def proxy_trades_root(request: Request):
    """Proxy /trades to the data service."""
    return await _proxy_request(request, "/trades")


@app.api_route(
    "/trades/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_trades(request: Request, path: str):
    """Proxy all /trades/* requests to the data service."""
    return await _proxy_request(request, f"/trades/{path}")


@app.api_route(
    "/log_trade",
    methods=["POST"],
)
async def proxy_log_trade(request: Request):
    """Proxy /log_trade to the data service."""
    return await _proxy_request(request, "/log_trade")


@app.api_route(
    "/risk/{path:path}",
    methods=["GET", "POST"],
)
async def proxy_risk(request: Request, path: str):
    """Proxy all /risk/* requests to the data service."""
    return await _proxy_request(request, f"/risk/{path}")


@app.api_route(
    "/audit/{path:path}",
    methods=["GET", "POST"],
)
async def proxy_audit(request: Request, path: str):
    """Proxy all /audit/* requests to the data service."""
    return await _proxy_request(request, f"/audit/{path}")


@app.api_route(
    "/journal/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_journal(request: Request, path: str):
    """Proxy all /journal/* requests to the data service."""
    return await _proxy_request(request, f"/journal/{path}")


@app.api_route(
    "/cnn/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_cnn(request: Request, path: str):
    """Proxy all /cnn/* requests to the data service (CNN model management).

    Methods include PUT and DELETE for the per-session CNN gate endpoints:
        GET    /cnn/gate                  — view all gate states
        PUT    /cnn/gate/{session_key}    — enable/disable gate for one session
        DELETE /cnn/gate/{session_key}    — remove Redis override for one session
        DELETE /cnn/gate                  — remove all overrides
        GET    /cnn/gate/html             — dashboard HTML fragment
    """
    return await _proxy_request(request, f"/cnn/{path}")


@app.api_route(
    "/kraken/{path:path}",
    methods=["GET"],
)
async def proxy_kraken(request: Request, path: str):
    """Proxy all /kraken/* requests to the data service (Kraken crypto exchange).

    Endpoints:
        GET /kraken/health          — Kraken connectivity + auth status
        GET /kraken/status          — WebSocket feed status + pair list
        GET /kraken/pairs           — Available Kraken pairs and their mappings
        GET /kraken/ticker/{pair}   — Current ticker snapshot for a pair
        GET /kraken/tickers         — All tracked pair tickers in one call
        GET /kraken/ohlcv/{pair}    — Historical OHLCV bars for a pair
        GET /kraken/health/html     — Dashboard HTML fragment for Kraken status
    """
    return await _proxy_request(request, f"/kraken/{path}")


@app.api_route(
    "/data/{path:path}",
    methods=["GET"],
)
async def proxy_data(request: Request, path: str):
    """Proxy all /data/* requests to the data service."""
    return await _proxy_request(request, f"/data/{path}")


@app.get("/metrics")
async def proxy_metrics_root(request: Request):
    """Proxy /metrics to the data service."""
    return await _proxy_request(request, "/metrics")


@app.get("/metrics/{path:path}")
async def proxy_metrics(request: Request, path: str):
    """Proxy /metrics/* to the data service."""
    return await _proxy_request(request, f"/metrics/{path}")


@app.get("/docs")
async def proxy_docs(request: Request):
    """Proxy /docs to the data service."""
    return await _proxy_request(request, "/docs")


@app.get("/openapi.json")
async def proxy_openapi(request: Request):
    """Proxy /openapi.json to the data service."""
    return await _proxy_request(request, "/openapi.json")


# ---------------------------------------------------------------------------
# Generic proxy helper
# ---------------------------------------------------------------------------


async def _proxy_request(request: Request, path: str) -> Response:
    """Forward an HTTP request to the data service and return the response.

    Handles all HTTP methods, query params, body, and headers.
    Returns the upstream response with matching status code and headers.
    """
    client = _get_client()

    # Build the target URL with query parameters
    url = path
    if request.query_params:
        url = f"{path}?{urlencode(dict(request.query_params))}"

    # Read the request body for POST/PUT/PATCH
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    headers = _proxy_headers(request)

    try:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            timeout=PROXY_TIMEOUT_DEFAULT,
        )

        # Filter response headers — strip hop-by-hop and server headers
        response_headers = {}
        skip_response = {
            "transfer-encoding",
            "connection",
            "keep-alive",
            "server",
        }
        for key, value in resp.headers.items():
            if key.lower() not in skip_response:
                response_headers[key] = value

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Data service unavailable",
                "detail": f"Cannot connect to {DATA_SERVICE_URL}",
            },
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Data service timeout",
                "detail": "Request to data service timed out",
            },
        )
    except Exception as exc:
        logger.error("Proxy error for %s %s: %s", request.method, path, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": "Proxy error",
                "detail": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Error page template
# ---------------------------------------------------------------------------


def _render_error_page(title: str, message: str) -> str:
    """Render a minimal error page with auto-refresh."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Futures Trading Co-Pilot</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #09090b;
            color: #a1a1aa;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            text-align: center;
            max-width: 480px;
            padding: 2rem;
        }}
        h1 {{
            color: #ef4444;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}
        p {{
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }}
        .retry {{
            color: #71717a;
            font-size: 0.875rem;
        }}
        .spinner {{
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid #3f3f46;
            border-top-color: #a1a1aa;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
        <h1>{title}</h1>
        <p>{message}</p>
        <div class="retry">
            <span class="spinner"></span>
            Auto-refreshing in 10 seconds...
        </div>
    </div>
</body>
</html>"""


def _sse_error_event(message: str) -> bytes:
    """Format an SSE error event as bytes."""
    import json

    payload = json.dumps({"type": "error", "message": message})
    event = f"event: error\ndata: {payload}\n\n"
    return event.encode("utf-8")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level=LOG_LEVEL,
    )
