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
import os
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from lib.core.logging_config import get_logger, setup_logging
from lib.services.web.auth import SessionAuthMiddleware, is_auth_enabled
from lib.services.web.auth import router as auth_router

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data:8000").rstrip("/")
CHARTING_SERVICE_URL = os.getenv("CHARTING_SERVICE_URL", "http://charting:8003").rstrip("/")

# Shared secret for authenticating with the data service.
# Must match the API_KEY set on the data service container.
# When empty, no header is injected (data service auth is also disabled).
_DATA_SERVICE_API_KEY: str = os.getenv("API_KEY", "").strip()

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

setup_logging(service="web")
logger = get_logger("web")

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


# ---------------------------------------------------------------------------
# Charting HTTP client — separate pool for the charting service (port 8003)
# ---------------------------------------------------------------------------

_charting_client: httpx.AsyncClient | None = None


def _get_charting_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client for charting service proxying."""
    global _charting_client
    if _charting_client is None or _charting_client.is_closed:
        _charting_client = httpx.AsyncClient(
            base_url=CHARTING_SERVICE_URL,
            timeout=httpx.Timeout(
                PROXY_TIMEOUT_DEFAULT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _charting_client


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
    logger.info("  Data service backend:     %s", DATA_SERVICE_URL)
    logger.info("  Charting service backend: %s", CHARTING_SERVICE_URL)
    logger.info("  Trainer routed via:       data service /trainer/*")
    logger.info("=" * 60)

    # Pre-warm the HTTP clients
    _get_client()
    _get_charting_client()

    yield

    # Shutdown
    logger.info("Web Service shutting down...")
    global _http_client, _sse_client, _charting_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
    if _sse_client is not None and not _sse_client.is_closed:
        await _sse_client.aclose()
        _sse_client = None
    if _charting_client is not None and not _charting_client.is_closed:
        await _charting_client.aclose()
        _charting_client = None
    logger.info("Web Service stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ruby — Web Dashboard",
    description=(
        "HTMX dashboard frontend for Ruby. "
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
# Auth — session cookie middleware + /login + /logout
# ---------------------------------------------------------------------------
# Registers routes first so the middleware can redirect to /login.
# When WEB_PASSWORD_HASH is not set, auth is fully disabled (dev mode).
# ---------------------------------------------------------------------------
app.include_router(auth_router)

if is_auth_enabled():
    app.add_middleware(SessionAuthMiddleware)
    logger.info("Session auth enabled — password hash configured")
else:
    logger.warning("Session auth DISABLED — set WEB_PASSWORD_HASH in .env to enable")


# ---------------------------------------------------------------------------
# Helper: build proxied headers
# ---------------------------------------------------------------------------


def _proxy_headers(request: Request) -> dict[str, str]:
    """Build headers to forward to the data service.

    Strips hop-by-hop headers and adds X-Forwarded-* headers.
    Always injects the server-side API_KEY as X-API-Key so the data
    service authenticates the web service regardless of what the
    browser sends.
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
        # Strip any key the browser may have sent — we always use the
        # server-side key so a missing/wrong browser key can't bypass auth.
        "x-api-key",
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

    # Inject the server-side API key so the data service accepts the request.
    if _DATA_SERVICE_API_KEY:
        headers["X-API-Key"] = _DATA_SERVICE_API_KEY

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
# /config — charting service discovery endpoint
#
# chart.js fetches GET /config from whatever origin the chart was loaded on.
# When the chart is embedded via /charting-proxy/ the origin is the web
# service (port 443 / 8080), so this endpoint must live here.
#
# Returns the data service URL (internal Docker address) — chart.js detects
# it as internal and routes all API calls through the web service proxy
# instead of trying to reach the data container directly from the browser.
# ---------------------------------------------------------------------------


@app.get("/config", include_in_schema=False)
async def charting_config():
    """Return charting service discovery config for chart.js.

    chart.js calls GET /config on boot to resolve the data service URL.
    We return the DATA_SERVICE_URL env var (which is the internal Docker
    address, e.g. http://data:8000).  chart.js will detect it as an
    internal address and route all bar/SSE requests through the web
    service proxy (/bars/*, /sse/*) which injects the API key header.
    """
    return JSONResponse(
        {
            "data_service_url": DATA_SERVICE_URL,
            "web_service_port": WEB_PORT,
            "proxy_via_web": True,
        }
    )


# ---------------------------------------------------------------------------
# SSE proxy helper (for long-lived streaming endpoints)
# ---------------------------------------------------------------------------


async def _proxy_sse_request(request: Request, path: str) -> StreamingResponse:
    """Forward an SSE request to the data service and stream the response.

    Uses the dedicated SSE client with longer timeouts for long-lived
    streaming connections (pipeline run, live price stream, etc.).
    """
    client = _get_sse_client()

    url = path
    if request.query_params:
        url = f"{path}?{request.query_params}"

    async def _stream():
        try:
            async with client.stream(
                "GET",
                url,
                headers=_proxy_headers(request),
                timeout=httpx.Timeout(
                    connect=PROXY_CONNECT_TIMEOUT,
                    read=SSE_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                ),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.ConnectError:
            yield b'data: {"type":"error","message":"Data service unavailable"}\n\n'
        except httpx.TimeoutException:
            yield b'data: {"type":"error","message":"Data service timeout"}\n\n'
        except Exception as exc:
            logger.error("SSE proxy error for %s: %s", path, exc)
            err_msg = str(exc).replace('"', '\\"')
            yield f'data: {{"type":"error","message":"{err_msg}"}}\n\n'.encode()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# SSE proxy routes — registered BEFORE the catch-all prefix routes so that
# /api/{path:path} does not shadow them.  SSE endpoints require the
# dedicated streaming client, not the regular _proxy_request helper.
# ---------------------------------------------------------------------------


@app.get("/api/pipeline/run")
async def proxy_pipeline_run(request: Request):
    """SSE proxy for the morning analysis pipeline."""
    return await _proxy_sse_request(request, "/api/pipeline/run")


@app.get("/api/live/stream")
async def proxy_live_stream(request: Request):
    """SSE proxy for live price/signal stream."""
    return await _proxy_sse_request(request, "/api/live/stream")


@app.get("/sse/chat")
async def proxy_sse_chat(request: Request):
    """Proxy the streaming SSE chat endpoint from the data service.

    Uses the dedicated SSE client (no keepalive expiry) so long-running
    chat streams are not reaped by the connection pool.
    """
    return await _proxy_sse_request(request, "/sse/chat")


@app.get("/sse/posint")
async def proxy_sse_posint(request: Request):
    """Proxy the position intelligence SSE stream from the data service.

    Forwards ``/sse/posint`` to the data service which polls Redis for
    ``engine:posint:*`` keys every 5 seconds and emits ``posint-update``
    events keyed by symbol.

    Browser usage::

        const es = new EventSource('/sse/posint');
        es.addEventListener('posint-update', (e) => {
            const data = JSON.parse(e.data);
            // data is keyed by symbol: { "MGC": {...}, "MES": {...} }
        });
    """
    return await _proxy_sse_request(request, "/sse/posint")


@app.get("/sse/bars/{symbol:path}")
async def proxy_sse_bars(request: Request, symbol: str):
    """Proxy the live 1-minute bar SSE stream for a symbol.

    Forwards ``/sse/bars/{symbol}`` to the data service which emits
    ``bar`` events as each candle closes.  Uses the dedicated SSE client
    with long read timeouts so the connection is not reaped by the pool.

    Browser usage::

        const es = new EventSource('/sse/bars/MGC');
        es.addEventListener('bar', (e) => {
            const bar = JSON.parse(e.data);  // {time, open, high, low, close, volume}
            candleSeries.update(bar);
        });
    """
    return await _proxy_sse_request(request, f"/sse/bars/{symbol}")


@app.get("/sse/strip")
async def proxy_sse_strip(request: Request):
    """Proxy the persistent strip SSE stream from the data service.

    The strip SSE stream pushes live price + P&L updates to the browser so
    the persistent top-strip cells update in real time without HTMX polling.
    Uses the dedicated SSE client (long timeouts, no keepalive expiry).
    """
    return await _proxy_sse_request(request, "/sse/strip")


# ---------------------------------------------------------------------------
# Special page handlers — routes with non-trivial logic that cannot be
# replaced by the generic catch-all proxy loop below.
# ---------------------------------------------------------------------------


@app.get("/rb-history", response_class=HTMLResponse)
async def proxy_rb_history(request: Request):
    """Canonical RB History path — proxies to /orb-history on the data service."""
    return await _proxy_request(request, "/orb-history")


@app.get("/trading", response_class=HTMLResponse)
async def serve_trading_page(request: Request):
    """Serve the trading page directly — no iframe wrapper.

    trading.html now has its own fully-styled .co-nav bar (display:flex,
    height:44px, etc.) so it renders the platform nav bar natively.
    We just fetch the raw HTML from the data service and return it.

    This avoids the iframe approach which caused height/sizing issues
    that made the content area invisible.
    """
    # Fetch trading.html from data service
    try:
        resp = await _proxy_request(request, "/trading/app")
        if resp.status_code == 200:
            return Response(
                content=resp.body,
                status_code=200,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"},
            )
    except Exception:
        pass

    # Fall back to the data service's shell+iframe version as last resort
    return await _proxy_request(request, "/trading")


@app.get("/chart", response_class=HTMLResponse)
async def serve_chart_page(request: Request):
    """Serve the TradingView Lightweight Charts v4 charting page.

    Reads ``static/chart.html`` directly from the web service's file system
    (same approach as the ``/charts`` route) so the page is available even
    when the data service is temporarily unreachable.

    API endpoints consumed by the page (all proxied through this service):
        GET  /bars/{symbol}/candles          — historical OHLCV bars
        GET  /api/chart/{symbol}/indicators  — pre-computed indicator series
        GET  /api/chart/{symbol}/regime      — current market regime
        GET  /api/chart/assets               — enabled asset list
        GET  /sse/bars/{symbol}              — live bar stream
        GET  /sse/dashboard                  — Janus signal events
    """
    import pathlib  # noqa: PLC0415

    _candidates = [
        pathlib.Path("/app/static/chart.html"),
        pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent / "static" / "chart.html",
        pathlib.Path(__file__).resolve().parent.parent.parent.parent / "static" / "chart.html",
        pathlib.Path.cwd() / "static" / "chart.html",
    ]
    for p in _candidates:
        if p.exists():
            return HTMLResponse(content=p.read_text(), headers={"Cache-Control": "no-cache"})

    logger.warning("chart.html not found in any candidate path")
    return HTMLResponse(
        content=(
            "<html><body style='background:#0a0a0f;color:#94a3b8;font-family:monospace;"
            "display:flex;align-items:center;justify-content:center;height:100vh'>"
            "<div style='text-align:center'><div style='font-size:2rem'>📈</div>"
            "chart.html not found — place it at <code>static/chart.html</code></div></body></html>"
        ),
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Charts page — local HTML with nav bar + charting iframe
# ---------------------------------------------------------------------------


@app.get("/charts", response_class=HTMLResponse)
async def proxy_charts(request: Request):
    """Serve the Charts page directly — nav bar + charting iframe.

    Previously this proxied to the data service which generated the same
    HTML via ``charts_page()``.  Generating it here removes a proxy hop
    and guarantees the nav bar always renders even if the data service is
    temporarily unreachable.
    """
    charting_proxy = "/charting-proxy/"
    return HTMLResponse(content=_charts_page_html(charting_proxy))


def _charts_page_html(charting_proxy: str) -> str:
    """Return complete Charts page HTML with nav bar + charting iframe."""
    return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>
<title>Charts — Ruby</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>"/>
<script>(function(){{
  var t=localStorage.getItem('theme');
  if(t==='light') document.documentElement.classList.remove('dark');
  else document.documentElement.classList.add('dark');
}})();</script>
<style>
/* ── Reset ── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{-webkit-text-size-adjust:100%}}

/* ── Theme variables ── */
:root{{
  --bg-body:#f4f4f5;
  --bg-panel:rgba(255,255,255,0.85);
  --bg-panel-inner:rgba(244,244,245,0.6);
  --bg-input:#e4e4e7;
  --border-panel:#d4d4d8;
  --border-subtle:#e4e4e7;
  --text-primary:#18181b;
  --text-secondary:#3f3f46;
  --text-muted:#71717a;
  --text-faint:#a1a1aa;
}}
.dark{{
  --bg-body:#09090b;
  --bg-panel:rgba(24,24,27,0.75);
  --bg-panel-inner:rgba(39,39,42,0.5);
  --bg-input:#27272a;
  --border-panel:#3f3f46;
  --border-subtle:#27272a;
  --text-primary:#f4f4f5;
  --text-secondary:#d4d4d8;
  --text-muted:#71717a;
  --text-faint:#52525b;
}}
body{{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg-body);
  color:var(--text-primary);
  min-height:100vh;
  line-height:1.5;
  font-size:13px;
}}
a{{color:inherit}}

/* ── Shared nav bar ── */
.co-nav{{
  display:flex;align-items:center;gap:0;
  padding:0 1rem;background:var(--bg-panel);
  border-bottom:1px solid var(--border-subtle);
  height:44px;position:sticky;top:0;z-index:200;
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
}}
.co-nav-brand{{
  font-weight:700;font-size:.9rem;color:var(--text-primary);
  text-decoration:none;margin-right:1.25rem;letter-spacing:-.02em;white-space:nowrap;
}}
.co-nav-tab{{
  display:inline-flex;align-items:center;gap:5px;
  padding:5px 12px;border-radius:6px;text-decoration:none;
  color:var(--text-muted);font-size:.78rem;font-weight:500;
  transition:background .12s,color .12s;white-space:nowrap;
}}
.co-nav-tab:hover{{background:var(--bg-input);color:var(--text-primary)}}
.co-nav-tab.active{{background:var(--bg-input);color:var(--text-primary);font-weight:650}}
.co-nav-right{{margin-left:auto;display:flex;align-items:center;gap:8px}}
.co-theme-btn{{
  background:none;border:1px solid var(--border-panel);border-radius:6px;
  padding:4px 8px;color:var(--text-muted);cursor:pointer;font-size:.75rem;
  transition:all .12s;font-family:inherit;
}}
.co-theme-btn:hover{{color:var(--text-primary);border-color:var(--text-primary)}}

/* ── Charts-specific layout ── */
html, body {{ height: 100%; overflow: hidden; }}
.co-page {{
  padding: 0 !important;
  max-width: 100% !important;
  height: calc(100vh - 44px);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
#charts-frame {{
  flex: 1;
  min-height: 0;
  width: 100%;
  border: none;
  background: #0a0a0f;
}}
#charts-offline {{
  display: none;
  flex: 1;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 12px;
  color: var(--text-muted, #71717a);
  font-size: 0.85rem;
}}
</style>
</head>
<body>
<nav class="co-nav">
  <a class="co-nav-brand" href="/">💎 FKS Ruby</a>
  <a class="co-nav-tab" href="/">📊 Dashboard</a>
  <a class="co-nav-tab" href="/trading">🚀 Trading</a>
  <a class="co-nav-tab  active" href="/charts">📈 Charts</a>
  <a class="co-nav-tab" href="/account">💰 Account</a>
  <a class="co-nav-tab" href="/signals">📡 Signals</a>
  <a class="co-nav-tab" href="/journal/page">📓 Journal</a>
  <a class="co-nav-tab" href="/trainer">🧠 Trainer</a>
  <a class="co-nav-tab" href="/settings">⚙️ Settings</a>
  <div class="co-nav-right">
    <button class="co-theme-btn" onclick="toggleTheme()">☀/🌙</button>
    <a href="/logout" class="co-nav-tab" style="font-size:11px;opacity:.6;margin-left:4px" title="Sign out">⏻</a>
  </div>
</nav>
<div class="co-page">

    <iframe
      id="charts-frame"
      src="{charting_proxy}"
      title="Ruby Charts"
      allow="fullscreen"
      loading="eager"
    ></iframe>

    <div id="charts-offline">
      <span style="font-size:2rem">📡</span>
      <div style="font-weight:600;color:var(--text-primary,#e4e4e7)">Charting service unavailable</div>
      <div>Make sure the <code>charting</code> container is running (proxied via <code>{charting_proxy}</code>)</div>
      <button onclick="document.getElementById('charts-frame').src=document.getElementById('charts-frame').src"
              style="margin-top:8px;padding:6px 16px;background:#2563eb;color:#fff;
                     border:none;border-radius:6px;cursor:pointer;font-size:0.8rem">
        ↺ Retry
      </button>
    </div>

    <script>
    (function() {{
      var frame = document.getElementById('charts-frame');
      var offline = document.getElementById('charts-offline');
      frame.addEventListener('error', function() {{
        frame.style.display = 'none';
        offline.style.display = 'flex';
      }});
      fetch('{charting_proxy}', {{method:'HEAD'}})
        .catch(function() {{
          frame.style.display = 'none';
          offline.style.display = 'flex';
        }});
    }})();
    </script>

</div>
<script>
function toggleTheme(){{
  var h=document.documentElement;
  if(h.classList.contains('dark')){{h.classList.remove('dark');localStorage.setItem('theme','light');}}
  else{{h.classList.add('dark');localStorage.setItem('theme','dark');}}
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Catch-all proxy routes — data service pass-through
#
# Instead of defining 100+ individual one-liner proxy functions, we use a
# factory pattern to register catch-all routes for each URL prefix.  Every
# request matching /{prefix}/{path} is forwarded as-is to the data service
# via _proxy_request().
#
# Routes that need special handling (SSE streaming, local HTML generation,
# path remapping) are registered explicitly ABOVE this section so they
# take priority over the catch-all patterns.
# ---------------------------------------------------------------------------

_ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def _make_prefix_proxy(prefix: str):
    """Create a catch-all proxy handler for ``/{prefix}/{path:path}``."""

    async def _handler(request: Request, path: str):
        return await _proxy_request(request, f"/{prefix}/{path}")

    name = f"proxy_{prefix.replace('-', '_')}"
    _handler.__name__ = name
    _handler.__qualname__ = name
    return _handler


def _make_root_proxy(prefix: str):
    """Create a proxy handler for the bare ``/{prefix}`` root (no sub-path)."""

    async def _handler(request: Request):
        return await _proxy_request(request, f"/{prefix}")

    name = f"proxy_{prefix.replace('-', '_')}_root"
    _handler.__name__ = name
    _handler.__qualname__ = name
    return _handler


def _make_fixed_proxy(target_path: str):
    """Create a proxy handler for a fixed (non-parameterized) path."""

    async def _handler(request: Request):
        return await _proxy_request(request, target_path)

    name = "proxy_" + target_path.strip("/").replace("/", "_").replace("-", "_").replace(".", "_")
    _handler.__name__ = name
    _handler.__qualname__ = name
    return _handler


# ── Prefix-based catch-all routes ──────────────────────────────────────────
# Each prefix gets two routes:
#   /{prefix}              — root (e.g. GET /trades, GET /settings)
#   /{prefix}/{path:path}  — catch-all sub-paths (e.g. GET /api/health/html)
#
# The data service decides which methods are valid; the web proxy simply
# forwards the request and relays whatever status code comes back.

_PROXY_PREFIXES = [
    "api",
    "analysis",
    "actions",
    "positions",
    "trades",
    "risk",
    "audit",
    "journal",
    "cnn",
    "kraken",
    "bars",
    "data",
    "metrics",
    "trainer",
    "settings",
    "workspaces",
    "partials",
    "trading",
]

for _pfx in _PROXY_PREFIXES:
    app.add_api_route(
        f"/{_pfx}",
        _make_root_proxy(_pfx),
        methods=_ALL_METHODS,
    )
    app.add_api_route(
        f"/{_pfx}/{{path:path}}",
        _make_prefix_proxy(_pfx),
        methods=_ALL_METHODS,
    )


# ── Fixed-path proxy routes ───────────────────────────────────────────────
# Standalone pages / endpoints that don't fall under any prefix catch-all.

_PROXY_PAGES = [
    "/log_trade",
    "/docs",
    "/openapi.json",
    "/orb-history",
    "/account",
    "/connections",
    "/signals",
    "/backup",
    "/net-worth",
    "/chat",
    "/pine",
]

for _page in _PROXY_PAGES:
    app.add_api_route(_page, _make_fixed_proxy(_page), methods=_ALL_METHODS)


# ---------------------------------------------------------------------------
# Charting proxy — forwards /charting-proxy/* and /charting/* to charting
#   service (nginx + chart.js on port 8003)
#
# The charting container serves static TradingView Lightweight Charts assets
# (HTML, JS, CSS, images) via Nginx.  These routes let the browser access
# charting assets through the web proxy instead of connecting directly to
# port 8003.
#
# Route map (web → charting service):
#   GET /charting-proxy/                → charting /
#   GET /charting-proxy/{path}          → charting /{path}
#   GET /charting/{path}                → charting /{path}
#   Any method /charting-proxy/api/*    → charting /api/*
# ---------------------------------------------------------------------------


async def _proxy_charting_request(request: Request, upstream_path: str) -> Response:
    """Forward an HTTP request to the charting service and return the response."""
    client = _get_charting_client()

    url = upstream_path
    if request.query_params:
        url = f"{upstream_path}?{urlencode(dict(request.query_params))}"

    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    # Forward only safe headers — strip hop-by-hop and host headers
    skip = {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
    }
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    if request.client:
        fwd_headers["X-Forwarded-For"] = request.client.host
    fwd_headers["X-Forwarded-Proto"] = request.url.scheme
    fwd_headers["X-Forwarded-Host"] = request.headers.get("host", "")

    try:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=fwd_headers,
            content=body,
        )

        excluded_resp = {
            "transfer-encoding",
            "content-encoding",
            "content-length",
            "connection",
            "keep-alive",
            "server",
        }
        resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_resp}

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Charting service unavailable",
                "detail": f"Cannot connect to {CHARTING_SERVICE_URL}",
                "hint": "Make sure the charting container is running: docker compose up -d charting",
            },
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Charting service timeout",
                "detail": "Request to charting service timed out",
            },
        )
    except Exception as exc:
        logger.error("Charting proxy error for %s %s: %s", request.method, upstream_path, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": "Charting proxy error",
                "detail": str(exc),
            },
        )


@app.api_route("/charting-proxy/", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_charting_root(request: Request):
    """Proxy charting root (index.html) from the charting service.

    HEAD and OPTIONS are included so the dashboard's fetch probe and CORS
    preflight requests don't get a 405 Method Not Allowed.
    """
    return await _proxy_charting_request(request, "/")


@app.api_route(
    "/charting-proxy/{path:path}",
    methods=["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_charting_path(request: Request, path: str):
    """Proxy all /charting-proxy/* requests to the charting service.

    Strips the ``/charting-proxy`` prefix before forwarding:
        GET  /charting-proxy/index.html  → charting GET /index.html
        GET  /charting-proxy/js/app.js   → charting GET /js/app.js
        GET  /charting-proxy/api/bars    → charting GET /api/bars
    """
    return await _proxy_charting_request(request, f"/{path}")


@app.api_route(
    "/charting/{path:path}",
    methods=["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_charting_assets(request: Request, path: str):
    """Proxy /charting/* for direct access to charting static assets.

    Strips the ``/charting`` prefix before forwarding:
        GET  /charting/index.html  → charting GET /index.html
        GET  /charting/js/app.js   → charting GET /js/app.js
    """
    return await _proxy_charting_request(request, f"/{path}")


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
    <title>{title} — Ruby</title>
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
