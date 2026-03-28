"""
Session Authentication — Ruby Web Service
==========================================
Dead-simple single-user session auth for the HTMX dashboard.

Design goals:
  - Zero extra containers (no Authelia, no DB)
  - One password stored as a bcrypt hash in WEB_PASSWORD_HASH env var
  - Signed HttpOnly session cookie — 30-day expiry by default
  - HMAC-SHA256 cookie signature using WEB_SESSION_SECRET
  - Constant-time comparisons throughout (no timing oracles)
  - /login and /logout routes injected directly into the FastAPI app
  - Starlette middleware intercepts every request before routing

Configuration (.env):
    WEB_PASSWORD_HASH    — bcrypt hash of your password.
                           Generate with:  ./run.sh web-hash-password
                           or:  python -c "import bcrypt; print(bcrypt.hashpw(b'pw', bcrypt.gensalt()).decode())"
                           When UNSET or empty, auth is disabled (dev mode).

    WEB_SESSION_SECRET   — Random secret for HMAC cookie signing.
                           Generate with:  python -c "import secrets; print(secrets.token_hex(32))"
                           Required when WEB_PASSWORD_HASH is set.

    WEB_SESSION_TTL_DAYS — Session lifetime in days (default: 30).

Cookie:
    Name:     fks_session
    Flags:    HttpOnly, SameSite=Lax, Secure (when behind HTTPS proxy)
    Value:    <timestamp>.<hmac_hex>
    Signing:  HMAC-SHA256(secret, "fks_session:<timestamp>")
"""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import secrets
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PASSWORD_HASH: str = os.getenv("WEB_PASSWORD_HASH", "").strip()
_SESSION_SECRET: str = os.getenv("WEB_SESSION_SECRET", "").strip()
_SESSION_TTL: int = int(os.getenv("WEB_SESSION_TTL_DAYS", "30")) * 86400

_COOKIE_NAME = "fks_session"

# Paths that are always accessible without a session.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/login",
        "/logout",
        "/health",
        "/favicon.ico",
    }
)

# ---------------------------------------------------------------------------
# Auth state helpers
# ---------------------------------------------------------------------------


def is_auth_enabled() -> bool:
    """Return True if a password hash has been configured."""
    return bool(_PASSWORD_HASH)


def _sign(timestamp: str) -> str:
    """Return HMAC-SHA256 hex signature for the given timestamp."""
    msg = f"fks_session:{timestamp}".encode()
    return hmac.new(_SESSION_SECRET.encode(), msg, hashlib.sha256).hexdigest()


def _make_cookie_value() -> str:
    """Generate a fresh signed session cookie value."""
    ts = str(int(time.time()))
    sig = _sign(ts)
    return f"{ts}.{sig}"


def _verify_cookie(value: str) -> bool:
    """Return True if the cookie value has a valid signature and is not expired."""
    try:
        ts_str, sig = value.split(".", 1)
        ts = int(ts_str)
    except (ValueError, AttributeError):
        return False

    # Constant-time signature check
    expected = _sign(ts_str)
    if not secrets.compare_digest(sig, expected):
        return False

    # Expiry check
    return not time.time() - ts > _SESSION_TTL


def is_authenticated(request: Request) -> bool:
    """Return True if the request carries a valid session cookie."""
    if not is_auth_enabled():
        return True
    cookie = request.cookies.get(_COOKIE_NAME, "")
    return bool(cookie) and _verify_cookie(cookie)


def _verify_password(plaintext: str) -> bool:
    """Return True if plaintext matches the configured bcrypt hash."""
    if not _PASSWORD_HASH:
        return False
    try:
        import bcrypt  # lazy import — only needed at login time

        return bcrypt.checkpw(plaintext.encode(), _PASSWORD_HASH.encode())
    except Exception:
        return False


def _set_session_cookie(response: Response, *, secure: bool = False) -> None:
    """Write a fresh signed session cookie onto response."""
    response.set_cookie(
        key=_COOKIE_NAME,
        value=_make_cookie_value(),
        max_age=_SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    """Expire the session cookie."""
    response.delete_cookie(key=_COOKIE_NAME, path="/")


# ---------------------------------------------------------------------------
# Login page HTML
# ---------------------------------------------------------------------------

_LOGIN_PAGE = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>FKS Trading — Sign In</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>💎</text></svg>"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {{
      --bg:      #0a0a0f;
      --card:    #12121a;
      --border:  #1e1e2e;
      --accent:  #00d4ff;
      --abg:     rgba(0,212,255,.08);
      --ab:      rgba(0,212,255,.25);
      --green:   #00e676;
      --red:     #ff4560;
      --text:    #e0e0e0;
      --dim:     #6b7394;
      --mono:    "JetBrains Mono", monospace;
      --head:    "Syne", sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      font-size: 13px;
    }}
    body {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,255,.06) 0%, transparent 60%),
        var(--bg);
    }}
    .card {{
      width: 100%;
      max-width: 380px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 40px 36px 36px;
      box-shadow: 0 24px 64px rgba(0,0,0,.6);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 32px;
    }}
    .brand-icon {{ font-size: 28px; line-height: 1; }}
    .brand-text {{
      font-family: var(--head);
      font-size: 20px;
      font-weight: 800;
      color: var(--accent);
      letter-spacing: .5px;
    }}
    .brand-sub {{
      font-size: 11px;
      color: var(--dim);
      margin-top: 1px;
      letter-spacing: .5px;
    }}
    label {{
      display: block;
      font-size: 11px;
      color: var(--dim);
      letter-spacing: .8px;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    input[type=password] {{
      width: 100%;
      background: #0d0d14;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 11px 14px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 14px;
      outline: none;
      transition: border-color .15s;
    }}
    input[type=password]:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--abg);
    }}
    .field {{ margin-bottom: 24px; }}
    button[type=submit] {{
      width: 100%;
      padding: 12px;
      background: var(--abg);
      border: 1px solid var(--ab);
      border-radius: 6px;
      color: var(--accent);
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: .5px;
      cursor: pointer;
      transition: background .15s, box-shadow .15s;
    }}
    button[type=submit]:hover {{
      background: rgba(0,212,255,.14);
      box-shadow: 0 0 16px rgba(0,212,255,.15);
    }}
    .error {{
      margin-bottom: 20px;
      padding: 10px 14px;
      background: rgba(255,69,96,.08);
      border: 1px solid rgba(255,69,96,.25);
      border-radius: 6px;
      color: #ff4560;
      font-size: 12px;
    }}
    .footer {{
      margin-top: 28px;
      text-align: center;
      font-size: 11px;
      color: var(--dim);
    }}
    .dot {{
      display: inline-block;
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--green);
      margin-right: 5px;
      vertical-align: middle;
      box-shadow: 0 0 6px var(--green);
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="brand">
      <span class="brand-icon">💎</span>
      <div>
        <div class="brand-text">FKS Trading</div>
        <div class="brand-sub">PERSONAL TRADING SYSTEM</div>
      </div>
    </div>

    {error_block}

    <form method="post" action="/login" autocomplete="off">
      <input type="hidden" name="next" value="{next_url}"/>
      <div class="field">
        <label for="password">Password</label>
        <input
          type="password"
          id="password"
          name="password"
          placeholder="••••••••••••"
          autofocus
          autocomplete="current-password"
        />
      </div>
      <button type="submit">SIGN IN →</button>
    </form>

    <div class="footer">
      <span class="dot"></span>Systems operational
    </div>
  </div>
</body>
</html>
"""

_ERROR_BLOCK = '<div class="error">⚠ {message}</div>'


def _login_page(*, error: str = "", next_url: str = "/") -> HTMLResponse:
    """Render the login page, optionally with an error message."""
    error_block = _ERROR_BLOCK.format(message=html.escape(error)) if error else ""
    body = _LOGIN_PAGE.format(
        error_block=error_block,
        next_url=html.escape(next_url, quote=True),
    )
    return HTMLResponse(content=body)


# ---------------------------------------------------------------------------
# Router — /login and /logout
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Auth"])


@router.get("/login", response_class=HTMLResponse, response_model=None, include_in_schema=False)
async def login_page(request: Request) -> HTMLResponse | RedirectResponse:
    """Render the login form.  Redirect to / if already authenticated."""
    if is_authenticated(request):
        next_url = request.query_params.get("next", "/")
        return RedirectResponse(url=next_url, status_code=303)
    next_url = request.query_params.get("next", "/")
    return _login_page(next_url=next_url)


@router.post("/login", include_in_schema=False)
async def login_submit(
    request: Request,
    password: str = Form(...),
    next: str = Form(default="/"),
) -> Response:
    """Validate password and issue session cookie."""
    # Reject obviously bad next URLs (open redirect guard)
    safe_next = next if (next.startswith("/") and not next.startswith("//")) else "/"

    if not _verify_password(password):
        # Brief sleep deters rapid brute-force attempts even without rate-limiting
        import asyncio

        await asyncio.sleep(0.4)
        logger.warning(
            "Failed login attempt from %s",
            request.client.host if request.client else "unknown",
        )
        return _login_page(error="Invalid password.", next_url=safe_next)

    logger.info(
        "Successful login from %s",
        request.client.host if request.client else "unknown",
    )

    # Determine whether the connection is HTTPS so we set the Secure flag
    # correctly.  Behind nginx the scheme comes through X-Forwarded-Proto.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    is_https = proto == "https"

    response = RedirectResponse(url=safe_next, status_code=303)
    _set_session_cookie(response, secure=is_https)
    return response


@router.get("/logout", include_in_schema=False)
async def logout(request: Request) -> Response:
    """Clear the session cookie and redirect to /login."""
    response = RedirectResponse(url="/login", status_code=303)
    _clear_session_cookie(response)
    logger.info(
        "Logout from %s",
        request.client.host if request.client else "unknown",
    )
    return response


# ---------------------------------------------------------------------------
# Middleware helper
# ---------------------------------------------------------------------------


class SessionAuthMiddleware:
    """ASGI middleware that enforces session authentication.

    Must be added to the FastAPI app AFTER the router is registered so
    that /login and /logout are available for redirects.

    Usage in main.py::

        from lib.services.web.auth import SessionAuthMiddleware, is_auth_enabled

        if is_auth_enabled():
            app.add_middleware(SessionAuthMiddleware)

    The middleware is a no-op when WEB_PASSWORD_HASH is unset (dev mode).
    """

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Build a minimal Request to inspect path and cookies
        request = Request(scope, receive)
        path = request.url.path

        # Always allow public paths and static assets through
        if path in _PUBLIC_PATHS or path.startswith("/static/"):
            await self._app(scope, receive, send)
            return

        if is_authenticated(request):
            await self._app(scope, receive, send)
            return

        # For HTMX partial requests return a 401 with a redirect header so
        # the browser can navigate to the login page without a broken partial.
        is_htmx = request.headers.get("hx-request") == "true"
        if is_htmx:
            response = Response(
                content="Session expired. Please log in.",
                status_code=401,
                headers={"HX-Redirect": f"/login?next={path}"},
            )
        else:
            response = RedirectResponse(
                url=f"/login?next={path}",
                status_code=303,
            )

        await response(scope, receive, send)
