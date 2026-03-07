"""
Settings Page API Router
=========================
Serves the full Settings page at GET /settings and provides HTMX fragments
for engine status, live feed controls, and service configuration.

Endpoints:
    GET  /settings                  — Full HTML settings page
    GET  /settings/status/html      — Engine status fragment (HTMX)
    POST /settings/update           — Update engine settings (account, interval, period)
    POST /settings/refresh          — Force refresh + cache flush
    POST /settings/optimize         — Force optimization cycle
    POST /settings/live_feed/start  — Start live feed
    POST /settings/live_feed/stop   — Stop live feed
    POST /settings/live_feed/upgrade   — Upgrade to quotes
    POST /settings/live_feed/downgrade — Downgrade to bars only
"""

import logging

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.settings")

router = APIRouter(tags=["Settings"])

# ---------------------------------------------------------------------------
# Engine accessor (injected by main.py after lifespan starts)
# ---------------------------------------------------------------------------

_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def _get_engine():
    return _engine  # may be None — callers handle gracefully


# ---------------------------------------------------------------------------
# Settings page HTML
# ---------------------------------------------------------------------------

_SETTINGS_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>
<title>Settings — Futures Co-Pilot</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚙️</text></svg>"/>
<script>(function(){var t=localStorage.getItem('theme');if(t==='light')document.documentElement.classList.remove('dark');else document.documentElement.classList.add('dark');})();</script>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
/* ── Reset & theme ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f4f4f5;--bg-panel:rgba(255,255,255,0.85);--bg-inner:rgba(244,244,245,0.6);
  --bg-input:#e4e4e7;--border:#d4d4d8;--border-s:#e4e4e7;
  --text:#18181b;--text2:#3f3f46;--muted:#71717a;--faint:#a1a1aa;
}
.dark{
  --bg:#09090b;--bg-panel:rgba(24,24,27,0.7);--bg-inner:rgba(39,39,42,0.5);
  --bg-input:#27272a;--border:#3f3f46;--border-s:#27272a;
  --text:#f4f4f5;--text2:#d4d4d8;--muted:#71717a;--faint:#52525b;
}
body{font-family:ui-monospace,'Cascadia Code','Fira Code',monospace;background:var(--bg);color:var(--text);min-height:100vh;font-size:13px}

/* ── Nav bar ── */
.nav{display:flex;align-items:center;gap:0;padding:0 1rem;background:var(--bg-panel);
     border-bottom:1px solid var(--border);height:42px;position:sticky;top:0;z-index:100;backdrop-filter:blur(10px)}
.nav-brand{font-weight:700;font-size:0.9rem;color:var(--text);text-decoration:none;margin-right:1.25rem;letter-spacing:-0.02em}
.nav-tab{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:6px;
         text-decoration:none;color:var(--muted);font-size:0.78rem;font-weight:500;transition:all .12s;white-space:nowrap}
.nav-tab:hover{background:var(--bg-input);color:var(--text)}
.nav-tab.active{background:var(--bg-input);color:var(--text);font-weight:650}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.theme-btn{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 8px;
           color:var(--muted);cursor:pointer;font-size:0.75rem;transition:all .12s;font-family:inherit}
.theme-btn:hover{color:var(--text);border-color:var(--text)}

/* ── Layout ── */
.page{padding:1rem;max-width:1400px;margin:0 auto}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
@media(max-width:900px){.grid-2{grid-template-columns:1fr}}
@media(max-width:600px){.grid-3{grid-template-columns:1fr 1fr}}

/* ── Card ── */
.card{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.card-title{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin-bottom:12px}

/* ── Form controls ── */
label.lbl{display:block;font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
input[type=number],select{
  background:var(--bg-input);border:1px solid var(--border);border-radius:6px;
  color:var(--text);padding:6px 10px;width:100%;font-size:0.8rem;outline:none;font-family:inherit}
input:focus,select:focus{border-color:#3b82f6;box-shadow:0 0 0 2px #3b82f620}
.field{margin-bottom:10px}

/* ── Buttons ── */
.btn{border-radius:7px;padding:7px 16px;font-size:0.8rem;font-weight:600;cursor:pointer;
     border:none;transition:opacity .12s;font-family:inherit;display:inline-flex;align-items:center;gap:5px}
.btn:hover{opacity:.85}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-primary{background:#2563eb;color:#fff}
.btn-danger{background:#dc2626;color:#fff}
.btn-success{background:#16a34a;color:#fff}
.btn-warning{background:#d97706;color:#fff}
.btn-neutral{background:var(--bg-input);border:1px solid var(--border);color:var(--text)}
.btn-sm{padding:4px 11px;font-size:0.74rem}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}

/* ── Status badge ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:9999px;
       font-size:0.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.b-ok{background:#14532d22;color:#4ade80;border:1px solid #14532d}
.b-warn{background:#3b270022;color:#fb923c;border:1px solid #3b2700}
.b-err{background:#450a0a22;color:#f87171;border:1px solid #450a0a}
.b-info{background:#1e3a5f22;color:#60a5fa;border:1px solid #1e3a5f}
.b-gray{background:#27272a22;color:#a1a1aa;border:1px solid #3f3f46}

/* ── Status row ── */
.status-row{display:flex;align-items:center;justify-content:space-between;
            padding:7px 10px;border-radius:7px;background:var(--bg-inner);
            border:1px solid var(--border-s);margin-bottom:6px;font-size:0.78rem}
.status-key{color:var(--muted);font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em}
.status-val{font-weight:600;color:var(--text);font-size:0.78rem;text-align:right}

/* ── Dot indicators ── */
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-green{background:#4ade80}.dot-red{background:#f87171}
.dot-yellow{background:#fbbf24}.dot-gray{background:#6b7280}
.dot-pulse{animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Feed status block ── */
.feed-block{padding:10px 12px;border-radius:8px;background:var(--bg-inner);
            border:1px solid var(--border-s);margin-bottom:10px}
.feed-title{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-bottom:6px}

/* ── Link row ── */
.link-row{display:flex;align-items:center;gap:10px;padding:8px 10px;
          border-radius:7px;background:var(--bg-inner);border:1px solid var(--border-s);
          margin-bottom:6px;font-size:0.78rem;text-decoration:none;color:var(--text);
          transition:background .12s}
.link-row:hover{background:var(--bg-input)}
.link-icon{font-size:1.1rem;width:22px;text-align:center}
.link-label{font-weight:600}
.link-desc{font-size:0.68rem;color:var(--muted);margin-top:1px}

/* ── Toast ── */
#toast{position:fixed;bottom:24px;right:24px;padding:10px 18px;border-radius:8px;
       font-size:0.8rem;font-weight:600;z-index:9999;transition:opacity .3s;
       opacity:0;pointer-events:none;max-width:320px}
#toast.show{opacity:1}
#toast.ok{background:#15803d;color:#fff;border:1px solid #16a34a}
#toast.err{background:#b91c1c;color:#fff;border:1px solid #dc2626}

/* ── Section divider ── */
hr.sep{border:none;border-top:1px solid var(--border-s);margin:12px 0}

/* ── Spinner ── */
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--border);
      border-top-color:#3b82f6;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Account pill selector ── */
.acct-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.acct-pill{padding:6px 16px;border-radius:7px;border:1px solid var(--border);
           background:var(--bg-input);color:var(--muted);font-size:0.78rem;
           cursor:pointer;transition:all .12s;font-family:inherit;font-weight:500}
.acct-pill.selected{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;font-weight:700}
.acct-pill:hover:not(.selected){border-color:var(--text);color:var(--text)}
</style>
</head>
<body>

<!-- Nav bar -->
<nav class="nav">
  <a class="nav-brand" href="/">📈 Co-Pilot</a>
  <a class="nav-tab" href="/">📊 Dashboard</a>
  <a class="nav-tab" href="/trainer">🧠 Trainer</a>
  <a class="nav-tab active" href="/settings">⚙️ Settings</a>
  <div class="nav-right">
    <button class="theme-btn" onclick="toggleTheme()">☀/🌙</button>
  </div>
</nav>

<div class="page">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
    <div>
      <div style="font-size:1.2rem;font-weight:700">⚙️ Settings</div>
      <div style="font-size:0.72rem;color:var(--muted);margin-top:2px">Engine configuration · live feed · service links</div>
    </div>
    <div id="engine-badge" class="badge b-gray">checking…</div>
  </div>

  <div class="grid-2">

    <!-- LEFT COLUMN -->
    <div>

      <!-- Engine runtime settings -->
      <div class="card">
        <div class="card-title">Engine Settings</div>

        <!-- Account size pills -->
        <div class="field">
          <label class="lbl">Account Size</label>
          <div class="acct-pills">
            <button class="acct-pill" id="acct-50"  onclick="selectAcct(50000)">$50 K</button>
            <button class="acct-pill" id="acct-100" onclick="selectAcct(100000)">$100 K</button>
            <button class="acct-pill" id="acct-150" onclick="selectAcct(150000)">$150 K</button>
          </div>
        </div>

        <div class="grid-2" style="gap:8px">
          <div class="field">
            <label class="lbl">Primary Interval</label>
            <select id="s-interval">
              <option value="1m">1m</option>
              <option value="2m">2m</option>
              <option value="5m" selected>5m</option>
              <option value="15m">15m</option>
              <option value="30m">30m</option>
              <option value="60m">60m</option>
              <option value="1h">1h</option>
              <option value="1d">1d</option>
            </select>
          </div>
          <div class="field">
            <label class="lbl">Lookback Period</label>
            <select id="s-period">
              <option value="1d">1d</option>
              <option value="3d">3d</option>
              <option value="5d" selected>5d</option>
              <option value="7d">7d</option>
              <option value="10d">10d</option>
              <option value="14d">14d</option>
              <option value="30d">30d</option>
            </select>
          </div>
        </div>

        <div class="btn-row">
          <button class="btn btn-primary" onclick="applySettings()">💾 Apply Settings</button>
          <button class="btn btn-neutral" onclick="loadStatus()">↺ Reload</button>
        </div>
        <div id="settings-msg" style="font-size:0.72rem;color:var(--muted);margin-top:7px;min-height:16px"></div>
      </div>

      <!-- Engine actions -->
      <div class="card">
        <div class="card-title">Engine Actions</div>
        <div class="btn-row" style="margin-top:0">
          <button class="btn btn-warning btn-sm" onclick="doAction('refresh')">🔄 Force Refresh</button>
          <button class="btn btn-neutral btn-sm" onclick="doAction('optimize')">🔬 Optimize Now</button>
        </div>
        <div id="action-msg" style="font-size:0.72rem;color:var(--muted);margin-top:8px;min-height:16px"></div>
      </div>

      <!-- Live feed controls -->
      <div class="card">
        <div class="card-title">Massive Live Feed</div>
        <div class="feed-block">
          <div class="feed-title">Feed Status</div>
          <div id="feed-status-row" class="status-row" style="margin-bottom:0">
            <span class="status-key">Connection</span>
            <span id="feed-status-val" class="status-val" style="display:flex;align-items:center;gap:6px">
              <span class="dot dot-gray dot-pulse" id="feed-dot"></span>
              <span id="feed-status-text">checking…</span>
            </span>
          </div>
        </div>
        <div class="btn-row">
          <button class="btn btn-success btn-sm" onclick="feedAction('start')">▶ Start</button>
          <button class="btn btn-danger btn-sm"  onclick="feedAction('stop')">■ Stop</button>
        </div>
        <hr class="sep"/>
        <div style="font-size:0.68rem;color:var(--muted);margin-bottom:6px">Feed quality</div>
        <div class="btn-row" style="margin-top:0">
          <button class="btn btn-neutral btn-sm" onclick="feedAction('upgrade')">⬆ Upgrade (quotes)</button>
          <button class="btn btn-neutral btn-sm" onclick="feedAction('downgrade')">⬇ Downgrade (bars only)</button>
        </div>
        <div id="feed-msg" style="font-size:0.72rem;color:var(--muted);margin-top:8px;min-height:16px"></div>
      </div>

    </div>

    <!-- RIGHT COLUMN -->
    <div>

      <!-- Engine status panel -->
      <div class="card">
        <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
          <span>Engine Status</span>
          <span class="spin" id="status-spin" style="display:none"></span>
        </div>
        <div id="engine-status-panel">
          <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:20px 0">Loading…</div>
        </div>
      </div>

      <!-- Service links -->
      <div class="card">
        <div class="card-title">Service Links</div>

        <a class="link-row" href="/" target="_self">
          <span class="link-icon">📊</span>
          <div>
            <div class="link-label">Main Dashboard</div>
            <div class="link-desc">ORB detection, asset focus cards, positions & P&L</div>
          </div>
        </a>

        <a class="link-row" href="/trainer" target="_self">
          <span class="link-icon">🧠</span>
          <div>
            <div class="link-label">CNN Trainer</div>
            <div class="link-desc">Train, validate, and export the CNN breakout model</div>
          </div>
        </a>

        <a class="link-row" href="/docs" target="_blank" rel="noopener">
          <span class="link-icon">📖</span>
          <div>
            <div class="link-label">API Docs (Swagger)</div>
            <div class="link-desc">Interactive REST API documentation</div>
          </div>
        </a>

        <a class="link-row" href="/redoc" target="_blank" rel="noopener">
          <span class="link-icon">📄</span>
          <div>
            <div class="link-label">ReDoc</div>
            <div class="link-desc">Alternative API documentation viewer</div>
          </div>
        </a>

        <a class="link-row" href="/metrics/prometheus" target="_blank" rel="noopener">
          <span class="link-icon">📈</span>
          <div>
            <div class="link-label">Prometheus Metrics</div>
            <div class="link-desc">Raw metrics endpoint for Prometheus scraping</div>
          </div>
        </a>

        <a class="link-row" href="/health" target="_blank" rel="noopener">
          <span class="link-icon">❤️</span>
          <div>
            <div class="link-label">Health Check</div>
            <div class="link-desc">JSON health status of all services</div>
          </div>
        </a>

        <a class="link-row" href="/api/info" target="_blank" rel="noopener">
          <span class="link-icon">ℹ️</span>
          <div>
            <div class="link-label">API Info</div>
            <div class="link-desc">Service version, status, and endpoint index</div>
          </div>
        </a>
      </div>

      <!-- About -->
      <div class="card">
        <div class="card-title">About</div>
        <div class="status-row">
          <span class="status-key">Service</span>
          <span class="status-val">Futures Trading Co-Pilot</span>
        </div>
        <div class="status-row">
          <span class="status-key">Data Service</span>
          <span class="status-val" id="about-data">port 8000</span>
        </div>
        <div class="status-row">
          <span class="status-key">Web Service</span>
          <span class="status-val">port 8080</span>
        </div>
        <div class="status-row">
          <span class="status-key">Trainer Service</span>
          <span class="status-val">port 8200</span>
        </div>
        <div class="status-row" style="margin-bottom:0">
          <span class="status-key">Theme</span>
          <span class="status-val" id="about-theme">dark</span>
        </div>
      </div>

    </div><!-- /right col -->
  </div><!-- /grid-2 -->
</div><!-- /page -->

<!-- Toast -->
<div id="toast"></div>

<script>
'use strict';

// ═══════════════════════════════════════════════════════
// Theme
// ═══════════════════════════════════════════════════════
function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  document.getElementById('about-theme').textContent = isDark ? 'dark' : 'light';
}

// ═══════════════════════════════════════════════════════
// Toast helper
// ═══════════════════════════════════════════════════════
let _toastTimer = null;
function toast(msg, isErr) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + (isErr ? 'err' : 'ok');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = ''; }, 3500);
}

// ═══════════════════════════════════════════════════════
// Load engine status
// ═══════════════════════════════════════════════════════
let _selectedAcct = null;

async function loadStatus() {
  const spin = document.getElementById('status-spin');
  spin.style.display = 'inline-block';
  try {
    const r = await fetch('/analysis/status', { signal: AbortSignal.timeout(6000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    renderStatus(d);
  } catch (e) {
    document.getElementById('engine-status-panel').innerHTML =
      '<div style="color:#f87171;font-size:0.78rem;padding:12px 0">Failed to load engine status: ' + e.message + '</div>';
    document.getElementById('engine-badge').className = 'badge b-err';
    document.getElementById('engine-badge').textContent = 'error';
  } finally {
    spin.style.display = 'none';
  }
}

function _statusRow(key, val) {
  return '<div class="status-row"><span class="status-key">' + key +
    '</span><span class="status-val">' + val + '</span></div>';
}

function renderStatus(d) {
  const badge = document.getElementById('engine-badge');
  if (d.running) {
    badge.className = 'badge b-ok'; badge.textContent = 'running';
  } else {
    badge.className = 'badge b-warn'; badge.textContent = 'idle';
  }

  // Sync form controls from current engine state
  const acct = d.account_size || d.settings?.account_size;
  if (acct && !_selectedAcct) selectAcct(acct, true);

  const intv = d.interval || d.settings?.interval;
  if (intv) {
    const sel = document.getElementById('s-interval');
    if (sel) { for (let o of sel.options) { if (o.value === intv) o.selected = true; } }
  }

  const per = d.period || d.settings?.period;
  if (per) {
    const sel = document.getElementById('s-period');
    if (sel) { for (let o of sel.options) { if (o.value === per) o.selected = true; } }
  }

  // Feed status
  const feed = d.live_feed || {};
  const feedRunning = feed.running || feed.connected || false;
  const feedDot = document.getElementById('feed-dot');
  const feedTxt = document.getElementById('feed-status-text');
  if (feedRunning) {
    feedDot.className = 'dot dot-green';
    feedTxt.textContent = feed.mode ? 'running (' + feed.mode + ')' : 'running';
  } else {
    feedDot.className = 'dot dot-gray';
    feedTxt.textContent = 'stopped';
  }

  // Build status rows
  let rows = '';
  if (d.last_refresh) rows += _statusRow('Last Refresh', _fmtTime(d.last_refresh));
  if (d.next_refresh) rows += _statusRow('Next Refresh', _fmtTime(d.next_refresh));
  if (d.refresh_interval_minutes) rows += _statusRow('Interval', d.refresh_interval_minutes + ' min');
  if (intv)  rows += _statusRow('Bar Interval', intv);
  if (per)   rows += _statusRow('Lookback', per);
  if (acct)  rows += _statusRow('Account', '$' + Number(acct).toLocaleString());
  if (d.data_source) rows += _statusRow('Data Source', d.data_source);
  if (d.assets_loaded !== undefined) rows += _statusRow('Assets Loaded', d.assets_loaded);

  if (feed.running !== undefined) {
    rows += _statusRow('Live Feed', feedRunning
      ? '<span style="color:#4ade80">● running</span>'
      : '<span style="color:#6b7280">● stopped</span>');
  }

  if (!rows) rows = '<div style="color:var(--faint);font-size:0.78rem;padding:8px 0">No status available</div>';

  document.getElementById('engine-status-panel').innerHTML = rows;
}

function _fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return iso; }
}

// ═══════════════════════════════════════════════════════
// Account size selector
// ═══════════════════════════════════════════════════════
function selectAcct(size, silent) {
  _selectedAcct = size;
  document.querySelectorAll('.acct-pill').forEach(p => p.classList.remove('selected'));
  const el = document.getElementById('acct-' + (size / 1000));
  if (el) el.classList.add('selected');
  if (!silent) document.getElementById('settings-msg').textContent = '';
}

// ═══════════════════════════════════════════════════════
// Apply engine settings
// ═══════════════════════════════════════════════════════
async function applySettings() {
  const body = {};
  if (_selectedAcct) body.account_size = _selectedAcct;
  const intv = document.getElementById('s-interval').value;
  const per  = document.getElementById('s-period').value;
  if (intv) body.interval = intv;
  if (per)  body.period   = per;

  if (!Object.keys(body).length) {
    document.getElementById('settings-msg').textContent = 'Nothing to update.';
    return;
  }

  const msg = document.getElementById('settings-msg');
  msg.textContent = 'Applying…';
  try {
    const r = await fetch('/actions/update_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(8000),
    });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || 'Settings updated');
      toast('Settings applied', false);
      setTimeout(loadStatus, 600);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Engine actions (refresh / optimize)
// ═══════════════════════════════════════════════════════
async function doAction(type) {
  const urls = { refresh: '/actions/force_refresh', optimize: '/actions/optimize_now' };
  const url = urls[type];
  if (!url) return;

  const msg = document.getElementById('action-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = (type === 'refresh' ? 'Triggering refresh…' : 'Triggering optimization…');

  try {
    const r = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(10000) });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || d.status || 'Done');
      toast(d.message || 'Done', false);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Live feed controls
// ═══════════════════════════════════════════════════════
async function feedAction(action) {
  const urls = {
    start:     '/actions/live_feed/start',
    stop:      '/actions/live_feed/stop',
    upgrade:   '/actions/live_feed/upgrade',
    downgrade: '/actions/live_feed/downgrade',
  };
  const url = urls[action];
  if (!url) return;

  const msg = document.getElementById('feed-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = action.charAt(0).toUpperCase() + action.slice(1) + 'ing…';

  try {
    const r = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(10000) });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || d.status || 'Done');
      toast(d.message || 'Done', false);
      setTimeout(loadStatus, 800);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  setInterval(loadStatus, 15000);

  const isDark = document.documentElement.classList.contains('dark');
  document.getElementById('about-theme').textContent = isDark ? 'dark' : 'light';

  // Show port from URL
  const port = location.port || (location.protocol === 'https:' ? '443' : '80');
  document.getElementById('about-data').textContent = 'port ' + port;
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/settings", response_class=HTMLResponse)
async def settings_page() -> HTMLResponse:
    """Serve the full Settings HTML page."""
    return HTMLResponse(content=_SETTINGS_PAGE_HTML)
