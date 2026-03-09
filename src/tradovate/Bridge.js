// =============================================================================
// tradovate_bridge.js  —  Node.js
// =============================================================================
//
// Drop this file into:
//   scripts/tradovate_bridge.js
//
// PURPOSE
// -------
// JavaScript equivalent of Bridge.cs for Tradovate accounts.
// Connects to Tradovate's WebSocket API, subscribes to live position /
// order / fill events via user/syncrequest, and pushes position snapshots
// to the Python engine — same endpoint and payload format as Bridge.cs so
// the existing positions.py API route requires zero changes.
//
// Responsibilities:
//   1. Authenticate with Tradovate (REST token → WS authorize)
//   2. Subscribe to real-time user events (positions, orders, fills, cash)
//      via user/syncrequest — no polling needed
//   3. Push position snapshots to the Python dashboard on every fill and
//      on a 15-second heartbeat
//   4. Parse risk-gate feedback from the dashboard response (can_trade,
//      block_reason) and surface it in /health
//   5. Expose a tiny HTTP /health endpoint (port 5681) for liveness probes
//   6. Auto-refresh the Tradovate access token before it expires (1h limit)
//   7. Exponential-backoff reconnection (max 10 attempts)
//
// QUICK START
// -----------
//   npm install ws node-fetch dotenv        # one-time
//   node scripts/tradovate_bridge.js
//
// Add to .env:
//   TRADOVATE_NAME=your_login
//   TRADOVATE_PASSWORD=your_password
//   TRADOVATE_APP_ID=YourApp
//   TRADOVATE_APP_VERSION=1.0
//   TRADOVATE_CID=your_client_id            # numeric, from Tradovate API settings
//   TRADOVATE_SEC=your_secret               # from Tradovate API settings
//   TRADOVATE_DEMO=false                    # true = demo/sim account
//   TRADOVATE_ACCOUNT_NAME=                 # leave blank to use first account
//   DASHBOARD_BASE_URL=http://localhost:8100 # matches ENGINE_PORT in your docker-compose
//   BRIDGE_PORT=5681                        # HTTP health port
//   HEARTBEAT_SEC=15                        # position push interval
//
// Run as a Docker service by adding to docker-compose.yml:
//   tradovate_bridge:
//     image: node:20-alpine
//     working_dir: /app
//     volumes: [".:/app"]
//     command: node scripts/tradovate_bridge.js
//     env_file: [.env]
//     restart: unless-stopped
//     network_mode: host
//
// =============================================================================

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Deps  (npm install ws node-fetch dotenv)
// ─────────────────────────────────────────────────────────────────────────────
require("dotenv").config();
const WebSocket = require("ws");
const http = require("http");
// node-fetch v2 is CommonJS compatible; v3+ is ESM-only.
// Use: npm install node-fetch@2
const fetch = require("node-fetch");

// ─────────────────────────────────────────────────────────────────────────────
// Configuration  (mirrors the #region Configuration block in Bridge.cs)
// ─────────────────────────────────────────────────────────────────────────────
const CFG = {
    // Tradovate credentials (all from .env)
    name: process.env.TRADOVATE_NAME || "",
    password: process.env.TRADOVATE_PASSWORD || "",
    appId: process.env.TRADOVATE_APP_ID || "FuturesCoPilot",
    appVersion: process.env.TRADOVATE_APP_VERSION || "1.0",
    cid: parseInt(process.env.TRADOVATE_CID || "0", 10),
    sec: process.env.TRADOVATE_SEC || "",

    // Demo vs live
    demo: process.env.TRADOVATE_DEMO === "true",

    // Account to watch (blank = first account in list)
    accountName: process.env.TRADOVATE_ACCOUNT_NAME || "",

    // Python engine push destination — same URL as Bridge.cs
    dashboardBaseUrl: (
        process.env.DASHBOARD_BASE_URL || "http://localhost:8100"
    ).replace(/\/$/, ""),

    // Local health-check port (use 5681 to avoid colliding with Bridge.cs on 5680)
    bridgePort: parseInt(process.env.BRIDGE_PORT || "5681", 10),

    // Position push interval in ms
    heartbeatMs: parseInt(process.env.HEARTBEAT_SEC || "15", 10) * 1000,

    // Bridge version (shown in /health)
    version: "1.0",
};

// Tradovate endpoints
const ENDPOINTS = {
    auth: CFG.demo
        ? "https://demo.tradovateapi.com/v1/auth/accesstokenrequest"
        : "https://live.tradovateapi.com/v1/auth/accesstokenrequest",
    ws: CFG.demo
        ? "wss://demo.tradovateapi.com/v1/websocket"
        : "wss://live.tradovateapi.com/v1/websocket",
};

// Token refresh: 85 minutes (tokens expire at 90m; refresh 5m early)
const TOKEN_REFRESH_MS = 85 * 60 * 1000;

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let ws = null;
let isAuthenticated = false;
let authSent = false;
let syncCompleted = false;
let requestIdCounter = 2; // 0=auth, 1=syncrequest, 2+ = all others

let tokenInfo = null; // { accessToken, expirationTime, userId, name }
let tokenRefreshTimer = null;

let accountId = null; // resolved after syncrequest
let accountName = "";
let positions = {}; // symbol → { direction, quantity, entryPrice, unrealizedPnl }
let cashBalance = 0;
let realizedPnl = 0;
let pendingOrderCount = 0;

let isRiskBlocked = false;
let riskBlockReason = "";

let heartbeatTimer = null; // 2.5s WS keepalive
let pushTimer = null; // 15s position push to Python engine
let lastPushSuccess = true;
let lastServerMsg = Date.now();

let reconnectAttempts = 0;
const MAX_RECONNECTS = 10;
let shouldReconnect = true;
let isReconnecting = false;

// Metrics (Prometheus-style counters)
const metrics = {
    fillsReceived: 0,
    orderEvents: 0,
    positionPushes: 0,
    pushErrors: 0,
    heartbeatsSent: 0,
    reconnects: 0,
    startTime: Date.now(),
};

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────
function log(msg) {
    console.log(`[TradovateBridge] ${msg}`);
}
function warn(msg) {
    console.warn(`[TradovateBridge] ⚠ ${msg}`);
}
function error(msg) {
    console.error(`[TradovateBridge] ✗ ${msg}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth — REST token fetch + auto-refresh
// ─────────────────────────────────────────────────────────────────────────────
async function fetchToken() {
    const body = {
        name: CFG.name,
        password: CFG.password,
        appId: CFG.appId,
        appVersion: CFG.appVersion,
        cid: CFG.cid,
        sec: CFG.sec,
    };

    const resp = await fetch(ENDPOINTS.auth, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        timeout: 10000,
    });

    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Auth failed ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    if (!data.accessToken)
        throw new Error(`No accessToken in response: ${JSON.stringify(data)}`);

    tokenInfo = data;
    log(`Token acquired — user: ${data.name}, expires: ${data.expirationTime}`);

    // Schedule refresh
    if (tokenRefreshTimer) clearTimeout(tokenRefreshTimer);
    tokenRefreshTimer = setTimeout(async () => {
        log("Refreshing access token…");
        try {
            await fetchToken();
        } catch (e) {
            error("Token refresh failed: " + e.message);
        }
    }, TOKEN_REFRESH_MS);
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket connection + Tradovate protocol
// ─────────────────────────────────────────────────────────────────────────────
async function connect() {
    if (!tokenInfo) await fetchToken();

    log(`Connecting to ${ENDPOINTS.ws} (${CFG.demo ? "DEMO" : "LIVE"})…`);
    ws = new WebSocket(ENDPOINTS.ws);

    ws.on("open", () => {
        log("WebSocket open");
        lastServerMsg = Date.now();
    });

    ws.on("message", onMessage);

    ws.on("close", (code, reason) => {
        log(`WebSocket closed. code=${code} reason=${reason}`);
        resetConnectionState();
        if (shouldReconnect && !isReconnecting) scheduleReconnect();
    });

    ws.on("error", (err) => {
        error("WebSocket error: " + err.message);
    });
}

// Tradovate frame protocol:
//   Server → client: "o" (open), "h" (heartbeat), "a[{...}]" (messages), "c" (close)
//   Client → server: "endpoint\nreqId\n\nbodyJson"  OR  "[]" (heartbeat)
function onMessage(raw) {
    lastServerMsg = Date.now();
    const msg = raw.toString();

    // Single-char frames
    if (msg === "o") {
        log("Server open frame — authenticating");
        sendAuth();
        return;
    }
    if (msg === "h") {
        return;
    } // server heartbeat — just reset timestamp (already done above)
    if (msg === "c") {
        log("Server close frame");
        return;
    }

    // Array frames: a[{...},{...}]
    if (msg.startsWith("a[")) {
        let messages;
        try {
            messages = JSON.parse(msg.slice(1));
        } catch (e) {
            warn("Failed to parse frame: " + msg.slice(0, 80));
            return;
        }
        for (const m of messages) handleSocketMessage(m);
        return;
    }
}

function sendAuth() {
    if (authSent) return;
    authSent = true;
    const authMsg = `authorize\n0\n\n${tokenInfo.accessToken}`;
    ws.send(authMsg);
    log("Auth message sent");
}

function sendWsRequest(endpoint, body, reqId) {
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("WS not open");
    if (!isAuthenticated) throw new Error("WS not authenticated");
    const id = reqId !== undefined ? reqId : requestIdCounter++;
    const msg = `${endpoint}\n${id}\n\n${body !== undefined ? JSON.stringify(body) : ""}`;
    ws.send(msg);
    return id;
}

function handleSocketMessage(m) {
    // Auth response (reqId=0)
    if (m.i === 0) {
        if (m.s === 200) {
            isAuthenticated = true;
            log("Authenticated ✓");
            startHeartbeat();
            subscribeUserSync();
        } else {
            error("Authentication rejected: " + JSON.stringify(m));
            shouldReconnect = false;
        }
        return;
    }

    // user/syncrequest response (reqId=1)
    if (m.i === 1 && !syncCompleted) {
        syncCompleted = true;
        log("user/syncrequest complete");
        if (m.d) processFullSync(m.d);
        // Start position push loop now that we have baseline data
        startPushTimer();
        return;
    }

    // Real-time event push (event name in m.e)
    if (m.e === "props") {
        if (m.d) processPropUpdates(m.d);
        return;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// user/syncrequest — subscribe to all user data events
// Tradovate: this is the correct way to get real-time fills, positions, orders
// ─────────────────────────────────────────────────────────────────────────────
function subscribeUserSync() {
    log("Sending user/syncrequest…");
    sendWsRequest("user/syncrequest", { users: [tokenInfo.userId] }, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sync data processors
// ─────────────────────────────────────────────────────────────────────────────

// Full baseline sync from syncrequest response
function processFullSync(data) {
    // Resolve account
    const accounts = data.accounts || [];
    let account = accounts[0];
    if (CFG.accountName) {
        account = accounts.find((a) => a.name === CFG.accountName) || account;
    }
    if (account) {
        accountId = account.id;
        accountName = account.name;
        log(`Account: ${accountName} (id=${accountId})`);
    }

    // Baseline cash
    const margins = data.marginSnapshots || [];
    const myMargin = margins.find((m) => m.accountId === accountId);
    if (myMargin) {
        cashBalance =
            myMargin.totalUsedMargin !== undefined
                ? myMargin.initialMargin || 0 // best available field
                : 0;
    }

    // Baseline positions
    positions = {};
    for (const pos of data.positions || []) {
        if (pos.accountId !== accountId) continue;
        updatePositionFromProp(pos);
    }

    // Baseline pending orders
    const openOrders = (data.orders || []).filter(
        (o) =>
            o.accountId === accountId &&
            (o.ordStatus === "Working" || o.ordStatus === "Accepted"),
    );
    pendingOrderCount = openOrders.length;

    log(
        `Baseline — positions: ${Object.keys(positions).length}, pending orders: ${pendingOrderCount}`,
    );
}

// Real-time prop update (fills, position changes, order state changes, P&L)
function processPropUpdates(data) {
    let positionChanged = false;
    let fillOccurred = false;

    // ── positions ──────────────────────────────────────────────────────────
    for (const pos of data.positions || []) {
        if (pos.accountId !== accountId) continue;
        updatePositionFromProp(pos);
        positionChanged = true;
    }

    // ── fills (executions) ─────────────────────────────────────────────────
    for (const fill of data.executionReports || data.fills || []) {
        if (fill.accountId !== accountId) continue;
        metrics.fillsReceived++;
        fillOccurred = true;
        log(
            `Fill: ${fill.side} ${fill.qty} ${fill.contractName || ""} @ ${fill.price}`,
        );
        // Update realized P&L if provided
        if (fill.totalFees !== undefined || fill.realizedPnl !== undefined) {
            realizedPnl += fill.realizedPnl || 0;
        }
    }

    // ── orders ─────────────────────────────────────────────────────────────
    const orderUpdates = data.orders || [];
    if (orderUpdates.length > 0) {
        metrics.orderEvents += orderUpdates.length;
        // Recount working orders for this account
        const workingStates = ["Working", "Accepted", "PendingNew"];
        // We don't have full order list in prop update — approximate with increments
        const newWorking = orderUpdates.filter(
            (o) =>
                o.accountId === accountId &&
                workingStates.includes(o.ordStatus),
        ).length;
        const terminated = orderUpdates.filter(
            (o) =>
                o.accountId === accountId &&
                !workingStates.includes(o.ordStatus),
        ).length;
        pendingOrderCount = Math.max(
            0,
            pendingOrderCount + newWorking - terminated,
        );
    }

    // ── account / cash ──────────────────────────────────────────────────────
    for (const acct of data.accounts || []) {
        if (acct.id !== accountId) continue;
        if (acct.balance !== undefined) cashBalance = acct.balance;
        if (acct.realizedPnl !== undefined) realizedPnl = acct.realizedPnl;
    }

    // ── push immediately on fills; debounce on position/order changes ──────
    if (fillOccurred || positionChanged) {
        pushPositionUpdate();
    }
}

function updatePositionFromProp(pos) {
    const symbol =
        pos.contractName || pos.symbol || String(pos.contractId || "");
    const qty = pos.netPos || pos.quantity || 0;

    if (qty === 0) {
        delete positions[symbol];
        return;
    }

    const direction = qty > 0 ? "long" : "short";
    const absQty = Math.abs(qty);
    const entry = pos.openPriceAvg || pos.averagePrice || pos.entryPrice || 0;
    const upnl = pos.unrealizedPnl || pos.openPnl || 0;

    positions[symbol] = {
        direction,
        quantity: absQty,
        entryPrice: entry,
        unrealizedPnl: upnl,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Position push  — same payload as Bridge.cs → /api/positions/update
// Python engine receives this at positions.py and does not need to change
// ─────────────────────────────────────────────────────────────────────────────
async function pushPositionUpdate() {
    if (!CFG.dashboardBaseUrl) return;

    metrics.positionPushes++;

    const totalUnrealized = Object.values(positions).reduce(
        (s, p) => s + p.unrealizedPnl,
        0,
    );

    // Mirror the exact JSON shape that Bridge.cs emits
    const payload = {
        account: accountName || "tradovate",
        positions: Object.entries(positions).map(([symbol, p]) => ({
            symbol,
            direction: p.direction,
            quantity: p.quantity,
            entry_price: p.entryPrice,
            unrealized_pnl: Math.round(p.unrealizedPnl * 100) / 100,
        })),
        cash_balance: Math.round(cashBalance * 100) / 100,
        realized_pnl: Math.round(realizedPnl * 100) / 100,
        unrealized_pnl: Math.round(totalUnrealized * 100) / 100,
        pending_orders: pendingOrderCount,
        risk_blocked: isRiskBlocked,
        // Tradovate-specific extras (Python engine can ignore unknown fields)
        source: "tradovate_bridge",
        bridge_version: CFG.version,
        demo: CFG.demo,
    };

    const url = `${CFG.dashboardBaseUrl}/api/positions/update`;

    try {
        const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            timeout: 5000,
        });

        if (!lastPushSuccess && resp.ok) {
            log("Position push connection restored ✓");
            lastPushSuccess = true;
        }

        // Parse risk-gate feedback (mirrors ParseRiskFeedback in Bridge.cs)
        if (resp.ok) {
            try {
                const feedback = await resp.json();
                parseRiskFeedback(feedback);
            } catch (_) {}
        }
    } catch (err) {
        metrics.pushErrors++;
        if (lastPushSuccess) {
            warn(`Position push failed: ${err.message}`);
            lastPushSuccess = false;
        }
    }
}

function parseRiskFeedback(data) {
    let canTrade = true;
    let blockReason = "";

    // Nested: { risk: { can_trade, block_reason } }
    const risk = data.risk || data;
    if (risk.can_trade !== undefined) canTrade = Boolean(risk.can_trade);
    if (risk.block_reason !== undefined)
        blockReason = String(risk.block_reason || "");

    const wasBlocked = isRiskBlocked;
    isRiskBlocked = !canTrade;
    riskBlockReason = blockReason;

    if (!canTrade && !wasBlocked) warn(`Risk BLOCKED: ${blockReason}`);
    if (canTrade && wasBlocked) log("Risk block cleared ✓");
}

// ─────────────────────────────────────────────────────────────────────────────
// Timers
// ─────────────────────────────────────────────────────────────────────────────

// WS keepalive (2.5s empty frame — Tradovate requirement)
function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send("[]");
            metrics.heartbeatsSent++;
        }
        // Dead connection detection
        if (Date.now() - lastServerMsg > 10000) {
            warn("No server message for 10s — forcing reconnect");
            if (ws) ws.close(4000, "server timeout");
        }
    }, 2500);
}

function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

// 15s position push loop
function startPushTimer() {
    if (pushTimer) clearInterval(pushTimer);
    pushPositionUpdate(); // immediate first push
    pushTimer = setInterval(pushPositionUpdate, CFG.heartbeatMs);
    log(`Position push timer started (${CFG.heartbeatMs / 1000}s interval)`);
}

function stopPushTimer() {
    if (pushTimer) {
        clearInterval(pushTimer);
        pushTimer = null;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconnection — exponential backoff (mirrors ReconnectionManager.ts)
// ─────────────────────────────────────────────────────────────────────────────
function resetConnectionState() {
    isAuthenticated = false;
    authSent = false;
    syncCompleted = false;
    stopHeartbeat();
    stopPushTimer();
}

function scheduleReconnect() {
    if (reconnectAttempts >= MAX_RECONNECTS) {
        error(`Max reconnects (${MAX_RECONNECTS}) reached — giving up`);
        return;
    }
    // min(initial * 2^attempt, 60s) + 10% jitter
    const base = 1000 * Math.pow(2, reconnectAttempts);
    const delay = Math.min(base, 60000) * (1 + Math.random() * 0.1);
    reconnectAttempts++;
    log(
        `Reconnecting in ${Math.round(delay / 1000)}s (attempt ${reconnectAttempts}/${MAX_RECONNECTS})…`,
    );
    setTimeout(doReconnect, delay);
}

async function doReconnect() {
    if (!shouldReconnect || isReconnecting) return;
    isReconnecting = true;
    metrics.reconnects++;
    try {
        if (ws) {
            ws.removeAllListeners();
            ws = null;
        }
        await fetchToken();
        await connect();
        reconnectAttempts = 0;
        isReconnecting = false;
        log("Reconnected ✓");
    } catch (err) {
        isReconnecting = false;
        error("Reconnect failed: " + err.message);
        scheduleReconnect();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Health HTTP server  (mirrors Bridge.cs /health + /metrics endpoints)
// ─────────────────────────────────────────────────────────────────────────────
const healthServer = http.createServer((req, res) => {
    const path = req.url.split("?")[0];
    res.setHeader("Content-Type", "application/json");

    if (path === "/health") {
        res.writeHead(200);
        res.end(
            JSON.stringify({
                status: "ok",
                bridge_version: CFG.version,
                source: "tradovate_bridge",
                demo: CFG.demo,
                authenticated: isAuthenticated,
                sync_complete: syncCompleted,
                account: accountName || null,
                risk_blocked: isRiskBlocked,
                risk_block_reason: riskBlockReason,
                reconnect_attempts: reconnectAttempts,
            }),
        );
        return;
    }

    if (path === "/status") {
        const totalUnrealized = Object.values(positions).reduce(
            (s, p) => s + p.unrealizedPnl,
            0,
        );
        res.writeHead(200);
        res.end(
            JSON.stringify({
                account: accountName,
                connected: isAuthenticated && syncCompleted,
                positions: Object.keys(positions).length,
                pending_orders: pendingOrderCount,
                cash_balance: Math.round(cashBalance * 100) / 100,
                realized_pnl: Math.round(realizedPnl * 100) / 100,
                unrealized_pnl: Math.round(totalUnrealized * 100) / 100,
                risk_blocked: isRiskBlocked,
                risk_block_reason: riskBlockReason,
                demo: CFG.demo,
                bridge_version: CFG.version,
                timestamp: new Date().toISOString(),
            }),
        );
        return;
    }

    if (path === "/metrics") {
        res.setHeader(
            "Content-Type",
            "text/plain; version=0.0.4; charset=utf-8",
        );
        const uptime = (Date.now() - metrics.startTime) / 1000;
        const totalUnrealized = Object.values(positions).reduce(
            (s, p) => s + p.unrealizedPnl,
            0,
        );
        const lines =
            [
                `# HELP tradovate_bridge_up Bridge connected and authenticated.`,
                `# TYPE tradovate_bridge_up gauge`,
                `tradovate_bridge_up ${isAuthenticated && syncCompleted ? 1 : 0}`,
                `# HELP tradovate_bridge_positions_count Open positions.`,
                `# TYPE tradovate_bridge_positions_count gauge`,
                `tradovate_bridge_positions_count ${Object.keys(positions).length}`,
                `# HELP tradovate_bridge_cash_balance Cash balance USD.`,
                `# TYPE tradovate_bridge_cash_balance gauge`,
                `tradovate_bridge_cash_balance ${Math.round(cashBalance * 100) / 100}`,
                `# HELP tradovate_bridge_unrealized_pnl Unrealized P&L USD.`,
                `# TYPE tradovate_bridge_unrealized_pnl gauge`,
                `tradovate_bridge_unrealized_pnl ${Math.round(totalUnrealized * 100) / 100}`,
                `# HELP tradovate_bridge_realized_pnl Realized P&L USD.`,
                `# TYPE tradovate_bridge_realized_pnl gauge`,
                `tradovate_bridge_realized_pnl ${Math.round(realizedPnl * 100) / 100}`,
                `# HELP tradovate_bridge_uptime_seconds Uptime.`,
                `# TYPE tradovate_bridge_uptime_seconds gauge`,
                `tradovate_bridge_uptime_seconds ${uptime.toFixed(0)}`,
                `# HELP tradovate_bridge_fills_total Fill events received.`,
                `# TYPE tradovate_bridge_fills_total counter`,
                `tradovate_bridge_fills_total ${metrics.fillsReceived}`,
                `# HELP tradovate_bridge_position_pushes_total Pushes sent to engine.`,
                `# TYPE tradovate_bridge_position_pushes_total counter`,
                `tradovate_bridge_position_pushes_total ${metrics.positionPushes}`,
                `# HELP tradovate_bridge_push_errors_total Push errors.`,
                `# TYPE tradovate_bridge_push_errors_total counter`,
                `tradovate_bridge_push_errors_total ${metrics.pushErrors}`,
                `# HELP tradovate_bridge_reconnects_total Reconnection attempts.`,
                `# TYPE tradovate_bridge_reconnects_total counter`,
                `tradovate_bridge_reconnects_total ${metrics.reconnects}`,
                `# HELP tradovate_bridge_info Bridge metadata.`,
                `# TYPE tradovate_bridge_info gauge`,
                `tradovate_bridge_info{version="${CFG.version}",account="${accountName}",demo="${CFG.demo}"} 1`,
            ].join("\n") + "\n";
        res.writeHead(200);
        res.end(lines);
        return;
    }

    res.writeHead(404);
    res.end(
        JSON.stringify({
            error: "not found",
            endpoints: ["/health", "/status", "/metrics"],
        }),
    );
});

// ─────────────────────────────────────────────────────────────────────────────
// Startup + graceful shutdown
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
    log(
        `Tradovate Bridge v${CFG.version} starting (${CFG.demo ? "DEMO" : "LIVE"})`,
    );

    // Validate config
    if (!CFG.name || !CFG.password || !CFG.cid || !CFG.sec) {
        error(
            "Missing credentials. Set TRADOVATE_NAME, TRADOVATE_PASSWORD, " +
                "TRADOVATE_CID, TRADOVATE_SEC in .env",
        );
        process.exit(1);
    }

    // Start health server
    healthServer.listen(CFG.bridgePort, () => {
        log(`Health endpoint: http://localhost:${CFG.bridgePort}/health`);
    });

    // Initial connection
    try {
        await fetchToken();
        await connect();
    } catch (err) {
        error("Initial connection failed: " + err.message);
        scheduleReconnect();
    }
}

function shutdown(sig) {
    log(`${sig} received — shutting down`);
    shouldReconnect = false;
    stopHeartbeat();
    stopPushTimer();
    if (tokenRefreshTimer) clearTimeout(tokenRefreshTimer);
    if (ws) ws.close(1000, "shutdown");
    healthServer.close();
    setTimeout(() => process.exit(0), 1000);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("unhandledRejection", (reason) =>
    warn("Unhandled rejection: " + reason),
);
process.on("uncaughtException", (err) =>
    error("Uncaught exception: " + err.message),
);

main();
