// =============================================================================
// tradovate_copier.js
// =============================================================================
//
// Trade copier for Tradovate accounts.
//
// ARCHITECTURE
// ────────────
//
//   ┌─────────────────────────────────────────────────────────────────┐
//   │  MASTER ACCOUNT                                                  │
//   │  (your main Tradovate / TradingView-connected account)           │
//   │                                                                  │
//   │  tradovate_bridge.js  →  Python engine (positions, risk)         │
//   │  tradovate_copier.js  →  watches fills via WS syncrequest        │
//   └───────────────────┬─────────────────────────────────────────────┘
//                       │ fill event
//              ┌────────┴────────┐
//              ▼                 ▼
//   ┌─────────────────┐  ┌─────────────────┐
//   │  FOLLOWER 1     │  │  FOLLOWER 2     │  ... (add more in .env)
//   │  Apex acct A    │  │  Apex acct B    │
//   │  REST placeorder│  │  REST placeorder│
//   └─────────────────┘  └─────────────────┘
//
// WHAT IT COPIES
// ──────────────
//  • Market entries  (Buy/Sell market)
//  • Limit entries
//  • Stop-market entries
//  • Exits / reversals (detected by position flip)
//  • Flatten (full position close propagated to all followers)
//
// WHAT IT DOES NOT COPY
// ─────────────────────
//  • Stop-loss / take-profit bracket orders (placed separately by follower
//    risk rules — too risky to blindly mirror across funded accounts)
//  • Order modifications or cancellations
//
// CONTRACT SCALING
// ────────────────
//  Each follower can have its own scale ratio and contract map.
//  Example: master trades ES → follower trades MES at 10× ratio (same $ risk).
//  Set per-follower in FOLLOWERS array below.
//
// TRADINGVIEW WEBHOOK
// ───────────────────
//  The copier also listens for TradingView JSON alerts on port 5682.
//  These go directly to the master account AND trigger a copy to all followers.
//  Alert JSON format:
//    { "action": "buy"|"sell"|"close", "symbol": "ESM5", "qty": 1,
//      "orderType": "Market"|"Limit"|"Stop", "price": 0, "stopPrice": 0 }
//
// QUICK START
// ───────────
//  1. npm install ws node-fetch@2 dotenv
//  2. Configure .env (see bottom of this file)
//  3. node scripts/tradovate_copier.js
//
// =============================================================================

"use strict";

require("dotenv").config();
const WebSocket = require("ws");
const http = require("http");
const fetch = require("node-fetch"); // npm install node-fetch@2

// ─────────────────────────────────────────────────────────────────────────────
// Config helpers
// ─────────────────────────────────────────────────────────────────────────────
function env(key, def = "") {
    return process.env[key] || def;
}
function envInt(key, def) {
    return parseInt(env(key, String(def)), 10);
}
function envBool(key) {
    return env(key, "false").toLowerCase() === "true";
}

// ─────────────────────────────────────────────────────────────────────────────
// Master account credentials
// ─────────────────────────────────────────────────────────────────────────────
const MASTER = {
    name: env("TRADOVATE_NAME"),
    password: env("TRADOVATE_PASSWORD"),
    appId: env("TRADOVATE_APP_ID", "FuturesCoPilot"),
    appVersion: env("TRADOVATE_APP_VERSION", "1.0"),
    cid: envInt("TRADOVATE_CID", 0),
    sec: env("TRADOVATE_SEC"),
    demo: envBool("TRADOVATE_DEMO"),
    // Override to watch a specific account name (blank = first account)
    accountName: env("TRADOVATE_ACCOUNT_NAME"),
};

// ─────────────────────────────────────────────────────────────────────────────
// Follower accounts
// ─────────────────────────────────────────────────────────────────────────────
// Define followers inline here OR via JSON in .env FOLLOWERS_JSON.
//
// Each follower object:
//   name        — Tradovate login
//   password    — Tradovate password
//   cid         — numeric client ID
//   sec         — secret key
//   demo        — boolean (can be different from master)
//   accountName — specific account name/spec (blank = first account)
//   scale       — multiplier applied to master qty  (1.0 = same size)
//   contractMap — optional symbol remapping, e.g. {"ES":"MES","NQ":"MNQ"}
//                 useful for trading micros on funded accounts
//   enabled     — set false to pause this follower without removing it
//
// Example FOLLOWERS_JSON in .env:
//   FOLLOWERS_JSON=[{"name":"apex1@email.com","password":"pw1","cid":123,"sec":"s1","demo":false,"accountName":"","scale":1,"contractMap":{"ES":"MES","NQ":"MNQ"},"enabled":true}]
//
const FOLLOWERS = JSON.parse(env("FOLLOWERS_JSON", "[]"));

// ─────────────────────────────────────────────────────────────────────────────
// Global settings
// ─────────────────────────────────────────────────────────────────────────────
const CFG = {
    demo: MASTER.demo,
    copierPort: envInt("COPIER_PORT", 5682), // HTTP / webhook port
    heartbeatMs: envInt("HEARTBEAT_SEC", 15) * 1000,
    copyFlattens: envBool("COPY_FLATTENS") !== false, // default true
    dashboardUrl: env("DASHBOARD_BASE_URL", "http://localhost:8100").replace(
        /\/$/,
        "",
    ),
    maxRetries: 3,
    retryDelayMs: 500,
};

const ENDPOINTS = {
    auth: MASTER.demo
        ? "https://demo.tradovateapi.com/v1/auth/accesstokenrequest"
        : "https://live.tradovateapi.com/v1/auth/accesstokenrequest",
    ws: MASTER.demo
        ? "wss://demo.tradovateapi.com/v1/websocket"
        : "wss://live.tradovateapi.com/v1/websocket",
    restBase: MASTER.demo
        ? "https://demo.tradovateapi.com/v1"
        : "https://live.tradovateapi.com/v1",
};

function followerRestBase(f) {
    return f.demo
        ? "https://demo.tradovateapi.com/v1"
        : "https://live.tradovateapi.com/v1";
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

// Master WS state
let ws = null;
let masterToken = null;
let masterAccountId = null;
let masterAccountSpec = "";
let isAuthenticated = false;
let authSent = false;
let syncCompleted = false;
let reqIdCounter = 2;
let lastServerMsg = Date.now();
let heartbeatTimer = null;

// Copy tracking — prevent duplicate copies on the same fill
const copiedFillIds = new Set(); // executionReport ids already processed
const masterPositions = {}; // symbol → { qty, avgPrice }  — tracks net position

// Follower sessions: each has token + accountId after init
// followerSessions[i] = { token, accountId, accountSpec, restBase }
const followerSessions = new Array(FOLLOWERS.length).fill(null);

// Metrics
const metrics = {
    fillsSeen: 0,
    copiesAttempted: 0,
    copiesOk: 0,
    copiesFailed: 0,
    tvWebhooks: 0,
    startTime: Date.now(),
};

// Reconnect
let reconnectAttempts = 0;
let shouldReconnect = true;
let isReconnecting = false;

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────
function log(m) {
    console.log(`[Copier] ${m}`);
}
function warn(m) {
    console.warn(`[Copier] ⚠ ${m}`);
}
function err(m) {
    console.error(`[Copier] ✗ ${m}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth helper (shared by master + all followers)
// ─────────────────────────────────────────────────────────────────────────────
async function authenticate(creds, restBase) {
    const resp = await fetch(`${restBase}/auth/accesstokenrequest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            name: creds.name,
            password: creds.password,
            appId: creds.appId || MASTER.appId,
            appVersion: creds.appVersion || MASTER.appVersion,
            cid: creds.cid,
            sec: creds.sec,
        }),
        timeout: 10000,
    });
    if (!resp.ok) throw new Error(`Auth ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();
    if (!data.accessToken)
        throw new Error("No accessToken: " + JSON.stringify(data));
    return data; // { accessToken, userId, name, expirationTime, … }
}

// ─────────────────────────────────────────────────────────────────────────────
// Follower initialisation — auth + resolve account
// ─────────────────────────────────────────────────────────────────────────────
async function initFollower(index) {
    const f = FOLLOWERS[index];
    const base = followerRestBase(f);

    log(`Initialising follower[${index}]: ${f.name}`);
    const tokenData = await authenticate(f, base);

    // Resolve account
    const acctResp = await fetch(`${base}/account/list`, {
        headers: { Authorization: `Bearer ${tokenData.accessToken}` },
        timeout: 10000,
    });
    if (!acctResp.ok)
        throw new Error(`account/list failed for follower ${index}`);
    const accounts = await acctResp.json();

    let account = accounts[0];
    if (f.accountName)
        account = accounts.find((a) => a.name === f.accountName) || account;
    if (!account) throw new Error(`No account found for follower ${index}`);

    followerSessions[index] = {
        token: tokenData.accessToken,
        userId: tokenData.userId,
        accountId: account.id,
        accountSpec: account.name,
        restBase: base,
        name: f.name,
        scale: f.scale || 1.0,
        contractMap: f.contractMap || {},
        enabled: f.enabled !== false,
    };

    log(
        `Follower[${index}] ready — account: ${account.name} (id=${account.id}), scale: ${f.scale || 1}`,
    );

    // Schedule token refresh at 85 minutes
    setTimeout(
        async () => {
            try {
                const refreshed = await authenticate(f, base);
                followerSessions[index].token = refreshed.accessToken;
                log(`Follower[${index}] token refreshed`);
            } catch (e) {
                warn(`Follower[${index}] token refresh failed: ${e.message}`);
            }
        },
        85 * 60 * 1000,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// REST order placement (used for all follower order submissions)
// ─────────────────────────────────────────────────────────────────────────────
async function placeFollowerOrder(session, orderParams) {
    // orderParams: { action, symbol, orderQty, orderType, price, stopPrice, timeInForce }
    const body = {
        accountSpec: session.accountSpec,
        accountId: session.accountId,
        action: orderParams.action, // "Buy" | "Sell"
        symbol: orderParams.symbol, // e.g. "ESM5" or mapped "MESM5"
        orderQty: Math.max(1, Math.round(orderParams.orderQty * session.scale)),
        orderType: orderParams.orderType || "Market",
        timeInForce: orderParams.timeInForce || "GTC",
    };

    // Add price fields only when relevant
    if (body.orderType === "Limit" && orderParams.price)
        body.price = orderParams.price;
    if (
        (body.orderType === "Stop" || body.orderType === "StopLimit") &&
        orderParams.stopPrice
    )
        body.stopPrice = orderParams.stopPrice;

    // Apply contract map (e.g. ES → MES)
    const mappedSymbol = mapSymbol(session.contractMap, body.symbol);
    body.symbol = mappedSymbol;

    for (let attempt = 1; attempt <= CFG.maxRetries; attempt++) {
        try {
            const resp = await fetch(`${session.restBase}/order/placeorder`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${session.token}`,
                },
                body: JSON.stringify(body),
                timeout: 8000,
            });

            const data = await resp.json();
            if (resp.ok) {
                metrics.copiesOk++;
                log(
                    `✓ Follower[${session.name}] ${body.action} ${body.orderQty} ${body.symbol} — orderId=${data.orderId || data.id}`,
                );
                return data;
            } else {
                throw new Error(`${resp.status}: ${JSON.stringify(data)}`);
            }
        } catch (e) {
            if (attempt < CFG.maxRetries) {
                warn(
                    `Follower[${session.name}] attempt ${attempt} failed: ${e.message} — retrying`,
                );
                await sleep(CFG.retryDelayMs * attempt);
            } else {
                metrics.copiesFailed++;
                err(
                    `Follower[${session.name}] FAILED after ${CFG.maxRetries} attempts: ${e.message}`,
                );
                throw e;
            }
        }
    }
}

// Close/flatten a follower position (liquidate by sending opposite order)
async function flattenFollower(session, symbol, currentQty, currentDirection) {
    if (currentQty === 0) return;
    const action = currentDirection === "long" ? "Sell" : "Buy";
    log(
        `Flattening follower[${session.name}] — ${action} ${currentQty} ${symbol}`,
    );
    await placeFollowerOrder(session, {
        action,
        symbol,
        orderQty: currentQty,
        orderType: "Market",
        timeInForce: "GTC",
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Core copy logic — called on every master fill event
// ─────────────────────────────────────────────────────────────────────────────
async function onMasterFill(fill) {
    // fill shape from Tradovate executionReport:
    //   { id, orderId, contractId, contractName, action (Buy/Sell), qty, price,
    //     cumQty, ordStatus, accountId, … }

    // Dedup
    if (copiedFillIds.has(fill.id)) return;
    copiedFillIds.add(fill.id);
    if (copiedFillIds.size > 5000) {
        // Trim oldest half
        const arr = [...copiedFillIds];
        arr.slice(0, 2500).forEach((id) => copiedFillIds.delete(id));
    }

    metrics.fillsSeen++;

    const symbol = fill.contractName || fill.symbol;
    const action = fill.action; // "Buy" | "Sell"
    const qty = fill.qty || fill.orderQty || 1;
    const fillPrice = fill.price || 0;

    log(`Master fill: ${action} ${qty} ${symbol} @ ${fillPrice}`);

    // Update master position tracker
    trackMasterPosition(symbol, action, qty);

    // Fire off copies to all enabled followers in parallel
    const tasks = followerSessions
        .filter((s, i) => s && s.enabled && FOLLOWERS[i]?.enabled !== false)
        .map((session) => {
            metrics.copiesAttempted++;
            return placeFollowerOrder(session, {
                action,
                symbol,
                orderQty: qty,
                orderType: "Market", // always market on copy (fill already happened on master)
                timeInForce: "GTC",
            }).catch((e) => {
                // Non-fatal — log and continue
                err(`Copy to ${session.name} failed: ${e.message}`);
            });
        });

    await Promise.allSettled(tasks);
}

// Called when master account sends a flatten/close-all signal
async function onMasterFlatten() {
    if (!CFG.copyFlattens) return;
    log("Master flatten detected — propagating to all followers");

    for (let i = 0; i < followerSessions.length; i++) {
        const session = followerSessions[i];
        if (!session || !session.enabled) continue;

        // Get follower positions via REST and close each
        try {
            const resp = await fetch(`${session.restBase}/position/list`, {
                headers: { Authorization: `Bearer ${session.token}` },
                timeout: 8000,
            });
            if (!resp.ok) continue;
            const positions = await resp.json();
            const myPositions = positions.filter(
                (p) =>
                    p.accountId === session.accountId &&
                    (p.netPos || p.quantity) !== 0,
            );
            for (const pos of myPositions) {
                const sym = pos.contractName || String(pos.contractId);
                const qty = Math.abs(pos.netPos || pos.quantity || 0);
                const dir = (pos.netPos || pos.quantity) > 0 ? "long" : "short";
                await flattenFollower(session, sym, qty, dir);
            }
        } catch (e) {
            err(`Flatten follower[${session.name}] error: ${e.message}`);
        }
    }
}

function trackMasterPosition(symbol, action, qty) {
    const pos = masterPositions[symbol] || { qty: 0, direction: "flat" };
    const delta = action === "Buy" ? qty : -qty;
    pos.qty += delta;
    if (pos.qty > 0) pos.direction = "long";
    else if (pos.qty < 0) pos.direction = "short";
    else pos.direction = "flat";
    masterPositions[symbol] = pos;
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbol mapping  (e.g. ES → MES, NQ → MNQ)
// ─────────────────────────────────────────────────────────────────────────────
function mapSymbol(contractMap, symbol) {
    if (!contractMap || !symbol) return symbol;
    // Exact match first
    if (contractMap[symbol]) return contractMap[symbol];
    // Root match — strip expiry suffix (e.g. "ESM5" → root "ES" → "MESM5")
    for (const [from, to] of Object.entries(contractMap)) {
        if (symbol.startsWith(from)) {
            return to + symbol.slice(from.length);
        }
    }
    return symbol;
}

// ─────────────────────────────────────────────────────────────────────────────
// Master WebSocket (user/syncrequest — same pattern as tradovate_bridge.js)
// ─────────────────────────────────────────────────────────────────────────────
async function connectMaster() {
    log(`Connecting master WS to ${ENDPOINTS.ws}…`);
    ws = new WebSocket(ENDPOINTS.ws);

    ws.on("open", () => {
        lastServerMsg = Date.now();
    });
    ws.on("message", onMasterMessage);
    ws.on("close", (code) => {
        log(`Master WS closed (${code})`);
        resetMasterState();
        if (shouldReconnect && !isReconnecting) scheduleReconnect();
    });
    ws.on("error", (e) => {
        err("Master WS error: " + e.message);
    });
}

function onMasterMessage(raw) {
    lastServerMsg = Date.now();
    const msg = raw.toString();

    if (msg === "o") {
        sendMasterAuth();
        return;
    }
    if (msg === "h" || msg === "c") return;

    if (msg.startsWith("a[")) {
        let frames;
        try {
            frames = JSON.parse(msg.slice(1));
        } catch (e) {
            return;
        }
        for (const f of frames) handleMasterFrame(f);
    }
}

function sendMasterAuth() {
    if (authSent) return;
    authSent = true;
    ws.send(`authorize\n0\n\n${masterToken.accessToken}`);
}

function handleMasterFrame(m) {
    // Auth response
    if (m.i === 0) {
        if (m.s === 200) {
            isAuthenticated = true;
            log("Master authenticated ✓");
            startMasterHeartbeat();
            sendMasterSync();
        } else {
            err("Master auth rejected — check credentials");
            shouldReconnect = false;
        }
        return;
    }

    // Sync response
    if (m.i === 1 && !syncCompleted) {
        syncCompleted = true;
        log("Master syncrequest complete ✓");
        if (m.d) processMasterSync(m.d);
        return;
    }

    // Real-time props events
    if (m.e === "props" && m.d) {
        processMasterProps(m.d);
    }
}

function sendMasterSync() {
    if (!masterToken) return;
    const body = { users: [masterToken.userId] };
    ws.send(`user/syncrequest\n1\n\n${JSON.stringify(body)}`);
}

function processMasterSync(data) {
    // Resolve account
    const accounts = data.accounts || [];
    let acct = accounts[0];
    if (MASTER.accountName)
        acct = accounts.find((a) => a.name === MASTER.accountName) || acct;
    if (acct) {
        masterAccountId = acct.id;
        masterAccountSpec = acct.name;
        log(`Master account: ${masterAccountSpec} (id=${masterAccountId})`);
    }
    // Baseline positions (don't copy on startup — just track)
    for (const pos of data.positions || []) {
        if (pos.accountId !== masterAccountId) continue;
        const sym = pos.contractName || String(pos.contractId);
        masterPositions[sym] = {
            qty: Math.abs(pos.netPos || 0),
            direction: (pos.netPos || 0) > 0 ? "long" : "short",
        };
    }
    log(
        `Master baseline — ${Object.keys(masterPositions).length} open positions`,
    );
}

function processMasterProps(data) {
    // ── Fills → copy ──────────────────────────────────────────────────────────
    const fills = data.executionReports || data.fills || [];
    for (const fill of fills) {
        if (fill.accountId !== masterAccountId) continue;
        if (fill.ordStatus !== "Filled" && fill.ordStatus !== "PartFill")
            continue;
        onMasterFill(fill).catch((e) => err("onMasterFill: " + e.message));
    }

    // ── Detect flatten: position goes to zero ────────────────────────────────
    if (CFG.copyFlattens) {
        for (const pos of data.positions || []) {
            if (pos.accountId !== masterAccountId) continue;
            if ((pos.netPos || pos.quantity || 0) === 0) {
                const sym = pos.contractName || String(pos.contractId);
                if (masterPositions[sym]?.qty > 0) {
                    log(`Master position closed: ${sym}`);
                    delete masterPositions[sym];
                    // Don't call full flatten — the individual fill copy already handles it.
                    // A position → 0 is caused by a fill we already copied above.
                }
            }
        }
    }
}

function startMasterHeartbeat() {
    stopMasterHeartbeat();
    heartbeatTimer = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send("[]");
        if (Date.now() - lastServerMsg > 10000) {
            warn("Master WS timeout — reconnecting");
            if (ws) ws.close(4000, "timeout");
        }
    }, 2500);
}

function stopMasterHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

function resetMasterState() {
    isAuthenticated = false;
    authSent = false;
    syncCompleted = false;
    stopMasterHeartbeat();
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconnection
// ─────────────────────────────────────────────────────────────────────────────
function scheduleReconnect() {
    if (reconnectAttempts >= 10) {
        err("Max reconnects reached");
        return;
    }
    const delay =
        Math.min(1000 * Math.pow(2, reconnectAttempts), 60000) *
        (1 + Math.random() * 0.1);
    reconnectAttempts++;
    log(
        `Reconnect in ${(delay / 1000).toFixed(1)}s (attempt ${reconnectAttempts}/10)…`,
    );
    setTimeout(doReconnect, delay);
}

async function doReconnect() {
    if (!shouldReconnect || isReconnecting) return;
    isReconnecting = true;
    try {
        if (ws) {
            ws.removeAllListeners();
            ws = null;
        }
        masterToken = await authenticate(
            MASTER,
            ENDPOINTS.restBase.replace("/v1", "") + "/v1",
        );
        await connectMaster();
        reconnectAttempts = 0;
    } catch (e) {
        err("Reconnect failed: " + e.message);
        scheduleReconnect();
    } finally {
        isReconnecting = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TradingView webhook receiver
// ─────────────────────────────────────────────────────────────────────────────
// Alert JSON from TradingView:
//   { "action": "buy"|"sell"|"close", "symbol": "ESM5",
//     "qty": 1, "orderType": "Market", "price": 0, "stopPrice": 0 }
//
// In TradingView Pine: alert("{{strategy.order.action}} ...", alert.freq_once_per_bar_close)
// Webhook URL: http://YOUR_SERVER_IP:5682/webhook
//
async function handleTvWebhook(body) {
    metrics.tvWebhooks++;
    log(`TradingView webhook: ${JSON.stringify(body)}`);

    const action = (body.action || "").toLowerCase();
    const symbol = body.symbol || "";
    const qty = parseInt(body.qty || body.quantity || "1", 10);
    const orderType = body.orderType || "Market";
    const price = body.price || 0;
    const stopPrice = body.stopPrice || 0;

    if (!symbol) {
        warn("Webhook missing symbol");
        return;
    }

    // Normalise action
    let tvAction;
    if (action === "buy" || action === "long") tvAction = "Buy";
    else if (action === "sell" || action === "short") tvAction = "Sell";
    else if (action === "close" || action === "flat") {
        // Close master + propagate flatten
        log(`TradingView close signal for ${symbol}`);
        await onMasterFlatten();
        return;
    } else {
        warn(`Unknown TradingView action: ${action}`);
        return;
    }

    // Place on master account (via REST — we're not on the WS path)
    try {
        const masterSession = {
            token: masterToken?.accessToken,
            accountSpec: masterAccountSpec,
            accountId: masterAccountId,
            restBase: ENDPOINTS.restBase,
            scale: 1,
            contractMap: {},
            name: "MASTER",
        };
        await placeFollowerOrder(masterSession, {
            action: tvAction,
            symbol,
            orderQty: qty,
            orderType,
            price,
            stopPrice,
        });
    } catch (e) {
        err(`TradingView → master order failed: ${e.message}`);
    }

    // Also copy directly to all followers (the fill event will arrive via WS
    // a moment later, but we copy eagerly here in case WS is slow)
    const tasks = followerSessions
        .filter((s) => s && s.enabled)
        .map((session) =>
            placeFollowerOrder(session, {
                action: tvAction,
                symbol,
                orderQty: qty,
                orderType,
                price,
                stopPrice,
            }).catch((e) =>
                err(
                    `TradingView → follower ${session.name} failed: ${e.message}`,
                ),
            ),
        );
    await Promise.allSettled(tasks);
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP server — health + TradingView webhook
// ─────────────────────────────────────────────────────────────────────────────
const httpServer = http.createServer(async (req, res) => {
    const path = req.url.split("?")[0];

    // ── POST /webhook  — TradingView alerts ───────────────────────────────────
    if (req.method === "POST" && path === "/webhook") {
        let rawBody = "";
        req.on("data", (chunk) => {
            rawBody += chunk.toString();
        });
        req.on("end", async () => {
            try {
                const body = JSON.parse(rawBody);
                await handleTvWebhook(body);
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ status: "ok" }));
            } catch (e) {
                warn("Webhook parse error: " + e.message);
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: e.message }));
            }
        });
        return;
    }

    res.setHeader("Content-Type", "application/json");

    // ── GET /health ────────────────────────────────────────────────────────────
    if (path === "/health") {
        res.writeHead(200);
        res.end(
            JSON.stringify({
                status: "ok",
                master: masterAccountSpec || null,
                authenticated: isAuthenticated,
                sync_complete: syncCompleted,
                followers: followerSessions.map((s, i) => ({
                    index: i,
                    name: s?.name || FOLLOWERS[i]?.name,
                    account: s?.accountSpec || null,
                    ready: !!s,
                    enabled: s?.enabled !== false,
                })),
            }),
        );
        return;
    }

    // ── GET /status ────────────────────────────────────────────────────────────
    if (path === "/status") {
        res.writeHead(200);
        res.end(
            JSON.stringify({
                master_account: masterAccountSpec,
                master_positions: masterPositions,
                metrics,
                followers: followerSessions.map((s, i) => ({
                    index: i,
                    name: s?.name || FOLLOWERS[i]?.name,
                    account: s?.accountSpec,
                    scale: s?.scale,
                    contract_map: s?.contractMap,
                    enabled: s?.enabled,
                })),
            }),
        );
        return;
    }

    // ── GET /metrics  — Prometheus ─────────────────────────────────────────────
    if (path === "/metrics") {
        res.setHeader(
            "Content-Type",
            "text/plain; version=0.0.4; charset=utf-8",
        );
        const uptime = (Date.now() - metrics.startTime) / 1000;
        res.writeHead(200);
        res.end(
            [
                `tradovate_copier_up ${isAuthenticated && syncCompleted ? 1 : 0}`,
                `tradovate_copier_fills_seen_total ${metrics.fillsSeen}`,
                `tradovate_copier_copies_attempted_total ${metrics.copiesAttempted}`,
                `tradovate_copier_copies_ok_total ${metrics.copiesOk}`,
                `tradovate_copier_copies_failed_total ${metrics.copiesFailed}`,
                `tradovate_copier_tv_webhooks_total ${metrics.tvWebhooks}`,
                `tradovate_copier_followers ${followerSessions.filter((s) => s?.enabled).length}`,
                `tradovate_copier_uptime_seconds ${uptime.toFixed(0)}`,
            ].join("\n") + "\n",
        );
        return;
    }

    res.writeHead(404);
    res.end(
        JSON.stringify({
            error: "not found",
            endpoints: ["/health", "/status", "/metrics", "/webhook"],
        }),
    );
});

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
    log("Tradovate Trade Copier starting…");
    log(`Master account: ${MASTER.name} (${MASTER.demo ? "DEMO" : "LIVE"})`);
    log(`Followers configured: ${FOLLOWERS.length}`);

    // Validate
    if (!MASTER.name || !MASTER.password || !MASTER.cid || !MASTER.sec) {
        err(
            "Missing master credentials — set TRADOVATE_NAME, TRADOVATE_PASSWORD, TRADOVATE_CID, TRADOVATE_SEC",
        );
        process.exit(1);
    }
    if (FOLLOWERS.length === 0) {
        warn("No followers configured — set FOLLOWERS_JSON in .env");
    }

    // Start health/webhook HTTP server
    httpServer.listen(CFG.copierPort, () => {
        log(`HTTP server on port ${CFG.copierPort}`);
        log(`  /health    — liveness`);
        log(`  /status    — full state`);
        log(`  /metrics   — Prometheus`);
        log(`  POST /webhook — TradingView alert receiver`);
    });

    // Init all followers in parallel
    const followerInits = FOLLOWERS.map((_, i) =>
        initFollower(i).catch((e) =>
            err(`Follower[${i}] init failed: ${e.message}`),
        ),
    );
    await Promise.allSettled(followerInits);

    // Auth master and connect WS
    try {
        masterToken = await authenticate(MASTER, ENDPOINTS.restBase);
        log(`Master token acquired — user: ${masterToken.name}`);
        await connectMaster();
    } catch (e) {
        err("Master init failed: " + e.message);
        scheduleReconnect();
    }

    // Master token refresh at 85m
    setInterval(
        async () => {
            try {
                masterToken = await authenticate(MASTER, ENDPOINTS.restBase);
                log("Master token refreshed ✓");
            } catch (e) {
                warn("Master token refresh failed: " + e.message);
            }
        },
        85 * 60 * 1000,
    );
}

function shutdown(sig) {
    log(`${sig} — shutting down`);
    shouldReconnect = false;
    stopMasterHeartbeat();
    if (ws) ws.close(1000, "shutdown");
    httpServer.close();
    setTimeout(() => process.exit(0), 1000);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("unhandledRejection", (r) => warn("Unhandled: " + r));
process.on("uncaughtException", (e) => err("Uncaught: " + e.message));

main();

// =============================================================================
// .env additions needed for the copier
// =============================================================================
//
// # ── Existing master creds (already in your .env from tradovate_bridge.js) ──
// TRADOVATE_NAME=your_login
// TRADOVATE_PASSWORD=your_password
// TRADOVATE_APP_ID=FuturesCoPilot
// TRADOVATE_APP_VERSION=1.0
// TRADOVATE_CID=12345
// TRADOVATE_SEC=your_secret
// TRADOVATE_DEMO=false
// TRADOVATE_ACCOUNT_NAME=          # blank = first account
//
// # ── Copier-specific ─────────────────────────────────────────────────────────
// COPIER_PORT=5682
// HEARTBEAT_SEC=15
// COPY_FLATTENS=true
//
// # ── Followers (JSON array) ──────────────────────────────────────────────────
// # Each object: name, password, cid, sec, demo, accountName, scale, contractMap, enabled
// #
// # Example 1: Same contracts, same size, live account
// # FOLLOWERS_JSON=[{"name":"apex_login","password":"pw","cid":999,"sec":"sec","demo":false,"accountName":"","scale":1,"contractMap":{},"enabled":true}]
// #
// # Example 2: Master trades ES/NQ, follower trades MES/MNQ at 1:1 $ risk
// # FOLLOWERS_JSON=[{"name":"apex_login","password":"pw","cid":999,"sec":"sec","demo":false,"accountName":"","scale":1,"contractMap":{"ES":"MES","NQ":"MNQ","GC":"MGC"},"enabled":true}]
// #
// # Example 3: Two followers
// # FOLLOWERS_JSON=[
// #   {"name":"acct1","password":"pw1","cid":101,"sec":"s1","demo":false,"scale":1,"contractMap":{},"enabled":true},
// #   {"name":"acct2","password":"pw2","cid":102,"sec":"s2","demo":false,"scale":0.5,"contractMap":{"ES":"MES"},"enabled":true}
// # ]
//
// # ── TradingView webhook ──────────────────────────────────────────────────────
// # In TradingView → Alerts → Webhook URL:
// #   http://YOUR_SERVER_IP_OR_TAILSCALE_IP:5682/webhook
// #
// # Alert message JSON (paste into TradingView alert message box):
// # {"action":"{{strategy.order.action}}","symbol":"{{ticker}}","qty":1,"orderType":"Market"}
// #
// # Or for manual alerts with fixed values:
// # {"action":"buy","symbol":"ESM5","qty":1,"orderType":"Market"}
