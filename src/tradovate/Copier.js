// =============================================================================
// tradovate_copier.js  —  Production Farm Copier
// =============================================================================
//
// Designed for:
//   Phase 1 — 5× TakeProfit Trader $150k
//   Phase 2 — add up to 20× Apex $300k
//
// ARCHITECTURE
// ─────────────
//
//   ┌──────────────────────────────────────────────────────────┐
//   │  MASTER  (your live Tradovate / TradingView account)     │
//   │  WebSocket user/syncrequest  →  fill events              │
//   └────────────────────────┬─────────────────────────────────┘
//                            │ fill event
//              ┌─────────────▼──────────────┐
//              │      Focus Tag (Redis)      │  ← Python engine writes focus:top_assets
//              │      Fan-Out Queue          │  batches of 5, serial per account
//              └─────────┬──────────────────┘
//                        │
//     ┌──────────────────┼──────────────────┐
//     ▼                  ▼                  ▼
//  [TPT_1]           [TPT_2..5]         [Apex_1..20]
//  CircuitBreaker    CircuitBreaker     CircuitBreaker
//  $4500 daily DD    $4500 daily DD     $4500 daily DD
//  $9000 trailing    $9000 trailing     $7500 trailing
//  Same contracts    Same contracts     Auto → micros
//
//   Python engine ──→ GET  /dashboard        — full farm health
//   Python engine ──→ POST /kill/:id         — kill switch per account
//   Python engine ──→ POST /kill/all
//   Python engine ──→ POST /revive/:id
//   Python engine ──→ POST /revive/all
//   Python engine ──→ POST /enable/:id | /disable/:id
//
// CONTRACT AUTO-MAP (Apex only, autoMicros: true)
//   ES → MES   NQ → MNQ   GC → MGC   RTY → M2K
//   YM → MYM   BTC → MBTC  6E → M6E  6J → M6J
//
// QUICK START
// ───────────
//   npm install ws node-fetch@2 dotenv ioredis
//   node scripts/tradovate_copier.js
//
// =============================================================================

"use strict";

require("dotenv").config();
const WebSocket = require("ws");
const http = require("http");
const fetch = require("node-fetch"); // npm install node-fetch@2

let Redis = null;
try {
    Redis = require("ioredis");
} catch (_) {}

// ─────────────────────────────────────────────────────────────────────────────
// Env helpers
// ─────────────────────────────────────────────────────────────────────────────
const env = (k, d = "") => process.env[k] || d;
const envInt = (k, d) => parseInt(env(k, String(d)), 10);
const envBool = (k, d = false) =>
    env(k, d ? "true" : "false").toLowerCase() === "true";

// ─────────────────────────────────────────────────────────────────────────────
// Broker rule configs
// ─────────────────────────────────────────────────────────────────────────────
const BROKER_RULES = {
    TPT_150K: {
        label: "TakeProfit Trader $150k",
        dailyLossLimit: 4500,
        trailingDD: 9000,
        autoMicros: false,
    },
    APEX_300K: {
        label: "Apex $300k",
        dailyLossLimit: 4500,
        trailingDD: 7500,
        autoMicros: true,
    },
    APEX_150K: {
        label: "Apex $150k",
        dailyLossLimit: 3000,
        trailingDD: 4500,
        autoMicros: true,
    },
    CUSTOM: {
        label: "Custom",
        dailyLossLimit: 99999,
        trailingDD: 99999,
        autoMicros: false,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Micro contract map  (applied when autoMicros = true)
// ─────────────────────────────────────────────────────────────────────────────
const MICRO_MAP = {
    ES: "MES",
    NQ: "MNQ",
    GC: "MGC",
    RTY: "M2K",
    YM: "MYM",
    BTC: "MBTC",
    "6E": "M6E",
    "6J": "M6J",
};

// ─────────────────────────────────────────────────────────────────────────────
// Master config
// ─────────────────────────────────────────────────────────────────────────────
const MASTER = {
    name: env("TRADOVATE_NAME"),
    password: env("TRADOVATE_PASSWORD"),
    appId: env("TRADOVATE_APP_ID", "FuturesCoPilot"),
    appVersion: env("TRADOVATE_APP_VERSION", "1.0"),
    cid: envInt("TRADOVATE_CID", 0),
    sec: env("TRADOVATE_SEC"),
    demo: envBool("TRADOVATE_DEMO"),
    accountName: env("TRADOVATE_ACCOUNT_NAME"),
};

const IS_DEMO = MASTER.demo;
const REST_BASE = IS_DEMO
    ? "https://demo.tradovateapi.com/v1"
    : "https://live.tradovateapi.com/v1";
const WS_URL = IS_DEMO
    ? "wss://demo.tradovateapi.com/v1/websocket"
    : "wss://live.tradovateapi.com/v1/websocket";

const followerBase = (demo) =>
    demo
        ? "https://demo.tradovateapi.com/v1"
        : "https://live.tradovateapi.com/v1";

// ─────────────────────────────────────────────────────────────────────────────
// Follower definitions  (FOLLOWERS_JSON in .env — see bottom of file)
// ─────────────────────────────────────────────────────────────────────────────
const FOLLOWER_DEFS = JSON.parse(env("FOLLOWERS_JSON", "[]"));

// ─────────────────────────────────────────────────────────────────────────────
// Global config
// ─────────────────────────────────────────────────────────────────────────────
const CFG = {
    copierPort: envInt("COPIER_PORT", 5682),
    maxConcurrentCopies: envInt("MAX_CONCURRENT_COPIES", 5),
    orderTimeoutMs: 8000,
    maxRetries: 3,
    retryBaseMs: 400,
    dailyLossWarnPct: 0.8,
};

// ─────────────────────────────────────────────────────────────────────────────
// Redis (optional — focus-asset tagging from Python engine)
// ─────────────────────────────────────────────────────────────────────────────
let redis = null;
if (Redis) {
    try {
        redis = new Redis({
            host: env("REDIS_HOST", "localhost"),
            port: envInt("REDIS_PORT", 6380),
            password: env("REDIS_PASSWORD") || undefined,
            lazyConnect: true,
        });
        redis.connect().catch(() => {
            redis = null;
        });
    } catch (_) {
        redis = null;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────
const ts = () => new Date().toISOString();
const log = (m) => console.log(`${ts()} [Copier] ${m}`);
const warn = (m) => console.warn(`${ts()} [Copier] ⚠ ${m}`);
const err = (m) => console.error(`${ts()} [Copier] ✗ ${m}`);

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────
const metrics = {
    fillsSeen: 0,
    copiesAttempted: 0,
    copiesOk: 0,
    copiesFailed: 0,
    circuitTrips: 0,
    killSwitches: 0,
    startTime: Date.now(),
    fillLog: [], // last 100 fills with per-account results
};

function recordFill(fill, focus, results) {
    metrics.fillLog.unshift({ ts: ts(), fill, focus, results });
    if (metrics.fillLog.length > 100) metrics.fillLog.pop();
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitBreaker  — per account
// ─────────────────────────────────────────────────────────────────────────────
class CircuitBreaker {
    constructor(id, rules) {
        this.id = id;
        this.rules = rules;
        this.killed = false; // manual
        this.autoTripped = false; // P&L guard
        this.tripReason = "";
        this.dailyPnl = 0;
        this.peakBalance = 0;
    }

    get blocked() {
        return this.killed || this.autoTripped;
    }
    get status() {
        return this.killed ? "KILLED" : this.autoTripped ? "TRIPPED" : "OK";
    }

    kill(reason = "manual") {
        this.killed = true;
        this.tripReason = reason;
        metrics.killSwitches++;
        warn(`CB[${this.id}] KILLED — ${reason}`);
    }

    revive() {
        this.killed = this.autoTripped = false;
        this.tripReason = "";
        log(`CB[${this.id}] revived`);
    }

    checkPnl(dailyPnl, balance) {
        if (this.killed) return;
        this.dailyPnl = dailyPnl;
        if (balance > this.peakBalance) this.peakBalance = balance;

        const loss = Math.abs(Math.min(dailyPnl, 0));
        const ddPeak = this.peakBalance > 0 ? this.peakBalance - balance : 0;

        if (
            !this.autoTripped &&
            loss >= this.rules.dailyLossLimit * CFG.dailyLossWarnPct
        )
            warn(
                `CB[${this.id}] daily loss warning — $${loss.toFixed(0)} / $${this.rules.dailyLossLimit}`,
            );

        if (loss >= this.rules.dailyLossLimit) {
            this.autoTripped = true;
            this.tripReason = `Daily loss $${loss.toFixed(0)} ≥ limit $${this.rules.dailyLossLimit}`;
            metrics.circuitTrips++;
            err(`CB[${this.id}] TRIPPED — ${this.tripReason}`);
        }
        if (ddPeak >= this.rules.trailingDD) {
            this.autoTripped = true;
            this.tripReason = `Trailing DD $${ddPeak.toFixed(0)} ≥ limit $${this.rules.trailingDD}`;
            metrics.circuitTrips++;
            err(`CB[${this.id}] TRIPPED — ${this.tripReason}`);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AccountSession  — one per follower account
// ─────────────────────────────────────────────────────────────────────────────
class AccountSession {
    constructor(def, index) {
        this.index = index;
        this.id = def.id || `acct_${index}`;
        this.label = def.label || def.name;
        this.def = def;
        this.rules = BROKER_RULES[def.broker] || BROKER_RULES.CUSTOM;
        this.restBase = followerBase(def.demo === true);
        this.token = null;
        this.accountId = null;
        this.accountSpec = null;
        this.scale = def.scale || 1;
        this.enabled = def.enabled !== false;
        this.ready = false;
        this.cb = new CircuitBreaker(this.id, this.rules);
        this._queue = Promise.resolve(); // serial per-account order queue
    }

    // Map symbol and calculate qty for this account
    resolveOrder(masterSymbol, masterQty) {
        let symbol = masterSymbol;
        const qty = Math.max(1, Math.round(masterQty * this.scale));

        if (this.rules.autoMicros) {
            const root = masterSymbol.replace(/[A-Z]\d+$/, "");
            const micro = MICRO_MAP[root];
            if (micro) symbol = micro + masterSymbol.slice(root.length);
        }

        return { symbol, qty };
    }

    // Enqueue — guarantees no double-fill per account even if two fills arrive quickly
    enqueue(params) {
        this._queue = this._queue
            .then(() => this._place(params))
            .catch(() => {});
        return this._queue;
    }

    async _place(params) {
        if (!this.ready || !this.enabled || this.cb.blocked)
            return {
                skipped: true,
                reason: this.cb.blocked ? this.cb.status : "not_ready",
            };

        const body = {
            accountSpec: this.accountSpec,
            accountId: this.accountId,
            action: params.action,
            symbol: params.symbol,
            orderQty: params.qty,
            orderType: "Market",
            timeInForce: "GTC",
        };

        for (let attempt = 1; attempt <= CFG.maxRetries; attempt++) {
            try {
                const resp = await fetch(`${this.restBase}/order/placeorder`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        Authorization: `Bearer ${this.token}`,
                    },
                    body: JSON.stringify(body),
                    timeout: CFG.orderTimeoutMs,
                });
                const data = await resp.json();
                if (resp.ok) {
                    metrics.copiesOk++;
                    log(
                        `✓ [${this.id}] ${body.action} ${body.orderQty} ${body.symbol} orderId=${data.orderId || data.id}`,
                    );
                    this._refreshPnl().catch(() => {});
                    return { ok: true, id: this.id, orderId: data.orderId };
                }
                throw new Error(`${resp.status}: ${JSON.stringify(data)}`);
            } catch (ex) {
                if (attempt < CFG.maxRetries)
                    await sleep(CFG.retryBaseMs * attempt);
                else {
                    metrics.copiesFailed++;
                    err(
                        `✗ [${this.id}] FAILED ${body.action} ${body.orderQty} ${body.symbol}: ${ex.message}`,
                    );
                    return { ok: false, id: this.id, error: ex.message };
                }
            }
        }
    }

    async _refreshPnl() {
        try {
            const resp = await fetch(
                `${this.restBase}/account/find?name=${encodeURIComponent(this.accountSpec)}`,
                {
                    headers: { Authorization: `Bearer ${this.token}` },
                    timeout: 5000,
                },
            );
            if (!resp.ok) return;
            const d = await resp.json();
            this.cb.checkPnl(
                d.dailyRealizedPnl || d.realizedPnl || 0,
                d.balance || d.cashValue || 0,
            );
        } catch (_) {}
    }

    async init() {
        log(`Init [${this.id}] ${this.label} (${this.rules.label})…`);
        const tok = await authRest(this.def, this.restBase);
        this.token = tok.accessToken;

        const r = await fetch(`${this.restBase}/account/list`, {
            headers: { Authorization: `Bearer ${this.token}` },
            timeout: 10000,
        });
        if (!r.ok) throw new Error(`account/list ${r.status}`);
        const accounts = await r.json();
        let acct = accounts[0];
        if (this.def.accountName)
            acct =
                accounts.find((a) => a.name === this.def.accountName) || acct;
        if (!acct) throw new Error("No account found");
        this.accountId = acct.id;
        this.accountSpec = acct.name;
        this.ready = true;
        log(`[${this.id}] ready — ${this.accountSpec}`);
        setTimeout(() => this._refreshToken(), 85 * 60 * 1000);
    }

    async _refreshToken() {
        try {
            const tok = await authRest(this.def, this.restBase);
            this.token = tok.accessToken;
            log(`[${this.id}] token refreshed`);
        } catch (ex) {
            warn(`[${this.id}] token refresh failed: ${ex.message}`);
        }
        setTimeout(() => this._refreshToken(), 85 * 60 * 1000);
    }

    toStatus() {
        return {
            id: this.id,
            label: this.label,
            broker: this.rules.label,
            account: this.accountSpec,
            ready: this.ready,
            enabled: this.enabled,
            circuit: this.cb.status,
            trip_reason: this.cb.tripReason || null,
            daily_pnl: this.cb.dailyPnl,
            auto_micros: this.rules.autoMicros,
            limits: {
                daily_loss: this.rules.dailyLossLimit,
                trailing_dd: this.rules.trailingDD,
            },
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Account pool
// ─────────────────────────────────────────────────────────────────────────────
const pool = FOLLOWER_DEFS.map((def, i) => new AccountSession(def, i));
const getSession = (id) => pool.find((s) => s.id === id);

// ─────────────────────────────────────────────────────────────────────────────
// Auth REST
// ─────────────────────────────────────────────────────────────────────────────
async function authRest(creds, base) {
    const r = await fetch(`${base}/auth/accesstokenrequest`, {
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
    if (!r.ok) throw new Error(`Auth ${r.status}: ${await r.text()}`);
    const d = await r.json();
    if (!d.accessToken) throw new Error("No accessToken");
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Focus-asset tag  — reads from Python engine's Redis keys
// ─────────────────────────────────────────────────────────────────────────────
async function checkFocus(symbol) {
    if (!redis) return { inFocus: null };
    try {
        const root = symbol.replace(/[A-Z]\d+$/, "");
        const inFocus = (await redis.sismember("focus:top_assets", root)) === 1;
        const bias = await redis.hget(`focus:bias:${root}`, "direction");
        return { inFocus, root, bias: bias || null };
    } catch (_) {
        return { inFocus: null };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fan-out  — called on every master fill
// ─────────────────────────────────────────────────────────────────────────────
async function fanOut(fill) {
    metrics.fillsSeen++;

    const masterSymbol = fill.contractName || fill.symbol || "";
    const action = fill.action;
    const masterQty = fill.qty || 1;
    const price = fill.price || 0;

    const focus = await checkFocus(masterSymbol);

    if (focus.inFocus === true)
        log(
            `Fill [IN FOCUS]: ${action} ${masterQty} ${masterSymbol} @ ${price} bias=${focus.bias || "?"}`,
        );
    else
        log(
            `Fill [off-focus]: ${action} ${masterQty} ${masterSymbol} @ ${price}`,
        );

    const active = pool.filter((s) => s.ready && s.enabled && !s.cb.blocked);
    const batches = chunk(active, CFG.maxConcurrentCopies);
    const results = [];

    for (const batch of batches) {
        const settled = await Promise.allSettled(
            batch.map((s) => {
                metrics.copiesAttempted++;
                const { symbol, qty } = s.resolveOrder(masterSymbol, masterQty);
                return s
                    .enqueue({ action, symbol, qty })
                    .then((r) => ({
                        ...r,
                        id: s.id,
                        label: s.label,
                        symbol,
                        qty,
                    }))
                    .catch((ex) => ({
                        ok: false,
                        id: s.id,
                        error: ex.message,
                    }));
            }),
        );
        results.push(
            ...settled.map((r) =>
                r.status === "fulfilled" ? r.value : r.reason,
            ),
        );
    }

    recordFill(
        { symbol: masterSymbol, action, qty: masterQty, price },
        { inFocus: focus.inFocus, bias: focus.bias },
        results,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Master WebSocket
// ─────────────────────────────────────────────────────────────────────────────
let ws = null;
let masterToken = null;
let masterAccountId = null;
let masterAccountSpec = "";
let isAuthenticated = false;
let authSent = false;
let syncCompleted = false;
let lastServerMsg = Date.now();
let heartbeatTimer = null;
let reconnectAttempts = 0;
let shouldReconnect = true;
let isReconnecting = false;
const copiedFillIds = new Set();

async function connectMaster() {
    log(`Connecting master WS → ${WS_URL}`);
    ws = new WebSocket(WS_URL);
    ws.on("open", () => {
        lastServerMsg = Date.now();
    });
    ws.on("message", onMsg);
    ws.on("close", (code) => {
        log(`Master WS closed (${code})`);
        resetMasterState();
        if (shouldReconnect && !isReconnecting) scheduleReconnect();
    });
    ws.on("error", (ex) => {
        err("Master WS: " + ex.message);
    });
}

function onMsg(raw) {
    lastServerMsg = Date.now();
    const m = raw.toString();
    if (m === "o") {
        sendAuth();
        return;
    }
    if (m === "h" || m === "c") return;
    if (!m.startsWith("a[")) return;
    let frames;
    try {
        frames = JSON.parse(m.slice(1));
    } catch (_) {
        return;
    }
    frames.forEach(handleFrame);
}

function sendAuth() {
    if (authSent) return;
    authSent = true;
    ws.send(`authorize\n0\n\n${masterToken.accessToken}`);
}

function handleFrame(f) {
    if (f.i === 0) {
        if (f.s === 200) {
            isAuthenticated = true;
            log("Master authenticated ✓");
            startHeartbeat();
            ws.send(
                `user/syncrequest\n1\n\n${JSON.stringify({ users: [masterToken.userId] })}`,
            );
        } else {
            err("Master auth rejected — check credentials");
            shouldReconnect = false;
        }
        return;
    }
    if (f.i === 1 && !syncCompleted) {
        syncCompleted = true;
        log("Master sync complete ✓");
        if (f.d) resolveAccount(f.d);
        return;
    }
    if (f.e === "props" && f.d) handleProps(f.d);
}

function resolveAccount(data) {
    const accounts = data.accounts || [];
    let acct = accounts[0];
    if (MASTER.accountName)
        acct = accounts.find((a) => a.name === MASTER.accountName) || acct;
    if (acct) {
        masterAccountId = acct.id;
        masterAccountSpec = acct.name;
        log(`Master account: ${masterAccountSpec} (id=${masterAccountId})`);
    }
}

function handleProps(data) {
    const fills = data.executionReports || data.fills || [];
    for (const fill of fills) {
        if (fill.accountId !== masterAccountId) continue;
        if (fill.ordStatus !== "Filled" && fill.ordStatus !== "PartFill")
            continue;
        if (copiedFillIds.has(fill.id)) continue;
        copiedFillIds.add(fill.id);
        if (copiedFillIds.size > 5000) {
            [...copiedFillIds]
                .slice(0, 2500)
                .forEach((id) => copiedFillIds.delete(id));
        }
        fanOut(fill).catch((ex) => err("fanOut: " + ex.message));
    }
}

function startHeartbeat() {
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) ws.send("[]");
        if (Date.now() - lastServerMsg > 10000) {
            warn("Master WS dead — forcing reconnect");
            ws?.close(4000, "timeout");
        }
    }, 2500);
}

function resetMasterState() {
    isAuthenticated = authSent = syncCompleted = false;
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

function scheduleReconnect() {
    if (reconnectAttempts >= 10) {
        err("Max reconnects reached");
        return;
    }
    const delay =
        Math.min(1000 * Math.pow(2, reconnectAttempts), 60000) *
        (1 + Math.random() * 0.1);
    reconnectAttempts++;
    log(`Reconnect in ${(delay / 1000).toFixed(1)}s (${reconnectAttempts}/10)`);
    setTimeout(doReconnect, delay);
}

async function doReconnect() {
    if (!shouldReconnect || isReconnecting) return;
    isReconnecting = true;
    try {
        ws?.removeAllListeners();
        ws = null;
        masterToken = await authRest(MASTER, REST_BASE);
        await connectMaster();
        reconnectAttempts = 0;
    } catch (ex) {
        err("Reconnect failed: " + ex.message);
        scheduleReconnect();
    } finally {
        isReconnecting = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP server  — dashboard + kill switches
// ─────────────────────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
    const path = req.url.split("?")[0];
    const method = req.method;
    res.setHeader("Content-Type", "application/json");

    // GET /health
    if (method === "GET" && path === "/health") {
        res.writeHead(200);
        return res.end(
            JSON.stringify({
                status: isAuthenticated && syncCompleted ? "ok" : "degraded",
                master: masterAccountSpec || null,
                authenticated: isAuthenticated,
                accounts: pool.length,
                ready: pool.filter((s) => s.ready).length,
                ok: pool.filter((s) => s.ready && !s.cb.blocked).length,
                killed: pool.filter((s) => s.cb.killed).length,
                tripped: pool.filter((s) => s.cb.autoTripped).length,
            }),
        );
    }

    // GET /dashboard  — full farm snapshot for Python engine
    if (method === "GET" && path === "/dashboard") {
        res.writeHead(200);
        return res.end(
            JSON.stringify({
                master: {
                    account: masterAccountSpec,
                    authenticated: isAuthenticated,
                    sync: syncCompleted,
                },
                accounts: pool.map((s) => s.toStatus()),
                metrics: {
                    fills_seen: metrics.fillsSeen,
                    copies_attempted: metrics.copiesAttempted,
                    copies_ok: metrics.copiesOk,
                    copies_failed: metrics.copiesFailed,
                    circuit_trips: metrics.circuitTrips,
                    kill_switches: metrics.killSwitches,
                    uptime_sec: Math.round(
                        (Date.now() - metrics.startTime) / 1000,
                    ),
                },
                recent_fills: metrics.fillLog.slice(0, 20),
            }),
        );
    }

    // GET /accounts
    if (method === "GET" && path === "/accounts") {
        res.writeHead(200);
        return res.end(JSON.stringify(pool.map((s) => s.toStatus())));
    }

    // POST /kill/:id  or  /kill/all
    if (method === "POST" && path.startsWith("/kill/")) {
        const id = path.slice(6);
        if (id === "all") {
            pool.forEach((s) => s.cb.kill("killall"));
            res.writeHead(200);
            return res.end(JSON.stringify({ ok: true, killed: pool.length }));
        }
        const s = getSession(id);
        if (!s) {
            res.writeHead(404);
            return res.end(JSON.stringify({ error: "not found" }));
        }
        s.cb.kill("dashboard");
        res.writeHead(200);
        return res.end(JSON.stringify({ ok: true, id, circuit: s.cb.status }));
    }

    // POST /revive/:id  or  /revive/all
    if (method === "POST" && path.startsWith("/revive/")) {
        const id = path.slice(8);
        if (id === "all") {
            pool.forEach((s) => s.cb.revive());
            res.writeHead(200);
            return res.end(JSON.stringify({ ok: true, revived: pool.length }));
        }
        const s = getSession(id);
        if (!s) {
            res.writeHead(404);
            return res.end(JSON.stringify({ error: "not found" }));
        }
        s.cb.revive();
        res.writeHead(200);
        return res.end(JSON.stringify({ ok: true, id, circuit: s.cb.status }));
    }

    // POST /enable/:id  |  /disable/:id  (pause without killing)
    if (
        method === "POST" &&
        (path.startsWith("/enable/") || path.startsWith("/disable/"))
    ) {
        const on = path.startsWith("/enable/");
        const id = path.slice(on ? 8 : 9);
        const targets = id === "all" ? pool : [getSession(id)].filter(Boolean);
        if (!targets.length) {
            res.writeHead(404);
            return res.end(JSON.stringify({ error: "not found" }));
        }
        targets.forEach((s) => {
            s.enabled = on;
        });
        res.writeHead(200);
        return res.end(
            JSON.stringify({ ok: true, enabled: on, count: targets.length }),
        );
    }

    // GET /metrics  — Prometheus
    if (method === "GET" && path === "/metrics") {
        res.setHeader(
            "Content-Type",
            "text/plain; version=0.0.4; charset=utf-8",
        );
        const uptime = (Date.now() - metrics.startTime) / 1000;
        const lines = [
            `tradovate_copier_up ${isAuthenticated && syncCompleted ? 1 : 0}`,
            `tradovate_copier_accounts_total ${pool.length}`,
            `tradovate_copier_accounts_ok ${pool.filter((s) => s.ready && !s.cb.blocked).length}`,
            `tradovate_copier_fills_seen_total ${metrics.fillsSeen}`,
            `tradovate_copier_copies_ok_total ${metrics.copiesOk}`,
            `tradovate_copier_copies_failed_total ${metrics.copiesFailed}`,
            `tradovate_copier_circuit_trips_total ${metrics.circuitTrips}`,
            `tradovate_copier_uptime_seconds ${uptime.toFixed(0)}`,
            ...pool.map(
                (s) =>
                    `tradovate_account_ok{id="${s.id}"} ${s.cb.blocked ? 0 : 1}`,
            ),
        ];
        res.writeHead(200);
        return res.end(lines.join("\n") + "\n");
    }

    res.writeHead(404);
    res.end(
        JSON.stringify({
            error: "not found",
            endpoints: [
                "GET /health",
                "GET /dashboard",
                "GET /accounts",
                "GET /metrics",
                "POST /kill/:id|all",
                "POST /revive/:id|all",
                "POST /enable/:id|all",
                "POST /disable/:id|all",
            ],
        }),
    );
});

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function chunk(arr, n) {
    const out = [];
    for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
    const tpt = pool.filter((s) => s.def.broker === "TPT_150K").length;
    const apex = pool.filter((s) => s.def.broker?.startsWith("APEX")).length;
    log("════════════════════════════════════════════");
    log(" Tradovate Farm Copier");
    log(`   Master  : ${MASTER.name} (${IS_DEMO ? "DEMO" : "LIVE"})`);
    log(
        `   Pool    : ${pool.length} accounts — ${tpt} TPT $150k  +  ${apex} Apex`,
    );
    log("════════════════════════════════════════════");

    if (!MASTER.name || !MASTER.cid || !MASTER.sec) {
        err(
            "Missing master credentials — set TRADOVATE_NAME / CID / SEC in .env",
        );
        process.exit(1);
    }

    server.listen(CFG.copierPort, () => {
        log(`HTTP on :${CFG.copierPort}`);
        log("  GET  /dashboard          — full farm health");
        log("  POST /kill/:id|all       — kill switch");
        log("  POST /revive/:id|all     — revive");
        log("  POST /disable/:id|all    — pause without killing");
    });

    // Init followers in batches of 5 (avoids rate-limiting Tradovate auth)
    for (const batch of chunk(pool, 5)) {
        await Promise.allSettled(
            batch.map((s) =>
                s.init().catch((ex) => err(`Init [${s.id}]: ${ex.message}`)),
            ),
        );
        await sleep(600);
    }
    log(`Pool ready: ${pool.filter((s) => s.ready).length} / ${pool.length}`);

    masterToken = await authRest(MASTER, REST_BASE);
    log(`Master token: ${masterToken.name}`);
    await connectMaster();

    setInterval(
        async () => {
            try {
                masterToken = await authRest(MASTER, REST_BASE);
                log("Master token refreshed");
            } catch (ex) {
                warn("Master token refresh: " + ex.message);
            }
        },
        85 * 60 * 1000,
    );
}

process.on("SIGINT", () => {
    shouldReconnect = false;
    ws?.close();
    server.close();
    process.exit(0);
});
process.on("SIGTERM", () => {
    shouldReconnect = false;
    ws?.close();
    server.close();
    process.exit(0);
});
process.on("unhandledRejection", (r) => warn("Unhandled: " + r));
process.on("uncaughtException", (e) => err("Uncaught: " + e.message));

main().catch((ex) => {
    err("Fatal: " + ex.message);
    process.exit(1);
});

// =============================================================================
// .env reference
// =============================================================================
//
// # ── Master ────────────────────────────────────────────────────────────────
// TRADOVATE_NAME=your_login
// TRADOVATE_PASSWORD=your_password
// TRADOVATE_APP_ID=FuturesCoPilot
// TRADOVATE_APP_VERSION=1.0
// TRADOVATE_CID=12345
// TRADOVATE_SEC=your_secret
// TRADOVATE_DEMO=false
// TRADOVATE_ACCOUNT_NAME=          # blank = first account
//
// # ── Copier ─────────────────────────────────────────────────────────────────
// COPIER_PORT=5682
// MAX_CONCURRENT_COPIES=5          # fan-out batch size (5 is safe for Tradovate rate limits)
//
// # ── Redis (your existing instance) ────────────────────────────────────────
// REDIS_HOST=localhost
// REDIS_PORT=6380
// REDIS_PASSWORD=your_redis_pw
//
// # Python engine writes focus like this (your scorer.py already has this info):
// #   await redis.delete("focus:top_assets")
// #   await redis.sadd("focus:top_assets", "NQ", "ES")
// #   await redis.hset("focus:bias:NQ", "direction", "long")  # from daily bias analyzer
//
// # ── Phase 1: 5× TPT $150k ──────────────────────────────────────────────────
// # ── Phase 2: add Apex accounts to the array below ─────────────────────────
// #
// # broker options: TPT_150K | APEX_300K | APEX_150K | CUSTOM
// #
// # TPT_150K  → same contracts as master (ES copies as ES)
// # APEX_300K → auto-maps to micros (ES→MES, NQ→MNQ, etc)  +  $7500 trailing DD guard
// #
// # scale: multiplier on top of the auto-micro map.
// #   scale:1 + APEX_300K = master trades 1 ES → follower gets 1 MES (10× less notional, correct)
// #   scale:2 + APEX_300K = master trades 1 ES → follower gets 2 MES
//
// FOLLOWERS_JSON=[
//   {"id":"tpt_1","label":"TPT 1","name":"tpt1@x.com","password":"pw","cid":101,"sec":"s","demo":false,"broker":"TPT_150K","scale":1,"enabled":true},
//   {"id":"tpt_2","label":"TPT 2","name":"tpt2@x.com","password":"pw","cid":102,"sec":"s","demo":false,"broker":"TPT_150K","scale":1,"enabled":true},
//   {"id":"tpt_3","label":"TPT 3","name":"tpt3@x.com","password":"pw","cid":103,"sec":"s","demo":false,"broker":"TPT_150K","scale":1,"enabled":true},
//   {"id":"tpt_4","label":"TPT 4","name":"tpt4@x.com","password":"pw","cid":104,"sec":"s","demo":false,"broker":"TPT_150K","scale":1,"enabled":true},
//   {"id":"tpt_5","label":"TPT 5","name":"tpt5@x.com","password":"pw","cid":105,"sec":"s","demo":false,"broker":"TPT_150K","scale":1,"enabled":true},
//   {"id":"apex_1","label":"Apex 1","name":"apex1@x.com","password":"pw","cid":201,"sec":"s","demo":false,"broker":"APEX_300K","scale":1,"enabled":true}
// ]
