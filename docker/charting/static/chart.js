/* ============================================================================
   Ruby Futures — Charting Service · chart.js
   ApexCharts — primary instance (candlestick + volume + overlays)
               + optional RSI sub-pane instance

   Flow:
     1. Boot       — resolve DATA_SERVICE_URL from /config (rendered by nginx)
     2. Assets     — GET /bars/assets → populate symbol tabs
     3. Bars       — GET /bars/{ticker}?interval=X&days_back=Y → split format
     4. Render     — primary ApexCharts instance:
                       series[0]  candlestick (price, left y-axis)
                       series[1]  bar         (volume, right y-axis, scaled 20%)
                       series[2+] line        (EMA9, EMA21, BB upper/mid/lower, VWAP)
                     optional RSI ApexCharts sub-pane (separate instance)
     5. Live       — SSE /sse/dashboard → in-place forming-candle update
                     → indicators recalculated and pushed incrementally
     6. Controls   — symbol / interval / days-back / indicator toggles
   ============================================================================ */

"use strict";

// ── Constants ──────────────────────────────────────────────────────────────────

const VALID_INTERVALS = {
    "1m": true,
    "5m": true,
    "15m": true,
    "1h": true,
    "1d": true,
};

const SSE_RECONNECT_BASE = 2_000;
const SSE_RECONNECT_MAX = 30_000;
const SSE_RECONNECT_MULT = 1.5;

// Max candles kept in memory — prune oldest when exceeded to bound RAM usage
const MAX_CANDLES = 2_000;

// Indicator series indices within the PRIMARY chart (after candle=0, volume=1)
const IDX = {
    EMA9: 2,
    EMA21: 3,
    BB_UPPER: 4,
    BB_MID: 5,
    BB_LOWER: 6,
    VWAP: 7,
};

// ── Palette — mirrors CSS custom properties ────────────────────────────────────
const C = {
    bg: "#0d0f14",
    bgSurface: "#13161e",
    bgElevated: "#1a1e2a",
    border: "#252a36",
    borderSubtle: "#1c2030",
    textPrimary: "#e2e8f0",
    textSecond: "#8892a4",
    textMuted: "#4a5568",
    accent: "#6366f1",
    green: "#22c55e",
    red: "#ef4444",
    volUp: "rgba(34,197,94,0.40)",
    volDown: "rgba(239,68,68,0.40)",
    ema9: "#f59e0b",
    ema21: "#3b82f6",
    bbUpper: "rgba(168,85,247,0.70)",
    bbMid: "rgba(168,85,247,0.40)",
    bbLower: "rgba(168,85,247,0.70)",
    vwap: "#06b6d4",
    rsiLine: "#a78bfa",
    rsiOB: "rgba(239,68,68,0.25)",
    rsiOS: "rgba(34,197,94,0.25)",
    fontMono: '"JetBrains Mono","Fira Code",ui-monospace,monospace',
};

// ── Application state ──────────────────────────────────────────────────────────
const state = {
    dataServiceUrl: "",
    assets: [], // [{name, ticker, bar_count, has_data}]
    activeTicker: null,
    activeName: null,
    activeInterval: "15m",
    activeDays: 14,
    liveEnabled: true,

    // Active indicator flags
    indicators: {
        ema9: true,
        ema21: true,
        bb: false,
        vwap: false,
        rsi: false,
    },

    // Primary ApexCharts instance (candlestick + volume + overlays)
    chart: null,
    // RSI sub-pane ApexCharts instance (separate element, shown/hidden)
    chartRsi: null,

    // Live data arrays — candleData & volumeData shared between chart + SSE
    // candleData : [{x: <unix ms>, y: [o, h, l, c]}]
    // volumeData : [{x: <unix ms>, y: <vol>, fillColor: string}]
    candleData: [],
    volumeData: [],

    // Computed indicator series (all [{x: ms, y: value}])
    ema9Data: [],
    ema21Data: [],
    bbUpperData: [],
    bbMidData: [],
    bbLowerData: [],
    vwapData: [],
    rsiData: [],

    // SSE
    sseSource: null,
    sseReconnectMs: SSE_RECONNECT_BASE,
    sseReconnectTimer: null,

    // AbortController for the current in-flight /bars fetch
    loadAbortCtrl: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const dom = {
    chartEl: $("chart"),
    chartRsiEl: $("chart-rsi"),
    overlay: $("chart-overlay"),
    overlayMsg: $("overlay-msg"),
    errorBanner: $("error-banner"),
    errorText: $("error-text"),
    errorDismiss: $("error-dismiss"),
    symbolTabs: $("symbol-tabs"),
    intervalBtns: document.querySelectorAll(".interval-btn"),
    indBtns: document.querySelectorAll(".ind-btn"),
    daysSelect: $("days-back"),
    liveToggle: $("live-toggle"),
    barCount: $("bar-count"),
    lastPrice: $("last-price"),
    lastChange: $("last-change"),
    lastTs: $("last-ts"),
    sseBadge: $("sse-status"),
    statusDot: $("status-dot"),
    statusLabel: $("status-label"),
};

// ── UI helpers ─────────────────────────────────────────────────────────────────
function setStatus(s, label) {
    dom.statusDot.className = "status-dot status-" + s;
    dom.statusLabel.textContent = label;
}

function showOverlay(msg) {
    dom.overlayMsg.textContent = msg || "Loading…";
    dom.overlay.classList.remove("hidden");
}

function hideOverlay() {
    dom.overlay.classList.add("hidden");
}

function showError(msg) {
    dom.errorText.textContent = msg;
    dom.errorBanner.classList.remove("hidden");
}

function hideError() {
    dom.errorBanner.classList.add("hidden");
}

function setSseBadge(mode) {
    dom.sseBadge.className = "stat sse-badge " + mode;
    const label = {
        live: "SSE live",
        reconnecting: "SSE reconnecting…",
        error: "SSE error",
        "": "SSE off",
    };
    dom.sseBadge.textContent = label[mode] ?? "SSE off";
}

// ── Formatters ─────────────────────────────────────────────────────────────────
function fmtPrice(v) {
    if (v == null || isNaN(v)) return "—";
    if (v >= 10_000)
        return v.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
    if (v >= 1) return v.toFixed(4);
    return v.toFixed(6);
}

function fmtVolume(v) {
    if (v == null || isNaN(v) || v === 0) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(Math.round(v));
}

function fmtTs(ms) {
    if (!ms) return "—";
    return new Date(ms).toLocaleString(undefined, {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    });
}

// ── Data parsing ───────────────────────────────────────────────────────────────
//
// /bars/{ticker} returns pandas "split" format:
//   { columns: ["open","high","low","close","volume",...],
//     index:   ["2024-03-01T09:30:00+00:00", ...],
//     data:    [[o,h,l,c,v,...], ...] }
//
// Primary series layout we build:
//   series[0]  candlestick : [{x: <ms>, y: [o, h, l, c]}]
//   series[1]  bar         : [{x: <ms>, y: <vol>, fillColor}]
//   series[2+] indicators  : [{x: <ms>, y: value}]   (appended by buildIndicatorSeries)

function splitToApex(payload) {
    if (!payload) return { candles: [], volumes: [] };

    const { columns = [], index = [], data = [] } = payload;

    // Column index map (case-insensitive)
    const col = {};
    columns.forEach((c, i) => {
        col[c.toLowerCase()] = i;
    });

    const iO = col["open"] ?? -1;
    const iH = col["high"] ?? -1;
    const iL = col["low"] ?? -1;
    const iC = col["close"] ?? -1;
    const iV = col["volume"] ?? -1;

    if (iO < 0 || iH < 0 || iL < 0 || iC < 0) {
        console.warn("[chart] Missing OHLC columns in payload. Got:", columns);
        return { candles: [], volumes: [] };
    }

    const candles = [];
    const volumes = [];

    for (let r = 0; r < data.length; r++) {
        const row = data[r];
        const ts = index[r];
        if (!ts && ts !== 0) continue;

        // Timestamp → Unix ms
        const ms =
            typeof ts === "number"
                ? ts > 1e12
                    ? ts
                    : ts * 1000
                : Date.parse(ts);
        if (!isFinite(ms)) continue;

        const o = +row[iO],
            h = +row[iH],
            l = +row[iL],
            c = +row[iC];
        if (!isFinite(o) || !isFinite(h) || !isFinite(l) || !isFinite(c))
            continue;
        if (h < l) continue; // guard corrupt rows

        candles.push({ x: ms, y: [o, h, l, c] });

        if (iV >= 0) {
            const v = +row[iV];
            volumes.push({
                x: ms,
                y: isFinite(v) ? v : 0,
                fillColor: c >= o ? C.volUp : C.volDown,
            });
        }
    }

    // Sort ascending (API may return newest-first)
    candles.sort((a, b) => a.x - b.x);
    volumes.sort((a, b) => a.x - b.x);

    // De-duplicate on timestamp — keep last occurrence
    const dedup = (arr) => {
        const map = new Map();
        for (const pt of arr) map.set(pt.x, pt);
        return Array.from(map.values());
    };

    return { candles: dedup(candles), volumes: dedup(volumes) };
}

// ── Indicator math ─────────────────────────────────────────────────────────────
//
// All functions accept a candleData array [{x, y:[o,h,l,c]}] (and optionally
// a volumeData array for VWAP) and return [{x, y}] series arrays.

/**
 * Exponential Moving Average
 * @param {Array}  candles   — [{x, y:[o,h,l,c]}]
 * @param {number} period
 * @returns {Array} [{x, y}]
 */
function calcEMA(candles, period) {
    if (candles.length < period) return [];
    const k = 2 / (period + 1);
    const out = [];
    let ema = null;

    for (let i = 0; i < candles.length; i++) {
        const close = candles[i].y[3];
        if (ema === null) {
            // Seed with a simple average over the first `period` bars
            if (i < period - 1) continue;
            let sum = 0;
            for (let j = i - period + 1; j <= i; j++) sum += candles[j].y[3];
            ema = sum / period;
        } else {
            ema = close * k + ema * (1 - k);
        }
        out.push({ x: candles[i].x, y: +ema.toFixed(6) });
    }
    return out;
}

/**
 * Bollinger Bands  (20-period, σ multiplier configurable, population std dev)
 * @param {Array}  candles
 * @param {number} period   default 20
 * @param {number} mult     default 2
 * @returns {{ upper, mid, lower }}  each [{x, y}]
 */
function calcBollingerBands(candles, period = 20, mult = 2) {
    if (candles.length < period) return { upper: [], mid: [], lower: [] };

    const upper = [],
        mid = [],
        lower = [];

    for (let i = period - 1; i < candles.length; i++) {
        // Simple mean
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += candles[j].y[3];
        const mean = sum / period;

        // Population std dev
        let sqDiff = 0;
        for (let j = i - period + 1; j <= i; j++) {
            const d = candles[j].y[3] - mean;
            sqDiff += d * d;
        }
        const sigma = Math.sqrt(sqDiff / period);

        const x = candles[i].x;
        upper.push({ x, y: +(mean + mult * sigma).toFixed(6) });
        mid.push({ x, y: +mean.toFixed(6) });
        lower.push({ x, y: +(mean - mult * sigma).toFixed(6) });
    }
    return { upper, mid, lower };
}

/**
 * Volume-Weighted Average Price — resets at the start of each session/day.
 * Since we don't know session boundaries from bar data alone, we reset VWAP
 * whenever a candle's date differs from the previous candle's date
 * (midnight-to-midnight daily reset — works well for intraday intervals;
 * for daily bars it just produces a running VWAP across all bars instead).
 *
 * @param {Array} candles   [{x, y:[o,h,l,c]}]
 * @param {Array} volumes   [{x, y}]
 * @returns {Array} [{x, y}]
 */
function calcVWAP(candles, volumes) {
    if (candles.length === 0) return [];

    // Build a quick lookup: timestamp → volume
    const volMap = new Map();
    for (const v of volumes) volMap.set(v.x, v.y);

    const out = [];
    let cumTPV = 0; // cumulative (typical price × volume)
    let cumVol = 0; // cumulative volume
    let lastDay = null;

    for (let i = 0; i < candles.length; i++) {
        const {
            x,
            y: [o, h, l, c],
        } = candles[i];
        const day = new Date(x).toDateString();

        if (day !== lastDay) {
            // Reset accumulators at the start of each new day
            cumTPV = 0;
            cumVol = 0;
            lastDay = day;
        }

        const tp = (h + l + c) / 3;
        const vol = volMap.get(x) ?? 0;

        cumTPV += tp * vol;
        cumVol += vol;

        const vwap = cumVol > 0 ? cumTPV / cumVol : tp;
        out.push({ x, y: +vwap.toFixed(6) });
    }
    return out;
}

/**
 * Relative Strength Index (Wilder smoothing)
 * @param {Array}  candles
 * @param {number} period  default 14
 * @returns {Array} [{x, y}]
 */
function calcRSI(candles, period = 14) {
    if (candles.length < period + 1) return [];

    const out = [];
    let avgGain = 0,
        avgLoss = 0;

    // Seed with simple average over first `period` changes
    for (let i = 1; i <= period; i++) {
        const diff = candles[i].y[3] - candles[i - 1].y[3];
        if (diff >= 0) avgGain += diff;
        else avgLoss -= diff;
    }
    avgGain /= period;
    avgLoss /= period;

    const rs0 = avgLoss === 0 ? Infinity : avgGain / avgLoss;
    const rsi0 = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs0);
    out.push({ x: candles[period].x, y: +rsi0.toFixed(2) });

    // Wilder smoothing for subsequent bars
    for (let i = period + 1; i < candles.length; i++) {
        const diff = candles[i].y[3] - candles[i - 1].y[3];
        const gain = diff > 0 ? diff : 0;
        const loss = diff < 0 ? -diff : 0;

        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;

        const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
        const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);
        out.push({ x: candles[i].x, y: +rsi.toFixed(2) });
    }
    return out;
}

/**
 * Recalculate ALL enabled indicators from current state.candleData / volumeData
 * and store results back into state.
 */
function recalcIndicators() {
    const c = state.candleData;
    const v = state.volumeData;

    state.ema9Data = state.indicators.ema9 ? calcEMA(c, 9) : [];
    state.ema21Data = state.indicators.ema21 ? calcEMA(c, 21) : [];

    const bb = state.indicators.bb
        ? calcBollingerBands(c, 20, 2)
        : { upper: [], mid: [], lower: [] };
    state.bbUpperData = bb.upper;
    state.bbMidData = bb.mid;
    state.bbLowerData = bb.lower;

    state.vwapData = state.indicators.vwap ? calcVWAP(c, v) : [];
    state.rsiData = state.indicators.rsi ? calcRSI(c, 14) : [];
}

// ── Timestamp format string for ApexCharts xaxis.labels ───────────────────────
function xAxisFormat(interval) {
    if (interval === "1d") return "dd MMM";
    if (interval === "1h") return "dd MMM HH:mm";
    return "HH:mm";
}

// ── Build PRIMARY chart options ────────────────────────────────────────────────
//
// Always 8 series slots (indices 0-7). Inactive indicator series carry empty
// data arrays — ApexCharts hides a line series with no data points gracefully.
// This lets us do chart.updateSeries() for live updates without destroying and
// recreating the chart instance.
//
// Series layout:
//   [0] candlestick  — OHLC
//   [1] bar          — Volume
//   [2] line         — EMA9
//   [3] line         — EMA21
//   [4] line         — BB Upper
//   [5] line         — BB Mid
//   [6] line         — BB Lower
//   [7] line         — VWAP

function buildSeries() {
    const ind = state.indicators;
    return [
        {
            name: state.activeName || "Price",
            type: "candlestick",
            data: state.candleData,
        },
        {
            name: "Volume",
            type: "bar",
            data: state.volumeData,
        },
        {
            name: "EMA 9",
            type: "line",
            data: ind.ema9 ? state.ema9Data : [],
        },
        {
            name: "EMA 21",
            type: "line",
            data: ind.ema21 ? state.ema21Data : [],
        },
        {
            name: "BB Upper",
            type: "line",
            data: ind.bb ? state.bbUpperData : [],
        },
        {
            name: "BB Mid",
            type: "line",
            data: ind.bb ? state.bbMidData : [],
        },
        {
            name: "BB Lower",
            type: "line",
            data: ind.bb ? state.bbLowerData : [],
        },
        {
            name: "VWAP",
            type: "line",
            data: ind.vwap ? state.vwapData : [],
        },
    ];
}

function buildOptions(title, interval) {
    const hasVolume = state.volumeData.length > 0;

    // ── Custom OHLC tooltip ────────────────────────────────────────────────
    const tooltipCustom = ({ seriesIndex, dataPointIndex, w }) => {
        if (seriesIndex !== 0) return "";
        const d = w.globals.initialSeries[0]?.data?.[dataPointIndex];
        if (!d) return "";

        const [o, h, l, c] = d.y;
        const isUp = c >= o;
        const color = isUp ? C.green : C.red;
        const chg = c - o;
        const pct = o !== 0 ? ((chg / o) * 100).toFixed(2) : "0.00";
        const sign = chg >= 0 ? "+" : "";

        const vp = state.volumeData[dataPointIndex];
        const volStr = vp ? fmtVolume(vp.y) : "—";
        const time = fmtTs(d.x);

        const row = (label, val, clr) => `
            <div style="display:flex;justify-content:space-between;gap:18px">
                <span style="color:${C.textSecond}">${label}</span>
                <span style="color:${clr ?? C.textPrimary};font-weight:600">${val}</span>
            </div>`;

        // Indicator values at this bar
        const indRows = [];
        const indLookup = (arr, label, colour) => {
            const pt =
                arr[dataPointIndex - (state.candleData.length - arr.length)];
            if (pt) indRows.push(row(label, fmtPrice(pt.y), colour));
        };
        if (state.indicators.ema9 && state.ema9Data.length)
            indLookup(state.ema9Data, "EMA9", C.ema9);
        if (state.indicators.ema21 && state.ema21Data.length)
            indLookup(state.ema21Data, "EMA21", C.ema21);
        if (state.indicators.vwap && state.vwapData.length)
            indLookup(state.vwapData, "VWAP", C.vwap);

        return `
            <div style="
                padding:8px 12px;
                font-family:${C.fontMono};
                font-size:11px;
                line-height:1.85;
                color:${C.textPrimary};
                min-width:168px;
            ">
                <div style="color:${C.textMuted};font-size:10px;margin-bottom:4px">${time}</div>
                ${row("O", fmtPrice(o), color)}
                ${row("H", fmtPrice(h), color)}
                ${row("L", fmtPrice(l), color)}
                ${row("C", fmtPrice(c), color)}
                <div style="border-top:1px solid ${C.border};margin:3px 0 2px"></div>
                ${row("Chg", `${sign}${fmtPrice(chg)} (${sign}${pct}%)`, color)}
                ${row("Vol", volStr)}
                ${indRows.join("")}
            </div>`;
    };

    // ── Y-axes — one per series slot to prevent axis bleed ────────────────
    // yaxis[0]  — price, left, visible
    // yaxis[1]  — volume, right, visible (scaled so bars use bottom 20%)
    // yaxis[2+] — indicator overlays: hidden but real so ApexCharts doesn't
    //             auto-scale the price axis to include indicator line values
    const priceYaxis = {
        seriesName: state.activeName || "Price",
        tooltip: { enabled: true },
        tickAmount: 6,
        labels: {
            formatter: (v) => fmtPrice(v),
            style: {
                colors: C.textMuted,
                fontSize: "10px",
                fontFamily: C.fontMono,
            },
        },
    };

    const volumeYaxis = {
        seriesName: "Volume",
        opposite: true,
        show: hasVolume,
        max: (max) => max * 5,
        min: 0,
        tickAmount: 3,
        labels: {
            formatter: (v) => fmtVolume(v),
            style: {
                colors: C.textMuted,
                fontSize: "9px",
                fontFamily: C.fontMono,
            },
        },
    };

    // Hidden y-axis shared by all overlay indicator series
    // — they use the same price scale as the candles, so we point them at
    //   the same seriesName as the price axis. ApexCharts treats seriesName
    //   as a linkage key; pointing all overlays at the price series name
    //   makes them scale correctly without polluting the axis labels.
    const overlayYaxis = {
        seriesName: state.activeName || "Price",
        show: false,
        labels: { formatter: (v) => fmtPrice(v) },
    };

    return {
        chart: {
            type: "candlestick",
            height: "100%",
            background: C.bg,
            foreColor: C.textSecond,
            fontFamily: C.fontMono,
            animations: {
                enabled: true,
                easing: "linear",
                speed: 200,
                animateGradually: { enabled: false },
                dynamicAnimation: { enabled: true, speed: 200 },
            },
            toolbar: {
                show: true,
                autoSelected: "zoom",
                tools: {
                    download: true,
                    selection: false,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true,
                },
            },
            zoom: { enabled: true, type: "x" },
            selection: { enabled: false },
        },

        theme: { mode: "dark" },

        title: {
            text: title || "",
            align: "left",
            offsetX: 12,
            offsetY: 4,
            style: {
                color: C.textSecond,
                fontSize: "11px",
                fontFamily: C.fontMono,
                fontWeight: 500,
            },
        },

        series: buildSeries(),

        plotOptions: {
            candlestick: {
                colors: { upward: C.green, downward: C.red },
                wick: { useFillColor: true },
            },
            bar: {
                columnWidth: "90%",
                distributed: true,
                borderRadius: 1,
                borderRadiusApplication: "end",
            },
        },

        dataLabels: { enabled: false },

        stroke: {
            // series: [candlestick, bar, EMA9, EMA21, BBU, BBM, BBL, VWAP]
            curve: [
                "smooth",
                "smooth",
                "smooth",
                "smooth",
                "smooth",
                "smooth",
                "smooth",
                "smooth",
            ],
            width: [1, 1, 1.5, 1.5, 1, 1, 1, 1.5],
            dashArray: [0, 0, 0, 0, 4, 2, 4, 3],
        },

        colors: [
            C.green, // candlestick — overridden by plotOptions.candlestick.colors
            C.volUp, // volume      — overridden by per-point fillColor
            C.ema9, // EMA9
            C.ema21, // EMA21
            C.bbUpper, // BB Upper
            C.bbMid, // BB Mid
            C.bbLower, // BB Lower
            C.vwap, // VWAP
        ],

        xaxis: {
            type: "datetime",
            tickAmount: 10,
            labels: {
                datetimeUTC: false,
                format: xAxisFormat(interval),
                style: {
                    colors: C.textMuted,
                    fontSize: "10px",
                    fontFamily: C.fontMono,
                },
            },
            axisBorder: { color: C.border },
            axisTicks: { color: C.border },
            crosshairs: {
                show: true,
                stroke: { color: C.textMuted, width: 1, dashArray: 3 },
            },
            tooltip: { enabled: false },
        },

        // y-axes: price, volume, then one hidden overlay axis per indicator series
        yaxis: [
            priceYaxis,
            volumeYaxis,
            overlayYaxis, // EMA9
            overlayYaxis, // EMA21
            overlayYaxis, // BB Upper
            overlayYaxis, // BB Mid
            overlayYaxis, // BB Lower
            overlayYaxis, // VWAP
        ],

        grid: {
            borderColor: C.borderSubtle,
            strokeDashArray: 3,
            xaxis: { lines: { show: false } },
            yaxis: { lines: { show: true } },
            padding: { top: 4, right: 16, bottom: 0, left: 4 },
        },

        tooltip: {
            enabled: true,
            theme: "dark",
            shared: false,
            intersect: false,
            custom: tooltipCustom,
        },

        legend: { show: false },

        noData: {
            text: "No data",
            style: {
                color: C.textMuted,
                fontFamily: C.fontMono,
                fontSize: "12px",
            },
        },
    };
}

// ── RSI sub-pane options ───────────────────────────────────────────────────────
function buildRsiOptions() {
    return {
        chart: {
            id: "rsi-pane",
            type: "line",
            height: "100%",
            background: C.bg,
            foreColor: C.textSecond,
            fontFamily: C.fontMono,
            animations: {
                enabled: true,
                dynamicAnimation: { enabled: true, speed: 200 },
            },
            toolbar: { show: false },
            zoom: { enabled: false },
            sparkline: { enabled: false },
        },

        theme: { mode: "dark" },

        series: [{ name: "RSI 14", data: state.rsiData }],

        stroke: { curve: "smooth", width: 1.5 },
        colors: [C.rsiLine],
        dataLabels: { enabled: false },

        xaxis: {
            type: "datetime",
            labels: { show: false },
            axisBorder: { color: C.border },
            axisTicks: { show: false },
            crosshairs: {
                show: true,
                stroke: { color: C.textMuted, width: 1, dashArray: 3 },
            },
            tooltip: { enabled: false },
        },

        yaxis: {
            min: 0,
            max: 100,
            tickAmount: 4,
            labels: {
                formatter: (v) => v.toFixed(0),
                style: {
                    colors: C.textMuted,
                    fontSize: "10px",
                    fontFamily: C.fontMono,
                },
            },
        },

        // Overbought / oversold reference bands
        annotations: {
            yaxis: [
                {
                    y: 70,
                    borderColor: C.red,
                    borderWidth: 1,
                    strokeDashArray: 4,
                    label: {
                        text: "OB 70",
                        style: {
                            background: "transparent",
                            color: C.red,
                            fontSize: "9px",
                            fontFamily: C.fontMono,
                        },
                        position: "right",
                        offsetX: -4,
                    },
                },
                {
                    y: 30,
                    borderColor: C.green,
                    borderWidth: 1,
                    strokeDashArray: 4,
                    label: {
                        text: "OS 30",
                        style: {
                            background: "transparent",
                            color: C.green,
                            fontSize: "9px",
                            fontFamily: C.fontMono,
                        },
                        position: "right",
                        offsetX: -4,
                    },
                },
                {
                    y: 50,
                    borderColor: C.textMuted,
                    borderWidth: 1,
                    strokeDashArray: 2,
                    label: { text: "" },
                },
            ],
        },

        fill: {
            type: "gradient",
            gradient: {
                shade: "dark",
                shadeIntensity: 0.3,
                opacityFrom: 0.4,
                opacityTo: 0.05,
                stops: [0, 100],
            },
        },

        grid: {
            borderColor: C.borderSubtle,
            strokeDashArray: 3,
            padding: { top: 2, right: 16, bottom: 0, left: 4 },
        },

        tooltip: {
            enabled: true,
            theme: "dark",
            shared: false,
            intersect: false,
            custom: ({ dataPointIndex, w }) => {
                const pt = w.globals.initialSeries[0]?.data?.[dataPointIndex];
                if (!pt) return "";
                const rsiVal = pt.y;
                const color =
                    rsiVal >= 70 ? C.red : rsiVal <= 30 ? C.green : C.rsiLine;
                return `
                    <div style="padding:6px 10px;font-family:${C.fontMono};font-size:11px;color:${C.textPrimary}">
                        <span style="color:${C.textMuted};font-size:10px">${fmtTs(pt.x)}</span><br/>
                        <span style="color:${color};font-weight:600">RSI&nbsp;${rsiVal.toFixed(2)}</span>
                    </div>`;
            },
        },

        legend: { show: false },
    };
}

// ── Chart lifecycle ────────────────────────────────────────────────────────────
function destroyChart() {
    if (state.chart) {
        state.chart.destroy();
        state.chart = null;
    }
}

function destroyRsiChart() {
    if (state.chartRsi) {
        state.chartRsi.destroy();
        state.chartRsi = null;
    }
}

function mountChart(title, interval) {
    destroyChart();
    state.chart = new ApexCharts(dom.chartEl, buildOptions(title, interval));
    state.chart.render();
}

function mountRsiChart() {
    destroyRsiChart();
    state.chartRsi = new ApexCharts(dom.chartRsiEl, buildRsiOptions());
    state.chartRsi.render();
    dom.chartRsiEl.classList.remove("hidden");
}

function unmountRsiChart() {
    destroyRsiChart();
    dom.chartRsiEl.classList.add("hidden");
}

/**
 * Sync visibility of the RSI pane based on state.indicators.rsi.
 * Called after any indicator toggle or after bars load.
 */
function syncRsiPane() {
    if (state.indicators.rsi && state.rsiData.length > 0) {
        if (!state.chartRsi) {
            mountRsiChart();
        } else {
            state.chartRsi.updateSeries(
                [{ name: "RSI 14", data: state.rsiData }],
                false,
            );
        }
    } else {
        unmountRsiChart();
    }
}

// ── Asset discovery ────────────────────────────────────────────────────────────
async function loadAssets() {
    try {
        const res = await fetch(`${state.dataServiceUrl}/bars/assets`, {
            headers: buildHeaders(),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();

        state.assets = json.assets || [];
        renderSymbolTabs();

        // Default symbol selection — respect ?symbol= param, then prefer Gold
        let initial = null;
        const qsSymbol = new URLSearchParams(window.location.search).get(
            "symbol",
        );
        if (qsSymbol) {
            initial =
                state.assets.find(
                    (a) => a.name === qsSymbol || a.ticker === qsSymbol,
                ) || null;
        }
        if (!initial) {
            const preferred = ["Gold", "S&P", "Nasdaq", "Crude Oil"];
            initial =
                state.assets.find(
                    (a) => preferred.includes(a.name) && a.has_data,
                ) ||
                state.assets.find((a) => a.has_data) ||
                state.assets[0] ||
                null;
        }

        if (initial) await selectSymbol(initial.ticker, initial.name);
        setStatus("ok", "Connected");
    } catch (err) {
        setStatus("error", "Data service unreachable");
        showError(`Failed to load assets: ${err.message}`);
        hideOverlay();
    }
}

function renderSymbolTabs() {
    dom.symbolTabs.innerHTML = "";
    for (const asset of state.assets) {
        const btn = document.createElement("button");
        btn.className = "symbol-btn";
        btn.dataset.ticker = asset.ticker;
        btn.dataset.name = asset.name;
        btn.setAttribute("role", "tab");
        btn.title = `${asset.name} (${asset.ticker}) — ${(asset.bar_count ?? 0).toLocaleString()} bars`;
        btn.innerHTML = `<span class="symbol-name">${asset.name}</span><span class="symbol-ticker">${asset.ticker}</span>`;
        if (!asset.has_data) btn.style.opacity = "0.4";
        btn.addEventListener("click", () =>
            selectSymbol(asset.ticker, asset.name),
        );
        dom.symbolTabs.appendChild(btn);
    }
}

function setActiveSymbolTab(ticker) {
    dom.symbolTabs.querySelectorAll(".symbol-btn").forEach((btn) => {
        const active = btn.dataset.ticker === ticker;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
    });
}

// ── Bar fetch ──────────────────────────────────────────────────────────────────
function buildHeaders() {
    const h = { Accept: "application/json" };
    if (window.DATA_API_KEY) h["X-API-Key"] = window.DATA_API_KEY;
    return h;
}

async function loadBars(ticker, interval, daysBack) {
    if (state.loadAbortCtrl) state.loadAbortCtrl.abort();
    state.loadAbortCtrl = new AbortController();

    const url =
        `${state.dataServiceUrl}/bars/${encodeURIComponent(ticker)}` +
        `?interval=${interval}&days_back=${daysBack}&auto_fill=false`;

    const res = await fetch(url, {
        headers: buildHeaders(),
        signal: state.loadAbortCtrl.signal,
    });

    if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} — ${body.slice(0, 120)}`);
    }
    return res.json();
}

async function selectSymbol(ticker, name) {
    if (state.activeTicker === ticker && state.chart) return;
    state.activeTicker = ticker;
    state.activeName = name || ticker;

    setActiveSymbolTab(ticker);
    sseDisconnect();
    showOverlay(`Loading ${state.activeName}…`);
    hideError();

    await renderBars();
}

async function renderBars() {
    const { activeTicker, activeInterval, activeDays, activeName } = state;
    if (!activeTicker) return;

    showOverlay(`Loading ${activeName}…`);
    hideError();

    try {
        const json = await loadBars(activeTicker, activeInterval, activeDays);
        const { candles, volumes } = splitToApex(json.data);

        if (candles.length === 0) {
            showError(`No bars stored for ${activeName} at ${activeInterval}`);
            hideOverlay();
            updateStatusBar([]);
            return;
        }

        state.candleData = candles;
        state.volumeData = volumes;

        // Calculate all enabled indicators from fresh data
        recalcIndicators();
        // Seed incremental EMA/RSI state for live bar updates
        seedLiveIndState();

        const title = `${activeName}  ·  ${activeInterval}`;
        mountChart(title, activeInterval);
        syncRsiPane();
        updateStatusBar(candles);
        hideOverlay();

        if (state.liveEnabled) sseConnect();
    } catch (err) {
        if (err.name === "AbortError") return;
        console.error("[chart] loadBars error:", err);
        showError(`Failed to load bars: ${err.message}`);
        hideOverlay();
        setStatus("error", "Load error");
    }
}

// ── Status bar ─────────────────────────────────────────────────────────────────
function updateStatusBar(candles) {
    const n = candles.length;
    dom.barCount.textContent = `${n.toLocaleString()} bars`;

    if (n === 0) {
        dom.lastPrice.textContent = "—";
        dom.lastChange.textContent = "—";
        dom.lastTs.textContent = "—";
        return;
    }

    const last = candles[n - 1];
    const prev = candles[n - 2];
    const c = last.y[3];

    dom.lastPrice.textContent = fmtPrice(c);

    if (prev) {
        const pc = prev.y[3];
        const chg = c - pc;
        const pct = pc !== 0 ? (chg / pc) * 100 : 0;
        const sign = chg >= 0 ? "+" : "";
        dom.lastChange.textContent = `${sign}${fmtPrice(chg)} (${sign}${pct.toFixed(2)}%)`;
        dom.lastChange.className =
            "stat " + (chg >= 0 ? "positive" : "negative");
    }

    dom.lastTs.textContent = fmtTs(last.x);
}

// ── SSE — live bar updates ─────────────────────────────────────────────────────
function sseConnect() {
    if (!state.liveEnabled) return;
    sseDisconnect();
    setSseBadge("reconnecting");

    try {
        const source = new EventSource(`${state.dataServiceUrl}/sse/dashboard`);
        state.sseSource = source;

        source.onopen = () => {
            setSseBadge("live");
            state.sseReconnectMs = SSE_RECONNECT_BASE;
        };

        source.onmessage = (e) => {
            try {
                handleSsePayload(JSON.parse(e.data));
            } catch (_) {}
        };

        source.addEventListener("bars_update", (e) => {
            try {
                handleSsePayload(JSON.parse(e.data));
            } catch (_) {}
        });

        source.addEventListener("heartbeat", () => {});

        source.onerror = () => {
            setSseBadge("reconnecting");
            source.close();
            state.sseSource = null;
            scheduleSSEReconnect();
        };
    } catch (err) {
        console.error("[chart] EventSource failed:", err);
        setSseBadge("error");
    }
}

function sseDisconnect() {
    if (state.sseReconnectTimer) {
        clearTimeout(state.sseReconnectTimer);
        state.sseReconnectTimer = null;
    }
    if (state.sseSource) {
        state.sseSource.close();
        state.sseSource = null;
    }
    setSseBadge("");
}

function scheduleSSEReconnect() {
    state.sseReconnectTimer = setTimeout(() => {
        state.sseReconnectTimer = null;
        if (state.liveEnabled && state.activeTicker) sseConnect();
    }, state.sseReconnectMs);

    state.sseReconnectMs = Math.min(
        state.sseReconnectMs * SSE_RECONNECT_MULT,
        SSE_RECONNECT_MAX,
    );
}

// Supported SSE payload shapes:
//   Shape A: { event:"bars_update", ticker:"MGC=F", bar:{timestamp,open,high,low,close,volume} }
//   Shape B: { type:"bar",          symbol:"MGC=F", data:{...} }
//   Shape C: { focus:{ ticker:"MGC=F", latest_bar:{...} } }
function handleSsePayload(payload) {
    if (!payload || typeof payload !== "object") return;

    if (payload.event === "bars_update" || payload.type === "bar") {
        const ticker = payload.ticker || payload.symbol;
        const raw = payload.bar || payload.data;
        if (ticker === state.activeTicker && raw) applyLiveBar(raw);
        return;
    }

    if (payload.focus) {
        const { ticker, latest_bar } = payload.focus;
        if (ticker === state.activeTicker && latest_bar)
            applyLiveBar(latest_bar);
    }
}

// ── In-place forming-candle update ────────────────────────────────────────────
//
// Same timestamp  → mutate the last candle in-place (forming candle).
// New timestamp   → push a new candle, prune if over MAX_CANDLES.
//
// After updating OHLCV data, re-append one indicator data point to the end of
// each enabled indicator series (EMA / BB / VWAP / RSI) using an incremental
// update — avoids recalculating from scratch on every tick.
//
// For EMA: the last value is updated using EMA(close, k, prevEMA).
// For BB:  recalc only the last window (rolling 20 bars).
// For VWAP: continue cumulative accumulators (state preserves them).
// For RSI:  update last avgGain/avgLoss with Wilder smoothing.
//
// If it's a brand-new candle, we push a new indicator point.
// If it's a forming-candle update, we replace the last indicator point.

// Incremental state for live indicator updates — populated on first render
// and kept in sync as bars arrive.
const liveInd = {
    emaK9: 2 / (9 + 1),
    emaK21: 2 / (21 + 1),
    ema9Last: null, // last computed EMA9 value
    ema21Last: null, // last computed EMA21 value

    // RSI Wilder state
    rsiAvgGain: null,
    rsiAvgLoss: null,
    rsiPeriod: 14,
};

/**
 * Seed liveInd from the fully-computed indicator arrays after a fresh bar load.
 */
function seedLiveIndState() {
    if (state.ema9Data.length > 0)
        liveInd.ema9Last = state.ema9Data[state.ema9Data.length - 1].y;
    if (state.ema21Data.length > 0)
        liveInd.ema21Last = state.ema21Data[state.ema21Data.length - 1].y;

    // RSI: seed avgGain/avgLoss by re-running Wilder from the end of the series
    // We just need the final state, so run the last `period` diffs.
    const c = state.candleData;
    const p = liveInd.rsiPeriod;
    if (c.length >= p + 1) {
        let ag = 0,
            al = 0;
        for (let i = 1; i <= p; i++) {
            const d = c[i].y[3] - c[i - 1].y[3];
            if (d >= 0) ag += d;
            else al -= d;
        }
        ag /= p;
        al /= p;

        for (let i = p + 1; i < c.length; i++) {
            const d = c[i].y[3] - c[i - 1].y[3];
            const gain = d > 0 ? d : 0;
            const loss = d < 0 ? -d : 0;
            ag = (ag * (p - 1) + gain) / p;
            al = (al * (p - 1) + loss) / p;
        }
        liveInd.rsiAvgGain = ag;
        liveInd.rsiAvgLoss = al;
    }
}

/**
 * Incrementally extend EMA9 / EMA21 / BB / VWAP / RSI for a new or updated close.
 * @param {boolean} isNewCandle  — true if we need to push a new point; false = replace last
 * @param {number}  ms           — bar timestamp in Unix ms
 * @param {number}  prevClose    — close of the previous finalized candle
 * @param {number}  close        — current close (may be forming)
 */
function updateIndicatorPoint(isNewCandle, ms, prevClose, close) {
    const push_or_replace = (arr, x, y) => {
        if (y == null || !isFinite(y)) return;
        const pt = { x, y: +y.toFixed(6) };
        if (isNewCandle || arr.length === 0) {
            arr.push(pt);
            if (arr.length > MAX_CANDLES)
                arr.splice(0, arr.length - MAX_CANDLES);
        } else {
            arr[arr.length - 1] = pt;
        }
    };

    // EMA9
    if (state.indicators.ema9 && state.ema9Data.length > 0) {
        const prev = liveInd.ema9Last;
        if (prev != null) {
            const next = close * liveInd.emaK9 + prev * (1 - liveInd.emaK9);
            push_or_replace(state.ema9Data, ms, next);
            liveInd.ema9Last = next;
        }
    }

    // EMA21
    if (state.indicators.ema21 && state.ema21Data.length > 0) {
        const prev = liveInd.ema21Last;
        if (prev != null) {
            const next = close * liveInd.emaK21 + prev * (1 - liveInd.emaK21);
            push_or_replace(state.ema21Data, ms, next);
            liveInd.ema21Last = next;
        }
    }

    // Bollinger Bands — recalc over the last 20 bars (cheap enough)
    if (state.indicators.bb && state.candleData.length >= 20) {
        const period = 20;
        const end = state.candleData.length - 1;
        const start = Math.max(0, end - period + 1);
        let sum = 0;
        for (let i = start; i <= end; i++) sum += state.candleData[i].y[3];
        const mean = sum / (end - start + 1);
        let sqDiff = 0;
        for (let i = start; i <= end; i++) {
            const d = state.candleData[i].y[3] - mean;
            sqDiff += d * d;
        }
        const sigma = Math.sqrt(sqDiff / (end - start + 1));
        push_or_replace(state.bbUpperData, ms, mean + 2 * sigma);
        push_or_replace(state.bbMidData, ms, mean);
        push_or_replace(state.bbLowerData, ms, mean - 2 * sigma);
    }

    // VWAP — recalc the current session (day) from scratch each tick.
    // Scan backward to find the first bar of today, then accumulate forward.
    // This is O(session_bars) — typically <500 bars — fast enough for live ticks.
    if (state.indicators.vwap && state.candleData.length > 0) {
        const volMap = new Map();
        for (const v of state.volumeData) volMap.set(v.x, v.y);

        const today = new Date(ms).toDateString();

        // Find start-of-day index (scan backward until we leave today)
        let startOfDay = state.candleData.length - 1;
        while (
            startOfDay > 0 &&
            new Date(state.candleData[startOfDay - 1].x).toDateString() ===
                today
        ) {
            startOfDay--;
        }

        // Accumulate typical-price × volume forward from startOfDay
        let cumTPV = 0,
            cumVol = 0;
        for (let j = startOfDay; j < state.candleData.length; j++) {
            const {
                x,
                y: [, h, l, c2],
            } = state.candleData[j];
            const tp = (h + l + c2) / 3;
            const vol = volMap.get(x) ?? 0;
            cumTPV += tp * vol;
            cumVol += vol;
        }
        const vwapVal = cumVol > 0 ? cumTPV / cumVol : close;
        push_or_replace(state.vwapData, ms, vwapVal);
    }

    // RSI
    if (state.indicators.rsi && state.rsiData.length > 0 && prevClose != null) {
        const p = liveInd.rsiPeriod;
        const ag = liveInd.rsiAvgGain;
        const al = liveInd.rsiAvgLoss;
        if (ag != null && al != null) {
            const diff = close - prevClose;
            const gain = diff > 0 ? diff : 0;
            const loss = diff < 0 ? -diff : 0;
            const newAG = (ag * (p - 1) + gain) / p;
            const newAL = (al * (p - 1) + loss) / p;
            const rs = newAL === 0 ? Infinity : newAG / newAL;
            const rsi = newAL === 0 ? 100 : 100 - 100 / (1 + rs);
            push_or_replace(state.rsiData, ms, rsi);

            if (isNewCandle) {
                liveInd.rsiAvgGain = newAG;
                liveInd.rsiAvgLoss = newAL;
            }
        }
    }
}

function applyLiveBar(raw) {
    if (!raw || !state.chart) return;

    // Normalise timestamp to Unix ms
    const tsRaw = raw.timestamp ?? raw.time ?? raw.ts;
    if (tsRaw == null) return;
    const ms =
        typeof tsRaw === "number"
            ? tsRaw > 1e12
                ? tsRaw
                : tsRaw * 1000
            : Date.parse(tsRaw);
    if (!isFinite(ms)) return;

    const o = +raw.open,
        h = +raw.high,
        l = +raw.low,
        c = +raw.close;
    if (!isFinite(o) || !isFinite(h) || !isFinite(l) || !isFinite(c)) return;

    // ── Candle data ──────────────────────────────────────────────────────────
    const lastCandle = state.candleData[state.candleData.length - 1];
    const isNewCandle = !lastCandle || lastCandle.x !== ms;
    const prevClose = lastCandle ? lastCandle.y[3] : null;

    if (!isNewCandle) {
        // Forming candle — merge OHLC in place
        lastCandle.y[1] = Math.max(lastCandle.y[1], h);
        lastCandle.y[2] = Math.min(lastCandle.y[2], l);
        lastCandle.y[3] = c;
    } else {
        state.candleData.push({ x: ms, y: [o, h, l, c] });
        if (state.candleData.length > MAX_CANDLES) {
            state.candleData.splice(0, state.candleData.length - MAX_CANDLES);
        }
    }

    // ── Volume data ──────────────────────────────────────────────────────────
    if (raw.volume != null) {
        const v = +raw.volume;
        const fillColor = c >= o ? C.volUp : C.volDown;
        const lastVol = state.volumeData[state.volumeData.length - 1];

        if (lastVol && lastVol.x === ms) {
            lastVol.y = isFinite(v) ? v : 0;
            lastVol.fillColor = fillColor;
        } else {
            state.volumeData.push({ x: ms, y: isFinite(v) ? v : 0, fillColor });
            if (state.volumeData.length > MAX_CANDLES) {
                state.volumeData.splice(
                    0,
                    state.volumeData.length - MAX_CANDLES,
                );
            }
        }
    }

    // ── Indicator incremental update ─────────────────────────────────────────
    updateIndicatorPoint(isNewCandle, ms, prevClose, c);

    // ── Push to primary chart — no animation on micro-updates ────────────────
    state.chart.updateSeries(buildSeries(), false);

    // ── Push to RSI sub-pane if mounted ──────────────────────────────────────
    if (state.chartRsi && state.indicators.rsi && state.rsiData.length > 0) {
        state.chartRsi.updateSeries(
            [{ name: "RSI 14", data: state.rsiData }],
            false,
        );
    }

    // ── Footer status bar ────────────────────────────────────────────────────
    dom.lastPrice.textContent = fmtPrice(c);
    dom.lastTs.textContent = fmtTs(ms);
}

// ── DATA_SERVICE_URL resolution ────────────────────────────────────────────────
//
// Priority:
//   1. window.DATA_SERVICE_URL   — injected by the embedding page
//   2. ?data_url= query param    — useful when iframed from any host
//   3. GET /config               — nginx renders {data_service_url, port}
//      → if the URL contains a docker-internal hostname, substitute real hostname
//   4. Same host, port 8001      — last-resort fallback

async function resolveDataServiceUrl() {
    if (window.DATA_SERVICE_URL)
        return window.DATA_SERVICE_URL.replace(/\/$/, "");

    const qs = new URLSearchParams(window.location.search);
    if (qs.get("data_url")) return qs.get("data_url").replace(/\/$/, "");

    try {
        const res = await fetch("/config", { cache: "no-store" });
        if (res.ok) {
            const { data_service_url } = await res.json();
            if (data_service_url) {
                const internalRe =
                    /^https?:\/\/(data|localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)/;
                if (!internalRe.test(data_service_url)) {
                    return data_service_url.replace(/\/$/, "");
                }
                const u = new URL(data_service_url);
                return `${window.location.protocol}//${window.location.hostname}:${u.port || 8001}`;
            }
        }
    } catch (_) {
        // /config unreachable — fall through to fallback
    }

    return `${window.location.protocol}//${window.location.hostname}:8001`;
}

// ── Control wiring ─────────────────────────────────────────────────────────────
function wireControls() {
    // ── Interval buttons ─────────────────────────────────────────────────────
    dom.intervalBtns.forEach((btn) => {
        btn.addEventListener("click", async () => {
            const iv = btn.dataset.interval;
            if (iv === state.activeInterval) return;
            state.activeInterval = iv;
            dom.intervalBtns.forEach((b) => {
                b.classList.toggle("active", b.dataset.interval === iv);
                b.setAttribute(
                    "aria-selected",
                    b.dataset.interval === iv ? "true" : "false",
                );
            });
            sseDisconnect();
            await renderBars();
        });
    });

    // ── Days-back select ─────────────────────────────────────────────────────
    dom.daysSelect.addEventListener("change", async () => {
        state.activeDays = parseInt(dom.daysSelect.value, 10);
        sseDisconnect();
        await renderBars();
    });

    // ── Live toggle ───────────────────────────────────────────────────────────
    dom.liveToggle.addEventListener("change", () => {
        state.liveEnabled = dom.liveToggle.checked;
        if (state.liveEnabled) sseConnect();
        else sseDisconnect();
    });

    // ── Indicator toggles ─────────────────────────────────────────────────────
    dom.indBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const ind = btn.dataset.ind;
            if (!(ind in state.indicators)) return;

            // Toggle the state flag and button active class
            state.indicators[ind] = !state.indicators[ind];
            btn.classList.toggle("active", state.indicators[ind]);

            if (state.chart) {
                if (ind === "rsi") {
                    // RSI lives in a separate pane — recalc and sync the sub-chart
                    if (state.indicators.rsi) {
                        state.rsiData = calcRSI(state.candleData, 14);
                        seedLiveIndState();
                    } else {
                        state.rsiData = [];
                    }
                    syncRsiPane();
                } else {
                    // Overlay indicator — recalc just the affected series
                    recalcSingleIndicator(ind);
                    seedLiveIndState();
                    // Update primary chart with new series data (no full re-mount)
                    state.chart.updateSeries(buildSeries(), false);
                }
            }
        });
    });

    // ── Error dismiss ─────────────────────────────────────────────────────────
    dom.errorDismiss.addEventListener("click", hideError);

    // ── Keyboard shortcuts ────────────────────────────────────────────────────
    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")
            return;
        if (e.key === "r" || e.key === "R") {
            if (state.chart) state.chart.resetSeries();
        } else if (e.key === "l" || e.key === "L") {
            dom.liveToggle.checked = !dom.liveToggle.checked;
            dom.liveToggle.dispatchEvent(new Event("change"));
        }
    });

    // ── Sync interval from ?interval= param ──────────────────────────────────
    const qsInterval = new URLSearchParams(window.location.search).get(
        "interval",
    );
    if (qsInterval && VALID_INTERVALS[qsInterval]) {
        state.activeInterval = qsInterval;
        dom.intervalBtns.forEach((b) => {
            const active = b.dataset.interval === qsInterval;
            b.classList.toggle("active", active);
            b.setAttribute("aria-selected", active ? "true" : "false");
        });
    }

    // ── Sync days from ?days= param ───────────────────────────────────────────
    const qsDays = parseInt(
        new URLSearchParams(window.location.search).get("days") || "",
        10,
    );
    if (isFinite(qsDays) && qsDays > 0) {
        state.activeDays = qsDays;
        const opts = Array.from(dom.daysSelect.options).map((o) =>
            parseInt(o.value, 10),
        );
        const closest = opts.reduce((a, b) =>
            Math.abs(b - qsDays) < Math.abs(a - qsDays) ? b : a,
        );
        dom.daysSelect.value = String(closest);
    }

    // ── Sync indicator toggle button states from state.indicators ────────────
    dom.indBtns.forEach((btn) => {
        const ind = btn.dataset.ind;
        if (ind in state.indicators) {
            btn.classList.toggle("active", state.indicators[ind]);
        }
    });
}

/**
 * Recalculate a single indicator by name and update state arrays.
 * Used by the toggle handler to avoid a full recalcIndicators() pass.
 */
function recalcSingleIndicator(ind) {
    const c = state.candleData;
    const v = state.volumeData;

    switch (ind) {
        case "ema9":
            state.ema9Data = state.indicators.ema9 ? calcEMA(c, 9) : [];
            break;
        case "ema21":
            state.ema21Data = state.indicators.ema21 ? calcEMA(c, 21) : [];
            break;
        case "bb": {
            const bb = state.indicators.bb
                ? calcBollingerBands(c, 20, 2)
                : { upper: [], mid: [], lower: [] };
            state.bbUpperData = bb.upper;
            state.bbMidData = bb.mid;
            state.bbLowerData = bb.lower;
            break;
        }
        case "vwap":
            state.vwapData = state.indicators.vwap ? calcVWAP(c, v) : [];
            break;
        case "rsi":
            state.rsiData = state.indicators.rsi ? calcRSI(c, 14) : [];
            break;
    }
}

// ── Boot ───────────────────────────────────────────────────────────────────────
async function boot() {
    setStatus("connecting", "Connecting…");
    showOverlay("Connecting to data service…");
    wireControls();

    state.dataServiceUrl = await resolveDataServiceUrl();
    console.log("[chart] data service:", state.dataServiceUrl);

    await loadAssets();
}

// Hook boot() to DOM ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
} else {
    boot();
}
