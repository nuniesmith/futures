// =============================================================================
// BreakoutStrategy — NinjaTrader 8 — SINGLE FILE EDITION
// =============================================================================
//
// Drop ONLY this file into:
//   Documents\NinjaTrader 8\bin\Custom\Strategies\
//
// All dependencies (SignalBus, BridgeOrderEngine, OrbCnnPredictor,
// CnnSessionThresholds, OrbChartRenderer) are inlined at the bottom of this
// file.  Do NOT also deploy the individual source files — that causes duplicate
// class compile errors inside NT8.
//
// Strategy overview:
//   Opening Range Breakout across 22 futures instruments in a single strategy
//   instance.  Runs its own ORB detection and/or relays signals from the Ruby
//   indicator via the inlined SignalBus.  Entries are risk-sized and bracketed
//   by the inlined BridgeOrderEngine.  An optional ONNX CNN filter gates every
//   breakout through a pre-trained HybridBreakoutCNN model.
//
// Instruments (15 active + 7 pending data subscription):
//   Equity index : MGC, MES, MNQ, M2K, MYM
//   FX           : 6E, 6B, 6J, 6A, 6C, 6S
//   Rates        : ZN, ZB
//   Crypto (micro): MBT, MET
//   Pending sub  : MCL, MNG, SIL, MHG, ZC, ZS, ZW
//                  (uncomment CPendingInstruments when feed provides them)
//
// CNN quick-start (optional):
//   1. Train:  python scripts/train_gpu.py
//   2. Export: python scripts/export_onnx.py
//   3. Copy breakout_cnn_best.onnx + OnnxRuntime DLLs to bin\Custom\
//      Required DLLs (NuGet: Microsoft.ML.OnnxRuntime 1.24.2, netstandard2.0):
//        Microsoft.ML.OnnxRuntime.dll
//        System.Buffers.dll  System.Memory.dll  System.Numerics.Vectors.dll
//        System.Runtime.CompilerServices.Unsafe.dll
//        onnxruntime.dll  onnxruntime_providers_shared.dll  (win-x64 native)
//   4. Set CCnnModelPath constant below, set EnableCnnFilter = true.
//
// Model path (edit this constant — no recompile needed for path changes):
//   See CCnnModelPath in the "Hardcoded settings" section (~line 310).
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Web.Script.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using static NinjaTrader.NinjaScript.CnnSessionThresholds;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    // ── Operation mode enum ───────────────────────────────────────────────────
    public enum BreakoutMode
    {
        BuiltIn,         // BreakoutStrategy's own ORB logic only
        SignalBusRelay,  // Execute signals from SignalBus (Ruby) only
        Both             // Run both simultaneously
    }

    // ── Take Profit Trader funded account tiers ───────────────────────────────
    // Fixed contract counts follow TPT's 25% rule (conservative end of the
    // allowed range) so a runaway breakout can never breach the daily drawdown.
    //
    //   Tier        TPT max (micros)   Fixed qty here   % of allowed
    //   $50k        60 micros          2                 3 %
    //   $100k       120 micros         3                 2.5 %
    //   $150k       150 micros         4                 2.7 %
    //
    // To go more aggressive, raise the qty in GetTptContracts() — but do NOT
    // exceed 25 % of the TPT-allowed maximum without reviewing their current rules.
    public enum TptAccountTier
    {
        FiftyK,           // $50k account  — 2 micros default
        HundredK,         // $100k account — 3 micros default
        HundredFiftyK,    // $150k account — 4 micros default
    }

    // ── Breakout types — stable ordinals matching Python IntEnum ──────────────
    // MUST stay in sync with rb/models/feature_contract.json breakout_type_map.
    // Do NOT reorder or insert — ordinals are baked into trained ONNX models.
    public enum BreakoutType
    {
        ORB = 0,   // Opening Range (was "Orb" in v4)
        PrevDay = 1,   // Previous day high/low
        InitialBalance = 2,   // First 60 min of RTH (institutional)
        Consolidation = 3,   // ATR contraction / squeeze (non-time-based)
        Weekly = 4,   // Weekly high/low range
        Monthly = 5,   // Monthly high/low range
        Asian = 6,   // Asian session range (19:00–01:00 ET)
        BollingerSqueeze = 7,   // Bollinger Band squeeze
        ValueArea = 8,   // Value Area high/low (volume profile)
        InsideDay = 9,   // Inside day pattern
        GapRejection = 10,  // Gap fill rejection
        PivotPoints = 11,  // Classic pivot point levels
        Fibonacci = 12,  // Fibonacci retracement levels
    }

    // ── 3-phase bracket walk state ────────────────────────────────────────────
    // Phase1: entry filled, SL + TP1 working.
    // Phase2: TP1 hit → SL moved to breakeven, TP2 working on remaining qty.
    // Phase3: TP2 hit → trail remaining qty with EMA9 toward TP3 price.
    //         Exit on whichever comes first: price crosses EMA9 (adverse) or
    //         reaches TP3.
    public enum BreakoutPhase { Phase1, Phase2, Phase3, Closed }

    /// <summary>
    /// Tracks the bracket walk state for a single position identified by its SignalId.
    /// One instance per FireEntry call; removed when phase reaches Closed.
    /// </summary>
    // =========================================================================
    // Per-instrument stop-and-reverse (SAR) state
    // =========================================================================
    /// <summary>
    /// Tracks the always-in micro position state per instrument for the
    /// stop-and-reverse strategy.  Mirrors Python PositionManager's
    /// MicroPosition / reversal-gate fields.
    /// </summary>
    public sealed class ReversalState
    {
        /// <summary>Current open direction: "long", "short", or "" (flat).</summary>
        public string ActiveDirection = "";

        /// <summary>SignalId of the position that is currently open (matches _positionPhases key).</summary>
        public string ActiveSignalId = "";

        /// <summary>Entry price of the currently open position.</summary>
        public double EntryPrice = 0;

        /// <summary>ATR at entry — used for R-multiple calculation.</summary>
        public double AtrAtEntry = 0;

        /// <summary>SL price of the currently open position (updated at phase transitions).</summary>
        public double SlPrice = 0;

        /// <summary>Time of the last entry or reversal — used for cooldown gate.</summary>
        public DateTime LastReversalTime = DateTime.MinValue;

        /// <summary>Number of reversals made (for logging).</summary>
        public int ReversalCount = 0;

        public bool IsFlat => string.IsNullOrEmpty(ActiveDirection);
        public bool IsLong => ActiveDirection == "long";
        public bool IsShort => ActiveDirection == "short";

        /// <summary>
        /// Approximate R-multiple for the current position given the current
        /// bar close price.  Positive = winning, negative = losing.
        /// </summary>
        public double RMultiple(double currentPrice)
        {
            if (IsFlat || AtrAtEntry <= 0) return 0;
            double raw = IsLong
                ? (currentPrice - EntryPrice) / AtrAtEntry
                : (EntryPrice - currentPrice) / AtrAtEntry;
            return raw;
        }

        /// <summary>Returns true when the position is currently profitable.</summary>
        public bool IsWinning(double currentPrice)
        {
            if (IsFlat) return false;
            return IsLong ? currentPrice > EntryPrice : currentPrice < EntryPrice;
        }

        public void Open(string direction, string signalId, double entryPrice,
                         double atr, double sl, DateTime time)
        {
            ActiveDirection = direction;
            ActiveSignalId = signalId;
            EntryPrice = entryPrice;
            AtrAtEntry = atr;
            SlPrice = sl;
            LastReversalTime = time;
        }

        public void Close()
        {
            ActiveDirection = "";
            ActiveSignalId = "";
            EntryPrice = 0;
            AtrAtEntry = 0;
            SlPrice = 0;
        }
    }

    public sealed class PositionPhase
    {
        public string SignalId { get; set; }
        public string Direction { get; set; }   // "long" | "short"
        public string Asset { get; set; }
        public int Bip { get; set; }
        public BreakoutPhase Phase { get; set; } = BreakoutPhase.Phase1;

        // Entry price and ATR captured at signal time
        public double EntryPrice { get; set; }
        public double AtrAtEntry { get; set; }

        // Phase1 targets (submitted as bracket orders)
        public double SlPrice { get; set; }   // current SL (modified at TP1)
        public double Tp1Price { get; set; }
        public double Tp2Price { get; set; }
        public double Tp3Price { get; set; }   // computed: entry ± ATR × Tp3Mult

        // Phase2/3 contract quantities
        public int TotalQty { get; set; }
        public int Tp1Qty { get; set; }
        public int Tp2Qty { get; set; }
        public int Tp3Qty { get; set; }   // = TotalQty - Tp1Qty - Tp2Qty (remainder)

        // OCO group for the current SL leg (updated at each phase transition)
        public string OcoGroup { get; set; }

        // Phase3: EMA9 ratcheted trail price (high-water mark, mirrors Python ema9_trail_price)
        // Only moves in the favourable direction — never retreats.
        // Initialised to 0 (unset); set to the first EMA9 value when Phase3 begins.
        public double Ema9TrailPrice { get; set; }

        // Phase3: EMA9-based trailing exit was triggered
        public bool Tp3Submitted { get; set; }
        public bool Ema9StopHit { get; set; }
    }

    public class BreakoutStrategy : Strategy
    {
        // =====================================================================
        // Range configuration (one per BreakoutType)
        // =====================================================================
        private struct RangeConfig
        {
            public BreakoutType Type;
            public int RangeBars;           // 0 = not bar-count-limited
            public double SqueezeThreshold;    // used only by Consolidation
            public int MinBarsRequired;     // minimum bars before "established"
            public string Description;

            public RangeConfig(BreakoutType t, int bars, double squeeze, int minBars, string desc)
            {
                Type = t;
                RangeBars = bars;
                SqueezeThreshold = squeeze;
                MinBarsRequired = minBars;
                Description = desc;
            }
        }

        // =====================================================================
        // Per-instrument ORB state
        // =====================================================================
        // One instance per BIP index, allocated at DataLoaded.
        private sealed class InstrumentState
        {
            // ── MTF (15-minute higher-timeframe) EMA / MACD state ─────────────
            // Updated by UpdateMtf() on every 15m bar close for the 15m BIP
            // that corresponds to this instrument.  The score and alignment
            // flags are read by ShouldReverse() to enforce Gate 4 (MTF ≥ 0.60).
            //
            // EMA periods mirror Python MTFAnalyzer defaults: fast=9, mid=21, slow=50.
            // MACD mirrors Python: fast=12, slow=26, signal=9.
            //
            // EMA state — incremental exponential moving average using:
            //   alpha = 2 / (period + 1)
            //   ema_new = alpha * close + (1 - alpha) * ema_prev
            public double MtfEmaFast = 0;        // EMA-9 on 15m
            public double MtfEmaMid = 0;        // EMA-21 on 15m
            public double MtfEmaSlow = 0;        // EMA-50 on 15m
            public int MtfEmaFilled = 0;        // bars consumed so far
            public bool MtfEmaReady = false;    // true once EMA-50 warmed up (≥50 bars)

            // MACD state — EMA-12, EMA-26, Signal EMA-9 of (fast−slow)
            public double MtfMacdFastEma = 0;
            public double MtfMacdSlowEma = 0;
            public double MtfMacdLine = 0;    // macdFastEma − macdSlowEma
            public double MtfMacdSigEma = 0;    // EMA-9 of MtfMacdLine
            public double MtfMacdHistogram = 0;    // MtfMacdLine − MtfMacdSigEma
            public int MtfMacdFilled = 0;    // bars consumed for MACD (needs ≥26)
            public bool MtfMacdReady = false;// true once signal line warmed (≥35 bars)

            // Histogram ring-buffer for slope (last 3 values → slope over 3 bars)
            public double[] MtfHistBuf = new double[3];
            public int MtfHistBufIdx = 0;
            public int MtfHistFilled = 0;

            // Computed outputs (written by UpdateMtf, read by ShouldReverse)
            // MtfScore: 0.0–1.0 matching Python's weighted sum:
            //   +0.30  EMA fully stacked in direction
            //   +0.15  EMA slope agrees with direction
            //   +0.25  MACD histogram polarity agrees
            //   +0.15  MACD histogram slope agrees
            //   +0.15  (no opposing divergence — always granted in C#, divergence detection omitted)
            public double MtfScore = 0;   // last computed score (direction-agnostic)
            public double MtfEmaSlopePerBar = 0;   // % change in slow EMA per bar (last 5 bars)
            public double[] MtfEmaSlowBuf = new double[5]; // ring buffer for slope calc
            public int MtfEmaSlowIdx = 0;
            public int MtfEmaSlowFill = 0;

            // BIP index of the corresponding 15m data series (-1 = not yet assigned)
            public int MtfBip = -1;

            // ── Multi-range support ───────────────────────────────────────────
            // One RangeState per BreakoutType, keyed by enum value.
            public Dictionary<BreakoutType, RangeState> Ranges =
                new Dictionary<BreakoutType, RangeState>();

            // ── Inner class: tracks the live range for a single BreakoutType ──
            public sealed class RangeState
            {
                public double RangeHigh = 0;
                public double RangeLow = double.MaxValue;
                public bool RangeEstablished = false;
                public DateTime SessionDate = DateTime.MinValue;  // for session-reset detection
                public DateTime RangeEndTime = DateTime.MinValue;  // ORB / IB window end
                public int BarsInRange = 0;
                public bool FiredLong = false;
                public bool FiredShort = false;
                // TP3 multiplier — overwritten from feature_contract.json after DataLoaded
                public double Tp3AtrMult = 5.0;
                // Scratch fields used by range builders that need to carry extra state
                // between bar updates without allocating heap objects per bar.
                public double AuxHigh = 0;   // e.g. yesterday_high / swing_high / pivot
                public double AuxLow = 0;   // e.g. yesterday_low  / swing_low  / s1
                public double AuxValue = 0;   // e.g. pivot point PP / gap_size / fib level
                public string AuxTag = "";  // e.g. gap direction "UP"/"DOWN"
            }

            // ── ORB (legacy fields kept for CNN feature extraction & VWAP guard) ──
            // These are populated from Ranges[BreakoutType.ORB] on every update
            // so that all existing consumers (CNN tabular prep, premarket range
            // tracking, GetStatusSummary) continue to work without modification.
            public double OrbHigh = 0;
            public double OrbLow = double.MaxValue;
            public bool OrbEstablished = false;
            public DateTime OrbSessionDate = DateTime.MinValue;
            public DateTime OrbEndTime = DateTime.MinValue;
            // Number of bars observed inside the ORB window for the current session.
            // Used to detect mid-session startup: if the strategy loads after the ORB
            // window has already closed, OrbBarCount stays at 0 and the ORB is marked
            // invalid so CheckBreakout skips it entirely for that session.
            public int OrbBarCount = 0;
            // Per-direction flags so both a long AND short can fire on news days.
            public bool BreakoutFiredLong = false;
            public bool BreakoutFiredShort = false;

            // ── VWAP (session-resetting) ──────────────────────────────────────
            public double CumTypicalVol = 0;
            public double CumVolume = 0;
            public double Vwap = 0;
            public DateTime VwapDate = DateTime.MinValue;

            // ── VWAP history (ring buffer for dynamic VWAP in CNN snapshots) ──
            // Stores per-bar VWAP values so the C# renderer can draw a dynamic
            // VWAP curve matching the Python training images.  Size matches
            // CnnLookbackBars (default 60).
            public double[] VwapHistory;
            public int VwapHistIdx = 0;
            public int VwapHistFilled = 0;

            // ── Volume baseline ───────────────────────────────────────────────
            // Running sum for a simple moving average of volume.
            public double[] VolBuffer;
            public int VolBufIdx = 0;
            public double VolSum = 0;   // running sum of VolBuffer
            public bool VolReady = false;
            public int VolFilled = 0;   // bars written so far (caps at period)

            // ── CVD proxy (cumulative volume delta from bar data) ─────────────
            // Approximates tick-level CVD using bar close vs open:
            //   delta_bar = volume * sign(close - open)
            // Accumulated from OR start to breakout bar, then normalised by
            // total volume to produce cvd_delta in [-1, 1].
            // Reset each session along with the ORB.
            public double CvdSignedVol = 0;    // Σ(vol × sign(close-open))
            public double CvdTotalVol = 0;     // Σ(vol)

            // ── Daily range history for NR7 detection ─────────────────────────
            // Stores the last 7 session daily ranges (high-low) so we can
            // detect the Narrow Range 7 pattern properly instead of the ATR
            // heuristic.  Updated once per new session.
            public double[] DailyRangeHistory = new double[7];
            public int DailyRangeIdx = 0;
            public int DailyRangeFilled = 0;
            // Track the previous session's high/low for computing daily range.
            public double SessionHigh = double.MinValue;
            public double SessionLow = double.MaxValue;

            // ── Premarket range tracking ──────────────────────────────────────
            // Tracks high/low from session open to OR start for the
            // premarket_range_ratio feature (pm_range / or_range).
            public double PremarketHigh = double.MinValue;
            public double PremarketLow = double.MaxValue;

            // ── Previous session ORB levels (for PrevDay range) ───────────────
            // Captured from the ORB range state at the moment UpdateRangeWindow(Orb)
            // detects a new session — i.e. before the legacy OrbHigh/OrbLow fields
            // are overwritten with the new session's first bar.  This is the only
            // reliable source for yesterday's actual price levels because:
            //   • st.SessionHigh/Low are wiped by UpdateVwap before UpdateRangeWindow runs
            //   • st.OrbHigh/Low are overwritten by UpdateRangeWindow(Orb) which runs
            //     before UpdateRangeWindow(PrevDay) in OnBarUpdate
            public double PrevOrbHigh = 0;
            public double PrevOrbLow = double.MaxValue;

            // ── Cooldown ──────────────────────────────────────────────────────
            public DateTime LastEntryTime = DateTime.MinValue;

            // ── Tick-safety helpers ───────────────────────────────────────────
            // LastVolume: volume of the most recently *closed* bar, captured at
            // IsFirstTickOfBar.  Used by the volume-surge filter so that
            // Calculate=OnEachTick doesn't compare a live accumulating tick-vol
            // (which starts at 1) against the historical average.
            public double LastVolume = 0;
            // LastLoggedBar: bar index of the most recent filter-rejection log
            // line.  Prevents the same rejection reason printing on every tick
            // within the same bar when Calculate=OnEachTick.
            public int LastLoggedBar = -1;
            // LastBarProcessed: bar index (Count - 1) of the most recently
            // processed closed bar for this BIP.  Used in place of IsFirstTickOfBar
            // because IsFirstTickOfBar is unreliable for secondary data series
            // (BIP > 0) and can also miss bars when the strategy starts mid-bar.
            // We detect a new closed bar by checking whether Count - 2 > LastBarProcessed,
            // which works correctly across all BIPs regardless of Calculate mode.
            public int LastBarProcessed = -1;

            // ── ATR — Wilder's smoothed, period 14 ───────────────────────────
            // During warmup (first <period> bars) we accumulate TR values in
            // AtrBuffer and compute a plain average.  Once AtrReady flips true
            // we switch to Wilder's exponential smoothing:
            //   AtrValue = (AtrValue * (period-1) + tr) / period
            // This matches the ATR(14) displayed by most platforms.
            public double AtrValue = 0;
            public double TrSum = 0;   // running TR sum (warmup only)
            public double PrevClose = 0;
            public double[] AtrBuffer;          // warmup ring-buffer
            public int AtrBufIdx = 0;
            public int AtrFilled = 0;   // bars written (caps at period)
            public bool AtrReady = false;

            // ── EMA9 — exponential moving average, period 9 ───────────────────
            // Used for Phase3 trailing stop logic.  Computed on every closed bar
            // using Wilder/standard EMA smoothing:
            //   EMA = prev_EMA + (2/(N+1)) * (close - prev_EMA)
            // Seeded from the simple average of the first 9 closes.
            public double Ema9Value = 0;
            public double Ema9Sum = 0;   // warmup: running sum of first 9 closes
            public int Ema9Filled = 0;   // bars written into warmup (caps at 9)
            public bool Ema9Ready = false;

            // ── ATR history ring-buffer for atr_trend feature (v7.1) ──────────
            // Stores the last 10 ATR values so we can detect whether ATR is
            // expanding or contracting.  Updated every bar by UpdateAtr.
            public double[] AtrHistory = new double[10];
            public int AtrHistIdx = 0;
            public int AtrHistFilled = 0;

            // ── Volume history ring-buffer for volume_trend feature (v7.1) ────
            // Stores the last 5 bar volumes for slope calculation.
            public double[] VolTrendBuf = new double[5];
            public int VolTrendIdx = 0;
            public int VolTrendFilled = 0;

            // ── Prior day OHLC for daily bias features (v7) ───────────────────
            // Captured at session rollover from the PrevDay range builder.
            // Used to compute daily_bias_direction, prior_day_pattern, etc.
            public double PrevDayOpen = 0;
            public double PrevDayHigh = 0;
            public double PrevDayLow = 0;
            public double PrevDayClose = 0;
            public double PrevDayVolume = 0;
            public bool PrevDayValid = false;

            // ── Prior week H/L for weekly_range_position (v7) ─────────────────
            public double PrevWeekHigh = 0;
            public double PrevWeekLow = double.MaxValue;
            public bool PrevWeekValid = false;

            // ── Session iterator (Globex-aware session detection) ─────────────
            public SessionIterator SessionIter = null;

            public InstrumentState(int volPeriod, int atrPeriod, int vwapHistorySize)
            {
                VolBuffer = new double[volPeriod];
                AtrBuffer = new double[atrPeriod];
                VwapHistory = new double[vwapHistorySize];

                // ── Register every breakout type ──────────────────────────────
                // All 13 types from the Python IntEnum are registered so that
                // UpdateRangeWindow / CheckBreakout can iterate them uniformly.
                // Types whose detection logic is not yet implemented will simply
                // never set RangeEstablished = true and are harmless.
                Ranges[BreakoutType.ORB] = new RangeState();
                Ranges[BreakoutType.PrevDay] = new RangeState();
                Ranges[BreakoutType.InitialBalance] = new RangeState();
                Ranges[BreakoutType.Consolidation] = new RangeState();
                Ranges[BreakoutType.Weekly] = new RangeState();
                Ranges[BreakoutType.Monthly] = new RangeState();
                Ranges[BreakoutType.Asian] = new RangeState();
                Ranges[BreakoutType.BollingerSqueeze] = new RangeState();
                Ranges[BreakoutType.ValueArea] = new RangeState();
                Ranges[BreakoutType.InsideDay] = new RangeState();
                Ranges[BreakoutType.GapRejection] = new RangeState();
                Ranges[BreakoutType.PivotPoints] = new RangeState();
                Ranges[BreakoutType.Fibonacci] = new RangeState();
            }
        }

        // ── BIP routing (built at DataLoaded) ─────────────────────────────────
        private readonly Dictionary<string, int> _symbolToBip =
            new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        private string[] _extraSymbols = new string[0];

        // ── Data-load guard ───────────────────────────────────────────────────
        // Prevents ORB/breakout logic from running until all active BIPs have
        // finished loading historical bars.  Without this guard, BIPs that
        // load slowly will have 0 bars and produce false ORB ranges.
        private bool _allDataLoaded = false;
        private DateTime _realtimeStart = DateTime.MinValue;
        private int DataLoadWaitSeconds = 15; // reduced now that non-subscribed symbols are removed
        private int _emptyBipsAtStart = 0;
        private DateTime _lastProgressLog = DateTime.MinValue;

        // BIP indices that loaded 0 bars after the wait window — skipped by
        // OnBarUpdate so they don't pollute ORB/breakout logic or stall startup.
        private HashSet<int> _skippedBips = new HashSet<int>();

        // ── Per-position phase tracking for 3-phase bracket walk ─────────────
        // Keyed by SignalId (the suffix used in order names).  Populated when
        // FireEntry creates an order; updated by OnOrderUpdate phase transitions.
        private Dictionary<string, PositionPhase> _positionPhases =
            new Dictionary<string, PositionPhase>(StringComparer.Ordinal);
        private object _phaseLock = new object();

        // ── Per-instrument SAR (stop-and-reverse) state ───────────────────────
        // One ReversalState per BIP index.  Tracks the always-in position
        // direction so CheckBreakout knows whether a new signal is a fresh
        // entry or a reversal candidate.
        private ReversalState[] _sarStates;

        // ── Per-instrument state array (index = BIP) ──────────────────────────
        private InstrumentState[] _states;

        // Property shims for SAR constants (keep call-sites readable)
        private double SarMinCnnProb => CSarMinCnnProb;
        private double SarWinningCnnProb => CSarWinningCnnProb;
        private double SarHighWinnerCnnProb => CSarHighWinnerCnnProb;
        private double SarMinMtfScore => CSarMinMtfScore;
        private int SarCooldownMinutes => CSarCooldownMinutes;
        private double SarChaseMaxAtrFraction => CSarChaseMaxAtrFraction;
        private double SarChaseMinCnnProb => CSarChaseMinCnnProb;
        private double SarHighWinnerRMult => CSarHighWinnerRMult;

        // ── Order engine (all SubmitOrderUnmanaged calls go through here) ─────
        private BridgeOrderEngine _engine;

        // ── CNN predictor (optional AI filter) ───────────────────────────────
        // Loaded from an ONNX file at DataLoaded; null when disabled or when
        // the model file is not found.  Thread-safe — ONNX Runtime's
        // InferenceSession allows concurrent Run() calls.
        private OrbCnnPredictor _cnn;

        // Temp folder for chart snapshots rendered before each CNN inference.
        // Snapshots are overwritten on every call (one per instrument per bar)
        // so disk usage stays bounded.
        private string _cnnSnapshotFolder;

        // ── Prometheus counters ───────────────────────────────────────────────
        private long _metricSignalsReceived;
        private long _metricSignalsExecuted;
        private long _metricSignalsRejected;
        private long _metricExitsExecuted;
        private long _metricBusDrained;
        private long _metricCnnAccepted;
        private long _metricCnnRejected;
        private long _metricOrdersRejected;

        // ── Active position tracking ─────────────────────────────────────
        // Tracks how many instruments currently have open positions.
        // Incremented on fill, decremented on flat.  Used to enforce
        // MaxConcurrentPositions gate before submitting new entries.
        private int _activePositionCount = 0;
        private readonly HashSet<string> _activeInstruments = new HashSet<string>();

        // ── Per-type TP3 multipliers loaded from feature_contract.json ────────
        // Populated at DataLoaded time. If the contract file is missing or a
        // type is absent, the global CTp3AtrMult constant is used as fallback.
        private Dictionary<BreakoutType, double> _tp3MultByType =
            new Dictionary<BreakoutType, double>();

        // ── MTF BIP routing ────────────────────────────────────────────────────
        // Maps instrument root name → BIP index of the *15m* data series for that
        // instrument.  Populated at DataLoaded once BarsArray is fully built.
        // Used by UpdateMtf() to read the correct 15m series.
        private readonly Dictionary<string, int> _mtfBipBySymbol =
            new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        // ── SAR sync — HTTP client for pushing reversals to the Python engine ──
        // Posts to POST http://100.100.84.48:8000/sar/sync after every reversal.
        // Fire-and-forget (ContinueWith, no await) so the bar-update thread is
        // never blocked.  Initialised in OnStateChange(DataLoaded).
        private System.Net.Http.HttpClient _sarHttpClient;
        private const string CSarSyncUrl = "http://100.100.84.48:8000/sar/sync";
        // MTF bars base URL — used by GetMtfBarsFromEngine() to fetch 15m bars
        // from the Pi's data engine as a fallback when NT8's own 15m BIP is missing.
        private const string CEngineBaseUrl = "http://100.100.84.48:8000";

        // ── Risk gate (mirrors Bridge.RiskBlocked so both respect it) ─────────
        // BreakoutStrategy has no HTTP listener of its own; it simply won't
        // allow entries when this is set.  Bridge can set it via SignalBus
        // or the caller can set it programmatically.
        internal volatile bool RiskBlocked = false;
        internal volatile string RiskBlockReason = "";

        // ── Stop-and-reverse constants (mirror Python PositionManager env vars) ──
        // Min CNN probability required to reverse a losing position.
        private const double CSarMinCnnProb = 0.85;
        // Min CNN probability required to reverse a *winning* position (higher bar).
        private const double CSarWinningCnnProb = 0.92;
        // Even higher threshold when position is at 1R+ profit — mirror Python Gate 6.
        private const double CSarHighWinnerCnnProb = 0.95;
        // Min MTF alignment score required for a reversal.
        private const double CSarMinMtfScore = 0.60;
        // Cooldown in minutes between reversals (matches Python 1800 s default).
        private const int CSarCooldownMinutes = 30;
        // Max ATR fraction for a market-chase entry (mirrors PM_CHASE_MAX_ATR_FRACTION 0.50).
        private const double CSarChaseMaxAtrFraction = 0.50;
        // Min CNN probability for a market-chase entry (mirrors PM_CHASE_MIN_CNN_PROB 0.90).
        private const double CSarChaseMinCnnProb = 0.90;
        // R-multiple threshold above which Gate 6 (high-winner protection) kicks in.
        private const double CSarHighWinnerRMult = 1.0;

        // =====================================================================
        // Properties
        // =====================================================================
        #region Properties

        // ── 1. Account ────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account", GroupName = "1. Account", Order = 1)]
        public string AccountName { get; set; } = "Sim101";

        // ── 2. Risk ───────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "TPT Mode (funded account)", GroupName = "2. Risk", Order = 1,
                 Description = "When enabled, uses fixed contract counts from the selected " +
                               "Take Profit Trader account tier instead of dynamic ATR-based sizing.")]
        public bool TptMode { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "TPT Account Tier", GroupName = "2. Risk", Order = 2,
                 Description = "$50k → 2 micros, $100k → 3 micros, $150k → 4 micros. " +
                               "Edit GetTptContracts() to adjust within TPT's allowed range.")]
        public TptAccountTier AccountTier { get; set; } = TptAccountTier.FiftyK;

        // ── 3. AI Filter ──────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Enable CNN Filter", GroupName = "3. AI Filter", Order = 1,
                 Description = "Gates every breakout candidate through the ONNX HybridBreakoutCNN model. " +
                               "Model path and all other CNN settings are hardcoded — edit the source to change them.")]
        public bool EnableCnnFilter { get; set; } = true;

        // ── 4. Diagnostics ────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Enable Debug Logging", GroupName = "4. Diagnostics", Order = 1,
                 Description = "Verbose per-bar ORB logging for all instruments (BIP + symbol).")]
        public bool EnableDebugLogging { get; set; } = true;

        // ── Hardcoded settings (edit source to change) ────────────────────────
        // Mode / instruments
        private const BreakoutMode CMode = BreakoutMode.Both;
        // Active instruments — confirmed working with current data subscription.
        // Core 5 assets — reduced from 15 to improve stability and reduce
        // concurrent position count.  Extended assets commented for future use.
        private const string CTrackedInstruments = "MGC,MES,MNQ,MYM,6E";
        // private const string CExtendedInstruments = "M2K,6B,6J,6A,6C,6S,ZN,ZB,MBT,MET";
        // Pending data subscription — uncomment and append to CTrackedInstruments
        // when your feed provides these symbols.  They were removed because NT8
        // returns 0 bars (data subscription issue, not a code bug).
        // private const string CPendingInstruments = "MCL,MNG,SIL,MHG,ZC,ZS,ZW";
        // ORB
        private const int COrbMinutes = 30;
        private const double CVolumeSurgeMult = 1.5;
        private const int CVolumeAvgPeriod = 20;
        private const bool CRequireVwap = true;
        private const double CMinOrbAtrRatio = 0.3;
        private const int CEntryCooldownMinutes = 10; // increased from 5 to reduce over-trading
        // Risk / sizing
        private const double CAccountSize = 50000;
        private const double CRiskPercentPerTrade = 0.5;
        private const int CMaxContracts = 5;
        private const int CMaxConcurrentPositions = 5; // max open trades across all instruments
        private const double CSlAtrMult = 1.5;
        private const double CTp1AtrMult = 2.0;
        private const double CTp2AtrMult = 3.5;
        private const double CTp3AtrMult = 5.0;   // Phase3 target: entry ± ATR × this
        private const bool CEnableTp3Trailing = true; // enable 3-phase EMA9 trailing walk
        private const bool CEnableAutoBrackets = true;
        private const int CDefaultSlTicks = 20;
        private const int CDefaultTpTicks = 40;
        // CNN
        // Number of tabular features C# builds and passes to OrbCnnPredictor.
        // Must stay in sync with PrepareCnnTabular() and feature_contract.json v7.1.
        // If the loaded ONNX model reports a different NumTabular, a warning is
        // printed at startup — retrain/re-export the model to clear it.
        private const int CNumTabularFeatures = 28;
        private static readonly string CCnnModelPath = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "NinjaTrader 8", "bin", "Custom", "Models", "breakout_cnn_best.onnx");
        private const double CCnnThresholdOverride = 0;
        private const string CCnnSessionKey = "us";
        private const string CCnnSnapshotFolder = "";
        private const int CCnnLookbackBars = 60;
        private const bool CCnnUseCuda = false;
        // Diagnostics
        private const int CDebugLogFrequency = 5;

        // ── Property shims — map old names to constants so all call-sites compile unchanged ──
        private BreakoutMode Mode => CMode;
        private string TrackedInstruments => CTrackedInstruments;
        private int OrbMinutes => COrbMinutes;
        private double VolumeSurgeMult => CVolumeSurgeMult;
        private int VolumeAvgPeriod => CVolumeAvgPeriod;
        private bool RequireVwap => CRequireVwap;
        private double MinOrbAtrRatio => CMinOrbAtrRatio;
        private int EntryCooldownMinutes => CEntryCooldownMinutes;
        private double AccountSize => CAccountSize;
        private double RiskPercentPerTrade => CRiskPercentPerTrade;
        private int MaxContracts => CMaxContracts;
        private int MaxConcurrentPositions => CMaxConcurrentPositions;
        private double SlAtrMult => CSlAtrMult;
        private double Tp1AtrMult => CTp1AtrMult;
        private double Tp2AtrMult => CTp2AtrMult;
        private double Tp3AtrMult => CTp3AtrMult;
        private bool EnableTp3Trailing => CEnableTp3Trailing;
        private bool EnableAutoBrackets => CEnableAutoBrackets;
        private int DefaultSlTicks => CDefaultSlTicks;
        private int DefaultTpTicks => CDefaultTpTicks;
        private string CnnModelPath => CCnnModelPath;
        private double CnnThresholdOverride => CCnnThresholdOverride;
        private string CnnSessionKey => CCnnSessionKey;
        private string CnnSnapshotFolder => CCnnSnapshotFolder;
        private int CnnLookbackBars => CCnnLookbackBars;
        private bool CnnUseCuda => CCnnUseCuda;
        private int DebugLogFrequency => CDebugLogFrequency;

        #endregion

        // =====================================================================
        // Lifecycle
        // =====================================================================
        #region Lifecycle

        protected override void OnStateChange()
        {
            // ── SetDefaults ───────────────────────────────────────────────────
            if (State == State.SetDefaults)
            {
                Description = "Standalone Opening Range Breakout strategy. Runs its own ORB detection " +
                              "and/or relays signals from Ruby via SignalBus. Uses BridgeOrderEngine " +
                              "for risk-sized order execution across 15 micro contracts (7 more pending data sub). " +
                              "Optionally gates entries through a HybridBreakoutCNN ONNX model.";
                Name = "BreakoutStrategy";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                IsUnmanaged = true;
                // NOTE: The "In order to display realized PnL per instrument the
                // strategy developer must..." message shown in the Strategies tab
                // for secondary BIP instruments (MES, MNQ, MYM, 6E) is a known
                // NT8 display limitation for unmanaged multi-instrument strategies.
                // There is no public API property to suppress it.  PnL is tracked
                // correctly on the primary instrument (MGC/BIP0); the message on
                // secondaries is cosmetic only and does not affect order execution,
                // risk management, or fill attribution.
                // Force BIP0 to 1-minute bars so it matches all AddDataSeries subscriptions.
                // Without this, BIP0 inherits whatever timeframe the chart is set to.
                BarsPeriod = new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1 };
                // BarsRequiredToTrade: keep low so the strategy activates quickly
                // after loading historical data — indicators self-warm via their
                // own filled-bar counters (AtrFilled, VolFilled).
                BarsRequiredToTrade = 20;
                // 60 calendar days ≈ 42 trading days of 1-min Globex bars.
                // Gives every BIP 10,000+ bars so ATR, volume avg, and VWAP
                // are fully warmed from the very first bar — matching the depth
                // that BIP0 (primary chart) loads automatically.
                DaysToLoad = 60;
            }

            // ── Configure ─────────────────────────────────────────────────────
            else if (State == State.Configure)
            {
                // Register as SignalBus consumer when running in relay or both modes.
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                {
                    SignalBus.RegisterConsumer();
                    Print("[BreakoutStrategy] SignalBus consumer registered");
                }

                // ── Subscribe additional instruments ──────────────────────────
                // Root cause of the "0 bars" problem:
                //
                // NT8's broker sync creates phantom MasterInstrument rows with
                // InstrumentType=Stock for futures symbols every time it connects,
                // even after the SQLite DB has been cleaned by fix_nt8_instruments.ps1.
                // NT8's AddDataSeries resolver picks the row with the LOWEST Id.
                // If the phantom Stock row has a lower Id than the correct futures
                // row, the BIP gets 0 bars regardless of what TradingHours is set,
                // because Stock instruments cannot load futures bar data.
                //
                // Two-step fix applied here at Configure time (before AddDataSeries):
                //
                //   Step 1 — REMOVE phantom Stock rows from MasterInstrument.All.
                //            This forces NT8's resolver to find only the correct
                //            futures row (InstrumentType != Stock) when AddDataSeries
                //            is called immediately after.
                //
                //   Step 2 — PATCH TradingHours on the surviving futures row.
                //            Belt-and-suspenders in case the DB row still has the
                //            wrong session template (RTH or FX on a non-FX futures).
                //
                // The permanent DB fix (fix_nt8_instruments.ps1) prevents the phantom
                // rows from being persisted across restarts, but the in-memory removal
                // here handles the broker-sync rows that appear at runtime.
                if (!string.IsNullOrWhiteSpace(TrackedInstruments))
                {
                    var requested = TrackedInstruments
                        .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    var extras = new List<string>();
                    string primaryRoot = Instrument?.MasterInstrument.Name.ToUpperInvariant() ?? "";

                    foreach (string sym in requested)
                    {
                        string root = sym.Trim().ToUpperInvariant();
                        if (string.IsNullOrEmpty(root)) continue;
                        if (root == primaryRoot) continue;
                        if (extras.Contains(root)) continue;

                        try
                        {
                            string th = GetTradingHoursTemplate(root);

                            // ── Resolve the real continuous-contract Future row ──────────
                            //
                            // NT8's broker sync creates phantom MasterInstrument rows with
                            // InstrumentType=Stock and expiry DEC99 for every futures symbol.
                            // These phantoms often have the LOWEST Id, so passing the bare
                            // root symbol to AddDataSeries resolves to the DEC99 phantom.
                            //
                            // Previous approaches (name-mangling, InstrumentType patching)
                            // all fail because NT8 uses MasterInstrument rows internally for
                            // path construction and lazy resolution.
                            //
                            // The correct fix: find the real Future MasterInstrument row,
                            // get its current front-month full name (e.g. "MES 06-25"),
                            // and pass that explicit name to AddDataSeries.  NT8 resolves
                            // an explicit full name unambiguously — no phantom can interfere.

                            var thObj = NinjaTrader.Data.TradingHours.Get(th);
                            string addName = root; // fallback: bare root symbol

                            // Scan MasterInstrument.All for the correct Future row.
                            // Pick the row with InstrumentType=Future whose expiry is
                            // in the future (or the furthest-dated if multiple exist).
                            // Exclude DEC99 phantoms that may have been patched to Future
                            // by earlier runs.
                            lock (NinjaTrader.Cbi.MasterInstrument.Sync)
                            {
                                NinjaTrader.Cbi.MasterInstrument bestMi = null;
                                DateTime bestExpiry = DateTime.MinValue;

                                foreach (var mi in NinjaTrader.Cbi.MasterInstrument.All)
                                {
                                    if (!string.Equals(mi.Name, root, StringComparison.OrdinalIgnoreCase))
                                        continue;
                                    if (mi.InstrumentType != NinjaTrader.Cbi.InstrumentType.Future)
                                        continue;

                                    // Get the next valid expiry for this MasterInstrument.
                                    DateTime expiry;
                                    try { expiry = mi.GetNextExpiry(DateTime.Now); }
                                    catch { continue; }

                                    // Skip DEC99 placeholders (expiry year 2099)
                                    if (expiry.Year >= 2099) continue;
                                    // Skip expired contracts (14-day grace for rollover —
                                    // futures remain tradable past listed expiry)
                                    if (expiry < DateTime.Now.Date.AddDays(-14)) continue;

                                    // Pick the nearest (front-month) valid contract
                                    if (bestMi == null || expiry < bestExpiry)
                                    {
                                        bestMi = mi;
                                        bestExpiry = expiry;
                                    }
                                }

                                if (bestMi != null)
                                {
                                    // Build the full instrument name NT8 expects:
                                    // "MES 06-25" format = root + " " + MM-YY
                                    addName = bestMi.Name + " " + bestExpiry.ToString("MM-yy");
                                    Print($"[BreakoutStrategy] Resolved {root} → '{addName}' (expiry {bestExpiry:yyyy-MM-dd})");

                                    // Patch TradingHours on the real Future row
                                    if (thObj != null && !string.Equals(bestMi.TradingHours?.Name, th, StringComparison.OrdinalIgnoreCase))
                                    {
                                        bestMi.TradingHours = thObj;
                                        Print($"[BreakoutStrategy] Patched TH on {addName} → '{th}'");
                                    }
                                }
                                else
                                {
                                    // No valid Future row found — log all rows for debugging
                                    Print($"[BreakoutStrategy] ⚠ No valid Future row for {root}. Falling back to root symbol.");
                                    foreach (var mi in NinjaTrader.Cbi.MasterInstrument.All
                                        .Where(m => string.Equals(m.Name, root, StringComparison.OrdinalIgnoreCase)))
                                    {
                                        DateTime exp;
                                        try { exp = mi.GetNextExpiry(DateTime.Now); }
                                        catch { exp = DateTime.MinValue; }
                                        Print($"[BreakoutStrategy]   → Id={mi.Id} type={mi.InstrumentType} expiry={exp:yyyy-MM-dd}");
                                    }
                                }
                            }

                            AddDataSeries(addName,
                                new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1 },
                                th);
                            extras.Add(root);
                            Print($"[BreakoutStrategy] AddDataSeries: {addName} → BIP {extras.Count} (1m)  session='{th}'");

                            // ── 15m HTF series for MTF alignment scoring ──────────────
                            // Added immediately after the 1m series so the BIP index is
                            // predictable: mtfBip = 1m_bip + extras.Count (N extra symbols).
                            // UpdateMtf() looks up the 15m BIP from _mtfBipBySymbol (built
                            // at DataLoaded by matching BarsPeriod.Value == 15).
                            try
                            {
                                AddDataSeries(addName,
                                    new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 15 },
                                    th);
                                Print($"[BreakoutStrategy] AddDataSeries: {addName} → BIP {extras.Count * 2} (15m)  session='{th}'");
                            }
                            catch (Exception ex15)
                            {
                                Print($"[BreakoutStrategy] 15m AddDataSeries failed for {root} (non-fatal): {ex15.Message}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"[BreakoutStrategy] AddDataSeries failed for {root}: {ex.Message}");
                        }
                    }

                    _extraSymbols = extras.ToArray();
                }

                // ── 15m series for the primary instrument (BIP0) ──────────────
                // Extra instruments get their 15m series inside the loop above.
                // The primary instrument only has BIP0 (1m) by default; we add
                // its 15m series here so UpdateMtf works for it too.
                try
                {
                    string primaryTh = GetTradingHoursTemplate(
                        Instrument?.MasterInstrument?.Name?.ToUpperInvariant() ?? "");
                    AddDataSeries(
                        Instrument?.MasterInstrument?.Name ?? "",
                        new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 15 },
                        primaryTh);
                    Print($"[BreakoutStrategy] AddDataSeries: primary 15m BIP added for " +
                          $"{Instrument?.MasterInstrument?.Name}");
                }
                catch (Exception exPrimary15)
                {
                    Print($"[BreakoutStrategy] Primary 15m AddDataSeries failed (non-fatal): {exPrimary15.Message}");
                }
            }

            // ── DataLoaded ────────────────────────────────────────────────────
            else if (State == State.DataLoaded)
            {
                // ── DataLoaded session verification ───────────────────────────
                // Log the session NT8 actually resolved for each BIP so we can
                // confirm the Instrument.GetInstrument() path in Configure worked.
                // No patching needed here — Configure already set the correct
                // TradingHours on the MasterInstrument before AddDataSeries ran.
                if (EnableDebugLogging)
                {
                    for (int i = 0; i < BarsArray.Length; i++)
                    {
                        if (BarsArray[i] == null) continue;
                        string sym = BarsArray[i].Instrument?.MasterInstrument?.Name ?? "?";
                        string thName = BarsArray[i].Instrument?.MasterInstrument?.TradingHours?.Name ?? "(null)";
                        int barCount = BarsArray[i].Count;
                        Print($"[BreakoutStrategy] DataLoaded BIP{i} {sym,-6}: {barCount,6} bars  session='{thName}'");
                    }
                }

                // Warn if BIP0 is not on a 1-minute timeframe — all AddDataSeries
                // subscriptions are hardcoded to 1-minute, so BIP0 must match.
                if (BarsArray.Length > 0)
                {
                    var p = BarsArray[0].BarsPeriod;
                    if (p.BarsPeriodType != BarsPeriodType.Minute || p.Value != 1)
                        Print($"[BreakoutStrategy] ⚠ BIP0 ({Instrument.MasterInstrument.Name}) " +
                              $"is on {p.Value}-{p.BarsPeriodType} bars, not 1-Minute. " +
                              $"All 21 additional instruments use 1-Minute bars. " +
                              $"Run this strategy on a 1-Minute MGC chart for consistent ORB timing.");
                    else
                        Print($"[BreakoutStrategy] BIP0 ({Instrument.MasterInstrument.Name}) confirmed on 1-Minute bars.");
                }

                // ── Preflight: trigger history download for any empty BIP ─────
                // NT8 only populates the root-symbol minute cache (e.g. db\minute\MES\)
                // after at least one continuous-contract chart has been opened for that
                // symbol.  When the cache is empty, BarsArray[bip].Count == 0 and the
                // session resolves to "US Equities RTH" (NT8's sentinel for "no data").
                //
                // NT8 does not expose a programmatic history-download API callable from
                // DataLoaded.  The strategy instead relies on NT8 streaming historical
                // bars in the background after transitioning to Realtime.
                //
                // Preflight: detect empty BIPs and extend the wait window.
                // 1. Count BIPs with 0 bars (root-symbol cache empty).
                // 2. Extend DataLoadWaitSeconds so OnBarUpdate keeps polling.
                // 3. Log empty symbols so the user knows which continuous-
                //    contract charts to open once if the wait expires empty.
                _emptyBipsAtStart = 0;
                for (int i = 0; i < BarsArray.Length; i++)
                {
                    if (BarsArray[i] == null) continue;
                    if (BarsArray[i].Count == 0)
                    {
                        _emptyBipsAtStart++;
                        string sym = BarsArray[i].Instrument.MasterInstrument.Name;
                        Print("[BreakoutStrategy] BIP" + i + " " + sym +
                              ": 0 bars at DataLoaded -- will retry briefly, then skip." +
                              " If persistent, add " + sym + " to CPendingInstruments" +
                              " and check your data subscription.");
                    }
                }

                if (_emptyBipsAtStart > 0)
                {
                    // Brief grace period for stragglers — 10s per empty BIP,
                    // capped at 30s.  Non-subscribed symbols won't magically
                    // appear, so a long wait just delays strategy startup.
                    int extraWait = Math.Min(_emptyBipsAtStart * 10, 30);
                    DataLoadWaitSeconds = Math.Max(DataLoadWaitSeconds, extraWait);
                    Print("[BreakoutStrategy] " + _emptyBipsAtStart +
                          " BIP(s) have empty cache. Waiting up to " +
                          DataLoadWaitSeconds + "s then skipping empties...");
                }

                // ── Build BIP routing table ───────────────────────────────────
                _symbolToBip.Clear();

                string primaryRoot = Instrument?.MasterInstrument.Name.ToUpperInvariant() ?? "";
                if (!string.IsNullOrEmpty(primaryRoot))
                    _symbolToBip[primaryRoot] = 0;

                for (int i = 0; i < _extraSymbols.Length; i++)
                {
                    int bip = i + 1;
                    if (bip < BarsArray.Length)
                    {
                        var bipBars = BarsArray[bip];
                        string confirmed = bipBars.Instrument.MasterInstrument.Name
                                                              .ToUpperInvariant();
                        // Log the trading-hours template NT8 actually resolved for this BIP.
                        // If a symbol has 0 bars, the template name here is the first thing
                        // to check — a mismatch (e.g. a forex template on a futures symbol)
                        // means NT8 couldn't find a matching session and silently skipped
                        // historical loading.
                        string thName = bipBars.Instrument.MasterInstrument
                                               .TradingHours?.Name ?? "(none)";
                        int bipCount = bipBars.Count;
                        string status = bipCount > 0 ? "OK" : "⚠ 0 BARS — check data subscription";

                        _symbolToBip[confirmed] = bip;
                        Print($"[BreakoutStrategy] BIP{bip} {confirmed,-6} " +
                              $"{bipCount,6} bars  [{status}]  " +
                              $"session='{thName}'");
                    }
                    else
                    {
                        // BarsArray is shorter than expected — NT8 dropped the subscription.
                        Print($"[BreakoutStrategy] ⚠ BIP{bip} ({_extraSymbols[i]}) not present " +
                              $"in BarsArray (length={BarsArray.Length}). " +
                              $"Symbol may be unknown to your data provider.");
                    }
                }

                // BIP 0 summary
                {
                    string thName0 = BarsArray[0].Instrument.MasterInstrument
                                                  .TradingHours?.Name ?? "(none)";
                    int cnt0 = BarsArray[0].Count;
                    Print($"[BreakoutStrategy] BIP0  {primaryRoot,-6} " +
                          $"{cnt0,6} bars  [primary]  session='{thName0}'");
                }

                Print($"[BreakoutStrategy] {_symbolToBip.Count} instruments mapped " +
                      $"across {BarsArray.Length} BIPs");

                // Allocate per-instrument state — one entry per BIP
                int count = BarsArray.Length;
                _states = new InstrumentState[count];
                _sarStates = new ReversalState[count];
                for (int i = 0; i < count; i++)
                {
                    _states[i] = new InstrumentState(VolumeAvgPeriod, 14, CnnLookbackBars);
                    _sarStates[i] = new ReversalState();
                    // Each BIP gets its own SessionIterator so Globex futures
                    // (which roll sessions at 18:00 ET, not midnight) reset their
                    // ORB and VWAP at the correct time rather than on calendar-date.
                    try { _states[i].SessionIter = new SessionIterator(BarsArray[i]); }
                    catch (Exception ex) { Print($"[BreakoutStrategy] SessionIterator BIP{i}: {ex.Message}"); }
                }

                // Warn if any BIP has suspiciously few bars — not enough to warm
                // the 14-bar ATR or the 20-bar volume average.
                int lowBarThreshold = Math.Max(VolumeAvgPeriod, 14) * 2; // 2× the longest lookback
                for (int i = 0; i < BarsArray.Length; i++)
                {
                    int c = BarsArray[i]?.Count ?? 0;
                    if (c > 0 && c < lowBarThreshold)
                    {
                        string n = BarsArray[i].Instrument.MasterInstrument.Name;
                        Print($"[BreakoutStrategy] ⚠ BIP{i} {n}: only {c} bars loaded " +
                              $"(need ≥{lowBarThreshold} to warm ATR+volume). " +
                              $"Consider increasing DaysToLoad (currently {DaysToLoad}).");
                    }
                }

                // ── Load per-type TP3 multipliers from feature_contract.json ──────────
                // The contract lives at models/feature_contract.json relative to the
                // NT8 Custom directory.  We look for it alongside the ONNX model.
                // Falls back to the global CTp3AtrMult constant for any missing type.
                try
                {
                    string modelDir = string.IsNullOrWhiteSpace(CnnModelPath)
                        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                                       "NinjaTrader 8", "bin", "Custom", "Models")
                        : Path.GetDirectoryName(CnnModelPath) ?? "";
                    string contractPath = Path.Combine(modelDir, "feature_contract.json");

                    if (File.Exists(contractPath))
                    {
                        string json = File.ReadAllText(contractPath);
                        _tp3MultByType = ParseTp3MultsFromContract(json);
                        Print($"[BreakoutStrategy] feature_contract.json loaded — " +
                              $"{_tp3MultByType.Count} per-type TP3 mults applied");

                        // Propagate to already-allocated RangeStates (if states built early)
                        if (_states != null)
                            ApplyTp3MultsToStates();
                    }
                    else
                    {
                        Print($"[BreakoutStrategy] feature_contract.json not found at '{contractPath}' " +
                              $"— using global CTp3AtrMult ({CTp3AtrMult}) for all types");
                    }
                }
                catch (Exception ex)
                {
                    Print($"[BreakoutStrategy] feature_contract.json parse error (non-fatal): {ex.Message}");
                }

                // ── CNN predictor ─────────────────────────────────────────────
                if (EnableCnnFilter)
                {
                    _cnnSnapshotFolder = string.IsNullOrWhiteSpace(CnnSnapshotFolder)
                        ? Path.Combine(Path.GetTempPath(), "NT8_OrbCnn")
                        : CnnSnapshotFolder;

                    try
                    {
                        Directory.CreateDirectory(_cnnSnapshotFolder);
                    }
                    catch (Exception ex)
                    {
                        Print($"[BreakoutStrategy] CNN snapshot folder error: {ex.Message}");
                        _cnnSnapshotFolder = Path.GetTempPath();
                    }

                    string modelPath = CnnModelPath?.Trim();
                    if (!string.IsNullOrEmpty(modelPath) && File.Exists(modelPath))
                    {
                        try
                        {
                            _cnn = new OrbCnnPredictor(modelPath, CnnUseCuda);
                            Print($"[BreakoutStrategy] CNN loaded: {modelPath}");
                        }
                        catch (Exception ex)
                        {
                            Print($"[BreakoutStrategy] CNN load failed — filter disabled: {ex.Message}");
                            _cnn = null;
                        }
                    }
                    else
                    {
                        Print($"[BreakoutStrategy] ⚠ CNN model not found at '{modelPath}' — filter disabled.");
                    }
                }

                if (EnableDebugLogging)
                {
                    Print("[Breakout DEBUG] === ALL TRACKED INSTRUMENTS ===");
                    foreach (var kv in _symbolToBip)
                        Print($"   BIP{kv.Value} → {kv.Key}");
                    Print($"   TPT Mode: {TptMode} | Tier: {AccountTier} | CNN: {(EnableCnnFilter && _cnn != null ? "ENABLED" : "DISABLED")}");
                    if (_cnn != null)
                    {
                        int modelDim = _cnn.NumTabular;
                        Print($"   CNN tabular dim: model expects {modelDim}, C# builds {CNumTabularFeatures}");
                    }

                    // ── CNN tabular dimension mismatch — warn loudly even outside debug mode ──
                    if (_cnn != null && _cnn.NumTabular != CNumTabularFeatures)
                    {
                        Print($"[BreakoutStrategy] ⚠ CNN TABULAR DIM MISMATCH: " +
                              $"model expects {_cnn.NumTabular} features, " +
                              $"C# builds {CNumTabularFeatures}. " +
                              $"The tabular vector will be silently truncated/zero-padded. " +
                              $"Re-train and re-export the ONNX model against " +
                              $"feature_contract.json v7.1 ({CNumTabularFeatures} features) " +
                              $"to resolve this. CNN predictions may be unreliable until fixed.");
                    }
                }

                // Instantiate order engine
                Account myAccount = null;
                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name == AccountName);

                if (myAccount != null)
                    Print($"[BreakoutStrategy] Account: {myAccount.Name}");
                else
                    Print($"[BreakoutStrategy] ⚠ Account '{AccountName}' not found");

                // Capture for closures
                Account capturedAccount = myAccount;

                // Apply TP3 mults to states now that they're allocated
                if (_states != null && _tp3MultByType.Count > 0)
                    ApplyTp3MultsToStates();

                // ── MTF BIP map — pair each 1m BIP with its 15m sibling ─────────
                // The 15m series are added in Configure immediately after the 1m series
                // for each extra symbol.  They land at BIP indices N+extra_count onward.
                // We match by instrument name: find every BIP whose BarsPeriod is 15m
                // and store it in _mtfBipBySymbol keyed by instrument root name.
                _mtfBipBySymbol.Clear();
                for (int bipIdx = 0; bipIdx < BarsArray.Length; bipIdx++)
                {
                    try
                    {
                        var bipBars = BarsArray[bipIdx];
                        if (bipBars == null) continue;
                        var bp = bipBars.BarsPeriod;
                        if (bp != null &&
                            bp.BarsPeriodType == BarsPeriodType.Minute &&
                            bp.Value == 15)
                        {
                            string rootName = bipBars.Instrument.MasterInstrument.Name;
                            if (!_mtfBipBySymbol.ContainsKey(rootName))
                            {
                                _mtfBipBySymbol[rootName] = bipIdx;
                                // Wire the MtfBip reference into the corresponding 1m state
                                if (_states != null)
                                {
                                    // Find the 1m state for this symbol
                                    int oneMBip;
                                    if (_symbolToBip.TryGetValue(rootName, out oneMBip) &&
                                        oneMBip < _states.Length)
                                    {
                                        _states[oneMBip].MtfBip = bipIdx;
                                    }
                                    else if (rootName.Equals(
                                        Instrument?.MasterInstrument?.Name ?? "",
                                        StringComparison.OrdinalIgnoreCase) &&
                                        _states.Length > 0)
                                    {
                                        _states[0].MtfBip = bipIdx;
                                    }
                                }
                                Print($"[BreakoutStrategy] MTF BIP{bipIdx} → {rootName} (15m)");
                            }
                        }
                    }
                    catch { /* non-fatal — MTF will fall back to 1.0 score for this symbol */ }
                }

                // ── SAR HTTP client (fire-and-forget push to Python engine) ─────
                try
                {
                    _sarHttpClient = new System.Net.Http.HttpClient
                    {
                        Timeout = TimeSpan.FromSeconds(3)
                    };
                    _sarHttpClient.DefaultRequestHeaders.Add("User-Agent", "NinjaTrader-SAR/1.0");
                    Print($"[BreakoutStrategy] SAR sync client initialised → {CSarSyncUrl}");
                }
                catch (Exception ex)
                {
                    Print($"[BreakoutStrategy] SAR HTTP client init failed (non-fatal): {ex.Message}");
                    _sarHttpClient = null;
                }

                _engine = new BridgeOrderEngine(
                    strategy: this,
                    symbolToBip: _symbolToBip,
                    getMyAccount: () => capturedAccount,
                    getAccountSize: () => AccountSize,
                    getRiskPercent: () => RiskPercentPerTrade,
                    getMaxContracts: () => MaxContracts,
                    getDefaultSlTicks: () => DefaultSlTicks,
                    getDefaultTpTicks: () => DefaultTpTicks,
                    getAutoBrackets: () => EnableAutoBrackets,
                    getRiskBlocked: () => RiskBlocked,
                    getRiskBlockReason: () => RiskBlockReason,
                    getRiskEnforcement: () => false, // BreakoutStrategy manages its own gate
                    onSignalReceived: () => Interlocked.Increment(ref _metricSignalsReceived),
                    onSignalExecuted: () => Interlocked.Increment(ref _metricSignalsExecuted),
                    onSignalRejected: () => Interlocked.Increment(ref _metricSignalsRejected),
                    onExitExecuted: () => Interlocked.Increment(ref _metricExitsExecuted),
                    onBusDrained: n => Interlocked.Add(ref _metricBusDrained, n),
                    sendPositionUpdate: () => { }, // BreakoutStrategy has no dashboard push
                    tag: "Breakout"
                );
            }

            // ── Realtime ──────────────────────────────────────────────────────
            // All BIPs should have historical bars loaded by the time the strategy
            // transitions to Realtime. Log each one so we can spot any that are
            // still at 0 (data subscription problem) vs properly loaded.
            else if (State == State.Realtime)
            {
                if (EnableDebugLogging && _states != null)
                {
                    Print("[Breakout DEBUG] === BIP BAR COUNTS AT REALTIME ===");
                    for (int i = 0; i < BarsArray.Length; i++)
                    {
                        string bipName = BarsArray[i].Instrument.MasterInstrument.Name;
                        int bipCount = BarsArray[i].Count;
                        string status = bipCount > 0 ? "OK" : "⚠ NO DATA";
                        Print($"[Breakout DEBUG]   BIP{i} {bipName,-6} {bipCount,6} bars  [{status}]");
                    }
                }
            }

            // ── Terminated ────────────────────────────────────────────────────
            else if (State == State.Terminated)
            {
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                    SignalBus.UnregisterConsumer();

                _cnn?.Dispose();
                _cnn = null;
                _engine = null;
                _states = null;
                _sarStates = null;
                try { _sarHttpClient?.Dispose(); } catch { }
                _sarHttpClient = null;
                _mtfBipBySymbol.Clear();
                lock (_phaseLock)
                    _positionPhases.Clear();
            }
        }

        #endregion


        // =====================================================================
        // Order / Execution / Position Update Handlers
        // =====================================================================
        #region Order Handlers

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice, OrderState orderState,
            DateTime time, ErrorCode error, string nativeError)
        {
            // ── SAR: close the ReversalState when a flatten or SL/TP fills ────
            // This keeps the SAR direction in sync with NT8's actual position.
            // We check for both the SAR flatten name prefix and the normal exit prefixes.
            if (orderState == OrderState.Filled)
            {
                string instrName = order.Instrument?.MasterInstrument?.Name ?? "";
                if (!string.IsNullOrEmpty(instrName) && _sarStates != null)
                {
                    bool isFlatOrder = order.Name != null && (
                        order.Name.StartsWith("SAR-Flat-") ||
                        order.Name.StartsWith("Flatten") ||
                        order.Name.StartsWith("Phase3Exit-") ||
                        order.Name.StartsWith("SL-") ||
                        order.Name.StartsWith("TP1-") ||
                        order.Name.StartsWith("TP2-"));

                    if (isFlatOrder)
                    {
                        // Find the BIP for this instrument and close its SAR state
                        // if NT8 shows the position as now flat.
                        for (int sarBip = 0; sarBip < BarsArray.Length && sarBip < _sarStates.Length; sarBip++)
                        {
                            if (BarsArray[sarBip] != null &&
                                BarsArray[sarBip].Instrument.MasterInstrument.Name == instrName)
                            {
                                try
                                {
                                    var ntPos = Positions[sarBip];
                                    if (ntPos == null || ntPos.MarketPosition == MarketPosition.Flat)
                                        _sarStates[sarBip]?.Close();
                                }
                                catch { /* Positions[] may throw on historical replay — harmless */ }
                                break;
                            }
                        }
                    }
                }
            }

            // ── Rejected orders: log and absorb instead of letting NT8 kill the strategy ──
            if (orderState == OrderState.Rejected)
            {
                Interlocked.Increment(ref _metricOrdersRejected);
                Print($"[Breakout] ⚠ ORDER REJECTED: {order.Name} {order.Instrument.FullName} " +
                      $"Action={order.OrderAction} Type={order.OrderType} " +
                      $"Limit={limitPrice} Stop={stopPrice} Qty={quantity} " +
                      $"Error={error} Native=\"{nativeError}\"");

                // For bracket orders (SL/TP), a rejection means the protective
                // order didn't get placed.  Log a warning but do NOT terminate.
                // The position is still open and the other leg of the bracket
                // may still be working.  The strategy should continue running.
                if (order.Name.StartsWith("SL-") || order.Name.StartsWith("TP1-") || order.Name.StartsWith("TP2-"))
                {
                    Print($"[Breakout] ⚠ BRACKET LEG REJECTED — position may be unprotected. " +
                          $"Instrument={order.Instrument.FullName} Order={order.Name}");
                }
                return; // absorb the rejection — do NOT let NT8 terminate the strategy
            }

            // ── Track fills for position counting ─────────────────────────────
            if (orderState == OrderState.Filled)
            {
                string instrName = order.Instrument.MasterInstrument.Name;

                // Entry fills: track position count + update PositionPhase actual qty
                if (order.Name.StartsWith("Signal-"))
                {
                    lock (_activeInstruments)
                    {
                        if (_activeInstruments.Add(instrName))
                            _activePositionCount = _activeInstruments.Count;
                    }

                    // Update PositionPhase with actual fill quantity so the phase
                    // split (tp1Qty / tp2Qty / tp3Qty) is based on real fills,
                    // not the estimated qty from FireEntry.
                    if (EnableTp3Trailing && filled > 0)
                    {
                        string signalId = ExtractSignalId(order.Name, "Signal-");
                        if (!string.IsNullOrEmpty(signalId))
                        {
                            lock (_phaseLock)
                            {
                                if (_positionPhases.TryGetValue(signalId, out var ph))
                                {
                                    int totalQty = filled;
                                    int tp1Qty = ph.Tp2Price > 0 ? Math.Max(1, totalQty / 2) : totalQty;
                                    int tp2Qty = ph.Tp2Price > 0 ? totalQty - tp1Qty : 0;
                                    int tp3Qty = ph.Tp3Price > 0 ? tp2Qty : 0;
                                    ph.TotalQty = totalQty;
                                    ph.Tp1Qty = tp1Qty;
                                    ph.Tp2Qty = tp2Qty;
                                    ph.Tp3Qty = tp3Qty;
                                    if (EnableDebugLogging)
                                        Print($"[Phase] Entry filled: id={signalId} qty={totalQty} " +
                                              $"tp1Qty={tp1Qty} tp2Qty={tp2Qty} tp3Qty={tp3Qty}");
                                }
                            }
                        }
                    }
                }

                // ── TP1 fill → Phase2: move SL to breakeven, let TP2 ride ────
                else if (order.Name.StartsWith("TP1-") && EnableTp3Trailing)
                {
                    string signalId = ExtractSignalId(order.Name, "TP1-");
                    if (!string.IsNullOrEmpty(signalId))
                    {
                        PositionPhase ph;
                        lock (_phaseLock)
                            _positionPhases.TryGetValue(signalId, out ph);

                        if (ph != null && ph.Phase == BreakoutPhase.Phase1 && ph.Tp2Price > 0 && ph.Tp2Qty > 0)
                        {
                            ph.Phase = BreakoutPhase.Phase2;

                            // Move the existing SL to breakeven by cancelling the old OCO
                            // SL order and submitting a new one at entry price.
                            double bePrice = ph.EntryPrice;
                            double tickSize = _engine.GetTickSize(ph.Bip);
                            // Add/subtract one tick so the stop is valid (not exactly at entry)
                            double newSl = ph.Direction == "long"
                                ? bePrice - tickSize
                                : bePrice + tickSize;
                            ph.SlPrice = newSl;

                            // Determine exit action
                            OrderAction exitAction = ph.Direction == "long"
                                ? OrderAction.Sell
                                : OrderAction.BuyToCover;

                            string newOco = "OCO-BE-" + signalId + "-" +
                                            Guid.NewGuid().ToString("N").Substring(0, 6);
                            ph.OcoGroup = newOco;

                            Print($"[Phase] TP1 HIT → Phase2 | id={signalId} BE_SL={newSl:F6} " +
                                  $"TP2={ph.Tp2Price:F6} qty={ph.Tp2Qty}");

                            try
                            {
                                // Submit new breakeven SL (remaining qty)
                                SubmitOrderUnmanaged(ph.Bip, exitAction,
                                    OrderType.StopMarket, ph.Tp2Qty,
                                    0, newSl, newOco, "SL-BE-" + signalId);

                                // TP2 limit already in flight from initial bracket —
                                // it will fill or SL will close the remainder.
                                // No need to re-submit TP2 (the original TP2-* order
                                // is still working from ExecuteEntryDirect/ProcessSignal).
                            }
                            catch (Exception ex)
                            {
                                Print($"[Phase] Phase2 SL submission failed for {signalId}: {ex.Message}");
                            }
                        }
                    }
                }

                // ── TP2 fill → Phase3: trail remainder with EMA9 toward TP3 ──
                else if (order.Name.StartsWith("TP2-") && EnableTp3Trailing)
                {
                    string signalId = ExtractSignalId(order.Name, "TP2-");
                    if (!string.IsNullOrEmpty(signalId))
                    {
                        PositionPhase ph;
                        lock (_phaseLock)
                            _positionPhases.TryGetValue(signalId, out ph);

                        if (ph != null && ph.Phase == BreakoutPhase.Phase2 && ph.Tp3Price > 0 && ph.Tp3Qty > 0)
                        {
                            ph.Phase = BreakoutPhase.Phase3;
                            // Reset the ratchet so CheckPhase3Exits initialises it from the
                            // live EMA9 on the very first bar it runs in Phase3.
                            ph.Ema9TrailPrice = 0;
                            Print($"[Phase] TP2 HIT → Phase3 | id={signalId} TP3={ph.Tp3Price:F6} " +
                                  $"qty={ph.Tp3Qty}  EMA9 trailing active");
                            // Phase3 exit is handled by CheckPhase3Exits() on each bar.
                        }
                        else if (ph != null)
                        {
                            // TP2 filled but no TP3 configured or already closed
                            ph.Phase = BreakoutPhase.Closed;
                            lock (_phaseLock)
                                _positionPhases.Remove(signalId);
                        }
                    }
                }

                // ── SL / BE-SL fills: mark position closed ───────────────────
                else if (order.Name.StartsWith("SL-") && EnableTp3Trailing)
                {
                    // Extract signal ID — handles both "SL-<id>" and "SL-BE-<id>"
                    string signalId = order.Name.StartsWith("SL-BE-")
                        ? order.Name.Substring("SL-BE-".Length)
                        : ExtractSignalId(order.Name, "SL-");

                    if (!string.IsNullOrEmpty(signalId))
                    {
                        lock (_phaseLock)
                        {
                            if (_positionPhases.TryGetValue(signalId, out var ph))
                            {
                                Print($"[Phase] SL HIT → Closed | id={signalId} phase={ph.Phase}");
                                ph.Phase = BreakoutPhase.Closed;
                                _positionPhases.Remove(signalId);
                            }
                        }
                    }
                }

                // ── Exit fills (SL/TP/Flatten): update active position count ──
                if (order.Name.StartsWith("SL-") || order.Name.StartsWith("TP") ||
                    order.Name.StartsWith("Flatten") || order.Name.StartsWith("Phase3Exit-"))
                {
                    try
                    {
                        for (int i = 0; i < BarsArray.Length; i++)
                        {
                            if (BarsArray[i] != null &&
                                BarsArray[i].Instrument.MasterInstrument.Name == instrName)
                            {
                                var pos = Positions[i];
                                if (pos == null || pos.MarketPosition == MarketPosition.Flat)
                                {
                                    lock (_activeInstruments)
                                    {
                                        _activeInstruments.Remove(instrName);
                                        _activePositionCount = _activeInstruments.Count;
                                    }
                                }
                                break;
                            }
                        }
                    }
                    catch { /* counter may drift; self-corrects on next fill */ }
                }
            }
        }

        /// <summary>
        /// Extract the SignalId suffix from an order name by stripping a known prefix.
        /// Returns empty string if the name does not start with the prefix.
        /// </summary>
        private string ExtractSignalId(string orderName, string prefix)
        {
            if (string.IsNullOrEmpty(orderName) || !orderName.StartsWith(prefix))
                return string.Empty;
            return orderName.Substring(prefix.Length);
        }

        /// <summary>
        /// Called once per primary bar (BIP0) when EnableTp3Trailing is true.
        /// For every position in Phase3:
        ///   1. Ratchet Ema9TrailPrice forward (only in the favourable direction),
        ///      mirroring Python PositionManager._update_bracket_phase(TRAILING).
        ///   2. Check TP3 hit (hard cap) — checked FIRST, same as Python update_all.
        ///   3. Check adverse EMA9 cross using the RATCHETED trail price (not raw EMA9),
        ///      so a brief dip in EMA9 during a choppy trend cannot pull the stop back.
        /// Removes closed phases from the dictionary.
        /// </summary>
        private void CheckPhase3Exits()
        {
            if (_positionPhases == null || _positionPhases.Count == 0) return;

            List<string> toRemove = null;

            lock (_phaseLock)
            {
                foreach (var kv in _positionPhases)
                {
                    var ph = kv.Value;
                    if (ph.Phase != BreakoutPhase.Phase3) continue;
                    if (ph.Tp3Submitted || ph.Ema9StopHit) continue;
                    if (ph.Bip >= _states.Length) continue;

                    var st = _states[ph.Bip];
                    if (!st.Ema9Ready) continue;

                    var bars = BarsArray[ph.Bip];
                    if (bars == null || bars.Count < 1) continue;

                    int last = bars.Count - 1;
                    double close = bars.GetClose(last);
                    double ema9 = st.Ema9Value;

                    // ── Step 1: Ratchet the trail price (mirrors Python ema9_trail_price) ──
                    // Initialise on the first bar of Phase3 (Ema9TrailPrice == 0).
                    if (ph.Ema9TrailPrice <= 0)
                    {
                        ph.Ema9TrailPrice = ema9;
                        if (EnableDebugLogging)
                            Print($"[Phase3] Trail INIT | id={ph.SignalId} EMA9={ema9:F6}");
                    }
                    else
                    {
                        bool shouldRatchet = ph.Direction == "long"
                            ? ema9 > ph.Ema9TrailPrice   // long: only move trail UP
                            : ema9 < ph.Ema9TrailPrice;  // short: only move trail DOWN
                        if (shouldRatchet)
                        {
                            if (EnableDebugLogging)
                                Print($"[Phase3] Trail ratchet | id={ph.SignalId} " +
                                      $"{ph.Ema9TrailPrice:F6} → {ema9:F6}");
                            ph.Ema9TrailPrice = ema9;
                        }
                    }

                    double trailStop = ph.Ema9TrailPrice;

                    OrderAction exitAction = ph.Direction == "long"
                        ? OrderAction.Sell
                        : OrderAction.BuyToCover;

                    // ── Step 2: TP3 hard exit (checked before EMA9 stop, same as Python) ──
                    bool tp3Hit = ph.Tp3Price > 0 && (ph.Direction == "long"
                        ? close >= ph.Tp3Price
                        : close <= ph.Tp3Price);

                    // ── Step 3: Adverse cross against the RATCHETED trail price ──
                    // Using trailStop (not raw ema9) means a momentary EMA9 dip during
                    // a choppy trend cannot pull the stop price backward, matching Python.
                    bool ema9Stop = ph.Direction == "long"
                        ? close < trailStop     // long: exit when close drops below trail
                        : close > trailStop;    // short: exit when close rises above trail

                    if (tp3Hit && !ph.Tp3Submitted)
                    {
                        // Price reached TP3 — submit limit exit
                        ph.Tp3Submitted = true;
                        Print($"[Phase3] TP3 REACHED | id={ph.SignalId} price={close:F6} " +
                              $"TP3={ph.Tp3Price:F6} trail={trailStop:F6} qty={ph.Tp3Qty}");
                        try
                        {
                            SubmitOrderUnmanaged(ph.Bip, exitAction,
                                OrderType.Limit, ph.Tp3Qty,
                                ph.Tp3Price, 0, "",
                                "Phase3Exit-" + ph.SignalId);
                        }
                        catch (Exception ex)
                        {
                            Print($"[Phase3] TP3 exit failed for {ph.SignalId}: {ex.Message}");
                            ph.Tp3Submitted = false;
                        }
                        ph.Phase = BreakoutPhase.Closed;
                        if (toRemove == null) toRemove = new List<string>();
                        toRemove.Add(ph.SignalId);
                    }
                    else if (ema9Stop)
                    {
                        // Close crossed the ratcheted EMA9 trail adversely — market exit
                        ph.Ema9StopHit = true;
                        Print($"[Phase3] EMA9 TRAIL STOP | id={ph.SignalId} price={close:F6} " +
                              $"trail={trailStop:F6} (raw EMA9={ema9:F6}) qty={ph.Tp3Qty}");
                        try
                        {
                            SubmitOrderUnmanaged(ph.Bip, exitAction,
                                OrderType.Market, ph.Tp3Qty,
                                0, 0, "",
                                "Phase3Exit-" + ph.SignalId);
                        }
                        catch (Exception ex)
                        {
                            Print($"[Phase3] EMA9 exit failed for {ph.SignalId}: {ex.Message}");
                            ph.Ema9StopHit = false;
                        }
                        ph.Phase = BreakoutPhase.Closed;
                        if (toRemove == null) toRemove = new List<string>();
                        toRemove.Add(ph.SignalId);
                    }
                }

                if (toRemove != null)
                    foreach (var id in toRemove)
                        _positionPhases.Remove(id);
            }
        }

        #endregion

        // =====================================================================
        // Bar Update
        // =====================================================================
        #region Bar Update

        protected override void OnBarUpdate()
        {
            if (_engine == null || _states == null) return;

            // ── Wait for all BIPs to finish loading historical bars ────────────
            // During the transition to Realtime, secondary BIPs can still be
            // streaming in their history.  We spin here (returning early) until
            // either DataLoadWaitSeconds have elapsed AND every BIP has at least
            // 800 bars (~13 h of 1-min data, enough to warm ATR + volume avg),
            // or until DataLoadWaitSeconds elapsed even if some BIPs are thin
            // (prevents hanging forever when a symbol has genuinely sparse data).
            if (!_allDataLoaded && BarsInProgress == 0)
            {
                if (_realtimeStart == DateTime.MinValue)
                    _realtimeStart = DateTime.Now;

                double elapsed = (DateTime.Now - _realtimeStart).TotalSeconds;

                if (elapsed < DataLoadWaitSeconds)
                {
                    // Log download progress every 15 seconds so it's clear data is arriving.
                    if ((DateTime.Now - _lastProgressLog).TotalSeconds >= 15)
                    {
                        _lastProgressLog = DateTime.Now;
                        int loaded = 0, empty = 0;
                        for (int i = 0; i < BarsArray.Length; i++)
                        {
                            if (BarsArray[i] != null && BarsArray[i].Count > 0) loaded++;
                            else empty++;
                        }
                        Print($"[BreakoutStrategy] ⏳ Waiting for history... " +
                              $"{loaded}/{BarsArray.Length} BIPs loaded  " +
                              $"({(int)elapsed}s / {DataLoadWaitSeconds}s)");
                        if (EnableDebugLogging)
                        {
                            for (int i = 0; i < BarsArray.Length; i++)
                            {
                                int c = BarsArray[i]?.Count ?? 0;
                                if (c == 0)
                                {
                                    string n = BarsArray[i]?.Instrument?.MasterInstrument?.Name ?? "?";
                                    Print($"[Breakout DEBUG]   BIP{i} {n,-6} still 0 bars");
                                }
                            }
                        }
                    }
                    return; // still within the wait window — keep polling
                }

                // Time window expired: check whether all BIPs are adequately loaded.
                bool allOk = true;
                for (int i = 0; i < BarsArray.Length; i++)
                {
                    if (BarsArray[i] == null || BarsArray[i].Count < 200)
                    {
                        allOk = false;
                        break;
                    }
                }

                // After the wait window expires, accept what we have.  BIPs
                // that are still at 0 bars are added to _skippedBips so
                // OnBarUpdate ignores them instead of stalling the strategy.
                _allDataLoaded = allOk || elapsed >= DataLoadWaitSeconds;

                if (_allDataLoaded)
                {
                    int okCount = 0, lowCount = 0, zeroCount = 0;
                    _skippedBips.Clear();
                    for (int i = 0; i < BarsArray.Length; i++)
                    {
                        int c = BarsArray[i]?.Count ?? 0;
                        if (c >= 200) okCount++;
                        else if (c > 0) lowCount++;
                        else
                        {
                            zeroCount++;
                            // Skip BIP0 (primary chart) — it will always eventually
                            // get data.  Only skip secondary BIPs with no data sub.
                            if (i > 0)
                            {
                                _skippedBips.Add(i);
                                string sn = BarsArray[i]?.Instrument?.MasterInstrument?.Name ?? "?";
                                Print($"[BreakoutStrategy] ⏭ Skipping BIP{i} {sn} — 0 bars after {(int)elapsed}s wait. " +
                                      $"Check data subscription or move to CPendingInstruments.");
                            }
                        }
                    }
                    Print($"[BreakoutStrategy] ✅ Preflight complete — " +
                          $"{okCount} OK, {lowCount} low-bar, {zeroCount} empty ({_skippedBips.Count} skipped)  " +
                          $"(waited {(int)elapsed}s)");
                    Print("[Breakout DEBUG] === POST-WAIT BIP BAR COUNTS ===");
                    for (int i = 0; i < BarsArray.Length; i++)
                    {
                        string n = BarsArray[i]?.Instrument?.MasterInstrument?.Name ?? "?";
                        int c = BarsArray[i]?.Count ?? 0;
                        string status = c >= 200 ? "OK" : c > 0 ? "⚠ LOW"
                            : _skippedBips.Contains(i) ? "⏭ SKIPPED" : "❌ EMPTY";
                        Print($"[Breakout DEBUG]   BIP{i} {n,-6} {c,6} bars  [{status}]");
                    }
                }
                else
                {
                    return; // not yet loaded and still within wait window
                }
            }

            int bip = BarsInProgress;

            // Guard: state array bounds (can lag BarsArray during early init)
            if (bip >= _states.Length) return;

            // Skip BIPs that were marked as no-data after the wait window.
            // This prevents 0-bar instruments from polluting ORB/breakout
            // logic or generating spurious log noise every bar.
            if (_skippedBips.Contains(bip)) return;

            var st = _states[bip];

            // ── SAR sync: detect external position closes (TP/SL fills) ─────
            // OnOrderUpdate already handles fill-driven SAR.Close(), but as a
            // belt-and-suspenders check we also reconcile with the live NT8
            // Positions[] every bar.  This catches edge cases where the fill
            // callback fires out of order or during historical replay.
            if (_sarStates != null && bip < _sarStates.Length)
            {
                var sarSync = _sarStates[bip];
                if (!sarSync.IsFlat)
                {
                    try
                    {
                        var ntPos = Positions[bip];
                        if (ntPos == null || ntPos.MarketPosition == MarketPosition.Flat)
                        {
                            if (EnableDebugLogging)
                                Print($"[SAR] BIP{bip} position closed externally — resetting SAR state " +
                                      $"(was {sarSync.ActiveDirection})");
                            sarSync.Close();
                        }
                    }
                    catch { /* Positions[] may throw during historical replay */ }
                }
            }

            // ── First-bar log: fires once per BIP the first time OnBarUpdate
            // is called for it, confirming the data subscription is alive.
            if (EnableDebugLogging && st.LastBarProcessed == -1 && BarsArray[bip].Count > 0)
            {
                string bipName = BarsArray[bip].Instrument.MasterInstrument.Name;
                int bipCount = BarsArray[bip].Count;
                Print($"[Breakout DEBUG] BIP{bip} {bipName} first bar received ({bipCount} bars loaded)");
            }

            // ── Per-bar indicator updates (once per closed bar, per BIP) ──────
            // IsFirstTickOfBar is unreliable for secondary data series (BIP > 0)
            // and can also miss bars when the strategy starts mid-bar.  Instead
            // we track the last processed bar index per-instrument state and
            // advance whenever Count - 2 has moved past it.  This is robust
            // across all BIPs, all Calculate modes, and mid-bar startup.
            var bipBars = BarsArray[bip];
            if (bipBars.Count >= 2)
            {
                int closedIdx = bipBars.Count - 2;
                if (closedIdx > st.LastBarProcessed)
                {
                    st.LastBarProcessed = closedIdx;

                    // Snapshot the closed bar's volume so CheckBreakout can compare
                    // against it even after UpdateVolumeAvg advances the ring-buffer.
                    st.LastVolume = bipBars.GetVolume(closedIdx);

                    UpdateVwap(bip, st);
                    UpdateAtr(bip, st);
                    UpdateVolumeAvg(bip, st);
                    UpdateEma9(bip, st);
                    // UpdateMtf is called separately for the 15m BIP below,
                    // NOT here (this path only runs for 1m BIPs).

                    // ── Multi-range window accumulation ───────────────────────
                    if (Mode == BreakoutMode.BuiltIn || Mode == BreakoutMode.Both)
                    {
                        // ORB must run first — PrevDay reads PrevOrbHigh/Low
                        // which are snapshotted inside UpdateRangeWindow(ORB).
                        UpdateRangeWindow(bip, st, BreakoutType.ORB);
                        UpdateRangeWindow(bip, st, BreakoutType.PrevDay);
                        UpdateRangeWindow(bip, st, BreakoutType.InitialBalance);
                        UpdateRangeWindow(bip, st, BreakoutType.Consolidation);
                        UpdateRangeWindow(bip, st, BreakoutType.Weekly);
                        UpdateRangeWindow(bip, st, BreakoutType.Monthly);
                        UpdateRangeWindow(bip, st, BreakoutType.Asian);
                        UpdateRangeWindow(bip, st, BreakoutType.BollingerSqueeze);
                        UpdateRangeWindow(bip, st, BreakoutType.ValueArea);
                        UpdateRangeWindow(bip, st, BreakoutType.InsideDay);
                        UpdateRangeWindow(bip, st, BreakoutType.GapRejection);
                        UpdateRangeWindow(bip, st, BreakoutType.PivotPoints);
                        UpdateRangeWindow(bip, st, BreakoutType.Fibonacci);
                    }
                }
            }

            // ── 15m MTF update: if this BIP is a 15m series, update the
            // corresponding 1m InstrumentState's MTF fields ──────────────────
            if (_states != null)
            {
                try
                {
                    var thisBars = BarsArray[bip];
                    if (thisBars != null)
                    {
                        var bp = thisBars.BarsPeriod;
                        if (bp != null && bp.BarsPeriodType == BarsPeriodType.Minute && bp.Value == 15)
                        {
                            // Find the 1m InstrumentState for this symbol
                            string mtfRoot = thisBars.Instrument.MasterInstrument.Name;
                            int oneMBip;
                            InstrumentState mtfSt = null;
                            if (_symbolToBip.TryGetValue(mtfRoot, out oneMBip) && oneMBip < _states.Length)
                                mtfSt = _states[oneMBip];
                            else if (mtfRoot.Equals(
                                    Instrument?.MasterInstrument?.Name ?? "",
                                    StringComparison.OrdinalIgnoreCase) &&
                                _states.Length > 0)
                                mtfSt = _states[0];

                            if (mtfSt != null)
                            {
                                int closedMtfIdx = thisBars.Count - 2;
                                if (closedMtfIdx > 0 && closedMtfIdx > mtfSt.LastBarProcessed - 1)
                                    UpdateMtf(bip, mtfSt);
                            }
                        }
                    }
                }
                catch { /* non-fatal — MTF update never blocks bar processing */ }
            }

            // ── Primary-series-only logic ─────────────────────────────────────
            if (bip == 0)
            {
                // Drain SignalBus (relay mode)
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                    _engine.DrainSignalBus(State);

                // Flush queued orders (Realtime only — backtest uses DirectExecute)
                _engine.FlushOrderQueue(State);

                // ── Sync risk gate from Bridge AddOn via SignalBus ────────────
                // Bridge.cs writes SignalBus.IsRiskBlocked / RiskBlockReason when
                // the dashboard's risk endpoint returns can_trade=false.
                // We read it once per primary bar — no reflection, no lock needed
                // (SignalBus fields are volatile; Bridge is the sole writer).
                // Works in backtest too: SignalBus.IsRiskBlocked defaults false.
                {
                    bool busBlocked = NinjaTrader.NinjaScript.SignalBus.IsRiskBlocked;
                    if (busBlocked != RiskBlocked)
                    {
                        RiskBlocked = busBlocked;
                        RiskBlockReason = NinjaTrader.NinjaScript.SignalBus.RiskBlockReason;
                        if (EnableDebugLogging)
                            Print($"[BreakoutStrategy] Risk gate → " +
                                  $"RiskBlocked={RiskBlocked}" +
                                  (RiskBlocked ? $" reason={RiskBlockReason}" : ""));
                    }
                }

                // ── Phase3 EMA9 trailing stop check ──────────────────────────
                // For each position in Phase3, check if price has crossed EMA9
                // adversely or reached TP3.  If so, submit a market exit.
                if (EnableTp3Trailing)
                    CheckPhase3Exits();

                // ── TPT hard stop — 4:00 PM ET flatten ───────────────────────
                // Take Profit Trader funded accounts prohibit overnight positions.
                // Holding past 4 PM ET = account termination.  This guard fires
                // on every primary bar once ET time ≥ 16:00 and any positions are
                // still open, and it also lifts the risk block at 18:00 ET when
                // the new Globex session is about to open.
                if (TptMode)
                    CheckTptHardStop();
            }

            // ── Breakout detection (runs for every BIP in built-in modes) ─────
            if (Mode == BreakoutMode.BuiltIn || Mode == BreakoutMode.Both)
                CheckBreakout(bip, st);
        }

        #endregion

        // =====================================================================
        // TPT hard stop — 4:00 PM ET session close safety
        // =====================================================================
        #region TPT hard stop

        /// <summary>
        /// Enforce Take Profit Trader's no-overnight-position rule.
        ///
        /// Called once per primary bar (BIP0) when TptMode == true.
        ///
        /// Behaviour:
        ///   16:00–17:59 ET — if any micro positions are open, flatten them all
        ///                     immediately and set RiskBlocked with reason
        ///                     "TPT_SESSION_CLOSED" until 18:00 ET.
        ///   18:00+ ET       — lift the TPT_SESSION_CLOSED block so the strategy
        ///                     can trade the new Globex session.
        ///
        /// The flatten fires on EVERY bar in the 16:00–17:59 window until the
        /// position count hits zero — this handles partial-fill edge cases where
        /// the first FlattenAll call only closes some legs.
        /// </summary>
        private void CheckTptHardStop()
        {
            try
            {
                // Convert the current bar's time to Eastern Time
                DateTime barTimeET = TimeZoneInfo.ConvertTimeFromUtc(
                    Time[0].Kind == DateTimeKind.Utc
                        ? Time[0]
                        : DateTime.SpecifyKind(Time[0], DateTimeKind.Utc),
                    TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));

                int hour = barTimeET.Hour;

                // ── Re-enable trading at 18:00 ET (new Globex session) ────────
                if (hour >= 18 && RiskBlockReason == "TPT_SESSION_CLOSED")
                {
                    RiskBlocked = false;
                    RiskBlockReason = "";
                    Print($"[TPT] Risk gate LIFTED at {barTimeET:HH:mm} ET — new Globex session open");
                    return;
                }

                // ── Hard flatten at 16:00 ET ──────────────────────────────────
                if (hour >= 16 && hour < 18)
                {
                    // Set the risk block immediately so no new entries fire
                    // even if the flatten takes a bar or two to confirm
                    if (!RiskBlocked || RiskBlockReason != "TPT_SESSION_CLOSED")
                    {
                        RiskBlocked = true;
                        RiskBlockReason = "TPT_SESSION_CLOSED";
                        Print($"[TPT] HARD STOP — session closed at {barTimeET:HH:mm} ET. " +
                              $"Risk gate BLOCKED (reason=TPT_SESSION_CLOSED). " +
                              $"No new entries until 18:00 ET.");
                    }

                    // Flatten if any positions are still open
                    if (_activePositionCount > 0)
                    {
                        Print($"[TPT] HARD STOP — flattening all {_activePositionCount} open position(s) " +
                              $"at {barTimeET:HH:mm} ET (no overnight holds allowed on TPT accounts)");

                        try
                        {
                            // FlattenAll submits market orders on all instruments
                            // in the strategy's account to close every open position.
                            _engine?.FlattenAll("TPT_HARD_STOP_16:00");
                        }
                        catch (Exception flatEx)
                        {
                            Print($"[TPT] FlattenAll error: {flatEx.Message} — retrying next bar");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Swallow: never let a timezone conversion failure crash the strategy
                Print($"[TPT] CheckTptHardStop error: {ex.Message}");
            }
        }

        #endregion

        // =====================================================================
        // Stop-and-reverse helpers
        // =====================================================================
        #region SAR helpers

        /// <summary>
        /// Evaluate reversal gates for an existing position.
        /// Mirrors Python PositionManager._should_reverse().
        ///
        /// Gate 1 — Direction must be opposite.
        /// Gate 2 — CNN probability ≥ threshold (higher bar for winning positions).
        /// Gate 3 — Cooldown since last entry/reversal.
        /// Gate 4 — MTF alignment score.
        /// Gate 5 — Winning position at 1R+: need CNN ≥ 0.95.
        /// </summary>
        /// <summary>
        /// Compute the real MTF score for a given instrument state and direction.
        ///
        /// UpdateMtf() stores two sentinel values in st.MtfScore:
        ///   1.0   — EMA or MACD not yet warmed (< 50 bars seen); pass-through gate.
        ///   -1.0  — Both EMA and MACD are warmed up; ComputeMtfScore() must be called
        ///           with the caller's direction to get the true 0.0–1.0 score.
        ///
        /// Any other value should not appear but is treated as pass-through to be safe.
        /// </summary>
        private double GetMtfScore(InstrumentState st, string direction)
        {
            if (st == null) return 1.0;

            // Sentinel -1: EMA+MACD warmed — compute the real directional score now.
            if (st.MtfScore < 0) return ComputeMtfScore(st, direction);

            // Sentinel 1.0 (or any positive value set during warm-up): pass-through.
            // This covers the initial default (0.0 from field initialiser) and the
            // explicit 1.0 written by UpdateMtf while bars are still accumulating.
            return st.MtfScore > 0 ? st.MtfScore : 1.0;
        }

        private bool ShouldReverse(
            ReversalState sar,
            string newDirection,
            double cnnProb,
            double mtfScore,
            double currentPrice,
            DateTime barTime)
        {
            // Gate 1: must be opposite direction
            if (newDirection == sar.ActiveDirection)
                return false;

            // Gate 2: CNN probability threshold
            bool isWinning = sar.IsWinning(currentPrice);
            double minProb = isWinning ? SarWinningCnnProb : SarMinCnnProb;
            if (cnnProb < minProb)
            {
                if (EnableDebugLogging)
                    Print($"[SAR] Gate2 FAIL: CNN={cnnProb:F3} < {minProb:F3} (winning={isWinning})");
                return false;
            }

            // Gate 3: cooldown
            if (sar.LastReversalTime != DateTime.MinValue)
            {
                double elapsedMin = (barTime - sar.LastReversalTime).TotalMinutes;
                if (elapsedMin < SarCooldownMinutes)
                {
                    if (EnableDebugLogging)
                        Print($"[SAR] Gate3 FAIL: cooldown {elapsedMin:F1}m < {SarCooldownMinutes}m");
                    return false;
                }
            }

            // Gate 4: MTF alignment
            if (mtfScore < SarMinMtfScore)
            {
                if (EnableDebugLogging)
                    Print($"[SAR] Gate4 FAIL: MTF={mtfScore:F2} < {SarMinMtfScore:F2}");
                return false;
            }

            // Gate 5: winning position at 1R+ needs exceptional CNN conviction
            if (isWinning && sar.RMultiple(currentPrice) > SarHighWinnerRMult && cnnProb < SarHighWinnerCnnProb)
            {
                if (EnableDebugLogging)
                    Print($"[SAR] Gate5 FAIL: won't flip +{sar.RMultiple(currentPrice):F2}R winner " +
                          $"without CNN ≥ {SarHighWinnerCnnProb:F2} (got {cnnProb:F3})");
                return false;
            }

            Print($"[SAR] Gates PASSED: {sar.ActiveDirection} → {newDirection} " +
                  $"CNN={cnnProb:F3} MTF={mtfScore:F2} R={sar.RMultiple(currentPrice):F2} " +
                  $"reversal#{sar.ReversalCount + 1}");
            return true;
        }

        /// <summary>
        /// Decide whether to use a limit order at the range edge or a market
        /// chase order.  Mirrors Python PositionManager._decide_entry_type().
        ///
        /// Returns:
        ///   orderType = "limit"  + limitPrice set to entryTarget when price is
        ///               below/above the trigger (haven't reached yet).
        ///   orderType = "market" when price is past trigger but within
        ///               SarChaseMaxAtrFraction × ATR AND cnnProb ≥ SarChaseMinCnnProb.
        ///   orderType = "limit"  at trigger price when price has moved too far.
        /// </summary>
        private string DecideEntryType(
            string direction,
            double entryTarget,
            double triggerPrice,
            double currentPrice,
            double atr,
            double cnnProb,
            out double resolvedPrice)
        {
            if (currentPrice <= 0 || atr <= 0)
            {
                resolvedPrice = triggerPrice;
                return "market";
            }

            double overshoot = direction == "long"
                ? currentPrice - triggerPrice
                : triggerPrice - currentPrice;

            double maxChase = SarChaseMaxAtrFraction * atr;

            if (overshoot <= 0)
            {
                // Price hasn't reached the trigger yet — queue a limit at range edge
                resolvedPrice = entryTarget;
                return "limit";
            }

            if (overshoot <= maxChase && cnnProb >= SarChaseMinCnnProb)
            {
                // Within chase window with high conviction — market at current price
                if (EnableDebugLogging)
                    Print($"[SAR] Chase entry: {direction} overshoot={overshoot:F4} " +
                          $"({(overshoot / atr * 100):F1}% ATR) CNN={cnnProb:F3} → MARKET");
                resolvedPrice = currentPrice;
                return "market";
            }

            if (overshoot > maxChase)
            {
                // Too far gone — limit at trigger (may not fill, better than chasing)
                if (EnableDebugLogging)
                    Print($"[SAR] Price moved too far ({(overshoot / atr * 100):F1}% ATR) " +
                          $"— limit at trigger {triggerPrice:F4}");
                resolvedPrice = triggerPrice;
                return "limit";
            }

            // Default: limit at entry target
            resolvedPrice = entryTarget;
            return "limit";
        }

        /// <summary>
        /// Attempt to reverse an existing position for <paramref name="instrName"/>
        /// (identified by <paramref name="bip"/>).
        ///
        /// Steps (mirrors Python _reverse_position):
        ///   1. Flatten the existing position with a market exit.
        ///   2. Cancel all working bracket orders for the old signalId.
        ///   3. Clean up _positionPhases.
        ///   4. Reset per-range fired flags so the new signal can fire.
        ///   5. Call FireEntry() for the new direction.
        ///   6. Update ReversalState.
        ///
        /// Returns true if the reversal was submitted.
        /// </summary>
        private bool TryReversePosition(
            string newDirection,
            int bip,
            InstrumentState st,
            ReversalState sar,
            double price,
            double atr,
            DateTime barTime,
            string instrName,
            BreakoutType breakoutType)
        {
            try
            {
                string oldSignalId = sar.ActiveSignalId;
                string oldDirection = sar.ActiveDirection;

                Print($"[SAR] REVERSE {oldDirection.ToUpper()} → {newDirection.ToUpper()} " +
                      $"{instrName} BIP{bip} @ {price:F4} reversal#{sar.ReversalCount + 1}");

                // Step 1: market-flatten the existing position
                OrderAction flattenAction = oldDirection == "long"
                    ? OrderAction.Sell
                    : OrderAction.BuyToCover;

                // Determine qty from the live NT8 position on this BIP
                int flatQty = 1; // fallback
                try
                {
                    var ntPos = Positions[bip];
                    if (ntPos != null && ntPos.MarketPosition != MarketPosition.Flat)
                        flatQty = Math.Abs(ntPos.Quantity);
                }
                catch { /* if Positions[bip] throws, fallback to 1 */ }

                string flatId = "SAR-Flat-" + barTime.ToString("yyyyMMdd-HHmmss") +
                                "-" + instrName + "-" + oldDirection[0];
                try
                {
                    SubmitOrderUnmanaged(bip, flattenAction, OrderType.Market,
                        flatQty, 0, 0, "", flatId);
                    Print($"[SAR] Flatten submitted: {flatId} qty={flatQty}");
                }
                catch (Exception ex)
                {
                    Print($"[SAR] Flatten submit failed for {instrName}: {ex.Message}");
                    return false;
                }

                // Step 2 & 3: clean up PositionPhase tracking for the old signal
                if (!string.IsNullOrEmpty(oldSignalId))
                {
                    lock (_phaseLock)
                    {
                        if (_positionPhases.TryGetValue(oldSignalId, out var oldPh))
                        {
                            oldPh.Phase = BreakoutPhase.Closed;
                            _positionPhases.Remove(oldSignalId);
                        }
                    }
                }

                // Step 4: reset per-range fired flags on ALL range types for this
                // instrument so the reversal signal can fire normally through FireEntry.
                foreach (var rsKv in st.Ranges)
                {
                    rsKv.Value.FiredLong = false;
                    rsKv.Value.FiredShort = false;
                }
                st.BreakoutFiredLong = false;
                st.BreakoutFiredShort = false;

                // Active position count will be decremented by OnOrderUpdate when the
                // flatten fill arrives.  We pre-decrement here so the MaxConcurrent
                // gate in FireEntry doesn't block the reversal entry immediately.
                lock (_activeInstruments)
                {
                    if (_activeInstruments.Remove(instrName))
                        _activePositionCount = _activeInstruments.Count;
                }

                // Step 5: update ReversalState *before* FireEntry so the cooldown
                // is stamped even if FireEntry fails (prevents rapid retries).
                sar.ReversalCount++;
                sar.LastReversalTime = barTime;
                // Direction / signalId will be re-set by the Open() call inside
                // the FireEntry → OnOrderUpdate fill path (handled in FireEntry wrapper).

                // Step 6: fire new entry — FireEntry will stamp the new direction
                // via the SAR open callback at the bottom of FireEntry.
                FireEntry(newDirection, bip, st, price, atr, barTime, instrName, breakoutType);

                // Step 7: push the reversal state to the Python engine on the Pi
                // (100.100.84.48:8000) so PositionManager stays in sync.
                // Fire-and-forget — never block the bar thread.
                PushSarSyncAsync(instrName, newDirection, sar, barTime);

                return true;
            }
            catch (Exception ex)
            {
                Print($"[SAR] TryReversePosition exception for {instrName}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Fire-and-forget POST to the Python engine's /sar/sync endpoint.
        /// Sends the new active direction + signalId + reversal count so the
        /// Python PositionManager's ReversalState stays in sync with C#.
        ///
        /// Endpoint: POST http://100.100.84.48:8000/sar/sync
        /// Body (JSON):
        /// {
        ///   "asset":          "MGC",
        ///   "direction":      "long",
        ///   "signal_id":      "brk-l-20250115-143022-MGC-ORB",
        ///   "reversal_count": 1,
        ///   "entry_price":    2650.5,
        ///   "atr_at_entry":   8.4,
        ///   "sl_price":       2637.9,
        ///   "timestamp":      "2025-01-15T14:30:22Z",
        ///   "source":         "NinjaTrader"
        /// }
        /// </summary>
        private void PushSarSyncAsync(string instrName, string newDirection,
                                      ReversalState sar, DateTime barTime)
        {
            var client = _sarHttpClient;
            if (client == null) return;

            try
            {
                // Build JSON manually — no LINQ/serializer dependency at call site
                string ts = barTime.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ");
                string signalId = sar?.ActiveSignalId ?? "";
                int revCount = sar?.ReversalCount ?? 0;
                double entryPrice = sar?.EntryPrice ?? 0;
                double atrAtEntry = sar?.AtrAtEntry ?? 0;
                double slPrice = sar?.SlPrice ?? 0;

                string json =
                    "{" +
                    $"\"asset\":\"{SarEsc(instrName)}\"," +
                    $"\"direction\":\"{SarEsc(newDirection)}\"," +
                    $"\"signal_id\":\"{SarEsc(signalId)}\"," +
                    $"\"reversal_count\":{revCount}," +
                    $"\"entry_price\":{entryPrice:R}," +
                    $"\"atr_at_entry\":{atrAtEntry:R}," +
                    $"\"sl_price\":{slPrice:R}," +
                    $"\"timestamp\":\"{ts}\"," +
                    "\"source\":\"NinjaTrader\"" +
                    "}";

                var content = new System.Net.Http.StringContent(
                    json, System.Text.Encoding.UTF8, "application/json");

                client.PostAsync(CSarSyncUrl, content).ContinueWith(t =>
                {
                    if (t.IsFaulted || t.IsCanceled)
                    {
                        // Only log throttled — network glitches are expected when Pi is unreachable
                        if (EnableDebugLogging)
                            Print($"[SAR] Sync push failed for {instrName}: " +
                                  $"{t.Exception?.GetBaseException()?.Message ?? "cancelled"}");
                    }
                    else
                    {
                        try
                        {
                            int code = (int)t.Result.StatusCode;
                            if (EnableDebugLogging)
                                Print($"[SAR] Sync pushed: {instrName} {newDirection} → HTTP {code}");
                        }
                        catch { }
                        finally { t.Result?.Dispose(); content?.Dispose(); }
                    }
                }, System.Threading.Tasks.TaskContinuationOptions.None);
            }
            catch (Exception ex)
            {
                Print($"[SAR] PushSarSyncAsync exception (non-fatal): {ex.Message}");
            }
        }

        /// <summary>Escape a string value for inline JSON (SAR sync payload).</summary>
        private static string SarEsc(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"")
                    .Replace("\n", "\\n").Replace("\r", "\\r");
        }

        #endregion

        // =====================================================================
        // VWAP — daily-resetting, computed inline from bar data
        // =====================================================================
        #region Per-Bar Indicators

        // ── Helper: detect a true session boundary using SessionIterator ──────
        // Returns true if barTime is in a new trading session relative to the
        // last date recorded in sessionDate.  Uses SessionIterator when available
        // (correctly handles Globex 18:00 ET roll); falls back to calendar date.
        private bool IsNewSession(InstrumentState st, Bars bars, DateTime barTime)
        {
            if (st.SessionIter != null)
            {
                try
                {
                    st.SessionIter.GetNextSession(barTime, false);
                    DateTime sessionStart = st.SessionIter.ActualSessionBegin;
                    // New session when the session-start date has advanced.
                    return sessionStart.Date != st.VwapDate;
                }
                catch { /* fall through to calendar-date fallback */ }
            }
            return barTime.Date != st.VwapDate;
        }

        private void UpdateVwap(int bip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[bip];
                // Need at least 2 bars: index Count-1 is the just-opened bar,
                // index Count-2 is the bar that just closed.
                if (bars.Count < 2) return;

                // Always read from the closed bar (Count-2).  This function is
                // only called on IsFirstTickOfBar, so Count-1 is the new live bar.
                int closed = bars.Count - 2;
                DateTime barTime = bars.GetTime(closed);

                // Reset on new session (Globex-aware via SessionIterator)
                bool newSession = IsNewSession(st, bars, barTime);
                if (newSession)
                {
                    // ── Archive previous session's daily range for NR7 ────────
                    if (st.SessionHigh > double.MinValue && st.SessionLow < double.MaxValue)
                    {
                        double prevRange = st.SessionHigh - st.SessionLow;
                        st.DailyRangeHistory[st.DailyRangeIdx] = prevRange;
                        st.DailyRangeIdx = (st.DailyRangeIdx + 1) % 7;
                        if (st.DailyRangeFilled < 7) st.DailyRangeFilled++;
                    }

                    st.CumTypicalVol = 0;
                    st.CumVolume = 0;
                    // Record the session-start date so subsequent bars in the same
                    // session don't trigger another reset.
                    st.VwapDate = st.SessionIter != null
                        ? st.SessionIter.ActualSessionBegin.Date
                        : barTime.Date;

                    // Reset session tracking for daily range and premarket
                    st.SessionHigh = double.MinValue;
                    st.SessionLow = double.MaxValue;
                    st.PremarketHigh = double.MinValue;
                    st.PremarketLow = double.MaxValue;

                    // Reset CVD accumulators for new session
                    st.CvdSignedVol = 0;
                    st.CvdTotalVol = 0;

                    if (EnableDebugLogging)
                    {
                        string vwapSymbol = bars.Instrument.MasterInstrument.Name;
                        Print($"[Breakout DEBUG] BIP{bip} {vwapSymbol} VWAP RESET (new session) at {barTime:HH:mm}");
                    }
                }

                double high = bars.GetHigh(closed);
                double low = bars.GetLow(closed);
                double close = bars.GetClose(closed);
                double open = bars.GetOpen(closed);
                double vol = bars.GetVolume(closed);
                double typical = (high + low + close) / 3.0;

                if (vol > 0)
                {
                    st.CumTypicalVol += typical * vol;
                    st.CumVolume += vol;
                }

                st.Vwap = st.CumVolume > 0 ? st.CumTypicalVol / st.CumVolume : close;

                // ── Update VWAP history ring buffer ───────────────────────────
                st.VwapHistory[st.VwapHistIdx] = st.Vwap;
                st.VwapHistIdx = (st.VwapHistIdx + 1) % st.VwapHistory.Length;
                if (st.VwapHistFilled < st.VwapHistory.Length) st.VwapHistFilled++;

                // ── Update session high/low for daily range (NR7) ─────────────
                if (high > st.SessionHigh) st.SessionHigh = high;
                if (low < st.SessionLow) st.SessionLow = low;

                // ── Update premarket range (bars before ORB established) ──────
                if (!st.OrbEstablished)
                {
                    if (high > st.PremarketHigh) st.PremarketHigh = high;
                    if (low < st.PremarketLow) st.PremarketLow = low;
                }

                // ── Update CVD proxy (cumulative signed volume) ───────────────
                if (vol > 0)
                {
                    double sign = close >= open ? 1.0 : -1.0;
                    st.CvdSignedVol += vol * sign;
                    st.CvdTotalVol += vol;
                }
            }
            catch (Exception ex) { Print($"[Breakout] UpdateVwap BIP{bip}: {ex.Message}"); }
        }

        private void UpdateAtr(int bip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[bip];
                // Need at least 2 bars: Count-1 is the just-opened bar,
                // Count-2 is the bar that just closed.
                if (bars.Count < 2) return;

                // Read from the closed bar (Count-2); this function is only
                // called on IsFirstTickOfBar so Count-1 is the new live bar.
                int closed = bars.Count - 2;
                double high = bars.GetHigh(closed);
                double low = bars.GetLow(closed);
                // PrevClose is the close of the bar before the closed bar.
                double prev = st.PrevClose > 0 ? st.PrevClose : (closed > 0 ? bars.GetClose(closed - 1) : bars.GetClose(closed));
                double tr = Math.Max(high - low,
                              Math.Max(Math.Abs(high - prev),
                                       Math.Abs(low - prev)));

                int period = st.AtrBuffer.Length; // 14

                if (!st.AtrReady)
                {
                    // ── Warmup phase: fill ring-buffer, use plain SMA ─────────
                    // TrSum tracks the running total; no foreach needed.
                    st.TrSum -= st.AtrBuffer[st.AtrBufIdx];
                    st.AtrBuffer[st.AtrBufIdx] = tr;
                    st.AtrBufIdx = (st.AtrBufIdx + 1) % period;
                    st.TrSum += tr;

                    if (st.AtrFilled < period)
                        st.AtrFilled++;

                    if (st.AtrFilled >= period)
                    {
                        // Seed Wilder's ATR with the plain average of the first <period> TRs.
                        st.AtrValue = st.TrSum / period;
                        st.AtrReady = true;
                    }
                    else
                    {
                        // Not enough bars yet — use whatever average we have so far.
                        st.AtrValue = st.AtrFilled > 0 ? st.TrSum / st.AtrFilled : tr;
                    }
                }
                else
                {
                    // ── Live phase: Wilder's exponential smoothing ────────────
                    // ── Wilder's smoothed ATR ──────────────────────────────────
                    // AtrValue = (AtrValue * (period-1) + tr) / period
                    st.AtrValue = (st.AtrValue * (period - 1) + tr) / period;
                }

                // ── Update ATR history ring-buffer for atr_trend feature (v7.1) ──
                st.AtrHistory[st.AtrHistIdx] = st.AtrValue;
                st.AtrHistIdx = (st.AtrHistIdx + 1) % st.AtrHistory.Length;
                if (st.AtrHistFilled < st.AtrHistory.Length)
                    st.AtrHistFilled++;

                // ── Update volume trend ring-buffer for volume_trend feature (v7.1) ──
                double barVol = bars.GetVolume(closed);
                st.VolTrendBuf[st.VolTrendIdx] = barVol;
                st.VolTrendIdx = (st.VolTrendIdx + 1) % st.VolTrendBuf.Length;
                if (st.VolTrendFilled < st.VolTrendBuf.Length)
                    st.VolTrendFilled++;

                st.PrevClose = bars.GetClose(closed);
            }
            catch (Exception ex) { Print($"[Breakout] UpdateAtr BIP{bip}: {ex.Message}"); }
        }

        private void UpdateVolumeAvg(int bip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[bip];
                // Need at least 2 bars: Count-1 is the just-opened bar,
                // Count-2 is the bar that just closed.
                if (bars.Count < 2) return;

                // Read volume from the closed bar (Count-2); this function is
                // only called on IsFirstTickOfBar so Count-1 is the new live bar.
                double vol = bars.GetVolume(bars.Count - 2);
                int period = st.VolBuffer.Length;

                // Subtract the value that is about to be overwritten, add new value.
                st.VolSum -= st.VolBuffer[st.VolBufIdx];
                st.VolBuffer[st.VolBufIdx] = vol;
                st.VolBufIdx = (st.VolBufIdx + 1) % period;
                st.VolSum += vol;

                if (st.VolFilled < period)
                    st.VolFilled++;

                if (!st.VolReady && st.VolFilled >= period)
                    st.VolReady = true;
            }
            catch (Exception ex) { Print($"[Breakout] UpdateVolumeAvg BIP{bip}: {ex.Message}"); }
        }

        /// <summary>
        /// Update the per-instrument EMA9 used for Phase3 trailing stop logic.
        /// Standard EMA formula: EMA = prev + k * (close - prev)  where k = 2/(N+1).
        /// Seeded from the SMA of the first 9 closes (same as most charting platforms).
        /// Called on every closed bar from the per-bar indicator update block in OnBarUpdate.
        /// </summary>
        private void UpdateEma9(int bip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[bip];
                if (bars.Count < 2) return;

                int closed = bars.Count - 2;
                double close = bars.GetClose(closed);
                const int period = 9;
                const double k = 2.0 / (period + 1);   // 0.2

                if (!st.Ema9Ready)
                {
                    st.Ema9Sum += close;
                    st.Ema9Filled++;
                    if (st.Ema9Filled >= period)
                    {
                        st.Ema9Value = st.Ema9Sum / period;
                        st.Ema9Ready = true;
                    }
                }
                else
                {
                    st.Ema9Value = st.Ema9Value + k * (close - st.Ema9Value);
                }
            }
            catch (Exception ex) { Print($"[Breakout] UpdateEma9 BIP{bip}: {ex.Message}"); }
        }

        /// <summary>
        /// Update the MTF (15-minute) EMA-9/21/50 and MACD state for <paramref name="bip"/>.
        /// Called from OnBarUpdate whenever the 15m BIP for this instrument closes a new bar.
        ///
        /// Mirrors Python MTFAnalyzer weights:
        ///   +0.30  EMA fully stacked in breakout direction
        ///   +0.15  EMA slow slope positive in direction (≥ 0.02% per bar)
        ///   +0.25  MACD histogram polarity agrees
        ///   +0.15  MACD histogram slope agrees
        ///   +0.15  No opposing divergence (always granted — divergence detection omitted)
        ///
        /// Result is stored in st.MtfScore.  ShouldReverse() reads it as Gate 4.
        /// </summary>
        private void UpdateMtf(int mtfBip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[mtfBip];
                if (bars == null || bars.Count < 2) return;

                int last = bars.Count - 1;
                double close = bars.GetClose(last);
                if (close <= 0) return;

                // ── EMA update (incremental, alpha = 2/(period+1)) ────────────
                double alphaFast = 2.0 / (9 + 1);   // EMA-9
                double alphaMid = 2.0 / (21 + 1);   // EMA-21
                double alphaSlow = 2.0 / (50 + 1);   // EMA-50

                if (st.MtfEmaFilled == 0)
                {
                    // Seed all three EMAs on the very first bar
                    st.MtfEmaFast = close;
                    st.MtfEmaMid = close;
                    st.MtfEmaSlow = close;
                }
                else
                {
                    st.MtfEmaFast = alphaFast * close + (1.0 - alphaFast) * st.MtfEmaFast;
                    st.MtfEmaMid = alphaMid * close + (1.0 - alphaMid) * st.MtfEmaMid;
                    st.MtfEmaSlow = alphaSlow * close + (1.0 - alphaSlow) * st.MtfEmaSlow;
                }
                st.MtfEmaFilled++;
                st.MtfEmaReady = st.MtfEmaFilled >= 50;

                // ── EMA slow ring-buffer for slope (last 5 bars) ─────────────
                st.MtfEmaSlowBuf[st.MtfEmaSlowIdx] = st.MtfEmaSlow;
                st.MtfEmaSlowIdx = (st.MtfEmaSlowIdx + 1) % 5;
                if (st.MtfEmaSlowFill < 5) st.MtfEmaSlowFill++;

                if (st.MtfEmaSlowFill >= 5)
                {
                    // Oldest value is at current index (ring buffer wraps around)
                    double oldest = st.MtfEmaSlowBuf[st.MtfEmaSlowIdx];
                    if (oldest > 0)
                        st.MtfEmaSlopePerBar = (st.MtfEmaSlow - oldest) / oldest / 5.0;
                }

                // ── MACD update ───────────────────────────────────────────────
                double alphaM12 = 2.0 / (12 + 1);
                double alphaM26 = 2.0 / (26 + 1);
                double alphaSig = 2.0 / (9 + 1);

                if (st.MtfMacdFilled == 0)
                {
                    st.MtfMacdFastEma = close;
                    st.MtfMacdSlowEma = close;
                    st.MtfMacdLine = 0;
                    st.MtfMacdSigEma = 0;
                }
                else
                {
                    st.MtfMacdFastEma = alphaM12 * close + (1.0 - alphaM12) * st.MtfMacdFastEma;
                    st.MtfMacdSlowEma = alphaM26 * close + (1.0 - alphaM26) * st.MtfMacdSlowEma;
                    double macdLine = st.MtfMacdFastEma - st.MtfMacdSlowEma;
                    st.MtfMacdLine = macdLine;

                    if (st.MtfMacdFilled < 26)
                    {
                        // Signal EMA not yet meaningful — seed it
                        st.MtfMacdSigEma = macdLine;
                    }
                    else
                    {
                        st.MtfMacdSigEma = alphaSig * macdLine + (1.0 - alphaSig) * st.MtfMacdSigEma;
                    }
                }
                st.MtfMacdFilled++;
                double histogram = st.MtfMacdLine - st.MtfMacdSigEma;
                st.MtfMacdHistogram = histogram;
                st.MtfMacdReady = st.MtfMacdFilled >= 35; // 26 slow + 9 signal warm-up

                // ── Histogram ring-buffer for slope ───────────────────────────
                st.MtfHistBuf[st.MtfHistBufIdx] = histogram;
                st.MtfHistBufIdx = (st.MtfHistBufIdx + 1) % 3;
                if (st.MtfHistFilled < 3) st.MtfHistFilled++;

                double histSlope = 0;
                if (st.MtfHistFilled >= 3)
                {
                    double oldest = st.MtfHistBuf[st.MtfHistBufIdx]; // oldest in ring
                    histSlope = histogram - oldest; // positive = histogram growing
                }

                // ── MTF score (direction-agnostic raw components) ─────────────
                // We store the score as a direction-agnostic value here and
                // then flip it in ComputeMtfScore(direction) at evaluation time.
                // Store components for ComputeMtfScore — store in MtfScore as a
                // placeholder (ComputeMtfScore will recompute with direction).
                // We write the raw components to MtfScore so callers can read it
                // without a direction.  Use 0.70 as the "warm-up pass-through" score
                // (same as passing with stacked EMA + agreeing MACD but no divergence data).
                if (!st.MtfEmaReady || !st.MtfMacdReady)
                {
                    // Not yet warmed — use conservative 1.0 pass-through
                    // (same behaviour as the old stand-in) until enough bars.
                    st.MtfScore = 1.0;
                }
                else
                {
                    // Compute a direction-agnostic intermediate score.
                    // Actual directional score is computed in ComputeMtfScore().
                    // Here we store -1 as a sentinel so ComputeMtfScore knows it
                    // needs to be computed fresh with the caller's direction.
                    st.MtfScore = -1; // sentinel: "ready but needs direction"
                }
            }
            catch (Exception ex)
            {
                Print($"[MTF] UpdateMtf BIP{mtfBip}: {ex.Message}");
            }
        }

        /// <summary>
        /// Compute the directional MTF score (0.0–1.0) for a given direction using
        /// the pre-computed EMA / MACD state in <paramref name="st"/>.
        ///
        /// Mirrors Python MTFAnalyzer score weights:
        ///   +0.30  EMA fully stacked (fast>mid>slow for LONG, reversed for SHORT)
        ///   +0.15  EMA slow slope direction agrees (≥ 0.02% per bar)
        ///   +0.25  MACD histogram polarity agrees
        ///   +0.15  MACD histogram slope agrees (growing in favourable direction)
        ///   +0.15  No opposing divergence (always granted — divergence omitted)
        /// </summary>
        private double ComputeMtfScore(InstrumentState st, string direction)
        {
            if (!st.MtfEmaReady || !st.MtfMacdReady) return 1.0; // warm-up pass-through

            bool isLong = direction == "long";

            // +0.30 — EMA stacked
            bool emaStacked = isLong
                ? (st.MtfEmaFast > st.MtfEmaMid && st.MtfEmaMid > st.MtfEmaSlow)
                : (st.MtfEmaFast < st.MtfEmaMid && st.MtfEmaMid < st.MtfEmaSlow);
            double score = emaStacked ? 0.30 : 0.0;

            // +0.15 — EMA slope direction agrees (threshold 0.0002 = 0.02% per bar)
            const double minSlopePct = 0.0002;
            bool slopeOk = isLong
                ? st.MtfEmaSlopePerBar >= minSlopePct
                : st.MtfEmaSlopePerBar <= -minSlopePct;
            if (slopeOk) score += 0.15;

            // +0.25 — MACD histogram polarity agrees
            bool histOk = isLong
                ? st.MtfMacdHistogram > 0
                : st.MtfMacdHistogram < 0;
            if (histOk) score += 0.25;

            // +0.15 — MACD histogram slope agrees (histogram growing in favourable direction)
            if (st.MtfHistFilled >= 3)
            {
                double oldest = st.MtfHistBuf[st.MtfHistBufIdx];
                double histSlope = st.MtfMacdHistogram - oldest;
                bool histSlopeOk = isLong ? histSlope > 0 : histSlope < 0;
                if (histSlopeOk) score += 0.15;
            }

            // +0.15 — No opposing divergence (always granted in C# — divergence detection
            // requires swing-point detection across many bars, which is omitted here to
            // keep the per-bar update cost low.  The CNN filter provides a second opinion.)
            score += 0.15;

            return score;
        }

        #endregion

        // =====================================================================
        // Multi-range window accumulation — replaces UpdateOrbWindow
        // =====================================================================
        // One method handles all BreakoutType values.  The ORB legacy fields
        // (OrbHigh, OrbLow, OrbEstablished, etc.) are kept in sync so that
        // every existing consumer (CNN tabular prep, VWAP guard, premarket
        // tracking, GetStatusSummary) continues to work without any changes.
        // =====================================================================
        #region Range Window

        // ── feature_contract.json helpers ────────────────────────────────────────

        /// <summary>
        /// Parse the "breakout_types" section of feature_contract.json and return
        /// a mapping of BreakoutType → tp3_atr_mult for all 13 types.
        /// Uses a hand-rolled parser so there's no System.Text.Json / Newtonsoft
        /// dependency — NT8's embedded CLR only guarantees mscorlib + NT8 assemblies.
        /// </summary>
        private Dictionary<BreakoutType, double> ParseTp3MultsFromContract(string json)
        {
            var result = new Dictionary<BreakoutType, double>();

            // Mapping from contract key names → BreakoutType enum values
            var nameMap = new Dictionary<string, BreakoutType>(StringComparer.OrdinalIgnoreCase)
            {
                { "ORB",             BreakoutType.ORB             },
                { "PrevDay",         BreakoutType.PrevDay          },
                { "InitialBalance",  BreakoutType.InitialBalance   },
                { "Consolidation",   BreakoutType.Consolidation    },
                { "Weekly",          BreakoutType.Weekly           },
                { "Monthly",         BreakoutType.Monthly          },
                { "Asian",           BreakoutType.Asian            },
                { "BollingerSqueeze",BreakoutType.BollingerSqueeze },
                { "ValueArea",       BreakoutType.ValueArea        },
                { "InsideDay",       BreakoutType.InsideDay        },
                { "GapRejection",    BreakoutType.GapRejection     },
                { "PivotPoints",     BreakoutType.PivotPoints      },
                { "Fibonacci",       BreakoutType.Fibonacci        },
            };

            // Find "breakout_types" object — locate its opening brace
            int btIdx = json.IndexOf("\"breakout_types\"", StringComparison.Ordinal);
            if (btIdx < 0) return result;
            int startBrace = json.IndexOf('{', btIdx + 16);
            if (startBrace < 0) return result;

            // Walk type-level objects inside "breakout_types"
            int pos = startBrace + 1;
            int depth = 1;
            while (pos < json.Length && depth > 0)
            {
                // Find the next quoted type name at depth==1
                int nameStart = json.IndexOf('"', pos);
                if (nameStart < 0) break;
                int nameEnd = json.IndexOf('"', nameStart + 1);
                if (nameEnd < 0) break;
                string typeName = json.Substring(nameStart + 1, nameEnd - nameStart - 1);
                pos = nameEnd + 1;

                // Skip to the opening brace of this type's object
                int typeObjStart = json.IndexOf('{', pos);
                if (typeObjStart < 0) break;

                // Find the matching closing brace
                int typeObjEnd = typeObjStart + 1;
                int innerDepth = 1;
                while (typeObjEnd < json.Length && innerDepth > 0)
                {
                    if (json[typeObjEnd] == '{') innerDepth++;
                    else if (json[typeObjEnd] == '}') innerDepth--;
                    typeObjEnd++;
                }
                string typeObj = json.Substring(typeObjStart, typeObjEnd - typeObjStart);

                // Extract "tp3_atr_mult" value from typeObj
                int tp3Idx = typeObj.IndexOf("\"tp3_atr_mult\"", StringComparison.Ordinal);
                if (tp3Idx >= 0)
                {
                    int colonIdx = typeObj.IndexOf(':', tp3Idx + 14);
                    if (colonIdx >= 0)
                    {
                        // Collect digits (including decimal point)
                        int numStart = colonIdx + 1;
                        while (numStart < typeObj.Length && (typeObj[numStart] == ' ' || typeObj[numStart] == '\t' || typeObj[numStart] == '\r' || typeObj[numStart] == '\n'))
                            numStart++;
                        int numEnd = numStart;
                        while (numEnd < typeObj.Length && (char.IsDigit(typeObj[numEnd]) || typeObj[numEnd] == '.' || typeObj[numEnd] == '-'))
                            numEnd++;
                        string numStr = typeObj.Substring(numStart, numEnd - numStart);
                        if (double.TryParse(numStr, System.Globalization.NumberStyles.Float,
                                            System.Globalization.CultureInfo.InvariantCulture, out double mult))
                        {
                            if (nameMap.TryGetValue(typeName, out BreakoutType bt))
                                result[bt] = mult;
                        }
                    }
                }

                pos = typeObjEnd;
                // Update outer depth tracking
                for (int i = typeObjStart; i < typeObjEnd; i++)
                {
                    if (json[i] == '{') depth++;
                    else if (json[i] == '}') depth--;
                }
            }

            return result;
        }

        /// <summary>
        /// Write loaded per-type TP3 mults into every InstrumentState's RangeStates.
        /// Called after _states is allocated and _tp3MultByType is populated.
        /// </summary>
        private void ApplyTp3MultsToStates()
        {
            if (_states == null || _tp3MultByType == null) return;
            foreach (var st in _states)
            {
                if (st == null) continue;
                foreach (var kv in _tp3MultByType)
                {
                    if (st.Ranges.TryGetValue(kv.Key, out var rs))
                        rs.Tp3AtrMult = kv.Value;
                }
            }
        }

        private RangeConfig GetRangeConfig(BreakoutType type)
        {
            switch (type)
            {
                case BreakoutType.ORB:
                    // Bar-count is derived from OrbMinutes (configurable parameter).
                    // MinBarsRequired = half the window, mirroring the old logic.
                    return new RangeConfig(type, OrbMinutes, 0.0, Math.Max(1, OrbMinutes / 2),
                        $"Opening Range {OrbMinutes} min");
                case BreakoutType.PrevDay:
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Previous Day High/Low");
                case BreakoutType.InitialBalance:
                    // First 60 minutes of RTH — fixed institutional convention.
                    return new RangeConfig(type, 60, 0.0, 30,
                        "Initial Balance 60 min");
                case BreakoutType.Consolidation:
                    // ATR contraction: bar range < 0.65 × ATR for at least 5 bars.
                    return new RangeConfig(type, 0, 0.65, 5,
                        "ATR Squeeze / Consolidation");
                case BreakoutType.Weekly:
                    // Weekly high/low — range set from previous week's H/L.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Weekly High/Low");
                case BreakoutType.Monthly:
                    // Monthly high/low — range set from previous month's H/L.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Monthly High/Low");
                case BreakoutType.Asian:
                    // Asian session range: 19:00–01:00 ET (360 min window).
                    return new RangeConfig(type, 360, 0.0, 60,
                        "Asian Session 19:00-01:00 ET");
                case BreakoutType.BollingerSqueeze:
                    // Bollinger Band squeeze: bandwidth < 0.50 × ATR for 10+ bars.
                    return new RangeConfig(type, 0, 0.50, 10,
                        "Bollinger Band Squeeze");
                case BreakoutType.ValueArea:
                    // Value Area high/low from volume profile — set from prior session VA.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Value Area High/Low");
                case BreakoutType.InsideDay:
                    // Inside day: current session range fully contained within prior session.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Inside Day");
                case BreakoutType.GapRejection:
                    // Gap fill rejection: gap open followed by reversal back into range.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Gap Fill Rejection");
                case BreakoutType.PivotPoints:
                    // Classic pivot point levels (PP, R1, S1) from prior session.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Pivot Points");
                case BreakoutType.Fibonacci:
                    // Fibonacci retracement levels from prior session range.
                    return new RangeConfig(type, 0, 0.0, 1,
                        "Fibonacci Retracement");
                default:
                    return new RangeConfig(type, OrbMinutes, 0.0, Math.Max(1, OrbMinutes / 2),
                        "Unknown");
            }
        }

        /// <summary>
        /// Resolve the session start date for a bar, using SessionIterator when
        /// available (Globex-aware) and falling back to calendar date otherwise.
        /// </summary>
        private DateTime GetSessionStartDate(InstrumentState st, DateTime barTime)
        {
            if (st.SessionIter != null)
            {
                try
                {
                    st.SessionIter.GetNextSession(barTime, false);
                    return st.SessionIter.ActualSessionBegin.Date;
                }
                catch { /* fall through */ }
            }
            return barTime.Date;
        }

        /// <summary>
        /// Update the range window for a single BreakoutType on each closed bar.
        /// Keeps the ORB legacy fields in sync for the Orb type so all downstream
        /// code (CNN prep, premarket tracking, status summary) is unaffected.
        /// </summary>
        private void UpdateRangeWindow(int bip, InstrumentState st, BreakoutType type)
        {
            try
            {
                var bars = BarsArray[bip];
                if (bars.Count < 2) return;

                int last = bars.Count - 2;          // most recently closed bar
                DateTime barTime = bars.GetTime(last);
                double high = bars.GetHigh(last);
                double low = bars.GetLow(last);

                var cfg = GetRangeConfig(type);
                var rs = st.Ranges[type];

                // ── Session boundary detection (Globex-aware) ─────────────────
                DateTime sessionDate = GetSessionStartDate(st, barTime);
                bool newSession = sessionDate != rs.SessionDate;

                if (newSession)
                {
                    // ── PrevDay: snapshot closing session's high/low before reset ─
                    if (type == BreakoutType.PrevDay && rs.RangeEstablished)
                    {
                        // We already have a valid previous-day range from the last
                        // session — nothing extra to do; the values carry forward
                        // into the new session and are only overwritten below.
                    }

                    // ── Compute PrevDay range from the last session's archived daily range ──
                    // IMPORTANT: UpdateVwap runs before UpdateRangeWindow and resets
                    // st.SessionHigh / st.SessionLow to sentinel values on newSession
                    // before we get here.  We must therefore read from DailyRangeHistory,
                    // which UpdateVwap archives *before* it wipes the session trackers.
                    //
                    // DailyRangeHistory stores the last 7 session (high-low) range sizes.
                    // That gives us the range magnitude but not the actual H/L prices, so
                    // we fall back to OrbHigh/OrbLow (legacy fields, fully synced) as the
                    // best available proxy for yesterday's high/low when the previous
                    // session's price levels are needed.
                    //
                    // On a genuine cold start (DailyRangeFilled == 0, no previous session
                    // data at all) we leave RangeEstablished = false so no signal fires
                    // until at least one complete session has been observed.

                    // ── Capture prior day OHLC for v7 daily bias features ─────
                    // Snapshot yesterday's levels before they get overwritten.
                    // Uses PrevOrbHigh/Low as the best proxy for prior session H/L
                    // (same source the PrevDay range builder uses).
                    if (type == BreakoutType.ORB)
                    {
                        // PrevOrbHigh/Low were just snapshotted above.
                        // We also need open/close for candle pattern analysis.
                        // Close = prior session's last bar close (bar just before this session start).
                        // Open = prior session's first bar open.  We approximate with PrevOrbLow
                        // as the session open proxy since we don't track session open separately.
                        if (st.PrevOrbHigh > 0 && st.PrevOrbLow < double.MaxValue && st.PrevOrbHigh > st.PrevOrbLow)
                        {
                            st.PrevDayHigh = st.PrevOrbHigh;
                            st.PrevDayLow = st.PrevOrbLow;
                            // Use the prior session's close (the bar just before session boundary)
                            st.PrevDayClose = st.PrevClose;
                            // Approximate open from session high/low + close position
                            // (imperfect but functional — the actual open isn't tracked)
                            st.PrevDayOpen = st.PrevClose; // best available proxy
                            st.PrevDayValid = true;
                        }

                        // ── Update prior week H/L for weekly_range_position (v7) ──
                        // Accumulate the maximum daily range seen across sessions.
                        // On Monday (new week), snapshot and reset.
                        if (barTime.DayOfWeek == DayOfWeek.Monday && st.PrevWeekHigh > 0
                            && st.PrevWeekLow < double.MaxValue)
                        {
                            // The accumulated week H/L becomes "prior week"
                            // (they'll be overwritten as the new week accumulates)
                        }
                        // Always accumulate into the current week tracker
                        if (st.PrevDayValid)
                        {
                            if (st.PrevDayHigh > st.PrevWeekHigh)
                                st.PrevWeekHigh = st.PrevDayHigh;
                            if (st.PrevDayLow < st.PrevWeekLow)
                                st.PrevWeekLow = st.PrevDayLow;
                            st.PrevWeekValid = true;
                        }
                    }

                    if (type == BreakoutType.PrevDay)
                    {
                        // The most recently archived daily range entry is at index
                        // (DailyRangeIdx - 1 + 7) % 7 because DailyRangeIdx points to
                        // the *next* write slot (ring buffer convention in UpdateVwap).
                        bool hasPrevSession = st.DailyRangeFilled > 0;

                        if (hasPrevSession)
                        {
                            // Use PrevOrbHigh/PrevOrbLow — snapshotted by
                            // UpdateRangeWindow(Orb) at the moment it detected
                            // the new session, before it overwrote st.OrbHigh/Low.
                            // This is safe because Orb runs before PrevDay in
                            // OnBarUpdate's explicit call order.
                            double prevH = st.PrevOrbHigh > 0 ? st.PrevOrbHigh : high;
                            double prevL = st.PrevOrbLow < double.MaxValue ? st.PrevOrbLow : low;
                            rs.RangeHigh = prevH;
                            rs.RangeLow = prevL;
                            rs.RangeEstablished = (prevH > prevL);
                        }
                        else
                        {
                            // Cold start — no complete session seen yet.  Suppress until
                            // UpdateVwap has archived at least one daily range entry.
                            rs.RangeEstablished = false;
                        }

                        rs.BarsInRange = hasPrevSession ? 1 : 0;
                        rs.FiredLong = false;
                        rs.FiredShort = false;
                        rs.SessionDate = sessionDate;

                        if (EnableDebugLogging)
                        {
                            string sym = bars.Instrument.MasterInstrument.Name;
                            if (rs.RangeEstablished)
                                Print($"[Breakout DEBUG] BIP{bip} {sym} PrevDay SET H={rs.RangeHigh:F6} L={rs.RangeLow:F6} (new session {sessionDate:yyyy-MM-dd})");
                            else
                                Print($"[Breakout DEBUG] BIP{bip} {sym} PrevDay WAITING — cold start, no prior session archived yet");
                        }
                        return;
                    }

                    // ── All other types: reset range for the new session ───────
                    rs.RangeHigh = high;
                    rs.RangeLow = low;
                    rs.RangeEstablished = false;
                    rs.BarsInRange = 0;
                    rs.FiredLong = false;
                    rs.FiredShort = false;
                    rs.SessionDate = sessionDate;

                    // ── Orb: compute window-end time ──────────────────────────
                    if (type == BreakoutType.ORB)
                    {
                        DateTime sessionBegin = barTime;
                        if (st.SessionIter != null)
                        {
                            try { sessionBegin = st.SessionIter.ActualSessionBegin; }
                            catch { /* keep barTime */ }
                        }
                        rs.RangeEndTime = sessionBegin.AddMinutes(OrbMinutes);

                        // ── Snapshot previous session ORB before overwriting ───
                        // PrevDay reads these after UpdateRangeWindow(Orb) has run,
                        // so we must save them here, before the reset below.
                        if (st.OrbHigh > 0 && st.OrbLow < double.MaxValue)
                        {
                            st.PrevOrbHigh = st.OrbHigh;
                            st.PrevOrbLow = st.OrbLow;
                        }

                        // ── Sync legacy ORB fields ─────────────────────────────
                        st.OrbHigh = high;
                        st.OrbLow = low;
                        st.OrbEstablished = false;
                        st.OrbBarCount = 0;
                        st.OrbSessionDate = sessionDate;
                        st.OrbEndTime = rs.RangeEndTime;
                        st.BreakoutFiredLong = false;
                        st.BreakoutFiredShort = false;

                        if (EnableDebugLogging)
                        {
                            string sym = bars.Instrument.MasterInstrument.Name;
                            Print($"[Breakout DEBUG] BIP{bip} {sym} → NEW SESSION {sessionDate:yyyy-MM-dd} | ORB window ends {rs.RangeEndTime:HH:mm:ss} ET | Initial H/L {high:F6}/{low:F6}");
                        }
                    }
                    else if (type == BreakoutType.InitialBalance)
                    {
                        // IB window is always 60 min from session open
                        DateTime sessionBegin = barTime;
                        if (st.SessionIter != null)
                        {
                            try { sessionBegin = st.SessionIter.ActualSessionBegin; }
                            catch { /* keep barTime */ }
                        }
                        rs.RangeEndTime = sessionBegin.AddMinutes(60);
                    }
                    else if (type == BreakoutType.Asian)
                    {
                        // Asian window: 19:00 ET to 01:00 ET next day (360 min).
                        // Use the session date's 19:00 as the start regardless of
                        // when the Globex session actually began.
                        DateTime asianStart = sessionDate.Date.AddHours(19);
                        rs.RangeEndTime = asianStart.AddMinutes(360);
                    }

                    return;
                }

                // ── PrevDay: range is fixed once set — no further bar expansion ─
                if (type == BreakoutType.PrevDay) return;

                // ── Type-specific range building ──────────────────────────────
                switch (type)
                {
                    // ── ORB: time-window accumulation ─────────────────────────
                    case BreakoutType.ORB:
                        {
                            if (barTime <= rs.RangeEndTime)
                            {
                                rs.RangeHigh = Math.Max(rs.RangeHigh, high);
                                rs.RangeLow = Math.Min(rs.RangeLow, low);
                                rs.BarsInRange++;

                                // Keep legacy fields current during accumulation
                                st.OrbHigh = rs.RangeHigh;
                                st.OrbLow = rs.RangeLow;
                                st.OrbBarCount = rs.BarsInRange;

                                if (EnableDebugLogging && last % DebugLogFrequency == 0)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] BIP{bip} {sym} ORB building → H={rs.RangeHigh:F6} L={rs.RangeLow:F6} Range={rs.RangeHigh - rs.RangeLow:F6} (bar {last})");
                                }
                                return;
                            }

                            // First bar after window closes — validate minimum bars
                            if (!rs.RangeEstablished)
                            {
                                if (rs.BarsInRange >= cfg.MinBarsRequired)
                                {
                                    rs.RangeEstablished = true;
                                    st.OrbEstablished = true;

                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 🔥 BIP{bip} {sym} ORB ESTABLISHED | H={rs.RangeHigh:F6} L={rs.RangeLow:F6} | Range={rs.RangeHigh - rs.RangeLow:F6} | ATR={st.AtrValue:F6} | Bars={rs.BarsInRange}");
                                    }
                                }
                                else
                                {
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] ⚠ BIP{bip} {sym} ORB SKIPPED (mid-session startup, only {rs.BarsInRange}/{cfg.MinBarsRequired} bars observed) — waiting for next session");
                                    }
                                    // Mark established-but-suppressed so this branch
                                    // never re-runs; fire flags block CheckBreakout.
                                    rs.RangeEstablished = true;
                                    rs.FiredLong = true;
                                    rs.FiredShort = true;
                                    st.OrbEstablished = true;
                                    st.BreakoutFiredLong = true;
                                    st.BreakoutFiredShort = true;
                                }
                            }
                            break;
                        }

                    // ── Initial Balance: time-window accumulation (60 min) ────
                    case BreakoutType.InitialBalance:
                        {
                            if (barTime <= rs.RangeEndTime)
                            {
                                rs.RangeHigh = Math.Max(rs.RangeHigh, high);
                                rs.RangeLow = Math.Min(rs.RangeLow, low);
                                rs.BarsInRange++;

                                if (EnableDebugLogging && last % DebugLogFrequency == 0)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] BIP{bip} {sym} IB building → H={rs.RangeHigh:F6} L={rs.RangeLow:F6} (bar {last})");
                                }
                                return;
                            }

                            // First bar after the 60-min window closes.
                            // Mirror the ORB mid-session startup guard: if the strategy
                            // loaded after the IB window had already passed (BarsInRange
                            // below the minimum), mark established-but-suppressed so
                            // CheckBreakout skips this BIP for the rest of the session.
                            if (!rs.RangeEstablished)
                            {
                                if (rs.BarsInRange >= cfg.MinBarsRequired)
                                {
                                    rs.RangeEstablished = true;
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 🔷 BIP{bip} {sym} IB ESTABLISHED | H={rs.RangeHigh:F6} L={rs.RangeLow:F6} | Bars={rs.BarsInRange}");
                                    }
                                }
                                else
                                {
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] ⚠ BIP{bip} {sym} IB SKIPPED (mid-session startup, only {rs.BarsInRange}/{cfg.MinBarsRequired} bars observed) — waiting for next session");
                                    }
                                    // Mark established-but-suppressed: prevents this branch
                                    // re-running every bar for the rest of the session.
                                    rs.RangeEstablished = true;
                                    rs.FiredLong = true;
                                    rs.FiredShort = true;
                                }
                            }
                            break;
                        }

                    // ── Consolidation: ATR contraction (squeeze) ──────────────
                    case BreakoutType.Consolidation:
                        {
                            double atr = st.AtrValue;
                            double barRange = high - low;

                            if (atr > 0 && barRange < atr * cfg.SqueezeThreshold)
                            {
                                // Bar is inside the squeeze — expand the range and count
                                rs.RangeHigh = Math.Max(rs.RangeHigh, high);
                                rs.RangeLow = Math.Min(rs.RangeLow, low);
                                rs.BarsInRange++;

                                if (!rs.RangeEstablished && rs.BarsInRange >= cfg.MinBarsRequired)
                                {
                                    rs.RangeEstablished = true;
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 🔶 BIP{bip} {sym} CONSOLIDATION ESTABLISHED | H={rs.RangeHigh:F6} L={rs.RangeLow:F6} | squeeze bars={rs.BarsInRange}");
                                    }
                                }
                            }
                            else
                            {
                                // This bar is wider than the squeeze threshold.
                                // Only reset the accumulator when the range was NOT yet
                                // established — i.e. we were still building toward the
                                // minimum bar count.  If the range IS established we leave
                                // it intact so CheckBreakout can detect the breakout on
                                // this very bar (close > RangeHigh or close < RangeLow).
                                // The accumulator will be reset on the next new session.
                                if (!rs.RangeEstablished)
                                {
                                    rs.RangeHigh = high;
                                    rs.RangeLow = low;
                                    rs.BarsInRange = 0;
                                }
                            }
                            break;
                        }

                    // ── Asian session: time-window accumulation (19:00–01:00 ET = 360 min) ──
                    case BreakoutType.Asian:
                        {
                            if (barTime <= rs.RangeEndTime)
                            {
                                // Only accumulate bars within the Asian window (19:00–01:00 ET)
                                int h = barTime.Hour;
                                bool inAsianWindow = (h >= 19) || (h <= 1);
                                if (inAsianWindow)
                                {
                                    rs.RangeHigh = Math.Max(rs.RangeHigh, high);
                                    rs.RangeLow = Math.Min(rs.RangeLow, low);
                                    rs.BarsInRange++;
                                }
                                return;
                            }

                            if (!rs.RangeEstablished)
                            {
                                if (rs.BarsInRange >= cfg.MinBarsRequired)
                                {
                                    rs.RangeEstablished = true;
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 🌏 BIP{bip} {sym} ASIAN ESTABLISHED | H={rs.RangeHigh:F6} L={rs.RangeLow:F6} | Bars={rs.BarsInRange}");
                                    }
                                }
                                else
                                {
                                    rs.RangeEstablished = true;
                                    rs.FiredLong = true;
                                    rs.FiredShort = true;
                                }
                            }
                            break;
                        }

                    // ── BollingerSqueeze: bandwidth contraction ───────────────
                    case BreakoutType.BollingerSqueeze:
                        {
                            // Similar to Consolidation but uses a tighter squeeze
                            // threshold (0.50 × ATR) and requires more bars (10).
                            double atr = st.AtrValue;
                            double barRange = high - low;

                            if (atr > 0 && barRange < atr * cfg.SqueezeThreshold)
                            {
                                rs.RangeHigh = Math.Max(rs.RangeHigh, high);
                                rs.RangeLow = Math.Min(rs.RangeLow, low);
                                rs.BarsInRange++;

                                if (!rs.RangeEstablished && rs.BarsInRange >= cfg.MinBarsRequired)
                                {
                                    rs.RangeEstablished = true;
                                    if (EnableDebugLogging)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 🔮 BIP{bip} {sym} BOLLINGER SQUEEZE ESTABLISHED | H={rs.RangeHigh:F6} L={rs.RangeLow:F6} | squeeze bars={rs.BarsInRange}");
                                    }
                                }
                            }
                            else
                            {
                                if (!rs.RangeEstablished)
                                {
                                    rs.RangeHigh = high;
                                    rs.RangeLow = low;
                                    rs.BarsInRange = 0;
                                }
                            }
                            break;
                        }

                    // ── Weekly: prior-week high/low ──────────────────────────────
                    case BreakoutType.Weekly:
                        {
                            // Strategy: accumulate a rolling 7-calendar-day window of H/L.
                            // On every bar, walk back through BarsArray[bip] to find bars
                            // whose date falls in the prior trading week (Mon–Fri before
                            // this week's Monday).  This is O(lookback) but runs once per
                            // 1-min bar and the window is capped at ~7*390 = ~2730 bars.
                            // Mirrors Python _build_weekly_range().
                            {
                                DateTime today = barTime.Date;
                                int weekday = (int)today.DayOfWeek; // Sun=0..Sat=6
                                // Convert to Mon=0 convention
                                int monOffset = weekday == 0 ? 6 : weekday - 1;
                                DateTime thisMonday = today.AddDays(-monOffset);
                                DateTime prevMonday = thisMonday.AddDays(-7);

                                double wHigh = 0, wLow = double.MaxValue;
                                int wBars = 0;

                                for (int k = last; k >= 0 && k >= last - 2730; k--)
                                {
                                    DateTime kt = bars.GetTime(k).Date;
                                    if (kt >= thisMonday) continue;  // this week — skip
                                    if (kt < prevMonday) break;     // older than prior week — stop
                                    wHigh = Math.Max(wHigh, bars.GetHigh(k));
                                    wLow = Math.Min(wLow, bars.GetLow(k));
                                    wBars++;
                                }

                                if (wBars >= cfg.MinBarsRequired && wHigh > wLow)
                                {
                                    rs.RangeHigh = wHigh;
                                    rs.RangeLow = wLow;
                                    rs.RangeEstablished = true;
                                    rs.BarsInRange = wBars;

                                    if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 📅 BIP{bip} {sym} WEEKLY " +
                                              $"H={rs.RangeHigh:F4} L={rs.RangeLow:F4} " +
                                              $"({wBars} prior-week bars, week starts {prevMonday:MM-dd})");
                                    }
                                }
                                else
                                {
                                    rs.RangeEstablished = false;
                                }
                            }
                            break;
                        }

                    // ── Monthly: prior-month high/low ─────────────────────────
                    case BreakoutType.Monthly:
                        {
                            // Walk back up to ~20*390 bars to find prior-month bars.
                            // "Prior month" = any bar whose date falls before the 1st
                            // of the current calendar month.  Mirrors Python _build_monthly_range().
                            {
                                DateTime firstOfMonth = new DateTime(barTime.Year, barTime.Month, 1);
                                DateTime lookbackStart = firstOfMonth.AddDays(-(cfg.RangeBars > 0
                                    ? cfg.RangeBars     // RangeBars repurposed as lookback days
                                    : 31));             // default ~1 month

                                double mHigh = 0, mLow = double.MaxValue;
                                int mBars = 0;

                                for (int k = last; k >= 0 && k >= last - 12000; k--)
                                {
                                    DateTime kt = bars.GetTime(k).Date;
                                    if (kt >= firstOfMonth) continue;  // current month — skip
                                    if (kt < lookbackStart) break;     // beyond lookback — stop
                                    mHigh = Math.Max(mHigh, bars.GetHigh(k));
                                    mLow = Math.Min(mLow, bars.GetLow(k));
                                    mBars++;
                                }

                                if (mBars >= cfg.MinBarsRequired && mHigh > mLow)
                                {
                                    rs.RangeHigh = mHigh;
                                    rs.RangeLow = mLow;
                                    rs.RangeEstablished = true;
                                    rs.BarsInRange = mBars;

                                    if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 📆 BIP{bip} {sym} MONTHLY " +
                                              $"H={rs.RangeHigh:F4} L={rs.RangeLow:F4} " +
                                              $"({mBars} prior-month bars, cutoff {firstOfMonth:MM-dd})");
                                    }
                                }
                                else
                                {
                                    rs.RangeEstablished = false;
                                }
                            }
                            break;
                        }

                    // ── ValueArea: prior-session VAH/VAL via volume profile ────
                    case BreakoutType.ValueArea:
                        {
                            // Approximate VAH/VAL from prior session (18:00 ET boundary)
                            // without a full volume profile library.  Strategy:
                            //   1. Collect prior-session bars (from last 18:00 ET boundary
                            //      backward until the one before that).
                            //   2. Sort close prices by their associated volume.
                            //   3. Value Area = price range covering the top 70% of volume.
                            //   4. VAH = highest price in VA, VAL = lowest price in VA.
                            // Mirrors Python _build_va_range() fallback (no volume profile lib).
                            {
                                // Find the most recent 18:00 ET session boundary
                                int sessionBoundary = -1;
                                int prevBoundary = -1;
                                for (int k = last; k >= 0 && k >= last - 2000; k--)
                                {
                                    DateTime kt = bars.GetTime(k);
                                    if (kt.Hour == 18 && kt.Minute == 0)
                                    {
                                        if (sessionBoundary < 0) sessionBoundary = k;
                                        else { prevBoundary = k; break; }
                                    }
                                }

                                // Fall back: use calendar day boundary if no 18:00 found
                                if (sessionBoundary < 0)
                                {
                                    DateTime today2 = bars.GetTime(last).Date;
                                    for (int k = last; k >= 0; k--)
                                        if (bars.GetTime(k).Date < today2) { sessionBoundary = k + 1; break; }
                                    if (sessionBoundary < 0) { rs.RangeEstablished = false; break; }
                                    for (int k = sessionBoundary - 1; k >= 0; k--)
                                        if (bars.GetTime(k).Date < bars.GetTime(sessionBoundary).Date) { prevBoundary = k + 1; break; }
                                }

                                if (prevBoundary < 0 || sessionBoundary <= prevBoundary)
                                {
                                    rs.RangeEstablished = false;
                                    break;
                                }

                                // Collect (close, volume) pairs for prior session bars
                                var priceVol = new List<(double price, double vol)>();
                                double totalVol = 0;
                                for (int k = prevBoundary; k < sessionBoundary; k++)
                                {
                                    double v = bars.GetVolume(k);
                                    double c = bars.GetClose(k);
                                    priceVol.Add((c, v));
                                    totalVol += v;
                                }

                                if (priceVol.Count < cfg.MinBarsRequired || totalVol <= 0)
                                {
                                    rs.RangeEstablished = false;
                                    break;
                                }

                                // Sort by price descending, accumulate volume until 70% reached
                                priceVol.Sort((a, b) => b.price.CompareTo(a.price));
                                double vaTarget = totalVol * 0.70;
                                double accumulated = 0;
                                double vah = priceVol[0].price;
                                double val = priceVol[priceVol.Count - 1].price;
                                double poc = vah;
                                double pocVol = 0;

                                // Find POC (highest-volume price)
                                foreach (var pv in priceVol)
                                    if (pv.vol > pocVol) { pocVol = pv.vol; poc = pv.price; }

                                // VA: prices from POC outward until 70% volume covered
                                int hi = 0, lo = priceVol.Count - 1;
                                // Find POC index in sorted list
                                for (int k = 0; k < priceVol.Count; k++)
                                    if (Math.Abs(priceVol[k].price - poc) < 1e-9) { hi = k; lo = k; break; }
                                accumulated += priceVol[hi].vol;
                                while (accumulated < vaTarget)
                                {
                                    bool canUp = hi > 0;
                                    bool canDown = lo < priceVol.Count - 1;
                                    if (!canUp && !canDown) break;
                                    double upVol = canUp ? priceVol[hi - 1].vol : 0;
                                    double downVol = canDown ? priceVol[lo + 1].vol : 0;
                                    if (canUp && (!canDown || upVol >= downVol)) { hi--; accumulated += priceVol[hi].vol; }
                                    else { lo++; accumulated += priceVol[lo].vol; }
                                }
                                vah = priceVol[hi].price;
                                val = priceVol[lo].price;

                                if (vah > val)
                                {
                                    rs.RangeHigh = vah;
                                    rs.RangeLow = val;
                                    rs.AuxValue = poc;
                                    rs.RangeEstablished = true;
                                    rs.BarsInRange = priceVol.Count;

                                    if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                    {
                                        string sym = bars.Instrument.MasterInstrument.Name;
                                        Print($"[Breakout DEBUG] 📊 BIP{bip} {sym} VALUE AREA " +
                                              $"VAH={vah:F4} VAL={val:F4} POC={poc:F4} " +
                                              $"({priceVol.Count} bars, {accumulated / totalVol * 100:F0}% vol covered)");
                                    }
                                }
                                else
                                {
                                    rs.RangeEstablished = false;
                                }
                            }
                            break;
                        }

                    // ── InsideDay: mother bar H/L when today is inside yesterday ──
                    case BreakoutType.InsideDay:
                        {
                            // Mirrors Python _build_inside_day_range().
                            // 1. Find the prior session boundary (18:00 ET).
                            // 2. Compute today_high/low and yesterday_high/low.
                            // 3. If today is fully inside yesterday, use yesterday's H/L as range.
                            {
                                int sessionBoundary2 = -1;
                                int prevBoundary2 = -1;
                                for (int k = last; k >= 0 && k >= last - 2000; k--)
                                {
                                    DateTime kt = bars.GetTime(k);
                                    if (kt.Hour == 18 && kt.Minute == 0)
                                    {
                                        if (sessionBoundary2 < 0) sessionBoundary2 = k;
                                        else { prevBoundary2 = k; break; }
                                    }
                                }
                                if (sessionBoundary2 < 0)
                                {
                                    // Calendar-day fallback
                                    DateTime today3 = bars.GetTime(last).Date;
                                    for (int k = last; k >= 0; k--)
                                        if (bars.GetTime(k).Date < today3) { sessionBoundary2 = k + 1; break; }
                                    if (sessionBoundary2 < 0) { rs.RangeEstablished = false; break; }
                                    for (int k = sessionBoundary2 - 1; k >= 0; k--)
                                        if (bars.GetTime(k).Date < bars.GetTime(sessionBoundary2).Date)
                                        { prevBoundary2 = k + 1; break; }
                                }

                                if (prevBoundary2 < 0 || sessionBoundary2 <= prevBoundary2)
                                { rs.RangeEstablished = false; break; }

                                // Today H/L (from session boundary to last bar)
                                double todayH = double.MinValue, todayL = double.MaxValue;
                                for (int k = sessionBoundary2; k <= last; k++)
                                {
                                    todayH = Math.Max(todayH, bars.GetHigh(k));
                                    todayL = Math.Min(todayL, bars.GetLow(k));
                                }

                                // Yesterday H/L
                                double yestH = double.MinValue, yestL = double.MaxValue;
                                for (int k = prevBoundary2; k < sessionBoundary2; k++)
                                {
                                    yestH = Math.Max(yestH, bars.GetHigh(k));
                                    yestL = Math.Min(yestL, bars.GetLow(k));
                                }

                                bool isInside = todayH != double.MinValue && yestH != double.MinValue
                                             && todayH <= yestH && todayL >= yestL;

                                if (!isInside)
                                {
                                    rs.RangeEstablished = false;
                                    rs.FiredLong = false;
                                    rs.FiredShort = false;
                                    break;
                                }

                                // Compression ratio guard (mirror Python 0.25–0.85)
                                double yestRange = yestH - yestL;
                                double todayRange = todayH - todayL;
                                double compression = yestRange > 0 ? todayRange / yestRange : 1.0;

                                if (compression < 0.25 || compression > 0.85)
                                { rs.RangeEstablished = false; break; }

                                rs.RangeHigh = yestH;
                                rs.RangeLow = yestL;
                                rs.AuxHigh = todayH;
                                rs.AuxLow = todayL;
                                rs.RangeEstablished = true;
                                rs.BarsInRange = sessionBoundary2 - prevBoundary2;

                                if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] 🔲 BIP{bip} {sym} INSIDE DAY " +
                                          $"MotherH={yestH:F4} MotherL={yestL:F4} " +
                                          $"TodayH={todayH:F4} TodayL={todayL:F4} " +
                                          $"Compression={compression:P0}");
                                }
                            }
                            break;
                        }

                    // ── GapRejection: overnight gap zone as range ─────────────
                    case BreakoutType.GapRejection:
                        {
                            // Mirrors Python _build_gap_rejection_range().
                            // Gap = today_open vs yesterday_close, must be >= 0.25 × ATR.
                            // Range = [min(yest_close, today_open), max(yest_close, today_open)].
                            {
                                double atr2 = st.AtrValue;
                                if (atr2 <= 0) { rs.RangeEstablished = false; break; }

                                // Find prior session boundary (18:00 ET)
                                int sessionBoundary3 = -1;
                                for (int k = last; k >= 0 && k >= last - 2000; k--)
                                {
                                    DateTime kt = bars.GetTime(k);
                                    if (kt.Hour == 18 && kt.Minute == 0) { sessionBoundary3 = k; break; }
                                }
                                if (sessionBoundary3 < 0)
                                {
                                    DateTime today4 = bars.GetTime(last).Date;
                                    for (int k = last; k >= 0; k--)
                                        if (bars.GetTime(k).Date < today4) { sessionBoundary3 = k + 1; break; }
                                }
                                if (sessionBoundary3 < 0 || sessionBoundary3 >= last)
                                { rs.RangeEstablished = false; break; }

                                // yesterday_close = last bar before session boundary
                                double yestClose = bars.GetClose(sessionBoundary3 - 1);
                                // today_open = first bar at/after boundary
                                double todayOpen = bars.GetOpen(sessionBoundary3);
                                double gapSize = todayOpen - yestClose;
                                double minGap = 0.25 * atr2;

                                if (Math.Abs(gapSize) < minGap)
                                { rs.RangeEstablished = false; rs.AuxTag = ""; break; }

                                string gapDir = gapSize > 0 ? "UP" : "DOWN";
                                double rHigh = Math.Max(yestClose, todayOpen);
                                double rLow = Math.Min(yestClose, todayOpen);

                                rs.RangeHigh = rHigh;
                                rs.RangeLow = rLow;
                                rs.AuxValue = gapSize;
                                rs.AuxLow = yestClose;
                                rs.AuxTag = gapDir;
                                rs.RangeEstablished = true;
                                rs.BarsInRange = last - sessionBoundary3 + 1;

                                if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] ⚡ BIP{bip} {sym} GAP {gapDir} " +
                                          $"zone=[{rLow:F4},{rHigh:F4}] " +
                                          $"gap={gapSize:F4} (ATR={atr2:F4} min={minGap:F4})");
                                }
                            }
                            break;
                        }

                    // ── PivotPoints: classic floor pivots R1/S1 as range ───────
                    case BreakoutType.PivotPoints:
                        {
                            // Mirrors Python _build_pivot_range() — classic formula.
                            // P = (H+L+C)/3,  R1 = 2P−L,  S1 = 2P−H.
                            // Uses prior-session H/L/C (18:00 ET boundary).
                            {
                                int pivotBoundary = -1;
                                for (int k = last; k >= 0 && k >= last - 2000; k--)
                                {
                                    DateTime kt = bars.GetTime(k);
                                    if (kt.Hour == 18 && kt.Minute == 0) { pivotBoundary = k; break; }
                                }
                                if (pivotBoundary < 0)
                                {
                                    DateTime today5 = bars.GetTime(last).Date;
                                    for (int k = last; k >= 0; k--)
                                        if (bars.GetTime(k).Date < today5) { pivotBoundary = k + 1; break; }
                                }
                                if (pivotBoundary < 0) { rs.RangeEstablished = false; break; }

                                // Prior session H/L/C
                                double pH = double.MinValue, pL = double.MaxValue, pC = 0;
                                for (int k = 0; k < pivotBoundary; k++)
                                {
                                    pH = Math.Max(pH, bars.GetHigh(k));
                                    pL = Math.Min(pL, bars.GetLow(k));
                                    pC = bars.GetClose(k);
                                }
                                // Better: walk only the prior session (previous 18:00 to current 18:00)
                                int prevPivotBoundary = -1;
                                for (int k = pivotBoundary - 1; k >= 0 && k >= pivotBoundary - 2000; k--)
                                    if (bars.GetTime(k).Hour == 18 && bars.GetTime(k).Minute == 0)
                                    { prevPivotBoundary = k; break; }

                                if (prevPivotBoundary >= 0)
                                {
                                    pH = double.MinValue; pL = double.MaxValue;
                                    for (int k = prevPivotBoundary; k < pivotBoundary; k++)
                                    {
                                        pH = Math.Max(pH, bars.GetHigh(k));
                                        pL = Math.Min(pL, bars.GetLow(k));
                                        pC = bars.GetClose(k);
                                    }
                                }

                                if (pH == double.MinValue || pL == double.MaxValue || pH <= pL)
                                { rs.RangeEstablished = false; break; }

                                double pivot = (pH + pL + pC) / 3.0;
                                double r1 = 2.0 * pivot - pL;
                                double s1 = 2.0 * pivot - pH;

                                if (r1 <= s1 || r1 <= 0)
                                { rs.RangeEstablished = false; break; }

                                rs.RangeHigh = r1;
                                rs.RangeLow = s1;
                                rs.AuxValue = pivot;
                                rs.AuxHigh = pH;
                                rs.AuxLow = pL;
                                rs.RangeEstablished = true;
                                rs.BarsInRange = pivotBoundary - (prevPivotBoundary >= 0 ? prevPivotBoundary : 0);

                                if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] 🎯 BIP{bip} {sym} PIVOT " +
                                          $"PP={pivot:F4} R1={r1:F4} S1={s1:F4}");
                                }
                            }
                            break;
                        }

                    // ── Fibonacci: 38.2%–61.8% retracement zone of prior swing ─
                    case BreakoutType.Fibonacci:
                        {
                            // Mirrors Python _build_fibonacci_range().
                            // Swing = highest high / lowest low over last 50 bars.
                            // Swing must be >= 1.5 × ATR.
                            // Retracement zone: [fib_618, fib_382] of that swing.
                            {
                                double atr3 = st.AtrValue;
                                if (atr3 <= 0) { rs.RangeEstablished = false; break; }

                                const int lookback = 50;
                                int start = Math.Max(0, last - lookback + 1);

                                double swingH = double.MinValue, swingL = double.MaxValue;
                                int swingHIdx = start, swingLIdx = start;
                                for (int k = start; k <= last; k++)
                                {
                                    double kH = bars.GetHigh(k);
                                    double kL = bars.GetLow(k);
                                    if (kH > swingH) { swingH = kH; swingHIdx = k; }
                                    if (kL < swingL) { swingL = kL; swingLIdx = k; }
                                }

                                double swingSize = swingH - swingL;
                                const double minSwingMult = 1.5;

                                if (swingSize < minSwingMult * atr3 || swingH <= swingL)
                                { rs.RangeEstablished = false; break; }

                                double rHigh, rLow, fib382, fib618;

                                if (swingHIdx > swingLIdx)
                                {
                                    // Upswing — retrace down from high
                                    fib382 = swingH - 0.382 * swingSize;
                                    fib618 = swingH - 0.618 * swingSize;
                                    rHigh = fib382;
                                    rLow = fib618;
                                }
                                else
                                {
                                    // Downswing — retrace up from low
                                    fib382 = swingL + 0.382 * swingSize;
                                    fib618 = swingL + 0.618 * swingSize;
                                    rHigh = fib618;
                                    rLow = fib382;
                                }

                                if (rHigh <= rLow || rHigh <= 0)
                                { rs.RangeEstablished = false; break; }

                                rs.RangeHigh = rHigh;
                                rs.RangeLow = rLow;
                                rs.AuxHigh = swingH;
                                rs.AuxLow = swingL;
                                rs.AuxValue = fib382;
                                rs.RangeEstablished = true;
                                rs.BarsInRange = last - start + 1;

                                if (EnableDebugLogging && !rs.FiredLong && !rs.FiredShort)
                                {
                                    string sym = bars.Instrument.MasterInstrument.Name;
                                    Print($"[Breakout DEBUG] 🌀 BIP{bip} {sym} FIBONACCI " +
                                          $"zone=[{rLow:F4},{rHigh:F4}] " +
                                          $"swing=[{swingL:F4},{swingH:F4}] size={swingSize:F4}");
                                }
                            }
                            break;
                        }
                }
            }
            catch (Exception ex)
            {
                Print($"[Breakout] UpdateRangeWindow BIP{bip} {type}: {ex.Message}");
            }
        }

        #endregion


        // =====================================================================
        // Breakout detection — evaluates all established ranges every bar
        // =====================================================================
        #region Breakout Detection

        private void CheckBreakout(int bip, InstrumentState st)
        {
            try
            {
                // Defensive: ensure SAR state array is sized (should always be
                // true after DataLoaded, but guard in case of early calls).
                ReversalState sar = (_sarStates != null && bip < _sarStates.Length)
                    ? _sarStates[bip]
                    : null;

                // Wait until at least the ORB is established (ensures ATR, VWAP,
                // and volume avg are all warmed up before any signal fires).
                var orbRs = st.Ranges[BreakoutType.ORB];
                if (!orbRs.RangeEstablished)
                {
                    // Heartbeat log so we can confirm all BIPs are alive
                    if (EnableDebugLogging && bip < BarsArray.Length)
                    {
                        var hbBars = BarsArray[bip];
                        int hbLast = hbBars.Count > 0 ? hbBars.Count - 1 : 0;
                        if (hbLast != st.LastLoggedBar && hbLast % DebugLogFrequency == 0)
                        {
                            st.LastLoggedBar = hbLast;
                            string bipName = hbBars.Instrument.MasterInstrument.Name;
                            string orbStatus = st.OrbSessionDate == DateTime.MinValue
                                ? "no session yet"
                                : $"session {st.OrbSessionDate:MM-dd} bars={st.OrbBarCount}";
                            Print($"[Breakout DEBUG] BIP{bip} {bipName} waiting for ORB ({orbStatus})");
                        }
                    }
                    return;
                }
                if (RiskBlocked) return;

                var bars = BarsArray[bip];
                if (bars.Count == 0) return;

                int last = bars.Count - 1;
                double close = bars.GetClose(last);
                double high = bars.GetHigh(last);
                double low = bars.GetLow(last);
                // Use the volume of the most recently *closed* bar rather than
                // the current bar's live accumulating tick-volume.  When
                // Calculate=OnBarClose the two are identical; when
                // Calculate=OnEachTick the live value starts at 1 and grows,
                // which would cause every tick to fail the volume-surge filter.
                double vol = st.LastVolume > 0 ? st.LastVolume : bars.GetVolume(last);
                // Use actual filled-bar count for the avg so early bars don't
                // compare against a zero-diluted average.
                double volAvg = st.VolFilled > 0
                    ? st.VolSum / Math.Min(st.VolFilled, st.VolBuffer.Length)
                    : 0;
                double atr = st.AtrValue;
                string instrName = bars.Instrument.MasterInstrument.Name;
                DateTime barTime = bars.GetTime(last);

                // ── Periodic status log ───────────────────────────────────────
                if (EnableDebugLogging && last != st.LastLoggedBar && last % DebugLogFrequency == 0)
                {
                    st.LastLoggedBar = last;
                    Print($"[Breakout DEBUG] BIP{bip} {instrName} | ORB H={st.OrbHigh:F6} L={st.OrbLow:F6} | Close={close:F6} | VWAP={st.Vwap:F6} | Vol={vol:F0} (avg={volAvg:F0}) | ATR={atr:F6}");
                }

                // ── Shared pre-flight filters (volume, cooldown) ──────────────
                // Volume surge filter: applied globally so all range types benefit.
                if (volAvg > 0 && vol < volAvg * VolumeSurgeMult)
                {
                    if (EnableDebugLogging && last != st.LastLoggedBar && (close > st.OrbHigh || close < st.OrbLow))
                    {
                        st.LastLoggedBar = last;
                        Print($"[Breakout DEBUG] BIP{bip} {instrName} FILTERED (volume surge) vol={vol:F0} < {VolumeSurgeMult}x avg={volAvg:F0}");
                    }
                    return;
                }

                // Cooldown: prevent multiple signals within EntryCooldownMinutes
                if ((barTime - st.LastEntryTime).TotalMinutes < EntryCooldownMinutes)
                {
                    if (EnableDebugLogging && last != st.LastLoggedBar && (close > st.OrbHigh || close < st.OrbLow))
                    {
                        st.LastLoggedBar = last;
                        Print($"[Breakout DEBUG] BIP{bip} {instrName} FILTERED (cooldown active, last entry {st.LastEntryTime:HH:mm:ss})");
                    }
                    return;
                }

                // ── Evaluate every established range type ─────────────────────
                foreach (var kv in st.Ranges)
                {
                    BreakoutType type = kv.Key;
                    var rs = kv.Value;

                    if (!rs.RangeEstablished) continue;

                    // ── ATR range filter (ORB only — IB/PrevDay are intrinsically large) ──
                    if (type == BreakoutType.ORB && MinOrbAtrRatio > 0 && atr > 0)
                    {
                        double rangeSize = rs.RangeHigh - rs.RangeLow;
                        if (rangeSize < atr * MinOrbAtrRatio)
                        {
                            if (EnableDebugLogging && last != st.LastLoggedBar)
                            {
                                st.LastLoggedBar = last;
                                Print($"[Breakout DEBUG] BIP{bip} {instrName} [{type}] FILTERED (range too small) {rangeSize:F6} < {MinOrbAtrRatio}xATR={atr * MinOrbAtrRatio:F6}");
                            }
                            continue;
                        }
                    }

                    bool longBreak = close > rs.RangeHigh;
                    bool shortBreak = close < rs.RangeLow;

                    // ── Long breakout ─────────────────────────────────────────
                    if (longBreak)
                    {
                        bool vwapOk = !RequireVwap || st.Vwap <= 0 || close > st.Vwap;

                        if (!rs.FiredLong)
                        {
                            // Fresh entry — standard path
                            if (vwapOk && PassesCnnFilter("long", bip, st, close, atr, barTime, instrName, type))
                            {
                                FireEntry("long", bip, st, close, atr, barTime, instrName, type);
                                rs.FiredLong = true;
                                if (type == BreakoutType.ORB) st.BreakoutFiredLong = true;
                            }
                        }
                        else if (sar != null && sar.IsShort)
                        {
                            // ── SAR path: existing SHORT position, new LONG breakout ──
                            // Retrieve CNN score so the reversal gates can be evaluated.
                            double cnnProb = 0;
                            bool cnnPass = !EnableCnnFilter || PassesCnnFilter(
                                "long", bip, st, close, atr, barTime, instrName, type,
                                out cnnProb);

                            // Real MTF score — computed from 15m EMA/MACD state.
                            // Falls back to 1.0 pass-through during the 50-bar EMA warm-up.
                            double mtfScore = GetMtfScore(st, "long");

                            if (vwapOk && ShouldReverse(sar, "long", cnnProb, mtfScore, close, barTime))
                            {
                                TryReversePosition("long", bip, st, sar, close, atr, barTime, instrName, type);
                                // FiredLong reset inside TryReversePosition; set it again after fire
                                rs.FiredLong = true;
                                if (type == BreakoutType.ORB) st.BreakoutFiredLong = true;
                            }
                        }
                    }

                    // ── Short breakout ────────────────────────────────────────
                    if (shortBreak)
                    {
                        bool vwapOk = !RequireVwap || st.Vwap <= 0 || close < st.Vwap;

                        if (!rs.FiredShort)
                        {
                            // Fresh entry — standard path
                            if (vwapOk && PassesCnnFilter("short", bip, st, close, atr, barTime, instrName, type))
                            {
                                FireEntry("short", bip, st, close, atr, barTime, instrName, type);
                                rs.FiredShort = true;
                                if (type == BreakoutType.ORB) st.BreakoutFiredShort = true;
                            }
                        }
                        else if (sar != null && sar.IsLong)
                        {
                            // ── SAR path: existing LONG position, new SHORT breakout ──
                            double cnnProb = 0;
                            bool cnnPass = !EnableCnnFilter || PassesCnnFilter(
                                "short", bip, st, close, atr, barTime, instrName, type,
                                out cnnProb);

                            // Real MTF score — computed from 15m EMA/MACD state.
                            double mtfScore = GetMtfScore(st, "short");

                            if (vwapOk && ShouldReverse(sar, "short", cnnProb, mtfScore, close, barTime))
                            {
                                TryReversePosition("short", bip, st, sar, close, atr, barTime, instrName, type);
                                rs.FiredShort = true;
                                if (type == BreakoutType.ORB) st.BreakoutFiredShort = true;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"[Breakout] CheckBreakout BIP{bip}: {ex.Message}");
            }
        }

        #endregion

        // =====================================================================
        // CNN filter gate
        // =====================================================================
        #region CNN Filter

        /// <summary>
        /// PassesCnnFilter overload that also returns the raw CNN probability.
        /// Used by the SAR reversal path so reversal gates can evaluate the
        /// numeric score without running inference twice.
        ///
        /// Internally delegates to the original overload after running inference
        /// directly so the probability is available.  When the CNN is disabled
        /// or unavailable, returns (true, 1.0) so gates evaluate conservatively.
        /// </summary>
        private bool PassesCnnFilter(string direction, int bip, InstrumentState st,
                                     double close, double atr, DateTime barTime,
                                     string instrName, BreakoutType type,
                                     out double cnnProbOut)
        {
            cnnProbOut = 1.0; // safe default: pass-through with full confidence

            // CNN disabled or model not loaded — pass through
            if (!EnableCnnFilter || _cnn == null)
                return true;

            // ATR not yet warmed — pass through (matches original fast path)
            if (!st.AtrReady)
                return true;

            try
            {
                string detectedSession = DetectSessionKey(barTime);
                float threshold = CnnThresholdOverride > 0
                    ? (float)CnnThresholdOverride
                    : GetSessionThreshold(detectedSession);

                // Use the same helper methods as the original overload
                float[] tabular = PrepareCnnTabular(bip, st, direction, close, atr, barTime,
                                                    breakoutType: type);
                if (tabular == null)
                    return true; // can't build features — pass through

                string snapshotPath = RenderCnnSnapshot(bip, st, direction, instrName, barTime, type);

                var pred = _cnn.Predict(snapshotPath ?? "", tabular, threshold);
                if (pred == null)
                    return true;

                cnnProbOut = pred.Probability;
                return pred.Signal;
            }
            catch (Exception ex)
            {
                Print($"[CNN] SAR prob-overload exception for {instrName}: {ex.Message}");
                return true; // fail-open
            }
        }

        /// <summary>
        /// Returns true if the breakout candidate passes the CNN filter (or if
        /// the filter is disabled / unavailable).  Handles all rendering and
        /// inference; callers just get a boolean.
        /// </summary>
        private bool PassesCnnFilter(
            string direction, int bip, InstrumentState st,
            double price, double atr, DateTime barTime, string instrName,
            BreakoutType breakoutType = BreakoutType.ORB)
        {
            // Fast path — filter disabled or model not loaded
            if (!EnableCnnFilter || _cnn == null) return true;

            // Fast path — ATR not yet warmed up (fewer than 14 bars seen).
            // The tabular feature vector uses atr_pct, quality_pct_norm, and
            // nr7_flag, all of which are meaningless while ATR is seeded from a
            // single bar.  Pass through rather than waste an inference call on
            // unreliable inputs.
            if (!st.AtrReady)
            {
                if (EnableDebugLogging)
                    Print($"[Breakout+CNN] {direction.ToUpper()} {instrName} BIP{bip} " +
                          $"SKIPPED (ATR warmup in progress, {st.AtrFilled}/14 bars)");
                return true;
            }

            try
            {
                // ── Resolve threshold — dynamic session detection ─────────────
                string detectedSession = DetectSessionKey(barTime);
                float threshold = CnnThresholdOverride > 0
                    ? (float)CnnThresholdOverride
                    : GetSessionThreshold(detectedSession);

                // ── Build tabular feature vector ──────────────────────────────
                float[] tabular = PrepareCnnTabular(bip, st, direction, price, atr, barTime,
                                                    breakoutType: breakoutType);
                if (tabular == null) return true; // can't build features → pass through

                // ── Render chart snapshot ─────────────────────────────────────
                string snapshotPath = RenderCnnSnapshot(bip, st, direction, instrName, barTime,
                                                        breakoutType);
                // snapshotPath may be null if rendering failed — predictor handles it gracefully

                // ── Run inference ─────────────────────────────────────────────
                var result = _cnn.Predict(snapshotPath ?? "", tabular, threshold);
                if (result == null) return true; // inference error → pass through

                if (result.Signal)
                {
                    Interlocked.Increment(ref _metricCnnAccepted);
                    Print($"[Breakout+CNN] {direction.ToUpper()} {instrName} BIP{bip} " +
                          $"ACCEPTED  session={detectedSession} {result}");
                    return true;
                }
                else
                {
                    Interlocked.Increment(ref _metricCnnRejected);
                    Print($"[Breakout+CNN] {direction.ToUpper()} {instrName} BIP{bip} " +
                          $"REJECTED  session={detectedSession} {result}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                Print($"[Breakout] CNN filter error BIP{bip}: {ex.Message}");
                return true; // fail-open: don't block on unexpected error
            }
        }

        /// <summary>
        /// Build the 28-element raw tabular feature vector per feature_contract.json v7.1.
        /// Raw values — normalisation is applied inside OrbCnnPredictor.NormaliseTabular.
        ///
        /// Order (must match feature_contract.json tabular_features exactly):
        ///   [0]  quality_pct_norm       — estimated signal quality / 100
        ///   [1]  volume_ratio           — current bar vol / rolling avg vol
        ///   [2]  atr_pct               — ATR / close price (fraction, not percent)
        ///   [3]  cvd_delta             — cumulative signed volume / total volume [-1,1]
        ///   [4]  nr7_flag              — 1 if today's range is narrowest of 7 sessions
        ///   [5]  direction_flag        — 1 = LONG, 0 = SHORT
        ///   [6]  session_ordinal       — dynamic from DetectSessionKey(barTime)
        ///   [7]  london_overlap_flag   — 1 if bar hour is 08:00–09:00 ET
        ///   [8]  or_range_atr_ratio    — ORB range / ATR (raw, normalised later)
        ///   [9]  premarket_range_ratio — premarket range / ORB range (raw)
        ///   [10] bar_of_day            — minutes since session open / 1380
        ///   [11] day_of_week           — weekday index / 4 (Mon=0, Fri=4)
        ///   [12] vwap_distance         — (price - vwap) / ATR (raw, normalised later)
        ///   [13] asset_class_id        — ordinal / 4.0 from asset class map
        ///   [14] breakout_type_ord     — (int)breakoutType / 12.0  (v6)
        ///   [15] asset_volatility_class — 0.0 low / 0.5 med / 1.0 high  (v6)
        ///   [16] hour_of_day           — barTime.Hour / 23.0  (v6)
        ///   [17] tp3_atr_mult_norm     — tp3AtrMult / 5.0  (v6)
        ///   [18] daily_bias_direction  — SHORT=0.0, NEUTRAL=0.5, LONG=1.0  (v7)
        ///   [19] daily_bias_confidence — 0.0–1.0 from daily bias analysis  (v7)
        ///   [20] prior_day_pattern     — candle pattern ordinal / 9  (v7)
        ///   [21] weekly_range_position — price in prior week H/L range [0,1]  (v7)
        ///   [22] monthly_trend_score   — EMA slope proxy [0,1]  (v7)
        ///   [23] crypto_momentum_score — 0.5 neutral (v7, placeholder)
        ///   [24] breakout_type_category — time=0, range=0.5, squeeze=1.0  (v7.1)
        ///   [25] session_overlap_flag  — 1.0 if London+NY overlap  (v7.1)
        ///   [26] atr_trend             — expanding=1.0, contracting=0.0  (v7.1)
        ///   [27] volume_trend          — 5-bar vol slope [0,1]  (v7.1)
        /// </summary>
        private float[] PrepareCnnTabular(
            int bip, InstrumentState st,
            string direction, double price, double atr, DateTime barTime,
            int expectedDim = CNumTabularFeatures,
            BreakoutType breakoutType = BreakoutType.ORB)
        {
            try
            {
                // [0] quality_pct_norm: proxy using ORB range relative to ATR.
                //     A wide, clean ORB (range >= 1×ATR) scores near 1.0.
                //     This is imperfect but keeps the tabular head useful while
                //     Ruby's quality_pct is not yet wired through to C#.
                float qualityNorm = 0.5f;
                double orbRange = st.OrbHigh - st.OrbLow;
                if (atr > 0 && orbRange > 0)
                {
                    qualityNorm = (float)Math.Min(orbRange / (atr * 1.5), 1.0);
                }

                // [1] volume_ratio — breakout bar volume / rolling average volume
                double volAvg = st.VolFilled > 0
                    ? st.VolSum / Math.Min(st.VolFilled, st.VolBuffer.Length)
                    : 0;
                float volRatio = volAvg > 0
                    ? (float)(BarsArray[bip].GetVolume(BarsArray[bip].Count - 1) / volAvg)
                    : 1f;

                // [2] atr_pct — ATR as a fraction of price
                float atrPct = price > 0 && atr > 0 ? (float)(atr / price) : 0f;

                // [3] cvd_delta — cumulative signed volume from session start,
                //     normalised by total volume.  Replaces the old stub (always 0).
                //     Positive = net buying pressure, Negative = net selling.
                float cvdDelta = 0f;
                if (st.CvdTotalVol > 0)
                    cvdDelta = (float)Math.Max(-1.0, Math.Min(1.0, st.CvdSignedVol / st.CvdTotalVol));

                // [4] nr7_flag — true NR7: today's session range is the narrowest
                //     of the last 7 sessions.  Uses actual daily range history
                //     instead of the old ATR heuristic.
                float nr7Flag = 0f;
                if (st.DailyRangeFilled >= 6 && orbRange > 0)
                {
                    // Current session range (use ORB range as proxy for current
                    // session if full session range not yet available)
                    double currentRange = (st.SessionHigh > double.MinValue && st.SessionLow < double.MaxValue)
                        ? st.SessionHigh - st.SessionLow
                        : orbRange;
                    bool isNarrowest = true;
                    for (int i = 0; i < st.DailyRangeFilled && i < 6; i++)
                    {
                        if (st.DailyRangeHistory[i] > 0 && currentRange >= st.DailyRangeHistory[i])
                        {
                            isNarrowest = false;
                            break;
                        }
                    }
                    nr7Flag = isNarrowest ? 1f : 0f;
                }
                else if (atr > 0)
                {
                    // Fallback to ATR heuristic when not enough daily history
                    nr7Flag = (orbRange < atr * 0.5) ? 1f : 0f;
                }

                // [5] direction_flag
                float dirFlag = direction.Equals("long", StringComparison.OrdinalIgnoreCase) ? 1f : 0f;

                // [6] session_ordinal — dynamically detected from bar time
                string detectedSession = DetectSessionKey(barTime);
                float sessionOrdinal = GetSessionOrdinal(detectedSession);

                // [7] london_overlap_flag — 08:00–09:00 ET
                float londonOverlap = (barTime.Hour >= 8 && barTime.Hour < 10) ? 1f : 0f;

                // [8] or_range_atr_ratio — ORB range / ATR (raw; normalised later)
                float orRangeAtrRatio = (atr > 0) ? (float)(orbRange / atr) : 0f;

                // [9] premarket_range_ratio — premarket range / ORB range (raw)
                float premarketRangeRatio = 0f;
                if (orbRange > 0 && st.PremarketHigh > double.MinValue && st.PremarketLow < double.MaxValue)
                {
                    double pmRange = st.PremarketHigh - st.PremarketLow;
                    premarketRangeRatio = (float)(pmRange / orbRange);
                }

                // [10] bar_of_day — minutes since Globex session open (18:00 ET) / 1380
                //      Globex session starts at 18:00 ET = 6pm.  If bar is after 18:00
                //      the minutes = (hour-18)*60+min.  If before, add 24h.
                int barHour = barTime.Hour;
                int barMinute = barTime.Minute;
                int minutesSinceOpen;
                if (barHour >= 18)
                    minutesSinceOpen = (barHour - 18) * 60 + barMinute;
                else
                    minutesSinceOpen = (barHour + 6) * 60 + barMinute; // +6 = 24-18
                float barOfDay = (float)(minutesSinceOpen / 1380.0);

                // [11] day_of_week — Monday=0, Friday=4, scaled /4.0
                int dow = (int)barTime.DayOfWeek;
                // DayOfWeek: Sun=0, Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Sat=6
                // Map to Mon=0..Fri=4 for contract compatibility
                float dayOfWeek = dow >= 1 && dow <= 5 ? (float)(dow - 1) / 4f : 0.5f;

                // [12] vwap_distance — (price - vwap) / ATR (raw; normalised later)
                float vwapDistance = 0f;
                if (atr > 0 && st.Vwap > 0)
                    vwapDistance = (float)((price - st.Vwap) / atr);

                // [13] asset_class_id — ordinal encoding / 4.0
                string instrRoot = BarsArray[bip].Instrument.MasterInstrument.Name;
                float assetClassId = GetAssetClassNorm(instrRoot);

                // ── v6 features [14]–[17] ────────────────────────────────────

                // [14] breakout_type_ord — ordinal of the BreakoutType enum / 12.0
                //      The caller (PassesCnnFilter via CheckBreakout) now passes the
                //      actual type that triggered the signal, so this is precise.
                float breakoutTypeOrd = (int)breakoutType / 12.0f;

                // [15] asset_volatility_class — 0.0 (low), 0.5 (med), 1.0 (high)
                float assetVolClass = GetVolatilityClass(instrRoot);

                // [16] hour_of_day — ET hour normalised to [0, 1]
                float hourOfDay = barTime.Hour / 23.0f;

                // [17] tp3_atr_mult_norm — TP3 multiplier / 5.0
                //      Read the per-type value that was loaded from feature_contract.json
                //      via ParseTp3MultsFromContract + ApplyTp3MultsToStates.
                //      Falls back to 5.0 (ORB default) when the range state is missing.
                double tp3AtrMult = 5.0;
                if (st.Ranges.TryGetValue(breakoutType, out var tp3Rs))
                    tp3AtrMult = tp3Rs.Tp3AtrMult;
                float tp3AtrMultNorm = (float)(tp3AtrMult / 5.0);

                // ── v7 features [18]–[23] — Daily Strategy layer ─────────────

                // [18] daily_bias_direction — SHORT=0.0, NEUTRAL=0.5, LONG=1.0
                //      Computed from prior day candle analysis.
                //      Uses a simplified heuristic matching Python bias_analyzer:
                //      if prior day closed in upper 25% → LONG (1.0),
                //      if prior day closed in lower 25% → SHORT (0.0),
                //      otherwise NEUTRAL (0.5).
                float dailyBiasDirection = 0.5f;
                if (st.PrevDayValid && st.PrevDayHigh > st.PrevDayLow)
                {
                    double dayRange = st.PrevDayHigh - st.PrevDayLow;
                    double closePos = (st.PrevDayClose - st.PrevDayLow) / dayRange;
                    if (closePos >= 0.75) dailyBiasDirection = 1.0f;       // LONG
                    else if (closePos <= 0.25) dailyBiasDirection = 0.0f;  // SHORT
                    // else NEUTRAL = 0.5
                }

                // [19] daily_bias_confidence — 0.0–1.0
                //      Higher when the prior day close is further from the midpoint
                //      and the range is meaningful relative to ATR.
                float dailyBiasConfidence = 0.0f;
                if (st.PrevDayValid && st.PrevDayHigh > st.PrevDayLow && atr > 0)
                {
                    double dayRange = st.PrevDayHigh - st.PrevDayLow;
                    double closePos = (st.PrevDayClose - st.PrevDayLow) / dayRange;
                    // Distance from midpoint (0.5) — max is 0.5
                    double distFromMid = Math.Abs(closePos - 0.5);
                    // Scale: 0.5 distance → 1.0 confidence, 0 distance → 0.0
                    double rangeQuality = Math.Min(dayRange / (atr * 1.5), 1.0);
                    dailyBiasConfidence = (float)Math.Min(1.0, distFromMid * 2.0 * rangeQuality);
                }

                // [20] prior_day_pattern — candle pattern ordinal / 9
                //      Simplified pattern detection matching Python bias_analyzer:
                //      inside=0, doji=1, engulfing_bull=2, engulfing_bear=3,
                //      hammer=4, shooting_star=5, strong_close_up=6,
                //      strong_close_down=7, outside_day=8, neutral=9
                float priorDayPattern = 1.0f; // default: neutral (9/9)
                if (st.PrevDayValid && st.PrevDayHigh > st.PrevDayLow)
                {
                    double body = Math.Abs(st.PrevDayClose - st.PrevDayOpen);
                    double dayRange = st.PrevDayHigh - st.PrevDayLow;
                    double bodyRatio = dayRange > 0 ? body / dayRange : 0;
                    double closePos = (st.PrevDayClose - st.PrevDayLow) / dayRange;

                    if (bodyRatio < 0.10)
                        priorDayPattern = 1.0f / 9.0f;  // doji
                    else if (closePos >= 0.75 && bodyRatio >= 0.60)
                        priorDayPattern = 6.0f / 9.0f;  // strong_close_up
                    else if (closePos <= 0.25 && bodyRatio >= 0.60)
                        priorDayPattern = 7.0f / 9.0f;  // strong_close_down
                    else if (closePos >= 0.70 && bodyRatio < 0.35)
                        priorDayPattern = 4.0f / 9.0f;  // hammer
                    else if (closePos <= 0.30 && bodyRatio < 0.35)
                        priorDayPattern = 5.0f / 9.0f;  // shooting_star
                    else
                        priorDayPattern = 9.0f / 9.0f;  // neutral
                }

                // [21] weekly_range_position — price in prior week H/L [0, 1]
                float weeklyRangePosition = 0.5f;
                if (st.PrevWeekValid && st.PrevWeekHigh > st.PrevWeekLow && price > 0)
                {
                    double weekRange = st.PrevWeekHigh - st.PrevWeekLow;
                    weeklyRangePosition = (float)Math.Max(0.0, Math.Min(1.0,
                        (price - st.PrevWeekLow) / weekRange));
                }

                // [22] monthly_trend_score — proxy using ATR trend as EMA slope
                //      Python uses 20-day EMA slope on daily bars.  In C# we
                //      approximate with the ATR expansion/contraction direction
                //      since we don't have daily EMA readily available.
                //      Maps to [0, 1]: 0.0 = strong downtrend, 0.5 = flat, 1.0 = strong uptrend.
                float monthlyTrendScore = 0.5f;
                if (st.PrevDayValid && price > 0 && st.PrevDayClose > 0)
                {
                    // Simple proxy: current price vs prior day close gives short-term trend
                    double pctChange = (price - st.PrevDayClose) / st.PrevDayClose;
                    // Normalise: ±2% → [0, 1]
                    monthlyTrendScore = (float)Math.Max(0.0, Math.Min(1.0,
                        (pctChange / 0.02) * 0.5 + 0.5));
                }

                // [23] crypto_momentum_score — placeholder (0.5 = neutral)
                //      In Python this reads from Kraken crypto data.  C# does not
                //      have access to Kraken feeds, so we use the neutral default.
                //      The model tolerates this — crypto momentum is additive, not
                //      critical for non-crypto assets.
                float cryptoMomentumScore = 0.5f;

                // ── v7.1 features [24]–[27] — Phase 4B sub-feature decomposition ──

                // [24] breakout_type_category — coarse grouping:
                //      time-based=0.0 (ORB, Asian, IB), range-based=0.5
                //      (PDR, Weekly, Monthly, VA, Inside, Gap, Pivot, Fib),
                //      squeeze-based=1.0 (Consolidation, BollingerSqueeze)
                float breakoutTypeCategory = 0.5f; // default: range-based
                switch (breakoutType)
                {
                    case BreakoutType.ORB:
                    case BreakoutType.Asian:
                    case BreakoutType.InitialBalance:
                        breakoutTypeCategory = 0.0f; // time-based
                        break;
                    case BreakoutType.Consolidation:
                    case BreakoutType.BollingerSqueeze:
                        breakoutTypeCategory = 1.0f; // squeeze-based
                        break;
                    default:
                        breakoutTypeCategory = 0.5f; // range-based
                        break;
                }

                // [25] session_overlap_flag — 1.0 if London+NY overlap (08:00–12:00 ET)
                //      Captures the highest-volume intraday window when both
                //      London and New York are fully active.
                float sessionOverlapFlag = (barTime.Hour >= 8 && barTime.Hour < 12) ? 1.0f : 0.0f;

                // [26] atr_trend — ATR expanding (1.0) or contracting (0.0)
                //      Compares the most recent ATR to the oldest in the 10-bar
                //      ring buffer.  If ATR is rising → expanding → 1.0.
                float atrTrend = 0.5f;
                if (st.AtrHistFilled >= 2)
                {
                    // Oldest value in ring buffer
                    int oldestIdx = st.AtrHistFilled >= 10
                        ? st.AtrHistIdx  // buffer is full, oldest is at current write position
                        : 0;             // buffer not full, oldest is at index 0
                    double oldestAtr = st.AtrHistory[oldestIdx];
                    double newestAtr = st.AtrValue;
                    if (oldestAtr > 0)
                    {
                        double atrChange = (newestAtr - oldestAtr) / oldestAtr;
                        // Map: -10% or worse → 0.0, 0% → 0.5, +10% or more → 1.0
                        atrTrend = (float)Math.Max(0.0, Math.Min(1.0,
                            (atrChange / 0.10) * 0.5 + 0.5));
                    }
                }

                // [27] volume_trend — 5-bar volume slope normalised [0, 1]
                //      1.0 = rising sharply, 0.5 = flat, 0.0 = declining sharply.
                float volumeTrend = 0.5f;
                if (st.VolTrendFilled >= 2)
                {
                    // Simple slope: compare newest vs oldest in the 5-bar buffer
                    int oldestVolIdx = st.VolTrendFilled >= 5
                        ? st.VolTrendIdx
                        : 0;
                    double oldestVol = st.VolTrendBuf[oldestVolIdx];
                    // Newest is the entry just before the current write index
                    int newestVolIdx = (st.VolTrendIdx - 1 + 5) % 5;
                    double newestVol = st.VolTrendBuf[newestVolIdx];
                    if (oldestVol > 0)
                    {
                        double volChange = (newestVol - oldestVol) / oldestVol;
                        // Map: -50% or worse → 0.0, 0% → 0.5, +50% or more → 1.0
                        volumeTrend = (float)Math.Max(0.0, Math.Min(1.0,
                            (volChange / 0.50) * 0.5 + 0.5));
                    }
                }

                return new float[]
                {
                    qualityNorm,          // [0]
                    volRatio,             // [1]
                    atrPct,               // [2]
                    cvdDelta,             // [3]
                    nr7Flag,              // [4]
                    dirFlag,              // [5]
                    sessionOrdinal,       // [6]
                    londonOverlap,        // [7]
                    orRangeAtrRatio,      // [8]
                    premarketRangeRatio,  // [9]
                    barOfDay,             // [10]
                    dayOfWeek,            // [11]
                    vwapDistance,          // [12]
                    assetClassId,         // [13]
                    breakoutTypeOrd,      // [14]  v6
                    assetVolClass,        // [15]  v6
                    hourOfDay,            // [16]  v6
                    tp3AtrMultNorm,       // [17]  v6
                    dailyBiasDirection,   // [18]  v7
                    dailyBiasConfidence,  // [19]  v7
                    priorDayPattern,      // [20]  v7
                    weeklyRangePosition,  // [21]  v7
                    monthlyTrendScore,    // [22]  v7
                    cryptoMomentumScore,  // [23]  v7
                    breakoutTypeCategory, // [24]  v7.1
                    sessionOverlapFlag,   // [25]  v7.1
                    atrTrend,             // [26]  v7.1
                    volumeTrend,          // [27]  v7.1
                };
            }
            catch (Exception ex)
            {
                Print($"[Breakout] PrepareCnnTabular BIP{bip}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Render a chart snapshot PNG for the current bar and return the path.
        /// Returns null if rendering fails — the predictor will zero the image
        /// tensor and rely on the tabular branch.
        /// The rangeHigh/rangeLow box is drawn using the actual range that triggered
        /// this signal (breakoutType), falling back to the ORB levels if unavailable.
        /// </summary>
        private string RenderCnnSnapshot(
            int bip, InstrumentState st,
            string direction, string instrName, DateTime barTime,
            BreakoutType breakoutType = BreakoutType.ORB)
        {
            try
            {
                var bars = BarsArray[bip];
                int numBars = Math.Min(CnnLookbackBars, bars.Count);
                if (numBars < 5) return null;

                int startIdx = bars.Count - numBars;
                var barArr = new OrbChartRenderer.Bar[numBars];

                for (int i = 0; i < numBars; i++)
                {
                    int idx = startIdx + i;
                    barArr[i] = new OrbChartRenderer.Bar(
                        bars.GetTime(idx),
                        bars.GetOpen(idx),
                        bars.GetHigh(idx),
                        bars.GetLow(idx),
                        bars.GetClose(idx),
                        bars.GetVolume(idx));
                }

                // Build a VWAP array parallel to barArr from the VWAP history
                // ring buffer.  This produces a dynamic VWAP curve matching
                // the Python training images instead of a flat line.
                var vwapArr = new double[barArr.Length];
                if (st.VwapHistFilled >= numBars)
                {
                    // Enough history — read the last numBars values from the ring buffer.
                    // VwapHistIdx points to the next write slot, so the most recent
                    // value is at (VwapHistIdx - 1), and we need numBars going back.
                    int histLen = st.VwapHistory.Length;
                    for (int vi = 0; vi < numBars; vi++)
                    {
                        // Map: vwapArr[numBars-1] = most recent = VwapHistIdx-1
                        //       vwapArr[0]        = oldest       = VwapHistIdx-numBars
                        int ringIdx = ((st.VwapHistIdx - numBars + vi) % histLen + histLen) % histLen;
                        vwapArr[vi] = st.VwapHistory[ringIdx];
                    }
                }
                else if (st.VwapHistFilled > 0)
                {
                    // Partial history — fill what we have, pad older bars with earliest VWAP
                    int available = st.VwapHistFilled;
                    int histLen = st.VwapHistory.Length;
                    int padCount = numBars - available;

                    // Read available values from ring buffer (oldest first)
                    double earliestVwap = st.Vwap;
                    for (int vi = 0; vi < available; vi++)
                    {
                        int ringIdx = ((st.VwapHistIdx - available + vi) % histLen + histLen) % histLen;
                        vwapArr[padCount + vi] = st.VwapHistory[ringIdx];
                        if (vi == 0) earliestVwap = st.VwapHistory[ringIdx];
                    }
                    // Pad older bars with earliest available VWAP
                    for (int vi = 0; vi < padCount; vi++)
                        vwapArr[vi] = earliestVwap;
                }
                else
                {
                    // No history at all — flat line at current VWAP (fallback)
                    for (int vi = 0; vi < vwapArr.Length; vi++) vwapArr[vi] = st.Vwap;
                }

                // Label e.g. "MGC_long_20260304_1330" -- used as the PNG filename prefix.
                // Format the datetime separately to avoid any compiler issues with
                // special characters inside interpolated format specifiers.
                string snapTime = barTime.ToString("yyyyMMdd") + "_" + barTime.ToString("HHmm");
                string snapLabel = instrName + "_" + direction + "_" + snapTime;

                // Use the range H/L for the actual breakout type so the box in the
                // snapshot matches what triggered the signal (e.g. Asian, IB, etc.).
                // Fall back to ORB legacy fields if the range state isn't available.
                double snapRangeH = st.OrbHigh;
                double snapRangeL = st.OrbLow;
                if (st.Ranges.TryGetValue(breakoutType, out var snapRs) && snapRs.RangeEstablished
                    && snapRs.RangeHigh > snapRs.RangeLow)
                {
                    snapRangeH = snapRs.RangeHigh;
                    snapRangeL = snapRs.RangeLow;
                }

                return OrbChartRenderer.RenderToTemp(
                    barArr,
                    snapRangeH,
                    snapRangeL,
                    vwapArr,
                    direction,
                    breakoutType,
                    _cnnSnapshotFolder,
                    snapLabel);
            }
            catch (Exception ex)
            {
                Print($"[Breakout] RenderCnnSnapshot BIP{bip}: {ex.Message}");
                return null;
            }
        }

        #endregion

        // =====================================================================
        // Entry submission
        // =====================================================================
        #region Entry

        // ── Trading-hours template lookup ─────────────────────────────────────
        // Returns the correct NT8 trading-hours template name for a given
        // instrument root symbol.  These template names ship with every NT8
        // install and are not broker-specific.
        //
        // Why this matters:
        //   AddDataSeries(instrumentName, barsPeriod, tradingHoursName) is used
        //   in Configure.  There is no useChartTradingHours parameter in any NT8
        //   overload — the tradingHoursName string always controls which session
        //   window NT8 uses for historical bar loading and session-boundary
        //   detection.  Passing the correct ETH template name here prevents NT8
        //   from applying "US Equities RTH" (the default for the primary chart)
        //   to secondary BIPs, which would produce zero bars for all futures.
        //
        //   The template names returned here are verified to exist in the local
        //   NT8 templates directory on this machine.
        //
        // Template name reference (verified present in NT8 templates directory):
        //   "CME US Index Futures ETH"   — MES, MNQ, M2K, MYM
        //   "CME Commodities ETH"        — MCL, MNG
        //   "Nymex Metals - Energy ETH"  — MGC, SIL, MHG
        //   "CME FX Futures ETH"         — 6E, 6B, 6J, 6A, 6C, 6S
        //   "CBOT Interest Rate ETH"     — ZN, ZB
        //   "CBOT Agriculturals ETH"     — ZC, ZS, ZW
        //   "Cryptocurrency"             — MBT, MET
        //   "Default 24 x 5"            — fallback for any unrecognised symbol
        // =====================================================================
        // Dynamic session detection — maps bar time to session key
        // =====================================================================
        // Replaces the hardcoded CCnnSessionKey = "us" with a time-of-day
        // lookup so session_ordinal and per-session thresholds reflect the
        // actual trading session.  Hour boundaries are in ET (Eastern Time),
        // matching the Globex schedule defined in feature_contract.json.

        /// <summary>
        /// Detect the active trading session from the bar's timestamp (ET).
        /// Returns a session key string matching feature_contract.json session_ordinals.
        /// </summary>
        private static string DetectSessionKey(DateTime barTimeET)
        {
            int h = barTimeET.Hour;

            // 18:00–18:29 ET → CME Globex re-open
            if (h == 18) return "cme";
            // 18:30–18:59 ET → Sydney
            // (18:30 is approximated as h==18 already captured above)
            // 19:00–20:59 ET → Tokyo
            if (h >= 19 && h <= 20) return "tokyo";
            // 21:00–22:59 ET → Shanghai
            if (h >= 21 && h <= 22) return "shanghai";
            // 23:00–02:59 ET → Sydney (late) / early Frankfurt transition
            if (h >= 23 || h <= 2) return "sydney";
            // 03:00–07:59 ET → Frankfurt / London
            if (h >= 3 && h <= 4) return "frankfurt";
            if (h >= 5 && h <= 7) return "london";
            // 08:00–09:29 ET → London-NY overlap
            if (h == 8 || (h == 9 && barTimeET.Minute < 30)) return "london_ny";
            // 09:30–13:59 ET → US equity session
            if ((h == 9 && barTimeET.Minute >= 30) || (h >= 10 && h <= 13)) return "us";
            // 14:00–15:59 ET → CME settlement
            if (h >= 14 && h <= 15) return "cme_settle";
            // 16:00–17:59 ET → gap between sessions
            return "us"; // default fallback
        }

        /// <summary>
        /// Returns the asset class normalised ordinal (0..1) for a given instrument root.
        /// Matches the asset_class_map in feature_contract.json:
        ///   0=equity_index, 1=fx, 2=metals_energy, 3=treasuries_ags, 4=crypto
        /// </summary>
        private static float GetAssetClassNorm(string root)
        {
            switch (root.ToUpperInvariant())
            {
                // Equity index
                case "MES":
                case "MNQ":
                case "M2K":
                case "MYM":
                    return 0f / 4f;
                // FX
                case "6E":
                case "6B":
                case "6J":
                case "6A":
                case "6C":
                case "6S":
                    return 1f / 4f;
                // Metals / Energy
                case "MGC":
                case "SIL":
                case "MHG":
                case "MCL":
                case "MNG":
                    return 2f / 4f;
                // Treasuries / Ags
                case "ZN":
                case "ZB":
                case "ZC":
                case "ZS":
                case "ZW":
                    return 3f / 4f;
                // Crypto
                case "MBT":
                case "MET":
                    return 4f / 4f;
                default:
                    return 0f / 4f; // default to equity index
            }
        }

        /// <summary>
        /// Returns the asset volatility class for a given instrument root.
        /// Must match Python ASSET_VOLATILITY_CLASS dict in feature_contract.json v6:
        ///   High (1.0): MBT, MET, MCL, MNG, MNQ
        ///   Medium (0.5): MES, M2K, MYM, 6B, 6J, SIL, MHG, ZC, ZS, ZW
        ///   Low (0.0): MGC, 6E, 6A, 6C, 6S, ZN, ZB
        /// </summary>
        private static float GetVolatilityClass(string root)
        {
            switch (root.ToUpperInvariant())
            {
                // High volatility (1.0)
                case "MBT":
                case "MET":
                case "MCL":
                case "MNG":
                case "MNQ":
                    return 1.0f;
                // Medium volatility (0.5)
                case "MES":
                case "M2K":
                case "MYM":
                case "6B":
                case "6J":
                case "SIL":
                case "MHG":
                case "ZC":
                case "ZS":
                case "ZW":
                    return 0.5f;
                // Low volatility (0.0)
                case "MGC":
                case "6E":
                case "6A":
                case "6C":
                case "6S":
                case "ZN":
                case "ZB":
                    return 0.0f;
                default:
                    return 0.5f; // default to medium
            }
        }

        private static string GetTradingHoursTemplate(string root)
        {
            // Template names are the exact filenames (minus .xml) present in
            //   Documents\NinjaTrader 8\templates\TradingHours\
            // on this machine.  They were verified by listing that directory.
            // The "CME US *" family that ships with a full NT8 desktop install
            // is NOT present here — only the older/generic CME names are.
            switch (root.ToUpperInvariant())
            {
                // ── Equity index micros ───────────────────────────────────────
                // CME US Index Futures ETH  IS present on this machine.
                case "MES":
                case "MNQ":
                case "M2K":
                case "MYM":
                    return "CME US Index Futures ETH";

                // ── Energy micros ─────────────────────────────────────────────
                // "CME Commodities ETH" covers energy (CL, NG, etc.) on Globex ETH.
                case "MCL":
                case "MNG":
                    return "CME Commodities ETH";

                // ── Metals micros + regular metals (CME/COMEX) ───────────────
                // "Nymex Metals - Energy ETH" is the metals ETH template that
                // IS present and is already used for MGC on this machine.
                case "MGC":
                case "SIL":
                case "MHG":
                    return "Nymex Metals - Energy ETH";

                // ── Forex micros + regular forex futures ──────────────────────
                // "CME FX Futures ETH" IS present and covers both micro (M6x)
                // and regular (6x) FX futures.
                case "M6E":
                case "M6B":
                case "M6J":
                case "M6A":
                case "M6C":
                case "M6S":
                case "6E":
                case "6B":
                case "6J":
                case "6A":
                case "6C":
                case "6S":
                    return "CME FX Futures ETH";

                // ── Crypto micros ─────────────────────────────────────────────
                // "Cryptocurrency" IS present and is the correct template for
                // MBT/MET (23-hour sessions, no Sunday open gap).
                case "MBT":
                case "MET":
                    return "Cryptocurrency";

                // ── Agricultural micros + regular grains (CBOT) ──────────────
                // "CBOT Agriculturals ETH" IS present and covers ZC, ZS, ZW
                // and their micro equivalents.
                case "MZC":
                case "MZS":
                case "MZW":
                case "ZC":
                case "ZS":
                case "ZW":
                    return "CBOT Agriculturals ETH";

                // ── Regular treasury futures (CBOT) ───────────────────────────
                // ZN (10-Year Note) and ZB (30-Year Bond) trade nearly 24 hours
                // on Globex.  "CBOT Interest Rate ETH" covers the full ETH session
                // (18:00–17:00 ET, Sun–Fri) and IS present on this machine.
                case "ZN":
                case "ZB":
                    return "CBOT Interest Rate ETH";

                // ── Unknown symbol ────────────────────────────────────────────
                // "Default 24 x 5" is always present in every NT8 install and
                // spans 00:00–24:00 Mon–Fri, which is wide enough to capture any
                // overnight futures session.  Better than "Nymex Metals - Energy ETH"
                // as a catch-all because it never produces 0-bar results due to a
                // session-time mismatch.
                default:
                    return "Default 24 x 5";
            }
        }

        // ── TPT fixed-quantity lookup ─────────────────────────────────────────
        // Returns the conservative fixed contract count for the selected TPT tier.
        // Adjust these values within TPT's allowed maximums if you want to be more
        // aggressive — but never exceed 25 % of their stated per-instrument cap.
        private int GetTptContracts()
        {
            switch (AccountTier)
            {
                case TptAccountTier.FiftyK: return 2;  // allowed up to 60
                case TptAccountTier.HundredK: return 3;  // allowed up to 120
                case TptAccountTier.HundredFiftyK: return 4;  // allowed up to 150
                default: return 2;
            }
        }

        private void FireEntry(string direction, int bip, InstrumentState st,
                                double price, double atr, DateTime barTime, string instrName,
                                BreakoutType breakoutType = BreakoutType.ORB)
        {
            // ── SAR: stamp the new active direction on the ReversalState ─────
            // We do this at the top of FireEntry (before any early returns) so
            // TryReversePosition's pre-decremented active count is consistent
            // and the cooldown is stamped even if the order later gets rejected.
            ReversalState sarRef = (_sarStates != null && bip < _sarStates.Length)
                ? _sarStates[bip]
                : null;

            double tickSize = _engine.GetTickSize(bip);

            // ── Resolve the range that triggered this entry ───────────────────
            var rs = st.Ranges[breakoutType];

            // ── Compute ATR-based SL/TP ───────────────────────────────────────
            double sl, tp1, tp2;

            if (atr > 0)
            {
                if (direction == "long")
                {
                    // SL below the range low (tighter of range-low or ATR-based)
                    double rangeSl = rs.RangeLow - tickSize * 2;
                    double atrSl = price - atr * SlAtrMult;
                    sl = Math.Max(rangeSl, atrSl); // tighter = higher for longs
                    tp1 = price + atr * Tp1AtrMult;
                    tp2 = Tp2AtrMult > 0 ? price + atr * Tp2AtrMult : 0;
                }
                else
                {
                    double rangeSl = rs.RangeHigh + tickSize * 2;
                    double atrSl = price + atr * SlAtrMult;
                    sl = Math.Min(rangeSl, atrSl); // tighter = lower for shorts
                    tp1 = price - atr * Tp1AtrMult;
                    tp2 = Tp2AtrMult > 0 ? price - atr * Tp2AtrMult : 0;
                }
            }
            else
            {
                // ATR not ready — use tick-based defaults (engine will also fall back)
                sl = 0;
                tp1 = 0;
                tp2 = 0;
            }

            // ── Build a SignalBus.Signal and execute directly ──────────────────
            // Using SignalBus.Signal keeps the API consistent with Ruby's path
            // and lets the engine apply risk sizing in one place.
            string signalId = "brk-" + direction[0] + "-" + barTime.ToString("yyyyMMdd-HHmmss")
                              + "-" + instrName + "-" + breakoutType;

            // ── Max concurrent positions gate ───────────────────────────────
            if (_activePositionCount >= MaxConcurrentPositions)
            {
                if (EnableDebugLogging)
                    Print($"[Breakout DEBUG] BIP{bip} {instrName} FILTERED (max concurrent positions: {_activePositionCount}/{MaxConcurrentPositions})");
                return;
            }

            // ── Quantity: TPT fixed tier vs dynamic engine sizing ─────────────
            // In TPT mode we send the exact fixed count directly in the signal.
            // BridgeOrderEngine will still cap at MaxContracts, so set MaxContracts
            // >= GetTptContracts() or the engine will silently reduce the qty.
            // In dynamic mode we send 1 and let the engine risk-size upward from
            // there, capped at MaxContracts.
            int signalQty;
            if (TptMode)
            {
                signalQty = GetTptContracts();
                Print($"[Breakout] TPT mode: tier={AccountTier} qty={signalQty}");
            }
            else
            {
                signalQty = 1; // engine risk-sizes up to MaxContracts
            }

            var sig = new SignalBus.Signal
            {
                Direction = direction,
                SignalType = "entry",
                Quantity = signalQty,
                OrderType = "market",
                StopLoss = sl,
                TakeProfit = tp1,
                TakeProfit2 = tp2,
                Strategy = "BreakoutStrategy",
                Asset = instrName,
                SignalId = signalId,
                Timestamp = barTime,
            };

            // ── TP3 price (Phase3 target) ─────────────────────────────────────
            double tp3 = 0;
            if (EnableTp3Trailing && atr > 0)
                tp3 = direction == "long"
                    ? price + atr * Tp3AtrMult
                    : price - atr * Tp3AtrMult;

            // ── Determine final quantity (mirrors BridgeOrderEngine risk sizing) ─
            // We compute the same risk-sized qty here so PositionPhase is populated
            // with the right split before the engine applies its own sizing.
            // If TptMode the qty is fixed; otherwise we use the signal qty = 1 as
            // a placeholder and let the engine size it — phase qty fields will be
            // updated from the actual fill qty in OnOrderUpdate.
            int estimatedQty = signalQty;   // refined at fill in OnOrderUpdate
            int tp1Qty = tp2 > 0 ? Math.Max(1, estimatedQty / 2) : estimatedQty;
            int tp2Qty = tp2 > 0 ? estimatedQty - tp1Qty : 0;
            int tp3Qty = tp3 > 0 ? tp2Qty : 0;   // Phase3 remainder = tp2Qty contracts

            // ── Register position phase tracking ─────────────────────────────
            if (EnableTp3Trailing)
            {
                var phase = new PositionPhase
                {
                    SignalId = signalId,
                    Direction = direction,
                    Asset = instrName,
                    Bip = bip,
                    Phase = BreakoutPhase.Phase1,
                    EntryPrice = price,
                    AtrAtEntry = atr,
                    SlPrice = sl,
                    Tp1Price = tp1,
                    Tp2Price = tp2,
                    Tp3Price = tp3,
                    TotalQty = estimatedQty,
                    Tp1Qty = tp1Qty,
                    Tp2Qty = tp2Qty,
                    Tp3Qty = tp3Qty,
                    OcoGroup = "",
                    Tp3Submitted = false,
                    Ema9StopHit = false,
                };
                lock (_phaseLock)
                    _positionPhases[signalId] = phase;
            }

            // ── SAR: record the position in ReversalState ────────────────────
            // Called after the PositionPhase is registered so sarRef.ActiveSignalId
            // matches the key in _positionPhases.
            sarRef?.Open(direction, signalId, price, atr, sl, barTime);

            Print($"[Breakout] {breakoutType} {direction.ToUpper()} {instrName} BIP{bip} " +
                  $"@ {price:F2} SL={sl:F2} TP1={tp1:F2} TP2={tp2:F2}" +
                  (tp3 > 0 ? $" TP3={tp3:F2}" : "") +
                  $" id={signalId}" +
                  $" [positions: {_activePositionCount}/{MaxConcurrentPositions}]" +
                  (sarRef != null && !sarRef.IsFlat ? $" [SAR active={sarRef.ActiveDirection}]" : ""));

            // Execute directly (backtest) or queue (realtime)
            if (State == State.Historical)
                _engine.ExecuteEntryDirect(sig);
            else
                _engine.ProcessSignal(sig.ToJson());

            // ── SAR sync: push every entry (fresh + reversal) to the Python engine ──
            // TryReversePosition() calls PushSarSyncAsync after FireEntry for reversals.
            // Here we handle fresh entries (new position, no existing SAR).
            // Reversals are already pushed by TryReversePosition after FireEntry returns,
            // so we only push here when this is NOT a reversal call (i.e. the ReversalState
            // was flat before this entry — sarRef was IsFlat before sarRef.Open() above).
            // We detect this by checking the reversal count: if ReversalCount == 0 and
            // direction matches what we just opened, it's a fresh entry.
            if (sarRef != null && sarRef.ReversalCount == 0 && State != State.Historical)
                PushSarSyncAsync(instrName, direction, sarRef, barTime);

            // Cooldown is shared across all range types — one entry per window.
            // The per-range FiredLong/FiredShort flags (set by CheckBreakout) and
            // the legacy ORB flags (kept in sync there) handle per-type suppression.
            st.LastEntryTime = barTime;
        }

        #endregion

        // =====================================================================
        // Diagnostics
        // =====================================================================
        #region Diagnostics

        /// <summary>
        /// Return a brief status string suitable for the NinjaTrader Output window
        /// or a dashboard endpoint.
        /// </summary>
        public string GetStatusSummary()
        {
            if (_states == null) return "[BreakoutStrategy] Not initialized";

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("[BreakoutStrategy] Status");
            sb.AppendLine($"  Mode          : {Mode}");
            sb.AppendLine($"  Instruments   : {_symbolToBip.Count}");
            sb.AppendLine($"  RiskBlocked   : {RiskBlocked}");
            // ── Sizing mode summary ───────────────────────────────────────────
            if (TptMode)
                sb.AppendLine($"  Sizing        : TPT fixed  tier={AccountTier}  qty={GetTptContracts()} contracts");
            else
                sb.AppendLine($"  Sizing        : Dynamic ATR  risk={RiskPercentPerTrade}%  cap={MaxContracts}");
            sb.AppendLine($"  Signals rcvd  : {Interlocked.Read(ref _metricSignalsReceived)}");
            sb.AppendLine($"  Signals exec  : {Interlocked.Read(ref _metricSignalsExecuted)}");
            sb.AppendLine($"  Signals rej   : {Interlocked.Read(ref _metricSignalsRejected)}");
            sb.AppendLine($"  Bus drained   : {Interlocked.Read(ref _metricBusDrained)}");
            sb.AppendLine($"  Exits exec    : {Interlocked.Read(ref _metricExitsExecuted)}");

            // ── CNN filter summary ────────────────────────────────────────────
            if (EnableCnnFilter)
            {
                long accepted = Interlocked.Read(ref _metricCnnAccepted);
                long rejected = Interlocked.Read(ref _metricCnnRejected);
                long total = accepted + rejected;
                float passRate = total > 0 ? (float)accepted / total * 100f : 0f;

                string cnnStatus = _cnn != null
                    ? $"LOADED  {Path.GetFileName(_cnn.ModelPath)}"
                    : "NOT LOADED (filter pass-through active)";

                float effectiveThreshold = CnnThresholdOverride > 0
                    ? (float)CnnThresholdOverride
                    : GetSessionThreshold("us"); // status summary uses default session

                sb.AppendLine($"  CNN filter    : {cnnStatus}");
                sb.AppendLine($"  CNN session   : {CnnSessionKey}  threshold={effectiveThreshold:P0}");
                sb.AppendLine($"  CNN accepted  : {accepted}  rejected={rejected}  pass-rate={passRate:F1}%");
                sb.AppendLine($"  CNN snapshots : {_cnnSnapshotFolder}");
            }
            else
            {
                sb.AppendLine("  CNN filter    : disabled");
            }

            sb.AppendLine("  Active breakout types : Orb, PrevDay, InitialBalance, Consolidation");
            sb.AppendLine("  Ranges per instrument:");

            foreach (var kv in _symbolToBip)
            {
                int bipIdx = kv.Value;
                if (bipIdx >= _states.Length) continue;
                var st = _states[bipIdx];

                // ── ORB summary (legacy compact line) ─────────────────────────
                string firedStr = "";
                if (st.BreakoutFiredLong) firedStr += "L";
                if (st.BreakoutFiredShort) firedStr += "S";
                if (firedStr == "") firedStr = "-";
                string orbInfo = st.OrbEstablished
                    ? $"SET  H:{st.OrbHigh:F2} L:{st.OrbLow:F2} ATR:{st.AtrValue:F4} fired:{firedStr}"
                    : (st.OrbSessionDate != DateTime.MinValue ? "BUILDING" : "WAITING");
                sb.AppendLine($"    {kv.Key,-6} BIP{bipIdx}  ORB: {orbInfo}");

                // ── Additional range types ─────────────────────────────────────
                foreach (var rkv in st.Ranges)
                {
                    if (rkv.Key == BreakoutType.ORB) continue; // already printed above
                    var rs = rkv.Value;
                    string rf = "";
                    if (rs.FiredLong) rf += "L";
                    if (rs.FiredShort) rf += "S";
                    if (rf == "") rf = "-";
                    string rInfo = rs.RangeEstablished
                        ? $"SET  H:{rs.RangeHigh:F2} L:{rs.RangeLow:F2} fired:{rf}"
                        : (rs.SessionDate != DateTime.MinValue ? $"BUILDING ({rs.BarsInRange} bars)" : "WAITING");
                    sb.AppendLine($"           {rkv.Key,-18}: {rInfo}");
                }
            }

            return sb.ToString();
        }

        #endregion
    }
}

// =============================================================================
// SignalBus — inlined from SignalBus.cs
// =============================================================================

namespace NinjaTrader.NinjaScript
{
    public static class SignalBus
    {
        // ── Risk gate — shared between Bridge AddOn and BreakoutStrategy ──────
        // Bridge.cs writes these when the dashboard's risk endpoint returns
        // can_trade=false.  BreakoutStrategy reads them directly (no reflection)
        // on every BIP0 bar.  Volatile ensures cross-thread visibility without
        // a full lock — one writer (Bridge heartbeat thread), one reader
        // (strategy bar-sync thread).
        private static volatile bool _isRiskBlocked = false;
        private static volatile string _riskBlockReason = "";

        public static bool IsRiskBlocked { get { return _isRiskBlocked; } set { _isRiskBlocked = value; } }
        public static string RiskBlockReason { get { return _riskBlockReason; } set { _riskBlockReason = value ?? ""; } }

        public class Signal
        {
            public string Direction { get; set; }
            public string SignalType { get; set; }
            public int Quantity { get; set; }
            public string OrderType { get; set; }
            public double LimitPrice { get; set; }
            public double StopLoss { get; set; }
            public double TakeProfit { get; set; }
            public double TakeProfit2 { get; set; }
            public string Strategy { get; set; }
            public string Asset { get; set; }
            public string SignalId { get; set; }
            public double SignalQuality { get; set; }
            public double WaveRatio { get; set; }
            public DateTime Timestamp { get; set; }
            public string ExitReason { get; set; }

            public string ToJson()
            {
                return "{"
                    + "\"direction\":\"" + EscapeJson(Direction ?? "long") + "\""
                    + ",\"signal_type\":\"" + EscapeJson(SignalType ?? "entry") + "\""
                    + ",\"quantity\":" + Quantity
                    + ",\"order_type\":\"" + EscapeJson(OrderType ?? "market") + "\""
                    + ",\"limit_price\":" + LimitPrice.ToString("F6")
                    + ",\"stop_loss\":" + StopLoss.ToString("F6")
                    + ",\"take_profit\":" + TakeProfit.ToString("F6")
                    + ",\"tp2\":" + TakeProfit2.ToString("F6")
                    + ",\"strategy\":\"" + EscapeJson(Strategy ?? "") + "\""
                    + ",\"asset\":\"" + EscapeJson(Asset ?? "") + "\""
                    + ",\"signal_id\":\"" + EscapeJson(SignalId ?? "") + "\""
                    + ",\"signal_quality\":" + SignalQuality.ToString("F4")
                    + ",\"wave_ratio\":" + WaveRatio.ToString("F4")
                    + ",\"exit_reason\":\"" + EscapeJson(ExitReason ?? "") + "\""
                    + "}";
            }

            private static string EscapeJson(string value)
            {
                if (string.IsNullOrEmpty(value)) return "";
                return value
                    .Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("\n", "\\n")
                    .Replace("\r", "\\r")
                    .Replace("\t", "\\t");
            }
        }

        private static readonly ConcurrentQueue<Signal> _queue = new ConcurrentQueue<Signal>();
        private static volatile bool _bridgeRegistered;
        private static volatile int _totalEnqueued;
        private static volatile int _totalDrained;

        public static bool Enqueue(Signal signal)
        {
            if (signal == null) return false;
            if (string.IsNullOrEmpty(signal.SignalId))
                signal.SignalId = "bus-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmssfff")
                    + "-" + (_totalEnqueued & 0xFFFF).ToString("X4");
            if (signal.Timestamp == DateTime.MinValue)
                signal.Timestamp = DateTime.UtcNow;
            if (signal.Quantity <= 0)
                signal.Quantity = 1;
            if (string.IsNullOrEmpty(signal.OrderType))
                signal.OrderType = "market";
            if (string.IsNullOrEmpty(signal.SignalType))
                signal.SignalType = "entry";
            _queue.Enqueue(signal);
            System.Threading.Interlocked.Increment(ref _totalEnqueued);
            return _bridgeRegistered;
        }

        public static bool EnqueueEntry(
            string direction, string asset,
            double stopLoss = 0, double takeProfit = 0, double takeProfit2 = 0,
            int quantity = 1, string orderType = "market", double limitPrice = 0,
            double signalQuality = 0, double waveRatio = 0, string strategy = "Ruby")
        {
            return Enqueue(new Signal
            {
                Direction = direction,
                SignalType = "entry",
                Quantity = quantity,
                OrderType = orderType,
                LimitPrice = limitPrice,
                StopLoss = stopLoss,
                TakeProfit = takeProfit,
                TakeProfit2 = takeProfit2,
                Strategy = strategy,
                Asset = asset,
                SignalQuality = signalQuality,
                WaveRatio = waveRatio,
            });
        }

        public static bool EnqueueExit(string asset, string reason = "signal", string strategy = "Ruby")
        {
            return Enqueue(new Signal
            {
                Direction = "flat",
                SignalType = "exit",
                Quantity = 0,
                OrderType = "market",
                Strategy = strategy,
                Asset = asset,
                ExitReason = reason,
            });
        }

        public static void RegisterConsumer() { _bridgeRegistered = true; }
        public static void UnregisterConsumer() { _bridgeRegistered = false; }

        public static List<Signal> DrainAll()
        {
            var results = new List<Signal>();
            while (_queue.TryDequeue(out Signal signal))
            {
                results.Add(signal);
                System.Threading.Interlocked.Increment(ref _totalDrained);
            }
            return results;
        }

        public static int PendingCount { get { return _queue.Count; } }
        public static bool HasConsumer { get { return _bridgeRegistered; } }
        public static int TotalEnqueued { get { return _totalEnqueued; } }
        public static int TotalDrained { get { return _totalDrained; } }

        public static void Reset()
        {
            while (_queue.TryDequeue(out _)) { }
            _totalEnqueued = 0;
            _totalDrained = 0;
            _bridgeRegistered = false;
            _isRiskBlocked = false;
            _riskBlockReason = "";
        }
    }
}

// =============================================================================
// BridgeOrderEngine — inlined from BridgeOrderEngine.cs
// =============================================================================

namespace NinjaTrader.NinjaScript.Strategies
{
    internal sealed class BridgeOrderEngine
    {
        private readonly Strategy _strategy;
        private readonly Dictionary<string, int> _symbolToBip;
        private readonly Func<NinjaTrader.Cbi.Account> _getMyAccount;
        private readonly Func<double> _getAccountSize;
        private readonly Func<double> _getRiskPercent;
        private readonly Func<int> _getMaxContracts;
        private readonly Func<int> _getDefaultSlTicks;
        private readonly Func<int> _getDefaultTpTicks;
        private readonly Func<bool> _getAutoBrackets;
        private readonly Func<bool> _getRiskBlocked;
        private readonly Func<string> _getRiskBlockReason;
        private readonly Func<bool> _getRiskEnforcement;
        private readonly Action _onSignalReceived;
        private readonly Action _onSignalExecuted;
        private readonly Action _onSignalRejected;
        private readonly Action _onExitExecuted;
        private readonly Action<long> _onBusDrained;
        private readonly Action _sendPositionUpdate;
        private readonly Queue<Action> _orderQueue = new Queue<Action>();
        private readonly object _queueLock = new object();
        private readonly string _tag;

        internal BridgeOrderEngine(
            Strategy strategy,
            Dictionary<string, int> symbolToBip,
            Func<NinjaTrader.Cbi.Account> getMyAccount,
            Func<double> getAccountSize,
            Func<double> getRiskPercent,
            Func<int> getMaxContracts,
            Func<int> getDefaultSlTicks,
            Func<int> getDefaultTpTicks,
            Func<bool> getAutoBrackets,
            Func<bool> getRiskBlocked,
            Func<string> getRiskBlockReason,
            Func<bool> getRiskEnforcement,
            Action onSignalReceived,
            Action onSignalExecuted,
            Action onSignalRejected,
            Action onExitExecuted,
            Action<long> onBusDrained,
            Action sendPositionUpdate,
            string tag = "Engine")
        {
            _strategy = strategy;
            _symbolToBip = symbolToBip;
            _getMyAccount = getMyAccount;
            _getAccountSize = getAccountSize;
            _getRiskPercent = getRiskPercent;
            _getMaxContracts = getMaxContracts;
            _getDefaultSlTicks = getDefaultSlTicks;
            _getDefaultTpTicks = getDefaultTpTicks;
            _getAutoBrackets = getAutoBrackets;
            _getRiskBlocked = getRiskBlocked;
            _getRiskBlockReason = getRiskBlockReason;
            _getRiskEnforcement = getRiskEnforcement;
            _onSignalReceived = onSignalReceived;
            _onSignalExecuted = onSignalExecuted;
            _onSignalRejected = onSignalRejected;
            _onExitExecuted = onExitExecuted;
            _onBusDrained = onBusDrained;
            _sendPositionUpdate = sendPositionUpdate;
            _tag = tag;
        }

        internal void DrainSignalBus(State currentState)
        {
            var signals = SignalBus.DrainAll();
            if (signals.Count == 0) return;
            _onBusDrained?.Invoke(signals.Count);
            foreach (var sig in signals)
            {
                try
                {
                    if (sig.SignalType == "exit")
                    {
                        string reason = !string.IsNullOrEmpty(sig.ExitReason) ? sig.ExitReason : "signal_bus_exit";
                        _onSignalReceived?.Invoke();
                        Log($"SignalBus EXIT: reason={reason} asset={sig.Asset}");
                        if (currentState == State.Historical) ExecuteFlattenDirect($"{sig.Strategy}:{reason}");
                        else FlattenAll($"{sig.Strategy}:{reason}");
                    }
                    else
                    {
                        Log($"SignalBus ENTRY: {sig.Direction?.ToUpper()} {sig.Asset} Q={sig.SignalQuality:P0}");
                        if (currentState == State.Historical) ExecuteEntryDirect(sig);
                        else ProcessSignal(sig.ToJson());
                    }
                }
                catch (Exception ex) { Log($"SignalBus error: {ex.Message}"); }
            }
        }

        internal void FlushOrderQueue(State currentState)
        {
            if (currentState != State.Realtime && currentState != State.Historical) return;
            lock (_queueLock)
            {
                while (_orderQueue.Count > 0)
                {
                    var action = _orderQueue.Dequeue();
                    try { action(); } catch (Exception ex) { Log($"Queue error: {ex.Message}"); }
                }
            }
        }

        internal Dictionary<string, object> ProcessSignal(string json)
        {
            var response = new Dictionary<string, object>();
            _onSignalReceived?.Invoke();
            try
            {
                var serializer = new JavaScriptSerializer();
                var signal = serializer.Deserialize<Dictionary<string, object>>(json);

                string signalId = GetStr(signal, "signal_id", NewId());
                string dir = GetStr(signal, "direction", "long").ToLower();
                int requestedQty = GetInt(signal, "quantity", 1);
                string typeStr = GetStr(signal, "order_type", "market").ToLower();
                double limitPrice = GetDbl(signal, "limit_price", 0);
                double slPrice = GetDbl(signal, "stop_loss", 0);
                double tpPrice = GetDbl(signal, "take_profit", 0);
                double tp2Price = GetDbl(signal, "tp2", 0);
                string strategy = GetStr(signal, "strategy", "");
                string asset = GetStr(signal, "asset", "");

                int bip = ResolveBip(asset);
                double tickSize = GetTickSize(bip);
                double pointVal = GetPointValue(bip);

                if (_getRiskEnforcement() && _getRiskBlocked())
                {
                    string msg = $"Signal rejected — risk blocked: {_getRiskBlockReason()}";
                    _onSignalRejected?.Invoke();
                    response["status"] = "rejected"; response["reason"] = msg; response["signal_id"] = signalId;
                    return response;
                }

                OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
                OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.BuyToCover;
                OrderType ot = OrderType.Market;
                double stopPrice = 0;
                if (typeStr == "limit") { ot = OrderType.Limit; }
                else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

                double balance = CurrentBalance();
                double riskDollars = balance * (_getRiskPercent() / 100.0);
                double entry = GetClose(bip);
                double slDist = slPrice > 0 && entry > 0 ? Math.Abs(entry - slPrice) : _getDefaultSlTicks() * tickSize;
                double riskPerContract = slDist * pointVal;
                int riskQty = riskPerContract > 0 ? (int)Math.Floor(riskDollars / riskPerContract) : 1;
                int finalQty = Math.Max(1, Math.Min(requestedQty, Math.Min(riskQty, _getMaxContracts())));

                Log($"Signal {signalId}: {dir.ToUpper()} {asset} BIP{bip} x{finalQty}");

                int cBip = bip; double cTickSize = tickSize; int cQty = finalQty;
                string cDir = dir; double cSl = slPrice; double cTp = tpPrice; double cTp2 = tp2Price;
                double cLimit = limitPrice; double cStop = stopPrice;
                string cId = signalId; string cStrategy = strategy; string cAsset = asset;

                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;
                        string entryName = $"Signal-{cDir}-{cId}";
                        if (entryName.Length > 49) entryName = entryName.Substring(0, 49);
                        try
                        {
                            _strategy.SubmitOrderUnmanaged(cBip, action, ot, cQty, cLimit, cStop, "", entryName);
                            if (_getAutoBrackets())
                            {
                                double be = GetClose(cBip);
                                if (be <= 0) return;
                                double sl = cSl > 0 ? cSl : cDir == "long" ? be - _getDefaultSlTicks() * cTickSize : be + _getDefaultSlTicks() * cTickSize;
                                double tp = cTp > 0 ? cTp : cDir == "long" ? be + _getDefaultTpTicks() * cTickSize : be - _getDefaultTpTicks() * cTickSize;

                                // ── Validate SL is on correct side of market ──────────
                                // BuyToCover stop (short exit) must be ABOVE market.
                                // Sell stop (long exit) must be BELOW market.
                                if (cDir == "short" && sl <= be)
                                {
                                    double corrected = be + cTickSize;
                                    Log($"⚠ SL CORRECTED (short): {sl:F6} was at/below market {be:F6}, moved to {corrected:F6}");
                                    sl = corrected;
                                }
                                else if (cDir == "long" && sl >= be)
                                {
                                    double corrected = be - cTickSize;
                                    Log($"⚠ SL CORRECTED (long): {sl:F6} was at/above market {be:F6}, moved to {corrected:F6}");
                                    sl = corrected;
                                }
                                int tp1Qty = cTp2 > 0 ? Math.Max(1, cQty / 2) : cQty;
                                int tp2Qty = cTp2 > 0 ? cQty - tp1Qty : 0;
                                string oco = $"OCO-{cId}-{Guid.NewGuid().ToString("N").Substring(0, 6)}";
                                _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.StopMarket, cQty, 0, sl, oco, $"SL-{cId}");
                                _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.Limit, tp1Qty, tp, 0, oco, $"TP1-{cId}");
                                if (cTp2 > 0 && tp2Qty > 0)
                                    _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.Limit, tp2Qty, cTp2, 0, "", $"TP2-{cId}");
                            }
                            _sendPositionUpdate?.Invoke();
                            _onSignalExecuted?.Invoke();
                            Log($"✅ Executed {cDir.ToUpper()} {cAsset} BIP{cBip} x{cQty} id={cId}");
                        }
                        catch (Exception submitEx) { Log($"⚠ Order submission failed for {cAsset}: {submitEx.Message}"); }
                    });
                }

                response["status"] = "queued"; response["signal_id"] = signalId;
                response["direction"] = dir; response["quantity"] = finalQty;
                response["asset"] = asset; response["bip"] = bip;
            }
            catch (Exception ex) { Log($"ProcessSignal error: {ex.Message}"); response["status"] = "error"; response["error"] = ex.Message; }
            return response;
        }

        internal void ExecuteEntryDirect(SignalBus.Signal sig)
        {
            _onSignalReceived?.Invoke();
            string dir = (sig.Direction ?? "long").ToLower();
            int requestedQty = sig.Quantity > 0 ? sig.Quantity : 1;
            string typeStr = (sig.OrderType ?? "market").ToLower();
            double limitPrice = sig.LimitPrice;
            double slPrice = sig.StopLoss;
            double tpPrice = sig.TakeProfit;
            double tp2Price = sig.TakeProfit2;
            string signalId = sig.SignalId ?? NewId();

            int bip = ResolveBip(sig.Asset);
            double tickSize = GetTickSize(bip);
            double pointVal = GetPointValue(bip);

            OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
            OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderType ot = OrderType.Market;
            double stopPrice = 0;
            if (typeStr == "limit") { ot = OrderType.Limit; }
            else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

            double balance = CurrentBalance();
            double riskDollars = balance * (_getRiskPercent() / 100.0);
            double entry = GetClose(bip);
            double slDist = slPrice > 0 && entry > 0 ? Math.Abs(entry - slPrice) : _getDefaultSlTicks() * tickSize;
            double riskPerContract = slDist * pointVal;
            int riskQty = riskPerContract > 0 ? (int)Math.Floor(riskDollars / riskPerContract) : 1;
            int finalQty = Math.Max(1, Math.Min(requestedQty, Math.Min(riskQty, _getMaxContracts())));

            _onSignalExecuted?.Invoke();
            string sigName = $"Signal-{dir}-{signalId}";
            if (sigName.Length > 49) sigName = sigName.Substring(0, 49);
            _strategy.SubmitOrderUnmanaged(bip, action, ot, finalQty, limitPrice, stopPrice, "", sigName);

            if (!_getAutoBrackets() || entry <= 0) return;
            double sl = slPrice > 0 ? slPrice : dir == "long" ? entry - _getDefaultSlTicks() * tickSize : entry + _getDefaultSlTicks() * tickSize;
            double tp = tpPrice > 0 ? tpPrice : dir == "long" ? entry + _getDefaultTpTicks() * tickSize : entry - _getDefaultTpTicks() * tickSize;

            // ── Validate SL is on correct side of market ──────────────────
            if (dir == "short" && sl <= entry)
            {
                double corrected = entry + tickSize;
                Log($"⚠ SL CORRECTED (short): {sl:F6} at/below market {entry:F6}, moved to {corrected:F6}");
                sl = corrected;
            }
            else if (dir == "long" && sl >= entry)
            {
                double corrected = entry - tickSize;
                Log($"⚠ SL CORRECTED (long): {sl:F6} at/above market {entry:F6}, moved to {corrected:F6}");
                sl = corrected;
            }
            int tp1Qty = tp2Price > 0 ? Math.Max(1, finalQty / 2) : finalQty;
            int tp2Qty = tp2Price > 0 ? finalQty - tp1Qty : 0;
            string oco2 = $"OCO-{signalId}-{Guid.NewGuid().ToString("N").Substring(0, 6)}";
            _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.StopMarket, finalQty, 0, sl, oco2, $"SL-{signalId}");
            _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.Limit, tp1Qty, tp, 0, oco2, $"TP1-{signalId}");
            if (tp2Price > 0 && tp2Qty > 0)
                _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.Limit, tp2Qty, tp2Price, 0, "", $"TP2-{signalId}");
        }

        internal void ExecuteFlattenDirect(string reason)
        {
            try
            {
                var pos = _strategy.Position;
                if (pos != null && pos.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction ca = pos.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
                    _strategy.SubmitOrderUnmanaged(0, ca, OrderType.Market, pos.Quantity, 0, 0, "", $"Flatten-{reason}");
                }
                else
                {
                    var acct = _getMyAccount();
                    if (acct?.Positions != null)
                        foreach (NinjaTrader.Cbi.Position p in acct.Positions)
                        {
                            if (p == null || p.Quantity == 0 || p.Instrument == null) continue;
                            OrderAction ca = p.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
                            int posBip = ResolveBip(p.Instrument.MasterInstrument.Name);
                            _strategy.SubmitOrderUnmanaged(posBip, ca, OrderType.Market, p.Quantity, 0, 0, "", $"Flatten-{p.Instrument.FullName}");
                        }
                }
                _onExitExecuted?.Invoke();
                Log($"🔴 FLATTEN ALL (direct) — reason: {reason}");
            }
            catch (Exception ex) { Log($"ExecuteFlattenDirect error: {ex.Message}"); }
        }

        internal Dictionary<string, object> FlattenAll(string reason)
        {
            var response = new Dictionary<string, object>();
            _onExitExecuted?.Invoke();
            try
            {
                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;
                        try
                        {
                            var acct = _getMyAccount();
                            if (acct?.Positions != null)
                                foreach (NinjaTrader.Cbi.Position p in acct.Positions)
                                {
                                    if (p == null || p.Quantity == 0 || p.Instrument == null) continue;
                                    OrderAction ca = p.MarketPosition == MarketPosition.Long ? OrderAction.Sell : OrderAction.BuyToCover;
                                    int posBip = ResolveBip(p.Instrument.MasterInstrument.Name);
                                    _strategy.SubmitOrderUnmanaged(posBip, ca, OrderType.Market, p.Quantity, 0, 0, "", $"Flatten-{p.Instrument.FullName}");
                                }
                            if (acct?.Orders != null)
                                foreach (NinjaTrader.Cbi.Order ord in acct.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working || ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                        acct.Cancel(new[] { ord });
                                }
                        }
                        catch (Exception ex) { Log($"FlattenAll queue error: {ex.Message}"); }
                        _sendPositionUpdate?.Invoke();
                        Log($"🔴 FLATTEN ALL — reason: {reason}");
                    });
                }
                response["status"] = "flatten_queued"; response["reason"] = reason;
            }
            catch (Exception ex) { response["status"] = "error"; response["error"] = ex.Message; }
            return response;
        }

        internal Dictionary<string, object> CancelAllOrders()
        {
            var response = new Dictionary<string, object>();
            try
            {
                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;
                        try
                        {
                            var acct = _getMyAccount();
                            if (acct?.Orders != null)
                                foreach (NinjaTrader.Cbi.Order ord in acct.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working || ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                        acct.Cancel(new[] { ord });
                                }
                        }
                        catch (Exception ex) { Log($"CancelAllOrders error: {ex.Message}"); }
                    });
                }
                response["status"] = "cancel_queued";
            }
            catch (Exception ex) { response["status"] = "error"; response["error"] = ex.Message; }
            return response;
        }

        internal int ResolveBip(string asset)
        {
            if (string.IsNullOrEmpty(asset)) return 0;
            string upper = asset.Trim().ToUpperInvariant();
            if (_symbolToBip.TryGetValue(upper, out int bip)) return bip;
            int space = upper.IndexOf(' ');
            if (space > 0 && _symbolToBip.TryGetValue(upper.Substring(0, space), out bip)) return bip;
            Log($"ResolveBip: no mapping for '{asset}', routing to BIP 0");
            return 0;
        }

        internal double GetTickSize(int bip)
        {
            try { if (bip < _strategy.BarsArray.Length) return _strategy.BarsArray[bip].Instrument.MasterInstrument.TickSize; } catch { }
            return _strategy.TickSize;
        }

        internal double GetPointValue(int bip)
        {
            try { if (bip < _strategy.BarsArray.Length) { double pv = _strategy.BarsArray[bip].Instrument.MasterInstrument.PointValue; if (pv > 0) return pv; } } catch { }
            return 10;
        }

        internal double GetClose(int bip)
        {
            try { if (bip < _strategy.BarsArray.Length && _strategy.BarsArray[bip].Count > 0) return _strategy.BarsArray[bip].GetClose(_strategy.BarsArray[bip].Count - 1); } catch { }
            try { if (_strategy.CurrentBar >= 0 && _strategy.Close?.Count > 0) return _strategy.Close[0]; } catch { }
            return 0;
        }

        private double CurrentBalance()
        {
            try { var acct = _getMyAccount(); if (acct != null) return acct.Get(NinjaTrader.Cbi.AccountItem.CashValue, NinjaTrader.Cbi.Currency.UsDollar); } catch { }
            return _getAccountSize();
        }

        private void Log(string message) => _strategy.Print($"[{_tag}] {message}");
        private static string NewId() => Guid.NewGuid().ToString("N").Substring(0, 8);

        private static string GetStr(Dictionary<string, object> d, string key, string def)
        { if (d.ContainsKey(key) && d[key] != null) return d[key].ToString(); return def; }
        private static int GetInt(Dictionary<string, object> d, string key, int def)
        { if (d.ContainsKey(key) && d[key] != null) try { return (int)Math.Round(Convert.ToDouble(d[key])); } catch { } return def; }
        private static double GetDbl(Dictionary<string, object> d, string key, double def)
        { if (d.ContainsKey(key) && d[key] != null) try { return Convert.ToDouble(d[key]); } catch { } return def; }
    }
}

// =============================================================================
// OrbCnnPredictor — inlined from OrbCnnPredictor.cs
// =============================================================================

namespace NinjaTrader.NinjaScript
{
    public class CnnPrediction
    {
        public float Probability { get; }
        public bool Signal { get; }
        public string Confidence { get; }
        public float Threshold { get; }

        public CnnPrediction(float prob, float threshold)
        {
            Probability = prob;
            Threshold = threshold;
            Signal = prob >= threshold;
            Confidence = prob >= 0.90f ? "HIGH"
                        : prob >= 0.75f ? "MEDIUM"
                        : prob >= 0.60f ? "LOW"
                        : "VERY LOW";
        }
        public override string ToString()
            => $"P={Probability:P1} signal={Signal} conf={Confidence} thresh={Threshold:P1}";
    }

    public static class CnnSessionThresholds
    {
        // ── Per-session thresholds ────────────────────────────────────────────
        // Must match feature_contract.json session_thresholds exactly.
        // Keys are the session names used by DetectSessionKey().
        private static readonly Dictionary<string, float> _thresholds = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
        {
            { "cme",        0.75f },
            { "sydney",     0.72f },
            { "tokyo",      0.74f },
            { "shanghai",   0.74f },
            { "frankfurt",  0.80f },
            { "london",     0.82f },
            { "london_ny",  0.82f },
            { "us",         0.82f },
            { "cme_settle", 0.78f },
            // Legacy keys for backward compatibility
            { "asia",       0.74f },
            { "overlap",    0.82f },
            { "futures",    0.80f },
            { "crypto",     0.74f },
            { "default",    0.82f },
        };
        public static readonly float Default = 0.82f;

        public static float GetSessionThreshold(string sessionKey)
        {
            if (string.IsNullOrEmpty(sessionKey)) return Default;
            return _thresholds.TryGetValue(sessionKey, out float t) ? t : Default;
        }

        // ── Per-session ordinals ──────────────────────────────────────────────
        // Must match feature_contract.json session_ordinals exactly.
        // Values encode position in the 24h Globex day cycle [0, 1].
        private static readonly Dictionary<string, float> _ordinals = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
        {
            { "cme",        0.000f },
            { "sydney",     0.125f },
            { "tokyo",      0.250f },
            { "shanghai",   0.375f },
            { "frankfurt",  0.500f },
            { "london",     0.625f },
            { "london_ny",  0.750f },
            { "us",         0.875f },
            { "cme_settle", 1.000f },
            // Legacy keys
            { "asia",       0.250f },
            { "overlap",    0.750f },
            { "futures",    0.500f },
            { "crypto",     0.375f },
            { "default",    0.875f },
        };

        public static float GetSessionOrdinal(string sessionKey)
        {
            if (string.IsNullOrEmpty(sessionKey)) return 0.875f; // default to US
            return _ordinals.TryGetValue(sessionKey, out float o) ? o : 0.875f;
        }
    }

    public sealed class OrbCnnPredictor : IDisposable
    {
        private InferenceSession _session;
        private readonly string _imageName;
        private readonly string _tabularName;
        private readonly string _outputName;
        private readonly string _modelPath;
        private bool _disposed;

        private readonly float[] _mean = { 0.485f, 0.456f, 0.406f };
        private readonly float[] _std = { 0.229f, 0.224f, 0.225f };

        private const int ImageSize = 224;
        // feature_contract.json: tabular feature count.
        // Default 18 (v6 contract), but auto-detected from model at load time.
        private int _numTabular = 28;
        private const int MaxTabular = 28; // C# always builds 28 features (v7.1 contract)
        private const int NumChannels = 3;
        private static readonly int ImageBufSize = NumChannels * ImageSize * ImageSize;
        public int NumTabular => _numTabular;

        public string ModelPath => _modelPath;

        public OrbCnnPredictor(string modelPath, bool useCuda = false)
        {
            if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentNullException(nameof(modelPath));
            if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}");
            _modelPath = modelPath;

            var opts = new SessionOptions();
            if (useCuda)
            {
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch { /* fall through to CPU */ }
            }
            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            opts.InterOpNumThreads = 1;
            opts.IntraOpNumThreads = 2;

            _session = new InferenceSession(modelPath, opts);

            // Resolve input/output names from the model's metadata
            var inputNames = new List<string>(_session.InputMetadata.Keys);
            var outputNames = new List<string>(_session.OutputMetadata.Keys);

            _imageName = inputNames.Count > 0 ? inputNames[0] : "image";
            _tabularName = inputNames.Count > 1 ? inputNames[1] : "tabular";

            // Auto-detect expected tabular dimension from model metadata
            if (_session.InputMetadata.ContainsKey(_tabularName))
            {
                var tabMeta = _session.InputMetadata[_tabularName];
                var dims = tabMeta.Dimensions;
                if (dims != null && dims.Length >= 2 && dims[1] > 0)
                {
                    _numTabular = dims[1];
                }
            }
            _outputName = outputNames.Count > 0 ? outputNames[0] : "output";
        }

        public CnnPrediction Predict(string imagePath, float[] tabular, float threshold)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(OrbCnnPredictor));
            if (tabular == null)
                throw new ArgumentException("tabular array must not be null");

            // Auto-adapt tabular vector to model's expected dimension.
            // C# always builds MaxTabular (14) features; if the model
            // expects fewer (e.g. 8 for v3), truncate.  If it somehow
            // expects more, zero-pad.
            if (tabular.Length != _numTabular)
            {
                float[] adapted = new float[_numTabular];
                int copyLen = Math.Min(tabular.Length, _numTabular);
                Array.Copy(tabular, adapted, copyLen);
                tabular = adapted;
            }

            float[] imgBuf = new float[ImageBufSize];
            LoadImageToBuffer(imagePath, imgBuf);
            float[] tabNorm = NormaliseTabular(tabular);

            var imgTensor = new DenseTensor<float>(imgBuf, new[] { 1, NumChannels, ImageSize, ImageSize });
            var tabTensor = new DenseTensor<float>(tabNorm, new[] { 1, _numTabular });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_imageName,   imgTensor),
                NamedOnnxValue.CreateFromTensor(_tabularName, tabTensor),
            };

            using (var results = _session.Run(inputs))
            {
                var output = results[0].AsEnumerable<float>().ToArray();
                float prob = output.Length >= 2 ? Softmax2(output[0], output[1]) : output[0];
                return new CnnPrediction(prob, threshold);
            }
        }

        /// <summary>
        /// Normalise 28 raw tabular features per feature_contract.json v7.1.
        /// Must produce identical output to _normalise_tabular_for_inference() in breakout_cnn.py.
        ///
        /// Index  Feature                  Normalisation
        /// ─────  ───────────────────────  ──────────────────────────────────────
        ///  [0]   quality_pct_norm         clamp [0, 1]
        ///  [1]   volume_ratio             min(log1p(raw) / log1p(10), 1.0)
        ///  [2]   atr_pct                  min(raw × 100, 1.0)
        ///  [3]   cvd_delta                clamp [-1, 1]
        ///  [4]   nr7_flag                 passthrough (0 or 1)
        ///  [5]   direction_flag           passthrough (0 or 1)
        ///  [6]   session_ordinal          passthrough [0, 1]
        ///  [7]   london_overlap_flag      passthrough (0 or 1)
        ///  [8]   or_range_atr_ratio       clamp(raw, 0, 3) / 3.0
        ///  [9]   premarket_range_ratio    clamp(raw, 0, 5) / 5.0
        ///  [10]  bar_of_day              passthrough [0, 1] (already normalised)
        ///  [11]  day_of_week             passthrough [0, 1] (already normalised)
        ///  [12]  vwap_distance           clamp(raw, -3, 3) / 3.0
        ///  [13]  asset_class_id          passthrough [0, 1] (already normalised)
        ///  [14]  breakout_type_ord       passthrough [0, 1] (already / 12)
        ///  [15]  asset_volatility_class  passthrough (0.0 / 0.5 / 1.0)
        ///  [16]  hour_of_day             passthrough [0, 1] (already / 23)
        ///  [17]  tp3_atr_mult_norm       passthrough [0, 1] (already / 5.0)
        ///  [18]  daily_bias_direction    passthrough [0, 1] (already normalised)
        ///  [19]  daily_bias_confidence   passthrough [0, 1]
        ///  [20]  prior_day_pattern       passthrough [0, 1] (already / 9)
        ///  [21]  weekly_range_position   passthrough [0, 1]
        ///  [22]  monthly_trend_score     passthrough [0, 1]
        ///  [23]  crypto_momentum_score   passthrough [0, 1]
        ///  [24]  breakout_type_category  passthrough {0.0, 0.5, 1.0}
        ///  [25]  session_overlap_flag    passthrough {0.0, 1.0}
        ///  [26]  atr_trend               passthrough [0, 1]
        ///  [27]  volume_trend            passthrough [0, 1]
        /// </summary>
        private float[] NormaliseTabular(float[] raw)
        {
            float[] norm = new float[_numTabular];

            // [0] quality_pct_norm — already 0..1, clamp
            norm[0] = Math.Max(0f, Math.Min(1f, raw[0]));

            // [1] volume_ratio — log-scale, match Python: min(log1p(raw) / log1p(10), 1.0)
            float volRaw = Math.Max(0.01f, raw[1]);
            norm[1] = Math.Min(1f, (float)(Math.Log(volRaw + 1.0) / Math.Log(11.0)));

            // [2] atr_pct — ×100, clamp to [0, 1]
            norm[2] = Math.Max(0f, Math.Min(1f, raw[2] * 100f));

            // [3] cvd_delta — clamp [-1, 1]
            norm[3] = Math.Max(-1f, Math.Min(1f, raw[3]));

            // [4] nr7_flag — passthrough
            norm[4] = raw[4];

            // [5] direction_flag — passthrough
            norm[5] = raw[5];

            // [6] session_ordinal — passthrough [0, 1]
            norm[6] = raw[6];

            // [7] london_overlap_flag — passthrough
            norm[7] = raw[7];

            // [8] onwards — only if model expects >= 9 features
            if (_numTabular > 8)
            {
                // [8] or_range_atr_ratio — clamp(raw, 0, 3) / 3.0
                norm[8] = Math.Max(0f, Math.Min(3f, raw[8])) / 3f;

                // [9] premarket_range_ratio — clamp(raw, 0, 5) / 5.0
                norm[9] = Math.Max(0f, Math.Min(5f, raw[9])) / 5f;

                // [10] bar_of_day — already normalised [0, 1]
                norm[10] = Math.Max(0f, Math.Min(1f, raw[10]));

                // [11] day_of_week — already normalised [0, 1]
                norm[11] = Math.Max(0f, Math.Min(1f, raw[11]));

                // [12] vwap_distance — clamp(raw, -3, 3) / 3.0 → [-1, 1]
                norm[12] = Math.Max(-3f, Math.Min(3f, raw[12])) / 3f;

                // [13] asset_class_id — already normalised [0, 1]
                if (_numTabular > 13) norm[13] = Math.Max(0f, Math.Min(1f, raw[13]));
            } // end guard for features [8]+

            // ── v6 features [14]–[17] — all pre-normalised in PrepareCnnTabular ──
            if (_numTabular > 14 && raw.Length > 14)
            {
                // [14] breakout_type_ord — already / 12.0, clamp [0, 1]
                norm[14] = Math.Max(0f, Math.Min(1f, raw[14]));

                // [15] asset_volatility_class — already 0.0 / 0.5 / 1.0
                if (_numTabular > 15 && raw.Length > 15)
                    norm[15] = Math.Max(0f, Math.Min(1f, raw[15]));

                // [16] hour_of_day — already / 23.0, clamp [0, 1]
                if (_numTabular > 16 && raw.Length > 16)
                    norm[16] = Math.Max(0f, Math.Min(1f, raw[16]));

                // [17] tp3_atr_mult_norm — already / 5.0, clamp [0, 1]
                if (_numTabular > 17 && raw.Length > 17)
                    norm[17] = Math.Max(0f, Math.Min(1f, raw[17]));
            } // end guard for features [14]+

            // ── v7 features [18]–[23] — Daily Strategy layer ─────────────────
            // All pre-normalised to [0, 1] in PrepareCnnTabular — passthrough with clamp.
            if (_numTabular > 18 && raw.Length > 18)
            {
                // [18] daily_bias_direction — already {0.0, 0.5, 1.0}
                norm[18] = Math.Max(0f, Math.Min(1f, raw[18]));

                // [19] daily_bias_confidence — already [0, 1]
                if (_numTabular > 19 && raw.Length > 19)
                    norm[19] = Math.Max(0f, Math.Min(1f, raw[19]));

                // [20] prior_day_pattern — already / 9, [0, 1]
                if (_numTabular > 20 && raw.Length > 20)
                    norm[20] = Math.Max(0f, Math.Min(1f, raw[20]));

                // [21] weekly_range_position — already [0, 1]
                if (_numTabular > 21 && raw.Length > 21)
                    norm[21] = Math.Max(0f, Math.Min(1f, raw[21]));

                // [22] monthly_trend_score — already [0, 1]
                if (_numTabular > 22 && raw.Length > 22)
                    norm[22] = Math.Max(0f, Math.Min(1f, raw[22]));

                // [23] crypto_momentum_score — already [0, 1]
                if (_numTabular > 23 && raw.Length > 23)
                    norm[23] = Math.Max(0f, Math.Min(1f, raw[23]));
            } // end guard for features [18]+

            // ── v7.1 features [24]–[27] — Phase 4B sub-feature decomposition ──
            // All pre-normalised to [0, 1] in PrepareCnnTabular — passthrough with clamp.
            if (_numTabular > 24 && raw.Length > 24)
            {
                // [24] breakout_type_category — already {0.0, 0.5, 1.0}
                norm[24] = Math.Max(0f, Math.Min(1f, raw[24]));

                // [25] session_overlap_flag — already {0.0, 1.0}
                if (_numTabular > 25 && raw.Length > 25)
                    norm[25] = Math.Max(0f, Math.Min(1f, raw[25]));

                // [26] atr_trend — already [0, 1]
                if (_numTabular > 26 && raw.Length > 26)
                    norm[26] = Math.Max(0f, Math.Min(1f, raw[26]));

                // [27] volume_trend — already [0, 1]
                if (_numTabular > 27 && raw.Length > 27)
                    norm[27] = Math.Max(0f, Math.Min(1f, raw[27]));
            } // end guard for features [24]+

            return norm;
        }

        private void LoadImageToBuffer(string path, float[] buf)
        {
            using (var bmp = new Bitmap(path))
            using (var resized = new Bitmap(bmp, new Size(ImageSize, ImageSize)))
            {
                var rect = new Rectangle(0, 0, ImageSize, ImageSize);
                var data = resized.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    int stride = data.Stride;
                    byte[] pixels = new byte[stride * ImageSize];
                    Marshal.Copy(data.Scan0, pixels, 0, pixels.Length);

                    for (int y = 0; y < ImageSize; y++)
                        for (int x = 0; x < ImageSize; x++)
                        {
                            int src = y * stride + x * 3;
                            float b = pixels[src + 0] / 255f;
                            float g = pixels[src + 1] / 255f;
                            float r = pixels[src + 2] / 255f;

                            int baseIdx = y * ImageSize + x;
                            buf[baseIdx] = (r - _mean[0]) / _std[0];
                            buf[ImageSize * ImageSize + baseIdx] = (g - _mean[1]) / _std[1];
                            buf[2 * ImageSize * ImageSize + baseIdx] = (b - _mean[2]) / _std[2];
                        }
                }
                finally { resized.UnlockBits(data); }
            }
        }

        private static float Softmax2(float a, float b)
        {
            float maxVal = Math.Max(a, b);
            float ea = (float)Math.Exp(a - maxVal);
            float eb = (float)Math.Exp(b - maxVal);
            return eb / (ea + eb);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _session?.Dispose();
            _session = null;
        }
    }

    // =========================================================================
    // OrbChartRenderer — renders a 224×224 candlestick PNG for CNN input
    // =========================================================================
    // Supports 13 distinct range-box styles, one per BreakoutType:
    //
    //   ORB              → gold semi-transparent fill + gold border (solid)
    //   PrevDay          → cyan border, no fill (dashed)
    //   InitialBalance   → blue border, light-blue fill (solid)
    //   Consolidation    → purple border, no fill (dashed)
    //   Weekly           → teal border, teal fill alpha 30 (solid)
    //   Monthly          → orange border, orange fill alpha 30 (solid)
    //   Asian            → red border, no fill (dashed)
    //   BollingerSqueeze → magenta border, no fill (dashed)
    //   ValueArea        → olive border, olive fill alpha 30 (solid)
    //   InsideDay        → lime border, no fill (dashed)
    //   GapRejection     → coral border, no fill (solid)
    //   PivotPoints      → steel-blue border, no fill (dashed)
    //   Fibonacci        → amber border, amber fill alpha 20 (solid)
    // =========================================================================

    public static class OrbChartRenderer
    {
        private const int W = 224;
        private const int H = 224;
        private const int VolPanelH = 40;
        private const int PriceH = H - VolPanelH - 4;
        private const int PriceTop = 4;
        private const int LeftPad = 4;
        private const int RightPad = 4;

        private static readonly Color BgColor = Color.FromArgb(0x0D, 0x0D, 0x0D);
        private static readonly Color BullCandle = Color.FromArgb(0x26, 0xA6, 0x9A);
        private static readonly Color BearCandle = Color.FromArgb(0xEF, 0x53, 0x50);
        private static readonly Color VwapLine = Color.FromArgb(0x00, 0xE5, 0xFF);
        private static readonly Color VolBull = Color.FromArgb(100, 0x26, 0xA6, 0x9A);
        private static readonly Color VolBear = Color.FromArgb(100, 0xEF, 0x53, 0x50);

        /// <summary>Describes how to paint one range box on the chart image.</summary>
        private sealed class BoxStyle
        {
            /// <summary>Semi-transparent fill color.  Alpha=0 means no fill is drawn.</summary>
            public Color Fill { get; }
            /// <summary>Opaque border/line color.</summary>
            public Color Border { get; }
            /// <summary>true = solid border lines, false = dashed (4px on / 4px off).</summary>
            public bool Solid { get; }

            public BoxStyle(Color fill, Color border, bool solid)
            { Fill = fill; Border = border; Solid = solid; }
        }

        /// <summary>
        /// Returns the <see cref="BoxStyle"/> for a given <see cref="BreakoutType"/>.
        /// Called once per Render() invocation; not perf-critical.
        /// </summary>
        private static BoxStyle GetBoxStyle(BreakoutType type)
        {
            switch (type)
            {
                // ORB — gold semi-transparent fill + gold solid border
                case BreakoutType.ORB:
                    return new BoxStyle(
                        Color.FromArgb(40, 0xFF, 0xD7, 0x00),
                        Color.FromArgb(100, 0xFF, 0xD7, 0x00),
                        solid: true);

                // PrevDay — cyan dashed border, no fill
                case BreakoutType.PrevDay:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0x00, 0xFF, 0xFF),
                        solid: false);

                // InitialBalance — blue solid border, light-blue fill
                case BreakoutType.InitialBalance:
                    return new BoxStyle(
                        Color.FromArgb(40, 0x29, 0xB6, 0xF6),
                        Color.FromArgb(180, 0x01, 0x88, 0xFF),
                        solid: true);

                // Consolidation — purple dashed border, no fill
                case BreakoutType.Consolidation:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0xAB, 0x47, 0xBC),
                        solid: false);

                // Weekly — teal solid border + teal fill alpha 30
                case BreakoutType.Weekly:
                    return new BoxStyle(
                        Color.FromArgb(30, 0x00, 0x96, 0x88),
                        Color.FromArgb(180, 0x00, 0x96, 0x88),
                        solid: true);

                // Monthly — orange solid border + orange fill alpha 30
                case BreakoutType.Monthly:
                    return new BoxStyle(
                        Color.FromArgb(30, 0xFF, 0x98, 0x00),
                        Color.FromArgb(180, 0xFF, 0x98, 0x00),
                        solid: true);

                // Asian — red dashed border, no fill
                case BreakoutType.Asian:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0xEF, 0x53, 0x50),
                        solid: false);

                // BollingerSqueeze — magenta dashed border, no fill
                case BreakoutType.BollingerSqueeze:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0xE0, 0x40, 0xFB),
                        solid: false);

                // ValueArea — olive solid border + olive fill alpha 30
                case BreakoutType.ValueArea:
                    return new BoxStyle(
                        Color.FromArgb(30, 0x9E, 0x9D, 0x24),
                        Color.FromArgb(180, 0x9E, 0x9D, 0x24),
                        solid: true);

                // InsideDay — lime dashed border, no fill
                case BreakoutType.InsideDay:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0x76, 0xFF, 0x03),
                        solid: false);

                // GapRejection — coral solid border, no fill
                case BreakoutType.GapRejection:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0xFF, 0x7F, 0x50),
                        solid: true);

                // PivotPoints — steel-blue dashed border, no fill
                case BreakoutType.PivotPoints:
                    return new BoxStyle(
                        Color.FromArgb(0, 0, 0, 0),
                        Color.FromArgb(180, 0x46, 0x82, 0xB4),
                        solid: false);

                // Fibonacci — amber solid border + amber fill alpha 20
                case BreakoutType.Fibonacci:
                    return new BoxStyle(
                        Color.FromArgb(20, 0xFF, 0xBF, 0x00),
                        Color.FromArgb(180, 0xFF, 0xBF, 0x00),
                        solid: true);

                // Fallback: gold ORB style
                default:
                    return new BoxStyle(
                        Color.FromArgb(40, 0xFF, 0xD7, 0x00),
                        Color.FromArgb(100, 0xFF, 0xD7, 0x00),
                        solid: true);
            }
        }

        public class Bar
        {
            public DateTime Time { get; set; }
            public double Open { get; set; }
            public double High { get; set; }
            public double Low { get; set; }
            public double Close { get; set; }
            public double Volume { get; set; }
            public Bar() { }
            public Bar(DateTime t, double o, double h, double l, double c, double v)
            { Time = t; Open = o; High = h; Low = l; Close = c; Volume = v; }
        }

        /// <summary>
        /// Render a 224×224 candlestick chart PNG.
        /// The range box (rangeHigh / rangeLow) is styled according to
        /// <paramref name="breakoutType"/> so each of the 13 types has a
        /// visually distinct appearance that the CNN can distinguish.
        /// </summary>
        public static Bitmap Render(
            IList<Bar> bars, double rangeHigh, double rangeLow,
            double[] vwapValues, string direction,
            BreakoutType breakoutType = BreakoutType.ORB,
            string label = "")
        {
            if (bars == null || bars.Count == 0)
                throw new ArgumentException("bars cannot be null or empty");

            var style = GetBoxStyle(breakoutType);

            var bmp = new Bitmap(W, H, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.Clear(BgColor);

                double priceMin = bars.Min(b => b.Low) * 0.9995;
                double priceMax = bars.Max(b => b.High) * 1.0005;
                double pRange = priceMax - priceMin;
                if (pRange <= 0) pRange = 1;

                double volMax = bars.Max(b => b.Volume);
                if (volMax <= 0) volMax = 1;

                int usableW = W - LeftPad - RightPad;
                float barW = Math.Max(1f, (float)usableW / bars.Count);

                // ── Range box ─────────────────────────────────────────────────
                if (rangeHigh > rangeLow && rangeHigh > priceMin && rangeLow < priceMax)
                {
                    // Clamp to price panel
                    double clampedH = Math.Min(rangeHigh, priceMax);
                    double clampedL = Math.Max(rangeLow, priceMin);

                    int yBoxH = PriceTop + (int)((priceMax - clampedH) / pRange * PriceH);
                    int yBoxL = PriceTop + (int)((priceMax - clampedL) / pRange * PriceH);
                    int boxH = Math.Max(1, yBoxL - yBoxH);

                    // Fill (only when alpha > 0)
                    if (style.Fill.A > 0)
                    {
                        using (var fillBrush = new SolidBrush(style.Fill))
                            g.FillRectangle(fillBrush, LeftPad, yBoxH, usableW, boxH);
                    }

                    // Border lines (top and bottom of range box)
                    using (var borderPen = new Pen(style.Border, 1f))
                    {
                        if (!style.Solid)
                            borderPen.DashPattern = new float[] { 4f, 4f };

                        g.DrawLine(borderPen, LeftPad, yBoxH, LeftPad + usableW, yBoxH);
                        g.DrawLine(borderPen, LeftPad, yBoxL, LeftPad + usableW, yBoxL);
                    }
                }

                // ── Candles ───────────────────────────────────────────────────
                for (int i = 0; i < bars.Count; i++)
                {
                    var bar = bars[i];
                    bool bull = bar.Close >= bar.Open;
                    Color col = bull ? BullCandle : BearCandle;

                    float xCenter = LeftPad + i * barW + barW / 2f;
                    float xL = LeftPad + i * barW + 1;
                    float xR = xL + barW - 2;

                    int yHigh = PriceTop + (int)((priceMax - bar.High) / pRange * PriceH);
                    int yLow = PriceTop + (int)((priceMax - bar.Low) / pRange * PriceH);
                    int yOpen = PriceTop + (int)((priceMax - bar.Open) / pRange * PriceH);
                    int yClose = PriceTop + (int)((priceMax - bar.Close) / pRange * PriceH);

                    int bodyTop = Math.Min(yOpen, yClose);
                    int bodyH = Math.Max(1, Math.Abs(yClose - yOpen));

                    using (var pen = new Pen(col, 1))
                        g.DrawLine(pen, xCenter, yHigh, xCenter, yLow);
                    using (var brush = new SolidBrush(col))
                        g.FillRectangle(brush, xL, bodyTop, Math.Max(1, xR - xL), bodyH);

                    // Volume panel
                    int volH = (int)(bar.Volume / volMax * (VolPanelH - 2));
                    int volTop = H - VolPanelH + (VolPanelH - volH);
                    using (var volBrush = new SolidBrush(bull ? VolBull : VolBear))
                        g.FillRectangle(volBrush, xL, volTop, Math.Max(1, xR - xL), volH);
                }

                // ── VWAP line ─────────────────────────────────────────────────
                if (vwapValues != null && vwapValues.Length == bars.Count)
                {
                    using (var vwapPen = new Pen(VwapLine, 1))
                    {
                        for (int i = 1; i < bars.Count; i++)
                        {
                            float x1 = LeftPad + (i - 1) * barW + barW / 2f;
                            float x2 = LeftPad + i * barW + barW / 2f;
                            int y1 = PriceTop + (int)((priceMax - vwapValues[i - 1]) / pRange * PriceH);
                            int y2 = PriceTop + (int)((priceMax - vwapValues[i]) / pRange * PriceH);
                            g.DrawLine(vwapPen, x1, y1, x2, y2);
                        }
                    }
                }
            }

            return bmp;
        }

        /// <summary>
        /// Convenience wrapper: renders to a temp PNG file and returns the path.
        /// Passes <paramref name="breakoutType"/> through to <see cref="Render"/>
        /// so the correct box style is applied.
        /// </summary>
        public static string RenderToTemp(
            IList<Bar> bars, double rangeHigh, double rangeLow,
            double[] vwapValues, string direction,
            BreakoutType breakoutType = BreakoutType.ORB,
            string outputFolder = null, string label = "")
        {
            string folder = string.IsNullOrWhiteSpace(outputFolder)
                ? Path.GetTempPath()
                : outputFolder;
            Directory.CreateDirectory(folder);
            string path = Path.Combine(folder,
                $"rng_{(int)breakoutType}_{DateTime.UtcNow:yyyyMMdd_HHmmssfff}.png");
            using (var bmp = Render(bars, rangeHigh, rangeLow, vwapValues,
                                    direction, breakoutType, label))
                bmp.Save(path, ImageFormat.Png);
            return path;
        }
    }
}
