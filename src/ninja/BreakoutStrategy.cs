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
        ORB              = 0,   // Opening Range (was "Orb" in v4)
        PrevDay          = 1,   // Previous day high/low
        InitialBalance   = 2,   // First 60 min of RTH (institutional)
        Consolidation    = 3,   // ATR contraction / squeeze (non-time-based)
        Weekly           = 4,   // Weekly high/low range
        Monthly          = 5,   // Monthly high/low range
        Asian            = 6,   // Asian session range (19:00–01:00 ET)
        BollingerSqueeze = 7,   // Bollinger Band squeeze
        ValueArea        = 8,   // Value Area high/low (volume profile)
        InsideDay        = 9,   // Inside day pattern
        GapRejection     = 10,  // Gap fill rejection
        PivotPoints      = 11,  // Classic pivot point levels
        Fibonacci        = 12,  // Fibonacci retracement levels
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
    public sealed class PositionPhase
    {
        public string   SignalId    { get; set; }
        public string   Direction   { get; set; }   // "long" | "short"
        public string   Asset       { get; set; }
        public int      Bip         { get; set; }
        public BreakoutPhase Phase  { get; set; } = BreakoutPhase.Phase1;

        // Entry price and ATR captured at signal time
        public double EntryPrice    { get; set; }
        public double AtrAtEntry    { get; set; }

        // Phase1 targets (submitted as bracket orders)
        public double SlPrice       { get; set; }   // current SL (modified at TP1)
        public double Tp1Price      { get; set; }
        public double Tp2Price      { get; set; }
        public double Tp3Price      { get; set; }   // computed: entry ± ATR × Tp3Mult

        // Phase2/3 contract quantities
        public int    TotalQty      { get; set; }
        public int    Tp1Qty        { get; set; }
        public int    Tp2Qty        { get; set; }
        public int    Tp3Qty        { get; set; }   // = TotalQty - Tp1Qty - Tp2Qty (remainder)

        // OCO group for the current SL leg (updated at each phase transition)
        public string OcoGroup      { get; set; }

        // Phase3: EMA9-based trailing exit was triggered
        public bool   Tp3Submitted  { get; set; }
        public bool   Ema9StopHit   { get; set; }
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
                // TP3 multiplier (loaded from RangeConfig; used for Phase3 target)
                public double Tp3AtrMult = 5.0;
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
            public double Ema9Value  = 0;
            public double Ema9Sum    = 0;   // warmup: running sum of first 9 closes
            public int    Ema9Filled = 0;   // bars written into warmup (caps at 9)
            public bool   Ema9Ready  = false;

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
                Ranges[BreakoutType.ORB]              = new RangeState();
                Ranges[BreakoutType.PrevDay]           = new RangeState();
                Ranges[BreakoutType.InitialBalance]    = new RangeState();
                Ranges[BreakoutType.Consolidation]     = new RangeState();
                Ranges[BreakoutType.Weekly]            = new RangeState();
                Ranges[BreakoutType.Monthly]           = new RangeState();
                Ranges[BreakoutType.Asian]             = new RangeState();
                Ranges[BreakoutType.BollingerSqueeze]  = new RangeState();
                Ranges[BreakoutType.ValueArea]         = new RangeState();
                Ranges[BreakoutType.InsideDay]         = new RangeState();
                Ranges[BreakoutType.GapRejection]      = new RangeState();
                Ranges[BreakoutType.PivotPoints]       = new RangeState();
                Ranges[BreakoutType.Fibonacci]         = new RangeState();
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

        // ── Per-instrument state array (index = BIP) ──────────────────────────
        private InstrumentState[] _states;

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

        // ── Risk gate (mirrors Bridge.RiskBlocked so both respect it) ─────────
        // BreakoutStrategy has no HTTP listener of its own; it simply won't
        // allow entries when this is set.  Bridge can set it via SignalBus
        // or the caller can set it programmatically.
        internal volatile bool RiskBlocked = false;
        internal volatile string RiskBlockReason = "";

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
        private const bool   CEnableTp3Trailing = true; // enable 3-phase EMA9 trailing walk
        private const bool CEnableAutoBrackets = true;
        private const int CDefaultSlTicks = 20;
        private const int CDefaultTpTicks = 40;
        // CNN
        // Number of tabular features C# builds and passes to OrbCnnPredictor.
        // Must stay in sync with PrepareCnnTabular() and feature_contract.json v6.
        // If the loaded ONNX model reports a different NumTabular, a warning is
        // printed at startup — retrain/re-export the model to clear it.
        private const int CNumTabularFeatures = 18;
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
                            Print($"[BreakoutStrategy] AddDataSeries: {addName} → BIP {extras.Count}  session='{th}'");
                        }
                        catch (Exception ex)
                        {
                            Print($"[BreakoutStrategy] AddDataSeries failed for {root}: {ex.Message}");
                        }
                    }

                    _extraSymbols = extras.ToArray();
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
                for (int i = 0; i < count; i++)
                {
                    _states[i] = new InstrumentState(VolumeAvgPeriod, 14, CnnLookbackBars);
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
                              $"feature_contract.json v6 ({CNumTabularFeatures} features) " +
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
                                    int tp1Qty   = ph.Tp2Price > 0 ? Math.Max(1, totalQty / 2) : totalQty;
                                    int tp2Qty   = ph.Tp2Price > 0 ? totalQty - tp1Qty : 0;
                                    int tp3Qty   = ph.Tp3Price > 0 ? tp2Qty : 0;
                                    ph.TotalQty  = totalQty;
                                    ph.Tp1Qty    = tp1Qty;
                                    ph.Tp2Qty    = tp2Qty;
                                    ph.Tp3Qty    = tp3Qty;
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
        /// For every position in Phase3, checks whether:
        ///   (a) price has crossed EMA9 adversely → submit market exit immediately, or
        ///   (b) price has reached or exceeded TP3 → submit limit exit at TP3.
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
                    double ema9  = st.Ema9Value;

                    OrderAction exitAction = ph.Direction == "long"
                        ? OrderAction.Sell
                        : OrderAction.BuyToCover;

                    bool ema9Stop = ph.Direction == "long"
                        ? close < ema9          // long: exit when price drops below EMA9
                        : close > ema9;         // short: exit when price rises above EMA9

                    bool tp3Hit = ph.Tp3Price > 0 && (ph.Direction == "long"
                        ? close >= ph.Tp3Price
                        : close <= ph.Tp3Price);

                    if (tp3Hit && !ph.Tp3Submitted)
                    {
                        // Price reached TP3 — submit limit exit
                        ph.Tp3Submitted = true;
                        Print($"[Phase3] TP3 REACHED | id={ph.SignalId} price={close:F6} TP3={ph.Tp3Price:F6} qty={ph.Tp3Qty}");
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
                        // Price crossed EMA9 adversely — trail stop triggered
                        ph.Ema9StopHit = true;
                        Print($"[Phase3] EMA9 TRAIL STOP | id={ph.SignalId} price={close:F6} EMA9={ema9:F6} qty={ph.Tp3Qty}");
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
                        RiskBlocked    = busBlocked;
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
            }

            // ── Breakout detection (runs for every BIP in built-in modes) ─────
            if (Mode == BreakoutMode.BuiltIn || Mode == BreakoutMode.Both)
                CheckBreakout(bip, st);
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
                    // AtrValue = (AtrValue * (period-1) + tr) / period
                    // This is identical to ATR(14) on most platforms.
                    st.AtrValue = (st.AtrValue * (period - 1) + tr) / period;
                }

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

                    // ── Weekly / Monthly / ValueArea / InsideDay / GapRejection / PivotPoints / Fibonacci ──
                    // These types follow the same pattern as PrevDay: the range is
                    // set once at session boundary from prior-period data, then held
                    // fixed.  Full detection logic (reading weekly/monthly bars,
                    // volume profile, pivot calculations, fib levels) is deferred to
                    // the medium-term roadmap.  For now they remain as registered-but-
                    // never-established placeholders so the enum, RangeConfig, and
                    // InstrumentState are all consistent with the v6 feature contract.
                    case BreakoutType.Weekly:
                    case BreakoutType.Monthly:
                    case BreakoutType.ValueArea:
                    case BreakoutType.InsideDay:
                    case BreakoutType.GapRejection:
                    case BreakoutType.PivotPoints:
                    case BreakoutType.Fibonacci:
                        {
                            // Stub: these types are registered in InstrumentState.Ranges
                            // but will never set RangeEstablished = true until their
                            // detection logic is implemented.  CheckBreakout skips
                            // any range where RangeEstablished == false, so this is safe.
                            //
                            // TODO: implement per-type detection logic matching Python
                            //       engine detectors in rb/breakout_engine.py.
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
                    if (rs.FiredLong && rs.FiredShort) continue;

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
                    if (!rs.FiredLong && longBreak)
                    {
                        bool vwapOk = !RequireVwap || st.Vwap <= 0 || close > st.Vwap;
                        if (vwapOk && PassesCnnFilter("long", bip, st, close, atr, barTime, instrName, type))
                        {
                            FireEntry("long", bip, st, close, atr, barTime, instrName, type);
                            rs.FiredLong = true;
                            // Keep legacy ORB flags in sync
                            if (type == BreakoutType.ORB) st.BreakoutFiredLong = true;
                        }
                    }

                    // ── Short breakout ────────────────────────────────────────
                    if (!rs.FiredShort && shortBreak)
                    {
                        bool vwapOk = !RequireVwap || st.Vwap <= 0 || close < st.Vwap;
                        if (vwapOk && PassesCnnFilter("short", bip, st, close, atr, barTime, instrName, type))
                        {
                            FireEntry("short", bip, st, close, atr, barTime, instrName, type);
                            rs.FiredShort = true;
                            // Keep legacy ORB flags in sync
                            if (type == BreakoutType.ORB) st.BreakoutFiredShort = true;
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
        /// Build the 18-element raw tabular feature vector per feature_contract.json v6.
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
        ///   [14] breakout_type_ord     — (int)breakoutType / 12.0  (v6 new)
        ///   [15] asset_volatility_class — 0.0 low / 0.5 med / 1.0 high  (v6 new)
        ///   [16] hour_of_day           — barTime.Hour / 23.0  (v6 new)
        ///   [17] tp3_atr_mult_norm     — tp3AtrMult / 5.0  (v6 new)
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

                // ── v6 new features [14]–[17] ────────────────────────────────

                // [14] breakout_type_ord — ordinal of the BreakoutType enum / 12.0
                //      The caller (PassesCnnFilter via CheckBreakout) now passes the
                //      actual type that triggered the signal, so this is precise.
                float breakoutTypeOrd = (int)breakoutType / 12.0f;

                // [15] asset_volatility_class — 0.0 (low), 0.5 (med), 1.0 (high)
                float assetVolClass = GetVolatilityClass(instrRoot);

                // [16] hour_of_day — ET hour normalised to [0, 1]
                float hourOfDay = barTime.Hour / 23.0f;

                // [17] tp3_atr_mult_norm — TP3 multiplier / 5.0
                //      Default TP3 = 5.0 × ATR for ORB.  When per-type RangeConfig
                //      gains a tp3_atr_mult field this will read from there.
                float tp3AtrMultNorm = 5.0f / 5.0f; // 1.0 for now (ORB default)

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
                    SignalId    = signalId,
                    Direction   = direction,
                    Asset       = instrName,
                    Bip         = bip,
                    Phase       = BreakoutPhase.Phase1,
                    EntryPrice  = price,
                    AtrAtEntry  = atr,
                    SlPrice     = sl,
                    Tp1Price    = tp1,
                    Tp2Price    = tp2,
                    Tp3Price    = tp3,
                    TotalQty    = estimatedQty,
                    Tp1Qty      = tp1Qty,
                    Tp2Qty      = tp2Qty,
                    Tp3Qty      = tp3Qty,
                    OcoGroup    = "",
                    Tp3Submitted = false,
                    Ema9StopHit  = false,
                };
                lock (_phaseLock)
                    _positionPhases[signalId] = phase;
            }

            Print($"[Breakout] {breakoutType} {direction.ToUpper()} {instrName} BIP{bip} " +
                  $"@ {price:F2} SL={sl:F2} TP1={tp1:F2} TP2={tp2:F2}" +
                  (tp3 > 0 ? $" TP3={tp3:F2}" : "") +
                  $" id={signalId}" +
                  $" [positions: {_activePositionCount}/{MaxConcurrentPositions}]");

            // Execute directly (backtest) or queue (realtime)
            if (State == State.Historical)
                _engine.ExecuteEntryDirect(sig);
            else
                _engine.ProcessSignal(sig.ToJson());

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
        private static volatile bool   _isRiskBlocked    = false;
        private static volatile string _riskBlockReason  = "";

        public static bool   IsRiskBlocked   { get { return _isRiskBlocked; }   set { _isRiskBlocked   = value; } }
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
            _isRiskBlocked   = false;
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
        private int _numTabular = 18;
        private const int MaxTabular = 18; // C# always builds 18 features (v6 contract)
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
        /// Normalise 18 raw tabular features per feature_contract.json v6.
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

        private static readonly Color BgColor    = Color.FromArgb(0x0D, 0x0D, 0x0D);
        private static readonly Color BullCandle = Color.FromArgb(0x26, 0xA6, 0x9A);
        private static readonly Color BearCandle = Color.FromArgb(0xEF, 0x53, 0x50);
        private static readonly Color VwapLine   = Color.FromArgb(0x00, 0xE5, 0xFF);
        private static readonly Color VolBull    = Color.FromArgb(100, 0x26, 0xA6, 0x9A);
        private static readonly Color VolBear    = Color.FromArgb(100, 0xEF, 0x53, 0x50);

        /// <summary>Describes how to paint one range box on the chart image.</summary>
        private sealed class BoxStyle
        {
            /// <summary>Semi-transparent fill color.  Alpha=0 means no fill is drawn.</summary>
            public Color Fill   { get; }
            /// <summary>Opaque border/line color.</summary>
            public Color Border { get; }
            /// <summary>true = solid border lines, false = dashed (4px on / 4px off).</summary>
            public bool  Solid  { get; }

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
                        Color.FromArgb(40,  0xFF, 0xD7, 0x00),
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
                        Color.FromArgb(40,  0x29, 0xB6, 0xF6),
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
                        Color.FromArgb(30,  0x00, 0x96, 0x88),
                        Color.FromArgb(180, 0x00, 0x96, 0x88),
                        solid: true);

                // Monthly — orange solid border + orange fill alpha 30
                case BreakoutType.Monthly:
                    return new BoxStyle(
                        Color.FromArgb(30,  0xFF, 0x98, 0x00),
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
                        Color.FromArgb(30,  0x9E, 0x9D, 0x24),
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
                        Color.FromArgb(20,  0xFF, 0xBF, 0x00),
                        Color.FromArgb(180, 0xFF, 0xBF, 0x00),
                        solid: true);

                // Fallback: gold ORB style
                default:
                    return new BoxStyle(
                        Color.FromArgb(40,  0xFF, 0xD7, 0x00),
                        Color.FromArgb(100, 0xFF, 0xD7, 0x00),
                        solid: true);
            }
        }

        public class Bar
        {
            public DateTime Time   { get; set; }
            public double   Open   { get; set; }
            public double   High   { get; set; }
            public double   Low    { get; set; }
            public double   Close  { get; set; }
            public double   Volume { get; set; }
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

                double priceMin = bars.Min(b => b.Low)  * 0.9995;
                double priceMax = bars.Max(b => b.High) * 1.0005;
                double pRange   = priceMax - priceMin;
                if (pRange <= 0) pRange = 1;

                double volMax = bars.Max(b => b.Volume);
                if (volMax <= 0) volMax = 1;

                int   usableW = W - LeftPad - RightPad;
                float barW    = Math.Max(1f, (float)usableW / bars.Count);

                // ── Range box ─────────────────────────────────────────────────
                if (rangeHigh > rangeLow && rangeHigh > priceMin && rangeLow < priceMax)
                {
                    // Clamp to price panel
                    double clampedH = Math.Min(rangeHigh, priceMax);
                    double clampedL = Math.Max(rangeLow,  priceMin);

                    int yBoxH = PriceTop + (int)((priceMax - clampedH) / pRange * PriceH);
                    int yBoxL = PriceTop + (int)((priceMax - clampedL) / pRange * PriceH);
                    int boxH  = Math.Max(1, yBoxL - yBoxH);

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
                    var   bar  = bars[i];
                    bool  bull = bar.Close >= bar.Open;
                    Color col  = bull ? BullCandle : BearCandle;

                    float xCenter = LeftPad + i * barW + barW / 2f;
                    float xL      = LeftPad + i * barW + 1;
                    float xR      = xL + barW - 2;

                    int yHigh  = PriceTop + (int)((priceMax - bar.High)  / pRange * PriceH);
                    int yLow   = PriceTop + (int)((priceMax - bar.Low)   / pRange * PriceH);
                    int yOpen  = PriceTop + (int)((priceMax - bar.Open)  / pRange * PriceH);
                    int yClose = PriceTop + (int)((priceMax - bar.Close) / pRange * PriceH);

                    int bodyTop = Math.Min(yOpen, yClose);
                    int bodyH   = Math.Max(1, Math.Abs(yClose - yOpen));

                    using (var pen = new Pen(col, 1))
                        g.DrawLine(pen, xCenter, yHigh, xCenter, yLow);
                    using (var brush = new SolidBrush(col))
                        g.FillRectangle(brush, xL, bodyTop, Math.Max(1, xR - xL), bodyH);

                    // Volume panel
                    int volH   = (int)(bar.Volume / volMax * (VolPanelH - 2));
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
                            float x2 = LeftPad +  i      * barW + barW / 2f;
                            int   y1 = PriceTop + (int)((priceMax - vwapValues[i - 1]) / pRange * PriceH);
                            int   y2 = PriceTop + (int)((priceMax - vwapValues[i])     / pRange * PriceH);
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
