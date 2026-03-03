#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
// OrbCnnPredictor, CnnSessionThresholds, OrbChartRenderer live in
// NinjaTrader.NinjaScript (OrbCnnPredictor.cs).  The using below is not
// strictly needed when all files compile together inside NT8, but it makes
// the dependency explicit and prevents "type not found" if compilation order
// changes.
using static NinjaTrader.NinjaScript.CnnSessionThresholds;
#endregion

// =============================================================================
// BreakoutStrategy — NinjaTrader 8 Strategy
// =============================================================================
//
// A standalone Opening Range Breakout (ORB) strategy that works independently
// of Bridge.  It can be dropped on any chart and will:
//
//   1. Build the opening range for the first N minutes of each session.
//   2. Detect breakouts confirmed by VWAP position + volume surge + ATR filter.
//   3. Execute entries with ATR-based SL/TP brackets via BridgeOrderEngine.
//   4. Optionally relay signals from the Ruby indicator via SignalBus instead
//      of running its own ORB logic (set Mode = SignalBusRelay).
//   5. Manage multi-instrument subscriptions via AddDataSeries so one strategy
//      instance covers all 22 micro contracts.
//   6. Optionally gate every breakout through a trained HybridBreakoutCNN model
//      loaded from an ONNX file — zero Python process, < 1 ms per inference.
//      See OrbCnnPredictor.cs and scripts/export_onnx.py for setup.
//
// Files required in Documents\NinjaTrader 8\bin\Custom\ (all compile together):
//   BreakoutStrategy.cs   — this file
//   BridgeOrderEngine.cs  — order execution engine
//   OrbCnnPredictor.cs    — ONNX inference wrapper + chart renderer
//   SignalBus.cs          — in-process signal queue
//
// Additional DLLs required in bin\Custom\ when EnableCnnFilter = true:
//   Microsoft.ML.OnnxRuntime.dll         (NuGet: Microsoft.ML.OnnxRuntime 1.18.1)
//   onnxruntime.dll                      (NuGet runtimes\win-x64\native\)
//   onnxruntime_providers_shared.dll     (same native folder)
//
// Modes:
//   BuiltIn      — BreakoutStrategy runs its own ORB detection on each bar.
//                  Does not require Ruby or Bridge to be running.
//   SignalBusRelay — Acts as a pure consumer of the static SignalBus, executing
//                    whatever signals Ruby (or any other producer) enqueues.
//                    Useful when Ruby is on the chart and you want its signal
//                    logic without attaching Bridge.
//   Both         — Runs built-in ORB AND drains SignalBus.  Both sources can
//                  fire entries; the per-instrument cooldown prevents doubles.
//
// Signal flow (BuiltIn mode):
//   OnBarUpdate (BIP 0)
//     → UpdateVwap()
//     → UpdateOrbWindow()
//     → if breakout confirmed → SubmitEntry() → BridgeOrderEngine.ExecuteEntryDirect()
//
// Signal flow (SignalBusRelay mode):
//   Ruby.OnBarUpdate() → SignalBus.Enqueue()
//   BreakoutStrategy.OnBarUpdate() → BridgeOrderEngine.DrainSignalBus()
//                                  → BridgeOrderEngine.ExecuteEntryDirect()
//
// Multi-instrument:
//   Same AddDataSeries / BIP architecture as Bridge.  See Bridge.cs header for
//   the full explanation.  OnBarUpdate fires for every BIP; all bar-level work
//   is indexed by BarsInProgress so each instrument is handled independently.
//
// =============================================================================
//
// CNN integration quick-start:
//   1. Train model in Python:  python scripts/train_gpu.py
//   2. Export to ONNX:         python scripts/export_onnx.py
//   3. Copy orb_breakout_cnn.onnx + OnnxRuntime DLLs to NT8\bin\Custom\
//   4. Set CnnModelPath, enable EnableCnnFilter in strategy properties.
//
// =============================================================================

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

    public class BreakoutStrategy : Strategy
    {
        // =====================================================================
        // Per-instrument ORB state
        // =====================================================================
        // One instance per BIP index, allocated at DataLoaded.
        private sealed class InstrumentState
        {
            // ── ORB ───────────────────────────────────────────────────────────
            public double   OrbHigh           = 0;
            public double   OrbLow            = double.MaxValue;
            public bool     OrbEstablished    = false;
            public DateTime OrbSessionDate    = DateTime.MinValue;
            public DateTime OrbEndTime        = DateTime.MinValue;
            // Per-direction flags so both a long AND short can fire on news days.
            public bool     BreakoutFiredLong  = false;
            public bool     BreakoutFiredShort = false;

            // ── VWAP (session-resetting) ──────────────────────────────────────
            public double   CumTypicalVol  = 0;
            public double   CumVolume      = 0;
            public double   Vwap           = 0;
            public DateTime VwapDate       = DateTime.MinValue;

            // ── Volume baseline ───────────────────────────────────────────────
            // Running sum for a simple moving average of volume.
            public double[] VolBuffer;
            public int      VolBufIdx    = 0;
            public double   VolSum       = 0;   // running sum of VolBuffer
            public bool     VolReady     = false;
            public int      VolFilled    = 0;   // bars written so far (caps at period)

            // ── Cooldown ──────────────────────────────────────────────────────
            public DateTime LastEntryTime = DateTime.MinValue;

            // ── ATR — Wilder's smoothed, period 14 ───────────────────────────
            // During warmup (first <period> bars) we accumulate TR values in
            // AtrBuffer and compute a plain average.  Once AtrReady flips true
            // we switch to Wilder's exponential smoothing:
            //   AtrValue = (AtrValue * (period-1) + tr) / period
            // This matches the ATR(14) displayed by most platforms.
            public double   AtrValue     = 0;
            public double   TrSum        = 0;   // running TR sum (warmup only)
            public double   PrevClose    = 0;
            public double[] AtrBuffer;          // warmup ring-buffer
            public int      AtrBufIdx    = 0;
            public int      AtrFilled    = 0;   // bars written (caps at period)
            public bool     AtrReady     = false;

            // ── Session iterator (Globex-aware session detection) ─────────────
            public SessionIterator SessionIter = null;

            public InstrumentState(int volPeriod, int atrPeriod)
            {
                VolBuffer = new double[volPeriod];
                AtrBuffer = new double[atrPeriod];
            }
        }

        // ── BIP routing (built at DataLoaded) ─────────────────────────────────
        private readonly Dictionary<string, int> _symbolToBip =
            new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        private string[] _extraSymbols = new string[0];

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

        // ── Risk gate (mirrors Bridge.RiskBlocked so both respect it) ─────────
        // BreakoutStrategy has no HTTP listener of its own; it simply won't
        // allow entries when this is set.  Bridge can set it via SignalBus
        // or the caller can set it programmatically.
        internal volatile bool  RiskBlocked      = false;
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

        // ── 2. Mode ───────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Mode", GroupName = "2. Mode", Order = 1,
                 Description = "BuiltIn = run own ORB logic. " +
                               "SignalBusRelay = execute signals from Ruby via SignalBus. " +
                               "Both = run both simultaneously.")]
        public BreakoutMode Mode { get; set; } = BreakoutMode.Both;

        // ── 3. Instruments ────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Tracked Instruments", GroupName = "3. Instruments", Order = 1,
                 Description = "Comma-separated instrument root names. Chart instrument is always BIP 0. " +
                               "Full list: MGC,SIL,MHG,MCL,MNG,MES,MNQ,M2K,MYM,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET")]
        public string TrackedInstruments { get; set; } =
            "MGC,MES,MNQ,MCL,MNG,M2K,MYM,SIL,MHG,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET";

        // ── 4. ORB Parameters ─────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Range(5, 120)]
        [Display(Name = "ORB Window (minutes)", GroupName = "4. ORB", Order = 1,
                 Description = "Number of minutes from session open used to build the Opening Range.")]
        public int OrbMinutes { get; set; } = 30;

        [NinjaScriptProperty]
        [Range(1, 5)]
        [Display(Name = "Volume Surge Multiplier", GroupName = "4. ORB", Order = 2,
                 Description = "Breakout bar volume must exceed average × this multiplier.")]
        public double VolumeSurgeMult { get; set; } = 1.5;

        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "Volume Average Period", GroupName = "4. ORB", Order = 3,
                 Description = "Bars used to compute the rolling volume average for the surge filter.")]
        public int VolumeAvgPeriod { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "Require VWAP Confirmation", GroupName = "4. ORB", Order = 4,
                 Description = "Long breakout requires close above VWAP; short requires close below.")]
        public bool RequireVwap { get; set; } = true;

        [NinjaScriptProperty]
        [Range(0.1, 5.0)]
        [Display(Name = "Min ATR Range Ratio", GroupName = "4. ORB", Order = 5,
                 Description = "ORB range must be at least this many × ATR(14) to be considered valid. " +
                               "Set to 0 to disable.")]
        public double MinOrbAtrRatio { get; set; } = 0.3;

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Entry Cooldown (minutes)", GroupName = "4. ORB", Order = 6,
                 Description = "Minimum minutes between entries on the same instrument.")]
        public int EntryCooldownMinutes { get; set; } = 5;

        // ── 5. Risk Management ────────────────────────────────────────────────
        // Two sizing modes are available:
        //
        //   TptMode = true  → Fixed contracts from GetTptContracts(), sized for
        //                     Take Profit Trader funded accounts.  AccountSize,
        //                     RiskPercentPerTrade, and MaxContracts are still
        //                     used by BridgeOrderEngine when it processes
        //                     SignalBus signals, but FireEntry bypasses the
        //                     engine's risk-sizing math and sends the fixed qty
        //                     directly in the signal.
        //
        //   TptMode = false → Dynamic ATR-based sizing via BridgeOrderEngine.
        //                     AccountSize × RiskPercent ÷ (ATR × SlMult × PointValue),
        //                     capped at MaxContracts.

        [NinjaScriptProperty]
        [Display(Name = "TPT Mode (funded account)", GroupName = "5. Risk", Order = 1,
                 Description = "When enabled, uses fixed contract counts from the selected " +
                               "Take Profit Trader account tier instead of dynamic ATR-based sizing. " +
                               "Recommended for all TPT funded accounts.")]
        public bool TptMode { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "TPT Account Tier", GroupName = "5. Risk", Order = 2,
                 Description = "$50k → 2 micros, $100k → 3 micros, $150k → 4 micros. " +
                               "Edit GetTptContracts() to adjust within TPT's allowed range.")]
        public TptAccountTier AccountTier { get; set; } = TptAccountTier.FiftyK;

        [NinjaScriptProperty]
        [Display(Name = "Account Size ($)", GroupName = "5. Risk", Order = 3,
                 Description = "Fallback account size used when live balance is unavailable. " +
                               "Also used by BridgeOrderEngine for SignalBus relay sizing.")]
        public double AccountSize { get; set; } = 50000;

        [NinjaScriptProperty]
        [Range(0.1, 2.0)]
        [Display(Name = "Risk % Per Trade", GroupName = "5. Risk", Order = 4,
                 Description = "Fraction of account balance risked per trade (0.5 = 0.5%). " +
                               "Used only when TptMode is disabled.")]
        public double RiskPercentPerTrade { get; set; } = 0.5;

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max Contracts", GroupName = "5. Risk", Order = 5,
                 Description = "Hard cap on contracts per signal for dynamic sizing mode. " +
                               "Not used when TptMode is enabled (tier qty is the cap).")]
        public int MaxContracts { get; set; } = 5;

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "SL ATR Multiplier", GroupName = "5. Risk", Order = 6,
                 Description = "Stop loss = entry ± ATR(14) × this multiplier. " +
                               "Used for bracket placement in both TPT and dynamic modes.")]
        public double SlAtrMult { get; set; } = 1.5;

        [NinjaScriptProperty]
        [Range(0.5, 10.0)]
        [Display(Name = "TP1 ATR Multiplier", GroupName = "5. Risk", Order = 7,
                 Description = "Take profit 1 = entry ± ATR(14) × this multiplier.")]
        public double Tp1AtrMult { get; set; } = 2.0;

        [NinjaScriptProperty]
        [Range(0.5, 15.0)]
        [Display(Name = "TP2 ATR Multiplier", GroupName = "5. Risk", Order = 8,
                 Description = "Take profit 2 = entry ± ATR(14) × this multiplier. Set to 0 to disable.")]
        public double Tp2AtrMult { get; set; } = 3.5;

        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Brackets", GroupName = "5. Risk", Order = 9,
                 Description = "Automatically submit SL + TP bracket orders alongside each entry.")]
        public bool EnableAutoBrackets { get; set; } = true;

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "Default SL Ticks (fallback)", GroupName = "5. Risk", Order = 10,
                 Description = "Used when no explicit SL price is available and ATR cannot be computed.")]
        public int DefaultSlTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Range(5, 200)]
        [Display(Name = "Default TP Ticks (fallback)", GroupName = "5. Risk", Order = 11,
                 Description = "Used when no explicit TP price is available and ATR cannot be computed.")]
        public int DefaultTpTicks { get; set; } = 40;

        // ── 6. AI Filter (CNN) ────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Enable CNN Filter", GroupName = "6. AI Filter", Order = 1,
                 Description = "When true every rule-based breakout candidate is passed through " +
                               "the ONNX-exported HybridBreakoutCNN model as a final confirmation gate. " +
                               "Requires OrbCnnPredictor.dll + OnnxRuntime DLLs in NT8\\bin\\Custom\\.")]
        public bool EnableCnnFilter { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "CNN Model Path", GroupName = "6. AI Filter", Order = 2,
                 Description = "Full path to the exported .onnx file (run scripts/export_onnx.py to generate it). " +
                               @"Example: C:\NT8\Models\orb_breakout_cnn.onnx")]
        public string CnnModelPath { get; set; } = @"C:\NT8\Models\orb_breakout_cnn.onnx";

        [NinjaScriptProperty]
        [Range(0.50, 0.99)]
        [Display(Name = "CNN Threshold Override", GroupName = "6. AI Filter", Order = 3,
                 Description = "Fixed probability threshold applied to all sessions. " +
                               "Set to 0 to use the per-session thresholds from CnnSessionThresholds " +
                               "(recommended: 0.82 US/London, 0.72–0.75 overnight).")]
        public double CnnThresholdOverride { get; set; } = 0;

        [NinjaScriptProperty]
        [Display(Name = "CNN Session Key", GroupName = "6. AI Filter", Order = 4,
                 Description = "Session key used to look up the per-session threshold when " +
                               "CnnThresholdOverride is 0. Valid keys: us, london, london_ny, " +
                               "frankfurt, tokyo, shanghai, sydney, cme, cme_settle.")]
        public string CnnSessionKey { get; set; } = "us";

        [NinjaScriptProperty]
        [Display(Name = "CNN Snapshot Folder", GroupName = "6. AI Filter", Order = 5,
                 Description = "Folder where per-bar chart PNGs are written before each inference. " +
                               @"Default: %TEMP%\NT8_OrbCnn.  Must be writable.")]
        public string CnnSnapshotFolder { get; set; } = "";

        [NinjaScriptProperty]
        [Range(10, 120)]
        [Display(Name = "CNN Lookback Bars", GroupName = "6. AI Filter", Order = 6,
                 Description = "Number of 1-min bars included in each chart snapshot. " +
                               "60 is the standard training size.")]
        public int CnnLookbackBars { get; set; } = 60;

        [NinjaScriptProperty]
        [Display(Name = "Use CUDA", GroupName = "6. AI Filter", Order = 7,
                 Description = "Attempt to use the CUDA GPU execution provider (requires onnxruntime-gpu DLLs). " +
                               "Falls back to CPU silently if CUDA is unavailable.")]
        public bool CnnUseCuda { get; set; } = false;

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
                              "for risk-sized order execution across 22 micro contracts. " +
                              "Optionally gates entries through a HybridBreakoutCNN ONNX model.";
                Name        = "BreakoutStrategy";
                Calculate   = Calculate.OnBarClose;
                IsOverlay   = false;
                IsUnmanaged = true;
                // BarsRequiredToTrade: keep low so the strategy activates quickly
                // after loading historical data — indicators self-warm via their
                // own filled-bar counters (AtrFilled, VolFilled).
                BarsRequiredToTrade = 20;
            }

            // ── Configure ─────────────────────────────────────────────────────
            else if (State == State.Configure)
            {
                // Register as SignalBus consumer when running in relay or both modes.
                // Bridge also registers; having two consumers is fine — DrainAll is
                // called by both, but each signal only appears once in the queue.
                // If Bridge is running, it will drain first; BreakoutStrategy drains
                // whatever is left (typically nothing unless Bridge is not attached).
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                {
                    SignalBus.RegisterConsumer();
                    Print("[BreakoutStrategy] SignalBus consumer registered");
                }

                // ── Subscribe additional instruments ──────────────────────────
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
                        if (root == primaryRoot)        continue;
                        if (extras.Contains(root))      continue;

                        try
                        {
                            AddDataSeries(root,
                                new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1 },
                                "", true);
                            extras.Add(root);
                            Print($"[BreakoutStrategy] AddDataSeries: {root} → BIP {extras.Count}");
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
                // Build BIP routing table
                _symbolToBip.Clear();

                string primaryRoot = Instrument?.MasterInstrument.Name.ToUpperInvariant() ?? "";
                if (!string.IsNullOrEmpty(primaryRoot))
                    _symbolToBip[primaryRoot] = 0;

                for (int i = 0; i < _extraSymbols.Length; i++)
                {
                    int bip = i + 1;
                    if (bip < BarsArray.Length)
                    {
                        string confirmed = BarsArray[bip].Instrument.MasterInstrument.Name
                                                         .ToUpperInvariant();
                        _symbolToBip[confirmed] = bip;
                        Print($"[BreakoutStrategy] Routing: {confirmed} → BIP {bip}");
                    }
                }

                Print($"[BreakoutStrategy] {_symbolToBip.Count} instruments mapped");

                // Allocate per-instrument state — one entry per BIP
                int count = BarsArray.Length;
                _states = new InstrumentState[count];
                for (int i = 0; i < count; i++)
                {
                    _states[i] = new InstrumentState(VolumeAvgPeriod, 14);
                    // Each BIP gets its own SessionIterator so Globex futures
                    // (which roll sessions at 18:00 ET, not midnight) reset their
                    // ORB and VWAP at the correct time rather than on calendar-date.
                    try { _states[i].SessionIter = new SessionIterator(BarsArray[i]); }
                    catch (Exception ex) { Print($"[BreakoutStrategy] SessionIterator BIP{i}: {ex.Message}"); }
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
                    strategy:           this,
                    symbolToBip:        _symbolToBip,
                    getMyAccount:       () => capturedAccount,
                    getAccountSize:     () => AccountSize,
                    getRiskPercent:     () => RiskPercentPerTrade,
                    getMaxContracts:    () => MaxContracts,
                    getDefaultSlTicks:  () => DefaultSlTicks,
                    getDefaultTpTicks:  () => DefaultTpTicks,
                    getAutoBrackets:    () => EnableAutoBrackets,
                    getRiskBlocked:     () => RiskBlocked,
                    getRiskBlockReason: () => RiskBlockReason,
                    getRiskEnforcement: () => false, // BreakoutStrategy manages its own gate
                    onSignalReceived:   () => Interlocked.Increment(ref _metricSignalsReceived),
                    onSignalExecuted:   () => Interlocked.Increment(ref _metricSignalsExecuted),
                    onSignalRejected:   () => Interlocked.Increment(ref _metricSignalsRejected),
                    onExitExecuted:     () => Interlocked.Increment(ref _metricExitsExecuted),
                    onBusDrained:       n  => Interlocked.Add(ref _metricBusDrained, n),
                    sendPositionUpdate: () => { }, // BreakoutStrategy has no dashboard push
                    tag:                "Breakout"
                );
            }

            // ── Terminated ────────────────────────────────────────────────────
            else if (State == State.Terminated)
            {
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                    SignalBus.UnregisterConsumer();

                _cnn?.Dispose();
                _cnn    = null;
                _engine = null;
                _states = null;
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

            int bip = BarsInProgress;

            // Guard: state array bounds (can lag BarsArray during early init)
            if (bip >= _states.Length) return;

            var st = _states[bip];

            // ── Per-bar indicator updates (runs for every BIP) ────────────────
            UpdateVwap(bip, st);
            UpdateAtr(bip, st);
            UpdateVolumeAvg(bip, st);

            // ── ORB window accumulation (runs for every BIP) ──────────────────
            if (Mode == BreakoutMode.BuiltIn || Mode == BreakoutMode.Both)
                UpdateOrbWindow(bip, st);

            // ── Primary-series-only logic ─────────────────────────────────────
            if (bip == 0)
            {
                // Drain SignalBus (relay mode)
                if (Mode == BreakoutMode.SignalBusRelay || Mode == BreakoutMode.Both)
                    _engine.DrainSignalBus(State);

                // Flush queued orders (Realtime only — backtest uses DirectExecute)
                _engine.FlushOrderQueue(State);
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
                if (bars.Count == 0) return;

                int      last    = bars.Count - 1;
                DateTime barTime = bars.GetTime(last);

                // Reset on new session (Globex-aware via SessionIterator)
                bool newSession = IsNewSession(st, bars, barTime);
                if (newSession)
                {
                    st.CumTypicalVol = 0;
                    st.CumVolume     = 0;
                    // Record the session-start date so subsequent bars in the same
                    // session don't trigger another reset.
                    st.VwapDate = st.SessionIter != null
                        ? st.SessionIter.ActualSessionBegin.Date
                        : barTime.Date;
                }

                double high    = bars.GetHigh(last);
                double low     = bars.GetLow(last);
                double close   = bars.GetClose(last);
                double vol     = bars.GetVolume(last);
                double typical = (high + low + close) / 3.0;

                if (vol > 0)
                {
                    st.CumTypicalVol += typical * vol;
                    st.CumVolume     += vol;
                }

                st.Vwap = st.CumVolume > 0 ? st.CumTypicalVol / st.CumVolume : close;
            }
            catch (Exception ex) { Print($"[Breakout] UpdateVwap BIP{bip}: {ex.Message}"); }
        }

        private void UpdateAtr(int bip, InstrumentState st)
        {
            try
            {
                var bars = BarsArray[bip];
                if (bars.Count < 2) return;

                int    last = bars.Count - 1;
                double high = bars.GetHigh(last);
                double low  = bars.GetLow(last);
                double prev = st.PrevClose > 0 ? st.PrevClose : bars.GetClose(last - 1);
                double tr   = Math.Max(high - low,
                              Math.Max(Math.Abs(high - prev),
                                       Math.Abs(low  - prev)));

                int period = st.AtrBuffer.Length; // 14

                if (!st.AtrReady)
                {
                    // ── Warmup phase: fill ring-buffer, use plain SMA ─────────
                    // TrSum tracks the running total; no foreach needed.
                    st.TrSum -= st.AtrBuffer[st.AtrBufIdx];
                    st.AtrBuffer[st.AtrBufIdx] = tr;
                    st.AtrBufIdx = (st.AtrBufIdx + 1) % period;
                    st.TrSum    += tr;

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

                st.PrevClose = bars.GetClose(last);
            }
            catch (Exception ex) { Print($"[Breakout] UpdateAtr BIP{bip}: {ex.Message}"); }
        }

        private void UpdateVolumeAvg(int bip, InstrumentState st)
        {
            try
            {
                var    bars = BarsArray[bip];
                if (bars.Count == 0) return;

                double vol    = bars.GetVolume(bars.Count - 1);
                int    period = st.VolBuffer.Length;

                // Subtract the value that is about to be overwritten, add new value.
                st.VolSum -= st.VolBuffer[st.VolBufIdx];
                st.VolBuffer[st.VolBufIdx] = vol;
                st.VolBufIdx = (st.VolBufIdx + 1) % period;
                st.VolSum   += vol;

                if (st.VolFilled < period)
                    st.VolFilled++;

                if (!st.VolReady && st.VolFilled >= period)
                    st.VolReady = true;
            }
            catch (Exception ex) { Print($"[Breakout] UpdateVolumeAvg BIP{bip}: {ex.Message}"); }
        }

        #endregion

        // =====================================================================
        // ORB window — accumulate high/low during the opening range period
        // =====================================================================
        #region ORB Window

        private void UpdateOrbWindow(int bip, InstrumentState st)
        {
            try
            {
                var      bars    = BarsArray[bip];
                if (bars.Count == 0) return;

                int      last    = bars.Count - 1;
                DateTime barTime = bars.GetTime(last);
                double   high    = bars.GetHigh(last);
                double   low     = bars.GetLow(last);

                // ── Session reset (Globex-aware) ──────────────────────────────
                // Use SessionIterator when available so Globex futures (which
                // roll at 18:00 ET rather than midnight) reset their ORB at the
                // true session open instead of the calendar-day boundary.
                DateTime sessionStartDate = barTime.Date; // calendar-date fallback
                if (st.SessionIter != null)
                {
                    try
                    {
                        st.SessionIter.GetNextSession(barTime, false);
                        sessionStartDate = st.SessionIter.ActualSessionBegin.Date;
                    }
                    catch { /* keep calendar-date fallback */ }
                }

                bool newSession = sessionStartDate != st.OrbSessionDate;
                if (newSession)
                {
                    st.OrbHigh        = high;
                    st.OrbLow         = low;
                    st.OrbEstablished = false;
                    st.OrbSessionDate = sessionStartDate;

                    // Anchor the ORB window end to the true session begin reported
                    // by SessionIterator (Globex-aware).  Falling back to barTime
                    // only when the iterator is unavailable avoids a skewed window
                    // when the first bar loaded has a gap from the actual open.
                    DateTime sessionBegin = barTime;
                    if (st.SessionIter != null)
                    {
                        try { sessionBegin = st.SessionIter.ActualSessionBegin; }
                        catch { /* keep barTime fallback */ }
                    }
                    st.OrbEndTime = sessionBegin.AddMinutes(OrbMinutes);

                    // Reset per-direction breakout flags for the new session.
                    st.BreakoutFiredLong  = false;
                    st.BreakoutFiredShort = false;
                    return;
                }

                // Still inside the ORB window — expand range
                if (barTime <= st.OrbEndTime)
                {
                    st.OrbHigh = Math.Max(st.OrbHigh, high);
                    st.OrbLow  = Math.Min(st.OrbLow,  low);
                    return;
                }

                // First bar after the window closes — mark established
                if (!st.OrbEstablished)
                    st.OrbEstablished = true;
            }
            catch (Exception ex) { Print($"[Breakout] UpdateOrbWindow BIP{bip}: {ex.Message}"); }
        }

        #endregion

        // =====================================================================
        // Breakout detection — called after ORB window is established
        // =====================================================================
        #region Breakout Detection

        private void CheckBreakout(int bip, InstrumentState st)
        {
            try
            {
                if (!st.OrbEstablished) return;
                if (RiskBlocked)        return;

                var bars = BarsArray[bip];
                if (bars.Count == 0) return;

                int    last   = bars.Count - 1;
                double close  = bars.GetClose(last);
                double high   = bars.GetHigh(last);
                double low    = bars.GetLow(last);
                double vol    = bars.GetVolume(last);
                // Use actual filled-bar count for the avg so early bars don't
                // compare against a zero-diluted average.
                double volAvg = st.VolFilled > 0
                    ? st.VolSum / Math.Min(st.VolFilled, st.VolBuffer.Length)
                    : 0;
                double atr    = st.AtrValue; // always available (seeded after 1st bar)

                // ── ATR range filter ──────────────────────────────────────────
                if (MinOrbAtrRatio > 0 && atr > 0)
                {
                    double orbRange = st.OrbHigh - st.OrbLow;
                    if (orbRange < atr * MinOrbAtrRatio) return;
                }

                // ── Volume filter ─────────────────────────────────────────────
                if (volAvg > 0 && vol < volAvg * VolumeSurgeMult) return;

                // ── Cooldown ──────────────────────────────────────────────────
                DateTime barTime = bars.GetTime(last);
                if ((barTime - st.LastEntryTime).TotalMinutes < EntryCooldownMinutes) return;

                string instrName = bars.Instrument.MasterInstrument.Name;

                // ── Long breakout ─────────────────────────────────────────────
                // Evaluated independently of the short block so that a failed
                // VWAP confirmation on the long side never silently suppresses
                // a valid short signal on the same bar.
                if (!st.BreakoutFiredLong && close > st.OrbHigh)
                {
                    bool vwapOk = !RequireVwap || st.Vwap <= 0 || close > st.Vwap;
                    if (vwapOk && PassesCnnFilter("long", bip, st, close, atr, barTime, instrName))
                        FireEntry("long", bip, st, close, atr, barTime, instrName);
                }

                // ── Short breakout ────────────────────────────────────────────
                // Independent if (not else-if) so both directions are always
                // considered.  In practice close cannot be both > OrbHigh and
                // < OrbLow on the same bar, but the structure is explicit and
                // future-proof against any ORB range adjustment logic.
                if (!st.BreakoutFiredShort && close < st.OrbLow)
                {
                    bool vwapOk = !RequireVwap || st.Vwap <= 0 || close < st.Vwap;
                    if (vwapOk && PassesCnnFilter("short", bip, st, close, atr, barTime, instrName))
                        FireEntry("short", bip, st, close, atr, barTime, instrName);
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
            double price, double atr, DateTime barTime, string instrName)
        {
            // Fast path — filter disabled or model not loaded
            if (!EnableCnnFilter || _cnn == null) return true;

            try
            {
                // ── Resolve threshold ─────────────────────────────────────────
                float threshold = CnnThresholdOverride > 0
                    ? (float)CnnThresholdOverride
                    : GetSessionThreshold(CnnSessionKey);

                // ── Build tabular feature vector ──────────────────────────────
                float[] tabular = PrepareCnnTabular(bip, st, direction, price, atr, barTime);
                if (tabular == null) return true; // can't build features → pass through

                // ── Render chart snapshot ─────────────────────────────────────
                string snapshotPath = RenderCnnSnapshot(bip, st, direction, instrName, barTime);
                // snapshotPath may be null if rendering failed — predictor handles it gracefully

                // ── Run inference ─────────────────────────────────────────────
                var result = _cnn.Predict(snapshotPath ?? "", tabular, threshold);
                if (result == null) return true; // inference error → pass through

                if (result.Signal)
                {
                    Interlocked.Increment(ref _metricCnnAccepted);
                    Print($"[Breakout+CNN] {direction.ToUpper()} {instrName} BIP{bip} " +
                          $"ACCEPTED  {result}");
                    return true;
                }
                else
                {
                    Interlocked.Increment(ref _metricCnnRejected);
                    Print($"[Breakout+CNN] {direction.ToUpper()} {instrName} BIP{bip} " +
                          $"REJECTED  {result}");
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
        /// Build the 8-element raw tabular feature vector in TABULAR_FEATURES order.
        /// Raw values — normalisation is applied inside OrbCnnPredictor.NormaliseTabular.
        ///
        /// Order (must match breakout_cnn.py TABULAR_FEATURES exactly):
        ///   [0] quality_pct_norm  — estimated signal quality / 100
        ///   [1] volume_ratio      — current bar vol / rolling avg vol
        ///   [2] atr_pct           — ATR / close price (fraction, not percent)
        ///   [3] cvd_delta         — stub 0 (CVD requires tick data; extend when available)
        ///   [4] nr7_flag          — 1 if today's ORB range is the narrowest of 7 days
        ///   [5] direction_flag    — 1 = LONG, 0 = SHORT
        ///   [6] session_ordinal   — CnnSessionThresholds.GetSessionOrdinal(CnnSessionKey)
        ///   [7] london_overlap    — 1 if bar hour is 08:00–09:00 ET
        /// </summary>
        private float[] PrepareCnnTabular(
            int bip, InstrumentState st,
            string direction, double price, double atr, DateTime barTime)
        {
            try
            {
                // [0] quality_pct_norm: proxy using ORB range relative to ATR.
                //     A wide, clean ORB (range >= 1×ATR) scores near 1.0.
                //     This is imperfect but keeps the tabular head useful while
                //     Ruby's quality_pct is not yet wired through to C#.
                float qualityNorm = 0.5f; // sensible default
                if (atr > 0)
                {
                    double orbRange = st.OrbHigh - st.OrbLow;
                    qualityNorm = (float)Math.Min(orbRange / (atr * 1.5), 1.0);
                }

                // [1] volume_ratio
                double volAvg = st.VolFilled > 0
                    ? st.VolSum / Math.Min(st.VolFilled, st.VolBuffer.Length)
                    : 0;
                float volRatio = volAvg > 0
                    ? (float)(BarsArray[bip].GetVolume(BarsArray[bip].Count - 1) / volAvg)
                    : 1f;

                // [2] atr_pct — ATR as a fraction of price
                float atrPct = price > 0 && atr > 0 ? (float)(atr / price) : 0f;

                // [3] cvd_delta — stub; requires tick-level data not available here.
                //     Zero is safe: the tabular head learned to work without it.
                float cvdDelta = 0f;

                // [4] nr7_flag — true when today's ORB range is tighter than the
                //     previous 6 sessions (NR7 pattern).  We approximate using the
                //     current ATR: if the range is below 0.5×ATR it is likely NR7.
                float nr7Flag = (atr > 0 && (st.OrbHigh - st.OrbLow) < atr * 0.5) ? 1f : 0f;

                // [5] direction_flag
                float dirFlag = direction.Equals("long", StringComparison.OrdinalIgnoreCase) ? 1f : 0f;

                // [6] session_ordinal
                float sessionOrdinal = GetSessionOrdinal(CnnSessionKey);

                // [7] london_overlap_flag — 08:00–09:00 ET
                float londonOverlap = (barTime.Hour >= 8 && barTime.Hour <= 9) ? 1f : 0f;

                return new float[]
                {
                    qualityNorm,
                    volRatio,
                    atrPct,
                    cvdDelta,
                    nr7Flag,
                    dirFlag,
                    sessionOrdinal,
                    londonOverlap,
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
        /// </summary>
        private string RenderCnnSnapshot(
            int bip, InstrumentState st,
            string direction, string instrName, DateTime barTime)
        {
            try
            {
                var bars    = BarsArray[bip];
                int numBars = Math.Min(CnnLookbackBars, bars.Count);
                if (numBars < 5) return null;

                int startIdx = bars.Count - numBars;
                var barArr   = new OrbChartRenderer.Bar[numBars];

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

                return OrbChartRenderer.RenderToTemp(
                    barArr,
                    st.OrbHigh,
                    st.OrbLow,
                    st.Vwap,
                    direction,
                    instrName,
                    barTime,
                    _cnnSnapshotFolder);
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

        // ── TPT fixed-quantity lookup ─────────────────────────────────────────
        // Returns the conservative fixed contract count for the selected TPT tier.
        // Adjust these values within TPT's allowed maximums if you want to be more
        // aggressive — but never exceed 25 % of their stated per-instrument cap.
        private int GetTptContracts()
        {
            switch (AccountTier)
            {
                case TptAccountTier.FiftyK:        return 2;  // allowed up to 60
                case TptAccountTier.HundredK:      return 3;  // allowed up to 120
                case TptAccountTier.HundredFiftyK: return 4;  // allowed up to 150
                default:                           return 2;
            }
        }

        private void FireEntry(string direction, int bip, InstrumentState st,
                                double price, double atr, DateTime barTime, string instrName)
        {
            double tickSize = _engine.GetTickSize(bip);

            // ── Compute ATR-based SL/TP ───────────────────────────────────────
            double sl, tp1, tp2;

            if (atr > 0)
            {
                if (direction == "long")
                {
                    // SL below the ORB low (tighter of ORB or ATR-based)
                    double orbSl  = st.OrbLow - tickSize * 2;
                    double atrSl  = price - atr * SlAtrMult;
                    sl  = Math.Max(orbSl, atrSl); // tighter = higher for longs
                    tp1 = price + atr * Tp1AtrMult;
                    tp2 = Tp2AtrMult > 0 ? price + atr * Tp2AtrMult : 0;
                }
                else
                {
                    double orbSl  = st.OrbHigh + tickSize * 2;
                    double atrSl  = price + atr * SlAtrMult;
                    sl  = Math.Min(orbSl, atrSl); // tighter = lower for shorts
                    tp1 = price - atr * Tp1AtrMult;
                    tp2 = Tp2AtrMult > 0 ? price - atr * Tp2AtrMult : 0;
                }
            }
            else
            {
                // ATR not ready — use tick-based defaults (engine will also fall back)
                sl  = 0;
                tp1 = 0;
                tp2 = 0;
            }

            // ── Build a SignalBus.Signal and execute directly ──────────────────
            // Using SignalBus.Signal keeps the API consistent with Ruby's path
            // and lets the engine apply risk sizing in one place.
            string signalId = "brk-" + direction[0] + "-" + barTime.ToString("yyyyMMdd-HHmmss")
                              + "-" + instrName;

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
                Direction   = direction,
                SignalType  = "entry",
                Quantity    = signalQty,
                OrderType   = "market",
                StopLoss    = sl,
                TakeProfit  = tp1,
                TakeProfit2 = tp2,
                Strategy    = "BreakoutStrategy",
                Asset       = instrName,
                SignalId    = signalId,
                Timestamp   = barTime,
            };

            Print($"[Breakout] ORB {direction.ToUpper()} {instrName} BIP{bip} " +
                  $"@ {price:F2} SL={sl:F2} TP1={tp1:F2} TP2={tp2:F2} id={signalId}");

            // Execute directly (backtest) or queue (realtime)
            if (State == State.Historical)
                _engine.ExecuteEntryDirect(sig);
            else
                _engine.ProcessSignal(sig.ToJson());

            // Mark the appropriate direction as fired and update cooldown.
            // The opposite direction can still fire later in the same session
            // (e.g. a failed long breakout followed by a reversal short).
            if (direction == "long")
                st.BreakoutFiredLong  = true;
            else
                st.BreakoutFiredShort = true;
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
                long total    = accepted + rejected;
                float passRate = total > 0 ? (float)accepted / total * 100f : 0f;

                string cnnStatus = _cnn != null
                    ? $"LOADED  {Path.GetFileName(_cnn.ModelPath)}"
                    : "NOT LOADED (filter pass-through active)";

                float effectiveThreshold = CnnThresholdOverride > 0
                    ? (float)CnnThresholdOverride
                    : GetSessionThreshold(CnnSessionKey);

                sb.AppendLine($"  CNN filter    : {cnnStatus}");
                sb.AppendLine($"  CNN session   : {CnnSessionKey}  threshold={effectiveThreshold:P0}");
                sb.AppendLine($"  CNN accepted  : {accepted}  rejected={rejected}  pass-rate={passRate:F1}%");
                sb.AppendLine($"  CNN snapshots : {_cnnSnapshotFolder}");
            }
            else
            {
                sb.AppendLine("  CNN filter    : disabled");
            }

            sb.AppendLine("  ORB per instrument:");

            foreach (var kv in _symbolToBip)
            {
                int bip = kv.Value;
                if (bip >= _states.Length) continue;
                var st = _states[bip];
                string firedStr = "";
                if (st.BreakoutFiredLong)  firedStr += "L";
                if (st.BreakoutFiredShort) firedStr += "S";
                if (firedStr == "")        firedStr  = "-";
                string orbInfo = st.OrbEstablished
                    ? $"SET  H:{st.OrbHigh:F2} L:{st.OrbLow:F2} ATR:{st.AtrValue:F4} fired:{firedStr}"
                    : (st.OrbSessionDate != DateTime.MinValue ? "BUILDING" : "WAITING");
                sb.AppendLine($"    {kv.Key,-6} BIP{bip}: {orbInfo}");
            }

            return sb.ToString();
        }

        #endregion
    }
}
