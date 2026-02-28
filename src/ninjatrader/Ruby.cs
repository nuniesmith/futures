#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Indicators;
using SharpDX;
using SharpDX.Direct2D1;
using SharpDX.DirectWrite;
#endregion

// =============================================================================
// Ruby — v2 : Opening Range Breakout + Session Bias
// =============================================================================
//
// Philosophy: QUALITY over QUANTITY.  One clean breakout per session, not 20
// noisy signals in the chop.
//
// Market-open breakout detection:
//   1. Track the Opening Range (OR) — high/low of the first N minutes
//   2. Declare a Session Bias (LONG / SHORT / AUTO)
//   3. Wait for price to break the OR in the bias direction
//   4. Confirm with VWAP cross + volume surge + AO alignment + wave dominance
//   5. Fire ONE high-conviction entry signal
//   6. Optionally allow ONE "ADD" on the first pullback to VWAP/EMA9
//   7. Exit on BB touch, reversal, or session close
//
// Visual philosophy:
//   - Arrows only for actionable entries (green ▲ BREAKOUT, cyan ▲ ADD)
//   - A single red ▼ EXIT arrow when position closes
//   - Clean info boxes with only what matters
//   - No TP/BE/ADD/LOW VOL text spam
//   - Opening range drawn as a shaded region
//   - Breakout level as a horizontal line
//
// Plot index map (15 plots — same as v1 for compatibility):
//   0  Resistance        (adaptive S/R high)     — hidden by default
//   1  MidBand           (adaptive S/R mid)      — hidden by default
//   2  Support           (adaptive S/R low)      — hidden by default
//   3  EMA9              (9-period EMA)           — ON by default
//   4  BB_Upper          (Bollinger upper)        — hidden by default
//   5  BB_Mid            (Bollinger middle)       — hidden by default
//   6  BB_Lower          (Bollinger lower)        — hidden by default
//   7  VWAP              (intraday VWAP)          — ON by default
//   8  VWAP_Upper1       (+1σ band)               — hidden by default
//   9  VWAP_Lower1       (-1σ band)               — hidden by default
//  10  VWAP_Upper2       (+2σ band)               — hidden by default
//  11  VWAP_Lower2       (-2σ band)               — hidden by default
//  12  POC               (rolling volume POC)     — ON by default
//  13  VAH               (value area high)        — hidden by default
//  14  VAL               (value area low)         — hidden by default
//
// Installation:
//   1. NinjaTrader 8 → New → NinjaScript Editor → Indicators
//   2. Paste this file → Compile (F5)
//   3. Drag Ruby onto any chart (MGC, MES, MNQ, MCL, etc.)
// =============================================================================

namespace NinjaTrader.NinjaScript.Indicators
{
    public class Ruby : Indicator
    {
        // =====================================================================
        // Internal indicator references (shared)
        // =====================================================================
        private EMA ema9;
        private Bollinger bb;
        private SMA volSMA;
        private SMA aoFastSMA;
        private SMA aoSlowSMA;

        // =====================================================================
        // Core — Wave tracking state
        // =====================================================================
        private List<double> bullWaves = new List<double>();
        private List<double> bearWaves = new List<double>();
        private double dynEMA;
        private double trendSpeed;
        private double currentWaveRatio;
        private double signalQuality;
        private bool inBullPhase;

        // =====================================================================
        // Core — AO cache
        // =====================================================================
        private double aoValue;
        private double aoPrevValue;

        // =====================================================================
        // Core — Signal cooldown
        // =====================================================================
        private DateTime lastBuySignalTime = DateTime.MinValue;
        private DateTime lastSellSignalTime = DateTime.MinValue;
        private DateTime lastExitSignalTime = DateTime.MinValue;

        // =====================================================================
        // Signal forwarding state
        // =====================================================================
        private bool lastBarWasLong;
        private bool lastBarWasShort;
        private int signalSequence;

        // =====================================================================
        // Core — Heatmap regression state
        // =====================================================================
        private Series<double> colorLevelSeries;
        private int regressionLength;

        // =====================================================================
        // Core — Volume label state
        // =====================================================================
        private int lowVolStreak;

        // =====================================================================
        // Volume — VWAP state
        // =====================================================================
        private double cumTypicalVol;
        private double cumVolume;
        private double cumTypicalVolSq;
        private DateTime lastSessionDate;

        // =====================================================================
        // Volume — Rolling Volume Profile state
        // =====================================================================
        private double[] vpBinVolumes;
        private double vpPriceMin;
        private double vpPriceMax;
        private double currentPOC;
        private double currentVAH;
        private double currentVAL;

        // =====================================================================
        // Volume — Session POC tracking (naked POCs)
        // =====================================================================
        private struct SessionPOCInfo
        {
            public double Price;
            public DateTime Date;
            public bool IsNaked;
            public string Tag;
        }
        private List<SessionPOCInfo> sessionPOCs;
        private double prevSessionPOC;
        private DateTime prevSessionDate;

        // =====================================================================
        // Volume — CVD state
        // =====================================================================
        private double cvdAccumulator;
        private DateTime cvdAnchorDate;

        // =====================================================================
        // Opening Range Breakout (ORB) state
        // =====================================================================
        private double orbHigh;
        private double orbLow;
        private bool orbEstablished;
        private DateTime orbSessionDate;
        private DateTime orbEndTime;       // time when OR period ends
        private bool orbBreakoutFired;     // only ONE breakout per session
        private bool orbAddFired;          // only ONE add per session
        private string orbBreakoutDir;     // "long" or "short" — direction of the breakout
        private int orbBreakoutBar;        // bar index when breakout fired
        private double orbBreakoutPrice;   // entry price of the breakout

        // VWAP cross tracking for breakout confirmation
        private bool prevBarAboveVwap;
        private bool vwapCrossedUp;
        private bool vwapCrossedDown;
        private int vwapCrossBar;          // bar of the most recent VWAP cross

        // =====================================================================
        // OnStateChange
        // =====================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Ruby v2 — Opening Range Breakout + Session Bias | Quality over Quantity";
                Name = "Ruby";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                IsSuspendedWhileInactive = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                PaintPriceMarkers = false;
                ScaleJustification = ScaleJustification.Right;
                BarsRequiredToPlot = 220;

                // Signal forwarding defaults
                SendSignalsToBridge = true;
                BridgeUrl = "http://localhost:5680";
                ExitCooldownMinutes = 3;
                ExitOnReversal = true;
                ExitOnBBTouch = true;
                SL_ATR_Mult = 1.5;
                TP1_ATR_Mult = 2.0;
                TP2_ATR_Mult = 3.5;

                // ── Plot definitions (15 overlay plots) ──────────────────
                // Core plots (0–6)
                AddPlot(new Stroke(Brushes.Red, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "Resistance");                          // 0
                AddPlot(new Stroke(Brushes.Magenta, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "MidBand");                             // 1
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "Support");                             // 2
                AddPlot(new Stroke(Brushes.DodgerBlue, 2),
                    PlotStyle.Line, "EMA9");                                // 3
                AddPlot(new Stroke(Brushes.Red, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Upper");                            // 4
                AddPlot(new Stroke(Brushes.Magenta, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Mid");                              // 5
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Lower");                            // 6

                // Volume plots (7–14)
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Solid, 2),
                    PlotStyle.Line, "VWAP");                                // 7
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "VWAP_Upper1");                         // 8
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "VWAP_Lower1");                         // 9
                AddPlot(new Stroke(Brushes.DarkGoldenrod, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VWAP_Upper2");                         // 10
                AddPlot(new Stroke(Brushes.DarkGoldenrod, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VWAP_Lower2");                         // 11
                AddPlot(new Stroke(Brushes.Cyan, DashStyleHelper.Solid, 2),
                    PlotStyle.Line, "POC");                                 // 12
                AddPlot(new Stroke(Brushes.DodgerBlue, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VAH");                                 // 13
                AddPlot(new Stroke(Brushes.DodgerBlue, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VAL");                                 // 14

                // ── Default parameters ───────────────────────────────────
                // Core
                SR_Lookback = 20;
                AO_Fast = 5;
                AO_Slow = 34;
                WaveLookback = 200;
                MinWaveRatio = 1.5;
                RegressionLength = 200;
                HeatSensitivity = 70;
                SignalCooldownMinutes = 5;

                // Volume detection (shared)
                VolumeAvgPeriod = 20;
                VolumeSpikeMult = 1.8;
                VolumeLowMult = 0.5;
                LowVolStreakBars = 3;

                // Volume profile
                VP_Lookback = 100;
                VP_Bins = 40;
                ValueAreaPct = 70;

                // Session POC
                SessionPOC_MaxDays = 5;
                NakedPOC_Enabled = true;

                // CVD
                CVD_AnchorDaily = true;

                // Absorption
                AbsorptionBodyRatio = 30;
                AbsorptionVolMult = 1.5;

                // Visibility defaults (clean chart)
                ShowBollingerBands = false;
                ShowAdaptiveSR = false;
                ShowEMA9 = true;
                ShowLabels = true;
                ShowVolumeLabels = false;  // OFF by default now — cleaner
                ShowVWAP = true;
                ShowVWAPBands = false;
                ShowPOC = true;
                ShowValueArea = false;
                ShowDeltaOutline = false;
                ShowOpeningRange = true;

                // ORB defaults
                SessionBias = RubySessionBias.Auto;
                ORB_Minutes = 30;
                ORB_MinQuality = 60;
                ORB_VolumeGate = 1.2;
                ORB_RequireVWAPCross = true;
                ORB_AllowAdd = true;
                ORB_AddPullbackATR = 0.5;
                ORB_MaxAddBarsAfterBreakout = 30;
            }
            else if (State == State.DataLoaded)
            {
                // Shared indicators
                ema9 = EMA(Close, 9);
                bb = Bollinger(Close, 2, 20);
                volSMA = SMA(Volume, VolumeAvgPeriod);
                aoFastSMA = SMA(Typical, AO_Fast);
                aoSlowSMA = SMA(Typical, AO_Slow);

                // Core state init
                dynEMA = 0;
                trendSpeed = 0;
                inBullPhase = true;
                lowVolStreak = 0;
                regressionLength = RegressionLength;
                colorLevelSeries = new Series<double>(this);

                // Signal forwarding init
                lastBarWasLong = false;
                lastBarWasShort = false;
                signalSequence = 0;
                lastExitSignalTime = DateTime.MinValue;

                // Volume state init
                cumTypicalVol = 0;
                cumVolume = 0;
                cumTypicalVolSq = 0;
                lastSessionDate = DateTime.MinValue;
                vpBinVolumes = new double[VP_Bins];
                currentPOC = 0;
                currentVAH = 0;
                currentVAL = 0;
                sessionPOCs = new List<SessionPOCInfo>();
                prevSessionPOC = 0;
                prevSessionDate = DateTime.MinValue;
                cvdAccumulator = 0;
                cvdAnchorDate = DateTime.MinValue;

                // ORB state init
                orbHigh = 0;
                orbLow = double.MaxValue;
                orbEstablished = false;
                orbSessionDate = DateTime.MinValue;
                orbEndTime = DateTime.MinValue;
                orbBreakoutFired = false;
                orbAddFired = false;
                orbBreakoutDir = "";
                orbBreakoutBar = 0;
                orbBreakoutPrice = 0;
                prevBarAboveVwap = false;
                vwapCrossedUp = false;
                vwapCrossedDown = false;
                vwapCrossBar = 0;

                // ── Hide plots that are off by default ───────────────────
                if (!ShowAdaptiveSR)
                {
                    Plots[0].Brush = Brushes.Transparent;   // Resistance
                    Plots[1].Brush = Brushes.Transparent;   // MidBand
                    Plots[2].Brush = Brushes.Transparent;   // Support
                }
                if (!ShowEMA9)
                    Plots[3].Brush = Brushes.Transparent;

                if (!ShowBollingerBands)
                {
                    Plots[4].Brush = Brushes.Transparent;   // BB_Upper
                    Plots[5].Brush = Brushes.Transparent;   // BB_Mid
                    Plots[6].Brush = Brushes.Transparent;   // BB_Lower
                }
                if (!ShowVWAP)
                    Plots[7].Brush = Brushes.Transparent;

                if (!ShowVWAPBands)
                {
                    Plots[8].Brush = Brushes.Transparent;   // VWAP_Upper1
                    Plots[9].Brush = Brushes.Transparent;   // VWAP_Lower1
                    Plots[10].Brush = Brushes.Transparent;  // VWAP_Upper2
                    Plots[11].Brush = Brushes.Transparent;  // VWAP_Lower2
                }
                if (!ShowPOC)
                    Plots[12].Brush = Brushes.Transparent;

                if (!ShowValueArea)
                {
                    Plots[13].Brush = Brushes.Transparent;  // VAH
                    Plots[14].Brush = Brushes.Transparent;  // VAL
                }
            }
            else if (State == State.Terminated)
            {
                bullWaves = null;
                bearWaves = null;
                vpBinVolumes = null;
                sessionPOCs = null;
            }
        }

        // =====================================================================
        // Properties — Core
        // =====================================================================
        #region Properties — Core

        [NinjaScriptProperty]
        [Range(5, 200)]
        [Display(Name = "S/R Lookback", GroupName = "1. Core", Order = 1)]
        public int SR_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(2, 20)]
        [Display(Name = "AO Fast Period", GroupName = "1. Core", Order = 2)]
        public int AO_Fast { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "AO Slow Period", GroupName = "1. Core", Order = 3)]
        public int AO_Slow { get; set; }

        [NinjaScriptProperty]
        [Range(20, 500)]
        [Display(Name = "Wave Lookback", GroupName = "1. Core", Order = 4)]
        public int WaveLookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Min Wave Ratio", GroupName = "1. Core", Order = 5)]
        public double MinWaveRatio { get; set; }

        [NinjaScriptProperty]
        [Range(20, 500)]
        [Display(Name = "Regression Length", GroupName = "1. Core", Order = 6)]
        public int RegressionLength { get; set; }

        #endregion

        #region Properties — Volume Profile

        [NinjaScriptProperty]
        [Range(20, 500)]
        [Display(Name = "VP Lookback Bars", GroupName = "2. Volume Profile", Order = 1)]
        public int VP_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "VP Number of Bins", GroupName = "2. Volume Profile", Order = 2)]
        public int VP_Bins { get; set; }

        [NinjaScriptProperty]
        [Range(50, 90)]
        [Display(Name = "Value Area %", GroupName = "2. Volume Profile", Order = 3)]
        public int ValueAreaPct { get; set; }

        #endregion

        #region Properties — Session POC

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max Session POC Days", GroupName = "3. Session POC", Order = 1)]
        public int SessionPOC_MaxDays { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Naked POCs", GroupName = "3. Session POC", Order = 2)]
        public bool NakedPOC_Enabled { get; set; }

        #endregion

        #region Properties — Volume Detection

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "Volume Average Period", GroupName = "4. Volume", Order = 1)]
        public int VolumeAvgPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "Volume Spike Multiplier", GroupName = "4. Volume", Order = 2)]
        public double VolumeSpikeMult { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Volume Low Multiplier", GroupName = "4. Volume", Order = 3)]
        public double VolumeLowMult { get; set; }

        [NinjaScriptProperty]
        [Range(2, 10)]
        [Display(Name = "Low Vol Streak Bars", GroupName = "4. Volume", Order = 4)]
        public int LowVolStreakBars { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "Absorption Vol Multiplier", GroupName = "4. Volume", Order = 5)]
        public double AbsorptionVolMult { get; set; }

        [NinjaScriptProperty]
        [Range(10, 80)]
        [Display(Name = "Absorption Body Ratio %", GroupName = "4. Volume", Order = 6)]
        public int AbsorptionBodyRatio { get; set; }

        #endregion

        #region Properties — CVD

        [NinjaScriptProperty]
        [Display(Name = "Anchor CVD Daily", Description = "Reset CVD at market open each day",
            GroupName = "5. CVD", Order = 1)]
        public bool CVD_AnchorDaily { get; set; }

        #endregion

        #region Properties — Visibility

        [NinjaScriptProperty]
        [Display(Name = "Show EMA9", GroupName = "6. Visibility", Order = 1)]
        public bool ShowEMA9 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Bollinger Bands", GroupName = "6. Visibility", Order = 2)]
        public bool ShowBollingerBands { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Adaptive S/R", GroupName = "6. Visibility", Order = 3)]
        public bool ShowAdaptiveSR { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Entry Labels", Description = "BREAKOUT/ADD text with quality info",
            GroupName = "6. Visibility", Order = 4)]
        public bool ShowLabels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Volume Labels", Description = "Legacy TP/BE, ADD, LOW VOL labels (noisy — off by default)",
            GroupName = "6. Visibility", Order = 5)]
        public bool ShowVolumeLabels { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "Heat Sensitivity", Description = "Lookback for heatmap gradient scaling",
            GroupName = "6. Visibility", Order = 6)]
        public int HeatSensitivity { get; set; }

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Signal Cooldown (min)",
            GroupName = "6. Visibility", Order = 7)]
        public int SignalCooldownMinutes { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show VWAP", GroupName = "6. Visibility", Order = 8)]
        public bool ShowVWAP { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show VWAP Bands", Description = "+/-1 and +/-2 sigma bands",
            GroupName = "6. Visibility", Order = 9)]
        public bool ShowVWAPBands { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show POC", GroupName = "6. Visibility", Order = 10)]
        public bool ShowPOC { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Value Area Lines", Description = "VAH/VAL",
            GroupName = "6. Visibility", Order = 11)]
        public bool ShowValueArea { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Delta Outline", Description = "Color candle outline by volume delta",
            GroupName = "6. Visibility", Order = 12)]
        public bool ShowDeltaOutline { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Opening Range", Description = "Draw the ORB high/low zone on the chart",
            GroupName = "6. Visibility", Order = 13)]
        public bool ShowOpeningRange { get; set; }

        #endregion

        #region Properties — Opening Range Breakout

        [NinjaScriptProperty]
        [Display(Name = "Session Bias", Description = "Directional bias for the day — LONG only takes longs, SHORT only takes shorts, AUTO detects from pre-market/wave ratio",
            GroupName = "7. ORB Strategy", Order = 1)]
        public RubySessionBias SessionBias { get; set; }

        [NinjaScriptProperty]
        [Range(5, 120)]
        [Display(Name = "Opening Range Minutes", Description = "How many minutes after session open to build the opening range",
            GroupName = "7. ORB Strategy", Order = 2)]
        public int ORB_Minutes { get; set; }

        [NinjaScriptProperty]
        [Range(30, 95)]
        [Display(Name = "Min Breakout Quality %", Description = "Minimum signal quality (0-100) to fire a breakout signal",
            GroupName = "7. ORB Strategy", Order = 3)]
        public int ORB_MinQuality { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 3.0)]
        [Display(Name = "Volume Gate (x avg)", Description = "Breakout bar volume must be >= this multiple of average volume",
            GroupName = "7. ORB Strategy", Order = 4)]
        public double ORB_VolumeGate { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Require VWAP Cross", Description = "Require price to be on the correct side of VWAP for breakout direction",
            GroupName = "7. ORB Strategy", Order = 5)]
        public bool ORB_RequireVWAPCross { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Allow ADD Signal", Description = "Allow one ADD entry on first pullback to VWAP/EMA9 after breakout",
            GroupName = "7. ORB Strategy", Order = 6)]
        public bool ORB_AllowAdd { get; set; }

        [NinjaScriptProperty]
        [Range(0.2, 2.0)]
        [Display(Name = "ADD Pullback ATR", Description = "Max pullback from breakout high/low in ATR multiples for ADD signal",
            GroupName = "7. ORB Strategy", Order = 7)]
        public double ORB_AddPullbackATR { get; set; }

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "ADD Max Bars After Breakout", Description = "Maximum bars after breakout to allow an ADD signal",
            GroupName = "7. ORB Strategy", Order = 8)]
        public int ORB_MaxAddBarsAfterBreakout { get; set; }

        #endregion

        #region Properties — Signal Forwarding

        [NinjaScriptProperty]
        [Display(Name = "Send Signals to Bridge", Description = "Forward breakout/exit signals to Bridge strategy via SignalBus (backtest) or HTTP (live)",
            GroupName = "8. Signal Forwarding", Order = 1)]
        public bool SendSignalsToBridge { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Bridge URL", Description = "Base URL for Bridge HTTP listener (live/sim only)",
            GroupName = "8. Signal Forwarding", Order = 2)]
        public string BridgeUrl { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Exit on Reversal", Description = "Send exit signal when opposite entry signal fires",
            GroupName = "8. Signal Forwarding", Order = 3)]
        public bool ExitOnReversal { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Exit on BB Touch", Description = "Send exit signal when price reaches Bollinger Band in profit direction",
            GroupName = "8. Signal Forwarding", Order = 4)]
        public bool ExitOnBBTouch { get; set; }

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Exit Cooldown (min)", Description = "Minimum minutes between exit signals",
            GroupName = "8. Signal Forwarding", Order = 5)]
        public int ExitCooldownMinutes { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "SL ATR Multiplier", Description = "Stop loss distance as multiple of ATR(14)",
            GroupName = "8. Signal Forwarding", Order = 6)]
        public double SL_ATR_Mult { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 10.0)]
        [Display(Name = "TP1 ATR Multiplier", Description = "Take profit 1 distance as multiple of ATR(14)",
            GroupName = "8. Signal Forwarding", Order = 7)]
        public double TP1_ATR_Mult { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 15.0)]
        [Display(Name = "TP2 ATR Multiplier", Description = "Take profit 2 distance as multiple of ATR(14)",
            GroupName = "8. Signal Forwarding", Order = 8)]
        public double TP2_ATR_Mult { get; set; }

        #endregion

        // =====================================================================
        // Plot accessors
        // =====================================================================
        #region Plot Accessors

        [Browsable(false)][XmlIgnore] public Series<double> Resistance => Values[0];
        [Browsable(false)][XmlIgnore] public Series<double> MidBand => Values[1];
        [Browsable(false)][XmlIgnore] public Series<double> Support => Values[2];
        [Browsable(false)][XmlIgnore] public Series<double> EMA9 => Values[3];
        [Browsable(false)][XmlIgnore] public Series<double> BB_Upper => Values[4];
        [Browsable(false)][XmlIgnore] public Series<double> BB_Mid => Values[5];
        [Browsable(false)][XmlIgnore] public Series<double> BB_Lower => Values[6];
        [Browsable(false)][XmlIgnore] public Series<double> VWAP_Line => Values[7];
        [Browsable(false)][XmlIgnore] public Series<double> VWAP_Upper1 => Values[8];
        [Browsable(false)][XmlIgnore] public Series<double> VWAP_Lower1 => Values[9];
        [Browsable(false)][XmlIgnore] public Series<double> VWAP_Upper2 => Values[10];
        [Browsable(false)][XmlIgnore] public Series<double> VWAP_Lower2 => Values[11];
        [Browsable(false)][XmlIgnore] public Series<double> POC_Line => Values[12];
        [Browsable(false)][XmlIgnore] public Series<double> VAH_Line => Values[13];
        [Browsable(false)][XmlIgnore] public Series<double> VAL_Line => Values[14];

        #endregion

        // =====================================================================
        // Heatmap helpers
        // =====================================================================
        #region Heatmap Helpers

        private double ComputeRegressionLine(int len)
        {
            if (CurrentBar < len) return 0;

            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            int n = len;

            for (int i = 0; i < n; i++)
            {
                double x = i;
                double y = Close[i];
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }

            double denom = n * sumX2 - sumX * sumX;
            if (Math.Abs(denom) < 0.00001) return Close[0];

            double slope = (n * sumXY - sumX * sumY) / denom;
            double intercept = (sumY - slope * sumX) / n;

            return intercept;
        }

        private double ComputeColorLevel(double regressionVal)
        {
            double diff = Close[0] - regressionVal;
            double stdev = StdDev(Close, regressionLength)[0];
            if (stdev < 0.0001) stdev = 0.0001;
            return diff / stdev;
        }

        private Color LerpColor(Color a, Color b, double t)
        {
            t = Math.Max(0, Math.Min(1, t));
            byte r = (byte)(a.R + (b.R - a.R) * t);
            byte g = (byte)(a.G + (b.G - a.G) * t);
            byte bl = (byte)(a.B + (b.B - a.B) * t);
            return Color.FromRgb(r, g, bl);
        }

        private Brush GetHeatmapBrush(double colorLevel, double maxLevel, double minLevel)
        {
            if (colorLevel >= 0)
            {
                double range = maxLevel > 0.0001 ? maxLevel : 0.0001;
                double t = Math.Min(1.0, colorLevel / range);

                Color aqua = Color.FromRgb(0, 255, 255);
                Color yellow = Color.FromRgb(255, 255, 0);
                Color red = Color.FromRgb(255, 50, 50);

                Color c = t < 0.5
                    ? LerpColor(aqua, yellow, t * 2.0)
                    : LerpColor(yellow, red, (t - 0.5) * 2.0);

                return new SolidColorBrush(c);
            }
            else
            {
                double range = Math.Abs(minLevel) > 0.0001 ? Math.Abs(minLevel) : 0.0001;
                double t = Math.Min(1.0, Math.Abs(colorLevel) / range);

                Color aqua = Color.FromRgb(0, 255, 255);
                Color deepBlue = Color.FromRgb(0, 50, 150);

                Color c = LerpColor(aqua, deepBlue, t);
                return new SolidColorBrush(c);
            }
        }

        #endregion

        // =====================================================================
        // Rolling Volume Profile
        // =====================================================================
        #region Volume Profile

        private void ComputeRollingVolumeProfile()
        {
            int lookback = Math.Min(VP_Lookback, CurrentBar);
            if (lookback < 10) return;

            double pMin = double.MaxValue;
            double pMax = double.MinValue;

            for (int i = 0; i < lookback; i++)
            {
                if (High[i] > pMax) pMax = High[i];
                if (Low[i] < pMin) pMin = Low[i];
            }

            if (pMax <= pMin) return;

            vpPriceMin = pMin;
            vpPriceMax = pMax;

            for (int b = 0; b < VP_Bins; b++)
                vpBinVolumes[b] = 0;

            double binSize = (pMax - pMin) / VP_Bins;
            if (binSize <= 0) return;

            for (int i = 0; i < lookback; i++)
            {
                double vol = Volume[i];
                double barHi = High[i];
                double barLo = Low[i];

                int loBin = (int)((barLo - pMin) / binSize);
                int hiBin = (int)((barHi - pMin) / binSize);

                loBin = Math.Max(0, Math.Min(loBin, VP_Bins - 1));
                hiBin = Math.Max(0, Math.Min(hiBin, VP_Bins - 1));

                int span = hiBin - loBin + 1;
                double volPerBin = span > 0 ? vol / span : vol;

                for (int b = loBin; b <= hiBin; b++)
                    vpBinVolumes[b] += volPerBin;
            }

            // Find POC (bin with highest volume)
            int pocBin = 0;
            double pocVol = 0;
            for (int b = 0; b < VP_Bins; b++)
            {
                if (vpBinVolumes[b] > pocVol)
                {
                    pocVol = vpBinVolumes[b];
                    pocBin = b;
                }
            }

            currentPOC = pMin + (pocBin + 0.5) * binSize;

            // Compute Value Area (70% of total volume around POC)
            double totalVol = 0;
            for (int b = 0; b < VP_Bins; b++)
                totalVol += vpBinVolumes[b];

            double vaTarget = totalVol * (ValueAreaPct / 100.0);
            double vaAccum = vpBinVolumes[pocBin];
            int vaLo = pocBin;
            int vaHi = pocBin;

            while (vaAccum < vaTarget && (vaLo > 0 || vaHi < VP_Bins - 1))
            {
                double addLo = vaLo > 0 ? vpBinVolumes[vaLo - 1] : 0;
                double addHi = vaHi < VP_Bins - 1 ? vpBinVolumes[vaHi + 1] : 0;

                if (addHi >= addLo && vaHi < VP_Bins - 1)
                {
                    vaHi++;
                    vaAccum += vpBinVolumes[vaHi];
                }
                else if (vaLo > 0)
                {
                    vaLo--;
                    vaAccum += vpBinVolumes[vaLo];
                }
                else if (vaHi < VP_Bins - 1)
                {
                    vaHi++;
                    vaAccum += vpBinVolumes[vaHi];
                }
                else break;
            }

            currentVAL = pMin + vaLo * binSize;
            currentVAH = pMin + (vaHi + 1) * binSize;
        }

        #endregion

        // =====================================================================
        // Session POC tracking (naked POCs)
        // =====================================================================
        #region Session POC Tracking

        private void SaveSessionPOC(DateTime sessionDate, double poc)
        {
            if (poc <= 0) return;

            bool exists = false;
            for (int i = 0; i < sessionPOCs.Count; i++)
            {
                if (sessionPOCs[i].Date.Date == sessionDate.Date)
                {
                    var updated = sessionPOCs[i];
                    updated.Price = poc;
                    updated.IsNaked = true;
                    sessionPOCs[i] = updated;
                    exists = true;
                    break;
                }
            }

            if (!exists)
            {
                sessionPOCs.Add(new SessionPOCInfo
                {
                    Price = poc,
                    Date = sessionDate,
                    IsNaked = true,
                    Tag = "SPOC_" + sessionDate.ToString("MMdd")
                });
            }

            // Trim old sessions
            while (sessionPOCs.Count > SessionPOC_MaxDays)
                sessionPOCs.RemoveAt(0);
        }

        private void UpdateNakedPOCs()
        {
            for (int i = 0; i < sessionPOCs.Count; i++)
            {
                var sp = sessionPOCs[i];
                if (!sp.IsNaked) continue;
                if (sp.Date.Date == Time[0].Date) continue;

                if (Low[0] <= sp.Price && High[0] >= sp.Price)
                {
                    sp.IsNaked = false;
                    sessionPOCs[i] = sp;
                    RemoveDrawObject(sp.Tag);
                }
                else
                {
                    Draw.HorizontalLine(this, sp.Tag, sp.Price, Brushes.DarkCyan,
                        DashStyleHelper.Dot, 1);
                }
            }
        }

        #endregion

        // =====================================================================
        // OnBarUpdate — main logic
        // =====================================================================
        protected override void OnBarUpdate()
        {
            // Need enough bars for all components
            int minBars = Math.Max(AO_Slow + 5, Math.Max(regressionLength + 10, WaveLookback));
            if (CurrentBar < minBars)
                return;

            // Shared volume average (computed once)
            double volAvg = volSMA[0];
            double vol = Volume[0];

            // ==================================================================
            // SECTION A: VWAP (daily-resetting)
            // ==================================================================
            DateTime barDate = Time[0].Date;

            if (barDate != lastSessionDate)
            {
                // Save prior session's POC before resetting
                if (lastSessionDate != DateTime.MinValue && currentPOC > 0)
                    SaveSessionPOC(lastSessionDate, currentPOC);

                cumTypicalVol = 0;
                cumVolume = 0;
                cumTypicalVolSq = 0;
                lastSessionDate = barDate;

                if (CVD_AnchorDaily)
                {
                    cvdAccumulator = 0;
                    cvdAnchorDate = barDate;
                }

                // ── Reset ORB for new session ──
                ResetORBForNewSession(barDate);
            }

            double typical = (High[0] + Low[0] + Close[0]) / 3.0;
            if (vol > 0)
            {
                cumTypicalVol += typical * vol;
                cumVolume += vol;
                cumTypicalVolSq += typical * typical * vol;
            }

            double vwap = cumVolume > 0 ? cumTypicalVol / cumVolume : Close[0];
            VWAP_Line[0] = vwap;

            if (cumVolume > 0)
            {
                double variance = (cumTypicalVolSq / cumVolume) - (vwap * vwap);
                double vwapStddev = variance > 0 ? Math.Sqrt(variance) : 0;
                VWAP_Upper1[0] = vwap + vwapStddev;
                VWAP_Lower1[0] = vwap - vwapStddev;
                VWAP_Upper2[0] = vwap + 2.0 * vwapStddev;
                VWAP_Lower2[0] = vwap - 2.0 * vwapStddev;
            }
            else
            {
                VWAP_Upper1[0] = vwap;
                VWAP_Lower1[0] = vwap;
                VWAP_Upper2[0] = vwap;
                VWAP_Lower2[0] = vwap;
            }

            // ==================================================================
            // SECTION B: Rolling Volume Profile (POC / VAH / VAL)
            // ==================================================================
            if (CurrentBar >= VP_Lookback)
                ComputeRollingVolumeProfile();

            POC_Line[0] = currentPOC > 0 ? currentPOC : Close[0];
            VAH_Line[0] = currentVAH > 0 ? currentVAH : Close[0];
            VAL_Line[0] = currentVAL > 0 ? currentVAL : Close[0];

            // Naked POC management
            if (NakedPOC_Enabled && sessionPOCs != null)
                UpdateNakedPOCs();

            // ==================================================================
            // SECTION C: CVD — Cumulative Volume Delta
            // ==================================================================
            double barRange = High[0] - Low[0];
            double delta = 0;

            if (barRange > 0 && vol > 0)
            {
                double buyPct = (Close[0] - Low[0]) / barRange;
                double buyVol = vol * buyPct;
                double sellVol = vol - buyVol;
                delta = buyVol - sellVol;
            }
            cvdAccumulator += delta;

            // ==================================================================
            // SECTION D: Dynamic Trend EMA + Wave Tracking
            // ==================================================================
            double alpha = 2.0 / (20.0 + 1.0);
            if (CurrentBar <= 200)
                dynEMA = Close[0];
            else
                dynEMA = alpha * Close[0] + (1.0 - alpha) * dynEMA;

            double barContribution = Close[0] - Open[0];
            trendSpeed += barContribution;

            bool aboveDyn = Close[0] > dynEMA;
            bool prevAboveDyn = Close[1] > dynEMA;

            // Bull → Bear transition
            if (!aboveDyn && prevAboveDyn)
            {
                if (inBullPhase && trendSpeed != 0)
                {
                    bullWaves.Insert(0, Math.Abs(trendSpeed));
                    if (bullWaves.Count > WaveLookback)
                        bullWaves.RemoveAt(bullWaves.Count - 1);
                }
                trendSpeed = 0;
                inBullPhase = false;
            }

            // Bear → Bull transition
            if (aboveDyn && !prevAboveDyn)
            {
                if (!inBullPhase && trendSpeed != 0)
                {
                    bearWaves.Insert(0, Math.Abs(trendSpeed));
                    if (bearWaves.Count > WaveLookback)
                        bearWaves.RemoveAt(bearWaves.Count - 1);
                }
                trendSpeed = 0;
                inBullPhase = true;
            }

            double bullAvg = bullWaves.Count > 0 ? bullWaves.Average() : 0.001;
            double bearAvg = bearWaves.Count > 0 ? bearWaves.Average() : 0.001;
            if (bearAvg < 0.0001) bearAvg = 0.0001;
            if (bullAvg < 0.0001) bullAvg = 0.0001;
            currentWaveRatio = bullAvg / bearAvg;

            bool bullDominant = currentWaveRatio >= 1.0;

            // ==================================================================
            // SECTION E: Adaptive Support / Resistance
            // ==================================================================
            double highest = MAX(High, SR_Lookback)[0];
            double lowest = MIN(Low, SR_Lookback)[0];
            double mid = (highest + lowest) / 2.0;

            Resistance[0] = highest;
            MidBand[0] = mid;
            Support[0] = lowest;

            // ==================================================================
            // SECTION F: EMA9 + Bollinger Bands
            // ==================================================================
            EMA9[0] = ema9[0];
            BB_Upper[0] = bb.Upper[0];
            BB_Mid[0] = bb.Middle[0];
            BB_Lower[0] = bb.Lower[0];

            // ==================================================================
            // SECTION G: Awesome Oscillator
            // ==================================================================
            aoPrevValue = aoValue;
            aoValue = aoFastSMA[0] - aoSlowSMA[0];

            bool aoBullish = aoValue > 0 && aoValue > aoPrevValue;
            bool aoBearish = aoValue < 0 && aoValue < aoPrevValue;

            // ==================================================================
            // SECTION H: VWAP cross detection
            // ==================================================================
            bool barAboveVwap = Close[0] > vwap;
            vwapCrossedUp = false;
            vwapCrossedDown = false;

            if (barAboveVwap && !prevBarAboveVwap)
            {
                vwapCrossedUp = true;
                vwapCrossBar = CurrentBar;
            }
            if (!barAboveVwap && prevBarAboveVwap)
            {
                vwapCrossedDown = true;
                vwapCrossBar = CurrentBar;
            }
            prevBarAboveVwap = barAboveVwap;

            // ==================================================================
            // SECTION I: Opening Range tracking
            // ==================================================================
            bool inORPeriod = orbSessionDate == barDate && Time[0] <= orbEndTime;

            if (inORPeriod)
            {
                // Still building the opening range
                if (High[0] > orbHigh) orbHigh = High[0];
                if (Low[0] < orbLow) orbLow = Low[0];
            }
            else if (orbSessionDate == barDate && !orbEstablished && orbHigh > 0 && orbLow < double.MaxValue)
            {
                // OR period just ended — freeze the range
                orbEstablished = true;

                if (ShowOpeningRange)
                {
                    // Draw the opening range zone
                    Draw.Rectangle(this, "ORB_Zone_" + barDate.ToString("MMdd"),
                        true, orbEndTime, orbHigh, Time[0], orbLow,
                        Brushes.Transparent, Brushes.DodgerBlue, 15);

                    Draw.HorizontalLine(this, "ORB_High_" + barDate.ToString("MMdd"),
                        orbHigh, Brushes.DodgerBlue, DashStyleHelper.Dash, 1);
                    Draw.HorizontalLine(this, "ORB_Low_" + barDate.ToString("MMdd"),
                        orbLow, Brushes.DodgerBlue, DashStyleHelper.Dash, 1);
                }
            }

            // ==================================================================
            // SECTION J: Signal Quality Score (enhanced for breakout)
            // ==================================================================
            signalQuality = 0;

            // Determine the effective bias direction for scoring
            string effectiveBias = GetEffectiveBias(bullDominant);
            bool biasIsLong = effectiveBias == "long";
            double waveRatioForDir = biasIsLong ? currentWaveRatio : (1.0 / Math.Max(currentWaveRatio, 0.001));

            // Wave strength component (0.20)
            if (waveRatioForDir > MinWaveRatio * 0.7)
            {
                signalQuality += 0.15;
                if (waveRatioForDir > MinWaveRatio)
                    signalQuality += 0.05;
            }

            // AO alignment (0.15)
            if ((biasIsLong && aoBullish) || (!biasIsLong && aoBearish))
                signalQuality += 0.15;

            // AO acceleration (0.05)
            double aoAccel = Math.Abs(aoValue) - Math.Abs(aoPrevValue);
            if (aoAccel > 0)
                signalQuality += 0.05;

            // Price vs EMA9 (0.10)
            if ((biasIsLong && Close[0] > ema9[0]) || (!biasIsLong && Close[0] < ema9[0]))
                signalQuality += 0.10;

            // VWAP position (0.15) — for longs, above VWAP is bullish
            if (vwap > 0)
            {
                if ((biasIsLong && Close[0] > vwap) || (!biasIsLong && Close[0] < vwap))
                    signalQuality += 0.15;
            }

            // Volume surge (0.15) — above average = conviction
            if (volAvg > 0 && vol > volAvg * ORB_VolumeGate)
                signalQuality += 0.15;

            // CVD alignment (0.10) — delta confirming direction
            if ((biasIsLong && cvdAccumulator > 0) || (!biasIsLong && cvdAccumulator < 0))
                signalQuality += 0.10;

            // POC proximity bonus (0.05)
            if (currentPOC > 0)
            {
                double pocDistance = Math.Abs(Close[0] - currentPOC);
                double pocThreshold = (highest - lowest) * 0.05;
                if (pocDistance <= pocThreshold)
                    signalQuality += 0.05;
            }

            // ORB context bonus (0.05) — price broke out of opening range
            if (orbEstablished && !orbBreakoutFired)
            {
                if ((biasIsLong && Close[0] > orbHigh) || (!biasIsLong && Close[0] < orbLow))
                    signalQuality += 0.05;
            }

            signalQuality = Math.Min(1.0, Math.Max(0.0, signalQuality));

            // ==================================================================
            // SECTION K: BREAKOUT Signal Detection (replaces old buy/sell)
            // ==================================================================
            bool breakoutSignal = false;
            bool addSignal = false;
            string signalDir = "";

            // ── Primary Breakout: requires ORB established, not yet fired ──
            if (orbEstablished && !orbBreakoutFired && !inORPeriod)
            {
                bool longBreakout = effectiveBias != "short"
                    && Close[0] > orbHigh
                    && Close[1] <= orbHigh;    // just crossed

                bool shortBreakout = effectiveBias != "long"
                    && Close[0] < orbLow
                    && Close[1] >= orbLow;     // just crossed

                // VWAP gate
                if (ORB_RequireVWAPCross)
                {
                    if (longBreakout && Close[0] < vwap) longBreakout = false;
                    if (shortBreakout && Close[0] > vwap) shortBreakout = false;
                }

                // Volume gate
                if (volAvg > 0)
                {
                    if (longBreakout && vol < volAvg * ORB_VolumeGate) longBreakout = false;
                    if (shortBreakout && vol < volAvg * ORB_VolumeGate) shortBreakout = false;
                }

                // Quality gate
                double minQ = ORB_MinQuality / 100.0;
                if (longBreakout && signalQuality < minQ) longBreakout = false;
                if (shortBreakout && signalQuality < minQ) shortBreakout = false;

                // AO confirmation
                if (longBreakout && !aoBullish && aoValue <= 0) longBreakout = false;
                if (shortBreakout && !aoBearish && aoValue >= 0) shortBreakout = false;

                if (longBreakout)
                {
                    breakoutSignal = true;
                    signalDir = "long";
                }
                else if (shortBreakout)
                {
                    breakoutSignal = true;
                    signalDir = "short";
                }
            }

            // ── ADD Signal: one pullback entry after breakout ──
            if (!breakoutSignal && orbBreakoutFired && !orbAddFired && ORB_AllowAdd
                && (CurrentBar - orbBreakoutBar) <= ORB_MaxAddBarsAfterBreakout
                && (CurrentBar - orbBreakoutBar) >= 3)  // give it at least 3 bars
            {
                double atr = ATR(14)[0];
                bool pullbackToSupport = false;

                if (orbBreakoutDir == "long")
                {
                    // Pullback: price came back near VWAP or EMA9, but still above ORB high
                    double pullbackLevel = Math.Max(vwap, ema9[0]);
                    bool touchedSupport = Low[0] <= pullbackLevel + atr * 0.2;
                    bool bouncedUp = Close[0] > Open[0] && Close[0] > pullbackLevel;
                    bool stillAboveOR = Close[0] > orbHigh;
                    bool notTooDeep = Close[0] >= orbBreakoutPrice - atr * ORB_AddPullbackATR;

                    pullbackToSupport = touchedSupport && bouncedUp && stillAboveOR && notTooDeep;

                    // Volume should be present (but doesn't need to be a spike)
                    if (pullbackToSupport && volAvg > 0 && vol < volAvg * 0.8)
                        pullbackToSupport = false;

                    if (pullbackToSupport)
                    {
                        addSignal = true;
                        signalDir = "long";
                    }
                }
                else if (orbBreakoutDir == "short")
                {
                    double pullbackLevel = Math.Min(vwap, ema9[0]);
                    bool touchedResist = High[0] >= pullbackLevel - atr * 0.2;
                    bool bouncedDown = Close[0] < Open[0] && Close[0] < pullbackLevel;
                    bool stillBelowOR = Close[0] < orbLow;
                    bool notTooDeep = Close[0] <= orbBreakoutPrice + atr * ORB_AddPullbackATR;

                    pullbackToSupport = touchedResist && bouncedDown && stillBelowOR && notTooDeep;

                    if (pullbackToSupport && volAvg > 0 && vol < volAvg * 0.8)
                        pullbackToSupport = false;

                    if (pullbackToSupport)
                    {
                        addSignal = true;
                        signalDir = "short";
                    }
                }
            }

            // ── Execute BREAKOUT entry ──
            if (breakoutSignal && signalDir != ""
                && (Time[0] - (signalDir == "long" ? lastBuySignalTime : lastSellSignalTime)).TotalMinutes >= SignalCooldownMinutes)
            {
                orbBreakoutFired = true;
                orbBreakoutDir = signalDir;
                orbBreakoutBar = CurrentBar;
                orbBreakoutPrice = Close[0];

                Brush arrowBrush = signalDir == "long" ? Brushes.Lime : Brushes.Red;
                double arrowY = signalDir == "long" ? Low[0] - TickSize * 4 : High[0] + TickSize * 4;
                string arrowTag = signalDir == "long" ? "BrkUp" : "BrkDn";

                if (signalDir == "long")
                    Draw.ArrowUp(this, arrowTag + CurrentBar, true, 0, arrowY, Brushes.Lime);
                else
                    Draw.ArrowDown(this, arrowTag + CurrentBar, true, 0, arrowY, Brushes.Red);

                if (ShowLabels)
                {
                    string label = string.Format("BREAKOUT\nQ:{0:0}%  W:{1:0.0}x",
                        signalQuality * 100, currentWaveRatio);
                    double lblY = signalDir == "long" ? Low[0] - TickSize * 12 : High[0] + TickSize * 12;
                    Draw.Text(this, "BrkLbl" + CurrentBar, label, 0, lblY, arrowBrush);
                }

                if (signalDir == "long")
                    lastBuySignalTime = Time[0];
                else
                    lastSellSignalTime = Time[0];

                // ── Forward to Bridge ──
                if (SendSignalsToBridge)
                {
                    // Exit opposite position first
                    if (signalDir == "long" && lastBarWasShort && ExitOnReversal)
                        ForwardExitSignal("reversal_to_long");
                    if (signalDir == "short" && lastBarWasLong && ExitOnReversal)
                        ForwardExitSignal("reversal_to_short");

                    ForwardEntrySignal(signalDir, "breakout");
                }

                lastBarWasLong = signalDir == "long";
                lastBarWasShort = signalDir == "short";
            }

            // ── Execute ADD entry ──
            if (addSignal && signalDir != ""
                && (Time[0] - (signalDir == "long" ? lastBuySignalTime : lastSellSignalTime)).TotalMinutes >= SignalCooldownMinutes)
            {
                orbAddFired = true;

                double arrowY = signalDir == "long" ? Low[0] - TickSize * 4 : High[0] + TickSize * 4;

                if (signalDir == "long")
                    Draw.ArrowUp(this, "Add" + CurrentBar, true, 0, arrowY, Brushes.Cyan);
                else
                    Draw.ArrowDown(this, "Add" + CurrentBar, true, 0, arrowY, Brushes.Orange);

                if (ShowLabels)
                {
                    string label = string.Format("ADD\nQ:{0:0}%", signalQuality * 100);
                    double lblY = signalDir == "long" ? Low[0] - TickSize * 12 : High[0] + TickSize * 12;
                    Brush lblBrush = signalDir == "long" ? Brushes.Cyan : Brushes.Orange;
                    Draw.Text(this, "AddLbl" + CurrentBar, label, 0, lblY, lblBrush);
                }

                if (signalDir == "long")
                    lastBuySignalTime = Time[0];
                else
                    lastSellSignalTime = Time[0];

                // ── Forward to Bridge ──
                if (SendSignalsToBridge)
                    ForwardEntrySignal(signalDir, "add");
            }

            // ── EXIT detection: BB touch, reversal ──
            if (SendSignalsToBridge && ExitOnBBTouch && orbBreakoutFired
                && (Time[0] - lastExitSignalTime).TotalMinutes >= ExitCooldownMinutes)
            {
                bool exitLong = lastBarWasLong && High[0] >= bb.Upper[0];
                bool exitShort = lastBarWasShort && Low[0] <= bb.Lower[0];

                if (exitLong)
                {
                    ForwardExitSignal("bb_upper_touch");
                    // Draw exit marker
                    Draw.ArrowDown(this, "Exit" + CurrentBar, true, 0, High[0] + TickSize * 4, Brushes.Yellow);
                    if (ShowLabels)
                        Draw.Text(this, "ExitLbl" + CurrentBar, "EXIT", 0, High[0] + TickSize * 10, Brushes.Yellow);
                    lastBarWasLong = false;
                }
                else if (exitShort)
                {
                    ForwardExitSignal("bb_lower_touch");
                    Draw.ArrowUp(this, "Exit" + CurrentBar, true, 0, Low[0] - TickSize * 4, Brushes.Yellow);
                    if (ShowLabels)
                        Draw.Text(this, "ExitLbl" + CurrentBar, "EXIT", 0, Low[0] - TickSize * 10, Brushes.Yellow);
                    lastBarWasShort = false;
                }
            }

            // ==================================================================
            // SECTION L: Heatmap Bar Coloring
            // ==================================================================
            double regressionVal = ComputeRegressionLine(regressionLength);
            double colorLevel = ComputeColorLevel(regressionVal);
            colorLevelSeries[0] = colorLevel;

            double maxColorLevel = double.MinValue;
            double minColorLevel = double.MaxValue;
            int lookback = Math.Min(HeatSensitivity, CurrentBar);
            for (int i = 0; i < lookback; i++)
            {
                double cl = colorLevelSeries[i];
                if (cl > maxColorLevel) maxColorLevel = cl;
                if (cl < minColorLevel) minColorLevel = cl;
            }
            if (maxColorLevel < 0.0001) maxColorLevel = 0.0001;
            if (minColorLevel > -0.0001) minColorLevel = -0.0001;

            BarBrush = GetHeatmapBrush(colorLevel, maxColorLevel, minColorLevel);

            // ==================================================================
            // SECTION M: Volume labels (legacy — off by default, kept for compat)
            // ==================================================================
            bool volSpike = volAvg > 0 && vol > volAvg * VolumeSpikeMult;
            bool volLow = volAvg > 0 && vol < volAvg * VolumeLowMult;

            if (ShowVolumeLabels && volAvg > 0)
            {
                if (volLow)
                    lowVolStreak++;
                else
                    lowVolStreak = 0;

                double bbProximity = (bb.Upper[0] - bb.Lower[0]) * 0.05;
                bool atUpperBB = High[0] >= bb.Upper[0] - bbProximity;
                bool atLowerBB = Low[0] <= bb.Lower[0] + bbProximity;
                bool trendingUp = Close[0] > dynEMA && aoBullish;
                bool trendingDown = Close[0] < dynEMA && aoBearish;

                if (volSpike && (atUpperBB || atLowerBB))
                {
                    double y = atUpperBB ? High[0] + TickSize * 6 : Low[0] - TickSize * 6;
                    Draw.Text(this, "VolTP" + CurrentBar, "TP/BE", 0, y, Brushes.Yellow);
                }
                else if (volSpike && (trendingUp || trendingDown))
                {
                    double y = trendingUp ? Low[0] - TickSize * 6 : High[0] + TickSize * 6;
                    Draw.Text(this, "VolADD" + CurrentBar, "ADD", 0, y, Brushes.Cyan);
                }

                if (lowVolStreak >= LowVolStreakBars)
                    Draw.Text(this, "VolLow" + CurrentBar, "LOW VOL", 0, Low[0] - TickSize * 4, Brushes.Gray);
            }
            else
            {
                // Track low vol streak even when labels are off (used in info box)
                if (volLow)
                    lowVolStreak++;
                else
                    lowVolStreak = 0;
            }

            // ==================================================================
            // SECTION N: Absorption Detection
            // ==================================================================
            if (CurrentBar >= VolumeAvgPeriod + 1 && volAvg > 0)
            {
                double absorptionVolThreshold = volAvg * AbsorptionVolMult;
                double body = Math.Abs(Close[0] - Open[0]);
                double bodyRatio = barRange > 0 ? (body / barRange) * 100.0 : 100.0;
                bool isAbsorption = vol > absorptionVolThreshold
                    && bodyRatio < AbsorptionBodyRatio
                    && barRange > 0;

                if (isAbsorption)
                {
                    double midBar = (High[0] + Low[0]) / 2.0;
                    Brush absBrush;
                    double absY;

                    if (Close[0] >= midBar)
                    {
                        absBrush = Brushes.Cyan;
                        absY = Low[0] - TickSize * 4;
                    }
                    else
                    {
                        absBrush = Brushes.Magenta;
                        absY = High[0] + TickSize * 4;
                    }

                    Draw.Diamond(this, "Abs" + CurrentBar, true, 0, absY, absBrush);
                }
            }

            // ==================================================================
            // SECTION O: Candle Outline
            // ==================================================================
            if (ShowDeltaOutline && barRange > 0 && vol > 0)
            {
                double deltaPct = Math.Abs(delta) / vol;
                if (delta > 0)
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Cyan : Brushes.DodgerBlue;
                else
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Magenta : Brushes.MediumOrchid;
            }
            else
            {
                double deviation = Close[0] - dynEMA;
                double stdev = StdDev(Close, 200)[0];
                if (stdev < 0.0001) stdev = 0.0001;
                double normDev = (deviation / stdev) * (HeatSensitivity / 100.0);

                if (normDev > 0)
                    CandleOutlineBrush = Brushes.Lime;
                else
                    CandleOutlineBrush = Brushes.Red;

                if (Math.Abs(normDev) > 1.5)
                    CandleOutlineBrush = normDev > 0 ? Brushes.Cyan : Brushes.OrangeRed;
            }
        }

        // =====================================================================
        // ORB helpers
        // =====================================================================
        #region ORB Helpers

        /// <summary>
        /// Reset the opening range state for a new trading session.
        /// Called when a new session date is detected.
        /// </summary>
        private void ResetORBForNewSession(DateTime sessionDate)
        {
            orbHigh = 0;
            orbLow = double.MaxValue;
            orbEstablished = false;
            orbSessionDate = sessionDate;
            orbBreakoutFired = false;
            orbAddFired = false;
            orbBreakoutDir = "";
            orbBreakoutBar = 0;
            orbBreakoutPrice = 0;

            // Compute the ORB end time: session open + ORB_Minutes
            // NinjaTrader session template determines the open time.
            // We use the first bar's time on the new date as the reference.
            orbEndTime = Time[0].AddMinutes(ORB_Minutes);

            // Start building OR with the current bar
            orbHigh = High[0];
            orbLow = Low[0];
        }

        /// <summary>
        /// Determine the effective session bias based on user setting + market context.
        /// AUTO mode uses wave ratio + CVD + VWAP position to pick direction.
        /// </summary>
        private string GetEffectiveBias(bool bullDominant)
        {
            if (SessionBias == RubySessionBias.Long) return "long";
            if (SessionBias == RubySessionBias.Short) return "short";

            // AUTO: use multi-factor consensus
            int bullVotes = 0;
            int bearVotes = 0;

            // Wave dominance
            if (bullDominant && currentWaveRatio > 1.2) bullVotes++;
            else if (!bullDominant && (1.0 / Math.Max(currentWaveRatio, 0.001)) > 1.2) bearVotes++;

            // CVD direction
            if (cvdAccumulator > 0) bullVotes++;
            else if (cvdAccumulator < 0) bearVotes++;

            // Price vs VWAP
            if (VWAP_Line.IsValidDataPoint(0))
            {
                if (Close[0] > VWAP_Line[0]) bullVotes++;
                else bearVotes++;
            }

            // Price vs EMA9
            if (Close[0] > ema9[0]) bullVotes++;
            else bearVotes++;

            // AO
            if (aoValue > 0) bullVotes++;
            else if (aoValue < 0) bearVotes++;

            return bullVotes >= bearVotes ? "long" : "short";
        }

        #endregion

        // =====================================================================
        // Signal Forwarding — entry + exit helpers
        // =====================================================================
        #region Signal Forwarding

        /// <summary>
        /// Forward an entry signal (long or short) to the Bridge strategy.
        /// Uses SignalBus for in-process delivery (works in backtest + live).
        /// Additionally fires an HTTP POST to Bridge in Realtime state.
        /// </summary>
        private void ForwardEntrySignal(string direction, string signalType)
        {
            try
            {
                signalSequence++;
                string asset = Instrument != null ? Instrument.FullName : "Unknown";
                string signalId = "ruby-" + signalType[0] + direction[0] + "-" + Time[0].ToString("yyyyMMdd-HHmmss")
                    + "-" + (signalSequence & 0xFFFF).ToString("X4");

                // Compute ATR-based SL/TP levels
                double atr = ATR(14)[0];
                double entry = Close[0];
                double sl, tp1, tp2;

                if (direction == "long")
                {
                    // For breakout longs, SL below ORB low or ATR-based (whichever is tighter)
                    double orbSl = orbEstablished && orbLow < double.MaxValue ? orbLow - TickSize * 2 : 0;
                    double atrSl = entry - atr * SL_ATR_Mult;
                    sl = orbSl > 0 ? Math.Max(orbSl, atrSl) : atrSl;  // tighter of the two
                    tp1 = entry + atr * TP1_ATR_Mult;
                    tp2 = entry + atr * TP2_ATR_Mult;
                }
                else
                {
                    double orbSl = orbEstablished && orbHigh > 0 ? orbHigh + TickSize * 2 : 0;
                    double atrSl = entry + atr * SL_ATR_Mult;
                    sl = orbSl > 0 ? Math.Min(orbSl, atrSl) : atrSl;
                    tp1 = entry - atr * TP1_ATR_Mult;
                    tp2 = entry - atr * TP2_ATR_Mult;
                }

                // Always enqueue to the in-process SignalBus (works in backtest + live)
                SignalBus.EnqueueEntry(
                    direction: direction,
                    asset: asset,
                    stopLoss: sl,
                    takeProfit: tp1,
                    takeProfit2: tp2,
                    quantity: 1,
                    orderType: "market",
                    signalQuality: signalQuality,
                    waveRatio: currentWaveRatio,
                    strategy: "Ruby:" + signalType
                );

                Print($"[Ruby] ➤ {signalType.ToUpper()} {direction.ToUpper()} → SignalBus | Q:{signalQuality:P0} W:{currentWaveRatio:F1}x SL:{sl:F2} TP1:{tp1:F2} TP2:{tp2:F2} id={signalId}");

                // In Realtime, also POST to Bridge HTTP listener for redundancy
                if (State == State.Realtime && !string.IsNullOrEmpty(BridgeUrl))
                    PostSignalHttpAsync(signalId, direction, sl, tp1, tp2, asset, signalId);
            }
            catch (Exception ex)
            {
                Print($"[Ruby] Signal forwarding error: {ex.Message}");
            }
        }

        /// <summary>
        /// Forward an exit/flatten signal to the Bridge strategy.
        /// </summary>
        private void ForwardExitSignal(string reason)
        {
            try
            {
                if ((Time[0] - lastExitSignalTime).TotalMinutes < ExitCooldownMinutes)
                    return;

                lastExitSignalTime = Time[0];

                string asset = Instrument != null ? Instrument.FullName : "Unknown";

                SignalBus.EnqueueExit(
                    asset: asset,
                    reason: reason,
                    strategy: "Ruby"
                );

                Print($"[Ruby] ✖ EXIT → SignalBus | reason={reason}");

                // In Realtime, also POST flatten to Bridge HTTP
                if (State == State.Realtime && !string.IsNullOrEmpty(BridgeUrl))
                    PostFlattenHttpAsync(reason);
            }
            catch (Exception ex)
            {
                Print($"[Ruby] Exit signal error: {ex.Message}");
            }
        }

        /// <summary>
        /// Fire-and-forget HTTP POST of an entry signal to Bridge /execute_signal.
        /// </summary>
        private void PostSignalHttpAsync(string signalId, string direction,
            double sl, double tp1, double tp2, string asset, string id)
        {
            string json = "{"
                + "\"direction\":\"" + direction + "\""
                + ",\"quantity\":1"
                + ",\"order_type\":\"market\""
                + ",\"limit_price\":0"
                + ",\"stop_loss\":" + sl.ToString("F6")
                + ",\"take_profit\":" + tp1.ToString("F6")
                + ",\"tp2\":" + tp2.ToString("F6")
                + ",\"strategy\":\"Ruby\""
                + ",\"asset\":\"" + asset.Replace("\"", "\\\"") + "\""
                + ",\"signal_id\":\"" + id + "\""
                + "}";

            string url = BridgeUrl.TrimEnd('/') + "/execute_signal";

            Task.Run(() =>
            {
                try
                {
                    using (var wc = new WebClient())
                    {
                        wc.Headers[HttpRequestHeader.ContentType] = "application/json";
                        wc.UploadString(url, "POST", json);
                    }
                }
                catch { /* non-critical: SignalBus is the primary path */ }
            });
        }

        /// <summary>
        /// Fire-and-forget HTTP POST to Bridge /flatten endpoint.
        /// </summary>
        private void PostFlattenHttpAsync(string reason)
        {
            string json = "{\"reason\":\"" + reason.Replace("\"", "\\\"") + "\"}";
            string url = BridgeUrl.TrimEnd('/') + "/flatten";

            Task.Run(() =>
            {
                try
                {
                    using (var wc = new WebClient())
                    {
                        wc.Headers[HttpRequestHeader.ContentType] = "application/json";
                        wc.UploadString(url, "POST", json);
                    }
                }
                catch { /* non-critical: SignalBus is the primary path */ }
            });
        }

        #endregion

        // =====================================================================
        // Rendering — Info boxes
        // =====================================================================
        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);

            if (CurrentBar < 50 || chartControl == null || RenderTarget == null) return;

            RenderUnifiedDashboard(chartControl, chartScale);
        }

        // =====================================================================
        // Unified SharpDX dashboard — right-hand side, colored, structured
        // =====================================================================
        private void RenderUnifiedDashboard(ChartControl chartControl, ChartScale chartScale)
        {
            // ── Gather all data ──────────────────────────────────────
            bool bullDominant = currentWaveRatio >= 1.0;
            string effectiveBias = GetEffectiveBias(bullDominant);

            int qualityPct = (int)(signalQuality * 100);
            double displayRatio = bullDominant ? currentWaveRatio : (1.0 / Math.Max(currentWaveRatio, 0.001));

            double cl = colorLevelSeries != null && colorLevelSeries.IsValidDataPoint(0) ? colorLevelSeries[0] : 0;

            // ORB data
            string orbStatus;
            if (!orbEstablished && orbSessionDate == Time[0].Date)
                orbStatus = "BUILDING";
            else if (orbEstablished && !orbBreakoutFired)
                orbStatus = string.Format("SET  H:{0:F1}  L:{1:F1}", orbHigh, orbLow);
            else if (orbBreakoutFired)
                orbStatus = string.Format("BROKE {0}", orbBreakoutDir.ToUpper());
            else
                orbStatus = "WAITING";

            string addStatus = orbBreakoutFired ? (orbAddFired ? "USED" : "READY") : "---";

            // Volume data
            double barRange = High[0] - Low[0];
            double vol = Volume[0];
            double delta = 0;
            if (barRange > 0 && vol > 0)
            {
                double buyPct = (Close[0] - Low[0]) / barRange;
                delta = vol * (2.0 * buyPct - 1.0);
            }

            double volAvg = (CurrentBar >= VolumeAvgPeriod && volSMA != null) ? volSMA[0] : 0;
            double volRatioVal = volAvg > 0 ? vol / volAvg : 0;

            int nakedCount = 0;
            if (sessionPOCs != null)
                foreach (var sp in sessionPOCs)
                    if (sp.IsNaked) nakedCount++;

            double price = Close[0];
            string vaPosition = "---";
            if (currentVAH > 0 && currentVAL > 0 && currentVAH != currentVAL)
            {
                if (price > currentVAH) vaPosition = "ABOVE VA";
                else if (price < currentVAL) vaPosition = "BELOW VA";
                else vaPosition = "INSIDE VA";
            }

            double vwapVal = VWAP_Line.IsValidDataPoint(0) ? VWAP_Line[0] : 0;

            // ── Layout constants ─────────────────────────────────────
            float panelWidth = 260f;
            float lineH = 20f;
            float sectionGap = 6f;
            float headerH = 26f;
            float padX = 12f;
            float padY = 10f;
            float cornerRadius = 6f;

            // Count lines: title(1) + bias section header(1) + 4 rows + gap
            //            + ORB header(1) + 3 rows + gap
            //            + volume header(1) + 7 rows + gap
            //            + flow header(1) + 4 rows
            int totalLines = 4 + 3 + 7 + 4;  // data rows
            int totalHeaders = 4;
            int totalGaps = 3;
            float titleH = 32f;
            float panelHeight = titleH + padY
                + totalHeaders * headerH
                + totalLines * lineH
                + totalGaps * sectionGap
                + padY;

            float panelX = chartControl.CanvasRight - panelWidth - 14f;
            float panelY = 14f;

            // ── Colors ───────────────────────────────────────────────
            var bgColor = new SharpDX.Color(18, 18, 24, 230);       // near-black
            var headerBg = new SharpDX.Color(30, 35, 50, 255);       // dark navy
            var accentBlue = new SharpDX.Color(80, 140, 255, 255);     // section accent
            var accentGreen = new SharpDX.Color(0, 220, 100, 255);      // bullish green
            var accentRed = new SharpDX.Color(255, 70, 70, 255);      // bearish red
            var accentGold = new SharpDX.Color(255, 200, 50, 255);     // gold / VWAP
            var accentCyan = new SharpDX.Color(0, 210, 230, 255);      // POC / cyan
            var dimWhite = new SharpDX.Color(180, 185, 200, 255);    // label text
            var brightWhite = new SharpDX.Color(240, 245, 255, 255);    // value text
            var mutedGray = new SharpDX.Color(100, 105, 120, 255);    // separator
            var titleColor = new SharpDX.Color(120, 180, 255, 255);    // title blue

            // ── Create SharpDX resources ─────────────────────────────
            var target = RenderTarget;
            SharpDX.DirectWrite.Factory dwFactory = null;

            try
            {
                dwFactory = new SharpDX.DirectWrite.Factory();

                using (var bgBrush = new SharpDX.Direct2D1.SolidColorBrush(target, bgColor))
                using (var headerBgBrush = new SharpDX.Direct2D1.SolidColorBrush(target, headerBg))
                using (var accentBlueBr = new SharpDX.Direct2D1.SolidColorBrush(target, accentBlue))
                using (var accentGreenBr = new SharpDX.Direct2D1.SolidColorBrush(target, accentGreen))
                using (var accentRedBr = new SharpDX.Direct2D1.SolidColorBrush(target, accentRed))
                using (var accentGoldBr = new SharpDX.Direct2D1.SolidColorBrush(target, accentGold))
                using (var accentCyanBr = new SharpDX.Direct2D1.SolidColorBrush(target, accentCyan))
                using (var dimWhiteBr = new SharpDX.Direct2D1.SolidColorBrush(target, dimWhite))
                using (var brightWhiteBr = new SharpDX.Direct2D1.SolidColorBrush(target, brightWhite))
                using (var mutedGrayBr = new SharpDX.Direct2D1.SolidColorBrush(target, mutedGray))
                using (var titleBr = new SharpDX.Direct2D1.SolidColorBrush(target, titleColor))
                using (var fontTitle = new SharpDX.DirectWrite.TextFormat(dwFactory, "Segoe UI", SharpDX.DirectWrite.FontWeight.Bold, SharpDX.DirectWrite.FontStyle.Normal, 15f))
                using (var fontHeader = new SharpDX.DirectWrite.TextFormat(dwFactory, "Segoe UI", SharpDX.DirectWrite.FontWeight.SemiBold, SharpDX.DirectWrite.FontStyle.Normal, 12f))
                using (var fontLabel = new SharpDX.DirectWrite.TextFormat(dwFactory, "Consolas", SharpDX.DirectWrite.FontWeight.Normal, SharpDX.DirectWrite.FontStyle.Normal, 12f))
                using (var fontValue = new SharpDX.DirectWrite.TextFormat(dwFactory, "Consolas", SharpDX.DirectWrite.FontWeight.Bold, SharpDX.DirectWrite.FontStyle.Normal, 12f))
                {
                    // ── Background panel with rounded corners ────────────
                    var panelRect = new SharpDX.RectangleF(panelX, panelY, panelWidth, panelHeight);
                    var roundedRect = new RoundedRectangle
                    {
                        Rect = panelRect,
                        RadiusX = cornerRadius,
                        RadiusY = cornerRadius
                    };
                    target.FillRoundedRectangle(roundedRect, bgBrush);

                    // Subtle border
                    using (var borderBrush = new SharpDX.Direct2D1.SolidColorBrush(target, new SharpDX.Color(60, 70, 100, 180)))
                    {
                        target.DrawRoundedRectangle(roundedRect, borderBrush, 1.0f);
                    }

                    float cx = panelX + padX;
                    float cy = panelY + padY;
                    float valueX = panelX + 120f;  // right-aligned column for values
                    float fullW = panelWidth - padX * 2;

                    // ── Title ────────────────────────────────────────────
                    DrawText(target, dwFactory, fontTitle, titleBr, cx, cy, fullW, titleH, "RUBY v2");
                    // Version tag
                    DrawTextRight(target, dwFactory, fontLabel, mutedGrayBr, panelX + panelWidth - padX, cy + 2f, "DASHBOARD");
                    cy += titleH;

                    // ══════════════════════════════════════════════════════
                    // SECTION 1: SESSION & BIAS
                    // ══════════════════════════════════════════════════════
                    DrawSectionHeader(target, dwFactory, fontHeader, accentBlueBr, headerBgBrush, cx - 4f, cy, fullW + 8f, headerH, "SESSION & BIAS");
                    cy += headerH;

                    // Bias direction
                    bool isLong = effectiveBias == "long";
                    string biasLabel = isLong ? "LONG" : "SHORT";
                    string biasArrow = isLong ? " ▲" : " ▼";
                    string biasSource = SessionBias == RubySessionBias.Auto ? "auto" : "manual";
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Bias");
                    DrawText(target, dwFactory, fontValue, isLong ? accentGreenBr : accentRedBr, valueX, cy, fullW, lineH, biasLabel + biasArrow);
                    DrawTextRight(target, dwFactory, fontLabel, mutedGrayBr, panelX + panelWidth - padX, cy, biasSource);
                    cy += lineH;

                    // Quality
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Quality");
                    var qualBrush = qualityPct >= 70 ? accentGreenBr : qualityPct >= 50 ? accentGoldBr : qualityPct >= 30 ? dimWhiteBr : accentRedBr;
                    string qualStars = qualityPct >= 70 ? " ★★★" : qualityPct >= 50 ? " ★★" : qualityPct >= 30 ? " ★" : "";
                    DrawText(target, dwFactory, fontValue, qualBrush, valueX, cy, fullW, lineH, qualityPct + "%" + qualStars);
                    cy += lineH;

                    // Wave ratio
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Waves");
                    var waveBrush = displayRatio >= 2.0 ? accentGreenBr : displayRatio >= 1.5 ? accentGoldBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, waveBrush, valueX, cy, fullW, lineH, string.Format("{0:0.00}x", displayRatio));
                    cy += lineH;

                    // Weather / momentum
                    string weatherTag;
                    SharpDX.Direct2D1.SolidColorBrush weatherBrush;
                    if (cl <= -1.0) { weatherTag = "FREEZING"; weatherBrush = accentRedBr; }
                    else if (cl <= -0.5) { weatherTag = "COLD"; weatherBrush = accentRedBr; }
                    else if (cl < 0.5) { weatherTag = "NEUTRAL"; weatherBrush = dimWhiteBr; }
                    else if (cl < 1.0) { weatherTag = "WARM"; weatherBrush = accentGoldBr; }
                    else { weatherTag = "HOT"; weatherBrush = accentGreenBr; }
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Weather");
                    DrawText(target, dwFactory, fontValue, weatherBrush, valueX, cy, fullW, lineH, weatherTag);
                    cy += lineH;

                    cy += sectionGap;

                    // ══════════════════════════════════════════════════════
                    // SECTION 2: OPENING RANGE
                    // ══════════════════════════════════════════════════════
                    SharpDX.Direct2D1.SolidColorBrush orbAccent;
                    if (orbBreakoutFired)
                        orbAccent = orbBreakoutDir == "long" ? accentGreenBr : accentRedBr;
                    else
                        orbAccent = accentGoldBr;
                    DrawSectionHeader(target, dwFactory, fontHeader, orbAccent, headerBgBrush, cx - 4f, cy, fullW + 8f, headerH, "OPENING RANGE");
                    cy += headerH;

                    // ORB Status
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Status");
                    var orbStatusBrush = orbBreakoutFired ? accentGreenBr : orbEstablished ? accentGoldBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, orbStatusBrush, valueX, cy, fullW, lineH, orbStatus);
                    cy += lineH;

                    // ADD status
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Add");
                    var addBrush = addStatus == "READY" ? accentCyanBr : addStatus == "USED" ? mutedGrayBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, addBrush, valueX, cy, fullW, lineH, addStatus);
                    cy += lineH;

                    // ORB range width (if established)
                    if (orbEstablished && orbHigh > 0 && orbLow < double.MaxValue && orbLow > 0)
                    {
                        double orbRange = orbHigh - orbLow;
                        DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Range");
                        DrawText(target, dwFactory, fontValue, brightWhiteBr, valueX, cy, fullW, lineH, string.Format("{0:F2}", orbRange));
                    }
                    else
                    {
                        DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Range");
                        DrawText(target, dwFactory, fontValue, mutedGrayBr, valueX, cy, fullW, lineH, "---");
                    }
                    cy += lineH;

                    cy += sectionGap;

                    // ══════════════════════════════════════════════════════
                    // SECTION 3: VOLUME PROFILE
                    // ══════════════════════════════════════════════════════
                    DrawSectionHeader(target, dwFactory, fontHeader, accentCyanBr, headerBgBrush, cx - 4f, cy, fullW + 8f, headerH, "VOLUME PROFILE");
                    cy += headerH;

                    // VWAP
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "VWAP");
                    DrawText(target, dwFactory, fontValue, accentGoldBr, valueX, cy, fullW, lineH, string.Format("{0:0.00}", vwapVal));
                    cy += lineH;

                    // POC
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "POC");
                    DrawText(target, dwFactory, fontValue, accentCyanBr, valueX, cy, fullW, lineH, string.Format("{0:0.00}", currentPOC));
                    cy += lineH;

                    // VAH
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "VAH");
                    DrawText(target, dwFactory, fontValue, brightWhiteBr, valueX, cy, fullW, lineH, string.Format("{0:0.00}", currentVAH));
                    cy += lineH;

                    // VAL
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "VAL");
                    DrawText(target, dwFactory, fontValue, brightWhiteBr, valueX, cy, fullW, lineH, string.Format("{0:0.00}", currentVAL));
                    cy += lineH;

                    // Position relative to VA
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Position");
                    var posBrush = vaPosition == "ABOVE VA" ? accentGreenBr
                        : vaPosition == "BELOW VA" ? accentRedBr
                        : vaPosition == "INSIDE VA" ? accentGoldBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, posBrush, valueX, cy, fullW, lineH, vaPosition);
                    cy += lineH;

                    // Naked POCs
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Naked POCs");
                    DrawText(target, dwFactory, fontValue, nakedCount > 0 ? accentCyanBr : mutedGrayBr, valueX, cy, fullW, lineH, nakedCount.ToString());
                    cy += lineH;

                    // Volume ratio
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Volume");
                    string volText;
                    SharpDX.Direct2D1.SolidColorBrush volBrush;
                    if (volAvg > 0)
                    {
                        if (volRatioVal > VolumeSpikeMult)
                        {
                            volText = string.Format("SPIKE {0:0.0}x", volRatioVal);
                            volBrush = accentGreenBr;
                        }
                        else if (volRatioVal < VolumeLowMult)
                        {
                            volText = string.Format("LOW {0:0.0}x", volRatioVal);
                            volBrush = accentRedBr;
                        }
                        else
                        {
                            volText = string.Format("{0:0.0}x", volRatioVal);
                            volBrush = dimWhiteBr;
                        }
                    }
                    else
                    {
                        volText = "---";
                        volBrush = mutedGrayBr;
                    }
                    DrawText(target, dwFactory, fontValue, volBrush, valueX, cy, fullW, lineH, volText);
                    cy += lineH;

                    cy += sectionGap;

                    // ══════════════════════════════════════════════════════
                    // SECTION 4: ORDER FLOW
                    // ══════════════════════════════════════════════════════
                    DrawSectionHeader(target, dwFactory, fontHeader, accentGoldBr, headerBgBrush, cx - 4f, cy, fullW + 8f, headerH, "ORDER FLOW");
                    cy += headerH;

                    // Delta
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "Delta");
                    string deltaText = barRange > 0 && vol > 0
                        ? string.Format("{0:+0;-0;0}", delta)
                        : "---";
                    var deltaBrush = delta > 0 ? accentGreenBr : delta < 0 ? accentRedBr : dimWhiteBr;
                    string deltaTag = delta > 0 ? "  BUY" : delta < 0 ? "  SELL" : "";
                    DrawText(target, dwFactory, fontValue, deltaBrush, valueX, cy, fullW, lineH, deltaText + deltaTag);
                    cy += lineH;

                    // CVD
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "CVD");
                    var cvdBrush = cvdAccumulator > 0 ? accentGreenBr : cvdAccumulator < 0 ? accentRedBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, cvdBrush, valueX, cy, fullW, lineH, string.Format("{0:+0;-0;0}", cvdAccumulator));
                    cy += lineH;

                    // CVD Trend
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "CVD Trend");
                    string cvdTrendText = cvdAccumulator > 0 ? "Rising ▲" : cvdAccumulator < 0 ? "Falling ▼" : "Flat";
                    DrawText(target, dwFactory, fontValue, cvdBrush, valueX, cy, fullW, lineH, cvdTrendText);
                    cy += lineH;

                    // AO momentum
                    DrawLabel(target, dwFactory, fontLabel, dimWhiteBr, cx, cy, "AO");
                    string aoText = string.Format("{0:+0.00;-0.00;0.00}", aoValue);
                    string aoDir = aoValue > aoPrevValue ? " ▲" : aoValue < aoPrevValue ? " ▼" : " ─";
                    var aoBrush = aoValue > 0 ? accentGreenBr : aoValue < 0 ? accentRedBr : dimWhiteBr;
                    DrawText(target, dwFactory, fontValue, aoBrush, valueX, cy, fullW, lineH, aoText + aoDir);
                    cy += lineH;
                }
            }
            finally
            {
                if (dwFactory != null)
                    dwFactory.Dispose();
            }
        }

        // ── Dashboard drawing helpers ────────────────────────────────────────

        private void DrawSectionHeader(SharpDX.Direct2D1.RenderTarget target,
            SharpDX.DirectWrite.Factory dwFactory,
            SharpDX.DirectWrite.TextFormat font,
            SharpDX.Direct2D1.SolidColorBrush accentBrush,
            SharpDX.Direct2D1.SolidColorBrush bgBrush,
            float x, float y, float w, float h, string text)
        {
            // Header background bar
            var rect = new SharpDX.RectangleF(x, y, w, h);
            var rounded = new RoundedRectangle { Rect = rect, RadiusX = 3f, RadiusY = 3f };
            target.FillRoundedRectangle(rounded, bgBrush);

            // Left accent bar
            target.FillRectangle(new SharpDX.RectangleF(x, y + 3f, 3f, h - 6f), accentBrush);

            // Header text
            var textRect = new SharpDX.RectangleF(x + 10f, y + 4f, w - 14f, h - 4f);
            using (var layout = new SharpDX.DirectWrite.TextLayout(dwFactory, text, font, textRect.Width, textRect.Height))
            {
                target.DrawTextLayout(new SharpDX.Vector2(textRect.Left, textRect.Top), layout, accentBrush);
            }
        }

        private void DrawLabel(SharpDX.Direct2D1.RenderTarget target,
            SharpDX.DirectWrite.Factory dwFactory,
            SharpDX.DirectWrite.TextFormat font,
            SharpDX.Direct2D1.SolidColorBrush brush,
            float x, float y, string text)
        {
            using (var layout = new SharpDX.DirectWrite.TextLayout(dwFactory, text, font, 110f, 20f))
            {
                target.DrawTextLayout(new SharpDX.Vector2(x, y + 1f), layout, brush);
            }
        }

        private void DrawText(SharpDX.Direct2D1.RenderTarget target,
            SharpDX.DirectWrite.Factory dwFactory,
            SharpDX.DirectWrite.TextFormat font,
            SharpDX.Direct2D1.SolidColorBrush brush,
            float x, float y, float maxW, float maxH, string text)
        {
            using (var layout = new SharpDX.DirectWrite.TextLayout(dwFactory, text, font, maxW, maxH))
            {
                target.DrawTextLayout(new SharpDX.Vector2(x, y + 1f), layout, brush);
            }
        }

        private void DrawTextRight(SharpDX.Direct2D1.RenderTarget target,
            SharpDX.DirectWrite.Factory dwFactory,
            SharpDX.DirectWrite.TextFormat font,
            SharpDX.Direct2D1.SolidColorBrush brush,
            float rightX, float y, string text)
        {
            using (var layout = new SharpDX.DirectWrite.TextLayout(dwFactory, text, font, 200f, 20f))
            {
                float textWidth = layout.Metrics.Width;
                target.DrawTextLayout(new SharpDX.Vector2(rightX - textWidth, y + 1f), layout, brush);
            }
        }
    }

    // =========================================================================
    // Session Bias enum — used by Ruby ORB strategy
    // =========================================================================
    public enum RubySessionBias
    {
        Auto,
        Long,
        Short
    }
}
