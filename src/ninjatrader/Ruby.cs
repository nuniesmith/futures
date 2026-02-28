#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
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
#endregion

// =============================================================================
// Ruby
// =============================================================================
// 
//   1. EMA(9) — blue trend line
//   2. Bollinger Bands (2σ, 20) — Upper=Red, Mid=Magenta, Lower=LimeGreen
//   3. Dynamic Trend EMA — wave tracking for bull/bear dominance
//   4. Wave Dominance — ratio of bull vs bear wave strengths
//   5. Adaptive S/R — MAX(High,N) / MIN(Low,N) / midpoint
//   6. Awesome Oscillator (5,34) — momentum confirmation
//   7. Signal Quality (0–100%) — composite score (now enhanced with VWAP/POC)
//   8. Buy/Sell arrows with quality labels (cooldown)
//   9. Heatmap bar coloring — aqua→yellow→red (regression deviation)
//  10. Candle outline heatmap — lime/red relative to dynEMA
//  11. Volume action labels — TP/BE, ADD, LOW VOL
//  12. Info box (top-right) — wave ratio, quality, AO, weather
//  13. Intraday VWAP — daily-resetting + optional σ bands
//  14. Rolling Volume Profile — POC / VAH / VAL
//  15. Session POC Tracking — naked POC horizontal rays
//  16. CVD Approximation — cumulative volume delta (OHLCV heuristic)
//  17. Absorption Detection — diamonds on absorption candles
//  18. Info box (top-left) — VWAP, POC, VAH/VAL, CVD, delta
//
// Plot index map (15 plots):
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
        private SMA volSMA;          // single shared volume average
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
        // State management
        // =====================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Ruby — Trend, Wave Dominance, Heatmap, VWAP, Volume Profile, CVD, Signals";
                Name = "Ruby";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                IsSuspendedWhileInactive = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                PaintPriceMarkers = false;
                ScaleJustification = ScaleJustification.Right;
                BarsRequiredToPlot = 220;

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
                ShowVolumeLabels = true;
                ShowVWAP = true;
                ShowVWAPBands = false;
                ShowPOC = true;
                ShowValueArea = false;
                ShowDeltaOutline = false;
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
        // Parameters
        // =====================================================================
        #region Properties — Core

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "S/R Lookback", Description = "Bars for adaptive support/resistance",
            GroupName = "1. Core — Trend & Waves", Order = 1)]
        public int SR_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(2, 20)]
        [Display(Name = "AO Fast Period",
            GroupName = "1. Core — Trend & Waves", Order = 2)]
        public int AO_Fast { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "AO Slow Period",
            GroupName = "1. Core — Trend & Waves", Order = 3)]
        public int AO_Slow { get; set; }

        [NinjaScriptProperty]
        [Range(50, 500)]
        [Display(Name = "Wave Lookback", Description = "Waves to keep for ratio calculation",
            GroupName = "1. Core — Trend & Waves", Order = 4)]
        public int WaveLookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Min Wave Ratio", Description = "Minimum wave ratio for signal generation",
            GroupName = "1. Core — Trend & Waves", Order = 5)]
        public double MinWaveRatio { get; set; }

        [NinjaScriptProperty]
        [Range(50, 500)]
        [Display(Name = "Regression Length", Description = "Bar count for heatmap regression line",
            GroupName = "1. Core — Trend & Waves", Order = 6)]
        public int RegressionLength { get; set; }

        #endregion

        #region Properties — Volume Profile

        [NinjaScriptProperty]
        [Range(20, 500)]
        [Display(Name = "VP Lookback (bars)", Description = "Bars for rolling volume profile",
            GroupName = "2. Volume Profile", Order = 1)]
        public int VP_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "VP Bins", Description = "Price bins for volume distribution",
            GroupName = "2. Volume Profile", Order = 2)]
        public int VP_Bins { get; set; }

        [NinjaScriptProperty]
        [Range(50, 90)]
        [Display(Name = "Value Area %", Description = "Percentage of volume for value area",
            GroupName = "2. Volume Profile", Order = 3)]
        public int ValueAreaPct { get; set; }

        #endregion

        #region Properties — Session POC

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Session POC Max Days",
            GroupName = "3. Session POC", Order = 1)]
        public int SessionPOC_MaxDays { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Naked POCs",
            GroupName = "3. Session POC", Order = 2)]
        public bool NakedPOC_Enabled { get; set; }

        #endregion

        #region Properties — Volume Detection

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "Volume Avg Period", Description = "SMA period for average volume",
            GroupName = "4. Volume Detection", Order = 1)]
        public int VolumeAvgPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "Volume Spike Mult", Description = "Volume > avg × this = spike",
            GroupName = "4. Volume Detection", Order = 2)]
        public double VolumeSpikeMult { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Low Volume Mult", Description = "Volume < avg × this = low volume",
            GroupName = "4. Volume Detection", Order = 3)]
        public double VolumeLowMult { get; set; }

        [NinjaScriptProperty]
        [Range(2, 10)]
        [Display(Name = "Low Vol Streak Bars",
            GroupName = "4. Volume Detection", Order = 4)]
        public int LowVolStreakBars { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "Absorption Vol Mult", Description = "Min volume as multiple of avg for absorption",
            GroupName = "4. Volume Detection", Order = 5)]
        public double AbsorptionVolMult { get; set; }

        [NinjaScriptProperty]
        [Range(10, 50)]
        [Display(Name = "Absorption Body Ratio %", Description = "Max body/range ratio for absorption candle",
            GroupName = "4. Volume Detection", Order = 6)]
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
        [Display(Name = "Show Entry Labels", Description = "BUY/SELL text with quality info",
            GroupName = "6. Visibility", Order = 4)]
        public bool ShowLabels { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Volume Labels", Description = "TP/BE, ADD, LOW VOL action labels",
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
        [Display(Name = "Show Value Area Lines", Description = "VAH/VAL (values still in info box)",
            GroupName = "6. Visibility", Order = 11)]
        public bool ShowValueArea { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Delta Outline", Description = "Color candle outline by volume delta (overrides dynEMA outline)",
            GroupName = "6. Visibility", Order = 12)]
        public bool ShowDeltaOutline { get; set; }

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
        // Heatmap helpers — ported from PineScript
        // =====================================================================
        #region Heatmap Helpers

        /// <summary>
        /// Linear regression line value at current bar.
        /// </summary>
        private double ComputeRegressionLine(int len)
        {
            if (CurrentBar < len) return Median[0];

            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
            for (int i = 0; i < len; i++)
            {
                double x = CurrentBar - i;
                double y = (High[i] + Low[i]) / 2.0;
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
                sumY2 += y * y;
            }

            double n = len;
            double meanX = sumX / n;
            double meanY = sumY / n;
            double varX = (sumX2 / n) - (meanX * meanX);

            if (varX < 1e-10) return Median[0];

            double covariance = (sumXY / n) - (meanX * meanY);
            double slope = covariance / varX;
            double intercept = meanY - slope * meanX;

            return (double)CurrentBar * slope + intercept;
        }

        /// <summary>
        /// Normalise deviation from regression line by stdev, clamped [-5, 5].
        /// </summary>
        private double ComputeColorLevel(double regressionVal)
        {
            double deviation = Close[0] - regressionVal;
            double stdev = StdDev(Close, 200)[0];
            if (stdev < 1e-10) return 0;
            double norm = deviation / stdev;
            return Math.Max(-5.0, Math.Min(5.0, norm));
        }

        private static Color LerpColor(Color a, Color b, double t)
        {
            t = Math.Max(0, Math.Min(1, t));
            byte r = (byte)(a.R + (b.R - a.R) * t);
            byte g = (byte)(a.G + (b.G - a.G) * t);
            byte bl = (byte)(a.B + (b.B - a.B) * t);
            return Color.FromRgb(r, g, bl);
        }

        /// <summary>
        /// Map color_level to aqua→yellow→red heatmap gradient.
        /// </summary>
        private SolidColorBrush GetHeatmapBrush(double colorLevel, double maxLevel, double minLevel)
        {
            Color aqua = Color.FromRgb(0, 255, 255);
            Color yellow = Color.FromRgb(255, 255, 0);
            Color red = Color.FromRgb(255, 0, 0);
            Color result;

            if (colorLevel > 0)
            {
                double maxVal = Math.Max(maxLevel, 0.0001);
                double t = Math.Min(colorLevel / maxVal, 1.0);
                result = LerpColor(yellow, red, t);
            }
            else
            {
                double minVal = Math.Min(minLevel, -0.0001);
                double t = 1.0 - Math.Min((colorLevel - minVal) / (0 - minVal), 1.0);
                t = 1.0 - t;
                result = LerpColor(aqua, yellow, t);
            }

            return new SolidColorBrush(result);
        }

        #endregion

        // =====================================================================
        // Volume Profile computation
        // =====================================================================
        #region Volume Profile

        private void ComputeRollingVolumeProfile()
        {
            double pMin = double.MaxValue;
            double pMax = double.MinValue;

            for (int i = 0; i < VP_Lookback; i++)
            {
                if (i >= CurrentBar) break;
                if (High[i] > pMax) pMax = High[i];
                if (Low[i] < pMin) pMin = Low[i];
            }

            if (pMax <= pMin || double.IsInfinity(pMin) || double.IsInfinity(pMax))
            {
                currentPOC = Close[0];
                currentVAH = Close[0];
                currentVAL = Close[0];
                return;
            }

            double padding = (pMax - pMin) * 0.001;
            pMin -= padding;
            pMax += padding;

            int nBins = VP_Bins;
            if (vpBinVolumes == null || vpBinVolumes.Length != nBins)
                vpBinVolumes = new double[nBins];
            else
                Array.Clear(vpBinVolumes, 0, nBins);

            double binWidth = (pMax - pMin) / nBins;
            if (binWidth <= 0) { currentPOC = Close[0]; return; }

            for (int i = 0; i < VP_Lookback; i++)
            {
                if (i >= CurrentBar) break;
                double barLow = Low[i];
                double barHigh = High[i];
                double barVol = Volume[i];
                if (barVol <= 0 || barHigh <= barLow) continue;

                double barRange = barHigh - barLow;
                for (int j = 0; j < nBins; j++)
                {
                    double binLo = pMin + j * binWidth;
                    double binHi = binLo + binWidth;
                    double overlapLo = Math.Max(barLow, binLo);
                    double overlapHi = Math.Min(barHigh, binHi);
                    if (overlapHi > overlapLo)
                        vpBinVolumes[j] += barVol * ((overlapHi - overlapLo) / barRange);
                }
            }

            // POC: bin with max volume
            int pocIdx = 0;
            double maxVol = 0;
            double totalVol = 0;

            for (int j = 0; j < nBins; j++)
            {
                totalVol += vpBinVolumes[j];
                if (vpBinVolumes[j] > maxVol)
                {
                    maxVol = vpBinVolumes[j];
                    pocIdx = j;
                }
            }

            currentPOC = pMin + (pocIdx + 0.5) * binWidth;

            // Value Area: expand outward from POC
            if (totalVol > 0)
            {
                double targetVol = totalVol * (ValueAreaPct / 100.0);
                double areaVol = vpBinVolumes[pocIdx];
                int upper = pocIdx;
                int lower = pocIdx;

                while (areaVol < targetVol && (upper < nBins - 1 || lower > 0))
                {
                    double volAbove = (upper < nBins - 1) ? vpBinVolumes[upper + 1] : 0;
                    double volBelow = (lower > 0) ? vpBinVolumes[lower - 1] : 0;

                    if (volAbove >= volBelow && upper < nBins - 1)
                    {
                        upper++;
                        areaVol += vpBinVolumes[upper];
                    }
                    else if (lower > 0)
                    {
                        lower--;
                        areaVol += vpBinVolumes[lower];
                    }
                    else if (upper < nBins - 1)
                    {
                        upper++;
                        areaVol += vpBinVolumes[upper];
                    }
                    else break;
                }

                currentVAH = pMin + (upper + 1) * binWidth;
                currentVAL = pMin + lower * binWidth;
            }
            else
            {
                currentVAH = currentPOC;
                currentVAL = currentPOC;
            }

            vpPriceMin = pMin;
            vpPriceMax = pMax;
        }

        #endregion

        // =====================================================================
        // Session POC management
        // =====================================================================
        #region Session POC

        private void SaveSessionPOC(DateTime sessionDate, double poc)
        {
            if (sessionPOCs == null) return;

            foreach (var sp in sessionPOCs)
                if (sp.Date == sessionDate) return;

            string tag = "SPOC_" + sessionDate.ToString("yyyyMMdd");
            sessionPOCs.Add(new SessionPOCInfo
            {
                Price = poc,
                Date = sessionDate,
                IsNaked = true,
                Tag = tag,
            });

            while (sessionPOCs.Count > SessionPOC_MaxDays)
            {
                var oldest = sessionPOCs[0];
                RemoveDrawObject(oldest.Tag);
                sessionPOCs.RemoveAt(0);
            }
        }

        private void UpdateNakedPOCs()
        {
            if (sessionPOCs == null || sessionPOCs.Count == 0) return;

            for (int i = sessionPOCs.Count - 1; i >= 0; i--)
            {
                var sp = sessionPOCs[i];
                if (!sp.IsNaked) continue;

                if (Low[0] <= sp.Price && High[0] >= sp.Price)
                {
                    var updated = sp;
                    updated.IsNaked = false;
                    sessionPOCs[i] = updated;
                    RemoveDrawObject(sp.Tag);
                    continue;
                }

                Draw.HorizontalLine(this, sp.Tag, sp.Price, Brushes.Yellow, DashStyleHelper.Dot, 1);
            }
        }

        #endregion

        // =====================================================================
        // Main computation — runs once per bar close
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
            // SECTION A: VWAP (daily-resetting) — needs to run early for signals
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
            // SECTION D: Dynamic Trend EMA + Wave Tracking (from Core)
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
            // SECTION H: Signal Quality Score (enhanced with VWAP/POC)
            // ==================================================================
            signalQuality = 0;

            double waveRatioForDir = bullDominant ? currentWaveRatio : (1.0 / currentWaveRatio);

            // Wave strength component (0.35 + 0.05 bonus)
            if (waveRatioForDir > MinWaveRatio)
            {
                signalQuality += 0.35;
                if (waveRatioForDir > MinWaveRatio * 1.5)
                    signalQuality += 0.05;
            }

            // AO alignment (0.25)
            if ((bullDominant && aoBullish) || (!bullDominant && aoBearish))
                signalQuality += 0.25;

            // AO acceleration (0.05)
            double aoAccel = Math.Abs(aoValue) - Math.Abs(aoPrevValue);
            if (aoAccel > 0)
                signalQuality += 0.05;

            // Price vs S/R midpoint (0.15 — reduced from 0.20 to make room for VWAP/POC)
            if ((bullDominant && Close[0] > mid) || (!bullDominant && Close[0] < mid))
                signalQuality += 0.15;

            // Price vs EMA9 (0.10)
            if ((bullDominant && Close[0] > ema9[0]) || (!bullDominant && Close[0] < ema9[0]))
                signalQuality += 0.10;

            // ── NEW: VWAP proximity bonus (0.05) ──
            // Buy signals near/below VWAP are higher quality; sell signals near/above
            if (vwap > 0)
            {
                if ((bullDominant && Close[0] <= vwap) || (!bullDominant && Close[0] >= vwap))
                    signalQuality += 0.05;
            }

            // ── NEW: POC proximity bonus (0.05) ──
            // Signals near POC (high-volume price) have structural support
            if (currentPOC > 0)
            {
                double pocDistance = Math.Abs(Close[0] - currentPOC);
                double pocThreshold = (highest - lowest) * 0.05; // within 5% of range
                if (pocDistance <= pocThreshold)
                    signalQuality += 0.05;
            }

            signalQuality = Math.Min(1.0, Math.Max(0.0, signalQuality));

            // ==================================================================
            // SECTION I: Buy / Sell Signals
            // ==================================================================
            double touchThreshold = highest * 0.003;
            double waveThresholdSignal = MinWaveRatio * 0.7;

            bool nearSupport = Low[0] <= lowest + touchThreshold;
            bool nearResistance = High[0] >= highest - touchThreshold;

            bool buySignal = nearSupport
                && aoBullish
                && currentWaveRatio > waveThresholdSignal
                && signalQuality >= 0.35;

            bool sellSignal = nearResistance
                && aoBearish
                && (1.0 / Math.Max(currentWaveRatio, 0.001)) > waveThresholdSignal
                && signalQuality >= 0.35;

            if (buySignal && (Time[0] - lastBuySignalTime).TotalMinutes >= SignalCooldownMinutes)
            {
                double arrowY = Low[0] - TickSize * 3;
                Draw.ArrowUp(this, "BuyArr" + CurrentBar, true, 0, arrowY, Brushes.Lime);

                if (ShowLabels)
                {
                    string label = string.Format("BUY\nQ: {0:0}%\nW: {1:0.0}x",
                        signalQuality * 100, currentWaveRatio);
                    Draw.Text(this, "BuyLbl" + CurrentBar, label,
                        0, Low[0] - TickSize * 10, Brushes.Lime);
                }
                lastBuySignalTime = Time[0];
            }

            if (sellSignal && (Time[0] - lastSellSignalTime).TotalMinutes >= SignalCooldownMinutes)
            {
                double arrowY = High[0] + TickSize * 3;
                Draw.ArrowDown(this, "SellArr" + CurrentBar, true, 0, arrowY, Brushes.Red);

                if (ShowLabels)
                {
                    string label = string.Format("SELL\nQ: {0:0}%\nW: {1:0.0}x",
                        signalQuality * 100, currentWaveRatio);
                    Draw.Text(this, "SellLbl" + CurrentBar, label,
                        0, High[0] + TickSize * 10, Brushes.Red);
                }
                lastSellSignalTime = Time[0];
            }

            // ==================================================================
            // SECTION J: Heatmap Bar Coloring
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
            // SECTION K: Volume spike/low detection + action labels
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

            // ==================================================================
            // SECTION L: Absorption Detection (from DynamicVolume)
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
            // SECTION M: Candle Outline
            // ==================================================================
            if (ShowDeltaOutline && barRange > 0 && vol > 0)
            {
                // Delta-based outline (from DynamicVolume)
                double deltaPct = Math.Abs(delta) / vol;
                if (delta > 0)
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Cyan : Brushes.DodgerBlue;
                else
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Magenta : Brushes.MediumOrchid;
            }
            else
            {
                // DynEMA-based outline (from Core — default)
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
        // On-chart info boxes
        // =====================================================================
        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);

            if (CurrentBar < 50) return;

            // ── Top-right: Core info (wave, quality, AO, weather) ────────
            RenderCoreInfoBox();

            // ── Top-left: Volume info (VWAP, POC, CVD, delta) ────────────
            RenderVolumeInfoBox();
        }

        private void RenderCoreInfoBox()
        {
            bool bullDominant = currentWaveRatio >= 1.0;
            string biasText = bullDominant ? "BULL" : "BEAR";
            string biasEmoji = bullDominant ? "▲" : "▼";

            int qualityPct = (int)(signalQuality * 100);
            string qualityTag = qualityPct >= 70 ? "HIGH" : qualityPct >= 50 ? "MED" : "LOW";

            string aoDir = aoValue > 0 ? "+" : "";
            double displayRatio = bullDominant ? currentWaveRatio : (1.0 / Math.Max(currentWaveRatio, 0.001));

            double cl = colorLevelSeries != null && colorLevelSeries.IsValidDataPoint(0) ? colorLevelSeries[0] : 0;
            string weatherTag;
            if (cl <= -1.0) weatherTag = "FREEZING";
            else if (cl <= -0.66) weatherTag = "FROZEN";
            else if (cl <= -0.33) weatherTag = "COLD";
            else if (cl < 0.33) weatherTag = "NEUTRAL";
            else if (cl < 0.66) weatherTag = "WARM";
            else if (cl < 1.0) weatherTag = "HOT";
            else weatherTag = "BURNING";

            double volAvg = volSMA != null ? volSMA[0] : 0;
            string volStatus;
            if (volAvg > 0 && Volume[0] > volAvg * VolumeSpikeMult)
                volStatus = "SPIKE ▲▲";
            else if (lowVolStreak >= LowVolStreakBars)
                volStatus = string.Format("LOW x{0} bars", lowVolStreak);
            else if (volAvg > 0 && Volume[0] < volAvg * VolumeLowMult)
                volStatus = "THIN";
            else
                volStatus = "NORMAL";

            string infoText = string.Format(
                "Ruby {0}\n" +
                "━━━━━━━━━━━━━━━━\n" +
                "Wave Ratio:    {1:0.00}x {2}\n" +
                "Signal Quality: {3}% ({4})\n" +
                "AO:            {5}{6:0.00}\n" +
                "DynEMA:        {7:0.00}\n" +
                "Weather:       {8}\n" +
                "Bulls: {9}  Bears: {10}\n" +
                "Volume:        {11}",
                biasEmoji + " " + biasText,
                displayRatio, biasEmoji,
                qualityPct, qualityTag,
                aoDir, aoValue,
                dynEMA,
                weatherTag,
                bullWaves != null ? bullWaves.Count : 0,
                bearWaves != null ? bearWaves.Count : 0,
                volStatus
            );

            Draw.TextFixed(this, "ruby_core_info",
                infoText,
                TextPosition.TopRight,
                Brushes.White,
                new SimpleFont("Consolas", 11),
                Brushes.Transparent,
                Brushes.Black,
                10);
        }

        private void RenderVolumeInfoBox()
        {
            double barRange = High[0] - Low[0];
            double vol = Volume[0];
            double delta = 0;
            string deltaDir = "---";

            if (barRange > 0 && vol > 0)
            {
                double buyPct = (Close[0] - Low[0]) / barRange;
                delta = vol * (2.0 * buyPct - 1.0);
                deltaDir = delta > 0 ? "▲ BUY" : "▼ SELL";
            }

            string cvdSlopeStr = delta > 0 ? "Rising ▲" : delta < 0 ? "Falling ▼" : "Flat ---";

            double volAvg = (CurrentBar >= VolumeAvgPeriod && volSMA != null) ? volSMA[0] : 0;
            string volStatus = "Normal";
            if (volAvg > 0)
            {
                double volRatio = vol / volAvg;
                if (volRatio > VolumeSpikeMult)
                    volStatus = string.Format("SPIKE ({0:0.0}x)", volRatio);
                else if (volRatio < VolumeLowMult)
                    volStatus = string.Format("LOW ({0:0.0}x)", volRatio);
                else
                    volStatus = string.Format("{0:0.0}x", volRatio);
            }

            int nakedCount = 0;
            if (sessionPOCs != null)
                foreach (var sp in sessionPOCs)
                    if (sp.IsNaked) nakedCount++;

            string vaPosition;
            double price = Close[0];
            if (currentVAH > 0 && currentVAL > 0 && currentVAH != currentVAL)
            {
                if (price > currentVAH) vaPosition = "ABOVE VA";
                else if (price < currentVAL) vaPosition = "BELOW VA";
                else vaPosition = "INSIDE VA";
            }
            else vaPosition = "---";

            string infoText = string.Format(
                "Ruby Volume\n" +
                "━━━━━━━━━━━━━━━━━━━━\n" +
                "VWAP:       {0:0.00}\n" +
                "POC:        {1:0.00}\n" +
                "VAH:        {2:0.00}\n" +
                "VAL:        {3:0.00}\n" +
                "Position:   {4}\n" +
                "━━━━━━━━━━━━━━━━━━━━\n" +
                "Delta:      {5:+0;-0;0} {6}\n" +
                "CVD:        {7:+0;-0;0}\n" +
                "CVD Trend:  {8}\n" +
                "Volume:     {9}\n" +
                "Naked POCs: {10}",
                VWAP_Line.IsValidDataPoint(0) ? VWAP_Line[0] : 0,
                currentPOC, currentVAH, currentVAL,
                vaPosition,
                delta, deltaDir,
                cvdAccumulator, cvdSlopeStr,
                volStatus, nakedCount
            );

            Draw.TextFixed(this, "ruby_vol_info",
                infoText,
                TextPosition.TopLeft,
                Brushes.White,
                new SimpleFont("Consolas", 10),
                Brushes.Transparent,
                Brushes.Black,
                10);
        }
    }
}
