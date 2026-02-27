#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows.Media;
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
// FKS_Core — Futures Kill-Shot Core Indicator (TASK-401)
// =============================================================================
//
// Mirrors the Python FKS analysis on-chart in NinjaTrader 8.
//
// Components:
//   1. EMA(9) — blue trend line (overlay)
//   2. Bollinger Bands (2σ, 20) — Upper=Red, Mid=Magenta, Lower=LimeGreen
//   3. Dynamic Trend EMA — wave tracking for bull/bear dominance
//   4. Wave Dominance — ratio of bull vs bear wave strengths
//   5. Adaptive S/R — MAX(High,N) / MIN(Low,N) / midpoint
//   6. Awesome Oscillator (5,34) — momentum confirmation
//   7. Signal Quality (0–100%) — composite score
//   8. Buy/Sell arrows with quality labels (5-min cooldown)
//   9. Volume bar coloring — green/lime(spike) bullish, red/orange(spike) bearish
//  10. Top-right info box — wave ratio, quality %, AO value
//  11. Candle outline heatmap — lime above dynEMA, red below
//  12. Dynamic Volume Analysis (TASK-403):
//        TP/BE  — volume spike at Bollinger Band → protect/take profit
//        ADD    — volume spike with trend (dynEMA + AO) → add to position
//        LOW VOL — volume < 0.5× avg for 3+ bars → thin market warning
//
// Installation:
//   1. NinjaTrader 8 → New → NinjaScript Editor → Indicators
//   2. Paste this file → Compile (F5)
//   3. Drag FKS_Core onto any chart (MGC, MES, MNQ, MCL, etc.)
//
// Parameters exposed via Properties panel:
//   SR_Lookback, AO_Fast, AO_Slow, WaveLookback, MinWaveRatio,
//   ShowLabels, HeatSensitivity, SignalCooldownMinutes,
//   VolumeSpikeMult, VolumeLowMult, ShowVolumeLabels, LowVolStreakBars
// =============================================================================

namespace NinjaTrader.NinjaScript.Indicators
{
    public class FKS_Core : Indicator
    {
        // ----- Internal indicator references -----
        private EMA ema9;
        private Bollinger bb;

        // ----- Wave tracking state -----
        private List<double> bullWaves = new List<double>();
        private List<double> bearWaves = new List<double>();
        private double dynEMA;
        private double trendSpeed;
        private double currentWaveRatio;
        private double signalQuality;
        private bool inBullPhase;

        // ----- Volume analysis state (TASK-403) -----
        private int lowVolStreak;

        // ----- Signal cooldown -----
        private DateTime lastBuySignalTime = DateTime.MinValue;
        private DateTime lastSellSignalTime = DateTime.MinValue;

        // ----- Volume tracking -----
        private SMA volSMA;

        // ----- Cached AO values (manual calculation) -----
        private SMA aoFastSMA;
        private SMA aoSlowSMA;
        private double aoValue;
        private double aoPrevValue;

        // =====================================================================
        // State management
        // =====================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "FKS Core — Trend Speed, Wave Dominance, Adaptive S/R, AO Signals, Colored Volume";
                Name = "FKS_Core";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                IsSuspendedWhileInactive = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                PaintPriceMarkers = false;
                ScaleJustification = ScaleJustification.Right;
                BarsRequiredToPlot = 220;

                // ---- Plot definitions (overlay) ----
                // Index 0: Resistance (adaptive high)
                AddPlot(new Stroke(Brushes.Red, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "Resistance");
                // Index 1: MidBand (adaptive midpoint)
                AddPlot(new Stroke(Brushes.Magenta, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "MidBand");
                // Index 2: Support (adaptive low)
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "Support");
                // Index 3: EMA9
                AddPlot(new Stroke(Brushes.DodgerBlue, 2),
                    PlotStyle.Line, "EMA9");
                // Index 4: Bollinger Upper
                AddPlot(new Stroke(Brushes.Red, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Upper");
                // Index 5: Bollinger Middle
                AddPlot(new Stroke(Brushes.Magenta, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Mid");
                // Index 6: Bollinger Lower
                AddPlot(new Stroke(Brushes.LimeGreen, DashStyleHelper.Solid, 1),
                    PlotStyle.Line, "BB_Lower");

                // ---- Default parameters ----
                SR_Lookback = 20;
                AO_Fast = 5;
                AO_Slow = 34;
                WaveLookback = 200;
                MinWaveRatio = 1.5;
                ShowLabels = true;
                HeatSensitivity = 70;
                SignalCooldownMinutes = 5;
                VolumeSpikeMult = 1.8;
                VolumeLowMult = 0.5;
                ShowVolumeLabels = true;
                LowVolStreakBars = 3;
            }
            else if (State == State.DataLoaded)
            {
                ema9 = EMA(Close, 9);
                bb = Bollinger(Close, 2, 20);

                // Volume average for spike detection
                volSMA = SMA(Volume, 20);

                // Awesome Oscillator: SMA(median, fast) - SMA(median, slow)
                // NT8 doesn't have a built-in AO, so we compute it manually
                // using SMA of the median price (H+L)/2
                aoFastSMA = SMA(Typical, AO_Fast);
                aoSlowSMA = SMA(Typical, AO_Slow);

                // Initialize dynamic EMA
                dynEMA = 0;
                trendSpeed = 0;
                inBullPhase = true;
                lowVolStreak = 0;
            }
            else if (State == State.Terminated)
            {
                bullWaves = null;
                bearWaves = null;
            }
        }

        // =====================================================================
        // Parameters
        // =====================================================================
        #region Properties

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "S/R Lookback", Description = "Bars for adaptive support/resistance",
            GroupName = "1. Core Parameters", Order = 1)]
        public int SR_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(2, 20)]
        [Display(Name = "AO Fast Period", Description = "Awesome Oscillator fast SMA period",
            GroupName = "1. Core Parameters", Order = 2)]
        public int AO_Fast { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "AO Slow Period", Description = "Awesome Oscillator slow SMA period",
            GroupName = "1. Core Parameters", Order = 3)]
        public int AO_Slow { get; set; }

        [NinjaScriptProperty]
        [Range(50, 500)]
        [Display(Name = "Wave Lookback", Description = "Number of waves to keep for ratio calculation",
            GroupName = "1. Core Parameters", Order = 4)]
        public int WaveLookback { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Min Wave Ratio", Description = "Minimum wave ratio for signal generation",
            GroupName = "1. Core Parameters", Order = 5)]
        public double MinWaveRatio { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Entry Labels", Description = "Show BUY/SELL text labels with quality info",
            GroupName = "2. Visual", Order = 1)]
        public bool ShowLabels { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "Heat Sensitivity", Description = "Sensitivity of the candle outline heatmap (higher = more sensitive)",
            GroupName = "2. Visual", Order = 2)]
        public int HeatSensitivity { get; set; }

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Signal Cooldown (min)", Description = "Minimum minutes between buy/sell signals",
            GroupName = "2. Visual", Order = 3)]
        public int SignalCooldownMinutes { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 5.0)]
        [Display(Name = "Volume Spike Multiplier", Description = "Volume > avg × this = spike",
            GroupName = "3. Volume", Order = 1)]
        public double VolumeSpikeMult { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 1.0)]
        [Display(Name = "Low Volume Multiplier", Description = "Volume < avg × this = low volume",
            GroupName = "3. Volume", Order = 2)]
        public double VolumeLowMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Volume Labels", Description = "Show TP/BE, ADD, LOW VOL action labels based on volume patterns",
            GroupName = "3. Volume", Order = 3)]
        public bool ShowVolumeLabels { get; set; }

        [NinjaScriptProperty]
        [Range(2, 10)]
        [Display(Name = "Low Vol Streak Bars", Description = "Consecutive low-volume bars required to show LOW VOL warning",
            GroupName = "3. Volume", Order = 4)]
        public int LowVolStreakBars { get; set; }

        #endregion

        // =====================================================================
        // Plot accessors
        // =====================================================================
        #region Plot Accessors

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> Resistance => Values[0];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> MidBand => Values[1];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> Support => Values[2];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> EMA9 => Values[3];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> BB_Upper => Values[4];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> BB_Mid => Values[5];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> BB_Lower => Values[6];

        #endregion

        // =====================================================================
        // Main computation — runs once per bar close
        // =====================================================================
        protected override void OnBarUpdate()
        {
            if (CurrentBar < Math.Max(AO_Slow + 5, Math.Max(200, WaveLookback)))
                return;

            // ==================================================================
            // 1. Dynamic Trend EMA (mirrors Python FKS trend speed)
            // ==================================================================
            double alpha = 2.0 / (20.0 + 1.0);
            if (CurrentBar <= 200)
                dynEMA = Close[0];
            else
                dynEMA = alpha * Close[0] + (1.0 - alpha) * dynEMA;

            // Accumulate trend speed (sum of bar ranges in current wave phase)
            double barContribution = Close[0] - Open[0];
            trendSpeed += barContribution;

            // Wave detection: track transitions across dynamic EMA
            bool aboveDyn = Close[0] > dynEMA;
            bool prevAboveDyn = Close[1] > dynEMA;

            // Bull → Bear transition: price crosses below dynEMA
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

            // Bear → Bull transition: price crosses above dynEMA
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

            // Wave ratio: average bull wave strength / average bear wave strength
            double bullAvg = bullWaves.Count > 0 ? bullWaves.Average() : 0.001;
            double bearAvg = bearWaves.Count > 0 ? bearWaves.Average() : 0.001;

            // Protect against division by zero
            if (bearAvg < 0.0001) bearAvg = 0.0001;
            if (bullAvg < 0.0001) bullAvg = 0.0001;

            currentWaveRatio = bullAvg / bearAvg;

            // Determine wave dominance direction
            bool bullDominant = currentWaveRatio >= 1.0;
            string dominanceDir = bullDominant ? "BULL" : "BEAR";

            // ==================================================================
            // 2. Adaptive Support / Resistance
            // ==================================================================
            double highest = MAX(High, SR_Lookback)[0];
            double lowest = MIN(Low, SR_Lookback)[0];
            double mid = (highest + lowest) / 2.0;

            Resistance[0] = highest;
            MidBand[0] = mid;
            Support[0] = lowest;

            // ==================================================================
            // 3. EMA9 + Bollinger Bands
            // ==================================================================
            EMA9[0] = ema9[0];
            BB_Upper[0] = bb.Upper[0];
            BB_Mid[0] = bb.Middle[0];
            BB_Lower[0] = bb.Lower[0];

            // ==================================================================
            // 4. Awesome Oscillator (manual: SMA(median,fast) - SMA(median,slow))
            // ==================================================================
            aoPrevValue = aoValue;
            aoValue = aoFastSMA[0] - aoSlowSMA[0];

            bool aoBullish = aoValue > 0 && aoValue > aoPrevValue;
            bool aoBearish = aoValue < 0 && aoValue < aoPrevValue;

            // ==================================================================
            // 5. Signal Quality Score (0.0 – 1.0)
            // ==================================================================
            //   - Wave ratio contribution:  +0.35 if ratio > MinWaveRatio
            //   - Wave ratio bonus:         +0.05 if ratio > MinWaveRatio * 1.5
            //   - AO confirmation:          +0.25 if AO matches direction
            //   - AO momentum:              +0.05 if AO is accelerating
            //   - Price vs midpoint:         +0.20 if price is on the "right side"
            //   - Price vs EMA9:            +0.10 if price confirms EMA trend
            // ==================================================================
            signalQuality = 0;

            double waveRatioForDir = bullDominant ? currentWaveRatio : (1.0 / currentWaveRatio);

            // Wave ratio contribution
            if (waveRatioForDir > MinWaveRatio)
            {
                signalQuality += 0.35;
                if (waveRatioForDir > MinWaveRatio * 1.5)
                    signalQuality += 0.05;
            }

            // AO confirmation
            if ((bullDominant && aoBullish) || (!bullDominant && aoBearish))
                signalQuality += 0.25;

            // AO momentum (accelerating)
            double aoAccel = Math.Abs(aoValue) - Math.Abs(aoPrevValue);
            if (aoAccel > 0)
                signalQuality += 0.05;

            // Price vs adaptive midpoint
            if ((bullDominant && Close[0] > mid) || (!bullDominant && Close[0] < mid))
                signalQuality += 0.20;

            // Price vs EMA9
            if ((bullDominant && Close[0] > ema9[0]) || (!bullDominant && Close[0] < ema9[0]))
                signalQuality += 0.10;

            signalQuality = Math.Min(1.0, Math.Max(0.0, signalQuality));

            // ==================================================================
            // 6. Buy / Sell Signals (arrows + labels)
            // ==================================================================
            // Buy: Low touches support zone + AO bullish + wave ratio strong enough
            // Sell: High touches resistance zone + AO bearish + wave ratio strong enough
            //
            // "Touches" means within 0.3% of the level to account for wicks
            // ==================================================================
            double touchThreshold = highest * 0.003;  // 0.3% proximity
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

            // Buy signal with cooldown
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

            // Sell signal with cooldown
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
            // 7. Volume-Colored Bars
            // ==================================================================
            bool bullishBar = Close[0] >= Close[1];
            double volAvg = volSMA[0];
            bool volSpike = volAvg > 0 && Volume[0] > volAvg * VolumeSpikeMult;
            bool volLow = volAvg > 0 && Volume[0] < volAvg * VolumeLowMult;

            if (bullishBar)
            {
                if (volSpike)
                    BarBrush = Brushes.Lime;          // Bullish spike — bright green
                else
                    BarBrush = Brushes.Green;         // Normal bullish — green
            }
            else
            {
                if (volSpike)
                    BarBrush = Brushes.OrangeRed;     // Bearish spike — orange
                else
                    BarBrush = Brushes.Red;           // Normal bearish — red
            }

            // Low volume warning: dim the bars
            if (volLow)
            {
                if (bullishBar)
                    BarBrush = Brushes.DarkGreen;
                else
                    BarBrush = Brushes.DarkRed;
            }

            // ==================================================================
            // 7b. Dynamic Volume Analysis Labels (TASK-403)
            // ==================================================================
            // TP/BE:   volume spike at upper BB → likely exhaustion, protect profits
            // ADD:     volume spike with trend confirmation → momentum, add to position
            // LOW VOL: volume dried up for N+ bars → caution, thin market
            // ==================================================================
            if (ShowVolumeLabels && volAvg > 0)
            {
                // Track consecutive low-volume bars
                if (volLow)
                    lowVolStreak++;
                else
                    lowVolStreak = 0;

                // Proximity threshold for "at Bollinger Band" (within 0.15% of band)
                double bbProximity = (bb.Upper[0] - bb.Lower[0]) * 0.05;

                bool atUpperBB = High[0] >= bb.Upper[0] - bbProximity;
                bool atLowerBB = Low[0] <= bb.Lower[0] + bbProximity;
                bool trendingUp = Close[0] > dynEMA && aoBullish;
                bool trendingDown = Close[0] < dynEMA && aoBearish;

                if (volSpike && (atUpperBB || atLowerBB))
                {
                    // Volume spike at a Bollinger Band → exhaustion signal
                    // Place label on the opposite side of the band being touched
                    if (atUpperBB)
                    {
                        Draw.Text(this, "VolTP" + CurrentBar, "TP/BE",
                            0, High[0] + TickSize * 6,
                            Brushes.Yellow);
                    }
                    else // atLowerBB
                    {
                        Draw.Text(this, "VolTP" + CurrentBar, "TP/BE",
                            0, Low[0] - TickSize * 6,
                            Brushes.Yellow);
                    }
                }
                else if (volSpike && (trendingUp || trendingDown))
                {
                    // Volume spike with trend confirmation → add to position
                    if (trendingUp)
                    {
                        Draw.Text(this, "VolADD" + CurrentBar, "ADD",
                            0, Low[0] - TickSize * 6,
                            Brushes.Cyan);
                    }
                    else // trendingDown
                    {
                        Draw.Text(this, "VolADD" + CurrentBar, "ADD",
                            0, High[0] + TickSize * 6,
                            Brushes.Cyan);
                    }
                }

                // Low volume streak warning
                if (lowVolStreak >= LowVolStreakBars)
                {
                    Draw.Text(this, "VolLow" + CurrentBar, "LOW VOL",
                        0, Low[0] - TickSize * 4,
                        Brushes.Gray);
                }
            }

            // ==================================================================
            // 8. Candle Outline Heatmap
            // ==================================================================
            // Colour the candle outline based on position relative to dynEMA.
            // The heat sensitivity parameter controls how quickly the colour
            // shifts — higher values make it more responsive to small deviations.
            // ==================================================================
            double deviation = Close[0] - dynEMA;

            // Normalise by recent standard deviation for consistent visual across instruments
            double stdev = StdDev(Close, 200)[0];
            if (stdev < 0.0001) stdev = 0.0001;

            double normDev = (deviation / stdev) * (HeatSensitivity / 100.0);

            if (normDev > 0)
                CandleOutlineBrush = Brushes.Lime;
            else
                CandleOutlineBrush = Brushes.Red;

            // Strong deviation: thicker outline via wick brush too
            if (Math.Abs(normDev) > 1.5)
            {
                CandleOutlineBrush = normDev > 0 ? Brushes.Cyan : Brushes.OrangeRed;
            }
        }

        // =====================================================================
        // On-chart info box (top-right corner)
        // =====================================================================
        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);

            if (CurrentBar < 50) return;

            // Determine bias text
            bool bullDominant = currentWaveRatio >= 1.0;
            string biasText = bullDominant ? "BULL" : "BEAR";
            string biasEmoji = bullDominant ? "▲" : "▼";

            // Quality category
            int qualityPct = (int)(signalQuality * 100);
            string qualityTag;
            if (qualityPct >= 70)
                qualityTag = "HIGH";
            else if (qualityPct >= 50)
                qualityTag = "MED";
            else
                qualityTag = "LOW";

            // AO direction
            string aoDir = aoValue > 0 ? "+" : "";

            // Effective wave ratio for display (always > 1 showing dominant side)
            double displayRatio = bullDominant ? currentWaveRatio : (1.0 / Math.Max(currentWaveRatio, 0.001));

            string infoText = string.Format(
                "FKS Core {0}\n" +
                "━━━━━━━━━━━━━━━━\n" +
                "Wave Ratio:    {1:0.00}x {2}\n" +
                "Signal Quality: {3}% ({4})\n" +
                "AO:            {5}{6:0.00}\n" +
                "DynEMA:        {7:0.00}\n" +
                "Bulls: {8}  Bears: {9}",
                biasEmoji + " " + biasText,
                displayRatio,
                biasEmoji,
                qualityPct,
                qualityTag,
                aoDir,
                aoValue,
                dynEMA,
                bullWaves != null ? bullWaves.Count : 0,
                bearWaves != null ? bearWaves.Count : 0
            );

            // Volume status line (TASK-403)
            string volStatus;
            if (volSMA != null && volSMA[0] > 0 && Volume[0] > volSMA[0] * VolumeSpikeMult)
                volStatus = "SPIKE ▲▲";
            else if (lowVolStreak >= LowVolStreakBars)
                volStatus = string.Format("LOW ×{0} bars", lowVolStreak);
            else if (volSMA != null && volSMA[0] > 0 && Volume[0] < volSMA[0] * VolumeLowMult)
                volStatus = "THIN";
            else
                volStatus = "NORMAL";

            infoText += string.Format("\nVolume:        {0}", volStatus);

            Draw.TextFixed(this, "fks_core_info",
                infoText,
                TextPosition.TopRight,
                Brushes.White,
                new SimpleFont("Consolas", 11),
                Brushes.Transparent,
                Brushes.Black,
                10);
        }
    }
}
