#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows;
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
// FKS_DynamicVolume — Dynamic Volume Analysis Indicator (TASK-403)
// =============================================================================
//
// Mirrors the Python volume_profile.py + cvd.py analysis on-chart in NT8.
//
// Components:
//   1. Intraday VWAP — daily-resetting volume-weighted average price (overlay)
//   2. VWAP Standard Deviation Bands — +/-1 and +/-2 sigma (OFF by default)
//   3. Rolling Volume Profile — POC / VAH / VAL computed over a lookback window
//   4. Session POC Tracking — prior-session POC levels drawn as horizontal rays
//      ("Naked POCs") that persist until price trades through them
//   5. CVD Approximation — Cumulative Volume Delta from OHLCV heuristic
//   6. Volume Spike / Absorption Detection — diamonds on absorption candles
//   7. On-chart info box — POC, VAH, VAL, VWAP, CVD slope, delta
//
// Visibility defaults (clean chart — only key levels shown):
//   - VWAP line: ON
//   - VWAP bands (+/-1, +/-2 sigma): OFF by default
//   - POC: ON
//   - VAH/VAL lines: OFF by default (values still shown in info box)
//   - Naked POCs: ON
//   - Absorption diamonds: ON
//   - Bar coloring: OFF (defers to FKS_Core heatmap when both loaded)
//
// Companion indicator: FKS_Core.cs (wave/trend analysis + heatmap bars)
// Python equivalents: volume_profile.py, cvd.py
// =============================================================================

namespace NinjaTrader.NinjaScript.Indicators
{
    public class FKS_DynamicVolume : Indicator
    {
        // =====================================================================
        // Internal state — VWAP
        // =====================================================================
        private double cumTypicalVol;
        private double cumVolume;
        private double cumTypicalVolSq;
        private DateTime lastSessionDate;

        // =====================================================================
        // Internal state — Rolling Volume Profile
        // =====================================================================
        private double[] vpBinVolumes;
        private double vpPriceMin;
        private double vpPriceMax;
        private double currentPOC;
        private double currentVAH;
        private double currentVAL;

        // =====================================================================
        // Internal state — Session POC tracking (naked POCs)
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
        // Internal state — CVD
        // =====================================================================
        private double cvdAccumulator;
        private DateTime cvdAnchorDate;

        // =====================================================================
        // Internal state — Volume average
        // =====================================================================
        private SMA volSMA;

        // =====================================================================
        // State management
        // =====================================================================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "FKS Dynamic Volume — VWAP, Volume Profile (POC/VAH/VAL), CVD, Absorption Detection";
                Name = "FKS_DynamicVolume";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                IsSuspendedWhileInactive = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                PaintPriceMarkers = false;
                ScaleJustification = ScaleJustification.Right;
                BarsRequiredToPlot = 50;

                // ----------------------------------------------------------
                // Overlay plots (price panel)
                // ----------------------------------------------------------
                // 0: VWAP
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Solid, 2),
                    PlotStyle.Line, "VWAP");
                // 1: VWAP +1 sigma — hidden by default
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "VWAP_Upper1");
                // 2: VWAP -1 sigma — hidden by default
                AddPlot(new Stroke(Brushes.Gold, DashStyleHelper.Dot, 1),
                    PlotStyle.Line, "VWAP_Lower1");
                // 3: VWAP +2 sigma — hidden by default
                AddPlot(new Stroke(Brushes.DarkGoldenrod, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VWAP_Upper2");
                // 4: VWAP -2 sigma — hidden by default
                AddPlot(new Stroke(Brushes.DarkGoldenrod, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VWAP_Lower2");
                // 5: Rolling POC
                AddPlot(new Stroke(Brushes.Cyan, DashStyleHelper.Solid, 2),
                    PlotStyle.Line, "POC");
                // 6: Value Area High — hidden by default
                AddPlot(new Stroke(Brushes.DodgerBlue, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VAH");
                // 7: Value Area Low — hidden by default
                AddPlot(new Stroke(Brushes.DodgerBlue, DashStyleHelper.Dash, 1),
                    PlotStyle.Line, "VAL");

                // ----------------------------------------------------------
                // Default parameter values (clean chart defaults)
                // ----------------------------------------------------------
                VP_Lookback = 100;
                VP_Bins = 40;
                ValueAreaPct = 70;
                SessionPOC_MaxDays = 5;
                NakedPOC_Enabled = true;
                VWAP_ShowBands = false;       // OFF by default — reduces clutter
                ShowValueArea = false;        // OFF by default — VAH/VAL hidden
                ShowDeltaOutline = false;     // OFF by default — let FKS_Core own bar visuals
                CVD_AnchorDaily = true;
                AbsorptionBodyRatio = 30;
                AbsorptionVolMult = 150;
                VolSpikeMultiplier = 180;
                VolumeAvgPeriod = 20;
            }
            else if (State == State.DataLoaded)
            {
                volSMA = SMA(Volume, VolumeAvgPeriod);

                // Initialize VWAP accumulators
                cumTypicalVol = 0;
                cumVolume = 0;
                cumTypicalVolSq = 0;
                lastSessionDate = DateTime.MinValue;

                // Initialize VP bins
                vpBinVolumes = new double[VP_Bins];
                currentPOC = 0;
                currentVAH = 0;
                currentVAL = 0;

                // Initialize session POC tracker
                sessionPOCs = new List<SessionPOCInfo>();
                prevSessionPOC = 0;
                prevSessionDate = DateTime.MinValue;

                // Initialize CVD
                cvdAccumulator = 0;
                cvdAnchorDate = DateTime.MinValue;

                // Hide plots that are off by default
                if (!VWAP_ShowBands)
                {
                    Plots[1].Brush = Brushes.Transparent;  // VWAP_Upper1
                    Plots[2].Brush = Brushes.Transparent;  // VWAP_Lower1
                    Plots[3].Brush = Brushes.Transparent;  // VWAP_Upper2
                    Plots[4].Brush = Brushes.Transparent;  // VWAP_Lower2
                }
                if (!ShowValueArea)
                {
                    Plots[6].Brush = Brushes.Transparent;  // VAH
                    Plots[7].Brush = Brushes.Transparent;  // VAL
                }
            }
            else if (State == State.Terminated)
            {
                vpBinVolumes = null;
                sessionPOCs = null;
            }
        }

        // =====================================================================
        // Parameters
        // =====================================================================
        #region Properties

        // --- Volume Profile ---
        [NinjaScriptProperty]
        [Range(20, 500)]
        [Display(Name = "VP Lookback (bars)", Description = "Number of bars for rolling volume profile",
            GroupName = "1. Volume Profile", Order = 1)]
        public int VP_Lookback { get; set; }

        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "VP Bins", Description = "Number of price bins for volume distribution",
            GroupName = "1. Volume Profile", Order = 2)]
        public int VP_Bins { get; set; }

        [NinjaScriptProperty]
        [Range(50, 90)]
        [Display(Name = "Value Area %", Description = "Percentage of volume for value area (e.g. 70)",
            GroupName = "1. Volume Profile", Order = 3)]
        public int ValueAreaPct { get; set; }

        // --- Session POC / Naked POC ---
        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Session POC Max Days", Description = "Number of prior sessions to track POC levels",
            GroupName = "2. Session POC", Order = 1)]
        public int SessionPOC_MaxDays { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Naked POCs", Description = "Draw prior-session POC levels that price hasn't revisited",
            GroupName = "2. Session POC", Order = 2)]
        public bool NakedPOC_Enabled { get; set; }

        // --- VWAP ---
        [NinjaScriptProperty]
        [Display(Name = "Show VWAP Bands", Description = "Show +/-1 and +/-2 sigma bands around VWAP (OFF by default to reduce clutter)",
            GroupName = "3. VWAP", Order = 1)]
        public bool VWAP_ShowBands { get; set; }

        // --- Visibility ---
        [NinjaScriptProperty]
        [Display(Name = "Show Value Area Lines", Description = "Show VAH/VAL lines on chart (OFF by default; values still in info box)",
            GroupName = "3. VWAP", Order = 2)]
        public bool ShowValueArea { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Show Delta Outline", Description = "Color candle outline by volume delta (OFF by default to not conflict with FKS_Core)",
            GroupName = "3. VWAP", Order = 3)]
        public bool ShowDeltaOutline { get; set; }

        // --- CVD ---
        [NinjaScriptProperty]
        [Display(Name = "Anchor CVD Daily", Description = "Reset CVD accumulation at market open each day",
            GroupName = "4. CVD", Order = 1)]
        public bool CVD_AnchorDaily { get; set; }

        // --- Volume Detection ---
        [NinjaScriptProperty]
        [Range(10, 50)]
        [Display(Name = "Absorption Body Ratio %", Description = "Max body/range ratio for absorption candle (e.g. 30 = 30%)",
            GroupName = "5. Volume Detection", Order = 1)]
        public int AbsorptionBodyRatio { get; set; }

        [NinjaScriptProperty]
        [Range(100, 500)]
        [Display(Name = "Absorption Vol Mult %", Description = "Min volume as % of average for absorption (e.g. 150 = 1.5x)",
            GroupName = "5. Volume Detection", Order = 2)]
        public int AbsorptionVolMult { get; set; }

        [NinjaScriptProperty]
        [Range(100, 500)]
        [Display(Name = "Volume Spike Mult %", Description = "Volume > avg x this% = spike highlight (e.g. 180 = 1.8x)",
            GroupName = "5. Volume Detection", Order = 3)]
        public int VolSpikeMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(5, 100)]
        [Display(Name = "Volume Avg Period", Description = "SMA period for average volume calculation",
            GroupName = "5. Volume Detection", Order = 4)]
        public int VolumeAvgPeriod { get; set; }

        #endregion

        // =====================================================================
        // Plot accessors
        // =====================================================================
        #region Plot Accessors

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VWAP_Line => Values[0];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VWAP_Upper1 => Values[1];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VWAP_Lower1 => Values[2];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VWAP_Upper2 => Values[3];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VWAP_Lower2 => Values[4];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> POC_Line => Values[5];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VAH_Line => Values[6];

        [Browsable(false)]
        [XmlIgnore]
        public Series<double> VAL_Line => Values[7];

        #endregion

        // =====================================================================
        // Main computation — runs once per bar close
        // =====================================================================
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 2)
                return;

            // ==================================================================
            // 1. Intraday VWAP (daily-resetting)
            // ==================================================================
            DateTime barDate = Time[0].Date;

            if (barDate != lastSessionDate)
            {
                if (lastSessionDate != DateTime.MinValue && currentPOC > 0)
                {
                    SaveSessionPOC(lastSessionDate, currentPOC);
                }

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
            double vol = Volume[0];

            if (vol > 0)
            {
                cumTypicalVol += typical * vol;
                cumVolume += vol;
                cumTypicalVolSq += typical * typical * vol;
            }

            double vwap = cumVolume > 0 ? cumTypicalVol / cumVolume : Close[0];
            VWAP_Line[0] = vwap;

            // VWAP standard deviation bands (computed even if hidden, for data access)
            if (cumVolume > 0)
            {
                double variance = (cumTypicalVolSq / cumVolume) - (vwap * vwap);
                double stddev = variance > 0 ? Math.Sqrt(variance) : 0;

                VWAP_Upper1[0] = vwap + stddev;
                VWAP_Lower1[0] = vwap - stddev;
                VWAP_Upper2[0] = vwap + 2.0 * stddev;
                VWAP_Lower2[0] = vwap - 2.0 * stddev;
            }
            else
            {
                VWAP_Upper1[0] = vwap;
                VWAP_Lower1[0] = vwap;
                VWAP_Upper2[0] = vwap;
                VWAP_Lower2[0] = vwap;
            }

            // ==================================================================
            // 2. Rolling Volume Profile (POC / VAH / VAL)
            // ==================================================================
            if (CurrentBar >= VP_Lookback)
            {
                ComputeRollingVolumeProfile();
            }

            POC_Line[0] = currentPOC > 0 ? currentPOC : Close[0];
            VAH_Line[0] = currentVAH > 0 ? currentVAH : Close[0];
            VAL_Line[0] = currentVAL > 0 ? currentVAL : Close[0];

            // ==================================================================
            // 3. Naked POC management
            // ==================================================================
            if (NakedPOC_Enabled && sessionPOCs != null)
            {
                UpdateNakedPOCs();
            }

            // ==================================================================
            // 4. CVD — Cumulative Volume Delta (OHLCV heuristic)
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
            // 5. Volume Spike & Absorption Detection
            // ==================================================================
            if (CurrentBar >= VolumeAvgPeriod + 1)
            {
                double volAvg = volSMA[0];
                double spikeThreshold = volAvg * (VolSpikeMultiplier / 100.0);
                double absorptionVolThreshold = volAvg * (AbsorptionVolMult / 100.0);

                bool isSpike = vol > spikeThreshold && volAvg > 0;

                // Absorption candle detection:
                //   high volume + small body relative to range
                double body = Math.Abs(Close[0] - Open[0]);
                double bodyRatio = barRange > 0 ? (body / barRange) * 100.0 : 100.0;
                bool isAbsorption = vol > absorptionVolThreshold
                    && bodyRatio < AbsorptionBodyRatio
                    && barRange > 0;

                if (isAbsorption)
                {
                    Brush absBrush;
                    double absY;

                    double midBar = (High[0] + Low[0]) / 2.0;
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
            // 6. Volume Delta Bar Coloring (candle outline) — optional
            // ==================================================================
            if (ShowDeltaOutline && barRange > 0 && vol > 0)
            {
                double deltaPct = Math.Abs(delta) / vol;
                if (delta > 0)
                {
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Cyan : Brushes.DodgerBlue;
                }
                else
                {
                    CandleOutlineBrush = deltaPct > 0.3 ? Brushes.Magenta : Brushes.MediumOrchid;
                }
            }
        }

        // =====================================================================
        // Rolling Volume Profile computation
        // =====================================================================
        private void ComputeRollingVolumeProfile()
        {
            double pMin = double.MaxValue;
            double pMax = double.MinValue;

            for (int i = 0; i < VP_Lookback; i++)
            {
                if (i >= CurrentBar) break;
                double h = High[i];
                double l = Low[i];
                if (h > pMax) pMax = h;
                if (l < pMin) pMin = l;
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
            if (binWidth <= 0)
            {
                currentPOC = Close[0];
                return;
            }

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
                    {
                        double overlapPct = (overlapHi - overlapLo) / barRange;
                        vpBinVolumes[j] += barVol * overlapPct;
                    }
                }
            }

            // POC: bin with maximum volume
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
                    else
                    {
                        break;
                    }
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

        // =====================================================================
        // Session POC management
        // =====================================================================
        private void SaveSessionPOC(DateTime sessionDate, double poc)
        {
            if (sessionPOCs == null) return;

            foreach (var sp in sessionPOCs)
            {
                if (sp.Date == sessionDate)
                    return;
            }

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

                Brush pocBrush = Brushes.Yellow;
                Draw.HorizontalLine(this, sp.Tag, sp.Price, pocBrush, DashStyleHelper.Dot, 1);
            }
        }

        // =====================================================================
        // On-chart info box (top-left corner to avoid overlap with FKS_Core)
        // =====================================================================
        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);

            if (CurrentBar < 50) return;

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

            string cvdSlopeStr;
            if (delta > 0)
                cvdSlopeStr = "Rising ▲";
            else if (delta < 0)
                cvdSlopeStr = "Falling ▼";
            else
                cvdSlopeStr = "Flat ---";

            double volAvg = (CurrentBar >= VolumeAvgPeriod && volSMA != null) ? volSMA[0] : 0;
            string volStatus = "Normal";
            if (volAvg > 0)
            {
                double volRatio = vol / volAvg;
                if (volRatio > VolSpikeMultiplier / 100.0)
                    volStatus = string.Format("SPIKE ({0:0.0}x)", volRatio);
                else if (volRatio < 0.5)
                    volStatus = string.Format("LOW ({0:0.0}x)", volRatio);
                else
                    volStatus = string.Format("{0:0.0}x", volRatio);
            }

            int nakedCount = 0;
            if (sessionPOCs != null)
            {
                foreach (var sp in sessionPOCs)
                    if (sp.IsNaked) nakedCount++;
            }

            string vaPosition;
            double price = Close[0];
            if (currentVAH > 0 && currentVAL > 0 && currentVAH != currentVAL)
            {
                if (price > currentVAH)
                    vaPosition = "ABOVE VA";
                else if (price < currentVAL)
                    vaPosition = "BELOW VA";
                else
                    vaPosition = "INSIDE VA";
            }
            else
            {
                vaPosition = "---";
            }

            string infoText = string.Format(
                "FKS Volume\n" +
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
                currentPOC,
                currentVAH,
                currentVAL,
                vaPosition,
                delta,
                deltaDir,
                cvdAccumulator,
                cvdSlopeStr,
                volStatus,
                nakedCount
            );

            Draw.TextFixed(this, "fks_vol_info",
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
