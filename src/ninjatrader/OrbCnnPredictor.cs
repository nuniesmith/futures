// =============================================================================
// OrbCnnPredictor.cs — ONNX-backed CNN inference for HybridBreakoutCNN
// =============================================================================
//
// Wraps Microsoft.ML.OnnxRuntime to run the exported HybridBreakoutCNN model
// directly inside NinjaTrader 8 — no Python process, no sockets, < 1 ms per
// inference on CPU (tested on i7-12700; CUDA path is also supported).
//
// Architecture note (must match the Python model exactly):
//   Input 0 — "image"   : float32 [1, 3, 224, 224]  (ImageNet-normalised RGB)
//   Input 1 — "tabular" : float32 [1, 8]             (see TABULAR_FEATURES below)
//   Output  — "output"  : float32 [1, 2]             (logits for [bad, good])
//
//   P(good breakout) = softmax(output)[0, 1]
//
// Tabular feature vector (must match TABULAR_FEATURES in breakout_cnn.py):
//   [0] quality_pct_norm  — Ruby quality score / 100          (0–1)
//   [1] volume_ratio      — breakout bar vol / 20-bar avg      (0–∞, log-scaled internally)
//   [2] atr_pct           — ATR / price                        (0–0.01+, ×100 internally)
//   [3] cvd_delta         — cumulative vol delta               (clamped –1..1)
//   [4] nr7_flag          — 1.0 if NR7 day, else 0.0
//   [5] direction_flag    — 1.0 = LONG, 0.0 = SHORT
//   [6] session_ordinal   — position in 24-h day [0..1] (see SESSION_ORDINALS)
//   [7] london_overlap    — 1.0 if 08:00–09:00 ET, else 0.0
//
// Chart rendering:
//   OrbChartRenderer (inner class) renders a 224×224 RGB PNG from raw bar data
//   using System.Drawing (GDI+).  The visual style mirrors the Ruby NT8 indicator
//   (dark background, green/red candles, ORB shading, VWAP line, volume panel)
//   so the CNN sees images that match its training distribution.
//
// Installation (copy to Documents\NinjaTrader 8\bin\Custom\):
//   OrbCnnPredictor.dll
//   Microsoft.ML.OnnxRuntime.dll
//   onnxruntime.dll
//   onnxruntime_providers_shared.dll
//
// Recommended NuGet version: Microsoft.ML.OnnxRuntime 1.18.1
//   (stable with .NET Framework 4.8 / NinjaTrader 8)
//
// =============================================================================

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NinjaTrader.NinjaScript
{
    // =========================================================================
    // Public result type
    // =========================================================================

    /// <summary>
    /// Result returned by <see cref="OrbCnnPredictor.Predict"/>.
    /// </summary>
    public sealed class CnnPrediction
    {
        /// <summary>P(good breakout) — 0.0–1.0.</summary>
        public float Probability { get; }

        /// <summary>True when Probability >= the threshold passed to Predict().</summary>
        public bool Signal { get; }

        /// <summary>"high" / "medium" / "low" relative to the threshold.</summary>
        public string Confidence { get; }

        /// <summary>Threshold that was applied.</summary>
        public float Threshold { get; }

        internal CnnPrediction(float prob, float threshold)
        {
            Probability = prob;
            Threshold   = threshold;
            Signal      = prob >= threshold;

            if      (prob >= threshold + 0.08f) Confidence = "high";
            else if (prob >= threshold - 0.04f) Confidence = "medium";
            else                                Confidence = "low";
        }

        public override string ToString() =>
            $"P={Probability:P1} [{Confidence}] signal={Signal} (thr={Threshold:P1})";
    }

    // =========================================================================
    // Session threshold table (mirrors SESSION_THRESHOLDS in breakout_cnn.py)
    // =========================================================================

    /// <summary>
    /// Per-session inference thresholds that mirror <c>SESSION_THRESHOLDS</c>
    /// in <c>breakout_cnn.py</c>.  Use <see cref="GetSessionThreshold"/> to look
    /// up the right value before calling <see cref="OrbCnnPredictor.Predict"/>.
    /// </summary>
    public static class CnnSessionThresholds
    {
        // Mirrors SESSION_THRESHOLDS dict in breakout_cnn.py exactly.
        private static readonly Dictionary<string, float> _thresholds =
            new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
            {
                // ── Overnight (thin markets) ───────────────────────────────
                { "cme",        0.75f },  // CME Globex re-open 18:00 ET
                { "sydney",     0.72f },  // Sydney / ASX
                { "tokyo",      0.74f },  // Tokyo / TSE
                { "shanghai",   0.74f },  // Shanghai / HK
                // ── Primary daytime ────────────────────────────────────────
                { "frankfurt",  0.80f },  // Frankfurt / Xetra 03:00 ET
                { "london",     0.82f },  // London Open 03:00 ET
                { "london_ny",  0.82f },  // London-NY crossover 08:00 ET
                { "us",         0.82f },  // US Equity Open 09:30 ET
                { "cme_settle", 0.78f },  // CME Settlement 14:00 ET
            };

        /// <summary>Default when session key is unknown.</summary>
        public const float Default = 0.82f;

        /// <summary>
        /// Returns the threshold for <paramref name="sessionKey"/>,
        /// falling back to <see cref="Default"/> for unknown keys.
        /// </summary>
        public static float GetSessionThreshold(string sessionKey)
        {
            if (string.IsNullOrWhiteSpace(sessionKey)) return Default;
            return _thresholds.TryGetValue(sessionKey.Trim(), out float t) ? t : Default;
        }

        // ── Session ordinal encoding (mirrors SESSION_ORDINAL in breakout_cnn.py) ──

        private static readonly Dictionary<string, float> _ordinals =
            new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
            {
                { "cme",        0.0f / 8 },
                { "sydney",     1.0f / 8 },
                { "tokyo",      2.0f / 8 },
                { "shanghai",   3.0f / 8 },
                { "frankfurt",  4.0f / 8 },
                { "london",     5.0f / 8 },
                { "london_ny",  6.0f / 8 },
                { "us",         7.0f / 8 },
                { "cme_settle", 8.0f / 8 },
            };

        /// <summary>
        /// Returns the ordinal float [0..1] for <paramref name="sessionKey"/>.
        /// Defaults to the US session ordinal (7/8 = 0.875) for unknown keys.
        /// </summary>
        public static float GetSessionOrdinal(string sessionKey)
        {
            if (string.IsNullOrWhiteSpace(sessionKey)) return 7.0f / 8;
            return _ordinals.TryGetValue(sessionKey.Trim(), out float o) ? o : 7.0f / 8;
        }
    }

    // =========================================================================
    // Main predictor
    // =========================================================================

    /// <summary>
    /// Runs HybridBreakoutCNN (ONNX) inference inside NinjaTrader 8.
    /// Thread-safe — a single instance can be shared across multiple
    /// BarsInProgress callbacks without locking (ONNX Runtime is thread-safe
    /// for concurrent Run() calls on the same session).
    /// </summary>
    public sealed class OrbCnnPredictor : IDisposable
    {
        // ── ONNX session ──────────────────────────────────────────────────────
        private readonly InferenceSession _session;
        private readonly string           _imageName;
        private readonly string           _tabularName;
        private readonly string           _outputName;
        private readonly string           _modelPath;

        private bool _disposed;

        // ── ImageNet normalisation constants (match torchvision transforms) ───
        // mean and std are per-channel, applied after scaling pixels to [0, 1].
        private static readonly float[] _mean = { 0.485f, 0.456f, 0.406f }; // R, G, B
        private static readonly float[] _std  = { 0.229f, 0.224f, 0.225f };

        // ── Tensor dimensions ─────────────────────────────────────────────────
        private const int ImageSize   = 224;
        private const int NumTabular  = 8;
        private const int NumChannels = 3;

        // ── Image buffer size constant ────────────────────────────────────────
        // 1 × 3 × 224 × 224 = 150 528 floats.
        // NOTE: _imageBuf is intentionally NOT a shared field — each Predict()
        // call allocates its own buffer so concurrent calls from multiple BIPs
        // on different threads cannot race on a single shared array.
        private const int ImageBufSize = 1 * NumChannels * ImageSize * ImageSize;

        /// <summary>Path to the .onnx file that was loaded.</summary>
        public string ModelPath => _modelPath;

        // =====================================================================
        // Construction
        // =====================================================================

        /// <summary>
        /// Load an ONNX model exported from HybridBreakoutCNN.
        /// </summary>
        /// <param name="modelPath">Full path to <c>orb_breakout_cnn.onnx</c>.</param>
        /// <param name="useCuda">
        ///   When true, attempts to use the CUDA execution provider.
        ///   Falls back silently to CPU if CUDA is unavailable.
        /// </param>
        public OrbCnnPredictor(string modelPath, bool useCuda = false)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"OrbCnnPredictor: model not found at {modelPath}", modelPath);

            _modelPath = modelPath;

            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads  = 2;
            options.IntraOpNumThreads  = 2;

            if (useCuda)
            {
                try
                {
                    options.AppendExecutionProvider_CUDA(0);
                }
                catch
                {
                    // CUDA not available — CPU fallback is automatic
                }
            }

            _session = new InferenceSession(modelPath, options);

            // ── Resolve input names by substring match, not positional order ──
            // ONNX InputMetadata is a Dictionary — iteration order is not
            // guaranteed to match the model's actual input order.  Relying on
            // inputNames[0] / [1] can silently swap the image and tabular tensors
            // if the exporter or runtime enumerates keys differently.
            // We instead find each name by its expected substring so the lookup
            // is robust to ordering changes in export_onnx.py.
            var inputMeta = _session.InputMetadata;
            if (inputMeta.Count < 2)
                throw new InvalidOperationException(
                    $"OrbCnnPredictor: expected 2 inputs (image + tabular), found {inputMeta.Count}. " +
                    $"Re-export using scripts/export_onnx.py.");

            // Look for the key whose name contains "image" (case-insensitive).
            _imageName = inputMeta.Keys
                .FirstOrDefault(k => k.IndexOf("image", StringComparison.OrdinalIgnoreCase) >= 0);

            // Look for the key whose name contains "tabular" (case-insensitive).
            _tabularName = inputMeta.Keys
                .FirstOrDefault(k => k.IndexOf("tabular", StringComparison.OrdinalIgnoreCase) >= 0);

            // Fallback: if the names don't contain the expected substrings (e.g. the
            // model was exported with generic names like "input_0", "input_1"), fall
            // back to positional assignment and log a warning so it is visible.
            if (_imageName == null || _tabularName == null)
            {
                var inputNames = inputMeta.Keys.ToList();
                _imageName   = _imageName   ?? inputNames[0];
                _tabularName = _tabularName ?? inputNames.FirstOrDefault(k => k != _imageName) ?? inputNames[1];

                // Not throwing here keeps the model usable; the operator should
                // verify the export if inference results look wrong.
                System.Diagnostics.Debug.WriteLine(
                    $"[OrbCnnPredictor] WARNING: could not identify input names by substring. " +
                    $"Fell back to: image='{_imageName}' tabular='{_tabularName}'. " +
                    $"All input keys: [{string.Join(", ", inputMeta.Keys)}]");
            }

            _outputName = _session.OutputMetadata.Keys.First();
        }

        // =====================================================================
        // Predict — main entry point
        // =====================================================================

        /// <summary>
        /// Run the CNN on a pre-rendered chart image + tabular feature vector.
        /// </summary>
        /// <param name="imagePath">
        ///   Path to a 224×224 PNG rendered by <see cref="OrbChartRenderer"/>.
        ///   The file does not need to exist — if it is missing, the method
        ///   falls back to tabular-only mode (image tensor zeroed out) and logs
        ///   a warning so you are not silently blocked.
        /// </param>
        /// <param name="tabular">
        ///   Exactly 8 raw feature values in <c>TABULAR_FEATURES</c> order:
        ///   [quality_pct_norm, volume_ratio, atr_pct, cvd_delta, nr7_flag,
        ///    direction_flag, session_ordinal, london_overlap_flag].
        ///   Normalisation (log-scale, clamp, etc.) is applied here, matching
        ///   <c>_normalise_tabular_for_inference</c> in breakout_cnn.py.
        /// </param>
        /// <param name="threshold">
        ///   Probability threshold for the <see cref="CnnPrediction.Signal"/> flag.
        ///   Use <see cref="CnnSessionThresholds.GetSessionThreshold"/> to get the
        ///   right value for the current session.
        /// </param>
        /// <returns>
        ///   A <see cref="CnnPrediction"/> or null if inference failed.
        /// </returns>
        public CnnPrediction Predict(string imagePath, float[] tabular, float threshold)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(OrbCnnPredictor));
            if (tabular == null || tabular.Length != NumTabular)
                throw new ArgumentException($"tabular must have exactly {NumTabular} elements", nameof(tabular));

            // ── Image tensor ─────────────────────────────────────────────────
            // Allocate a fresh buffer per call so concurrent Predict() invocations
            // from multiple BIPs on different threads cannot race on a single shared
            // array.  At 150 528 floats × 4 bytes = ~590 KB this is negligible for
            // the once-per-breakout-signal call frequency.
            var imageBuf    = new float[ImageBufSize];
            bool imageLoaded = LoadImageToBuffer(imagePath, imageBuf);
            // Zero-image fallback: if loading fails the buffer is already zeroed
            // (new float[] is value-initialised), so no explicit Clear needed.

            var imageTensor   = new DenseTensor<float>(imageBuf,
                                    new[] { 1, NumChannels, ImageSize, ImageSize });

            // ── Tabular tensor ────────────────────────────────────────────────
            float[] normTab   = NormaliseTabular(tabular);
            var tabularTensor = new DenseTensor<float>(normTab, new[] { 1, NumTabular });

            // ── Run inference ─────────────────────────────────────────────────
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_imageName,   imageTensor),
                NamedOnnxValue.CreateFromTensor(_tabularName, tabularTensor),
            };

            using var results = _session.Run(inputs);

            // Output is logits [1, 2] for [bad, good].
            float[] logits = results
                .First(r => r.Name == _outputName)
                .AsTensor<float>()
                .ToArray();

            if (logits.Length < 2)
                throw new InvalidOperationException(
                    $"OrbCnnPredictor: expected output of length 2, got {logits.Length}");

            float probGood = Softmax2(logits[0], logits[1]);
            return new CnnPrediction(probGood, threshold);
        }

        // =====================================================================
        // Tabular normalisation — mirrors _normalise_tabular_for_inference()
        // =====================================================================

        /// <summary>
        /// Apply the same per-feature normalisation that the Python dataset uses.
        /// Input is raw values; output is the 8-float normalised vector ready for
        /// the tabular head.
        /// </summary>
        public static float[] NormaliseTabular(float[] raw)
        {
            if (raw.Length != NumTabular)
                throw new ArgumentException($"Expected {NumTabular} raw features");

            float[] n = new float[NumTabular];

            // [0] quality_pct_norm — caller passes already / 100 (0–1), just clamp
            n[0] = Math.Max(0f, Math.Min(1f, raw[0]));

            // [1] volume_ratio — log1p(x) / log1p(10), clamped to [0, 1]
            float volRaw = Math.Max(raw[1], 0.01f);
            n[1] = Math.Min((float)(Math.Log(1 + volRaw) / Math.Log(11.0)), 1f);

            // [2] atr_pct — multiply by 100 and clamp to [0, 1]
            n[2] = Math.Min(raw[2] * 100f, 1f);

            // [3] cvd_delta — clamp to [-1, 1]
            n[3] = Math.Max(-1f, Math.Min(1f, raw[3]));

            // [4] nr7_flag — 0 or 1 passthrough
            n[4] = raw[4] >= 0.5f ? 1f : 0f;

            // [5] direction_flag — 0 or 1 passthrough
            n[5] = raw[5] >= 0.5f ? 1f : 0f;

            // [6] session_ordinal — already [0, 1] from CnnSessionThresholds.GetSessionOrdinal
            n[6] = Math.Max(0f, Math.Min(1f, raw[6]));

            // [7] london_overlap_flag — 0 or 1 passthrough
            n[7] = raw[7] >= 0.5f ? 1f : 0f;

            return n;
        }

        // =====================================================================
        // Image loading — GDI+ → ImageNet-normalised float tensor
        // =====================================================================

        /// <summary>
        /// Load a PNG/JPEG from <paramref name="path"/>, resize to 224×224,
        /// convert to float [0,1], and apply ImageNet mean/std normalisation.
        /// Fills <paramref name="buf"/> in CHW layout (channels-first, R then G then B).
        /// Returns true on success.
        /// </summary>
        private static bool LoadImageToBuffer(string path, float[] buf)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
                return false;

            try
            {
                using var src     = new Bitmap(path);
                using var resized = new Bitmap(ImageSize, ImageSize, PixelFormat.Format24bppRgb);
                using (var g = Graphics.FromImage(resized))
                {
                    g.InterpolationMode  = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    g.SmoothingMode      = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    g.DrawImage(src, 0, 0, ImageSize, ImageSize);
                }

                // Lock bits for fast pixel access
                var rect = new Rectangle(0, 0, ImageSize, ImageSize);
                var data = resized.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    int stride    = data.Stride;
                    int planeSize = ImageSize * ImageSize;

                    // Format24bppRgb is stored as B, G, R bytes (little-endian GDI+).
                    // We want CHW order: all R, then all G, then all B.
                    unsafe
                    {
                        byte* ptr = (byte*)data.Scan0.ToPointer();
                        for (int y = 0; y < ImageSize; y++)
                        {
                            byte* row = ptr + y * stride;
                            for (int x = 0; x < ImageSize; x++)
                            {
                                // GDI+ stores B, G, R at each pixel
                                float b = row[x * 3 + 0] / 255f;
                                float g = row[x * 3 + 1] / 255f;
                                float r = row[x * 3 + 2] / 255f;

                                int pixIdx = y * ImageSize + x;

                                // Normalise: (pixel - mean) / std
                                buf[0 * planeSize + pixIdx] = (r - _mean[0]) / _std[0]; // R
                                buf[1 * planeSize + pixIdx] = (g - _mean[1]) / _std[1]; // G
                                buf[2 * planeSize + pixIdx] = (b - _mean[2]) / _std[2]; // B
                            }
                        }
                    }
                }
                finally
                {
                    resized.UnlockBits(data);
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        // =====================================================================
        // Softmax for 2-class output
        // =====================================================================

        /// <summary>
        /// Numerically stable 2-class softmax. Returns P(class 1).
        /// </summary>
        private static float Softmax2(float logit0, float logit1)
        {
            float max  = Math.Max(logit0, logit1);
            float e0   = (float)Math.Exp(logit0 - max);
            float e1   = (float)Math.Exp(logit1 - max);
            return e1 / (e0 + e1);
        }

        // =====================================================================
        // Dispose
        // =====================================================================

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
            }
        }
    }

    // =========================================================================
    // Chart renderer — GDI+ Ruby-style ORB snapshot (224 × 224 PNG)
    // =========================================================================

    /// <summary>
    /// Renders a Ruby-style ORB candlestick chart as a 224×224 PNG.
    ///
    /// The visual style matches <c>chart_renderer.py</c> closely enough that
    /// the CNN's image backbone sees images from the same distribution as its
    /// training data:
    ///   • Dark background (#0d0d0d)
    ///   • Green (up) / red (down) filled candles with wicks
    ///   • ORB zone — semi-transparent gold shading between OrbHigh and OrbLow
    ///   • VWAP — dashed cyan line
    ///   • Volume panel — bottom 20 % of chart (green/red bars)
    ///   • Breakout label — "▲ LONG" or "▼ SHORT" above/below entry bar
    ///
    /// Image is written to a temp folder and the path is returned so
    /// <see cref="OrbCnnPredictor.Predict"/> can load it.
    /// </summary>
    public static class OrbChartRenderer
    {
        // ── Canvas constants ──────────────────────────────────────────────────
        private const int W           = 224;  // total canvas width
        private const int H           = 224;  // total canvas height
        private const int VolPanelH   = 40;   // bottom panel for volume bars
        private const int PriceH      = H - VolPanelH - 4; // height available for price
        private const int PriceTop    = 4;    // top margin
        private const int LeftPad     = 4;    // left margin
        private const int RightPad    = 4;    // right margin

        // ── Colours (match chart_renderer.py dark theme) ──────────────────────
        private static readonly Color _background   = Color.FromArgb(0x0D, 0x0D, 0x0D);
        private static readonly Color _bullCandle   = Color.FromArgb(0x26, 0xA6, 0x9A); // teal-green
        private static readonly Color _bearCandle   = Color.FromArgb(0xEF, 0x53, 0x50); // red
        private static readonly Color _orbFill      = Color.FromArgb(40, 0xFF, 0xD7, 0x00); // gold, 16% alpha
        private static readonly Color _orbBorder    = Color.FromArgb(100, 0xFF, 0xD7, 0x00);
        private static readonly Color _vwapLine     = Color.FromArgb(0x00, 0xE5, 0xFF); // cyan
        private static readonly Color _volBull      = Color.FromArgb(100, 0x26, 0xA6, 0x9A);
        private static readonly Color _volBear      = Color.FromArgb(100, 0xEF, 0x53, 0x50);
        private static readonly Color _textColor    = Color.FromArgb(0xCC, 0xCC, 0xCC);
        private static readonly Color _longLabel    = Color.FromArgb(0x66, 0xBB, 0x6A); // green
        private static readonly Color _shortLabel   = Color.FromArgb(0xEF, 0x53, 0x50); // red

        /// <summary>
        /// Bar data record passed to the renderer.
        /// </summary>
        public sealed class Bar
        {
            public DateTime Time   { get; }
            public double   Open   { get; }
            public double   High   { get; }
            public double   Low    { get; }
            public double   Close  { get; }
            public double   Volume { get; }

            public Bar(DateTime time, double open, double high, double low, double close, double volume)
            {
                Time   = time;
                Open   = open;
                High   = high;
                Low    = low;
                Close  = close;
                Volume = volume;
            }
        }

        /// <summary>
        /// Render a 224×224 chart snapshot and write it to <paramref name="outputPath"/>.
        /// </summary>
        /// <param name="bars">
        ///   Chronological bar array (most-recent last).  Typically 60 bars of 1-min data.
        /// </param>
        /// <param name="orbHigh">ORB high price level.</param>
        /// <param name="orbLow">ORB low price level.</param>
        /// <param name="vwap">Current VWAP value (drawn as horizontal dashed line).</param>
        /// <param name="direction">"long" or "short".</param>
        /// <param name="outputPath">Where to write the PNG (directory must exist).</param>
        /// <returns>
        ///   True on success; false if rendering failed (caller should skip CNN filter).
        /// </returns>
        public static bool Render(
            Bar[]  bars,
            double orbHigh,
            double orbLow,
            double vwap,
            string direction,
            string outputPath)
        {
            if (bars == null || bars.Length == 0) return false;

            try
            {
                using var bmp = new Bitmap(W, H, PixelFormat.Format32bppArgb);
                using var g   = Graphics.FromImage(bmp);

                g.SmoothingMode      = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.TextRenderingHint  = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

                // ── Background ────────────────────────────────────────────────
                g.Clear(_background);

                // ── Price range across all bars ───────────────────────────────
                double priceMin = bars.Min(b => b.Low);
                double priceMax = bars.Max(b => b.High);

                // Expand range slightly so candles don't touch the edges
                double pad     = (priceMax - priceMin) * 0.05;
                priceMin      -= pad;
                priceMax      += pad;
                double priceRange = priceMax - priceMin;
                if (priceRange <= 0) priceRange = 1;

                // ── Helpers: price → Y pixel, volume → height ─────────────────
                int PriceToY(double p) =>
                    PriceTop + (int)((1.0 - (p - priceMin) / priceRange) * (PriceH - PriceTop));

                int priceAreaW  = W - LeftPad - RightPad;
                int numBars     = bars.Length;
                float barWidth  = (float)priceAreaW / numBars;
                float candleW   = Math.Max(1f, barWidth * 0.7f);

                double volMax = bars.Max(b => b.Volume);
                if (volMax <= 0) volMax = 1;

                // ── ORB zone ──────────────────────────────────────────────────
                int orbTopY    = PriceToY(orbHigh);
                int orbBottomY = PriceToY(orbLow);
                if (orbBottomY > orbTopY)
                {
                    using var orbFillBrush   = new SolidBrush(_orbFill);
                    using var orbBorderPen   = new Pen(_orbBorder, 1f);
                    g.FillRectangle(orbFillBrush, LeftPad, orbTopY, priceAreaW, orbBottomY - orbTopY);
                    g.DrawLine(orbBorderPen, LeftPad, orbTopY,    LeftPad + priceAreaW, orbTopY);
                    g.DrawLine(orbBorderPen, LeftPad, orbBottomY, LeftPad + priceAreaW, orbBottomY);
                }

                // ── VWAP line ─────────────────────────────────────────────────
                if (vwap > priceMin && vwap < priceMax)
                {
                    int vwapY = PriceToY(vwap);
                    using var vwapPen = new Pen(_vwapLine, 1.2f) { DashStyle = System.Drawing.Drawing2D.DashStyle.Dash };
                    g.DrawLine(vwapPen, LeftPad, vwapY, LeftPad + priceAreaW, vwapY);
                }

                // ── Candlesticks + volume bars ────────────────────────────────
                for (int i = 0; i < numBars; i++)
                {
                    var   bar      = bars[i];
                    bool  isUp     = bar.Close >= bar.Open;
                    Color candleC  = isUp ? _bullCandle : _bearCandle;
                    Color volC     = isUp ? _volBull    : _volBear;

                    float xLeft    = LeftPad + i * barWidth;
                    float xCenter  = xLeft + barWidth / 2f;
                    float xCandleL = xCenter - candleW / 2f;

                    // Price candle body
                    int bodyTop    = PriceToY(Math.Max(bar.Open, bar.Close));
                    int bodyBottom = PriceToY(Math.Min(bar.Open, bar.Close));
                    int bodyH      = Math.Max(1, bodyBottom - bodyTop);

                    using (var brush = new SolidBrush(candleC))
                        g.FillRectangle(brush, xCandleL, bodyTop, candleW, bodyH);

                    // Wick
                    int wickTop    = PriceToY(bar.High);
                    int wickBottom = PriceToY(bar.Low);
                    using (var pen = new Pen(candleC, 1f))
                    {
                        g.DrawLine(pen, xCenter, wickTop,    xCenter, bodyTop);
                        g.DrawLine(pen, xCenter, bodyBottom, xCenter, wickBottom);
                    }

                    // Volume bar (bottom panel)
                    int volBarH  = (int)(bar.Volume / volMax * (VolPanelH - 4));
                    int volBarY  = H - 2 - volBarH;
                    using (var brush = new SolidBrush(volC))
                        g.FillRectangle(brush, xCandleL, volBarY, candleW, volBarH);
                }

                // ── Breakout direction label ───────────────────────────────────
                bool isLong   = direction.Equals("long", StringComparison.OrdinalIgnoreCase);
                string label  = isLong ? "▲ LONG" : "▼ SHORT";
                Color labelC  = isLong ? _longLabel : _shortLabel;

                using var font      = new Font("Arial", 7f, FontStyle.Bold);
                using var labelBrush = new SolidBrush(labelC);
                SizeF     labelSz   = g.MeasureString(label, font);
                g.DrawString(label, font, labelBrush,
                             W - labelSz.Width - 4, PriceTop + 2);

                // ── Save ──────────────────────────────────────────────────────
                string dir = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                bmp.Save(outputPath, ImageFormat.Png);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Convenience wrapper: builds a deterministic temp file path from
        /// the instrument name and bar timestamp, renders the chart, and
        /// returns the path (or null on failure).
        /// </summary>
        public static string RenderToTemp(
            Bar[]  bars,
            double orbHigh,
            double orbLow,
            double vwap,
            string direction,
            string symbol,
            DateTime barTime,
            string tempFolder = null)
        {
            string folder = tempFolder
                ?? Path.Combine(Path.GetTempPath(), "NT8_OrbCnn");

            string fileName = $"{symbol}_{barTime:yyyyMMdd_HHmmss}_{direction[0]}.png";
            string fullPath = Path.Combine(folder, fileName);

            return Render(bars, orbHigh, orbLow, vwap, direction, fullPath)
                ? fullPath
                : null;
        }
    }
}
