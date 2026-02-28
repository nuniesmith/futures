#region Using declarations
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
#endregion

namespace NinjaTrader.NinjaScript
{
    /// <summary>
    /// SignalBus — static in-memory signal relay for indicator → strategy communication.
    ///
    /// This enables Ruby (indicator) to forward entry/exit signals to Bridge (strategy)
    /// without requiring HTTP in backtest mode. In live/realtime mode the HTTP path
    /// (/execute_signal) is preferred, but SignalBus serves as the universal fallback
    /// and is the *only* reliable path inside Strategy Analyzer backtests where the
    /// Bridge HTTP listener is not running.
    ///
    /// Architecture:
    ///   Ruby.OnBarUpdate()  →  SignalBus.Enqueue(signal)
    ///   Bridge.OnBarUpdate()  →  SignalBus.DrainAll()  →  Bridge.ProcessSignal(json)
    ///
    /// Thread-safety: all public methods are safe to call from any thread.
    /// The internal queue is a ConcurrentQueue so no explicit locking is needed.
    /// </summary>
    public static class SignalBus
    {
        // ── Signal payload ────────────────────────────────────────────
        public class Signal
        {
            public string Direction { get; set; }  // "long" | "short"
            public string SignalType { get; set; }  // "entry" | "exit"
            public int Quantity { get; set; }  // requested qty (Bridge will risk-size)
            public string OrderType { get; set; }  // "market" | "limit" | "stop"
            public double LimitPrice { get; set; }
            public double StopLoss { get; set; }  // exact SL price (0 = use Bridge default)
            public double TakeProfit { get; set; }  // exact TP1 price (0 = use Bridge default)
            public double TakeProfit2 { get; set; }  // optional TP2 price
            public string Strategy { get; set; }  // originating strategy/indicator name
            public string Asset { get; set; }  // instrument symbol or friendly name
            public string SignalId { get; set; }  // unique ID for tracking
            public double SignalQuality { get; set; }  // 0.0–1.0 quality score from Ruby
            public double WaveRatio { get; set; }  // wave dominance ratio from Ruby
            public DateTime Timestamp { get; set; }  // bar time when signal was generated
            public string ExitReason { get; set; }  // for exit signals: reason string

            /// <summary>
            /// Serialize to the JSON format that Bridge.ProcessSignal() expects.
            /// Hand-rolled to avoid taking a dependency on a JSON library that
            /// may or may not be loaded in the NinjaTrader process.
            /// </summary>
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

        // ── Internal state ────────────────────────────────────────────
        private static readonly ConcurrentQueue<Signal> _queue = new ConcurrentQueue<Signal>();
        private static volatile bool _bridgeRegistered;
        private static volatile int _totalEnqueued;
        private static volatile int _totalDrained;

        // ── Producer API (called by Ruby / any indicator) ─────────────

        /// <summary>
        /// Enqueue a signal for the Bridge strategy to pick up on its next
        /// OnBarUpdate() tick.  Returns true if a Bridge consumer is registered
        /// (i.e. someone is listening), false otherwise.  The signal is enqueued
        /// regardless so it won't be lost if Bridge registers a moment later.
        /// </summary>
        public static bool Enqueue(Signal signal)
        {
            if (signal == null) return false;

            // Assign defaults for anything the caller didn't set
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

        /// <summary>
        /// Convenience: build and enqueue an entry signal in one call.
        /// </summary>
        public static bool EnqueueEntry(
            string direction,
            string asset,
            double stopLoss = 0,
            double takeProfit = 0,
            double takeProfit2 = 0,
            int quantity = 1,
            string orderType = "market",
            double limitPrice = 0,
            double signalQuality = 0,
            double waveRatio = 0,
            string strategy = "Ruby")
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

        /// <summary>
        /// Convenience: enqueue a flat/exit signal.
        /// </summary>
        public static bool EnqueueExit(
            string asset,
            string reason = "signal",
            string strategy = "Ruby")
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

        // ── Consumer API (called by Bridge strategy) ──────────────────

        /// <summary>
        /// Register that a Bridge instance is running and will consume signals.
        /// Call this from Bridge.OnStateChange (State.Configure or State.Realtime).
        /// </summary>
        public static void RegisterConsumer()
        {
            _bridgeRegistered = true;
        }

        /// <summary>
        /// Unregister the Bridge consumer. Call from Bridge.OnStateChange
        /// (State.Terminated).
        /// </summary>
        public static void UnregisterConsumer()
        {
            _bridgeRegistered = false;
        }

        /// <summary>
        /// Drain all pending signals from the queue.  Returns an empty list
        /// if nothing is waiting.  This is non-blocking and lock-free.
        /// Bridge should call this on every OnBarUpdate().
        /// </summary>
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

        /// <summary>
        /// Peek at how many signals are waiting without draining.
        /// </summary>
        public static int PendingCount
        {
            get { return _queue.Count; }
        }

        // ── Diagnostics ───────────────────────────────────────────────

        /// <summary>True if a Bridge consumer is currently registered.</summary>
        public static bool HasConsumer
        {
            get { return _bridgeRegistered; }
        }

        /// <summary>Total signals ever enqueued across the process lifetime.</summary>
        public static int TotalEnqueued
        {
            get { return _totalEnqueued; }
        }

        /// <summary>Total signals ever drained/consumed.</summary>
        public static int TotalDrained
        {
            get { return _totalDrained; }
        }

        /// <summary>
        /// Clear all pending signals and reset counters.
        /// Useful between backtest runs to avoid stale signal bleed.
        /// </summary>
        public static void Reset()
        {
            while (_queue.TryDequeue(out _)) { }
            _totalEnqueued = 0;
            _totalDrained = 0;
            _bridgeRegistered = false;
        }
    }
}
