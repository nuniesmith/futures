#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class FuturesDashboardBridge : Strategy
    {
        private Account myAccount;
        private HttpClient httpClient;
        private HttpListener listener;
        private Thread listenerThread;
        private bool lastPushSuccess = true;
        private DateTime lastErrorLog = DateTime.MinValue;

        // Order queue: signals are queued here, then processed on the main
        // thread in OnBarUpdate to avoid cross-thread NinjaScript exceptions.
        private readonly Queue<Action> orderQueue = new Queue<Action>();
        private readonly object queueLock = new object();

        // Throttle: minimum seconds between error log messages
        private const int ERROR_LOG_THROTTLE_SECONDS = 15;

        #region Properties
        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account to Monitor", GroupName = "1. Account", Order = 1)]
        public string AccountName { get; set; } = "Sim101";

        [NinjaScriptProperty]
        [Display(Name = "Position Update URL", GroupName = "2. Web App", Order = 1)]
        public string PositionUpdateUrl { get; set; } = "http://localhost:8000/positions/update";

        [NinjaScriptProperty]
        [Display(Name = "Signal Listener Port", GroupName = "2. Web App", Order = 2)]
        public int SignalListenerPort { get; set; } = 8080;

        [NinjaScriptProperty]
        [Display(Name = "Enable Position Push", GroupName = "3. Options", Order = 1)]
        public bool EnablePositionPush { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Brackets", GroupName = "3. Options", Order = 2)]
        public bool EnableAutoBrackets { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Default SL Ticks (fallback)", GroupName = "3. Options", Order = 3,
                 Description = "Used when signal JSON does not include stop_loss price")]
        public int DefaultStopLossTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "Default TP Ticks (fallback)", GroupName = "3. Options", Order = 4,
                 Description = "Used when signal JSON does not include take_profit price")]
        public int DefaultTakeProfitTicks { get; set; } = 40;
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Two-way bridge: pushes live positions to dashboard, receives order signals with targets";
                Name = "FuturesDashboardBridge";
                Calculate = Calculate.OnEachTick;
                IsOverlay = false;
            }
            else if (State == State.Configure)
            {
                httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name == AccountName);

                if (myAccount != null)
                {
                    myAccount.PositionUpdate += OnPositionUpdate;
                    Print($"[Bridge] Monitoring account: {myAccount.Name}");
                }

                StartSignalListener();
                SendPositionUpdate();
            }
            else if (State == State.Terminated)
            {
                try { if (myAccount != null) myAccount.PositionUpdate -= OnPositionUpdate; } catch { }
                StopSignalListener();
                try { httpClient?.Dispose(); } catch { }
                httpClient = null;
            }
        }

        #region Position Push
        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            try { SendPositionUpdate(); }
            catch (Exception ex) { ThrottledLog($"OnPositionUpdate error: {ex.Message}"); }
        }

        private void SendPositionUpdate()
        {
            if (!EnablePositionPush || myAccount == null) return;
            var client = httpClient;
            if (client == null) return;

            try
            {
                double lastPrice = SafeGetClose();

                var sb = new StringBuilder();
                sb.Append("{\"account\":\"").Append(myAccount.Name).Append("\",\"positions\":[");
                bool first = true;

                var positions = myAccount.Positions;
                if (positions != null)
                {
                    foreach (Position pos in positions)
                    {
                        try
                        {
                            if (pos == null || pos.Quantity == 0) continue;
                            if (pos.Instrument == null) continue;

                            double pnl = 0;
                            try { if (lastPrice > 0) pnl = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, lastPrice); } catch { }

                            if (!first) sb.Append(",");
                            sb.Append("{")
                              .Append("\"symbol\":\"").Append(pos.Instrument.FullName).Append("\",")
                              .Append("\"side\":\"").Append(pos.MarketPosition).Append("\",")
                              .Append("\"quantity\":").Append(pos.Quantity).Append(",")
                              .Append("\"avgPrice\":").Append(pos.AveragePrice).Append(",")
                              .Append("\"unrealizedPnL\":").Append(Math.Round(pnl, 2)).Append(",")
                              .Append("\"instrument\":\"").Append(pos.Instrument.MasterInstrument.Name).Append("\",")
                              .Append("\"tickSize\":").Append(pos.Instrument.MasterInstrument.TickSize).Append(",")
                              .Append("\"pointValue\":").Append(pos.Instrument.MasterInstrument.PointValue).Append(",")
                              .Append("\"lastUpdate\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"")
                              .Append("}");
                            first = false;
                        }
                        catch (Exception ex) { ThrottledLog($"Error reading position: {ex.Message}"); }
                    }
                }

                sb.Append("],\"timestamp\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"}");

                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                client.PostAsync(PositionUpdateUrl, content).ContinueWith(t =>
                {
                    if (t.IsFaulted)
                    {
                        if (lastPushSuccess || (DateTime.Now - lastErrorLog).TotalSeconds > 30)
                        {
                            lastErrorLog = DateTime.Now;
                            lastPushSuccess = false;
                        }
                    }
                    else if (!lastPushSuccess)
                    {
                        Print("[Bridge] Position push connection restored");
                        lastPushSuccess = true;
                    }
                });
            }
            catch (Exception ex) { ThrottledLog($"SendPositionUpdate error: {ex.Message}"); }
        }
        #endregion

        #region Order Execution
        protected override void OnBarUpdate()
        {
            try
            {
                if (State != State.Realtime) return;

                lock (queueLock)
                {
                    while (orderQueue.Count > 0)
                    {
                        var action = orderQueue.Dequeue();
                        try { action(); }
                        catch (Exception ex) { ThrottledLog($"Queued order error: {ex.Message}"); }
                    }
                }
            }
            catch (Exception ex) { ThrottledLog($"OnBarUpdate error: {ex.Message}"); }
        }

        /// <summary>
        /// Process incoming signal JSON and queue the order for main-thread execution.
        ///
        /// Signal JSON format:
        /// {
        ///   "direction": "long" | "short",
        ///   "quantity": 1,
        ///   "order_type": "market" | "limit" | "stop",
        ///   "limit_price": 0,
        ///   "stop_loss": 5200.00,      // exact SL price (optional — falls back to DefaultStopLossTicks)
        ///   "take_profit": 5225.00,    // exact TP1 price (optional — falls back to DefaultTakeProfitTicks)
        ///   "tp2": 5240.00            // exact TP2 price (optional — no fallback)
        /// }
        /// </summary>
        private void ProcessSignal(string json)
        {
            try
            {
                var serializer = new JavaScriptSerializer();
                var signal = serializer.Deserialize<Dictionary<string, object>>(json);

                string dir = GetSignalString(signal, "direction", "long").ToLower();
                int quantity = GetSignalInt(signal, "quantity", 1);
                string typeStr = GetSignalString(signal, "order_type", "market").ToLower();
                double limitPrice = GetSignalDouble(signal, "limit_price", 0);
                double slPrice = GetSignalDouble(signal, "stop_loss", 0);
                double tpPrice = GetSignalDouble(signal, "take_profit", 0);
                double tp2Price = GetSignalDouble(signal, "tp2", 0);

                OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
                OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.Buy;
                OrderType ot = OrderType.Market;
                double stopPrice = 0;

                if (typeStr == "limit") ot = OrderType.Limit;
                else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

                // Capture for closure
                string capturedDir = dir;
                int capturedQty = quantity;
                double capturedSl = slPrice;
                double capturedTp = tpPrice;
                double capturedTp2 = tp2Price;

                lock (queueLock)
                {
                    orderQueue.Enqueue(() =>
                    {
                        if (State != State.Realtime) return;

                        // Submit entry order
                        SubmitOrderUnmanaged(0, action, ot, capturedQty, limitPrice, stopPrice, "", "Dashboard-" + capturedDir);

                        // Submit bracket orders (SL + TP)
                        if (EnableAutoBrackets)
                        {
                            double entry = SafeGetClose();
                            if (entry <= 0) return;

                            // Stop Loss: use exact price if provided, else fall back to tick offset
                            double sl;
                            if (capturedSl > 0)
                                sl = capturedSl;
                            else
                                sl = dir == "long" ? entry - DefaultStopLossTicks * TickSize : entry + DefaultStopLossTicks * TickSize;

                            // Take Profit 1: use exact price if provided, else fall back to tick offset
                            double tp;
                            if (capturedTp > 0)
                                tp = capturedTp;
                            else
                                tp = dir == "long" ? entry + DefaultTakeProfitTicks * TickSize : entry - DefaultTakeProfitTicks * TickSize;

                            // Determine quantities for split targets
                            int slQty = capturedQty;
                            int tp1Qty = capturedTp2 > 0 ? Math.Max(1, capturedQty / 2) : capturedQty;
                            int tp2Qty = capturedTp2 > 0 ? capturedQty - tp1Qty : 0;

                            // SL covers full position
                            SubmitOrderUnmanaged(0, exitAction, OrderType.StopMarket, slQty, 0, sl, "OCO-Bracket", "SL-" + capturedDir);

                            // TP1
                            SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp1Qty, tp, 0, "OCO-Bracket", "TP1-" + capturedDir);

                            // TP2 (only if provided via exact price)
                            if (capturedTp2 > 0 && tp2Qty > 0)
                                SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp2Qty, capturedTp2, 0, "", "TP2-" + capturedDir);

                            Print($"[Bridge] Brackets set: SL={sl:F2} TP1={tp:F2}" + (capturedTp2 > 0 ? $" TP2={capturedTp2:F2}" : ""));
                        }

                        SendPositionUpdate();
                        Print($"[Bridge] Executed {capturedDir.ToUpper()} x{capturedQty}");
                    });
                }
            }
            catch (Exception ex) { ThrottledLog($"Signal error: {ex.Message}"); }
        }
        #endregion

        #region Signal Listener (HTTP)
        private void StartSignalListener()
        {
            string prefix = $"http://localhost:{SignalListenerPort}/";
            try
            {
                listener = new HttpListener();
                listener.Prefixes.Add(prefix);
                listener.Start();
                listenerThread = new Thread(ListenLoop) { IsBackground = true };
                listenerThread.Start();
                Print($"[Bridge] Listening for signals on port {SignalListenerPort}");
            }
            catch (Exception ex)
            {
                Print($"[Bridge] Listener failed on port {SignalListenerPort}: {ex.Message}");
            }
        }

        private void StopSignalListener()
        {
            try { listener?.Stop(); } catch { }
        }

        private void ListenLoop()
        {
            while (listener?.IsListening == true)
            {
                try
                {
                    var context = listener.GetContext();
                    string path = context.Request.Url.AbsolutePath;
                    string method = context.Request.HttpMethod;

                    if (method == "POST" && path == "/execute_signal")
                    {
                        using var reader = new System.IO.StreamReader(context.Request.InputStream);
                        string json = reader.ReadToEnd();
                        ProcessSignal(json);
                        SendResponse(context.Response, 200, "{\"status\":\"queued\"}");
                    }
                    else if (method == "GET" && path == "/status")
                    {
                        // Return current account status for dashboard health check
                        string status = BuildStatusJson();
                        SendResponse(context.Response, 200, status);
                    }
                    else
                    {
                        SendResponse(context.Response, 404, "{\"error\":\"not found\"}");
                    }
                }
                catch { }
            }
        }

        private string BuildStatusJson()
        {
            try
            {
                string acctName = myAccount?.Name ?? "disconnected";
                int posCount = 0;
                try { if (myAccount?.Positions != null) posCount = myAccount.Positions.Count(p => p.Quantity != 0); } catch { }

                return $"{{\"account\":\"{acctName}\",\"positions\":{posCount},\"state\":\"{State}\",\"connected\":{(myAccount != null ? "true" : "false")}}}";
            }
            catch { return "{\"error\":\"status unavailable\"}"; }
        }
        #endregion

        #region Helpers
        private double SafeGetClose()
        {
            try
            {
                if (CurrentBar >= 0 && Close != null && Close.Count > 0)
                    return Close[0];
            }
            catch { }
            return 0;
        }

        private static string GetSignalString(Dictionary<string, object> signal, string key, string defaultValue)
        {
            if (signal.ContainsKey(key) && signal[key] != null)
                return signal[key].ToString();
            return defaultValue;
        }

        private static int GetSignalInt(Dictionary<string, object> signal, string key, int defaultValue)
        {
            if (signal.ContainsKey(key) && signal[key] != null)
            {
                try { return (int)Math.Round(Convert.ToDouble(signal[key])); } catch { }
            }
            return defaultValue;
        }

        private static double GetSignalDouble(Dictionary<string, object> signal, string key, double defaultValue)
        {
            if (signal.ContainsKey(key) && signal[key] != null)
            {
                try { return Convert.ToDouble(signal[key]); } catch { }
            }
            return defaultValue;
        }

        private void SendResponse(HttpListenerResponse resp, int code, string msg)
        {
            try
            {
                resp.StatusCode = code;
                resp.ContentType = "application/json";
                byte[] buf = Encoding.UTF8.GetBytes(msg);
                resp.ContentLength64 = buf.Length;
                resp.OutputStream.Write(buf, 0, buf.Length);
                resp.OutputStream.Close();
            }
            catch { }
        }

        private void ThrottledLog(string message)
        {
            if ((DateTime.Now - lastErrorLog).TotalSeconds >= ERROR_LOG_THROTTLE_SECONDS)
            {
                Print($"[Bridge] {message}");
                lastErrorLog = DateTime.Now;
            }
        }
        #endregion
    }
}
