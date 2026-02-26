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

        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account to Monitor", GroupName = "1. Account", Order = 1)]
        public string AccountName { get; set; } = "Sim101";

        [NinjaScriptProperty]
        [Display(Name = "Position Update URL", GroupName = "2. Web App", Order = 1)]
        public string PositionUpdateUrl { get; set; } = "http://localhost:8000/update_positions";

        [NinjaScriptProperty]
        [Display(Name = "Signal Listener Port", GroupName = "2. Web App", Order = 2)]
        public int SignalListenerPort { get; set; } = 8080;

        [NinjaScriptProperty]
        [Display(Name = "Enable Position Push", GroupName = "3. Options", Order = 1)]
        public bool EnablePositionPush { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Brackets", GroupName = "3. Options", Order = 2)]
        public bool EnableAutoBrackets { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "Stop Loss Ticks", GroupName = "3. Options", Order = 3)]
        public int StopLossTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "Take Profit Ticks", GroupName = "3. Options", Order = 4)]
        public int TakeProfitTicks { get; set; } = 40;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Two-way bridge - Silent when dashboard offline";
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
                    Print($"[FuturesDashboardBridge] Monitoring account: {myAccount.Name}");
                }

                StartSignalListener();
                SendPositionUpdate();   // one silent attempt at startup
            }
            else if (State == State.Terminated)
            {
                if (myAccount != null) myAccount.PositionUpdate -= OnPositionUpdate;
                StopSignalListener();
                httpClient?.Dispose();
            }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e) => SendPositionUpdate();

        private async void SendPositionUpdate()
        {
            if (!EnablePositionPush || myAccount == null) return;

            // Build JSON
            var sb = new StringBuilder();
            sb.Append("{\"account\":\"" + myAccount.Name + "\",\"positions\":[");
            bool first = true;
            foreach (Position pos in myAccount.Positions)
            {
                if (pos.Quantity == 0) continue;
                double pnl = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);
                if (!first) sb.Append(",");
                sb.Append("{\"symbol\":\"" + pos.Instrument.FullName + "\",\"side\":\"" + pos.MarketPosition + "\",\"quantity\":" + pos.Quantity + ",\"avgPrice\":" + pos.AveragePrice + ",\"unrealizedPnL\":" + pnl + ",\"lastUpdate\":\"" + DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") + "\"}");
                first = false;
            }
            sb.Append("],\"timestamp\":\"" + DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") + "\"}");

            try
            {
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                await httpClient.PostAsync(PositionUpdateUrl, content);

                if (!lastPushSuccess)
                {
                    Print("[FuturesDashboardBridge] ✅ Position push connection restored");
                    lastPushSuccess = true;
                }
            }
            catch (Exception ex)
            {
                if (lastPushSuccess || (DateTime.Now - lastErrorLog).TotalSeconds > 30)
                {
                    Print($"[FuturesDashboardBridge] Position push failed (will retry silently): {ex.Message}");
                    lastErrorLog = DateTime.Now;
                    lastPushSuccess = false;
                }
            }
        }

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
                Print($"[FuturesDashboardBridge] ✅ Listening on port {SignalListenerPort}");
            }
            catch (Exception ex)
            {
                Print($"[FuturesDashboardBridge] ❌ Listener failed on port {SignalListenerPort}: {ex.Message}");
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
                    if (context.Request.HttpMethod == "POST" && context.Request.Url.AbsolutePath == "/execute_signal")
                    {
                        using var reader = new System.IO.StreamReader(context.Request.InputStream);
                        string json = reader.ReadToEnd();
                        ProcessSignal(json);
                        SendResponse(context.Response, 200, "OK");
                    }
                    else
                        SendResponse(context.Response, 404, "Not found");
                }
                catch { }
            }
        }

        private void ProcessSignal(string json)
        {
            try
            {
                var serializer = new JavaScriptSerializer();
                dynamic signal = serializer.DeserializeObject(json);

                string dir = (signal["direction"] ?? "long").ToString().ToLower();
                int quantity = (int)Math.Round(Convert.ToDouble(signal["quantity"] ?? 1));
                string typeStr = (signal["order_type"] ?? "market").ToString().ToLower();

                OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
                OrderType ot = OrderType.Market;
                double limitPrice = 0, stopPrice = 0;

                if (typeStr == "limit") { ot = OrderType.Limit; limitPrice = Convert.ToDouble(signal["limit_price"] ?? 0); }
                else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = Convert.ToDouble(signal["limit_price"] ?? 0); }

                SubmitOrderUnmanaged(0, action, ot, quantity, limitPrice, stopPrice, "", "Dashboard-" + dir);

                if (EnableAutoBrackets)
                {
                    double entry = Close[0];
                    double sl = action == OrderAction.Buy ? entry - StopLossTicks * TickSize : entry + StopLossTicks * TickSize;
                    double tp = action == OrderAction.Buy ? entry + TakeProfitTicks * TickSize : entry - TakeProfitTicks * TickSize;

                    SubmitOrderUnmanaged(0, action == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy, OrderType.StopMarket, quantity, 0, sl, "OCO-SL", "SL-" + dir);
                    SubmitOrderUnmanaged(0, action == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy, OrderType.Limit, quantity, tp, 0, "OCO-TP", "TP-" + dir);
                }

                SendPositionUpdate();
                Print($"[FuturesDashboardBridge] Executed {dir.ToUpper()} {quantity}");
            }
            catch (Exception ex)
            {
                Print($"[FuturesDashboardBridge] Signal error: {ex.Message}");
            }
        }

        private void SendResponse(HttpListenerResponse resp, int code, string msg)
        {
            resp.StatusCode = code;
            byte[] buf = Encoding.UTF8.GetBytes(msg);
            resp.ContentLength64 = buf.Length;
            resp.OutputStream.Write(buf, 0, buf.Length);
            resp.OutputStream.Close();
        }

        protected override void OnBarUpdate() { }
    }
}
