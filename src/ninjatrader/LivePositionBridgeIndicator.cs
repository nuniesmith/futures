#region Using declarations
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

// === REQUIRED FOR DROPDOWN ===
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Gui.Tools;          // AccountNameConverter
// ============================

using NinjaTrader.Cbi;
using NinjaTrader.Data;               // PerformanceUnit
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class LivePositionBridge : Indicator
    {
        private Account myAccount;
        private HttpClient httpClient;
        private const string API_URL = "http://localhost:8000/update_positions";

        // Throttle: minimum seconds between error log messages
        private DateTime lastErrorLog = DateTime.MinValue;
        private const int ERROR_LOG_THROTTLE_SECONDS = 15;

        // Track push status for connection restored messages
        private bool lastPushSuccess = true;

        // ============== ACCOUNT DROPDOWN ==============
        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account to Monitor",
                 Description = "Select Sim or Live account",
                 Order = 1,
                 GroupName = "Parameters")]
        public string AccountName { get; set; } = "Sim101";
        // ==============================================

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Sends live positions to web dashboard with account selector";
                Name = "LivePositionBridge";
                Calculate = Calculate.OnEachTick;
                IsOverlay = false;
                DisplayInDataBox = false;
            }
            else if (State == State.Configure)
            {
                httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                // Show all accounts for easy debugging
                try
                {
                    Print("[LivePositionBridge] Available accounts on this platform:");
                    lock (Account.All)
                    {
                        foreach (Account acc in Account.All)
                            Print($"   → {acc.Name}");
                    }
                }
                catch { }

                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name == AccountName);

                Print($"[LivePositionBridge] Monitoring selected account: {myAccount?.Name ?? "NONE FOUND - check dropdown"}");

                if (myAccount != null)
                    myAccount.PositionUpdate += OnPositionUpdate;
            }
            else if (State == State.Terminated)
            {
                try
                {
                    if (myAccount != null)
                        myAccount.PositionUpdate -= OnPositionUpdate;
                }
                catch { }
                try { httpClient?.Dispose(); } catch { }
                httpClient = null;
            }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            try
            {
                if (myAccount == null) return;

                var client = httpClient;
                if (client == null) return;

                // Safely read Close[0] — may be null/invalid during session breaks
                double lastPrice = 0;
                try
                {
                    if (CurrentBar >= 0 && Close != null && Close.Count > 0)
                        lastPrice = Close[0];
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append($"\"account\":\"{myAccount.Name}\",");
                sb.Append("\"positions\":[");

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

                            double unrealizedPnL = 0;
                            try
                            {
                                if (lastPrice > 0)
                                    unrealizedPnL = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, lastPrice);
                            }
                            catch { }

                            if (!first) sb.Append(",");
                            sb.Append("{");
                            sb.Append($"\"symbol\":\"{pos.Instrument.FullName}\",");
                            sb.Append($"\"side\":\"{pos.MarketPosition}\",");
                            sb.Append($"\"quantity\":{pos.Quantity},");
                            sb.Append($"\"avgPrice\":{pos.AveragePrice},");
                            sb.Append($"\"unrealizedPnL\":{unrealizedPnL},");
                            sb.Append($"\"lastUpdate\":\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"");
                            sb.Append("}");
                            first = false;
                        }
                        catch (Exception ex)
                        {
                            ThrottledLog($"Error reading position: {ex.Message}");
                        }
                    }
                }

                sb.Append("],");
                sb.Append($"\"timestamp\":\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"");
                sb.Append("}");

                // Fire-and-forget: NT8 hates await in indicator event handlers
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                client.PostAsync(API_URL, content).ContinueWith(t =>
                {
                    if (t.IsFaulted)
                    {
                        if (lastPushSuccess || (DateTime.Now - lastErrorLog).TotalSeconds > 30)
                        {
                            lastErrorLog = DateTime.Now;
                            lastPushSuccess = false;
                        }
                    }
                    else
                    {
                        if (!lastPushSuccess)
                        {
                            lastPushSuccess = true;
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                ThrottledLog($"OnPositionUpdate error: {ex.Message}");
            }
        }

        protected override void OnBarUpdate() { }

        private void ThrottledLog(string message)
        {
            if ((DateTime.Now - lastErrorLog).TotalSeconds >= ERROR_LOG_THROTTLE_SECONDS)
            {
                Print($"[LivePositionBridge] {message}");
                lastErrorLog = DateTime.Now;
            }
        }
    }
}
