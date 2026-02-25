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
                Print("[LivePositionBridge] Available accounts on this platform:");
                lock (Account.All)
                {
                    foreach (Account acc in Account.All)
                        Print($"   → {acc.Name}");
                }

                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name == AccountName);

                Print($"[LivePositionBridge] Monitoring selected account: {myAccount?.Name ?? "NONE FOUND - check dropdown"}");

                if (myAccount != null)
                    myAccount.PositionUpdate += OnPositionUpdate;
            }
            else if (State == State.Terminated)
            {
                if (myAccount != null)
                    myAccount.PositionUpdate -= OnPositionUpdate;
                httpClient?.Dispose();
            }
        }

        private async void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (myAccount == null) return;

            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"account\":\"{myAccount.Name}\",");
            sb.Append("\"positions\":[");

            bool first = true;
            foreach (Position pos in myAccount.Positions)
            {
                if (pos.Quantity == 0) continue;

                double unrealizedPnL = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);

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

            sb.Append("],");
            sb.Append($"\"timestamp\":\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"");
            sb.Append("}");

            try
            {
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                await httpClient.PostAsync(API_URL, content);
                Print($"✅ LivePositionBridge sent {myAccount.Positions.Count(p => p.Quantity != 0)} position(s) from {myAccount.Name}");
            }
            catch (Exception ex)
            {
                Print($"❌ LivePositionBridge error: {ex.Message}");
            }
        }

        protected override void OnBarUpdate() { }
    }
}
