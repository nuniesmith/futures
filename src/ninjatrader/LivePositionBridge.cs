#region Using declarations
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;  // Add reference if needed (see setup)
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class LivePositionBridge : Indicator
    {
        private Account myAccount;
        private HttpClient httpClient;
        private const string API_URL = "http://localhost:8000/update_positions";  // Change if remote

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Sends live account positions to web dashboard";
                Name = "LivePositionBridge";
                Calculate = Calculate.OnEachTick;  // Fast reaction
                IsOverlay = false;
                DisplayInDataBox = false;
            }
            else if (State == State.Configure)
            {
                httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                // Get default account (Sim101 or your live account name)
                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name.Contains("Sim") || a.Name.Contains("Live"));

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

            var positions = new List<object>();
            foreach (Position pos in myAccount.Positions)
            {
                if (pos.Quantity == 0) continue;

                positions.Add(new
                {
                    symbol = pos.Instrument.FullName,     // e.g. "MESZ5"
                    side = pos.MarketPosition.ToString(), // Long / Short
                    quantity = pos.Quantity,
                    avgPrice = pos.AveragePrice,
                    unrealizedPnL = pos.UnrealizedPnL,
                    lastUpdate = DateTime.UtcNow
                });
            }

            var payload = new { account = myAccount.Name, positions = positions, timestamp = DateTime.UtcNow };

            try
            {
                var json = JsonConvert.SerializeObject(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                await httpClient.PostAsync(API_URL, content);
                // Optional: Print to NT output window for debugging
                Print($"Positions sent: {positions.Count} open");
            }
            catch (Exception ex)
            {
                Print($"Position bridge error: {ex.Message}");
            }
        }

        protected override void OnBarUpdate() { }  // Not needed
    }
}
