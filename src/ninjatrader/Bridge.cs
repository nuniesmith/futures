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
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class Bridge : Strategy
    {
        private Account myAccount;
        private HttpClient httpClient;
        private HttpListener listener;
        private Thread listenerThread;
        private bool lastPushSuccess = true;
        private DateTime lastErrorLog = DateTime.MinValue;
        private DateTime lastHeartbeat = DateTime.MinValue;

        // Ruby indicator reference — instantiated during Configure so that
        // Ruby.OnBarUpdate() runs on every bar, pushing signals into SignalBus.
        // This is what makes backtesting via Strategy Analyzer work.
        private Ruby rubyIndicator;

        // Order queue: signals are queued here, then processed on the main
        // thread in OnBarUpdate to avoid cross-thread NinjaScript exceptions.
        private readonly Queue<Action> orderQueue = new Queue<Action>();
        private readonly object queueLock = new object();

        private readonly object orderLock = new object();

        // Track recent order events for the /orders endpoint
        private readonly List<Dictionary<string, object>> recentOrderEvents = new List<Dictionary<string, object>>();
        private const int MAX_ORDER_EVENTS = 50;

        // Risk feedback from Python — when the risk engine says stop, we stop
        private volatile bool riskBlocked = false;
        private volatile string riskBlockReason = "";

        // Throttle: minimum seconds between error log messages
        private const int ERROR_LOG_THROTTLE_SECONDS = 15;

        // Heartbeat interval in seconds
        private const int HEARTBEAT_INTERVAL_SECONDS = 15;

        #region Properties
        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account to Monitor", GroupName = "1. Account", Order = 1)]
        public string AccountName { get; set; } = "Sim101";

        [NinjaScriptProperty]
        [Display(Name = "Dashboard Base URL", GroupName = "2. Web App", Order = 1,
                 Description = "Base URL of the Python data service (e.g. http://localhost:8000)")]
        public string DashboardBaseUrl { get; set; } = "http://localhost:8000";

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
        [Display(Name = "Enable Risk Enforcement", GroupName = "3. Options", Order = 3,
                 Description = "When true, the bridge will block new orders if the Python risk engine says to stop")]
        public bool EnableRiskEnforcement { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Default SL Ticks (fallback)", GroupName = "3. Options", Order = 4,
                 Description = "Used when signal JSON does not include stop_loss price")]
        public int DefaultStopLossTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "Default TP Ticks (fallback)", GroupName = "3. Options", Order = 5,
                 Description = "Used when signal JSON does not include take_profit price")]
        public int DefaultTakeProfitTicks { get; set; } = 40;

        #region Risk Management (Micro Contracts)
        [NinjaScriptProperty]
        [Display(Name = "Account Size ($)", GroupName = "4. Risk Management", Order = 1)]
        public double AccountSize { get; set; } = 50000;

        [NinjaScriptProperty]
        [Range(0.1, 2.0)]
        [Display(Name = "Risk % Per Trade", GroupName = "4. Risk Management", Order = 2)]
        public double RiskPercentPerTrade { get; set; } = 0.5;

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max Micro Contracts", GroupName = "4. Risk Management", Order = 3,
                 Description = "Hard cap on number of micro contracts per signal")]
        public int MaxMicroContracts { get; set; } = 10;
        #endregion

        [NinjaScriptProperty]
        [Display(Name = "Enable SignalBus", GroupName = "3. Options", Order = 6,
                 Description = "Consume in-process signals from Ruby indicator via SignalBus (required for backtest, also works live)")]
        public bool EnableSignalBus { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Attach Ruby Indicator", GroupName = "3. Options", Order = 7,
                 Description = "Instantiate Ruby indicator on the chart so its signals feed the SignalBus during backtests")]
        public bool AttachRuby { get; set; } = true;

        #region Ruby Indicator Parameters (forwarded)
        // These are forwarded to the Ruby indicator when AttachRuby is true.
        // They mirror Ruby's ORB parameters so you can tune from Bridge's property grid.
        [NinjaScriptProperty]
        [Display(Name = "Ruby: Session Bias", GroupName = "5. Ruby ORB", Order = 1)]
        public RubySessionBias RubySessionBias { get; set; } = RubySessionBias.Auto;

        [NinjaScriptProperty]
        [Range(5, 120)]
        [Display(Name = "Ruby: ORB Minutes", GroupName = "5. Ruby ORB", Order = 2)]
        public int RubyORB_Minutes { get; set; } = 30;

        [NinjaScriptProperty]
        [Range(30, 95)]
        [Display(Name = "Ruby: Min Quality %", GroupName = "5. Ruby ORB", Order = 3)]
        public int RubyORB_MinQuality { get; set; } = 60;

        [NinjaScriptProperty]
        [Range(0.5, 3.0)]
        [Display(Name = "Ruby: Volume Gate (x avg)", GroupName = "5. Ruby ORB", Order = 4)]
        public double RubyORB_VolumeGate { get; set; } = 1.2;

        [NinjaScriptProperty]
        [Display(Name = "Ruby: Require VWAP Cross", GroupName = "5. Ruby ORB", Order = 5)]
        public bool RubyORB_RequireVWAPCross { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Ruby: Allow ADD Signal", GroupName = "5. Ruby ORB", Order = 6)]
        public bool RubyORB_AllowAdd { get; set; } = true;

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Ruby: SL ATR Mult", GroupName = "5. Ruby ORB", Order = 7)]
        public double RubySL_ATR_Mult { get; set; } = 1.5;

        [NinjaScriptProperty]
        [Range(0.5, 10.0)]
        [Display(Name = "Ruby: TP1 ATR Mult", GroupName = "5. Ruby ORB", Order = 8)]
        public double RubyTP1_ATR_Mult { get; set; } = 2.0;

        [NinjaScriptProperty]
        [Range(0.5, 15.0)]
        [Display(Name = "Ruby: TP2 ATR Mult", GroupName = "5. Ruby ORB", Order = 9)]
        public double RubyTP2_ATR_Mult { get; set; } = 3.5;

        [NinjaScriptProperty]
        [Range(1, 60)]
        [Display(Name = "Ruby: Signal Cooldown (min)", GroupName = "5. Ruby ORB", Order = 10)]
        public int RubySignalCooldownMinutes { get; set; } = 5;
        #endregion

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Two-way bridge: pushes live positions/account to dashboard, receives order signals with targets and risk enforcement. Attach Ruby indicator for ORB backtest.";
                Name = "Bridge";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                IsUnmanaged = true;
                EnableSignalBus = true;
                AttachRuby = true;
            }
            else if (State == State.Configure)
            {
                httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                // Attach Ruby indicator so it runs on every bar and pushes
                // signals to SignalBus.  This is what makes backtesting work —
                // without this, Ruby never executes during Strategy Analyzer.
                if (AttachRuby && EnableSignalBus)
                {
                    rubyIndicator = Ruby(
                        /* SR_Lookback */        20,
                        /* AO_Fast */            5,
                        /* AO_Slow */            34,
                        /* WaveLookback */       200,
                        /* MinWaveRatio */       1.5,
                        /* RegressionLength */   200,
                        /* VP_Lookback */        100,
                        /* VP_Bins */            40,
                        /* ValueAreaPct */       70,
                        /* SessionPOC_MaxDays */ 5,
                        /* NakedPOC_Enabled */   true,
                        /* VolumeAvgPeriod */    20,
                        /* VolumeSpikeMult */    1.8,
                        /* VolumeLowMult */      0.5,
                        /* LowVolStreakBars */   3,
                        /* AbsorptionVolMult */  1.5,
                        /* AbsorptionBodyRatio */ 30,
                        /* CVD_AnchorDaily */    true,
                        /* ShowEMA9 */           true,
                        /* ShowBollingerBands */ false,
                        /* ShowAdaptiveSR */     false,
                        /* ShowLabels */         true,
                        /* ShowVolumeLabels */   false,
                        /* HeatSensitivity */    70,
                        /* SignalCooldownMin */   RubySignalCooldownMinutes,
                        /* ShowVWAP */           true,
                        /* ShowVWAPBands */      false,
                        /* ShowPOC */            true,
                        /* ShowValueArea */      false,
                        /* ShowDeltaOutline */   false,
                        /* ShowOpeningRange */   true,
                        /* SessionBias */        RubySessionBias,
                        /* ORB_Minutes */        RubyORB_Minutes,
                        /* ORB_MinQuality */     RubyORB_MinQuality,
                        /* ORB_VolumeGate */     RubyORB_VolumeGate,
                        /* ORB_RequireVWAPCross */ RubyORB_RequireVWAPCross,
                        /* ORB_AllowAdd */       RubyORB_AllowAdd,
                        /* ORB_AddPullbackATR */ 0.5,
                        /* ORB_MaxAddBarsAfterBreakout */ 30,
                        /* SendSignalsToBridge */ true,
                        /* BridgeUrl */          "http://localhost:" + SignalListenerPort,
                        /* ExitOnReversal */     true,
                        /* ExitOnBBTouch */      true,
                        /* ExitCooldownMinutes */ 3,
                        /* SL_ATR_Mult */        RubySL_ATR_Mult,
                        /* TP1_ATR_Mult */       RubyTP1_ATR_Mult,
                        /* TP2_ATR_Mult */       RubyTP2_ATR_Mult
                    );
                    Print("[Bridge] Ruby indicator attached for SignalBus integration");
                }

                lock (Account.All)
                    myAccount = Account.All.FirstOrDefault(a => a.Name == AccountName);

                if (myAccount != null)
                {
                    myAccount.PositionUpdate += OnPositionUpdate;
                    myAccount.OrderUpdate += OnOrderUpdate;
                    Print($"[Bridge] Monitoring account: {myAccount.Name}");
                }

                // Register as a SignalBus consumer so Ruby (and other indicators)
                // can forward signals in-process — works in both live and backtest.
                if (EnableSignalBus)
                {
                    SignalBus.Reset();
                    SignalBus.RegisterConsumer();
                    Print("[Bridge] SignalBus consumer registered");
                }

                // Only start HTTP listener in live/sim — it would fail or be
                // pointless during a Strategy Analyzer backtest.
                // (Listener start is deferred to State.Realtime below.)
            }
            else if (State == State.Realtime)
            {
                // Start the HTTP signal listener only when we reach Realtime.
                // During Historical (backtest) the listener is not needed
                // because signals arrive via SignalBus.
                StartSignalListener();
                SendPositionUpdate();
                Print("[Bridge] Realtime — HTTP listener started");
            }
            else if (State == State.Terminated)
            {
                try
                {
                    if (myAccount != null)
                    {
                        myAccount.PositionUpdate -= OnPositionUpdate;
                        myAccount.OrderUpdate -= OnOrderUpdate;
                    }
                }
                catch { }

                if (EnableSignalBus)
                {
                    SignalBus.UnregisterConsumer();
                    Print($"[Bridge] SignalBus unregistered (total enqueued={SignalBus.TotalEnqueued}, drained={SignalBus.TotalDrained})");
                }

                StopSignalListener();
                try { httpClient?.Dispose(); } catch { }
                httpClient = null;
                rubyIndicator = null;
            }
        }

        #region Order Tracking
        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            try
            {
                if (e.Order == null) return;

                var orderEvent = new Dictionary<string, object>
                {
                    { "orderId", e.Order.OrderId },
                    { "name", e.Order.Name ?? "" },
                    { "instrument", e.Order.Instrument != null ? e.Order.Instrument.FullName : "" },
                    { "action", e.Order.OrderAction.ToString() },
                    { "type", e.Order.OrderType.ToString() },
                    { "quantity", e.Order.Quantity },
                    { "filled", e.Order.Filled },
                    { "avgFillPrice", e.Order.AverageFillPrice },
                    { "limitPrice", e.Order.LimitPrice },
                    { "stopPrice", e.Order.StopPrice },
                    { "state", e.Order.OrderState.ToString() },
                    { "time", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") }
                };

                lock (orderLock)
                {
                    recentOrderEvents.Add(orderEvent);
                    while (recentOrderEvents.Count > MAX_ORDER_EVENTS)
                        recentOrderEvents.RemoveAt(0);
                }

                // Log significant state changes
                if (e.Order.OrderState == NinjaTrader.Cbi.OrderState.Filled)
                {
                    Print($"[Bridge] Order FILLED: {e.Order.Name} {e.Order.OrderAction} x{e.Order.Filled} @ {e.Order.AverageFillPrice:F2}");
                    // Push updated positions after a fill
                    SendPositionUpdate();
                }
                else if (e.Order.OrderState == NinjaTrader.Cbi.OrderState.Rejected)
                {
                    Print($"[Bridge] Order REJECTED: {e.Order.Name} — check NinjaTrader log");
                }
                else if (e.Order.OrderState == NinjaTrader.Cbi.OrderState.Cancelled)
                {
                    Print($"[Bridge] Order CANCELLED: {e.Order.Name}");
                }
            }
            catch (Exception ex)
            {
                ThrottledLog($"OnOrderUpdate error: {ex.Message}");
            }
        }
        #endregion

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

                // Gather account-level info
                double cashBalance = 0;
                double realizedPnL = 0;
                try
                {
                    cashBalance = myAccount.Get(AccountItem.CashValue, Currency.UsDollar);
                    realizedPnL = myAccount.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(myAccount.Name).Append("\",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realizedPnL\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"bridge_version\":\"2.0\",");
                sb.Append("\"positions\":[");

                bool first = true;
                double totalUnrealizedPnL = 0;

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
                            try
                            {
                                if (lastPrice > 0)
                                    pnl = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, lastPrice);
                            }
                            catch { }

                            totalUnrealizedPnL += pnl;

                            if (!first) sb.Append(",");
                            sb.Append("{");
                            sb.Append("\"symbol\":\"").Append(pos.Instrument.FullName).Append("\",");
                            sb.Append("\"side\":\"").Append(pos.MarketPosition).Append("\",");
                            sb.Append("\"quantity\":").Append(pos.Quantity).Append(",");
                            sb.Append("\"avgPrice\":").Append(pos.AveragePrice).Append(",");
                            sb.Append("\"unrealizedPnL\":").Append(Math.Round(pnl, 2)).Append(",");
                            sb.Append("\"instrument\":\"").Append(pos.Instrument.MasterInstrument.Name).Append("\",");
                            sb.Append("\"tickSize\":").Append(pos.Instrument.MasterInstrument.TickSize).Append(",");
                            sb.Append("\"pointValue\":").Append(pos.Instrument.MasterInstrument.PointValue).Append(",");
                            sb.Append("\"lastUpdate\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                            sb.Append("}");
                            first = false;
                        }
                        catch (Exception ex) { ThrottledLog($"Error reading position: {ex.Message}"); }
                    }
                }

                sb.Append("],");

                // Pending orders section
                sb.Append("\"pendingOrders\":[");
                try
                {
                    bool firstOrder = true;
                    var orders = myAccount.Orders;
                    if (orders != null)
                    {
                        foreach (Order ord in orders)
                        {
                            try
                            {
                                if (ord == null) continue;
                                if (ord.OrderState != NinjaTrader.Cbi.OrderState.Working
                                    && ord.OrderState != NinjaTrader.Cbi.OrderState.Accepted)
                                    continue;

                                if (!firstOrder) sb.Append(",");
                                sb.Append("{");
                                sb.Append("\"orderId\":\"").Append(ord.OrderId).Append("\",");
                                sb.Append("\"name\":\"").Append(EscapeJson(ord.Name ?? "")).Append("\",");
                                sb.Append("\"instrument\":\"").Append(ord.Instrument != null ? ord.Instrument.FullName : "").Append("\",");
                                sb.Append("\"action\":\"").Append(ord.OrderAction).Append("\",");
                                sb.Append("\"type\":\"").Append(ord.OrderType).Append("\",");
                                sb.Append("\"quantity\":").Append(ord.Quantity).Append(",");
                                sb.Append("\"limitPrice\":").Append(ord.LimitPrice).Append(",");
                                sb.Append("\"stopPrice\":").Append(ord.StopPrice).Append(",");
                                sb.Append("\"state\":\"").Append(ord.OrderState).Append("\"");
                                sb.Append("}");
                                firstOrder = false;
                            }
                            catch { }
                        }
                    }
                }
                catch { }
                sb.Append("],");

                sb.Append("\"totalUnrealizedPnL\":").Append(Math.Round(totalUnrealizedPnL, 2)).Append(",");
                sb.Append("\"riskBlocked\":").Append(riskBlocked ? "true" : "false").Append(",");
                sb.Append("\"riskBlockReason\":\"").Append(EscapeJson(riskBlockReason)).Append("\",");
                sb.Append("\"timestamp\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");

                string url = DashboardBaseUrl.TrimEnd('/') + "/positions/update";
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                client.PostAsync(url, content).ContinueWith(t =>
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
                            Print("[Bridge] Position push connection restored");
                            lastPushSuccess = true;
                        }

                        // Process risk feedback from the Python response
                        ProcessRiskFeedback(t.Result);
                    }
                });
            }
            catch (Exception ex) { ThrottledLog($"SendPositionUpdate error: {ex.Message}"); }
        }

        /// <summary>
        /// Parse the risk feedback from the Python /positions/update response.
        /// The response JSON includes:
        ///   { "risk": { "can_trade": true/false, "block_reason": "...", "warnings": [...] } }
        /// </summary>
        private void ProcessRiskFeedback(HttpResponseMessage response)
        {
            if (!EnableRiskEnforcement) return;

            try
            {
                string body = response.Content.ReadAsStringAsync().Result;
                if (string.IsNullOrEmpty(body)) return;

                var serializer = new JavaScriptSerializer();
                var result = serializer.Deserialize<Dictionary<string, object>>(body);

                if (result != null && result.ContainsKey("risk"))
                {
                    var riskObj = result["risk"];
                    if (riskObj is Dictionary<string, object> risk)
                    {
                        bool canTrade = true;
                        if (risk.ContainsKey("can_trade"))
                        {
                            try { canTrade = Convert.ToBoolean(risk["can_trade"]); } catch { }
                        }

                        string blockReason = "";
                        if (risk.ContainsKey("block_reason") && risk["block_reason"] != null)
                            blockReason = risk["block_reason"].ToString();

                        bool wasBlocked = riskBlocked;
                        riskBlocked = !canTrade;
                        riskBlockReason = blockReason;

                        if (!canTrade && !wasBlocked)
                            Print($"[Bridge] ⚠️ Risk BLOCKED: {blockReason}");
                        else if (canTrade && wasBlocked)
                            Print("[Bridge] ✅ Risk block cleared — trading allowed");

                        // Log warnings
                        if (risk.ContainsKey("warnings") && risk["warnings"] is System.Collections.ArrayList warnings)
                        {
                            foreach (var w in warnings)
                            {
                                if (w != null)
                                    Print($"[Bridge] ⚠️ Risk warning: {w}");
                            }
                        }
                    }
                }
            }
            catch { } // Non-fatal — don't let risk parsing crash the position push
        }
        #endregion

        #region Heartbeat
        private void SendHeartbeat()
        {
            var client = httpClient;
            if (client == null || myAccount == null) return;

            try
            {
                double cashBalance = 0;
                try { cashBalance = myAccount.Get(AccountItem.CashValue, Currency.UsDollar); } catch { }

                int openPositionCount = 0;
                try
                {
                    if (myAccount.Positions != null)
                        openPositionCount = myAccount.Positions.Count(p => p.Quantity != 0);
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(myAccount.Name).Append("\",");
                sb.Append("\"state\":\"").Append(State).Append("\",");
                sb.Append("\"connected\":true,");
                sb.Append("\"positions\":").Append(openPositionCount).Append(",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"riskBlocked\":").Append(riskBlocked ? "true" : "false").Append(",");
                sb.Append("\"bridge_version\":\"2.0\",");
                sb.Append("\"listenerPort\":").Append(SignalListenerPort).Append(",");
                sb.Append("\"timestamp\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");

                string url = DashboardBaseUrl.TrimEnd('/') + "/positions/heartbeat";
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                client.PostAsync(url, content).ContinueWith(t =>
                {
                    // Silently ignore heartbeat failures — position push already handles reconnection logging
                });
            }
            catch { }
        }
        #endregion

        #region Order Execution
        protected override void OnBarUpdate()
        {
            try
            {
                // ── SignalBus: drain in-process signals from Ruby / other indicators ──
                // This runs in ALL states (Historical, Realtime, etc.) so that
                // backtesting via Strategy Analyzer works correctly.
                if (EnableSignalBus)
                    DrainSignalBus();

                // Process queued order actions on the main thread.
                // In Realtime the queue is fed by the HTTP listener and by DrainSignalBus.
                // In Historical the queue is fed only by DrainSignalBus (for legacy path).
                bool canSubmit = State == State.Realtime || State == State.Historical;
                if (canSubmit)
                {
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

                if (State != State.Realtime && State != State.Historical) return;

                // Periodic heartbeat (live/sim only — skip during backtest)
                if (State == State.Realtime
                    && (DateTime.Now - lastHeartbeat).TotalSeconds >= HEARTBEAT_INTERVAL_SECONDS)
                {
                    lastHeartbeat = DateTime.Now;
                    SendHeartbeat();
                }
            }
            catch (Exception ex) { ThrottledLog($"OnBarUpdate error: {ex.Message}"); }
        }

        /// <summary>
        /// Drain all pending signals from the static SignalBus queue and route
        /// them to ProcessSignal (entries) or FlattenAll (exits).
        ///
        /// This enables the Ruby indicator (or any other producer) to forward
        /// signals to Bridge without HTTP — critical for Strategy Analyzer
        /// backtests where the HTTP listener is not running.
        ///
        /// In Historical (backtest) mode, orders are submitted directly on the
        /// current OnBarUpdate call rather than being queued, because:
        ///   1. We are already on the main NinjaScript thread.
        ///   2. The queue-dequeue lambda in ProcessSignal guards with
        ///      `if (State != State.Realtime) return;` which would silently
        ///      discard every order during a backtest.
        ///
        /// In Realtime mode, the normal ProcessSignal → queue → dequeue path
        /// is used so that order submission happens on the correct thread.
        /// </summary>
        private void DrainSignalBus()
        {
            var signals = SignalBus.DrainAll();
            if (signals.Count == 0) return;

            foreach (var sig in signals)
            {
                try
                {
                    if (sig.SignalType == "exit")
                    {
                        string reason = !string.IsNullOrEmpty(sig.ExitReason) ? sig.ExitReason : "signal_bus_exit";
                        Print($"[Bridge] SignalBus EXIT: reason={reason} strategy={sig.Strategy} asset={sig.Asset}");

                        if (State == State.Historical)
                            ExecuteFlattenDirect($"Ruby:{reason}");
                        else
                            FlattenAll($"Ruby:{reason}");
                    }
                    else
                    {
                        Print($"[Bridge] SignalBus ENTRY: {sig.Direction} strategy={sig.Strategy} Q={sig.SignalQuality:P0} id={sig.SignalId}");

                        if (State == State.Historical)
                            ExecuteEntryDirect(sig);
                        else
                            ProcessSignal(sig.ToJson());
                    }
                }
                catch (Exception ex)
                {
                    ThrottledLog($"SignalBus processing error: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Submit an entry order directly (no queue) — used during Historical
        /// backtest mode where we are already on the main NinjaScript thread
        /// and the queued-lambda approach would be skipped.
        ///
        /// Performs the same risk-sizing logic as ProcessSignal but executes
        /// SubmitOrderUnmanaged immediately.
        /// </summary>
        private void ExecuteEntryDirect(SignalBus.Signal sig)
        {
            string dir = (sig.Direction ?? "long").ToLower();
            int requestedQty = sig.Quantity > 0 ? sig.Quantity : 1;
            string typeStr = (sig.OrderType ?? "market").ToLower();
            double limitPrice = sig.LimitPrice;
            double slPrice = sig.StopLoss;
            double tpPrice = sig.TakeProfit;
            double tp2Price = sig.TakeProfit2;
            string signalId = sig.SignalId ?? Guid.NewGuid().ToString("N").Substring(0, 8);

            OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
            OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderType ot = OrderType.Market;
            double stopPrice = 0;

            if (typeStr == "limit") ot = OrderType.Limit;
            else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

            // === RISK SIZING (same logic as ProcessSignal) ===
            double currentBalance = AccountSize;
            try
            {
                if (myAccount != null)
                    currentBalance = myAccount.Get(AccountItem.CashValue, Currency.UsDollar);
            }
            catch { }

            double riskDollars = currentBalance * (RiskPercentPerTrade / 100.0);
            double entry = SafeGetClose();

            double slDistancePoints = slPrice > 0 && entry > 0
                ? Math.Abs(entry - slPrice)
                : DefaultStopLossTicks * TickSize;

            double pointValue = 10;
            try
            {
                if (Instrument != null && Instrument.MasterInstrument.PointValue > 0)
                    pointValue = Instrument.MasterInstrument.PointValue;
            }
            catch { }

            double riskPerContract = slDistancePoints * pointValue;
            int riskBasedQty = riskPerContract > 0
                ? (int)Math.Floor(riskDollars / riskPerContract)
                : 1;

            int finalQty = Math.Max(1, Math.Min(requestedQty, Math.Min(riskBasedQty, MaxMicroContracts)));

            Print($"[Bridge BT] {dir.ToUpper()} x{finalQty} (req={requestedQty}, risk={riskBasedQty}, cap={MaxMicroContracts}) id={signalId}");

            string entryName = $"Signal-{dir}-{signalId}";

            // Submit entry directly — we are on the main thread during backtest
            SubmitOrderUnmanaged(0, action, ot, finalQty, limitPrice, stopPrice, "", entryName);

            // Submit bracket orders (SL + TP)
            if (EnableAutoBrackets)
            {
                double bracketEntry = SafeGetClose();
                if (bracketEntry <= 0) return;

                double sl;
                if (slPrice > 0)
                    sl = slPrice;
                else
                    sl = dir == "long"
                        ? bracketEntry - DefaultStopLossTicks * TickSize
                        : bracketEntry + DefaultStopLossTicks * TickSize;

                double tp;
                if (tpPrice > 0)
                    tp = tpPrice;
                else
                    tp = dir == "long"
                        ? bracketEntry + DefaultTakeProfitTicks * TickSize
                        : bracketEntry - DefaultTakeProfitTicks * TickSize;

                int slQty = finalQty;
                int tp1Qty = tp2Price > 0 ? Math.Max(1, finalQty / 2) : finalQty;
                int tp2Qty = tp2Price > 0 ? finalQty - tp1Qty : 0;

                string oco = $"OCO-{signalId}";

                SubmitOrderUnmanaged(0, exitAction, OrderType.StopMarket, slQty, 0, sl, oco, $"SL-{signalId}");
                SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp1Qty, tp, 0, oco, $"TP1-{signalId}");

                if (tp2Price > 0 && tp2Qty > 0)
                    SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp2Qty, tp2Price, 0, "", $"TP2-{signalId}");

                Print($"[Bridge BT] Brackets: SL={sl:F2} TP1={tp:F2}" + (tp2Price > 0 ? $" TP2={tp2Price:F2}" : ""));
            }
        }

        /// <summary>
        /// Flatten all positions directly (no queue) — used during Historical
        /// backtest mode.  Uses the strategy's own Position object (not
        /// myAccount.Positions which is empty during Strategy Analyzer runs).
        /// Submits a market order in the opposite direction to close.
        /// </summary>
        private void ExecuteFlattenDirect(string reason)
        {
            try
            {
                // During backtest, use the strategy's own Position property
                // which tracks the simulated position on the primary instrument.
                if (Position != null && Position.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction closeAction = Position.MarketPosition == MarketPosition.Long
                        ? OrderAction.Sell
                        : OrderAction.BuyToCover;

                    int qty = Position.Quantity;

                    SubmitOrderUnmanaged(0, closeAction, OrderType.Market, qty, 0, 0, "", $"Flatten-{reason}");
                    Print($"[Bridge BT] Flattening {Position.MarketPosition} x{qty} reason={reason}");
                }
                else
                {
                    // Fallback: try account-level positions (works in live/sim)
                    if (myAccount != null && myAccount.Positions != null)
                    {
                        foreach (Position pos in myAccount.Positions)
                        {
                            if (pos == null || pos.Quantity == 0 || pos.Instrument == null) continue;

                            OrderAction closeAction = pos.MarketPosition == MarketPosition.Long
                                ? OrderAction.Sell
                                : OrderAction.BuyToCover;

                            SubmitOrderUnmanaged(0, closeAction, OrderType.Market, pos.Quantity, 0, 0, "", $"Flatten-{pos.Instrument.FullName}");
                            Print($"[Bridge BT] Flattening {pos.Instrument.FullName} {pos.MarketPosition} x{pos.Quantity} reason={reason}");
                        }
                    }
                }

                Print($"[Bridge BT] 🔴 FLATTEN ALL — reason: {reason}");
            }
            catch (Exception ex)
            {
                Print($"[Bridge BT] Flatten error: {ex.Message}");
            }
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
        ///   "tp2": 5240.00,           // exact TP2 price (optional — no fallback)
        ///   "strategy": "TrendEMA",   // strategy name for logging
        ///   "asset": "Gold",          // asset name for logging
        ///   "signal_id": "abc123"     // unique signal ID for tracking
        /// }
        /// </summary>
        private Dictionary<string, object> ProcessSignal(string json)
        {
            var response = new Dictionary<string, object>();

            try
            {
                var serializer = new JavaScriptSerializer();
                var signal = serializer.Deserialize<Dictionary<string, object>>(json);

                string signalId = GetSignalString(signal, "signal_id", Guid.NewGuid().ToString("N").Substring(0, 8));
                string dir = GetSignalString(signal, "direction", "long").ToLower();
                int requestedQty = GetSignalInt(signal, "quantity", 1);
                string typeStr = GetSignalString(signal, "order_type", "market").ToLower();
                double limitPrice = GetSignalDouble(signal, "limit_price", 0);
                double slPrice = GetSignalDouble(signal, "stop_loss", 0);
                double tpPrice = GetSignalDouble(signal, "take_profit", 0);
                double tp2Price = GetSignalDouble(signal, "tp2", 0);
                string strategy = GetSignalString(signal, "strategy", "");
                string asset = GetSignalString(signal, "asset", "");

                // Check risk enforcement
                if (EnableRiskEnforcement && riskBlocked)
                {
                    string msg = $"Signal rejected — risk blocked: {riskBlockReason}";
                    Print($"[Bridge] ⚠️ {msg}");
                    response["status"] = "rejected";
                    response["reason"] = msg;
                    response["signal_id"] = signalId;
                    return response;
                }

                OrderAction action = dir == "long" ? OrderAction.Buy : OrderAction.SellShort;
                OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.Buy;
                OrderType ot = OrderType.Market;
                double stopPrice = 0;

                if (typeStr == "limit") ot = OrderType.Limit;
                else if (typeStr == "stop") { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

                // === RISK MANAGEMENT FOR MICRO CONTRACTS ===
                double currentBalance = AccountSize;
                try
                {
                    if (myAccount != null)
                        currentBalance = myAccount.Get(AccountItem.CashValue, Currency.UsDollar);
                }
                catch { }

                double riskDollars = currentBalance * (RiskPercentPerTrade / 100.0);
                double entry = SafeGetClose();

                double slDistancePoints = slPrice > 0 && entry > 0
                    ? Math.Abs(entry - slPrice)
                    : DefaultStopLossTicks * TickSize;

                double pointValue = 10; // safe default for most micros
                try
                {
                    if (Instrument != null && Instrument.MasterInstrument.PointValue > 0)
                        pointValue = Instrument.MasterInstrument.PointValue;
                }
                catch { }

                double riskPerContract = slDistancePoints * pointValue;
                int riskBasedQty = riskPerContract > 0
                    ? (int)Math.Floor(riskDollars / riskPerContract)
                    : 1;

                int finalQty = Math.Max(1, Math.Min(requestedQty, Math.Min(riskBasedQty, MaxMicroContracts)));

                Print($"[Bridge] Signal {signalId}: {dir.ToUpper()} x{finalQty} (requested {requestedQty}, risk-sized {riskBasedQty}, cap {MaxMicroContracts})");
                Print($"[Bridge Risk] Balance:${currentBalance:0} | Risk:${riskDollars:0} ({RiskPercentPerTrade}%) | SL:{slDistancePoints:F2}pts | PointVal:{pointValue}");

                // Capture for closure
                int capturedQty = finalQty;
                string capturedDir = dir;
                double capturedSl = slPrice;
                double capturedTp = tpPrice;
                double capturedTp2 = tp2Price;
                double capturedLimitPrice = limitPrice;
                double capturedStopPrice = stopPrice;
                string capturedSignalId = signalId;
                string capturedStrategy = strategy;

                lock (queueLock)
                {
                    orderQueue.Enqueue(() =>
                    {
                        if (State != State.Realtime) return;

                        string entryName = $"Signal-{capturedDir}-{capturedSignalId}";

                        // Submit entry order
                        SubmitOrderUnmanaged(0, action, ot, capturedQty, capturedLimitPrice, capturedStopPrice, "", entryName);

                        // Submit bracket orders (SL + TP)
                        if (EnableAutoBrackets)
                        {
                            double bracketEntry = SafeGetClose();
                            if (bracketEntry <= 0) return;

                            // Stop Loss: use exact price if provided, else fall back to tick offset
                            double sl;
                            if (capturedSl > 0)
                                sl = capturedSl;
                            else
                                sl = capturedDir == "long"
                                    ? bracketEntry - DefaultStopLossTicks * TickSize
                                    : bracketEntry + DefaultStopLossTicks * TickSize;

                            // Take Profit 1: use exact price if provided, else fall back to tick offset
                            double tp;
                            if (capturedTp > 0)
                                tp = capturedTp;
                            else
                                tp = capturedDir == "long"
                                    ? bracketEntry + DefaultTakeProfitTicks * TickSize
                                    : bracketEntry - DefaultTakeProfitTicks * TickSize;

                            // Determine quantities for split targets
                            int slQty = capturedQty;
                            int tp1Qty = capturedTp2 > 0 ? Math.Max(1, capturedQty / 2) : capturedQty;
                            int tp2Qty = capturedTp2 > 0 ? capturedQty - tp1Qty : 0;

                            string oco = $"OCO-{capturedSignalId}";

                            // SL covers full position
                            SubmitOrderUnmanaged(0, exitAction, OrderType.StopMarket, slQty, 0, sl, oco, $"SL-{capturedSignalId}");

                            // TP1
                            SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp1Qty, tp, 0, oco, $"TP1-{capturedSignalId}");

                            // TP2 (only if provided via exact price)
                            if (capturedTp2 > 0 && tp2Qty > 0)
                                SubmitOrderUnmanaged(0, exitAction, OrderType.Limit, tp2Qty, capturedTp2, 0, "", $"TP2-{capturedSignalId}");

                            Print($"[Bridge] Brackets set: SL={sl:F2} TP1={tp:F2}" + (capturedTp2 > 0 ? $" TP2={capturedTp2:F2}" : ""));
                        }

                        // Push updated positions after signal processing
                        SendPositionUpdate();
                        Print($"[Bridge] ✅ Executed {capturedDir.ToUpper()} x{capturedQty} [{capturedStrategy}] id={capturedSignalId}");
                    });
                }

                response["status"] = "queued";
                response["signal_id"] = signalId;
                response["direction"] = dir;
                response["quantity"] = finalQty;
                response["requested_quantity"] = requestedQty;
                response["risk_sized_quantity"] = riskBasedQty;
                response["strategy"] = strategy;
            }
            catch (Exception ex)
            {
                ThrottledLog($"Signal error: {ex.Message}");
                response["status"] = "error";
                response["error"] = ex.Message;
            }

            return response;
        }

        /// <summary>
        /// Flatten all positions — close everything at market.
        /// Called from /flatten endpoint.
        /// </summary>
        private Dictionary<string, object> FlattenAll(string reason)
        {
            var response = new Dictionary<string, object>();

            try
            {
                int positionCount = 0;

                lock (queueLock)
                {
                    orderQueue.Enqueue(() =>
                    {
                        if (State != State.Realtime) return;

                        try
                        {
                            if (myAccount != null && myAccount.Positions != null)
                            {
                                foreach (Position pos in myAccount.Positions)
                                {
                                    if (pos == null || pos.Quantity == 0 || pos.Instrument == null) continue;

                                    OrderAction closeAction = pos.MarketPosition == MarketPosition.Long
                                        ? OrderAction.Sell
                                        : OrderAction.BuyToCover;

                                    SubmitOrderUnmanaged(0, closeAction, OrderType.Market, pos.Quantity, 0, 0, "", $"Flatten-{pos.Instrument.FullName}");
                                    Print($"[Bridge] Flattening {pos.Instrument.FullName} {pos.MarketPosition} x{pos.Quantity}");
                                }
                            }

                            // Cancel all working orders
                            if (myAccount != null && myAccount.Orders != null)
                            {
                                foreach (Order ord in myAccount.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working
                                        || ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                    {
                                        myAccount.Cancel(new[] { ord });
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"[Bridge] Flatten error: {ex.Message}");
                        }

                        SendPositionUpdate();
                        Print($"[Bridge] 🔴 FLATTEN ALL — reason: {reason}");
                    });
                }

                try
                {
                    if (myAccount?.Positions != null)
                        positionCount = myAccount.Positions.Count(p => p.Quantity != 0);
                }
                catch { }

                response["status"] = "flatten_queued";
                response["positions_to_close"] = positionCount;
                response["reason"] = reason;
            }
            catch (Exception ex)
            {
                response["status"] = "error";
                response["error"] = ex.Message;
            }

            return response;
        }

        /// <summary>
        /// Cancel all working orders without closing positions.
        /// </summary>
        private Dictionary<string, object> CancelAllOrders()
        {
            var response = new Dictionary<string, object>();

            try
            {
                int cancelCount = 0;

                lock (queueLock)
                {
                    orderQueue.Enqueue(() =>
                    {
                        if (State != State.Realtime) return;

                        try
                        {
                            if (myAccount != null && myAccount.Orders != null)
                            {
                                foreach (Order ord in myAccount.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working
                                        || ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                    {
                                        myAccount.Cancel(new[] { ord });
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"[Bridge] Cancel all orders error: {ex.Message}");
                        }

                        Print("[Bridge] Cancel all working orders requested");
                    });
                }

                try
                {
                    if (myAccount?.Orders != null)
                    {
                        cancelCount = myAccount.Orders.Count(o =>
                            o.OrderState == NinjaTrader.Cbi.OrderState.Working
                            || o.OrderState == NinjaTrader.Cbi.OrderState.Accepted);
                    }
                }
                catch { }

                response["status"] = "cancel_queued";
                response["orders_to_cancel"] = cancelCount;
            }
            catch (Exception ex)
            {
                response["status"] = "error";
                response["error"] = ex.Message;
            }

            return response;
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
            var serializer = new JavaScriptSerializer();

            while (listener?.IsListening == true)
            {
                try
                {
                    var context = listener.GetContext();

                    // Add CORS headers to every response so the browser dashboard
                    // (or Python proxy) can call the Bridge directly if needed.
                    AddCorsHeaders(context.Response, context.Request);

                    // Handle preflight OPTIONS requests
                    if (context.Request.HttpMethod == "OPTIONS")
                    {
                        SendResponse(context.Response, 204, "");
                        continue;
                    }

                    string path = context.Request.Url.AbsolutePath;
                    string method = context.Request.HttpMethod;

                    if (method == "POST" && path == "/execute_signal")
                    {
                        using (var reader = new System.IO.StreamReader(context.Request.InputStream))
                        {
                            string json = reader.ReadToEnd();
                            var result = ProcessSignal(json);
                            SendResponse(context.Response, 200, serializer.Serialize(result));
                        }
                    }
                    else if (method == "POST" && path == "/flatten")
                    {
                        string reason = "dashboard";
                        try
                        {
                            using (var reader = new System.IO.StreamReader(context.Request.InputStream))
                            {
                                string body = reader.ReadToEnd();
                                if (!string.IsNullOrEmpty(body))
                                {
                                    var payload = serializer.Deserialize<Dictionary<string, object>>(body);
                                    if (payload != null && payload.ContainsKey("reason"))
                                        reason = payload["reason"].ToString();
                                }
                            }
                        }
                        catch { }
                        var result = FlattenAll(reason);
                        SendResponse(context.Response, 200, serializer.Serialize(result));
                    }
                    else if (method == "POST" && path == "/cancel_orders")
                    {
                        var result = CancelAllOrders();
                        SendResponse(context.Response, 200, serializer.Serialize(result));
                    }
                    else if (method == "GET" && path == "/status")
                    {
                        string status = BuildStatusJson();
                        SendResponse(context.Response, 200, status);
                    }
                    else if (method == "GET" && path == "/orders")
                    {
                        string ordersJson = BuildOrdersJson();
                        SendResponse(context.Response, 200, ordersJson);
                    }
                    else if (method == "GET" && path == "/health")
                    {
                        // Lightweight health check — just returns 200 with minimal JSON
                        SendResponse(context.Response, 200, "{\"status\":\"ok\",\"bridge_version\":\"2.0\"}");
                    }
                    else
                    {
                        SendResponse(context.Response, 404, "{\"error\":\"not found\",\"endpoints\":[\"/execute_signal\",\"/flatten\",\"/cancel_orders\",\"/status\",\"/orders\",\"/health\"]}");
                    }
                }
                catch (HttpListenerException)
                {
                    // Listener was stopped — exit cleanly
                    break;
                }
                catch (Exception ex)
                {
                    ThrottledLog($"ListenLoop error: {ex.Message}");
                }
            }
        }

        private void AddCorsHeaders(HttpListenerResponse response, HttpListenerRequest request)
        {
            // Allow the Python dashboard (typically localhost:8000) and any local dev origin
            string origin = request.Headers["Origin"];
            if (!string.IsNullOrEmpty(origin))
            {
                response.Headers.Add("Access-Control-Allow-Origin", origin);
            }
            else
            {
                response.Headers.Add("Access-Control-Allow-Origin", "*");
            }
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
            response.Headers.Add("Access-Control-Max-Age", "86400");
        }

        private string BuildStatusJson()
        {
            try
            {
                string acctName = myAccount?.Name ?? "disconnected";
                int posCount = 0;
                double cashBalance = 0;
                double realizedPnL = 0;
                double unrealizedPnL = 0;
                int pendingOrderCount = 0;

                try
                {
                    if (myAccount != null)
                    {
                        cashBalance = myAccount.Get(AccountItem.CashValue, Currency.UsDollar);
                        realizedPnL = myAccount.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    }
                }
                catch { }

                try
                {
                    if (myAccount?.Positions != null)
                    {
                        foreach (Position pos in myAccount.Positions)
                        {
                            if (pos != null && pos.Quantity != 0)
                            {
                                posCount++;
                                try
                                {
                                    double lastPrice = SafeGetClose();
                                    if (lastPrice > 0)
                                        unrealizedPnL += pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency, lastPrice);
                                }
                                catch { }
                            }
                        }
                    }
                }
                catch { }

                try
                {
                    if (myAccount?.Orders != null)
                    {
                        pendingOrderCount = myAccount.Orders.Count(o =>
                            o.OrderState == NinjaTrader.Cbi.OrderState.Working
                            || o.OrderState == NinjaTrader.Cbi.OrderState.Accepted);
                    }
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(acctName).Append("\",");
                sb.Append("\"connected\":").Append(myAccount != null ? "true" : "false").Append(",");
                sb.Append("\"state\":\"").Append(State).Append("\",");
                sb.Append("\"positions\":").Append(posCount).Append(",");
                sb.Append("\"pendingOrders\":").Append(pendingOrderCount).Append(",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realizedPnL\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"unrealizedPnL\":").Append(Math.Round(unrealizedPnL, 2)).Append(",");
                sb.Append("\"riskBlocked\":").Append(riskBlocked ? "true" : "false").Append(",");
                sb.Append("\"riskBlockReason\":\"").Append(EscapeJson(riskBlockReason)).Append("\",");
                sb.Append("\"riskPercent\":").Append(RiskPercentPerTrade).Append(",");
                sb.Append("\"maxContracts\":").Append(MaxMicroContracts).Append(",");
                sb.Append("\"bridge_version\":\"2.0\",");
                sb.Append("\"listenerPort\":").Append(SignalListenerPort).Append(",");
                sb.Append("\"dashboardUrl\":\"").Append(EscapeJson(DashboardBaseUrl)).Append("\",");
                sb.Append("\"timestamp\":\"").Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");
                return sb.ToString();
            }
            catch { return "{\"error\":\"status unavailable\"}"; }
        }

        private string BuildOrdersJson()
        {
            try
            {
                var serializer = new JavaScriptSerializer();
                List<Dictionary<string, object>> events;
                lock (orderLock)
                {
                    events = new List<Dictionary<string, object>>(recentOrderEvents);
                }

                var result = new Dictionary<string, object>
                {
                    { "events", events },
                    { "count", events.Count },
                    { "timestamp", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") }
                };

                return serializer.Serialize(result);
            }
            catch { return "{\"events\":[],\"error\":\"orders unavailable\"}"; }
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

        /// <summary>
        /// Minimal JSON string escaping for values embedded in hand-built JSON.
        /// Handles backslash, double-quote, and control characters.
        /// </summary>
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

        private void SendResponse(HttpListenerResponse resp, int code, string msg)
        {
            try
            {
                resp.StatusCode = code;
                resp.ContentType = "application/json";
                byte[] buf = Encoding.UTF8.GetBytes(msg ?? "");
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
