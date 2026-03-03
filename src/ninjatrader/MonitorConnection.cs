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
using System.Web.Script.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

// =============================================================================
// MonitorConnection — NinjaTrader 8 Strategy  (telemetry + command layer)
// =============================================================================
//
// Responsibilities (everything Bridge did except ORB and Ruby):
//   1. Monitor a live/sim account and push positions, P&L, and pending orders
//      to the Python dashboard on every fill and on a 15-second heartbeat.
//   2. Host an HTTP listener (default port 5680) that accepts:
//        POST /execute_signal   — forward signal JSON to BridgeOrderEngine
//        POST /flatten          — flatten all positions immediately
//        POST /cancel_orders    — cancel all working orders
//        GET  /status           — JSON health + account snapshot
//        GET  /orders           — recent order event log
//        GET  /health           — lightweight liveness probe
//        GET  /metrics          — Prometheus exposition format
//   3. Drain the static SignalBus queue each bar so backtests work:
//        BreakoutStrategy enqueues → MonitorConnection drains + executes.
//   4. Parse risk-gate feedback from the Python /positions/update response
//      and toggle RiskBlocked to prevent new entries.
//   5. Expose Prometheus counters for Grafana.
//
// What this file does NOT do:
//   - Any ORB or breakout signal generation  → BreakoutStrategy.cs
//   - Any order-submission logic             → BridgeOrderEngine.cs
//   - Attaching the Ruby indicator           → attach Ruby directly on a chart
//
// Typical setup:
//   Drop MonitorConnection on any 1-min chart.
//   Drop BreakoutStrategy on the same (or different) chart.
//   Both share SignalBus — BreakoutStrategy fires signals, MonitorConnection
//   pushes telemetry back to Python, Python responds with risk feedback.
//
// Multi-instrument order routing:
//   MonitorConnection does NOT need AddDataSeries itself.  BridgeOrderEngine
//   uses the _symbolToBip table populated from the TrackedInstruments property
//   to route SubmitOrderUnmanaged to the correct BIP.  If you want MonitorConnection
//   to submit orders for non-primary instruments, add them to TrackedInstruments
//   and they will be subscribed via AddDataSeries automatically.
//
// HTTP access from Docker / WSL2:
//   netsh http add urlacl url=http://+:5680/ user=Everyone
//   (run once as Administrator; listener binds to all interfaces)
//
// =============================================================================

namespace NinjaTrader.NinjaScript.Strategies
{
    public class MonitorConnection : Strategy
    {
        // =====================================================================
        // Private fields
        // =====================================================================

        // ── Account + HTTP ────────────────────────────────────────────────────
        private Account        _account;
        private HttpClient     _httpClient;

        // ── Inbound HTTP listener ─────────────────────────────────────────────
        private HttpListener              _listener;
        private Thread                    _listenerThread;
        private volatile bool             _listenerStopped;
        private readonly ManualResetEventSlim _listenerExited = new ManualResetEventSlim(false);

        // ── Lifecycle guard ───────────────────────────────────────────────────
        private bool _cleanedUp;

        // ── Outbound push state ───────────────────────────────────────────────
        private bool     _lastPushSuccess = true;
        private DateTime _lastErrorLog    = DateTime.MinValue;
        private DateTime _lastHeartbeat   = DateTime.MinValue;

        // ── Order execution engine ────────────────────────────────────────────
        private BridgeOrderEngine _engine;

        // ── Multi-instrument BIP routing ──────────────────────────────────────
        private readonly Dictionary<string, int> _symbolToBip =
            new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        private string[] _extraSymbols = new string[0];

        // ── Recent order event log (for /orders endpoint) ─────────────────────
        private readonly List<Dictionary<string, object>> _orderEvents =
            new List<Dictionary<string, object>>();
        private readonly object _orderLock = new object();
        private const int MAX_ORDER_EVENTS = 50;

        // ── Risk gate ─────────────────────────────────────────────────────────
        internal volatile bool   RiskBlocked      = false;
        internal volatile string RiskBlockReason  = "";

        // ── Throttle + timing constants ───────────────────────────────────────
        private const int ERROR_LOG_THROTTLE_SECONDS = 15;
        private const int HEARTBEAT_INTERVAL_SECONDS = 15;

        // ── Prometheus counters ───────────────────────────────────────────────
        private long _metricSignalsReceived;
        private long _metricSignalsExecuted;
        private long _metricSignalsRejected;
        private long _metricExitsExecuted;
        private long _metricPositionPushes;
        private long _metricPositionPushErrors;
        private long _metricHeartbeatsSent;
        private long _metricHttpRequests;
        private long _metricSignalBusDrained;
        private long _metricOrdersFilled;
        private long _metricOrdersRejected;
        private readonly DateTime _startTime = DateTime.UtcNow;

        // =====================================================================
        // Properties
        // =====================================================================
        #region Properties

        // ── 1. Account ────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [TypeConverter(typeof(AccountNameConverter))]
        [Display(Name = "Account", GroupName = "1. Account", Order = 1,
                 Description = "Account to monitor and trade on.")]
        public string AccountName { get; set; } = "Sim101";

        // ── 2. Web App ────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Dashboard Base URL", GroupName = "2. Web App", Order = 1,
                 Description = "Base URL of the Python data service. " +
                               "Example: http://100.69.78.116:8000  " +
                               "The trainer service on port 8200 is separate and not required here.")]
        public string DashboardBaseUrl { get; set; } = "http://100.69.78.116:8000";

        [NinjaScriptProperty]
        [Display(Name = "Signal Listener Port", GroupName = "2. Web App", Order = 2,
                 Description = "HTTP port this strategy listens on for signals and control commands. " +
                               "Must match NT_BRIDGE_PORT in the Python service .env file.")]
        public int SignalListenerPort { get; set; } = 5680;

        // ── 3. Options ────────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Enable Position Push", GroupName = "3. Options", Order = 1,
                 Description = "Push live positions to the Python dashboard on every fill and heartbeat.")]
        public bool EnablePositionPush { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Auto Brackets", GroupName = "3. Options", Order = 2,
                 Description = "Automatically submit SL + TP bracket orders alongside each entry.")]
        public bool EnableAutoBrackets { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable Risk Enforcement", GroupName = "3. Options", Order = 3,
                 Description = "Block new orders when the Python risk engine returns can_trade=false.")]
        public bool EnableRiskEnforcement { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Enable SignalBus", GroupName = "3. Options", Order = 4,
                 Description = "Drain in-process signals from BreakoutStrategy via SignalBus each bar. " +
                               "Required for Strategy Analyzer backtests.")]
        public bool EnableSignalBus { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Default SL Ticks (fallback)", GroupName = "3. Options", Order = 5,
                 Description = "Used when a signal does not include an explicit stop_loss price.")]
        public int DefaultStopLossTicks { get; set; } = 20;

        [NinjaScriptProperty]
        [Display(Name = "Default TP Ticks (fallback)", GroupName = "3. Options", Order = 6,
                 Description = "Used when a signal does not include an explicit take_profit price.")]
        public int DefaultTakeProfitTicks { get; set; } = 40;

        // ── 4. Instruments ────────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Tracked Instruments", GroupName = "4. Instruments", Order = 1,
                 Description = "Comma-separated instrument root names. The chart instrument is BIP 0. " +
                               "Each extra symbol becomes BIP 1, 2, … so the engine can route orders " +
                               "for all 22 micros from one strategy instance. " +
                               "Full list: MGC,SIL,MHG,MCL,MNG,MES,MNQ,M2K,MYM,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET")]
        public string TrackedInstruments { get; set; } =
            "MGC,MES,MNQ,MCL,MNG,M2K,MYM,SIL,MHG,6E,6B,6J,6A,6C,6S,ZN,ZB,ZC,ZS,ZW,MBT,MET";

        // ── 5. Risk Management ────────────────────────────────────────────────
        [NinjaScriptProperty]
        [Display(Name = "Account Size ($)", GroupName = "5. Risk", Order = 1,
                 Description = "Fallback account size when live balance is unavailable.")]
        public double AccountSize { get; set; } = 50000;

        [NinjaScriptProperty]
        [Range(0.1, 2.0)]
        [Display(Name = "Risk % Per Trade", GroupName = "5. Risk", Order = 2,
                 Description = "Fraction of account balance risked per trade (0.5 = 0.5 %).")]
        public double RiskPercentPerTrade { get; set; } = 0.5;

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max Contracts", GroupName = "5. Risk", Order = 3,
                 Description = "Hard cap on contracts per signal regardless of risk-sized quantity.")]
        public int MaxContracts { get; set; } = 10;

        #endregion

        // =====================================================================
        // Lifecycle
        // =====================================================================
        #region Lifecycle

        protected override void OnStateChange()
        {
            // ── SetDefaults ───────────────────────────────────────────────────
            if (State == State.SetDefaults)
            {
                Description = "Telemetry and command bridge. Pushes live positions to the Python " +
                              "dashboard, hosts HTTP endpoints, drains SignalBus. " +
                              "Order logic → BridgeOrderEngine.cs   " +
                              "ORB signals → BreakoutStrategy.cs";
                Name         = "MonitorConnection";
                Calculate    = Calculate.OnBarClose;
                IsOverlay    = false;
                IsUnmanaged  = true;
                BarsRequiredToTrade = 1;
            }

            // ── Configure ────────────────────────────────────────────────────
            else if (State == State.Configure)
            {
                _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                // Subscribe extra instruments so BridgeOrderEngine can route
                // SubmitOrderUnmanaged to the correct BIP for each symbol.
                if (!string.IsNullOrWhiteSpace(TrackedInstruments))
                {
                    var requested = TrackedInstruments
                        .Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    var extras = new List<string>();
                    string primaryRoot = Instrument?.MasterInstrument.Name.ToUpperInvariant() ?? "";

                    foreach (string sym in requested)
                    {
                        string root = sym.Trim().ToUpperInvariant();
                        if (string.IsNullOrEmpty(root)) continue;
                        if (root == primaryRoot)        continue;
                        if (extras.Contains(root))      continue;

                        try
                        {
                            AddDataSeries(root,
                                new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1 },
                                "", true);
                            extras.Add(root);
                            Print($"[Monitor] AddDataSeries: {root} → BIP {extras.Count}");
                        }
                        catch (Exception ex)
                        {
                            Print($"[Monitor] AddDataSeries failed for {root}: {ex.Message}");
                        }
                    }

                    _extraSymbols = extras.ToArray();
                }

                // Account subscription for fill/position events
                lock (Account.All)
                    _account = Account.All.FirstOrDefault(a => a.Name == AccountName);

                if (_account != null)
                {
                    _account.PositionUpdate += OnPositionUpdate;
                    _account.OrderUpdate    += OnOrderUpdate;
                    Print($"[Monitor] Account: {_account.Name}");
                }
                else
                {
                    Print($"[Monitor] ⚠ Account '{AccountName}' not found — position push disabled.");
                }

                // Register as SignalBus consumer so BreakoutStrategy signals reach us
                if (EnableSignalBus)
                {
                    SignalBus.Reset();
                    SignalBus.RegisterConsumer();
                    Print("[Monitor] SignalBus consumer registered");
                }
            }

            // ── DataLoaded ───────────────────────────────────────────────────
            else if (State == State.DataLoaded)
            {
                // Build BIP routing table from confirmed instrument names
                _symbolToBip.Clear();

                string primaryRoot = Instrument?.MasterInstrument.Name.ToUpperInvariant() ?? "";
                if (!string.IsNullOrEmpty(primaryRoot))
                    _symbolToBip[primaryRoot] = 0;

                for (int i = 0; i < _extraSymbols.Length; i++)
                {
                    int bip = i + 1;
                    if (bip < BarsArray.Length)
                    {
                        string confirmed = BarsArray[bip].Instrument.MasterInstrument.Name
                                                         .ToUpperInvariant();
                        _symbolToBip[confirmed] = bip;
                        Print($"[Monitor] Routing: {confirmed} → BIP {bip}");
                    }
                }

                Print($"[Monitor] {_symbolToBip.Count} instruments mapped");

                // Create the order engine — all execution goes through here
                Account capturedAccount = _account;

                _engine = new BridgeOrderEngine(
                    strategy:           this,
                    symbolToBip:        _symbolToBip,
                    getMyAccount:       () => capturedAccount,
                    getAccountSize:     () => AccountSize,
                    getRiskPercent:     () => RiskPercentPerTrade,
                    getMaxContracts:    () => MaxContracts,
                    getDefaultSlTicks:  () => DefaultStopLossTicks,
                    getDefaultTpTicks:  () => DefaultTakeProfitTicks,
                    getAutoBrackets:    () => EnableAutoBrackets,
                    getRiskBlocked:     () => RiskBlocked,
                    getRiskBlockReason: () => RiskBlockReason,
                    getRiskEnforcement: () => EnableRiskEnforcement,
                    onSignalReceived:   () => Interlocked.Increment(ref _metricSignalsReceived),
                    onSignalExecuted:   () => Interlocked.Increment(ref _metricSignalsExecuted),
                    onSignalRejected:   () => Interlocked.Increment(ref _metricSignalsRejected),
                    onExitExecuted:     () => Interlocked.Increment(ref _metricExitsExecuted),
                    onBusDrained:       n  => Interlocked.Add(ref _metricSignalBusDrained, n),
                    sendPositionUpdate: SendPositionUpdate,
                    tag:                "Monitor"
                );
            }

            // ── Realtime ─────────────────────────────────────────────────────
            else if (State == State.Realtime)
            {
                StartListener();
                SendPositionUpdate();
                Print($"[Monitor] Realtime — listener started on port {SignalListenerPort}");
            }

            // ── Terminated ───────────────────────────────────────────────────
            else if (State == State.Terminated)
            {
                if (_cleanedUp) return;
                _cleanedUp = true;

                StopListener();

                try
                {
                    if (_account != null)
                    {
                        _account.PositionUpdate -= OnPositionUpdate;
                        _account.OrderUpdate    -= OnOrderUpdate;
                    }
                }
                catch { }

                if (EnableSignalBus)
                {
                    SignalBus.UnregisterConsumer();
                    Print($"[Monitor] SignalBus unregistered " +
                          $"(enqueued={SignalBus.TotalEnqueued}, drained={SignalBus.TotalDrained})");
                }

                try { _httpClient?.Dispose(); } catch { }
                _httpClient = null;
                _engine     = null;
            }
        }

        #endregion

        // =====================================================================
        // Bar update — drain SignalBus, flush order queue, heartbeat
        // =====================================================================
        #region Bar Update

        protected override void OnBarUpdate()
        {
            // All per-bar logic runs once per primary-series bar only.
            // Secondary BIP bars are needed by NT8 internally for order routing
            // but do not require any processing here.
            if (BarsInProgress != 0) return;

            try
            {
                if (EnableSignalBus && _engine != null)
                    _engine.DrainSignalBus(State);

                if (_engine != null &&
                    (State == State.Realtime || State == State.Historical))
                    _engine.FlushOrderQueue(State);

                if (State != State.Realtime) return;

                if ((DateTime.Now - _lastHeartbeat).TotalSeconds >= HEARTBEAT_INTERVAL_SECONDS)
                {
                    _lastHeartbeat = DateTime.Now;
                    SendHeartbeat();
                }
            }
            catch (Exception ex) { ThrottledLog($"OnBarUpdate: {ex.Message}"); }
        }

        #endregion

        // =====================================================================
        // Account events
        // =====================================================================
        #region Account Events

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            try
            {
                var evt = new Dictionary<string, object>
                {
                    { "orderId",    e.Order.OrderId                                    },
                    { "name",       e.Order.Name ?? ""                                 },
                    { "instrument", e.Order.Instrument?.FullName ?? ""                 },
                    { "action",     e.Order.OrderAction.ToString()                     },
                    { "type",       e.Order.OrderType.ToString()                       },
                    { "quantity",   e.Order.Quantity                                   },
                    { "state",      e.Order.OrderState.ToString()                      },
                    { "limitPrice", e.Order.LimitPrice                                 },
                    { "stopPrice",  e.Order.StopPrice                                  },
                    { "timestamp",  DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")  },
                };

                lock (_orderLock)
                {
                    _orderEvents.Add(evt);
                    if (_orderEvents.Count > MAX_ORDER_EVENTS)
                        _orderEvents.RemoveAt(0);
                }

                switch (e.Order.OrderState)
                {
                    case NinjaTrader.Cbi.OrderState.Filled:
                        Interlocked.Increment(ref _metricOrdersFilled);
                        Print($"[Monitor] FILLED  {e.Order.Name} " +
                              $"{e.Order.OrderAction} {e.Order.Quantity} " +
                              $"{e.Order.Instrument?.FullName} @ {e.Order.AverageFillPrice}");
                        SendPositionUpdate();
                        break;

                    case NinjaTrader.Cbi.OrderState.Rejected:
                        Interlocked.Increment(ref _metricOrdersRejected);
                        Print($"[Monitor] REJECTED {e.Order.Name}: {e.Order.Instrument?.FullName}");
                        break;

                    case NinjaTrader.Cbi.OrderState.Cancelled:
                        Print($"[Monitor] CANCELLED {e.Order.Name}");
                        break;
                }
            }
            catch (Exception ex) { ThrottledLog($"OnOrderUpdate: {ex.Message}"); }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            try   { SendPositionUpdate(); }
            catch (Exception ex) { ThrottledLog($"OnPositionUpdate: {ex.Message}"); }
        }

        #endregion

        // =====================================================================
        // Outbound pushes — position update + heartbeat
        // =====================================================================
        #region Outbound Pushes

        private void SendPositionUpdate()
        {
            if (!EnablePositionPush || _account == null) return;
            var client = _httpClient;
            if (client == null) return;

            Interlocked.Increment(ref _metricPositionPushes);
            try
            {
                double lastPrice    = SafeClose();
                double cashBalance  = 0;
                double realizedPnL  = 0;

                try
                {
                    cashBalance = _account.Get(AccountItem.CashValue,          Currency.UsDollar);
                    realizedPnL = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(_account.Name).Append("\",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realizedPnL\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"bridge_version\":\"3.0\",");
                sb.Append("\"positions\":[");

                bool   firstPos           = true;
                double totalUnrealizedPnL = 0;

                if (_account.Positions != null)
                {
                    foreach (Position pos in _account.Positions)
                    {
                        try
                        {
                            if (pos == null || pos.Quantity == 0 || pos.Instrument == null)
                                continue;

                            double pnl = 0;
                            try
                            {
                                if (lastPrice > 0)
                                    pnl = pos.GetUnrealizedProfitLoss(
                                        PerformanceUnit.Currency, lastPrice);
                            }
                            catch { }

                            totalUnrealizedPnL += pnl;

                            if (!firstPos) sb.Append(",");
                            sb.Append("{");
                            sb.Append("\"symbol\":\"").Append(pos.Instrument.FullName).Append("\",");
                            sb.Append("\"side\":\"").Append(pos.MarketPosition).Append("\",");
                            sb.Append("\"quantity\":").Append(pos.Quantity).Append(",");
                            sb.Append("\"avgPrice\":").Append(pos.AveragePrice).Append(",");
                            sb.Append("\"unrealizedPnL\":").Append(Math.Round(pnl, 2)).Append(",");
                            sb.Append("\"instrument\":\"")
                              .Append(pos.Instrument.MasterInstrument.Name).Append("\",");
                            sb.Append("\"tickSize\":")
                              .Append(pos.Instrument.MasterInstrument.TickSize).Append(",");
                            sb.Append("\"pointValue\":")
                              .Append(pos.Instrument.MasterInstrument.PointValue).Append(",");
                            sb.Append("\"lastUpdate\":\"")
                              .Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                            sb.Append("}");
                            firstPos = false;
                        }
                        catch (Exception ex) { ThrottledLog($"Position read: {ex.Message}"); }
                    }
                }

                sb.Append("],\"pendingOrders\":[");

                bool firstOrd = true;
                try
                {
                    if (_account.Orders != null)
                    {
                        foreach (Order ord in _account.Orders)
                        {
                            try
                            {
                                if (ord == null) continue;
                                if (ord.OrderState != NinjaTrader.Cbi.OrderState.Working &&
                                    ord.OrderState != NinjaTrader.Cbi.OrderState.Accepted)
                                    continue;

                                if (!firstOrd) sb.Append(",");
                                sb.Append("{");
                                sb.Append("\"orderId\":\"").Append(ord.OrderId).Append("\",");
                                sb.Append("\"name\":\"").Append(Esc(ord.Name)).Append("\",");
                                sb.Append("\"instrument\":\"")
                                  .Append(ord.Instrument?.FullName ?? "").Append("\",");
                                sb.Append("\"action\":\"").Append(ord.OrderAction).Append("\",");
                                sb.Append("\"type\":\"").Append(ord.OrderType).Append("\",");
                                sb.Append("\"quantity\":").Append(ord.Quantity).Append(",");
                                sb.Append("\"limitPrice\":").Append(ord.LimitPrice).Append(",");
                                sb.Append("\"stopPrice\":").Append(ord.StopPrice).Append(",");
                                sb.Append("\"state\":\"").Append(ord.OrderState).Append("\"");
                                sb.Append("}");
                                firstOrd = false;
                            }
                            catch { }
                        }
                    }
                }
                catch { }

                sb.Append("],");
                sb.Append("\"totalUnrealizedPnL\":")
                  .Append(Math.Round(totalUnrealizedPnL, 2)).Append(",");
                sb.Append("\"riskBlocked\":")
                  .Append(RiskBlocked ? "true" : "false").Append(",");
                sb.Append("\"riskBlockReason\":\"")
                  .Append(Esc(RiskBlockReason)).Append("\",");
                sb.Append("\"timestamp\":\"")
                  .Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");

                string url     = DashboardBaseUrl.TrimEnd('/') + "/positions/update";
                var    content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");

                client.PostAsync(url, content).ContinueWith(t =>
                {
                    if (t.IsFaulted)
                    {
                        Interlocked.Increment(ref _metricPositionPushErrors);
                        if (_lastPushSuccess || (DateTime.Now - _lastErrorLog).TotalSeconds > 30)
                        {
                            _lastErrorLog    = DateTime.Now;
                            _lastPushSuccess = false;
                        }
                    }
                    else
                    {
                        if (!_lastPushSuccess)
                        {
                            Print("[Monitor] Position push connection restored");
                            _lastPushSuccess = true;
                        }
                        ParseRiskFeedback(t.Result);
                    }
                });
            }
            catch (Exception ex) { ThrottledLog($"SendPositionUpdate: {ex.Message}"); }
        }

        private void ParseRiskFeedback(HttpResponseMessage response)
        {
            if (!EnableRiskEnforcement) return;
            try
            {
                string body = response.Content.ReadAsStringAsync().Result;
                if (string.IsNullOrEmpty(body)) return;

                var data = new JavaScriptSerializer()
                    .Deserialize<Dictionary<string, object>>(body);

                if (data == null || !data.ContainsKey("risk")) return;
                if (!(data["risk"] is Dictionary<string, object> risk)) return;

                bool canTrade    = true;
                string blockReason = "";

                if (risk.ContainsKey("can_trade"))
                    try { canTrade = Convert.ToBoolean(risk["can_trade"]); } catch { }

                if (risk.ContainsKey("block_reason") && risk["block_reason"] != null)
                    blockReason = risk["block_reason"].ToString();

                bool wasBlocked = RiskBlocked;
                RiskBlocked      = !canTrade;
                RiskBlockReason  = blockReason;

                if (!canTrade && !wasBlocked)
                    Print($"[Monitor] ⚠ Risk BLOCKED: {blockReason}");
                else if (canTrade && wasBlocked)
                    Print("[Monitor] ✓ Risk block cleared");

                if (risk.ContainsKey("warnings") &&
                    risk["warnings"] is System.Collections.ArrayList warnings)
                    foreach (var w in warnings)
                        if (w != null) Print($"[Monitor] ⚠ Risk warning: {w}");
            }
            catch { }
        }

        private void SendHeartbeat()
        {
            Interlocked.Increment(ref _metricHeartbeatsSent);
            var client = _httpClient;
            if (client == null || _account == null) return;

            try
            {
                double cashBalance   = 0;
                int    openPositions = 0;

                try { cashBalance   = _account.Get(AccountItem.CashValue, Currency.UsDollar); } catch { }
                try
                {
                    if (_account.Positions != null)
                        openPositions = _account.Positions.Count(p => p != null && p.Quantity != 0);
                }
                catch { }

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(_account.Name).Append("\",");
                sb.Append("\"state\":\"").Append(State).Append("\",");
                sb.Append("\"connected\":true,");
                sb.Append("\"positions\":").Append(openPositions).Append(",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"riskBlocked\":").Append(RiskBlocked ? "true" : "false").Append(",");
                sb.Append("\"bridge_version\":\"3.0\",");
                sb.Append("\"listenerPort\":").Append(SignalListenerPort).Append(",");
                sb.Append("\"timestamp\":\"")
                  .Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");

                string url     = DashboardBaseUrl.TrimEnd('/') + "/positions/heartbeat";
                var    content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");
                client.PostAsync(url, content).ContinueWith(_ => { });
            }
            catch { }
        }

        #endregion

        // =====================================================================
        // HTTP listener — start / stop / request loop
        // =====================================================================
        #region HTTP Listener

        private void StartListener()
        {
            // Try wildcard binding first (accessible from Docker / WSL2 / Tailscale).
            // Requires a one-time URL reservation as Administrator:
            //   netsh http add urlacl url=http://+:5680/ user=Everyone
            // Falls back to localhost if wildcard binding is refused.
            string[] prefixes =
            {
                $"http://+:{SignalListenerPort}/",
                $"http://localhost:{SignalListenerPort}/"
            };

            foreach (string prefix in prefixes)
            {
                for (int attempt = 1; attempt <= 3; attempt++)
                {
                    try
                    {
                        _listenerStopped = false;
                        _listenerExited.Reset();

                        _listener = new HttpListener();
                        _listener.Prefixes.Add(prefix);
                        _listener.Start();

                        _listenerThread = new Thread(ListenLoop)
                            { IsBackground = true, Name = "MonitorListener" };
                        _listenerThread.Start();

                        bool wildcard = prefix.Contains("+");
                        Print($"[Monitor] Listening on port {SignalListenerPort}" +
                              (wildcard ? " (all interfaces)" : " (localhost only)"));

                        if (!wildcard)
                            Print("[Monitor] For remote access run once as admin: " +
                                  $"netsh http add urlacl url=http://+:{SignalListenerPort}/ user=Everyone");
                        return;
                    }
                    catch (Exception ex)
                    {
                        Print($"[Monitor] Listener {attempt}/3 on {prefix}: {ex.Message}");
                        try { _listener?.Close(); } catch { }
                        _listener = null;
                        if (attempt < 3) Thread.Sleep(800);
                    }
                }
            }

            Print("[Monitor] All listener attempts failed — HTTP unavailable; SignalBus still active.");
        }

        private void StopListener()
        {
            _listenerStopped = true;
            try { _listener?.Stop();  } catch { }
            try { _listener?.Close(); } catch { }
            _listener = null;

            if (_listenerThread != null && _listenerThread.IsAlive)
            {
                if (!_listenerExited.Wait(3000))
                    Print("[Monitor] Warning: listener thread did not exit within 3 s");
                _listenerThread = null;
            }
        }

        private void ListenLoop()
        {
            var serializer = new JavaScriptSerializer();
            try
            {
                while (!_listenerStopped && _listener?.IsListening == true)
                {
                    try
                    {
                        var ctx = _listener.GetContext();
                        Interlocked.Increment(ref _metricHttpRequests);

                        AddCorsHeaders(ctx.Response, ctx.Request);

                        if (ctx.Request.HttpMethod == "OPTIONS")
                        {
                            SendResponse(ctx.Response, 204, "");
                            continue;
                        }

                        string path   = ctx.Request.Url.AbsolutePath.TrimEnd('/');
                        string method = ctx.Request.HttpMethod;

                        // ── POST /execute_signal ──────────────────────────────
                        if (method == "POST" && path == "/execute_signal")
                        {
                            using (var r = new System.IO.StreamReader(ctx.Request.InputStream))
                            {
                                string json = r.ReadToEnd();
                                var result = _engine != null
                                    ? _engine.ProcessSignal(json)
                                    : Error("engine not ready");
                                SendResponse(ctx.Response, 200, serializer.Serialize(result));
                            }
                        }

                        // ── POST /flatten ─────────────────────────────────────
                        else if (method == "POST" && path == "/flatten")
                        {
                            string reason = "dashboard";
                            try
                            {
                                using (var r = new System.IO.StreamReader(ctx.Request.InputStream))
                                {
                                    string body = r.ReadToEnd();
                                    if (!string.IsNullOrEmpty(body))
                                    {
                                        var p = serializer.Deserialize<Dictionary<string, object>>(body);
                                        if (p != null && p.ContainsKey("reason"))
                                            reason = p["reason"].ToString();
                                    }
                                }
                            }
                            catch { }

                            var result = _engine != null
                                ? _engine.FlattenAll(reason)
                                : Error("engine not ready");
                            SendResponse(ctx.Response, 200, serializer.Serialize(result));
                        }

                        // ── POST /cancel_orders ───────────────────────────────
                        else if (method == "POST" && path == "/cancel_orders")
                        {
                            var result = _engine != null
                                ? _engine.CancelAllOrders()
                                : Error("engine not ready");
                            SendResponse(ctx.Response, 200, serializer.Serialize(result));
                        }

                        // ── GET /status ───────────────────────────────────────
                        else if (method == "GET" && path == "/status")
                        {
                            SendResponse(ctx.Response, 200, BuildStatusJson());
                        }

                        // ── GET /orders ───────────────────────────────────────
                        else if (method == "GET" && path == "/orders")
                        {
                            SendResponse(ctx.Response, 200, BuildOrdersJson(serializer));
                        }

                        // ── GET /health ───────────────────────────────────────
                        else if (method == "GET" && path == "/health")
                        {
                            SendResponse(ctx.Response, 200,
                                "{\"status\":\"ok\",\"bridge_version\":\"3.0\"}");
                        }

                        // ── GET /metrics ──────────────────────────────────────
                        else if (method == "GET" && path == "/metrics")
                        {
                            ctx.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                            SendResponse(ctx.Response, 200, BuildPrometheusMetrics());
                        }

                        // ── 404 ───────────────────────────────────────────────
                        else
                        {
                            SendResponse(ctx.Response, 404,
                                "{\"error\":\"not found\"," +
                                "\"endpoints\":[" +
                                "\"/execute_signal\",\"/flatten\",\"/cancel_orders\"," +
                                "\"/status\",\"/orders\",\"/health\",\"/metrics\"]}");
                        }
                    }
                    catch (HttpListenerException)   { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception ex)
                    {
                        if (_listenerStopped) break;
                        ThrottledLog($"ListenLoop: {ex.Message}");
                    }
                }
            }
            finally { _listenerExited.Set(); }
        }

        private static void AddCorsHeaders(HttpListenerResponse resp, HttpListenerRequest req)
        {
            string origin = req.Headers["Origin"];
            resp.Headers.Add("Access-Control-Allow-Origin",
                !string.IsNullOrEmpty(origin) ? origin : "*");
            resp.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp.Headers.Add("Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-API-Key");
            resp.Headers.Add("Access-Control-Max-Age", "86400");
        }

        #endregion

        // =====================================================================
        // Response builders
        // =====================================================================
        #region Response Builders

        private string BuildStatusJson()
        {
            try
            {
                // Instrument list
                var instrSb = new StringBuilder("[");
                bool firstI = true;
                foreach (var kv in _symbolToBip)
                {
                    if (!firstI) instrSb.Append(",");
                    instrSb.Append("{\"symbol\":\"").Append(kv.Key)
                           .Append("\",\"bip\":").Append(kv.Value).Append("}");
                    firstI = false;
                }
                instrSb.Append("]");

                int    posCount        = 0;
                double cashBalance     = 0;
                double realizedPnL     = 0;
                double unrealizedPnL   = 0;
                int    pendingOrdCount = 0;

                try
                {
                    if (_account != null)
                    {
                        cashBalance = _account.Get(AccountItem.CashValue,          Currency.UsDollar);
                        realizedPnL = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    }
                }
                catch { }

                try
                {
                    if (_account?.Positions != null)
                        foreach (Position pos in _account.Positions)
                        {
                            if (pos == null || pos.Quantity == 0) continue;
                            posCount++;
                            try
                            {
                                double lp = SafeClose();
                                if (lp > 0)
                                    unrealizedPnL += pos.GetUnrealizedProfitLoss(
                                        PerformanceUnit.Currency, lp);
                            }
                            catch { }
                        }
                }
                catch { }

                try
                {
                    if (_account?.Orders != null)
                        pendingOrdCount = _account.Orders.Count(o =>
                            o.OrderState == NinjaTrader.Cbi.OrderState.Working ||
                            o.OrderState == NinjaTrader.Cbi.OrderState.Accepted);
                }
                catch { }

                bool listenerUp   = _listener != null && !_listenerStopped;
                bool busActive    = EnableSignalBus && SignalBus.HasConsumer;

                var sb = new StringBuilder();
                sb.Append("{");
                sb.Append("\"account\":\"").Append(_account?.Name ?? "disconnected").Append("\",");
                sb.Append("\"connected\":").Append(_account != null ? "true" : "false").Append(",");
                sb.Append("\"state\":\"").Append(State).Append("\",");
                sb.Append("\"positions\":").Append(posCount).Append(",");
                sb.Append("\"pendingOrders\":").Append(pendingOrdCount).Append(",");
                sb.Append("\"cashBalance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realizedPnL\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"unrealizedPnL\":").Append(Math.Round(unrealizedPnL, 2)).Append(",");
                sb.Append("\"riskBlocked\":").Append(RiskBlocked ? "true" : "false").Append(",");
                sb.Append("\"riskBlockReason\":\"").Append(Esc(RiskBlockReason)).Append("\",");
                sb.Append("\"riskPercent\":").Append(RiskPercentPerTrade).Append(",");
                sb.Append("\"maxContracts\":").Append(MaxContracts).Append(",");
                sb.Append("\"bridge_version\":\"3.0\",");
                sb.Append("\"listenerUp\":").Append(listenerUp ? "true" : "false").Append(",");
                sb.Append("\"listenerPort\":").Append(SignalListenerPort).Append(",");
                sb.Append("\"dashboardUrl\":\"").Append(Esc(DashboardBaseUrl)).Append("\",");
                sb.Append("\"signalBusActive\":").Append(busActive ? "true" : "false").Append(",");
                sb.Append("\"signalBusEnqueued\":").Append(SignalBus.TotalEnqueued).Append(",");
                sb.Append("\"signalBusDrained\":").Append(SignalBus.TotalDrained).Append(",");
                sb.Append("\"signalBusPending\":").Append(SignalBus.PendingCount).Append(",");
                sb.Append("\"trackedInstruments\":").Append(instrSb).Append(",");
                sb.Append("\"metricsUrl\":\"http://localhost:")
                  .Append(SignalListenerPort).Append("/metrics\",");
                sb.Append("\"timestamp\":\"")
                  .Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")).Append("\"");
                sb.Append("}");
                return sb.ToString();
            }
            catch { return "{\"error\":\"status unavailable\"}"; }
        }

        private string BuildOrdersJson(JavaScriptSerializer serializer)
        {
            try
            {
                List<Dictionary<string, object>> copy;
                lock (_orderLock)
                    copy = new List<Dictionary<string, object>>(_orderEvents);

                return serializer.Serialize(new Dictionary<string, object>
                {
                    { "events",    copy },
                    { "count",     copy.Count },
                    { "timestamp", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") }
                });
            }
            catch { return "{\"events\":[],\"error\":\"orders unavailable\"}"; }
        }

        private string BuildPrometheusMetrics()
        {
            double uptime      = (DateTime.UtcNow - _startTime).TotalSeconds;
            bool   connected   = _account != null && State == NinjaTrader.NinjaScript.State.Realtime;
            bool   listenerUp  = _listener != null && !_listenerStopped;
            bool   busActive   = EnableSignalBus && SignalBus.HasConsumer;

            int    posCount    = 0;
            double cash        = 0;
            double unrealized  = 0;
            double realized    = 0;

            try
            {
                if (_account != null)
                {
                    cash     = _account.Get(AccountItem.CashValue,          Currency.UsDollar);
                    realized = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    if (_account.Positions != null)
                        foreach (Position pos in _account.Positions)
                            if (pos != null && pos.Quantity != 0)
                            {
                                posCount++;
                                try
                                {
                                    double lp = SafeClose();
                                    if (lp > 0)
                                        unrealized += pos.GetUnrealizedProfitLoss(
                                            PerformanceUnit.Currency, lp);
                                }
                                catch { }
                            }
                }
            }
            catch { }

            var sb = new StringBuilder();

            void Gauge(string name, string help, string val)
            {
                sb.AppendLine($"# HELP {name} {help}");
                sb.AppendLine($"# TYPE {name} gauge");
                sb.AppendLine($"{name} {val}");
            }
            void Counter(string name, string help, long val)
            {
                sb.AppendLine($"# HELP {name} {help}");
                sb.AppendLine($"# TYPE {name} counter");
                sb.AppendLine($"{name} {val}");
            }

            Gauge  ("monitor_up",               "Strategy connected and in Realtime state.",  connected   ? "1" : "0");
            Gauge  ("monitor_listener_up",       "HTTP listener running.",                     listenerUp  ? "1" : "0");
            Gauge  ("monitor_signalbus_active",  "SignalBus consumer registered.",             busActive   ? "1" : "0");
            Gauge  ("monitor_risk_blocked",      "Risk engine has blocked trading.",           RiskBlocked ? "1" : "0");
            Gauge  ("monitor_uptime_seconds",    "Seconds since strategy started.",            uptime.ToString("F0"));
            Gauge  ("monitor_positions_count",   "Number of open positions.",                  posCount.ToString());
            Gauge  ("monitor_cash_balance",      "Account cash balance USD.",                  Math.Round(cash,       2).ToString());
            Gauge  ("monitor_unrealized_pnl",    "Total unrealized P&L USD.",                  Math.Round(unrealized, 2).ToString());
            Gauge  ("monitor_realized_pnl",      "Total realized P&L USD.",                    Math.Round(realized,   2).ToString());
            Gauge  ("monitor_signalbus_pending", "Signals waiting in SignalBus queue.",        SignalBus.PendingCount.ToString());

            Counter("monitor_signals_received_total",      "Signals received.",                       Interlocked.Read(ref _metricSignalsReceived));
            Counter("monitor_signals_executed_total",      "Entry signals executed.",                  Interlocked.Read(ref _metricSignalsExecuted));
            Counter("monitor_signals_rejected_total",      "Signals rejected.",                        Interlocked.Read(ref _metricSignalsRejected));
            Counter("monitor_exits_executed_total",        "Flatten/exit operations.",                 Interlocked.Read(ref _metricExitsExecuted));
            Counter("monitor_position_pushes_total",       "Position updates pushed to dashboard.",    Interlocked.Read(ref _metricPositionPushes));
            Counter("monitor_position_push_errors_total",  "Failed position push attempts.",           Interlocked.Read(ref _metricPositionPushErrors));
            Counter("monitor_heartbeats_total",            "Heartbeats sent to dashboard.",            Interlocked.Read(ref _metricHeartbeatsSent));
            Counter("monitor_http_requests_total",         "Inbound HTTP requests handled.",           Interlocked.Read(ref _metricHttpRequests));
            Counter("monitor_orders_filled_total",         "Orders that reached Filled state.",        Interlocked.Read(ref _metricOrdersFilled));
            Counter("monitor_orders_rejected_total",       "Orders rejected by broker.",               Interlocked.Read(ref _metricOrdersRejected));
            Counter("monitor_signalbus_enqueued_total",    "Signals ever enqueued into SignalBus.",    SignalBus.TotalEnqueued);
            Counter("monitor_signalbus_drained_total",     "Signals drained from SignalBus.",          SignalBus.TotalDrained);

            string acct = _account?.Name ?? "disconnected";
            sb.AppendLine("# HELP monitor_info Strategy metadata.");
            sb.AppendLine("# TYPE monitor_info gauge");
            sb.AppendLine($"monitor_info{{version=\"3.0\",account=\"{acct}\"," +
                          $"port=\"{SignalListenerPort}\"}} 1");

            return sb.ToString();
        }

        #endregion

        // =====================================================================
        // Helpers
        // =====================================================================
        #region Helpers

        private double SafeClose()
        {
            try
            {
                if (CurrentBar >= 0 && Close != null && Close.Count > 0)
                    return Close[0];
            }
            catch { }
            return 0;
        }

        private static string Esc(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            return value
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\n", "\\n")
                .Replace("\r", "\\r")
                .Replace("\t", "\\t");
        }

        private static void SendResponse(HttpListenerResponse resp, int code, string body)
        {
            try
            {
                resp.StatusCode      = code;
                resp.ContentType     = resp.ContentType ?? "application/json";
                byte[] buf           = Encoding.UTF8.GetBytes(body ?? "");
                resp.ContentLength64 = buf.Length;
                resp.OutputStream.Write(buf, 0, buf.Length);
                resp.OutputStream.Close();
            }
            catch { }
        }

        private static Dictionary<string, object> Error(string msg) =>
            new Dictionary<string, object> { { "status", "error" }, { "error", msg } };

        private void ThrottledLog(string message)
        {
            if ((DateTime.Now - _lastErrorLog).TotalSeconds >= ERROR_LOG_THROTTLE_SECONDS)
            {
                Print($"[Monitor] {message}");
                _lastErrorLog = DateTime.Now;
            }
        }

        #endregion
    }
}
