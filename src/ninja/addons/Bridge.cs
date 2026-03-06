// =============================================================================
// Bridge.cs  —  NinjaTrader 8 AddOn
// =============================================================================
//
// Drop this file into:
//   Documents\NinjaTrader 8\bin\Custom\AddOns\
//
// PURPOSE
// -------
// Revives the HTTP bridge from the archive MonitorConnection.cs as a proper
// NT8 AddOn (inherits AddOnBase, not Strategy) so it runs automatically when
// NT8 connects to data — independent of any chart or strategy instance.
//
// Responsibilities:
//   1. Host an HTTP listener (default port 5680) with endpoints:
//        GET  /health          — liveness probe
//        GET  /status          — full account snapshot
//        GET  /orders          — recent order event log (last 50)
//        GET  /metrics         — Prometheus exposition format
//        POST /execute_signal  — forward signal JSON to BreakoutStrategy via SignalBus
//        POST /flatten         — flatten all positions immediately
//        POST /cancel_orders   — cancel all working orders
//   2. Push position snapshots to the Python dashboard on every fill and on a
//      15-second heartbeat timer.
//   3. Parse risk-gate feedback from the dashboard response and expose a
//      static RiskBlocked flag that BreakoutStrategy can read.
//   4. Expose Prometheus counters for Grafana (bridge_* metrics matching the
//      ninjatrader-bridge job in futures repo prometheus.yml).
//
// HTTP access from Docker / WSL2 / Tailscale:
//   netsh http add urlacl url=http://+:5680/ user=Everyone
//   (run once as Administrator; listener binds to all interfaces)
//
// CONFIGURATION
// -------------
// Edit the constants in the #region Configuration block below.
//   ListenerPort       — HTTP listener port (default 5680)
//   DashboardBaseUrl   — Python dashboard URL for position push
//   AccountName        — NT8 account name to monitor
//   EnablePositionPush — toggle outbound position push
//   HeartbeatSec       — seconds between heartbeat pushes
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Core;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
    public class Bridge : AddOnBase
    {
        // =====================================================================
        #region Configuration
        // =====================================================================

        /// <summary>HTTP listener port.  Must match the Prometheus scrape
        /// config and dashboard proxy route.</summary>
        private const int ListenerPort = 5680;

        /// <summary>Python dashboard base URL for position push.
        /// Set to empty string to disable outbound push.</summary>
        private const string DashboardBaseUrl = "http://localhost:8000";

        /// <summary>NT8 account name to monitor for positions and orders.</summary>
        private const string AccountName = "Sim101";

        /// <summary>Enable outbound position push to the dashboard.</summary>
        private const bool EnablePositionPush = true;

        /// <summary>Seconds between heartbeat position pushes.</summary>
        private const int HeartbeatSec = 15;

        /// <summary>Bridge version string returned in /health and /status.</summary>
        private const string BridgeVersion = "4.0";

        /// <summary>Max order events to keep in the /orders ring buffer.</summary>
        private const int MaxOrderEvents = 50;

        /// <summary>Seconds to throttle repeated error log messages.</summary>
        private const int ErrorLogThrottleSec = 15;

        #endregion

        // =====================================================================
        #region State
        // =====================================================================

        // ── Account ───────────────────────────────────────────────────────
        private Account _account;

        // ── HTTP listener ─────────────────────────────────────────────────
        private HttpListener _listener;
        private Thread _listenerThread;
        private volatile bool _listenerStopped;
        private readonly ManualResetEventSlim _listenerExited = new ManualResetEventSlim(false);

        // ── Outbound HTTP client ──────────────────────────────────────────
        private HttpClient _httpClient;

        // ── Heartbeat timer ───────────────────────────────────────────────
        private Timer _heartbeatTimer;

        // ── Lifecycle ─────────────────────────────────────────────────────
        private bool _started;
        private bool _cleanedUp;
        private readonly object _startLock = new object();

        // ── Push state ────────────────────────────────────────────────────
        private bool _lastPushSuccess = true;
        private DateTime _lastErrorLog = DateTime.MinValue;

        // ── Order event ring buffer ───────────────────────────────────────
        private readonly List<Dictionary<string, object>> _orderEvents =
            new List<Dictionary<string, object>>();
        private readonly object _orderLock = new object();

        // ── Risk gate ─────────────────────────────────────────────────────
        // Stored on SignalBus (NinjaTrader.NinjaScript.SignalBus) which is
        // compiled into the same NinjaTrader.Custom assembly as both this
        // AddOn and BreakoutStrategy.  Using SignalBus avoids reflection and
        // works correctly in backtest (defaults false) and multi-instance
        // scenarios.  Bridge is the sole writer; BreakoutStrategy reads on
        // every BIP0 bar.
        private static bool IsRiskBlocked
        {
            get { return NinjaTrader.NinjaScript.SignalBus.IsRiskBlocked; }
            set { NinjaTrader.NinjaScript.SignalBus.IsRiskBlocked = value; }
        }
        private static string RiskBlockReason
        {
            get { return NinjaTrader.NinjaScript.SignalBus.RiskBlockReason; }
            set { NinjaTrader.NinjaScript.SignalBus.RiskBlockReason = value; }
        }


        // ── Prometheus counters ───────────────────────────────────────────
        private long _metricSignalsReceived;
        private long _metricSignalsExecuted;
        private long _metricSignalsRejected;
        private long _metricOrdersFilled;
        private long _metricOrdersRejected;
        private long _metricPositionPushes;
        private long _metricPositionPushErrors;
        private long _metricHeartbeatsSent;
        private long _metricHttpRequests;
        private readonly DateTime _startTime = DateTime.UtcNow;

        #endregion

        // =====================================================================
        #region Lifecycle
        // =====================================================================

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "HTTP bridge AddOn — hosts REST endpoints for dashboard " +
                              "integration, pushes live positions to the Python engine, " +
                              "and exposes Prometheus metrics. Runs independently of any " +
                              "chart or strategy instance.";
                Name = "Bridge";
            }
            else if (State == State.Configure)
            {
                _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            }
            else if (State == State.Active)
            {
                Connection.ConnectionStatusUpdate += OnConnectionStatusChanged;
                Out("[Bridge] Active — waiting for data connection on port " + ListenerPort + "...");

                // Handle the case where NT8 was already connected before this
                // AddOn compiled / activated (e.g. after a hot recompile).
                bool alreadyConnected = false;
                lock (Connection.Connections)
                {
                    alreadyConnected = Connection.Connections.Any(c =>
                        c.Status == ConnectionStatus.Connected &&
                        c.Options != null &&
                        c.Options.Provider != Provider.Playback);
                }

                if (alreadyConnected)
                    StartBridge("connection already established at activation");
            }
            else if (State == State.Terminated)
            {
                Connection.ConnectionStatusUpdate -= OnConnectionStatusChanged;
                Cleanup();
            }
        }

        // NT8 AddOns do not have OnWindowCreated/OnWindowDestroyed unless they
        // create NTWindows.  Cleanup is handled in OnStateChange(Terminated).

        private void OnConnectionStatusChanged(object sender, ConnectionStatusEventArgs e)
        {
            try
            {
                if (e.Status == ConnectionStatus.Connected &&
                    e.Connection != null &&
                    e.Connection.Options != null &&
                    e.Connection.Options.Provider != Provider.Playback)
                {
                    StartBridge("connection " + e.Connection.Options.Name + " established");
                }
            }
            catch (Exception ex)
            {
                Out("[Bridge] ConnectionStatusChanged error: " + ex.Message);
            }
        }

        private void StartBridge(string reason)
        {
            lock (_startLock)
            {
                if (_started) return;
                _started = true;
            }

            Out("[Bridge] Starting — " + reason);

            // ── Resolve account ───────────────────────────────────────────
            try
            {
                lock (Account.All)
                    _account = Account.All.FirstOrDefault(a => a.Name == AccountName);

                if (_account != null)
                {
                    _account.OrderUpdate += OnOrderUpdate;
                    _account.PositionUpdate += OnPositionUpdate;
                    Out("[Bridge] Account: " + _account.Name);
                }
                else
                {
                    Out("[Bridge] ⚠ Account '" + AccountName + "' not found — position push disabled.");
                }
            }
            catch (Exception ex)
            {
                Out("[Bridge] Account resolution error: " + ex.Message);
            }

            // ── Start HTTP listener ───────────────────────────────────────
            StartListener();

            // ── Start heartbeat timer ─────────────────────────────────────
            if (EnablePositionPush)
            {
                _heartbeatTimer = new Timer(HeartbeatCallback, null,
                    TimeSpan.FromSeconds(HeartbeatSec),
                    TimeSpan.FromSeconds(HeartbeatSec));
                Out("[Bridge] Heartbeat timer started (" + HeartbeatSec + "s interval)");
            }

            // ── Initial position push ─────────────────────────────────────
            if (EnablePositionPush)
                SendPositionUpdate();
        }

        private void Cleanup()
        {
            if (_cleanedUp) return;
            _cleanedUp = true;

            StopListener();

            try { _heartbeatTimer?.Dispose(); } catch { }
            _heartbeatTimer = null;

            try
            {
                if (_account != null)
                {
                    _account.OrderUpdate -= OnOrderUpdate;
                    _account.PositionUpdate -= OnPositionUpdate;
                }
            }
            catch { }

            try { _httpClient?.Dispose(); } catch { }
            _httpClient = null;
            _account = null;

            Out("[Bridge] Shutdown complete.");
        }

        #endregion

        // =====================================================================
        #region HTTP Listener
        // =====================================================================

        private void StartListener()
        {
            // Try wildcard binding first (accessible from Docker / WSL2 / Tailscale).
            // Requires a one-time URL reservation as Administrator:
            //   netsh http add urlacl url=http://+:5680/ user=Everyone
            // Falls back to localhost if wildcard binding is refused.
            string[] prefixes =
            {
                "http://+:" + ListenerPort + "/",
                "http://localhost:" + ListenerPort + "/"
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
                        { IsBackground = true, Name = "BridgeListener" };
                        _listenerThread.Start();

                        bool wildcard = prefix.Contains("+");
                        Out("[Bridge] Listening on port " + ListenerPort +
                            (wildcard ? " (all interfaces)" : " (localhost only)"));

                        if (!wildcard)
                            Out("[Bridge] For remote access run once as admin: " +
                                "netsh http add urlacl url=http://+:" + ListenerPort + "/ user=Everyone");
                        return;
                    }
                    catch (Exception ex)
                    {
                        Out("[Bridge] Listener " + attempt + "/3 on " + prefix + ": " + ex.Message);
                        try { _listener?.Close(); } catch { }
                        _listener = null;
                        if (attempt < 3) Thread.Sleep(800);
                    }
                }
            }

            Out("[Bridge] All listener attempts failed — HTTP unavailable.");
        }

        private void StopListener()
        {
            _listenerStopped = true;
            try { _listener?.Stop(); } catch { }
            try { _listener?.Close(); } catch { }
            _listener = null;

            if (_listenerThread != null && _listenerThread.IsAlive)
            {
                if (!_listenerExited.Wait(3000))
                    Out("[Bridge] Warning: listener thread did not exit within 3 s");
                _listenerThread = null;
            }
        }

        private void ListenLoop()
        {
            var serializer = new JavaScriptSerializer();
            try
            {
                while (!_listenerStopped && _listener != null && _listener.IsListening)
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

                        string path = ctx.Request.Url.AbsolutePath.TrimEnd('/');
                        string method = ctx.Request.HttpMethod;

                        // ── POST /execute_signal ──────────────────────────
                        if (method == "POST" && path == "/execute_signal")
                        {
                            using (var r = new System.IO.StreamReader(ctx.Request.InputStream))
                            {
                                string json = r.ReadToEnd();
                                Interlocked.Increment(ref _metricSignalsReceived);

                                // Forward to BreakoutStrategy via the static SignalBus.
                                // The strategy's DrainSignalBus will pick it up on the
                                // next bar and route it to BridgeOrderEngine.
                                try
                                {
                                    var signal = new SignalBus.Signal
                                    {
                                        Direction = "long",
                                        SignalType = "entry",
                                        Quantity = 1,
                                        OrderType = "market",
                                        Strategy = "Bridge",
                                        Timestamp = DateTime.UtcNow
                                    };

                                    // Parse the incoming JSON and populate the signal
                                    var data = serializer.Deserialize<Dictionary<string, object>>(json);
                                    if (data != null)
                                    {
                                        signal.Direction = GetStr(data, "direction", "long");
                                        signal.SignalType = GetStr(data, "signal_type", "entry");
                                        signal.Quantity = GetInt(data, "quantity", 1);
                                        signal.OrderType = GetStr(data, "order_type", "market");
                                        signal.LimitPrice = GetDbl(data, "limit_price", 0);
                                        signal.StopLoss = GetDbl(data, "stop_loss", 0);
                                        signal.TakeProfit = GetDbl(data, "take_profit", 0);
                                        signal.TakeProfit2 = GetDbl(data, "take_profit_2", 0);
                                        signal.Strategy = GetStr(data, "strategy", "Bridge");
                                        signal.Asset = GetStr(data, "asset", "");
                                        signal.SignalId = GetStr(data, "signal_id",
                                            "brg-" + Guid.NewGuid().ToString("N").Substring(0, 8));
                                        signal.SignalQuality = GetDbl(data, "signal_quality", 0);
                                    }

                                    SignalBus.Enqueue(signal);
                                    Interlocked.Increment(ref _metricSignalsExecuted);

                                    var result = new Dictionary<string, object>
                                    {
                                        { "status", "queued" },
                                        { "signal_id", signal.SignalId },
                                        { "bus_pending", SignalBus.PendingCount }
                                    };
                                    SendResponse(ctx.Response, 200, serializer.Serialize(result));
                                }
                                catch (Exception ex)
                                {
                                    Interlocked.Increment(ref _metricSignalsRejected);
                                    var err = new Dictionary<string, object>
                                    {
                                        { "status", "error" },
                                        { "message", ex.Message }
                                    };
                                    SendResponse(ctx.Response, 500, serializer.Serialize(err));
                                }
                            }
                        }

                        // ── POST /flatten ─────────────────────────────────
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

                            var result = FlattenAllPositions(reason);
                            SendResponse(ctx.Response, 200, serializer.Serialize(result));
                        }

                        // ── POST /cancel_orders ───────────────────────────
                        else if (method == "POST" && path == "/cancel_orders")
                        {
                            var result = CancelAllWorkingOrders();
                            SendResponse(ctx.Response, 200, serializer.Serialize(result));
                        }

                        // ── GET /status ───────────────────────────────────
                        else if (method == "GET" && path == "/status")
                        {
                            SendResponse(ctx.Response, 200, BuildStatusJson());
                        }

                        // ── GET /orders ───────────────────────────────────
                        else if (method == "GET" && path == "/orders")
                        {
                            SendResponse(ctx.Response, 200, BuildOrdersJson(serializer));
                        }

                        // ── GET /health ───────────────────────────────────
                        else if (method == "GET" && path == "/health")
                        {
                            SendResponse(ctx.Response, 200,
                                "{\"status\":\"ok\",\"bridge_version\":\"" + BridgeVersion + "\"}");
                        }

                        // ── GET /metrics ──────────────────────────────────
                        else if (method == "GET" && path == "/metrics")
                        {
                            ctx.Response.ContentType = "text/plain; version=0.0.4; charset=utf-8";
                            SendResponse(ctx.Response, 200, BuildPrometheusMetrics());
                        }

                        // ── 404 ───────────────────────────────────────────
                        else
                        {
                            SendResponse(ctx.Response, 404,
                                "{\"error\":\"not found\"," +
                                "\"endpoints\":[" +
                                "\"/execute_signal\",\"/flatten\",\"/cancel_orders\"," +
                                "\"/status\",\"/orders\",\"/health\",\"/metrics\"]}");
                        }
                    }
                    catch (HttpListenerException) { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception ex)
                    {
                        if (_listenerStopped) break;
                        ThrottledLog("ListenLoop: " + ex.Message);
                    }
                }
            }
            finally { _listenerExited.Set(); }
        }

        #endregion

        // =====================================================================
        #region Account Events
        // =====================================================================

        private void OnOrderUpdate(object sender, OrderEventArgs e)
        {
            try
            {
                if (e.Order == null) return;

                var evt = new Dictionary<string, object>
                {
                    { "time",       DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") },
                    { "orderId",    e.Order.OrderId },
                    { "name",       e.Order.Name ?? "" },
                    { "instrument", e.Order.Instrument?.FullName ?? "" },
                    { "action",     e.Order.OrderAction.ToString() },
                    { "type",       e.Order.OrderType.ToString() },
                    { "quantity",   e.Order.Quantity },
                    { "state",      e.Order.OrderState.ToString() },
                    { "limitPrice", e.Order.LimitPrice },
                    { "stopPrice",  e.Order.StopPrice }
                };

                lock (_orderLock)
                {
                    _orderEvents.Add(evt);
                    while (_orderEvents.Count > MaxOrderEvents)
                        _orderEvents.RemoveAt(0);
                }

                // Track fills and rejections
                if (e.Order.OrderState == OrderState.Filled)
                {
                    Interlocked.Increment(ref _metricOrdersFilled);
                    if (EnablePositionPush) SendPositionUpdate();
                }
                else if (e.Order.OrderState == OrderState.Rejected)
                {
                    Interlocked.Increment(ref _metricOrdersRejected);
                }
            }
            catch (Exception ex)
            {
                ThrottledLog("OnOrderUpdate: " + ex.Message);
            }
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            try
            {
                if (EnablePositionPush) SendPositionUpdate();
            }
            catch (Exception ex)
            {
                ThrottledLog("OnPositionUpdate: " + ex.Message);
            }
        }

        private void HeartbeatCallback(object state)
        {
            try
            {
                Interlocked.Increment(ref _metricHeartbeatsSent);
                if (EnablePositionPush) SendPositionUpdate();
            }
            catch (Exception ex)
            {
                ThrottledLog("Heartbeat: " + ex.Message);
            }
        }

        #endregion

        // =====================================================================
        #region Position Push
        // =====================================================================

        private void SendPositionUpdate()
        {
            if (!EnablePositionPush || _account == null) return;
            var client = _httpClient;
            if (client == null) return;
            if (string.IsNullOrWhiteSpace(DashboardBaseUrl)) return;

            Interlocked.Increment(ref _metricPositionPushes);
            try
            {
                double cashBalance = 0;
                double realizedPnL = 0;

                try
                {
                    cashBalance = _account.Get(AccountItem.CashValue, Currency.UsDollar);
                    realizedPnL = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                }
                catch { }

                var sb = new StringBuilder(2048);
                sb.Append("{");
                sb.Append("\"account\":\"").Append(_account.Name).Append("\",");
                sb.Append("\"positions\":[");

                bool firstPos = true;
                double totalUnrealizedPnL = 0;
                int posCount = 0;

                if (_account.Positions != null)
                {
                    foreach (Position pos in _account.Positions)
                    {
                        try
                        {
                            if (pos == null || pos.Quantity == 0 || pos.Instrument == null)
                                continue;

                            posCount++;
                            double pnl = 0;
                            try
                            {
                                pnl = pos.GetUnrealizedProfitLoss(PerformanceUnit.Currency);
                            }
                            catch { }

                            totalUnrealizedPnL += pnl;

                            string dir = pos.MarketPosition == MarketPosition.Long ? "long" : "short";

                            if (!firstPos) sb.Append(",");
                            sb.Append("{");
                            sb.Append("\"symbol\":\"")
                              .Append(pos.Instrument.MasterInstrument.Name).Append("\",");
                            sb.Append("\"direction\":\"").Append(dir).Append("\",");
                            sb.Append("\"quantity\":").Append(pos.Quantity).Append(",");
                            sb.Append("\"entry_price\":").Append(pos.AveragePrice).Append(",");
                            sb.Append("\"unrealized_pnl\":").Append(Math.Round(pnl, 2));
                            sb.Append("}");
                            firstPos = false;
                        }
                        catch (Exception ex) { ThrottledLog("Position read: " + ex.Message); }
                    }
                }

                sb.Append("],");
                sb.Append("\"cash_balance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realized_pnl\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"unrealized_pnl\":").Append(Math.Round(totalUnrealizedPnL, 2)).Append(",");

                // Pending orders count
                int pendingOrdCount = 0;
                try
                {
                    if (_account.Orders != null)
                        pendingOrdCount = _account.Orders.Count(o =>
                            o.OrderState == OrderState.Working ||
                            o.OrderState == OrderState.Accepted);
                }
                catch { }

                sb.Append("\"pending_orders\":").Append(pendingOrdCount).Append(",");
                sb.Append("\"risk_blocked\":").Append(IsRiskBlocked ? "true" : "false");
                sb.Append("}");

                string url = DashboardBaseUrl.TrimEnd('/') + "/api/positions/update";
                var content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json");

                client.PostAsync(url, content).ContinueWith(t =>
                {
                    if (t.IsFaulted)
                    {
                        Interlocked.Increment(ref _metricPositionPushErrors);
                        if (_lastPushSuccess || (DateTime.Now - _lastErrorLog).TotalSeconds > 30)
                        {
                            _lastErrorLog = DateTime.Now;
                            _lastPushSuccess = false;
                        }
                    }
                    else
                    {
                        if (!_lastPushSuccess)
                        {
                            Out("[Bridge] Position push connection restored");
                            _lastPushSuccess = true;
                        }
                        ParseRiskFeedback(t.Result);
                    }
                });
            }
            catch (Exception ex) { ThrottledLog("SendPositionUpdate: " + ex.Message); }
        }

        private void ParseRiskFeedback(HttpResponseMessage response)
        {
            try
            {
                string body = response.Content.ReadAsStringAsync().Result;
                if (string.IsNullOrEmpty(body)) return;

                var data = new JavaScriptSerializer()
                    .Deserialize<Dictionary<string, object>>(body);

                if (data == null) return;

                // Check top-level can_trade
                bool canTrade = true;
                string blockReason = "";

                // Try nested risk object first (futures engine format)
                if (data.ContainsKey("risk") && data["risk"] is Dictionary<string, object> risk)
                {
                    if (risk.ContainsKey("can_trade"))
                        try { canTrade = Convert.ToBoolean(risk["can_trade"]); } catch { }

                    if (risk.ContainsKey("block_reason") && risk["block_reason"] != null)
                        blockReason = risk["block_reason"].ToString();
                }
                // Try flat format
                else
                {
                    if (data.ContainsKey("can_trade"))
                        try { canTrade = Convert.ToBoolean(data["can_trade"]); } catch { }

                    if (data.ContainsKey("block_reason") && data["block_reason"] != null)
                        blockReason = data["block_reason"].ToString();
                }

                bool wasBlocked = IsRiskBlocked;
                IsRiskBlocked = !canTrade;
                RiskBlockReason = blockReason;

                if (!canTrade && !wasBlocked)
                    Out("[Bridge] ⚠ Risk BLOCKED: " + blockReason);
                else if (canTrade && wasBlocked)
                    Out("[Bridge] ✓ Risk block cleared");
            }
            catch { }
        }

        #endregion

        // =====================================================================
        #region Order Actions (flatten / cancel)
        // =====================================================================

        private Dictionary<string, object> FlattenAllPositions(string reason)
        {
            var result = new Dictionary<string, object>();
            try
            {
                if (_account == null)
                {
                    result["status"] = "error";
                    result["message"] = "account not connected";
                    return result;
                }

                int flattenedCount = 0;

                if (_account.Positions != null)
                {
                    foreach (Position pos in _account.Positions)
                    {
                        try
                        {
                            if (pos == null || pos.Quantity == 0 || pos.Instrument == null)
                                continue;

                            _account.Flatten(new[] { pos.Instrument });
                            flattenedCount++;
                            Out("[Bridge] Flattened " + pos.Instrument.MasterInstrument.Name +
                                " (" + pos.Quantity + " @ " + pos.AveragePrice + ") reason=" + reason);
                        }
                        catch (Exception ex)
                        {
                            Out("[Bridge] Flatten error for " +
                                (pos.Instrument?.MasterInstrument?.Name ?? "?") + ": " + ex.Message);
                        }
                    }
                }

                result["status"] = "ok";
                result["flattened"] = flattenedCount;
                result["reason"] = reason;
            }
            catch (Exception ex)
            {
                result["status"] = "error";
                result["message"] = ex.Message;
            }
            return result;
        }

        private Dictionary<string, object> CancelAllWorkingOrders()
        {
            var result = new Dictionary<string, object>();
            try
            {
                if (_account == null)
                {
                    result["status"] = "error";
                    result["message"] = "account not connected";
                    return result;
                }

                int cancelledCount = 0;

                if (_account.Orders != null)
                {
                    foreach (Order ord in _account.Orders)
                    {
                        try
                        {
                            if (ord == null) continue;
                            if (ord.OrderState != OrderState.Working &&
                                ord.OrderState != OrderState.Accepted)
                                continue;

                            _account.Cancel(new[] { ord });
                            cancelledCount++;
                            Out("[Bridge] Cancelled order " + ord.OrderId + " " +
                                ord.Name + " " + ord.Instrument?.MasterInstrument?.Name);
                        }
                        catch (Exception ex)
                        {
                            Out("[Bridge] Cancel error for order " + ord.OrderId + ": " + ex.Message);
                        }
                    }
                }

                result["status"] = "ok";
                result["cancelled"] = cancelledCount;
            }
            catch (Exception ex)
            {
                result["status"] = "error";
                result["message"] = ex.Message;
            }
            return result;
        }

        #endregion

        // =====================================================================
        #region JSON Builders
        // =====================================================================

        private string BuildStatusJson()
        {
            try
            {
                int posCount = 0;
                double cashBalance = 0;
                double realizedPnL = 0;
                double unrealizedPnL = 0;
                int pendingOrdCount = 0;

                try
                {
                    if (_account != null)
                    {
                        cashBalance = _account.Get(AccountItem.CashValue, Currency.UsDollar);
                        realizedPnL = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    }
                }
                catch { }

                try
                {
                    if (_account != null && _account.Positions != null)
                        foreach (Position pos in _account.Positions)
                        {
                            if (pos == null || pos.Quantity == 0) continue;
                            posCount++;
                            try
                            {
                                unrealizedPnL += pos.GetUnrealizedProfitLoss(
                                    PerformanceUnit.Currency);
                            }
                            catch { }
                        }
                }
                catch { }

                try
                {
                    if (_account != null && _account.Orders != null)
                        pendingOrdCount = _account.Orders.Count(o =>
                            o.OrderState == OrderState.Working ||
                            o.OrderState == OrderState.Accepted);
                }
                catch { }

                bool listenerUp = _listener != null && !_listenerStopped;

                var sb = new StringBuilder(1024);
                sb.Append("{");
                sb.Append("\"account\":\"").Append(_account?.Name ?? "disconnected").Append("\",");
                sb.Append("\"connected\":").Append(_account != null ? "true" : "false").Append(",");
                sb.Append("\"bridge_version\":\"").Append(BridgeVersion).Append("\",");
                sb.Append("\"positions\":").Append(posCount).Append(",");
                sb.Append("\"pending_orders\":").Append(pendingOrdCount).Append(",");
                sb.Append("\"cash_balance\":").Append(Math.Round(cashBalance, 2)).Append(",");
                sb.Append("\"realized_pnl\":").Append(Math.Round(realizedPnL, 2)).Append(",");
                sb.Append("\"unrealized_pnl\":").Append(Math.Round(unrealizedPnL, 2)).Append(",");
                sb.Append("\"risk_blocked\":").Append(IsRiskBlocked ? "true" : "false").Append(",");
                sb.Append("\"risk_block_reason\":\"").Append(Esc(RiskBlockReason)).Append("\",");
                sb.Append("\"listener_up\":").Append(listenerUp ? "true" : "false").Append(",");
                sb.Append("\"listener_port\":").Append(ListenerPort).Append(",");
                sb.Append("\"dashboard_url\":\"").Append(Esc(DashboardBaseUrl)).Append("\",");
                sb.Append("\"signalbus_pending\":").Append(SignalBus.PendingCount).Append(",");
                sb.Append("\"signalbus_enqueued\":").Append(SignalBus.TotalEnqueued).Append(",");
                sb.Append("\"signalbus_drained\":").Append(SignalBus.TotalDrained).Append(",");
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
            double uptime = (DateTime.UtcNow - _startTime).TotalSeconds;
            bool connected = _account != null;
            bool listenerUp = _listener != null && !_listenerStopped;

            int posCount = 0;
            double cash = 0;
            double unrealized = 0;
            double realized = 0;

            try
            {
                if (_account != null)
                {
                    cash = _account.Get(AccountItem.CashValue, Currency.UsDollar);
                    realized = _account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                    if (_account.Positions != null)
                        foreach (Position pos in _account.Positions)
                            if (pos != null && pos.Quantity != 0)
                            {
                                posCount++;
                                try
                                {
                                    unrealized += pos.GetUnrealizedProfitLoss(
                                        PerformanceUnit.Currency);
                                }
                                catch { }
                            }
                }
            }
            catch { }

            var sb = new StringBuilder(4096);

            // Gauge helper
            Action<string, string, string> Gauge = (name, help, val) =>
            {
                sb.AppendLine("# HELP " + name + " " + help);
                sb.AppendLine("# TYPE " + name + " gauge");
                sb.AppendLine(name + " " + val);
            };
            // Counter helper
            Action<string, string, long> Counter = (name, help, val) =>
            {
                sb.AppendLine("# HELP " + name + " " + help);
                sb.AppendLine("# TYPE " + name + " counter");
                sb.AppendLine(name + " " + val.ToString());
            };

            Gauge("bridge_up", "Bridge connected to account.", connected ? "1" : "0");
            Gauge("bridge_listener_up", "HTTP listener running.", listenerUp ? "1" : "0");
            Gauge("bridge_positions_count", "Number of open positions.", posCount.ToString());
            Gauge("bridge_cash_balance", "Account cash balance USD.", Math.Round(cash, 2).ToString());
            Gauge("bridge_unrealized_pnl", "Total unrealized P&L USD.", Math.Round(unrealized, 2).ToString());
            Gauge("bridge_realized_pnl", "Total realized P&L USD.", Math.Round(realized, 2).ToString());
            Gauge("bridge_uptime_seconds", "Seconds since bridge started.", uptime.ToString("F0"));

            Counter("bridge_signals_received_total", "Signals received via /execute_signal.",
                Interlocked.Read(ref _metricSignalsReceived));
            Counter("bridge_signals_executed_total", "Signals queued to SignalBus.",
                Interlocked.Read(ref _metricSignalsExecuted));
            Counter("bridge_signals_rejected_total", "Signals rejected.",
                Interlocked.Read(ref _metricSignalsRejected));
            Counter("bridge_orders_filled_total", "Orders that reached Filled state.",
                Interlocked.Read(ref _metricOrdersFilled));
            Counter("bridge_orders_rejected_total", "Orders rejected by broker.",
                Interlocked.Read(ref _metricOrdersRejected));
            Counter("bridge_heartbeats_total", "Heartbeat position pushes sent.",
                Interlocked.Read(ref _metricHeartbeatsSent));

            // Info metric
            string acct = _account?.Name ?? "disconnected";
            sb.AppendLine("# HELP bridge_info Bridge metadata.");
            sb.AppendLine("# TYPE bridge_info gauge");
            sb.AppendLine("bridge_info{version=\"" + BridgeVersion + "\",account=\"" + acct +
                          "\",port=\"" + ListenerPort + "\"} 1");

            return sb.ToString();
        }

        #endregion

        // =====================================================================
        #region Helpers
        // =====================================================================

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

        private static void SendResponse(HttpListenerResponse resp, int code, string body)
        {
            try
            {
                resp.StatusCode = code;
                if (string.IsNullOrEmpty(resp.ContentType))
                    resp.ContentType = "application/json; charset=utf-8";
                byte[] buf = Encoding.UTF8.GetBytes(body ?? "");
                resp.ContentLength64 = buf.Length;
                resp.OutputStream.Write(buf, 0, buf.Length);
                resp.Close();
            }
            catch { try { resp.Close(); } catch { } }
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

        private void ThrottledLog(string message)
        {
            var now = DateTime.Now;
            if ((now - _lastErrorLog).TotalSeconds >= ErrorLogThrottleSec)
            {
                _lastErrorLog = now;
                Out("[Bridge] " + message);
            }
        }

        private void Out(string message)
        {
            try
            {
                NinjaTrader.Code.Output.Process(message, PrintTo.OutputTab1);
            }
            catch { }
        }

        private static string GetStr(Dictionary<string, object> d, string key, string def)
        {
            if (d.ContainsKey(key) && d[key] != null) return d[key].ToString();
            return def;
        }

        private static int GetInt(Dictionary<string, object> d, string key, int def)
        {
            if (d.ContainsKey(key) && d[key] != null)
                try { return (int)Math.Round(Convert.ToDouble(d[key])); } catch { }
            return def;
        }

        private static double GetDbl(Dictionary<string, object> d, string key, double def)
        {
            if (d.ContainsKey(key) && d[key] != null)
                try { return Convert.ToDouble(d[key]); } catch { }
            return def;
        }

        #endregion
    }
}
