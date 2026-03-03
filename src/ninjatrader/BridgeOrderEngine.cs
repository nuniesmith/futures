#region Using declarations
using System;
using System.Collections.Generic;
using System.Web.Script.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

// =============================================================================
// BridgeOrderEngine — Shared order execution engine
// =============================================================================
//
// Contains all order submission logic that was previously embedded in Bridge.
// Both Bridge and BreakoutStrategy instantiate one of these to handle:
//
//   - Signal JSON parsing (ProcessSignal)
//   - Risk sizing (balance × risk% ÷ SL distance × point value)
//   - BIP-aware order routing (SubmitOrderUnmanaged with correct series index)
//   - Auto-bracket submission (SL + TP1 + optional TP2)
//   - SignalBus draining (DrainSignalBus — called every OnBarUpdate)
//   - Order queue flushing (FlushOrderQueue — called every OnBarUpdate)
//   - FlattenAll / CancelAllOrders helpers
//
// The engine holds no state beyond the order queue and the callbacks injected
// at construction time.  It never touches NinjaScript directly — all order
// submission is done via the Strategy reference passed in, which keeps the
// engine testable and decoupled.
//
// Thread-safety:
//   _orderQueue is only ever written from the HTTP listener thread and only
//   ever read/drained from the NinjaScript main thread (OnBarUpdate).
//   The queueLock guards all access.
//
// Usage:
//   // In State.DataLoaded:
//   _engine = new BridgeOrderEngine(
//       strategy:           this,
//       symbolToBip:        _symbolToBip,
//       getMyAccount:       () => myAccount,
//       getAccountSize:     () => AccountSize,
//       getRiskPercent:     () => RiskPercentPerTrade,
//       getMaxContracts:    () => MaxMicroContracts,
//       getDefaultSlTicks:  () => DefaultStopLossTicks,
//       getDefaultTpTicks:  () => DefaultTakeProfitTicks,
//       getAutoBrackets:    () => EnableAutoBrackets,
//       getRiskBlocked:     () => RiskBlocked,
//       getRiskBlockReason: () => RiskBlockReason,
//       getRiskEnforcement: () => EnableRiskEnforcement,
//       onSignalReceived:   () => Interlocked.Increment(ref metricSignalsReceived),
//       onSignalExecuted:   () => Interlocked.Increment(ref metricSignalsExecuted),
//       onSignalRejected:   () => Interlocked.Increment(ref metricSignalsRejected),
//       onExitExecuted:     () => Interlocked.Increment(ref metricExitsExecuted),
//       onBusDrained:       n  => Interlocked.Add(ref metricSignalBusDrained, n),
//       sendPositionUpdate: SendPositionUpdate
//   );
//
//   // In OnBarUpdate (BIP 0 only):
//   _engine.DrainSignalBus(State);
//   _engine.FlushOrderQueue(State);
//
// =============================================================================

namespace NinjaTrader.NinjaScript.Strategies
{
    internal sealed class BridgeOrderEngine
    {
        // ── Strategy reference ────────────────────────────────────────────────
        // Used solely for SubmitOrderUnmanaged, BarsArray, TickSize, and Print.
        private readonly Strategy _strategy;

        // ── BIP routing table — injected from Bridge/BreakoutStrategy ─────────
        private readonly Dictionary<string, int> _symbolToBip;

        // ── Callbacks for reading caller-owned properties ─────────────────────
        private readonly Func<Account>  _getMyAccount;
        private readonly Func<double>   _getAccountSize;
        private readonly Func<double>   _getRiskPercent;
        private readonly Func<int>      _getMaxContracts;
        private readonly Func<int>      _getDefaultSlTicks;
        private readonly Func<int>      _getDefaultTpTicks;
        private readonly Func<bool>     _getAutoBrackets;
        private readonly Func<bool>     _getRiskBlocked;
        private readonly Func<string>   _getRiskBlockReason;
        private readonly Func<bool>     _getRiskEnforcement;

        // ── Metric callbacks ──────────────────────────────────────────────────
        private readonly Action         _onSignalReceived;
        private readonly Action         _onSignalExecuted;
        private readonly Action         _onSignalRejected;
        private readonly Action         _onExitExecuted;
        private readonly Action<long>   _onBusDrained;

        // ── Position push callback ────────────────────────────────────────────
        private readonly Action _sendPositionUpdate;

        // ── Order queue (entries queued by HTTP thread, drained by NT thread) ─
        private readonly Queue<Action> _orderQueue = new Queue<Action>();
        private readonly object        _queueLock  = new object();

        // ── Strategy name tag for log lines ──────────────────────────────────
        private readonly string _tag;

        // =====================================================================
        // Constructor
        // =====================================================================

        internal BridgeOrderEngine(
            Strategy            strategy,
            Dictionary<string, int> symbolToBip,
            Func<Account>       getMyAccount,
            Func<double>        getAccountSize,
            Func<double>        getRiskPercent,
            Func<int>           getMaxContracts,
            Func<int>           getDefaultSlTicks,
            Func<int>           getDefaultTpTicks,
            Func<bool>          getAutoBrackets,
            Func<bool>          getRiskBlocked,
            Func<string>        getRiskBlockReason,
            Func<bool>          getRiskEnforcement,
            Action              onSignalReceived,
            Action              onSignalExecuted,
            Action              onSignalRejected,
            Action              onExitExecuted,
            Action<long>        onBusDrained,
            Action              sendPositionUpdate,
            string              tag = "Engine")
        {
            _strategy           = strategy;
            _symbolToBip        = symbolToBip;
            _getMyAccount       = getMyAccount;
            _getAccountSize     = getAccountSize;
            _getRiskPercent     = getRiskPercent;
            _getMaxContracts    = getMaxContracts;
            _getDefaultSlTicks  = getDefaultSlTicks;
            _getDefaultTpTicks  = getDefaultTpTicks;
            _getAutoBrackets    = getAutoBrackets;
            _getRiskBlocked     = getRiskBlocked;
            _getRiskBlockReason = getRiskBlockReason;
            _getRiskEnforcement = getRiskEnforcement;
            _onSignalReceived   = onSignalReceived;
            _onSignalExecuted   = onSignalExecuted;
            _onSignalRejected   = onSignalRejected;
            _onExitExecuted     = onExitExecuted;
            _onBusDrained       = onBusDrained;
            _sendPositionUpdate = sendPositionUpdate;
            _tag                = tag;
        }

        // =====================================================================
        // SignalBus drain — call from OnBarUpdate (BIP 0 only)
        // =====================================================================

        /// <summary>
        /// Drain all pending signals from SignalBus and route them to the
        /// appropriate execution path based on current NinjaScript State.
        ///
        /// Historical (backtest): execute directly on the calling thread —
        ///   SubmitOrderUnmanaged must be called synchronously from OnBarUpdate.
        ///
        /// Realtime: parse the signal JSON and enqueue a lambda; the lambda
        ///   is dequeued and executed in FlushOrderQueue on the next bar.
        /// </summary>
        internal void DrainSignalBus(State currentState)
        {
            var signals = SignalBus.DrainAll();
            if (signals.Count == 0) return;

            _onBusDrained?.Invoke(signals.Count);

            foreach (var sig in signals)
            {
                try
                {
                    if (sig.SignalType == "exit")
                    {
                        string reason = !string.IsNullOrEmpty(sig.ExitReason)
                            ? sig.ExitReason
                            : "signal_bus_exit";

                        _onSignalReceived?.Invoke();
                        Log($"SignalBus EXIT: reason={reason} strategy={sig.Strategy} asset={sig.Asset}");

                        if (currentState == State.Historical)
                            ExecuteFlattenDirect($"{sig.Strategy}:{reason}");
                        else
                            FlattenAll($"{sig.Strategy}:{reason}");
                    }
                    else
                    {
                        Log($"SignalBus ENTRY: {sig.Direction?.ToUpper()} {sig.Asset} " +
                            $"Q={sig.SignalQuality:P0} id={sig.SignalId}");

                        if (currentState == State.Historical)
                            ExecuteEntryDirect(sig);
                        else
                            ProcessSignal(sig.ToJson());
                    }
                }
                catch (Exception ex)
                {
                    Log($"SignalBus processing error: {ex.Message}");
                }
            }
        }

        // =====================================================================
        // Order queue flush — call from OnBarUpdate (BIP 0 only)
        // =====================================================================

        /// <summary>
        /// Dequeue and execute all pending order lambdas on the NinjaScript
        /// main thread.  Must only be called from OnBarUpdate.
        /// </summary>
        internal void FlushOrderQueue(State currentState)
        {
            if (currentState != State.Realtime && currentState != State.Historical)
                return;

            lock (_queueLock)
            {
                while (_orderQueue.Count > 0)
                {
                    var action = _orderQueue.Dequeue();
                    try   { action(); }
                    catch (Exception ex) { Log($"Queued order error: {ex.Message}"); }
                }
            }
        }

        // =====================================================================
        // ProcessSignal — parse JSON, risk-size, enqueue order lambda
        // =====================================================================

        /// <summary>
        /// Parse an incoming signal JSON string, apply risk sizing, and enqueue
        /// an order lambda for execution on the next FlushOrderQueue call.
        ///
        /// Signal JSON format:
        /// {
        ///   "direction":   "long" | "short",
        ///   "quantity":    1,
        ///   "order_type":  "market" | "limit" | "stop",
        ///   "limit_price": 0.0,
        ///   "stop_loss":   5200.00,   // exact SL price  (0 = use default ticks)
        ///   "take_profit": 5225.00,   // exact TP1 price (0 = use default ticks)
        ///   "tp2":         5240.00,   // exact TP2 price (0 = no TP2)
        ///   "strategy":    "Ruby",
        ///   "asset":       "MGC",     // used to resolve BIP index
        ///   "signal_id":   "abc123"
        /// }
        ///
        /// Returns a response dictionary suitable for JSON serialisation.
        /// </summary>
        internal Dictionary<string, object> ProcessSignal(string json)
        {
            var response = new Dictionary<string, object>();
            _onSignalReceived?.Invoke();

            try
            {
                var serializer = new JavaScriptSerializer();
                var signal     = serializer.Deserialize<Dictionary<string, object>>(json);

                string signalId     = GetStr(signal, "signal_id",   NewId());
                string dir          = GetStr(signal, "direction",   "long").ToLower();
                int    requestedQty = GetInt(signal, "quantity",    1);
                string typeStr      = GetStr(signal, "order_type",  "market").ToLower();
                double limitPrice   = GetDbl(signal, "limit_price", 0);
                double slPrice      = GetDbl(signal, "stop_loss",   0);
                double tpPrice      = GetDbl(signal, "take_profit", 0);
                double tp2Price     = GetDbl(signal, "tp2",         0);
                string strategy     = GetStr(signal, "strategy",    "");
                string asset        = GetStr(signal, "asset",       "");

                int    bip       = ResolveBip(asset);
                double tickSize  = GetTickSize(bip);
                double pointVal  = GetPointValue(bip);

                // ── Risk gate ─────────────────────────────────────────────────
                if (_getRiskEnforcement() && _getRiskBlocked())
                {
                    string msg = $"Signal rejected — risk blocked: {_getRiskBlockReason()}";
                    Log($"⚠️  {msg}");
                    _onSignalRejected?.Invoke();
                    response["status"]    = "rejected";
                    response["reason"]    = msg;
                    response["signal_id"] = signalId;
                    return response;
                }

                // ── Order type ────────────────────────────────────────────────
                OrderAction action     = dir == "long" ? OrderAction.Buy       : OrderAction.SellShort;
                OrderAction exitAction = dir == "long" ? OrderAction.Sell      : OrderAction.BuyToCover;
                OrderType   ot         = OrderType.Market;
                double      stopPrice  = 0;

                if      (typeStr == "limit") { ot = OrderType.Limit; }
                else if (typeStr == "stop")  { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

                // ── Risk sizing ───────────────────────────────────────────────
                double balance     = CurrentBalance();
                double riskDollars = balance * (_getRiskPercent() / 100.0);
                double entry       = GetClose(bip);

                double slDist = slPrice > 0 && entry > 0
                    ? Math.Abs(entry - slPrice)
                    : _getDefaultSlTicks() * tickSize;

                double riskPerContract = slDist * pointVal;
                int    riskQty         = riskPerContract > 0
                    ? (int)Math.Floor(riskDollars / riskPerContract)
                    : 1;

                int finalQty = Math.Max(1, Math.Min(requestedQty,
                                   Math.Min(riskQty, _getMaxContracts())));

                Log($"Signal {signalId}: {dir.ToUpper()} {asset} BIP{bip} x{finalQty} " +
                    $"(req={requestedQty} risk={riskQty} cap={_getMaxContracts()})");
                Log($"Risk: balance=${balance:0} riskDollars=${riskDollars:0} " +
                    $"({_getRiskPercent()}%) slDist={slDist:F2}pts pointVal={pointVal}");

                // ── Capture locals for the queued lambda ──────────────────────
                int    cBip       = bip;
                double cTickSize  = tickSize;
                int    cQty       = finalQty;
                string cDir       = dir;
                double cSl        = slPrice;
                double cTp        = tpPrice;
                double cTp2       = tp2Price;
                double cLimit     = limitPrice;
                double cStop      = stopPrice;
                string cId        = signalId;
                string cStrategy  = strategy;
                string cAsset     = asset;
                // action/exitAction/ot are value-types — safe to close over directly

                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;

                        string entryName = $"Signal-{cDir}-{cId}";
                        _strategy.SubmitOrderUnmanaged(cBip, action, ot, cQty, cLimit, cStop, "", entryName);

                        if (_getAutoBrackets())
                        {
                            double bracketEntry = GetClose(cBip);
                            if (bracketEntry <= 0) return;

                            double sl = cSl > 0 ? cSl
                                : cDir == "long"
                                    ? bracketEntry - _getDefaultSlTicks() * cTickSize
                                    : bracketEntry + _getDefaultSlTicks() * cTickSize;

                            double tp = cTp > 0 ? cTp
                                : cDir == "long"
                                    ? bracketEntry + _getDefaultTpTicks() * cTickSize
                                    : bracketEntry - _getDefaultTpTicks() * cTickSize;

                            int slQty  = cQty;
                            int tp1Qty = cTp2 > 0 ? Math.Max(1, cQty / 2) : cQty;
                            int tp2Qty = cTp2 > 0 ? cQty - tp1Qty          : 0;

                            string oco = $"OCO-{cId}";

                            _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.StopMarket,
                                slQty,  0,   sl, oco,  $"SL-{cId}");
                            _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.Limit,
                                tp1Qty, tp,   0, oco, $"TP1-{cId}");

                            if (cTp2 > 0 && tp2Qty > 0)
                                _strategy.SubmitOrderUnmanaged(cBip, exitAction, OrderType.Limit,
                                    tp2Qty, cTp2, 0, "", $"TP2-{cId}");

                            Log($"Brackets {cAsset} BIP{cBip}: SL={sl:F2} TP1={tp:F2}" +
                                (cTp2 > 0 ? $" TP2={cTp2:F2}" : ""));
                        }

                        _sendPositionUpdate?.Invoke();
                        _onSignalExecuted?.Invoke();
                        Log($"✅ Executed {cDir.ToUpper()} {cAsset} BIP{cBip} x{cQty} [{cStrategy}] id={cId}");
                    });
                }

                response["status"]             = "queued";
                response["signal_id"]          = signalId;
                response["direction"]          = dir;
                response["quantity"]           = finalQty;
                response["requested_quantity"] = requestedQty;
                response["risk_sized_quantity"]= riskQty;
                response["strategy"]           = strategy;
                response["asset"]              = asset;
                response["bip"]                = bip;
            }
            catch (Exception ex)
            {
                Log($"ProcessSignal error: {ex.Message}");
                response["status"] = "error";
                response["error"]  = ex.Message;
            }

            return response;
        }

        // =====================================================================
        // ExecuteEntryDirect — backtest path (called on main NT thread)
        // =====================================================================

        /// <summary>
        /// Submit an entry + brackets immediately without queuing.
        /// Only valid when called from OnBarUpdate during Historical state.
        /// </summary>
        internal void ExecuteEntryDirect(SignalBus.Signal sig)
        {
            _onSignalReceived?.Invoke();

            string dir         = (sig.Direction  ?? "long").ToLower();
            int    requestedQty= sig.Quantity > 0 ? sig.Quantity : 1;
            string typeStr     = (sig.OrderType  ?? "market").ToLower();
            double limitPrice  = sig.LimitPrice;
            double slPrice     = sig.StopLoss;
            double tpPrice     = sig.TakeProfit;
            double tp2Price    = sig.TakeProfit2;
            string signalId    = sig.SignalId ?? NewId();

            int    bip       = ResolveBip(sig.Asset);
            double tickSize  = GetTickSize(bip);
            double pointVal  = GetPointValue(bip);

            OrderAction action     = dir == "long" ? OrderAction.Buy  : OrderAction.SellShort;
            OrderAction exitAction = dir == "long" ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderType   ot         = OrderType.Market;
            double      stopPrice  = 0;

            if      (typeStr == "limit") { ot = OrderType.Limit; }
            else if (typeStr == "stop")  { ot = OrderType.StopMarket; stopPrice = limitPrice; limitPrice = 0; }

            // ── Risk sizing ───────────────────────────────────────────────────
            // Cache GetClose once so risk sizing and bracket placement both use
            // the same price snapshot — avoids a theoretical race in live mode
            // and keeps log output consistent.
            double balance     = CurrentBalance();
            double riskDollars = balance * (_getRiskPercent() / 100.0);
            double entry       = GetClose(bip);

            double slDist = slPrice > 0 && entry > 0
                ? Math.Abs(entry - slPrice)
                : _getDefaultSlTicks() * tickSize;

            double riskPerContract = slDist * pointVal;
            int    riskQty         = riskPerContract > 0
                ? (int)Math.Floor(riskDollars / riskPerContract)
                : 1;

            int finalQty = Math.Max(1, Math.Min(requestedQty,
                               Math.Min(riskQty, _getMaxContracts())));

            _onSignalExecuted?.Invoke();
            Log($"BT {dir.ToUpper()} {sig.Asset} BIP{bip} x{finalQty} " +
                $"(req={requestedQty} risk={riskQty} cap={_getMaxContracts()}) id={signalId}");

            _strategy.SubmitOrderUnmanaged(bip, action, ot, finalQty,
                limitPrice, stopPrice, "", $"Signal-{dir}-{signalId}");

            if (!_getAutoBrackets()) return;

            // Reuse the already-fetched entry price for bracket placement.
            // A second GetClose() call here would risk returning a different
            // value if the market moved between the two calls in live mode.
            double bracketEntry = entry;
            if (bracketEntry <= 0) return;

            double sl = slPrice > 0 ? slPrice
                : dir == "long"
                    ? bracketEntry - _getDefaultSlTicks() * tickSize
                    : bracketEntry + _getDefaultSlTicks() * tickSize;

            double tp = tpPrice > 0 ? tpPrice
                : dir == "long"
                    ? bracketEntry + _getDefaultTpTicks() * tickSize
                    : bracketEntry - _getDefaultTpTicks() * tickSize;

            int slQty  = finalQty;
            int tp1Qty = tp2Price > 0 ? Math.Max(1, finalQty / 2) : finalQty;
            int tp2Qty = tp2Price > 0 ? finalQty - tp1Qty          : 0;

            string oco = $"OCO-{signalId}";

            _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.StopMarket,
                slQty,  0,       sl,  oco,  $"SL-{signalId}");
            _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.Limit,
                tp1Qty, tp,       0,  oco, $"TP1-{signalId}");

            if (tp2Price > 0 && tp2Qty > 0)
                _strategy.SubmitOrderUnmanaged(bip, exitAction, OrderType.Limit,
                    tp2Qty, tp2Price, 0, "", $"TP2-{signalId}");

            Log($"BT Brackets {sig.Asset} BIP{bip}: SL={sl:F2} TP1={tp:F2}" +
                (tp2Price > 0 ? $" TP2={tp2Price:F2}" : ""));
        }

        // =====================================================================
        // ExecuteFlattenDirect — backtest flatten path
        // =====================================================================

        /// <summary>
        /// Close all positions immediately (no queue).
        /// Uses Strategy.Position for the primary instrument in backtest;
        /// falls back to account positions for live/sim.
        /// </summary>
        internal void ExecuteFlattenDirect(string reason)
        {
            try
            {
                // Primary instrument — works in Strategy Analyzer backtest
                var pos = _strategy.Position;
                if (pos != null && pos.MarketPosition != MarketPosition.Flat)
                {
                    OrderAction closeAction = pos.MarketPosition == MarketPosition.Long
                        ? OrderAction.Sell : OrderAction.BuyToCover;

                    _strategy.SubmitOrderUnmanaged(0, closeAction, OrderType.Market,
                        pos.Quantity, 0, 0, "", $"Flatten-{reason}");
                    Log($"BT Flattening {pos.MarketPosition} x{pos.Quantity} reason={reason}");
                }
                else
                {
                    // Fallback: account positions (live/sim)
                    var acct = _getMyAccount();
                    if (acct?.Positions != null)
                    {
                        foreach (Position p in acct.Positions)
                        {
                            if (p == null || p.Quantity == 0 || p.Instrument == null) continue;
                            OrderAction ca = p.MarketPosition == MarketPosition.Long
                                ? OrderAction.Sell : OrderAction.BuyToCover;
                            int posBip = ResolveBip(p.Instrument.MasterInstrument.Name);
                            _strategy.SubmitOrderUnmanaged(posBip, ca, OrderType.Market,
                                p.Quantity, 0, 0, "", $"Flatten-{p.Instrument.FullName}");
                            Log($"BT Flattening {p.Instrument.FullName} BIP{posBip} " +
                                $"{p.MarketPosition} x{p.Quantity} reason={reason}");
                        }
                    }
                }

                _onExitExecuted?.Invoke();
                Log($"🔴 FLATTEN ALL (direct) — reason: {reason}");
            }
            catch (Exception ex) { Log($"ExecuteFlattenDirect error: {ex.Message}"); }
        }

        // =====================================================================
        // FlattenAll — live path (queues a lambda)
        // =====================================================================

        /// <summary>
        /// Enqueue a flatten-all action for execution on the next bar.
        /// Safe to call from the HTTP listener thread.
        /// </summary>
        internal Dictionary<string, object> FlattenAll(string reason)
        {
            var response = new Dictionary<string, object>();
            _onExitExecuted?.Invoke();

            try
            {
                // Count open positions now, before enqueuing, so the response
                // field reflects reality rather than always returning 0.
                int posCount = 0;
                try
                {
                    var acct = _getMyAccount();
                    if (acct?.Positions != null)
                    {
                        foreach (Position p in acct.Positions)
                        {
                            if (p != null && p.Quantity > 0 &&
                                p.MarketPosition != MarketPosition.Flat)
                                posCount++;
                        }
                    }
                }
                catch { }

                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;
                        try
                        {
                            var acct = _getMyAccount();
                            if (acct?.Positions != null)
                            {
                                foreach (Position pos in acct.Positions)
                                {
                                    if (pos == null || pos.Quantity == 0 || pos.Instrument == null)
                                        continue;
                                    OrderAction ca = pos.MarketPosition == MarketPosition.Long
                                        ? OrderAction.Sell : OrderAction.BuyToCover;
                                    int posBip = ResolveBip(pos.Instrument.MasterInstrument.Name);
                                    _strategy.SubmitOrderUnmanaged(posBip, ca, OrderType.Market,
                                        pos.Quantity, 0, 0, "", $"Flatten-{pos.Instrument.FullName}");
                                    Log($"Flattening {pos.Instrument.FullName} BIP{posBip} " +
                                        $"{pos.MarketPosition} x{pos.Quantity}");
                                }
                            }

                            // Cancel working orders
                            if (acct?.Orders != null)
                            {
                                foreach (Order ord in acct.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working ||
                                        ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                        acct.Cancel(new[] { ord });
                                }
                            }
                        }
                        catch (Exception ex) { Log($"FlattenAll queue error: {ex.Message}"); }

                        _sendPositionUpdate?.Invoke();
                        Log($"🔴 FLATTEN ALL — reason: {reason}");
                    });
                }

                response["status"]             = "flatten_queued";
                response["positions_to_close"] = posCount;
                response["reason"]             = reason;
            }
            catch (Exception ex)
            {
                response["status"] = "error";
                response["error"]  = ex.Message;
            }

            return response;
        }

        // =====================================================================
        // CancelAllOrders — cancel working orders without closing positions
        // =====================================================================

        internal Dictionary<string, object> CancelAllOrders()
        {
            var response = new Dictionary<string, object>();
            try
            {
                int count = 0;

                lock (_queueLock)
                {
                    _orderQueue.Enqueue(() =>
                    {
                        if (_strategy.State != State.Realtime) return;
                        try
                        {
                            var acct = _getMyAccount();
                            if (acct?.Orders != null)
                            {
                                foreach (Order ord in acct.Orders)
                                {
                                    if (ord == null) continue;
                                    if (ord.OrderState == NinjaTrader.Cbi.OrderState.Working ||
                                        ord.OrderState == NinjaTrader.Cbi.OrderState.Accepted)
                                        acct.Cancel(new[] { ord });
                                }
                            }
                        }
                        catch (Exception ex) { Log($"CancelAllOrders queue error: {ex.Message}"); }
                        Log("Cancel all working orders executed");
                    });
                }

                try
                {
                    var acct = _getMyAccount();
                    if (acct?.Orders != null)
                        count = 0; // count at enqueue time is approximate
                }
                catch { }

                response["status"]           = "cancel_queued";
                response["orders_to_cancel"] = count;
            }
            catch (Exception ex)
            {
                response["status"] = "error";
                response["error"]  = ex.Message;
            }

            return response;
        }

        // =====================================================================
        // BIP routing helpers
        // =====================================================================

        /// <summary>
        /// Resolve a signal's asset string to a BarsInProgress index.
        /// Handles both root names ("MGC") and full names with contract codes
        /// ("MGC 12-25").  Falls back to BIP 0 if no match is found.
        /// </summary>
        internal int ResolveBip(string asset)
        {
            if (string.IsNullOrEmpty(asset)) return 0;

            string upper = asset.Trim().ToUpperInvariant();

            if (_symbolToBip.TryGetValue(upper, out int bip))
                return bip;

            // Strip contract month/year suffix ("MGC 12-25" → "MGC")
            int space = upper.IndexOf(' ');
            if (space > 0 && _symbolToBip.TryGetValue(upper.Substring(0, space), out bip))
                return bip;

            Log($"ResolveBip: no mapping for '{asset}', routing to BIP 0");
            return 0;
        }

        // =====================================================================
        // Per-BIP instrument helpers
        // =====================================================================

        internal double GetTickSize(int bip)
        {
            try
            {
                if (bip < _strategy.BarsArray.Length)
                    return _strategy.BarsArray[bip].Instrument.MasterInstrument.TickSize;
            }
            catch { }
            return _strategy.TickSize;
        }

        internal double GetPointValue(int bip)
        {
            try
            {
                if (bip < _strategy.BarsArray.Length)
                {
                    double pv = _strategy.BarsArray[bip].Instrument.MasterInstrument.PointValue;
                    if (pv > 0) return pv;
                }
            }
            catch { }
            return 10; // safe micro default
        }

        internal double GetClose(int bip)
        {
            try
            {
                if (bip < _strategy.BarsArray.Length && _strategy.BarsArray[bip].Count > 0)
                    return _strategy.BarsArray[bip].GetClose(_strategy.BarsArray[bip].Count - 1);
            }
            catch { }

            // Primary series fallback
            try
            {
                if (_strategy.CurrentBar >= 0 && _strategy.Close?.Count > 0)
                    return _strategy.Close[0];
            }
            catch { }

            return 0;
        }

        // =====================================================================
        // Private helpers
        // =====================================================================

        private double CurrentBalance()
        {
            try
            {
                var acct = _getMyAccount();
                if (acct != null)
                    return acct.Get(AccountItem.CashValue, Currency.UsDollar);
            }
            catch { }
            return _getAccountSize();
        }

        private void Log(string message)
            => _strategy.Print($"[{_tag}] {message}");

        private static string NewId()
            => Guid.NewGuid().ToString("N").Substring(0, 8);

        // ── Signal JSON parsing helpers ───────────────────────────────────────

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
    }
}
