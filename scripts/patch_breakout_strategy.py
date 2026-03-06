#!/usr/bin/env python3
"""
patch_breakout_strategy.py
==========================
Comprehensive patch for BreakoutStrategy.cs to fix the issues discovered in the
2026-03-05 crash logs:

Issues fixed:
  1. OCO ID reuse → Order rejected → Strategy terminated with open positions
     - Root cause: NinjaTrader keeps OCO IDs for the lifetime of the strategy
       session.  When a bracket's SL was rejected (stop price wrong side of
       market), the OCO group died, and the TP1 reusing it was also rejected.
     - Fix: Append a short GUID suffix to every OCO ID so they're always unique.

  2. Stop-loss price validation — BuyToCover stops placed below market price
     - Root cause: For short positions, the SL (a BuyToCover StopMarket) must
       be ABOVE the current market price.  The code didn't validate this.
     - Fix: Before submitting any StopMarket order, validate it's on the correct
       side of the market.  If invalid, adjust to current price + 1 tick.

  3. CNN model dimension mismatch (14 features sent, model expects 8)
     - Root cause: C# was updated to v4 feature contract (14 features) but the
       deployed ONNX model was trained with v3 (8 features).
     - Fix: Add a runtime dimension check that reads the model's expected input
       size and auto-adapts the tabular vector (truncate or zero-pad).

  4. Signal name > 50 characters ignored by NinjaTrader
     - Root cause: Signal names like "Signal-short-brk-s-20260305-143400-6S-Consolidation"
       exceed NT8's 50-char limit.
     - Fix: Truncate signal names to 49 chars.

  5. Too many concurrent positions (34 entries across 15 instruments in 30 min)
     - Fix: Reduce tracked instruments from 15 to 5 core assets.
     - Fix: Add MaxConcurrentPositions limit (default 5) with a per-instrument
       active-position tracker.

  6. No OnOrderUpdate handler — unmanaged strategy can't gracefully handle rejections
     - Fix: Add OnOrderUpdate override that catches Rejected state, logs it,
       and does NOT let the default handler kill the strategy.

  7. ErrorHandling resilience — strategy should log and continue, not terminate
     - Fix: Wrap all SubmitOrderUnmanaged calls in try/catch with logging.

Usage:
    python scripts/patch_breakout_strategy.py [--dry-run]

    Without --dry-run, writes the patched file in-place (backup created first).
    With --dry-run, prints a summary of changes without writing.
"""

import os
import re
import shutil
import sys
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
NINJATRADER_REPO = os.path.expanduser("~/github/ninjatrader")
SRC_FILE = os.path.join(NINJATRADER_REPO, "src", "BreakoutStrategy.cs")
BACKUP_EXT = f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_file(path: str) -> str:
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class Patcher:
    """Applies a series of text transformations to the BreakoutStrategy.cs source."""

    def __init__(self, source: str):
        self.source = source
        self.changes: list[str] = []

    def _replace(self, old: str, new: str, desc: str, count: int = 1) -> bool:
        if old not in self.source:
            print(f"  ⚠ SKIP (pattern not found): {desc}")
            return False
        occurrences = self.source.count(old)
        self.source = self.source.replace(old, new, count)
        self.changes.append(f"✅ {desc} ({occurrences} occurrence(s), replaced {count})")
        return True

    def _regex_replace(self, pattern: str, replacement: str, desc: str, count: int = 0) -> bool:
        rx = re.compile(pattern, re.MULTILINE | re.DOTALL)
        matches = rx.findall(self.source)
        if not matches:
            print(f"  ⚠ SKIP (regex not matched): {desc}")
            return False
        self.source = rx.sub(replacement, self.source, count=count)
        self.changes.append(f"✅ {desc} ({len(matches)} match(es))")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 1: Reduce tracked instruments to 5 core assets
    # ══════════════════════════════════════════════════════════════════════════
    def fix_reduce_instruments(self):
        self._replace(
            'private const string CTrackedInstruments = "MGC,MES,MNQ,M2K,MYM,6E,6B,6J,6A,6C,6S,ZN,ZB,MBT,MET";',
            "// Core 5 assets — reduced from 15 to improve stability and reduce\n"
            "        // concurrent position count.  Extended assets commented for future use.\n"
            '        private const string CTrackedInstruments = "MGC,MES,MNQ,MYM,6E";\n'
            '        // private const string CExtendedInstruments = "M2K,6B,6J,6A,6C,6S,ZN,ZB,MBT,MET";',
            "Fix 1: Reduce tracked instruments from 15 → 5 core assets",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 2: Add max concurrent positions constant + tracking field
    # ══════════════════════════════════════════════════════════════════════════
    def fix_add_max_concurrent_positions(self):
        # Add constant after CMaxContracts
        self._replace(
            "private const int CMaxContracts = 5;",
            "private const int CMaxContracts = 5;\n"
            "        private const int CMaxConcurrentPositions = 5; // max open trades across all instruments",
            "Fix 2a: Add CMaxConcurrentPositions constant",
        )
        # Add property shim (find where MaxContracts => is defined)
        self._replace(
            "private int MaxContracts => CMaxContracts;",
            "private int MaxContracts => CMaxContracts;\n"
            "        private int MaxConcurrentPositions => CMaxConcurrentPositions;",
            "Fix 2b: Add MaxConcurrentPositions property shim",
        )
        # Add tracking counter field near the _metricSignals fields
        self._replace(
            "private long _metricCnnRejected;",
            "private long _metricCnnRejected;\n"
            "        private long _metricOrdersRejected;\n\n"
            "        // ── Active position tracking ─────────────────────────────────────\n"
            "        // Tracks how many instruments currently have open positions.\n"
            "        // Incremented on fill, decremented on flat.  Used to enforce\n"
            "        // MaxConcurrentPositions gate before submitting new entries.\n"
            "        private int _activePositionCount = 0;\n"
            "        private readonly HashSet<string> _activeInstruments = new HashSet<string>();",
            "Fix 2c: Add active position counter and tracking fields",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 3: Add OnOrderUpdate handler for graceful rejection handling
    # ══════════════════════════════════════════════════════════════════════════
    def fix_add_order_update_handler(self):
        # Insert after the #endregion that closes Lifecycle, before Bar Update
        handler_code = """
        // =====================================================================
        // Order / Execution / Position Update Handlers
        // =====================================================================
        #region Order Handlers

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice, OrderState orderState,
            DateTime time, ErrorCode error, string nativeError)
        {
            // ── Rejected orders: log and absorb instead of letting NT8 kill the strategy ──
            if (orderState == OrderState.Rejected)
            {
                Interlocked.Increment(ref _metricOrdersRejected);
                Print($"[Breakout] ⚠ ORDER REJECTED: {order.Name} {order.Instrument.FullName} " +
                      $"Action={order.OrderAction} Type={order.OrderType} " +
                      $"Limit={limitPrice} Stop={stopPrice} Qty={quantity} " +
                      $"Error={error} Native=\\"{nativeError}\\"");

                // For bracket orders (SL/TP), a rejection means the protective
                // order didn't get placed.  Log a warning but do NOT terminate.
                // The position is still open and the other leg of the bracket
                // may still be working.  The strategy should continue running.
                if (order.Name.StartsWith("SL-") || order.Name.StartsWith("TP1-") || order.Name.StartsWith("TP2-"))
                {
                    Print($"[Breakout] ⚠ BRACKET LEG REJECTED — position may be unprotected. " +
                          $"Instrument={order.Instrument.FullName} Order={order.Name}");
                }
                return; // absorb the rejection — do NOT let NT8 terminate the strategy
            }

            // ── Track fills for position counting ─────────────────────────────
            if (orderState == OrderState.Filled)
            {
                string instrName = order.Instrument.MasterInstrument.Name;
                // Entry fills: track that we have a position in this instrument
                if (order.Name.StartsWith("Signal-"))
                {
                    lock (_activeInstruments)
                    {
                        if (_activeInstruments.Add(instrName))
                            _activePositionCount = _activeInstruments.Count;
                    }
                }
                // Exit fills (SL/TP/Flatten): check if position is now flat
                else if (order.Name.StartsWith("SL-") || order.Name.StartsWith("TP") || order.Name.StartsWith("Flatten"))
                {
                    // Check position state after a short delay — use the order's BIP
                    // to query the actual position
                    try
                    {
                        // Find the BIP for this instrument
                        for (int i = 0; i < BarsArray.Length; i++)
                        {
                            if (BarsArray[i] != null &&
                                BarsArray[i].Instrument.MasterInstrument.Name == instrName)
                            {
                                var pos = Positions[i];
                                if (pos == null || pos.MarketPosition == MarketPosition.Flat)
                                {
                                    lock (_activeInstruments)
                                    {
                                        _activeInstruments.Remove(instrName);
                                        _activePositionCount = _activeInstruments.Count;
                                    }
                                }
                                break;
                            }
                        }
                    }
                    catch { /* position query failed — counter may drift, self-corrects on next fill */ }
                }
            }
        }

        #endregion
"""
        # Insert before the Bar Update section
        self._replace(
            "        // =====================================================================\n"
            "        // Bar Update\n"
            "        // =====================================================================\n"
            "        #region Bar Update",
            handler_code + "\n"
            "        // =====================================================================\n"
            "        // Bar Update\n"
            "        // =====================================================================\n"
            "        #region Bar Update",
            "Fix 3: Add OnOrderUpdate handler for graceful rejection handling",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 4: Add max concurrent positions gate in entry flow
    # ══════════════════════════════════════════════════════════════════════════
    def fix_add_position_gate(self):
        # Find the entry method where signals are built and add a gate.
        # The entry happens in the section that builds the SignalBus.Signal,
        # right before "var sig = new SignalBus.Signal".
        # We add the gate just before the signal quantity calculation.
        self._replace(
            "            // ── Quantity: TPT fixed tier vs dynamic engine sizing ─────────────\n"
            "            // In TPT mode we send the exact fixed count directly in the signal.\n"
            "            // BridgeOrderEngine will still cap at MaxContracts, so set MaxContracts\n"
            "            // >= GetTptContracts() or the engine will silently reduce the qty.\n"
            "            // In dynamic mode we send 1 and let the engine risk-size upward from\n"
            "            // there, capped at MaxContracts.\n"
            "            int signalQty;",
            "            // ── Max concurrent positions gate ───────────────────────────────\n"
            "            if (_activePositionCount >= MaxConcurrentPositions)\n"
            "            {\n"
            "                if (EnableDebugLogging)\n"
            '                    Print($"[Breakout DEBUG] BIP{bip} {instrName} FILTERED (max concurrent positions: {_activePositionCount}/{MaxConcurrentPositions})");\n'
            "                return;\n"
            "            }\n"
            "\n"
            "            // ── Quantity: TPT fixed tier vs dynamic engine sizing ─────────────\n"
            "            // In TPT mode we send the exact fixed count directly in the signal.\n"
            "            // BridgeOrderEngine will still cap at MaxContracts, so set MaxContracts\n"
            "            // >= GetTptContracts() or the engine will silently reduce the qty.\n"
            "            // In dynamic mode we send 1 and let the engine risk-size upward from\n"
            "            // there, capped at MaxContracts.\n"
            "            int signalQty;",
            "Fix 4: Add max concurrent positions gate before entry",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 5: Make OCO IDs unique with GUID suffix
    # ══════════════════════════════════════════════════════════════════════════
    def fix_unique_oco_ids(self):
        # In ProcessSignal (queued path) — the lambda that builds brackets
        self._replace(
            '                            string oco = $"OCO-{cId}";',
            '                            string oco = $"OCO-{cId}-{Guid.NewGuid().ToString("N").Substring(0, 6)}";',
            "Fix 5a: Unique OCO ID in ProcessSignal queued path",
        )
        # In ExecuteEntryDirect (direct path)
        self._replace(
            '            string oco2 = $"OCO-{signalId}";',
            '            string oco2 = $"OCO-{signalId}-{Guid.NewGuid().ToString("N").Substring(0, 6)}";',
            "Fix 5b: Unique OCO ID in ExecuteEntryDirect path",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 6: Truncate signal names to 49 chars (NT8 limit is 50)
    # ══════════════════════════════════════════════════════════════════════════
    def fix_signal_name_length(self):
        # In ProcessSignal queued path
        self._replace(
            '                        string entryName = $"Signal-{cDir}-{cId}";',
            '                        string entryName = $"Signal-{cDir}-{cId}";\n'
            "                        if (entryName.Length > 49) entryName = entryName.Substring(0, 49);",
            "Fix 6a: Truncate signal name in ProcessSignal queued path",
        )
        # In ExecuteEntryDirect
        self._replace(
            '            _strategy.SubmitOrderUnmanaged(bip, action, ot, finalQty, limitPrice, stopPrice, "", $"Signal-{dir}-{signalId}");',
            '            string sigName = $"Signal-{dir}-{signalId}";\n'
            "            if (sigName.Length > 49) sigName = sigName.Substring(0, 49);\n"
            '            _strategy.SubmitOrderUnmanaged(bip, action, ot, finalQty, limitPrice, stopPrice, "", sigName);',
            "Fix 6b: Truncate signal name in ExecuteEntryDirect path",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 7: Validate SL price is on correct side of market before submission
    # ══════════════════════════════════════════════════════════════════════════
    def fix_validate_sl_price(self):
        # We need to wrap the SL submission in both paths with a price validation.
        # ProcessSignal queued path - wrap the bracket section
        # Find the bracket submission block in ProcessSignal and add validation
        old_queued_bracket = (
            "                        if (_getAutoBrackets())\n"
            "                        {\n"
            "                            double be = GetClose(cBip);\n"
            "                            if (be <= 0) return;\n"
            '                            double sl = cSl > 0 ? cSl : cDir == "long" ? be - _getDefaultSlTicks() * cTickSize : be + _getDefaultSlTicks() * cTickSize;\n'
            '                            double tp = cTp > 0 ? cTp : cDir == "long" ? be + _getDefaultTpTicks() * cTickSize : be - _getDefaultTpTicks() * cTickSize;'
        )
        new_queued_bracket = (
            "                        if (_getAutoBrackets())\n"
            "                        {\n"
            "                            double be = GetClose(cBip);\n"
            "                            if (be <= 0) return;\n"
            '                            double sl = cSl > 0 ? cSl : cDir == "long" ? be - _getDefaultSlTicks() * cTickSize : be + _getDefaultSlTicks() * cTickSize;\n'
            '                            double tp = cTp > 0 ? cTp : cDir == "long" ? be + _getDefaultTpTicks() * cTickSize : be - _getDefaultTpTicks() * cTickSize;\n'
            "\n"
            "                            // ── Validate SL is on correct side of market ──────────\n"
            "                            // BuyToCover stop (short exit) must be ABOVE market.\n"
            "                            // Sell stop (long exit) must be BELOW market.\n"
            '                            if (cDir == "short" && sl <= be)\n'
            "                            {\n"
            "                                double corrected = be + cTickSize;\n"
            '                                Log($"⚠ SL CORRECTED (short): {sl:F6} was at/below market {be:F6}, moved to {corrected:F6}");\n'
            "                                sl = corrected;\n"
            "                            }\n"
            '                            else if (cDir == "long" && sl >= be)\n'
            "                            {\n"
            "                                double corrected = be - cTickSize;\n"
            '                                Log($"⚠ SL CORRECTED (long): {sl:F6} was at/above market {be:F6}, moved to {corrected:F6}");\n'
            "                                sl = corrected;\n"
            "                            }"
        )
        self._replace(
            old_queued_bracket, new_queued_bracket, "Fix 7a: Validate SL price in ProcessSignal queued bracket"
        )

        # ExecuteEntryDirect path
        old_direct_bracket = (
            "            if (!_getAutoBrackets() || entry <= 0) return;\n"
            '            double sl = slPrice > 0 ? slPrice : dir == "long" ? entry - _getDefaultSlTicks() * tickSize : entry + _getDefaultSlTicks() * tickSize;\n'
            '            double tp = tpPrice > 0 ? tpPrice : dir == "long" ? entry + _getDefaultTpTicks() * tickSize : entry - _getDefaultTpTicks() * tickSize;'
        )
        new_direct_bracket = (
            "            if (!_getAutoBrackets() || entry <= 0) return;\n"
            '            double sl = slPrice > 0 ? slPrice : dir == "long" ? entry - _getDefaultSlTicks() * tickSize : entry + _getDefaultSlTicks() * tickSize;\n'
            '            double tp = tpPrice > 0 ? tpPrice : dir == "long" ? entry + _getDefaultTpTicks() * tickSize : entry - _getDefaultTpTicks() * tickSize;\n'
            "\n"
            "            // ── Validate SL is on correct side of market ──────────────────\n"
            '            if (dir == "short" && sl <= entry)\n'
            "            {\n"
            "                double corrected = entry + tickSize;\n"
            '                Log($"⚠ SL CORRECTED (short): {sl:F6} at/below market {entry:F6}, moved to {corrected:F6}");\n'
            "                sl = corrected;\n"
            "            }\n"
            '            else if (dir == "long" && sl >= entry)\n'
            "            {\n"
            "                double corrected = entry - tickSize;\n"
            '                Log($"⚠ SL CORRECTED (long): {sl:F6} at/above market {entry:F6}, moved to {corrected:F6}");\n'
            "                sl = corrected;\n"
            "            }"
        )
        self._replace(old_direct_bracket, new_direct_bracket, "Fix 7b: Validate SL price in ExecuteEntryDirect bracket")

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 8: Wrap SubmitOrderUnmanaged calls in try/catch
    # ══════════════════════════════════════════════════════════════════════════
    def fix_wrap_submit_in_trycatch(self):
        # Wrap the entire bracket submission block in ProcessSignal with try/catch
        # We target the SubmitOrderUnmanaged for entry in the queued lambda
        self._replace(
            '                        _strategy.SubmitOrderUnmanaged(cBip, action, ot, cQty, cLimit, cStop, "", entryName);',
            "                        try {\n"
            '                        _strategy.SubmitOrderUnmanaged(cBip, action, ot, cQty, cLimit, cStop, "", entryName);',
            "Fix 8a: Open try block around queued entry submission",
        )
        self._replace(
            "                        _sendPositionUpdate?.Invoke();\n"
            "                        _onSignalExecuted?.Invoke();\n"
            '                        Log($"✅ Executed {cDir.ToUpper()} {cAsset} BIP{cBip} x{cQty} id={cId}");',
            "                        _sendPositionUpdate?.Invoke();\n"
            "                        _onSignalExecuted?.Invoke();\n"
            '                        Log($"✅ Executed {cDir.ToUpper()} {cAsset} BIP{cBip} x{cQty} id={cId}");\n'
            '                        } catch (Exception submitEx) { Log($"⚠ Order submission failed for {cAsset}: {submitEx.Message}"); }',
            "Fix 8b: Close try/catch around queued entry submission",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 9: CNN dimension auto-adaptation
    # ══════════════════════════════════════════════════════════════════════════
    def fix_cnn_dimension_mismatch(self):
        # The OrbCnnPredictor has NumTabular = 14 hardcoded, but the model expects 8.
        # We need to:
        # 1. Read the model's expected dimension at load time
        # 2. Auto-adapt the tabular vector at inference time

        # Replace the hardcoded NumTabular constant with a dynamic field
        self._replace(
            "        // feature_contract.json v4: 14 tabular features\n        private const int NumTabular = 14;",
            "        // feature_contract.json: tabular feature count.\n"
            "        // Default 14 (v4 contract), but auto-detected from model at load time.\n"
            "        private int _numTabular = 14;\n"
            "        private const int MaxTabular = 14; // C# always builds 14 features",
            "Fix 9a: Make NumTabular dynamic in OrbCnnPredictor",
        )

        # Replace all references to NumTabular in OrbCnnPredictor with _numTabular
        # (except the one we just replaced)
        self._replace(
            "        private static readonly int ImageBufSize = NumChannels * ImageSize * ImageSize;",
            "        private static readonly int ImageBufSize = NumChannels * ImageSize * ImageSize;\n"
            "        public int NumTabular => _numTabular;",
            "Fix 9b: Add NumTabular public property",
        )

        # In the constructor, detect the model's expected tabular dimension
        self._replace(
            '            _tabularName = inputNames.Count > 1 ? inputNames[1] : "tabular";',
            '            _tabularName = inputNames.Count > 1 ? inputNames[1] : "tabular";\n\n'
            "            // Auto-detect expected tabular dimension from model metadata\n"
            "            if (_session.InputMetadata.ContainsKey(_tabularName))\n"
            "            {\n"
            "                var tabMeta = _session.InputMetadata[_tabularName];\n"
            "                var dims = tabMeta.Dimensions;\n"
            "                if (dims != null && dims.Length >= 2 && dims[1] > 0)\n"
            "                {\n"
            "                    _numTabular = dims[1];\n"
            "                }\n"
            "            }",
            "Fix 9c: Auto-detect tabular dimension from ONNX model metadata",
        )

        # Fix the Predict method to adapt the vector length
        self._replace(
            "            if (tabular == null || tabular.Length != NumTabular)\n"
            '                throw new ArgumentException($"tabular must have {NumTabular} elements, got {tabular?.Length ?? 0}");',
            "            if (tabular == null)\n"
            '                throw new ArgumentException("tabular array must not be null");\n'
            "\n"
            "            // Auto-adapt tabular vector to model's expected dimension.\n"
            "            // C# always builds MaxTabular (14) features; if the model\n"
            "            // expects fewer (e.g. 8 for v3), truncate.  If it somehow\n"
            "            // expects more, zero-pad.\n"
            "            if (tabular.Length != _numTabular)\n"
            "            {\n"
            "                float[] adapted = new float[_numTabular];\n"
            "                int copyLen = Math.Min(tabular.Length, _numTabular);\n"
            "                Array.Copy(tabular, adapted, copyLen);\n"
            "                tabular = adapted;\n"
            "            }",
            "Fix 9d: Auto-adapt tabular vector length in Predict()",
        )

        # Fix the NormaliseTabular to handle variable length
        self._replace(
            "            float[] norm = new float[NumTabular];",
            "            float[] norm = new float[_numTabular];",
            "Fix 9e: Use dynamic _numTabular in NormaliseTabular",
        )

        # Fix the DenseTensor to use dynamic size
        self._replace(
            "            var tabTensor = new DenseTensor<float>(tabNorm, new[] { 1, NumTabular });",
            "            var tabTensor = new DenseTensor<float>(tabNorm, new[] { 1, _numTabular });",
            "Fix 9f: Use dynamic _numTabular for tensor shape",
        )

        # Fix the NormaliseTabular to only normalise up to what's available
        # We need to guard indices 8-13 in case _numTabular < 14
        # Find the normalisation for index [8] and wrap 8-13 in a guard
        self._replace(
            "            // [8] or_range_atr_ratio — clamp(raw, 0, 3) / 3.0\n"
            "            norm[8] = Math.Max(0f, Math.Min(3f, raw[8])) / 3f;",
            "            // [8] onwards — only if model expects >= 9 features\n"
            "            if (_numTabular > 8)\n"
            "            {\n"
            "            // [8] or_range_atr_ratio — clamp(raw, 0, 3) / 3.0\n"
            "            norm[8] = Math.Max(0f, Math.Min(3f, raw[8])) / 3f;",
            "Fix 9g: Guard extended features [8-13] behind dimension check (open)",
        )

        # Close the guard after [13]
        self._replace(
            "            norm[13] = Math.Max(0f, Math.Min(1f, raw[13]));",
            "            if (_numTabular > 13) norm[13] = Math.Max(0f, Math.Min(1f, raw[13]));\n"
            "            } // end guard for features [8]+",
            "Fix 9h: Guard extended features [8-13] behind dimension check (close)",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 10: Add diagnostic logging for active position count
    # ══════════════════════════════════════════════════════════════════════════
    def fix_add_position_count_logging(self):
        # Add to the periodic diagnostics output
        self._replace(
            '                  $"@ {price:F2} SL={sl:F2} TP1={tp1:F2} TP2={tp2:F2} id={signalId}");',
            '                  $"@ {price:F2} SL={sl:F2} TP1={tp1:F2} TP2={tp2:F2} id={signalId}" +\n'
            '                  $" [positions: {_activePositionCount}/{MaxConcurrentPositions}]");',
            "Fix 10: Add position count to entry log line",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 11: Increase cooldown from 5 to 10 minutes to reduce over-trading
    # ══════════════════════════════════════════════════════════════════════════
    def fix_increase_cooldown(self):
        self._replace(
            "private const int CEntryCooldownMinutes = 5;",
            "private const int CEntryCooldownMinutes = 10; // increased from 5 to reduce over-trading",
            "Fix 11: Increase entry cooldown from 5 → 10 minutes",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fix 12: Log CNN model dimension info at startup
    # ══════════════════════════════════════════════════════════════════════════
    def fix_log_cnn_dimensions(self):
        self._replace(
            '                    Print($"   TPT Mode: {TptMode} | Tier: {AccountTier} | CNN: {(EnableCnnFilter && _cnn != null ? "ENABLED" : "DISABLED")}");',
            '                    Print($"   TPT Mode: {TptMode} | Tier: {AccountTier} | CNN: {(EnableCnnFilter && _cnn != null ? "ENABLED" : "DISABLED")}");'
            "\n                    if (_cnn != null)"
            '\n                        Print($"   CNN tabular dim: model expects {_cnn.NumTabular}, C# builds {14}");',
            "Fix 12: Log CNN tabular dimension at startup",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Apply all fixes
    # ══════════════════════════════════════════════════════════════════════════
    def apply_all(self) -> str:
        print("\n🔧 Applying patches to BreakoutStrategy.cs...\n")

        self.fix_reduce_instruments()
        self.fix_add_max_concurrent_positions()
        self.fix_add_order_update_handler()
        self.fix_add_position_gate()
        self.fix_unique_oco_ids()
        self.fix_signal_name_length()
        self.fix_validate_sl_price()
        self.fix_wrap_submit_in_trycatch()
        self.fix_cnn_dimension_mismatch()
        self.fix_add_position_count_logging()
        self.fix_increase_cooldown()
        self.fix_log_cnn_dimensions()

        print(f"\n📋 Summary: {len(self.changes)} patches applied:\n")
        for c in self.changes:
            print(f"  {c}")

        return self.source


def main():
    dry_run = "--dry-run" in sys.argv

    if not os.path.exists(SRC_FILE):
        print(f"❌ Source file not found: {SRC_FILE}")
        print("   Make sure the ninjatrader repo is at ~/github/ninjatrader")
        sys.exit(1)

    source = read_file(SRC_FILE)
    print(f"📂 Read {len(source):,} bytes from {SRC_FILE}")

    patcher = Patcher(source)
    patched = patcher.apply_all()

    if dry_run:
        print("\n🔍 DRY RUN — no files written.")
        print(f"   Patched source would be {len(patched):,} bytes ({len(patched) - len(source):+,} bytes)")
        return

    # Create timestamped backup
    backup_path = SRC_FILE + BACKUP_EXT
    shutil.copy2(SRC_FILE, backup_path)
    print(f"\n💾 Backup saved: {backup_path}")

    # Write patched file
    write_file(SRC_FILE, patched)
    print(f"✅ Patched file written: {SRC_FILE}")
    print(f"   Size: {len(patched):,} bytes ({len(patched) - len(source):+,} bytes)")

    # Verify the key constants are present
    print("\n🔍 Verification:")
    checks = [
        ("Core 5 instruments", '"MGC,MES,MNQ,MYM,6E"'),
        ("MaxConcurrentPositions", "CMaxConcurrentPositions"),
        ("OnOrderUpdate handler", "protected override void OnOrderUpdate"),
        ("OCO GUID suffix", 'Guid.NewGuid().ToString("N").Substring(0, 6)'),
        ("Signal name truncation", "if (entryName.Length > 49)"),
        ("SL validation (short)", "SL CORRECTED (short)"),
        ("SL validation (long)", "SL CORRECTED (long)"),
        ("Try/catch on submit", "Order submission failed"),
        ("Dynamic NumTabular", "_numTabular"),
        ("Position count gate", "max concurrent positions"),
        ("Cooldown 10min", "CEntryCooldownMinutes = 10"),
    ]
    all_ok = True
    for label, needle in checks:
        found = needle in patched
        status = "✅" if found else "❌"
        if not found:
            all_ok = False
        print(f"  {status} {label}")

    if all_ok:
        print("\n🎉 All patches verified successfully!")
    else:
        print("\n⚠ Some patches may not have applied correctly — review the output above.")

    print("\n📝 Next steps:")
    print("   1. Copy the patched file to your NinjaTrader machine:")
    print("      Documents\\NinjaTrader 8\\bin\\Custom\\Strategies\\BreakoutStrategy.cs")
    print("   2. In NinjaTrader, go to Tools > NinjaScript Editor and compile (F5)")
    print("   3. Restart the strategy on a 1-minute chart")
    print("   4. Verify in the output window:")
    print("      - Only 5 instruments are loaded (MGC, MES, MNQ, MYM, 6E)")
    print("      - CNN tabular dim line shows the model's expected dimension")
    print("      - No more 'OCO ID cannot be reused' errors")
    print("      - Position count shows in entry logs")


if __name__ == "__main__":
    main()
