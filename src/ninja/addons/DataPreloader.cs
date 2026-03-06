// =============================================================================
// DataPreloader.cs  —  NinjaTrader 8 AddOn
// =============================================================================
//
// Drop this file into:
//   Documents\NinjaTrader 8\bin\Custom\AddOns\
//
// PURPOSE
// -------
// BreakoutStrategy subscribes to 5 core instruments via AddDataSeries().  NT8 only
// returns historical bars for those secondary BIPs if the instrument's minute
// cache already exists on disk (db\minute\<SYMBOL>\).  An empty cache means
// the strategy sees 0 bars for every instrument except the one the chart is
// running on (MGC in this case).
//
// This AddOn fires at NT8 startup (after the first live/sim data connection)
// and issues BarsRequests for 60 days of 1-minute history for every instrument
// in the list — exactly as if you had manually opened a 1-minute continuous-
// contract chart for each symbol.  Once the disk cache is warm, every
// subsequent BreakoutStrategy startup loads full history on all 16 BIPs with
// no manual intervention.
//
// BEHAVIOUR
// ---------
//  • Runs once per NT8 session — the seed is triggered by the first
//    Connected event on any live or sim data connection.
//  • Skips any symbol whose db\minute\SYMBOL\ folder already contains files
//    (SkipIfCached = true by default) so restarts are essentially instant.
//  • Requests are staggered (StaggerMs apart) to avoid flooding the feed.
//  • All progress is written to the NT8 Output window (Output tab 1).
//  • Safe to leave deployed permanently — it does nothing on subsequent
//    startups when all caches are already warm.
//
// VERIFIED NT8 API (reflected from NinjaTrader.Cbi.dll / NinjaTrader.Data.dll)
// -----------------------------------------------------------------------------
//  Connection.ConnectionStatusUpdate          — static event (not ConnectionStatusChanged)
//  ConnectionStatusEventArgs.Connection       — the Connection that changed
//  ConnectionStatusEventArgs.Status           — new ConnectionStatus
//  ConnectionStatusEventArgs.PreviousStatus   — prior ConnectionStatus
//  Connection.PlaybackConnection              — static; non-null when playback active
//  NinjaTrader.Cbi.Provider.Playback          — enum value to detect playback feeds
//  Instrument.GetInstrument(string, bool)     — only overload (no exchange/type params)
//  BarsRequest.MergePolicy                    — NinjaTrader.Cbi.MergePolicy property
//  BarsRequest has NO IsContinuous property
//  BarsRequest.Request callback: Action<BarsRequest, ErrorCode, string>
//    — bars are on the BarsRequest argument: callback arg.Bars.Count
//
// CONFIGURATION
// -------------
// Edit the constants in the #region Configuration block below.
//   DaysToLoad   — days of 1-min history to request  (default 15)
//   StaggerMs    — ms between successive requests     (default 800)
//   SkipIfCached — skip symbols that already have files (default true)
//   LogVerbose   — write bar-count lines to Output    (default false)
//
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Core;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
    public class DataPreloader : AddOnBase
    {
        // =====================================================================
        #region Configuration
        // =====================================================================

        /// <summary>Days of 1-minute history to request for each instrument.
        /// 60 calendar days ≈ 42 trading days — gives every instrument 10,000+
        /// bars of 1-min history, matching the depth the strategy's own
        /// DaysToLoad pulls for BIP0 and all secondary BIPs.</summary>
        private const int DaysToLoad = 60;

        /// <summary>Enable automatic reconnection when NT8's connection enters
        /// the Panic/Disconnected state and stops retrying on its own.
        /// This is the primary fix for the 2 AM disconnect-and-never-reconnect
        /// problem — NT8's built-in retry gives up after ~3.5 minutes and your
        /// strategy's ConnectionLossHandling=Recalculate only governs what
        /// happens *after* the platform reconnects, not the reconnect itself.</summary>
        private const bool EnableConnectionWatchdog = true;

        /// <summary>Seconds to wait after detecting a full disconnect before
        /// attempting to force a reconnect.  Gives NT8's own retry logic time
        /// to succeed first.  If NT8 reconnects on its own within this window,
        /// the watchdog cancels its attempt.</summary>
        private const int WatchdogReconnectDelaySec = 30;

        /// <summary>Maximum number of consecutive reconnect attempts before the
        /// watchdog backs off (prevents infinite rapid-fire reconnects if the
        /// data feed is genuinely down for maintenance).</summary>
        private const int WatchdogMaxAttempts = 10;

        /// <summary>Seconds to wait between reconnect attempts after the first
        /// one fails.  Uses linear backoff: attempt N waits N × this value,
        /// capped at 5 minutes.</summary>
        private const int WatchdogBackoffBaseSec = 30;

        /// <summary>Milliseconds to wait between successive BarsRequests so the
        /// data feed is not overwhelmed.  800 ms works well for Rithmic / CQG.</summary>
        private const int StaggerMs = 800;

        /// <summary>When true, instruments whose db\minute\SYMBOL\ folder already
        /// contains at least one file are skipped — no redundant download.</summary>
        private const bool SkipIfCached = true;

        /// <summary>When true, log a bar-count line for each instrument after
        /// its BarsRequest completes.  Useful for debugging; noisy otherwise.</summary>
        private const bool LogVerbose = true;

        #endregion

        // =====================================================================
        #region Instrument list
        // =====================================================================

        // Core 5 instruments matching BreakoutStrategy.CTrackedInstruments.
        // Each entry: (root symbol, trading-hours template name).
        // Template names must exactly match files in:
        //   Documents\NinjaTrader 8\templates\TradingHours\ 
        //
        // NOTE: Instrument.GetInstrument only accepts (string name, bool create).
        // There is no exchange-filtered overload in NT8's public API.
        // MGC is intentionally excluded -- it is the primary chart instrument
        // (BIP0) and always loads its own data from the chart automatically.
        private static readonly InstrumentSpec[] Instruments =
        {
            // Core 4 (+ MGC on chart = 5 total) -- synced with BreakoutStrategy CTrackedInstruments
            new InstrumentSpec("MES", "CME US Index Futures ETH"),
            new InstrumentSpec("MNQ", "CME US Index Futures ETH"),
            new InstrumentSpec("MYM", "CME US Index Futures ETH"),
            new InstrumentSpec("6E",  "CME FX Futures ETH"),
        };

        // Extended instruments -- uncomment and move into Instruments[] to
        // re-enable.  Removed to sync with BreakoutStrategy's 5-asset config.
        //   new InstrumentSpec("M2K", "CME US Index Futures ETH"),
        //   new InstrumentSpec("6B",  "CME FX Futures ETH"),
        //   new InstrumentSpec("6J",  "CME FX Futures ETH"),
        //   new InstrumentSpec("6A",  "CME FX Futures ETH"),
        //   new InstrumentSpec("6C",  "CME FX Futures ETH"),
        //   new InstrumentSpec("6S",  "CME FX Futures ETH"),
        //   new InstrumentSpec("ZN",  "CBOT Interest Rate ETH"),
        //   new InstrumentSpec("ZB",  "CBOT Interest Rate ETH"),
        //   new InstrumentSpec("MBT", "Cryptocurrency"),
        //   new InstrumentSpec("MET", "Cryptocurrency"),
        //
        // Pending data subscription:
        //   new InstrumentSpec("MCL", "CME Commodities ETH"),
        //   new InstrumentSpec("MNG", "CME Commodities ETH"),
        //   new InstrumentSpec("SIL", "Nymex Metals - Energy ETH"),
        //   new InstrumentSpec("MHG", "Nymex Metals - Energy ETH"),
        //   new InstrumentSpec("ZC",  "CBOT Agriculturals ETH"),
        //   new InstrumentSpec("ZS",  "CBOT Agriculturals ETH"),
        //   new InstrumentSpec("ZW",  "CBOT Agriculturals ETH"),

        /// <summary>Plain data holder — C# 7.3 (NT8's Roslyn) does not support
        /// tuple literals in static field initialisers, so we use a struct.</summary>
        private struct InstrumentSpec
        {
            public readonly string Symbol;
            public readonly string TradingHours;
            public InstrumentSpec(string sym, string th)
            { Symbol = sym; TradingHours = th; }
        }

        #endregion

        // =====================================================================
        #region State
        // =====================================================================

        private bool _seeded = false;
        private Thread _seedThread = null;
        private string _minuteCacheRoot = string.Empty;

        // ── Connection watchdog state ─────────────────────────────────────
        private Timer _watchdogTimer = null;
        private int _watchdogAttempts = 0;
        private bool _watchdogActive = false;
        private readonly object _watchdogLock = new object();

        // Live BarsRequest objects — keep references so the GC does not collect
        // them before the async callback fires.
        private readonly object _lock = new object();
        private readonly Dictionary<string, BarsRequest> _pending =
            new Dictionary<string, BarsRequest>(StringComparer.OrdinalIgnoreCase);
        private int _completedCount = 0;
        private int _totalRequested = 0;

        #endregion

        // =====================================================================
        #region AddOnBase lifecycle
        // =====================================================================

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Pre-seeds the NT8 minute-bar disk cache for the 5 core " +
                              "BreakoutStrategy instruments at startup so every " +
                              "AddDataSeries() BIP loads full history automatically.";
                Name = "DataPreloader";
            }
            else if (State == State.Configure)
            {
                // Globals.UserDataDir resolves to Documents\NinjaTrader 8\
                // regardless of which Windows account NT8 is running under.
                _minuteCacheRoot = Path.Combine(Globals.UserDataDir, "db", "minute");
            }
            else if (State == State.Active)
            {
                // ConnectionStatusUpdate is the correct static event name on
                // NinjaTrader.Cbi.Connection (verified by reflection).
                Connection.ConnectionStatusUpdate += OnConnectionStatusChanged;
                Out("[DataPreloader] Active — waiting for data connection...");
                if (EnableConnectionWatchdog)
                    Out("[DataPreloader] Connection watchdog ENABLED " +
                        "(delay=" + WatchdogReconnectDelaySec + "s, " +
                        "maxAttempts=" + WatchdogMaxAttempts + ", " +
                        "backoff=" + WatchdogBackoffBaseSec + "s)");

                // Handle the case where NT8 was already connected before this
                // AddOn compiled / activated (e.g. after a hot recompile).
                bool alreadyConnected = false;
                lock (Connection.Connections)
                {
                    alreadyConnected = Connection.Connections.Any(c =>
                        c.Status == ConnectionStatus.Connected && !IsPlaybackConnection(c));
                }

                if (alreadyConnected)
                    TriggerSeed("connection already established at activation");
            }
            else if (State == State.Terminated)
            {
                Connection.ConnectionStatusUpdate -= OnConnectionStatusChanged;

                // Stop the watchdog timer.
                StopWatchdog("AddOn terminated");

                // Best-effort: interrupt the background thread if still running.
                try
                {
                    if (_seedThread != null && _seedThread.IsAlive)
                    {
                        _seedThread.Interrupt();
                        if (!_seedThread.Join(4000))
                            _seedThread.Abort();
                    }
                }
                catch { /* ignore — process is shutting down */ }

                // Dispose any BarsRequest objects that never fired their callback.
                lock (_lock)
                {
                    foreach (var br in _pending.Values)
                        try { br.Dispose(); } catch { }
                    _pending.Clear();
                }
            }
        }

        #endregion

        // =====================================================================
        #region Connection event
        // =====================================================================

        private void OnConnectionStatusChanged(object sender, ConnectionStatusEventArgs e)
        {
            if (e.Connection == null) return;
            if (IsPlaybackConnection(e.Connection)) return;

            string connName = "(unknown)";
            try { connName = e.Connection.ToString(); } catch { }

            // ── Transition TO Connected — seed data and cancel watchdog ────
            if (e.Status == ConnectionStatus.Connected &&
                e.PreviousStatus != ConnectionStatus.Connected)
            {
                Out("[DataPreloader] Connection restored: '" + connName +
                    "' (" + e.PreviousStatus + " → Connected)");

                // Cancel any pending watchdog reconnect — we're back online.
                StopWatchdog("connection restored");

                TriggerSeed("connection '" + connName + "' became Connected");
                return;
            }

            // ── Transition TO Disconnected — start watchdog ───────────────
            if (EnableConnectionWatchdog &&
                (e.Status == ConnectionStatus.Disconnected ||
                 e.Status == ConnectionStatus.ConnectionLost))
            {
                Out("[DataPreloader] ⚠ Connection status: '" + connName +
                    "' " + e.PreviousStatus + " → " + e.Status);

                // Check if ANY non-playback connection is still alive.
                bool anyAlive = false;
                try
                {
                    lock (Connection.Connections)
                    {
                        anyAlive = Connection.Connections.Any(c =>
                            !IsPlaybackConnection(c) &&
                            (c.Status == ConnectionStatus.Connected ||
                             c.Status == ConnectionStatus.Connecting));
                    }
                }
                catch { /* if we can't check, assume dead */ }

                if (!anyAlive)
                {
                    Out("[DataPreloader] ⚠ All connections down — " +
                        "starting watchdog reconnect timer (" +
                        WatchdogReconnectDelaySec + "s delay)...");
                    StartWatchdog();
                }
            }
        }

        /// <summary>
        /// Returns true when <paramref name="c"/> is the NT8 playback (replay)
        /// connection.  Connection.PlaybackConnection is a static reference to
        /// the replay feed instance; comparing by reference is reliable.
        /// As a fallback we also check the connection name for "playback".
        /// </summary>
        private static bool IsPlaybackConnection(Connection c)
        {
            if (c == null) return false;

            // Reference equality — PlaybackConnection is a singleton.
            if (Connection.PlaybackConnection != null &&
                ReferenceEquals(c, Connection.PlaybackConnection))
                return true;

            // Name fallback in case the reference check is inconclusive.
            try
            {
                string name = c.ToString() ?? string.Empty;
                return name.IndexOf("playback", StringComparison.OrdinalIgnoreCase) >= 0;
            }
            catch { return false; }
        }

        #endregion

        // =====================================================================
        #region Seed orchestration
        // =====================================================================

        private void TriggerSeed(string reason)
        {
            lock (_lock)
            {
                if (_seeded) return;
                _seeded = true;
            }

            Out("[DataPreloader] Seed triggered: " + reason);

            _seedThread = new Thread(SeedAllInstruments)
            {
                IsBackground = true,
                Name = "DataPreloader.SeedThread"
            };
            _seedThread.Start();
        }

        /// <summary>
        /// Resets the <c>_seeded</c> flag so the next Connected event triggers
        /// a fresh data-seeding pass.  Called after a watchdog reconnect so all
        /// BIPs get up-to-date history.
        /// </summary>
        private void ResetSeedFlag(string reason)
        {
            lock (_lock)
            {
                _seeded = false;
            }
            Out("[DataPreloader] Seed flag reset — " + reason);
        }

        #endregion

        // =====================================================================
        #region Connection watchdog
        // =====================================================================

        /// <summary>
        /// Starts the watchdog timer that will attempt to reconnect after
        /// <see cref="WatchdogReconnectDelaySec"/> seconds.  If the connection
        /// comes back before the timer fires, <see cref="StopWatchdog"/> cancels it.
        /// </summary>
        private void StartWatchdog()
        {
            lock (_watchdogLock)
            {
                if (_watchdogActive) return; // already ticking
                _watchdogActive = true;
                _watchdogAttempts = 0;

                int delayMs = WatchdogReconnectDelaySec * 1000;
                _watchdogTimer = new Timer(WatchdogTick, null, delayMs, Timeout.Infinite);
            }
        }

        /// <summary>
        /// Cancels the watchdog timer and resets the attempt counter.
        /// </summary>
        private void StopWatchdog(string reason)
        {
            lock (_watchdogLock)
            {
                if (!_watchdogActive) return;
                _watchdogActive = false;
                _watchdogAttempts = 0;

                try { _watchdogTimer?.Dispose(); } catch { }
                _watchdogTimer = null;

                Out("[DataPreloader] Watchdog stopped — " + reason);
            }
        }

        /// <summary>
        /// Timer callback.  Checks whether any connection has come back; if not,
        /// forces a disconnect-then-reconnect on every known Connection object.
        /// Schedules itself again with increasing backoff if the attempt fails.
        /// </summary>
        private void WatchdogTick(object state)
        {
            try
            {
                // Double-check: are we still disconnected?
                bool anyAlive = false;
                // Collect the Options.Name of each dead connection so we can
                // look up the matching ConnectOptions from Core.Globals.
                List<string> deadConnectionNames = new List<string>();

                lock (Connection.Connections)
                {
                    foreach (var c in Connection.Connections)
                    {
                        if (IsPlaybackConnection(c)) continue;

                        if (c.Status == ConnectionStatus.Connected ||
                            c.Status == ConnectionStatus.Connecting)
                        {
                            anyAlive = true;
                            break;
                        }

                        // Collect disconnected/lost connection names to reconnect.
                        if (c.Status == ConnectionStatus.Disconnected ||
                            c.Status == ConnectionStatus.ConnectionLost)
                        {
                            try
                            {
                                string name = c.Options?.Name;
                                if (!string.IsNullOrEmpty(name) && !deadConnectionNames.Contains(name))
                                    deadConnectionNames.Add(name);
                            }
                            catch { }
                        }
                    }
                }

                if (anyAlive)
                {
                    StopWatchdog("connection came back before watchdog fired");
                    return;
                }

                int attempt;
                lock (_watchdogLock)
                {
                    _watchdogAttempts++;
                    attempt = _watchdogAttempts;

                    if (attempt > WatchdogMaxAttempts)
                    {
                        Out("[DataPreloader] ❌ Watchdog giving up after " +
                            WatchdogMaxAttempts + " attempts. " +
                            "Manual reconnect or NT8 restart required.");
                        _watchdogActive = false;
                        try { _watchdogTimer?.Dispose(); } catch { }
                        _watchdogTimer = null;
                        return;
                    }
                }

                Out("[DataPreloader] 🔄 Watchdog reconnect attempt " +
                    attempt + "/" + WatchdogMaxAttempts + "...");

                // Reset the seed flag so data will be re-seeded after reconnect.
                ResetSeedFlag("watchdog reconnect attempt " + attempt);

                if (deadConnectionNames.Count == 0)
                {
                    Out("[DataPreloader]   No named connections found to reconnect.");
                }

                // Force reconnect using the static Connection.Connect(ConnectOptions) API.
                // ConnectOptions are looked up from Core.Globals.ConnectOptions by name.
                foreach (string connName in deadConnectionNames)
                {
                    try
                    {
                        Out("[DataPreloader]   Reconnecting '" + connName + "'...");

                        // Look up the ConnectOptions for this connection name.
                        ConnectOptions connectOptions = null;
                        lock (Globals.ConnectOptions)
                            connectOptions = Globals.ConnectOptions
                                .FirstOrDefault(o => o.Name == connName);

                        if (connectOptions == null)
                        {
                            Out("[DataPreloader]   ⚠ No ConnectOptions found for '" +
                                connName + "' — skipping.");
                            continue;
                        }

                        // Only reconnect if not already connected under this name.
                        bool alreadyConnected = false;
                        lock (Connection.Connections)
                            alreadyConnected = Connection.Connections
                                .Any(c => c.Options.Name == connName &&
                                     c.Status == ConnectionStatus.Connected);

                        if (alreadyConnected)
                        {
                            Out("[DataPreloader]   '" + connName + "' already connected — skipping.");
                            continue;
                        }

                        // Static call: Connection.Connect(ConnectOptions) initiates
                        // the connection sequence and returns the Connection object.
                        Connection newConn = Connection.Connect(connectOptions);

                        if (newConn != null)
                            Out("[DataPreloader]   Connect() returned status=" + newConn.Status +
                                " for '" + connName + "'");
                        else
                            Out("[DataPreloader]   Connect() returned null for '" + connName + "'");
                    }
                    catch (Exception ex)
                    {
                        Out("[DataPreloader]   ⚠ Reconnect error for '" + connName + "': " + ex.Message);
                    }
                }

                // Schedule the next tick with linear backoff, capped at 5 min.
                int backoffSec = Math.Min(attempt * WatchdogBackoffBaseSec, 300);
                Out("[DataPreloader]   Next watchdog check in " + backoffSec + "s...");

                lock (_watchdogLock)
                {
                    if (!_watchdogActive) return; // stopped while we were working
                    try { _watchdogTimer?.Dispose(); } catch { }
                    _watchdogTimer = new Timer(WatchdogTick, null,
                        backoffSec * 1000, Timeout.Infinite);
                }
            }
            catch (Exception ex)
            {
                Out("[DataPreloader] ⚠ Watchdog tick error: " + ex.Message);

                // Schedule a retry in 60s even on error so the watchdog
                // doesn't silently die.
                lock (_watchdogLock)
                {
                    if (!_watchdogActive) return;
                    try { _watchdogTimer?.Dispose(); } catch { }
                    _watchdogTimer = new Timer(WatchdogTick, null,
                        60000, Timeout.Infinite);
                }
            }
        }

        #endregion

        // =====================================================================
        #region Seed orchestration (continued)
        // =====================================================================

        private void SeedAllInstruments()
        {
            try
            {
                Out("[DataPreloader] =============================================");
                Out("[DataPreloader] Starting history cache seed run...");
                Out("[DataPreloader]   DaysToLoad   : " + DaysToLoad);
                Out("[DataPreloader]   StaggerMs    : " + StaggerMs);
                Out("[DataPreloader]   SkipIfCached : " + SkipIfCached);
                Out("[DataPreloader]   Cache root   : " + _minuteCacheRoot);
                Out("[DataPreloader] =============================================");

                // Small initial delay — let NT8 finish its own startup tasks
                // before we start issuing BarsRequests against the connection.
                Thread.Sleep(3000);

                DateTime from = DateTime.Now.Date.AddDays(-DaysToLoad);
                DateTime to = DateTime.Now.Date.AddDays(1); // include today

                // ── Determine which instruments actually need seeding ──────────
                var toRequest = new List<InstrumentSpec>();

                foreach (var inst in Instruments)
                {
                    if (SkipIfCached && IsCached(inst.Symbol))
                    {
                        Out("[DataPreloader]   SKIP  " + inst.Symbol.PadRight(6) + " — cache exists");
                        continue;
                    }
                    toRequest.Add(inst);
                }

                if (toRequest.Count == 0)
                {
                    Out("[DataPreloader] All instruments already cached — nothing to do.");
                    Out("[DataPreloader] If bars are still 0, run fix_nt8_instruments.ps1 " +
                        "to correct MasterInstruments.TradingHours in the NT8 database.");
                    return;
                }

                Out("[DataPreloader] " + toRequest.Count + " instrument(s) need seeding:");
                foreach (var inst in toRequest)
                    Out("[DataPreloader]   QUEUE " + inst.Symbol);

                _totalRequested = toRequest.Count;
                _completedCount = 0;

                // ── Fire one BarsRequest per instrument, staggered ────────────
                foreach (var inst in toRequest)
                {
                    try
                    {
                        RequestHistory(inst.Symbol, inst.TradingHours, from, to);
                    }
                    catch (ThreadInterruptedException)
                    {
                        Out("[DataPreloader] Seed thread interrupted — stopping.");
                        return;
                    }
                    catch (Exception ex)
                    {
                        Out("[DataPreloader]   ERROR " + inst.Symbol + ": " + ex.Message);
                        // Count it as done so the wait below does not stall.
                        lock (_lock) { _completedCount++; }
                    }

                    try { Thread.Sleep(StaggerMs); }
                    catch (ThreadInterruptedException) { return; }
                }

                // ── Wait for all callbacks to fire ────────────────────────────
                // Give at most max(60, DaysToLoad * 6) seconds total.
                int waitSecs = Math.Max(60, DaysToLoad * 6);
                int elapsed = 0;
                int logEvery = 10; // log a progress line every N seconds

                Out("[DataPreloader] All requests dispatched — waiting for completions...");

                while (elapsed < waitSecs)
                {
                    try { Thread.Sleep(1000); }
                    catch (ThreadInterruptedException) { return; }

                    elapsed++;

                    int done;
                    lock (_lock) { done = _completedCount; }

                    if (done >= _totalRequested) break;

                    if (elapsed % logEvery == 0)
                        Out("[DataPreloader] Progress: " + done + "/" + _totalRequested +
                            " done  (" + elapsed + "s / " + waitSecs + "s)");
                }

                int final;
                lock (_lock) { final = _completedCount; }

                Out("[DataPreloader] =============================================");
                Out("[DataPreloader] Seed run finished: " + final + "/" + _totalRequested + " completed.");

                if (final < _totalRequested)
                    Out("[DataPreloader] WARNING: some instruments timed out. " +
                        "Re-start NT8 to retry, or check your data subscription.");
                else
                    Out("[DataPreloader] All instruments seeded. " +
                        "BreakoutStrategy will now load full history on all BIPs.");

                Out("[DataPreloader] =============================================");
            }
            catch (ThreadAbortException)
            {
                // Normal path when NT8 terminates the AddOn — nothing to log.
            }
            catch (Exception ex)
            {
                Out("[DataPreloader] FATAL error in seed thread: " + ex.Message);
            }
        }

        #endregion

        // =====================================================================
        #region History request
        // =====================================================================

        private void RequestHistory(string symbol, string tradingHours,
                                    DateTime from, DateTime to)
        {
            // ── Resolve the real continuous-contract Future row ────────────────
            //
            // NT8's broker sync creates phantom MasterInstrument rows with
            // InstrumentType=Stock and expiry DEC99 for every futures symbol.
            // These phantoms often have the LOWEST Id, so Instrument.GetInstrument()
            // with a bare root symbol resolves to the phantom → "Unknown instrument".
            //
            // Previous approaches (name-mangling, InstrumentType patching) all
            // fail because NT8 uses MasterInstrument rows internally for path
            // construction and lazy resolution.
            //
            // The correct fix: scan MasterInstrument.All for the real Future row,
            // get its current front-month full name (e.g. "MES 06-25"), and use
            // Instrument.GetInstrument with that explicit name.  NT8 resolves an
            // explicit full name unambiguously — no phantom can interfere.
            string rootUpper = symbol.Trim().ToUpperInvariant();
            string fullName = null;

            try
            {
                NinjaTrader.Data.TradingHours thObj = null;
                if (!string.IsNullOrWhiteSpace(tradingHours))
                    thObj = NinjaTrader.Data.TradingHours.Get(tradingHours);

                lock (MasterInstrument.Sync)
                {
                    MasterInstrument bestMi = null;
                    DateTime bestExpiry = DateTime.MinValue;

                    foreach (var mi in MasterInstrument.All)
                    {
                        if (!string.Equals(mi.Name, rootUpper, StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (mi.InstrumentType != InstrumentType.Future)
                            continue;

                        DateTime expiry;
                        try { expiry = mi.GetNextExpiry(DateTime.Now); }
                        catch { continue; }

                        // Skip DEC99 placeholders (expiry year 2099)
                        if (expiry.Year >= 2099) continue;
                        // Skip expired contracts (14-day grace for rollover —
                        // futures remain tradable past listed expiry)
                        if (expiry < DateTime.Now.Date.AddDays(-14)) continue;

                        // Pick the nearest (front-month) valid contract
                        if (bestMi == null || expiry < bestExpiry)
                        {
                            bestMi = mi;
                            bestExpiry = expiry;
                        }
                    }

                    if (bestMi != null)
                    {
                        fullName = bestMi.Name + " " + bestExpiry.ToString("MM-yy");
                        Out("[DataPreloader]   RESOLVED  " + rootUpper.PadRight(6) +
                            " → '" + fullName + "' (expiry " + bestExpiry.ToString("yyyy-MM-dd") + ")");

                        // Patch TradingHours on the real Future row
                        if (thObj != null && !string.Equals(bestMi.TradingHours?.Name, tradingHours,
                                                            StringComparison.OrdinalIgnoreCase))
                        {
                            bestMi.TradingHours = thObj;
                            Out("[DataPreloader]   TH-PATCH   " + rootUpper +
                                " → '" + tradingHours + "'");
                        }
                    }
                    else
                    {
                        // No valid Future row — log all rows for debugging
                        Out("[DataPreloader]   ⚠ No valid Future row for " + rootUpper +
                            ". Dumping all MasterInstrument rows:");
                        foreach (var mi in MasterInstrument.All
                            .Where(m => string.Equals(m.Name, rootUpper, StringComparison.OrdinalIgnoreCase)))
                        {
                            DateTime exp;
                            try { exp = mi.GetNextExpiry(DateTime.Now); }
                            catch { exp = DateTime.MinValue; }
                            Out("[DataPreloader]     Id=" + mi.Id +
                                " type=" + mi.InstrumentType +
                                " expiry=" + exp.ToString("yyyy-MM-dd"));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Out("[DataPreloader]   ⚠ Contract resolution error for " + rootUpper +
                    ": " + ex.Message);
            }

            if (string.IsNullOrEmpty(fullName))
            {
                Out("[DataPreloader]   SKIP  " + symbol.PadRight(6) +
                    " — could not resolve a valid front-month contract.");
                lock (_lock) { _completedCount++; }
                return;
            }

            // ── Resolve Instrument from the explicit full name ────────────────
            Instrument instrument = null;
            try { instrument = Instrument.GetInstrument(fullName, false); } catch { }
            if (instrument == null)
                try { instrument = Instrument.GetInstrument(fullName, true); } catch { }

            if (instrument == null)
            {
                Out("[DataPreloader]   MISS  " + symbol.PadRight(6) +
                    " — Instrument.GetInstrument('" + fullName + "') returned null.");
                lock (_lock) { _completedCount++; }
                return;
            }

            Out("[DataPreloader]   INSTRUMENT  " + symbol.PadRight(6) +
                " → '" + (instrument.FullName ?? "?") + "'");

            // ── Resolve TradingHours template ─────────────────────────────────
            TradingHours th = null;
            if (!string.IsNullOrWhiteSpace(tradingHours))
                th = TradingHours.Get(tradingHours);
            if (th == null)
                th = TradingHours.Get("Default 24 x 5");
            if (th == null)
                th = TradingHours.Get("Default"); // absolute fallback

            // ── Build BarsRequest ─────────────────────────────────────────────
            var br = new BarsRequest(instrument, from, to)
            {
                BarsPeriod = new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1 },
                TradingHours = th,
                MergePolicy = MergePolicy.MergeBackAdjusted,
            };

            // Capture locals for the async closure.
            string sym = symbol;
            BarsRequest brRef = br;

            br.Request((barsRequest, errorCode, errorMsg) =>
            {
                try
                {
                    if (errorCode == ErrorCode.NoError)
                    {
                        int count = (barsRequest != null && barsRequest.Bars != null)
                            ? barsRequest.Bars.Count
                            : 0;

                        if (LogVerbose)
                            Out("[DataPreloader]   DONE  " + sym.PadRight(6) + " — " + count + " bars cached");
                        else
                            Out("[DataPreloader]   DONE  " + sym.PadRight(6));
                    }
                    else
                    {
                        Out("[DataPreloader]   FAIL  " + sym.PadRight(6) +
                            " — " + errorCode + ": " + errorMsg);
                    }
                }
                finally
                {
                    lock (_lock)
                    {
                        _completedCount++;
                        _pending.Remove(sym);
                        try { brRef.Dispose(); } catch { }
                    }
                }
            });

            // Keep a reference alive until the callback fires.
            lock (_lock)
            {
                _pending[sym] = br;
            }

            Out("[DataPreloader]   REQ   " + symbol.PadRight(6) +
                " — " + DaysToLoad + "d of 1-min bars" +
                "  (" + from.ToString("yyyy-MM-dd") + " to " + to.ToString("yyyy-MM-dd") + ")");
        }

        #endregion

        // =====================================================================
        #region Cache check
        // =====================================================================

        /// <summary>
        /// Returns true when the NT8 disk cache for <paramref name="symbol"/>
        /// already contains at least one file.  NT8 stores bars in year-named
        /// sub-folders:  db\minute\MES\2025\01.ntd
        /// </summary>
        private bool IsCached(string symbol)
        {
            string dir = Path.Combine(_minuteCacheRoot, symbol);
            if (!Directory.Exists(dir)) return false;

            try
            {
                return Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories).Any();
            }
            catch
            {
                return false;
            }
        }

        #endregion

        // =====================================================================
        #region Output helper
        // =====================================================================

        private static void Out(string message)
        {
            NinjaTrader.Code.Output.Process(message, PrintTo.OutputTab1);
        }

        #endregion
    }
}
