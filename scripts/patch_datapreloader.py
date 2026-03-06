#!/usr/bin/env python3
"""
patch_datapreloader.py
======================
Syncs the DataPreloader AddOn's instrument list with BreakoutStrategy's
5 core assets (MGC, MES, MNQ, MYM, 6E).

MGC is the primary chart instrument (BIP0) so it's excluded from the
preloader — the chart loads its own data automatically. That leaves
4 instruments for the preloader to seed: MES, MNQ, MYM, 6E.

Usage:
    python scripts/patch_datapreloader.py [--dry-run]
"""

import os
import re
import shutil
import sys
from datetime import datetime

NINJATRADER_REPO = os.path.expanduser("~/github/ninjatrader")
SRC_FILE = os.path.join(NINJATRADER_REPO, "src", "addons", "DataPreloader.cs")
BACKUP_EXT = ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")


def read_file(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    dry_run = "--dry-run" in sys.argv

    if not os.path.exists(SRC_FILE):
        print("ERROR: Source file not found: " + SRC_FILE)
        sys.exit(1)

    src = read_file(SRC_FILE)
    original_len = len(src)
    changes = []

    # ── Fix 1: Header comment "15 instruments" → "5 core instruments" ────
    old1 = "BreakoutStrategy subscribes to 15 instruments via AddDataSeries()"
    new1 = "BreakoutStrategy subscribes to 5 core instruments via AddDataSeries()"
    if old1 in src:
        src = src.replace(old1, new1)
        changes.append("Fix 1: Header comment 15 -> 5 instruments")

    # ── Fix 2: Header comment "15 days" → "60 days" (stale comment) ─────
    old2 = "and issues BarsRequests for 15 days of 1-minute history for every instrument"
    new2 = "and issues BarsRequests for 60 days of 1-minute history for every instrument"
    if old2 in src:
        src = src.replace(old2, new2)
        changes.append("Fix 2: Header comment 15 days -> 60 days")

    # ── Fix 3: Replace the instrument array + comments via regex ─────────
    # Match from the comment block above Instruments = { ... }; through the
    # pending instruments comment block.
    pattern = (
        r"(        // Active instruments used by BreakoutStrategy \(15\)\..*?"
        r'new InstrumentSpec\("MET",\s*"Cryptocurrency"\),\s*\};'
        r".*?"
        r'//   new InstrumentSpec\("ZW",\s*"CBOT Agriculturals ETH"\),)'
    )

    # Build replacement as a plain string — use a lambda with re.sub to
    # avoid re interpreting backslashes in the replacement text.
    _repl_lines = [
        r"        // Core 5 instruments matching BreakoutStrategy.CTrackedInstruments.",
        r"        // Each entry: (root symbol, trading-hours template name).",
        r"        // Template names must exactly match files in:",
        r"        //   Documents\NinjaTrader 8\templates\TradingHours\ ",
        r"        //",
        r"        // NOTE: Instrument.GetInstrument only accepts (string name, bool create).",
        r"        // There is no exchange-filtered overload in NT8's public API.",
        r"        // MGC is intentionally excluded -- it is the primary chart instrument",
        r"        // (BIP0) and always loads its own data from the chart automatically.",
        r"        private static readonly InstrumentSpec[] Instruments =",
        r"        {",
        r"            // Core 4 (+ MGC on chart = 5 total) -- synced with BreakoutStrategy CTrackedInstruments",
        r'            new InstrumentSpec("MES", "CME US Index Futures ETH"),',
        r'            new InstrumentSpec("MNQ", "CME US Index Futures ETH"),',
        r'            new InstrumentSpec("MYM", "CME US Index Futures ETH"),',
        r'            new InstrumentSpec("6E",  "CME FX Futures ETH"),',
        r"        };",
        r"",
        r"        // Extended instruments -- uncomment and move into Instruments[] to",
        r"        // re-enable.  Removed to sync with BreakoutStrategy's 5-asset config.",
        r'        //   new InstrumentSpec("M2K", "CME US Index Futures ETH"),',
        r'        //   new InstrumentSpec("6B",  "CME FX Futures ETH"),',
        r'        //   new InstrumentSpec("6J",  "CME FX Futures ETH"),',
        r'        //   new InstrumentSpec("6A",  "CME FX Futures ETH"),',
        r'        //   new InstrumentSpec("6C",  "CME FX Futures ETH"),',
        r'        //   new InstrumentSpec("6S",  "CME FX Futures ETH"),',
        r'        //   new InstrumentSpec("ZN",  "CBOT Interest Rate ETH"),',
        r'        //   new InstrumentSpec("ZB",  "CBOT Interest Rate ETH"),',
        r'        //   new InstrumentSpec("MBT", "Cryptocurrency"),',
        r'        //   new InstrumentSpec("MET", "Cryptocurrency"),',
        r"        //",
        r"        // Pending data subscription:",
        r'        //   new InstrumentSpec("MCL", "CME Commodities ETH"),',
        r'        //   new InstrumentSpec("MNG", "CME Commodities ETH"),',
        r'        //   new InstrumentSpec("SIL", "Nymex Metals - Energy ETH"),',
        r'        //   new InstrumentSpec("MHG", "Nymex Metals - Energy ETH"),',
        r'        //   new InstrumentSpec("ZC",  "CBOT Agriculturals ETH"),',
        r'        //   new InstrumentSpec("ZS",  "CBOT Agriculturals ETH"),',
        r'        //   new InstrumentSpec("ZW",  "CBOT Agriculturals ETH"),',
    ]
    _repl_text = "\n".join(_repl_lines)

    rx = re.compile(pattern, re.DOTALL)
    if rx.search(src):
        src = rx.sub(lambda m: _repl_text, src)
        changes.append("Fix 3: Replaced instrument list (14 -> 4 active, rest commented)")
    else:
        print("WARNING: Instrument block regex did not match. Trying line-by-line fallback...")
        # Fallback: just check if already patched
        if "Core 5 instruments matching BreakoutStrategy" in src:
            print("  -> Already patched, skipping.")
        else:
            print("  -> Could not find instrument block to patch!")
            sys.exit(1)

    # ── Fix 4: Description string ────────────────────────────────────────
    old4 = 'Pre-seeds the NT8 minute-bar disk cache for all "'
    new4 = 'Pre-seeds the NT8 minute-bar disk cache for the 5 core "'
    if old4 in src:
        src = src.replace(old4, new4)
        changes.append("Fix 4: Description string 'all' -> '5 core'")

    # Also fix the second half if it exists on next line
    old4b = '"active BreakoutStrategy instruments at startup'
    new4b = '"BreakoutStrategy instruments at startup'
    if old4b in src:
        src = src.replace(old4b, new4b)
        changes.append("Fix 4b: Description string trim 'active'")

    # ── Report ───────────────────────────────────────────────────────────
    print("")
    print("Patches applied: " + str(len(changes)))
    for c in changes:
        print("  + " + c)
    print("")
    print("Size: " + str(original_len) + " -> " + str(len(src)) + " bytes (" + str(len(src) - original_len) + ")")

    if dry_run:
        print("")
        print("DRY RUN -- no files written.")
        return

    # Backup and write
    backup_path = SRC_FILE + BACKUP_EXT
    shutil.copy2(SRC_FILE, backup_path)
    print("Backup: " + backup_path)

    write_file(SRC_FILE, src)
    print("Written: " + SRC_FILE)

    # ── Verify ───────────────────────────────────────────────────────────
    print("")
    print("Verification:")
    result = read_file(SRC_FILE)
    checks = [
        ("4 active instruments", 'new InstrumentSpec("6E"'),
        ("MES present", 'new InstrumentSpec("MES"'),
        ("MNQ present", 'new InstrumentSpec("MNQ"'),
        ("MYM present", 'new InstrumentSpec("MYM"'),
        ("M2K commented out", '//   new InstrumentSpec("M2K"'),
        ("6B commented out", '//   new InstrumentSpec("6B"'),
        ("MBT commented out", '//   new InstrumentSpec("MBT"'),
        ("5 core in header", "5 core instruments"),
        ("Core 5 comment", "Core 5 instruments matching BreakoutStrategy"),
    ]
    all_ok = True
    for label, needle in checks:
        found = needle in result
        status = "OK" if found else "FAIL"
        if not found:
            all_ok = False
        print("  [" + status + "] " + label)

    if all_ok:
        print("")
        print("All checks passed!")
    else:
        print("")
        print("WARNING: Some checks failed -- review the output above.")

    print("")
    print("Next: Copy DataPreloader.cs to your NT8 machine:")
    print(
        "  Documents"
        + chr(92)
        + "NinjaTrader 8"
        + chr(92)
        + "bin"
        + chr(92)
        + "Custom"
        + chr(92)
        + "AddOns"
        + chr(92)
        + "DataPreloader.cs"
    )
    print("Then recompile in NinjaScript Editor (F5).")
    print("On next startup you should see '4 instrument(s) need seeding' instead of 14.")


if __name__ == "__main__":
    main()
