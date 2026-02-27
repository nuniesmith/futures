#!/usr/bin/env python3
"""
One-time migration script: SQLite → PostgreSQL
================================================

Copies all data from the local SQLite journal database into the
Postgres instance running in Docker.  Safe to run multiple times —
uses ON CONFLICT DO NOTHING to avoid duplicates.

Now also migrates audit tables (risk_events, orb_events) and includes
a post-migration verification step that compares row counts between
SQLite and Postgres.

Usage:
    # From project root, with .env loaded:
    python scripts/migrate_to_postgres.py

    # Or specify paths explicitly:
    python scripts/migrate_to_postgres.py \
        --sqlite data/futures_journal.db \
        --pg "postgresql+psycopg://futures_user:futures_dev_pass@localhost:5432/futures_db"

    # Dry-run (read SQLite, print counts, don't write to Postgres):
    python scripts/migrate_to_postgres.py --dry-run

    # Clean migration (truncate Postgres tables first, then re-insert):
    python scripts/migrate_to_postgres.py --clean

    # Verify only (no migration, just compare counts):
    python scripts/migrate_to_postgres.py --verify-only

    # Skip verification after migration:
    python scripts/migrate_to_postgres.py --no-verify

Prerequisites:
    - Postgres container is running: docker compose up -d postgres
    - Tables have been created by the data-service (init_db)
    - pip install psycopg[binary] sqlalchemy
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    """Return a list of user table names in the SQLite database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [r[0] for r in rows]


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    """Return the number of rows in a SQLite table."""
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0] if row else 0


def _pg_row_count(pg_conn, table: str) -> int:
    """Return the number of rows in a Postgres table, or -1 if table missing."""
    from sqlalchemy import text

    try:
        row = pg_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
        return row[0] if row else 0
    except Exception:
        return -1


def _pg_truncate(pg_conn, table: str) -> None:
    """Truncate a Postgres table and restart its identity sequence."""
    from sqlalchemy import text

    pg_conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))


def _pg_table_exists(pg_conn, table: str) -> bool:
    """Check whether a table exists in the Postgres public schema."""
    from sqlalchemy import text

    row = pg_conn.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_schema = 'public' AND table_name = :tbl"
            ")"
        ),
        {"tbl": table},
    ).fetchone()
    return bool(row and row[0])


# ---------------------------------------------------------------------------
# Generic table migrator
# ---------------------------------------------------------------------------


def _migrate_table(
    src: sqlite3.Connection,
    pg_conn,
    table: str,
    order_by: str = "id",
    conflict_clause: str = "ON CONFLICT DO NOTHING",
    drop_cols: list[str] | None = None,
    verbose: bool = False,
    label: str = "",
) -> tuple[int, list[str]]:
    """Migrate a single table from SQLite to Postgres.

    Returns (migrated_count, error_list).
    """
    from sqlalchemy import text

    drop_cols = drop_cols or ["id"]
    label = label or table
    migrated = 0
    errors = []

    rows = src.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()

    for row in rows:
        d = dict(row)
        row_id = d.get("id", "?")
        for col in drop_cols:
            d.pop(col, None)

        cols = list(d.keys())
        col_str = ", ".join(cols)
        placeholders = ", ".join(f":{k}" for k in cols)

        try:
            pg_conn.execute(
                text(
                    f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) "
                    f"{conflict_clause}"
                ),
                d,
            )
            migrated += 1
            if verbose:
                # Build a short summary of the row for display
                summary_keys = [
                    k
                    for k in cols
                    if k
                    in (
                        "asset",
                        "direction",
                        "entry",
                        "trade_date",
                        "net_pnl",
                        "event_type",
                        "symbol",
                        "reason",
                        "timestamp",
                        "or_high",
                        "or_low",
                        "breakout_detected",
                    )
                ]
                summary = ", ".join(f"{k}={d.get(k, '?')}" for k in summary_keys[:4])
                print(f"   ✓ {label} #{row_id}: {summary}")
        except Exception as exc:
            errors.append(f"{label} #{row_id}: {exc}")
            print(f"   ✗ {label} #{row_id}: {exc}")

    return migrated, errors


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    sqlite_path: str,
    pg_url: str,
    verbose: bool = False,
) -> dict:
    """Compare row counts between SQLite and Postgres for all migratable tables.

    Returns a dict:
        {
            "ok": bool,
            "timestamp": str,
            "tables": {
                "<name>": {
                    "sqlite": int,
                    "postgres": int,
                    "match": bool,
                    "note": str,
                }
            },
            "summary": str,
        }
    """
    from sqlalchemy import create_engine

    result = {
        "ok": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tables": {},
        "summary": "",
    }

    # ── SQLite ──────────────────────────────────────────────────────────
    if not os.path.exists(sqlite_path):
        result["ok"] = False
        result["summary"] = f"SQLite file not found: {sqlite_path}"
        return result

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    sqlite_tables = get_sqlite_tables(src)

    # Tables we care about migrating (order matters for display)
    migratable = ["trades_v2", "daily_journal", "risk_events", "orb_events"]

    # ── Postgres ────────────────────────────────────────────────────────
    try:
        engine = create_engine(pg_url)
        pg_conn = engine.connect()
    except Exception as exc:
        src.close()
        result["ok"] = False
        result["summary"] = f"Could not connect to Postgres: {exc}"
        return result

    all_match = True
    for table in migratable:
        entry = {"sqlite": 0, "postgres": 0, "match": False, "note": ""}

        if table not in sqlite_tables:
            entry["note"] = "not in SQLite (skipped)"
            entry["match"] = True  # nothing to migrate → OK
        else:
            entry["sqlite"] = get_row_count(src, table)

        if not _pg_table_exists(pg_conn, table):
            entry["note"] = "table missing in Postgres!"
            entry["match"] = False
        else:
            entry["postgres"] = _pg_row_count(pg_conn, table)

        if not entry["note"]:
            entry["match"] = entry["sqlite"] <= entry["postgres"]
            if entry["sqlite"] == entry["postgres"]:
                entry["note"] = "✓ exact match"
            elif entry["postgres"] > entry["sqlite"]:
                entry["note"] = "✓ Postgres has more rows (OK — may have live data)"
            else:
                entry["note"] = (
                    f"✗ MISMATCH — {entry['sqlite'] - entry['postgres']} rows missing"
                )

        if not entry["match"]:
            all_match = False

        result["tables"][table] = entry

    # Also note skipped tables
    skip_tables = {"trades_v1_backup"}
    for table in sqlite_tables:
        if table in skip_tables:
            count = get_row_count(src, table)
            result["tables"][table] = {
                "sqlite": count,
                "postgres": 0,
                "match": True,
                "note": "skipped (legacy backup)",
            }

    pg_conn.close()
    engine.dispose()
    src.close()

    result["ok"] = all_match
    if all_match:
        result["summary"] = "All tables verified — Postgres has all migrated data."
    else:
        mismatched = [t for t, v in result["tables"].items() if not v["match"]]
        result["summary"] = (
            f"MISMATCH in {len(mismatched)} table(s): {', '.join(mismatched)}"
        )

    # ── Print ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 Post-Migration Verification")
    print("=" * 60)
    print(f"   SQLite:   {sqlite_path}")
    display_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    print(f"   Postgres: {display_url}")
    print()

    col_w = max(len(t) for t in result["tables"]) + 2
    print(f"   {'Table':<{col_w}} {'SQLite':>8} {'Postgres':>10}  Status")
    print(f"   {'-' * col_w} {'-' * 8} {'-' * 10}  {'-' * 30}")
    for table, info in result["tables"].items():
        sqlite_str = str(info["sqlite"]) if info["sqlite"] >= 0 else "N/A"
        pg_str = str(info["postgres"]) if info["postgres"] >= 0 else "N/A"
        print(f"   {table:<{col_w}} {sqlite_str:>8} {pg_str:>10}  {info['note']}")

    print()
    if result["ok"]:
        print("   ✅ " + result["summary"])
    else:
        print("   ❌ " + result["summary"])
    print("=" * 60)

    if verbose:
        print("\n📄 Full verification JSON:")
        print(json.dumps(result, indent=2))

    return result


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate(
    sqlite_path: str,
    pg_url: str,
    dry_run: bool = False,
    verbose: bool = False,
    run_verify: bool = True,
    clean: bool = False,
) -> dict:
    """Migrate data from SQLite to Postgres.

    Returns a dict with migration statistics.
    """
    from sqlalchemy import create_engine, text

    # ── Validate SQLite source ──────────────────────────────────────────
    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite file not found: {sqlite_path}")
        return {"error": f"File not found: {sqlite_path}"}

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row

    tables = get_sqlite_tables(src)
    print(f"\n📂 SQLite database: {sqlite_path}")
    print(f"   Tables found: {', '.join(tables) or '(none)'}")

    stats = {
        "trades": 0,
        "journal": 0,
        "risk_events": 0,
        "orb_events": 0,
        "skipped": 0,
        "errors": [],
    }

    for table in tables:
        count = get_row_count(src, table)
        print(f"   {table}: {count} rows")

    if not tables:
        print("\n⚠️  No tables to migrate.")
        src.close()
        return stats

    # ── Dry run? ────────────────────────────────────────────────────────
    if dry_run:
        print("\n🔍 DRY RUN — no data will be written to Postgres")
        src.close()

        if run_verify:
            verify(sqlite_path, pg_url, verbose=verbose)

        return stats

    # ── Connect to Postgres ─────────────────────────────────────────────
    print(f"\n🐘 Connecting to Postgres...")
    # Mask password in output
    display_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    print(f"   Target: {display_url}")

    try:
        engine = create_engine(pg_url)
        pg_conn = engine.connect()
    except Exception as exc:
        print(f"ERROR: Could not connect to Postgres: {exc}")
        src.close()
        return {"error": str(exc)}

    # ── Helper: decide whether to migrate a table ───────────────────────
    #    Tables without a natural unique key (trades_v2, risk_events,
    #    orb_events) will produce duplicates if migrated twice.  Default
    #    behaviour: skip if the Postgres table already has rows.  Pass
    #    --clean to truncate-and-reinsert instead.
    # ────────────────────────────────────────────────────────────────────

    def _should_migrate(table: str, has_unique_key: bool) -> bool:
        """Return True if we should proceed with migrating *table*."""
        pg_count = _pg_row_count(pg_conn, table)
        if pg_count <= 0:
            return True  # empty or missing — always safe

        if clean:
            print(f"   ⚠️  --clean: truncating {table} ({pg_count} existing rows)")
            _pg_truncate(pg_conn, table)
            pg_conn.commit()
            return True

        if has_unique_key:
            # ON CONFLICT … DO NOTHING will handle dedup for us
            return True

        # No unique key and table already has data → skip to avoid dups
        print(
            f"   ⏭️  Skipping {table} — Postgres already has {pg_count} rows. "
            f"Use --clean to truncate and re-migrate."
        )
        stats["skipped_tables"] = stats.get("skipped_tables", [])
        stats["skipped_tables"].append(table)
        return False

    # ── 1. Migrate trades_v2 ────────────────────────────────────────────
    if "trades_v2" in tables:
        print("\n📋 Migrating trades_v2...")
        if _should_migrate("trades_v2", has_unique_key=False):
            count, errs = _migrate_table(
                src,
                pg_conn,
                table="trades_v2",
                order_by="id",
                conflict_clause="ON CONFLICT DO NOTHING",
                drop_cols=["id"],
                verbose=verbose,
                label="Trade",
            )
            stats["trades"] = count
            stats["errors"].extend(errs)
            print(f"   → {count} trades migrated")

    # ── 2. Migrate daily_journal ────────────────────────────────────────
    if "daily_journal" in tables:
        print("\n📓 Migrating daily_journal...")
        if _should_migrate("daily_journal", has_unique_key=True):
            count, errs = _migrate_table(
                src,
                pg_conn,
                table="daily_journal",
                order_by="trade_date",
                conflict_clause="ON CONFLICT (trade_date) DO NOTHING",
                drop_cols=["id"],
                verbose=verbose,
                label="Journal",
            )
            stats["journal"] = count
            stats["errors"].extend(errs)
            print(f"   → {count} journal entries migrated")

    # ── 3. Migrate risk_events ──────────────────────────────────────────
    if "risk_events" in tables:
        print("\n🛡️  Migrating risk_events...")
        if _should_migrate("risk_events", has_unique_key=False):
            count, errs = _migrate_table(
                src,
                pg_conn,
                table="risk_events",
                order_by="id",
                conflict_clause="ON CONFLICT DO NOTHING",
                drop_cols=["id"],
                verbose=verbose,
                label="Risk Event",
            )
            stats["risk_events"] = count
            stats["errors"].extend(errs)
            print(f"   → {count} risk events migrated")

    # ── 4. Migrate orb_events ──────────────────────────────────────────
    if "orb_events" in tables:
        print("\n📊 Migrating orb_events...")
        if _should_migrate("orb_events", has_unique_key=False):
            count, errs = _migrate_table(
                src,
                pg_conn,
                table="orb_events",
                order_by="id",
                conflict_clause="ON CONFLICT DO NOTHING",
                drop_cols=["id"],
                verbose=verbose,
                label="ORB Event",
            )
            stats["orb_events"] = count
            stats["errors"].extend(errs)
            print(f"   → {count} ORB events migrated")

    # ── Skip v1 backup ──────────────────────────────────────────────────
    if "trades_v1_backup" in tables:
        count = get_row_count(src, "trades_v1_backup")
        print(f"\n📦 Found trades_v1_backup ({count} rows) — skipping (already in v2)")
        stats["skipped"] += count

    # ── Commit and clean up ─────────────────────────────────────────────
    pg_conn.commit()
    pg_conn.close()
    engine.dispose()
    src.close()

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("✅ Migration complete!")
    print(f"   Trades migrated:    {stats['trades']}")
    print(f"   Journal entries:    {stats['journal']}")
    print(f"   Risk events:        {stats['risk_events']}")
    print(f"   ORB events:         {stats['orb_events']}")
    if stats["skipped"]:
        print(f"   Skipped (v1):       {stats['skipped']}")
    if stats.get("skipped_tables"):
        print(f"   Skipped tables:     {', '.join(stats['skipped_tables'])}")
    if stats["errors"]:
        print(f"   Errors:             {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"     - {err}")
    else:
        print("   Errors:             0")
    print("=" * 50)

    # ── Post-migration verification ─────────────────────────────────────
    if run_verify:
        verify_result = verify(sqlite_path, pg_url, verbose=verbose)
        stats["verification"] = verify_result
        if not verify_result["ok"]:
            print("\n⚠️  Post-migration verification detected mismatches!")
            print("   Run with --verify-only to re-check after resolving issues.")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Migrate futures trading data from SQLite to PostgreSQL",
    )
    parser.add_argument(
        "--sqlite",
        default=os.getenv("DB_PATH", "data/futures_journal.db"),
        help="Path to the SQLite database file (default: data/futures_journal.db)",
    )
    parser.add_argument(
        "--pg",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://futures_user:futures_dev_pass@localhost:5432/futures_db",
        ),
        help="PostgreSQL connection URL (default: from DATABASE_URL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read SQLite and show counts without writing to Postgres",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip migration — only run the post-migration verification check",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-migration verification step",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Truncate Postgres tables before migrating (avoids duplicates on re-run)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each migrated row and detailed verification JSON",
    )
    args = parser.parse_args()

    # ── Verify-only mode ────────────────────────────────────────────────
    if args.verify_only:
        print("=" * 60)
        print("🔍 SQLite ↔ PostgreSQL Verification (no migration)")
        print("=" * 60)
        result = verify(
            sqlite_path=args.sqlite,
            pg_url=args.pg,
            verbose=args.verbose,
        )
        sys.exit(0 if result["ok"] else 1)

    # ── Full migration ──────────────────────────────────────────────────
    print("=" * 50)
    print("🔄 SQLite → PostgreSQL Migration")
    print("=" * 50)

    result = migrate(
        sqlite_path=args.sqlite,
        pg_url=args.pg,
        dry_run=args.dry_run,
        verbose=args.verbose,
        run_verify=not args.no_verify,
        clean=args.clean,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
