#!/usr/bin/env python3
"""
One-time migration script: SQLite → PostgreSQL
================================================

Copies all data from the local SQLite journal database into the
Postgres instance running in Docker.  Safe to run multiple times —
uses ON CONFLICT DO NOTHING to avoid duplicates.

Usage:
    # From project root, with .env loaded:
    python scripts/migrate_to_postgres.py

    # Or specify paths explicitly:
    python scripts/migrate_to_postgres.py \
        --sqlite data/futures_journal.db \
        --pg "postgresql+psycopg://futures_user:futures_dev_pass@localhost:5432/futures_db"

    # Dry-run (read SQLite, print counts, don't write to Postgres):
    python scripts/migrate_to_postgres.py --dry-run

Prerequisites:
    - Postgres container is running: docker compose up -d postgres
    - Tables have been created by the data-service (init_db)
    - pip install psycopg[binary] sqlalchemy
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Ensure src/ is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))


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


def migrate(
    sqlite_path: str,
    pg_url: str,
    dry_run: bool = False,
    verbose: bool = False,
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

    stats = {"trades": 0, "journal": 0, "skipped": 0, "errors": []}

    for table in tables:
        count = get_row_count(src, table)
        print(f"   {table}: {count} rows")

    if not tables:
        print("\n⚠️  No tables to migrate.")
        src.close()
        return stats

    # ── Connect to Postgres ─────────────────────────────────────────────
    if dry_run:
        print("\n🔍 DRY RUN — no data will be written to Postgres")
        src.close()
        return stats

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

    # ── Migrate trades_v2 ───────────────────────────────────────────────
    if "trades_v2" in tables:
        print("\n📋 Migrating trades_v2...")
        rows = src.execute("SELECT * FROM trades_v2 ORDER BY id").fetchall()

        for row in rows:
            d = dict(row)
            row_id = d.pop("id", None)

            cols = list(d.keys())
            col_str = ", ".join(cols)
            placeholders = ", ".join(f":{k}" for k in cols)

            try:
                pg_conn.execute(
                    text(
                        f"INSERT INTO trades_v2 ({col_str}) VALUES ({placeholders}) "
                        f"ON CONFLICT DO NOTHING"
                    ),
                    d,
                )
                stats["trades"] += 1
                if verbose:
                    print(
                        f"   ✓ Trade #{row_id}: {d.get('asset', '?')} "
                        f"{d.get('direction', '?')} @ {d.get('entry', '?')}"
                    )
            except Exception as exc:
                stats["errors"].append(f"Trade #{row_id}: {exc}")
                print(f"   ✗ Trade #{row_id}: {exc}")

        print(f"   → {stats['trades']} trades migrated")

    # ── Migrate daily_journal ───────────────────────────────────────────
    if "daily_journal" in tables:
        print("\n📓 Migrating daily_journal...")
        rows = src.execute("SELECT * FROM daily_journal ORDER BY trade_date").fetchall()

        for row in rows:
            d = dict(row)
            d.pop("id", None)

            cols = list(d.keys())
            col_str = ", ".join(cols)
            placeholders = ", ".join(f":{k}" for k in cols)

            try:
                pg_conn.execute(
                    text(
                        f"INSERT INTO daily_journal ({col_str}) VALUES ({placeholders}) "
                        f"ON CONFLICT (trade_date) DO NOTHING"
                    ),
                    d,
                )
                stats["journal"] += 1
                if verbose:
                    print(
                        f"   ✓ {d.get('trade_date', '?')}: "
                        f"net=${d.get('net_pnl', 0):,.2f}"
                    )
            except Exception as exc:
                stats["errors"].append(f"Journal {d.get('trade_date', '?')}: {exc}")
                print(f"   ✗ {d.get('trade_date', '?')}: {exc}")

        print(f"   → {stats['journal']} journal entries migrated")

    # ── Migrate v1 backup if present ────────────────────────────────────
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
    print(f"   Trades migrated:  {stats['trades']}")
    print(f"   Journal entries:  {stats['journal']}")
    if stats["skipped"]:
        print(f"   Skipped (v1):     {stats['skipped']}")
    if stats["errors"]:
        print(f"   Errors:           {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"     - {err}")
    else:
        print("   Errors:           0")
    print("=" * 50)

    return stats


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
        "-v",
        "--verbose",
        action="store_true",
        help="Print each migrated row",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("🔄 SQLite → PostgreSQL Migration")
    print("=" * 50)

    result = migrate(
        sqlite_path=args.sqlite,
        pg_url=args.pg,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
