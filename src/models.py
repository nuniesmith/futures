"""
Account configurations, contract specifications, asset mappings,
and database helpers with OPEN/CLOSED status tracking.

Supports dual-database backends:
  - PostgreSQL via DATABASE_URL (production / Docker)
  - SQLite via DB_PATH (local dev / tests)

The active backend is chosen automatically at module load time:
  - If DATABASE_URL is set and starts with "postgresql", use Postgres.
  - Otherwise, fall back to SQLite at DB_PATH.

All CRUD functions use the same interface regardless of backend.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

_EST = ZoneInfo("America/New_York")

logger = logging.getLogger("models")

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "futures_journal.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Detect which backend to use
_USE_POSTGRES = DATABASE_URL.startswith("postgresql")

# SQLAlchemy engine (lazy-initialised for Postgres)
_sa_engine = None

# ---------------------------------------------------------------------------
# Daily journal schema — simple end-of-day P&L entries
# ---------------------------------------------------------------------------
_SCHEMA_DAILY_JOURNAL_SQLITE = """
CREATE TABLE IF NOT EXISTS daily_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT    NOT NULL UNIQUE,
    account_size    INTEGER NOT NULL,
    gross_pnl       REAL    NOT NULL DEFAULT 0.0,
    net_pnl         REAL    NOT NULL DEFAULT 0.0,
    commissions     REAL    NOT NULL DEFAULT 0.0,
    num_contracts   INTEGER DEFAULT 0,
    instruments     TEXT    DEFAULT '',
    notes           TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
"""

_SCHEMA_DAILY_JOURNAL_PG = """
CREATE TABLE IF NOT EXISTS daily_journal (
    id              SERIAL PRIMARY KEY,
    trade_date      TEXT    NOT NULL UNIQUE,
    account_size    INTEGER NOT NULL,
    gross_pnl       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    net_pnl         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    commissions     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    num_contracts   INTEGER DEFAULT 0,
    instruments     TEXT    DEFAULT '',
    notes           TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Contract mode: "micro" (default) or "full"
# Set via environment variable CONTRACT_MODE or toggle at runtime.
# ---------------------------------------------------------------------------
CONTRACT_MODE = os.getenv("CONTRACT_MODE", "micro").lower()

# ---------------------------------------------------------------------------
# Account profiles – 50k is 1/3 of 150k, 100k is 2/3
# ---------------------------------------------------------------------------
ACCOUNT_PROFILES = {
    "50k": {
        "size": 50_000,
        "risk_pct": 0.01,
        "risk_dollars": 500,
        "max_contracts": 2,
        "max_contracts_micro": 10,
        "soft_stop": -500,
        "hard_stop": -750,
        "eod_dd": 1_500,
        "label": "$50k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 1-2 full / 10 micro contracts on $50k (25% rule). "
            "Daily Loss Removed: $1,500."
        ),
    },
    "100k": {
        "size": 100_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_000,
        "max_contracts": 3,
        "max_contracts_micro": 20,
        "soft_stop": -1_000,
        "hard_stop": -1_500,
        "eod_dd": 3_000,
        "label": "$100k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 2-3 full / 20 micro contracts on $100k (25% rule). "
            "Daily Loss Removed: $3,000."
        ),
    },
    "150k": {
        "size": 150_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_500,
        "max_contracts": 4,
        "max_contracts_micro": 30,
        "soft_stop": -1_500,
        "hard_stop": -2_250,
        "eod_dd": 4_500,
        "label": "$150k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 3-4 full / 30 micro contracts on $150k (25% rule). "
            "Daily Loss Removed: $4,500."
        ),
    },
}

# ---------------------------------------------------------------------------
# Contract specifications — Full-size CME contracts
# ---------------------------------------------------------------------------
FULL_CONTRACT_SPECS = {
    "Gold": {"ticker": "GC=F", "point": 100, "tick": 0.10, "margin": 11_000},
    "Silver": {"ticker": "SI=F", "point": 5_000, "tick": 0.005, "margin": 9_000},
    "Copper": {"ticker": "HG=F", "point": 25_000, "tick": 0.0005, "margin": 6_000},
    "Crude Oil": {"ticker": "CL=F", "point": 1_000, "tick": 0.01, "margin": 7_000},
    "S&P": {"ticker": "ES=F", "point": 50, "tick": 0.25, "margin": 12_000},
    "Nasdaq": {"ticker": "NQ=F", "point": 20, "tick": 0.25, "margin": 17_000},
}

# ---------------------------------------------------------------------------
# Contract specifications — Micro CME contracts
# Micro contracts are 1/10 of full size (except Silver which is 1/5).
# These give more granularity for position sizing and scaling.
# ---------------------------------------------------------------------------
MICRO_CONTRACT_SPECS = {
    "Gold": {
        "ticker": "MGC=F",
        "data_ticker": "GC=F",
        "point": 10,
        "tick": 0.10,
        "margin": 1_100,
    },
    "Silver": {
        "ticker": "SIL=F",
        "data_ticker": "SI=F",
        "point": 1_000,
        "tick": 0.005,
        "margin": 1_800,
    },
    "Copper": {
        "ticker": "MHG=F",
        "data_ticker": "HG=F",
        "point": 2_500,
        "tick": 0.0005,
        "margin": 600,
    },
    "Crude Oil": {
        "ticker": "MCL=F",
        "data_ticker": "CL=F",
        "point": 100,
        "tick": 0.01,
        "margin": 700,
    },
    "S&P": {
        "ticker": "MES=F",
        "data_ticker": "ES=F",
        "point": 5,
        "tick": 0.25,
        "margin": 1_500,
    },
    "Nasdaq": {
        "ticker": "MNQ=F",
        "data_ticker": "NQ=F",
        "point": 2,
        "tick": 0.25,
        "margin": 2_100,
    },
}

# ---------------------------------------------------------------------------
# Active contract specs — selected by CONTRACT_MODE env var
# ---------------------------------------------------------------------------
CONTRACT_SPECS = (
    MICRO_CONTRACT_SPECS if CONTRACT_MODE == "micro" else FULL_CONTRACT_SPECS
)

# Convenience: name → data ticker (for Yahoo Finance fetching).
# Micro contracts track the same underlying price as full-size, so we always
# fetch from the full-size ticker which Yahoo reliably supports.
ASSETS: dict[str, str] = {
    name: str(spec.get("data_ticker", spec["ticker"]))
    for name, spec in CONTRACT_SPECS.items()
}

# Reverse lookup: data ticker → name
TICKER_TO_NAME: dict[str, str] = {
    str(spec.get("data_ticker", spec["ticker"])): name
    for name, spec in CONTRACT_SPECS.items()
}


def set_contract_mode(mode: str) -> dict:
    """Switch between 'micro' and 'full' contract specs at runtime.

    Updates the module-level CONTRACT_SPECS, ASSETS, and TICKER_TO_NAME dicts
    in place so all importers see the change.

    Returns the newly active CONTRACT_SPECS.
    """
    global CONTRACT_MODE
    mode = mode.lower()
    if mode not in ("micro", "full"):
        raise ValueError(f"Invalid contract mode '{mode}'. Use 'micro' or 'full'.")

    CONTRACT_MODE = mode
    source = MICRO_CONTRACT_SPECS if mode == "micro" else FULL_CONTRACT_SPECS

    CONTRACT_SPECS.clear()
    CONTRACT_SPECS.update(source)

    ASSETS.clear()
    ASSETS.update(
        {
            name: spec.get("data_ticker", spec["ticker"])
            for name, spec in CONTRACT_SPECS.items()
        }
    )

    TICKER_TO_NAME.clear()
    TICKER_TO_NAME.update(
        {
            spec.get("data_ticker", spec["ticker"]): name
            for name, spec in CONTRACT_SPECS.items()
        }
    )

    return CONTRACT_SPECS


# ---------------------------------------------------------------------------
# Trade statuses
# ---------------------------------------------------------------------------
STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"
STATUS_CANCELLED = "CANCELLED"


# ═══════════════════════════════════════════════════════════════════════════
# Database abstraction layer
# ═══════════════════════════════════════════════════════════════════════════
#
# Two backends, same interface:
#   - SQLite: uses sqlite3 directly (zero dependencies beyond stdlib)
#   - Postgres: uses psycopg via SQLAlchemy's raw connection interface
#
# The key difference is parameter placeholders: SQLite uses `?` while
# Postgres uses `%s`.  We normalise by writing SQL with `?` and
# converting at execution time when using Postgres.
# ═══════════════════════════════════════════════════════════════════════════


def _get_sa_engine():
    """Lazily create the SQLAlchemy engine for Postgres."""
    global _sa_engine
    if _sa_engine is None:
        try:
            from sqlalchemy import create_engine

            _sa_engine = create_engine(
                DATABASE_URL,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            logger.info("Postgres engine created: %s", DATABASE_URL.split("@")[-1])
        except Exception as exc:
            logger.error("Failed to create Postgres engine: %s", exc)
            raise
    return _sa_engine


def _convert_placeholders(sql: str) -> str:
    """Convert SQLite-style `?` placeholders to Postgres-style `%s`."""
    return sql.replace("?", "%s")


class _RowProxy:
    """Lightweight dict-like wrapper around a Postgres row tuple.

    Provides item access by column name (row["col"]) and dict()
    conversion, matching sqlite3.Row behaviour.
    """

    __slots__ = ("_data",)

    def __init__(self, columns: list[str], values: tuple):
        self._data = dict(zip(columns, values))

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str):
        return key in self._data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)


class _PgCursorWrapper:
    """Wraps a psycopg/DBAPI cursor to auto-convert `?` → `%s`
    and return _RowProxy objects for dict-like access."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql: str, params=None):
        converted = _convert_placeholders(sql)
        if params:
            self._cursor.execute(converted, params)
        else:
            self._cursor.execute(converted)
        return self

    def executescript(self, sql: str):
        """Execute a multi-statement script.  Postgres doesn't have
        executescript, so we just execute it as a single string."""
        # Replace SQLite-specific syntax if present
        pg_sql = sql.replace("AUTOINCREMENT", "")
        # SERIAL PRIMARY KEY already handles auto-increment in Postgres
        self._cursor.execute(pg_sql)
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        if self._cursor.description:
            columns = [desc[0] for desc in self._cursor.description]
            return _RowProxy(columns, row)
        return row

    def fetchall(self):
        rows = self._cursor.fetchall()
        if not rows or not self._cursor.description:
            return rows
        columns = [desc[0] for desc in self._cursor.description]
        return [_RowProxy(columns, r) for r in rows]

    @property
    def lastrowid(self):
        # psycopg3 doesn't always set lastrowid; we handle this
        # in the CRUD functions with RETURNING clauses
        return getattr(self._cursor, "lastrowid", None)

    @property
    def description(self):
        return self._cursor.description


class _PgConnectionWrapper:
    """Wraps a SQLAlchemy raw connection to match sqlite3.Connection API.

    Provides execute(), executescript(), commit(), close(), and
    row_factory-like behaviour via _RowProxy.
    """

    def __init__(self, raw_conn):
        self._conn = raw_conn
        self._cursor = raw_conn.cursor()

    def execute(self, sql: str, params=None):
        wrapper = _PgCursorWrapper(self._cursor)
        wrapper.execute(sql, params)
        return wrapper

    def executescript(self, sql: str):
        wrapper = _PgCursorWrapper(self._cursor)
        wrapper.executescript(sql)
        return wrapper

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        try:
            self._cursor.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


def _get_sqlite_conn() -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_conn():
    """Get a database connection (Postgres or SQLite).

    Returns an object with execute(), executescript(), commit(), close()
    methods.  Rows are accessible by column name via dict-style access.
    """
    if _USE_POSTGRES:
        try:
            engine = _get_sa_engine()
            raw = engine.raw_connection()
            return _PgConnectionWrapper(raw)
        except Exception as exc:
            logger.warning(
                "Postgres connection failed, falling back to SQLite: %s", exc
            )
            return _get_sqlite_conn()
    return _get_sqlite_conn()


def _is_using_postgres() -> bool:
    """Return True if Postgres is the active backend."""
    return _USE_POSTGRES


# ---------------------------------------------------------------------------
# Trades schema
# ---------------------------------------------------------------------------

_SCHEMA_V2_SQLITE = """
CREATE TABLE IF NOT EXISTS trades_v2 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    account_size    INTEGER NOT NULL,
    asset           TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry           REAL    NOT NULL,
    sl              REAL,
    tp              REAL,
    contracts       INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL DEFAULT 'OPEN',
    close_price     REAL,
    close_time      TEXT,
    pnl             REAL,
    rr              REAL,
    notes           TEXT    DEFAULT '',
    strategy        TEXT    DEFAULT ''
);
"""

_SCHEMA_V2_PG = """
CREATE TABLE IF NOT EXISTS trades_v2 (
    id              SERIAL PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    account_size    INTEGER NOT NULL,
    asset           TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry           DOUBLE PRECISION NOT NULL,
    sl              DOUBLE PRECISION,
    tp              DOUBLE PRECISION,
    contracts       INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL DEFAULT 'OPEN',
    close_price     DOUBLE PRECISION,
    close_time      TEXT,
    pnl             DOUBLE PRECISION,
    rr              DOUBLE PRECISION,
    notes           TEXT    DEFAULT '',
    strategy        TEXT    DEFAULT ''
);
"""

_MIGRATE_V1 = """
INSERT INTO trades_v2
    (id, created_at, account_size, asset, direction, entry, sl, tp,
     contracts, status, close_price, close_time, pnl, rr, notes, strategy)
SELECT
    id,
    date,
    150000,
    asset,
    direction,
    entry,
    sl,
    tp,
    contracts,
    CASE WHEN exit_price IS NOT NULL AND exit_price != 0 THEN 'CLOSED' ELSE 'OPEN' END,
    CASE WHEN exit_price != 0 THEN exit_price ELSE NULL END,
    CASE WHEN exit_price IS NOT NULL AND exit_price != 0 THEN date ELSE NULL END,
    pnl,
    rr,
    COALESCE(notes, ''),
    COALESCE(strategy, '')
FROM trades;
"""


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Initialise the trades_v2 and daily_journal tables.

    For SQLite: migrates from v1 schema if needed.
    For Postgres: creates tables idempotently (CREATE TABLE IF NOT EXISTS).
    """
    conn = _get_conn()

    if _USE_POSTGRES:
        try:
            conn.executescript(_SCHEMA_V2_PG)
            conn.executescript(_SCHEMA_DAILY_JOURNAL_PG)
            conn.commit()
            logger.info("Postgres tables initialised (trades_v2, daily_journal)")
        except Exception as exc:
            logger.error("Postgres init_db failed: %s", exc)
            conn.rollback()
        finally:
            conn.close()
        return

    # --- SQLite path ---
    # Check if new table already exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trades_v2'"
    )
    if cur.fetchone() is not None:
        # trades_v2 exists — just ensure daily_journal also exists
        conn.executescript(_SCHEMA_DAILY_JOURNAL_SQLITE)
        conn.close()
        return

    # Create v2 schema
    conn.executescript(_SCHEMA_V2_SQLITE)

    # Create daily journal schema
    conn.executescript(_SCHEMA_DAILY_JOURNAL_SQLITE)

    # Migrate from v1 if it exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
    )
    if cur.fetchone() is not None:
        try:
            conn.executescript(_MIGRATE_V1)
            conn.execute("ALTER TABLE trades RENAME TO trades_v1_backup")
            conn.commit()
        except Exception:
            conn.rollback()

    conn.close()


# ---------------------------------------------------------------------------
# Helper: convert row to dict (works for both sqlite3.Row and _RowProxy)
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> dict:
    """Convert a database row to a plain dict."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    return dict(row)


# ---------------------------------------------------------------------------
# Helper: insert with RETURNING for Postgres, lastrowid for SQLite
# ---------------------------------------------------------------------------


def _insert_returning_id(
    conn, sql: str, params: tuple, table: str = "trades_v2"
) -> int:
    """Execute an INSERT and return the new row's id.

    For Postgres, appends RETURNING id to the SQL.
    For SQLite, uses cursor.lastrowid.
    """
    if _USE_POSTGRES:
        pg_sql = _convert_placeholders(sql) + " RETURNING id"
        cur = conn._cursor if hasattr(conn, "_cursor") else conn.execute(pg_sql, params)
        if hasattr(conn, "_cursor"):
            conn._cursor.execute(pg_sql, params)
            row = conn._cursor.fetchone()
        else:
            row = cur.fetchone()
        return row[0] if row else 0
    else:
        cur = conn.execute(sql, params)
        return cur.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def create_trade(
    account_size: int,
    asset: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    contracts: int,
    strategy: str = "",
    notes: str = "",
) -> int:
    """Insert a new OPEN trade. Returns the new trade id."""
    conn = _get_conn()
    now = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")

    sql = """INSERT INTO trades_v2
           (created_at, account_size, asset, direction, entry, sl, tp,
            contracts, status, strategy, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    params = (
        now,
        account_size,
        asset,
        direction,
        entry,
        sl,
        tp,
        contracts,
        STATUS_OPEN,
        strategy,
        notes,
    )

    trade_id = _insert_returning_id(conn, sql, params, "trades_v2")
    conn.commit()
    conn.close()
    return trade_id


def close_trade(trade_id: int, close_price: float) -> dict:
    """Close an open trade and calculate realised P&L.

    Returns a dict with the trade details including pnl.
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM trades_v2 WHERE id = ?", (trade_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Trade {trade_id} not found")
    if row["status"] != STATUS_OPEN:
        conn.close()
        raise ValueError(f"Trade {trade_id} is already {row['status']}")

    asset = row["asset"]
    direction = row["direction"]
    entry = row["entry"]
    contracts = row["contracts"]

    spec = CONTRACT_SPECS.get(asset)
    point_value = spec["point"] if spec else 1.0

    if direction.upper() == "LONG":
        pnl = (close_price - entry) * point_value * contracts
        rr = (
            abs((close_price - entry) / (entry - row["sl"]))
            if row["sl"] and row["sl"] != entry
            else 0.0
        )
    else:
        pnl = (entry - close_price) * point_value * contracts
        rr = (
            abs((entry - close_price) / (row["sl"] - entry))
            if row["sl"] and row["sl"] != entry
            else 0.0
        )

    close_time = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")

    conn.execute(
        """UPDATE trades_v2
           SET status = ?, close_price = ?, close_time = ?, pnl = ?, rr = ?
           WHERE id = ?""",
        (STATUS_CLOSED, close_price, close_time, round(pnl, 2), round(rr, 2), trade_id),
    )
    conn.commit()

    result = _row_to_dict(row)
    result.update(
        status=STATUS_CLOSED,
        close_price=close_price,
        close_time=close_time,
        pnl=round(pnl, 2),
        rr=round(rr, 2),
    )
    conn.close()
    return result


def cancel_trade(trade_id: int) -> None:
    """Cancel an open trade (never filled)."""
    conn = _get_conn()
    conn.execute(
        "UPDATE trades_v2 SET status = ? WHERE id = ? AND status = ?",
        (STATUS_CANCELLED, trade_id, STATUS_OPEN),
    )
    conn.commit()
    conn.close()


def _query_to_list(conn, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return a list of dicts.

    Works with both SQLite and Postgres backends.  For SQLite, we use
    pd.read_sql for convenience.  For Postgres, we fetch rows directly
    and convert to dicts, avoiding pd.read_sql connection issues.
    """
    if _USE_POSTGRES:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]
    else:
        df = pd.read_sql(sql, conn, params=params)
        return df.to_dict(orient="records")


def get_open_trades(account_size: Optional[int] = None) -> list[dict]:
    """Return all OPEN trades, optionally filtered by account size."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY created_at DESC",
            (STATUS_OPEN, account_size),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY created_at DESC",
            (STATUS_OPEN,),
        )
    conn.close()
    return results


def get_closed_trades(account_size: Optional[int] = None) -> list[dict]:
    """Return all CLOSED trades for the journal."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY close_time DESC",
            (STATUS_CLOSED, account_size),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY close_time DESC",
            (STATUS_CLOSED,),
        )
    conn.close()
    return results


def get_all_trades(account_size: Optional[int] = None) -> list[dict]:
    """Return all trades regardless of status."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE account_size = ? ORDER BY created_at DESC",
            (account_size,),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 ORDER BY created_at DESC",
        )
    conn.close()
    return results


def get_today_pnl(account_size: Optional[int] = None) -> float:
    """Sum of realised P&L for trades closed today."""
    today = datetime.now(tz=_EST).strftime("%Y-%m-%d")
    conn = _get_conn()
    if account_size:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades_v2 WHERE status = ? AND close_time LIKE ? AND account_size = ?",
            (STATUS_CLOSED, f"{today}%", account_size),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades_v2 WHERE status = ? AND close_time LIKE ?",
            (STATUS_CLOSED, f"{today}%"),
        ).fetchone()
    conn.close()
    # For Postgres _RowProxy, access by index via values; for SQLite Row, use index
    if row is None:
        return 0.0
    if hasattr(row, "values"):
        vals = list(row.values())
        return float(vals[0]) if vals else 0.0
    return float(row[0]) if row else 0.0


def get_today_trades(account_size: Optional[int] = None) -> list[dict]:
    """Return all trades created or closed today."""
    today = datetime.now(tz=_EST).strftime("%Y-%m-%d")
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            """SELECT * FROM trades_v2
               WHERE (created_at LIKE ? OR close_time LIKE ?)
                 AND account_size = ?
               ORDER BY created_at DESC""",
            (f"{today}%", f"{today}%", account_size),
        )
    else:
        results = _query_to_list(
            conn,
            """SELECT * FROM trades_v2
               WHERE created_at LIKE ? OR close_time LIKE ?
               ORDER BY created_at DESC""",
            (f"{today}%", f"{today}%"),
        )
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------


def calc_max_contracts(
    entry: float,
    sl: float,
    asset: str,
    risk_dollars: float,
    hard_max: int,
) -> int:
    """Calculate max contracts respecting risk-per-trade and account cap.

    Uses the currently active CONTRACT_SPECS (micro or full).
    The hard_max should come from account profile's max_contracts_micro
    when trading micros, or max_contracts when trading full-size.
    """
    spec = CONTRACT_SPECS.get(asset)
    if spec is None:
        return 1
    risk_per_contract = abs(entry - sl) * spec["point"]
    if risk_per_contract <= 0:
        return 1
    raw = int(risk_dollars // risk_per_contract)
    return max(1, min(raw, hard_max))


def get_max_contracts_for_profile(profile_key: str) -> int:
    """Return the appropriate max contracts limit for the active contract mode."""
    profile = ACCOUNT_PROFILES.get(profile_key)
    if profile is None:
        return 4
    if CONTRACT_MODE == "micro":
        return profile.get("max_contracts_micro", profile["max_contracts"])
    return profile["max_contracts"]


def calc_pnl(
    asset: str,
    direction: str,
    entry: float,
    close_price: float,
    contracts: int,
) -> float:
    """Calculate P&L for a given trade."""
    spec = CONTRACT_SPECS.get(asset)
    point_value = spec["point"] if spec else 1.0
    if direction.upper() == "LONG":
        return (close_price - entry) * point_value * contracts
    else:
        return (entry - close_price) * point_value * contracts


# ---------------------------------------------------------------------------
# Daily Journal CRUD
# ---------------------------------------------------------------------------


def save_daily_journal(
    trade_date: str,
    account_size: int,
    gross_pnl: float,
    net_pnl: float,
    num_contracts: int = 0,
    instruments: str = "",
    notes: str = "",
) -> int:
    """Save or update a daily journal entry.

    Commissions are auto-calculated as gross_pnl - net_pnl.
    If an entry already exists for the given date, it is updated.
    Returns the row id.
    """
    commissions = round(gross_pnl - net_pnl, 2)
    now = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_conn()

    # Check if entry exists for this date
    existing = conn.execute(
        "SELECT id FROM daily_journal WHERE trade_date = ?", (trade_date,)
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE daily_journal
               SET account_size = ?, gross_pnl = ?, net_pnl = ?,
                   commissions = ?, num_contracts = ?, instruments = ?,
                   notes = ?, created_at = ?
               WHERE trade_date = ?""",
            (
                account_size,
                round(gross_pnl, 2),
                round(net_pnl, 2),
                commissions,
                num_contracts,
                instruments,
                notes,
                now,
                trade_date,
            ),
        )
        row_id = existing["id"]
    else:
        insert_sql = """INSERT INTO daily_journal
               (trade_date, account_size, gross_pnl, net_pnl, commissions,
                num_contracts, instruments, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        insert_params = (
            trade_date,
            account_size,
            round(gross_pnl, 2),
            round(net_pnl, 2),
            commissions,
            num_contracts,
            instruments,
            notes,
            now,
        )
        row_id = _insert_returning_id(conn, insert_sql, insert_params, "daily_journal")

    conn.commit()
    conn.close()
    return row_id  # type: ignore[return-value]


def get_daily_journal(
    limit: int = 30,
    account_size: Optional[int] = None,
) -> pd.DataFrame:
    """Return recent daily journal entries as a DataFrame."""
    conn = _get_conn()

    if _USE_POSTGRES:
        # For Postgres, query and convert to DataFrame manually
        if account_size:
            rows = _query_to_list(
                conn,
                """SELECT * FROM daily_journal
                   WHERE account_size = ?
                   ORDER BY trade_date DESC LIMIT ?""",
                (account_size, limit),
            )
        else:
            rows = _query_to_list(
                conn,
                "SELECT * FROM daily_journal ORDER BY trade_date DESC LIMIT ?",
                (limit,),
            )
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        # SQLite: use pd.read_sql directly
        if account_size:
            df = pd.read_sql(
                """SELECT * FROM daily_journal
                   WHERE account_size = ?
                   ORDER BY trade_date DESC LIMIT ?""",
                conn,
                params=(account_size, limit),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM daily_journal ORDER BY trade_date DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        conn.close()
        return df


def get_journal_stats(account_size: Optional[int] = None) -> dict:
    """Compute aggregate stats from the daily journal.

    Returns dict with total_days, total_gross, total_net, total_commissions,
    win_days, loss_days, win_rate, avg_daily_pnl, best_day, worst_day,
    current_streak.
    """
    df = get_daily_journal(limit=9999, account_size=account_size)
    if df.empty:
        return {
            "total_days": 0,
            "total_gross": 0.0,
            "total_net": 0.0,
            "total_commissions": 0.0,
            "win_days": 0,
            "loss_days": 0,
            "break_even_days": 0,
            "win_rate": 0.0,
            "avg_daily_net": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "current_streak": 0,
        }

    total_days = len(df)
    total_gross = float(df["gross_pnl"].sum())
    total_net = float(df["net_pnl"].sum())
    total_commissions = float(df["commissions"].sum())
    win_days = int((df["net_pnl"] > 0).sum())
    loss_days = int((df["net_pnl"] < 0).sum())
    break_even_days = int((df["net_pnl"] == 0).sum())
    win_rate = win_days / total_days * 100 if total_days > 0 else 0.0
    avg_daily_net = total_net / total_days if total_days > 0 else 0.0
    best_day = float(df["net_pnl"].max())
    worst_day = float(df["net_pnl"].min())

    # Current streak (sorted by date ascending for streak calc)
    sorted_df = df.sort_values("trade_date", ascending=True)
    streak = 0
    for pnl in reversed(sorted_df["net_pnl"].tolist()):
        if pnl > 0:
            if streak >= 0:
                streak += 1
            else:
                break
        elif pnl < 0:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break

    return {
        "total_days": total_days,
        "total_gross": round(total_gross, 2),
        "total_net": round(total_net, 2),
        "total_commissions": round(total_commissions, 2),
        "win_days": win_days,
        "loss_days": loss_days,
        "break_even_days": break_even_days,
        "win_rate": round(win_rate, 1),
        "avg_daily_net": round(avg_daily_net, 2),
        "best_day": round(best_day, 2),
        "worst_day": round(worst_day, 2),
        "current_streak": streak,
    }


# ---------------------------------------------------------------------------
# SQLite → Postgres one-time migration helper
# ---------------------------------------------------------------------------


def migrate_sqlite_to_postgres(
    sqlite_path: Optional[str] = None,
    pg_url: Optional[str] = None,
) -> dict:
    """One-time migration: copy all data from SQLite to Postgres.

    Call this manually when transitioning from SQLite to Postgres:
        from models import migrate_sqlite_to_postgres
        migrate_sqlite_to_postgres("data/futures_journal.db")

    Returns a dict with counts of migrated records.
    """
    import sqlite3 as _sqlite3

    src_path = sqlite_path or DB_PATH
    target_url = pg_url or DATABASE_URL

    if not target_url.startswith("postgresql"):
        raise ValueError("DATABASE_URL must point to a Postgres database")

    # Read from SQLite
    src_conn = _sqlite3.connect(src_path)
    src_conn.row_factory = _sqlite3.Row

    result = {"trades": 0, "journal": 0, "errors": []}

    # Check source tables
    tables = [
        r[0]
        for r in src_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    from sqlalchemy import create_engine, text

    pg_engine = create_engine(target_url)

    with pg_engine.connect() as pg_conn:
        # Migrate trades_v2
        if "trades_v2" in tables:
            rows = src_conn.execute("SELECT * FROM trades_v2").fetchall()
            for row in rows:
                d = dict(row)
                trade_id = d.pop("id", None)
                try:
                    cols = ", ".join(d.keys())
                    placeholders = ", ".join(f":{k}" for k in d.keys())
                    pg_conn.execute(
                        text(f"INSERT INTO trades_v2 ({cols}) VALUES ({placeholders})"),
                        d,
                    )
                    result["trades"] += 1
                except Exception as exc:
                    result["errors"].append(f"Trade: {exc}")

        # Migrate daily_journal
        if "daily_journal" in tables:
            rows = src_conn.execute("SELECT * FROM daily_journal").fetchall()
            for row in rows:
                d = dict(row)
                d.pop("id", None)
                try:
                    cols = ", ".join(d.keys())
                    placeholders = ", ".join(f":{k}" for k in d.keys())
                    pg_conn.execute(
                        text(
                            f"INSERT INTO daily_journal ({cols}) VALUES ({placeholders}) "
                            f"ON CONFLICT (trade_date) DO NOTHING"
                        ),
                        d,
                    )
                    result["journal"] += 1
                except Exception as exc:
                    result["errors"].append(f"Journal: {exc}")

        pg_conn.commit()

    src_conn.close()
    logger.info(
        "Migration complete: %d trades, %d journal entries, %d errors",
        result["trades"],
        result["journal"],
        len(result["errors"]),
    )
    return result
