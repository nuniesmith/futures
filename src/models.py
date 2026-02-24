"""
Account configurations, contract specifications, asset mappings,
and SQLite trade helpers with OPEN/CLOSED status tracking.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "futures_journal.db")

# ---------------------------------------------------------------------------
# Account profiles – 50k is 1/3 of 150k, 100k is 2/3
# ---------------------------------------------------------------------------
ACCOUNT_PROFILES = {
    "50k": {
        "size": 50_000,
        "risk_pct": 0.01,
        "risk_dollars": 500,
        "max_contracts": 2,
        "soft_stop": -500,
        "hard_stop": -750,
        "eod_dd": 1_500,
        "label": "$50k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 1-2 contracts on $50k (25% rule). "
            "Daily Loss Removed: $1,500."
        ),
    },
    "100k": {
        "size": 100_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_000,
        "max_contracts": 3,
        "soft_stop": -1_000,
        "hard_stop": -1_500,
        "eod_dd": 3_000,
        "label": "$100k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 2-3 contracts on $100k (25% rule). "
            "Daily Loss Removed: $3,000."
        ),
    },
    "150k": {
        "size": 150_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_500,
        "max_contracts": 4,
        "soft_stop": -1_500,
        "hard_stop": -2_250,
        "eod_dd": 4_500,
        "label": "$150k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 3-4 contracts on $150k (25% rule). "
            "Daily Loss Removed: $4,500."
        ),
    },
}

# ---------------------------------------------------------------------------
# Contract specifications
# ---------------------------------------------------------------------------
CONTRACT_SPECS = {
    "Gold": {"ticker": "GC=F", "point": 100, "tick": 0.10, "margin": 11_000},
    "Silver": {"ticker": "SI=F", "point": 5_000, "tick": 0.005, "margin": 9_000},
    "Copper": {"ticker": "HG=F", "point": 250, "tick": 0.0005, "margin": 6_000},
    "Crude Oil": {"ticker": "CL=F", "point": 1_000, "tick": 0.01, "margin": 7_000},
    "S&P": {"ticker": "ES=F", "point": 50, "tick": 0.25, "margin": 12_000},
    "Nasdaq": {"ticker": "NQ=F", "point": 20, "tick": 0.25, "margin": 17_000},
}

# Convenience: name → ticker
ASSETS = {name: spec["ticker"] for name, spec in CONTRACT_SPECS.items()}

# Reverse lookup: ticker → name
TICKER_TO_NAME = {spec["ticker"]: name for name, spec in CONTRACT_SPECS.items()}

# ---------------------------------------------------------------------------
# Trade statuses
# ---------------------------------------------------------------------------
STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"
STATUS_CANCELLED = "CANCELLED"

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_SCHEMA_V2 = """
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


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Initialise the trades_v2 table, migrating from v1 if needed."""
    conn = _get_conn()

    # Check if new table already exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trades_v2'"
    )
    if cur.fetchone() is not None:
        conn.close()
        return

    # Create v2 schema
    conn.executescript(_SCHEMA_V2)

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
    cur = conn.execute(
        """INSERT INTO trades_v2
           (created_at, account_size, asset, direction, entry, sl, tp,
            contracts, status, strategy, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        ),
    )
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id  # type: ignore[return-value]


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

    close_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute(
        """UPDATE trades_v2
           SET status = ?, close_price = ?, close_time = ?, pnl = ?, rr = ?
           WHERE id = ?""",
        (STATUS_CLOSED, close_price, close_time, round(pnl, 2), round(rr, 2), trade_id),
    )
    conn.commit()

    result = dict(row)
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


def get_open_trades(account_size: Optional[int] = None) -> pd.DataFrame:
    """Return all OPEN trades, optionally filtered by account size."""
    conn = _get_conn()
    if account_size:
        df = pd.read_sql(
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY created_at DESC",
            conn,
            params=(STATUS_OPEN, account_size),
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY created_at DESC",
            conn,
            params=(STATUS_OPEN,),
        )
    conn.close()
    return df


def get_closed_trades(account_size: Optional[int] = None) -> pd.DataFrame:
    """Return all CLOSED trades for the journal."""
    conn = _get_conn()
    if account_size:
        df = pd.read_sql(
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY close_time DESC",
            conn,
            params=(STATUS_CLOSED, account_size),
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY close_time DESC",
            conn,
            params=(STATUS_CLOSED,),
        )
    conn.close()
    return df


def get_all_trades(account_size: Optional[int] = None) -> pd.DataFrame:
    """Return all trades regardless of status."""
    conn = _get_conn()
    if account_size:
        df = pd.read_sql(
            "SELECT * FROM trades_v2 WHERE account_size = ? ORDER BY created_at DESC",
            conn,
            params=(account_size,),
        )
    else:
        df = pd.read_sql("SELECT * FROM trades_v2 ORDER BY created_at DESC", conn)
    conn.close()
    return df


def get_today_pnl(account_size: Optional[int] = None) -> float:
    """Sum of realised P&L for trades closed today."""
    today = datetime.now().strftime("%Y-%m-%d")
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
    return float(row[0]) if row else 0.0


def get_today_trades(account_size: Optional[int] = None) -> pd.DataFrame:
    """Return all trades created or closed today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    if account_size:
        df = pd.read_sql(
            """SELECT * FROM trades_v2
               WHERE (created_at LIKE ? OR close_time LIKE ?)
                 AND account_size = ?
               ORDER BY created_at DESC""",
            conn,
            params=(f"{today}%", f"{today}%", account_size),
        )
    else:
        df = pd.read_sql(
            """SELECT * FROM trades_v2
               WHERE created_at LIKE ? OR close_time LIKE ?
               ORDER BY created_at DESC""",
            conn,
            params=(f"{today}%", f"{today}%"),
        )
    conn.close()
    return df


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
    """Calculate max contracts respecting risk-per-trade and account cap."""
    spec = CONTRACT_SPECS.get(asset)
    if spec is None:
        return 1
    risk_per_contract = abs(entry - sl) * spec["point"]
    if risk_per_contract <= 0:
        return 1
    raw = int(risk_dollars // risk_per_contract)
    return max(1, min(raw, hard_max))


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
