"""
FastAPI server for receiving trade data from NinjaTrader.

Run alongside the Streamlit dashboard:
    python api_server.py

Listens on port 8000. NinjaTrader sends POST /log_trade with trade details,
which are written to the same SQLite journal used by app.py.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from datetime import datetime

app = FastAPI(title="Futures Dashboard Trade API")

DB_PATH = "futures_journal.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            date TEXT,
            asset TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            contracts INTEGER,
            exit_price REAL,
            pnl REAL,
            rr REAL,
            notes TEXT,
            strategy TEXT
        )"""
    )
    conn.commit()
    conn.close()


_init_db()


class Trade(BaseModel):
    asset: str
    direction: str
    entry: float
    exit_price: float
    contracts: int
    pnl: float
    strategy: str = ""
    notes: str = ""


@app.post("/log_trade")
def log_trade(trade: Trade):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO trades
           (date, asset, direction, entry, sl, tp, contracts, exit_price, pnl, rr, notes, strategy)
           VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, 0, ?, ?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            trade.asset,
            trade.direction,
            trade.entry,
            trade.contracts,
            trade.exit_price,
            trade.pnl,
            trade.notes,
            trade.strategy,
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "logged", "asset": trade.asset, "pnl": trade.pnl}


@app.get("/trades")
def get_trades(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT * FROM trades ORDER BY date DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
