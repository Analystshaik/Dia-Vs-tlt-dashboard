from __future__ import annotations
import sqlite3
from pathlib import Path

def connect_db(path: str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=60000")
    return con

def init_schema(con: sqlite3.Connection, source_table: str, analysis_table: str, snapshot_table: str):
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {source_table} (
        symbol TEXT NOT NULL,
        bar_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL NOT NULL,
        volume REAL,
        source TEXT DEFAULT 'IBKR_ADJUSTED_LAST',
        loaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(symbol, bar_date)
    )""")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{source_table}_symbol_date ON {source_table}(symbol, bar_date)")

    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {analysis_table} (
        asof_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        weeks INTEGER NOT NULL,
        corr_13w REAL,
        corr_26w REAL,
        corr_52w REAL,
        beta_26w REAL,
        lag0_corr REAL,
        lag1_corr REAL,
        lag2_corr REAL,
        lag3_corr REAL,
        lag4_corr REAL,
        lag5_corr REAL,
        lag6_corr REAL,
        best_lag INTEGER,
        best_lag_corr REAL,
        best_lag_beta REAL,
        corr_stability REAL,
        rs_4w REAL,
        rs_13w REAL,
        stock_ret_1w REAL,
        stock_ret_4w REAL,
        tlt_ret_1w REAL,
        tlt_ret_4w REAL,
        expected_direction INTEGER,
        lag_gap REAL,
        relationship_type TEXT,
        relationship_score REAL,
        follower_alert INTEGER DEFAULT 0,
        PRIMARY KEY(asof_date, ticker)
    )""")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{analysis_table}_date_score ON {analysis_table}(asof_date, relationship_score DESC)")

    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {snapshot_table} (
        ticker TEXT PRIMARY KEY,
        asof_date TEXT NOT NULL,
        best_lag INTEGER,
        best_lag_corr REAL,
        relationship_type TEXT,
        relationship_score REAL,
        follower_alert INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
