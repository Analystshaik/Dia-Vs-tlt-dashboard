from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from db import connect_db
from holdings_loader import load_holdings

def load_daily(con, table, symbols):
    placeholders = ",".join("?" for _ in symbols)
    q = f"""SELECT symbol, bar_date, close
            FROM {table}
            WHERE symbol IN ({placeholders})
            ORDER BY symbol, bar_date"""
    df = pd.read_sql_query(q, con, params=symbols)
    if df.empty:
        raise RuntimeError(f"No price data found in {table}. Run ibkr_backfill.py first.")
    df["bar_date"] = pd.to_datetime(df["bar_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["bar_date","close"])

def safe_corr(a, b, min_n=20):
    d = pd.concat([a,b], axis=1).dropna()
    if len(d) < min_n or d.iloc[:,0].std() == 0 or d.iloc[:,1].std() == 0:
        return np.nan
    return float(d.iloc[:,0].corr(d.iloc[:,1]))

def beta(y, x, min_n=20):
    d = pd.concat([y,x], axis=1).dropna()
    if len(d) < min_n:
        return np.nan
    xv = d.iloc[:,1].var(ddof=1)
    if not np.isfinite(xv) or xv == 0:
        return np.nan
    return float(d.iloc[:,0].cov(d.iloc[:,1]) / xv)

def compounded(ret, n):
    x = ret.dropna().tail(n)
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)

def classify(score, best_corr):
    if not np.isfinite(score):
        return "INSUFFICIENT_DATA"
    if abs(score) < 25 or not np.isfinite(best_corr) or abs(best_corr) < .20:
        return "LOW_OR_UNSTABLE"
    return "POSITIVE_TLT" if score > 0 else "INVERSE_TLT"

def ensure_source_schema(con, table):
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table} (
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
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_symbol_date ON {table}(symbol, bar_date)")
    con.commit()

def _table_columns(con, table):
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}

def ensure_daily_schema(con, table, snapshot):
    # Version 1.0 could accidentally create this table with the weekly schema
    # because ibkr_backfill.py called the shared weekly init_schema(). Detect
    # that layout and repair it automatically.
    existing = _table_columns(con, table)
    required = {
        "asof_date","ticker","days","corr_20d","corr_60d","corr_120d","beta_60d",
        "lag0_corr","lag1_corr","lag2_corr","lag3_corr","lag4_corr","lag5_corr","lag6_corr",
        "best_lag","best_lag_corr","best_lag_beta","corr_stability","rs_5d","rs_20d",
        "stock_ret_1d","stock_ret_5d","tlt_ret_1d","tlt_ret_5d","expected_direction",
        "lag_gap","relationship_type","relationship_score","follower_alert"
    }

    if existing and not required.issubset(existing):
        # The malformed table has no valid daily rows because inserts into it
        # failed before completion. Rename it instead of silently deleting it.
        backup = table + "_legacy_bad_schema"
        con.execute(f"DROP TABLE IF EXISTS {backup}")
        con.execute(f"ALTER TABLE {table} RENAME TO {backup}")
        con.commit()
        print(f"Repaired incompatible table schema: renamed {table} -> {backup}")

    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        asof_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        days INTEGER NOT NULL,
        corr_20d REAL,
        corr_60d REAL,
        corr_120d REAL,
        beta_60d REAL,
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
        rs_5d REAL,
        rs_20d REAL,
        stock_ret_1d REAL,
        stock_ret_5d REAL,
        tlt_ret_1d REAL,
        tlt_ret_5d REAL,
        expected_direction INTEGER,
        lag_gap REAL,
        relationship_type TEXT,
        relationship_score REAL,
        follower_alert INTEGER DEFAULT 0,
        PRIMARY KEY(asof_date, ticker)
    )""")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_date_score ON {table}(asof_date, relationship_score DESC)")

    # Snapshot schema is compatible across weekly/daily versions, but keep it
    # explicit and self-contained.
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {snapshot} (
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

def calc_one(ticker, stock_px, tlt_px, cfg):
    px = pd.concat([stock_px.rename("stock"), tlt_px.rename("tlt")], axis=1).dropna()
    ret = px.pct_change().replace([np.inf,-np.inf], np.nan).dropna()
    if len(ret) < int(cfg.get("min_history_days", 140)):
        return None

    corrs = {}
    for w in cfg.get("daily_correlation_windows", cfg.get("correlation_windows", [20,60,120])):
        corrs[w] = safe_corr(ret["stock"].tail(w), ret["tlt"].tail(w), min_n=max(15, int(w*.6)))

    beta60 = beta(ret["stock"].tail(60), ret["tlt"].tail(60), 30)

    lag_corr, lag_beta = {}, {}
    for k in cfg.get("daily_lags", cfg.get("lags", [0,1,2,3,4,5,6])):
        x = ret["tlt"].shift(k)
        lag_corr[k] = safe_corr(ret["stock"].tail(120), x.tail(120), 40)
        lag_beta[k] = beta(ret["stock"].tail(120), x.tail(120), 40)

    valid = {k:v for k,v in lag_corr.items() if np.isfinite(v)}
    best_lag = max(valid, key=lambda k: abs(valid[k])) if valid else None
    best_corr = valid.get(best_lag, np.nan) if best_lag is not None else np.nan
    best_beta = lag_beta.get(best_lag, np.nan) if best_lag is not None else np.nan

    vals = np.array([corrs.get(20,np.nan), corrs.get(60,np.nan), corrs.get(120,np.nan)], dtype=float)
    finite = vals[np.isfinite(vals)]
    stability = float(max(0.0, 1.0 - np.std(finite))) if len(finite) >= 2 else np.nan

    corr60 = corrs.get(60, np.nan)
    corr120 = corrs.get(120, np.nan)
    sign_src = best_corr if np.isfinite(best_corr) else corr60
    sign = np.sign(sign_src) if np.isfinite(sign_src) and sign_src != 0 else 0
    strength = (
        0.30 * abs(corr60 if np.isfinite(corr60) else 0) +
        0.20 * abs(corr120 if np.isfinite(corr120) else 0) +
        0.30 * abs(best_corr if np.isfinite(best_corr) else 0) +
        0.20 * (stability if np.isfinite(stability) else 0)
    )
    score = float(np.clip(sign * 100.0 * strength, -100, 100))

    sret, tret = ret["stock"], ret["tlt"]
    rs5 = compounded(sret,5) - compounded(tret,5)
    rs20 = compounded(sret,20) - compounded(tret,20)
    stock1, stock5 = compounded(sret,1), compounded(sret,5)
    tlt1, tlt5 = compounded(tret,1), compounded(tret,5)

    expected_direction, lag_gap, follower = 0, np.nan, 0
    if best_lag is not None and best_lag >= 1 and np.isfinite(best_corr) and np.isfinite(best_beta):
        expected = best_beta * tlt1
        lag_gap = float(expected - stock1)
        if abs(best_corr) >= .40 and abs(tlt1) >= .005 and abs(lag_gap) >= .004:
            expected_direction = 1 if expected > 0 else (-1 if expected < 0 else 0)
            follower = 1

    return {
        "asof_date": ret.index.max().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "days": int(len(ret)),
        "corr_20d": corrs.get(20),
        "corr_60d": corrs.get(60),
        "corr_120d": corrs.get(120),
        "beta_60d": beta60,
        **{f"lag{k}_corr": lag_corr.get(k) for k in range(7)},
        "best_lag": best_lag,
        "best_lag_corr": best_corr,
        "best_lag_beta": best_beta,
        "corr_stability": stability,
        "rs_5d": rs5,
        "rs_20d": rs20,
        "stock_ret_1d": stock1,
        "stock_ret_5d": stock5,
        "tlt_ret_1d": tlt1,
        "tlt_ret_5d": tlt5,
        "expected_direction": expected_direction,
        "lag_gap": lag_gap,
        "relationship_type": classify(score, best_corr),
        "relationship_score": score,
        "follower_alert": follower,
    }

def save(con, table, snapshot, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows calculated.")
    cols = [
        "asof_date","ticker","days","corr_20d","corr_60d","corr_120d","beta_60d",
        "lag0_corr","lag1_corr","lag2_corr","lag3_corr","lag4_corr","lag5_corr","lag6_corr",
        "best_lag","best_lag_corr","best_lag_beta","corr_stability","rs_5d","rs_20d",
        "stock_ret_1d","stock_ret_5d","tlt_ret_1d","tlt_ret_5d","expected_direction",
        "lag_gap","relationship_type","relationship_score","follower_alert"
    ]
    sql = f"""INSERT OR REPLACE INTO {table} ({','.join(cols)})
              VALUES ({','.join('?' for _ in cols)})"""
    vals = []
    for _, r in df[cols].iterrows():
        vals.append(tuple(None if pd.isna(v) else v.item() if hasattr(v, "item") else v for v in r))
    con.executemany(sql, vals)

    asof = df["asof_date"].max()
    for _, r in df[df["asof_date"] == asof].iterrows():
        con.execute(f"""
          INSERT INTO {snapshot}(ticker,asof_date,best_lag,best_lag_corr,relationship_type,relationship_score,follower_alert)
          VALUES(?,?,?,?,?,?,?)
          ON CONFLICT(ticker) DO UPDATE SET
            asof_date=excluded.asof_date,best_lag=excluded.best_lag,
            best_lag_corr=excluded.best_lag_corr,relationship_type=excluded.relationship_type,
            relationship_score=excluded.relationship_score,follower_alert=excluded.follower_alert,
            updated_at=CURRENT_TIMESTAMP
        """, (
            r["ticker"], r["asof_date"],
            None if pd.isna(r["best_lag"]) else int(r["best_lag"]),
            None if pd.isna(r["best_lag_corr"]) else float(r["best_lag_corr"]),
            r["relationship_type"],
            None if pd.isna(r["relationship_score"]) else float(r["relationship_score"]),
            int(r["follower_alert"])
        ))
    con.commit()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    con = connect_db(cfg["database"])
    ensure_source_schema(con, cfg["source_table"])
    ensure_daily_schema(con, cfg.get("daily_analysis_table", cfg["analysis_table"]), cfg.get("daily_snapshot_table", cfg["snapshot_table"]))

    holdings = load_holdings(args.config, refresh=False)
    universe = holdings["ticker"].tolist()
    symbols = [cfg["reference_symbol"]] + universe
    daily = load_daily(con, cfg["source_table"], symbols)

    piv = daily.pivot(index="bar_date", columns="symbol", values="close").sort_index()
    ref = cfg["reference_symbol"]
    if ref not in piv.columns:
        raise RuntimeError(f"{ref} data missing.")

    rows = []
    for ticker in universe:
        if ticker not in piv.columns:
            print(f"SKIP {ticker}: no data")
            continue
        row = calc_one(ticker, piv[ticker], piv[ref], cfg)
        if row:
            rows.append(row)
        else:
            print(f"SKIP {ticker}: insufficient daily history")

    df = save(con, cfg.get("daily_analysis_table", cfg["analysis_table"]), cfg.get("daily_snapshot_table", cfg["snapshot_table"]), rows)
    con.close()

    print("\nTLT / Dow 30 stocks DAILY relationship ranking")
    print(df.sort_values("relationship_score", key=lambda s: s.abs(), ascending=False)[
        ["ticker","relationship_score","best_lag","best_lag_corr","corr_20d","corr_60d","corr_120d","follower_alert"]
    ].to_string(index=False, float_format=lambda x: f"{x: .3f}"))

if __name__ == "__main__":
    main()
