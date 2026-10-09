from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from db import connect_db, init_schema
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

def to_weekly(df):
    out = []
    for symbol, g in df.groupby("symbol"):
        s = g.set_index("bar_date")["close"].sort_index()
        # Friday-ending week; last trading close in each week.
        w = s.resample("W-FRI").last().dropna()
        out.append(pd.DataFrame({"symbol": symbol, "week": w.index, "close": w.values}))
    return pd.concat(out, ignore_index=True)

def safe_corr(a, b, min_n=8):
    d = pd.concat([a,b], axis=1).dropna()
    if len(d) < min_n or d.iloc[:,0].std() == 0 or d.iloc[:,1].std() == 0:
        return np.nan
    return float(d.iloc[:,0].corr(d.iloc[:,1]))

def beta(y, x, min_n=8):
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
    mag = abs(score)
    if mag < 25 or not np.isfinite(best_corr) or abs(best_corr) < .20:
        return "LOW_OR_UNSTABLE"
    if score > 0:
        return "POSITIVE_TLT"
    return "INVERSE_TLT"

def calc_one(ticker, stock_px, tlt_px, cfg):
    px = pd.concat([stock_px.rename("stock"), tlt_px.rename("tlt")], axis=1).dropna()
    ret = px.pct_change().replace([np.inf,-np.inf], np.nan).dropna()
    if len(ret) < int(cfg.get("min_history_weeks", 60)):
        return None

    windows = cfg.get("weekly_correlation_windows", cfg.get("correlation_windows", [13,26,52]))
    corrs = {}
    for w in windows:
        corrs[w] = safe_corr(ret["stock"].tail(w), ret["tlt"].tail(w), min_n=max(8, int(w*.6)))

    beta26 = beta(ret["stock"].tail(26), ret["tlt"].tail(26), 12)

    lag_corr = {}
    lag_beta = {}
    # TLT lead k: corr(stock[t], TLT[t-k]).
    for k in cfg.get("weekly_lags", cfg.get("lags", [0,1,2,3,4,5,6])):
        x = ret["tlt"].shift(k)
        lag_corr[k] = safe_corr(ret["stock"].tail(52), x.tail(52), 20)
        lag_beta[k] = beta(ret["stock"].tail(52), x.tail(52), 20)

    valid_lags = {k:v for k,v in lag_corr.items() if np.isfinite(v)}
    best_lag = max(valid_lags, key=lambda k: abs(valid_lags[k])) if valid_lags else None
    best_corr = valid_lags.get(best_lag, np.nan) if best_lag is not None else np.nan
    best_beta = lag_beta.get(best_lag, np.nan) if best_lag is not None else np.nan

    vals = np.array([corrs.get(13,np.nan), corrs.get(26,np.nan), corrs.get(52,np.nan)], dtype=float)
    finite = vals[np.isfinite(vals)]
    stability = float(max(0.0, 1.0 - np.std(finite))) if len(finite) >= 2 else np.nan

    # Signed 0..100 score: sign = relationship direction; magnitude = confidence/strength.
    corr26 = corrs.get(26, np.nan)
    corr52 = corrs.get(52, np.nan)
    base_sign_source = best_corr if np.isfinite(best_corr) else corr26
    sign = np.sign(base_sign_source) if np.isfinite(base_sign_source) and base_sign_source != 0 else 0

    strength = (
        0.30 * abs(corr26 if np.isfinite(corr26) else 0) +
        0.20 * abs(corr52 if np.isfinite(corr52) else 0) +
        0.30 * abs(best_corr if np.isfinite(best_corr) else 0) +
        0.20 * (stability if np.isfinite(stability) else 0)
    )
    score = float(np.clip(sign * 100.0 * strength, -100, 100))

    sret, tret = ret["stock"], ret["tlt"]
    rs4 = compounded(sret,4) - compounded(tret,4)
    rs13 = compounded(sret,13) - compounded(tret,13)
    stock1 = compounded(sret,1)
    stock4 = compounded(sret,4)
    tlt1 = compounded(tret,1)
    tlt4 = compounded(tret,4)

    # Follower alert: non-zero best lead, strong historical lead relationship,
    # and TLT has made a material 1-week move while stock has not yet moved
    # proportionally in the historically implied direction.
    expected_direction = 0
    lag_gap = np.nan
    follower = 0
    if best_lag is not None and best_lag >= 1 and np.isfinite(best_corr) and np.isfinite(best_beta):
        source_move = tlt1
        expected = best_beta * source_move
        lag_gap = float(expected - stock1)
        if abs(best_corr) >= .45 and abs(source_move) >= .01 and abs(lag_gap) >= .0075:
            expected_direction = 1 if expected > 0 else (-1 if expected < 0 else 0)
            follower = 1

    row = {
        "asof_date": ret.index.max().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "weeks": int(len(ret)),
        "corr_13w": corrs.get(13),
        "corr_26w": corrs.get(26),
        "corr_52w": corrs.get(52),
        "beta_26w": beta26,
        **{f"lag{k}_corr": lag_corr.get(k) for k in range(7)},
        "best_lag": best_lag,
        "best_lag_corr": best_corr,
        "best_lag_beta": best_beta,
        "corr_stability": stability,
        "rs_4w": rs4,
        "rs_13w": rs13,
        "stock_ret_1w": stock1,
        "stock_ret_4w": stock4,
        "tlt_ret_1w": tlt1,
        "tlt_ret_4w": tlt4,
        "expected_direction": expected_direction,
        "lag_gap": lag_gap,
        "relationship_type": classify(score, best_corr),
        "relationship_score": score,
        "follower_alert": follower,
    }
    return row

def save(con, table, snapshot, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows calculated.")

    cols = [
        "asof_date","ticker","weeks","corr_13w","corr_26w","corr_52w","beta_26w",
        "lag0_corr","lag1_corr","lag2_corr","lag3_corr","lag4_corr","lag5_corr","lag6_corr",
        "best_lag","best_lag_corr","best_lag_beta","corr_stability","rs_4w","rs_13w",
        "stock_ret_1w","stock_ret_4w","tlt_ret_1w","tlt_ret_4w","expected_direction",
        "lag_gap","relationship_type","relationship_score","follower_alert"
    ]
    sql = f"""INSERT OR REPLACE INTO {table} ({','.join(cols)})
              VALUES ({','.join('?' for _ in cols)})"""
    vals = []
    for _, r in df[cols].iterrows():
        vals.append(tuple(None if pd.isna(v) else v.item() if hasattr(v, "item") else v for v in r))
    con.executemany(sql, vals)

    asof = df["asof_date"].max()
    snap = df[df["asof_date"] == asof]
    for _, r in snap.iterrows():
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
    init_schema(con, cfg["source_table"], cfg.get("weekly_analysis_table", cfg["analysis_table"]), cfg.get("weekly_snapshot_table", cfg["snapshot_table"]))
    holdings = load_holdings(args.config, refresh=False)
    universe = holdings["ticker"].tolist()
    symbols = [cfg["reference_symbol"]] + universe
    daily = load_daily(con, cfg["source_table"], symbols)
    weekly = to_weekly(daily)

    piv = weekly.pivot(index="week", columns="symbol", values="close").sort_index()
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
            print(f"SKIP {ticker}: insufficient weekly history")

    df = save(con, cfg.get("weekly_analysis_table", cfg["analysis_table"]), cfg.get("weekly_snapshot_table", cfg["snapshot_table"]), rows)
    con.close()

    print("\nTLT / Dow 30 stocks weekly relationship ranking")
    display = df.sort_values("relationship_score", key=lambda s: s.abs(), ascending=False)[
        ["ticker","relationship_score","best_lag","best_lag_corr","corr_13w","corr_26w","corr_52w","follower_alert"]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

if __name__ == "__main__":
    main()
