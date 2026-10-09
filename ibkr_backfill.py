from __future__ import annotations
import argparse, json, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from db import connect_db
from holdings_loader import load_holdings

@dataclass
class ReqState:
    ticker: str
    ibkr_symbol: str
    done: threading.Event = field(default_factory=threading.Event)
    rows: List[tuple] = field(default_factory=list)
    error: str | None = None

class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.next_id = 1
        self.req: Dict[int, ReqState] = {}
        self.ready = threading.Event()

    def nextValidId(self, orderId):
        self.next_id = max(self.next_id, int(orderId))
        self.ready.set()

    def historicalData(self, reqId, bar):
        st = self.req.get(reqId)
        if st:
            st.rows.append((st.ticker, str(bar.date), float(bar.open), float(bar.high),
                            float(bar.low), float(bar.close), float(bar.volume or 0)))

    def historicalDataEnd(self, reqId, start, end):
        st = self.req.get(reqId)
        if st:
            st.done.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in (2104,2106,2107,2108,2158):
            print(f"IBKR status {errorCode}: {errorString}")
            return
        print(f"IBKR error req={reqId} code={errorCode}: {errorString}")
        st = self.req.get(reqId)
        if st and errorCode not in (162,):
            st.error = f"{errorCode}: {errorString}"

def smart_stock(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c

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

def upsert_rows(con, table, rows):
    if not rows: return 0
    con.executemany(f"""
      INSERT INTO {table}(symbol,bar_date,open,high,low,close,volume,source)
      VALUES(?,?,?,?,?,?,?,'IBKR_ADJUSTED_LAST')
      ON CONFLICT(symbol,bar_date) DO UPDATE SET
       open=excluded.open,high=excluded.high,low=excluded.low,
       close=excluded.close,volume=excluded.volume,
       source=excluded.source,loaded_at=CURRENT_TIMESTAMP
    """, rows)
    con.commit()
    return len(rows)

def request_symbol(app, ticker, ibkr_symbol, cfg):
    rid = app.next_id
    app.next_id += 1
    st = ReqState(ticker=ticker, ibkr_symbol=ibkr_symbol)
    app.req[rid] = st
    app.reqHistoricalData(
        rid, smart_stock(ibkr_symbol), "",
        f"{int(cfg['history_years'])} Y", "1 day",
        cfg["ibkr"]["what_to_show"], int(cfg["ibkr"]["use_rth"]),
        1, False, []
    )
    if not st.done.wait(float(cfg["ibkr"]["request_timeout_seconds"])):
        app.cancelHistoricalData(rid)
        st.error = st.error or "timeout"
    return st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--no-refresh-holdings", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    holdings = load_holdings(args.config, refresh=False)
    if args.symbols:
        wanted = {x.upper() for x in args.symbols}
        holdings = holdings[holdings["ticker"].isin(wanted)]

    # Always include TLT as the reference series.
    records = [("TLT","TLT")] + list(holdings[["ticker","ibkr_symbol"]].itertuples(index=False, name=None))

    con = connect_db(cfg["database"])
    ensure_source_schema(con, cfg["source_table"])

    ib = cfg["ibkr"]
    app = IBKRApp()
    app.connect(ib["host"], int(ib["port"]), clientId=int(ib["client_id"]))
    th = threading.Thread(target=app.run, daemon=True); th.start()
    if not app.ready.wait(10):
        raise RuntimeError("IBKR did not return nextValidId. Check TWS/Gateway API settings.")

    failed = []
    print(f"Connected. Universe={cfg['universe']}; requests={len(records)} including TLT.")
    try:
        for i,(ticker,ibkr_symbol) in enumerate(records,1):
            last_error = None
            for attempt in range(int(ib["retry_count"])+1):
                print(f"[{i}/{len(records)}] {ticker} ({ibkr_symbol}) attempt {attempt+1}")
                st = request_symbol(app,ticker,ibkr_symbol,cfg)
                if st.rows:
                    n = upsert_rows(con,cfg["source_table"],st.rows)
                    print(f"  stored/checkpointed {n} daily bars")
                    last_error = None
                    break
                last_error = st.error or "no rows"
                if attempt < int(ib["retry_count"]):
                    time.sleep(3*(attempt+1))
            if last_error:
                failed.append((ticker,last_error))
                print(f"  FAILED {ticker}: {last_error}")
                if str(last_error).startswith("200:"):
                    print("  Non-fatal: IBKR could not resolve this cached constituent; analysis will skip it.")
            time.sleep(float(ib["pause_between_requests_seconds"]))
    finally:
        app.disconnect(); con.close()

    if failed:
        print("\nFailed symbols:")
        for t,e in failed: print(f"  {t}: {e}")
        print("Rerun the script; successfully downloaded symbols are already checkpointed in SQLite.")

if __name__ == "__main__":
    main()
