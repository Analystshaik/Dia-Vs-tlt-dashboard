from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from db import connect_db
from holdings_loader import load_holdings

def ensure_source_schema(con, table):
    con.execute(f'''CREATE TABLE IF NOT EXISTS {table} (
        symbol TEXT NOT NULL, bar_date TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
        source TEXT DEFAULT 'IBKR_ADJUSTED_LAST',
        loaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(symbol, bar_date))''')
    con.commit()

def ensure_schema(con, table, snapshot):
    con.execute(f'''CREATE TABLE IF NOT EXISTS {table} (
        asof_date TEXT NOT NULL, ticker TEXT NOT NULL, months INTEGER NOT NULL,
        corr_6m REAL, corr_12m REAL, corr_24m REAL, beta_12m REAL,
        lag0_corr REAL, lag1_corr REAL, lag2_corr REAL, lag3_corr REAL,
        lag4_corr REAL, lag5_corr REAL, lag6_corr REAL,
        best_lag INTEGER, best_lag_corr REAL, best_lag_beta REAL,
        corr_stability REAL, rs_3m REAL, rs_6m REAL,
        stock_ret_1m REAL, stock_ret_3m REAL, tlt_ret_1m REAL, tlt_ret_3m REAL,
        expected_direction INTEGER, lag_gap REAL, relationship_type TEXT,
        relationship_score REAL, follower_alert INTEGER DEFAULT 0,
        PRIMARY KEY(asof_date, ticker))''')
    con.execute(f'''CREATE TABLE IF NOT EXISTS {snapshot} (
        ticker TEXT PRIMARY KEY, asof_date TEXT NOT NULL,
        best_lag INTEGER, best_lag_corr REAL, relationship_type TEXT,
        relationship_score REAL, follower_alert INTEGER,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    con.commit()

def safe_corr(a,b,min_n):
    d=pd.concat([a,b],axis=1).dropna()
    if len(d)<min_n or d.iloc[:,0].std()==0 or d.iloc[:,1].std()==0:
        return np.nan
    return float(d.iloc[:,0].corr(d.iloc[:,1]))

def beta(y,x,min_n):
    d=pd.concat([y,x],axis=1).dropna()
    if len(d)<min_n: return np.nan
    v=d.iloc[:,1].var(ddof=1)
    if not np.isfinite(v) or v==0: return np.nan
    return float(d.iloc[:,0].cov(d.iloc[:,1])/v)

def compounded(r,n):
    x=r.dropna().tail(n)
    return np.nan if x.empty else float((1+x).prod()-1)

def calc_one(ticker, spx, tpx, cfg):
    px=pd.concat([spx.rename('stock'),tpx.rename('tlt')],axis=1).dropna()
    ret=px.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    if len(ret) < int(cfg.get("min_history_months", 30)): return None
    c6=safe_corr(ret.stock.tail(6),ret.tlt.tail(6),4)
    c12=safe_corr(ret.stock.tail(12),ret.tlt.tail(12),7)
    c24=safe_corr(ret.stock.tail(24),ret.tlt.tail(24),12)
    b12=beta(ret.stock.tail(12),ret.tlt.tail(12),7)
    lc,lb={},{}
    for k in range(7):
        x=ret.tlt.shift(k)
        lc[k]=safe_corr(ret.stock.tail(36),x.tail(36),18)
        lb[k]=beta(ret.stock.tail(36),x.tail(36),18)
    valid={k:v for k,v in lc.items() if np.isfinite(v)}
    best_lag=max(valid,key=lambda k:abs(valid[k])) if valid else None
    best_corr=valid.get(best_lag,np.nan) if best_lag is not None else np.nan
    best_beta=lb.get(best_lag,np.nan) if best_lag is not None else np.nan
    vals=np.array([c6,c12,c24],dtype=float); finite=vals[np.isfinite(vals)]
    stability=float(max(0,1-np.std(finite))) if len(finite)>=2 else np.nan
    sign_src=best_corr if np.isfinite(best_corr) else c12
    sign=np.sign(sign_src) if np.isfinite(sign_src) and sign_src!=0 else 0
    strength=.30*abs(c12 if np.isfinite(c12) else 0)+.20*abs(c24 if np.isfinite(c24) else 0)+.30*abs(best_corr if np.isfinite(best_corr) else 0)+.20*(stability if np.isfinite(stability) else 0)
    score=float(np.clip(sign*100*strength,-100,100))
    sr,tr=ret.stock,ret.tlt
    rs3=compounded(sr,3)-compounded(tr,3); rs6=compounded(sr,6)-compounded(tr,6)
    s1,s3=compounded(sr,1),compounded(sr,3); t1,t3=compounded(tr,1),compounded(tr,3)
    follower=0; gap=np.nan; expdir=0
    if best_lag is not None and best_lag>=1 and np.isfinite(best_corr) and np.isfinite(best_beta):
        expected=best_beta*t1; gap=float(expected-s1)
        if abs(best_corr)>=.40 and abs(t1)>=.02 and abs(gap)>=.015:
            follower=1; expdir=1 if expected>0 else (-1 if expected<0 else 0)
    rtype='LOW_OR_UNSTABLE'
    if np.isfinite(score) and np.isfinite(best_corr) and abs(score)>=25 and abs(best_corr)>=.20:
        rtype='POSITIVE_TLT' if score>0 else 'INVERSE_TLT'
    return {
        'asof_date':ret.index.max().strftime('%Y-%m-%d'),'ticker':ticker,'months':int(len(ret)),
        'corr_6m':c6,'corr_12m':c12,'corr_24m':c24,'beta_12m':b12,
        **{f'lag{k}_corr':lc.get(k) for k in range(7)},
        'best_lag':best_lag,'best_lag_corr':best_corr,'best_lag_beta':best_beta,
        'corr_stability':stability,'rs_3m':rs3,'rs_6m':rs6,
        'stock_ret_1m':s1,'stock_ret_3m':s3,'tlt_ret_1m':t1,'tlt_ret_3m':t3,
        'expected_direction':expdir,'lag_gap':gap,'relationship_type':rtype,
        'relationship_score':score,'follower_alert':follower}

def save(con,table,snapshot,rows):
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No monthly rows calculated.')
    cols=list(df.columns)
    sql=f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})"
    vals=[]
    for _,r in df.iterrows():
        vals.append(tuple(None if pd.isna(r[c]) else r[c].item() if hasattr(r[c],'item') else r[c] for c in cols))
    con.executemany(sql,vals)
    for _,r in df.iterrows():
        con.execute(f'''INSERT INTO {snapshot}
        (ticker,asof_date,best_lag,best_lag_corr,relationship_type,relationship_score,follower_alert)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(ticker) DO UPDATE SET
        asof_date=excluded.asof_date,best_lag=excluded.best_lag,best_lag_corr=excluded.best_lag_corr,
        relationship_type=excluded.relationship_type,relationship_score=excluded.relationship_score,
        follower_alert=excluded.follower_alert,updated_at=CURRENT_TIMESTAMP''',
        (r.ticker,r.asof_date,None if pd.isna(r.best_lag) else int(r.best_lag),
         None if pd.isna(r.best_lag_corr) else float(r.best_lag_corr),r.relationship_type,
         float(r.relationship_score),int(r.follower_alert)))
    con.commit(); return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.json'); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    con=connect_db(cfg['database']); ensure_source_schema(con,cfg['source_table'])
    ensure_schema(con,cfg['monthly_analysis_table'],cfg['monthly_snapshot_table'])
    h=load_holdings(args.config,refresh=False); universe=h['ticker'].tolist(); symbols=['TLT']+universe
    ph=','.join('?' for _ in symbols)
    d=pd.read_sql_query(f"SELECT symbol,bar_date,close FROM {cfg['source_table']} WHERE symbol IN ({ph}) ORDER BY symbol,bar_date",con,params=symbols)
    d['bar_date']=pd.to_datetime(d['bar_date'],errors='coerce'); d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna()
    pieces=[]
    for sym,g in d.groupby('symbol'):
        s=g.set_index('bar_date')['close'].sort_index().resample('ME').last().dropna()
        pieces.append(pd.DataFrame({'symbol':sym,'month':s.index,'close':s.values}))
    m=pd.concat(pieces,ignore_index=True); piv=m.pivot(index='month',columns='symbol',values='close').sort_index()
    rows=[]
    for t in universe:
        if t not in piv.columns: continue
        r=calc_one(t,piv[t],piv['TLT'],cfg)
        if r: rows.append(r)
    df=save(con,cfg['monthly_analysis_table'],cfg['monthly_snapshot_table'],rows); con.close()
    print(df.sort_values('relationship_score',key=lambda s:s.abs(),ascending=False)[['ticker','relationship_score','best_lag','best_lag_corr','corr_6m','corr_12m','corr_24m','follower_alert']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))

if __name__=='__main__':
    main()
