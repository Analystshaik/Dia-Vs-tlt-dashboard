from __future__ import annotations
import pandas as pd

DOW_30 = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GOOGL","GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","WMT"
]

def load_holdings(config_path="config.json", refresh=False):
    return pd.DataFrame({
        "ticker": DOW_30,
        "company": DOW_30,
        "ibkr_symbol": [x.replace("."," ") for x in DOW_30],
        "universe": "DOW30",
        "source_url": "fixed_dow30_universe"
    })
