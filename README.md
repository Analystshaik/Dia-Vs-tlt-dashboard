# Dow 30 vs TLT — Multi-Timeframe Leadership

This package provides two leadership boards for each timeframe:

1. Concurrent Leaders — best_lag = 0
2. True Followers — best_lag >= 1

Timeframes:
- Daily: 20D / 60D / 120D correlations, TLT leads 0–6 trading days
- Weekly: 13W / 26W / 52W correlations, TLT leads 0–6 weeks
- Monthly: 6M / 12M / 24M correlations, TLT leads 0–6 months

The Streamlit dashboard has Daily, Weekly, Monthly, and Cross-Timeframe Followers tabs.

Run:
python ibkr_backfill.py --no-refresh-holdings
python analyze_daily.py
python analyze_weekly.py
python analyze_monthly.py
streamlit run dashboard.py

Or use run_all.bat.


## Configuration hardening
This release adds explicit daily/weekly/monthly history settings and analyzer defaults, so older or partially merged config files do not fail with missing-key errors such as `min_history_days`.
"# Dia-Vs-tlt-dashboard" 
