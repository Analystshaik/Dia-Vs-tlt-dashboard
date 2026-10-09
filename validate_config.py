from pathlib import Path
import json

cfg = json.loads(Path("config.json").read_text())

required = {
    "min_history_days": 140,
    "min_history_weeks": 60,
    "min_history_months": 30,
    "daily_correlation_windows": [20,60,120],
    "weekly_correlation_windows": [13,26,52],
    "monthly_correlation_windows": [6,12,24],
    "daily_lags": [0,1,2,3,4,5,6],
    "weekly_lags": [0,1,2,3,4,5,6],
    "monthly_lags": [0,1,2,3,4,5,6],
}
changed = False
for k,v in required.items():
    if k not in cfg:
        cfg[k] = v
        changed = True
if changed:
    Path("config.json").write_text(json.dumps(cfg, indent=2))
    print("config.json repaired.")
else:
    print("config.json already contains all required multi-timeframe settings.")
