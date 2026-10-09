@echo off
cd /d %~dp0
python validate_config.py
if errorlevel 1 goto :error
python ibkr_backfill.py
if errorlevel 1 goto :error
python analyze_daily.py
if errorlevel 1 goto :error
python analyze_weekly.py
if errorlevel 1 goto :error
python analyze_monthly.py
if errorlevel 1 goto :error
echo.
echo Daily, Weekly and Monthly analyses complete.
echo streamlit run dashboard.py
goto :eof
:error
echo.
echo ERROR - see messages above.
exit /b 1
