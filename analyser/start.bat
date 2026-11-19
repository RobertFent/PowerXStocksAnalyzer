@echo off
REM StrategyAnalyser launcher (Windows): double-click to start. Sets up Python packages on first run.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3 is not installed. Get it from https://www.python.org/downloads/ ^(tick "Add Python to PATH"^) and run this again. & pause & exit /b 1)
if not exist .venv (echo First run: setting up ^(1-2 minutes^) ... & python -m venv .venv || (pause & exit /b 1))
call .venv\Scripts\activate.bat
if not exist .venv\deps_ok (pip install -q -r requirements.txt && echo ok > .venv\deps_ok)
echo Starting the Strategy Workbench ... keep this window open while you use it (Ctrl+C to stop).
python -m backtest.app
pause
