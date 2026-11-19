#!/bin/bash
# StrategyAnalyser launcher (Mac): double-click to start. Sets up Python packages on first run.
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close."; exit 1
fi
if [ ! -d .venv ]; then
  echo "First run: setting up (1-2 minutes) ..."
  python3 -m venv .venv || { read -r -p "Could not create the environment. Press Enter to close."; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f .venv/.deps_ok ] || [ requirements.txt -nt .venv/.deps_ok ]; then
  pip install -q -r requirements.txt && touch .venv/.deps_ok
fi
echo "Starting the Strategy Workbench ... keep this window open while you use it (Ctrl+C to stop)."
python3 -m backtest.app
