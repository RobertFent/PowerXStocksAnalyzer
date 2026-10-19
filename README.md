# PowerXStocksAnalyzer

Python backend of the stock screener:

1. **Daily scraper** (`indicators.py`): downloads the last ~300 days for sp100 / nasdaq100 / sp500,
   computes the indicators, upserts the last 7 days into Postgres (`stock_data`) and revalidates the
   screener cache.
2. **Strategy research** (`backtest/`): rule-based long-only swing strategies. You can backtest them,
   tune them with walk-forward validation, stress-test them, and print daily signals.

```
indicators.py          daily scraper (cron / docker)
utils.py               refresh index symbol lists in symbols/ from Wikipedia
symbols/               index constituents (input for scraper + backtests)
database/schema.sql    Postgres table used by the scraper
docker/                container for the daily scraper
backtest/              research package (python3 -m backtest.<command>)
strategies/            chosen rule sets as JSON (current.json = the one to trade)
docs/                  research log and findings
tests/                 engine sanity tests (no network)
data/, output/         downloaded history / run results (git-ignored)
```

## Setup
```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env      # DATABASE_URL, REVALIDATE_SECRET, STOCK_SCREENER_REVALIDATE_URL
```

## Daily scraper
```sh
python3 indicators.py                                          # local
docker compose -f docker/indicators.docker-compose.yml up      # docker
# cron: 0 7 * * * cd /home/robot/repos/PowerXStocksAnalyzer && /usr/bin/docker compose -f docker/indicators.docker-compose.yml up
python3 utils.py                                               # refresh symbols/*.csv when index members change
```

## Strategy research workflow
All commands run from the repo root. Each command prints a report and writes it to `output/<command>_<timestamp>/`.

| Step | Command | What it does |
|---|---|---|
| 1. Data | `python3 -m backtest.fetch` / `--update` | ~14 years of adjusted daily OHLCV + SPY/QQQ into `data/` |
| 2. Tune | `python3 -m backtest.tune --space default --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar` | random/grid search with walk-forward validation (train 3y, test 1y), then one run on the untouched holdout |
| 3. Analyze | `python3 -m backtest.analyze --params output/tuning_<ts>/best_params.json` | changes one parameter at a time, cost sensitivity, ranked vs random slot choice, signal vs random entry, regimes, other indices |
| 4. Backtest | `python3 -m backtest.run --params strategies/current.json --start 2016-01-01` | full report, trades.csv and equity.csv vs SPY |
| 5. Adopt | copy a `best_params.json` to `strategies/current.json` | only if it passes the checks below |
| 6. Trade | `python3 -m backtest.fetch --update && python3 -m backtest.signals --holding AAPL ...` | entry candidates for the next open and exit signals for your holdings |
| Tests | `python3 tests/test_backtest.py` | engine sanity checks |

**Search spaces** live in `backtest/search_space.py`: `default`, `breadth` (only buy when many stocks are oversold at once, with fewer slots) and `exits` (trailing stops, close-above-yesterday's-high exit). Narrow or extend any space with `--only key=v1,v2`, e.g. `--only entry_indicator=willr max_positions=5`.

**Rules** are `StrategyParams` in `backtest/strategy.py`; `describe()` prints them in plain English. Oscillators use a 0..100 scale where low = oversold, so Williams %R −90 is `entry_threshold=10`.

**Execution model:** the signal is taken at the close and filled at the next open. Stops fill at the stop price, or at the open if the price gaps below it. Costs are 5 bps per side. Positions are equal-weight slots.

### Before adopting a rule set
- Walk-forward "Stitched OOS" Sharpe and drawdown are at least as good as SPY's. This is the honest number, not the in-sample top score.
- The holdout result is positive. Look at it once and don't re-tune afterwards.
- In `analyze`, neighboring parameter values perform similarly (a plateau, not a spike), and results survive 10 bps costs.
- Caveat: `symbols/` holds today's index members, so stocks that dropped out are missing (survivorship bias). Expect live results to be lower.

Current findings and open ideas: [docs/strategy-research.md](docs/strategy-research.md).
