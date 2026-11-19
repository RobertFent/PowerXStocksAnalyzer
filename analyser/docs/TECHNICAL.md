# Technical notes

Searching rule sets, stress tests and how the engine works. Non-technical guide: [GUIDE.md](GUIDE.md).

```
backtest/
  fetch.py, symbols.py      data download (Yahoo Finance, Wikipedia index lists)
  features.py               indicators (oscillators on a 0..100 scale, low = oversold), momentum,
                            cross-sectional rank, rebalance days, point-in-time eligibility
  membership.py             S&P 500 membership history -> symbols/sp500_membership.csv
  significance.py           bootstrap excess vs SPY, deflated Sharpe, random-entry Monte Carlo
  strategy.py               StrategyParams: the rule set + plain-English describe()
  engine.py                 trade generation (numba) and portfolio simulation
  metrics.py, benchmarks.py CAGR, drawdown, Sharpe, Calmar; SPY buy & hold, SPY timing
  criteria.py               the 7 pass/fail checks used by the verdict
  app.py, workbench.py, jobs.py  web app: workbench (every step as a button, background subprocess jobs) + planner
  plan.py, report.py        terminal questionnaire, HTML report
  compare.py                multi-strategy comparison page
  run.py, analyze.py, tune.py, signals.py
  spreads.py                SPY bull call spreads: Black-Scholes marks with VIX as IV, exact payoff at expiry; --tune grid
  search_space.py           what tune searches
strategies/                 rule sets as JSON
tests/                      python3 tests/test_backtest.py (no network)
```

## Research commands
All commands run from the repo root and write to `output/<command>_<timestamp>/`. The Workbench (`backtest.app`)
runs the same commands as background jobs.

| Command | What it does |
|---|---|
| `python3 -m backtest.fetch --historical` / `--update` | ~14 years of adjusted daily OHLCV for past and present S&P 500 members + SPY/QQQ into `data/` |
| `python3 -m backtest.report --params strategies/<plan>.json` | one-page HTML report with verdict; `backtest.compare` puts every plan on one page |
| `python3 -m backtest.analyze --params <plan>` (`--quick` = verdict and luck only) | stress test: verdict, bootstrap and deflated Sharpe, one parameter changed at a time, costs, ranked vs random slot choice, signal vs random entry, regimes |
| `python3 -m backtest.tune --space default --metric calmar --n-iter 400` | random/grid search with walk-forward validation (train 3y, test 1y), then one run on the holdout; `--only key=v1,v2` narrows a space |
| `python3 -m backtest.run --params <plan>` | text backtest with trades.csv and equity.csv |
| `python3 -m backtest.signals --params <plan> --holding AAPL` | today's entry candidates and exit signals for a plan |
| `python3 -m backtest.spreads --fetch` / `--params strategies/bcs_495.json` / `--tune --grid dte=30,60` / `--save "name"` | SPY call spreads: data, report, variants, save a routine as a plan |
| `python3 tests/test_backtest.py` | engine sanity checks, no network |

**Search spaces** live in `backtest/search_space.py`: `default` (all oscillators), `indicators` (only MFI, Bollinger %B, Connors RSI, Ultimate Oscillator, MACD and CCI), `breadth` (only buy when many stocks are oversold at once, with fewer slots) and `exits` (trailing stops, close-above-yesterday's-high exit). Narrow or extend any space with `--only key=v1,v2`, e.g. `--only entry_indicator=willr max_positions=5`.

**Rules** are `StrategyParams` in `backtest/strategy.py`; `describe()` prints them in plain English. Entry oscillators: RSI, Williams %R, Stochastic, IBS, MFI, Bollinger %B, Connors RSI, Ultimate Oscillator, MACD histogram and CCI. MACD and CCI are converted to a percent rank over the last `entry_length` days so they fit the same scale. Formulas are in `backtest/features.py`. Oscillators use a 0..100 scale where low = oversold, so Williams %R −90 is `entry_threshold=10`.

**Execution model:** the signal is taken at the close and filled at the next open. Stops fill at the stop price, or at the open if the price gaps below it. Costs are 5 bps per side. Positions are equal-weight slots.

### Before trusting a rule set
- **Walk-forward:** for a tuned set, the "Stitched OOS" result in the tune report must be at least as good as SPY and SPY timing. This is the honest number, not the in-sample top score.
- **Verdict:** `backtest.analyze` must pass all criteria in `backtest/criteria.py`. Set the thresholds before looking at results.
- **Neighborhood:** values next to the chosen parameters must perform similarly (a plateau, not a spike).
- **Caveat:** membership is point-in-time, but prices of companies delisted after leaving the index are mostly unavailable (about 88% of member-days covered), so results are still slightly optimistic.

**Benchmarks:** "SPY timing" means holding SPY only while it is above its 200-day average, otherwise cash. A stock-picking strategy that can't beat it gets its edge from the trend filter, not from picking stocks.

## Strategy families
- **Mean reversion:** `entry_indicator` = oscillator, entry when it is below `entry_threshold` (first day or any day).
- **Momentum:** `entry_indicator='rs'` ranks all eligible stocks by `momentum(entry_length, skip_days)` on rebalance
  days and buys the top `entry_threshold`; `'breakout'` buys a close above the previous `entry_length`-day high.
  Exits: `rank` (dropped below rank `exit_threshold`), `ma`, `channel`, trailing stop, time.
- **Membership:** `FeatureStore.eligible` marks rows where the ticker was in the index; entries, breadth and
  momentum ranks use only eligible rows. Positions are held through a removal until the exit rule fires.

## Rule-book mechanics (RSI(4) classic)
`universe` ('sp100+nasdaq100' etc., today's members for anything but sp500), `max_price`, `min_history_days`,
`max_new_per_day` (portfolio level), `rank_by='symbol_roi'` (free slots go to symbols whose earlier completed
trades had the best average return; no look-ahead), `obe_min_gap` (entry only if the open is that far below the
signal close), `add_drop_pct` (second equal block at entry × (1 − pct), filled at the open on a gap, skipped within
`add_skip_near_stop` of the stop; the portfolio funds it from cash if available).
The exit rule of the rule book is not in the excerpt we have; `strategies/rsi4_*.json` use a placeholder.

## Significance
`backtest.significance`: `bootstrap_excess` (stationary block bootstrap, 20-day blocks, of daily strategy − SPY),
`deflated_sharpe` (PSR/DSR with skew and kurtosis; `trials` sets the expected maximum Sharpe of zero-skill
strategies), `random_entry_test` (same trades, random entry days among days that pass the same filters, run
through the portfolio simulation). The verdict uses the smaller of the bootstrap and the deflated probability.

## Option spreads (`backtest/spreads.py`)
Black-Scholes marks with the VIX of the day as implied vol (no skew), the 13-week T-bill as rate and a 1.4% dividend
yield; payoff at expiry exact from SPY's unadjusted close. `strike_mode='debit'` picks, per entry, the whole-dollar
strikes whose model price is closest to `fixed_debit`; `'offset'` places the long strike at `offset` from the price.
Cadence `expiry` = third Friday with expiry on the monthly expiration `dte/30` months out. Early closes (target,
stop) use the model marks; when the debit paid differs from the model price, the difference fades linearly to expiry.
Sizing: `risk` of the account across the spreads that overlap. `--tune` enumerates `GRID` (editable with `--grid`).

## How the engine works
- **Signals** are computed at the close of day t from data up to t only. A test checks that changing future prices never changes past trades.
- **Trades:** one position per stock at a time. Exits are checked in this order: exits scheduled for the open, entries at the open, intraday stop or trailing stop, then close-based exits for the next open.
- **Portfolio:** equal-weight slots, marked to market daily. When there are more signals than slots, `rank_by` decides which get them. Positions still open at the end of a window are closed at that day's close.
- **Walk-forward** (`tune`): train 3 years, test the next year, roll forward. The stitched test years are the honest estimate. The `--holdout-start` period is evaluated once at the end.
