# Strategy research log

Setup for every run: sp500 (today's members), 2016-01-01 to 2024-12-31 for tuning, 2025-01-01 to
2026-10-16 as holdout, 5 bps per side, next-open fills. Walk-forward picks the best rule set on 3 years
and trades it in the following year (2019–2024). Full reports are in `output/` (git-ignored); re-run them
with the commands below.

## Summary (2026-10-19)

| Run | Walk-forward 2019–24 (CAGR / max DD / Sharpe) | Best set, holdout 2025+ | Median holdout of top 10 |
|---|---|---|---|
| SPY buy & hold | 17.1% / −33.7% / 0.90 | 17.4% / −18.8% | – |
| `--space default` (no constraints) | 10.9% / −17.1% / 0.78 | −2.1% | – |
| `--space default --only market_ma=200 trend_ma=100,200` | 12.4% / −21.5% / 0.92 | **+8.0% / −12.8%** | 5.8% |
| `--space breadth` | 6.2% / −39.3% / 0.42 | +18.3% / −16.8% | 8.1% |
| `--space exits` | 19.8% / −12.9% / 1.22 * | −5.4% / −26.3% | −0.2% |

\* The `exits` space was narrowed using what earlier 2016–2024 runs showed, so its walk-forward result is optimistic. The holdout shows it doesn't hold up.

**Current rule set** (`strategies/current.json`): Williams %R(10) below −90 (any close),
Close > SMA200, SPY > SMA200. Exit when Williams %R(10) goes above −60, at a 4×ADR(14) stop, or after
20 days. 10 slots, most oversold signals first.
2016–2024: 18.2% CAGR, −18.1% DD, Sharpe 1.28. Realistic expectation: about 10–14% CAGR with a drawdown well
below SPY's. It does not beat SPY on return in 2025–26.

**Candidate** (`strategies/candidate_breadth.json`): RSI(5) below 30, only when at least 15% of sp500 is oversold the
same day, 5 slots. Good holdout, but tuning in this space was unstable. Paper-trade it first.

## What we learned
1. **The trend filter (Close > SMA200) is what matters.** Without it: 8% CAGR and −37% DD instead of 18% and −18%.
   The SPY>SMA200 filter and stop losses add little.
2. **A single-stock oversold signal has about zero edge on its own.** Signal trades average +0.49%, the same as
   random entries in the same stock, year and holding time. The signal raises the win rate (69% vs 56%)
   but fattens the left tail. Its edge per trade has been slightly negative since 2022.
3. **The edge sits on broad pullbacks** (more than 60 stocks oversold at once: +0.5% over random, 76% win rate) and
   in high-volatility names. Filling slots with the most oversold signals beats random choice (18% vs 12% CAGR).
4. **Breadth-only trading** (step 1) improves each trade but cuts exposure to 15–45%. Tuning in that
   space overfits: the parameter choices jump from fold to fold, and the out-of-sample drawdown is −39%.
5. **Alternative exits** (step 2): tight trailing stops (1.5–2×ADR) are clearly worse. A close above yesterday's
   high is worse than the oscillator exit. "No exit signal + time stop" is about equal to the oscillator exit.
   No exit change improved the holdout.
6. **Costs matter:** at 20 bps per side the current set falls from 18% to 7% CAGR.
7. **Other indices are weaker:** the current set makes 5–7% CAGR on nasdaq100 and sp100.

## Next ideas (in order)
1. **Paper-trade** `current.json` and `candidate_breadth.json` side by side with `backtest.signals` and compare
   with the backtest every month.
2. **Sector cap:** signals cluster in one sector (e.g. 2026-10-16: almost only tech). Add max N positions per sector
   (needs sector data in `symbols/`).
3. **Combine both:** current rule set with 10 slots, plus bigger position sizes on broad-pullback days (breadth ≥ 15%).
4. **Fix survivorship bias** with historical index membership (point-in-time constituents), then re-run all of the above.
5. **Volatility-adjusted position size** (risk a fixed % per trade using ADR) instead of equal weight.

## Reproduce
```sh
python3 -m backtest.fetch
python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600
python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600 --only market_ma=200 trend_ma=100,200
python3 -m backtest.tune --space breadth --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 500
python3 -m backtest.tune --space exits   --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600
python3 -m backtest.analyze --params strategies/current.json
```
Random search uses `--seed 42`, so runs are repeatable as long as the data is unchanged.
