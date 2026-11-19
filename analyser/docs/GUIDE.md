# Guide

The tool answers one question: **would these rules have made money, and would they have beaten simply holding
the index?** Everything runs on your own computer from the **Workbench** (double-click `start.command` or
`start.bat`; the README has the step tables). This guide covers what the Workbench does not say: how to test
fairly, how to read a report, and the math behind each button.

## 1. Setup by hand (if the launcher does not work)

Install Python 3.10+ ([python.org/downloads](https://www.python.org/downloads/); Windows: tick *"Add Python to PATH"*),
open a Terminal in this folder (Mac: right-click the folder → *Services* → *New Terminal at Folder*; Windows: type
`cmd` in the folder's address bar) and run:
```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m backtest.app            # opens the Workbench; python3 -m backtest lists every command
```
Next time only the `source` line is needed.

## 2. Testing fairly

- **Write the plan down before you test it, then test it once.** With enough tweaking every plan looks great on
  past data and then fails with real money; that is the main lesson of this research.
- **Compare variants only if you decided them all up front** (RSI(2), RSI(4) and RSI(7), say) and report every one.
  Don't keep only the best.
- **The test period** (2025 until today) was never used to pick any rule. It is the honest number in every report.
- **A search finds luck.** The more rule sets the computer tries, the better the best one looks by chance; the
  stress test's "variants tried" field raises the bar accordingly.

The Planner (Stocks, step 2) asks for a name, when to buy (*buy the dip*: an indicator such as RSI, Williams %R,
Stochastic, Connors RSI or CCI, its length and oversold level, first day or any day; *follow the trend*: the top N
stocks by 3–12-month performance, or breakouts), filters (the stock above or deep below its 200-day average, SPY above
its 200-day average), when to sell (indicator recovers, close above yesterday's high, stop loss, trailing stop,
maximum days) and the number of positions. The box on the right shows the rules in plain English while you type.
Option routines are built in the Options side, step 2.

## 3. How to read a report

| In the report | Meaning | Good when |
|---|---|---|
| Return per year | average yearly growth after costs | higher than SPY, or close to it with smaller drops |
| Worst drop | biggest fall from a previous high (−20% = $10,000 became $8,000) | clearly smaller than SPY's |
| Sharpe | return per unit of ups and downs (SPY ≈ 0.9) | above 1.1 |
| Calmar | return per year ÷ worst drop | higher than SPY timing |
| SPY timing | hold SPY only while it's above its 200-day average | the strategy should beat this, otherwise the stock picking adds nothing |
| Development period | the years used to find the rules | always looks good, so don't trust it alone |
| **Test period** | data the rules have never seen | **this is the honest number** |
| **Verdict** | 7 checks agreed on before testing | **all 7 must pass** |
| Unlikely to be luck | probability that the edge over SPY is more than chance; the more variants you tried, the higher the bar | at least 90% |
| Stitched OOS (searches) | rules re-picked each year and traded the next | compare with the SPY lines below it, not with the in-sample score |

## 4. Good to know

- **Survivorship bias:** the backtest knows which stocks were in the S&P 500 on each day and only buys those.
  Prices of companies delisted after leaving the index are mostly unavailable, so results are still slightly
  optimistic (*Check history coverage* prints how much is covered).
- **Costs:** 0.05% per buy or sell for stocks; $0.05 slippage per side and $0.65 per contract and leg for spreads.
  The verdict also checks double costs.
- **Long only, daily bars:** the tool only buys, never shorts; a signal at the close is filled at the next open.
- **Option prices are modelled** (Black-Scholes from the VIX of the day, the T-bill rate and SPY's dividend yield);
  only the payoff at expiry is exact. There is no volatility skew, so the model is a little kind to call spreads.
- **Problems:** error messages usually say what is missing, e.g. "Run python3 -m backtest.fetch first". Otherwise ask Robert.

## 5. What happens behind each button

All steps share one engine: for every stock and every day the rules say *buy* or *sell*, a signal at the close is
executed at the next day's open, 0.05% cost per buy and per sell, 10% of the account per position, cash earns nothing.
A stock is only tradeable on days it was actually in the index. Everything below is computed on that simulated account.

### Stocks, step 1: price data
Downloads daily open / high / low / close / volume from Yahoo Finance for every ticker that was in the S&P 500 since
2016 (past members included, from the published membership history) plus SPY. Prices are adjusted for splits and
dividends. Indicators (RSI, Williams %R, moving averages, momentum ranks…) are computed from these once and cached.

### Stocks, step 2: build your own plan
No math; it writes the rules to a file in `strategies/`. The plain-English box is the same text the report shows.

### Step 3 (both sides): reports
The account's daily value over the whole period is the basis for every number:

| Number | How it is computed |
|---|---|
| Return per year (CAGR) | the constant yearly growth rate that turns the start value into the end value |
| Worst drop (max drawdown) | the largest fall from any previous high of the account, in percent |
| Volatility | standard deviation of daily returns × √252 (yearly) |
| Sharpe | average daily return ÷ its standard deviation, × √252. Return per unit of wobble; SPY ≈ 0.9 |
| Sortino | like Sharpe, but only down days count as wobble |
| Calmar | return per year ÷ worst drop. Return per unit of pain |
| Profit factor | sum of all winning trades ÷ sum of all losing trades |
| Win rate, average trade, days held | straight from the trade list |
| SPY buy & hold | the same account holding only SPY |
| SPY timing | holding SPY only on days it closed above its 200-day average, otherwise cash |
| Verdict | the 7 checks of section 3, each a comparison of these numbers between the plan, SPY and SPY timing; the luck check is the quick stress test (below) |

Development period = backtest start to the test-period start; test period = test-period start to today, computed
separately with the same rules. *Compare* runs the same thing for every plan and adds the correlation of daily
returns between plans (1 = move together, 0 = unrelated).

### Stocks, step 4: stress test
Asks whether the report could be a fluke. **Quick** runs the first two, **Full** all four:

1. **Bootstrap.** Take the plan's daily returns minus SPY's. Build 2,000 alternative histories by re-shuffling
   20-day blocks of those excess returns, and count how often the average stays above zero. 95% means the
   edge over SPY survived in 95% of re-shuffles.
2. **Deflated Sharpe ratio** (Bailey & López de Prado). The probability that the excess Sharpe is above zero,
   after correcting for the number of variants tried: with 400 variants the best one is expected to look good by
   chance alone, so the bar rises with the "variants tried" field. The verdict's *Unlikely to be luck* is the
   smaller of 1 and 2 and must be at least 90%.
3. **Neighbourhood.** Change one setting at a time (RSI length 4 → 3 or 5, threshold 15 → 10 or 20, stop
   3 → 4 × ADR…) and re-run. A real edge degrades gently; a lucky one collapses next door.
4. **Costs and random entries.** Re-run at double costs (0.10% per side). Then keep every trade's stock, year and
   holding time but move its entry to a random day that passes the same filters (trend, market, liquidity),
   30 times. If random days do as well, the indicator's timing adds nothing; the filters did the work.

### Stocks, step 5: search
Tries many rule sets and tells you how good "pick the best" really is:

1. Draw N rule sets at random from the chosen space (e.g. RSI/Williams/Stochastic lengths × thresholds × filters × exits).
2. **Walk-forward.** Split the development period into folds: 3 years to pick, the following 1 year to trade, then
   move forward a year. In each fold every rule set is backtested on the 3 years and ranked by the chosen metric;
   the winner is run on the next year it has never seen. Chaining those unseen years gives the **stitched OOS**
   line: what you would have earned by re-picking the best rules every year. This, not the in-sample score, is
   the honest result of the search.
3. **Best set.** The rule set with the best score over the whole development period is saved as
   `best_params.json` and run once on the test period. Its development-period numbers are in-sample and flattered
   by the search; its test-period and stitched-OOS numbers are not.
4. Constraints: at least 20 trades per window and a worst drop no deeper than 35%, otherwise the set is discarded.

### Options, steps 1, 2 and 4: SPY call spreads
No option quotes exist for 14 years, so each spread is priced with Black-Scholes from the VIX of the day (plus the
T-bill rate and SPY's dividend yield); the payoff at expiry is exact from SPY's unadjusted close. The strikes are
chosen the way you trade: *the spread that costs the debit you pay* (4.95 on a $10 width, say) lands wherever the
market prices that on the day; alternatively a fixed distance from the price. *Save as a plan* writes the setup to
`strategies/` (`"type": "spreads"`); step 3 then reports and compares it like a stock plan, with the same verdict and
luck check (the stock stress test does not apply). The text backtest also shows per-spread statistics, the routine at
2 / 5 / 10 / 20% of the account at risk and the result of one constant contract. *Try N setups* backtests every
combination you type in (strikes as `debit:0.5` or `offset:0.017`, widths, expiries, close rules, stops, cadence,
trend filter), or a random sample when you cap the number, ranked on the development period with the test period
next to it. The model's weak spot is any early close (target or stop): it assumes the spread trades at model value,
so a routine held to expiry rests on the model only for its strikes.

**Rank by** (search and spread variants) decides what "best" means: *Calmar* = most return per unit of worst drop (favours
smooth equity, the default), *Sharpe* = most return per unit of daily wobble, *Return* = highest growth regardless
of drops (tends to pick wild sets). The search also records the number of variants tried, which the stress test
uses to deflate the luck check.
