# StrategyAnalyser

An analysis tool: it backtests trading rules over 10+ years and compares them honestly with simply holding the
index. Two kinds of rules: indicator-based buy and sell rules on S&P 500 stocks, and SPY call spread routines
(options). No programming needed.

## Quick start

You need Python 3.10 or newer ([python.org/downloads](https://www.python.org/downloads/); on Windows tick
*"Add Python to PATH"* in the installer).

1. **Double-click `start.command`** (Mac) or **`start.bat`** (Windows). The first run installs the packages
   (1–2 minutes); keep the window open while you work. *Mac: if the file "can't be opened", right-click → Open once.*
2. Your browser opens the **Strategy Workbench**. The switch at the top picks **Stocks** or **Options**; each side has
   numbered steps, every step is a button. Long steps run in the background and show under *Jobs* on the right.

**Stocks: buy & sell S&P 500 shares**

| Step | Button | What happens |
|---|---|---|
| 1 | **Download all price data** | ~14 years of daily prices for every stock that was in the S&P 500 since 2016, plus SPY (1–2 min). Later only *Add the latest days*. Backtests only buy a stock while it was in the index. |
| 2 | **Open the Planner** | describe your own rules (dip buying or trend following), save. Write the rules down before you look at any result. |
| 3 | **Report for this plan** / **Compare all plans with SPY** | one-page report (verdict, key numbers, growth of $10,000 vs SPY, drawdowns, year by year), or every saved plan on one page against SPY buy & hold and SPY timing |
| 4 | **Quick / Full** stress test | luck checks (1 min), or also neighbouring settings, costs and random entries (5 min) |
| 5 | **Start search** | hundreds of rule sets with walk-forward validation (5–15 min); then *Report of the best set* or *Save best set as a plan* |

**Options: SPY call spreads**

| Step | Button | What happens |
|---|---|---|
| 1 | **Download SPY + VIX** | SPY, VIX and T-bill history since 2010 (seconds). Spreads are priced with Black-Scholes from the VIX of the day; the payoff at expiry is exact. |
| 2 | **Backtest this setup** / **Save as a plan** | your routine: width, the debit you pay (the strikes are wherever that debit lands that day), expiry, cadence, close rules, stop, trend filter |
| 3 | **Report for this plan** / **Compare all plans with SPY** | the same report and comparison as for stock plans |
| 4 | **Try N setups** | every combination of the strikes, widths, expiries, rules, cadences and filters you type in; your own setup's rank is printed below the table |

**Reading a result:** the **test period** (2025 – today, never used to pick any rule) and the **verdict** (7 checks,
all must pass) are the honest numbers; everything else can look good by chance. Details and the math behind each
button: **[docs/GUIDE.md](docs/GUIDE.md)**. Terminal equivalents: `python3 -m backtest`.

## Main finding (21 October 2026)

Several thousand stock rule sets were tested (10 oversold indicators, momentum and breakouts, trend and market
filters, stops, exits), year by year on 2019–2024 and on 2025–2026, with point-in-time index membership and a
check for luck. **No stock rule set beats holding SPY**; the 200-day trend filter does the work, the indicators add
little. **The SPY call spread routine is the only plan that passes every check**, but its option prices are modelled
from the VIX and six months of real trading were flat, so it is not confirmed.

| 2016 – Oct 2026 | Return / year | Worst drop | Verdict |
|---|---|---|---|
| SPY buy & hold | 15.1% | −33.7% | – |
| SPY timing (hold SPY only above its 200-day average) | 11.4% | −20.7% | – |
| Dip-buying reference (`willr.json`) | 7.6% | −27.4% | 2 of 7, luck 5% |
| Momentum, best of 400 (`best_strategy.json`) | 7.6% walk-forward | −29.2% | luck 0% |
| RSI(4) rule book, best variant: deep below the 200-day average (`rsi4_deep.json`) | 8.6% | −30.7% | 3 of 7, luck 4% |
| **SPY call spread, $4.95 on $10, 2 monthly expirations (`bcs_495.json`)** | **23.8%** | **−22.2%** | **7 of 7, luck 93%**, prices modelled |

Full results and reasoning: **[docs/strategy-research.md](docs/strategy-research.md)**. Engine and research
commands: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## What's in here
| Folder | Contents |
|---|---|
| `strategies/` | saved plans: `willr.json`, `best_strategy.json`, `rsi4_*.json` (RSI(4) rule book variants), `bcs_*.json` (SPY call spread routine and three tuned variants), `candidate_breadth.json`, `coach_plan.json` (template), plus your own |
| `symbols/` | index lists and `sp500_membership.csv` (who was in the S&P 500 when) |
| `docs/` | guide, findings, technical notes |
| `backtest/` | the code (`python3 -m backtest.<command>`) |
| `output/` | reports and results |

*Backtests are not a promise. Prices of companies delisted after leaving the index are mostly unavailable, so stock
results are slightly optimistic; option prices are modelled, not quoted.*
