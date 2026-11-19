# Strategy research: findings and conclusion

*Status: 21 October 2026 (sections 1–6 are the research log; section 7 has the current comparison and conclusion)*

## Question
Is there an indicator-based, long-only swing strategy with clear entry and exit rules on
S&P 500 stocks that is clearly better than simply holding SPY?

## Setup
- **Universe:** S&P 500 (today's members, about 495 stocks), daily bars, adjusted for splits and dividends.
- **Periods:**
  - 2016–2024 for tuning.
  - **2025 to 16 Oct 2026 as holdout**: never used for tuning, looked at once.
- **Execution:** the signal comes at the close, the fill is at the next open. Stops fill at the stop price (or at the open on a gap). 0.05% cost per side.
- **Sizing:** up to 10 equal-weight positions. If there are more signals than free slots, the most oversold get the slots.
- **Honest test:** walk-forward. For each year from 2019 to 2024, pick the best rules on the 3 years before and trade them unchanged in that year. Then chain those years together ("stitched out-of-sample").
- **Benchmarks:**
  - SPY buy & hold.
  - **SPY timing**: hold SPY only while it is above its 200-day average, otherwise cash. This shows how much comes from the trend filter alone.

## What was tested
Several thousand rule sets in total. Each is listed with the command that reproduces it.

| Run | What varied | Command |
|---|---|---|
| Free search | RSI, Williams %R, Stochastic, IBS (lengths 2–14, thresholds), trend/market filter, stop, time stop | `python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600` |
| Filters forced | same, but stock above 100/200-day average and SPY filter always on | `... --only market_ma=200 trend_ma=100,200` |
| Broad pullbacks | only buy when 5–30% of the S&P 500 is oversold the same day, 3/5/10 positions | `... --space breadth --n-iter 500` |
| Exits | trailing stops, close above yesterday's high, time stop only | `... --space exits --n-iter 600` |
| More indicators | MFI, Bollinger %B, Connors RSI, Ultimate Oscillator, MACD, CCI | `... --space indicators --n-iter 800` |

## Results

| | 2019–2024 walk-forward (CAGR / max DD / Sharpe) | 2025–Oct 2026 holdout (CAGR / max DD) |
|---|---|---|
| **SPY buy & hold** | 17.1% / −33.7% / 0.90 | 17.4% / −18.8% |
| **SPY timing (>200-day avg)** | 12.2% / −20.7% / 0.97 | 11.2% / −10.2% |
| Free search | 10.9% / −17.1% / 0.78 | −2.1% / – |
| Filters forced → `strategies/willr.json` | 12.4% / −21.5% / 0.92 | 7.3% / −12.8% |
| Broad pullbacks → `strategies/candidate_breadth.json` | 6.2% / −39.3% / 0.42 | 18.3% / −16.8% * |
| Exits | 19.8% / −12.9% / 1.22 ** | −5.4% / −26.3% |
| More indicators (best: CCI) | 6.6% / −26.3% / 0.52 | −3.5% / −23.7% |
| Momentum family, point-in-time members (20 Oct) | 7.6% / −29.2% / 0.42 | +22.0% / −30.5% * |

All rows except the last use today's index members (survivorship bias, see the update below); the reference
set drops to 7.6% / yr over 2016–2026 with point-in-time members.

\* Its walk-forward result is the worst of all, so the good holdout is most likely luck.
\*\* This space was narrowed using earlier 2016–2024 results, so its walk-forward number is too optimistic. The holdout shows it.

**Verdict against the pass/fail criteria** (`backtest/criteria.py`, set before the final check):

| Criterion | willr.json | candidate_breadth | best CCI |
|---|---|---|---|
| Makes money in the test period | PASS | PASS | FAIL |
| Test period: at least 70% of SPY's return | FAIL (7.3% vs 17.4%) | PASS | FAIL |
| Clearly smoother ride than SPY (Sharpe +0.2, 2016–now) | PASS (1.13 vs 0.88) | FAIL (1.01) | FAIL |
| Test period: beats simple SPY timing | FAIL (Sharpe 0.54 vs 0.99) | FAIL (Calmar 1.09 vs 1.10) | FAIL |
| Still works with double trading costs | PASS (just) | PASS | FAIL |
| Smaller worst drop than SPY | PASS | PASS | PASS |
| **Result** | **4/6 – do not adopt** | **4/6 – do not adopt** | **1/6 – do not adopt** |

## Update 20 October 2026: survivorship bias fixed, momentum tested, luck check added

### 1. Survivorship bias is now (mostly) fixed
The backtest knows who was in the S&P 500 on every day since 1996 (`symbols/sp500_membership.csv`, built by
`python3 -m backtest.membership` from a public daily membership history) and only buys a stock while it was
a member. Effect on the reference rule set, 2016 – Oct 2026, same rules, same data:

| | Today's members (old) | Point-in-time members (new) |
|---|---|---|
| Return / year | 16.4% | **7.6%** |
| Worst drop | −18.1% | −27.4% |
| Sharpe | 1.15 | 0.60 |
| Luck check (edge over SPY real?) | 55% | 5% |
| Verdict | 4 of 6 | **2 of 7 – do not adopt** |

Half of the "edge" was buying dips in stocks that we only knew in hindsight would make it into the index.
`python3 -m backtest.fetch --historical` added prices for 115 of the 270 companies that left the index since
2016; 88.4% of all member-days now have prices. The remaining gap is mostly companies that were taken over or
renamed (Yahoo serves the history under the new ticker only). Remaining bias: small, and in the strategy's favour.

### 2. Momentum / trend following does not beat SPY here either
New strategy family: buy the strongest stocks (top N by 3–12-month performance, rebalanced weekly or monthly,
with rank / moving-average / trailing-stop exits) and breakouts (new 50–252-day high). 400 rule sets, same
walk-forward, point-in-time membership:

| 2019–2024 walk-forward | CAGR | Worst drop | Sharpe |
|---|---|---|---|
| Momentum family (rules re-picked each year) | 7.6% | −29.2% | 0.42 |
| SPY buy & hold | 17.1% | −33.7% | 0.90 |
| SPY timing | 12.2% | −20.7% | 0.97 |

The best in-sample set (top 30 by 3-month performance, weekly, exit below the 50-day average: 12% / yr on
2016–2024) made +22% / yr in the 2025–26 holdout with a −31% drop, but its excess over SPY across 2016–2026 is
−0.4% / yr and the deflated luck check is 0%: the holdout is one good year of a rule that failed its walk-forward. With today's members instead of point-in-time ones the
same family showed 36% / yr out-of-sample: momentum is the family that survivorship bias flatters most.
Concentrated momentum in large caps had two bad reversal years (2022, 2024) and 10–20 trades a year is too
few to be stable.

### 3. "Could this be luck?" is now part of the verdict
Every report checks whether the edge over SPY is distinguishable from chance:
- a block bootstrap of the daily difference to SPY gives the probability that the yearly excess return is really positive;
- a deflated Sharpe ratio raises the bar for the number of variants tried before the chosen one (`--trials`, and
  automatically in `tune`): a lucky best-of-400 looks like Sharpe 0.9 by itself.
- `analyze` also runs a random-entry test: the same trades with random entry days under the same filters.

Results: reference set 5% (excess −7.3% / yr), best momentum set 0% after 400 variants. Nothing tested so far
clears the 90% bar.

### 4. Corrected finding on the oversold signal
Earlier we wrote that an oversold entry earns the same as a random entry. Conditioning on the filters, the
picture is: days that pass the trend filter (stock and SPY above their 200-day average) are calm days with
*below*-average 5-day returns (0.27% vs 0.35% for all days); oversold signals on those days return 0.46%.
So the indicator does add about +0.2% per trade *given* the filter, and the filter gives it back. Net ≈ SPY,
and after survivorship bias, below SPY.

### 5. Comparison page and the RSI(4) rule book (20 Oct, evening)
`python3 -m backtest.compare` puts several rule sets and SPY on one page. The current table is in 7.1; what the first run showed:

Notes:
- The RSI(4) rule book (S&P 100 / Nasdaq-100, $5–$250, above the 200-day average, RSI(4) below 15, max 3 new
  trades a day, 10 positions, prioritised by each symbol's past ROI, stop loss = entry − 4 × ADR(20)) is implemented
  as written, **but the profit exit was not in the excerpt**. The files use a placeholder (sell when RSI(4) closes
  back above 55). Replace it with the real rule before reading anything into these numbers.
- OBE entries (only buy if the open is 2%+ below the signal close) cut the trades from ~1,600 to ~145 and the
  return with them; 5% adds raise the return (7.1% vs 4.4%) with the 4 × ADR stop capping the damage of averaging down.
- The momentum set's 18.5% / yr includes its own tuning period; its honest number is the 7.6% / yr walk-forward above.
- The daily returns of the dip-buying and momentum rules correlate only ~0.5, so mixing them would smooth the ride,
  but none of them earns more than SPY.

### 6. The momentum plan in practice (21 Oct)
`best_strategy.json` is the best of 400 momentum / breakout rule sets. The rules:

- **Buy:** on the first trading day of each month, every S&P 500 stock that newly entered the top 30 by 6-month
  performance (price ≥ $5, ≥ $5M traded per day). Buy at the next open, 10% of the account each, max 10 positions;
  with more signals than free slots, take the strongest.
- **Sell:** when the close is below the 200-day average, at the next open. No stop loss, no target.
- In practice: one look per month, ~15 trades a year, median hold about 6 months, 46% of trades win; the few big
  winners (STX +363%, AMD +204%, FIX +45% in 2025–26) pay for many −20…−30% losers (HOOD, LUV, DOW, SMCI).

| | Return / yr | Worst drop | Not luck |
|---|---|---|---|
| 2016 – Oct 2026 incl. the years the rules were picked on | 18.5% | −28.6% | 1% |
| Walk-forward 2019–24 (rules re-picked each year, traded the next) | 6.8% | −38% | – |
| Test period 2025 – today (never used to pick rules) | −4.6% | −21.8% | – |
| SPY buy & hold, same periods | 15.1% / 17.1% / 17.6% | −33.7% | – |

Verdict 3 / 7. The 18.5% is the number that made it "best", and it is the one not to trust: it was selected on
that data. Every honest view (walk-forward, test period, luck check) is below SPY. Kept as a reference for what momentum looks like here.

## Update 21 October 2026: all plans on one page, deep-below RSI(4), SPY call spreads

Everything below is one run on the data of 21 October 2026 (`python3 -m backtest.compare`; stock plans on
point-in-time S&P 500 members or today's S&P 100 + Nasdaq-100, 5 bps per side; the spread plan on SPY with
modelled option prices, see 7.3). Test period = 2025 – Oct 2026, never used to pick any rule.

### 7.1 The comparison

| Plan | Return / yr | Worst drop | Sharpe | Test return / yr | Test drop | Trades | Win | Verdict | Not luck |
|---|---|---|---|---|---|---|---|---|---|
| **SPY buy & hold** | 15.1% | −33.7% | 0.88 | 17.8% | −18.8% | – | – | – | – |
| SPY timing (above its 200-day average) | 11.4% | −20.7% | 0.98 | 11.7% | −10.2% | – | – | – | – |
| SPY bull call spread, $4.95 on $10, 2 monthly expirations (`bcs_495.json`) | 23.8% | −22.2% | 1.44 | 25.9% | −12.6% | 101 | 75% | **7 / 7** | 93% |
| Momentum top 30, best of 400 (`best_strategy.json`) | 18.6% | −28.6% | 0.82 | 27.2% | −38.3% | 164 | 46% | 3 / 7 | 1% |
| Broad market pullbacks (`candidate_breadth.json`) | 9.6% | −18.7% | 0.76 | 17.1% | −16.8% | 738 | 70% | 3 / 7 | 10% |
| RSI(4) deep below the 200-day average (`rsi4_deep.json`) | 8.6% | −30.7% | 0.65 | 14.7% | −9.9% | 724 | 66% | 3 / 7 | 4% |
| Williams %R dip-buying, reference (`willr.json`) | 7.8% | −27.4% | 0.61 | 9.0% | −13.3% | 3,571 | 65% | 2 / 7 | 5% |
| RSI(4) + 5% adds | 7.2% | −16.3% | 0.63 | 10.5% | −11.2% | 1,487 | 69% | 2 / 7 | 3% |
| RSI(4) classic rule book | 4.5% | −14.9% | 0.46 | 5.1% | −11.1% | 1,583 | 65% | 2 / 7 | 1% |
| RSI(4) + OBE 2% / 3% / 4% entries | 1.3–1.8% | −4% to −9% | 0.4–0.55 | 0.5–2.0% | −3% | 40–146 | 60–68% | 2 / 7 | 0% |
| RSI(4) + OBE + 5% adds | 1.9% | −12.8% | 0.35 | 0.9% | −4.1% | 143 | 77% | 2 / 7 | 0% |

"Not luck" = probability that the edge over SPY buy & hold is real, after the bootstrap and the deflated Sharpe
check (`best_strategy.json` is deflated for its 400 tried variants). All RSI(4) files still use the placeholder exit
(RSI(4) back above 55); the rule book's real profit exit is not implemented.

### 7.2 Stock plans: the picture has not changed

- **Nothing that buys and sells S&P 500 shares on indicator rules beats holding SPY.** The best of them (momentum,
  broad pullbacks) match or beat SPY in a good year and give it all back in a reversal year; their luck checks are
  1–10%. Every RSI(4) variant trails SPY by a wide margin in both periods.
- **Deep below the average (new).** Buying RSI(4) < 15 only when the stock is at least 15% below its 200-day
  average, i.e. washed-out names, is the best RSI(4) variant: 8.6% a year, 14.7% in the test period, 66% winners,
  and it is the only RSI(4) rule that made money in 2022 (+15.6% while SPY lost 18.7%) because crashed stocks
  bounce hardest. But the worst drop is −30.7% (March 2020: stops hit on FANG −43%, BA −32%), the stress test
  says the edge over SPY is −6.4% a year with a 4% chance of being positive, and random entries under the same
  filters did as well as the signal (+2.5% vs +1.7% per trade): the rebound comes from *being long beaten-down
  stocks*, not from the RSI timing. Trades held 1–5 days are the profitable ones (+4%); holding 10–20 days loses.
- **OBE entries** (open 2–4% below the signal close) cut the trade count to 40–146 in ten years and the return
  with it; whatever edge the rule book has is not in the gap.
- Conclusion for the stock side is unchanged: the 200-day trend filter does the work, the indicator does not,
  and more searching only finds more luck. SPY timing remains the honest benchmark for anyone who wants smaller drops.

### 7.3 The SPY call spread routine: the only plan that passes, and why that is not yet a result

**The routine.** Every third Friday, while SPY is above its 200-day average, buy the $10-wide SPY call spread that
costs $4.95 that day, expiring on the monthly expiration two months out. Close when it is worth $9.90 ($9.50 for
spreads expiring in March/June/September/December, when SPY goes ex-dividend on expiration day) or let it expire.
10% of the account at risk in total (two spreads open at once, 5% each).

**The backtest** (2016 – Oct 2026, 103 spreads): 23.8% a year, worst drop −22.2%, Sharpe 1.44, 76% winners;
test period 25.9% a year with a −12.6% drop against SPY's 17.8% / −18.8%. Per spread: average +49% of the debit,
winners +90%, losers −80%. All seven checks pass, luck check 93%. Year by year it beat SPY in 9 of 11 years and
lost only in 2022 (−12.7% vs −18.2%) and 2023 (+7% vs +26%).

**Why it works in the model.** A $4.95 spread on a $10 width is priced as a 50/50 bet, but SPY has risen over
about three quarters of all two-month windows that started above its 200-day average. The spread pays roughly
1:1, so a 75% win rate turns into +49% of the debit per trade: the routine harvests the market's upward drift with
leverage and a defined loss. That is a sound idea; holding SPY does the same without the leverage. The sizing table
shows exactly this: 2% at risk gives 3.9% a year, 10% gives 23.8%, 20% gives 49% with a −41% drop. The return
is a dial on the risk, not a free lunch.

**Why the number is not yet a result.**
1. **The option prices are modelled.** No fourteen years of SPY option quotes exist, so every spread is priced with
   Black-Scholes from the VIX of the day (with the T-bill rate and SPY's dividend yield); only the payoff at expiry
   is exact. The model has no volatility skew. In the real market calls above the price are cheaper than the model
   thinks, so **$4.95 buys strikes a little higher than the model picks**: the midpoint sat on average 0.06% above
   the SPY price in the model, 0.3% in the test period. Moving the strikes up by a few tenths of a percent costs
   several points of win rate, and the whole edge is the win rate. The early close at $9.90 also assumes the
   spread trades at model value.
2. **Your own six months were flat** while the model says 2026 is +20%. That is the most important data point we
   have, and it has to be reconciled before anyone trusts the backtest: either the real strikes sit higher than
   the model's (see 1), the fills are worse, or six months is simply too short (21 trades in the whole test period;
   a 76%-win routine has a 1-in-5 chance of a flat half year).
3. **Fragility in the tune.** 720 variants priced the same way: 91% made money in the test period, the median
   setup made 15% against SPY's 17.8%, and neighbouring setups (strike one notch apart) swing between +7% and
   +61% in the test period. The family looks fine; any single number does not.

**What to do with it.**
- The one check that settles it: **compare real fills with the model**. For each real spread (date, SPY price,
  both strikes, debit paid) look at the strikes the model would have chosen that day (the "strikes:" line of
  `python3 -m backtest.spreads --params strategies/bcs_495.json`). If the real strikes are consistently higher,
  the backtest is too kind and the win rate will be lower than 75%. Six months of past fills are enough to start.
- Keep the size honest: 10% of the account at risk is the backtest's −22% worst drop; 5% is −11% for 11% a year.
- The comparison that matters for the coach is **same risk, which earns more**: at a −20% worst drop the spread
  routine (10% at risk) backtests at 23.8% a year, SPY timing at 11.4%, SPY itself at 15.1% with −34%. If the
  fills confirm the model, that is a real improvement; if they don't, the routine is a leveraged SPY position
  with extra costs.

### 7.4 Which is the best?

On the numbers, the SPY call spread routine, and it is not close: it is the only plan that passes all seven
checks, beats SPY in both periods, and survives the luck check. On the evidence, it is the only plan whose
numbers rest on a model rather than on prices, and the one plan where real trading has so far disagreed with the
backtest. Among the stock plans nothing beats SPY; the deep-below RSI(4) rule is the most interesting of them
because it works in crashes, but it is a bet on rebounds, not on the indicator.

Ranking for the coach: 1. SPY buy & hold as the thing to beat; 2. the spread routine, pending the fill check;
3. SPY timing if smaller drops are the goal; 4. everything else as research, not as a plan.
The spread variants found by the tune (7.5) rank with the routine: more return in the model, more dependence on the model.

### 7.5 Other SPY call spread variants (from the 720-setup tune)

The tune (`python3 -m backtest.spreads --tune`) tried strikes by debit (40 / 50 / 60% of the width) or at a fixed
distance, widths 5 / 10 / 20, 30 / 60 / 90 days, close at 80 or 99% of the width, 50% stop or none, monthly
(third Friday) or weekly entries, with and without the 200-day filter. 91% of the 720 setups made money in the test
period; the median made 15% a year against SPY's 17.8%. Three families from the top of the table are saved as plans
and run on the same data as your routine:

| Plan | Return / yr | Worst drop | Sharpe | Win | Trades | Test return / yr | Test drop | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Yours**: $4.95 on $10, 2 months, 99%, above 200-day (`bcs_495.json`) | 23.8% | −22.2% | 1.44 | 76% | 103 | 25.9% | −12.6% | 7 / 7 |
| In the money: $6.00 on $10, 1 month, 99%, 50% stop, no filter (`bcs_itm30.json`) | 46.2% | −19.0% | 2.02 | 77% | 130 | 75.1% | −12.4% | 7 / 7 |
| Weekly ladder: $5.00 on $10 every week, 1 month, 99%, 50% stop, no filter (`bcs_weekly.json`) | 35.7% | −19.7% | 1.74 | 62% | 564 | 30.6% | −17.6% | 7 / 7 |
| Smooth: $6.00 on $10, 2 months, take profit at 80%, 50% stop, above 200-day (`bcs_smooth.json`) | 14.2% | −8.0% | 1.66 | 87% | 103 | 11.6% | −8.0% | 5 / 7 |
| SPY buy & hold | 15.1% | −33.7% | 0.88 | | | 17.8% | −18.8% | |

What the tune says, read with the caveats of 7.3 in mind:

- **The direction is consistent, the sizes are not.** Shorter spreads (one month instead of two), strikes below the
  price (pay $6 instead of $4.95: the bet is "SPY does not fall 1–1.5%" rather than "SPY rises"), a 50% stop and
  no trend filter all raise the model's return; each is a step towards selling insurance on a rising market. The
  in-the-money variant's 75% a year in the test period is the model enjoying 2025–26; its development-period 41%
  is the number to argue about, and even that is the best of 720.
- **The stop is the most model-dependent piece.** Every top row has the 50% stop, which turns −100% losers into
  −55…−60% ones (24 of the in-the-money plan's 130 trades were stopped). A stop fires on the *modelled* mark of a
  spread in the middle of a sell-off, when real quotes are widest and skew is steepest. Your routine, held to expiry,
  depends on the model only for the strikes; the smooth variant depends on it for both exits and its results should
  be taken as the least reliable of the four despite the best-looking drawdown.
- **The weekly ladder is the most robust evidence.** 564 trades instead of 103, four to five spreads open at once,
  62% winners paying about 1.5:1, 36% a year in the development period and 31% in the test period with no cliff
  between them. It is also the hardest to trade by hand (a new spread every week, four open positions, a stop to
  watch) and the most sensitive to costs: at four times the trades, slippage is four times the drag.
- **Your routine is in the middle of the pack by Calmar (rank 424 of 688) and near the top by test-period return.**
  That is what a setup that was *not* found by searching should look like. The variants above were found by
  searching 720 combinations and are flattered accordingly.

**If one variant is worth checking against real fills next to yours, it is the one-month in-the-money spread without the stop**
(`debit:0.6, $10, 30 days, 99%, expiry, no filter`: 41% a year, −21% drop in the development period, 61% in the
test period, held to expiry so the model only picks the strikes). The gap between model and market shows up in
the debit the market asks for the model's strikes.

## What we learned
1. **The 200-day trend filter does most of the work.** Without it, returns halve and drawdowns double. Holding only SPY above its 200-day average gets a similar Sharpe (0.97) with no stock picking at all.
2. **An oversold reading has no edge on its own.** Trades entered on an oversold signal average +0.49%. Random entries in the same stock, year and holding time average +0.47%. The signal raises the win rate (69% vs 55%) but the losses get bigger, so the average stays the same.
3. **The indicator hardly matters.** All 10 oscillators reach similar in-sample results (13–18% per year, Sharpe about 1.0), because they all measure the same short pullback. The walk-forward picked a different indicator almost every year.
4. **The more we searched, the better the in-sample results and the worse the honest ones.** Best in-sample results of 15–22% per year shrink to 6–12% in walk-forward. The holdout was negative for most winners.
5. **What does show some edge:** buying on broad market pullbacks (many stocks oversold at once), more volatile stocks, and sizing slots toward the most oversold names. All of it is too weak and unstable to trade on its own.
6. **Costs and universe matter.** At 0.20% cost per side, 18% per year becomes 7%. On the Nasdaq-100 and S&P 100 the same rules make 5–7%.
7. **Results are optimistic.** Using today's index members leaves out stocks that fell out of the index (survivorship bias). This favours dip-buying.

## Conclusion
- **No rule set found here is clearly better than SPY, and with survivorship bias removed the best ones are clearly worse.** The best ones roughly match SPY's return per unit of risk (Sharpe) and cut the drawdown by about half. They earn less in total, and in 2025–26 they also lost to the simple "SPY above its 200-day average" rule.
- **A pure "buy the oversold dip" strategy on S&P 500 stocks has no robust edge in 2016–2026.** Its value is lower drawdown, and that comes from the trend filter.
- **Searching more parameters or indicators won't change this.** It only produces more lucky in-sample winners.
- **(21 Oct) The SPY call spread routine is the only plan that passes every check** (23.8% a year, −22% worst drop, 7 / 7), but its option prices are modelled and six months of real trades were flat; section 7.3 says what to check before trusting it.

## Next steps (analysis only)
1. **Check the spread model against real fills** (7.3): real strikes vs model strikes for the same debit. This decides whether the one passing plan is real.
2. **Implement the RSI(4) rule book's real profit exit**; every `rsi4_*.json` still uses the placeholder.
3. **Add volatility skew to the option model** if the fill check shows the strikes are consistently off.
4. **Test any new idea as a fixed hypothesis** (Planner, written down first, reported once), not as a new search.

## Reproduce
```sh
python3 -m backtest.fetch
python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600
python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600 --only market_ma=200 trend_ma=100,200
python3 -m backtest.tune --space breadth    --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 500
python3 -m backtest.tune --space exits      --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 600
python3 -m backtest.tune --space indicators --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 800
python3 -m backtest.tune --space momentum   --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar --n-iter 400 --min-trades 20
python3 -m backtest.analyze --params strategies/willr.json            # add --quick for verdict only
python3 -m backtest.report  --params strategies/willr.json            # one-page HTML report
```
Random search uses `--seed 42`. Numbers can move slightly when the data is re-downloaded, because adjusted prices and index members change.
