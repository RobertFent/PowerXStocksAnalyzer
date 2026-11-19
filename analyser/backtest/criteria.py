"""Pass/fail criteria for adopting a rule set - decided BEFORE looking at results.

Used by `python3 -m backtest.analyze` (section VERDICT). Edit the thresholds here,
not after seeing a result. Meant for a rule set chosen in advance (e.g. your trading
plan or strategies/willr.json); for a freshly tuned set the tuning period is in-sample,
so judge it by the walk-forward result of `backtest.tune` first.
"""

HOLDOUT_MIN_CAGR = 0.0              # holdout must make money
HOLDOUT_MIN_SHARE_OF_SPY = 0.7      # ... and at least 70% of SPY's holdout CAGR
SHARPE_MARGIN_OVER_SPY = 0.2        # full-period Sharpe >= SPY buy & hold Sharpe + 0.2
STRESS_COST_BPS = 10                # at this cost per side Sharpe must still be >= SPY's
LUCK_MIN_PROB = 0.90                # probability that the edge over SPY is real (deflated for tested variants)


def evaluate(full: dict, holdout: dict, stressed: dict,
             spy_full: dict, spy_holdout: dict, timing_holdout: dict,
             luck: dict | None = None) -> list[tuple[str, bool, str]]:
    """Returns (criterion, passed, detail) rows. `luck` = backtest.significance.luck_summary(...)."""
    pct = lambda x: f'{x * 100:.1f}%'
    rows = [
        ('Makes money in the test period', holdout['cagr'] > HOLDOUT_MIN_CAGR,
         f'{pct(holdout["cagr"])} per year'),
        (f'Test period: at least {HOLDOUT_MIN_SHARE_OF_SPY:.0%} of SPY\'s return',
         holdout['cagr'] >= HOLDOUT_MIN_SHARE_OF_SPY * spy_holdout['cagr'],
         f'{pct(holdout["cagr"])} vs SPY {pct(spy_holdout["cagr"])} per year'),
        ('Clearly smoother ride than SPY', full['sharpe'] >= spy_full['sharpe'] + SHARPE_MARGIN_OVER_SPY,
         f'Sharpe {full["sharpe"]:.2f} vs SPY {spy_full["sharpe"]:.2f} (needs +{SHARPE_MARGIN_OVER_SPY})'),
        ('Test period: beats simple SPY timing',
         holdout['sharpe'] > timing_holdout['sharpe'] and holdout['calmar'] > timing_holdout['calmar'],
         f'Sharpe {holdout["sharpe"]:.2f} vs {timing_holdout["sharpe"]:.2f}, '
         f'Calmar {holdout["calmar"]:.2f} vs {timing_holdout["calmar"]:.2f}'),
        ('Still works with double trading costs', stressed['sharpe'] >= spy_full['sharpe'],
         f'Sharpe {stressed["sharpe"]:.2f} at {STRESS_COST_BPS:g} bps vs SPY {spy_full["sharpe"]:.2f}'),
        ('Smaller worst drop than SPY', full['max_drawdown'] > spy_full['max_drawdown'],
         f'{pct(full["max_drawdown"])} vs SPY {pct(spy_full["max_drawdown"])}'),
    ]
    if luck is not None:
        d, b = luck['dsr'], luck['bootstrap']
        prob = min(d['prob'], b['p_positive'])
        trials = d.get('trials', 1)
        detail = (f'probability the edge over SPY is real: {prob * 100:.0f}% '
                  f'(excess {b["excess_yearly"] * 100:+.1f}%/yr'
                  + (f', after {trials} tested variants' if trials > 1 else '') + ')')
        rows.append(('Unlikely to be luck', bool(prob >= LUCK_MIN_PROB), detail))
    return rows


def format_verdict(rows: list[tuple[str, bool, str]]) -> list[str]:
    lines = [f'  [{"PASS" if ok else "FAIL"}] {name:<42} {detail}' for name, ok, detail in rows]
    passed = sum(ok for _, ok, _ in rows)
    lines.append(f'  => {passed}/{len(rows)} passed: '
                 + ('PASSES every check' if passed == len(rows) else 'DOES NOT PASS'))
    return lines
