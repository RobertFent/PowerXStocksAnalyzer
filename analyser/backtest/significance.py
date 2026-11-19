"""Could the result be luck? Three complementary checks.

1. bootstrap_excess   Block-bootstrap of daily (strategy - benchmark) returns. Gives the probability
                      that the true yearly excess return over the benchmark is positive, and a range.
2. deflated_sharpe    Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado 2014): probability
                      that the true Sharpe of the excess returns is above the Sharpe you would expect
                      from the BEST of `trials` zero-skill strategies. Accounts for how many rule sets
                      were tried, the length of the test and fat tails. trials=1 for a plan tested once.
3. random_entry_test  Monte Carlo at portfolio level: the same trades (ticker, year, holding time) but
                      random entry days, run through the same portfolio simulation. p-value = share of
                      random portfolios that did at least as well. Tests whether the entry timing adds
                      anything beyond simply being long those stocks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation, good to ~1e-9)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def excess_returns(strategy: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Daily strategy return minus benchmark return, aligned on dates."""
    s, b = daily_returns(strategy), daily_returns(benchmark)
    df = pd.concat([s, b], axis=1, join='inner').dropna()
    return df.iloc[:, 0] - df.iloc[:, 1]


def bootstrap_excess(excess: pd.Series, n: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Stationary block bootstrap of the mean daily excess return.
    Returns p_positive (prob. the true yearly excess > 0), the point estimate and a 90% range (yearly)."""
    x = excess.to_numpy(dtype=float)
    T = len(x)
    if T < 60:
        return {'p_positive': float('nan'), 'excess_yearly': float('nan'), 'low': float('nan'), 'high': float('nan'), 'days': T}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    n_blocks = T // block + 1
    for i in range(n):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        means[i] = x[idx[:T]].mean()
    yearly = means * TRADING_DAYS
    return {'p_positive': float((means > 0).mean()), 'excess_yearly': float(x.mean() * TRADING_DAYS),
            'low': float(np.percentile(yearly, 5)), 'high': float(np.percentile(yearly, 95)), 'days': T}


def deflated_sharpe(excess: pd.Series, trials: int = 1) -> dict:
    """Probability that the true (daily) Sharpe of `excess` exceeds the Sharpe expected from the best of
    `trials` strategies with zero skill (= 0 when trials == 1). Uses skew and kurtosis of the returns."""
    x = excess.to_numpy(dtype=float)
    T = len(x)
    if T < 60 or x.std(ddof=1) == 0:
        return {'prob': float('nan'), 'sharpe_yearly': float('nan'), 'sharpe0_yearly': float('nan'), 'trials': trials}
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=1)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    trials = max(1, int(trials))
    if trials > 1:
        var_sr = 1.0 / T  # variance of a zero-skill daily Sharpe estimate
        sr0 = math.sqrt(var_sr) * ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / trials)
                                   + EULER_GAMMA * _norm_ppf(1 - 1.0 / (trials * math.e)))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    stat = (sr - sr0) * math.sqrt(T - 1) / denom
    return {'prob': _norm_cdf(stat), 'sharpe_yearly': sr * math.sqrt(TRADING_DAYS),
            'sharpe0_yearly': sr0 * math.sqrt(TRADING_DAYS), 'trials': trials, 'days': T}


def random_entry_test(fs, trades: pd.DataFrame, params, start, end, n: int = 30,
                      cost_per_side: float = 0.0005, seed: int = 0, metric: str = 'sharpe') -> dict:
    """Portfolio-level Monte Carlo: keep each trade's ticker, year and holding time, randomise its entry
    day among days that pass the same filters (trend, market, liquidity), re-run the portfolio simulation.
    p-value = share of random runs that did at least as well. Tests whether the indicator's timing adds
    anything beyond being long those stocks under those filters."""
    from backtest.engine import simulate_portfolio
    from backtest.metrics import compute_metrics
    from backtest.strategy import filter_mask

    rng = np.random.default_rng(seed)
    res = simulate_portfolio(fs, trades, start, end, max_positions=params.max_positions, max_new_per_day=params.max_new_per_day, cost_per_side=cost_per_side)
    actual = compute_metrics(res.equity, res.trades, res.exposure)[metric]

    t = trades[(trades['exit_reason'] != 'open')].reset_index(drop=True)
    years = pd.DatetimeIndex(fs.dates).year.to_numpy()
    tick = fs.tickers
    allowed = filter_mask(fs, params)
    allowed[-1] = False
    allowed[fs.first_row[1:].nonzero()[0]] = False  # row before a ticker change can't be a signal day
    idx_all = np.arange(len(tick))[allowed]
    groups = pd.Series(idx_all).groupby([tick[idx_all], years[idx_all]])
    pools = {k: v.to_numpy() for k, v in groups}
    n_rows = len(tick)
    values = []
    for _ in range(n):
        e_idx, x_idx = [], []
        for row in t.itertuples(index=False):
            pool = pools.get((row.ticker, row.entry_date.year))
            if pool is None or len(pool) == 0:
                continue
            hold = max(int(row.bars_held), 1)
            sig = int(pool[rng.integers(0, len(pool))])  # signal day (passes the filters)
            e, x = sig + 1, sig + 1 + hold                # entry next open, exit `hold` bars later
            if x >= n_rows or tick[x] != row.ticker:
                continue
            e_idx.append(e)
            x_idx.append(x)
        e_idx, x_idx = np.array(e_idx, dtype=np.int64), np.array(x_idx, dtype=np.int64)
        r = pd.DataFrame({
            'ticker': tick[e_idx], 'signal_date': fs.dates[e_idx - 1],
            'entry_date': fs.dates[e_idx], 'exit_date': fs.dates[x_idx],
            'entry_price': fs.open[e_idx], 'exit_price': fs.open[x_idx],
            'exit_reason': 'signal', 'bars_held': x_idx - e_idx,
            'rank': rng.random(len(e_idx)), 'entry_idx': e_idx, 'exit_idx': x_idx,
            'add_idx': -1, 'add_price': np.nan,
        })
        r['gross_return'] = r['exit_price'] / r['entry_price'] - 1.0
        r = r.sort_values(['entry_date', 'rank']).reset_index(drop=True)
        rr = simulate_portfolio(fs, r, start, end, max_positions=params.max_positions, max_new_per_day=params.max_new_per_day, cost_per_side=cost_per_side)
        values.append(compute_metrics(rr.equity, rr.trades, rr.exposure)[metric])
    values = np.array(values)
    return {'actual': float(actual), 'random_median': float(np.median(values)),
            'random_p95': float(np.percentile(values, 95)), 'p_value': float((values >= actual).mean()),
            'runs': n, 'metric': metric}


def luck_summary(strategy_equity: pd.Series, benchmark_equity: pd.Series, trials: int = 1) -> dict:
    """The two cheap checks in one dict (used by the verdict)."""
    ex = excess_returns(strategy_equity, benchmark_equity)
    return {'bootstrap': bootstrap_excess(ex), 'dsr': deflated_sharpe(ex, trials), 'trials': trials}


def format_luck(l: dict, benchmark_name: str = 'SPY') -> list[str]:
    b, d = l['bootstrap'], l['dsr']
    lines = []
    if not np.isnan(b['p_positive']):
        lines.append(f'  Excess return over {benchmark_name}: {b["excess_yearly"] * 100:+.1f}% per year '
                     f'(90% range {b["low"] * 100:+.1f}% .. {b["high"] * 100:+.1f}%)  '
                     f'-> probability it is really positive: {b["p_positive"] * 100:.0f}%')
    if not np.isnan(d['prob']):
        extra = (f' after accounting for {d["trials"]} tested variants (a lucky best-of-{d["trials"]} would show '
                 f'Sharpe {d["sharpe0_yearly"]:.2f})' if d['trials'] > 1 else '')
        lines.append(f'  Sharpe of the excess returns: {d["sharpe_yearly"]:.2f}  -> probability it is really > 0: '
                     f'{d["prob"] * 100:.0f}%{extra}')
    return lines or ['  not enough data']
