"""Stress-test ONE rule set: is the edge real and robust, or curve-fit?

    python3 -m backtest.analyze --params output/tuning_<ts>/best_params.json --start 2016-01-01 --holdout-start 2025-01-01

Sections
  1. Baseline        tuning period vs holdout vs SPY
  2. Neighborhood    change one parameter at a time (incl. max_positions) - robust rules sit on a plateau
  3. Costs           0 / 5 / 10 / 20 bps per side
  4. Slot choice     ranked choice vs random choice among the day's signals
  5. Random entries  signal trades vs random entries in the same stocks/years with the same holding time
  6. Regimes         SPY above / below its 200d average, per year, by holding time, worst trades
  7. Universes       same rules on nasdaq100 / sp100

Writes output/analysis_<ts>/report.txt (+ csv tables).
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.data import OUTPUT_DIR, load_benchmark, load_prices, load_universe
from backtest.engine import generate_trades, simulate_portfolio
from backtest.features import FeatureStore
from backtest.metrics import compute_metrics
from backtest.search_space import NEIGHBORHOOD
from backtest.strategy import StrategyParams

COLS = ['cagr', 'max_drawdown', 'sharpe', 'calmar', 'trades', 'win_rate', 'avg_trade', 'profit_factor']


def fmt_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c in ('cagr', 'max_drawdown', 'win_rate', 'avg_trade', 'exposure', 'mean', 'median', 'p05', 'p95', 'spy_cagr', 'spy_maxdd'):
            out[c] = out[c].map(lambda x: f'{x * 100:7.2f}%' if pd.notna(x) else '')
        elif out[c].dtype.kind == 'f':
            out[c] = out[c].map(lambda x: f'{x:7.2f}' if pd.notna(x) else '')
    return out.to_string()


class Ctx:
    def __init__(self, fs, start, end, sim):
        self.fs, self.start, self.end, self.sim = fs, start, end, sim

    def run(self, p: StrategyParams, trades=None, **sim_over) -> dict:
        trades = generate_trades(self.fs, p) if trades is None else trades
        kw = {'max_positions': p.max_positions, **self.sim, **sim_over}
        res = simulate_portfolio(self.fs, trades, self.start, self.end, **kw)
        return compute_metrics(res.equity, res.trades, res.exposure)


def neighborhood(ctx: Ctx, p: StrategyParams) -> pd.DataFrame:
    rows = []
    for key, candidates in NEIGHBORHOOD.items():
        if key == 'entry_length' and p.entry_indicator == 'ibs':
            continue
        if key == 'exit_threshold' and p.exit_rule != 'osc':
            continue
        values = sorted(set(candidates) | {getattr(p, key)}, key=lambda v: (str(type(v)), v))
        for v in values:
            m = ctx.run(p.with_(**{key: v}))
            rows.append({'param': key, 'value': v, 'current': '<--' if v == getattr(p, key) else '',
                         **{c: m[c] for c in COLS}})
    return pd.DataFrame(rows).set_index(['param', 'value'])


def random_entry_edge(fs: FeatureStore, trades: pd.DataFrame, start, end, seed=0) -> pd.DataFrame:
    """For every signal trade pick a random entry day in the same ticker & year, hold the same
    number of bars (open -> open). Shows whether the signal beats just being long that stock."""
    rng = np.random.default_rng(seed)
    t = trades[(trades['entry_date'] >= pd.Timestamp(start)) & (trades['entry_date'] <= pd.Timestamp(end))]
    t = t[t['exit_reason'] != 'open']
    years = pd.DatetimeIndex(fs.dates).year.to_numpy()
    tick = fs.tickers
    # row ranges per (ticker, year)
    key = pd.Series(np.arange(len(tick))).groupby([tick, years])
    ranges = {k: (v.min(), v.max()) for k, v in key}
    rand = []
    for row in t.itertuples(index=False):
        lo, hi = ranges[(row.ticker, row.entry_date.year)]
        hold = max(int(row.bars_held), 1)
        hi = min(hi, len(tick) - 1 - hold)
        if hi <= lo:
            continue
        i = int(rng.integers(lo, hi + 1))
        j = i + hold
        if tick[j] != row.ticker:
            continue
        rand.append(fs.open[j] / fs.open[i] - 1)
    sig = t['gross_return']
    rand = pd.Series(rand)
    def stats(s):
        return {'n': len(s), 'mean': s.mean(), 'median': s.median(), 'win_rate': (s > 0).mean(),
                'p05': s.quantile(0.05), 'p95': s.quantile(0.95)}
    df = pd.DataFrame({'signal trades': stats(sig), 'random entries': stats(rand)}).T
    df['n'] = df['n'].astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', help='best_params.json (default: StrategyParams defaults)')
    ap.add_argument('--set', nargs='*', default=[], help='override params key=value')
    ap.add_argument('--index', default='sp500')
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--holdout-start', default='2025-01-01')
    ap.add_argument('--cost-bps', type=float, default=5.0)
    ap.add_argument('--random-runs', type=int, default=20)
    args = ap.parse_args()

    p = StrategyParams.load(args.params) if args.params else StrategyParams()
    if args.set:
        p = StrategyParams.from_dict({**p.to_dict(), **dict(s.split('=', 1) for s in args.set)})

    bench = load_benchmark('SPY')
    fs = FeatureStore(load_prices(args.index), bench)
    tune_end = (pd.Timestamp(args.holdout_start) - pd.Timedelta(days=1)) if args.holdout_start else None
    sim = {'cost_per_side': args.cost_bps / 1e4}
    ctx = Ctx(fs, args.start, tune_end, sim)
    trades = generate_trades(fs, p)
    out = OUTPUT_DIR / f'analysis_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True, exist_ok=True)
    lines = [f'ANALYSIS  universe={args.index}  tuning period {args.start}..{tune_end.date() if tune_end is not None else "end"}'
             f'  holdout {args.holdout_start}..end', '', p.describe(), '']

    # 1 baseline
    rows = {}
    for name, (s, e) in {'tuning period': (args.start, tune_end), 'holdout': (args.holdout_start, None),
                         'full': (args.start, None)}.items():
        res = simulate_portfolio(fs, trades, s, e, max_positions=p.max_positions, **sim)
        m = compute_metrics(res.equity, res.trades, res.exposure)
        b = bench.set_index('Date')['Close'].reindex(res.equity.index).ffill()
        bm = compute_metrics(b / b.iloc[0], pd.DataFrame(columns=['net_return', 'pnl', 'bars_held']))
        rows[name] = {**{c: m[c] for c in COLS}, 'exposure': m['exposure'],
                      'spy_cagr': bm['cagr'], 'spy_maxdd': bm['max_drawdown'], 'spy_sharpe': bm['sharpe']}
        if name == 'full':
            full_res = res
    base = pd.DataFrame(rows).T
    lines += ['1. BASELINE', fmt_table(base), '']

    # 2 neighborhood
    nb = neighborhood(ctx, p)
    nb.to_csv(out / 'neighborhood.csv')
    lines += ['2. NEIGHBORHOOD (tuning period, one parameter changed at a time)', fmt_table(nb), '']

    # 3 costs
    cost = pd.DataFrame({f'{bps} bps': ctx.run(p, trades, cost_per_side=bps / 1e4) for bps in (0, 5, 10, 20)}).T[COLS]
    lines += ['3. COSTS per side (tuning period)', fmt_table(cost), '']

    # 4 slot choice
    rnd = []
    rng = np.random.default_rng(1)
    for _ in range(args.random_runs):
        t2 = trades.copy()
        t2['rank'] = rng.random(len(t2))
        t2 = t2.sort_values(['entry_date', 'rank']).reset_index(drop=True)
        rnd.append(ctx.run(p, t2))
    rnd = pd.DataFrame(rnd)[COLS]
    sel = pd.DataFrame({'most oversold first': ctx.run(p, trades),
                        'random (median)': rnd.median(), 'random (worst)': rnd.min(), 'random (best)': rnd.max()}).T[COLS]
    lines += [f'4. SLOT CHOICE (tuning period, {args.random_runs} random runs)', fmt_table(sel), '']

    # 5 random entries
    edge = random_entry_edge(fs, trades, args.start, tune_end or fs.dates.max())
    lines += ['5. SIGNAL vs RANDOM ENTRY (gross per trade, same ticker/year/holding time)', fmt_table(edge), '']

    # 6 regimes
    t = full_res.trades.copy()
    spy = bench.set_index('Date')['Close']
    up = (spy > spy.rolling(200).mean())
    t['spy_regime'] = np.where(up.reindex(t['signal_date']).to_numpy(dtype=bool), 'SPY > SMA200', 'SPY < SMA200')
    reg = t.groupby('spy_regime')['net_return'].agg(['count', 'mean', 'median', lambda s: (s > 0).mean()])
    reg.columns = ['trades', 'mean', 'median', 'win_rate']
    yr = t.groupby(t['entry_date'].dt.year)['net_return'].agg(['count', 'mean', lambda s: (s > 0).mean()])
    yr.columns = ['trades', 'mean', 'win_rate']
    eq = full_res.equity
    yr['strategy_return'] = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
    spy_eq = spy.reindex(eq.index).ffill()
    yr['spy_return'] = spy_eq.groupby(spy_eq.index.year).last() / spy_eq.groupby(spy_eq.index.year).first() - 1
    hold = t.groupby(pd.cut(t['bars_held'], [-1, 1, 3, 5, 10, 20, 1000]), observed=True)['net_return'].agg(['count', 'mean'])
    lines += ['6. REGIMES (full period, trades taken)', fmt_table(reg), '',
              'per year', yr.to_string(float_format=lambda x: f'{x * 100:.2f}%' if abs(x) < 50 else f'{x:.0f}'), '',
              'by holding days', hold.to_string(float_format=lambda x: f'{x * 100:.2f}%' if abs(x) < 5 else f'{x:.0f}'), '',
              'worst 10 trades', t.nsmallest(10, 'net_return')[['ticker', 'entry_date', 'exit_date', 'net_return', 'bars_held', 'exit_reason']].to_string(index=False), '']

    # 7 other universes
    uni_rows = {}
    universes = load_universe()['Index'].unique()
    for idx in [u for u in universes if u != args.index]:
        fs2 = FeatureStore(load_prices(idx), bench)
        ctx2 = Ctx(fs2, args.start, None, sim)
        uni_rows[idx] = ctx2.run(p)
    if uni_rows:
        lines += ['7. OTHER UNIVERSES (full period from --start)', fmt_table(pd.DataFrame(uni_rows).T[COLS]), '']

    report = '\n'.join(lines)
    print(report)
    (out / 'report.txt').write_text(report)
    full_res.trades.to_csv(out / 'trades.csv', index=False)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
