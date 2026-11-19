"""Find the best rule parameters with walk-forward validation.

    python3 -m backtest.tune                                  # random search, 400 combos, sp500
    python3 -m backtest.tune --method grid                    # full grid (can be large!)
    python3 -m backtest.tune --start 2016-01-01 --holdout-start 2025-01-01 --metric calmar
    python3 -m backtest.tune --only entry_indicator=willr max_positions=5   # restrict / extend the space
    python3 -m backtest.tune --space breadth                  # use SPACES['breadth'] from search_space.py

How it works
  1. The period before --holdout-start is split into rolling folds:
        train (--train-years)  ->  test (--test-years), moved forward by test-years.
  2. Every parameter set is backtested on every window (portfolio sim incl. costs).
  3. Per fold the set with the best TRAIN score is picked and its TEST result recorded.
     Chaining the test results = honest out-of-sample (OOS) estimate of the whole
     "tune -> trade" process. If this is bad, the best in-sample result is luck.
  4. Final params = best score over the whole tuning period. They are then run once
     on the untouched holdout period (if given). Do not re-tune after looking at it.

Score = --metric (sharpe | sortino | calmar | cagr | profit_factor) with constraints
--min-trades per window and --max-dd; violating sets get -inf.

Outputs: output/tuning_<ts>/{leaderboard.csv, all_results.csv, walk_forward.csv,
         best_params.json, report.txt}
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest.benchmarks import benchmark_metrics, format_benchmarks, spy_buy_hold
from backtest.significance import format_luck, luck_summary
from backtest.data import OUTPUT_DIR, load_benchmark, load_prices
from backtest.engine import generate_trades, simulate_portfolio
from backtest.features import FeatureStore
from backtest.membership import load_membership
from backtest.metrics import compute_metrics, format_metrics, score
from backtest.search_space import SPACES
from backtest.strategy import StrategyParams

PARAM_NAMES = list(StrategyParams().to_dict().keys())

# ---------------------------------------------------------------- search space


def _coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def filtered_space(only: list[str], base_space: list[dict]) -> list[dict]:
    fixed = {}
    for item in only or []:
        k, v = item.split('=', 1)
        fixed[k] = [_coerce(x) for x in v.split(',')]
    space = []
    for sub in base_space:
        sub = dict(sub)
        ok = True
        for k, vals in fixed.items():
            if k in sub:
                keep = [x for x in sub[k] if x in vals] or ([] if k == 'entry_indicator' else vals)
                if not keep:
                    ok = False
                sub[k] = keep
            else:
                sub[k] = vals
        if ok:
            space.append(sub)
    return space


def grid(space: list[dict]) -> list[dict]:
    combos = []
    for sub in space:
        keys = list(sub)
        for vals in itertools.product(*(sub[k] for k in keys)):
            combos.append(dict(zip(keys, vals)))
    return combos


def random_sample(space: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sizes = [int(np.prod([len(v) for v in sub.values()])) for sub in space]
    total = sum(sizes)
    if n >= total:
        return grid(space)
    seen, combos = set(), []
    while len(combos) < n:
        sub = rng.choices(space, weights=sizes)[0]
        c = {k: rng.choice(v) for k, v in sub.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            combos.append(c)
    return combos


# ---------------------------------------------------------------- windows


def make_windows(dates: pd.DatetimeIndex, tune_end: pd.Timestamp, warmup_years: float,
                 train_years: float, test_years: float, anchored: bool,
                 first_date: pd.Timestamp | None = None) -> tuple[list, list]:
    start = dates.min() + pd.DateOffset(months=int(warmup_years * 12))
    if first_date is not None:
        start = max(start, first_date)
    folds = []
    tr_start = start
    while True:
        tr_end = tr_start + pd.DateOffset(months=int(train_years * 12))
        te_end = tr_end + pd.DateOffset(months=int(test_years * 12))
        if tr_end >= tune_end:
            break
        te_end = min(te_end, tune_end)
        if (te_end - tr_end).days < 120:  # skip tiny last fold
            break
        folds.append({'fold': len(folds) + 1,
                      'train': ((start if anchored else tr_start), tr_end - pd.Timedelta(days=1)),
                      'test': (tr_end, te_end - pd.Timedelta(days=1))})
        tr_start = tr_start + pd.DateOffset(months=int(test_years * 12))
    windows = [('full', start, tune_end - pd.Timedelta(days=1))]
    for f in folds:
        windows.append((f'train_{f["fold"]}', *f['train']))
        windows.append((f'test_{f["fold"]}', *f['test']))
    return windows, folds


# ---------------------------------------------------------------- worker

_FS: FeatureStore | None = None
_CFG: dict = {}


def _init(cfg: dict) -> None:
    global _FS, _CFG
    _CFG = cfg
    prices = load_prices(cfg['index'])
    _FS = FeatureStore(prices, load_benchmark('SPY'), load_membership(cfg['index']))


def _evaluate(job: tuple[int, dict]) -> list[dict]:
    combo_id, combo = job
    params = StrategyParams.from_dict({**_CFG['base'], **combo})
    rows = []
    try:
        trades = generate_trades(_FS, params)
        for name, ws, we in _CFG['windows']:
            res = simulate_portfolio(_FS, trades, ws, we, max_positions=params.max_positions, max_new_per_day=params.max_new_per_day, **_CFG['sim'])
            m = compute_metrics(res.equity, res.trades, res.exposure)
            rows.append({'combo_id': combo_id, 'window': name, **params.to_dict(), **m})
    except Exception as e:  # keep tuning even if one combo explodes
        rows.append({'combo_id': combo_id, 'window': 'error', 'error': repr(e), **combo})
    return rows


# ---------------------------------------------------------------- main


def stitched_oos(fs: FeatureStore, wf: pd.DataFrame, base: dict, sim: dict, capital: float) -> pd.Series:
    parts = []
    for _, row in wf.iterrows():
        p = StrategyParams.from_dict({**base, **{k: row[k] for k in PARAM_NAMES if k in row}})
        res = simulate_portfolio(fs, generate_trades(fs, p), row['test_start'], row['test_end'],
                                 max_positions=p.max_positions, max_new_per_day=p.max_new_per_day, **sim)
        parts.append(res.equity.pct_change().fillna(0.0))
    rets = pd.concat(parts)
    return capital * (1 + rets).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', default='sp500', help='sp500 | nasdaq100 | sp100 | all')
    ap.add_argument('--method', choices=['random', 'grid'], default='random')
    ap.add_argument('--n-iter', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--space', default='default', choices=sorted(SPACES), help='search space from search_space.py')
    ap.add_argument('--only', nargs='*', default=[], help='restrict / override space: key=v1,v2 ...')
    ap.add_argument('--metric', default='sharpe',
                    choices=['sharpe', 'sortino', 'calmar', 'cagr', 'profit_factor'])
    ap.add_argument('--min-trades', type=int, default=30, help='min trades per window')
    ap.add_argument('--max-dd', type=float, default=0.35, help='max drawdown allowed, e.g. 0.35')
    ap.add_argument('--train-years', type=float, default=3)
    ap.add_argument('--test-years', type=float, default=1)
    ap.add_argument('--warmup-years', type=float, default=1, help='skip first year (MA200 warm-up)')
    ap.add_argument('--start', help='e.g. 2020-01-01: only trade/evaluate from this date on '
                                    '(older data is still used to warm up indicators)')
    ap.add_argument('--anchored', action='store_true', help='expanding instead of rolling train window')
    ap.add_argument('--holdout-start', help='e.g. 2025-01-01: never used for tuning, evaluated once at the end')
    ap.add_argument('--cost-bps', type=float, default=5.0)
    ap.add_argument('--capital', type=float, default=100_000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    index = None if args.index == 'all' else args.index
    prices = load_prices(index)
    bench = load_benchmark('SPY')
    dates = pd.DatetimeIndex(prices['Date'].unique()).sort_values()
    data_end = dates.max() + pd.Timedelta(days=1)
    tune_end = pd.Timestamp(args.holdout_start) if args.holdout_start else data_end
    windows, folds = make_windows(dates, tune_end, args.warmup_years,
                                  args.train_years, args.test_years, args.anchored,
                                  pd.Timestamp(args.start) if args.start else None)
    if not folds:
        raise SystemExit('Not enough history for one train+test fold. '
                         'Fetch more data or lower --train-years.')

    space = filtered_space(args.only, SPACES[args.space])
    combos = grid(space) if args.method == 'grid' else random_sample(space, args.n_iter, args.seed)
    sim = {'cost_per_side': args.cost_bps / 10_000,
           'initial_capital': args.capital}
    base = StrategyParams().to_dict()
    cfg = {'index': index, 'windows': windows, 'sim': sim, 'base': base}

    print(f'Universe {args.index}: {prices["Ticker"].nunique()} tickers, {dates.min().date()} .. {dates.max().date()}')
    print(f'{len(folds)} walk-forward folds, {len(combos)} parameter sets, {args.workers} workers')
    for f in folds:
        print(f'  fold {f["fold"]}: train {f["train"][0].date()}..{f["train"][1].date()}  '
              f'test {f["test"][0].date()}..{f["test"][1].date()}')

    rows: list[dict] = []
    jobs = list(enumerate(combos))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cfg,)) as ex:
            for r in tqdm(ex.map(_evaluate, jobs, chunksize=4), total=len(jobs)):
                rows.extend(r)
    else:
        _init(cfg)
        for job in tqdm(jobs):
            rows.extend(_evaluate(job))

    res = pd.DataFrame(rows)
    errors = res[res['window'] == 'error']
    res = res[res['window'] != 'error'].copy()
    if len(errors):
        print(f'WARNING: {errors["combo_id"].nunique()} parameter sets failed, e.g. {errors["error"].iloc[0]}')
    res['score'] = [score(m, args.metric, args.min_trades, args.max_dd) for m in res.to_dict('records')]

    # ---- walk-forward selection
    wf_rows = []
    for f in folds:
        tr = res[res['window'] == f'train_{f["fold"]}']
        te = res[res['window'] == f'test_{f["fold"]}'].set_index('combo_id')
        valid = tr[np.isfinite(tr['score'])]
        if valid.empty:
            continue
        best = valid.sort_values('score', ascending=False).iloc[0]
        t = te.loc[best['combo_id']]
        wf_rows.append({'fold': f['fold'], 'test_start': f['test'][0].date(), 'test_end': f['test'][1].date(),
                        'train_score': best['score'], 'test_score': t['score'],
                        'test_cagr': t['cagr'], 'test_max_dd': t['max_drawdown'],
                        'test_sharpe': t['sharpe'], 'test_trades': t['trades'],
                        **{k: best[k] for k in PARAM_NAMES}})
    wf = pd.DataFrame(wf_rows)

    # ---- leaderboard over the full tuning period + fold consistency
    full = res[res['window'] == 'full'].set_index('combo_id')
    tr_scores = res[res['window'].str.startswith('train_')].pivot(index='combo_id', columns='window', values='sharpe')
    te_scores = res[res['window'].str.startswith('test_')].pivot(index='combo_id', columns='window', values='sharpe')
    full['folds_train_sharpe_median'] = tr_scores.median(axis=1)
    full['folds_test_sharpe_positive'] = (te_scores > 0).mean(axis=1)
    leaderboard = full.sort_values('score', ascending=False)

    out = OUTPUT_DIR / f'tuning_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / 'all_results.csv', index=False)
    leaderboard.to_csv(out / 'leaderboard.csv')
    wf.to_csv(out / 'walk_forward.csv', index=False)

    lines = [f'TUNING  universe={args.index}  metric={args.metric}  sets={len(combos)}  '
             f'min_trades={args.min_trades}  max_dd={args.max_dd}  cost={args.cost_bps}bps/side  '
             f'space={args.space}', '']

    fs = FeatureStore(prices, bench, load_membership(index))
    if len(wf):
        oos = stitched_oos(fs, wf, base, sim, args.capital)
        empty = pd.DataFrame(columns=['net_return', 'pnl', 'bars_held'])
        om = compute_metrics(oos, empty)
        b = bench.set_index('Date')['Close'].reindex(oos.index).ffill()
        bm = compute_metrics(b / b.iloc[0], empty)
        lines += ['WALK-FORWARD (best-in-train params traded in the following test window)',
                  wf[['fold', 'test_start', 'test_end', 'entry_indicator', 'entry_length', 'entry_threshold',
                      'exit_threshold', 'test_cagr', 'test_max_dd', 'test_sharpe', 'test_trades']]
                  .to_string(index=False, float_format=lambda x: f'{x:.3f}'),
                  '',
                  f'Stitched OOS:  CAGR {om["cagr"] * 100:.2f}%  MaxDD {om["max_drawdown"] * 100:.2f}%  '
                  f'Sharpe {om["sharpe"]:.2f}  Calmar {om["calmar"]:.2f}',
                  'Benchmarks, same period:',
                  *format_benchmarks(benchmark_metrics(bench, oos.index, sim['cost_per_side'])),
                  'Could the walk-forward result be luck? (vs SPY buy & hold; the yearly re-selection is one process, so trials = 1)',
                  *format_luck(luck_summary(oos, spy_buy_hold(bench, oos.index), 1)),
                  '']
        pd.DataFrame({'oos_equity': oos}).to_csv(out / 'walk_forward_equity.csv')

    valid_lb = leaderboard[np.isfinite(leaderboard['score'])]
    if valid_lb.empty:
        lines.append('No parameter set satisfied the constraints. Loosen --min-trades / --max-dd.')
    else:
        best = valid_lb.iloc[0]
        best_params = StrategyParams.from_dict({k: best[k] for k in PARAM_NAMES})
        top_cols = ['score', 'cagr', 'max_drawdown', 'sharpe', 'trades', 'win_rate', 'profit_factor',
                    'folds_train_sharpe_median', 'folds_test_sharpe_positive',
                    'entry_indicator', 'entry_length', 'entry_threshold', 'entry_mode', 'exit_threshold',
                    'trend_ma', 'market_ma', 'stop_adr_mult', 'max_hold_days']
        lines += ['TOP 15 over the full tuning period',
                  valid_lb[top_cols].head(15).to_string(float_format=lambda x: f'{x:.3f}'), '',
                  'BEST PARAMETERS', best_params.describe(), '',
                  f'Tuning period ({windows[0][1].date()} .. {windows[0][2].date()})',
                  format_metrics(best.to_dict())]
        payload = {'params': best_params.to_dict(), 'metric': args.metric, 'trials': len(combos), 'tuning_metrics':
                   {k: float(best[k]) for k in ['cagr', 'max_drawdown', 'sharpe', 'trades', 'win_rate']}}
        if args.holdout_start:
            res_h = simulate_portfolio(fs, generate_trades(fs, best_params), args.holdout_start, None,
                                       max_positions=best_params.max_positions, max_new_per_day=best_params.max_new_per_day, **sim)
            hm = compute_metrics(res_h.equity, res_h.trades, res_h.exposure)
            lines += ['', f'HOLDOUT ({args.holdout_start} .. end) - never seen during tuning', format_metrics(hm)]
            res_f = simulate_portfolio(fs, generate_trades(fs, best_params), windows[0][1], None,
                                       max_positions=best_params.max_positions, max_new_per_day=best_params.max_new_per_day, **sim)
            lines += ['', f'COULD THE BEST SET BE LUCK? (tuning + holdout, vs SPY buy & hold, deflated for {len(combos)} tested sets)',
                      *format_luck(luck_summary(res_f.equity, spy_buy_hold(bench, res_f.equity.index), len(combos)))]
            payload['holdout_metrics'] = {k: float(hm[k]) for k in ['cagr', 'max_drawdown', 'sharpe', 'trades', 'win_rate']}
        (out / 'best_params.json').write_text(json.dumps(payload, indent=2, default=str))

    report = '\n'.join(lines)
    print('\n' + report)
    (out / 'report.txt').write_text(report)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
