"""Today's signals for a rule set (run after `python3 -m backtest.fetch --update`).

    python3 -m backtest.signals                                  # strategies/willr.json
    python3 -m backtest.signals --params strategies/candidate_breadth.json --holding AAPL MSFT

Prints new entry candidates for the next open (ranked like the backtest fills its slots)
and, for tickers you hold, whether the exit rule fired at the last close.
Stops / trailing stops are price levels - set them in your broker.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.data import ROOT_DIR, load_benchmark, load_prices
from backtest.features import FeatureStore
from backtest.membership import load_membership
from backtest.strategy import StrategyParams, build_signals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', default=str(ROOT_DIR / 'strategies' / 'willr.json'))
    ap.add_argument('--index', default='sp500')
    ap.add_argument('--holding', nargs='*', default=[], help='tickers you currently hold')
    args = ap.parse_args()

    p = StrategyParams.load(args.params)
    idx = p.universe or (None if args.index == 'all' else args.index)
    fs = FeatureStore(load_prices(idx), load_benchmark('SPY'), load_membership(idx))
    entry, exit_, rank = build_signals(fs, p)
    last_date = fs.dates.max()
    today = fs.dates == last_date
    adr = fs.get('adr', p.adr_length)

    print(p.describe())
    print(f'\nLast close: {pd.Timestamp(last_date).date()}\n')

    idx = np.where(today & entry)[0]
    cand = pd.DataFrame({'ticker': fs.tickers[idx], 'close': fs.close[idx], 'rank': rank[idx],
                         'adr_pct': adr[idx] / fs.close[idx] * 100})
    if p.stop_adr_mult:
        cand['stop_dist'] = p.stop_adr_mult * adr[idx]
    cand = cand.sort_values('rank').reset_index(drop=True)
    print(f'ENTRY candidates for the next open: {len(cand)} (take the top free slots of {p.max_positions})')
    print(cand.head(25).to_string(float_format=lambda x: f'{x:.2f}') if len(cand) else '  none')

    if args.holding:
        held = {t.upper() for t in args.holding}
        idx = np.where(today & np.isin(fs.tickers, list(held)))[0]
        print('\nHOLDINGS')
        for i in idx:
            print(f'  {fs.tickers[i]:<6} close {fs.close[i]:8.2f}  exit signal: {"SELL next open" if exit_[i] else "hold"}')
        missing = held - set(fs.tickers[idx])
        if missing:
            print(f'  no data for: {", ".join(sorted(missing))}')
        if p.max_hold_days:
            print(f'  remember the time stop: sell after {p.max_hold_days} trading days')


if __name__ == '__main__':
    main()
