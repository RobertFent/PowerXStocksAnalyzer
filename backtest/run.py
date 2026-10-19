"""Backtest ONE strategy and write a full report.

    python3 -m backtest.run --params strategies/current.json
    python3 -m backtest.run --params output/tuning_<ts>/best_params.json --start 2016-01-01
    python3 -m backtest.run --set entry_indicator=rsi entry_length=2 entry_threshold=10 max_positions=5

Writes output/backtest_<ts>/{trades.csv, equity.csv, summary.txt}
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from backtest.data import OUTPUT_DIR, load_benchmark, load_prices
from backtest.engine import generate_trades, simulate_portfolio
from backtest.features import FeatureStore
from backtest.metrics import compute_metrics, format_metrics, yearly_returns
from backtest.strategy import StrategyParams


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        k, v = item.split('=', 1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', help='json file with strategy params (e.g. best_params.json)')
    ap.add_argument('--set', nargs='*', default=[], help='override params: key=value ...')
    ap.add_argument('--index', default='sp500', help="sp500 | nasdaq100 | sp100 | all")
    ap.add_argument('--start', help='first entry date (indicators still use earlier data)')
    ap.add_argument('--end')
    ap.add_argument('--cost-bps', type=float, default=5.0, help='cost per side in basis points (commission + slippage)')
    ap.add_argument('--capital', type=float, default=100_000)
    args = ap.parse_args()

    params = StrategyParams.load(args.params) if args.params else StrategyParams()
    if args.set:
        params = StrategyParams.from_dict({**params.to_dict(), **parse_overrides(args.set)})

    prices = load_prices(None if args.index == 'all' else args.index)
    bench = load_benchmark('SPY')
    fs = FeatureStore(prices, bench)

    trades_all = generate_trades(fs, params)
    res = simulate_portfolio(fs, trades_all, args.start, args.end,
                             max_positions=params.max_positions,
                             cost_per_side=args.cost_bps / 10_000,
                             initial_capital=args.capital)
    m = compute_metrics(res.equity, res.trades, res.exposure)

    b = bench.set_index('Date')['Close'].reindex(res.equity.index).ffill()
    bench_eq = b / b.iloc[0] * args.capital
    bm = compute_metrics(bench_eq, res.trades.iloc[0:0], None)

    period = f'{res.equity.index[0].date()} .. {res.equity.index[-1].date()}'
    lines = [
        f'Strategy backtest  |  universe: {args.index}  |  {period}',
        f'max positions: {params.max_positions}, cost: {args.cost_bps:g} bps/side, capital: ${args.capital:,.0f}',
        '',
        params.describe(),
        '',
        'PORTFOLIO',
        format_metrics(m),
        f'  Signals skipped (no free slot): {res.skipped}',
        '',
        'BENCHMARK SPY buy & hold',
        f'  CAGR {bm["cagr"] * 100:.2f}%  |  Max DD {bm["max_drawdown"] * 100:.2f}%  |  Sharpe {bm["sharpe"]:.2f}',
        '',
        'YEARLY RETURNS (strategy vs SPY)',
    ]
    yr = pd.DataFrame({'strategy': yearly_returns(res.equity), 'spy': yearly_returns(bench_eq)})
    for y, row in yr.iterrows():
        lines.append(f'  {y}  {row["strategy"] * 100:7.2f}%  {row["spy"] * 100:7.2f}%')
    lines += ['', 'EXIT REASONS']
    if len(res.trades):
        grp = res.trades.groupby('exit_reason')['net_return'].agg(['count', 'mean'])
        for reason, row in grp.iterrows():
            lines.append(f'  {reason:<11} {int(row["count"]):6d} trades, avg {row["mean"] * 100:6.2f}%')
    lines += ['', 'ALL SIGNALS (unconstrained, gross, before portfolio limits)',
              f'  {len(trades_all)} trades, avg {trades_all["gross_return"].mean() * 100:.2f}%, '
              f'win rate {(trades_all["gross_return"] > 0).mean() * 100:.1f}%']
    report = '\n'.join(lines)
    print(report)

    out = OUTPUT_DIR / f'backtest_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / 'trades.csv', index=False)
    pd.DataFrame({'equity': res.equity, 'exposure': res.exposure, 'spy': bench_eq}).to_csv(out / 'equity.csv')
    (out / 'summary.txt').write_text(report)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
