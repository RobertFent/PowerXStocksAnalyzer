"""Simple benchmarks every strategy has to beat.

    SPY buy & hold     always invested in SPY
    SPY timing         hold SPY only while it closed above its 200-day average the day
                       before, otherwise cash (0%). Costs are charged on each switch.

If a stock-picking strategy cannot beat "SPY timing", its edge comes from the trend
filter, not from picking stocks.
"""
from __future__ import annotations

import pandas as pd

from backtest.metrics import compute_metrics

_EMPTY = pd.DataFrame(columns=['net_return', 'pnl', 'bars_held'])


def spy_buy_hold(bench: pd.DataFrame, index: pd.DatetimeIndex, capital: float = 1.0) -> pd.Series:
    close = bench.set_index('Date')['Close'].reindex(index).ffill()
    return close / close.iloc[0] * capital


def spy_timing(bench: pd.DataFrame, index: pd.DatetimeIndex, ma: int = 200,
               cost_per_side: float = 0.0005, capital: float = 1.0) -> pd.Series:
    close = bench.set_index('Date')['Close']
    invested = (close > close.rolling(ma, min_periods=ma).mean()).astype(float).shift(1).fillna(0.0)
    switches = invested.diff().abs().fillna(invested)
    daily = invested * close.pct_change().fillna(0.0) - switches * cost_per_side
    daily = daily.reindex(index).fillna(0.0)
    daily.iloc[0] = 0.0
    return capital * (1 + daily).cumprod()


def benchmark_metrics(bench: pd.DataFrame, index: pd.DatetimeIndex,
                      cost_per_side: float = 0.0005) -> dict[str, dict]:
    return {
        'SPY buy & hold': compute_metrics(spy_buy_hold(bench, index), _EMPTY),
        'SPY timing (>SMA200)': compute_metrics(spy_timing(bench, index, cost_per_side=cost_per_side), _EMPTY),
    }


def format_benchmarks(bm: dict[str, dict]) -> list[str]:
    return [f'  {name:<22} CAGR {m["cagr"] * 100:6.2f}%  |  Max DD {m["max_drawdown"] * 100:7.2f}%  |  '
            f'Sharpe {m["sharpe"]:.2f}  |  Calmar {m["calmar"]:.2f}' for name, m in bm.items()]
