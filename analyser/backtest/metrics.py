"""Performance metrics from an equity curve + list of trades."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                    exposure: pd.Series | None = None) -> dict:
    rets = equity.pct_change().dropna()
    years = max(len(equity) / TRADING_DAYS, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    vol = float(rets.std() * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = float(rets.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    n = len(trades)
    if n:
        r = trades['net_return']
        wins, losses = r[r > 0], r[r <= 0]
        gross_win = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
        gross_loss = -trades.loc[trades['pnl'] <= 0, 'pnl'].sum()
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float('inf')
        win_rate = float(len(wins) / n)
        avg_trade = float(r.mean())
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        avg_bars = float(trades['bars_held'].mean())
    else:
        profit_factor = win_rate = avg_trade = avg_win = avg_loss = avg_bars = 0.0

    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': mdd,
        'calmar': calmar,
        'exposure': float(exposure.mean()) if exposure is not None and len(exposure) else np.nan,
        'trades': n,
        'trades_per_year': n / years,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_bars_held': avg_bars,
    }


def yearly_returns(equity: pd.Series) -> pd.Series:
    by_year = equity.groupby(equity.index.year).last()
    first = equity.iloc[0]
    prev = pd.concat([pd.Series([first], index=[by_year.index[0] - 1]), by_year]).shift(1).iloc[1:]
    return by_year / prev - 1.0


def score(m: dict, metric: str = 'sharpe', min_trades: int = 30, max_dd: float = 0.35,
          min_trades_per_year: float = 0.0) -> float:
    """Ranking score for tuning. Parameter sets violating constraints get -inf."""
    if m['trades'] < min_trades or m['trades_per_year'] < min_trades_per_year:
        return float('-inf')
    if m['max_drawdown'] < -abs(max_dd):
        return float('-inf')
    v = m[metric]
    if metric == 'profit_factor' and not np.isfinite(v):
        v = 10.0
    return float(v)


def fmt_pct(x: float) -> str:
    return f'{x * 100:6.2f}%'


def format_metrics(m: dict) -> str:
    return '\n'.join([
        f'  Total return     {fmt_pct(m["total_return"])}',
        f'  CAGR             {fmt_pct(m["cagr"])}',
        f'  Max drawdown     {fmt_pct(m["max_drawdown"])}',
        f'  Sharpe / Sortino {m["sharpe"]:6.2f} / {m["sortino"]:.2f}',
        f'  Calmar           {m["calmar"]:6.2f}',
        f'  Exposure         {fmt_pct(m["exposure"])}',
        f'  Trades           {m["trades"]:6d}  ({m["trades_per_year"]:.0f}/year)',
        f'  Win rate         {fmt_pct(m["win_rate"])}',
        f'  Avg trade        {fmt_pct(m["avg_trade"])}   (win {fmt_pct(m["avg_win"])}, loss {fmt_pct(m["avg_loss"])})',
        f'  Profit factor    {m["profit_factor"]:6.2f}',
        f'  Avg days held    {m["avg_bars_held"]:6.1f}',
    ])
