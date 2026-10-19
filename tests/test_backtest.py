"""Sanity tests for the backtest engine (no network needed).

    python3 -m pytest tests/        or        python3 tests/test_backtest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import generate_trades, simulate_portfolio  # noqa: E402
from backtest.features import FeatureStore  # noqa: E402
from backtest.metrics import compute_metrics  # noqa: E402
from backtest.strategy import StrategyParams  # noqa: E402


def synthetic_prices(n_tickers=20, start='2015-01-01', end='2020-12-31', seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for t in range(n_tickers):
        r = rng.normal(0.0004, 0.02, len(dates))
        close = 50 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.005, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, len(dates))))
        frames.append(pd.DataFrame({'Date': dates, 'Ticker': f'T{t:02d}', 'Open': open_, 'High': high,
                                    'Low': low, 'Close': close, 'Volume': 1_000_000}))
    return pd.concat(frames, ignore_index=True)


def synthetic_benchmark(prices: pd.DataFrame) -> pd.DataFrame:
    b = prices.groupby('Date')['Close'].mean().reset_index()
    b['Ticker'] = 'SPY'
    for c in ('Open', 'High', 'Low'):
        b[c] = b['Close']
    b['Volume'] = 0
    return b


def test_rsi_matches_reference():
    p = synthetic_prices(2)
    fs = FeatureStore(p)
    rsi = fs.get('rsi', 4)
    one = p[p['Ticker'] == 'T01'].reset_index(drop=True)
    d = one['Close'].diff()
    ag = d.clip(lower=0).ewm(alpha=1 / 4, adjust=False, min_periods=4).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / 4, adjust=False, min_periods=4).mean()
    ref = (100 * ag / (ag + al)).to_numpy()
    ours = rsi[fs.tickers == 'T01']
    assert np.allclose(ours[10:], ref[10:], equal_nan=True)
    # no leakage across tickers: first rows of T01 must be NaN
    assert np.isnan(ours[:4]).all()


def test_willr_scale():
    p = synthetic_prices(1)
    fs = FeatureStore(p)
    w = fs.get('willr', 4)
    hh = p['High'].rolling(4).max()
    ll = p['Low'].rolling(4).min()
    ref = ((hh - p['Close']) / (hh - ll) * -100 + 100).to_numpy()
    assert np.allclose(w, ref, equal_nan=True)


def test_no_lookahead():
    """Changing prices after a cutoff must not change trades that closed before it."""
    p = synthetic_prices(10)
    cutoff = pd.Timestamp('2018-01-01')
    p2 = p.copy()
    m = p2['Date'] >= cutoff
    for c in ('Open', 'High', 'Low', 'Close'):
        p2.loc[m, c] *= 1.3
    params = StrategyParams(trend_ma=100, stop_adr_mult=3, max_hold_days=10)
    t1 = generate_trades(FeatureStore(p), params)
    t2 = generate_trades(FeatureStore(p2), params)
    a = t1[t1['exit_date'] < cutoff].reset_index(drop=True)
    b = t2[t2['exit_date'] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) > 20


def test_trade_rules_handmade():
    # 1 ticker, IBS signal: low close in range -> entry next open; exit when IBS > 50
    rows = [
        # O,    H,    L,    C
        (10.0, 10.5, 9.5, 10.4),  # 0 ibs 90
        (10.0, 10.5, 9.5, 9.6),   # 1 ibs 10  -> signal (cross below 20)
        (9.7, 10.0, 9.4, 9.5),    # 2 entry @9.7, ibs 16.7 (hold)
        (9.6, 10.2, 9.5, 10.1),   # 3 ibs 85.7 -> exit signal
        (10.3, 10.6, 10.0, 10.5),  # 4 exit @10.3 open
        (10.5, 10.6, 10.0, 10.1),  # 5 ibs 16.7 -> new signal
        (10.0, 10.1, 8.0, 8.5),   # 6 entry @10.0, stop hit intraday
        (8.5, 9.0, 8.4, 8.9),
    ]
    p = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'])
    p['Date'] = pd.bdate_range('2020-01-01', periods=len(p))
    p['Ticker'] = 'X'
    p['Volume'] = 1e9
    fs = FeatureStore(p)
    params = StrategyParams(entry_indicator='ibs', entry_length=0, entry_threshold=20, exit_threshold=50,
                            stop_adr_mult=1.0, min_price=0, min_dollar_volume=0)
    # ADR(14) needs 14 rows - patch it for this tiny example
    fs._cache[('adr', 14)] = np.full(len(p), 1.0)
    t = generate_trades(fs, params)
    assert len(t) == 2, t
    assert t.loc[0, 'entry_price'] == 9.7 and t.loc[0, 'exit_price'] == 10.3
    assert t.loc[0, 'exit_reason'] == 'signal'
    assert t.loc[1, 'entry_price'] == 10.0 and t.loc[1, 'exit_price'] == 9.0  # stop = 10 - 1*ADR
    assert t.loc[1, 'exit_reason'] == 'stop'


def _frame(rows):
    p = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'], dtype=float)
    p['Date'] = pd.bdate_range('2020-01-01', periods=len(p))
    p['Ticker'] = 'X'
    p['Volume'] = 1e9
    return p


def test_trailing_stop_and_prev_high_exit():
    rows = [
        (10, 10.5, 9.5, 10.4),   # 0 ibs 90
        (10, 10.5, 9.5, 9.6),    # 1 ibs 10 -> signal
        (10, 11, 9.9, 10.9),     # 2 entry @10, close 10.9 > prev high 10.5 -> prev_high exit next open
        (11, 13, 10.9, 12.8),    # 3 peak 13
        (12.8, 12.9, 11.5, 11.8),  # 4 trail = 13 - 1 = 12 -> hit, filled at 12
        (11.8, 12, 11.7, 11.9),
    ]
    p = _frame(rows)
    base = dict(entry_indicator='ibs', entry_length=0, entry_threshold=20, min_price=0, min_dollar_volume=0)
    fs = FeatureStore(p)
    fs._cache[('adr', 14)] = np.full(len(p), 1.0)
    t = generate_trades(fs, StrategyParams(**base, exit_rule='none', trail_adr_mult=1.0))
    assert t.loc[0, 'exit_reason'] == 'trail' and t.loc[0, 'exit_price'] == 12.0, t
    t = generate_trades(fs, StrategyParams(**base, exit_rule='prev_high'))
    assert t.loc[0, 'exit_reason'] == 'signal' and t.loc[0, 'exit_price'] == 11.0, t


def test_portfolio_single_trade_accounting():
    rows = [(10, 10, 10, 10)] * 3 + [(10, 11, 10, 11), (12, 12, 12, 12), (12, 12, 12, 12)]
    p = pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'], dtype=float)
    p['Date'] = pd.bdate_range('2020-01-01', periods=len(p))
    p['Ticker'] = 'X'
    p['Volume'] = 1e9
    fs = FeatureStore(p)
    trades = pd.DataFrame([{'ticker': 'X', 'signal_date': p['Date'][2], 'entry_date': p['Date'][3],
                            'exit_date': p['Date'][4], 'entry_price': 10.0, 'exit_price': 12.0,
                            'exit_reason': 'signal', 'bars_held': 1, 'rank': 1.0,
                            'entry_idx': 3, 'exit_idx': 4}])
    res = simulate_portfolio(fs, trades, max_positions=2, cost_per_side=0.0, initial_capital=1000)
    # half the capital in, +20% -> +10% total
    assert abs(res.equity.iloc[-1] - 1100) < 1e-9
    assert abs(res.equity.iloc[3] - 1050) < 1e-9  # marked at close 11
    assert abs(res.trades['net_return'].iloc[0] - 0.2) < 1e-12


def test_end_to_end_runs():
    p = synthetic_prices(30)
    fs = FeatureStore(p, synthetic_benchmark(p))
    params = StrategyParams(trend_ma=200, market_ma=200, stop_adr_mult=3, max_hold_days=10)
    trades = generate_trades(fs, params)
    res = simulate_portfolio(fs, trades, '2016-01-01', None, max_positions=5)
    m = compute_metrics(res.equity, res.trades, res.exposure)
    assert m['trades'] > 0 and np.isfinite(m['sharpe'])
    assert (res.exposure <= 1.0001).all()
    # never more than 5 positions open at once
    events = pd.concat([pd.Series(1, res.trades['entry_date']), pd.Series(-1, res.trades['exit_date'])])
    opened = events.groupby(level=0).sum().cumsum()
    assert opened.max() <= 5


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
