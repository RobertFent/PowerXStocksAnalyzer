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


def test_new_oscillators_are_causal_and_bounded():
    p = synthetic_prices(3)
    fs = FeatureStore(p)
    for name, n in [('mfi', 14), ('crsi', 100), ('uo', 0), ('macd', 100), ('cci', 100), ('bbp', 20)]:
        v = fs.oscillator(name, n)
        ok = v[~np.isnan(v)]
        assert len(ok) > 1000, name
        if name != 'bbp':
            assert ok.min() >= 0 and ok.max() <= 100, name
    # causality: changing the future must not change past values
    p2 = p.copy()
    m = p2['Date'] >= '2018-01-01'
    p2.loc[m, ['Open', 'High', 'Low', 'Close']] *= 1.5
    fs2 = FeatureStore(p2)
    past = (fs.dates < np.datetime64('2018-01-01'))
    for name, n in [('mfi', 14), ('crsi', 100), ('uo', 0), ('macd', 100), ('cci', 100), ('bbp', 20)]:
        assert np.allclose(fs.oscillator(name, n)[past], fs2.oscillator(name, n)[past], equal_nan=True), name


def test_describe_all_indicators():
    from backtest.features import OSCILLATORS
    for ind in OSCILLATORS:
        text = StrategyParams(entry_indicator=ind, entry_length=14, trend_ma=200, market_ma=200,
                              stop_adr_mult=3, trail_adr_mult=2, max_hold_days=10, breadth_min=0.1).describe()
        assert text.startswith('BUY') and 'SELL' in text and 'POSITIONS' in text


def test_planner_form_conversion():
    from backtest.app import params_from_form
    p = params_from_form({'entry_indicator': 'willr', 'entry_length': '10', 'entry_level': '-90',
                          'exit_rule': 'osc', 'exit_level': '-60', 'trend_ma': '200', 'max_positions': '5'})
    assert p.entry_threshold == 10 and p.exit_threshold == 40 and p.max_positions == 5
    try:
        params_from_form({'entry_indicator': 'rsi', 'exit_rule': 'none'})
        assert False, 'plan without any exit must be rejected'
    except ValueError:
        pass


def test_momentum_features_and_membership():
    p = synthetic_prices(5)
    fs = FeatureStore(p)
    one = p[p['Ticker'] == 'T02'].reset_index(drop=True)
    rows = fs.tickers == 'T02'
    # highest high of the PREVIOUS 20 days (today excluded)
    ref = one['High'].rolling(20).max().shift(1).to_numpy()
    assert np.allclose(fs.get('hh', 20)[rows], ref, equal_nan=True)
    # momentum with skip: close[t-21] / close[t-21-126] - 1
    refm = (one['Close'].shift(21) / one['Close'].shift(147) - 1).to_numpy()
    assert np.allclose(fs.momentum(126, 21)[rows], refm, equal_nan=True)
    # cross-sectional rank: 1..5 on every day where all 5 have data
    rk = fs.mom_rank(126, 0)
    d = pd.DataFrame({'d': fs.dates, 'r': rk}).dropna()
    counts = d.groupby('d')['r'].agg(['min', 'max', 'count'])
    full = counts[counts['count'] == 5]
    assert (full['min'] == 1).all() and (full['max'] == 5).all()
    # rebalance: exactly one True per month per ticker
    rb = fs.rebalance_mask('monthly')
    per_month = pd.Series(rb[rows]).groupby(pd.DatetimeIndex(fs.dates[rows]).to_period('M')).sum()
    assert (per_month == 1).all()
    # membership: T02 eligible only in 2017, T03 never (not listed), others always
    # T03 left the index before our data starts (never eligible); a ticker with no record at all is always eligible
    mem = pd.DataFrame({'Ticker': ['T00', 'T01', 'T02', 'T03'],
                        'Start': pd.to_datetime(['2010-01-01', '2010-01-01', '2017-01-01', '2000-01-01']),
                        'End': pd.to_datetime([None, None, '2017-12-31', '2009-12-31'])})
    fs2 = FeatureStore(p, membership=mem)
    yr = pd.DatetimeIndex(fs2.dates).year
    assert fs2.eligible[(fs2.tickers == 'T02') & (yr == 2017)].all()
    assert not fs2.eligible[(fs2.tickers == 'T02') & (yr != 2017)].any()
    assert not fs2.eligible[fs2.tickers == 'T03'].any() and fs2.eligible[fs2.tickers == 'T00'].all()
    assert fs2.eligible[fs2.tickers == 'T04'].all()  # no record -> treated as eligible
    # rank only among eligible: T03 must never have a rank
    assert np.isnan(fs2.mom_rank(126, 0)[fs2.tickers == 'T03']).all()
    # strategy only buys eligible stocks
    t = generate_trades(fs2, StrategyParams(entry_indicator='rs', entry_length=63, entry_threshold=2,
                                            exit_rule='rank', exit_threshold=3, rebalance='monthly', min_dollar_volume=0))
    assert len(t) > 0 and 'T03' not in set(t['ticker'])
    assert (t.loc[t['ticker'] == 'T02', 'signal_date'].dt.year == 2017).all()


def test_breakout_rules_handmade():
    rows = [(10, 10.5, 9.5, 10)] * 12 + [(10, 11, 9.9, 10.8)] + [(11, 11.5, 10.8, 11.2)] * 3 + [(11, 11.2, 9.0, 9.2), (9.2, 9.5, 9.0, 9.3)]
    p = _frame(rows)
    fs = FeatureStore(p)
    fs._cache[('adr', 14)] = np.full(len(p), 1.0)
    t = generate_trades(fs, StrategyParams(entry_indicator='breakout', entry_length=10, exit_rule='channel',
                                           exit_length=5, min_price=0, min_dollar_volume=0))
    assert len(t) == 1
    assert t.loc[0, 'entry_date'] == p['Date'][13] and t.loc[0, 'entry_price'] == 11.0  # breakout close 10.8 > 10.5
    assert t.loc[0, 'exit_reason'] == 'signal' and t.loc[0, 'exit_price'] == 9.2       # close 9.2 < 5-day low 9.9


def test_rulebook_mechanics():
    # OBE: entry only if open <= signal close * (1 - gap); add: second block 5% below entry
    rows = [
        (10.0, 10.5, 9.5, 10.4),   # 0
        (10.0, 10.5, 9.5, 9.6),    # 1 ibs 10 -> signal, close 9.6
        (9.5, 9.8, 9.0, 9.1),      # 2 open 9.5 = 1% below -> OBE 2% rejects; without OBE: entry 9.5
        (9.1, 9.3, 8.9, 9.0),      # 3 low 8.9 <= 9.5*0.95=9.025 -> add at 9.025 (ibs 25, no exit)
        (9.2, 10.0, 9.1, 9.9),     # 4 ibs 89 -> exit signal
        (10.0, 10.2, 9.8, 10.1),   # 5 exit at open 10.0
        (10.0, 10.2, 9.8, 10.1),
    ]
    p = _frame(rows)
    fs = FeatureStore(p)
    fs._cache[('adr', 14)] = np.full(len(p), 1.0)
    base = dict(entry_indicator='ibs', entry_length=0, entry_threshold=20, exit_threshold=50, min_price=0, min_dollar_volume=0)
    assert len(generate_trades(fs, StrategyParams(**base, obe_min_gap=0.02))) == 0
    t = generate_trades(fs, StrategyParams(**base, add_drop_pct=0.05))
    assert len(t) == 1 and t.loc[0, 'entry_price'] == 9.5 and t.loc[0, 'add_idx'] == 3
    assert abs(t.loc[0, 'add_price'] - 9.025) < 1e-6 and t.loc[0, 'exit_price'] == 10.0
    res = simulate_portfolio(fs, t, max_positions=2, cost_per_side=0.0, initial_capital=1000)
    tr = res.trades.iloc[0]
    assert abs(tr['cost'] - (500 + 500 / 9.5 * 9.025)) < 1e-6          # second block, same share count
    assert abs(tr['proceeds'] - 2 * (500 / 9.5) * 10.0) < 1e-6
    # daily limit: 3 candidates on one day, max 2 new per day
    p2 = synthetic_prices(6)
    fs2 = FeatureStore(p2)
    q = StrategyParams(entry_indicator='ibs', entry_length=0, entry_threshold=40, exit_threshold=60,
                       min_price=0, min_dollar_volume=0, max_positions=10, max_new_per_day=2)
    r2 = simulate_portfolio(fs2, generate_trades(fs2, q), max_positions=10, max_new_per_day=2)
    assert r2.trades.groupby('entry_date').size().max() <= 2
    # symbol_roi ranking: no look-ahead -> first 3 trades of each symbol rank 0
    t3 = generate_trades(fs2, q.with_(rank_by='symbol_roi'))
    first = t3.sort_values('entry_date').groupby('ticker').head(3)
    assert (first['rank'] == 0).all() and (t3['rank'] != 0).any()


def test_significance_sanity():
    from backtest.significance import bootstrap_excess, deflated_sharpe, excess_returns
    rng = np.random.default_rng(3)
    idx = pd.bdate_range('2016-01-01', periods=2000)
    bench = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, 2000)), idx)
    same = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, 2000)), idx)
    better = pd.Series(np.cumprod(1 + rng.normal(0.0020, 0.01, 2000)), idx)
    assert bootstrap_excess(excess_returns(same, bench), n=300)['p_positive'] < 0.9
    ex = excess_returns(better, bench)
    assert bootstrap_excess(ex, n=300)['p_positive'] > 0.95
    assert deflated_sharpe(ex, 1)['prob'] > 0.95
    assert deflated_sharpe(ex, 1)['prob'] > deflated_sharpe(ex, 500)['prob']  # more trials -> less confidence


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


def test_spreads_pricing_and_backtest():
    from backtest.spreads import call_price, spread_value, _expiry, SpreadParams, backtest
    import numpy as np, pandas as pd
    s = np.array([100.0, 110.0, 90.0])
    assert np.allclose(call_price(s, 100.0, np.zeros(3), np.full(3, 0.2)), [0.0, 10.0, 0.0])   # intrinsic at expiry
    v = spread_value(s, 100.0, 110.0, np.full(3, 0.25), np.full(3, 0.2))
    assert (v >= 0).all() and (v <= 10).all() and v[1] > v[0] > v[2]
    assert _expiry(pd.Timestamp('2026-10-20'), 60).weekday() == 4                             # a Friday
    rng = np.random.default_rng(1)
    dates = pd.bdate_range('2015-01-01', periods=900)
    close = 200 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    df = pd.DataFrame({'Date': dates, 'Close': close, 'AdjClose': close, 'VIX': 16.0})
    p = SpreadParams(offset=0.017, width=10, dte=60, target=0.99, cadence='monthly', trend_ma=200, strike_mode='offset', fixed_debit=0.0)
    assert p.concurrent() == 2
    eq, tr = backtest(df, p, '2016-01-01', None, 100000)
    assert len(tr) > 10 and (eq > 0).all() and eq.iloc[0] == 100000
    assert tr['debit'].between(0, 10).all() and tr['exit'].between(0, 10).all()
    assert set(tr['exit_reason']) <= {'target', 'expiry', 'stop', 'open'}
    closed = tr[tr.exit_reason != 'open']
    assert ((closed['exit_date'] <= closed['expiry'] + pd.Timedelta(days=3))).all()
    # cash accounting: final equity = capital + sum of trade P&L (open ones at their mark)
    assert abs(eq.iloc[-1] - (100000 + tr['pnl'].sum())) < 1e-6
    # strikes chosen by debit: every spread costs what you pay, strikes move with the market
    q = SpreadParams(width=10, dte=60, cadence='expiry', trend_ma=0, strike_mode='debit', fixed_debit=4.95)
    eq2, tr2 = backtest(df, q, '2016-01-01', None, 100000)
    assert len(tr2) > 10 and (tr2['debit'] == 4.95).all() and tr2['k1'].nunique() > 5
    assert tr2['entry_date'].dt.weekday.isin([3, 4]).all()                       # third Fridays (or the Thursday before)
    from backtest.spreads import third_friday
    assert third_friday(2026, 10) == pd.Timestamp('2026-10-16')
    # jobs build a whitelisted command for it
    from backtest.jobs import build_command
    argv, _ = build_command('spreads', {'offset': '1.7', 'width': '10', 'dte': '60', 'target': '99', 'debit': '4.95',
                                        'cadence': 'monthly', 'trend_ma': '200', 'stop': '0'})
    assert '--debit' in argv and argv[argv.index('--offset') + 1] == '0.0170'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
