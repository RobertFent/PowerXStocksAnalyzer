"""Backtest engine.

Step 1  generate_trades():  walks every ticker bar by bar and produces ALL trades
        the rules would take (one open position per ticker at a time).
Step 2  simulate_portfolio(): replays those trades through a portfolio with
        limited capital / max positions, costs, and daily mark-to-market, which
        gives the equity curve used for Sharpe, drawdown, CAGR, ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.features import FeatureStore
from backtest.strategy import StrategyParams, build_signals

try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

EXIT_REASONS = {0: 'signal', 1: 'stop', 2: 'time', 3: 'open', 4: 'trail'}  # open = still open at data end


@njit(cache=True)
def _trade_loop(open_, high, low, close, first_row, entry_sig, exit_sig, adr,
                stop_mult, trail_mult, max_hold, next_open, obe_gap, add_drop, add_skip):
    n = close.shape[0]
    out_entry = np.empty(n, np.int64)
    out_exit = np.empty(n, np.int64)
    out_sig = np.empty(n, np.int64)
    out_ep = np.empty(n, np.float64)
    out_xp = np.empty(n, np.float64)
    out_reason = np.empty(n, np.int64)
    out_add_idx = np.empty(n, np.int64)
    out_add_px = np.empty(n, np.float64)
    k = 0
    add_idx = -1
    add_px = np.nan

    in_pos = False
    pending_entry = False
    pending_exit = False
    pending_reason = 0
    entry_i = 0
    sig_i = 0
    entry_price = 0.0
    stop = -np.inf
    trail = -np.inf
    peak = 0.0

    for i in range(n):
        if first_row[i]:
            if in_pos:  # previous ticker ran out of data while in a trade
                out_entry[k] = entry_i; out_exit[k] = i - 1; out_sig[k] = sig_i
                out_ep[k] = entry_price; out_xp[k] = close[i - 1]; out_reason[k] = 3
                out_add_idx[k] = add_idx; out_add_px[k] = add_px
                k += 1
            in_pos = False
            pending_entry = False
            pending_exit = False

        # 1) exits scheduled for this open
        if in_pos and pending_exit:
            out_entry[k] = entry_i; out_exit[k] = i; out_sig[k] = sig_i
            out_ep[k] = entry_price; out_xp[k] = open_[i]; out_reason[k] = pending_reason
            out_add_idx[k] = add_idx; out_add_px[k] = add_px
            k += 1
            in_pos = False
            pending_exit = False

        # 2) entries scheduled for this open (OBE: only if the open is far enough below the signal close)
        just_exited = False
        if pending_entry and not in_pos:
            if obe_gap <= 0.0 or open_[i] <= close[sig_i] * (1.0 - obe_gap):
                in_pos = True
                entry_i = i
                entry_price = open_[i]
                stop = entry_price - stop_mult * adr[sig_i] if stop_mult > 0 else -np.inf
                trail = -np.inf
                peak = entry_price
                add_idx = -1
                add_px = np.nan
        pending_entry = False

        # 3) intraday stop loss / trailing stop (whichever is higher)
        level = stop if stop >= trail else trail
        if in_pos and low[i] <= level:
            fill = level
            if open_[i] < level:  # gapped below the stop
                fill = open_[i]
            out_entry[k] = entry_i; out_exit[k] = i; out_sig[k] = sig_i
            out_ep[k] = entry_price; out_xp[k] = fill
            out_reason[k] = 1 if stop >= trail else 4
            out_add_idx[k] = add_idx; out_add_px[k] = add_px
            k += 1
            in_pos = False

        # 3b) add a second block when the price fell add_drop below the entry (limit order, not on the entry day)
        if in_pos and add_drop > 0.0 and add_idx < 0 and i > entry_i:
            lvl = entry_price * (1.0 - add_drop)
            near_stop = level > -np.inf and lvl <= level * (1.0 + add_skip)
            if low[i] <= lvl and not near_stop:
                add_idx = i
                add_px = lvl if open_[i] > lvl else open_[i]

        # 4) close based exits
        if in_pos:
            held = i - entry_i
            reason = -1
            if exit_sig[i]:
                reason = 0
            elif max_hold > 0 and held >= max_hold:
                reason = 2
            if reason >= 0:
                if next_open:
                    pending_exit = True
                    pending_reason = reason
                else:
                    out_entry[k] = entry_i; out_exit[k] = i; out_sig[k] = sig_i
                    out_ep[k] = entry_price; out_xp[k] = close[i]; out_reason[k] = reason
                    out_add_idx[k] = add_idx; out_add_px[k] = add_px
                    k += 1
                    in_pos = False
                    just_exited = True

        # trailing stop for the next bar: highest high since entry - trail_mult * ADR
        if in_pos and trail_mult > 0:
            if high[i] > peak:
                peak = high[i]
            t = peak - trail_mult * adr[i]
            if t > trail:
                trail = t

        # 5) new entry signal at the close (only when flat)
        if not in_pos and not just_exited and entry_sig[i]:
            sig_i = i
            if next_open:
                # only if the ticker has a next bar
                if i + 1 < n and not first_row[i + 1]:
                    pending_entry = True
            else:
                in_pos = True
                entry_i = i
                entry_price = close[i]
                stop = entry_price - stop_mult * adr[i] if stop_mult > 0 else -np.inf
                trail = -np.inf
                peak = entry_price
                add_idx = -1
                add_px = np.nan

    if in_pos:
        out_entry[k] = entry_i; out_exit[k] = n - 1; out_sig[k] = sig_i
        out_ep[k] = entry_price; out_xp[k] = close[n - 1]; out_reason[k] = 3
        out_add_idx[k] = add_idx; out_add_px[k] = add_px
        k += 1

    return (out_entry[:k], out_exit[:k], out_sig[:k], out_ep[:k], out_xp[:k], out_reason[:k],
            out_add_idx[:k], out_add_px[:k])


def generate_trades(fs: FeatureStore, p: StrategyParams) -> pd.DataFrame:
    """All trades the rules generate, before any capital / position limits."""
    entry, exit_, rank = build_signals(fs, p)
    adr = fs.get('adr', p.adr_length)
    e, x, s, ep, xp, r, ai, apx = _trade_loop(
        fs.open, fs.high, fs.low, fs.close, fs.first_row,
        np.ascontiguousarray(entry, dtype=np.bool_),
        np.ascontiguousarray(exit_, dtype=np.bool_),
        np.nan_to_num(adr, nan=0.0),
        float(p.stop_adr_mult), float(p.trail_adr_mult), int(p.max_hold_days), p.execution == 'next_open',
        float(p.obe_min_gap), float(p.add_drop_pct), float(p.add_skip_near_stop))
    trades = pd.DataFrame({
        'ticker': fs.tickers[e],
        'signal_date': fs.dates[s],
        'entry_date': fs.dates[e],
        'exit_date': fs.dates[x],
        'entry_price': ep,
        'exit_price': xp,
        'exit_reason': pd.Categorical.from_codes(r, list(EXIT_REASONS.values())),
        'bars_held': x - e,
        'rank': rank[s],
        'entry_idx': e,
        'exit_idx': x,
        'add_idx': ai,
        'add_price': apx,
    })
    trades['gross_return'] = trades['exit_price'] / trades['entry_price'] - 1.0
    if p.rank_by == 'symbol_roi':
        trades['rank'] = _symbol_roi_rank(trades)
    return trades.sort_values(['entry_date', 'rank']).reset_index(drop=True)


def _symbol_roi_rank(trades: pd.DataFrame, min_trades: int = 3) -> np.ndarray:
    """Rank = minus the average gross return of the symbol's EARLIER completed trades (no look-ahead);
    symbols with fewer than `min_trades` earlier trades rank neutral (0)."""
    t = trades.sort_values('entry_date')
    rank = np.zeros(len(trades))
    for ticker, grp in t.groupby('ticker', sort=False):
        idx = grp.index.to_numpy()
        entries = grp['entry_date'].to_numpy()
        exits = grp['exit_date'].to_numpy()
        rets = grp['gross_return'].to_numpy()
        for j in range(len(idx)):
            done = exits < entries[j]
            if done.sum() >= min_trades:
                rank[idx[j]] = -rets[done].mean()
    return rank


@dataclass
class PortfolioResult:
    equity: pd.Series          # daily equity, index = date
    exposure: pd.Series        # fraction of equity invested at close
    trades: pd.DataFrame       # trades actually taken (with net return, pnl)
    skipped: int               # signals skipped because no slot / cash


def simulate_portfolio(fs: FeatureStore, trades: pd.DataFrame,
                       start=None, end=None,
                       max_positions: int = 10,
                       cost_per_side: float = 0.0005,
                       initial_capital: float = 100_000.0,
                       max_new_per_day: int = 0) -> PortfolioResult:
    """Equal-weight slots: each new position gets equity/max_positions (if cash allows).
    Only trades with entry_date in [start, end] are taken; positions still open at
    `end` are closed at that day's close. Trades with an add (add_idx >= 0) buy a second
    block of the same size at add_price on that day if cash allows."""
    all_dates = np.unique(fs.dates)
    if start is not None:
        all_dates = all_dates[all_dates >= np.datetime64(pd.Timestamp(start))]
    if end is not None:
        all_dates = all_dates[all_dates <= np.datetime64(pd.Timestamp(end))]
    if len(all_dates) == 0:
        raise ValueError('no trading days in the requested window')
    first_d, last_d = all_dates[0], all_dates[-1]

    t = trades[(trades['entry_date'] >= first_d) & (trades['entry_date'] <= last_d)]
    # group candidate trades by entry date (already sorted by entry_date, rank)
    cand_by_date: dict = {}
    for row in t.itertuples(index=False):
        cand_by_date.setdefault(row.entry_date, []).append(row)

    close = fs.close
    dates = fs.dates
    cash = float(initial_capital)
    positions: list[dict] = []
    taken: list[dict] = []
    skipped = 0
    equity_vals = np.empty(len(all_dates))
    expo_vals = np.empty(len(all_dates))
    prev_equity = cash

    for di, d in enumerate(all_dates):
        is_last = di == len(all_dates) - 1
        # exits on day d
        still = []
        for pos in positions:
            tr = pos['trade']
            if tr.exit_date <= d and tr.exit_reason != 'open':
                proceeds = pos['shares'] * tr.exit_price * (1 - cost_per_side)
                cash += proceeds
                pos['exit_value'] = proceeds
                pos['exit_date'] = tr.exit_date
                pos['exit_price'] = tr.exit_price
                pos['exit_reason'] = tr.exit_reason
                taken.append(pos)
            else:
                still.append(pos)
        positions = still

        # adds on day d (second block at the add price)
        for pos in positions:
            tr = pos['trade']
            add_i = getattr(tr, 'add_idx', -1)
            if add_i >= 0 and not pos['added'] and dates[add_i] <= d:
                pos['added'] = True
                block = pos['shares0'] * tr.add_price * (1 + cost_per_side)
                if block <= cash:
                    cash -= block
                    pos['shares'] += pos['shares0']
                    pos['cost'] += block
                    pos['add_price'] = tr.add_price

        # entries on day d
        new_today = 0
        for tr in cand_by_date.get(d, ()):
            if len(positions) >= max_positions or (max_new_per_day and new_today >= max_new_per_day):
                skipped += 1
                continue
            alloc = min(prev_equity / max_positions, cash)
            if alloc <= 0.5 * prev_equity / max_positions:
                skipped += 1
                continue
            shares = alloc / (tr.entry_price * (1 + cost_per_side))
            cash -= alloc
            new_today += 1
            positions.append({'trade': tr, 'shares': shares, 'shares0': shares, 'cost': alloc,
                              'ptr': tr.entry_idx, 'added': False, 'add_price': np.nan})

        # mark to market at close of d
        invested = 0.0
        for pos in positions:
            tr = pos['trade']
            ptr = pos['ptr']
            while ptr + 1 <= tr.exit_idx and dates[ptr + 1] <= d:
                ptr += 1
            pos['ptr'] = ptr
            invested += pos['shares'] * close[ptr]

        if is_last:  # liquidate at the end of the window
            for pos in positions:
                tr = pos['trade']
                px = close[pos['ptr']]
                proceeds = pos['shares'] * px * (1 - cost_per_side)
                cash += proceeds
                pos.update(exit_value=proceeds, exit_date=dates[pos['ptr']],
                           exit_price=px, exit_reason='window_end')
                taken.append(pos)
            positions = []
            equity = cash
        else:
            equity = cash + invested
        equity_vals[di] = equity
        expo_vals[di] = invested / equity if equity > 0 else 0.0
        prev_equity = equity

    idx = pd.DatetimeIndex(all_dates)
    rows = []
    for pos in taken:
        tr = pos['trade']
        rows.append({
            'ticker': tr.ticker, 'signal_date': tr.signal_date, 'entry_date': tr.entry_date,
            'exit_date': pos['exit_date'], 'entry_price': tr.entry_price,
            'exit_price': pos['exit_price'], 'exit_reason': pos['exit_reason'],
            'bars_held': int(pos['ptr'] - tr.entry_idx) if pos['exit_reason'] == 'window_end' else tr.bars_held,
            'cost': pos['cost'], 'proceeds': pos['exit_value'], 'add_price': pos.get('add_price', np.nan),
        })
    taken_df = pd.DataFrame(rows, columns=['ticker', 'signal_date', 'entry_date', 'exit_date',
                                           'entry_price', 'exit_price', 'exit_reason',
                                           'bars_held', 'cost', 'proceeds', 'add_price'])
    taken_df['pnl'] = taken_df['proceeds'] - taken_df['cost']
    taken_df['net_return'] = taken_df['proceeds'] / taken_df['cost'] - 1.0
    taken_df = taken_df.sort_values('entry_date').reset_index(drop=True)
    return PortfolioResult(pd.Series(equity_vals, idx, name='equity'),
                           pd.Series(expo_vals, idx, name='exposure'),
                           taken_df, skipped)
