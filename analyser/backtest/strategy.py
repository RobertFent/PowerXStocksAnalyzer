"""Rule-based long-only strategy definition. Two families share the same fields:

MEAN REVERSION (entry_indicator = an oscillator)
    oscillator(entry_length) < entry_threshold
        entry_mode='cross': and it was >= threshold the day before (first day below)
        entry_mode='level': any day below while flat
MOMENTUM / TREND (entry_indicator = 'rs' or 'breakout')
    'rs':       on rebalance days (rebalance = daily | weekly | monthly) the stock ranks in the top
                entry_threshold of all stocks by momentum = return over entry_length days,
                skipping the last skip_days (classic 12-1 momentum: 252 and 21)
    'breakout': close > highest high of the previous entry_length days (today excluded)

ENTRY also requires (evaluated at the close of day t, executed at next open by default)
    Close > SMA(trend_ma)            if trend_ma > 0 (or Close <= SMA * (1 - trend_below) if trend_below > 0)
    AND SPY Close > SPY SMA(market_ma)   if market_ma > 0
    AND Close >= min_price AND 20d avg dollar volume >= min_dollar_volume
    AND (optional) >= breadth_min of the universe oversold the same day

EXIT (first that triggers)
    stop:   Low <= entry - stop_adr_mult * ADR(14)          -> filled at the stop (or open on a gap)
    trail:  Low <= highest high - trail_adr_mult * ADR(14)  -> filled at the stop (or open on a gap)
    signal: at close -> next open; exit_rule is one of
            'osc'       oscillator(exit_len) > exit_threshold
            'prev_high' close > yesterday's high
            'channel'   close < lowest low of the previous exit_length days
            'ma'        close < SMA(exit_length)
            'rank'      on rebalance days: momentum rank worse than exit_threshold
            'none'      only stops / time exit
    time:   held >= max_hold_days at close                    -> next open

PORTFOLIO
    max_positions equal-weight slots; when more signals than free slots the best
    `rank_by` (oversold | volatility | momentum) get them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from backtest.features import FeatureStore

ADR_LENGTH = 14  # default lookback of the average daily range (stops, adds); per strategy: adr_length
DOLLAR_VOLUME_LENGTH = 20


@dataclass(frozen=True)
class StrategyParams:
    entry_indicator: str = 'willr'      # oscillators: rsi | willr | stoch | ibs | mfi | bbp | crsi | uo | macd | cci
                                        # momentum:    rs (relative-strength rank) | breakout (n-day high)
    entry_length: int = 4               # oscillator length / momentum lookback / breakout channel (days)
    entry_threshold: float = 15.0       # oscillator: 0..100, low = oversold (WILLR -85 == 15); rs: top N stocks
    entry_mode: str = 'cross'           # cross | level (oscillators only)
    exit_rule: str = 'osc'              # osc | prev_high | channel | ma | rank | none (see module docstring)
    exit_indicator: str = ''            # '' = same as entry
    exit_length: int = 0                # 0 = same as entry
    exit_threshold: float = 50.0
    trend_ma: int = 0                   # 0 = off, e.g. 200 -> Close > SMA200
    trend_below: float = 0.0            # if > 0: flip the trend filter: Close <= SMA(trend_ma) * (1 - trend_below), e.g. 0.15 = at least 15% below
    market_ma: int = 0                  # 0 = off, e.g. 200 -> SPY > SMA200
    stop_adr_mult: float = 0.0          # 0 = no stop
    trail_adr_mult: float = 0.0         # 0 = off; trailing stop = highest high since entry - mult * ADR(14)
    max_hold_days: int = 0              # 0 = no time stop
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000
    execution: str = 'next_open'        # next_open | close
    breadth_min: float = 0.0            # 0 = off; e.g. 0.10 -> only buy when >= 10% of the universe
                                        # is oversold the same day (broad pullback, not a single-stock dip)
    rank_by: str = 'oversold'           # which signals get the free slots: oversold | volatility | momentum
    max_positions: int = 10             # equal-weight slots, each gets equity / max_positions
    rebalance: str = 'daily'            # daily | weekly | monthly: when 'rs' entries and 'rank' exits are checked
    skip_days: int = 0                  # momentum: ignore the most recent n days (e.g. 21 for 12-1 momentum)
    universe: str = ''                  # '' = whatever --index says; e.g. 'sp100+nasdaq100'
    max_price: float = 0.0              # 0 = off; e.g. 250 -> only stocks priced <= $250
    min_history_days: int = 0           # 0 = off; e.g. 756 -> the stock needs 3 years of price history
    max_new_per_day: int = 0            # 0 = off; e.g. 3 -> at most 3 new positions per day
    obe_min_gap: float = 0.0            # 0 = off; e.g. 0.02 -> only enter if the open is >= 2% below the signal close
    add_drop_pct: float = 0.0           # 0 = off; e.g. 0.05 -> buy a second equal block if price falls 5% below entry
    add_skip_near_stop: float = 0.005   # skip the add if its price is within 0.5% of the stop
    adr_length: int = ADR_LENGTH        # days in the average daily range used for stops / trailing stops

    MOMENTUM_ENTRIES = ('rs', 'breakout')

    @property
    def is_momentum(self) -> bool:
        return self.entry_indicator in self.MOMENTUM_ENTRIES

    @property
    def exit_ind(self) -> str:
        return self.exit_indicator or self.entry_indicator

    @property
    def exit_len(self) -> int:
        return self.exit_length or self.entry_length

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyParams':
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in names and v is not None
                 and not (isinstance(v, float) and np.isnan(v))}
        types = {f.name: f.type for f in fields(cls)}
        for k, v in clean.items():
            t = types[k]
            if t == 'int' or t is int:
                clean[k] = int(v)
            elif t == 'float' or t is float:
                clean[k] = float(v)
            elif t == 'str' or t is str:
                clean[k] = '' if v is None else str(v)
        return cls(**clean)

    @classmethod
    def load(cls, path: str | Path) -> 'StrategyParams':
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data.get('params', data))

    def with_(self, **kw) -> 'StrategyParams':
        return replace(self, **kw)

    def describe(self) -> str:
        """The rules in plain English."""
        name = {'rsi': 'RSI', 'willr': 'Williams %R', 'stoch': 'Stochastic %K', 'ibs': 'IBS (close within the day\'s range)',
                'mfi': 'Money Flow Index', 'bbp': 'Bollinger %B', 'crsi': 'Connors RSI', 'uo': 'Ultimate Oscillator',
                'macd': 'MACD histogram rank', 'cci': 'CCI rank'}

        def lvl(ind: str, v: float) -> str:
            return f'{v - 100:g}' if ind == 'willr' else f'{v:g}'

        def label(ind: str, n: int) -> str:
            if ind in ('ibs', 'uo'):
                return name[ind]
            if ind in ('macd', 'cci', 'crsi'):
                return f'{name[ind]} ({n}-day window)'
            return f'{name[ind]}({n})'

        def months(days: int) -> str:
            if days >= 21 and days % 21 == 0:
                m = days // 21
                return f'{m} month' + ('s' if m > 1 else '')
            return f'{days} trading days'

        ind, xind = self.entry_indicator, self.exit_ind
        nxt = 'next day\'s open' if self.execution == 'next_open' else 'same close'
        when = {'daily': 'every day', 'weekly': 'on the first trading day of each week',
                'monthly': 'on the first trading day of each month'}[self.rebalance]
        buy = []
        if ind == 'rs':
            skip = f', ignoring the last {months(self.skip_days)}' if self.skip_days else ''
            buy.append(f'{when}: the stock is among the top {self.entry_threshold:g} of all stocks by '
                       f'performance over the last {months(self.entry_length)}{skip}')
        elif ind == 'breakout':
            buy.append(f'the close is above the highest high of the previous {self.entry_length} trading days'
                       + (f' (checked {when})' if self.rebalance != 'daily' else ''))
        else:
            how = 'drops below' if self.entry_mode == 'cross' else 'is below'
            buy.append(f'{label(ind, self.entry_length)} {how} {lvl(ind, self.entry_threshold)}')
            if self.breadth_min:
                buy.append(f'at least {self.breadth_min * 100:g}% of all stocks are that oversold on the same day')
        if self.trend_ma and self.trend_below:
            buy.append(f'the stock is at least {self.trend_below * 100:g}% below its {self.trend_ma}-day average (deeply beaten down)')
        elif self.trend_ma:
            buy.append(f'the stock is above its {self.trend_ma}-day average')
        if self.market_ma:
            buy.append(f'SPY is above its {self.market_ma}-day average')
        price = f'price at least ${self.min_price:g}' + (f' and at most ${self.max_price:g}' if self.max_price else '')
        if self.min_dollar_volume:
            price += f', average daily trading value at least ${self.min_dollar_volume / 1e6:g}M'
        buy.append(price)
        if self.min_history_days:
            buy.append(f'at least {self.min_history_days / 252:g} years of price history')
        if self.universe:
            buy.append(f'listed in {self.universe.replace("+", " or ").replace("sp", "S&P ").replace("nasdaq", "Nasdaq ")}')
        sell = []
        if self.exit_rule == 'osc':
            sell.append(f'{label(xind, self.exit_len)} rises above {lvl(xind, self.exit_threshold)} '
                        f'-> sell at the {nxt}')
        elif self.exit_rule == 'prev_high':
            sell.append(f"the close is above yesterday's high -> sell at the {nxt}")
        elif self.exit_rule == 'channel':
            sell.append(f'the close is below the lowest low of the previous {self.exit_len} trading days -> sell at the {nxt}')
        elif self.exit_rule == 'ma':
            sell.append(f'the close is below its {self.exit_len}-day average -> sell at the {nxt}')
        elif self.exit_rule == 'rank':
            sell.append(f'{when}: the stock has dropped out of the top {self.exit_threshold:g} by performance -> sell at the {nxt}')
        if self.stop_adr_mult:
            sell.append(f'stop loss: entry price - {self.stop_adr_mult:g} x average daily range ({self.adr_length} days)')
        if self.trail_adr_mult:
            sell.append(f'trailing stop: highest high since entry - {self.trail_adr_mult:g} x average daily range')
        if self.max_hold_days:
            sell.append(f'after {self.max_hold_days} trading days -> sell at the {nxt}')
        rank_by = 'momentum' if (self.is_momentum and self.rank_by == 'oversold') else self.rank_by
        pick = {'oversold': 'the most oversold', 'volatility': 'the most volatile',
                'momentum': 'the strongest performers',
                'symbol_roi': 'the symbols with the best past results (average return of their earlier trades)'}[rank_by]
        lines = ['BUY when all of these are true at the close:']
        lines += [f'  - {b}' for b in buy]
        lines.append(f'  -> buy at the {nxt}'
                     + (f', but only if the open is at least {self.obe_min_gap * 100:g}% below the signal close (OBE)'
                        if self.obe_min_gap else ''))
        if self.add_drop_pct:
            lines.append(f'  - if the price falls {self.add_drop_pct * 100:g}% below the entry, buy a second block of the same '
                         f'size (skipped within {self.add_skip_near_stop * 100:g}% of the stop)')
        lines.append('SELL at the first of:')
        lines += [f'  - {x}' for x in (sell or ['never (no exit rule set)'])]
        lines.append(f'POSITIONS: up to {self.max_positions} at once, {100 / self.max_positions:g}% of the account each'
                     + (f', at most {self.max_new_per_day} new per day' if self.max_new_per_day else '')
                     + f'; if there are more signals than free spots, take {pick}')
        return '\n'.join(lines)


def filter_mask(fs: FeatureStore, p: StrategyParams) -> np.ndarray:
    """Rows that pass every entry condition EXCEPT the indicator itself (trend, market, liquidity, membership)."""
    ok = ~np.isnan(fs.get('adr', p.adr_length))
    if p.trend_ma and p.trend_below:
        ok &= fs.close <= fs.get('sma', p.trend_ma) * (1.0 - p.trend_below)   # deep below the average
    elif p.trend_ma:
        ok &= fs.close > fs.get('sma', p.trend_ma)  # NaN compares False -> no trade during warm-up
    if p.market_ma:
        ok &= fs.get('market_ok', p.market_ma)
    if p.min_price:
        ok &= fs.close >= p.min_price
    if p.min_dollar_volume:
        ok &= fs.get('dollar_volume', DOLLAR_VOLUME_LENGTH) >= p.min_dollar_volume
    if p.max_price:
        ok &= fs.close <= p.max_price
    if p.min_history_days:
        ok &= fs.bars_since_start() >= p.min_history_days
    if fs.eligible is not None:
        ok &= fs.eligible  # point-in-time index membership (no survivorship bias)
    return ok


def build_signals(fs: FeatureStore, p: StrategyParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (entry_signal, exit_signal, rank) arrays aligned with fs rows.
    rank: lower = better candidate when more signals than free slots."""
    if p.entry_indicator == 'rs':
        rk = fs.mom_rank(p.entry_length, p.skip_days)
        entry = (rk <= p.entry_threshold) & fs.rebalance_mask(p.rebalance)
        osc = None
    elif p.entry_indicator == 'breakout':
        entry = fs.close > fs.get('hh', p.entry_length)
        if p.rebalance != 'daily':
            entry &= fs.rebalance_mask(p.rebalance)
        osc = None
    else:
        osc = fs.oscillator(p.entry_indicator, p.entry_length)
        below = osc < p.entry_threshold
        if p.entry_mode == 'cross':
            entry = below & (fs.shift(osc) >= p.entry_threshold)
        elif p.entry_mode == 'level':
            entry = below.copy()
        else:
            raise ValueError(f'unknown entry_mode {p.entry_mode!r}')
        if p.breadth_min:
            entry &= fs.breadth(p.entry_indicator, p.entry_length, p.entry_threshold) >= p.breadth_min

    entry &= filter_mask(fs, p)

    if p.exit_rule == 'osc':
        if p.exit_ind in StrategyParams.MOMENTUM_ENTRIES:
            raise ValueError("exit_rule 'osc' needs an oscillator: set exit_indicator (e.g. 'rsi') for momentum entries")
        exit_ = fs.oscillator(p.exit_ind, p.exit_len) > p.exit_threshold
    elif p.exit_rule == 'prev_high':
        exit_ = fs.close > fs.shift(fs.high)
    elif p.exit_rule == 'channel':
        exit_ = fs.close < fs.get('ll', p.exit_len)
    elif p.exit_rule == 'ma':
        exit_ = fs.close < fs.get('sma', p.exit_len)
    elif p.exit_rule == 'rank':
        rk = fs.mom_rank(p.entry_length if p.entry_indicator == 'rs' else max(p.exit_len, 21), p.skip_days)
        exit_ = (np.nan_to_num(rk, nan=1e9) > p.exit_threshold) & fs.rebalance_mask(p.rebalance)
    elif p.exit_rule == 'none':
        exit_ = np.zeros(len(fs), dtype=bool)
    else:
        raise ValueError(f'unknown exit_rule {p.exit_rule!r}')

    if p.rank_by == 'volatility':
        rank = -(fs.get('adr', p.adr_length) / fs.close)
    elif p.rank_by == 'momentum' or (p.rank_by == 'oversold' and osc is None):
        n = p.entry_length if p.entry_indicator == 'rs' else 126
        rank = -np.nan_to_num(fs.momentum(n, p.skip_days), nan=-1e9)
    elif p.rank_by == 'oversold':
        rank = osc
    elif p.rank_by == 'symbol_roi':
        rank = osc if osc is not None else np.zeros(len(fs))  # replaced per trade in generate_trades
    else:
        raise ValueError(f'unknown rank_by {p.rank_by!r}')
    return entry, exit_, rank
