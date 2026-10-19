"""Rule-based long-only swing strategy definition.

ENTRY (evaluated at the close of day t, executed at next open by default)
    oscillator(entry_length) < entry_threshold
        entry_mode='cross': and it was >= threshold the day before (first day below)
        entry_mode='level': any day below while flat
    AND Close > SMA(trend_ma)            if trend_ma > 0
    AND SPY Close > SPY SMA(market_ma)   if market_ma > 0
    AND Close >= min_price AND 20d avg dollar volume >= min_dollar_volume
    AND (optional) >= breadth_min of the universe oversold the same day

EXIT (first that triggers)
    stop:   Low <= entry - stop_adr_mult * ADR(14)          -> filled at the stop (or open on a gap)
    trail:  Low <= highest high - trail_adr_mult * ADR(14)  -> filled at the stop (or open on a gap)
    signal: exit_rule 'osc' (oscillator > exit_threshold) or 'prev_high' (close > yesterday's high)
            at close -> next open
    time:   held >= max_hold_days at close                    -> next open

PORTFOLIO
    max_positions equal-weight slots; when more signals than free slots the best
    `rank_by` (oversold | volatility) get them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from backtest.features import FeatureStore

ADR_LENGTH = 14
DOLLAR_VOLUME_LENGTH = 20


@dataclass(frozen=True)
class StrategyParams:
    entry_indicator: str = 'willr'      # rsi | willr | stoch | ibs
    entry_length: int = 4
    entry_threshold: float = 15.0       # 0..100 scale, low = oversold (WILLR -85 == 15)
    entry_mode: str = 'cross'           # cross | level
    exit_rule: str = 'osc'              # osc: oscillator > exit_threshold | prev_high: close > yesterday's high
                                        # | none: only stops / time exit
    exit_indicator: str = ''            # '' = same as entry
    exit_length: int = 0                # 0 = same as entry
    exit_threshold: float = 50.0
    trend_ma: int = 0                   # 0 = off, e.g. 200 -> Close > SMA200
    market_ma: int = 0                  # 0 = off, e.g. 200 -> SPY > SMA200
    stop_adr_mult: float = 0.0          # 0 = no stop
    trail_adr_mult: float = 0.0         # 0 = off; trailing stop = highest high since entry - mult * ADR(14)
    max_hold_days: int = 0              # 0 = no time stop
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000
    execution: str = 'next_open'        # next_open | close
    breadth_min: float = 0.0            # 0 = off; e.g. 0.10 -> only buy when >= 10% of the universe
                                        # is oversold the same day (broad pullback, not a single-stock dip)
    rank_by: str = 'oversold'           # which signals get the free slots: oversold | volatility
    max_positions: int = 10             # equal-weight slots, each gets equity / max_positions

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
        name = {'rsi': 'RSI', 'willr': 'Williams %R', 'stoch': 'Stoch %K', 'ibs': 'IBS'}
        def lvl(ind: str, v: float) -> str:
            return f'{v - 100:g}' if ind == 'willr' else f'{v:g}'
        ind, xind = self.entry_indicator, self.exit_ind
        e_len = '' if ind == 'ibs' else f'({self.entry_length})'
        x_len = '' if xind == 'ibs' else f'({self.exit_len})'
        when = 'first close' if self.entry_mode == 'cross' else 'any close'
        lines = [f'ENTRY: {when} with {name[ind]}{e_len} < {lvl(ind, self.entry_threshold)}']
        if self.trend_ma:
            lines.append(f'       and Close > SMA({self.trend_ma})')
        if self.market_ma:
            lines.append(f'       and SPY > SMA({self.market_ma})')
        lines.append(f'       and price >= ${self.min_price:g}, avg $vol(20) >= ${self.min_dollar_volume / 1e6:g}M')
        lines.append('       -> buy ' + ('next day at the open' if self.execution == 'next_open' else 'at the close'))
        sell = 'next open' if self.execution == 'next_open' else 'at the close'
        exits = []
        if self.exit_rule == 'osc':
            exits.append(f'{name[xind]}{x_len} > {lvl(xind, self.exit_threshold)} at close -> sell {sell}')
        elif self.exit_rule == 'prev_high':
            exits.append(f"close > yesterday's high -> sell {sell}")
        if self.stop_adr_mult:
            exits.append(f'stop loss at entry - {self.stop_adr_mult:g} x ADR({ADR_LENGTH})')
        if self.trail_adr_mult:
            exits.append(f'trailing stop at highest high - {self.trail_adr_mult:g} x ADR({ADR_LENGTH})')
        lines.append('EXIT:  ' + (exits[0] if exits else 'time stop only'))
        lines += [f'       or {e}' for e in exits[1:]]
        if self.max_hold_days:
            lines.append(f'       or after {self.max_hold_days} trading days')
        if self.breadth_min:
            lines.insert(1, f'       and >= {self.breadth_min * 100:g}% of the universe is below that level the same day')
        lines.append(f'SLOTS: {self.max_positions} equal-weight positions, '
                     + ('most oversold signals first' if self.rank_by == 'oversold'
                        else 'most volatile (ADR%) signals first'))
        return '\n'.join(lines)


def build_signals(fs: FeatureStore, p: StrategyParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (entry_signal, exit_signal, rank) arrays aligned with fs rows.
    rank: lower = better candidate when more signals than free slots (most oversold first)."""
    osc = fs.oscillator(p.entry_indicator, p.entry_length)
    below = osc < p.entry_threshold
    if p.entry_mode == 'cross':
        prev = fs.shift(osc)
        entry = below & (prev >= p.entry_threshold)
    elif p.entry_mode == 'level':
        entry = below.copy()
    else:
        raise ValueError(f'unknown entry_mode {p.entry_mode!r}')

    if p.trend_ma:
        entry &= fs.close > fs.get('sma', p.trend_ma)  # NaN compares False -> no trade during warm-up
    if p.market_ma:
        entry &= fs.get('market_ok', p.market_ma)
    if p.min_price:
        entry &= fs.close >= p.min_price
    if p.min_dollar_volume:
        entry &= fs.get('dollar_volume', DOLLAR_VOLUME_LENGTH) >= p.min_dollar_volume
    entry &= ~np.isnan(fs.get('adr', ADR_LENGTH))
    if p.breadth_min:
        entry &= fs.breadth(p.entry_indicator, p.entry_length, p.entry_threshold) >= p.breadth_min

    if p.exit_rule == 'osc':
        exit_ = fs.oscillator(p.exit_ind, p.exit_len) > p.exit_threshold
    elif p.exit_rule == 'prev_high':
        exit_ = fs.close > fs.shift(fs.high)
    elif p.exit_rule == 'none':
        exit_ = np.zeros(len(fs), dtype=bool)
    else:
        raise ValueError(f'unknown exit_rule {p.exit_rule!r}')
    if p.rank_by == 'volatility':
        rank = -(fs.get('adr', ADR_LENGTH) / fs.close)
    elif p.rank_by == 'oversold':
        rank = osc
    else:
        raise ValueError(f'unknown rank_by {p.rank_by!r}')
    return entry, exit_, rank
