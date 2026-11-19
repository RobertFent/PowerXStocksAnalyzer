"""Search spaces for `python3 -m backtest.tune --space <name>` - edit freely.

A space is a list of sub-spaces; each sub-space maps param -> candidate values.
Grid search = union of the cartesian products, random search samples from them.
Oscillators use a 0..100 scale, low = oversold (Williams %R -90 == 10).
`--only key=v1,v2` narrows (or adds) values on top of the chosen space.
"""

_OSC = [
    {'entry_indicator': ['willr'], 'entry_length': [2, 3, 4, 5, 7, 10, 14],
     'entry_threshold': [5, 10, 15, 20, 25], 'exit_threshold': [40, 50, 60, 70, 80]},
    {'entry_indicator': ['rsi'], 'entry_length': [2, 3, 4, 5, 7],
     'entry_threshold': [5, 10, 15, 20, 25, 30], 'exit_threshold': [40, 50, 60, 70]},
    {'entry_indicator': ['stoch'], 'entry_length': [5, 7, 10, 14],
     'entry_threshold': [5, 10, 15, 20], 'exit_threshold': [40, 50, 60, 70, 80]},
    {'entry_indicator': ['ibs'], 'entry_length': [0],
     'entry_threshold': [5, 10, 15, 20, 25], 'exit_threshold': [50, 60, 70, 80]},
]

# additional oscillators (same 0..100 / low = oversold convention, see features.py)
_OSC_EXTRA = [
    {'entry_indicator': ['mfi'], 'entry_length': [3, 5, 7, 10, 14],
     'entry_threshold': [5, 10, 15, 20, 25], 'exit_threshold': [50, 60, 70, 80]},
    {'entry_indicator': ['bbp'], 'entry_length': [10, 20],
     'entry_threshold': [-10, 0, 5, 10, 20], 'exit_threshold': [50, 60, 80, 100]},
    {'entry_indicator': ['crsi'], 'entry_length': [50, 100],          # length = percent-rank window
     'entry_threshold': [5, 10, 15, 20], 'exit_threshold': [50, 60, 70, 80]},
    {'entry_indicator': ['uo'], 'entry_length': [0],
     'entry_threshold': [20, 25, 30, 35], 'exit_threshold': [50, 60, 70]},
    {'entry_indicator': ['macd'], 'entry_length': [50, 100, 252],     # length = percent-rank window
     'entry_threshold': [2, 5, 10, 15], 'exit_threshold': [50, 60, 70, 80]},
    {'entry_indicator': ['cci'], 'entry_length': [50, 100, 252],      # length = percent-rank window
     'entry_threshold': [2, 5, 10, 15], 'exit_threshold': [50, 60, 70, 80]},
]


def _with(common: dict, subs: list[dict]) -> list[dict]:
    return [{**sub, **common} for sub in subs]


_COMMON = {
    'entry_mode': ['cross', 'level'],
    'trend_ma': [0, 100, 200],
    'market_ma': [0, 200],
    'stop_adr_mult': [0, 2, 3, 4],
    'max_hold_days': [0, 5, 10, 20],
}

# 1) broad search over all oscillators, filters and exits
DEFAULT = _with(_COMMON, _OSC + _OSC_EXTRA)

# 1b) only the additional oscillators (MFI, Bollinger %B, Connors RSI, Ultimate Osc, MACD, CCI)
INDICATORS = _with(_COMMON, _OSC_EXTRA)

# 2) only buy broad pullbacks (many stocks oversold at once) with fewer, bigger positions
BREADTH = _with({
    'entry_mode': ['cross', 'level'],
    'trend_ma': [100, 200],
    'market_ma': [0, 200],
    'breadth_min': [0.05, 0.1, 0.15, 0.2, 0.3],
    'max_positions': [3, 5, 10],
    'stop_adr_mult': [0, 4],
    'max_hold_days': [0, 10, 20],
}, _OSC[:3])

# 3) different exits for the oscillator entries that held up best
EXITS = [{
    'entry_indicator': ['willr', 'stoch'],
    'entry_length': [7, 10, 14],
    'entry_threshold': [10, 15, 20],
    'entry_mode': ['level'],
    'trend_ma': [200],
    'market_ma': [0, 200],
    'exit_rule': ['osc', 'prev_high', 'none'],
    'exit_threshold': [40, 60, 80],
    'trail_adr_mult': [0, 1.5, 2, 3, 4],
    'stop_adr_mult': [0, 3, 4],
    'max_hold_days': [0, 5, 10, 20],
}]

# 4) momentum / trend following: relative-strength ranking with periodic rebalancing, and breakouts
MOMENTUM = [
    {
        'entry_indicator': ['rs'],
        'entry_length': [63, 126, 252],          # momentum lookback (3, 6, 12 months)
        'skip_days': [0, 21],                    # ignore the last month (12-1 momentum)
        'entry_threshold': [10, 20, 30, 50],     # buy if among the top N stocks
        'rebalance': ['weekly', 'monthly'],
        'exit_rule': ['rank', 'ma', 'none'],
        'exit_threshold': [30, 50, 100],         # rank exit: sell when worse than this rank
        'exit_length': [50, 100, 200],           # ma exit: sell below this average
        'trend_ma': [0, 200],
        'market_ma': [0, 200],
        'trail_adr_mult': [0, 4, 6],
        'max_positions': [5, 10, 20],
        'rank_by': ['momentum'],
    },
    {
        'entry_indicator': ['breakout'],
        'entry_length': [50, 100, 252],          # n-day high
        'exit_rule': ['channel', 'ma', 'none'],
        'exit_length': [20, 50, 100],            # channel low / moving average
        'trend_ma': [0, 200],
        'market_ma': [0, 200],
        'trail_adr_mult': [0, 3, 4, 6],
        'max_hold_days': [0, 60],
        'max_positions': [10, 20],
        'rank_by': ['momentum'],
    },
]

SPACES = {'default': DEFAULT, 'indicators': INDICATORS, 'breadth': BREADTH, 'exits': EXITS, 'momentum': MOMENTUM}

# values tried by `backtest.analyze` when changing one parameter at a time
NEIGHBORHOOD_MOMENTUM = {
    'entry_length': [42, 63, 126, 189, 252],
    'skip_days': [0, 10, 21],
    'entry_threshold': [5, 10, 20, 30, 50],
    'rebalance': ['weekly', 'monthly'],
    'exit_rule': ['rank', 'ma', 'channel', 'none'],
    'exit_threshold': [20, 30, 50, 100],
    'exit_length': [20, 50, 100, 200],
    'trend_ma': [0, 100, 200],
    'market_ma': [0, 200],
    'trail_adr_mult': [0, 3, 4, 6],
    'max_hold_days': [0, 60, 120],
    'max_positions': [5, 10, 20],
}

NEIGHBORHOOD = {
    'entry_length': [2, 3, 4, 5, 7, 10, 14],
    'entry_threshold': [5, 10, 15, 20, 25],
    'exit_threshold': [40, 50, 60, 70, 80],
    'entry_mode': ['cross', 'level'],
    'exit_rule': ['osc', 'prev_high'],
    'trend_ma': [0, 100, 200],
    'market_ma': [0, 200],
    'breadth_min': [0, 0.1, 0.2, 0.3],
    'stop_adr_mult': [0, 2, 3, 4],
    'trail_adr_mult': [0, 2, 3, 4],
    'max_hold_days': [0, 5, 10, 20],
    'max_positions': [3, 5, 10, 20],
}
