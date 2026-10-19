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


def _with(common: dict, subs: list[dict]) -> list[dict]:
    return [{**sub, **common} for sub in subs]


# 1) broad search over all oscillators, filters and exits
DEFAULT = _with({
    'entry_mode': ['cross', 'level'],
    'trend_ma': [0, 100, 200],
    'market_ma': [0, 200],
    'stop_adr_mult': [0, 2, 3, 4],
    'max_hold_days': [0, 5, 10, 20],
}, _OSC)

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

SPACES = {'default': DEFAULT, 'breadth': BREADTH, 'exits': EXITS}

# values tried by `backtest.analyze` when changing one parameter at a time
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
