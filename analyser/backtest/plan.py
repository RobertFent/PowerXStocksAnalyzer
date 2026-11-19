"""Write down a trading plan as a rule set - by answering questions, no coding.

    python3 -m backtest.plan

Asks for entry, filters, exits and position size, shows the rules in plain English
and saves them to strategies/<name>.json. Then run:

    python3 -m backtest.report --params strategies/<name>.json

Press Enter to accept the suggested value in [brackets].
"""
from __future__ import annotations

import json
import re

from backtest.data import ROOT_DIR
from backtest.strategy import StrategyParams

INDICATORS = [
    # key, name, length question (None = no length), default length, oversold level text, default entry, default exit
    ('rsi', 'RSI', 'RSI length in days', 4, 'RSI level (0-100, e.g. 20)', 20, 55),
    ('willr', 'Williams %R', 'Williams %R length in days', 10, 'Williams %R level (-100 to 0, e.g. -90)', -90, -50),
    ('stoch', 'Stochastic %K (slow)', 'Stochastic length in days', 14, 'Stochastic level (0-100, e.g. 20)', 20, 60),
    ('crsi', 'Connors RSI', None, 100, 'Connors RSI level (0-100, e.g. 10)', 10, 70),
    ('mfi', 'Money Flow Index', 'MFI length in days', 14, 'MFI level (0-100, e.g. 20)', 20, 60),
    ('bbp', 'Bollinger %B (2 std)', 'Bollinger period in days', 20, 'Bollinger %B level (0 = lower band, e.g. 0)', 0, 60),
    ('cci', 'CCI (as rank vs. last year)', None, 252, 'CCI rank level (0-100, e.g. 5 = lowest 5%)', 5, 60),
    ('macd', 'MACD histogram (as rank vs. last year)', None, 252, 'MACD rank level (0-100, e.g. 5)', 5, 60),
    ('ibs', 'IBS: close near the low of the day', None, 0, 'IBS level (0-100, e.g. 15)', 15, 70),
    ('uo', 'Ultimate Oscillator', None, 0, 'Ultimate Oscillator level (0-100, e.g. 30)', 30, 60),
]


def ask(question: str, default, cast=str):
    while True:
        raw = input(f'{question} [{default}]: ').strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print('  Please enter a valid value.')


def yes(question: str, default: bool = True) -> bool:
    d = 'y' if default else 'n'
    return ask(f'{question} (y/n)', d).lower().startswith('y')


def main() -> None:
    print(__doc__.split('\n\n')[0], '\n')
    title = ask('Name of this plan', 'My trading plan')

    print('\nENTRY INDICATOR (buy when it shows "oversold")')
    for i, ind in enumerate(INDICATORS, 1):
        print(f'  {i:2}. {ind[1]}')
    choice = ask('Choose a number', 1, int)
    key, _, len_q, len_d, lvl_q, entry_d, exit_d = INDICATORS[max(1, min(choice, len(INDICATORS))) - 1]

    length = ask(len_q, len_d, int) if len_q else len_d
    entry = ask(f'Buy when the {lvl_q} is below', entry_d, float)
    first_day = yes('Only on the FIRST day it gets there? (n = any day it is there)', True)

    print('\nFILTERS')
    trend = ask('Stock must be above its N-day average (0 = no filter)', 200, int)
    market = ask('SPY must be above its N-day average (0 = no filter)', 200, int)

    print('\nEXIT')
    use_osc = yes('Sell when the indicator recovers above a level?', True)
    exit_lvl = ask('  Sell when it rises above', exit_d, float) if use_osc else exit_d
    stop = ask('Stop loss in multiples of the average daily range (0 = none, e.g. 3)', 3.0, float)
    trail = ask('Trailing stop in multiples of the average daily range (0 = none)', 0.0, float)
    days = ask('Sell after N trading days at the latest (0 = no limit)', 10, int)

    print('\nPOSITIONS')
    slots = ask('Maximum positions at the same time (each gets an equal share)', 10, int)

    def scale(v: float) -> float:  # Williams %R is stored as 0..100 like all others
        return v + 100 if key == 'willr' else v

    p = StrategyParams(entry_indicator=key, entry_length=length, entry_threshold=scale(entry),
                       entry_mode='cross' if first_day else 'level',
                       exit_rule='osc' if use_osc else 'none', exit_threshold=scale(exit_lvl),
                       trend_ma=trend, market_ma=market, stop_adr_mult=stop, trail_adr_mult=trail,
                       max_hold_days=days, max_positions=max(1, slots))
    print('\n' + p.describe() + '\n')
    if not yes('Save this plan?', True):
        print('Not saved.')
        return

    slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_') or 'plan'
    path = ROOT_DIR / 'strategies' / f'{slug}.json'
    path.write_text(json.dumps({'name': title,
                                'note': 'Created with python3 -m backtest.plan. Decided before testing - do not tweak after seeing results.',
                                'params': p.to_dict()}, indent=2) + '\n')
    rel = path.relative_to(ROOT_DIR)
    print(f'Saved to {rel}\n\nNext:\n  python3 -m backtest.report --params {rel}\n'
          f'Then open the newest report_*.html in the output folder.')


if __name__ == '__main__':
    main()
