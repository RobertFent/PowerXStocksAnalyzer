"""python3 -m backtest  ->  overview of all commands."""

COMMANDS = [
    ('Getting data', [
        ('python3 -m backtest.fetch --historical', 'download ~14 years of daily prices incl. past S&P 500 members (first time)'),
        ('python3 -m backtest.fetch --update', 'add the latest days (run before signals)'),
        ('python3 -m backtest.symbols', 'refresh the index member lists from Wikipedia'),
        ('python3 -m backtest.membership [--check]', 'rebuild the S&P 500 membership history / check price coverage'),
    ]),
    ('Testing a plan (no coding)', [
        ('python3 -m backtest.app', 'workbench + planner in your browser: every step as a button (or double-click start.command / start.bat)'),
        ('python3 -m backtest.plan', 'same in the Terminal: answer questions -> strategies/<name>.json'),
        ('python3 -m backtest.report --params strategies/<name>.json', 'one-page HTML report with verdict and charts'),
        ('python3 -m backtest.compare [--params a.json b.json]', 'all strategies on one page against SPY (default: every file in strategies/)'),
    ]),
    ('Daily use', [
        ('python3 -m backtest.signals --params strategies/<name>.json', "today's buy candidates (add --holding AAPL MSFT for sell signals)"),
    ]),
    ('Research (for the curious)', [
        ('python3 -m backtest.run --params ...', 'detailed text backtest, trades.csv, equity.csv'),
        ('python3 -m backtest.analyze --params ... [--quick]', 'stress tests: verdict, costs, neighbors, random entries'),
        ('python3 -m backtest.tune --space default|momentum', 'search many rule sets with walk-forward validation'),
    ]),
    ('Options', [
        ('python3 -m backtest.spreads --fetch', 'download SPY (unadjusted) and VIX history once'),
        ('python3 -m backtest.spreads --debit 4.95 [--offset 0.017 --width 10 --dte 60]', 'backtest a SPY bull call spread routine, scaled to the debit you pay'),
        ('python3 -m backtest.spreads --tune', 'try 720 spread setups (strikes, width, expiry, close rule, stop, cadence, filter)'),
    ]),
]

if __name__ == '__main__':
    print('StrategyAnalyser - commands (run from this folder)\n')
    for title, rows in COMMANDS:
        print(title)
        for cmd, what in rows:
            print(f'  {cmd}\n      {what}')
        print()
    print('Guide: docs/GUIDE.md   Findings: docs/strategy-research.md')
