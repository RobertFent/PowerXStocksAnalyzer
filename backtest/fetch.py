"""Download multi-year daily history for backtesting.

    python3 -m backtest.fetch                        # sp500, nasdaq100, sp100 since 2012
    python3 -m backtest.fetch --start 2005-01-01
    python3 -m backtest.fetch --update               # only fetch the last days and merge

Prices are split/dividend adjusted (auto_adjust=True) so returns are correct
across splits. Indicators are NOT stored - the backtester computes them with
whatever lengths are being tuned.

Note: symbol lists are *today's* index members -> survivorship bias (delisted /
removed stocks are missing). Results will look somewhat better than reality.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from backtest.data import DATA_DIR, PRICE_COLUMNS, ROOT_DIR, load_table, save_table

DEFAULT_INDICES = ['sp500', 'nasdaq100', 'sp100']
BENCHMARK_TICKERS = ['SPY', 'QQQ']
BATCH_SIZE = 50


def read_symbols(index: str) -> list[str]:
    path = os.path.join(ROOT_DIR, 'symbols', f'{index}.csv')
    with open(path, 'r', encoding='UTF-8') as f:
        return [row['Symbol'].strip().upper() for row in csv.DictReader(f) if row['Symbol'].strip()]


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frames = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        if multi:
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker].copy()
        else:
            sub = raw.copy()
        sub = sub.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any')
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={'index': 'Date', 'Datetime': 'Date'})
        sub['Ticker'] = ticker
        frames.append(sub[PRICE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out['Date'] = pd.to_datetime(out['Date']).dt.tz_localize(None).dt.normalize()
    return out


def download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for i in tqdm(range(0, len(tickers), BATCH_SIZE), desc='downloading'):
        batch = tickers[i:i + BATCH_SIZE]
        raw = yf.download(batch, start=start, end=end, interval='1d',
                          auto_adjust=True, group_by='ticker',
                          threads=True, progress=False)
        frames.append(_to_long(raw, batch))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
    return df.sort_values(['Ticker', 'Date']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--indices', nargs='+', default=DEFAULT_INDICES)
    parser.add_argument('--start', default='2012-01-01')
    parser.add_argument('--end', default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument('--update', action='store_true',
                        help='only download the last 10 days and merge into existing data. '
                             'Adjusted prices shift after dividends/splits, so re-run a full '
                             'download now and then.')
    args = parser.parse_args()

    universe_rows = []
    for index in args.indices:
        for symbol in read_symbols(index):
            universe_rows.append({'Ticker': symbol, 'Index': index})
    universe = pd.DataFrame(universe_rows).drop_duplicates()
    tickers = sorted(universe['Ticker'].unique())

    start = args.start
    if args.update:
        start = (date.today() - timedelta(days=10)).isoformat()

    print(f'Fetching {len(tickers)} tickers from {start} to {args.end}')
    prices = download(tickers, start, args.end)
    bench = download(BENCHMARK_TICKERS, start, args.end)

    if args.update:
        prices = merge(load_table('prices'), prices)
        bench = merge(load_table('benchmark'), bench)

    DATA_DIR.mkdir(exist_ok=True)
    universe.to_csv(DATA_DIR / 'universe.csv', index=False)
    p1 = save_table(prices, 'prices')
    p2 = save_table(bench, 'benchmark')

    missing = sorted(set(tickers) - set(prices['Ticker'].unique()))
    print(f'Saved {len(prices):,} rows for {prices["Ticker"].nunique()} tickers -> {p1}')
    print(f'Saved benchmark ({", ".join(BENCHMARK_TICKERS)}) -> {p2}')
    print(f'Date range: {prices["Date"].min().date()} .. {prices["Date"].max().date()}')
    if missing:
        print(f'No data for {len(missing)} tickers: {", ".join(missing[:30])}'
              f'{" ..." if len(missing) > 30 else ""}')


if __name__ == '__main__':
    main()
