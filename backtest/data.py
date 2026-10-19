"""Loading / saving of the historical price data used for backtesting.

Layout (all inside ./data, git-ignored):
    prices.parquet     long format: Date, Ticker, Open, High, Low, Close, Volume (split/dividend adjusted)
    benchmark.parquet  same columns for benchmark tickers (SPY, QQQ)
    universe.csv       Ticker, Index  (a ticker can be in several indices)

Falls back to .csv.gz when pyarrow is not installed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

PRICE_COLUMNS = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _path(name: str) -> Path:
    if _has_pyarrow():
        return DATA_DIR / f'{name}.parquet'
    return DATA_DIR / f'{name}.csv.gz'


def _find(name: str) -> Path:
    for candidate in (DATA_DIR / f'{name}.parquet', DATA_DIR / f'{name}.csv.gz'):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No {name} data found in {DATA_DIR}. Run `python3 -m backtest.fetch` first.')


def save_table(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = _path(name)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_table(name: str) -> pd.DataFrame:
    path = _find(name)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    return df


def load_universe() -> pd.DataFrame:
    path = DATA_DIR / 'universe.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing. Run `python3 -m backtest.fetch` first.')
    return pd.read_csv(path)


def load_prices(index: str | None = None,
                start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    """Load adjusted OHLCV. `index` filters to tickers of e.g. 'sp500' (None = all)."""
    df = load_table('prices')
    if index:
        universe = load_universe()
        tickers = set(universe.loc[universe['Index'] == index, 'Ticker'])
        if not tickers:
            raise ValueError(f'No tickers for index {index!r}. '
                             f'Available: {sorted(universe["Index"].unique())}')
        df = df[df['Ticker'].isin(tickers)]
    if start:
        df = df[df['Date'] >= pd.Timestamp(start)]
    if end:
        df = df[df['Date'] <= pd.Timestamp(end)]
    return df.sort_values(['Ticker', 'Date']).reset_index(drop=True)


def load_benchmark(ticker: str = 'SPY') -> pd.DataFrame:
    df = load_table('benchmark')
    return df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
