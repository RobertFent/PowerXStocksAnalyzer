"""Refresh the index member lists in symbols/ from Wikipedia.

    python3 -m backtest.symbols          # sp500, sp100, nasdaq100

Run it when index members change, then `python3 -m backtest.fetch` to download their history.
"""
from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from backtest.data import ROOT_DIR

SOURCES = {
    'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    'sp100': 'https://en.wikipedia.org/wiki/S%26P_100',
    'nasdaq100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
}

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/118.0.0.0 Safari/537.36'
    )
}


def get_symbols_from_wiki(url: str) -> list[str]:
    """Symbols from the 'constituents' table of a Wikipedia page, in Yahoo Finance style."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    table = BeautifulSoup(response.text, 'html.parser').find('table', {'id': 'constituents'})
    if not table:
        raise ValueError(f'Could not find the constituents table on {url}')
    symbols = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if cols:
            symbols.append(cols[0].text.strip().replace('.', '-'))
    return symbols


def main() -> None:
    out_dir = ROOT_DIR / 'symbols'
    out_dir.mkdir(exist_ok=True)
    for index, url in SOURCES.items():
        symbols = get_symbols_from_wiki(url)
        pd.DataFrame(symbols, columns=['Symbol']).to_csv(out_dir / f'{index}.csv', index=False)
        print(f'Saved {len(symbols)} symbols to symbols/{index}.csv')


if __name__ == '__main__':
    main()
