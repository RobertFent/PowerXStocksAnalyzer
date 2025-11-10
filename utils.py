from io import BytesIO
import requests
import pandas as pd
from bs4 import BeautifulSoup

SP500_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
NASDAQ100_WIKI_URL = 'https://en.wikipedia.org/wiki/Nasdaq-100'
# DOW_WIKI_URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/118.0.0.0 Safari/537.36'
    )
}


def get_symbols_from_wiki(url: str) -> list:
    """Return a list of symbols from Wikipedia."""
    response = requests.get(url, headers=HEADERS, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})

    if not table:
        raise ValueError("Could not find the S&P 500 table on Wikipedia.")

    symbols = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if cols:
            symbol = cols[0].text.strip()
            # match Yahoo Finance style
            symbols.append(symbol.replace('.', '-'))

    return symbols


def save_symbols_to_csv(symbols: list, filename: str) -> None:
    """Save the list of symbols to a CSV file."""
    df = pd.DataFrame(symbols, columns=['Symbol'])
    df.to_csv(filename, index=False)
    print(f"Saved {len(symbols)} symbols to {filename}")


def write_nyse_symbols() -> None:
    csv_url = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
    response = requests.get(csv_url, headers=HEADERS, timeout=5)
    response.raise_for_status()

    nyse_csv_df = pd.read_csv(BytesIO(response.content))
    nyse_csv_df.rename(
        columns={'ACT Symbol': 'Symbol'}, inplace=True)
    nyse_csv_df.drop(columns=['Company Name'], inplace=True)

    nyse_csv_df.to_csv('symbols/nyse.csv', index=False)


if __name__ == '__main__':
    sp500_symbols = get_symbols_from_wiki(SP500_WIKI_URL)
    nasdaq_100_symbols = get_symbols_from_wiki(NASDAQ100_WIKI_URL)
    save_symbols_to_csv(sp500_symbols, 'symbols/sp500.csv')
    save_symbols_to_csv(nasdaq_100_symbols, 'symbols/nasdaq100.csv')
    write_nyse_symbols()
