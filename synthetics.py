import csv
from datetime import datetime, time, timedelta
from pathlib import Path
import threading

import concurrent
import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm
import yfinance as yf
import pandas_market_calendars as mcal
import pandas_ta as ta

LOCK = threading.Lock()
winning_stocks = []

MIN_VOLUME = 1000000
MAX_RSI = 60
MIN_IV = 30
MAX_IV = 70


def get_symbols_from_csv():
    '''returns list of symbols from csv.
    '''
    symbols = []
    pathlist = Path('symbols/').rglob('*.csv')
    for path in pathlist:
        path_in_str = str(path)

        with open(path_in_str, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['Symbol']
                symbols.append(symbol)

    # use set to remove duplicates
    unique_symbols = set(symbols)
    print(f'Read {len(unique_symbols)} symbols from .csv files')
    return list(unique_symbols)


def analyze_symbols_single_threaded(symbols, start_timestamp, end_timestamp):
    for symbol in symbols:
        analyze_symbol(symbol, start_timestamp, end_timestamp)


def analyze_symbols_multi_threaded(symbols, start_timestamp, end_timestamp):
    '''Analyze all stocks in concurrent threads with progress bar.

    Keyword arguments:
    tickers -- list of all symbols
    '''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_symbol, symbol, start_timestamp, end_timestamp)
                   for symbol in symbols]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


def analyze_symbol(symbol: str, start_timestamp: int, end_timestamp: int):
    try:
        symbol_df = get_ticker_data_yahoo(
            symbol, start_timestamp, end_timestamp)

        add_ema_data(symbol_df)
        add_macd_data(symbol_df)
        add_rsi_data(symbol_df)
        add_implied_volatility(symbol_df)
        # print(symbol_df)
        # todo: bid/ask spread
        # todo: add bollinger

        # print(f'Analyzing {symbol}...')
        is_symbol_legit = analyze_technicals(symbol_df)
        if is_symbol_legit:
            with LOCK:
                winning_stocks.append(symbol)

    except Exception as e:
        pass
        # print(f'Error analyzing symbol: {symbol}; {str(e)}')


def get_ticker_data_yahoo(symbol, start_timestamp, end_timestamp):
    start = pd.to_datetime(start_timestamp, unit='s')
    end = pd.to_datetime(end_timestamp, unit='s')
    df = yf.download(symbol, start=start, end=end,
                     interval='1d', progress=False, auto_adjust=False)

    # flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Ticker'] = symbol.upper()

    # reset index so Date is a column
    df.reset_index(inplace=True)
    return df


def add_ema_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    ema_20 = dataframe['Close'].ewm(span=20, adjust=False).mean()
    ema_50 = dataframe['Close'].ewm(span=50, adjust=False).mean()

    dataframe['EMA20'] = ema_20
    dataframe['EMA50'] = ema_50


def add_macd_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds macd(26, 12, 9) values.
    '''
    ema_12 = dataframe['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe['Close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    dataframe['MACD Line'] = macd_line
    dataframe['Signal Line'] = signal_line


def add_rsi_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds rsi(14) values.
    '''
    rsi = ta.rsi(dataframe['Close'], length=14)
    dataframe['RSI'] = rsi


def add_implied_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds implied volatility values.
    '''
    dataframe['Daily Return in Percent'] = dataframe['Close'].pct_change()
    std_dev = dataframe['Daily Return in Percent'].std()
    # trading_days_per_year = 252
    annualized_std_dev = std_dev * np.sqrt(252)
    dataframe['IV'] = np.nan
    dataframe.loc[dataframe.index[-1],
                  'IV'] = annualized_std_dev * 100


def analyze_technicals(dataframe: pd.DataFrame) -> bool:
    latest_technicals = dataframe.loc[dataframe.index[-1]]
    second_latest_technicals = dataframe.loc[dataframe.index[-2]]
    close = latest_technicals['Close']
    ema20 = latest_technicals['EMA20']
    ema50 = latest_technicals['EMA50']
    rsi = latest_technicals['RSI']
    iv = latest_technicals['IV']
    volume = latest_technicals['Volume']
    macd_line = latest_technicals['MACD Line']
    macd_line_prev = second_latest_technicals['MACD Line']

    is_volume_bigger_than_mio = volume > MIN_VOLUME
    is_close_higher_than_ema20_than_ema50 = close > ema20 > ema50
    # todo: signal line
    is_macd_increasing = macd_line > macd_line_prev
    is_rsi_smaller_60 = rsi < MAX_RSI
    is_iv_between_30_and_70 = MIN_IV < iv < MAX_IV

    are_technicals_legit = is_volume_bigger_than_mio and is_close_higher_than_ema20_than_ema50 and is_macd_increasing and is_rsi_smaller_60 and is_iv_between_30_and_70
    return bool(are_technicals_legit)


def get_trading_time_range_timestamps() -> tuple[int, int]:
    # NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    tz = pytz.timezone('US/Eastern')

    # Get valid trading days up to today
    today = datetime.now(tz).date()
    schedule = nyse.schedule(
        start_date=today - timedelta(days=100), end_date=today)
    trading_days = schedule.index.date

    # Get 50 trading days ago (or earliest available if less)
    start_day = trading_days[-51]  # 0-based index: yesterday = -1
    end_day = trading_days[-2]     # yesterday’s trading day

    # Combine with NYSE open and close times
    start_dt = tz.localize(datetime.combine(start_day, time(9, 30)))
    end_dt = tz.localize(datetime.combine(end_day, time(16, 0)))

    # Convert to int POSIX timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    return start_ts, end_ts


if __name__ == '__main__':
    symbols_from_csv = get_symbols_from_csv()
    start_timestamp, end_timestamp = get_trading_time_range_timestamps()

    start = datetime.now()
    analyze_symbols_single_threaded(
        symbols_from_csv, start_timestamp, end_timestamp)
    duration_in_seconds = (datetime.now() - start).total_seconds()
    print(f'Processing took {duration_in_seconds} seconds')
    # analyze_symbols_multi_threaded(
    #     symbols_from_csv, start_timestamp, end_timestamp)
    print('Criteria:')
    print(f'- Min. Volume: {MIN_VOLUME}')
    print(f'- Max. RSI: {MAX_RSI}')
    print(f'- IV between {MIN_IV} and {MAX_IV}')
    print('- MACD increasing')
    print('- Close > EVA 20 > EVA 50')
    print(f'{len(winning_stocks)} stocks fulfilling criteria!')
    print(winning_stocks)
