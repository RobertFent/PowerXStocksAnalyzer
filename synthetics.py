import csv
from datetime import datetime, time, timedelta
import concurrent
import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm
import yfinance as yf
import pandas_market_calendars as mcal
import pandas_ta as ta

MIN_VOLUME = 1000000
MAX_RSI = 60
MIN_IV = 30
MAX_IV = 70
TRADING_DAYS_USED = 200
TRADING_DAYS_PER_YEAR = 252

# must match filename in symbols folder
INDEX_LIST = ['dow', 'nasdaq100', 'sp500']


def get_symbols_from_csv(indices: list[str] | None):
    '''returns list of symbols from csv.
    '''
    if (indices is None):
        indices = ['dow', 'nasdaq100', 'nyse', 'sp500']

    symbols = []

    for index in indices:
        path_in_str = f'symbols/{index}.csv'

        with open(path_in_str, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['Symbol']
                symbols.append(symbol)

    # use set to remove duplicates
    unique_symbols = set(symbols)
    print(
        f'Read {len(unique_symbols)} symbols from .csv files from {', '.join(indices)}')
    return list(unique_symbols)


def analyze_symbols_multi_process(symbols: list[str], start_timestamp: int, end_timestamp: int) -> list[str]:
    winning_stocks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(return_symbol_df_if_is_winner, symbol,
                            start_timestamp, end_timestamp)
            for symbol in symbols
        ]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            symbol_df = f.result()
            if symbol_df is not None:
                winning_stocks.append(symbol_df)
    return winning_stocks


def return_symbol_df_if_is_winner(symbol: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
    try:
        symbol_df = get_ticker_data_yahoo(
            symbol, start_timestamp, end_timestamp)

        symbol_df = add_ema_data(symbol_df)
        symbol_df = add_macd_data(symbol_df)
        symbol_df = add_rsi_data(symbol_df)
        symbol_df = add_implied_volatility(symbol_df)
        # print(symbol_df)
        # todo: bid/ask spread
        # todo: add bollinger

        # additional indicators
        symbol_df = add_williams_percent_r(symbol_df)

        # print(f'Analyzing {symbol}...')
        is_symbol_legit = analyze_technicals(symbol_df)
        return symbol_df if is_symbol_legit else None

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
    '''adds EMA20 and EMA50 values
    '''
    ema_20 = dataframe['Close'].ewm(span=20, adjust=False).mean()
    ema_50 = dataframe['Close'].ewm(span=50, adjust=False).mean()

    modified_df = dataframe.copy()
    modified_df['EMA20'] = ema_20
    modified_df['EMA50'] = ema_50

    return modified_df


def add_macd_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds macd(26, 12, 9) values.
    '''
    ema_12 = dataframe['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe['Close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    modified_df = dataframe.copy()
    modified_df['MACD Line'] = macd_line
    modified_df['Signal Line'] = signal_line

    return modified_df


def add_rsi_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds rsi(14) values.
    '''
    rsi = ta.rsi(dataframe['Close'], length=14)

    modified_df = dataframe.copy()
    modified_df['RSI'] = rsi

    return modified_df


def add_implied_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds IV(30) values.
    '''
    modified_df = dataframe.copy()
    modified_df['Daily Return in Percent'] = np.log(
        modified_df['Close']).diff()

    # get latest 30 days
    sample = modified_df['Daily Return in Percent'].iloc[-30:]
    std_dev = sample.std()
    annualized_std_dev = std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)

    modified_df['IV'] = np.nan
    modified_df.loc[modified_df.index[-1],
                    'IV'] = annualized_std_dev * 100

    return modified_df


def add_williams_percent_r(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds William %R values; length = 14
    '''
    willr = ta.willr(
        dataframe['High'], dataframe['Low'], dataframe['Close'], length=14)

    modified_df = dataframe.copy()
    modified_df['WILLR'] = willr

    return modified_df


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
    nyse = mcal.get_calendar('NYSE')
    tz = pytz.timezone('US/Eastern')

    # get latest 200 trading days
    today = datetime.now(tz).date()
    schedule = nyse.schedule(
        start_date=today - timedelta(days=TRADING_DAYS_USED), end_date=today)
    trading_days = schedule.index.date

    # get earliest and latest possible dates
    start_day = trading_days[0]  # 0-based index
    end_day = trading_days[-2]     # yesterday’s trading day

    # Combine with NYSE open and close times
    start_dt = tz.localize(datetime.combine(start_day, time(9, 30)))
    end_dt = tz.localize(datetime.combine(end_day, time(16, 0)))

    # Convert to int POSIX timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    return start_ts, end_ts


def get_symbol_names_as_list_from_winning_dfs(df_list: list[pd.DataFrame]) -> list[str]:
    return [symbol_df['Ticker'].iloc[0] for symbol_df in df_list]


if __name__ == '__main__':
    print('Processing symbols and check for indicator matches...')
    symbols_from_csv = get_symbols_from_csv(INDEX_LIST)
    start_timestamp, end_timestamp = get_trading_time_range_timestamps()

    # debug statement for testing out winning stocks
    symbol_df = return_symbol_df_if_is_winner(
        'APA', start_timestamp, end_timestamp)

    start = datetime.now()
    winning_stock_dfs = analyze_symbols_multi_process(
        symbols_from_csv, start_timestamp, end_timestamp)
    duration_in_seconds = (datetime.now() - start).total_seconds()

    winning_symbols = get_symbol_names_as_list_from_winning_dfs(
        winning_stock_dfs)

    # todo: parse last col. into database

    print(f'Processing took {duration_in_seconds} seconds')
    print('Criteria:')
    print(f'- Min. Volume: {MIN_VOLUME}')
    print(f'- Max. RSI: {MAX_RSI}')
    print(f'- IV between {MIN_IV} and {MAX_IV}')
    print('- MACD increasing')
    print('- Close > EVA 20 > EVA 50')
    print(f'{len(winning_stock_dfs)} stocks fulfilling criteria!')
    print(winning_symbols)
