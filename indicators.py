import os
import sys
import csv
from datetime import datetime, time, timedelta
import concurrent
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import pandas as pd
import pytz
import requests
from tqdm import tqdm
import yfinance as yf
import pandas_market_calendars as mcal
import pandas_ta as ta

# load envs from .env
load_dotenv()

TRADING_DAYS_PER_YEAR = 252
# change this to 250 more than days to update in db for more accurate results
DAYS_IN_PAST_FOR_PROCESSING = 300
DAYS_TO_UPDATE_IN_DATABASE = 7  # change this to increase the db insert window

# must match filename in symbols folder
INDEX_LIST = ['dow', 'nasdaq100', 'sp500']

DATABASE_URL = os.getenv('DATABASE_URL')
REVALIDATE_SECRET = os.getenv('REVALIDATE_SECRET')
STOCK_SCREENER_REVALIDATE_URL = os.getenv('STOCK_SCREENER_REVALIDATE_URL')

# init logger
logger = logging.getLogger('indicators.py')


def verify_environment_variables_are_set() -> None:
    if DATABASE_URL is None:
        logger.error('DATABASE_URL missing! Exiting...')
        sys.exit(1)


def get_symbols_from_csv(indices: list[str] | None) -> list[str]:
    """returns list of symbols from csv.
    """
    if (indices is None):
        indices = ['dow', 'nasdaq100', 'nyse', 'sp500']

    symbols = []

    for index in indices:
        path_in_str = os.path.join('symbols', f'{index}.csv')
        with open(path_in_str, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['Symbol']
                symbols.append(symbol)

    # use set to remove duplicates
    unique_symbols = set(symbols)
    logger.info(
        'Read %d symbols from .csv files from %s', len(unique_symbols), ', '.join(indices))
    return list(unique_symbols)


def analyze_symbols_multi_process(symbols: list[str], start_timestamp: int, end_timestamp: int) -> list[pd.DataFrame]:
    analyzed_stocks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(return_analyzed_symbol_df, symbol,
                            start_timestamp, end_timestamp)
            for symbol in symbols
        ]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            symbol_df = f.result()
            analyzed_stocks.append(symbol_df)
    return analyzed_stocks


def return_analyzed_symbol_df(symbol: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame | None:
    try:
        symbol_df = get_ticker_data_yahoo(
            symbol, start_timestamp, end_timestamp)

        symbol_df = add_ema_data(symbol_df)
        symbol_df = add_ma_200_data(symbol_df)
        symbol_df = add_macd_data(symbol_df)
        symbol_df = add_rsi_14_data(symbol_df)
        symbol_df = add_rsi_4_data(symbol_df)
        symbol_df = add_implied_volatility(symbol_df)
        # logger.debug(symbol_df)
        # todo: bid/ask spread (option related -> without tradier api not possible here)
        # todo: add bollinger

        # additional indicators
        symbol_df = add_williams_percent_r_4(symbol_df)
        symbol_df = add_williams_percent_r_14(symbol_df)
        symbol_df = add_stochastic_slow(symbol_df)
        symbol_df = add_adr_20(symbol_df)

        # loger.debug(f'Analyzing {symbol}...')
        return symbol_df

    except Exception as e:
        return None
        # logger.error(f'Error analyzing symbol: {symbol}; {str(e)}')


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
    """adds EMA20 and EMA50 values
    """
    ema_20 = dataframe['Close'].ewm(span=20, adjust=False).mean()
    ema_50 = dataframe['Close'].ewm(span=50, adjust=False).mean()

    modified_df = dataframe.copy()
    modified_df['EMA20'] = ema_20
    modified_df['EMA50'] = ema_50

    return modified_df


def add_ma_200_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds MA_200 values"""
    ma_200 = dataframe['Close'].rolling(window=200).mean()

    modified_df = dataframe.copy()
    modified_df['MA_200'] = ma_200

    return modified_df


def add_macd_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds macd(26, 12, 9) values."""
    ema_12 = dataframe['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe['Close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    modified_df = dataframe.copy()
    modified_df['MACD Line'] = macd_line
    modified_df['Signal Line'] = signal_line

    return modified_df


def add_rsi_14_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds rsi(14) values."""
    rsi = ta.rsi(dataframe['Close'], length=14)

    modified_df = dataframe.copy()
    modified_df['RSI_14'] = rsi

    return modified_df


def add_rsi_4_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds rsi(4) values."""
    rsi = ta.rsi(dataframe['Close'], length=4)

    modified_df = dataframe.copy()
    modified_df['RSI_4'] = rsi

    return modified_df


def add_implied_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds IV(30) values."""
    modified_df = dataframe.copy()

    # daily log returns
    modified_df['Daily Return in Percent'] = np.log(
        modified_df['Close']).diff()

    # rolling std with min_periods=1 to calculate IV even for first days
    rolling_std = modified_df['Daily Return in Percent'].rolling(
        window=30, min_periods=1).std()

    modified_df['IV'] = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    return modified_df


def add_williams_percent_r_4(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds William %R values; length = 4"""
    willr = ta.willr(
        dataframe['High'], dataframe['Low'], dataframe['Close'], length=4)

    modified_df = dataframe.copy()
    modified_df['WILLR_4'] = willr

    return modified_df


def add_williams_percent_r_14(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds William %R values; length = 14"""
    willr = ta.willr(
        dataframe['High'], dataframe['Low'], dataframe['Close'], length=14)

    modified_df = dataframe.copy()
    modified_df['WILLR_14'] = willr

    return modified_df


def add_stochastic_slow(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds stochastic slow(14, 3, 3) values.
        if %K crosses above %D -> buy signal
        if %K crosses below %D -> sell signal
    """
    low14 = dataframe['Low'].rolling(14).min()
    high14 = dataframe['High'].rolling(14).max()

    fast_k = (dataframe['Close'] - low14) / (high14 - low14) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan).fillna(0)

    fast_d = fast_k.rolling(3).mean()

    modified_df = dataframe.copy()
    modified_df['%K'] = fast_d
    modified_df['%D'] = modified_df['%K'].rolling(3).mean()

    return modified_df


def add_adr_20(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds ADR_20 (Average Daily Range over 20 days).
        ADR_20 = rolling mean of (High - Low) over 20 days.
    """
    adr20 = (dataframe['High'] - dataframe['Low']).rolling(window=20).mean()
    modified_df = dataframe.copy()
    modified_df['ADR_20'] = adr20
    return modified_df


def get_trading_time_range_timestamps() -> tuple[int, int]:
    nyse = mcal.get_calendar('NYSE')
    tz = pytz.timezone('US/Eastern')

    # get latest 200 trading days
    today = datetime.now(tz).date()
    schedule = nyse.schedule(
        start_date=today - timedelta(days=DAYS_IN_PAST_FOR_PROCESSING), end_date=today)
    trading_days = schedule.index.date

    # get earliest and latest possible dates
    # last trading day is yesterday if today is Mon-Fri else its just the last trading timestamp
    last_prev_trading_day_index = -2 if today.weekday() <= 4 else -1
    start_day = trading_days[0]  # 0-based index
    end_day = trading_days[last_prev_trading_day_index]

    # Combine with NYSE open and close times
    start_dt = tz.localize(datetime.combine(start_day, time(9, 30)))
    end_dt = tz.localize(datetime.combine(end_day, time(16, 0)))

    # Convert to int POSIX timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    return start_ts, end_ts


def insert_symbols_data_into_database(dfs: list[pd.DataFrame]) -> None:
    try:
        connection = psycopg2.connect(DATABASE_URL)
        for df in dfs:
            bulk_insert_symbol_data(df, connection)
    except Exception as e:
        logger.error('Error inserting stock: %s', str(e))
    finally:
        connection.close()


def bulk_insert_symbol_data(df: pd.DataFrame, connection) -> None:
    """Bulk-insert all rows of a winner DataFrame into PostgreSQL."""

    query = """
        INSERT INTO stock_data (
            ticker, date, close, high, low, open, volume,
            ema20, ema50, macd_line, signal_line, rsi_14, rsi_4,
            iv, willr_4, willr_14, stoch_percent_k, stoch_percent_d, adr_20, ma_200
        )
        VALUES %s
        ON CONFLICT (ticker, date)
        DO UPDATE SET
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            open = EXCLUDED.open,
            volume = EXCLUDED.volume,
            ema20 = EXCLUDED.ema20,
            ema50 = EXCLUDED.ema50,
            macd_line = EXCLUDED.macd_line,
            signal_line = EXCLUDED.signal_line,
            rsi_14 = EXCLUDED.rsi_14,
            rsi_4 = EXCLUDED.rsi_4,
            iv = EXCLUDED.iv,
            willr_4 = EXCLUDED.willr_4,
            willr_14 = EXCLUDED.willr_14,
            stoch_percent_k = EXCLUDED.stoch_percent_k,
            stoch_percent_d = EXCLUDED.stoch_percent_d,
            adr_20 = EXCLUDED.adr_20,
            ma_200 = EXCLUDED.ma_200,
            last_updated_at = NOW();
    """

    # convert entire DataFrame to list of tuples for bulk inserting
    values = []
    for _, row in df.tail(DAYS_TO_UPDATE_IN_DATABASE).iterrows():
        values.append((
            row['Ticker'],
            row['Date'],
            to_float(row['Close']),
            to_float(row['High']),
            to_float(row['Low']),
            to_float(row['Open']),
            int(row['Volume']) if row['Volume'] is not None else None,
            to_float(row['EMA20']),
            to_float(row['EMA50']),
            to_float(row['MACD Line']),
            to_float(row['Signal Line']),
            to_float(row['RSI_14']),
            to_float(row['RSI_4']),
            to_float(row['IV']),
            to_float(row['WILLR_4']),
            to_float(row['WILLR_14']),
            to_float(row['%K']),
            to_float(row['%D']),
            to_float(row['ADR_20']),
            to_float(row['MA_200'])
        ))

    with connection.cursor() as cur:
        execute_values(cur, query, values)
        connection.commit()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def setup_logger():
    os.makedirs('logs', exist_ok=True)
    date_str = datetime.now().strftime('%d-%m-%Y')
    logging.basicConfig(
        filename=f'logs/{date_str}.log', level=logging.INFO)

    # set console handler too
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # console.setFormatter(formatter)
    # logger.addHandler(console)


def save_dfs_to_excel(stock_dfs: list[pd.DataFrame]) -> None:
    combined_df = pd.concat(stock_dfs, ignore_index=True)
    combined_df.to_excel('combined_stock_data.xlsx',
                         engine='openpyxl', index=False)
    combined_df.to_csv('combined_stock_data.csv', index=False)


def revalidate_stock_screener_cache() -> None:
    headers = {'x-revalidate-secret': REVALIDATE_SECRET}
    response = requests.post(
        STOCK_SCREENER_REVALIDATE_URL, headers=headers, timeout=5000)
    logger.info('Stock Screener revalidation response: %s', response.json())


if __name__ == '__main__':
    verify_environment_variables_are_set()
    setup_logger()

    logger.info('\nStarting indicator script...')
    logger.info(datetime.now().strftime('%d/%m/%Y, %H:%M:%S'))
    logger.info('Processing symbols and check for indicator matches...')
    symbols_from_csv = get_symbols_from_csv(INDEX_LIST)
    start_timestamp, end_timestamp = get_trading_time_range_timestamps()

    # debug statement for testing out winning stocks
    symbol_df = return_analyzed_symbol_df(
        'QCOM', start_timestamp, end_timestamp)
    logger.debug(symbol_df)

    start_analyzing = datetime.now()
    stock_dfs = analyze_symbols_multi_process(
        symbols_from_csv, start_timestamp, end_timestamp)
    analyzing_duration_in_seconds = (
        datetime.now() - start_analyzing).total_seconds()

    start_inserting = datetime.now()
    insert_symbols_data_into_database(stock_dfs)
    # save_dfs_to_excel(stock_dfs)

    inserting_duration_in_seconds = (
        datetime.now() - start_inserting).total_seconds()

    logger.info('Processing took %s seconds', analyzing_duration_in_seconds)
    logger.info('Inserting / saving took %s seconds',
                inserting_duration_in_seconds)

    # send request to refresh cache after new data is inserted
    revalidate_stock_screener_cache()
