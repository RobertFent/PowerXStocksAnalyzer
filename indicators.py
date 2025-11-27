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
from tqdm import tqdm
import yfinance as yf
import pandas_market_calendars as mcal
import pandas_ta as ta

# load envs from .env
load_dotenv()

MIN_VOLUME = 1000000
MAX_RSI = 60
MIN_IV = 30
MAX_IV = 70
TRADING_DAYS_PER_YEAR = 252
# change this to 250 more than days to update in db for more accurate results
DAYS_IN_PAST_FOR_PROCESSING = 300
DAYS_TO_UPDATE_IN_DATABASE = 7  # change this to increase the db insert window

# must match filename in symbols folder
INDEX_LIST = ['dow', 'nasdaq100', 'sp500']

DATABASE_URL = os.getenv("DATABASE_URL")

# init logger
logger = logging.getLogger('indicators.py')


def verify_environment_variables_are_set() -> None:
    if DATABASE_URL is None:
        logger.error('DATABASE_URL missing! Exiting...')
        sys.exit(1)


def get_symbols_from_csv(indices: list[str] | None) -> list[str]:
    '''returns list of symbols from csv.
    '''
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


def analyze_symbols_multi_process(symbols: list[str], start_timestamp: int, end_timestamp: int) -> list[str]:
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


def return_analyzed_symbol_df(symbol: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
    try:
        symbol_df = get_ticker_data_yahoo(
            symbol, start_timestamp, end_timestamp)

        symbol_df = add_ema_data(symbol_df)
        symbol_df = add_macd_data(symbol_df)
        symbol_df = add_rsi_data(symbol_df)
        symbol_df = add_implied_volatility(symbol_df)
        # logger.debug(symbol_df)
        # todo: bid/ask spread (option related -> without tradier api not possible here)
        # todo: add bollinger

        # additional indicators
        symbol_df = add_williams_percent_r(symbol_df)
        symbol_df = add_stochastic_slow(symbol_df)

        # loger.debug(f'Analyzing {symbol}...')
        return symbol_df

    except Exception as e:
        pass
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

    # daily log returns
    modified_df['Daily Return in Percent'] = np.log(
        modified_df['Close']).diff()

    # rolling std with min_periods=1 to calculate IV even for first days
    rolling_std = modified_df['Daily Return in Percent'].rolling(
        window=30, min_periods=1).std()

    modified_df['IV'] = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    return modified_df


def add_williams_percent_r(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds William %R values; length = 14
    '''
    willr = ta.willr(
        dataframe['High'], dataframe['Low'], dataframe['Close'], length=14)

    modified_df = dataframe.copy()
    modified_df['WILLR'] = willr

    return modified_df


def add_stochastic_slow(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''adds stochastic slow(14, 3, 3) values.
        if %K crosses above %D -> buy signal
        if %K crosses below %D -> sell signal
    '''
    low14 = dataframe['Low'].rolling(14).min()
    high14 = dataframe['High'].rolling(14).max()

    fast_k = (dataframe['Close'] - low14) / (high14 - low14) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan).fillna(0)

    fast_d = fast_k.rolling(3).mean()

    modified_df = dataframe.copy()
    modified_df['%K'] = fast_d
    modified_df['%D'] = modified_df['%K'].rolling(3).mean()

    return modified_df


def is_winner_symbol(dataframe: pd.DataFrame) -> bool:
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
        start_date=today - timedelta(days=DAYS_IN_PAST_FOR_PROCESSING), end_date=today)
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


def filter_symbol_dfs_for_winners(df_list: list[pd.DataFrame]) -> list[pd.DataFrame]:
    winning_symbols = []
    for symbol_df in df_list:
        try:
            if is_winner_symbol(symbol_df):
                winning_symbols.append(symbol_df)
        except:
            pass  # todo
    return winning_symbols


def get_symbol_names_as_list_from_winning_dfs(df_list: list[pd.DataFrame]) -> list[str]:
    return [symbol_df['Ticker'].iloc[0] for symbol_df in df_list]


def insert_symbols_data_into_database(dfs: list[str]) -> None:
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
        INSERT INTO stock_winners (
            ticker, date, close, high, low, open, volume,
            ema20, ema50, macd_line, signal_line, rsi,
            iv, willr, stoch_percent_k, stoch_percent_d
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
            rsi = EXCLUDED.rsi,
            iv = EXCLUDED.iv,
            willr = EXCLUDED.willr,
            stoch_percent_k = EXCLUDED.stoch_percent_k,
            stoch_percent_d = EXCLUDED.stoch_percent_d,
            last_updated_at = NOW();
    """

    # convert entire DataFrame to list of tuples for bulk inserting
    values = []
    for _, row in df.tail(DAYS_TO_UPDATE_IN_DATABASE).iterrows():
        values.append((
            row["Ticker"],
            row["Date"],
            to_float(row["Close"]),
            to_float(row["High"]),
            to_float(row["Low"]),
            to_float(row["Open"]),
            int(row["Volume"]) if row["Volume"] is not None else None,
            to_float(row["EMA20"]),
            to_float(row["EMA50"]),
            to_float(row["MACD Line"]),
            to_float(row["Signal Line"]),
            to_float(row["RSI"]),
            to_float(row["IV"]),
            to_float(row["WILLR"]),
            to_float(row["%K"]),
            to_float(row["%D"])
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
    date_str = datetime.now().strftime("%d-%m-%Y")
    logging.basicConfig(
        filename=f'logs/{date_str}.log', level=logging.INFO)

    # set console handler too
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # console.setFormatter(formatter)
    # logger.addHandler(console)


if __name__ == '__main__':
    verify_environment_variables_are_set()
    setup_logger()

    logger.info('\nStarting indicator script...')
    logger.info(datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    logger.info('Processing symbols and check for indicator matches...')
    symbols_from_csv = get_symbols_from_csv(INDEX_LIST)
    start_timestamp, end_timestamp = get_trading_time_range_timestamps()

    # debug statement for testing out winning stocks
    # symbol_df = return_analyzed_symbol_df(
    #     'QCOM', start_timestamp, end_timestamp)
    # logger.debug(symbol_df)

    start_analyzing = datetime.now()
    stock_dfs = analyze_symbols_multi_process(
        symbols_from_csv, start_timestamp, end_timestamp)
    analyzing_duration_in_seconds = (
        datetime.now() - start_analyzing).total_seconds()

    start_inserting = datetime.now()
    insert_symbols_data_into_database(stock_dfs)
    inserting_duration_in_seconds = (
        datetime.now() - start_inserting).total_seconds()

    winning_symbols = filter_symbol_dfs_for_winners(stock_dfs)
    winning_symbols_str = get_symbol_names_as_list_from_winning_dfs(
        winning_symbols)

    logger.info('Processing took %s seconds', analyzing_duration_in_seconds)
    logger.info('Inserting took %s seconds', inserting_duration_in_seconds)
    logger.info('Criteria:')
    logger.info('- Min. Volume: %s', MIN_VOLUME)
    logger.info('- Max. RSI: %s', MAX_RSI)
    logger.info('- IV between %s and %s', MIN_IV, MAX_IV)
    logger.info('- MACD increasing')
    logger.info('- Close > EVA 20 > EVA 50')
    logger.info('%d stocks fulfilling criteria!', len(winning_symbols))
    logger.info('%s', winning_symbols_str)
