"""Daily scraper, the only script that has to run: once per trading day after the US close.

1. prices + indicators of every index member                   -> stock_data
2. option positioning (walls, gamma / vanna / charm exposure,
   implied volatility, skew, max pain, earnings date) of the
   most liquid symbols                                         -> gamma_levels

Option greeks are computed with Black-Scholes from Yahoo's open interest and implied volatility.
Dealer positioning is the usual naive assumption (dealers long calls, short puts): good enough for
index ETFs, rough for single stocks. Yahoo empties the open interest overnight US time, so a run
before the US session gets no usable option chains.

    python3 indicators.py                              # everything
    python3 indicators.py --only prices                # or: --only options
    python3 indicators.py --dry-run SPY AAPL           # print the option levels, no database
"""
import argparse
import os
import sys
import csv
import time as time_module
from datetime import date, datetime, time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
import logging
from math import erf
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
DAYS_TO_UPDATE_IN_DATABASE = 14  # change this to increase the db insert window

# must match filename in symbols folder
INDEX_LIST = ['sp100', 'nasdaq100', 'sp500']

# Fail the run when more than this share of symbols could not be analyzed.
# Individual delistings are normal; a tenth of the index disappearing is not,
# and a run that exits 0 with a third of the data missing is worse than one
# that fails loudly.
MAX_FAILURE_RATE = 0.05

# yfinance keeps its timezone and cookie caches in SQLite. Every worker process
# would otherwise open the same files and they race — "database is locked",
# "UNIQUE constraint failed: _cookieschema.strategy". Worse, opening one in the
# parent before the fork leaves a file descriptor that every child inherits,
# which shows up as "disk I/O error" across the whole run. So: the parent
# creates the root and never opens it, and each worker gets a subdirectory of
# its own (see prepare_yfinance_cache and init_worker_cache).
YF_CACHE_DIR = os.getenv(
    'YF_CACHE_DIR', os.path.join(os.getcwd(), '.cache', 'py-yfinance'))

DATABASE_URL = os.getenv('DATABASE_URL')
REVALIDATE_SECRET = os.getenv('REVALIDATE_SECRET')
STOCK_SCREENER_REVALIDATE_URL = os.getenv('STOCK_SCREENER_REVALIDATE_URL', '')

# ---- option positioning ----
# must match filename in symbols folder
OPTIONS_INDEX_LIST = ['sp100']
OPTIONS_EXTRA_SYMBOLS = ['SPY', 'QQQ']

MAX_DAYS_TO_EXPIRATION = 45
MIN_TOTAL_OPEN_INTEREST = 1000  # below this the walls are meaningless
# Yahoo empties the open interest overnight (US time) until the session starts. If less than
# this share of the strikes within 10% around spot has open interest the chain is not usable.
MIN_OPEN_INTEREST_COVERAGE = 0.5
RISK_FREE_RATE = 0.04
CONTRACT_SIZE = 100
# Yahoo returns ~0 or absurd IVs for strikes without a quote (most of them outside market hours)
MIN_VALID_IV = 0.03
MAX_VALID_IV = 4.0
# gamma flip is searched between -30% and +30% around spot
FLIP_GRID = np.linspace(0.7, 1.3, 121)
OPTIONS_MAX_WORKERS = 4  # keep low, Yahoo rate limits option chain requests
OPTIONS_RETRIES = 3

CREATE_GAMMA_LEVELS_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS gamma_levels (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        spot DOUBLE PRECISION,
        call_wall DOUBLE PRECISION,
        put_wall DOUBLE PRECISION,
        gamma_flip DOUBLE PRECISION,
        net_gex DOUBLE PRECISION,
        call_gex DOUBLE PRECISION,
        put_gex DOUBLE PRECISION,
        call_oi BIGINT,
        put_oi BIGINT,
        expirations SMALLINT,
        last_updated_at TIMESTAMP DEFAULT NOW(),

        UNIQUE (ticker, date)
    );
    ALTER TABLE gamma_levels
        ADD COLUMN IF NOT EXISTS net_vanna DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS net_charm DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS atm_iv_30d DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS max_pain DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS max_pain_expiration DATE,
        ADD COLUMN IF NOT EXISTS skew_25_delta DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS realized_volatility_30d DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS call_volume BIGINT,
        ADD COLUMN IF NOT EXISTS put_volume BIGINT,
        ADD COLUMN IF NOT EXISTS next_earnings_date DATE,
        ADD COLUMN IF NOT EXISTS ex_dividend_date DATE;
"""

# init logger
logger = logging.getLogger('indicators.py')


def verify_environment_variables_are_set() -> None:
    if DATABASE_URL is None:
        logger.error('DATABASE_URL missing! Exiting...')
        sys.exit(1)


def prepare_yfinance_cache() -> None:
    """Create the cache root. Deliberately does NOT open it.

    Opening a yfinance cache in the parent — a single yf.download is enough —
    leaves an open SQLite file descriptor that ProcessPoolExecutor duplicates
    into every child, and concurrent use of an inherited connection fails as
    'disk I/O error'.
    """
    try:
        os.makedirs(YF_CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning('Could not create %s: %s', YF_CACHE_DIR, str(e))


def init_worker_cache() -> None:
    """Point this worker at a cache of its own, after the fork."""
    try:
        worker_dir = os.path.join(YF_CACHE_DIR, f'w{os.getpid()}')
        os.makedirs(worker_dir, exist_ok=True)
        yf.set_tz_cache_location(worker_dir)
    except Exception:
        # No cache is fine: yfinance falls back to fetching the timezone.
        pass


def get_symbols_from_csv(indices: list[str]) -> dict[str, list[str]]:
    """returns dict where key is index and values are symbols listed there"""

    symbols: dict[str, list[str]] = {}

    for index in indices:
        symbols[index] = []
        path_in_str = os.path.join('symbols', f'{index}.csv')
        with open(path_in_str, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['Symbol']
                symbols[index].append(symbol)

    return symbols


def analyze_symbols_multi_process(symbols: dict[str, list[str]], start_timestamp: int, end_timestamp: int) -> tuple[list[pd.DataFrame], list[str]]:
    """Returns the analyzed frames and the symbols that could not be analyzed."""
    analyzed_stocks: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker_cache) as executor:
        futures = {}
        for index, listed_symbols in symbols.items():
            for symbol in listed_symbols:
                future = executor.submit(
                    return_analyzed_symbol_df, index, symbol,
                    start_timestamp, end_timestamp)
                futures[future] = symbol

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            symbol = futures[f]
            try:
                symbol_df = f.result()
            except Exception as e:
                logger.error('Worker died on %s: %s', symbol, str(e))
                failed_symbols.append(symbol)
                continue

            if symbol_df is None:
                failed_symbols.append(symbol)
            else:
                analyzed_stocks.append(symbol_df)

    return analyzed_stocks, failed_symbols


def return_analyzed_symbol_df(index: str, symbol: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame | None:
    try:
        symbol_df = get_ticker_data_yahoo(
            symbol, start_timestamp, end_timestamp)

        symbol_df = add_index_data(symbol_df, index)
        symbol_df = add_ema_data(symbol_df)
        symbol_df = add_ma_200_data(symbol_df)
        symbol_df = add_macd_data(symbol_df)
        symbol_df = add_rsi_14_data(symbol_df)
        symbol_df = add_rsi_4_data(symbol_df)
        symbol_df = add_implied_volatility(symbol_df)
        # todo: bid/ask spread (option related -> without tradier api not possible here)
        # todo: add bollinger

        # additional indicators
        symbol_df = add_williams_percent_r_4(symbol_df)
        symbol_df = add_williams_percent_r_14(symbol_df)
        symbol_df = add_stochastic_slow(symbol_df)
        symbol_df = add_adr_values(symbol_df)

        return symbol_df

    except Exception as e:
        # Returning None silently is how a run reports success with a third of
        # the index missing.
        logger.error('Error analyzing symbol %s: %s: %s',
                     symbol, type(e).__name__, str(e))
        return None


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


def add_index_data(dataframe: pd.DataFrame, index: str) -> pd.DataFrame:
    """adds index"""
    modified_df = dataframe.copy()
    modified_df['Index'] = index

    return modified_df


def add_ema_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds EMA20 and EMA50 values"""
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


def add_adr_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """adds ADR_7 and ADR_14 (Average Daily Range over 14 days).
        ADR_14 = rolling mean of (High - Low) over 14 days.
    """
    adr7 = (dataframe['High'] - dataframe['Low']).rolling(window=7).mean()
    adr14 = (dataframe['High'] - dataframe['Low']).rolling(window=14).mean()
    modified_df = dataframe.copy()
    modified_df['ADR_7'] = adr7
    modified_df['ADR_14'] = adr14
    return modified_df


def get_trading_time_range_timestamps() -> tuple[int, int]:
    nyse = mcal.get_calendar('NYSE')
    tz = pytz.timezone('US/Eastern')

    today = datetime.now(tz).date()
    schedule = nyse.schedule(
        start_date=today - timedelta(days=DAYS_IN_PAST_FOR_PROCESSING), end_date=today)

    # the last session that is already closed: today when run after the close, else the previous one
    completed_sessions = schedule[schedule['market_close'] <= pd.Timestamp.now(tz='UTC')]
    start_day = schedule.index[0].date()
    end_day = completed_sessions.index[-1].date()

    # Combine with NYSE open and close times
    start_dt = tz.localize(datetime.combine(start_day, time(9, 30)))
    end_dt = tz.localize(datetime.combine(end_day, time(16, 0)))

    # Convert to int POSIX timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    return start_ts, end_ts


def insert_symbols_data_into_database(dfs: list[pd.DataFrame]) -> None:
    connection = None
    try:
        connection = psycopg2.connect(DATABASE_URL)
        for df in dfs:
            bulk_insert_symbol_data(df, connection)
    except Exception as e:
        # Previously this was logged and swallowed, and the finally block then
        # raised NameError when the connect itself had failed.
        logger.error('Error inserting stock data: %s', str(e))
        raise
    finally:
        if connection is not None:
            connection.close()


def bulk_insert_symbol_data(df: pd.DataFrame, connection) -> None:
    """Bulk-insert all rows of a winner DataFrame into PostgreSQL."""

    query = """
        INSERT INTO stock_data (
            ticker, index, date, close, high, low, open, volume,
            ema20, ema50, macd_line, signal_line, rsi_14, rsi_4,
            iv, willr_4, willr_14, stoch_percent_k, stoch_percent_d, adr_7, adr_14, ma_200
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
            adr_7 = EXCLUDED.adr_7,
            adr_14 = EXCLUDED.adr_14,
            ma_200 = EXCLUDED.ma_200,
            last_updated_at = NOW();
    """

    # convert entire DataFrame to list of tuples for bulk inserting
    values = []
    for _, row in df.tail(DAYS_TO_UPDATE_IN_DATABASE).iterrows():
        values.append((
            row['Ticker'],
            row['Index'],
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
            to_float(row['ADR_7']),
            to_float(row['ADR_14']),
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
    # combined_df.to_excel('combined_stock_data_with_index.xlsx',
    #                      engine='openpyxl', index=False)
    combined_df.to_csv('combined_stock_data_with_index.csv', index=False)


def revalidate_stock_screener_cache() -> None:
    if not STOCK_SCREENER_REVALIDATE_URL:
        logger.info('STOCK_SCREENER_REVALIDATE_URL is not set - skipping')
        return

    headers = {'x-revalidate-secret': REVALIDATE_SECRET}
    # 5000 was seconds, not milliseconds: effectively no timeout at all, so a
    # hanging endpoint would hang the whole nightly run.
    response = requests.post(
        STOCK_SCREENER_REVALIDATE_URL, headers=headers, timeout=30)
    logger.info('Stock Screener revalidation response: %s', response.json())


# ---- option positioning ----

class OpenInterestNotPopulated(Exception):
    pass


def black_scholes_gamma(spot, strikes, iv, years_to_expiration):
    """gamma is identical for calls and puts; spot may be a scalar or a column vector"""
    sqrt_t = np.sqrt(years_to_expiration)
    d1 = (np.log(spot / strikes) + (RISK_FREE_RATE + 0.5 * iv ** 2)
          * years_to_expiration) / (iv * sqrt_t)
    return np.exp(-0.5 * d1 ** 2) / (np.sqrt(2 * np.pi) * spot * iv * sqrt_t)


def gamma_exposure(spot, contracts: pd.DataFrame):
    """signed dealer gamma exposure per contract row in USD per 1% move of the underlying"""
    gamma = black_scholes_gamma(spot, contracts['strike'].to_numpy(),
                                contracts['iv'].to_numpy(), contracts['t'].to_numpy())
    return (contracts['sign'].to_numpy() * gamma * contracts['oi'].to_numpy()
            * CONTRACT_SIZE * spot ** 2 * 0.01)


def vanna_and_charm_exposure(spot: float, contracts: pd.DataFrame) -> tuple[float, float]:
    """net dealer vanna and charm exposure, both as USD of delta the dealers have to trade:
        vanna per 1 volatility point the IV rises, charm per calendar day that passes.
        without dividends vanna and charm are identical for calls and puts.
    """
    strikes = contracts['strike'].to_numpy()
    iv = contracts['iv'].to_numpy()
    t = contracts['t'].to_numpy()
    sqrt_t = np.sqrt(t)

    d1 = (np.log(spot / strikes) + (RISK_FREE_RATE + 0.5 * iv ** 2) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    density = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)

    vanna = -density * d2 / iv
    charm = -density * (2 * RISK_FREE_RATE * t - d2 * iv * sqrt_t) / (2 * t * iv * sqrt_t)

    delta_notional = contracts['sign'].to_numpy() * contracts['oi'].to_numpy() * CONTRACT_SIZE * spot
    return float((vanna * 0.01 * delta_notional).sum()), float((charm / 365 * delta_notional).sum())


def find_max_pain(contracts: pd.DataFrame) -> tuple[float, date]:
    """strike at which the option holders of the expiration with the most open interest lose the most"""
    expiration = contracts.groupby('expiration')['oi'].sum().idxmax()
    chain = contracts[contracts['expiration'] == expiration]
    strikes = chain['strike'].to_numpy()
    oi = chain['oi'].to_numpy()
    is_call = chain['sign'].to_numpy() > 0

    candidates = np.unique(strikes)
    settle = candidates[:, None]
    payout = np.where(is_call, np.maximum(settle - strikes, 0), np.maximum(strikes - settle, 0)) * oi
    return float(candidates[payout.sum(axis=1).argmin()]), expiration


def find_atm_iv_30d(spot: float, contracts: pd.DataFrame) -> float | None:
    """at-the-money implied volatility in percent, interpolated to 30 days in total variance.
        only IVs Yahoo really delivered are used, so this is empty when the chains had no quotes.
    """
    quoted = contracts[~contracts['iv_estimated']]
    points = []
    for t, chain in quoted.groupby('t'):
        smile = chain.groupby('strike')['iv'].mean()
        strikes = smile.index.to_numpy()
        if len(smile) < 2 or spot < strikes[0] or spot > strikes[-1]:
            continue
        points.append((t, float(np.interp(spot, strikes, smile.to_numpy()))))
    if not points:
        return None

    years = np.array([p[0] for p in points])
    total_variance = np.array([p[1] ** 2 * p[0] for p in points])
    target = 30 / 365
    if len(points) == 1 or target <= years[0] or target >= years[-1]:
        # no expiration on one side of 30 days: take the closest one as it is
        return points[int(np.argmin(np.abs(years - target)))][1] * 100
    return float(np.sqrt(np.interp(target, years, total_variance) / target) * 100)


def find_skew_25_delta(contracts: pd.DataFrame) -> float | None:
    """IV of the 25 delta put minus IV of the 25 delta call in volatility points, taken from the
        expiration closest to 30 days. positive = downside protection is the more expensive side.
        only IVs Yahoo really delivered are used.
    """
    quoted = contracts[~contracts['iv_estimated']]
    if quoted.empty:
        return None

    years = quoted['t'].unique()
    chain = quoted[quoted['t'] == years[np.argmin(np.abs(years - 30 / 365))]]
    iv = chain['iv'].to_numpy()
    t = chain['t'].to_numpy()
    spot = chain['spot'].to_numpy()
    d1 = (np.log(spot / chain['strike'].to_numpy()) + (RISK_FREE_RATE + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    call_delta = 0.5 * (1 + np.vectorize(erf)(d1 / np.sqrt(2)))
    is_call = chain['sign'].to_numpy() > 0

    wing_iv = []
    for side, delta in ((is_call, call_delta), (~is_call, 1 - call_delta)):
        order = np.argsort(delta[side])
        deltas = delta[side][order]
        if len(deltas) < 2 or deltas[0] > 0.25 or deltas[-1] < 0.25:
            return None
        wing_iv.append(np.interp(0.25, deltas, iv[side][order]))
    return float((wing_iv[1] - wing_iv[0]) * 100)


def find_gamma_flip(spot: float, contracts: pd.DataFrame) -> float | None:
    """price closest to spot where total net gamma exposure changes its sign"""
    prices = spot * FLIP_GRID
    profile = gamma_exposure(prices[:, None], contracts).sum(axis=1)

    crossings = np.where(np.sign(profile[:-1]) * np.sign(profile[1:]) < 0)[0]
    if len(crossings) == 0:
        return None

    # linear interpolation between the two grid points around each crossing
    flips = prices[crossings] - profile[crossings] * (
        prices[crossings + 1] - prices[crossings]) / (profile[crossings + 1] - profile[crossings])
    return float(flips[np.argmin(np.abs(flips - spot))])


def compute_levels(spot: float, contracts: pd.DataFrame) -> dict | None:
    """contracts: one row per option with strike, oi, iv, iv_estimated, t (years),
        expiration (date) and sign (+1 call, -1 put)
    """
    if contracts.empty or contracts['oi'].sum() < MIN_TOTAL_OPEN_INTEREST:
        return None

    contracts = contracts.assign(gex=gamma_exposure(spot, contracts))
    calls = contracts[contracts['sign'] > 0]
    puts = contracts[contracts['sign'] < 0]
    if calls.empty or puts.empty:
        return None

    net_vanna, net_charm = vanna_and_charm_exposure(spot, contracts)
    max_pain, max_pain_expiration = find_max_pain(contracts)

    return {
        'spot': spot,
        'call_wall': float(calls.groupby('strike')['gex'].sum().idxmax()),
        'put_wall': float(puts.groupby('strike')['gex'].sum().idxmin()),
        'gamma_flip': find_gamma_flip(spot, contracts),
        'net_gex': float(contracts['gex'].sum()),
        'call_gex': float(calls['gex'].sum()),
        'put_gex': float(puts['gex'].sum()),
        'call_oi': int(calls['oi'].sum()),
        'put_oi': int(puts['oi'].sum()),
        'expirations': int(contracts['t'].nunique()),
        'net_vanna': net_vanna,
        'net_charm': net_charm,
        'atm_iv_30d': find_atm_iv_30d(spot, contracts),
        'skew_25_delta': find_skew_25_delta(contracts.assign(spot=spot)),
        'max_pain': max_pain,
        'max_pain_expiration': max_pain_expiration,
    }


def clean_expiration(calls: pd.DataFrame, puts: pd.DataFrame, fallback_iv: float) -> pd.DataFrame:
    """returns the open contracts of one expiration with a usable IV on every row.
        unusable IVs are interpolated over the strikes of the same expiration,
        if there is nothing to interpolate from the realized volatility is used.
    """
    frames = []
    for chain, sign in ((calls, 1), (puts, -1)):
        frames.append(pd.DataFrame({
            'strike': chain['strike'].astype(float),
            'oi': chain['openInterest'].fillna(0).astype(float),
            'iv': chain['impliedVolatility'].astype(float),
            'sign': sign,
        }))
    contracts = pd.concat(frames, ignore_index=True)
    contracts = contracts[contracts['oi'] > 0]

    valid = contracts['iv'].between(MIN_VALID_IV, MAX_VALID_IV)
    smile = contracts[valid].groupby('strike')['iv'].mean()
    if len(smile) >= 2:
        interpolated = np.interp(
            contracts['strike'], smile.index.to_numpy(), smile.to_numpy())
    else:
        interpolated = np.full(len(contracts), fallback_iv)
    contracts['iv'] = np.where(valid, contracts['iv'], interpolated)
    contracts['iv_estimated'] = ~valid

    return contracts


def next_date(value, today: date) -> date | None:
    """first date from today on out of a date or a list of dates as Yahoo's calendar delivers them"""
    values = value if isinstance(value, (list, tuple)) else [value]
    values = [v.date() if isinstance(v, datetime) else v for v in values]
    upcoming = sorted(v for v in values if isinstance(v, date) and v >= today)
    return upcoming[0] if upcoming else None


def fetch_calendar(ticker: yf.Ticker, today: date) -> dict:
    """ETFs have no calendar and Yahoo changes this endpoint now and then, so it must never fail the symbol"""
    try:
        calendar = ticker.calendar or {}
        return {
            'next_earnings_date': next_date(calendar.get('Earnings Date'), today),
            'ex_dividend_date': next_date(calendar.get('Ex-Dividend Date'), today),
        }
    except Exception:
        return {'next_earnings_date': None, 'ex_dividend_date': None}


def fetch_contracts(symbol: str, today: date) -> tuple[float, date, pd.DataFrame, dict]:
    ticker = yf.Ticker(symbol)

    history = ticker.history(period='3mo', auto_adjust=False)
    spot = float(history['Close'].iloc[-1])
    as_of = history.index[-1].date()
    realized_volatility = float(
        np.log(history['Close']).diff().tail(30).std() * np.sqrt(252))

    extras = {
        'realized_volatility_30d': realized_volatility * 100,
        'call_volume': 0,
        'put_volume': 0,
        **fetch_calendar(ticker, today),
    }

    frames = []
    near_spot_rows = 0
    near_spot_rows_with_oi = 0
    for expiration in ticker.options:
        days_to_expiration = (
            datetime.strptime(expiration, '%Y-%m-%d').date() - today).days
        if days_to_expiration < 0 or days_to_expiration > MAX_DAYS_TO_EXPIRATION:
            continue

        chain = ticker.option_chain(expiration)
        extras['call_volume'] += int(chain.calls['volume'].fillna(0).sum())
        extras['put_volume'] += int(chain.puts['volume'].fillna(0).sum())
        for side in (chain.calls, chain.puts):
            near_spot = side[side['strike'].between(spot * 0.9, spot * 1.1)]
            near_spot_rows += len(near_spot)
            near_spot_rows_with_oi += int(
                (near_spot['openInterest'].fillna(0) > 0).sum())
        contracts = clean_expiration(
            chain.calls, chain.puts, realized_volatility)
        # expiration day itself counts as half a day
        contracts['t'] = max(days_to_expiration, 0.5) / 365
        contracts['expiration'] = today + timedelta(days=days_to_expiration)
        frames.append(contracts)

    if not frames:
        return spot, as_of, pd.DataFrame(), extras
    if near_spot_rows_with_oi < MIN_OPEN_INTEREST_COVERAGE * near_spot_rows:
        raise OpenInterestNotPopulated(
            f'{near_spot_rows_with_oi} of {near_spot_rows} strikes near spot have open interest')
    return spot, as_of, pd.concat(frames, ignore_index=True), extras


def return_symbol_levels(symbol: str, today: date) -> dict | None:
    for attempt in range(1, OPTIONS_RETRIES + 1):
        try:
            spot, as_of, contracts, extras = fetch_contracts(symbol, today)
            levels = compute_levels(spot, contracts)
            if levels is None:
                logger.warning('%s: not enough open interest', symbol)
                return None

            estimated_oi = contracts.loc[contracts['iv_estimated'], 'oi'].sum()
            levels['iv_estimated_share'] = float(
                estimated_oi / contracts['oi'].sum())
            return {'ticker': symbol.upper(), 'date': as_of, **levels, **extras}
        except OpenInterestNotPopulated as e:
            logger.warning('%s: open interest not populated by Yahoo (%s)',
                           symbol, str(e))
            return None
        except Exception as e:
            if attempt == OPTIONS_RETRIES:
                logger.error('Error fetching options of %s: %s',
                             symbol, str(e))
            else:
                time_module.sleep(5 * attempt)

    return None


def analyze_option_symbols(symbols: list[str]) -> list[dict]:
    today = datetime.now(pytz.timezone('US/Eastern')).date()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=OPTIONS_MAX_WORKERS) as executor:
        futures = [executor.submit(return_symbol_levels, symbol, today)
                   for symbol in symbols]
        for f in tqdm(as_completed(futures), total=len(futures)):
            levels = f.result()
            if levels is not None:
                results.append(levels)

    return results


def insert_levels_into_database(levels: list[dict]) -> None:
    query = """
        INSERT INTO gamma_levels (
            ticker, date, spot, call_wall, put_wall, gamma_flip,
            net_gex, call_gex, put_gex, call_oi, put_oi, expirations,
            net_vanna, net_charm, atm_iv_30d, max_pain, max_pain_expiration,
            skew_25_delta, realized_volatility_30d, call_volume, put_volume,
            next_earnings_date, ex_dividend_date
        )
        VALUES %s
        ON CONFLICT (ticker, date)
        DO UPDATE SET
            spot = EXCLUDED.spot,
            call_wall = EXCLUDED.call_wall,
            put_wall = EXCLUDED.put_wall,
            gamma_flip = EXCLUDED.gamma_flip,
            net_gex = EXCLUDED.net_gex,
            call_gex = EXCLUDED.call_gex,
            put_gex = EXCLUDED.put_gex,
            call_oi = EXCLUDED.call_oi,
            put_oi = EXCLUDED.put_oi,
            expirations = EXCLUDED.expirations,
            net_vanna = EXCLUDED.net_vanna,
            net_charm = EXCLUDED.net_charm,
            atm_iv_30d = EXCLUDED.atm_iv_30d,
            max_pain = EXCLUDED.max_pain,
            max_pain_expiration = EXCLUDED.max_pain_expiration,
            skew_25_delta = EXCLUDED.skew_25_delta,
            realized_volatility_30d = EXCLUDED.realized_volatility_30d,
            call_volume = EXCLUDED.call_volume,
            put_volume = EXCLUDED.put_volume,
            next_earnings_date = EXCLUDED.next_earnings_date,
            ex_dividend_date = EXCLUDED.ex_dividend_date,
            last_updated_at = NOW();
    """
    values = [(
        row['ticker'], row['date'], to_float(row['spot']), to_float(
            row['call_wall']), to_float(row['put_wall']),
        to_float(row['gamma_flip']), to_float(
            row['net_gex']), to_float(row['call_gex']),
        to_float(row['put_gex']), row['call_oi'], row['put_oi'], row['expirations'],
        to_float(row['net_vanna']), to_float(row['net_charm']), to_float(row['atm_iv_30d']),
        to_float(row['max_pain']), row['max_pain_expiration'],
        to_float(row['skew_25_delta']), to_float(row['realized_volatility_30d']),
        row['call_volume'], row['put_volume'], row['next_earnings_date'], row['ex_dividend_date']
    ) for row in levels]

    connection = psycopg2.connect(DATABASE_URL)
    try:
        with connection.cursor() as cur:
            cur.execute(CREATE_GAMMA_LEVELS_TABLE_QUERY)
            execute_values(cur, query, values)
        connection.commit()
    finally:
        connection.close()


def get_option_symbols() -> list[str]:
    symbols = list(OPTIONS_EXTRA_SYMBOLS)
    for listed_symbols in get_symbols_from_csv(OPTIONS_INDEX_LIST).values():
        symbols.extend(listed_symbols)
    return list(dict.fromkeys(symbols))  # unique, keeps order


def scrape_prices_and_indicators() -> float:
    """returns the share of symbols that failed"""
    logger.info('Processing symbols and check for indicator matches...')
    symbols_from_csv = get_symbols_from_csv(INDEX_LIST)
    start_timestamp, end_timestamp = get_trading_time_range_timestamps()

    total_symbols = sum(len(s) for s in symbols_from_csv.values())

    start_analyzing = datetime.now()
    stock_dfs, failed = analyze_symbols_multi_process(
        symbols_from_csv, start_timestamp, end_timestamp)
    analyzing_duration_in_seconds = (
        datetime.now() - start_analyzing).total_seconds()

    failure_rate = len(failed) / total_symbols if total_symbols else 0.0
    logger.info('Analyzed %s of %s symbols (%s failed, %.1f%%)',
                len(stock_dfs), total_symbols, len(failed), failure_rate * 100)
    if failed:
        logger.warning('Failed symbols: %s', ', '.join(sorted(failed)))

    start_inserting = datetime.now()
    insert_symbols_data_into_database(stock_dfs)
    # save_dfs_to_excel(stock_dfs)
    inserting_duration_in_seconds = (
        datetime.now() - start_inserting).total_seconds()

    logger.info('Processing took %s seconds', analyzing_duration_in_seconds)
    logger.info('Inserting / saving took %s seconds',
                inserting_duration_in_seconds)

    return failure_rate


def scrape_option_levels(symbols: list[str], dry_run: bool) -> None:
    start = datetime.now()
    symbol_levels = analyze_option_symbols(symbols)
    if len(symbol_levels) < len(symbols) / 2:
        logger.error('Only %s of %s symbols delivered usable option chains. Yahoo has no open '
                     'interest outside of / before the US session, check the time of the run.',
                     len(symbol_levels), len(symbols))
    logger.info('Option levels for %s symbols took %s seconds',
                len(symbol_levels), (datetime.now() - start).total_seconds())

    estimated = [row['iv_estimated_share'] for row in symbol_levels]
    if estimated:
        logger.info('Share of open interest with estimated IV: median %.0f%%, max %.0f%%',
                    np.median(estimated) * 100, max(estimated) * 100)

    if dry_run:
        print(f'{len(symbol_levels)} of {len(symbols)} symbols usable, see logs/ for skipped ones')
        # one symbol per column, there are too many values for one row
        print(pd.DataFrame(symbol_levels).set_index('ticker').T.to_string()
              if symbol_levels else 'no data')
    elif symbol_levels:
        try:
            insert_levels_into_database(symbol_levels)
        except Exception as e:
            logger.error('Error inserting option levels: %s', str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('symbols', nargs='*',
                        help='with --dry-run: option symbols instead of the configured lists')
    parser.add_argument('--only', choices=['prices', 'options'],
                        help='run just one of the two parts')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the option levels of the given symbols, nothing is written')
    args = parser.parse_args()

    setup_logger()
    logger.info('\nStarting indicator script...')
    logger.info(datetime.now().strftime('%d/%m/%Y, %H:%M:%S'))

    prepare_yfinance_cache()

    if args.dry_run:
        scrape_option_levels(args.symbols or get_option_symbols(), dry_run=True)
        sys.exit(0)

    verify_environment_variables_are_set()

    # one after the other, so both parts don't hit Yahoo's rate limit together
    failure_rate = 0.0
    if args.only != 'options':
        failure_rate = scrape_prices_and_indicators()
    if args.only != 'prices':
        scrape_option_levels(get_option_symbols(), dry_run=False)

    # send request to refresh cache after new data is inserted
    try:
        revalidate_stock_screener_cache()
    except Exception as e:
        logger.error('Revalidation failed: %s', str(e))

    # Exit non-zero so systemd - and a Kuma push monitor - see a run that
    # technically finished but produced a fraction of the data.
    if failure_rate > MAX_FAILURE_RATE:
        logger.error('%.1f%% of symbols failed, above the %.1f%% threshold',
                     failure_rate * 100, MAX_FAILURE_RATE * 100)
        sys.exit(1)
