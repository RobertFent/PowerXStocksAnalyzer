"""stock analyzer based on PowerXStrategy
"""
import concurrent.futures
import ftplib
import threading
import io
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

# only stocks below 80 bucks
MAX_STOCKPRICE = 80
# only stocks with at least this daily volume
MIN_VOLUME = 20000000
# value of my deposit
DEPOSIT_VALUE = 10000
# RSI > 80 may indicate overbought -> use 85 because of difference to yahoos charts
MAX_RSI = 85
# RSI > 50 -> use 60 because of difference to yahoos charts
MIN_RSI = 50
# %K of stochastic slow should be over 50 -> use 60 because of difference to yahoos charts
MIN_STOCH_K = 50
# url of API
BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/'


winning_stocks = []
LOCK = threading.Lock()


def get_sp500_symbols():
    """returns list of s&p 500 listed symbols.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []

    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[0].text.strip()
        symbols.append(symbol)

    return symbols


def get_nasdaq_symbols():
    """returns list of nasdaq listed symbols.
    """
    ftp = ftplib.FTP('ftp.nasdaqtrader.com')
    ftp.login()
    ftp.cwd('SymbolDirectory')

    r = io.BytesIO()
    ftp.retrbinary('RETR nasdaqlisted.txt', r.write)

    info = r.getvalue().decode()
    splits = info.split('|')

    symbols = [x for x in splits if '\r\n' in x]
    symbols = [x.split('\r\n')[1]
               for x in symbols if 'NASDAQ' not in x != '\r\n']
    symbols = [ticker for ticker in symbols if 'File' not in ticker]

    ftp.close()

    return symbols


def get_dow_jones_symbols():
    """returns list of dow jones listed symbols.
    """
    url = 'https://www.investing.com/indices/us-30-components'
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'cr1'})

    symbols = []

    for row in table.findAll('tr'):
        symbol = row.find(
            'td', {'class': 'bold left noWrap elp name'}).text.strip()
        symbols.append(symbol)

    return symbols

# todo
# dow_jones_symbols = get_dow_jones_symbols()
# print(dow_jones_symbols)


def get_ticker_data(ticker):
    """returns dataframe with all needed data of given symbol.

    Keyword arguments:
    ticker -- symbol of stock
    """
    current_date, past_date = get_dates()
    url = BASE_URL + ticker
    params = {
        'period1': int(pd.Timestamp(past_date).timestamp()),
        'period2': int(pd.Timestamp(current_date).timestamp()),
        'interval': '1d',
        # 'events': 'div,splits'
    }

    # send request
    response = requests.get(url, params,
                            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; ' +
                                     'Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, ' +
                                     'like Gecko) Chrome/39.0.2171.95 Safari/537.36'},
                            timeout=5)

    # get JSON response
    data = response.json()

    # get open / high / low / close data
    frame = pd.DataFrame(data['chart']['result'][0]['indicators']['quote'][0])

    # get the date info
    temp_time = data['chart']['result'][0]['timestamp']

    # add time
    frame.index = pd.to_datetime(temp_time, unit='s')
    frame.index = frame.index.map(lambda dt: dt.floor('d'))

    # reorder frame
    frame = frame[['open', 'high', 'low', 'close', 'volume']]
    frame['ticker'] = ticker.upper()

    return frame


# MACD
def add_macd_data(dataframe):
    """adds macd(26, 12, 9) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """
    ema_12 = dataframe['close'].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe['close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    dataframe['MACD Line'] = macd_line
    dataframe['Signal Line'] = signal_line

# RSI


def add_rsi_data(dataframe):
    """adds rsi(7) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """
    price_changes = dataframe['close'].diff()

    # Separate gains and losses
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)

    # Calculate average gain and average loss over 7-day period
    avg_gain = gains.rolling(window=7, min_periods=1).mean()
    avg_loss = losses.rolling(window=7, min_periods=1).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    dataframe['RSI'] = rsi


# todo: not always the most accurate result for %K
# stochastic slow
def add_stochastic_slow(dataframe):
    """adds stochastic slow(14, 3, 3) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """
    # Calculate the lowest low over the past 14 days
    lowest_low = dataframe['low'].rolling(window=14).min()

    # Calculate the highest high over the past 14 days
    highest_high = dataframe['high'].rolling(window=14).max()

    # Calculate %K
    percent_k = ((dataframe['close'] - lowest_low) /
                 (highest_high - lowest_low)) * 100
    # Calculate %D (3-day simple moving average of %K)
    percent_d = percent_k.rolling(window=3, min_periods=0).mean()

    dataframe['%K'] = percent_k
    dataframe['%D'] = percent_d


def get_dates(time_format='%m/%d/%Y'):
    """returns dates in est time zone.

    Keyword arguments:
    time_format -- wanted format of time
    """
    current_date = datetime.now()
    est_timezone = pytz.timezone('US/Eastern')

    # convert to est
    current_date_est = current_date.astimezone(est_timezone)

    formated_date = current_date_est.strftime(time_format)
    past_month_date = current_date_est - timedelta(days=30)
    formated_past_month_date = past_month_date.strftime(time_format)
    return formated_date, formated_past_month_date


def add_color_of_days(dataframe):
    """colors day to green, black or red.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """

    colors = []
    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        green_indicators = 0

        # macd
        if dataframe.iloc[[i]]['MACD Line'].item() - dataframe.iloc[[i]]['Signal Line'].item():
            green_indicators += 1
        # rsi
        if dataframe.iloc[[i]]['RSI'].item() >= MIN_RSI:
            green_indicators += 1
        # stochastic slow
        if dataframe.iloc[[i]]['%K'].item() >= MIN_STOCH_K:
            green_indicators += 1

        if green_indicators == 3:
            colors.append('green')
        elif (green_indicators < 3 and green_indicators > 0):
            colors.append('black')
        else: colors.append('red')

    dataframe['color'] = colors


def is_winner(dataframe):
    """decide if stock is a winner.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """
    # current day should be green and the day before not
    return (dataframe.iloc[[len(dataframe)-1]]['color'].item() == 'green' and
            dataframe.iloc[[len(dataframe)-2]]['color'].item() != 'green' and
            dataframe.iloc[[len(dataframe)-1]]['close'].item() < MAX_STOCKPRICE and
            dataframe.iloc[[len(dataframe)-1]]['volume'].item() > MIN_VOLUME and
            dataframe.iloc[[len(dataframe)-1]]['RSI'].item() < MAX_RSI)


def add_order_values(dataframe):
    """add stock and option trading data.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    """

    adr = [None, None, None, None, None, None]
    entry = [None, None, None, None, None, None]
    stop_loss = [None, None, None, None, None, None]
    limit_order = [None, None, None, None, None, None]
    shares_to_buy = [None, None, None, None, None, None]

    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        if i > 5:
            sum_values = 0
            for j in range(i-7, i):
                sum_values += dataframe.iloc[[j]]['high'].item() - \
                    dataframe.iloc[[j]]['low'].item()

            # adr of current day
            current_adr = sum_values / 7
            # entry for next day
            next_entry = dataframe.iloc[[i]]['high'].item() + 0.01
            # stop_loss and limit order
            stop_loss_current_day = next_entry - current_adr
            limit_order_current_day = next_entry + 2 * current_adr
            # max amount of shares to buy to not loose more than 2% of depot in case of stop-loss
            max_shares_current_day = (
                DEPOSIT_VALUE*0.02) / (next_entry - stop_loss_current_day)
            shares_to_buy.append(max_shares_current_day)
            adr.append(current_adr)
            entry.append(next_entry)
            stop_loss.append(stop_loss_current_day)
            limit_order.append(limit_order_current_day)

            # todo: option prices

    # dataframe['ADR'] = adr
    dataframe['Next-Entry'] = entry
    dataframe['Stop-Loss'] = stop_loss
    dataframe['Limit-Order'] = limit_order
    dataframe['Max-Shares'] = shares_to_buy


def analyze_ticker(ticker):
    """analyze add all indicators etc. to stock data.

    Keyword arguments:
    tickers -- list of all symbols
    """
    try:
        # get data starting past month and ending current date
        ticker_data = get_ticker_data(ticker)
        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        return ticker_data

    except Exception:
        pass


def analyze_ticker_wrapper(ticker):
    """analyze stock and return if winner.

    Keyword arguments:
    tickers -- list of all symbols
    """
    ticker_data = analyze_ticker(ticker)
    if ticker_data is not None and is_winner(ticker_data):
        return ticker_data.iloc[[len(ticker_data)-1]]['ticker'].item()


def analyze_stocks(tickers):
    """analyze all stocks in concurrent threads.

    Keyword arguments:
    tickers -- list of all symbols
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(analyze_ticker_wrapper, tickers), total=len(tickers)))
        with LOCK:
            winning_stocks.extend(
                [ticker for ticker in results if ticker is not None])


def set_winners():
    """set all winners of each index.
    """
    print('Analyzing NASDAQ...')
    tickers = get_nasdaq_symbols()
    # print(tickers)
    analyze_stocks(tickers)
    print('%d stocks in NASDAQ analyzed' % len(tickers))

    print('Analyzing S&P500...')
    tickers = get_sp500_symbols()
    # print(tickers)
    analyze_stocks(tickers)
    print('%d stocks in S&P500 analyzed' % len(tickers))


def get_info(ticker, options=False):
    """prints the ticker data.

    Keyword arguments:
    ticker -- symbol of ticker (e.g. AAPL)
    options -- wether ticker data should be for stocks or option trading
    """
    try:
        ticker_data = get_ticker_data(ticker)
        add_order_values(ticker_data)

        # debug
        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        if options:
            # todo: get data for options
            ticker_data = ticker_data[[
                'ticker', 'high', 'Next-Entry', 'Stop-Loss', 'Limit-Order', 'Max-Shares', 'color']]
            print('Details for OPTIONS-Trading:')
        else:
            ticker_data = ticker_data[[
                'ticker', 'high', 'Next-Entry', 'Stop-Loss', 'Limit-Order', 'Max-Shares', 'color']]
            print('Details for Stock-Trading:')

        print(ticker_data)
        print('Check out the chart for further details: https://finance.yahoo.com/quote/%s?p=%s' %
              (ticker, ticker))
    except Exception as e:
        print(str(e))

def main():
    """starts program based on user input.
    """
    choice = input(
        'Welcome to PowerXStocksAnalyzer!\nWhat do you want to do? 1: get a list of stocks in ' +
        'buy zone or {ticker symbol}: get specific info of a given symbol?: ')
    if choice == '1':
        set_winners()
        print('Stocks in buy zone:\n' + ', '.join(winning_stocks) +
              '\nCheck out the stocks here:')
        for winner in winning_stocks:
            print('https://finance.yahoo.com/chart/' + winner)
    else:
        choice1 = input(
            'Do you trade stocks or options? 1=stocks; 2=options: ')
        if choice1 == '1':
            get_info(choice)
        elif choice1 == '2':
            get_info(choice, True)
        else:
            print('Wrong input!')


main()
