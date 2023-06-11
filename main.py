'''stock analyzer based on PowerXStrategy
'''
import concurrent.futures
import ftplib
import threading
import io
from datetime import datetime, timedelta
import math
import pytz
import requests
import pandas_ta as ta
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

# only stocks below 80 bucks
MAX_STOCKPRICE = 80
# only stocks with at least this daily volume
MIN_VOLUME = 10000000
# value of my deposit
DEPOSIT_VALUE = 10000
# RSI > 80 may indicate overbought -> use 85 because of difference to yahoos charts
MAX_RSI = 85
# RSI > 50 -> use 60 because of difference to yahoos charts
MIN_RSI = 50
# %K of stochastic slow should be over 50 -> use 60 because of difference to yahoos charts
MIN_STOCH_D = 50
# url of API
BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/'
# header for request to not get forbidden error
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36' +
           ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
LOCK = threading.Lock()

# global dynamic params
current_date = None
past_date = None
params = None
winning_stocks = []
losing_stocks = []


def set_dates(time_format='%m/%d/%Y'):
    '''returns dates in est time zone.

    Keyword arguments:
    time_format -- wanted format of time
    '''
    global current_date, past_date
    curr_date = datetime.now()
    est_timezone = pytz.timezone('US/Eastern')
    # convert to est
    current_date_est = curr_date.astimezone(est_timezone)

    formated_date = current_date_est.strftime(time_format)
    past_month_date = current_date_est - timedelta(days=50)
    formated_past_month_date = past_month_date.strftime(time_format)

    current_date = formated_date
    past_date = formated_past_month_date

def init_request_params():
    '''inits global params used for sending a request.
    '''
    global params
    params = {
        'period1': int(pd.Timestamp(past_date).timestamp()),
        'period2': int(pd.Timestamp(current_date).timestamp()),
        'interval': '1d',
        # 'events': 'div,splits'
    }


def get_sp500_symbols():
    '''returns list of s&p 500 listed symbols.
    '''
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
    '''returns list of nasdaq listed symbols.
    '''
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
    '''returns list of dow jones listed symbols.
    '''
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
    '''returns dataframe with all needed data of given symbol.

    Keyword arguments:
    ticker -- symbol of stock
    '''
    url = BASE_URL + ticker

    # send request
    response = requests.get(url, params, headers=HEADERS, timeout=5)

    if response.status_code == 200:
        # get JSON response
        data = response.json()

        # get open / high / low / close data
        frame = pd.DataFrame(data['chart']['result']
                             [0]['indicators']['quote'][0])

        # get the date info
        temp_time = data['chart']['result'][0]['timestamp']

        # add time
        frame.index = pd.to_datetime(temp_time, unit='s')
        frame.index = frame.index.map(lambda dt: dt.floor('d'))

        # reorder frame
        frame = frame[['open', 'high', 'low', 'close', 'volume']]
        frame['ticker'] = ticker.upper()

        return frame

    return None


def get_option_strike_price(ticker, target_price):
    '''todo: exp. etc.
    '''
    url = 'https://query2.finance.yahoo.com/v7/finance/options/' + ticker
    response = requests.get(url, params, headers=HEADERS, timeout=5)
    if response.status_code == 200:
        data = response.json()
        options = data['optionChain']['result'][0]['options'][0]['calls']
        # iterate over inverted list to start with max call and break at first call lower
        for option in options[::-1]:
            strike_price = option['strike']
            """
            expiration_date = option['expiration']
            # contract should expire within 30-45 days
            future1 = (datetime.now() + timedelta(days=10)).timestamp()
            future2 = (datetime.now() + timedelta(days=45)).timestamp()
            if (strike_price < target_price and
                expiration_date > future1 and
                    expiration_date < future2):
                print(option)
                return option['strike'], option['contractSymbol']
            """
            if strike_price < target_price:
                return strike_price

    return None


# MACD
def add_macd_data(dataframe):
    '''adds macd(26, 12, 9) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    ema_12 = dataframe['close'].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe['close'].ewm(span=26, adjust=False).mean()

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    dataframe['MACD Line'] = macd_line
    dataframe['Signal Line'] = signal_line

# RSI
def add_rsi_data(dataframe):
    '''adds rsi(7) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    '''old code
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
    '''
    rsi = ta.rsi(dataframe['close'], length=7)
    dataframe['RSI'] = rsi


# todo: not always the most accurate result for %D
# stochastic slow
def add_stochastic_slow(dataframe):
    '''adds stochastic slow(14, 3, 3) values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    # Calculate the lowest low over the past 14 days
    lowest_low = dataframe['low'].rolling(window=14).min()

    # Calculate the highest high over the past 14 days
    highest_high = dataframe['high'].rolling(window=14).max()

    # Calculate %K (fast stochastic)
    percent_k = ((dataframe['close'] - lowest_low) /
                 (highest_high - lowest_low)) * 100
    # Calculate %D  (3-day simple moving average of %K) (fast stochastic)
    percent_d = percent_k.rolling(window=3).mean()

    # slow stochastic
    dataframe['%K'] = percent_d
    dataframe['%D'] = dataframe['%K'].rolling(window=3).mean()


def add_color_of_days(dataframe):
    '''colors day to green, black or red.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''

    colors = []
    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        green_indicators = 0

        # macd
        if dataframe.iloc[[i]]['MACD Line'].item() - dataframe.iloc[[i]]['Signal Line'].item() > 0:
            green_indicators += 1
        # rsi
        if dataframe.iloc[[i]]['RSI'].item() >= MIN_RSI:
            green_indicators += 1
        # stochastic slow
        if dataframe.iloc[[i]]['%D'].item() >= MIN_STOCH_D:
            green_indicators += 1

        if math.isnan(dataframe.iloc[[i]]['%D'].item()):
            colors.append('NaN')
            continue

        if green_indicators == 3:
            colors.append('green')
        elif (green_indicators < 3 and green_indicators > 0):
            colors.append('black')
        else:
            colors.append('red')

    dataframe['color'] = colors


def is_winner(dataframe):
    '''decide if stock is a winner.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    # current day should be green and the day before not
    return (dataframe.iloc[[len(dataframe)-1]]['color'].item() == 'green' and
            dataframe.iloc[[len(dataframe)-2]]['color'].item() != 'green' and
            dataframe.iloc[[len(dataframe)-1]]['close'].item() < MAX_STOCKPRICE and
            dataframe.iloc[[len(dataframe)-1]]['volume'].item() > MIN_VOLUME and
            dataframe.iloc[[len(dataframe)-1]]['RSI'].item() < MAX_RSI)


def is_loser(dataframe):
    '''decide if stock is a loser.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    # current day should be red and the day before not
    return (dataframe.iloc[[len(dataframe)-1]]['color'].item() == 'red' and
            dataframe.iloc[[len(dataframe)-2]]['color'].item() != 'red' and
            dataframe.iloc[[len(dataframe)-1]]['close'].item() < MAX_STOCKPRICE and
            dataframe.iloc[[len(dataframe)-1]]['volume'].item() > MIN_VOLUME)


def add_order_values(dataframe):
    '''add stock and option trading data.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''

    adr = [None] * 6
    entry = []
    stop_loss = [None] * 6
    limit_order = [None] * 6
    shares_to_buy = [None] * 6
    strike_price = [None] * (len(dataframe) - 1)

    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        # entry for next day
        next_entry = dataframe.iloc[[i]]['high'].item() + 0.01
        entry.append(next_entry)
        if i > 5:
            sum_values = 0
            for j in range(i-7, i):
                sum_values += dataframe.iloc[[j]]['high'].item() - \
                    dataframe.iloc[[j]]['low'].item()

            # adr of current day
            current_adr = sum_values / 7
            # stop_loss and limit order
            stop_loss_current_day = next_entry - current_adr
            limit_order_current_day = next_entry + 2 * current_adr
            # max amount of shares to buy to not loose more than 2% of depot in case of stop-loss
            max_shares_current_day = (
                DEPOSIT_VALUE*0.02) / (next_entry - stop_loss_current_day)

            # append all needed values
            shares_to_buy.append(max_shares_current_day)
            adr.append(current_adr)
            stop_loss.append(stop_loss_current_day)
            limit_order.append(limit_order_current_day)

    strike_price.append(get_option_strike_price(
        dataframe.iloc[[len(dataframe)-1]]['ticker'].item(), entry[len(entry)-1]))

    # dataframe['ADR'] = adr
    dataframe['Next-Entry'] = entry
    dataframe['Stop-Loss'] = stop_loss
    dataframe['Limit-Order'] = limit_order
    dataframe['Max-Shares'] = shares_to_buy
    dataframe['Strike-Price'] = strike_price


def analyze_ticker(ticker):
    '''analyze add all indicators etc. to stock data.

    Keyword arguments:
    tickers -- list of all symbols
    '''
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
    '''Analyze stock and determine if it's a winner or loser.

    Keyword arguments:
    ticker -- stock ticker symbol
    '''
    ticker_data = analyze_ticker(ticker)
    if ticker_data is not None:
        if is_winner(ticker_data):
            with LOCK:
                winning_stocks.append(ticker)
        elif is_loser(ticker_data):
            with LOCK:
                losing_stocks.append(ticker)


def analyze_stocks(tickers):
    '''Analyze all stocks in concurrent threads with progress bar.

    Keyword arguments:
    tickers -- list of all symbols
    '''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_ticker_wrapper, ticker)
                   for ticker in tickers]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


def calc_attribs():
    '''set all winners and loosers of each index.
    '''
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

def get_info(ticker, options=False, debug=False):
    '''prints the ticker data.

    Keyword arguments:
    ticker -- symbol of ticker (e.g. AAPL)
    options -- wether ticker data should be for stocks or option trading
    '''
    try:
        ticker_data = get_ticker_data(ticker)

        add_order_values(ticker_data)

        # for debugging
        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        if options:
            # todo: get data for options
            ticker_data = ticker_data[[
                'ticker', 'high', 'close', 'Next-Entry', 'Strike-Price', 'color']]
            print('Details for OPTIONS-Trading:')
        elif not options and not debug:
            ticker_data = ticker_data[[
                'ticker', 'high', 'close', 'Next-Entry', 'Stop-Loss',
                'Limit-Order', 'Max-Shares', 'color', 'volume']]
            print('Details for Stock-Trading:')
        else:
            ticker_data = ticker_data[[
                'ticker', 'high', 'close', 'open', 'low','MACD Line', 'Signal Line', '%K', '%D', 'RSI', 'volume', 'color']]
            print('Details for debugging:')

        print(ticker_data)
        print('Check out the chart for further details: https://finance.yahoo.com/quote/%s?p=%s' %
              (ticker, ticker))
    except Exception as e:
        print(str(e))


def main():
    '''starts program based on user input.
    '''
    set_dates()
    init_request_params()
    choice = input(
        'Welcome to PowerXStocksAnalyzer!\nWhat do you want to do? 1: get a list of stocks in ' +
        'buy zone | short position or {ticker symbol}: get specific info of a given symbol?: ')
    if choice == '1':
        calc_attribs()
        # winner
        print('Stocks in buy zone:\n' + ', '.join(winning_stocks))
        if len(winning_stocks) > 0:
            print('\nBut watch out -> do not trade stocks with gaps in the chart!' +
                  '\nCheck out the stocks here:')
            for winner in winning_stocks:
                print('https://finance.yahoo.com/chart/' + winner)
        # loser
        print('Stocks in short position:\n' + ', '.join(losing_stocks))
        if len(losing_stocks) > 0:
            print('\nCheck out the stocks here:')
            for loser in losing_stocks:
                print('https://finance.yahoo.com/chart/' + loser)
    else:
        choice1 = input(
            'Do you trade stocks or options? 1=stocks; 2=options; 3=debugging: ')
        if choice1 == '1':
            get_info(choice)
        elif choice1 == '2':
            get_info(choice, True)
        elif choice1 == '3':
            get_info(choice, False, True)
        else:
            print('Wrong input!')


main()
