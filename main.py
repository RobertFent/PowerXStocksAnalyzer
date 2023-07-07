'''stock analyzer based on PowerXStrategy

may use tradier or yahoo finance data (based on vars in .env)

example usage:
python3 ./main.py

of if run as cronjob:
python3 main.py --cron 1
'''
import argparse
import concurrent.futures
import csv
import ftplib
import threading
import io
import json
from datetime import datetime, timedelta
import math
from pathlib import Path
from dotenv import dotenv_values
import pytz
import requests
import pandas_ta as ta
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np

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
# how long the dataframe should be
DAYS=120
# max implied volatility value -> don't trade stocks that are too volatile
MAX_IV = 100
# timeout for request in seconds
TIMEOUT = 3

LOCK = threading.Lock()

# global dynamic params
current_date = None
past_date = None
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

    # current date is prev. day if using Tradier
    curr_days = 1 if TRADIER == 'True' else 0
    current_date = (current_date_est  - timedelta(days=curr_days)).strftime(time_format)
    past_date = (current_date_est - timedelta(days=DAYS)).strftime(time_format)


def get_symbols_from_csv():
    '''returns list of symbols from csv.
    '''
    symbols = []

    pathlist = Path('csv/').rglob('*.csv')
    for path in pathlist:
        path_in_str = str(path)

        with open(path_in_str, 'r', encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row['Symbol']
                symbols.append(symbol)

    # use set to remove duplicates
    return set(symbols)


def get_sp500_symbols():
    '''returns list of s&p 500 listed symbols.
    '''
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, timeout=TIMEOUT)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []

    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[0].text.strip()
        symbols.append(symbol)

    return symbols


# not working anymore as problem with ftp currently happens -> using fallback .csv
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


def get_ticker_data_tradier(symbol):
    '''returns dataframe with all needed data of given symbol from tradier api.

    Keyword arguments:
    ticker -- symbol of stock
    '''

    # Set the endpoint URL and parameters
    url = 'https://api.tradier.com/v1/markets/history'

    # Make the API request
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36' +
        ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

    params = {
        'symbol': symbol,
        'start': past_date,
        'end': current_date
    }

    response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)

    # Process the response
    if response.status_code == 200:
        data = json.loads(response.text)

        quotes = data['history']['day']

        # Create a pandas DataFrame to store the data
        frame = pd.DataFrame(quotes)
        frame.index = pd.to_datetime(frame['date'])

        # Extract required columns
        frame = frame[['open', 'high', 'low', 'close', 'volume']]
        frame['ticker'] = symbol.upper()

        return frame
    else:
        #print("Error occurred while fetching stock data.")
        return None


def set_earnings_date_tradier(dataframe, symbol):

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36' +
        ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

    response = requests.get(f'https://api.tradier.com/beta/markets/fundamentals/calendars?symbols={symbol}', headers=headers, timeout=TIMEOUT)
    quote_data = response.json()

    corporate_data = quote_data[0]['results'][0]['tables']['corporate_calendars']

    curr_date = datetime.now().date()
    start_date = curr_date - timedelta(days=120)
    end_date = curr_date + timedelta(days=120)

    dataframe['next_earnings_event'] = np.nan
    dataframe['latest_earnings_event'] = np.nan

    if corporate_data is not None:
        # filter data on date
        date_filtered_data = [item for item in corporate_data if start_date <= datetime.fromisoformat(item['begin_date_time']).date() <= end_date]

        # check if earnings event is present
        earnings_filtered_data = [datetime.fromisoformat(item['begin_date_time']).date() for item in date_filtered_data if "Earnings" in item['event']]

        if len(earnings_filtered_data) > 0:

            lastest_date, next_date = None, None
            for date in earnings_filtered_data:

                # get latest event
                if lastest_date is None:
                    if date < curr_date:
                        lastest_date = date
                else:
                    if date < curr_date and date > lastest_date:
                        lastest_date = date

                # get next event
                if next_date is None:
                    if date >= curr_date:
                        next_date = date
                else:
                    if date > curr_date and date < next_date:
                        next_date = date

                
            dataframe.loc[dataframe.index[-1], 'next_earnings_event'] = next_date
            dataframe.loc[dataframe.index[-1], 'latest_earnings_event'] = lastest_date
        
        #print(earnings_filtered_data)
        #has_earnings = earnings_filtered_data is not None
        #print(has_earnings)



def get_ticker_data_yahoo(symbol):
    '''returns dataframe with all needed data of given symbol from yahoo.finance api.

    Keyword arguments:
    ticker -- symbol of stock
    '''

    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol

    # header for request to not get forbidden error
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36' +
               ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    params = {
        'period1': int(pd.Timestamp(past_date).timestamp()),
        'period2': int(pd.Timestamp(current_date).timestamp()),
        'interval': '1d',
    }

    # send request
    response = requests.get(url, params, headers=headers, timeout=TIMEOUT)

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
        frame['ticker'] = symbol.upper()

        return frame

    return None


def set_option_data_tradier(ticker, dataframe, target_price):
    '''sets next price below current price, fitting exp. date, spreads etc.

    Keyword arguments:
    ticker -- symbol of stock
    dataframe -- ticker data as pd dataframe
    target_price -- current price of stock
    '''

    dataframe['Strike-Price'] = np.nan
    dataframe['Exp.-Date'] = np.nan
    dataframe['Option-High'] = np.nan
    dataframe['bid'] = np.nan
    dataframe['ask'] = np.nan
    dataframe['effective-spread'] = np.nan

    response = requests.get('https://api.tradier.com/v1/markets/options/expirations',
        params={'symbol': ticker, 'includeAllRoots': 'true', 'strikes': 'true'},
        headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'},
        timeout=TIMEOUT
    )

    options = response.json()

    if response.status_code == 200:

        fitting_options = []

        curr_date = datetime.now().date()
        min_date = curr_date + timedelta(days=30)
        max_date = curr_date + timedelta(days=45)
        for option in options['expirations']['expiration']:
            curr_option_date = datetime.fromisoformat(option['date']).date()
            if curr_option_date > min_date and curr_option_date < max_date:
                fitting_options.append(option)

        if len(fitting_options) > 0:
            next_price = 0
            for strike in fitting_options[0]['strikes']['strike']:
                if strike < target_price and strike > next_price:
                    next_price = strike

            response = requests.get('https://api.tradier.com/v1/markets/options/chains',
            params={'symbol': ticker, 'expiration': fitting_options[0]['date'], 'greeks': 'true'},
            headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'},
            timeout=TIMEOUT)

            options = response.json()

            if response.status_code == 200:
                for option in options['options']['option']:
                    # should always be only one
                    if(option['strike'] == next_price and "Call" in option['description']):
                        if option['close'] is not None:
                            mid = option['ask'] - option['bid']
                            # https://en.wikipedia.org/wiki/Bid–ask_spread#Effective_spread
                            effective_spread = 2 * (np.abs(option['close'] - mid)/mid) * 100
                        else:
                            effective_spread = None
                        dataframe.loc[dataframe.index[-1], 'ask'] = option['ask']
                        dataframe.loc[dataframe.index[-1], 'bid'] = option['bid']
                        dataframe.loc[dataframe.index[-1], 'effective_spread'] = effective_spread
                        dataframe.loc[dataframe.index[-1], 'symbol'] = option['symbol']
                        dataframe.loc[dataframe.index[-1], 'opt-close'] = option['close']
                        dataframe.loc[dataframe.index[-1], 'opt-high'] = option['high']

            dataframe.loc[dataframe.index[-1], 'Strike-Price'] = next_price
            dataframe.loc[dataframe.index[-1], 'Exp.-Date'] = fitting_options[0]['date']


# todo: set instead of return
def set_option_data_yahoo(ticker, dataframe, target_price):
    '''todo: exp. etc.
    '''
    url = 'https://query2.finance.yahoo.com/v7/finance/options/' + ticker

    # header for request to not get forbidden error
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36' +
               ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    params = {
        'period1': int(pd.Timestamp(past_date).timestamp()),
        'period2': int(pd.Timestamp(current_date).timestamp()),
        'interval': '1d',
    }
    response = requests.get(url, params, headers=headers, timeout=TIMEOUT)

    if response.status_code == 200:
        data = response.json()
        options = data['optionChain']['result'][0]['options']
        if options:
            calls = data['optionChain']['result'][0]['options'][0]['calls']
            # iterate over inverted list to start with max call and break at first call lower
            for call in calls[::-1]:
                strike_price = call['strike']
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


def add_stochastic_slow(dataframe):
    '''adds stochastic slow(14, 3, 3) values.

    # todo?: maybe use this info
    if %K crosses above %D -> buy signal
    if %K crosses below %D -> sell signal

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


def add_implied_volatility(dataframe):
    '''adds implied volatility values.

    Keyword arguments:
    dataframe -- ticker data as pd dataframe
    '''
    dataframe['returns'] = dataframe['close'].pct_change()
    std_dev = dataframe['returns'].std()
    # trading_days_per_year = 252
    annualized_std_dev = std_dev * np.sqrt(252)
    dataframe['implied_volatilitiy'] = np.nan
    dataframe.loc[dataframe.index[-1], 'implied_volatilitiy'] = annualized_std_dev * 100


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
            dataframe.iloc[[len(dataframe)-1]]['RSI'].item() < MAX_RSI and
            dataframe.iloc[[len(dataframe)-1]]['implied_volatilitiy'].item() < MAX_IV)


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

    # add data for options trading
    (set_option_data_tradier(dataframe.iloc[[len(dataframe)-1]]['ticker'].item(),
                             dataframe, entry[len(entry)-1]) if TRADIER == 'True' else set_option_data_yahoo(
        dataframe.iloc[[len(dataframe)-1]]['ticker'].item(), dataframe, entry[len(entry)-1]))

    # dataframe['ADR'] = adr
    dataframe['Next-Entry'] = entry
    dataframe['Stop-Loss'] = stop_loss
    dataframe['Limit-Order'] = limit_order
    dataframe['Max-Shares'] = shares_to_buy


def analyze_ticker(ticker):
    '''analyze add all indicators etc. to stock data.

    Keyword arguments:
    tickers -- list of all symbols
    '''
    try:
        ticker_data = (get_ticker_data_tradier(ticker) if TRADIER == 'True'
                       else get_ticker_data_yahoo(ticker))
        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)
        add_implied_volatility(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        return ticker_data

    except Exception as e:
        #print(str(e))
        pass


def analyze_ticker_wrapper(ticker):
    '''Analyze stock and determine if it's a winner or loser.

    Keyword arguments:
    ticker -- stock ticker symbol
    '''
    ticker_data = analyze_ticker(ticker)
    #print(ticker_data)
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


def process_algorithm():
    '''set all winners and loosers of each index.
    '''
    ''' way of getting symbols online
    print('Analyzing NASDAQ...')
    tickers = get_nasdaq_symbols()
    analyze_stocks(tickers)
    print('%d stocks in NASDAQ analyzed' % len(tickers))

    print('Analyzing S&P500...')
    tickers = get_sp500_symbols()
    # print(tickers)
    analyze_stocks(tickers)
    print('%d stocks in S&P500 analyzed' % len(tickers))
    '''
    print('Analyzing Nasdaq, NYSE, DJIA and S&P500...')
    tickers = get_symbols_from_csv()
    #print(tickers)
    analyze_stocks(tickers)
    print('%d stocks in  Nasdaq, NYSE, DJIA and S&P500 analyzed' % len(tickers))


def get_info(ticker, options=False, debug=False, mobile=False):
    '''prints the ticker data.

    Keyword arguments:
    ticker -- symbol of ticker (e.g. AAPL)
    options -- wether ticker data should be for stocks or option trading
    '''
    try:
        ticker_data = (get_ticker_data_tradier(ticker) if TRADIER == 'True'
                       else get_ticker_data_yahoo(ticker))

        add_order_values(ticker_data)

        # for debugging
        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)
        add_implied_volatility(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        # todo: yahoo finance data
        if TRADIER == 'True':
            set_earnings_date_tradier(ticker_data, ticker)
        else:
            ticker_data['next_earnings_event'] = np.nan
            ticker_data['latest_earnings_event'] = np.nan
        if options:
            ticker_data = ticker_data[[
                'ticker', 'symbol', 'Strike-Price', 'Exp.-Date', 'opt-high', 'opt-close',
                'bid', 'ask', 'effective_spread', 'color', 'implied_volatilitiy', 'next_earnings_event', 'latest_earnings_event']]
            print('Details for OPTIONS-Trading:')
        elif mobile:
            ticker_data = ticker_data[[
                'ticker', 'close', 'Next-Entry', 'Stop-Loss', 'Limit-Order']]
            print('Details for Stock-Trading (mobile format):')
        elif debug:
            ticker_data = ticker_data[[
                'ticker', 'high', 'close', 'open', 'low', 'MACD Line',
                'Signal Line', '%K', '%D', 'RSI', 'volume', 'color']]
            print('Details for debugging:')
        else:
            ticker_data = ticker_data[[
                'ticker', 'high', 'close', 'Next-Entry', 'Stop-Loss',
                'Limit-Order', 'Max-Shares', 'color', 'volume', 'implied_volatilitiy', 'next_earnings_event', 'latest_earnings_event']]
            print('Details for Stock-Trading:')
            
        print(ticker_data)
        print('Check out the chart for further details: https://finance.yahoo.com/quote/%s?p=%s' %
              (ticker, ticker))
    except Exception as e:
        print(str(e))
        print(e.with_traceback())


def save_output_to_file(text):
    '''saves text into file with current date as title
    '''
    # todo: create output folder if not existing
    log_date = (datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
                if TRADIER == 'True' else current_date)
    filename = 'output/%s.txt' % log_date
    path = Path(filename)

    # file exists
    if path.is_file():
        with open(filename, 'a', encoding='UTF-8') as f:
            print(text, file=f)

    # create file if not existing
    else:
        with open(filename, 'w', encoding='UTF-8') as f:
            print(filename, file=f)
            print(text, file=f)


def main(cron):
    '''starts program based on user input.
    '''
    # set dates properly for yahoo or tradier
    set_dates('%Y-%m-%d') if TRADIER == 'True' else set_dates()
    print('Welcome to PowerXStocksAnalyzer!\nLast day to analyze is: %s' % (current_date))
    choice = (input(
        'What do you want to do? 1: get a list of stocks in ' +
        'buy zone | put position or {ticker symbol}: get specific info of a given symbol?: ')
        if cron != 1 else None)
    if choice == '1' or cron == 1:
        process_algorithm()
        print('Processing done! Check output here: output/%s.txt' % current_date)
        # winner
        save_output_to_file('Stocks in buy zone:\n' + ', '.join(winning_stocks))
        if len(winning_stocks) > 0:
            save_output_to_file('\nCheck out the stocks here:')
            for winner in winning_stocks:
                save_output_to_file('https://finance.yahoo.com/chart/' + winner)
        # loser
        save_output_to_file('\nStocks in put position:\n' + ', '.join(losing_stocks))
        if len(losing_stocks) > 0:
            save_output_to_file('\nCheck out the stocks here:')
            for loser in losing_stocks:
                save_output_to_file('https://finance.yahoo.com/chart/' + loser)

        save_output_to_file('\nBut watch out -> do not trade stocks with gaps in the chart!')
    else:
        choice1 = input(
            'Do you trade stocks or options? 1=stocks (short format); ' +
            '2=stocks (long format); 3=stocks (mobile format); 4=options; 5=debugging: ')
        if choice1 == '1':
            get_info(choice)
        elif choice1 == '2':
            pd.set_option('display.max_rows', None)
            get_info(choice)
        elif choice1 == '3':
            get_info(choice, mobile=True)
        elif choice1 == '4':
            get_info(choice, True)
        elif choice1 == '5':
            get_info(choice, False, True)
        else:
            print('Wrong input!')


if __name__ == '__main__':

    # read vars from .env file
    env_vars = dotenv_values('.env')
    TRADIER = env_vars['TRADIER']
    API_KEY = env_vars['API_KEY']

    parser = argparse.ArgumentParser()
    parser.add_argument("--cron", type=int, default=0, help="0 if not cronjob 1 else")
    args = parser.parse_args()

    main(args.cron)
