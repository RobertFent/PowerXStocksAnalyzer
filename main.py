from yahoo_fin.stock_info import get_data, tickers_nasdaq, tickers_dow, tickers_sp500
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import ftplib
import io
import concurrent.futures
import threading

# only stocks below 80 bucks
MAX_STOCKPRICE = 80
# only stocks with at least this volume
MIN_VOLUME = 1000000
# value of my deposit
DEPOSIT_VALUE=10000
# url of API
BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"


winning_stocks = []
LOCK = threading.Lock()

def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []

    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[0].text.strip()
        symbols.append(symbol)

    return symbols

def get_nasdaq_symbols():
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")
    
    r = io.BytesIO()
    ftp.retrbinary('RETR nasdaqlisted.txt', r.write)
    
    info = r.getvalue().decode()
    splits = info.split("|")
    
    
    symbols = [x for x in splits if "\r\n" in x]
    symbols = [x.split("\r\n")[1] for x in symbols if "NASDAQ" not in x != "\r\n"]
    symbols = [ticker for ticker in symbols if "File" not in ticker]    
    
    ftp.close()    

    return symbols

def get_dow_jones_symbols():
    url = 'https://www.investing.com/indices/us-30-components'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'cr1'})

    symbols = []

    for row in table.findAll('tr'):
        symbol = row.find('td', {'class': 'bold left noWrap elp name'}).text.strip()
        symbols.append(symbol)

    return symbols

# todo
#dow_jones_symbols = get_dow_jones_symbols()
#print(dow_jones_symbols)


def get_ticker_data(ticker):
    current_date, past_date = get_dates()
    url = BASE_URL + ticker
    params = {
        "period1": int(pd.Timestamp(past_date).timestamp()),
        "period2": int(pd.Timestamp(current_date).timestamp()),
        "interval": "1d",
        #"events": "div,splits"
    }

    # send request
    response = requests.get(url, params, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})

    # get JSON response
    data = response.json()

    # get open / high / low / close data
    frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])

    # get the date info
    temp_time = data["chart"]["result"][0]["timestamp"]
    
    # add time
    frame.index = pd.to_datetime(temp_time, unit = "s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))

    # reorder frame
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame['ticker'] = ticker.upper()

    return frame


# MACD
def add_macd_data(dataframe):
    ema_12 = dataframe["close"].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe["close"].ewm(span=26, adjust=False).mean()

    macd_line= ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    dataframe['MACD Line'] = macd_line
    dataframe['Signal Line'] = signal_line

# RSI
def add_rsi_data(dataframe):
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

    # Calculate the lowest low over the past 14 days
    lowest_low = dataframe['low'].rolling(window=14).min()

    # Calculate the highest high over the past 14 days
    highest_high = dataframe['high'].rolling(window=14).max()

    # Calculate %K
    percent_k = ((dataframe['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    # Calculate %D (3-day simple moving average of %K)
    percent_d = percent_k.rolling(window=3, min_periods=0).mean()

    dataframe['%K'] = percent_k
    dataframe['%D'] = percent_d

def get_dates(format='%m/%d/%Y'):
    current_date = datetime.now()
    formated_date = current_date.strftime(format)
    past_month_date = current_date - timedelta(days=30)
    formated_past_month_date = past_month_date.strftime(format)
    return formated_date, formated_past_month_date

def add_color_of_days(dataframe):
    # first day has no color
    colors = [None]
    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        if(i > 0):
            green_indicators = 0

            # macd analysis: macd higher then signal, uptrend
            macd_anal = [False, False]
            # macd of current day
            current_diff_macd = dataframe.iloc[[i]]['MACD Line'].item() - dataframe.iloc[[i]]['Signal Line'].item()
            if current_diff_macd > 0: macd_anal[0] = True

            diff_to_prev_day_macd = dataframe.iloc[[i]]['MACD Line'].item() - dataframe.iloc[[i-1]]['MACD Line'].item()
            if diff_to_prev_day_macd > 0: macd_anal[1] = True

            if (macd_anal[0] & macd_anal[1]):
                # print('MACD: green day')
                green_indicators += 1

            # rsi analysis: rsi higher then 50, uptrend
            rsi_anal = [False, False]
            if dataframe.iloc[[i]]['RSI'].item() >= 50: rsi_anal[0] = True
            if dataframe.iloc[[i]]['RSI'].item() - dataframe.iloc[[i-1]]['RSI'].item() > 0 : rsi_anal[1] = True

            if (rsi_anal[0] & rsi_anal[1]):
                # print('RSI: green day')
                green_indicators += 1

            # stochastic slow analysis: %K higher then 50, uptrend
            stochastic_slow_anal = [False, False]
            if dataframe.iloc[[i]]['%K'].item() >= 50: stochastic_slow_anal[0] = True
            if dataframe.iloc[[i]]['%K'].item() - dataframe.iloc[[i-1]]['%K'].item() > 0 : stochastic_slow_anal[1] = True

            if (stochastic_slow_anal[0] & stochastic_slow_anal[1]):
                # print('Stochastic Slow: green day')
                green_indicators += 1

            
            if (green_indicators == 3): colors.append('green')
            elif (green_indicators < 3 and green_indicators > 0): colors.append('black')
            else: colors.append('red')
    
    dataframe['color'] = colors

def is_winner(dataframe):
    # current day should be green and the day before not
    return (dataframe.iloc[[len(dataframe)-1]]['color'].item() == 'green' and
            dataframe.iloc[[len(dataframe)-2]]['color'].item() != 'green' and
            dataframe.iloc[[len(dataframe)-1]]['close'].item() < MAX_STOCKPRICE and
            dataframe.iloc[[len(dataframe)-1]]['volume'].item() > MIN_VOLUME)


# todo: test if correct
def add_order_values(dataframe):
    
    adr = [None, None, None, None, None, None]
    entry = [None, None, None, None, None, None]
    stop_loss = [None, None, None, None, None, None]
    limit_order = [None, None, None, None, None, None]
    shares_to_buy = [None, None, None, None, None, None]
    
    for i in range(len(dataframe)):
        # print(dataframe.iloc[[i]])
        if(i > 5):
            sum_values = 0
            for j in range(i-7, i):
                sum_values += dataframe.iloc[[j]]['high'].item() - dataframe.iloc[[j]]['low'].item()
            
            # adr of current day
            current_adr = sum_values / 7
            # entry for next day
            next_entry = dataframe.iloc[[i]]['high'].item() + 0.01
            # stop_loss and limit order
            stop_loss_current_day = next_entry - 1.5 * current_adr
            limit_order_current_day = next_entry + 3 * current_adr
            # max amount of shares to buy to not loose more than 2% of depot in case stop-loss triggeres
            max_shares_current_day = (DEPOSIT_VALUE*0.02) / (next_entry - stop_loss_current_day)
            shares_to_buy.append(max_shares_current_day)
            adr.append(current_adr)
            entry.append(next_entry)
            stop_loss.append(stop_loss_current_day)
            limit_order.append(limit_order_current_day)

            # todo: option prices

    dataframe['ADR'] = adr
    dataframe['Next-Entry'] = entry
    dataframe['Stop-Loss'] = stop_loss
    dataframe['Limit-Order'] = limit_order
    dataframe['Max-Shares'] = shares_to_buy

def analyze_ticker(ticker):
    current_date, past_date = get_dates()

    try:
        # get data starting past month and ending current date
        # ticker_data = get_data(ticker, start_date=past_date, end_date=current_date, index_as_date = True, interval="1d")
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
    ticker_data = analyze_ticker(ticker)
    if ticker_data is not None and is_winner(ticker_data):
        return ticker_data.iloc[[len(ticker_data)-1]]['ticker'].item()

def analyze_stocks(tickers):
    global winning_stocks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_ticker_wrapper, tickers), total=len(tickers)))
        with LOCK:
            winning_stocks.extend([ticker for ticker in results if ticker is not None])


def set_winners():
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


def get_info(ticker):
    current_date, past_date = get_dates()

    try:
        ticker_data = get_data(ticker, start_date=past_date, end_date=current_date, index_as_date = True, interval="1d")
        add_order_values(ticker_data)

        # debug
        # add needed values
        # add_macd_data(ticker_data)
        # add_rsi_data(ticker_data)
        # add_stochastic_slow(ticker_data)

        # add color of days
        # add_color_of_days(ticker_data)

        print(ticker_data)
    except Exception as e:
        print(str(e))
    

def main():
    choice = input('Welcome to PowerXStocksAnalyzer!\nWhat do you want to do? 1: get a list of stocks in buy zone or {ticker symbol}: get specific info of a given symbol?: ')
    if choice == '1':
        set_winners()
        print('Stocks in buy zone:\n' + ', '.join(winning_stocks) + '\nCheck out if the analysis is correct at: https://finance.yahoo.com')
    else:
        get_info(choice)


main()













"""
deprecated
def get_winners():
    
    winning_stocks = []
    iter = 0

    print('Analyzing NASDAQ...')
    for ticker in tqdm(get_nasdaq_symbols()):

        iter += 1
        ticker_data = analyze_ticker(ticker)

        
        # if (iter > 250): break

        # check for buy condition
        if (ticker_data is not None and is_winner(ticker_data)):
            winning_stocks.append(ticker_data.iloc[[len(ticker_data)-1]]['ticker'].item())
            # add_order_values(ticker_data)
            # print(ticker_data)
            # print('Winner: ' + ticker_data.iloc[[len(ticker_data)-1]]['ticker'].item())
        
        #if (iter > 199): break
        
    print('%d stocks in NASDAQ analyzed' % iter)

    iter = 0
    print('Analyzing S&P 500...')
    for ticker in tqdm(get_sp500_symbols()):
        iter += 1

        ticker_data = analyze_ticker(ticker)

        # check for buy condition
        if (ticker_data is not None and is_winner(ticker_data)):
            winning_stocks.append(ticker_data.iloc[[len(ticker_data)-1]]['ticker'].item())
    
    print('%d stocks in S&P 500 analyzed' % iter)

    return winning_stocks  
"""