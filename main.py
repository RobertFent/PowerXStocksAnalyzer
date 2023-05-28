from yahoo_fin.stock_info import get_data, tickers_nasdaq, tickers_dow, tickers_sp500
import pandas as pd
from datetime import datetime, timedelta

# MACD
def add_macd_data(dataframe):
    ema_12 = dataframe["close"].ewm(span=12, adjust=False).mean()
    ema_26 = dataframe["close"].ewm(span=26, adjust=False).mean()

    macd_line= ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    # macd_histogram = macd_line - signal_line

    # macd_df = pd.DataFrame({
    #     'MACD Line': macd_line,
    #     'Signal Line': signal_line,
    #     'MACD Histogram': macd_histogram
    # })

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


# stochastic slow
def add_stochastic_slow(dataframe):

    # Calculate the lowest low over the past 14 days
    lowest_low = dataframe['low'].rolling(window=14).min()

    # Calculate the highest high over the past 14 days
    highest_high = dataframe['high'].rolling(window=14).max()

    # Calculate %K
    percent_k = ((dataframe['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    # Calculate %D (3-day simple moving average of %K)
    percent_d = percent_k.rolling(window=3).mean()

    dataframe['%K'] = percent_k
    dataframe['%D'] = percent_d

    # todo: get this right
    # print(lowest_low.values)

def get_dates():
    current_date = datetime.now()
    formated_date = current_date.strftime("%m/%d/%Y")
    past_month_date = current_date - timedelta(days=30)
    formated_past_month_date = past_month_date.strftime("%m/%d/%Y")
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

def print_if_winner(dataframe):
    # current day should be green and the day before not
    if (dataframe.iloc[[len(dataframe)-1]]['color'].item() == 'green' and dataframe.iloc[[len(dataframe)-2]]['color'].item() != 'green'):
        print(dataframe)
        print('Winner: ' + dataframe.iloc[[len(dataframe)-1]]['ticker'].item())
            


def main():
    current_date, past_date = get_dates()
    
    loops = 0
    for ticker in tickers_nasdaq():
        loops += 1
        print(ticker)
        # get data starting past month and ending current date
        ticker_data = get_data(ticker, start_date=past_date, end_date=current_date, index_as_date = True, interval="1d")

        # add needed values
        add_macd_data(ticker_data)
        add_rsi_data(ticker_data)
        add_stochastic_slow(ticker_data)

        # add color of days
        add_color_of_days(ticker_data)

        # check for buy condition
        print_if_winner(ticker_data)
        if loops == 20: break

main()