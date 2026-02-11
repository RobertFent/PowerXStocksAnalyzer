import pandas as pd
import math

# dipping first time below entry threshold is an entry signal; first time dipping above exit is an exit signal
WILL_R_4_ENTRY_SIGNAL = -85
WILL_R_4_EXIT_SIGNAL = -60

RSI_4_ENTRY_SIGNAL = 15
RSI_4_EXIT_SIGNAL = 45

MAX_POSITION_SIZE_IN_DOLLAR = 5000

df = pd.read_csv('combined_stock_data.csv')

pd.options.mode.chained_assignment = None  # turn off value assignment warning


# If I buy at close today how long does it take until price rises by 2%?
# Start with entry signal and calc days; later on maybe at ADR stop loss checks

total_sum_of_profits_willr4: list[float] = []
total_sum_of_profits_rsi4: list[float] = []


def calculate_trade_profits_for_symbol(entry_rows: pd.DataFrame, indicator_key: str, exit_signal: float) -> list[float]:
    trade_profits = []
    for entry_idx in entry_rows.index:
        entry_close = entry_rows.loc[entry_idx]['Close']
        num_shares = math.floor(MAX_POSITION_SIZE_IN_DOLLAR / entry_close)
        # look forward from entry to find first exit
        after_entry = ticker_data.loc[entry_idx+1:]
        exit_mask = after_entry[indicator_key] > exit_signal
        if exit_mask.any():
            exit_idx = exit_mask.idxmax()
            try:
                # the day after exit signal occurs will be taken at the market open
                exit_row = ticker_data.loc[exit_idx+1]
                trade_profit = (exit_row['Open'] -
                                ticker_data.loc[entry_idx, 'Close']) * num_shares
            except:
                pass  # when calc error occurs just ignore the trade
            # print(f'Entry: {ticker_data.loc[entry_idx, 'Date']} @ {ticker_data.loc[entry_idx, 'Close']}, '
            #       f'Exit: {exit_row['Date']} @ {exit_row['Close']}; profit: {trade_profit}')

            trade_profits.append(trade_profit)
        else:
            # print(
            #     f'Entry: {ticker_data.loc[entry_idx, 'Date']} @ {ticker_data.loc[entry_idx, 'Close']}, Exit: Not found')
            pass  # skip not exited trades for now
    return trade_profits


total_trade_profits_willr = []
total_trade_profits_rsi = []

for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    ticker_data.loc[:, 'prev_WILLR_4'] = ticker_data['WILLR_4'].shift(1)
    ticker_data.loc[:, 'prev_RSI_4'] = ticker_data['RSI_4'].shift(1)

    # entry: first dip below threshold
    entry_mask_willr = (ticker_data['prev_WILLR_4'] > WILL_R_4_ENTRY_SIGNAL) & \
        (ticker_data['WILLR_4'] <= WILL_R_4_ENTRY_SIGNAL)
    entry_rows_willr = ticker_data[entry_mask_willr]

    entry_mask_rsi = (ticker_data['prev_RSI_4'] < RSI_4_ENTRY_SIGNAL) & (
        ticker_data['RSI_4'] >= RSI_4_ENTRY_SIGNAL)
    entry_rows_rsi = ticker_data[entry_mask_rsi]

    trade_profits_willr = calculate_trade_profits_for_symbol(
        entry_rows_willr, 'WILLR_4', WILL_R_4_EXIT_SIGNAL)

    trade_profits_rsi = calculate_trade_profits_for_symbol(
        entry_rows_rsi, 'RSI_4', RSI_4_EXIT_SIGNAL)

    total_trade_profits_willr.append(sum(trade_profits_willr))
    total_trade_profits_rsi.append(sum(trade_profits_rsi))

    lastest_share_price = ticker_data['Close'].iloc[-1]
    first_share_price = ticker_data['Close'].iloc[0]

    # print()
    # print(f'######### Symbol: {ticker} ############')
    # print(
    #     f'Total sum of willr profits: ${round(sum(trade_profits_willr), 2)}')
    # print(
    #     f'Total sum of rsi profits: ${round(sum(trade_profits_rsi), 2)}')
    # print(
    #     f'Lastest share price on {ticker_data['Date'].iloc[-1]} is ${round(lastest_share_price, 2)} which diffs ${round(lastest_share_price - first_share_price, 2)} compared to the initial price of ${round(first_share_price, 2)} on {ticker_data['Date'].iloc[0]}')

print('\n############ RESULTS ############\n')
print('RSI entry: first RSI(4) trading day below 15; RSI exit: the day after RSI(4) above 45')
print('WILLR entry: first WILLR(4) trading day below -80; WILLR exit: the day after WILLR(4) above -60')
print('If you would have taken every entry signal with around $5000 per position since 2017 for all S&P500 and Nasdaq stocks these would be your profits')
print('Caveat: There is no stop-loss considered so far. The only exit is the crossover of the exit threshold')
print(f'WILLR4: ${round(sum(total_trade_profits_willr))}')
print(f'RSI4: ${round(sum(total_trade_profits_rsi))}')
