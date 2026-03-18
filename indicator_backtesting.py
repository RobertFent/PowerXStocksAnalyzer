import pandas as pd
import math

# dipping first time below entry threshold is an entry signal; first time dipping above exit is an exit signal
WILL_R_4_ENTRY_SIGNAL = -85
WILL_R_4_EXIT_SIGNAL = -50

RSI_4_ENTRY_SIGNAL = 15
RSI_4_EXIT_SIGNAL = 45

# todo: ADR 3 or 4 better?
ADR_MULT = 4

MAX_POSITION_SIZE_IN_DOLLAR = 5000

df = pd.read_csv('combined_stock_data.csv')

pd.options.mode.chained_assignment = None  # turn off value assignment warning

trading_metrics_result_objects_willr4: list[dict] = []
trading_metrics_result_objects_rsi4: list[dict] = []


def calculate_trade_profits_for_symbol(entry_rows: pd.DataFrame, indicator_key: str, exit_signal: float) -> tuple[list[float], float]:
    # todo: ROI based on number of days in trade and profit
    trade_profits = []
    avg_days_in_trade = 0
    for entry_idx in entry_rows.index:
        entry_data = entry_rows.loc[entry_idx]
        entry_adr = entry_data['ADR_14']
        entry_close = entry_data['Close']
        stop_loss_amount = ADR_MULT * entry_adr
        stop_loss_data_point = (entry_close - stop_loss_amount)

        # calc number of shares based on max. position size in dollar
        num_shares = math.floor(MAX_POSITION_SIZE_IN_DOLLAR / entry_close)

        # look forward from entry to find first exit
        after_entry = ticker_data.loc[entry_idx+1:]

        # exit is either based on indicator exit signal or stop loss is broken
        exit_mask = ((after_entry[indicator_key] > exit_signal) |
                     (after_entry['Low'] < stop_loss_data_point))
        if exit_mask.any():
            exit_idx = exit_mask.idxmax()
            try:
                # todo: handle stop loss better maybe
                # pass ROI, days in trade etc in result object
                # the day after exit signal occurs will be taken at the market open
                exit_row = ticker_data.loc[exit_idx+1]
                trade_profit = (exit_row['Open'] -
                                ticker_data.loc[entry_idx, 'Close']) * num_shares
                current_days_in_trade = exit_idx - entry_idx

                avg_days_in_trade = (
                    current_days_in_trade + avg_days_in_trade) / 2
                trade_profits.append(trade_profit)
            except:
                pass  # when calc error occurs just ignore the trade
            # print(f'Entry: {ticker_data.loc[entry_idx, 'Date']} @ {ticker_data.loc[entry_idx, 'Close']}, '
            #       f'Exit: {exit_row['Date']} @ {exit_row['Close']}; profit: {trade_profit}')

        else:
            # print(
            #     f'Entry: {ticker_data.loc[entry_idx, 'Date']} @ {ticker_data.loc[entry_idx, 'Close']}, Exit: Not found')
            pass  # skip not exited trades for now

    return trade_profits, avg_days_in_trade


total_trade_profits_willr: list[float] = []
total_trade_profits_rsi: list[float] = []
avg_days_in_trade_willr: float = 0
avg_days_in_trade_rsi: float = 0

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

    trade_profits_willr, curr_avg_days_in_trade_willr = calculate_trade_profits_for_symbol(
        entry_rows_willr, 'WILLR_4', WILL_R_4_EXIT_SIGNAL)

    trade_profits_rsi, curr_avg_days_in_trade_rsi = calculate_trade_profits_for_symbol(
        entry_rows_rsi, 'RSI_4', RSI_4_EXIT_SIGNAL)

    total_trade_profits_willr.append(sum(trade_profits_willr))
    total_trade_profits_rsi.append(sum(trade_profits_rsi))

    avg_days_in_trade_willr = (
        curr_avg_days_in_trade_willr + avg_days_in_trade_willr) / 2
    avg_days_in_trade_rsi = (
        curr_avg_days_in_trade_rsi + avg_days_in_trade_rsi) / 2

    # lastest_share_price = ticker_data['Close'].iloc[-1]
    # first_share_price = ticker_data['Close'].iloc[0]

    # print()
    # print(f'######### Symbol: {ticker} ############')
    # print(
    #     f'Total sum of willr profits: ${round(sum(trade_profits_willr), 2)}')
    # print(
    #     f'Total sum of rsi profits: ${round(sum(trade_profits_rsi), 2)}')
    # print(
    #     f'Lastest share price on {ticker_data['Date'].iloc[-1]} is ${round(lastest_share_price, 2)} which diffs ${round(lastest_share_price - first_share_price, 2)} compared to the initial price of ${round(first_share_price, 2)} on {ticker_data['Date'].iloc[0]}')

print('\n############ RESULTS ############\n')
print(
    f'RSI entry: first RSI(4) trading day below {RSI_4_ENTRY_SIGNAL}; RSI exit: the day after RSI(4) above {RSI_4_EXIT_SIGNAL} or stop loss hit {ADR_MULT}*ADR(14)')
print(
    f'WILLR entry: first WILLR(4) trading day below {WILL_R_4_ENTRY_SIGNAL}; WILLR exit: the day after WILLR(4) above {WILL_R_4_EXIT_SIGNAL} or stop loss hit {ADR_MULT}*ADR(14)')
print(
    f'If you would have taken every entry signal with around ${MAX_POSITION_SIZE_IN_DOLLAR} per position since 2017 for all S&P500 and Nasdaq stocks these would be your profits')
print('\n######### WILLR #########\n')
print(f'- total trade profits: ${round(sum(total_trade_profits_willr))}')
print(f'- average days in trade: {round(avg_days_in_trade_willr, 2)}')

print('\n######### RSI #########\n')
print(f'- total trade profits: ${round(sum(total_trade_profits_rsi))}')
print(f'- average days in trade: {round(avg_days_in_trade_rsi, 2)}')
