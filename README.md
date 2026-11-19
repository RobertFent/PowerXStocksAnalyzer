# PowerXStocksAnalyzer

Daily scraper for the stock screener. `indicators.py` is the only script that has to run, once per
trading day after the US close:

1. **Prices + indicators** for sp100 / nasdaq100 / sp500: downloads the last ~300 days, computes
   EMA20/50, MA200, MACD, RSI 4/14, IV, Williams %R 4/14, slow stochastic, ADR 7/14 and upserts the
   last 7 days into `stock_data`.
2. **Option positioning** for sp100 + SPY/QQQ: reads the option chains (expirations within 45 days)
   and upserts one row per symbol and day into `gamma_levels`: call wall, put wall, gamma flip, net
   gamma / vanna / charm exposure, 30 day at-the-money IV, 25 delta skew, realized volatility, max
   pain, put/call open interest and volume, next earnings and ex-dividend date.
3. Revalidates the screener cache.

Greeks are computed with Black-Scholes from Yahoo's open interest and IV. Dealer positioning is the
naive "long calls, short puts" assumption, so treat single-stock levels as rough. Yahoo empties the
open interest overnight US time: a run before the US session gets no usable option chains (the
script then skips the symbols and logs it).

Strategy research (backtesting, tuning, signals) lives in [`analyser/`](analyser/README.md), the former
**StrategyAnalyser** repo. The folder is self-contained: it has its own requirements, symbols and start
scripts and does not use the scraper or the database, so it can be handed on as a folder.

```
indicators.py          the scraper
utils.py               refresh index symbol lists in symbols/ from Wikipedia
symbols/               index constituents
database/schema.sql    Postgres tables used by the scraper
docker/                container for the scraper (home server)
analyser/              strategy backtesting workbench (own README, start.command / start.bat)
```

## Setup
```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env      # DATABASE_URL, REVALIDATE_SECRET, STOCK_SCREENER_REVALIDATE_URL
```

## Run
```sh
python3 indicators.py                                          # everything, local
python3 indicators.py --only prices                            # or: --only options
python3 indicators.py --dry-run SPY AAPL                       # print option levels, no database
docker compose -f docker/indicators.docker-compose.yml up      # docker
```

Cron on the home server (22:30 Berlin time, after the US close):
```
30 22 * * 1-5 cd /home/robot/repos/PowerXStocksAnalyzer && /usr/bin/docker compose -f docker/indicators.docker-compose.yml up
```

Refresh `symbols/*.csv` when index members change (and copy them to `analyser/symbols/`):
```sh
python3 utils.py
```
