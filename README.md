# PowerXStocksAnalyzer

Daily scraper for the stock screener: downloads the last ~300 trading days for sp100 / nasdaq100 / sp500,
computes the indicators (EMA20/50, MA200, MACD, RSI 4/14, IV, Williams %R 4/14, slow stochastic, ADR 7/14),
upserts the last 7 days into Postgres (`stock_data`) and revalidates the screener cache.

Strategy research (backtesting, tuning, signals) moved to the separate **StrategyAnalyser** repo.

```
indicators.py          daily scraper
utils.py               refresh index symbol lists in symbols/ from Wikipedia
symbols/               index constituents
database/schema.sql    Postgres table used by the scraper
docker/                container for the scraper (home server)
```

## Setup
```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env      # DATABASE_URL, REVALIDATE_SECRET, STOCK_SCREENER_REVALIDATE_URL
```

## Run
```sh
python3 indicators.py                                          # local
docker compose -f docker/indicators.docker-compose.yml up      # docker
```

Cron on the home server:
```
0 7 * * * cd /home/robot/repos/PowerXStocksAnalyzer && /usr/bin/docker compose -f docker/indicators.docker-compose.yml up
```

Refresh `symbols/*.csv` when index members change (and copy them to StrategyAnalyser):
```sh
python3 utils.py
```
