# PowerXStocksAnalyzer
Automation Tool for PowerXStrategy

## setup venv
```sh
python3 -m pip install virtualenv
python3 -m virtualenv venv
source .venv/bin/activate
```

## Synthetic Long Indicators
### install requirements
```sh
pip install -r requirements.txt
```

### start the tool locally
```sh
python3 indicators.py
```

### or via docker
```sh
docker compose -f docker/indicators.docker-compose.yml up
```

### run as cronjob
```sh
crontab -e
```
add the following line
```
0 7 * * * cd /home/robot/repos/PowerXStocksAnalyzer && /usr/bin/docker compose -f docker/indicators.docker-compose.yml up
```
## PowerX Strategy (outdate)
### setup variable
create .env
```sh
cp example.env .env
```

fill out .env with proper values



### install requirements
```sh
pip install -r requirements.txt
```

### start the tool
```sh
python3 powerx.py
```