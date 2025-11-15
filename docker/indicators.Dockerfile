FROM python:3.13.7

WORKDIR /indicators

COPY indicators.py requirements.txt ./
COPY symbols ./symbols

RUN pip install -r requirements.txt

CMD ["python3", "./indicators.py"]