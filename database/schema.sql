CREATE TABLE IF NOT EXISTS stock_winners (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    close DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    open DOUBLE PRECISION,
    volume BIGINT,
    ema20 DOUBLE PRECISION,
    ema50 DOUBLE PRECISION,
    macd_line DOUBLE PRECISION,
    signal_line DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    iv DOUBLE PRECISION,
    willr DOUBLE PRECISION,
    stoch_percent_k DOUBLE PRECISION,
    stoch_percent_d DOUBLE PRECISION,
    last_updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (ticker, date)
);
