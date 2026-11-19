CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    index VARCHAR(10) NOT NULL,
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
    rsi_14 DOUBLE PRECISION,
    rsi_4 DOUBLE PRECISION,
    iv DOUBLE PRECISION,
    willr_4 DOUBLE PRECISION,
    willr_14 DOUBLE PRECISION,
    stoch_percent_k DOUBLE PRECISION,
    stoch_percent_d DOUBLE PRECISION,
    adr_7 NUMERIC(6,2),
    adr_14 NUMERIC(6,2),
    ma_200 NUMERIC(6,2),
    last_updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (ticker, date)
);

CREATE INDEX CONCURRENTLY idx_stock_data_ticker_date ON stock_data (ticker, date DESC);

-- option positioning, written by indicators.py (which also creates the table and missing columns)
-- net_gex / call_gex / put_gex: dealer gamma exposure in USD per 1% move of the underlying
CREATE TABLE IF NOT EXISTS gamma_levels (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    spot DOUBLE PRECISION,
    call_wall DOUBLE PRECISION,
    put_wall DOUBLE PRECISION,
    gamma_flip DOUBLE PRECISION,
    net_gex DOUBLE PRECISION,
    call_gex DOUBLE PRECISION,
    put_gex DOUBLE PRECISION,
    call_oi BIGINT,
    put_oi BIGINT,
    expirations SMALLINT,
    -- USD of delta dealers have to trade per 1 vol point IV rise / per calendar day
    net_vanna DOUBLE PRECISION,
    net_charm DOUBLE PRECISION,
    -- at-the-money implied volatility in percent, interpolated to 30 days
    atm_iv_30d DOUBLE PRECISION,
    -- max pain strike of the expiration with the most open interest
    max_pain DOUBLE PRECISION,
    max_pain_expiration DATE,
    -- IV of the 25 delta put minus IV of the 25 delta call in volatility points (~30 days)
    skew_25_delta DOUBLE PRECISION,
    -- 30 day realized volatility in percent, to compare against atm_iv_30d
    realized_volatility_30d DOUBLE PRECISION,
    -- option volume of the last session over all scanned expirations
    call_volume BIGINT,
    put_volume BIGINT,
    next_earnings_date DATE,
    ex_dividend_date DATE,
    last_updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (ticker, date)
);
