"""Indicator calculation with caching, so tuning can vary indicator lengths cheaply.

All oscillators are returned on a 0..100 scale where LOW = OVERSOLD:
    rsi    RSI(n), Wilder smoothing
    willr  Williams %R(n) + 100        (WILLR -85  ==  15 here)
    stoch  slow stochastic %K(n, 3)    (same definition as indicators.py)
    ibs    internal bar strength (Close-Low)/(High-Low)*100, no length

Every value at row t only uses data up to and including the close of t
(no look-ahead). Rows are sorted by Ticker, Date and each ticker's rows are
contiguous - the engine relies on that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OSCILLATORS = ('rsi', 'willr', 'stoch', 'ibs')


class FeatureStore:
    def __init__(self, prices: pd.DataFrame, benchmark: pd.DataFrame | None = None):
        df = prices.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        self.df = df
        self.tickers = df['Ticker'].to_numpy()
        self.dates = df['Date'].to_numpy(dtype='datetime64[ns]')
        self.open = df['Open'].to_numpy(dtype=np.float64)
        self.high = df['High'].to_numpy(dtype=np.float64)
        self.low = df['Low'].to_numpy(dtype=np.float64)
        self.close = df['Close'].to_numpy(dtype=np.float64)
        self.volume = df['Volume'].to_numpy(dtype=np.float64)
        self.first_row = np.r_[True, self.tickers[1:] != self.tickers[:-1]]
        self._key = df['Ticker']
        self.benchmark = None
        if benchmark is not None and not benchmark.empty:
            self.benchmark = benchmark.sort_values('Date').set_index('Date')['Close']
        self._cache: dict[tuple, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.close)

    # ---------- helpers ----------
    def _grouped_rolling(self, values: np.ndarray, n: int, how: str) -> np.ndarray:
        s = pd.Series(values)
        r = s.groupby(self._key.values, sort=False).rolling(n, min_periods=n)
        r = getattr(r, how)()
        return r.reset_index(level=0, drop=True).sort_index().to_numpy()

    def _grouped_ewm(self, values: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
        s = pd.Series(values)
        return s.groupby(self._key.values, sort=False).transform(
            lambda x: x.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
        ).to_numpy()

    def shift(self, values: np.ndarray, k: int = 1) -> np.ndarray:
        """Shift by k rows within each ticker (previous value), NaN at ticker start."""
        out = np.full_like(values, np.nan, dtype=np.float64)
        out[k:] = values[:-k]
        # invalidate rows whose k-th predecessor belongs to another ticker
        ticker_id = np.cumsum(self.first_row)
        same = np.zeros(len(values), dtype=bool)
        same[k:] = ticker_id[k:] == ticker_id[:-k]
        out[~same] = np.nan
        return out

    # ---------- public ----------
    def get(self, name: str, n: int = 0) -> np.ndarray:
        key = (name, int(n))
        if key not in self._cache:
            self._cache[key] = getattr(self, f'_calc_{name}')(int(n))
        return self._cache[key]

    def oscillator(self, name: str, n: int) -> np.ndarray:
        if name not in OSCILLATORS:
            raise ValueError(f'unknown oscillator {name!r}, use one of {OSCILLATORS}')
        return self.get(name, n)

    def breadth(self, name: str, n: int, threshold: float) -> np.ndarray:
        """Per row: share of tickers (with data) whose oscillator is below `threshold` that day."""
        key = ('breadth', name, int(n), float(threshold))
        if key not in self._cache:
            osc = self.oscillator(name, n)
            valid = ~np.isnan(osc)
            df = pd.DataFrame({'d': self.dates[valid], 'below': osc[valid] < threshold})
            share = df.groupby('d')['below'].mean()
            mapped = share.reindex(pd.DatetimeIndex(self.dates)).to_numpy()
            self._cache[key] = np.nan_to_num(mapped.astype(np.float64), nan=0.0)
        return self._cache[key]

    def _calc_sma(self, n: int) -> np.ndarray:
        return self._grouped_rolling(self.close, n, 'mean')

    def _calc_rsi(self, n: int) -> np.ndarray:
        delta = self.close - self.shift(self.close)
        gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0, None))
        loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0, None))
        avg_gain = self._grouped_ewm(gain, 1.0 / n, n)
        avg_loss = self._grouped_ewm(loss, 1.0 / n, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            rsi = 100.0 * avg_gain / (avg_gain + avg_loss)
        rsi[(avg_gain + avg_loss) == 0] = 50.0
        return rsi

    def _hh_ll(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return (self._grouped_rolling(self.high, n, 'max'),
                self._grouped_rolling(self.low, n, 'min'))

    def _calc_willr(self, n: int) -> np.ndarray:
        hh, ll = self._hh_ll(n)
        rng = hh - ll
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (self.close - ll) / rng * 100.0
        v[rng == 0] = 50.0
        return v

    def _calc_stoch(self, n: int) -> np.ndarray:
        fast_k = self._calc_willr(n)  # identical formula to fast %K
        return self._grouped_rolling(fast_k, 3, 'mean')

    def _calc_ibs(self, _n: int = 0) -> np.ndarray:
        rng = self.high - self.low
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (self.close - self.low) / rng * 100.0
        v[rng == 0] = 50.0
        return v

    def _calc_adr(self, n: int) -> np.ndarray:
        return self._grouped_rolling(self.high - self.low, n, 'mean')

    def _calc_dollar_volume(self, n: int) -> np.ndarray:
        return self._grouped_rolling(self.close * self.volume, n, 'mean')

    def _calc_market_ok(self, n: int) -> np.ndarray:
        """True where the benchmark closed above its SMA(n) on that date."""
        if self.benchmark is None:
            raise ValueError('market filter requested but no benchmark data loaded')
        ok = self.benchmark > self.benchmark.rolling(n, min_periods=n).mean()
        mapped = ok.reindex(pd.DatetimeIndex(self.dates)).to_numpy()
        return np.where(pd.isna(mapped), False, mapped).astype(bool)
