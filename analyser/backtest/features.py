"""Indicator calculation with caching, so tuning can vary indicator lengths cheaply.

All oscillators are returned on a 0..100 scale where LOW = OVERSOLD:
    rsi    RSI(n), Wilder smoothing
    willr  Williams %R(n) + 100        (WILLR -85  ==  15 here)
    stoch  slow stochastic %K(n, 3)    (%K smoothed by 3)
    ibs    internal bar strength (Close-Low)/(High-Low)*100, no length
    mfi    Money Flow Index(n) - volume-weighted RSI
    bbp    Bollinger %B(n, 2 std) * 100 (can go below 0 / above 100)
    crsi   Connors RSI: mean of RSI(3), RSI(2) of the up/down streak and the
           percent rank of today's return over the last n days (n=100 classic)
    uo     Ultimate Oscillator(7, 14, 28), no length
    macd   percent rank of the MACD(12,26,9) histogram over the last n days
           (low = momentum weakest vs. its own recent history)
    cci    percent rank of CCI(20) over the last n days

Trend / momentum features (not oscillators):
    hh(n), ll(n)      highest high / lowest low of the PREVIOUS n days (today excluded)
    momentum(n, skip) close[t-skip] / close[t-skip-n] - 1
    mom_rank(n, skip) cross-sectional rank of momentum that day: 1 = strongest stock
    rebalance_mask    True on the first trading day of each month / week
    eligible          True while the ticker is a member of the index (point-in-time), if known

Every value at row t only uses data up to and including the close of t
(no look-ahead). Rows are sorted by Ticker, Date and each ticker's rows are
contiguous - the engine relies on that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _rolling_mad(values, first_row, n):
    """Rolling mean absolute deviation over n rows, restarting at each ticker."""
    out = np.full(values.shape[0], np.nan)
    start = 0
    for i in range(values.shape[0]):
        if first_row[i]:
            start = i
        if i - start + 1 >= n:
            w = values[i - n + 1:i + 1]
            m = w.mean()
            out[i] = np.abs(w - m).mean()
    return out


@njit(cache=True)
def _streak(close, first_row):
    """Consecutive up (+) / down (-) closes, reset at each ticker start."""
    out = np.zeros(close.shape[0])
    for i in range(close.shape[0]):
        if first_row[i]:
            out[i] = 0.0
        elif close[i] > close[i - 1]:
            out[i] = out[i - 1] + 1.0 if out[i - 1] > 0 else 1.0
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - 1.0 if out[i - 1] < 0 else -1.0
        else:
            out[i] = 0.0
    return out

OSCILLATORS = ('rsi', 'willr', 'stoch', 'ibs', 'mfi', 'bbp', 'crsi', 'uo', 'macd', 'cci')


class FeatureStore:
    def __init__(self, prices: pd.DataFrame, benchmark: pd.DataFrame | None = None,
                 membership: pd.DataFrame | None = None):
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
        self.eligible: np.ndarray | None = None
        if membership is not None and len(membership):
            self.eligible = self._membership_mask(membership)

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
            if self.eligible is not None:
                valid &= self.eligible
            df = pd.DataFrame({'d': self.dates[valid], 'below': osc[valid] < threshold})
            share = df.groupby('d')['below'].mean()
            mapped = share.reindex(pd.DatetimeIndex(self.dates)).to_numpy()
            self._cache[key] = np.nan_to_num(mapped.astype(np.float64), nan=0.0)
        return self._cache[key]

    def _calc_sma(self, n: int) -> np.ndarray:
        return self._grouped_rolling(self.close, n, 'mean')

    def _calc_rsi(self, n: int) -> np.ndarray:
        return self._rsi_of(self.close, n)

    def _rsi_of(self, values: np.ndarray, n: int) -> np.ndarray:
        delta = values - self.shift(values)
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

    def _grouped_pct_rank(self, values: np.ndarray, n: int) -> np.ndarray:
        """Percent rank (0..100) of today's value within the last n values of the same ticker."""
        s = pd.Series(values)
        r = s.groupby(self._key.values, sort=False).rolling(n, min_periods=n).rank(pct=True)
        return r.reset_index(level=0, drop=True).sort_index().to_numpy() * 100.0

    def _calc_mfi(self, n: int) -> np.ndarray:
        tp = (self.high + self.low + self.close) / 3.0
        tp_prev = self.shift(tp)
        flow = tp * self.volume
        pos = np.where(tp > tp_prev, flow, 0.0)
        neg = np.where(tp < tp_prev, flow, 0.0)
        pos[np.isnan(tp_prev)] = np.nan
        neg[np.isnan(tp_prev)] = np.nan
        sp = self._grouped_rolling(pos, n, 'sum')
        sn = self._grouped_rolling(neg, n, 'sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            v = 100.0 * sp / (sp + sn)
        v[(sp + sn) == 0] = 50.0
        return v

    def _calc_bbp(self, n: int) -> np.ndarray:
        mean = self._grouped_rolling(self.close, n, 'mean')
        std = self._grouped_rolling(self.close, n, 'std')
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (self.close - (mean - 2 * std)) / (4 * std) * 100.0
        v[std == 0] = 50.0
        return v

    def _calc_crsi(self, n: int) -> np.ndarray:
        n = n or 100
        streak = _streak(self.close, self.first_row)
        ret = self.close / self.shift(self.close) - 1.0
        return (self._rsi_of(self.close, 3) + self._rsi_of(streak, 2)
                + self._grouped_pct_rank(ret, n)) / 3.0

    def _calc_uo(self, _n: int = 0) -> np.ndarray:
        prev_close = self.shift(self.close)
        low = np.fmin(self.low, prev_close)
        high = np.fmax(self.high, prev_close)
        bp = self.close - low
        tr = high - low
        avgs = []
        for k in (7, 14, 28):
            sb = self._grouped_rolling(bp, k, 'sum')
            st = self._grouped_rolling(tr, k, 'sum')
            with np.errstate(divide='ignore', invalid='ignore'):
                avgs.append(np.where(st > 0, sb / st, 0.5))
        return 100.0 * (4 * avgs[0] + 2 * avgs[1] + avgs[2]) / 7.0

    def _grouped_ema(self, values: np.ndarray, span: int) -> np.ndarray:
        s = pd.Series(values)
        return s.groupby(self._key.values, sort=False).transform(
            lambda x: x.ewm(span=span, adjust=False).mean()).to_numpy()

    def _calc_macd(self, n: int) -> np.ndarray:
        n = n or 100
        line = self._grouped_ema(self.close, 12) - self._grouped_ema(self.close, 26)
        hist = line - self._grouped_ema(line, 9)
        return self._grouped_pct_rank(hist, n)

    def _calc_cci(self, n: int) -> np.ndarray:
        n = n or 100
        tp = (self.high + self.low + self.close) / 3.0
        mean = self._grouped_rolling(tp, 20, 'mean')
        mad = _rolling_mad(tp, self.first_row, 20)
        with np.errstate(divide='ignore', invalid='ignore'):
            cci = (tp - mean) / (0.015 * mad)
        return self._grouped_pct_rank(cci, n)

    # ---------- trend / momentum ----------
    def _calc_hh(self, n: int) -> np.ndarray:
        return self.shift(self._grouped_rolling(self.high, n, 'max'))

    def _calc_ll(self, n: int) -> np.ndarray:
        return self.shift(self._grouped_rolling(self.low, n, 'min'))

    def momentum(self, n: int, skip: int = 0) -> np.ndarray:
        key = ('momentum', int(n), int(skip))
        if key not in self._cache:
            end = self.shift(self.close, skip) if skip else self.close
            start = self.shift(self.close, n + skip)
            with np.errstate(divide='ignore', invalid='ignore'):
                self._cache[key] = end / start - 1.0
        return self._cache[key]

    def mom_rank(self, n: int, skip: int = 0) -> np.ndarray:
        """1 = strongest momentum among all tickers with data that day; NaN where momentum is NaN."""
        key = ('mom_rank', int(n), int(skip))
        if key not in self._cache:
            m = self.momentum(n, skip)
            if self.eligible is not None:
                m = np.where(self.eligible, m, np.nan)  # rank only among index members that day
            s = pd.Series(m)
            r = s.groupby(self.dates, sort=False).rank(ascending=False, method='first')
            self._cache[key] = r.to_numpy(dtype=np.float64)
        return self._cache[key]

    def bars_since_start(self) -> np.ndarray:
        """0 on a ticker's first row, 1 on the next, ... (how much history the stock has)."""
        key = ('bars_since_start',)
        if key not in self._cache:
            idx = np.arange(len(self.dates))
            starts = np.maximum.accumulate(np.where(self.first_row, idx, 0))
            self._cache[key] = idx - starts
        return self._cache[key]

    def rebalance_mask(self, freq: str) -> np.ndarray:
        """True on the first trading day of each month ('monthly') / week ('weekly'); all True for 'daily'."""
        key = ('rebalance', freq)
        if key not in self._cache:
            if freq == 'daily':
                self._cache[key] = np.ones(len(self.dates), dtype=bool)
            else:
                days = pd.DatetimeIndex(np.unique(self.dates))
                period = days.to_period('M' if freq == 'monthly' else 'W')
                first = pd.Series(days).groupby(period).first()
                first_days = set(np.asarray(first.values, dtype='datetime64[ns]'))
                self._cache[key] = np.isin(self.dates, list(first_days))
        return self._cache[key]

    def _membership_mask(self, membership: pd.DataFrame) -> np.ndarray:
        """membership: columns Ticker, Start, End (End NaT = still a member)."""
        mask = np.zeros(len(self.dates), dtype=bool)
        starts = np.flatnonzero(self.first_row)
        ends = np.r_[starts[1:], len(self.dates)]
        rows = {t: (a, b) for t, a, b in zip(self.tickers[starts], starts, ends)}
        known = set(membership['Ticker'])
        for t, (a, b) in rows.items():  # tickers without any membership record (other index): always eligible
            if t not in known:
                mask[a:b] = True
        for row in membership.itertuples(index=False):
            span = rows.get(row.Ticker)
            if span is None:
                continue
            a, b = span
            d = self.dates[a:b]
            lo = np.datetime64(pd.Timestamp(row.Start))
            ok = d >= lo
            if pd.notna(row.End):
                ok &= d <= np.datetime64(pd.Timestamp(row.End))
            mask[a:b] |= ok
        return mask

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
