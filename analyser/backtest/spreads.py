"""SPY bull call spreads: backtest and tune a debit-spread routine without options data.

    python3 -m backtest.spreads --fetch            # SPY (unadjusted) + VIX -> data/options_underlying.csv.gz
    python3 -m backtest.spreads                    # backtest the default setup (below), prints a report
    python3 -m backtest.spreads --params strategies/bcs_495.json     # HTML report for a saved option plan
    python3 -m backtest.spreads --width 10 --dte 60 --fixed-debit 4.95 --save "My spread plan"   # save as a plan
    python3 -m backtest.spreads --tune             # grid over strikes, widths, expiries, targets, stops, cadence, filter

How it works
  - No option quotes exist for 14 years, so every spread is priced with Black-Scholes using the VIX of the
    day as implied volatility, the 13-week T-bill as the rate and a 1.4% dividend yield (times --iv-mult, default 1.0; OTM calls usually trade a bit below VIX).
    The entry debit, the daily mark and the early close all come from that model; the payoff at expiry
    is exact (intrinsic value from the SPY close). SPY is unadjusted so dividends hit the calls like real life.
  - A spread: buy a call, sell the call `width` dollars higher. strike_mode 'debit' (default) picks, each entry
    day, the whole-dollar strikes whose model price is closest to the debit you pay (fixed_debit), i.e. "I always pay
    4.95 for a $10 spread" - the strikes land wherever the market prices that; 'offset' puts the long strike at
    SPY * (1 + offset) instead. Spreads expire
    on the Friday closest to `dte` calendar days ahead. Close when the spread is worth at least
    target * width (e.g. 9.90 on a 10-wide), at a stop (value <= debit * (1 - stop)), or at expiry.
  - Cadence: a new spread on the first trading day of each month / week, or the day after the last one closed;
    only when SPY is above its --trend-ma day average (0 = always).
  - Sizing: --risk of the account is at risk in total (debit * contracts), split over the spreads that overlap
    (a 60-day spread opened monthly = 2 open at once = half of --risk each). Costs: --slip dollars per spread
    per side (bid-ask) and --commission dollars per contract per side.
  - Returns are reported on the whole account (cash sits idle) and per trade on the debit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data import DATA_DIR, OUTPUT_DIR, ROOT_DIR
from backtest.metrics import compute_metrics, yearly_returns, fmt_pct

UNDERLYING = DATA_DIR / 'options_underlying.csv.gz'


# ----------------------------------------------------------------------------- data
def fetch(start: str = '2010-01-01') -> pd.DataFrame:
    import yfinance as yf
    spy = yf.download('SPY', start=start, auto_adjust=False, progress=False)
    vix = yf.download('^VIX', start=start, auto_adjust=False, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        vix.columns = vix.columns.get_level_values(0)
    df = pd.DataFrame({'Date': spy.index, 'Close': spy['Close'].to_numpy(), 'AdjClose': spy['Adj Close'].to_numpy()})
    df['VIX'] = vix['Close'].reindex(spy.index).ffill().to_numpy()
    irx = yf.download('^IRX', start=start, auto_adjust=False, progress=False)
    if isinstance(irx.columns, pd.MultiIndex):
        irx.columns = irx.columns.get_level_values(0)
    df['Rate'] = (irx['Close'].reindex(spy.index).ffill().bfill() / 100.0).to_numpy()   # 13-week T-bill, annual
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(UNDERLYING, index=False, compression='gzip')
    return df


def load() -> pd.DataFrame:
    if not UNDERLYING.exists():
        raise FileNotFoundError(f'{UNDERLYING} missing. Run `python3 -m backtest.spreads --fetch` first.')
    df = pd.read_csv(UNDERLYING, parse_dates=['Date']).dropna()
    if 'Rate' not in df:
        print('NOTE: no T-bill rates in the data file (older download); using 0%. Run `python3 -m backtest.spreads --fetch` again.')
    return df.sort_values('Date').reset_index(drop=True)


# ----------------------------------------------------------------------------- pricing
def _ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def call_price(s: np.ndarray, k: float, t_years: np.ndarray, iv: np.ndarray, r=0.0, q: float = 0.0) -> np.ndarray:
    """Black-Scholes call with continuous dividend yield q; at t=0 returns intrinsic value. r may be an array."""
    s, t, iv = np.asarray(s, float), np.maximum(np.asarray(t_years, float), 0.0), np.maximum(np.asarray(iv, float), 1e-4)
    r = np.broadcast_to(np.asarray(r, float), s.shape)
    out = np.maximum(s - k, 0.0)
    live = t > 0
    if live.any():
        st, tt, vt, rt = s[live], t[live], iv[live], r[live]
        d1 = (np.log(st / k) + (rt - q + 0.5 * vt ** 2) * tt) / (vt * np.sqrt(tt))
        d2 = d1 - vt * np.sqrt(tt)
        out[live] = st * np.exp(-q * tt) * _ncdf(d1) - k * np.exp(-rt * tt) * _ncdf(d2)
    return out


def spread_value(s, k1, k2, t_years, iv, r=0.0, q=0.0):
    return call_price(s, k1, t_years, iv, r, q) - call_price(s, k2, t_years, iv, r, q)


# ----------------------------------------------------------------------------- strategy
@dataclass
class SpreadParams:
    offset: float = 0.017      # long strike = SPY * (1 + offset); 0.017 ~ 767 -> 780
    width: float = 10.0        # dollars between the strikes
    dte: int = 60              # calendar days to expiry (nearest Friday)
    target: float = 0.99       # close when the spread is worth target * width (9.90 on a 10-wide)
    target_div: float = 0.95   # same, for spreads expiring in a dividend month (Mar/Jun/Sep/Dec: SPY goes ex-dividend on expiration day)
    stop: float = 0.0          # close when the spread lost this share of the debit (0 = never)
    cadence: str = 'expiry'    # expiry (third Friday, when the old one expires) | monthly (1st day) | weekly | next
    trend_ma: int = 200        # open only when SPY > its N-day average (0 = always)
    risk: float = 0.10         # share of the account at risk across all open spreads (split by how many overlap)
    iv_mult: float = 1.0       # implied vol = VIX/100 * iv_mult
    slip: float = 0.05         # dollars per spread per side
    commission: float = 0.65   # dollars per contract per leg per side
    rate: float = -1.0         # risk-free rate; -1 = the 13-week T-bill of the day (needs --fetch), else a fixed number
    div_yield: float = 0.014   # SPY dividend yield: pulls the forward (and the strikes that cost a given debit) down
    fixed_debit: float = 4.95  # the debit you pay per spread (0 = pay the model price)
    strike_mode: str = 'debit' # debit: pick the strikes that cost fixed_debit that day | offset: strikes at `offset` from the price

    @classmethod
    def load(cls, path) -> 'SpreadParams':
        data = json.loads(Path(path).read_text())
        d = data.get('params', data)
        return cls(**{k: v for k, v in d.items() if k in asdict(cls())})

    def to_dict(self) -> dict:
        return asdict(self)

    def concurrent(self) -> int:
        """How many spreads are normally open at once (expiry / cadence)."""
        step = {'expiry': 30, 'monthly': 30, 'weekly': 7}.get(self.cadence)
        return max(1, math.ceil(self.dte / step)) if step else 1

    def describe(self) -> str:
        where = ('out of the money' if self.offset > 0 else 'in the money' if self.offset < 0 else 'at the money')
        cad = {'expiry': 'the third Friday of each month (monthly expiration day)', 'monthly': 'first trading day of each month',
               'weekly': 'first trading day of each week', 'next': 'the day after the previous one closed'}[self.cadence]
        exp = (f'expiring on the monthly expiration about {self.dte} days out' if self.cadence == 'expiry' else f'expiring in about {self.dte} days')
        if self.strike_mode == 'debit' and self.fixed_debit:
            strikes = (f'  buy the SPY call and sell the call ${self.width:g} higher, {exp}, at the strikes that cost '
                       f'${self.fixed_debit:.2f} that day (the model picks them from the VIX; usually around the money)')
        else:
            strikes = (f'  buy the SPY call {self.offset * 100:+.1f}% from the price ({where}), sell the call ${self.width:g} higher, {exp}'
                       + (f'; you pay ${self.fixed_debit:.2f} for it' if self.fixed_debit else ''))
        lines = [f'OPEN on {cad}' + (f', only while SPY is above its {self.trend_ma}-day average' if self.trend_ma else '') + ':',
                 strikes,
                 f'CLOSE when the spread is worth {self.target * 100:.0f}% of its width (${self.target * self.width:.2f} on a ${self.width:g}-wide)'
                 + (f', {self.target_div * 100:.0f}% (${self.target_div * self.width:.2f}) for spreads expiring in Mar/Jun/Sep/Dec because SPY goes ex-dividend on that expiration day'
                    if self.target_div and self.target_div != self.target else '')
                 + (f', when it has lost {self.stop * 100:.0f}% of the debit' if self.stop else '') + ', or at expiry',
                 f'SIZE: {self.risk * 100:.0f}% of the account at risk in total, about {self.concurrent()} spread(s) open at once '
                 f'({self.risk / self.concurrent() * 100:.1f}% each); ${self.slip:.2f} slippage per side, '
                 f'${self.commission:.2f} commission per contract and leg']
        return '\n'.join(lines)


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def _expiry_third_friday(d: pd.Timestamp, dte: int) -> pd.Timestamp:
    months = max(1, round(dte / 30))
    m = d.month - 1 + months
    return third_friday(d.year + m // 12, m % 12 + 1)


def _strike_for_debit(s: float, width: float, t: float, iv: float, debit: float, r: float = 0.0, q: float = 0.0) -> float:
    """The whole-dollar long strike whose spread costs closest to `debit` in the model that day."""
    ks = np.arange(math.floor(s * 0.9), math.ceil(s * 1.1) + 1, dtype=float)
    vals = np.array([spread_value(np.array([s]), k, k + width, np.array([t]), np.array([iv]), r, q)[0] for k in ks])
    return float(ks[int(np.argmin(np.abs(vals - debit)))])


def _expiry(d: pd.Timestamp, dte: int) -> pd.Timestamp:
    target = d + pd.Timedelta(days=dte)
    shift = (4 - target.weekday()) % 7          # next Friday
    nxt = target + pd.Timedelta(days=shift)
    prv = nxt - pd.Timedelta(days=7)
    return nxt if (nxt - target) <= (target - prv) else prv


def backtest(df: pd.DataFrame, p: SpreadParams, start: str, end: str | None = None, capital: float = 100000.0):
    d = df[(df['Date'] >= start) & ((df['Date'] <= end) if end else True)].reset_index(drop=True)
    full = df.set_index('Date')['Close']
    ma = full.rolling(p.trend_ma, min_periods=p.trend_ma).mean() if p.trend_ma else None
    dates = d['Date']
    if p.cadence == 'expiry':      # third Friday of each month, or the last trading day before it
        tf = {third_friday(y, m) for y in range(dates.iloc[0].year, dates.iloc[-1].year + 1) for m in range(1, 13)}
        dnp = dates.to_numpy()
        sched = np.zeros(len(d), bool)
        for f in tf:
            k = int(np.searchsorted(dnp, np.datetime64(f), side='right')) - 1
            if 0 <= k < len(d) and (f - dates.iloc[k]).days <= 3:
                sched[k] = True
    elif p.cadence == 'monthly':
        sched = dates.dt.to_period('M').ne(dates.dt.to_period('M').shift(1)).fillna(True).to_numpy().astype(bool)
    elif p.cadence == 'weekly':
        wk = dates.dt.isocalendar().week.astype(int)
        sched = wk.ne(wk.shift(1)).fillna(True).to_numpy().astype(bool)
    else:
        sched = np.ones(len(d), bool)
    trend_ok = (full.reindex(dates) > ma.reindex(dates)).fillna(False).to_numpy().astype(bool) if ma is not None else np.ones(len(d), bool)
    close = d['Close'].to_numpy(); iv = d['VIX'].to_numpy() / 100.0 * p.iv_mult
    rate = (d['Rate'].to_numpy() if 'Rate' in d and p.rate < 0 else np.full(len(d), max(p.rate, 0.0)))
    all_dates = full.index; n_days = len(d)
    # 1) every spread's daily mark until it closes (vectorised per spread)
    legs = []
    i = 0
    while i < n_days - 1:
        if not (sched[i] and trend_ok[i]) or np.isnan(iv[i]):
            i += 1; continue
        day = dates.iloc[i]
        expiry = _expiry_third_friday(day, p.dte) if p.cadence == 'expiry' else _expiry(day, p.dte)
        if expiry <= all_dates[-1]:
            expiry = all_dates[all_dates <= expiry][-1]
        j_end = int(np.searchsorted(dates.to_numpy(), np.datetime64(expiry), side='right'))   # exclusive
        j_end = min(max(j_end, i + 2), n_days)
        t_years = np.maximum((expiry - dates.iloc[i:j_end]).dt.days.to_numpy(), 0) / 365.0
        if expiry > all_dates[-1]:
            t_years = np.maximum(t_years, 1 / 365)
        if p.strike_mode == 'debit' and p.fixed_debit:
            k1 = _strike_for_debit(close[i], p.width, t_years[0], iv[i], p.fixed_debit - p.slip, rate[i], p.div_yield)
        else:
            k1 = float(round(close[i] * (1 + p.offset)))
        k2 = k1 + p.width
        marks = spread_value(close[i:j_end], k1, k2, t_years, iv[i:j_end], rate[i:j_end], p.div_yield)
        debit = p.fixed_debit if p.fixed_debit else float(marks[0]) + p.slip
        if p.fixed_debit:
            # what you paid above the model fades out linearly towards expiry: day 0 = your price, expiry = exact payoff
            fade = t_years / t_years[0] if t_years[0] > 0 else np.zeros_like(t_years)
            marks = np.clip(marks + (p.fixed_debit - float(marks[0])) * fade, 0.0, p.width)
        target = p.target_div if (p.target_div and expiry.month in (3, 6, 9, 12)) else p.target
        hit_t = np.where(marks[1:] >= target * p.width)[0]
        hit_s = np.where(marks[1:] <= debit * (1 - p.stop))[0] if p.stop else np.array([], int)
        j_close = j_end - 1; reason = 'expiry' if expiry <= all_dates[-1] else 'open'
        if len(hit_t) and (i + hit_t[0] + 1 < j_close or reason == 'open'):
            j_close, reason = i + hit_t[0] + 1, 'target'
        if len(hit_s) and i + hit_s[0] + 1 < j_close:
            j_close, reason = i + hit_s[0] + 1, 'stop'
        val = float(marks[j_close - i])
        exit_px = val if reason in ('expiry', 'open') else max(val - p.slip, 0.0)
        legs.append({'i': i, 'j': j_close, 'reason': reason, 'entry': day, 'expiry': expiry, 'k1': k1, 'k2': k2,
                     'debit': debit, 'exit': exit_px, 'marks': marks[: j_close - i + 1]})
        i = j_close + 1 if p.cadence == 'next' else i + 1
    # 2) account: size each spread at entry, mark open ones daily
    concurrent = p.concurrent()
    flow = np.zeros(n_days); trades = []          # cash in/out per day
    open_marks = np.zeros(n_days)                 # dollar value of open spreads per day
    for leg in sorted(legs, key=lambda l: l['i']):
        i, j = leg['i'], leg['j']
        cash_now = capital + flow[:i + 1].sum()   # only flows up to today count
        equity_now = cash_now + open_marks[i]
        n = int(equity_now * p.risk / concurrent // (leg['debit'] * 100)) if leg['debit'] > 0 else 0
        cost = leg['debit'] * 100 * n + p.commission * 2 * n
        if n <= 0 or cost > cash_now:
            continue
        expired_worthless = leg['reason'] == 'expiry' and leg['exit'] == 0
        proceeds = leg['exit'] * 100 * n - (0 if expired_worthless or leg['reason'] == 'open' else p.commission * 2 * n)
        flow[i] -= cost; flow[j] += proceeds
        open_marks[i:j] += leg['marks'][:-1] * 100 * n
        pnl = proceeds - cost
        trades.append({'ticker': 'SPY', 'entry_date': leg['entry'], 'exit_date': dates.iloc[j], 'expiry': leg['expiry'],
                       'spy_entry': close[i], 'k1': leg['k1'], 'k2': leg['k2'], 'contracts': n, 'debit': leg['debit'],
                       'exit': leg['exit'], 'pnl': pnl, 'net_return': pnl / cost, 'bars_held': j - i, 'exit_reason': leg['reason']})
    equity = capital + np.cumsum(flow) + open_marks
    eq = pd.Series(equity, index=pd.DatetimeIndex(dates), name='equity')
    tr = pd.DataFrame(trades)
    if len(tr):
        tr['pnl_1'] = (tr['exit'] - tr['debit']) * 100 - p.commission * 4   # one contract, constant size
    return eq, tr


# ----------------------------------------------------------------------------- reporting
def _spy_lines(df: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    from backtest.benchmarks import benchmark_metrics
    bench = df[['Date', 'AdjClose']].rename(columns={'AdjClose': 'Close'})
    return benchmark_metrics(bench, index)


def _max_buying_power(tr: pd.DataFrame) -> float:
    """Largest sum of debits (one contract each) open at the same time."""
    ev = []
    for r in tr.itertuples():
        ev.append((r.entry_date, r.debit * 100)); ev.append((r.exit_date, -r.debit * 100))
    cur = best = 0.0
    for _, v in sorted(ev, key=lambda x: (x[0], x[1])):
        cur += v; best = max(best, cur)
    return best


def _avg_debit(df, p, start) -> float:
    _, tr = backtest(df, p, start)
    return float(tr['debit'].mean()) if len(tr) else 0.0


def calibrate(df: pd.DataFrame, p: SpreadParams, start: str, debit: float) -> tuple[SpreadParams, str]:
    """Match the model to the debit you really pay. First scales the implied vol (VIX x 0.7 .. 1.6); if that
    cannot reach the debit, keeps VIX x 1.0 and moves the long strike instead (the strikes that cost `debit`)."""
    lo, hi = 0.7, 1.6
    if _avg_debit(df, SpreadParams(**{**asdict(p), 'iv_mult': lo}), start) <= debit <= _avg_debit(df, SpreadParams(**{**asdict(p), 'iv_mult': hi}), start):
        for _ in range(14):
            mid = (lo + hi) / 2
            if _avg_debit(df, SpreadParams(**{**asdict(p), 'iv_mult': mid}), start) < debit:
                lo = mid
            else:
                hi = mid
        q = SpreadParams(**{**asdict(p), 'iv_mult': round((lo + hi) / 2, 3)})
        return q, f'Calibrated: implied vol = VIX x {q.iv_mult} gives an average debit of ${debit:.2f} for this setup'
    base = SpreadParams(**{**asdict(p), 'iv_mult': 1.0})
    model = _avg_debit(df, base, start)
    q = SpreadParams(**{**asdict(base), 'fixed_debit': debit})
    return q, (f'NOTE: with the VIX as implied vol, a spread {p.offset * 100:+.1f}% from the price costs about ${model:.2f} in the model, '
               f'not ${debit:.2f}, and no plausible volatility gets there. The backtest therefore pays exactly ${debit:.2f} per spread '
               f'(your strikes, your price; marks scaled accordingly, payoff at expiry exact). If your fills are really ${debit:.2f}, '
               f'you are paying about ${debit - model:.2f} more than the model thinks fair - check whether your strikes sit closer to the money.')


def report(df: pd.DataFrame, p: SpreadParams, start: str, holdout: str, capital: float = 100000.0) -> tuple[list[str], dict]:
    eq, tr = backtest(df, p, start, None, capital)
    out = {}
    lines = [p.describe(), '']
    periods = [('Full period', start, None), ('Development period', start, holdout), ('Test period', holdout, None)]
    for name, a, b in periods:
        e = eq[(eq.index >= a) & ((eq.index < b) if b else True)]
        t = tr[(tr['exit_date'] >= a) & ((tr['exit_date'] < b) if b else True)] if len(tr) else tr
        if len(e) < 30:
            continue
        m = compute_metrics(e / e.iloc[0] * capital, t)
        spy = _spy_lines(df, e.index)
        out[name] = {'strategy': m, **spy}
        lines += [f'{name} {e.index[0].date()} .. {e.index[-1].date()}',
                  f'  {"":28s} {"return/yr":>10s} {"worst drop":>11s} {"Sharpe":>7s} {"Calmar":>7s} {"trades":>7s} {"win":>5s}']
        for label, mm in [('Call spreads (whole account)', m)] + list(spy.items()):
            lines.append(f'  {label:28s} {fmt_pct(mm["cagr"]):>10s} {fmt_pct(mm["max_drawdown"]):>11s} '
                         f'{mm["sharpe"]:7.2f} {mm["calmar"]:7.2f} {mm["trades"]:7d} {fmt_pct(mm["win_rate"]) if mm["trades"] else "":>5s}')
        if len(t):
            closed = t[t['exit_reason'] != 'open']
            yrs = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
            bp = _max_buying_power(t)
            lines.append(f'  one contract per spread, constant size: P&L ${closed["pnl_1"].sum():,.0f} in {yrs:.1f} years '
                         f'on at most ${bp:,.0f} tied up at once = {closed["pnl_1"].sum() / yrs / bp * 100 if bp else 0:+.0f}% per year on that buying power'
                         + (f' ({len(t) - len(closed)} still open)' if len(t) > len(closed) else ''))
            rs = t['exit_reason'].value_counts()
            dist = ((t['k1'] + p.width / 2) / t['spy_entry'] - 1) * 100
            lines.append(f'  strikes: the spread midpoint sat on average {dist.mean():+.2f}% from the SPY price (range {dist.min():+.1f}% .. {dist.max():+.1f}%)')
            lines.append(f'  per spread: avg {t["net_return"].mean() * 100:+.1f}% of the debit, '
                         f'avg win {t.loc[t.pnl > 0, "net_return"].mean() * 100 if (t.pnl > 0).any() else 0:+.0f}%, '
                         f'avg loss {t.loc[t.pnl <= 0, "net_return"].mean() * 100 if (t.pnl <= 0).any() else 0:+.0f}%, '
                         f'debit ${t["debit"].mean():.2f} avg, closed at target {rs.get("target", 0)}, '
                         f'expiry {rs.get("expiry", 0)}, stop {rs.get("stop", 0)}; '
                         f'total P&L ${t["pnl"].sum():,.0f} on ${(t["debit"] * 100 * t["contracts"]).mean():,.0f} avg at risk')
        lines.append('')
    lines.append(f'Position size (whole account of ${capital:,.0f}, full period): the same spreads with a different share of the account at risk in total')
    for risk in (0.02, 0.05, 0.10, 0.20):
        e2, t2 = backtest(df, SpreadParams(**{**asdict(p), 'risk': risk}), start, None, capital)
        m2 = compute_metrics(e2 / e2.iloc[0] * capital, t2)
        lines.append(f'  {risk * 100:3.0f}% at risk: {fmt_pct(m2["cagr"]):>8s} per year, worst drop {fmt_pct(m2["max_drawdown"])}, Sharpe {m2["sharpe"]:.2f}')
    lines.append('')
    if len(eq) > 252:
        yr = yearly_returns(eq)
        spy_yr = yearly_returns(df.set_index('Date')['AdjClose'].reindex(eq.index).ffill())
        lines.append('Year by year (account vs SPY): ' + ', '.join(f'{y}: {fmt_pct(v)} / {fmt_pct(spy_yr.get(y, float("nan")))}' for y, v in yr.items()))
        lines.append('')
    if len(tr):
        lines.append('Last trades:')
        show = tr.tail(8)[['entry_date', 'spy_entry', 'k1', 'k2', 'debit', 'exit_date', 'exit', 'net_return', 'exit_reason']].copy()
        show['entry_date'] = show['entry_date'].dt.date; show['exit_date'] = show['exit_date'].dt.date
        show['net_return'] = (show['net_return'] * 100).round(0)
        lines += ['  ' + l for l in show.round(2).to_string(index=False).split('\n')]
    return lines, {'equity': eq, 'trades': tr, 'metrics': out}


def is_spread_file(path) -> bool:
    try:
        return json.loads(Path(path).read_text()).get('type') == 'spreads'
    except (OSError, ValueError):
        return False


def run_one(path, start: str, holdout_start: str, capital: float = 100000.0) -> dict:
    """Same shape as compare.run_one: equity, metrics, verdict count and luck for an option plan."""
    from backtest import criteria
    from backtest.benchmarks import benchmark_metrics, spy_buy_hold
    from backtest.significance import luck_summary
    path = Path(path); p = SpreadParams.load(path); meta = json.loads(path.read_text())
    df = load()
    eq, tr = backtest(df, p, start, None, capital)
    hold = eq[eq.index >= holdout_start]
    stressed, trs = backtest(df, SpreadParams(**{**asdict(p), 'slip': p.slip * 2, 'commission': p.commission * 2}), start, None, capital)
    closed = tr[tr.exit_reason != 'open'] if len(tr) else tr
    fm = compute_metrics(eq, closed); hm = compute_metrics(hold / hold.iloc[0] * capital, closed[closed.exit_date >= holdout_start] if len(closed) else closed)
    sm = compute_metrics(stressed, trs[trs.exit_reason != 'open'] if len(trs) else trs)
    bench = df[['Date', 'AdjClose']].rename(columns={'AdjClose': 'Close'})
    fb = benchmark_metrics(bench, eq.index); hb = benchmark_metrics(bench, hold.index)
    trials = max(1, int(meta.get('trials', 1) or 1))
    luck = luck_summary(eq, spy_buy_hold(bench, eq.index), trials)
    verdict = criteria.evaluate(fm, hm, sm, fb['SPY buy & hold'], hb['SPY buy & hold'], hb['SPY timing (>SMA200)'], luck)
    return {'file': path.name, 'name': meta.get('name') or path.stem, 'index': 'SPY options', 'params': p,
            'equity': eq, 'trades': tr, 'full': fm, 'holdout': hm, 'passed': sum(ok for _, ok, _ in verdict), 'checks': len(verdict),
            'luck': min(luck['dsr']['prob'], luck['bootstrap']['p_positive']), 'trials': trials, 'verdict': verdict}


def build_report(path, start: str, holdout_start: str, capital: float = 100000.0) -> Path:
    """The same one-page HTML report as backtest.report, for an option plan."""
    from backtest.report import render
    from backtest.benchmarks import benchmark_metrics, spy_buy_hold, spy_timing
    path = Path(path); p = SpreadParams.load(path); meta = json.loads(path.read_text()); df = load()
    r = run_one(path, start, holdout_start, capital)
    eq, tr = r['equity'], r['trades']
    closed = tr[tr.exit_reason != 'open'] if len(tr) else tr
    tune_end = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    dev = eq[eq.index <= tune_end]
    has_dev = tune_end > pd.Timestamp(start) + pd.Timedelta(days=60) and len(dev) > 30
    tune_m = compute_metrics(dev / dev.iloc[0] * capital, closed[closed.exit_date <= tune_end] if len(closed) else closed) if has_dev else None
    bench = df[['Date', 'AdjClose']].rename(columns={'AdjClose': 'Close'})
    tune_b = benchmark_metrics(bench, dev.index) if has_dev else None
    full_b = benchmark_metrics(bench, eq.index)
    hold_b = benchmark_metrics(bench, eq.index[eq.index >= pd.Timestamp(holdout_start)])
    curves = {'Strategy': eq / eq.iloc[0] * 10_000, 'SPY buy & hold': spy_buy_hold(bench, eq.index, 10_000),
              'SPY timing': spy_timing(bench, eq.index, capital=10_000)}
    for m in (r['full'], r['holdout'], tune_m or {}):
        m['exposure'] = float('nan')
    return render(
        out_stem=path.stem, title=r['name'], subtitle_what='Backtest on SPY options (Black-Scholes marks from the VIX, exact payoff at expiry)',
        cost_text=f'${p.slip:.2f} slippage per side, ${p.commission:.2f} commission per contract and leg, {p.risk * 100:.0f}% of the account at risk',
        note=meta.get('note', ''), rules=p.describe(), verdict=r['verdict'],
        full_m=r['full'], hold_m=r['holdout'], tune_m=tune_m, full_b=full_b, hold_b=hold_b, tune_b=tune_b,
        curves=curves, trades=closed, start=start, holdout_start=holdout_start, tune_end=tune_end,
        reason_names={'target': 'Closed at the target', 'expiry': 'Expired', 'stop': 'Stop', 'open': 'Still open'},
        stress_text='double slippage and commissions')


def _html(text: str, eq: pd.Series, df: pd.DataFrame, holdout: str, title: str = 'SPY call spreads') -> str:
    """Self-contained page: growth of the account vs SPY (log scale) above the text report."""
    import html as _h
    spy = df.set_index('Date')['AdjClose'].reindex(eq.index).ffill()
    spy = spy / spy.iloc[0] * eq.iloc[0]
    W, H, L, R, T, B = 900, 360, 70, 20, 20, 40
    lo = min(eq.min(), spy.min()) * 0.95; hi = max(eq.max(), spy.max()) * 1.05
    n = len(eq)
    x = lambda i: L + (W - L - R) * i / max(n - 1, 1)
    y = lambda v: T + (H - T - B) * (1 - (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    def path(s):
        return 'M' + ' L'.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(s.to_numpy()))
    parts = []
    hi_idx = int(np.searchsorted(eq.index.to_numpy(), np.datetime64(pd.Timestamp(holdout))))
    if 0 < hi_idx < n:
        parts.append(f'<rect x="{x(hi_idx):.1f}" y="{T}" width="{x(n - 1) - x(hi_idx):.1f}" height="{H - T - B}" fill="#2a78d6" opacity="0.07"/>')
        parts.append(f'<text x="{x(hi_idx) + 6:.1f}" y="{T + 14}" fill="#52514e" font-size="12">test period</text>')
    for v in sorted({eq.iloc[0] * m for m in (0.5, 1, 2, 4, 8) if lo <= eq.iloc[0] * m <= hi}):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e3e2dd"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="12" fill="#52514e">${v:,.0f}</text>')
    years = eq.index.year
    for yr in sorted(set(years)):
        i = int(np.argmax(years == yr))
        parts.append(f'<text x="{x(i):.1f}" y="{H - 10}" text-anchor="middle" font-size="12" fill="#52514e">{yr}</text>')
    parts.append(f'<path d="{path(spy)}" fill="none" stroke="#7a7974" stroke-width="2"/>')
    parts.append(f'<path d="{path(eq)}" fill="none" stroke="#2a78d6" stroke-width="2.2"/>')
    parts.append(f'<text x="{W - R}" y="{y(spy.iloc[-1]) + 4:.1f}" text-anchor="end" font-size="12" fill="#7a7974">SPY</text>')
    parts.append(f'<text x="{W - R}" y="{y(eq.iloc[-1]) - 8:.1f}" text-anchor="end" font-size="12" fill="#2a78d6" font-weight="600">Call spreads</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:960px;display:block">' + ''.join(parts) + '</svg>'
    import html as _hh
    return ('<!doctype html><meta charset="utf-8"><title>' + _hh.escape(title) + '</title>'
            '<body style="margin:0;background:#f4f3f0;color:#0b0b0b;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
            '<main style="max-width:1000px;margin:0 auto;padding:28px 16px">'
            '<h1 style="font-size:24px;margin:0 0 4px">' + _hh.escape(title) + '</h1>'
            '<p style="color:#52514e;margin:0 0 16px">Growth of the account vs SPY buy &amp; hold, log scale. Shaded: test period.</p>'
            f'<div style="background:#fcfcfb;border:1px solid #e3e2dd;border-radius:14px;padding:16px">{svg}</div>'
            f'<pre style="white-space:pre-wrap;font:13px/1.5 ui-monospace,Menlo,monospace;background:#fcfcfb;border:1px solid #e3e2dd;border-radius:14px;padding:16px;margin-top:18px">{_h.escape(text)}</pre>'
            '</main></body>')


# What --tune tries. 'strike' is either "debit:<share of width>" (the strikes that cost that debit, e.g. 0.5 = 4.95 on a
# $10 width) or "offset:<fraction>" (long strike at that distance from the price). Every value can be changed with
# --grid key=v1,v2,...  e.g. --grid width=5,10 --grid dte=30,45,60 --grid strike=debit:0.4,debit:0.5,offset:0.01
GRID = {
    'strike': ['debit:0.4', 'debit:0.5', 'debit:0.6', 'offset:0.0', 'offset:0.017'],
    'width': [5.0, 10.0, 20.0],
    'dte': [30, 60, 90],
    'target': [0.8, 0.99],
    'stop': [0.0, 0.5],
    'cadence': ['expiry', 'weekly'],
    'trend_ma': [0, 200],
}


def parse_grid(items: list[str] | None) -> dict:
    """--grid key=v1,v2 overrides one row of GRID."""
    grid = {k: list(v) for k, v in GRID.items()}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'--grid expects key=v1,v2 (got {item!r})')
        key, vals = item.split('=', 1)
        key = key.strip()
        if key not in grid:
            raise SystemExit(f'--grid: unknown key {key!r}; choose from {", ".join(grid)}')
        parsed = []
        for v in vals.split(','):
            v = v.strip()
            if not v:
                continue
            if key in ('strike', 'cadence'):
                parsed.append(v)
            elif key in ('dte', 'trend_ma'):
                parsed.append(int(float(v)))
            else:
                parsed.append(float(v))
        if parsed:
            grid[key] = parsed
    for v in grid['strike']:
        if not (v.startswith('debit:') or v.startswith('offset:')):
            raise SystemExit(f'strike values look like debit:0.5 or offset:0.017 (got {v!r})')
    for v in grid['cadence']:
        if v not in ('expiry', 'monthly', 'weekly', 'next'):
            raise SystemExit(f'cadence must be expiry, monthly, weekly or next (got {v!r})')
    return grid


def _apply(base: SpreadParams, combo: dict) -> SpreadParams:
    d = {**asdict(base), **{k: v for k, v in combo.items() if k != 'strike'}}
    kind, val = combo['strike'].split(':')
    if kind == 'debit':
        d.update(strike_mode='debit', fixed_debit=round(float(val) * d['width'], 2))
    else:
        d.update(strike_mode='offset', fixed_debit=0.0, offset=float(val))
    return SpreadParams(**d)


def tune(df: pd.DataFrame, base: SpreadParams, start: str, holdout: str, metric: str = 'calmar',
         grid: dict | None = None, max_setups: int = 0, seed: int = 0) -> pd.DataFrame:
    """Backtest every combination of the grid (or a random sample of `max_setups` of them)."""
    grid = grid or GRID
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    if max_setups and len(combos) > max_setups:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), max_setups, replace=False))]
    rows = []
    for i, vals in enumerate(combos):
        combo = dict(zip(keys, vals))
        p = _apply(base, combo)
        eq, tr = backtest(df, p, start)
        dev = eq[eq.index < holdout]; hold = eq[eq.index >= holdout]
        m_dev = compute_metrics(dev / dev.iloc[0], tr[tr['exit_date'] < holdout] if len(tr) else tr)
        m_hold = compute_metrics(hold / hold.iloc[0], tr[tr['exit_date'] >= holdout] if len(tr) else tr) if len(hold) > 30 else {}
        rows.append({**combo, 'debit': round(float(tr['debit'].mean()), 2) if len(tr) else np.nan, 'dev_cagr': m_dev['cagr'], 'dev_dd': m_dev['max_drawdown'], 'dev_sharpe': m_dev['sharpe'],
                     'dev_calmar': m_dev['calmar'], 'trades': m_dev['trades'], 'win': m_dev['win_rate'],
                     'hold_cagr': m_hold.get('cagr', np.nan), 'hold_dd': m_hold.get('max_drawdown', np.nan),
                     'hold_trades': m_hold.get('trades', 0)})
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(combos)}', flush=True)
    res = pd.DataFrame(rows)
    res = res[res['trades'] >= 20].sort_values('dev_' + metric, ascending=False)
    return res


def _match_row(res: pd.DataFrame, p: SpreadParams) -> pd.DataFrame:
    """The leaderboard row closest to a given setup (same width/expiry/rules, strike within a notch)."""
    if not len(res):
        return res
    def strike_ok(sv: str) -> bool:
        kind, val = sv.split(':')
        if p.strike_mode == 'debit' and p.fixed_debit:
            return kind == 'debit' and abs(float(val) - p.fixed_debit / p.width) < 0.03
        return kind == 'offset' and abs(float(val) - p.offset) < 1e-6
    mask = res.apply(lambda r: strike_ok(r['strike']) and abs(r['width'] - p.width) < 1e-9 and int(r['dte']) == p.dte
                     and abs(r['target'] - p.target) < 1e-9 and abs(r['stop'] - p.stop) < 1e-9
                     and r['cadence'] == p.cadence and int(r['trend_ma']) == p.trend_ma, axis=1)
    return res[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--metric', default='calmar', choices=['calmar', 'sharpe', 'cagr'])
    ap.add_argument('--grid', action='append', metavar='KEY=V1,V2', help='change what --tune tries, e.g. --grid dte=30,45,60 (repeatable)')
    ap.add_argument('--max-setups', type=int, default=0, help='--tune: random sample of at most this many setups (0 = all)')
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--holdout-start', default='2025-01-01')
    ap.add_argument('--params', help='an option plan file from strategies/ (type "spreads"); makes the HTML report like backtest.report')
    ap.add_argument('--save', help='save this setup as strategies/<name>.json so Reports and Compare pick it up')
    ap.add_argument('--capital', type=float, default=100000, help='account size; contracts are whole, so small accounts cannot size small risks')
    ap.add_argument('--debit', type=float, default=0.0, help='what you pay per spread (same as --fixed-debit)')
    for f, v in asdict(SpreadParams()).items():
        ap.add_argument('--' + f.replace('_', '-'), type=type(v), default=v)
    a = ap.parse_args()
    if a.fetch:
        df = fetch(); print(f'Saved {UNDERLYING}: {df.Date.iloc[0].date()} .. {df.Date.iloc[-1].date()}, {len(df)} days'); return
    if a.params:
        from backtest.criteria import format_verdict
        r = run_one(a.params, a.start, a.holdout_start, a.capital)
        print('\n'.join([f'VERDICT ({r["passed"]}/{r["checks"]} checks passed)'] + format_verdict(r['verdict'])))
        lines, _ = report(load(), SpreadParams.load(a.params), a.start, a.holdout_start, a.capital)
        print('\n' + '\n'.join(lines))
        out = build_report(a.params, a.start, a.holdout_start, a.capital)
        print(f'\nReport saved: {out}')
        return
    df = load()
    p = SpreadParams(**{f: getattr(a, f) for f in asdict(SpreadParams())})
    if a.debit:
        p.fixed_debit = a.debit
    note = ''
    if a.save:
        from backtest.app import slugify
        f = ROOT_DIR / 'strategies' / f'{slugify(a.save)}.json'
        f.write_text(json.dumps({'name': a.save, 'type': 'spreads', 'params': p.to_dict()}, indent=2) + '\n')
        print(f'Saved plan: {f.relative_to(ROOT_DIR)}  (now in the plan list of Reports / Compare)')
        return
    out_dir = OUTPUT_DIR / f'spreads_{datetime.now():%Y%m%d_%H%M%S}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.tune:
        grid = parse_grid(a.grid)
        total = int(np.prod([len(v) for v in grid.values()]))
        n = min(total, a.max_setups) if a.max_setups else total
        print(f'Tuning {n} of {total} setups on {a.start}..{a.holdout_start}, ranked by {a.metric} (development period)')
        print('  grid: ' + '; '.join(f'{k} = {", ".join(str(x) for x in v)}' for k, v in grid.items()))
        res = tune(df, p, a.start, a.holdout_start, a.metric, grid, a.max_setups)
        res.to_csv(out_dir / 'leaderboard.csv', index=False)
        lines = ['SPY CALL SPREADS - TUNING', f'{n} setups, all priced by the model (VIX x {p.iv_mult} as implied vol, plus slippage); '
                 f'development {a.start}..{a.holdout_start}, test period {a.holdout_start}..today. '
                 'Development numbers are in-sample (the best of many); trust the test period and the spread of results more than the top row.',
                 'strike = debit:X means the strikes that cost X of the width that day (debit:0.5 = $4.95 on a $10 width); '
                 'offset:X = long strike X from the price. debit = average price paid per spread.', '',
                 'Tried: ' + '; '.join(f'{k} = {", ".join(str(x) for x in v)}' for k, v in grid.items()), '',
                 'Top 15 by development ' + a.metric + ':']
        show = res.head(15).copy()
        for c in ('dev_cagr', 'dev_dd', 'win', 'hold_cagr', 'hold_dd'):
            show[c] = (show[c] * 100).round(1)
        lines += ['  ' + l for l in show.round(2).to_string(index=False).split('\n')]
        yours = _match_row(res, p)
        if len(yours):
            r = yours.iloc[0]; rank = int((res['dev_' + a.metric] > r['dev_' + a.metric]).sum()) + 1
            lines += ['', f'Your setup ({r.strike}, ${p.width:g} wide, {p.dte} days, close at {p.target * 100:.0f}%, {p.cadence}, '
                          f'trend {p.trend_ma or "none"}): rank {rank} of {len(res)}, development {r.dev_cagr * 100:.1f}%/yr, '
                          f'test period {r.hold_cagr * 100:.1f}%/yr']
        tp = res['hold_trades'].median() if 'hold_trades' in res else 0
        lines += ['', f'Note: the test period holds only ~{tp:.0f} trades per setup; a few trades decide a 2025+ number, so neighbouring rows '
                      'that disagree wildly (same setup, strike one notch apart) are telling you the result is noise, not that one is better.']
        lines += ['', 'Share of setups that made money in the test period: '
                  f'{(res.hold_cagr > 0).mean() * 100:.0f}%; median test-period return {res.hold_cagr.median() * 100:.1f}%/yr']
        spy = _spy_lines(df, df[df.Date >= a.holdout_start].set_index('Date').index)
        lines.append(f'SPY buy & hold in the test period: {spy["SPY buy & hold"]["cagr"] * 100:.1f}%/yr')
    else:
        lines, res = report(df, p, a.start, a.holdout_start, a.capital)
        res['trades'].to_csv(out_dir / 'trades.csv', index=False)
        res['equity'].to_csv(out_dir / 'equity.csv')
        lines = ['SPY CALL SPREADS - BACKTEST (Black-Scholes marks with VIX as implied vol; payoff at expiry exact)', ''] + lines
    if note:
        lines.insert(2, note); lines.insert(3, '')
    text = '\n'.join(lines)
    (out_dir / 'report.txt').write_text(text)
    print(text); print(f'\nSaved to {out_dir}')
    if not a.tune:
        html_path = out_dir / 'report.html'
        html_path.write_text(_html(text, res['equity'], df, a.holdout_start))
        print(f'Report saved: {html_path}')


if __name__ == '__main__':
    main()
