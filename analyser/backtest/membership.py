"""Point-in-time S&P 500 membership, to avoid survivorship bias.

    python3 -m backtest.membership            # rebuild symbols/sp500_membership.csv from the source
    python3 -m backtest.membership --check    # how much of the history do we have prices for?

Source: daily membership snapshots 1996-today maintained in the public GitHub repo
fja05680/sp500 (built from Andreas Clenow's list plus Wikipedia changes). It is turned into
intervals  Ticker, Start, End  (End empty = still a member) and stored in symbols/ so the repo
works offline. `python3 -m backtest.fetch --historical` then downloads prices for every ticker
that was a member since the backtest start, and the backtest only buys a stock while it was
actually in the index.

Limitation: Yahoo Finance has no prices for most delisted companies (taken over, bankrupt), so
some removed members stay missing. `--check` reports the coverage so you know how much.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

from backtest.data import DATA_DIR, ROOT_DIR, load_table

SOURCE_URL = ('https://raw.githubusercontent.com/fja05680/sp500/master/'
              'S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv')
MEMBERSHIP_FILE = ROOT_DIR / 'symbols' / 'sp500_membership.csv'


def download_source() -> pd.DataFrame:
    r = requests.get(SOURCE_URL, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def intervals_from_snapshots(snap: pd.DataFrame) -> pd.DataFrame:
    """snap: columns date, tickers ('A,AAPL,...'). Returns Ticker, Start, End intervals."""
    snap = snap.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(snap['date'])
    sets = [set(t.strip().upper().replace('.', '-') for t in str(row).split(',') if t.strip())
            for row in snap['tickers']]
    rows = []
    open_since: dict[str, pd.Timestamp] = {}
    prev: set[str] = set()
    for d, members in zip(dates, sets):
        for t in members - prev:
            open_since[t] = d
        for t in prev - members:
            rows.append((t, open_since.pop(t), d - pd.Timedelta(days=1)))
        prev = members
    for t, start in open_since.items():
        rows.append((t, start, pd.NaT))
    df = pd.DataFrame(rows, columns=['Ticker', 'Start', 'End']).sort_values(['Ticker', 'Start'])
    return df.reset_index(drop=True)


def reconcile_with_current(df: pd.DataFrame, current: set[str], as_of: pd.Timestamp) -> pd.DataFrame:
    """The source lags Wikipedia by a few weeks: close intervals of tickers no longer in today's
    list at the source's last snapshot, and open one for today's members the source doesn't have."""
    df = df.copy()
    open_now = df['End'].isna()
    stale = open_now & ~df['Ticker'].isin(current)
    df.loc[stale, 'End'] = as_of
    missing = sorted(current - set(df.loc[df['End'].isna(), 'Ticker']))
    add = pd.DataFrame({'Ticker': missing, 'Start': as_of + pd.Timedelta(days=1), 'End': pd.NaT})
    return pd.concat([df, add], ignore_index=True).sort_values(['Ticker', 'Start']).reset_index(drop=True)


def build(path: Path = MEMBERSHIP_FILE) -> pd.DataFrame:
    snap = download_source()
    df = intervals_from_snapshots(snap)
    current_file = ROOT_DIR / 'symbols' / 'sp500.csv'
    if current_file.exists():
        current = set(pd.read_csv(current_file)['Symbol'].str.strip().str.upper())
        df = reconcile_with_current(df, current, pd.to_datetime(snap['date']).max())
    path.parent.mkdir(exist_ok=True)
    out = df.copy()
    out['Start'] = out['Start'].dt.strftime('%Y-%m-%d')
    out['End'] = out['End'].dt.strftime('%Y-%m-%d')
    out.to_csv(path, index=False)
    return df


def load_membership(index: str | None = 'sp500', path: Path = MEMBERSHIP_FILE) -> pd.DataFrame | None:
    """Intervals for the index, or None when no membership file exists (then today's list is used)."""
    parts = (index or '').split('+')
    if not any(i in ('sp500', 'sp100') for i in parts) or not path.exists():
        return None  # tickers without a record are treated as always eligible (see FeatureStore)
    df = pd.read_csv(path)
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])
    return df


def tickers_since(membership: pd.DataFrame, start: str) -> list[str]:
    """Every ticker that was a member on or after `start`."""
    m = membership[membership['End'].isna() | (membership['End'] >= pd.Timestamp(start))]
    return sorted(m['Ticker'].unique())


def coverage(membership: pd.DataFrame, prices: pd.DataFrame, start: str) -> dict:
    """How much of the member-days since `start` do we have prices for?"""
    start_ts = pd.Timestamp(start)
    have = prices.groupby('Ticker')['Date'].agg(['min', 'max'])
    end_all = prices['Date'].max()
    rows = []
    for r in membership.itertuples(index=False):
        s = max(r.Start, start_ts)
        e = r.End if pd.notna(r.End) else end_all
        if e < s:
            continue
        days = (e - s).days
        if r.Ticker in have.index:
            hs, he = have.loc[r.Ticker, 'min'], have.loc[r.Ticker, 'max']
            got = max(0, (min(e, he) - max(s, hs)).days)
        else:
            got = 0
        rows.append({'Ticker': r.Ticker, 'Start': s.date(), 'End': e.date(), 'days': days, 'have': got,
                     'removed': pd.notna(r.End)})
    df = pd.DataFrame(rows)
    removed = df[df['removed']]
    missing = df[df['have'] < df['days'] * 0.5]
    return {
        'member_days_covered': df['have'].sum() / max(1, df['days'].sum()),
        'tickers': df['Ticker'].nunique(),
        'removed_tickers': removed['Ticker'].nunique(),
        'removed_with_prices': removed[removed['have'] > 0]['Ticker'].nunique(),
        'missing': sorted(missing['Ticker'].unique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='report price coverage of the membership history')
    ap.add_argument('--start', default='2016-01-01')
    args = ap.parse_args()
    if args.check:
        m = load_membership()
        if m is None:
            raise SystemExit('No membership file. Run python3 -m backtest.membership first.')
        prices = load_table('prices')
        c = coverage(m, prices, args.start)
        print(f'Since {args.start}: {c["tickers"]} tickers were S&P 500 members, {c["removed_tickers"]} of them were removed.')
        print(f'Prices available for {c["removed_with_prices"]} of the removed ones.')
        print(f'Member-days with prices: {c["member_days_covered"] * 100:.1f}%')
        if c['missing']:
            print(f'{len(c["missing"])} tickers with (almost) no prices: {", ".join(c["missing"][:40])}'
                  f'{" ..." if len(c["missing"]) > 40 else ""}')
        return
    df = build()
    n_removed = df['End'].notna().sum()
    print(f'Saved {len(df)} membership intervals for {df["Ticker"].nunique()} tickers to {MEMBERSHIP_FILE.relative_to(ROOT_DIR)}')
    print(f'{n_removed} removals recorded, snapshots {df["Start"].min().date()} .. today.')
    print('Next: python3 -m backtest.fetch --historical')


if __name__ == '__main__':
    main()
