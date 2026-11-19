"""Planner web app: build a trading plan in the browser and backtest it with one click.

    python3 -m backtest.app            # opens http://127.0.0.1:8765 in your browser
    python3 -m backtest.app --port 9000 --no-open

Runs only on your own computer (nothing is published). Stop it with Ctrl+C.
Uses only the Python standard library on top of the backtest package.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from backtest.data import OUTPUT_DIR, ROOT_DIR, load_benchmark, load_prices
from backtest import jobs
from backtest.features import FeatureStore
from backtest.membership import load_membership
from backtest.workbench import PAGE as WORKBENCH_PAGE
from backtest.strategy import StrategyParams

STRATEGY_DIR = ROOT_DIR / 'strategies'

# key, name, short explanation, has length, default length, length label, entry default, exit default, scale hint
INDICATORS = [
    ('rsi', 'RSI', 'Relative Strength Index: momentum of recent closes.', True, 4, 'Length (days)', 20, 55, '0 to 100, oversold is low (e.g. 20)'),
    ('willr', 'Williams %R', 'Where the close sits in the recent high-low range.', True, 10, 'Length (days)', -90, -50, '-100 to 0, oversold is near -100 (e.g. -90)'),
    ('stoch', 'Stochastic %K (slow)', 'Smoothed position of the close in the recent range.', True, 14, 'Length (days)', 20, 60, '0 to 100 (e.g. 20)'),
    ('crsi', 'Connors RSI', 'Mix of RSI(3), up/down streak and today\'s move vs. the last 100 days.', False, 100, '', 10, 70, '0 to 100 (e.g. 10)'),
    ('mfi', 'Money Flow Index', 'RSI weighted by volume.', True, 14, 'Length (days)', 20, 60, '0 to 100 (e.g. 20)'),
    ('bbp', 'Bollinger %B', 'Close relative to the Bollinger Bands (2 std).', True, 20, 'Band period (days)', 0, 60, '0 = lower band, 100 = upper band; can go below 0'),
    ('cci', 'CCI (rank)', 'Commodity Channel Index, ranked against the last year.', True, 252, 'Rank window (days)', 5, 60, '0 to 100, 5 = lowest 5% of the window'),
    ('macd', 'MACD histogram (rank)', 'MACD histogram ranked against the last year.', True, 252, 'Rank window (days)', 5, 60, '0 to 100, 5 = lowest 5% of the window'),
    ('ibs', 'IBS', 'Close near the low of the day (internal bar strength).', False, 0, '', 15, 70, '0 = closed at the low, 100 = at the high'),
    ('uo', 'Ultimate Oscillator', 'Buying pressure over 7, 14 and 28 days.', False, 0, '', 30, 60, '0 to 100 (e.g. 30)'),
]
IND_KEYS = {i[0] for i in INDICATORS}
MOMENTUM_ENTRIES = {
    'rs': 'Relative strength: the stock is among the top N of all stocks by performance over the lookback.',
    'breakout': 'Breakout: the close is above the highest high of the previous N days.',
}
EXIT_RULES = ('osc', 'prev_high', 'channel', 'ma', 'rank', 'none')

_FS_CACHE: dict[str, FeatureStore] = {}
_FS_LOCK = threading.Lock()


def feature_store(index: str = 'sp500') -> FeatureStore:
    with _FS_LOCK:
        if index not in _FS_CACHE:
            _FS_CACHE[index] = FeatureStore(load_prices(index), load_benchmark('SPY'), load_membership(index))
        return _FS_CACHE[index]


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:60] or 'plan'


def trend_from(value) -> dict:
    """Form value '200' -> above the 200-day average; '200:-0.15' -> at least 15% below it."""
    s = str(value or '0')
    if ':' in s:
        ma, gap = s.split(':', 1)
        return {'trend_ma': int(float(ma)), 'trend_below': abs(float(gap))}
    return {'trend_ma': int(float(s)), 'trend_below': 0.0}


def params_from_form(f: dict) -> StrategyParams:
    """Form values (native indicator scale) -> StrategyParams (0..100 scale)."""
    family = f.get('family', 'reversion')
    num = lambda k, d=0.0: float(f.get(k, d) or 0)
    if family == 'momentum':
        ind = f.get('mom_entry', 'rs')
        if ind not in MOMENTUM_ENTRIES:
            raise ValueError('Unknown momentum entry')
        exit_rule = f.get('mom_exit', 'rank')
        if exit_rule not in ('rank', 'ma', 'channel', 'none'):
            raise ValueError('Unknown exit rule')
        if ind == 'breakout' and exit_rule == 'rank':
            exit_rule = 'ma'
        p = StrategyParams(
            entry_indicator=ind,
            entry_length=int(num('mom_lookback', 126)),
            entry_threshold=num('mom_top', 20),
            skip_days=int(num('skip_days', 0)),
            rebalance=('daily' if ind == 'breakout' else
                       f.get('rebalance', 'monthly') if f.get('rebalance') in ('daily', 'weekly', 'monthly') else 'monthly'),
            exit_rule=exit_rule,
            exit_threshold=num('mom_exit_rank', 50),
            exit_length=int(num('mom_exit_length', 100)),
            **trend_from(f.get('trend_ma')), market_ma=int(num('market_ma')),
            stop_adr_mult=num('stop_adr_mult'), trail_adr_mult=num('trail_adr_mult'),
            max_hold_days=int(num('max_hold_days')),
            rank_by='momentum',
            max_positions=max(1, int(num('max_positions', 10))),
            min_price=num('min_price', 5),
            min_dollar_volume=num('min_dollar_volume_m', 5) * 1e6,
            universe=f.get('universe', '') if f.get('universe') in ('', 'sp100', 'nasdaq100', 'sp100+nasdaq100') else '',
            max_price=num('max_price'), max_new_per_day=int(num('max_new_per_day')),
            min_history_days=int(num('min_history_years') * 252),
            obe_min_gap=num('obe_pct') / 100.0, add_drop_pct=num('add_pct') / 100.0,
        )
        if p.entry_length < 5 or p.entry_threshold < 1 or p.max_positions > 50:
            raise ValueError('Please check the numbers (lookback at least 5 days, top N at least 1, at most 50 positions).')
        if p.exit_rule == 'none' and not (p.stop_adr_mult or p.trail_adr_mult or p.max_hold_days):
            raise ValueError('Without an exit signal you need a stop, a trailing stop or a maximum number of days.')
        return p
    ind = f.get('entry_indicator', 'rsi')
    if ind not in IND_KEYS:
        raise ValueError('Unknown indicator')
    shift = 100.0 if ind == 'willr' else 0.0
    p = StrategyParams(
        entry_indicator=ind,
        entry_length=int(num('entry_length')),
        entry_threshold=num('entry_level') + shift,
        entry_mode='cross' if f.get('entry_mode') == 'cross' else 'level',
        exit_rule=f.get('exit_rule', 'osc') if f.get('exit_rule') in ('osc', 'prev_high', 'none') else 'osc',
        exit_threshold=num('exit_level', 50) + shift,
        **trend_from(f.get('trend_ma')), market_ma=int(num('market_ma')),
        stop_adr_mult=num('stop_adr_mult'), trail_adr_mult=num('trail_adr_mult'),
        max_hold_days=int(num('max_hold_days')),
        breadth_min=num('breadth_pct') / 100.0,
        rank_by='volatility' if f.get('rank_by') == 'volatility' else 'oversold',
        max_positions=max(1, int(num('max_positions', 10))),
        min_price=num('min_price', 5),
        min_dollar_volume=num('min_dollar_volume_m', 5) * 1e6,
        universe=f.get('universe', '') if f.get('universe') in ('', 'sp100', 'nasdaq100', 'sp100+nasdaq100') else '',
        max_price=num('max_price'), max_new_per_day=int(num('max_new_per_day')),
        min_history_days=int(num('min_history_years') * 252),
        obe_min_gap=num('obe_pct') / 100.0, add_drop_pct=num('add_pct') / 100.0,
    )
    if p.entry_length < 0 or p.max_positions > 50 or p.max_hold_days < 0:
        raise ValueError('Please check the numbers (no negative values, at most 50 positions).')
    if p.exit_rule == 'none' and not (p.stop_adr_mult or p.trail_adr_mult or p.max_hold_days):
        raise ValueError('Without an exit signal you need a stop, a trailing stop or a maximum number of days.')
    return p


def _best_params_path(rel: str) -> Path | None:
    """Accept output/tuning_*/best_params.json (and nothing else outside strategies/)."""
    if not rel or 'best_params.json' not in rel:
        return None
    f = (ROOT_DIR / rel).resolve()
    if f.is_file() and f.name == 'best_params.json' and str(f).startswith(str(OUTPUT_DIR.resolve())):
        return f
    return None


def data_status() -> dict:
    """What the workbench shows under 'Price data'."""
    from backtest.data import DATA_DIR, load_universe
    out = {'has_data': False, 'strategies': [{'file': s['file'], 'name': s['name'], 'type': s.get('type', 'stocks')} for s in list_strategies()],
           'reports': list_reports(), 'options': {'has_data': False}}
    try:
        from backtest.spreads import load as load_options
        od = load_options()
        out['options'] = {'has_data': True, 'first': str(od['Date'].iloc[0].date()), 'last': str(od['Date'].iloc[-1].date()),
                          'has_rate': 'Rate' in od}
    except FileNotFoundError:
        pass
    try:
        from backtest.data import load_table
        prices = load_table('prices')
    except FileNotFoundError:
        return out
    out.update({'has_data': True, 'tickers': int(prices['Ticker'].nunique()),
                'first': str(prices['Date'].min().date()), 'last': str(prices['Date'].max().date())})
    m = load_membership('sp500')
    out['membership'] = m is not None
    if m is not None:
        try:
            from backtest.membership import coverage
            out['coverage'] = round(coverage(m, prices, '2016-01-01')['member_days_covered'] * 100, 1)
        except Exception:
            out['coverage'] = None
    return out


def list_strategies() -> list[dict]:
    out = []
    for path in sorted(STRATEGY_DIR.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            if data.get('type') == 'spreads':
                from backtest.spreads import SpreadParams
                sp = SpreadParams.load(path)
                out.append({'file': path.name, 'name': data.get('name') or path.stem, 'note': data.get('note', ''),
                            'type': 'spreads', 'params': sp.to_dict(), 'rules': sp.describe()})
                continue
            p = StrategyParams.from_dict(data.get('params', data))
            out.append({'file': path.name, 'name': data.get('name') or path.stem, 'note': data.get('note', ''),
                        'type': 'stocks', 'params': p.to_dict(), 'rules': p.describe()})
        except Exception:  # skip broken files instead of failing the page
            continue
    return out


def list_reports(limit: int = 12) -> list[dict]:
    files = sorted(list(OUTPUT_DIR.glob('report_*.html')) + list(OUTPUT_DIR.glob('compare_*.html')),
                   key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        m = re.fullmatch(r'(?:report_(.+)|(compare))_(\d{8})_(\d{6})', f.stem)
        label = (f'{(m.group(1) or "comparison of all").replace("_", " ")} · '
                 f'{datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S"):%d %b %Y %H:%M}' if m else f.stem)
        out.append({'file': f.name, 'label': label})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = 'StrategyPlanner/1.0'

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), 'application/json')

    def _body(self) -> dict:
        n = int(self.headers.get('Content-Length', 0) or 0)
        return json.loads(self.rfile.read(n) or b'{}') if n else {}

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == '/':
            page = PAGE.replace('{{INDICATORS}}', json.dumps([
                {'key': k, 'name': n, 'help': h, 'hasLength': hl, 'length': dl, 'lengthLabel': ll,
                 'entry': e, 'exit': x, 'scale': sc} for k, n, h, hl, dl, ll, e, x, sc in INDICATORS]))
            return self._send(200, page.encode(), 'text/html; charset=utf-8')
        if path == '/workbench':
            return self._send(200, WORKBENCH_PAGE.encode(), 'text/html; charset=utf-8')
        if path == '/api/state':
            return self._json({'strategies': list_strategies(), 'reports': list_reports()})
        if path == '/api/status':
            return self._json(data_status())
        if path == '/api/jobs':
            return self._json(jobs.list_jobs())
        if path.startswith('/text/'):
            rel = unquote(path[len('/text/'):])
            f = (ROOT_DIR / rel).resolve()
            if f.is_file() and f.suffix in ('.txt', '.log', '.csv', '.json') and str(f).startswith(str(OUTPUT_DIR.resolve())):
                return self._send(200, f.read_bytes(), 'text/plain; charset=utf-8')
        if path.startswith('/reports/'):
            name = Path(unquote(path[len('/reports/'):])).name
            f = OUTPUT_DIR / name
            if re.fullmatch(r'(report|compare)_[\w.-]+\.html', name) and f.is_file():
                return self._send(200, f.read_bytes(), 'text/html; charset=utf-8')
        self._send(404, b'Not found', 'text/plain')

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == '/api/describe':
                return self._json({'rules': params_from_form(body).describe()})
            if path == '/api/save':
                p = params_from_form(body)
                name = (body.get('name') or 'My trading plan').strip()[:80]
                file = STRATEGY_DIR / f'{slugify(name)}.json'
                file.write_text(json.dumps({'name': name, 'note': (body.get('note') or '').strip()[:500],
                                            'params': p.to_dict()}, indent=2) + '\n')
                return self._json({'file': file.name})
            if path == '/api/compare':
                from backtest.compare import build as build_compare
                files = [STRATEGY_DIR / Path(f).name for f in body.get('files', [])]
                files = [f for f in files if f.is_file()]
                if not files:
                    files = sorted(p for p in STRATEGY_DIR.glob('*.json') if p.stem != 'coach_plan')
                for f in files:  # warm the caches for every universe used
                    feature_store(StrategyParams.load(f).universe or 'sp500')
                out = build_compare(files, 'sp500', body.get('start') or '2016-01-01',
                                    body.get('holdout_start') or '2025-01-01', float(body.get('cost_bps') or 5))
                return self._json({'url': f'/reports/{out.name}'})
            if path == '/api/jobs':
                return self._json(jobs.start_job(body.get('kind', ''), body.get('opts') or {}))
            m = re.fullmatch(r'/api/jobs/([\w.-]+)/cancel', path)
            if m:
                return self._json({'ok': jobs.cancel_job(m.group(1))})
            if path == '/api/adopt':
                src = _best_params_path(body.get('file', ''))
                if src is None:
                    return self._json({'error': 'File not found.'}, 404)
                name = (body.get('name') or 'Best of search').strip()[:80]
                data = json.loads(src.read_text())
                p = StrategyParams.from_dict(data.get('params', data))
                out = STRATEGY_DIR / f'{slugify(name)}.json'
                out.write_text(json.dumps({'name': name, 'note': f'Adopted from {src.parent.name} (best in-sample set of a search; '
                                           f'judge it by the walk-forward result, not by the tuning period).',
                                           'params': p.to_dict(), 'trials': data.get('trials', 1)}, indent=2) + '\n')
                return self._json({'file': out.name})
            if path == '/api/run':
                file = _best_params_path(body.get('file', '')) or STRATEGY_DIR / Path(body.get('file', '')).name
                if not file.is_file():
                    return self._json({'error': 'Strategy file not found.'}, 404)
                from backtest.spreads import is_spread_file, build_report
                if is_spread_file(file):
                    out = build_report(file, body.get('start') or '2016-01-01', body.get('holdout_start') or '2025-01-01')
                    return self._json({'url': f'/reports/{out.name}'})
                from backtest.report import build  # imported lazily: keeps start-up fast
                uni = StrategyParams.load(file).universe or 'sp500'
                out = build(file, uni, body.get('start') or '2016-01-01',
                            body.get('holdout_start') or '2025-01-01', float(body.get('cost_bps') or 5),
                            fs=feature_store(uni), trials=max(1, int(body.get('trials') or 1)))
                return self._json({'url': f'/reports/{out.name}'})
        except FileNotFoundError as e:
            return self._json({'error': f'{e}'}, 400)
        except (ValueError, KeyError, TypeError) as e:
            return self._json({'error': str(e) or 'Invalid input.'}, 400)
        except Exception as e:  # show the problem in the page instead of a silent failure
            traceback.print_exc()
            return self._json({'error': f'Unexpected error: {e}'}, 500)
        self._send(404, b'Not found', 'text/plain')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-open', action='store_true')
    args = ap.parse_args()
    STRATEGY_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit('No free port found.')
    url = f'http://127.0.0.1:{server.server_port}/workbench'
    print(f'Strategy planner running at {url}\nStop it with Ctrl+C.')
    threading.Thread(target=lambda: feature_store('sp500'), daemon=True).start()  # warm up data in the background
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Planner</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f3f0; --surface: #fcfcfb; --border: #e3e2dd; --field: #ffffff;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --accent: #2a78d6; --accent-ink: #ffffff; --accent-soft: rgba(42,120,214,0.10);
  --bad-text: #a32222; --bad-soft: rgba(208,59,59,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2e2e2c; --field: #222221;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86;
    --accent: #3987e5; --accent-ink: #ffffff; --accent-soft: rgba(57,135,229,0.16);
    --bad-text: #f07a7a; --bad-soft: rgba(208,59,59,0.16);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 64px; }
header h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
header p { margin: 0; color: var(--text-2); }
.layout { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 18px; margin-top: 20px; align-items: start; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px; }
.card + .card { margin-top: 18px; }
.sticky { position: sticky; top: 16px; }
@media (max-width: 900px) { .sticky { position: static; } }
h2 { font-size: 18px; margin: 0 0 4px; }
.step { display: flex; align-items: center; gap: 10px; margin: 22px 0 10px; }
.step:first-of-type { margin-top: 8px; }
.step .n { width: 24px; height: 24px; border-radius: 50%; background: var(--accent-soft); color: var(--accent);
  font-weight: 700; font-size: 13px; display: grid; place-items: center; flex: none; }
.step h3 { margin: 0; font-size: 16px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 14px; }
@media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
label.f { display: flex; flex-direction: column; gap: 4px; font-size: 14px; font-weight: 600; }
label.f small, .hint { font-weight: 400; color: var(--text-3); font-size: 13px; }
input[type=text], input[type=number], select, textarea {
  font: inherit; font-weight: 400; color: var(--text); background: var(--field); border: 1px solid var(--border);
  border-radius: 9px; padding: 9px 11px; width: 100%; }
input:focus, select:focus, textarea:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: transparent; }
textarea { resize: vertical; min-height: 60px; }
.full { grid-column: 1 / -1; }
.seg { display: inline-flex; background: var(--bg); border-radius: 10px; padding: 3px; gap: 3px; flex-wrap: wrap; }
.seg label { cursor: pointer; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg span { display: block; padding: 7px 12px; border-radius: 8px; font-size: 14px; color: var(--text-2); }
.seg input:checked + span { background: var(--surface); color: var(--text); font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.seg input:focus-visible + span { outline: 2px solid var(--accent); }
details { margin-top: 18px; border-top: 1px solid var(--border); padding-top: 12px; }
summary { cursor: pointer; font-weight: 600; color: var(--text-2); }
.rules { white-space: pre-wrap; background: var(--bg); border-radius: 10px; padding: 14px 16px; font-size: 14.5px;
  line-height: 1.65; min-height: 120px; margin: 10px 0 14px; }
.btn { font: inherit; font-weight: 650; border: 0; border-radius: 10px; padding: 12px 18px; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); width: 100%; font-size: 16px; }
.btn:disabled { opacity: .6; cursor: progress; }
.btn.ghost { background: var(--bg); color: var(--text); width: auto; padding: 8px 12px; font-size: 14px; font-weight: 600; }
.msg { margin-top: 12px; font-size: 14px; border-radius: 9px; padding: 10px 12px; display: none; }
.msg.err { display: block; background: var(--bad-soft); color: var(--bad-text); }
.msg.info { display: block; background: var(--accent-soft); color: var(--text); }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--accent); border-right-color: transparent;
  border-radius: 50%; animation: spin .8s linear infinite; vertical-align: -2px; margin-right: 8px; }
@keyframes spin { to { transform: rotate(360deg); } }
ul.list { list-style: none; padding: 0; margin: 8px 0 0; }
ul.list li { border-top: 1px solid var(--border); padding: 10px 0; display: flex; gap: 10px; align-items: center; justify-content: space-between; }
ul.list .t { font-weight: 600; font-size: 15px; }
ul.list .s { color: var(--text-3); font-size: 13px; }
ul.list .acts { display: flex; gap: 6px; flex: none; }
a { color: var(--accent); }
.empty { color: var(--text-3); font-size: 14px; padding: 8px 0; }
</style>
</head>
<body>
<main>
<nav style="display:flex;gap:18px;margin-bottom:6px;font-size:14px"><a href="/workbench" style="color:var(--accent);text-decoration:none;font-weight:600">Workbench</a><a href="/" style="color:var(--text);text-decoration:none;font-weight:600">Planner</a></nav>
<header>
  <h1>Strategy Planner</h1>
  <p>Describe your trading plan, then test it on 10 years of S&amp;P 500 stocks against simply holding the index.</p>
</header>

<div class="layout">
<form class="card" id="form" autocomplete="off">
  <div class="step"><span class="n">1</span><h3>Name</h3></div>
  <div class="grid">
    <label class="f full">Plan name<input type="text" name="name" value="My trading plan" maxlength="80"></label>
  </div>

  <div class="step"><span class="n">2</span><h3>When to buy</h3></div>
  <div class="grid">
    <div class="full">
      <div class="hint" style="margin-bottom:6px">Strategy type</div>
      <div class="seg">
        <label><input type="radio" name="family" value="reversion" checked><span>Buy the dip (oversold indicator)</span></label>
        <label><input type="radio" name="family" value="momentum"><span>Follow the trend (momentum / breakout)</span></label>
      </div>
    </div>
  </div>
  <div class="grid" id="momBlock" style="display:none">
    <label class="f full">Entry<select name="mom_entry" id="momEntry">
      <option value="rs">Relative strength: among the strongest stocks</option>
      <option value="breakout">Breakout: new N-day high</option></select><small id="momHelp"></small></label>
    <label class="f"><span id="lookbackLabel">Performance over the last</span><select name="mom_lookback">
      <option value="63">3 months</option><option value="126" selected>6 months</option>
      <option value="189">9 months</option><option value="252">12 months</option>
      <option value="50">50 days</option><option value="100">100 days</option><option value="20">20 days</option></select></label>
    <label class="f" id="topWrap">Buy if among the top<select name="mom_top">
      <option value="5">5 stocks</option><option value="10">10 stocks</option><option value="20" selected>20 stocks</option>
      <option value="30">30 stocks</option><option value="50">50 stocks</option></select></label>
    <label class="f" id="skipWrap">Ignore the most recent<select name="skip_days">
      <option value="0" selected>nothing</option><option value="10">2 weeks</option><option value="21">1 month</option></select>
      <small>classic "12-1 momentum" skips the last month</small></label>
    <label class="f" id="rebWrap">Check and rebalance<select name="rebalance">
      <option value="monthly" selected>monthly</option><option value="weekly">weekly</option><option value="daily">daily</option></select></label>
  </div>
  <div class="grid" id="oscBlock">
    <label class="f full">Indicator<select name="entry_indicator" id="ind"></select><small id="indHelp"></small></label>
    <label class="f" id="lenWrap"><span id="lenLabel">Length (days)</span><input type="number" name="entry_length" min="1" step="1"></label>
    <label class="f">Buy when it is below<input type="number" name="entry_level" step="any"><small id="scaleHint"></small></label>
    <div class="full">
      <div class="hint" style="margin-bottom:6px">Which day?</div>
      <div class="seg">
        <label><input type="radio" name="entry_mode" value="cross" checked><span>First day it gets there</span></label>
        <label><input type="radio" name="entry_mode" value="level"><span>Any day it is there</span></label>
      </div>
    </div>
  </div>

  <div class="step"><span class="n">3</span><h3>Filters</h3></div>
  <div class="grid">
    <label class="f">Stock trend filter<select name="trend_ma">
      <option value="0">No filter</option><option value="50">Above its 50-day average</option>
      <option value="100">Above its 100-day average</option><option value="200" selected>Above its 200-day average</option>
      <option value="200:-0.10">At least 10% below its 200-day average</option><option value="200:-0.15">At least 15% below its 200-day average</option>
      <option value="200:-0.20">At least 20% below its 200-day average</option><option value="200:-0.30">At least 30% below its 200-day average</option></select></label>
    <label class="f">Market filter (SPY)<select name="market_ma">
      <option value="0">No filter</option><option value="50">SPY above its 50-day average</option>
      <option value="200" selected>SPY above its 200-day average</option></select></label>
  </div>

  <div class="step"><span class="n">4</span><h3>When to sell</h3></div>
  <div class="grid" id="momExitBlock" style="display:none">
    <label class="f">Exit signal<select name="mom_exit" id="momExit">
      <option value="rank" selected>Drops out of the strongest stocks</option>
      <option value="ma">Close below a moving average</option>
      <option value="channel">Close below the lowest low of N days</option>
      <option value="none">No signal (stops / time only)</option></select></label>
    <label class="f" id="momExitRankWrap">Sell when no longer in the top<select name="mom_exit_rank">
      <option value="20">20</option><option value="30">30</option><option value="50" selected>50</option><option value="100">100</option></select></label>
    <label class="f" id="momExitLenWrap" style="display:none">Length (days)<select name="mom_exit_length">
      <option value="20">20</option><option value="50">50</option><option value="100" selected>100</option><option value="200">200</option></select></label>
  </div>
  <div class="grid">
    <label class="f" id="oscExitRule">Exit signal<select name="exit_rule" id="exitRule">
      <option value="osc">Indicator rises above a level</option>
      <option value="prev_high">Close above yesterday's high</option>
      <option value="none">No signal (stops / time only)</option></select></label>
    <label class="f" id="exitWrap">Sell when it rises above<input type="number" name="exit_level" step="any"></label>
    <label class="f">Stop loss<select name="stop_adr_mult">
      <option value="0">None</option><option value="2">2 × average daily range</option>
      <option value="3" selected>3 × average daily range</option><option value="4">4 × average daily range</option>
      <option value="5">5 × average daily range</option></select></label>
    <label class="f">Trailing stop<select name="trail_adr_mult">
      <option value="0" selected>None</option><option value="2">2 × average daily range</option>
      <option value="3">3 × average daily range</option><option value="4">4 × average daily range</option></select></label>
    <label class="f">Sell after at most<select name="max_hold_days">
      <option value="0">No limit</option><option value="5">5 trading days</option>
      <option value="10" selected>10 trading days</option><option value="20">20 trading days</option>
      <option value="40">40 trading days</option><option value="60">60 trading days</option>
      <option value="120">120 trading days</option><option value="250">250 trading days</option></select></label>
  </div>

  <div class="step"><span class="n">5</span><h3>Positions</h3></div>
  <div class="grid">
    <label class="f">Positions at the same time<select name="max_positions">
      <option value="3">3 (33% each)</option><option value="5">5 (20% each)</option>
      <option value="10" selected>10 (10% each)</option><option value="20">20 (5% each)</option></select></label>
    <label class="f" id="rankWrap">More signals than free spots? Take<select name="rank_by">
      <option value="oversold" selected>the most oversold</option><option value="volatility">the most volatile</option></select></label>
  </div>

  <details>
    <summary>More options</summary>
    <div class="grid" style="margin-top:12px">
      <label class="f">Only on broad pullbacks<select name="breadth_pct">
        <option value="0" selected>Off</option><option value="5">at least 5% of stocks oversold</option>
        <option value="10">at least 10% of stocks oversold</option><option value="20">at least 20% of stocks oversold</option></select></label>
      <label class="f">Trading costs per buy / sell<select name="cost_bps">
        <option value="2">0.02%</option><option value="5" selected>0.05%</option><option value="10">0.10%</option><option value="20">0.20%</option></select></label>
      <label class="f">Minimum stock price ($)<input type="number" name="min_price" value="5" min="0" step="any"></label>
      <label class="f">Min. daily trading value ($ millions)<input type="number" name="min_dollar_volume_m" value="5" min="0" step="any"></label>
      <label class="f">Universe<select name="universe">
        <option value="" selected>S&P 500 (point-in-time members)</option><option value="sp100">S&P 100</option>
        <option value="nasdaq100">Nasdaq-100</option><option value="sp100+nasdaq100">S&P 100 + Nasdaq-100</option></select></label>
      <label class="f">Maximum stock price ($)<input type="number" name="max_price" value="0" min="0" step="any"><small>0 = no limit</small></label>
      <label class="f">Max. new positions per day<input type="number" name="max_new_per_day" value="0" min="0" step="1"><small>0 = no limit</small></label>
      <label class="f">Min. price history (years)<select name="min_history_years">
        <option value="0" selected>none</option><option value="1">1 year</option><option value="3">3 years</option><option value="5">5 years</option></select></label>
      <label class="f">OBE entries only<select name="obe_pct">
        <option value="0" selected>off</option><option value="2">open at least 2% below the signal close</option>
        <option value="3">open at least 3% below the signal close</option>
        <option value="4">open at least 4% below the signal close</option></select></label>
      <label class="f">Add a second block<select name="add_pct">
        <option value="0" selected>off</option><option value="5">if the price falls 5% below entry</option>
        <option value="10">if the price falls 10% below entry</option></select></label>
      <label class="f">Variants tried before this one<input type="number" name="trials" value="1" min="1" step="1"><small>be honest: each variant you tried raises the bar for "not luck"</small></label>
      <label class="f">Backtest from<input type="text" name="start" value="2016-01-01"><small>YYYY-MM-DD</small></label>
      <label class="f">Test period from<input type="text" name="holdout_start" value="2025-01-01"><small>never used to pick rules</small></label>
      <label class="f full">Notes (optional)<textarea name="note" maxlength="500" placeholder="Where does this plan come from?"></textarea></label>
    </div>
  </details>
</form>

<div>
  <div class="card sticky">
    <h2>Your rules</h2>
    <div class="rules" id="rules">…</div>
    <button class="btn" id="run" type="button">Save &amp; run backtest</button>
    <div class="msg" id="msg" role="status"></div>
    <p class="hint" style="margin:12px 0 0">Decide the rules first, test once. Tweaking numbers until the result looks good
      makes every plan look great on the past – and fail with real money.</p>
  </div>
  <div class="card">
    <h2>Saved plans</h2>
    <ul class="list" id="plans"></ul>
    <button class="btn ghost" id="compare" type="button" style="margin-top:10px">Compare all plans with SPY</button>
  </div>
  <div class="card">
    <h2>Recent reports</h2>
    <ul class="list" id="reports"></ul>
  </div>
</div>
</div>
</main>
<script>
const IND = {{INDICATORS}};
const form = document.getElementById('form');
const $ = id => document.getElementById(id);
const sel = $('ind');
IND.forEach(i => sel.add(new Option(i.name, i.key)));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

function values() {
  const d = Object.fromEntries(new FormData(form).entries());
  return d;
}
function applyIndicator(setDefaults) {
  const i = IND.find(x => x.key === sel.value);
  $('indHelp').textContent = i.help;
  $('scaleHint').textContent = i.scale;
  $('lenWrap').style.display = i.hasLength ? '' : 'none';
  $('lenLabel').textContent = i.lengthLabel || 'Length';
  if (setDefaults) {
    form.entry_length.value = i.length;
    form.entry_level.value = i.entry;
    form.exit_level.value = i.exit;
  }
}
function applyExit() { $('exitWrap').style.display = $('exitRule').value === 'osc' ? '' : 'none'; }
const MOM_HELP = {rs: 'The stock is among the strongest N of all S&P 500 stocks by performance over the lookback.',
                  breakout: 'The close is above the highest high of the previous N days (use the day lengths).'};
function applyFamily() {
  const mom = form.family.value === 'momentum';
  $('momBlock').style.display = mom ? '' : 'none';
  $('oscBlock').style.display = mom ? 'none' : '';
  $('momExitBlock').style.display = mom ? '' : 'none';
  $('oscExitRule').style.display = mom ? 'none' : '';
  $('exitWrap').style.display = (!mom && $('exitRule').value === 'osc') ? '' : 'none';
  $('rankWrap').style.display = mom ? 'none' : '';
  const entry = $('momEntry').value, isRs = entry === 'rs';
  $('momHelp').textContent = MOM_HELP[entry];
  $('lookbackLabel').textContent = isRs ? 'Performance over the last' : 'Highest high of the previous';
  $('topWrap').style.display = isRs ? '' : 'none';
  $('skipWrap').style.display = isRs ? '' : 'none';
  $('rebWrap').style.display = isRs ? '' : 'none';
  const rankOpt = $('momExit').querySelector('option[value=rank]');
  rankOpt.disabled = !isRs; if (!isRs && $('momExit').value === 'rank') $('momExit').value = 'ma';
  const x = $('momExit').value;
  $('momExitRankWrap').style.display = x === 'rank' ? '' : 'none';
  $('momExitLenWrap').style.display = (x === 'ma' || x === 'channel') ? '' : 'none';
}

let t;
async function preview() {
  clearTimeout(t);
  t = setTimeout(async () => {
    try {
      const r = await fetch('/api/describe', {method: 'POST', body: JSON.stringify(values())});
      const j = await r.json();
      $('rules').textContent = j.rules || j.error;
    } catch (e) { $('rules').textContent = 'The planner stopped – restart it in the Terminal.'; }
  }, 150);
}
function msg(text, kind, busy) {
  const m = $('msg'); m.className = 'msg ' + kind;
  m.innerHTML = (busy ? '<span class="spinner"></span>' : '') + esc(text);
}

async function runFile(file, extra) {
  $('run').disabled = true;
  msg('Running the backtest … this takes about 30–60 seconds (longer the first time).', 'info', true);
  try {
    const v = values();
    const r = await fetch('/api/run', {method: 'POST', body: JSON.stringify(Object.assign({file,
      start: v.start, holdout_start: v.holdout_start, cost_bps: v.cost_bps, trials: v.trials}, extra || {}))});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    msg('Done – opening the report.', 'info');
    loadState();
    window.location.href = j.url;
  } catch (e) { msg(e.message, 'err'); }
  finally { $('run').disabled = false; }
}

$('compare').addEventListener('click', async () => {
  $('compare').disabled = true;
  msg('Comparing all saved plans … about a minute per plan the first time.', 'info', true);
  try {
    const v = values();
    const r = await fetch('/api/compare', {method: 'POST', body: JSON.stringify({start: v.start, holdout_start: v.holdout_start, cost_bps: v.cost_bps})});
    const j = await r.json(); if (!r.ok) throw new Error(j.error);
    loadState(); window.location.href = j.url;
  } catch (e) { msg(e.message, 'err'); } finally { $('compare').disabled = false; }
});
$('run').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/save', {method: 'POST', body: JSON.stringify(values())});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    await runFile(j.file);
  } catch (e) { msg(e.message, 'err'); }
});

function loadPlan(s) {
  const p = s.params, shift = p.entry_indicator === 'willr' ? 100 : 0;
  form.elements['name'].value = s.name; form.elements['note'].value = s.note || '';
  const setSel = (name, v) => { const s = form[name]; if (![...s.options].some(o => +o.value === +v || o.value === String(v)))
    s.add(new Option(String(v), v)); s.value = String(v); };
  const mom = p.entry_indicator === 'rs' || p.entry_indicator === 'breakout';
  form.querySelector(`input[name=family][value=${mom ? 'momentum' : 'reversion'}]`).checked = true;
  if (mom) {
    $('momEntry').value = p.entry_indicator; setSel('mom_lookback', p.entry_length); setSel('mom_top', p.entry_threshold);
    setSel('skip_days', p.skip_days); setSel('rebalance', p.rebalance);
    $('momExit').value = ['rank', 'ma', 'channel', 'none'].includes(p.exit_rule) ? p.exit_rule : 'none';
    setSel('mom_exit_rank', p.exit_threshold); setSel('mom_exit_length', p.exit_length || p.entry_length);
  } else {
    sel.value = p.entry_indicator; applyIndicator(false);
    form.entry_length.value = p.entry_length;
    form.entry_level.value = +(p.entry_threshold - shift).toFixed(4);
    form.exit_level.value = +(p.exit_threshold - shift).toFixed(4);
    form.querySelector(`input[name=entry_mode][value=${p.entry_mode}]`).checked = true;
    form.exit_rule.value = ['osc', 'prev_high', 'none'].includes(p.exit_rule) ? p.exit_rule : 'osc';
    form.rank_by.value = p.rank_by === 'volatility' ? 'volatility' : 'oversold';
  }
  setSel('trend_ma', p.trend_below ? `${p.trend_ma}:-${p.trend_below.toFixed(2)}` : p.trend_ma); setSel('market_ma', p.market_ma); setSel('stop_adr_mult', p.stop_adr_mult);
  setSel('trail_adr_mult', p.trail_adr_mult); setSel('max_hold_days', p.max_hold_days);
  setSel('max_positions', p.max_positions); setSel('breadth_pct', Math.round(p.breadth_min * 100));
  form.min_price.value = p.min_price; form.min_dollar_volume_m.value = p.min_dollar_volume / 1e6;
  form.universe.value = p.universe || ''; form.max_price.value = p.max_price || 0; form.max_new_per_day.value = p.max_new_per_day || 0;
  setSel('min_history_years', Math.round((p.min_history_days || 0) / 252)); setSel('obe_pct', Math.round((p.obe_min_gap || 0) * 100));
  setSel('add_pct', Math.round((p.add_drop_pct || 0) * 100));
  applyExit(); applyFamily(); preview(); window.scrollTo({top: 0, behavior: 'smooth'});
}

async function loadState() {
  const j = await (await fetch('/api/state')).json();
  const plans = $('plans'); plans.innerHTML = '';
  if (!j.strategies.length) plans.innerHTML = '<li class="empty">No saved plans yet.</li>';
  j.strategies.forEach(s => {
    const li = document.createElement('li');
    li.innerHTML = `<div><div class="t">${esc(s.name)}</div><div class="s">${esc(s.file)}</div></div>
      <div class="acts">${s.type === 'spreads' ? '' : '<button class="btn ghost" data-a="edit">Edit</button>'}<button class="btn ghost" data-a="run">Report</button></div>`;
    if (s.type !== 'spreads') li.querySelector('[data-a=edit]').onclick = () => loadPlan(s);
    li.querySelector('[data-a=run]').onclick = () => runFile(s.file);
    plans.appendChild(li);
  });
  const reps = $('reports'); reps.innerHTML = '';
  if (!j.reports.length) reps.innerHTML = '<li class="empty">No reports yet.</li>';
  j.reports.forEach(r => {
    const li = document.createElement('li');
    li.innerHTML = `<a href="/reports/${encodeURIComponent(r.file)}">${esc(r.label)}</a>`;
    reps.appendChild(li);
  });
}

sel.addEventListener('change', () => { applyIndicator(true); preview(); });
$('exitRule').addEventListener('change', () => { applyExit(); preview(); });
form.querySelectorAll('input[name=family]').forEach(r => r.addEventListener('change', () => {
  const mom = form.family.value === 'momentum';   // sensible defaults per family
  form.max_hold_days.value = mom ? '0' : '10'; form.stop_adr_mult.value = mom ? '0' : '3';
  applyFamily(); preview(); }));
$('momEntry').addEventListener('change', () => { applyFamily(); preview(); });
$('momExit').addEventListener('change', () => { applyFamily(); preview(); });
form.addEventListener('input', preview);
form.addEventListener('change', preview);
applyIndicator(true); applyExit(); applyFamily(); preview(); loadState();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
