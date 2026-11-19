"""Compare several rule sets with each other and with SPY on one page.

    python3 -m backtest.compare                                   # every file in strategies/
    python3 -m backtest.compare --params strategies/willr.json strategies/best_strategy.json strategies/rsi4_*.json
    python3 -m backtest.compare --start 2016-01-01 --holdout-start 2025-01-01 --no-open

Each strategy is backtested on its own universe (the 'universe' in its file, else --index).
The page shows: growth of $10,000 for all strategies + SPY buy & hold + SPY timing, a metrics table for
the full period and the test period, how many verdict checks each passes, year-by-year returns, and
how similar the strategies are to each other (correlation of daily returns).
Writes output/compare_<timestamp>.html.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import criteria
from backtest.benchmarks import benchmark_metrics, spy_buy_hold, spy_timing
from backtest.data import OUTPUT_DIR, ROOT_DIR, load_benchmark, load_prices
from backtest.engine import generate_trades, simulate_portfolio
from backtest.features import FeatureStore
from backtest.membership import load_membership
from backtest.metrics import compute_metrics, yearly_returns
from backtest.significance import luck_summary
from backtest.strategy import StrategyParams
from backtest.spreads import is_spread_file

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300', '#e34948']
COLORS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#008300', '#e66767']


def _pct(x: float, d: int = 1) -> str:
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x * 100:.{d}f}%'


def run_one(path: Path, default_index: str, start: str, holdout_start: str, cost: float,
            stores: dict) -> dict:
    if is_spread_file(path):
        from backtest.spreads import run_one as run_spreads
        return run_spreads(path, start, holdout_start)
    p = StrategyParams.load(path)
    meta = json.loads(path.read_text())
    index = p.universe or default_index
    if index not in stores:
        stores[index] = FeatureStore(load_prices(None if index == 'all' else index), load_benchmark('SPY'),
                                     load_membership(index))
    fs = stores[index]
    trades = generate_trades(fs, p)
    kw = dict(max_positions=p.max_positions, max_new_per_day=p.max_new_per_day, cost_per_side=cost)
    full = simulate_portfolio(fs, trades, start, None, **kw)
    hold = simulate_portfolio(fs, trades, holdout_start, None, **kw)
    stress = simulate_portfolio(fs, trades, start, None, **{**kw, 'cost_per_side': criteria.STRESS_COST_BPS / 1e4})
    fm = compute_metrics(full.equity, full.trades, full.exposure)
    hm = compute_metrics(hold.equity, hold.trades, hold.exposure)
    sm = compute_metrics(stress.equity, stress.trades, stress.exposure)
    bench = load_benchmark('SPY')
    fb = benchmark_metrics(bench, full.equity.index, cost)
    hb = benchmark_metrics(bench, hold.equity.index, cost)
    trials = max(1, int(meta.get('trials', 1) or 1))
    luck = luck_summary(full.equity, spy_buy_hold(bench, full.equity.index), trials)
    verdict = criteria.evaluate(fm, hm, sm, fb['SPY buy & hold'], hb['SPY buy & hold'], hb['SPY timing (>SMA200)'], luck)
    return {'file': path.name, 'name': meta.get('name') or path.stem, 'index': index, 'params': p,
            'equity': full.equity, 'full': fm, 'holdout': hm, 'passed': sum(ok for _, ok, _ in verdict),
            'checks': len(verdict), 'luck': min(luck['dsr']['prob'], luck['bootstrap']['p_positive']),
            'trials': trials}


def build(paths: list[Path], default_index: str, start: str, holdout_start: str, cost_bps: float) -> Path:
    stores: dict = {}
    runs = [run_one(p, default_index, start, holdout_start, cost_bps / 1e4, stores) for p in paths]
    bench = load_benchmark('SPY')
    idx = runs[0]['equity'].index
    for r in runs:  # common calendar (universes can differ slightly)
        idx = idx.intersection(r['equity'].index)
    curves = {r['name']: r['equity'].reindex(idx).ffill() for r in runs}
    curves = {k: v / v.iloc[0] * 10_000 for k, v in curves.items()}
    curves['SPY buy & hold'] = spy_buy_hold(bench, idx, 10_000)
    curves['SPY timing'] = spy_timing(bench, idx, cost_per_side=cost_bps / 1e4, capital=10_000)
    names = list(curves)
    short = lambda n: n if len(n) <= 26 else n[:24].rstrip() + '…'
    weekly = pd.DataFrame(curves).resample('W-FRI').last().dropna()
    dd = weekly / weekly.cummax() - 1
    yr = pd.DataFrame({k: yearly_returns(v) for k, v in curves.items()})
    rets = pd.DataFrame({r['name']: r['equity'].reindex(idx).ffill().pct_change() for r in runs}).dropna()
    corr = rets.corr() if len(runs) > 1 else None
    hold_idx = idx[idx >= pd.Timestamp(holdout_start)]
    bm_full = benchmark_metrics(bench, idx, cost_bps / 1e4)
    bm_hold = benchmark_metrics(bench, hold_idx, cost_bps / 1e4)

    def row(label_html, fm, hm, extra=''):
        return (f'<tr><th scope="row">{label_html}</th>'
                f'<td>{_pct(fm["cagr"])}</td><td>{_pct(fm["max_drawdown"])}</td><td>{fm["sharpe"]:.2f}</td><td>{fm["calmar"]:.2f}</td>'
                f'<td class="sep">{_pct(hm["cagr"])}</td><td>{_pct(hm["max_drawdown"])}</td><td>{hm["sharpe"]:.2f}</td>'
                f'{extra}</tr>')
    rows = ''
    for i, r in enumerate(runs):
        extra = (f'<td class="sep">{r["full"]["trades"]:,}</td><td>{_pct(r["full"]["win_rate"], 0)}</td>'
                 f'<td class="{"ok" if r["passed"] == r["checks"] else "bad"}">{r["passed"]} / {r["checks"]}</td>'
                 f'<td>{_pct(r["luck"], 0)}</td>')
        rows += row(f'<i class="sw" style="--c:{COLORS[i % 8]}"></i>{html.escape(r["name"])}', r['full'], r['holdout'], extra)
    empty = '<td class="sep"></td><td></td><td></td><td></td>'
    rows += row('<i class="sw" style="--c:var(--spy)"></i>SPY buy &amp; hold', bm_full['SPY buy & hold'], bm_hold['SPY buy & hold'], empty)
    rows += row('<i class="sw" style="--c:var(--timing)"></i>SPY timing', bm_full['SPY timing (>SMA200)'], bm_hold['SPY timing (>SMA200)'], empty)

    rule_blocks = ''.join(
        f'<details><summary><i class="sw" style="--c:{COLORS[i % 8]}"></i>{html.escape(r["name"])} '
        f'<span class="muted">· {html.escape(r["file"])} · {html.escape(r["index"])}</span></summary>'
        f'<pre>{html.escape(r["params"].describe())}</pre></details>' for i, r in enumerate(runs))
    yearly_rows = ''.join(
        f'<tr><th scope="row">{y}</th>' + ''.join(f'<td>{_pct(yr.loc[y, n])}</td>' for n in names) + '</tr>'
        for y in yr.index)
    yearly_head = ''.join(f'<th>{html.escape(short(n))}</th>' for n in names)
    corr_html = ''
    if corr is not None:
        sn = [r['name'] for r in runs]
        corr_html = ('<section><h2>How similar are the strategies?</h2><p class="lead">Correlation of daily returns. '
                     'Near 1 = they move together (little diversification); near 0 = independent.</p><div class="tbl"><table>'
                     '<thead><tr><th></th>' + ''.join(f'<th>{html.escape(short(n))}</th>' for n in sn) + '</tr></thead><tbody>'
                     + ''.join(f'<tr><th scope="row">{html.escape(short(a))}</th>' + ''.join(f'<td>{corr.loc[a, b]:.2f}</td>' for b in sn) + '</tr>' for a in sn)
                     + '</tbody></table></div></section>')
    data = {
        'dates': [d.strftime('%Y-%m-%d') for d in weekly.index],
        'names': names,
        'equity': {n: [round(v, 2) for v in weekly[n]] for n in names},
        'drawdown': {n: [round(v * 100, 2) for v in dd[n]] for n in names},
        'colors': [COLORS[i % 8] for i in range(len(runs))],
        'colorsDark': [COLORS_DARK[i % 8] for i in range(len(runs))],
        'holdout': holdout_start,
    }
    page = TEMPLATE
    for k, v in {
        '{{PERIOD}}': f'{idx[0]:%d %b %Y} – {idx[-1]:%d %b %Y}',
        '{{HOLDOUT}}': f'{pd.Timestamp(holdout_start):%b %Y}',
        '{{COST}}': f'{cost_bps:g}',
        '{{ROWS}}': rows, '{{RULES}}': rule_blocks,
        '{{YEARLY_HEAD}}': yearly_head, '{{YEARLY_ROWS}}': yearly_rows,
        '{{CORR}}': corr_html,
        '{{LEGEND}}': ''.join(f'<span style="--c:{COLORS[i % 8]}">{html.escape(r["name"])}</span>' for i, r in enumerate(runs))
                      + '<span style="--c:var(--spy)">SPY buy &amp; hold</span><span style="--c:var(--timing)">SPY timing</span>',
        '{{DATA}}': json.dumps(data),
        '{{GENERATED}}': f'{datetime.now():%d %b %Y %H:%M}',
    }.items():
        page = page.replace(k, v)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f'compare_{datetime.now():%Y%m%d_%H%M%S}.html'
    out.write_text(page, encoding='utf-8')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', nargs='*', help='strategy files (default: all in strategies/ except the template)')
    ap.add_argument('--index', default='sp500', help="default universe for files without a 'universe'")
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--holdout-start', default='2025-01-01')
    ap.add_argument('--cost-bps', type=float, default=5.0)
    ap.add_argument('--no-open', action='store_true')
    args = ap.parse_args()
    paths = [Path(p) for p in args.params] if args.params else \
        sorted(p for p in (ROOT_DIR / 'strategies').glob('*.json') if p.stem != 'coach_plan')
    if not paths:
        raise SystemExit('No strategy files found.')
    print(f'Comparing {len(paths)} strategies ... (about a minute each the first time)')
    out = build(paths, args.index, args.start, args.holdout_start, args.cost_bps)
    print(f'Report saved: {out.relative_to(ROOT_DIR)}')
    if not args.no_open:
        import webbrowser
        webbrowser.open(out.as_uri())


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Comparison</title>
<style>
:root { color-scheme: light; --bg:#f4f3f0; --surface:#fcfcfb; --border:#e3e2dd; --text:#0b0b0b; --text-2:#52514e; --text-3:#7a7974;
  --grid:#e9e8e4; --shade:rgba(42,120,214,0.06); --spy:#52514e; --timing:#9a9890; --good:#006300; --bad:#a32222; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#121211; --surface:#1a1a19; --border:#2e2e2c;
  --text:#f0efec; --text-2:#c3c2b7; --text-3:#8f8e86; --grid:#2a2a28; --shade:rgba(57,135,229,0.10); --spy:#c3c2b7; --timing:#8f8e86; --good:#4cc94c; --bad:#f07a7a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:30px; margin:0 0 4px; letter-spacing:-0.01em; } .sub { color:var(--text-2); margin:0; }
section { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:22px 24px; margin-top:18px; }
h2 { font-size:19px; margin:0 0 12px; } p.lead { color:var(--text-2); margin:-4px 0 14px; font-size:15px; }
.tbl { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:7px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } thead th { color:var(--text-2); font-weight:600; } tbody th { font-weight:500; }
td.sep, th.sep { border-left:1px solid var(--border); } td.ok { color:var(--good); font-weight:600; } td.bad { color:var(--bad); font-weight:600; }
.sw { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:8px; background:var(--c); }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:14px; color:var(--text-2); margin:0 0 8px; }
.legend span::before { content:""; display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; background:var(--c); }
.chart { position:relative; } .chart svg { display:block; width:100%; height:auto; overflow:visible; }
.chart text { font-size:12px; fill:var(--text-3); font-family:inherit; } .chart .lbl { font-size:12px; font-weight:600; fill:var(--text-2); }
.tip { position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:13px;
  box-shadow:0 4px 16px rgba(0,0,0,.12); opacity:0; transition:opacity .1s; white-space:nowrap; z-index:2; }
.tip b { display:block; margin-bottom:2px; } .tip .r { display:flex; justify-content:space-between; gap:14px; color:var(--text-2); }
.tip .r i { display:inline-block; width:10px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; }
details { border-top:1px solid var(--border); padding:8px 0; } summary { cursor:pointer; font-weight:600; } .muted { color:var(--text-3); font-weight:400; font-size:13px; }
pre { white-space:pre-wrap; font:14px/1.6 inherit; background:var(--bg); border-radius:10px; padding:12px 14px; margin:8px 0 4px; }
footer { color:var(--text-3); font-size:13px; margin-top:18px; text-align:center; }
@media (max-width:720px) { section { padding:18px 16px; } th, td { padding:6px 6px; font-size:13px; } h1 { font-size:24px; } }
</style>
</head>
<body>
<main>
<h1>Strategy Comparison</h1>
<p class="sub">{{PERIOD}} · test period from {{HOLDOUT}} (never used to pick rules) · costs {{COST}} bps per trade side</p>

<section>
  <h2>Results</h2>
  <p class="lead">Left: whole period. Middle: test period only. Right: trades, win rate, verdict checks passed, probability the edge over SPY is real.</p>
  <div class="tbl"><table>
    <thead><tr><th></th><th>Return/yr</th><th>Worst drop</th><th>Sharpe</th><th>Calmar</th>
      <th class="sep">Test ret/yr</th><th>Test drop</th><th>Test Sharpe</th>
      <th class="sep">Trades</th><th>Win</th><th>Verdict</th><th>Not luck</th></tr></thead>
    <tbody>{{ROWS}}</tbody></table></div>
</section>

<section>
  <h2>Growth of $10,000</h2>
  <p class="lead">Shaded area = test period. Hover for values.</p>
  <div class="legend">{{LEGEND}}</div>
  <div class="chart" id="equity"></div>
</section>

<section>
  <h2>Drops from the previous high</h2>
  <div class="legend">{{LEGEND}}</div>
  <div class="chart" id="drawdown"></div>
</section>

<section>
  <h2>Year by year</h2>
  <div class="tbl"><table><thead><tr><th>Year</th>{{YEARLY_HEAD}}</tr></thead><tbody>{{YEARLY_ROWS}}</tbody></table></div>
</section>

{{CORR}}

<section>
  <h2>The rules</h2>
  {{RULES}}
</section>

<footer>Generated {{GENERATED}} · StrategyAnalyser · Past results do not guarantee future results.</footer>
</main>
<div class="tip" id="tip"></div>
<script>
const D = {{DATA}};
const NS = 'http://www.w3.org/2000/svg';
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const dark = () => window.matchMedia('(prefers-color-scheme: dark)').matches;
const color = k => { const n = D.names.length - 2; if (k < n) return (dark() ? D.colorsDark : D.colors)[k]; return k === n ? css('--spy') : css('--timing'); };
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const tip = document.getElementById('tip');
const fmtMoney = v => '$' + Math.round(v).toLocaleString('en-US');
function showTip(evt, title, rows) {
  tip.innerHTML = '<b>' + title + '</b>' + rows.map(r => '<div class="r"><span><i style="background:' + r.c + '"></i>' + r.n + '</span><span>' + r.v + '</span></div>').join('');
  tip.style.opacity = 1; const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.pageX + pad, y = evt.pageY - h - pad;
  if (x + w > window.scrollX + document.documentElement.clientWidth - 8) x = evt.pageX - w - pad;
  if (y < window.scrollY + 8) y = evt.pageY + pad; tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.style.opacity = 0; };
function niceTicks(lo, hi, n) { const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v); return out; }
function lineChart(id, key, fmt, opts) {
  const host = document.getElementById(id); host.innerHTML = '';
  const series = D.names.map(n => D[key][n]);
  const W = Math.max(host.clientWidth, 320), H = opts.h || 320, m = {l: 64, r: 10, t: 10, b: 28};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.label}, host);
  const n = D.dates.length, all = series.flat();
  let lo = opts.min !== undefined ? opts.min : Math.min(...all), hi = opts.max !== undefined ? opts.max : Math.max(...all);
  const ticks = niceTicks(lo, hi, 5); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const x = i => m.l + i / (n - 1) * (W - m.l - m.r), y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  const hIdx = D.dates.findIndex(d => d >= D.holdout);
  if (hIdx > 0) el('rect', {x: x(hIdx), y: m.t, width: x(n - 1) - x(hIdx), height: H - m.t - m.b, fill: css('--shade')}, svg);
  ticks.forEach(t => { el('line', {x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: css('--grid'), 'stroke-width': 1}, svg);
    el('text', {x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end'}, svg).textContent = fmt(t); });
  const years = [...new Set(D.dates.map(d => d.slice(0, 4)))];
  const every = Math.ceil(years.length / Math.max(2, Math.floor((W - m.l - m.r) / 70)));
  years.forEach((yr, k) => { if (k % every) return; const i = D.dates.findIndex(d => d.startsWith(yr)); el('text', {x: x(i), y: H - 8, 'text-anchor': 'middle'}, svg).textContent = yr; });
  series.forEach((s, k) => { const d = s.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1)).join('');
    const bench = k >= D.names.length - 2;
    el('path', {d, fill: 'none', stroke: color(k), 'stroke-width': bench ? 1.5 : 2, 'stroke-dasharray': bench ? '4 3' : 'none', 'stroke-linejoin': 'round'}, svg); });
  const cross = el('line', {y1: m.t, y2: H - m.b, stroke: css('--text-3'), 'stroke-width': 1, opacity: 0}, svg);
  const hit = el('rect', {x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: 'transparent'}, svg);
  hit.addEventListener('mousemove', evt => { const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
    cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('opacity', 0.6);
    showTip(evt, new Date(D.dates[i]).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric'}),
      series.map((s, k) => ({n: D.names[k], v: fmt(s[i]), c: color(k)}))); });
  hit.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); });
}
function draw() { lineChart('equity', 'equity', fmtMoney, {label: 'Growth of $10,000', h: 340}); lineChart('drawdown', 'drawdown', v => v.toFixed(0) + '%', {label: 'Drawdowns', h: 240, max: 0}); }
draw(); let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
