"""One-page HTML report for a rule set - made for reading, not for coding.

    python3 -m backtest.report                                    # strategies/willr.json
    python3 -m backtest.report --params strategies/coach_plan.json

Writes output/report_<name>_<timestamp>.html and opens it in the browser (--no-open to skip).
Contains: the rules in plain English, the pass/fail verdict, key numbers vs SPY and
SPY timing, growth of $10,000, drawdowns, yearly returns, trade statistics and a glossary.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import criteria
from backtest.benchmarks import benchmark_metrics, spy_buy_hold, spy_timing
from backtest.significance import luck_summary
from backtest.data import OUTPUT_DIR, ROOT_DIR, load_benchmark, load_prices
from backtest.engine import generate_trades, simulate_portfolio
from backtest.features import FeatureStore
from backtest.membership import load_membership
from backtest.metrics import compute_metrics, yearly_returns
from backtest.strategy import StrategyParams

SERIES = ['Strategy', 'SPY buy & hold', 'SPY timing']


def universe_label(index: str | None) -> str:
    names = {'sp500': 'S&P 500', 'sp100': 'S&P 100', 'nasdaq100': 'Nasdaq-100', None: 'all', 'all': 'all'}
    parts = [names.get(i, i) for i in (index or 'all').split('+')]
    label = ' + '.join(parts) + ' stocks'
    if index != 'sp500':
        label += " (today's members)"
    return label


def _pct(x: float, digits: int = 1) -> str:
    return f'{x * 100:.{digits}f}%'


def _row(label: str, m: dict) -> str:
    return (f'<tr><th scope="row">{html.escape(label)}</th><td>{_pct(m["cagr"])}</td>'
            f'<td>{_pct(m["max_drawdown"])}</td><td>{m["sharpe"]:.2f}</td><td>{m["calmar"]:.2f}</td></tr>')


def _period_table(title: str, strat: dict, bms: dict) -> str:
    return (f'<h3>{html.escape(title)}</h3><div class="tbl"><table><thead><tr><th></th><th>Return / year</th>'
            f'<th>Worst drop</th><th>Sharpe</th><th>Calmar</th></tr></thead><tbody>'
            + _row('Strategy', strat) + _row('SPY buy & hold', bms['SPY buy & hold'])
            + _row('SPY timing', bms['SPY timing (>SMA200)']) + '</tbody></table></div>')


def build(params_path: Path, index: str | None, start: str, holdout_start: str, cost_bps: float,
          fs: FeatureStore | None = None, trials: int = 1) -> Path:
    """Backtest the rule set in params_path and write the HTML report. Pass a cached `fs` to skip loading data."""
    p = StrategyParams.load(params_path)
    index = p.universe or index
    meta = json.loads(params_path.read_text())
    note = meta.get('note', '')
    trials = max(int(trials or 1), int(meta.get('trials', 1) or 1))  # a strategy file can record how it was found
    bench = load_benchmark('SPY')
    if fs is None:
        fs = FeatureStore(load_prices(index), bench, load_membership(index))
    cost = cost_bps / 1e4
    trades = generate_trades(fs, p)
    tune_end = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)

    def sim(s, e, c=cost):
        r = simulate_portfolio(fs, trades, s, e, max_positions=p.max_positions, max_new_per_day=p.max_new_per_day, cost_per_side=c)
        return r, compute_metrics(r.equity, r.trades, r.exposure)

    full_res, full_m = sim(start, None)
    has_dev = tune_end > pd.Timestamp(start) + pd.Timedelta(days=60)
    tune_m = sim(start, tune_end)[1] if has_dev else None
    _, hold_m = sim(holdout_start, None)
    _, stress_m = sim(start, None, criteria.STRESS_COST_BPS / 1e4)
    full_b = benchmark_metrics(bench, full_res.equity.index, cost)
    tune_b = benchmark_metrics(bench, full_res.equity.index[full_res.equity.index <= tune_end], cost) if has_dev else None
    hold_b = benchmark_metrics(bench, full_res.equity.index[full_res.equity.index >= pd.Timestamp(holdout_start)], cost)
    luck = luck_summary(full_res.equity, spy_buy_hold(bench, full_res.equity.index), trials)
    verdict = criteria.evaluate(full_m, hold_m, stress_m, full_b['SPY buy & hold'],
                                hold_b['SPY buy & hold'], hold_b['SPY timing (>SMA200)'], luck)
    passed = sum(ok for _, ok, _ in verdict)

    curves = {
        'Strategy': full_res.equity / full_res.equity.iloc[0] * 10_000,
        'SPY buy & hold': spy_buy_hold(bench, full_res.equity.index, 10_000),
        'SPY timing': spy_timing(bench, full_res.equity.index, cost_per_side=cost, capital=10_000),
    }
    title = meta.get('name') or (f'Best set of {params_path.parent.name}' if params_path.stem == 'best_params'
                                 else params_path.stem.replace('_', ' ').title())
    return render(
        out_stem=params_path.stem, title=title, subtitle_what=f'Backtest on {universe_label(index)}',
        cost_text=f'costs {cost_bps:g} bps per trade side', note=note, rules=p.describe(), verdict=verdict,
        full_m=full_m, hold_m=hold_m, tune_m=tune_m, full_b=full_b, hold_b=hold_b, tune_b=tune_b,
        curves=curves, trades=full_res.trades, start=start, holdout_start=holdout_start, tune_end=tune_end,
        reason_names={'signal': 'Exit signal', 'stop': 'Stop loss', 'trail': 'Trailing stop',
                      'time': 'Time limit', 'window_end': 'Still open at the end'})


def render(*, out_stem: str, title: str, subtitle_what: str, cost_text: str, note: str, rules: str,
           verdict: list, full_m: dict, hold_m: dict, tune_m: dict | None, full_b: dict, hold_b: dict,
           tune_b: dict | None, curves: dict, trades: pd.DataFrame, start: str, holdout_start: str,
           tune_end: pd.Timestamp, reason_names: dict, stress_text: str = f'{criteria.STRESS_COST_BPS:g} bps costs') -> Path:
    """Write the one-page HTML report from already computed results (shared by stock and option plans).
    `curves`: {'Strategy', 'SPY buy & hold', 'SPY timing'} -> daily growth of $10,000 on the same index."""
    passed = sum(ok for _, ok, _ in verdict)
    idx = curves['Strategy'].index
    weekly = pd.DataFrame(curves).resample('W-FRI').last().dropna()
    dd = weekly / weekly.cummax() - 1
    yr = pd.DataFrame({k: yearly_returns(v) for k, v in curves.items()})
    data = {
        'dates': [d.strftime('%Y-%m-%d') for d in weekly.index],
        'equity': {k: [round(v, 2) for v in weekly[k]] for k in SERIES},
        'drawdown': {k: [round(v * 100, 2) for v in dd[k]] for k in SERIES},
        'years': [str(y) for y in yr.index],
        'yearly': {k: [round(v * 100, 2) for v in yr[k]] for k in SERIES},
        'holdout': holdout_start,
    }
    t = trades
    reasons = t.groupby('exit_reason')['net_return'].agg(['count', 'mean']) if len(t) else pd.DataFrame()
    reason_rows = ''.join(
        f'<tr><th scope="row">{reason_names.get(r, r)}</th><td>{int(row["count"])}</td><td>{_pct(row["mean"], 2)}</td></tr>'
        for r, row in reasons.iterrows())
    yearly_rows = ''.join(
        f'<tr><th scope="row">{y}</th>' + ''.join(f'<td>{_pct(yr.loc[int(y), k])}</td>' for k in SERIES) + '</tr>'
        for y in data['years'])
    verdict_rows = ''.join(
        f'<li class="{"ok" if ok else "bad"}"><span class="badge">{"✓ PASS" if ok else "✕ FAIL"}</span>'
        f'<span class="crit">{html.escape(name)}</span><span class="detail">{html.escape(detail)}</span></li>'
        for name, ok, detail in verdict)
    period = f'{idx[0].date():%d %b %Y} – {idx[-1].date():%d %b %Y}'
    headline_ok = passed == len(verdict)
    has_dev = tune_m is not None
    exposure = full_m.get('exposure')
    exposure_txt = _pct(exposure, 0) if exposure is not None and exposure == exposure else '–'

    page = TEMPLATE
    for key, val in {
        '{{TITLE}}': html.escape(title),
        '{{PERIOD}}': period,
        '{{WHAT}}': html.escape(subtitle_what), '{{COST_TEXT}}': html.escape(cost_text),
        '{{NOTE}}': f'<p class="note">{html.escape(note)}</p>' if note else '',
        '{{RULES}}': html.escape(rules),
        '{{VERDICT_CLASS}}': 'ok' if headline_ok else 'bad',
        '{{VERDICT_HEAD}}': (f'{passed} of {len(verdict)} checks passed – '
                             + ('passes every check' if headline_ok else 'does not pass')),
        '{{VERDICT_ROWS}}': verdict_rows,
        '{{K_CAGR}}': _pct(full_m['cagr']), '{{K_SPY_CAGR}}': _pct(full_b['SPY buy & hold']['cagr']),
        '{{K_DD}}': _pct(full_m['max_drawdown']), '{{K_SPY_DD}}': _pct(full_b['SPY buy & hold']['max_drawdown']),
        '{{K_WIN}}': _pct(full_m['win_rate'], 0), '{{K_TRADES}}': f'{full_m["trades"]:,}',
        '{{K_TPY}}': f'{full_m["trades_per_year"]:.0f}', '{{K_HOLD}}': f'{full_m["avg_bars_held"]:.1f}',
        '{{K_HOLDOUT}}': _pct(hold_m['cagr']), '{{K_SPY_HOLDOUT}}': _pct(hold_b['SPY buy & hold']['cagr']),
        '{{TABLE_TUNE}}': (_period_table(f'Development period ({pd.Timestamp(start):%Y} – {tune_end:%Y})', tune_m, tune_b)
                           if has_dev else ''),
        '{{TABLE_HOLD}}': _period_table(f'Test period ({pd.Timestamp(holdout_start):%b %Y} – today, never used to pick the rules)',
                                        hold_m, hold_b),
        '{{YEARLY_ROWS}}': yearly_rows,
        '{{REASON_ROWS}}': reason_rows,
        '{{AVG_TRADE}}': _pct(full_m['avg_trade'], 2), '{{AVG_WIN}}': _pct(full_m['avg_win'], 2),
        '{{AVG_LOSS}}': _pct(full_m['avg_loss'], 2), '{{EXPOSURE}}': exposure_txt,
        '{{STRESS}}': html.escape(stress_text),
        '{{DATA}}': json.dumps(data),
        '{{GENERATED}}': f'{datetime.now():%d %b %Y %H:%M}',
    }.items():
        page = page.replace(key, val)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f'report_{out_stem}_{datetime.now():%Y%m%d_%H%M%S}.html'
    out.write_text(page, encoding='utf-8')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', default=str(ROOT_DIR / 'strategies' / 'willr.json'))
    ap.add_argument('--index', default='sp500', help="sp500 | sp100 | nasdaq100 | sp100+nasdaq100 | all; a 'universe' in the strategy file wins")
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--holdout-start', default='2025-01-01')
    ap.add_argument('--cost-bps', type=float, default=5.0)
    ap.add_argument('--trials', type=int, default=1, help='rule sets tried before picking this one (deflates the luck check)')
    ap.add_argument('--no-open', action='store_true', help='do not open the report in the browser')
    args = ap.parse_args()
    print(f'Backtesting {args.params} ... (about a minute)')
    out = build(Path(args.params), None if args.index == 'all' else args.index,
                args.start, args.holdout_start, args.cost_bps, trials=args.trials)
    print(f'Report saved: {out.relative_to(ROOT_DIR)}')
    if not args.no_open:
        import webbrowser
        webbrowser.open(out.as_uri())


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} – Strategy Report</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f3f0; --surface: #fcfcfb; --border: #e3e2dd;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --good: #0ca30c; --good-text: #006300; --bad: #d03b3b; --bad-text: #a32222;
  --grid: #e9e8e4; --shade: rgba(42,120,214,0.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2e2e2c;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --good: #0ca30c; --good-text: #4cc94c; --bad: #d03b3b; --bad-text: #f07a7a;
    --grid: #2a2a28; --shade: rgba(57,135,229,0.10);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
header h1 { font-size: 30px; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }
header .sub { color: var(--text-2); margin: 0; }
.note { color: var(--text-2); font-size: 14px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 14px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 22px 24px; margin-top: 18px; }
h2 { font-size: 19px; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 18px 0 8px; color: var(--text-2); font-weight: 600; }
p.lead { color: var(--text-2); margin: -4px 0 14px; font-size: 15px; }
pre.rules { white-space: pre-wrap; font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); border-radius: 10px; padding: 14px 16px; margin: 0; overflow-x: auto; }
.verdict-head { display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 600; margin-bottom: 12px; }
.verdict-head .dot { width: 14px; height: 14px; border-radius: 50%; flex: none; }
.verdict-head.ok .dot { background: var(--good); } .verdict-head.bad .dot { background: var(--bad); }
ul.checks { list-style: none; padding: 0; margin: 0; }
ul.checks li { display: grid; grid-template-columns: 84px 1fr; gap: 2px 12px; padding: 9px 0;
  border-top: 1px solid var(--border); }
ul.checks .badge { font-weight: 700; font-size: 13px; grid-row: span 2; padding-top: 2px; }
ul.checks li.ok .badge { color: var(--good-text); } ul.checks li.bad .badge { color: var(--bad-text); }
ul.checks .crit { font-weight: 600; } ul.checks .detail { color: var(--text-2); font-size: 14px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.kpi { background: var(--bg); border-radius: 12px; padding: 14px 16px; }
.kpi .label { color: var(--text-2); font-size: 13px; }
.kpi .value { font-size: 26px; font-weight: 650; line-height: 1.25; margin-top: 2px; font-variant-numeric: tabular-nums; }
.kpi .cmp { color: var(--text-3); font-size: 13px; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--text-2); font-weight: 600; }
tbody th { font-weight: 500; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; color: var(--text-2); margin: 0 0 8px; }
.legend span::before { content: ""; display: inline-block; width: 14px; height: 3px; border-radius: 2px;
  vertical-align: middle; margin-right: 6px; background: var(--c); }
.chart { position: relative; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { font-size: 12px; fill: var(--text-3); font-family: inherit; }
.chart .lbl { font-size: 12px; font-weight: 600; fill: var(--text-2); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
  opacity: 0; transition: opacity .1s; white-space: nowrap; z-index: 2; }
.tip b { display: block; margin-bottom: 2px; }
.tip .r { display: flex; justify-content: space-between; gap: 14px; color: var(--text-2); }
.tip .r i { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } section { padding: 18px 16px; }
  th, td { padding: 6px 6px; font-size: 13px; } header h1 { font-size: 24px; } .kpi .value { font-size: 22px; } }
dl.gloss { display: grid; grid-template-columns: 170px 1fr; gap: 8px 16px; margin: 0; font-size: 15px; }
dl.gloss dt { font-weight: 600; } dl.gloss dd { margin: 0; color: var(--text-2); }
@media (max-width: 600px) { dl.gloss { grid-template-columns: 1fr; } dl.gloss dd { margin-bottom: 8px; } }
footer { color: var(--text-3); font-size: 13px; margin-top: 18px; text-align: center; }
</style>
</head>
<body>
<main>
<header>
  <a id="back" href="/" style="display:none;color:var(--s1);font-size:14px;text-decoration:none">&larr; Back to the planner</a>
  <h1>{{TITLE}}</h1>
  <p class="sub">{{WHAT}} · {{PERIOD}} · {{COST_TEXT}}</p>
</header>
{{NOTE}}

<section>
  <h2>The rules</h2>
  <pre class="rules">{{RULES}}</pre>
</section>

<section>
  <h2>Verdict</h2>
  <div class="verdict-head {{VERDICT_CLASS}}"><span class="dot"></span>{{VERDICT_HEAD}}</div>
  <ul class="checks">{{VERDICT_ROWS}}</ul>
</section>

<section>
  <h2>Key numbers</h2>
  <div class="kpis">
    <div class="kpi"><div class="label">Return per year</div><div class="value">{{K_CAGR}}</div><div class="cmp">SPY {{K_SPY_CAGR}}</div></div>
    <div class="kpi"><div class="label">Worst drop from a high</div><div class="value">{{K_DD}}</div><div class="cmp">SPY {{K_SPY_DD}}</div></div>
    <div class="kpi"><div class="label">Test period return / year</div><div class="value">{{K_HOLDOUT}}</div><div class="cmp">SPY {{K_SPY_HOLDOUT}}</div></div>
    <div class="kpi"><div class="label">Winning trades</div><div class="value">{{K_WIN}}</div><div class="cmp">{{K_TRADES}} trades, ~{{K_TPY}} per year</div></div>
    <div class="kpi"><div class="label">Days per trade</div><div class="value">{{K_HOLD}}</div><div class="cmp">invested {{EXPOSURE}} of the time</div></div>
  </div>
  {{TABLE_TUNE}}
  {{TABLE_HOLD}}
</section>

<section>
  <h2>Growth of $10,000</h2>
  <p class="lead">Shaded area = test period, never used to choose the rules. Hover for values.</p>
  <div class="legend"><span style="--c:var(--s1)">Strategy</span><span style="--c:var(--s2)">SPY buy &amp; hold</span><span style="--c:var(--s3)">SPY timing (only above its 200-day average)</span></div>
  <div class="chart" id="equity"></div>
</section>

<section>
  <h2>Drops from the previous high</h2>
  <p class="lead">How deep each approach fell before recovering. Smaller is easier to sit through.</p>
  <div class="legend"><span style="--c:var(--s1)">Strategy</span><span style="--c:var(--s2)">SPY buy &amp; hold</span><span style="--c:var(--s3)">SPY timing</span></div>
  <div class="chart" id="drawdown"></div>
</section>

<section>
  <h2>Year by year</h2>
  <div class="legend"><span style="--c:var(--s1)">Strategy</span><span style="--c:var(--s2)">SPY buy &amp; hold</span><span style="--c:var(--s3)">SPY timing</span></div>
  <div class="chart" id="yearly"></div>
  <div class="tbl"><table><thead><tr><th>Year</th><th>Strategy</th><th>SPY buy &amp; hold</th><th>SPY timing</th></tr></thead>
  <tbody>{{YEARLY_ROWS}}</tbody></table></div>
</section>

<section>
  <h2>Trades</h2>
  <div class="cols">
    <div class="tbl"><table><thead><tr><th>How trades ended</th><th>Trades</th><th>Avg result</th></tr></thead><tbody>{{REASON_ROWS}}</tbody></table></div>
    <div class="tbl"><table><tbody>
      <tr><th scope="row">Average trade</th><td>{{AVG_TRADE}}</td></tr>
      <tr><th scope="row">Average winner</th><td>{{AVG_WIN}}</td></tr>
      <tr><th scope="row">Average loser</th><td>{{AVG_LOSS}}</td></tr>
      <tr><th scope="row">Time invested</th><td>{{EXPOSURE}}</td></tr>
    </tbody></table></div>
  </div>
</section>

<section>
  <h2>How to read this</h2>
  <dl class="gloss">
    <dt>Return per year</dt><dd>Average yearly growth (CAGR), after trading costs.</dd>
    <dt>Worst drop</dt><dd>The biggest fall from a previous high (max drawdown). −20% means $10,000 became $8,000 at some point.</dd>
    <dt>Sharpe</dt><dd>Return per unit of ups and downs. Higher is smoother. SPY is usually around 0.8–1.0.</dd>
    <dt>Calmar</dt><dd>Return per year divided by the worst drop. Higher means more return for the pain.</dd>
    <dt>SPY timing</dt><dd>Simple benchmark: hold the S&amp;P 500 ETF only while it is above its 200-day average, otherwise cash. If a strategy can't beat this, its stock picking adds nothing.</dd>
    <dt>Development period</dt><dd>The years used to find the rules. Results here always look good.</dd>
    <dt>Test period</dt><dd>Data the rules have never seen. This is the honest number.</dd>
    <dt>Verdict</dt><dd>Seven checks agreed on before testing (strong enough vs SPY, holds up in the test period, survives {{STRESS}}, smaller drops than SPY, unlikely to be luck). All must pass before the result means anything.</dd>
    <dt>Unlikely to be luck</dt><dd>Probability that the edge over SPY is real, from the day-to-day ups and downs of the difference between the strategy and SPY. Below 90% the result is within what chance alone produces. If many variants were tried before this one, the bar rises (a lucky best-of-100 looks impressive by itself).</dd>
  </dl>
</section>

<footer>Generated {{GENERATED}} · StrategyAnalyser · Past results do not guarantee future results.</footer>
</main>
<div class="tip" id="tip"></div>
<script>
const D = {{DATA}};
const NAMES = ['Strategy', 'SPY buy & hold', 'SPY timing'];
const VARS = ['--s1', '--s2', '--s3'];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const tip = document.getElementById('tip');
const fmtMoney = v => '$' + Math.round(v).toLocaleString('en-US');
const fmtPct = v => (v > 0 ? '+' : '') + v.toFixed(1) + '%';

function showTip(evt, title, rows) {
  tip.innerHTML = '<b>' + title + '</b>' + rows.map(r =>
    '<div class="r"><span><i style="background:' + r.c + '"></i>' + r.n + '</span><span>' + r.v + '</span></div>').join('');
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.pageX + pad, y = evt.pageY - h - pad;
  if (x + w > window.scrollX + document.documentElement.clientWidth - 8) x = evt.pageX - w - pad;
  if (y < window.scrollY + 8) y = evt.pageY + pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.style.opacity = 0; };

function niceTicks(lo, hi, n) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v); return out;
}

function lineChart(id, series, fmt, opts) {
  const host = document.getElementById(id); host.innerHTML = '';
  const W = Math.max(host.clientWidth, 320), H = opts.h || 300, m = {l: 64, r: 96, t: 10, b: 28};
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
  years.forEach((yr, k) => { if (k % every) return; const i = D.dates.findIndex(d => d.startsWith(yr));
    el('text', {x: x(i), y: H - 8, 'text-anchor': 'middle'}, svg).textContent = yr; });
  const labels = [];
  series.forEach((s, k) => {
    const d = s.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1)).join('');
    el('path', {d, fill: 'none', stroke: css(VARS[k]), 'stroke-width': 2, 'stroke-linejoin': 'round'}, svg);
    labels.push({k, y: y(s[n - 1])});
  });
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
  labels.forEach(l => { el('text', {x: W - m.r + 8, y: l.y + 4, class: 'lbl'}, svg).textContent =
    ['Strategy', 'SPY', 'SPY timing'][l.k]; });
  const cross = el('line', {y1: m.t, y2: H - m.b, stroke: css('--text-3'), 'stroke-width': 1, opacity: 0}, svg);
  const dots = series.map((s, k) => el('circle', {r: 4, fill: css(VARS[k]), stroke: css('--surface'), 'stroke-width': 2, opacity: 0}, svg));
  const hit = el('rect', {x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: 'transparent'}, svg);
  hit.addEventListener('mousemove', evt => {
    const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
    cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('opacity', 0.6);
    dots.forEach((c, k) => { c.setAttribute('cx', x(i)); c.setAttribute('cy', y(series[k][i])); c.setAttribute('opacity', 1); });
    showTip(evt, new Date(D.dates[i]).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric'}),
      series.map((s, k) => ({n: NAMES[k], v: fmt(s[i]), c: css(VARS[k])})));
  });
  hit.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); dots.forEach(c => c.setAttribute('opacity', 0)); });
}

function barChart(id) {
  const host = document.getElementById(id); host.innerHTML = '';
  const W = Math.max(host.clientWidth, 320), H = 280, m = {l: 52, r: 8, t: 10, b: 28};
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Yearly returns'}, host);
  const S = NAMES.map(k => D.yearly[k]), all = S.flat();
  const ticks = niceTicks(Math.min(0, ...all), Math.max(0, ...all), 5);
  const lo = Math.min(ticks[0], ...all), hi = Math.max(ticks[ticks.length - 1], ...all);
  const y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  ticks.forEach(t => { el('line', {x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: css('--grid'), 'stroke-width': 1}, svg);
    el('text', {x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end'}, svg).textContent = t + '%'; });
  el('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: css('--text-3'), 'stroke-width': 1}, svg);
  const ny = D.years.length, band = (W - m.l - m.r) / ny, bw = Math.max(3, Math.min(18, (band - 10) / 3 - 2));
  D.years.forEach((yr, j) => {
    const cx = m.l + band * (j + 0.5);
    const lab = band < 34 ? "'" + yr.slice(2) : yr, skip = band < 22 ? 2 : 1;
    if (j % skip === 0) el('text', {x: cx, y: H - 8, 'text-anchor': 'middle'}, svg).textContent = lab;
    S.forEach((s, k) => {
      const v = s[j], x0 = cx + (k - 1) * (bw + 2) - bw / 2, top = y(Math.max(v, 0)), bot = y(Math.min(v, 0));
      const h = Math.max(1, bot - top), r = Math.min(4, bw / 2, h);
      const d = v >= 0
        ? `M${x0},${bot} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x0 + bw - r} Q${x0 + bw},${top} ${x0 + bw},${top + r} V${bot} Z`
        : `M${x0},${top} V${bot - r} Q${x0},${bot} ${x0 + r},${bot} H${x0 + bw - r} Q${x0 + bw},${bot} ${x0 + bw},${bot - r} V${top} Z`;
      el('path', {d, fill: css(VARS[k])}, svg);
    });
    const hit = el('rect', {x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, fill: 'transparent'}, svg);
    hit.addEventListener('mousemove', evt => showTip(evt, yr, S.map((s, k) => ({n: NAMES[k], v: fmtPct(s[j]), c: css(VARS[k])}))));
    hit.addEventListener('mouseleave', hideTip);
  });
}

function draw() {
  lineChart('equity', NAMES.map(k => D.equity[k]), fmtMoney, {label: 'Growth of $10,000', h: 320});
  lineChart('drawdown', NAMES.map(k => D.drawdown[k]), v => v.toFixed(0) + '%', {label: 'Drawdowns', h: 240, max: 0});
  barChart('yearly');
}
draw();
if (location.protocol.startsWith('http')) document.getElementById('back').style.display = 'inline-block';
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
