"""Background jobs for the web app: run a backtest command as a subprocess, keep its log.

Only a fixed set of job kinds exists (no free-form commands). Each job writes
output/jobs/<id>.log and reports status, the last log lines and, when done, where its result is.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from backtest.data import OUTPUT_DIR, ROOT_DIR
from backtest.spreads import is_spread_file

JOB_DIR = OUTPUT_DIR / 'jobs'
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

SPACES = ('default', 'indicators', 'breadth', 'exits', 'momentum')


def _strategy_file(name: str) -> str:
    p = ROOT_DIR / 'strategies' / Path(name).name
    if not p.is_file():
        raise ValueError('Strategy file not found.')
    return str(p.relative_to(ROOT_DIR))


def build_command(kind: str, opts: dict) -> tuple[list[str], str]:
    """Returns (argv, label) for a job kind. Raises ValueError for bad input."""
    py = [sys.executable, '-m']
    start = opts.get('start') or '2016-01-01'
    holdout = opts.get('holdout_start') or '2025-01-01'
    for d in (start, holdout):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            raise ValueError('Dates must look like 2016-01-01.')
    if kind == 'fetch':
        return py + ['backtest.fetch', '--historical'], 'Download all price data (incl. past index members)'
    if kind == 'update':
        return py + ['backtest.fetch', '--update'], 'Add the latest trading days'
    if kind == 'membership':
        return py + ['backtest.membership', '--check'], 'Check price coverage of the index history'
    if kind == 'tests':
        return [sys.executable, 'tests/test_backtest.py'], 'Self-test of the engine'
    if kind == 'report':
        f = _strategy_file(opts.get('file', 'willr.json'))
        if is_spread_file(ROOT_DIR / f):
            return py + ['backtest.spreads', '--params', f, '--start', start, '--holdout-start', holdout], f'Report: {Path(f).stem}'
        return py + ['backtest.report', '--params', f, '--start', start, '--holdout-start', holdout, '--no-open'], f'Report: {Path(f).stem}'
    if kind == 'compare':
        files = [_strategy_file(f) for f in opts.get('files', [])]
        return py + ['backtest.compare', '--start', start, '--holdout-start', holdout, '--no-open'] + (['--params'] + files if files else []), \
            'Compare ' + (f'{len(files)} plans' if files else 'all plans') + ' with SPY'
    if kind == 'analyze':
        f = _strategy_file(opts.get('file', 'willr.json'))
        if is_spread_file(ROOT_DIR / f):
            raise ValueError('The stress test is for stock plans; an option plan gets its verdict and luck check in its report.')
        quick = bool(opts.get('quick', True))
        trials = max(1, int(opts.get('trials') or 1))
        cmd = py + ['backtest.analyze', '--params', f, '--start', start, '--holdout-start', holdout, '--trials', str(trials)]
        return cmd + (['--quick'] if quick else []), f'Stress test ({"quick" if quick else "full"}): {Path(f).stem}'
    if kind == 'tune':
        space = opts.get('space', 'default')
        if space not in SPACES:
            raise ValueError('Unknown search space.')
        n = max(20, min(5000, int(opts.get('n_iter') or 400)))
        metric = opts.get('metric', 'calmar')
        if metric not in ('sharpe', 'sortino', 'calmar', 'cagr', 'profit_factor'):
            raise ValueError('Unknown metric.')
        cmd = py + ['backtest.tune', '--space', space, '--start', start, '--holdout-start', holdout,
                    '--metric', metric, '--n-iter', str(n), '--min-trades', '20']
        return cmd, f'Search {n} rule sets ({space})'
    if kind == 'spreads_fetch':
        return py + ['backtest.spreads', '--fetch'], 'Download SPY and VIX history'
    if kind in ('spreads', 'spreads_tune', 'spreads_save'):
        def num(key, lo, hi, scale=1.0):
            v = float(opts.get(key) or 0)
            if not (lo <= v <= hi):
                raise ValueError(f'{key} out of range.')
            return v * scale
        cadence = opts.get('cadence', 'expiry')
        if cadence not in ('expiry', 'monthly', 'weekly', 'next'):
            raise ValueError('Unknown cadence.')
        strike_mode = opts.get('strike_mode', 'debit')
        if strike_mode not in ('debit', 'offset'):
            raise ValueError('Unknown strike mode.')
        cmd = py + ['backtest.spreads', '--start', start, '--holdout-start', holdout, '--strike-mode', strike_mode,
                    '--offset', f'{num("offset", -10, 10, 0.01):.4f}', '--width', f'{num("width", 1, 100):g}',
                    '--dte', f'{int(num("dte", 7, 365))}', '--target', f'{num("target", 10, 100, 0.01):.2f}',
                    '--target-div', f'{num("target_div", 0, 100, 0.01):.2f}' if opts.get('target_div') else f'{num("target", 10, 100, 0.01):.2f}',
                    '--stop', f'{num("stop", 0, 100, 0.01):.2f}', '--cadence', cadence,
                    '--trend-ma', f'{int(num("trend_ma", 0, 400))}']
        if float(opts.get('debit') or 0) > 0:
            cmd += ['--debit', f'{num("debit", 0.01, 1000):.2f}']
        if kind == 'spreads_save':
            name = re.sub(r'[^\w .,:$%()+-]', '', str(opts.get('name') or 'SPY call spread'))[:80].strip() or 'SPY call spread'
            return cmd + ['--save', name], f'Save option plan: {name}'
        if kind == 'spreads_tune':
            metric = opts.get('metric', 'calmar')
            if metric not in ('calmar', 'sharpe', 'cagr'):
                raise ValueError('Unknown metric.')
            cmd += ['--tune', '--metric', metric]
            grid = opts.get('grid') or {}
            if not isinstance(grid, dict):
                raise ValueError('grid must be an object.')
            count = 1
            for key, val in grid.items():
                if key not in ('strike', 'width', 'dte', 'target', 'stop', 'cadence', 'trend_ma'):
                    raise ValueError(f'Unknown grid key {key}.')
                vals = [v.strip() for v in str(val).split(',') if v.strip()]
                if not vals or any(not re.fullmatch(r'[\w.:+-]{1,20}', v) for v in vals):
                    raise ValueError(f'Bad values for {key}.')
                count *= len(vals)
                cmd += ['--grid', f'{key}={",".join(vals)}']
            m = int(opts.get('max_setups') or 0)
            if m < 0 or m > 20000:
                raise ValueError('max setups out of range.')
            if m:
                cmd += ['--max-setups', str(m)]
            n = min(count, m) if m else count
            if n > 5000:
                raise ValueError(f'{n} setups is too many for one run (about {n // 120} minutes); cut the lists or set "at most".')
            return cmd, f'Try {n} spread setups'
        return cmd, 'Backtest SPY call spreads'
    raise ValueError('Unknown job.')


def start_job(kind: str, opts: dict) -> dict:
    argv, label = build_command(kind, opts)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f'{datetime.now():%Y%m%d_%H%M%S}_{kind}'
    with _LOCK:
        if job_id in _JOBS:
            job_id += f'_{len(_JOBS)}'
    log_path = JOB_DIR / f'{job_id}.log'
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'}
    log = open(log_path, 'wb')
    proc = subprocess.Popen(argv, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
    job = {'id': job_id, 'kind': kind, 'label': label, 'argv': argv, 'started': time.time(),
           'finished': None, 'status': 'running', 'returncode': None, 'log': str(log_path), 'result': None}
    with _LOCK:
        _JOBS[job_id] = job

    def wait():
        rc = proc.wait()
        log.close()
        with _LOCK:
            job['returncode'] = rc
            job['finished'] = time.time()
            job['status'] = 'done' if rc == 0 else 'failed'
            job['result'] = find_result(kind, log_path)
    threading.Thread(target=wait, daemon=True).start()
    job['_proc'] = proc
    return public(job)


def cancel_job(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job or job['status'] != 'running':
        return False
    job['_proc'].terminate()
    job['status'] = 'cancelled'
    return True


def find_result(kind: str, log_path: Path) -> dict | None:
    """Parse the job's log for where it saved its result."""
    try:
        text = log_path.read_text(errors='replace')
    except OSError:
        return None
    text = text.replace('\r', '\n')
    m = re.search(r'Report saved: (\S+\.html)', text)
    if m:
        return {'type': 'html', 'path': m.group(1), 'url': '/reports/' + Path(m.group(1)).name}
    m = re.findall(r'Saved to (\S+)', text)
    if m:
        p = Path(m[-1])
        rel = p.relative_to(ROOT_DIR) if p.is_absolute() else p
        res = {'type': 'dir', 'path': str(rel)}
        if (ROOT_DIR / rel / 'report.txt').exists():
            res['report'] = f'/text/{rel}/report.txt'
        if (ROOT_DIR / rel / 'best_params.json').exists():
            res['best_params'] = str(rel / 'best_params.json')
        return res
    if kind in ('fetch', 'update', 'membership', 'tests', 'spreads_fetch'):
        return {'type': 'log'}
    return None


def tail(log_path: str, n: int = 12) -> list[str]:
    try:
        data = Path(log_path).read_bytes()[-20000:].decode('utf-8', errors='replace')
    except OSError:
        return []
    lines = [l.rstrip() for l in data.replace('\r', '\n').split('\n')]
    lines = [l for l in lines if l.strip()]
    out = []
    for l in lines:  # collapse tqdm progress spam: keep only the latest progress line in a row
        if out and '%|' in l and '%|' in out[-1]:
            out[-1] = l
        else:
            out.append(l)
    return out[-n:]


def public(job: dict) -> dict:
    d = {k: v for k, v in job.items() if not k.startswith('_') and k != 'argv'}
    d['elapsed'] = int((job['finished'] or time.time()) - job['started'])
    d['log_url'] = '/text/' + str(Path(job['log']).relative_to(ROOT_DIR))
    d['tail'] = tail(job['log'])
    return d


def list_jobs() -> list[dict]:
    with _LOCK:
        jobs = list(_JOBS.values())
    return [public(j) for j in sorted(jobs, key=lambda j: j['started'], reverse=True)]
