"""P2: is the failure a lack of information, or our own discarding of it?

Under a verification budget the goal arms throw away every firing they did not
pay to check: the history does not extend and nothing is recorded. The baselines
here keep the visible event stream instead and **spend no queries at all**,
because which label fired is always visible; only whether it advanced costs
money. If keeping everything beats paying for progress labels, the failure was
self-inflicted. If it does not, the information really is insufficient.

  rawhistory  every observed event, no automaton, zero queries
  window k    the last k observed events, zero queries
  count       how many events have fired, zero queries
  history     the progress-event history, which needs the labels
  goal        the goal reading, which needs the labels
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
ARMS = {
    'raw_free':    (['--method', 'rawhistory'], 0),
    'window2_free':(['--method', 'window', '--window', '2'], 0),
    'window4_free':(['--method', 'window', '--window', '4'], 0),
    'count_free':  (['--method', 'count'], 0),
    'history_80':  (['--method', 'history'], 80),
    'history_320': (['--method', 'history'], 320),
    'goal_80':     (['--method', 'goal', '--goal-select', 'learned'], 80),
    'goal_160':    (['--method', 'goal', '--goal-select', 'learned'], 160),
    'goal_320':    (['--method', 'goal', '--goal-select', 'learned'], 320),
    'goal_free':   (['--method', 'goal', '--goal-select', 'learned'], -1),
}


def one(job):
    tag, arm, seed = job
    extra, budget = ARMS[arm]
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', '250',
           '--theta', '20', '--quota', '1', '--strict-children', '0',
           '--verify-budget', str(budget)] + extra
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'arm': arm, 'seed': seed, 'budget': budget,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'used': r['verifications'], 'states': r['features'],
            'overflow': r['overflow_hits'], 'unknown': r['unknown_firings']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], a, s) for f in fams for a in ARMS for s in range(10100, 10106)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 200 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/p2_information.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_p2.py', families=len(fams), arms=list(ARMS), seeds=6,
        seed0=10100, budget=BUDGET, runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
