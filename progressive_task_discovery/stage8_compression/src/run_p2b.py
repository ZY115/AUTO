"""P2 corrected, after a review found two defects in the first version.

Defect one: `count` and `window` read the verification-filtered progress history,
not the raw event stream, because the history source was tied to the method name.
At zero queries all three saw an empty history and produced identical
trajectories. Source and encoding are now declared separately (`--history raw`),
and `--window-pure` stops a window from also hashing the total prefix length.

Defect two, and the more important one: the experiment that was asked for was
"keep the same goal controller, file the query evidence under the complete
visible prefix instead". The first version instead swapped the controller for
raw-history-indexed Q-learning, changing two things at once. Part B below is the
comparison as asked: one controller, three filing keys.

  node        the believed progress history (the standing default)
  fullprefix  the complete visible event prefix, which costs no queries to know
  truestate   the true task state, privileged, an attribution ceiling
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
FREE = [
    ('count_raw',   ['--method', 'count', '--history', 'raw']),
    ('window2_raw', ['--method', 'window', '--window', '2', '--window-pure', '1', '--history', 'raw']),
    ('window4_raw', ['--method', 'window', '--window', '4', '--window-pure', '1', '--history', 'raw']),
    ('raw',         ['--method', 'rawhistory']),
]
KEYS = ['node', 'fullprefix', 'truestate']
LEVELS = [20, 40, 80, 160, 320]
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']


def one(job):
    tag, kind, name, extra, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', '250'] + extra
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'kind': kind, 'arm': name, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'], 'used': r['verifications'],
            'states': r['features'], 'overflow': r['overflow_hits'],
            'desync': r['desync_steps'], 'trace': r['trace_hash']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = []
    for f in fams:
        for s in range(10300, 10306):
            for name, extra in FREE:
                jobs.append((f['tag'], 'free', name, extra + ['--verify-budget', '0'], s))
            for key in KEYS:
                for lv in LEVELS:
                    jobs.append((f['tag'], 'cache', f'{key}_{lv}',
                                 ['--method', 'goal', '--goal-select', 'learned',
                                  '--cache-key', key, '--verify-budget', str(lv)] + MERGE, s))
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 300 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/p2_corrected.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_p2b.py', families=len(fams), free_arms=[n for n, _ in FREE],
        cache_keys=KEYS, levels=LEVELS, seeds=6, seed0=10300, budget=BUDGET,
        runs=len(rows), note='supersedes results/p2_information.jsonl, whose count/window arms read the wrong history',
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
