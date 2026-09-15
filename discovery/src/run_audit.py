"""P0: how much of the negative result is wrong caching rather than the query policy?

A label is filed under the agent's history node. When the agent skips a query on
a firing that did advance, its history stops matching the task, and from then on
labels are filed under, and read from, a node that describes a different task
state. Step 24 named that mechanism; this measures it and separates it from the
policy.

Three cache settings against the same policies and budgets:

  node / reuse      what every experiment so far did
  node / no reuse   never re-use a recorded label, so each firing is paid for or
                    unknown; isolates the policy from the cache entirely
  truestate / reuse privileged: file labels under the true task state. This is
                    what perfect evidence attribution would look like, so the gap
                    to it is the ceiling on what any structure could recover.

Instrumentation is measurement only: no arm's behaviour depends on the counters.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
LEVELS = [160, 320]
POLICIES = ['arrival', 'random', 'decision', 'novel']
CACHES = [('node', 1), ('node', 0), ('truestate', 1)]


def one(job):
    tag, lv, pol, key, reuse, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--method', 'goal', '--goal-select', 'learned', '--seed', str(seed),
           '--budget', str(BUDGET), '--every', '250',
           '--theta', '20', '--quota', '1', '--strict-children', '0',
           '--merge-rule', 'edsm', '--verify-budget', str(lv),
           '--verify-policy', pol, '--verify-rate', '0.5',
           '--cache-key', key, '--cache-reuse', str(reuse)]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'level': lv, 'policy': pol, 'key': key, 'reuse': reuse,
            'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'], 'used': r['verifications'],
            'desync': r['desync_steps'], 'misattributed': r['misattributed'],
            'cache_hits': r['cache_hits'], 'stale': r['stale_uses'],
            'stale_missed': r['stale_missed_advance'], 'unknown': r['unknown_firings']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], lv, p, k, ru, s) for f in fams for lv in LEVELS
            for p in POLICIES for (k, ru) in CACHES for s in range(10000, 10004)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 300 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/audit.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_audit.py', families=len(fams), levels=LEVELS,
        policies=POLICIES, caches=[list(c) for c in CACHES], seeds=4, seed0=10000,
        budget=BUDGET, merge_rule='edsm', runs=len(rows),
        note='truestate cache is privileged and exists only as an attribution ceiling',
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
