"""P1 + P2: the attribution-only benchmark, and the decomposition of the gap.

One goal controller, one query rule, one environment, one reward, one set of
training parameters. The single thing that varies is **who a paid answer is
shared with**. No `novel`/`aligned`/`decision` heuristics here: mixing attribution
with query selection is what made the earlier results hard to read.

The query rule is the most mechanical one available: the first time an
(attribution state, event) pair is seen, ask; afterwards reuse.

  fullprefix   every complete visible event history stands alone
  node         the believed progress history (the standing baseline)
  mergedclass  the learned partition: an answer is shared across a class
  truestate    the true task state, privileged, the ceiling

Every query is also classified: **necessary** the first time that (true task
state, event) pair is paid for anywhere, **redundant** otherwise. Redundant
queries are exactly what a perfect abstraction would have saved, so
`necessary + redundant` decomposes each key's spend against the ceiling.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
# Each entry is (name, extra flags). `mergedclass` files a paid answer against
# the learned class, so the merge rule decides what the abstraction is.
KEYS = [('fullprefix', ['--cache-key', 'fullprefix']),
        ('node',       ['--cache-key', 'node']),
        ('edsm',       ['--cache-key', 'mergedclass', '--merge-rule', 'edsm']),
        ('sig2',       ['--cache-key', 'mergedclass', '--merge-rule', 'signature', '--sig-support', '2']),
        ('sig3',       ['--cache-key', 'mergedclass', '--merge-rule', 'signature', '--sig-support', '3']),
        ('sig4',       ['--cache-key', 'mergedclass', '--merge-rule', 'signature', '--sig-support', '4']),
        ('truestate',  ['--cache-key', 'truestate'])]
LEVELS = [80, 160, 320, 2000]          # 2000 is effectively unlimited here
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']


def one(job):
    tag, key, lv, seed = job
    name, extra = key
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--method', 'goal', '--goal-select', 'learned', '--seed', str(seed),
           '--budget', str(BUDGET), '--every', '250',
           '--verify-policy', 'arrival',
           '--verify-budget', str(lv)] + MERGE + extra
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'key': name, 'level': lv, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'], 'used': r['verifications'],
            'necessary': r['necessary_queries'], 'redundant': r['redundant_queries'],
            'stale': r['stale_uses'], 'stale_missed': r['stale_missed_advance'],
            'desync': r['desync_steps'], 'classes': r['classes'], 'nodes': r['nodes']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], k, lv, s) for f in fams for k in KEYS for lv in LEVELS
            for s in range(10600, 10606)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 300 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/attribution2.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_attribution.py', families=len(fams), keys=[k for k,_ in KEYS],
        levels=LEVELS, seeds=6, seed0=10600, budget=BUDGET,
        query_rule='arrival: ask once per (attribution state, event), reuse after',
        runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
