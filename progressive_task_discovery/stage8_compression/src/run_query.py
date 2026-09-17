"""Given a query budget, which questions should the agent spend it on, and what
should it believe about a firing it did not pay to check?

Those are two separate choices and are crossed here rather than bundled.

  which to ask   arrival  every pair it has never checked, first come first served
                 random   a coin flip on each unchecked firing
                 decision only where it does not yet know what to do at all
                 novel    only where the structure has no opinion, so the
                          generalisation covers the rest
  what to assume own      an unchecked pair did not advance
                 class    adopt the conservative class's verdict when it has one

`novel` and the `class` fallback are the sharp form of this project's original
question: does building the automaton reduce how much the agent has to ask? Both
need a partition that actually generalises, so every arm that consults one
consults the same blue-fringe partition.

The informative band is 40 to 320 queries, which Step 23 measured as the whole
transition from total failure to saturation.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
LEVELS = [40, 80, 160, 320]
POLICIES = ['arrival', 'random', 'decision', 'novel']
FALLBACKS = ['own', 'class']


def one(job):
    tag, level, pol, fb, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--method', 'goal', '--goal-select', 'learned', '--seed', str(seed),
           '--budget', str(BUDGET), '--every', '250',
           '--theta', '20', '--quota', '1', '--strict-children', '0',
           '--merge-rule', 'edsm', '--verify-budget', str(level),
           '--verify-policy', pol, '--verify-fallback', fb, '--verify-rate', '0.5']
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'level': level, 'policy': pol, 'fallback': fb, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'used': r['verifications'], 'declined': r['declined'],
            'generalised': r['generalised_firings'], 'unknown': r['unknown_firings'],
            'nodes': r['nodes'], 'classes': r['classes']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], lv, p, fb, s) for f in fams for lv in LEVELS
            for p in POLICIES for fb in FALLBACKS for s in range(9800, 9806)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 400 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/query.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_query.py', families=len(fams), levels=LEVELS,
        policies=POLICIES, fallbacks=FALLBACKS, rate=0.5, merge_rule='edsm',
        seeds=6, seed0=9800, budget=BUDGET, runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
