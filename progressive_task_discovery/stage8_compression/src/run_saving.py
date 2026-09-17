"""How far can the structure cut the number of questions?

Step 24 got 21% with a crude rule: do not ask where the class already has an
explicit opinion. Two things were wrong with it. It skipped even where the class
said the event *advances*, and skipping means believing it did not, so the agent
acted against its own model exactly where the model was most informative. And it
ignored the closed-world reading of the same evidence, which Step 19 measured as
the single largest source of correct answers.

  arrival  ask about every pair never checked
  novel    skip where the class has any explicit opinion (Step 24's rule)
  aligned  skip only where the class says the event does NOT advance, so the
           skip and the fallback agree
  closed   aligned, plus the closed-world completion: in a class that has seen
           something advance, an event never seen to advance is asserted not to

crossed with which partition is consulted (conservative, over-merging, and the
true one as a privileged ceiling) and how much evidence a "does not advance"
verdict needs behind it.

A saving only counts if the outcome does not degrade, so steps to 90% and the
solved count are reported beside the query count every time.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 200000
LEVELS = [160, 320]
POLICIES = ['arrival', 'novel', 'aligned', 'closed']
PARTS = ['sweep', 'aggressive', 'true']
SUPPORTS = [1, 3]


def one(job):
    tag, lv, pol, part, sup, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--method', 'goal', '--goal-select', 'learned', '--seed', str(seed),
           '--budget', str(BUDGET), '--every', '250',
           '--theta', '20', '--quota', '1', '--strict-children', '0',
           '--merge-rule', 'edsm', '--verify-budget', str(lv),
           '--verify-policy', pol, '--verify-partition', part,
           '--verify-support', str(sup)]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'level': lv, 'policy': pol, 'part': part, 'support': sup,
            'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'used': r['verifications'], 'unknown': r['unknown_firings'],
            'nodes': r['nodes']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], lv, p, pt, sup, s) for f in fams for lv in LEVELS
            for p in POLICIES for pt in PARTS for sup in SUPPORTS
            for s in range(9900, 9904)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows.append(r)
            if i % 400 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/saving.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_saving.py', families=len(fams), levels=LEVELS,
        policies=POLICIES, partitions=PARTS, supports=SUPPORTS, seeds=4,
        seed0=9900, budget=BUDGET, merge_rule='edsm', runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
