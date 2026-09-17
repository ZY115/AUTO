"""What does the task structure cost when checking progress is not free?

Every experiment up to here handed the agent a progress signal on every step: the
task state changed, and it was told. That is why the goal reading looked so
strong and why knowing the task in advance looked worth so little — the structure
was being given away. Here the event labels stay visible and accurate, the
terminal signal stays free, and only one question is metered: *did that firing
advance the task?*

A pair the agent has not paid to check stays **unknown**, not "ignored": nothing
is recorded, and the agent acts as though it did not advance. That guess is its
own, so the history is never filtered by a state change it cannot see.

The reference arm is given the rules outright and needs no verification at all,
so the gap between the two lines is what the feedback was worth.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']
BUDGET = 200000
LEVELS = [0, 5, 10, 20, 40, 80, 160, 320, 640, -1]


def one(job):
    tag, level, seed, arm = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/ceiling/{tag}.task'),
           '--method', 'goal', '--seed', str(seed), '--budget', str(BUDGET),
           '--every', '250'] + MERGE
    if arm == 'learned':
        cmd += ['--goal-select', 'learned', '--verify-budget', str(level)]
    else:
        # Given the rules, so it never asks the question and is unmetered.
        cmd += ['--goal-select', 'value_ranked']
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'level': level, 'seed': seed, 'arm': arm,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'verifications': r['verifications'], 'unknown': r['unknown_firings'],
            'nodes': r['nodes'], 'first_success': r['first_success']}


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    jobs = [(f['tag'], lv, s, 'learned') for f in fams for lv in LEVELS
            for s in range(9700, 9706)]
    jobs += [(f['tag'], -1, s, 'given') for f in fams for s in range(9700, 9706)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            rows.append(r)
            if i % 200 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/verify.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_verify.py', families=len(fams), levels=LEVELS, seeds=6,
        seed0=9700, budget=BUDGET, every=250, runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
