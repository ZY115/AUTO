"""P3 under the published protocol, after a review showed the derived one biased it.

The first OfficeWorld run used a balanced start set, a horizon of twice the
shortest path, a negative step cost and epsilon .02. The native environment gives
1 on success and 0 otherwise, a single starting cell, an episode length of 250,
and ISA's configuration uses epsilon .1 and learning rate .1. Under the derived
protocol CRM solved 0/90; under the published one it solves. The negative step
cost combined with fast failure termination was biasing early exploration, which
the review flagged as a hypothesis and this tests.

Everything here uses the published reward, exploration, start and horizon, so the
arms differ only in how they consume the task structure.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 600000
NATIVE = ['--cost', '0', '--reward-success', '1', '--epsilon', '0.1', '--alpha', '0.1']
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']
ARMS = {
    'oracle_goal':  ['--method', 'goal', '--goal-select', 'value_ranked'] + MERGE,
    'oracle_index': ['--method', 'automaton'],
    'oracle_crm':   ['--method', 'automaton', '--crm', '1'],
    'learned_goal': ['--method', 'goal', '--goal-select', 'learned'] + MERGE,
    'learned_merged': ['--method', 'merged'] + MERGE,
    'history':      ['--method', 'history'],
}


def one(job):
    tag, arm, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/officeworld/{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', '1000'] + NATIVE + ARMS[arm]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'arm': arm, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'censored': r['first90'] <= 0,
            'auc': r['auc'], 'final': r['final_success'],
            'first_success': r['first_success'], 'ever': r['first_success'] > 0,
            'failures': r['failures']}


if __name__ == '__main__':
    tasks = [t['tag'] for t in json.loads((ROOT / 'results/officeworld/tasks.json').read_text())
             if t.get('native')]
    jobs = [(t, a, s) for t in tasks for a in ARMS for s in range(10500, 10508)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            rows.append(r)
            if i % 100 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/p3_native.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_p3c.py', protocol='published OfficeWorld reward/start/horizon, ISA exploration',
        tasks=len(tasks), arms=list(ARMS), seeds=8, seed0=10500, budget=BUDGET,
        runs=len(rows),
        note='supersedes results/p3_officeworld.jsonl for cross-method claims; that run used a derived protocol',
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
