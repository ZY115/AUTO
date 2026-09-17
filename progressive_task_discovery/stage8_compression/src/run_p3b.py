"""P3 follow-up: OfficeWorld's own scarce-query region.

The first sweep used 160, 320 and 640 on a domain that needs about 17 queries, so
every level was in the saturated region and they came out identical. That says
the budget never bound; it does not say the domain has no transition. This sweeps
0, 4, 8, 16, 32, 64 to find where it actually is, and asks the same question
there: when queries are genuinely scarce, does sharing evidence change anything?
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 400000
LEVELS = [0, 4, 8, 16, 32, 64, 160]
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']
ARMS = {
    'arrival': ['--verify-policy', 'arrival'],
    'novel':   ['--verify-policy', 'novel', '--merge-rule', 'edsm'],
    'truestate_cache': ['--verify-policy', 'arrival', '--cache-key', 'truestate'],
}


def one(job):
    tag, arm, lv, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/officeworld/{tag}.task'),
           '--method', 'goal', '--goal-select', 'learned', '--seed', str(seed),
           '--budget', str(BUDGET), '--every', '500',
           '--verify-budget', str(lv)] + MERGE + ARMS[arm]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'arm': arm, 'level': lv, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'], 'used': r['verifications'],
            'nodes': r['nodes'], 'desync': r['desync_steps'],
            'first_success': r['first_success']}


if __name__ == '__main__':
    tasks = [t['tag'] for t in json.loads((ROOT / 'results/officeworld/tasks.json').read_text())]
    jobs = [(t, a, lv, s) for t in tasks for a in ARMS for lv in LEVELS
            for s in range(10400, 10406)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            rows.append(r)
            if i % 200 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/p3_small_budgets.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_p3b.py', domain='OfficeWorld-derived protocol',
        tasks=len(tasks), arms=list(ARMS), levels=LEVELS, seeds=6, seed0=10400,
        budget=BUDGET, runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
