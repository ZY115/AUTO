"""P3: the core comparison, rerun on a public domain.

OfficeWorld, exported exactly by `src/export_officeworld.py`: the same grid,
walls, labels and ground-truth automaton the ISA papers use. Two layers, as the
review asked.

Layer one keeps the standard task and standard feedback and asks whether the
reading still separates. Layer two meters progress verification and asks what the
labels are worth. Layer two must be called **a paid-feedback extension on a public
domain**, never "the query saving is validated on the published benchmark".

Arms are grouped by what they are given, because a method handed the rules may
not be compared with one that has to learn them:

  known rules     goal/value_ranked, automaton index, index+CRM
  unknown rules   goal/learned at several verification budgets
  no task memory  raw event history, free
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 400000
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']
ARMS = {
    # known rules
    'oracle_goal':  (['--method', 'goal', '--goal-select', 'value_ranked'] + MERGE, -1),
    'oracle_index': (['--method', 'automaton'], -1),
    'oracle_crm':   (['--method', 'automaton', '--crm', '1'], -1),
    # unknown rules
    'learned_free': (['--method', 'goal', '--goal-select', 'learned'] + MERGE, -1),
    'learned_640':  (['--method', 'goal', '--goal-select', 'learned'] + MERGE, 640),
    'learned_320':  (['--method', 'goal', '--goal-select', 'learned'] + MERGE, 320),
    'learned_160':  (['--method', 'goal', '--goal-select', 'learned'] + MERGE, 160),
    'learned_novel_640': (['--method', 'goal', '--goal-select', 'learned',
                           '--merge-rule', 'edsm', '--verify-policy', 'novel'] + MERGE, 640),
    # no task memory
    'raw_free':     (['--method', 'rawhistory'], 0),
}


def one(job):
    tag, arm, seed = job
    extra, vb = ARMS[arm]
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/officeworld/{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', '500',
           '--verify-budget', str(vb)] + extra
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'arm': arm, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'first_success': r['first_success'], 'used': r['verifications'],
            'states': r['features'], 'failures': r['failures']}


if __name__ == '__main__':
    tasks = [t['tag'] for t in json.loads((ROOT / 'results/officeworld/tasks.json').read_text())]
    jobs = [(t, a, s) for t in tasks for a in ARMS for s in range(10200, 10206)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            rows.append(r)
            if i % 100 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / 'results/p3_officeworld.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_p3.py', domain='OfficeWorld (gym-subgoal-automata)',
        exporter='src/export_officeworld.py', tasks=len(tasks), arms=list(ARMS),
        seeds=6, seed0=10200, budget=BUDGET, runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))
