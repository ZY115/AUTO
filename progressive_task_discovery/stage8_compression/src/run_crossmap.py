"""The minimal cross-map comparison: what task knowledge is worth carrying, and
how much of it should be believed.

Task semantics and event labels are fixed across the pair. Everything
map-specific restarts at zero on the target: navigation skills, the map model,
every value table. Only task knowledge crosses, and the arms differ in what that
knowledge is and how far it is trusted.

  scratch      nothing crosses
  dict         the frozen source tree, consulted by exact history
  structure    the same tree, folded through the conservative partition
  aggressive   the same tree, folded through the over-merging partition
  selective    the same tree and both partitions, used in tiers: observed
               evidence first, then what the conservative class rules out, and
               the over-merging partition's extra answers only while nothing
               fatal has ever been observed
  oracle       the true task machine

Groups two to five share one frozen batch per source seed, so nothing separates
them except the use made of it. Verification spends the target budget like any
other step; there are no free queries.
"""
import hashlib, json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from structure import convert

MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']
SOURCE_BUDGET = 40000
TARGET_BUDGET = 100000
ARMS = {
    'scratch':    ('learned', False),
    'dict':       ('learned', True),
    'structure':  ('merged', True),
    'aggressive': ('aggressive', True),
    'selective':  ('selective', True),
    'oracle':     ('value', False),
}


def seed_file(tag, source_seed, out):
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/overlap/{tag}.task'),
           '--method', 'merged', '--seed', str(source_seed), '--budget', str(SOURCE_BUDGET),
           '--every', str(SOURCE_BUDGET), '--dump-tree', '1',
           '--theta', '20', '--quota', '10000000', '--strict-children', '0']
    raw = subprocess.check_output(cmd, text=True)
    Path(out + '.json').write_text(raw)
    return convert(out + '.json', out)


def one(job):
    pair, source_seed, target_seed, every = job
    src, tgt = pair['mapA']['tag'], pair['mapB']['tag']
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'seed.txt')
        nodes, _ = seed_file(src, source_seed, path)
        for arm, (sel, seeded) in ARMS.items():
            cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/overlap/{tgt}.task'),
                   '--method', 'goal', '--goal-select', sel, '--seed', str(target_seed),
                   '--budget', str(TARGET_BUDGET), '--every', str(every)] + MERGE
            if seeded:
                cmd += ['--seed-structure', path]
            r = json.loads(subprocess.check_output(cmd, text=True))
            first90 = r['first90'] if r['first90'] > 0 else TARGET_BUDGET
            rows.append({'pair': src, 'overlap': pair['overlap'], 'band': pair['band'],
                         'source_seed': source_seed, 'seed': target_seed, 'arm': arm,
                         'first90': first90, 'auc': r['auc'], 'final': r['final_success'],
                         'first_success': r['first_success'], 'failures': r['failures'],
                         'seed_nodes': nodes if seeded else 0,
                         # Total cost charges the source sampling to the arms that
                         # used it, which is the only comparison a practitioner
                         # could act on.
                         'total_cost': first90 + (SOURCE_BUDGET if seeded else 0)})
    return rows


def main(source_seeds=4, target_seeds=4, out='results/crossmap.jsonl', every=250):
    source_seeds, target_seeds, every = int(source_seeds), int(target_seeds), int(every)
    pairs = json.loads((ROOT / 'results/overlap/pairs.json').read_text())
    jobs = [(p, 700 + i, 800 + j, every) for p in pairs
            for i in range(source_seeds) for j in range(target_seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.extend(r)
            if i % 50 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    (ROOT / out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (ROOT / out).with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_crossmap.py', pairs=len(pairs), source_seeds=source_seeds,
        target_seeds=target_seeds, source_budget=SOURCE_BUDGET, target_budget=TARGET_BUDGET,
        arms=list(ARMS), runs=len(rows), every=every,
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:])
