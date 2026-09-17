"""Which part of a transferred structure is task knowledge, and which is the
source map's taste?

The tree carries two different things at once. That an event advanced after a
history is a fact about the task. How *often* it advanced is a fact about the
source map's layout and the policy that collected the data, and the program was
using that count to pick the next goal on a map where every skill had been reset.
Two genuinely equivalent histories can both allow A and B with counts (9,1) and
(1,99); no merge is wrong, and the decision still changes.

So: same frozen batch, same partition, same target-side selection rule, and the
source counts either may or may not narrow the candidate set. Everything else is
held fixed, including the ranking machinery itself, which the "pure" arms still
use with counts accumulated on the target map.

The oracle arm uses the true allowed set through the same selection rule, so that
"perfect rules" and "learned rules" differ in one thing rather than two.
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
    'scratch':     ('learned', False, 1),
    'dict_freq':   ('learned', True, 1),
    'dict_pure':   ('learned', True, 0),
    'struct_freq': ('merged', True, 1),
    'struct_pure': ('merged', True, 0),
    'oracle':      ('value_ranked', False, 1),
}


def one(job):
    pair, source_seed, target_seed, every, folder = job
    src, tgt = pair['mapA']['tag'], pair['mapB']['tag']
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'seed.txt')
        raw = subprocess.check_output(
            [str(ROOT / 'compress'), '--task', str(ROOT / f'results/{folder}/{src}.task'),
             '--method', 'merged', '--seed', str(source_seed),
             '--budget', str(SOURCE_BUDGET), '--every', str(SOURCE_BUDGET),
             '--dump-tree', '1', '--theta', '20', '--quota', '10000000',
             '--strict-children', '0'], text=True)
        Path(path + '.json').write_text(raw)
        nodes, _ = convert(path + '.json', path)
        for arm, (sel, seeded, freq) in ARMS.items():
            cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/{folder}/{tgt}.task'),
                   '--method', 'goal', '--goal-select', sel, '--seed', str(target_seed),
                   '--budget', str(TARGET_BUDGET), '--every', str(every),
                   '--source-frequency', str(freq)] + MERGE
            if seeded:
                cmd += ['--seed-structure', path]
            r = json.loads(subprocess.check_output(cmd, text=True))
            ck = [c for c in r['checkpoints'] if c[0] >= TARGET_BUDGET * .8 and c[1] > 0]
            rows.append({'pair': src, 'overlap': pair['overlap'], 'band': pair['band'],
                         'source_seed': source_seed, 'seed': target_seed, 'arm': arm,
                         'first90': r['first90'] if r['first90'] > 0 else TARGET_BUDGET,
                         'auc': r['auc'], 'final': r['final_success'],
                         'failures': r['failures'],
                         'rank_decisions': r['rank_decisions'],
                         'rank_changed': r['rank_changed'],
                         'route': sum(c[2] for c in ck) / len(ck) if ck else None,
                         'seed_nodes': nodes if seeded else 0})
    return rows


def main(stage='pilot', source_seeds=4, target_seeds=4, every=25):
    source_seeds, target_seeds, every = int(source_seeds), int(target_seeds), int(every)
    # Mechanism pilot on a few high- and low-overlap pairs of the development
    # batch; confirmation on a batch drawn from map seeds the pilot never saw,
    # because only five low-overlap pairs exist in the development batch and the
    # effect lives there.
    if stage == 'pilot':
        pairs = json.loads((ROOT / 'results/overlap/pairs.json').read_text())
        keep = ([p for p in pairs if p['band'] == 'high'][:5]
                + [p for p in pairs if p['band'] == 'low'][:5])
        folder = 'overlap'
    else:
        keep = json.loads((ROOT / 'results/overlap2/pairs.json').read_text())
        folder = 'overlap2'
    jobs = [(p, 600 + i, 900 + j, every, folder) for p in keep
            for i in range(source_seeds) for j in range(target_seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=1)):
            rows.extend(r)
            if i % 40 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    out = ROOT / f'results/purity_{stage}.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps(dict(
        runner='src/run_purity.py', stage=stage, pairs=len(keep), folder=folder,
        source_seeds=source_seeds, target_seeds=target_seeds, every=every,
        source_budget=SOURCE_BUDGET, target_budget=TARGET_BUDGET, arms=list(ARMS),
        runs=len(rows),
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print('wrote', len(rows), 'to', out.name)


if __name__ == '__main__':
    main(*sys.argv[1:])
