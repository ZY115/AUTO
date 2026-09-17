"""Is knowing the task worth anything anywhere in the reachable design space?

Step 20 answered the cross-map question negatively and then found the reason: the
whole prize was 1.05, because the goal reading learns this task on a new map in
about nine hundred steps. The prize is one free division, oracle over scratch, and
it should have been computed before that experiment rather than after.

So compute it, across the family parameters this generator can reach, and across
both readings. Nothing here trains a new algorithm: two arms per family, and the
ratio between them is the ceiling on anything transfer or structural inference
could ever buy in that family.
"""
import itertools, json, os, subprocess, sys, statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_maps import Task, evaluate_layout
from spoke_maps import build_spokes, funnel_ok
from emit_tasks import emit

OUT = ROOT / 'results/ceiling'
BUDGET = 300000
# Two readings, two arms each. The ratio inside a reading is what a perfect task
# structure is worth there; comparing the ratios says whether a low ceiling is a
# property of the family or of how the structure is consumed.
ARMS = {
    # The oracle Step 21 used, kept so the correction is visible: it takes the
    # true allowed set and picks by skill value, skipping the narrowing every
    # learned arm does. It therefore differed from them in two ways at once.
    'goal_oracle_old': ['--method', 'goal', '--goal-select', 'value'],
    # The matched oracle: true rules, same selection rule as the learned arms.
    'goal_oracle':  ['--method', 'goal', '--goal-select', 'value_ranked',
                     '--theta', '20', '--quota', '1', '--strict-children', '0'],
    'goal_scratch': ['--method', 'goal', '--goal-select', 'learned',
                     '--theta', '20', '--quota', '1', '--strict-children', '0'],
    'index_oracle': ['--method', 'automaton'],
    'index_scratch': ['--method', 'history'],
}


def build_family(n, d, m, m2, L, arm, seeds=range(90000, 90060)):
    """First map seed that yields a valid, count-aliased layout for this task."""
    pair = (0, 1)
    task = Task(n, pair, d, m, m2)
    for seed in seeds:
        layout = build_spokes(n, L, seed, arm=arm)
        if layout is None or not funnel_ok(layout):
            continue
        info = evaluate_layout(layout, task, 24)
        if info is None or not info['count_aliased'] or info['orders'] < 4:
            continue
        return dict(n=n, d=d, m=m, m2=m2, pair=list(pair), L=L, arm=arm, seed=seed)
    return None


def run(job):
    tag, arm, seed = job
    cmd = [str(ROOT / 'compress'), '--task', str(OUT / f'{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', '250'] + ARMS[arm]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'tag': tag, 'arm': arm, 'seed': seed,
            'first90': r['first90'] if r['first90'] > 0 else BUDGET,
            'auc': r['auc'], 'final': r['final_success'],
            'states': r['features'], 'horizon': r['horizon']}


def main(seeds=8):
    seeds = int(seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    grid = []
    # n: how many events may be done in any order. d: bridge length.
    # m/m2: the two tail lengths. L: cells per labelled corridor. arm: corridor
    # length, which is the physical cost of visiting a label.
    for n in (3, 4, 5):
        for d in (2, 4):
            for m, m2 in ((1, 4), (3, 6)):
                for L, arm in ((2, 7), (3, 11)):
                    grid.append((n, d, m, m2, L, arm))
    built = []
    for cfg in grid:
        got = build_family(*cfg)
        if got is None:
            print('no layout for', cfg, file=sys.stderr)
            continue
        tag = 'c_n{n}d{d}m{m}_{m2}L{L}a{arm}'.format(**got)
        emit(dict(got, tag=tag, target=tag), OUT)
        got['tag'] = tag
        built.append(got)
    (OUT / 'families.json').write_text(json.dumps(built, indent=2) + '\n')
    print('built', len(built), 'families', file=sys.stderr)
    jobs = [(b['tag'], a, s) for b in built for a in ARMS for s in range(9500, 9500 + seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(run, jobs, chunksize=2)):
            rows.append(r)
            if i % 100 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    (ROOT / 'results/ceiling.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('wrote', len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:])
