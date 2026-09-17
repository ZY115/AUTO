"""Separate two things an automaton could be doing: compressing states, and
telling the agent which (state, event) pairs it has not yet tried."""
import collections, concurrent.futures, hashlib, json, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPLORE = {'beta': .15, 'target': 'scarcest', 'quota': 1}
SHAPE = {'shape': 'learned', 'shape_scale': .1}          # chosen on development AUC
SHAPE_ORACLE = {'shape': 'oracle', 'shape_scale': .05}   # same criterion, reference only

ARMS = [
    ('history', 'history', {'beta': 0}),
    ('history+random', 'history', {'beta': .15, 'target': 'random'}),
    ('history+scarcest', 'history', dict(EXPLORE)),
    ('history+shape', 'history', dict(SHAPE)),
    ('history+shape+scarcest', 'history', {**EXPLORE, **SHAPE}),
    ('automaton', 'automaton', {'beta': 0}),
    ('automaton+random', 'automaton', {'beta': .15, 'target': 'random'}),
    ('automaton+scarcest', 'automaton', dict(EXPLORE)),
    ('automaton+shape+scarcest', 'automaton', {**EXPLORE, **SHAPE}),
    ('history+shape_oracle', 'history', dict(SHAPE_ORACLE)),
    ('merged+scarcest', 'merged_explore', {**EXPLORE, 'theta': 200}),
]


def run(job):
    tag, arm, method, kw, seed, budget = job
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/family_maps/{tag}.task'),
           '--method', method, '--seed', str(seed), '--budget', str(budget), '--every', '1000']
    for k, v in kw.items():
        cmd += ['--' + k.replace('_', '-'), str(v)]
    rec = json.loads(subprocess.check_output(cmd, text=True))
    rec.update(rung=tag, arm=arm)
    return rec


def main(name='explore', budget=200000, seeds=32, workers=8, seed0=300):
    budget, seeds, workers, seed0 = int(budget), int(seeds), int(workers), int(seed0)
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m['tag'], arm, method, kw, s, budget)
            for m in metas for arm, method, kw in ARMS
            for s in range(seed0, seed0 + seeds)]
    (out / 'spec.json').write_text(json.dumps(
        dict(name=name, budget=budget, seeds=seeds, seed0=seed0, arms=[a for a, _, _ in ARMS],
             jobs=len(jobs),
             source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    started = time.monotonic()
    with (out / 'raw.jsonl').open('w') as fh, \
         concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for rec in pool.map(run, jobs):
            fh.write(json.dumps(rec) + '\n')
            rows.append(rec)
    print(f'{len(jobs)} runs in {round(time.monotonic() - started, 1)}s')

    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    names = [a for a, _, _ in ARMS]
    print(f"{'臂':>26} {'均值步数':>9} {'均值AUC':>8} {'最终成功':>8} {'结构步占比':>9} {'达标':>9}")
    for n in names:
        rs = [x for m in metas for x in g[(m['tag'], n)]]
        per = [cap(g[(m['tag'], n)]) for m in metas]
        solved = sum(1 for x in rs if x['first90'] > 0)
        print(f"{n:>26} {statistics.mean(per):>9.0f} {statistics.mean(x['auc'] for x in rs):>8.3f}"
              f" {statistics.mean(x['final_success'] for x in rs):>8.2f}"
              f" {statistics.mean(x['structural_steps'] / x['budget'] for x in rs):>9.2f}"
              f" {solved:>4}/{len(rs)}")
    print()
    print(f"{'台阶':>9} " + " ".join(f"{n[:16]:>17}" for n in names))
    for m in metas:
        print(f"{m['tag']:>9} " + " ".join(f"{cap(g[(m['tag'], n)]):>17.0f}" for n in names))


if __name__ == '__main__':
    main(*sys.argv[1:])
