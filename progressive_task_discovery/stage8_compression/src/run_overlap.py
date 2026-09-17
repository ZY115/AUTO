"""Transfer benefit as a function of how novel the target map's orderings are.

Adds the control the literature note asks for: carry the same source evidence
but represent it without merging, so "having seen the source events" is
separated from "having generalised them". Merging is switched off by a quota no
run can meet, which leaves the seeded prefix tree in place while forbidding any
class to form.
"""
import collections, concurrent.futures, hashlib, json, random, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from structure import convert

MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0', '--revision', 'transfer']
NOMERGE = ['--theta', '20', '--quota', '1000000', '--strict-children', '0', '--revision', 'transfer']


def compress(task, extra, seed, budget, every, dump=False):
    cmd = [str(ROOT / 'compress'), '--task', str(task), '--seed', str(seed),
           '--budget', str(budget), '--every', str(every)] + extra
    if dump:
        cmd += ['--dump-tree', '1']
    return json.loads(subprocess.check_output(cmd, text=True))


def job(args):
    entry, arm, seed, source_budget, budget = args
    a = ROOT / f"results/overlap/{entry['mapA']['tag']}.task"
    b = ROOT / f"results/overlap/{entry['mapB']['tag']}.task"
    rec = dict(task=entry['task'], band=entry['band'], overlap=entry['overlap'],
               pair=f"{entry['mapA']['tag']}->{entry['mapB']['tag']}", arm=arm, seed=seed)
    if arm.startswith('transfer'):
        flags = MERGE if arm == 'transfer_merged' else NOMERGE
        with tempfile.TemporaryDirectory() as tmp:
            out = compress(a, ['--method', 'merged'] + MERGE, seed, source_budget,
                           source_budget, dump=True)
            dumped = Path(tmp) / 'dump.json'
            dumped.write_text(json.dumps(out))
            table = Path(tmp) / 'structure.txt'
            nodes, accepting = convert(dumped, table)
            rec.update(source_nodes=nodes, source_accepting=accepting,
                       source_precision=out['merge_precision'], source_recall=out['merge_recall'])
            res = compress(b, ['--method', 'merged'] + flags + ['--seed-structure', str(table)],
                           seed, budget, 1000)
    elif arm == 'cold_merged':
        res = compress(b, ['--method', 'merged'] + MERGE, seed, budget, 1000)
    elif arm == 'cold_history':
        res = compress(b, ['--method', 'history'], seed, budget, 1000)
    else:
        res = compress(b, ['--method', 'automaton'], seed, budget, 1000)
    rec.update({k: res[k] for k in ('first90', 'auc', 'final_success', 'first_success',
                                    'nodes', 'classes', 'merge_precision', 'merge_recall',
                                    'revisions', 'true_reached', 'seeded_nodes', 'budget')})
    return rec


def main(name='overlap_run', source_budget=10000, budget=200000, seeds=16, seed0=3000, workers=8):
    source_budget, budget, seeds, seed0, workers = (int(source_budget), int(budget),
                                                    int(seeds), int(seed0), int(workers))
    pairs = json.loads((ROOT / 'results/overlap/pairs.json').read_text())
    arms = ['cold_history', 'cold_merged', 'transfer_prefix', 'transfer_merged', 'oracle']
    jobs = [(e, arm, s, source_budget, budget)
            for e in pairs for arm in arms for s in range(seed0, seed0 + seeds)]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, jobs))
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'manifest.json').write_text(json.dumps(dict(
        runs=len(rows), pairs=len(pairs), arms=arms, seeds=seeds, seed0=seed0,
        source_budget=source_budget, budget=budget,
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print(f'{len(rows)} runs in {round(time.monotonic() - started, 1)}s\n')

    g = collections.defaultdict(list)
    for r in rows:
        g[(r['pair'], r['arm'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    bands = ['high', 'mid', 'low']
    print(f"{'重合档':>6} {'组数':>4} {'重合度':>6} " + " ".join(f'{a:>16}' for a in arms))
    for band in bands:
        ps = [e for e in pairs if e['band'] == band]
        if not ps:
            continue
        keys = [f"{e['mapA']['tag']}->{e['mapB']['tag']}" for e in ps]
        cells = [statistics.mean(cap(g[(k, a)]) for k in keys) for a in arms]
        print(f"{band:>6} {len(ps):>4} {statistics.mean(e['overlap'] for e in ps):>6.2f} "
              + " ".join(f'{c:>16.0f}' for c in cells))

    print(f"\n迁移相对在新地图从零学结构，按重合档")
    for band in bands:
        ps = [e for e in pairs if e['band'] == band]
        if not ps:
            continue
        keys = [f"{e['mapA']['tag']}->{e['mapB']['tag']}" for e in ps]
        cold = {k: [(x['first90'] if x['first90'] > 0 else x['budget']) for x in g[(k, 'cold_merged')]] for k in keys}
        for arm in ('transfer_prefix', 'transfer_merged'):
            hot = {k: [(x['first90'] if x['first90'] > 0 else x['budget']) for x in g[(k, arm)]] for k in keys}
            rng = random.Random(61)
            point = statistics.mean(v for k in keys for v in cold[k]) / statistics.mean(v for k in keys for v in hot[k])
            draws = []
            for _ in range(10000):
                pick = [keys[rng.randrange(len(keys))] for _ in keys]
                draws.append(statistics.mean(v for k in pick for v in cold[k]) /
                             statistics.mean(v for k in pick for v in hot[k]))
            draws.sort()
            print(f'  {band:>5} {arm:>17} {point:>5.2f}  [{draws[250]:.2f}, {draws[9750]:.2f}]')

    print(f"\n目标地图上被推翻的合并，以及结构质量")
    print(f"{'档':>6} {'迁移臂修订次数':>12} {'从零臂修订次数':>12} {'迁移精度':>8} {'从零精度':>8}")
    for band in bands:
        ps = [e for e in pairs if e['band'] == band]
        if not ps:
            continue
        keys = [f"{e['mapA']['tag']}->{e['mapB']['tag']}" for e in ps]
        t = [x for k in keys for x in g[(k, 'transfer_merged')]]
        c = [x for k in keys for x in g[(k, 'cold_merged')]]
        print(f"{band:>6} {statistics.mean(x['revisions'] for x in t):>12.2f} "
              f"{statistics.mean(x['revisions'] for x in c):>12.2f} "
              f"{statistics.mean(x['merge_precision'] for x in t):>8.2f} "
              f"{statistics.mean(x['merge_precision'] for x in c):>8.2f}")


if __name__ == '__main__':
    main(*sys.argv[1:])
