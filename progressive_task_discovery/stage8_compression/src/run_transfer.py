"""Does a learned task automaton transfer to a new map?

Phase one trains on map A and writes out the map-independent structure. Phase
two trains on map B, either from nothing or seeded with that structure. Only the
structure crosses; every table indexed by a cell starts at zero on map B, and so
does the map itself.
"""
import collections, concurrent.futures, hashlib, json, random, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from structure import convert

MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0', '--revision', 'transfer']


def compress(task, extra, seed, budget, every=1000, dump=None):
    cmd = [str(ROOT / 'compress'), '--task', str(task), '--seed', str(seed),
           '--budget', str(budget), '--every', str(every)] + extra
    if dump:
        cmd += ['--dump-tree', '1']
    return json.loads(subprocess.check_output(cmd, text=True))


def job(args):
    entry, arm, seed, source_budget, budget = args
    a = ROOT / f"results/transfer/{entry['mapA']['tag']}.task"
    b = ROOT / f"results/transfer/{entry['mapB']['tag']}.task"
    record = {'pair_id': entry['pair_id'], 'arm': arm, 'seed': seed}
    if arm == 'transfer':
        with tempfile.TemporaryDirectory() as tmp:
            dumped = Path(tmp) / 'dump.json'
            out = compress(a, ['--method', 'merged'] + MERGE, seed, source_budget, source_budget, dump=True)
            dumped.write_text(json.dumps(out))
            table = Path(tmp) / 'structure.txt'
            nodes, accepting = convert(dumped, table)
            record.update(source_nodes=nodes, source_accepting=accepting,
                          source_budget=source_budget,
                          source_precision=out['merge_precision'], source_recall=out['merge_recall'])
            res = compress(b, ['--method', 'merged'] + MERGE + ['--seed-structure', str(table)],
                           seed, budget)
    elif arm == 'cold_merged':
        res = compress(b, ['--method', 'merged'] + MERGE, seed, budget)
    elif arm == 'cold_history':
        res = compress(b, ['--method', 'history'], seed, budget)
    elif arm == 'oracle':
        res = compress(b, ['--method', 'automaton'], seed, budget)
    else:
        raise ValueError(arm)
    record.update({k: res[k] for k in ('first90', 'stable90', 'auc', 'final_success',
                                       'first_success', 'features', 'nodes', 'classes',
                                       'merge_precision', 'merge_recall', 'true_reached',
                                       'seeded_nodes', 'updates', 'budget')})
    return record


def main(name='transfer_run', source_budget=200000, budget=200000, seeds=32, seed0=2000, workers=8):
    source_budget, budget, seeds, seed0, workers = (int(source_budget), int(budget),
                                                    int(seeds), int(seed0), int(workers))
    pairs = json.loads((ROOT / 'results/transfer/pairs.json').read_text())
    arms = ['cold_history', 'cold_merged', 'transfer', 'oracle']
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
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((ROOT / 'compress').read_bytes()).hexdigest()), indent=2) + '\n')
    print(f'{len(rows)} runs in {round(time.monotonic() - started, 1)}s')

    g = collections.defaultdict(list)
    for r in rows:
        g[(r['pair_id'], r['arm'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    ids = [e['pair_id'] for e in pairs]
    print(f"\n在地图 B 上的表现，{len(pairs)} 对地图 × {seeds} 个种子")
    print(f"{'臂':>14} {'达标步数':>9} {'首次成功':>9} {'AUC':>7} {'最终':>6} {'精度':>6} {'召回':>6}")
    for arm in arms:
        rs = [x for i in ids for x in g[(i, arm)]]
        per = [cap(g[(i, arm)]) for i in ids]
        extra = ''
        if 'merged' in arm or arm == 'transfer':
            extra = (f"{statistics.mean(x['merge_precision'] for x in rs):>6.2f} "
                     f"{statistics.mean(x['merge_recall'] for x in rs):>6.2f}")
        print(f"{arm:>14} {statistics.mean(per):>9.0f} "
              f"{statistics.mean(x['first_success'] for x in rs if x['first_success'] > 0):>9.0f} "
              f"{statistics.mean(x['auc'] for x in rs):>7.3f} "
              f"{statistics.mean(x['final_success'] for x in rs):>6.3f} {extra}")

    def blocks(arm):
        return {i: [(x['first90'] if x['first90'] > 0 else x['budget']) for x in g[(i, arm)]] for i in ids}

    def pb(a, b, boots=10000, seed=53):
        rng = random.Random(seed)
        point = statistics.mean(v for i in ids for v in a[i]) / statistics.mean(v for i in ids for v in b[i])
        draws = []
        for _ in range(boots):
            pick = [ids[rng.randrange(len(ids))] for _ in ids]
            draws.append(statistics.mean(v for i in pick for v in a[i]) /
                         statistics.mean(v for i in pick for v in b[i]))
        draws.sort()
        return point, draws[int(.025 * boots)], draws[int(.975 * boots)]

    print('\n按地图对成块的配对 bootstrap')
    for a, b, note in [('cold_history', 'cold_merged', '在新地图上从零学结构'),
                       ('cold_merged', 'transfer', '把旧地图学到的结构搬过来'),
                       ('transfer', 'oracle', '搬来的结构距离真实自动机'),
                       ('cold_history', 'oracle', '真实自动机相对完整历史')]:
        p, lo, hi = pb(blocks(a), blocks(b))
        print(f'  {note:>24} {p:>5.2f}  [{lo:.2f}, {hi:.2f}]')


if __name__ == '__main__':
    main(*sys.argv[1:])
