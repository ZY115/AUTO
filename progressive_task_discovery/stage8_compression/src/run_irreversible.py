"""Same task, same map, two rules: a wrong commit ignored or a wrong commit fatal.

The commit point is the one place where the two histories stand on the same cell
with the same progress count, so an unrecoverable mistake there is a memory
mistake. Navigation mistakes stay free in both versions.
"""
import collections, concurrent.futures, hashlib, json, random, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0', '--revision', 'transfer']
ARMS = [('count', ['--method', 'count']),
        ('history', ['--method', 'history']),
        ('merged', ['--method', 'merged'] + MERGE),
        ('oracle', ['--method', 'automaton'])]


def job(args):
    entry, kind, arm, extra, seed, budget = args
    task = ROOT / f"results/irreversible/{entry[kind]['tag']}.task"
    cmd = [str(ROOT / 'compress'), '--task', str(task), '--seed', str(seed),
           '--budget', str(budget), '--every', '1000'] + extra
    r = json.loads(subprocess.check_output(cmd, text=True))
    r.update(task_name=entry['task'], map_seed=entry['seed'], kind=kind, arm=arm)
    return r


def main(name='irreversible_run', budget=200000, seeds=16, seed0=4000, workers=8):
    budget, seeds, seed0, workers = int(budget), int(seeds), int(seed0), int(workers)
    maps = json.loads((ROOT / 'results/irreversible/maps.json').read_text())
    jobs = [(e, kind, arm, extra, s, budget)
            for e in maps for kind in ('safe', 'fatal') for arm, extra in ARMS
            for s in range(seed0, seed0 + seeds)]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, jobs))
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'manifest.json').write_text(json.dumps(dict(
        runs=len(rows), maps=len(maps), arms=[a for a, _ in ARMS], seeds=seeds, seed0=seed0,
        budget=budget,
        source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2) + '\n')
    print(f'{len(rows)} runs in {round(time.monotonic() - started, 1)}s\n')

    g = collections.defaultdict(list)
    for r in rows:
        g[(r['map_seed'], r['kind'], r['arm'])].append(r)
    keys = sorted({r['map_seed'] for r in rows})
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    print(f"{'规则':>6} {'臂':>9} {'达标步数':>9} {'AUC':>7} {'最终':>6} {'失败次数':>9} "
          f"{'首次失败':>9} {'每次成功的失败代价':>12}")
    for kind in ('safe', 'fatal'):
        for arm, _ in ARMS:
            rs = [x for k in keys for x in g[(k, kind, arm)]]
            per = [cap(g[(k, kind, arm)]) for k in keys]
            fails = statistics.mean(x['failures'] for x in rs)
            ff = [x['first_failure'] for x in rs if x['first_failure'] > 0]
            eps = statistics.mean(x['episodes'] for x in rs)
            print(f"{kind:>6} {arm:>9} {statistics.mean(per):>9.0f} "
                  f"{statistics.mean(x['auc'] for x in rs):>7.3f} "
                  f"{statistics.mean(x['final_success'] for x in rs):>6.3f} {fails:>9.0f} "
                  f"{(statistics.mean(ff) if ff else 0):>9.0f} {fails / max(1, eps - fails):>12.3f}")
        print()

    def blocks(kind, arm):
        return {k: [(x['first90'] if x['first90'] > 0 else x['budget']) for x in g[(k, kind, arm)]]
                for k in keys}

    def pb(a, b, boots=10000, seed=83):
        rng = random.Random(seed)
        point = statistics.mean(v for k in keys for v in a[k]) / statistics.mean(v for k in keys for v in b[k])
        draws = []
        for _ in range(boots):
            pick = [keys[rng.randrange(len(keys))] for _ in keys]
            draws.append(statistics.mean(v for k in pick for v in a[k]) /
                         statistics.mean(v for k in pick for v in b[k]))
        draws.sort()
        return point, draws[int(.025 * boots)], draws[int(.975 * boots)]

    print('结构的价值，在两种规则下分别计算')
    for kind in ('safe', 'fatal'):
        for a, b, note in [('history', 'merged', '学到的结构相对完整历史'),
                           ('history', 'oracle', '真实自动机相对完整历史'),
                           ('merged', 'oracle', '学到的结构距离真实自动机')]:
            p, lo, hi = pb(blocks(kind, a), blocks(kind, b))
            print(f'  {kind:>5} {note:>22} {p:>5.2f}  [{lo:.2f}, {hi:.2f}]')
    print('\n同一臂，不可逆相对可逆的代价')
    for arm, _ in ARMS:
        p, lo, hi = pb(blocks('fatal', arm), blocks('safe', arm))
        print(f'  {arm:>9} {p:>5.2f}  [{lo:.2f}, {hi:.2f}]')


if __name__ == '__main__':
    main(*sys.argv[1:])
