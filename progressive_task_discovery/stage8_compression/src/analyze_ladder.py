"""Read the ladder runs and test whether sample cost tracks representation size."""
import collections, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    rows = [json.loads(l) for l in (ROOT / 'results' / name / 'raw.jsonl').open()]
    metas = {m['tag']: m for m in json.loads((ROOT / 'results/family_maps/rungs.json').read_text())}
    return rows, metas


def capped(rs):
    return [(x['first90'] if x['first90'] > 0 else x['budget']) for x in rs]


def paired_ratio(a, b, boots=2000, seed=7):
    """Paired seed bootstrap on the ratio of mean capped costs."""
    import random
    rng = random.Random(seed)
    seeds = sorted(set(a) & set(b))
    point = statistics.mean(a[s] for s in seeds) / statistics.mean(b[s] for s in seeds)
    draws = []
    for _ in range(boots):
        pick = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        draws.append(statistics.mean(a[s] for s in pick) / statistics.mean(b[s] for s in pick))
    draws.sort()
    return point, draws[int(.025 * boots)], draws[int(.975 * boots)]


def main(name='ladder32'):
    rows, metas = load(name)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)
    rungs = sorted(metas.values(), key=lambda m: m['honest_ratio'])

    print(f"{'台阶':>9} {'比值':>5} {'臂':>10} {'达标步数':>9} {'达标':>7} {'AUC':>6} {'最终':>5} {'特征':>5}")
    points = []
    for meta in rungs:
        tag = meta['tag']
        arms = sorted({a for (t, a) in g if t == tag},
                      key=lambda a: (not a.startswith('count'), a.startswith('window'),
                                     int(a[6:]) if a.startswith('window') else 0, a))
        for arm in arms:
            rs = g[(tag, arm)]
            f = capped(rs)
            solved = sum(1 for x in rs if x['first90'] > 0)
            feats = statistics.mean(x['features'] for x in rs)
            print(f"{tag:>9} {meta['honest_ratio']:>5.2f} {arm:>10} {statistics.mean(f):>9.0f}"
                  f" {solved:>3}/{len(rs)} {statistics.mean(x['auc'] for x in rs):>6.3f}"
                  f" {statistics.mean(x['final_success'] for x in rs):>5.2f} {feats:>5.0f}")
            if solved == len(rs):
                points.append((tag, arm, feats, statistics.mean(f)))
        print()

    print('每个台阶内，达标步数对特征数的仿射拟合')
    print(f"{'台阶':>9} {'点数':>4} {'斜率':>7} {'截距':>8} {'截距占oracle':>12}")
    for meta in rungs:
        tag = meta['tag']
        pts = [(x[2], x[3]) for x in points if x[0] == tag]
        auto = [x[3] for x in points if x[0] == tag and x[1] == 'automaton']
        if len(pts) < 3 or not auto:
            print(f"{tag:>9} {len(pts):>4}   点数不足")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        var = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in pts) / var
        inter = my - slope * mx
        print(f"{tag:>9} {len(pts):>4} {slope:>7.0f} {inter:>8.0f} {inter / auto[0]:>11.0%}")

    print('\n分母臂对自动机臂的配对成本比')
    print(f"{'台阶':>9} {'预测比值':>7} {'实测比 window':>13} {'95%区间':>18} {'实测比 history':>14}")
    for meta in rungs:
        tag = meta['tag']
        w = f"window{meta['min_sufficient_window']}"
        auto = {x['seed']: c for x, c in zip(g[(tag, 'automaton')], capped(g[(tag, 'automaton')]))}
        line = f"{tag:>9} {meta['honest_ratio']:>7.2f}"
        for arm in (w, 'history'):
            rs = g[(tag, arm)]
            if not rs:
                line += f" {'n/a':>13}"
                continue
            d = {x['seed']: c for x, c in zip(rs, capped(rs))}
            p, lo, hi = paired_ratio(d, auto)
            if arm == w:
                line += f" {p:>13.2f} {f'[{lo:.2f}, {hi:.2f}]':>18}"
            else:
                line += f" {p:>14.2f}"
        print(line)


if __name__ == '__main__':
    main(*sys.argv[1:])
