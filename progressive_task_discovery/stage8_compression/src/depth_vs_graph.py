"""Build tasks where progress depth and task-graph distance disagree.

In the earlier family every accepted history had the same length, so the depth
of a prefix was also its distance to acceptance and a depth potential could not
be told apart from a graph potential. Giving the two branch tails different
lengths breaks that: at the end of the shared bridge both branches stand on the
same cell with the same progress count, yet owe different numbers of
transitions.

Two properties are checked exactly before a layout is used. The first is the one
carried over: position and progress count must not substitute for history. The
second is new: among the histories the forced starts actually produce, there
must be pairs at equal depth whose remaining task distance differs, and the
larger that gap the more room a graph potential has over a depth potential.
"""
import itertools, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, evaluate_layout, product_shortest
from spoke_maps import build_spokes, funnel_ok

ROOT = Path(__file__).resolve().parents[1]


def remaining(task, prefix):
    """Transitions still owed, walking the task machine, not the map."""
    st = task.run(prefix)
    n = 0
    while not task.done(st):
        st = task.step(st, task.required(st)[0])
        n += 1
    return n


def depth_gap(task, histories):
    """Largest disagreement between depth and remaining distance, and where."""
    by_depth = {}
    for h in histories:
        for k in range(len(h) + 1):
            by_depth.setdefault(k, set()).add(remaining(task, tuple(h[:k])))
    gaps = {k: max(v) - min(v) for k, v in by_depth.items() if len(v) > 1}
    return (max(gaps.values()) if gaps else 0), sorted(gaps)


def search(n, d, m, m2, arm, instances, seeds):
    best = None
    for seed in seeds:
        layout = build_spokes(n, instances, seed, arm=arm)
        if layout is None or not funnel_ok(layout):
            continue
        for pair in itertools.combinations(range(n), 2):
            task = Task(n, pair, d, m, m2)
            info = evaluate_layout(layout, task, 24)
            if info is None or not info['count_aliased']:
                continue
            gap, depths = depth_gap(task, info['optimal_histories'])
            if gap < 1:
                continue
            dist, N = product_shortest(layout, task)
            lengths = [dist[s] for s in info['starts']]
            score = (-gap, -info['orders'], max(lengths))
            if best is None or score < best[0]:
                best = (score, dict(layout=layout, task=task, info=info, seed=seed,
                                    pair=pair, gap=gap, gap_depths=depths,
                                    lengths=lengths, n=n, d=d, m=m, m2=m2,
                                    L=instances, arm=arm))
    return best[1] if best else None


def main(out='results/depth_vs_graph'):
    started = time.monotonic()
    outdir = ROOT / out
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [('gapA', 3, 3, 1, 4, 7, 2), ('gapB', 4, 3, 1, 4, 8, 2),
             ('gapC', 3, 4, 1, 5, 8, 2), ('gapD', 4, 2, 1, 5, 8, 3)]
    rows = []
    for tag, n, d, m, m2, arm, L in cells:
        found = search(n, d, m, m2, arm, L,
                       range(74000 + n * 311 + d * 17 + m2, 74000 + n * 311 + d * 17 + m2 + 18))
        if found is None:
            print(f'{tag}: 没有找到满足条件的布局')
            continue
        info, task = found['info'], found['task']
        rows.append(dict(tag=tag, n=n, d=d, m=m, m2=m2, L=L, arm=arm, seed=found['seed'],
                         pair=list(found['pair']), gap=found['gap'], gap_depths=found['gap_depths'],
                         orders=info['orders'], history_states=info['history_states'],
                         minimal_states=info['minimal_states'],
                         physical_lengths=found['lengths'],
                         task_lengths=[len(h) for h in info['optimal_histories']],
                         cells=len(found['layout']['cells']),
                         min_sufficient_window=info['min_sufficient_window']))
        print(f"{tag}: n={n} d={d} 尾长 {m}/{m2} 顺序数 {info['orders']} "
              f"深度距离最大分歧 {found['gap']} 出现在深度 {found['gap_depths']} "
              f"物理路径 {min(found['lengths'])}-{max(found['lengths'])} "
              f"任务长度 {sorted({len(h) for h in info['optimal_histories']})} 格子 {len(found['layout']['cells'])}")
    (outdir / 'search.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\n{round(time.monotonic() - started, 1)} 秒，写入 {outdir}/search.json')


if __name__ == '__main__':
    main(*sys.argv[1:])
