"""Map pairs graded by how much their forced event orderings overlap.

The previous transfer experiment paired two layouts without asking whether the
target map exercises orderings the source map never produced. If the two force
the same orderings, a carried structure is trivially applicable and the result is
closer to relearning a route than to generalising a task rule. Overlap is
therefore made an explicit axis: the same hidden task on two maps whose forced
orderings agree almost completely, partly, or not at all.
"""
import itertools, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, evaluate_layout
from spoke_maps import build_spokes, funnel_ok
from emit_tasks import emit

ROOT = Path(__file__).resolve().parents[1]


def candidates(n, d, m, m2, pair, L, arm, seeds):
    out = []
    task = Task(n, pair, d, m, m2)
    for seed in seeds:
        layout = build_spokes(n, L, seed, arm=arm)
        if layout is None or not funnel_ok(layout):
            continue
        info = evaluate_layout(layout, task, 24)
        if info is None or not info['count_aliased'] or info['orders'] < 4:
            continue
        orders = {tuple(h[:n]) for h in info['optimal_histories']}
        out.append(dict(seed=seed, layout=layout, info=info, orders=orders))
    return out


def overlap(a, b):
    """Share of the target's forced orderings that the source also produced."""
    return len(a['orders'] & b['orders']) / len(b['orders'])


def main(out='results/overlap', seed0=80000, span=260, specs=None):
    # Parameterised so an independent confirmation batch can be drawn from
    # map seeds the development batch never touched.
    seed0, span = int(seed0), int(span)
    started = time.monotonic()
    outdir = ROOT / out
    outdir.mkdir(parents=True, exist_ok=True)
    if specs is None:
        specs = [('taskP', 3, 3, 1, 4, (0, 1), 2, 7), ('taskQ', 4, 2, 1, 4, (0, 1), 2, 8)]
    bands = [('high', .75, 1.01), ('mid', .3, .6), ('low', -.01, .2)]
    rows = []
    for name, n, d, m, m2, pair, L, arm in specs:
        pool = candidates(n, d, m, m2, pair, L, arm, range(seed0, seed0 + span))
        if len(pool) < 4:
            print(f'{name}: 候选地图只有 {len(pool)} 张')
            continue
        # Consider every ordered pair, not just one fixed source, or the low
        # band stays empty for want of candidates rather than for want of maps.
        options = {band: [] for band, _, _ in bands}
        for source, target in itertools.permutations(pool, 2):
            share = overlap(source, target)
            for band, lo, hi in bands:
                if lo <= share < hi:
                    options[band].append((source, target, share))
        for band, lo, hi in bands:
            picks = []
            used = set()
            for source, target, share in sorted(options[band], key=lambda z: -z[2] if band == 'high' else z[2]):
                if source['seed'] in used or target['seed'] in used:
                    continue
                used.add(source['seed'])
                used.add(target['seed'])
                picks.append((source, target))
                if len(picks) >= 5:
                    break
            if not picks:
                print(f'{name}/{band}: 没有落在这一档的目标地图')
                continue
            for k, (source, target) in enumerate(picks):
                entry = dict(task=name, band=band, n=n, d=d, m=m, m2=m2, pair=list(pair),
                             L=L, arm=arm, overlap=overlap(source, target),
                             source_seed=source['seed'], target_seed=target['seed'],
                             source_orders=sorted(map(list, source['orders'])),
                             target_orders=sorted(map(list, target['orders'])))
                for role, c in (('A', source), ('B', target)):
                    tag = f'{name}_{band}{k}{role}'
                    meta = emit(dict(entry, seed=c['seed'], tag=tag, target=tag), outdir)
                    entry[f'map{role}'] = dict(tag=tag, seed=c['seed'], orders=c['info']['orders'],
                                               horizon=meta['horizon'], lengths=meta['lengths'],
                                               cells=len(c['layout']['cells']))
                rows.append(entry)
                print(f"{name}/{band}{k}: 顺序重合 {entry['overlap']:.2f}  "
                      f"源 {source['info']['orders']} 种顺序，目标 {target['info']['orders']} 种，"
                      f"其中 {len(source['orders'] & target['orders'])} 种源图见过")
    (outdir / 'pairs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\n{len(rows)} 组，{round(time.monotonic() - started, 1)} 秒')


if __name__ == '__main__':
    main(*sys.argv[1:])
