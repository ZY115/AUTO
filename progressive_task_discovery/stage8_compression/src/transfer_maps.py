"""Pairs of maps that carry the same hidden task.

The whole point of a task automaton is that it is the half of the problem that
does not mention the floor plan. Every experiment so far learned one automaton
for one map and threw it away, which is the harshest possible test: the
structure has to repay its own discovery cost inside a single run. Here the same
task is posed on two different layouts, so what was learned on the first can be
carried to the second while everything indexed by a cell must be relearned.
"""
import itertools, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, evaluate_layout, product_shortest
from spoke_maps import build_spokes, funnel_ok
from emit_tasks import emit

ROOT = Path(__file__).resolve().parents[1]


def usable(n, d, m, m2, pair, L, arm, seed):
    layout = build_spokes(n, L, seed, arm=arm)
    if layout is None or not funnel_ok(layout):
        return None
    task = Task(n, pair, d, m, m2)
    info = evaluate_layout(layout, task, 24)
    if info is None or not info['count_aliased'] or info['orders'] < 4:
        return None
    return layout, task, info


def main(out='results/transfer'):
    started = time.monotonic()
    outdir = ROOT / out
    outdir.mkdir(parents=True, exist_ok=True)
    specs = [(3, 3, 1, 4, (0, 1), 2, 7), (3, 2, 1, 3, (0, 1), 2, 7),
             (4, 3, 1, 4, (0, 1), 2, 8), (4, 2, 1, 4, (0, 1), 2, 8),
             (3, 4, 1, 4, (0, 2), 2, 8), (4, 2, 1, 5, (0, 2), 3, 8)]
    rows = []
    for idx, (n, d, m, m2, pair, L, arm) in enumerate(specs):
        found = []
        base = 78000 + idx * 977
        for seed in range(base, base + 40):
            got = usable(n, d, m, m2, pair, L, arm, seed)
            if got:
                found.append((seed, got))
            if len(found) >= 2:
                break
        if len(found) < 2:
            print(f'pair{idx:02d}: 只找到 {len(found)} 张可用地图')
            continue
        entry = {'pair_id': f'pair{idx:02d}', 'n': n, 'd': d, 'm': m, 'm2': m2,
                 'pair': list(pair), 'L': L, 'arm': arm}
        for role, (seed, (layout, task, info)) in zip('AB', found):
            cfg = dict(entry, seed=seed, tag=f'pair{idx:02d}{role}', target=f'pair{idx:02d}{role}')
            meta = emit(cfg, outdir)
            entry[f'map{role}'] = dict(seed=seed, tag=meta['tag'], orders=info['orders'],
                                       cells=len(layout['cells']), horizon=meta['horizon'],
                                       lengths=meta['lengths'],
                                       history_states=info['history_states'],
                                       minimal_states=info['minimal_states'])
        a, b = entry['mapA'], entry['mapB']
        print(f"{entry['pair_id']}: n={n} d={d} 尾长{m}/{m2}  "
              f"地图A {a['cells']}格 顺序{a['orders']} 物理{min(a['lengths'])}-{max(a['lengths'])}  "
              f"地图B {b['cells']}格 顺序{b['orders']} 物理{min(b['lengths'])}-{max(b['lengths'])}")
        rows.append(entry)
    (outdir / 'pairs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\n{len(rows)} 对地图，{round(time.monotonic() - started, 1)} 秒')


if __name__ == '__main__':
    main(*sys.argv[1:])
