"""Reversible and irreversible versions of one task on one forked map.

The only difference between the versions is what a wrong commit does. Both use
the same geometry, the same starts and the same optimal routes, which is checked
rather than assumed: if the two versions disagreed about the optimal routes, the
comparison would be measuring a change of map as well as a change of rule.

The fork exists because with the suffix labels strung along one corridor,
reaching one branch's first label means walking over the other's. Harmless while
wrong events are ignored, fatal once they are not, and it deletes a branch
instead of punishing a memory mistake.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, evaluate_layout
from spoke_maps import build_forked, funnel_ok
from emit_tasks import emit

ROOT = Path(__file__).resolve().parents[1]


def candidate(n, d, m, m2, pair, L, arm, seed, want=4):
    safe = Task(n, pair, d, m, m2, fatal=False)
    fatal = Task(n, pair, d, m, m2, fatal=True)
    layout = build_forked(n, L, seed, arm=arm, bridge=safe.bridge, tails=safe.tails)
    if layout is None or not funnel_ok(layout):
        return None
    a = evaluate_layout(layout, safe, 24)
    b = evaluate_layout(layout, fatal, 24)
    if a is None or b is None or not a['count_aliased'] or not b['count_aliased']:
        return None
    if a['orders'] < want:
        return None
    # Same routes under both rules, or the comparison is not about the rule.
    if a['optimal_histories'] != b['optimal_histories'] or a['starts'] != b['starts']:
        return None
    return layout, a, b


def main(out='results/irreversible'):
    started = time.monotonic()
    outdir = ROOT / out
    outdir.mkdir(parents=True, exist_ok=True)
    specs = [('taskR', 3, 3, 1, 4, (0, 1), 2, 7), ('taskS', 4, 2, 1, 4, (0, 1), 2, 8),
             ('taskT', 3, 2, 1, 3, (0, 1), 2, 7)]
    rows = []
    for name, n, d, m, m2, pair, L, arm in specs:
        found = 0
        for seed in range(82000, 82400):
            got = candidate(n, d, m, m2, pair, L, arm, seed)
            if not got:
                continue
            layout, a, _ = got
            entry = dict(task=name, n=n, d=d, m=m, m2=m2, pair=list(pair), L=L, arm=arm,
                         seed=seed, orders=a['orders'], cells=len(layout['cells']),
                         forked=True)
            for fatal in (False, True):
                kind = 'fatal' if fatal else 'safe'
                tag = f'{name}_{found:02d}_{kind}'
                meta = emit(dict(entry, tag=tag, target=tag, fatal=fatal), outdir)
                entry[kind] = dict(tag=tag, horizon=meta['horizon'], lengths=meta['lengths'],
                                   failing_states=meta['failing_states'],
                                   minimal_states=meta['minimal_states'])
            assert entry['safe']['lengths'] == entry['fatal']['lengths']
            rows.append(entry)
            found += 1
            if found >= 5:
                break
        print(f"{name}: {found} 张地图，顺序数 {rows[-1]['orders'] if found else 0}，"
              f"最短 {min(rows[-1]['safe']['lengths']) if found else 0}"
              f"-{max(rows[-1]['safe']['lengths']) if found else 0}")
    (outdir / 'maps.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\n{len(rows)} 张地图 × 两个版本，{round(time.monotonic() - started, 1)} 秒')


if __name__ == '__main__':
    main(*sys.argv[1:])
