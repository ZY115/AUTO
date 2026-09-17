"""What would full structural coverage cost in environment steps, and is it worth it?

To separate two task states a merger needs negative evidence: event e tried at
state q and seen not to advance. Task-driven behaviour does not produce that,
so it has to be bought with directed detours. This prices the purchase against
the prize, which is the measured gap between the full-history arm and the
oracle-automaton arm.

An episode can amortise one prefix walk over several event trials, so the
estimate charges each node its shortest prefix walk once, then charges a
nearest-neighbour tour over the event cells that still need trying.
"""
import itertools, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task, product_shortest
from emit_tasks import make_layout

ROOT = Path(__file__).resolve().parents[1]


def cell_distances(layout):
    """All-pairs shortest path over the physical map, ignoring the task."""
    n = len(layout['moves'])
    INF = 10 ** 6
    dist = [[INF] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        frontier = [s]
        seen = {s}
        d = 0
        while frontier:
            d += 1
            nxt = []
            for c in frontier:
                for ns, _ in layout['moves'][c]:
                    if ns not in seen:
                        seen.add(ns)
                        dist[s][ns] = d
                        nxt.append(ns)
            frontier = nxt
    return dist


def main(out='results/coverage_budget.json'):
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    confirm = [json.loads(l) for l in (ROOT / 'results/confirm2/raw.jsonl').open()]
    by_arm = {}
    for r in confirm:
        by_arm.setdefault((r['rung'], r['arm']), []).append(r)
    capped = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)

    rows = []
    print(f"{'台阶':>9} {'节点':>5} {'事件':>4} {'需试次数':>8} {'前缀成本':>8} {'巡回成本':>8} "
          f"{'总预算':>8} {'奖金':>8} {'预算/奖金':>9}")
    for meta in metas:
        layout = make_layout(meta)
        task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
        dist, N = product_shortest(layout, task)
        cell = cell_distances(layout)
        label_cells = {}
        for c, g in layout['labels']:
            label_cells.setdefault(g, []).append(c)

        # Nodes the learner actually meets: prefixes of the optimal histories.
        prefixes = set()
        for h in meta['optimal_histories']:
            for i in range(len(h) + 1):
                prefixes.add(tuple(h[:i]))
        prefix_steps = tour_steps = 0
        for p in sorted(prefixes):
            # Where the agent physically stands after this prefix, and what it
            # cost to walk here from the cheapest start.
            best = None
            for start, hist in zip(meta['starts'], meta['optimal_histories']):
                if tuple(hist[:len(p)]) != p:
                    continue
                q = task.index[task.run(p)]
                here = layout['labels'][0][0] if not p else None
                cand = dist[q * N + start] if q * N + start < len(dist) else None
                if cand is not None and cand >= 0 and (best is None or cand < best[0]):
                    pos = start if not p else label_cells[p[-1]][0]
                    best = (cand, pos)
            if best is None:
                continue
            prefix_steps += max(0, dist[task.index[task.run(p)] * N + meta['starts'][0]] -
                                dist[task.index[task.run(p)] * N + meta['starts'][0]]) + best[0]
            pos = best[1]
            remaining = sorted(label_cells)
            here = pos
            for g in remaining:
                step = min(cell[here][c] for c in label_cells[g])
                tour_steps += step
                here = min(label_cells[g], key=lambda c: cell[pos][c])
        trials = len(prefixes) * task.alphabet
        budget = prefix_steps + tour_steps
        prize = capped(by_arm[(meta['tag'], 'history')]) - capped(by_arm[(meta['tag'], 'automaton')])
        rows.append(dict(tag=meta['tag'], nodes=len(prefixes), alphabet=task.alphabet,
                         trials=trials, prefix_steps=prefix_steps, tour_steps=tour_steps,
                         budget=budget, prize=prize, ratio=budget / prize if prize else None))
        print(f"{meta['tag']:>9} {len(prefixes):>5} {task.alphabet:>4} {trials:>8} {prefix_steps:>8} "
              f"{tour_steps:>8} {budget:>8} {prize:>8.0f} {budget / prize:>9.2f}")
    Path(ROOT / out).write_text(json.dumps(rows, indent=2) + '\n')
    print(f"\n中位 预算/奖金 = {statistics.median(r['ratio'] for r in rows):.2f}")


if __name__ == '__main__':
    main(*sys.argv[1:])
