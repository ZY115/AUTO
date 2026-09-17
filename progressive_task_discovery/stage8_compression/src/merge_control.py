"""Control: is the merge rule correct, or is the evidence the binding constraint?

Runs the same evidence-driven rule the C++ learner uses, but on complete
evidence: every event tried at every reachable history. If the rule then
recovers the true Myhill-Nerode partition exactly, the rule is sound and any
shortfall in a training run is a coverage problem, not an algorithm problem.

Then repeats the rule on partial evidence, keeping a random fraction of the
(node, event) observations, to show how precision degrades with coverage.
"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_maps import Task


def reachable_prefixes(task, layout_histories):
    pre = set()
    for h in layout_histories:
        for i in range(len(h) + 1):
            pre.add(tuple(h[:i]))
    return sorted(pre, key=lambda p: (len(p), p))


def evidence(task, prefixes, keep=1.0, rng=None):
    """advanced[h][e], ignored[h][e] as the learner would record them."""
    adv, ign, succ = {}, {}, {}
    index = {p: i for i, p in enumerate(prefixes)}
    for p in prefixes:
        st = task.run(p)
        adv[p], ign[p], succ[p] = set(), set(), {}
        for e in range(task.alphabet):
            if rng is not None and rng.random() > keep:
                continue
            nxt = task.step(st, e)
            if nxt != st:
                adv[p].add(e)
                child = p + (e,)
                if child in index:
                    succ[p][e] = child
            else:
                ign[p].add(e)
    return adv, ign, succ


def merge(prefixes, adv, ign, succ):
    parent = {p: p for p in prefixes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def conflict(a, b):
        return bool((adv[a] & ign[b]) or (ign[a] & adv[b]))

    def compatible(i, j):
        stack, seen, touched = [(i, j)], set(), []
        while stack:
            a, b = stack.pop()
            a, b = find(a), find(b)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            touched.append((a, b))
            if conflict(a, b):
                return None
            for e in range(max(len(adv[a]) + len(ign[a]), 1) and 16):
                sa, sb = succ[a].get(e), succ[b].get(e)
                if sa is not None and sb is not None:
                    stack.append((sa, sb))
        return touched

    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            a, b = prefixes[i], prefixes[j]
            if find(a) == find(b) or conflict(a, b):
                continue
            touched = compatible(a, b)
            if touched is None:
                continue
            for x, y in touched:
                x, y = find(x), find(y)
                if x != y:
                    parent[max(x, y)] = min(x, y)
    return {p: find(p) for p in prefixes}


def score(task, prefixes, blocks):
    agree = learned = truth = 0
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            a, b = prefixes[i], prefixes[j]
            same_true = task.minimal[task.index[task.run(a)]] == task.minimal[task.index[task.run(b)]]
            same_learned = blocks[a] == blocks[b]
            learned += same_learned
            truth += same_true
            agree += same_true and same_learned
    return (agree / learned if learned else 1.0,
            agree / truth if truth else 1.0,
            len(set(blocks.values())),
            len({task.minimal[task.index[task.run(p)]] for p in prefixes}))


def main(out='results/merge_control.json'):
    metas = json.loads((Path(__file__).resolve().parents[1] / 'results/family_maps/rungs.json').read_text())
    rows = []
    print(f"{'台阶':>9} {'覆盖':>5} {'精度':>5} {'召回':>5} {'类数':>5} {'真实':>5}")
    for meta in metas:
        task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
        prefixes = reachable_prefixes(task, [tuple(h) for h in meta['optimal_histories']])
        for keep in (1.0, 0.9, 0.8, 0.65, 0.5):
            rng = None if keep == 1.0 else random.Random(11)
            adv, ign, succ = evidence(task, prefixes, keep, rng)
            blocks = merge(prefixes, adv, ign, succ)
            p, r, k, t = score(task, prefixes, blocks)
            rows.append(dict(tag=meta['tag'], keep=keep, precision=p, recall=r,
                             classes=k, true_classes=t))
            print(f"{meta['tag']:>9} {keep:>5.2f} {p:>5.2f} {r:>5.2f} {k:>5} {t:>5}")
        print()
    Path(out).write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
