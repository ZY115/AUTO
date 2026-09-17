"""Can the merge rule recover the compact task state when evidence is not the
limiting factor?

The earlier control answered a weaker question than it claimed: its evidence set
was the prefixes of the selected optimal trajectories, not the whole reachable
task language, and it used the merge rule that has since been shown to drop
member evidence. This audit fixes both. Policy is out of the picture entirely:
every rule sees exactly the same fixed evidence.

Gate 1 asks whether exhaustive evidence yields the minimal partition. If it does
not, the inference rule is the problem and no amount of exploration will help.
Gate 2 grades the evidence down to see where, and how sharply, it degrades.
"""
from itertools import permutations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_maps import Task
from inference import Node, rebuild, score

sys.path.insert(0, str(ROOT / 'tests'))


def reachable(task):
    """Every prefix of every accepted history, not just the optimal routes."""
    out = set()
    for perm in permutations(range(task.n)):
        st = task.states[0]
        bit = 0 if perm.index(task.pair[0]) < perm.index(task.pair[1]) else 1
        h = tuple(perm) + tuple(task.suffix(bit))
        for k in range(len(h) + 1):
            out.add(h[:k])
        del st
    return sorted(out, key=lambda p: (len(p), p))


def build_nodes(task, prefixes, keep=1.0, rng=None):
    index = {p: i for i, p in enumerate(prefixes)}
    nodes = []
    for p in prefixes:
        st = task.run(p)
        nd = Node(task.alphabet, visits=1000,
                  truth=task.minimal[task.index[st]])
        for e in range(task.alphabet):
            if rng is not None and rng.random() > keep:
                continue
            if task.step(st, e) != st:
                nd.advanced[e] = 1
                child = p + (e,)
                if child in index:
                    nd.succ[e] = index[child]
            else:
                nd.ignored[e] = 1
        nodes.append(nd)
    return nodes


def legacy(nodes, alphabet):
    from test_inference import legacy_rebuild
    return legacy_rebuild(nodes, alphabet)


def main(out='results/inference_audit.json'):
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    rows = []
    print('Gate 1  完整可达任务语言，完整证据')
    print(f"{'台阶':>9} {'前缀数':>6} {'真实类':>6} | "
          f"{'旧规则 精度/召回/类':>24} | {'新规则宽松':>20} | {'新规则严格':>20}")
    for meta in metas:
        task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
        pre = reachable(task)
        nodes = build_nodes(task, pre)
        cells = []
        results = {}
        for label, blocks in [('legacy', legacy(nodes, task.alphabet)),
                              ('loose', rebuild(nodes, task.alphabet, quota=0, strict_children=False)),
                              ('strict', rebuild(nodes, task.alphabet, quota=1, strict_children=True))]:
            p, r, k, t = score(nodes, blocks)
            results[label] = dict(precision=p, recall=r, classes=k, true_classes=t)
            cells.append(f'{p:.2f} / {r:.2f} / {k}')
        rows.append(dict(tag=meta['tag'], prefixes=len(pre), true_classes=results['legacy']['true_classes'],
                         full_evidence=results))
        print(f"{meta['tag']:>9} {len(pre):>6} {results['legacy']['true_classes']:>6} | "
              f"{cells[0]:>24} | {cells[1]:>20} | {cells[2]:>20}")

    print('\nGate 2  证据分级，新规则宽松版')
    print(f"{'台阶':>9} " + " ".join(f"{k:>16}" for k in (1.0, .95, .9, .8, .65, .5)))
    for row, meta in zip(rows, metas):
        task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
        pre = reachable(task)
        cells, graded = [], {}
        for keep in (1.0, .95, .9, .8, .65, .5):
            ps, rs, ks = [], [], []
            for trial in range(5):
                rng = None if keep == 1.0 else random.Random(100 + trial)
                nodes = build_nodes(task, pre, keep, rng)
                p, r, k, _ = score(nodes, rebuild(nodes, task.alphabet, quota=0, strict_children=False))
                ps.append(p); rs.append(r); ks.append(k)
            graded[str(keep)] = dict(precision=sum(ps) / 5, recall=sum(rs) / 5, classes=sum(ks) / 5)
            cells.append(f'{sum(ps)/5:.2f} / {sum(rs)/5:.2f}')
        row['graded'] = graded
        print(f"{meta['tag']:>9} " + " ".join(f"{c:>16}" for c in cells))
    (ROOT / out).write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
