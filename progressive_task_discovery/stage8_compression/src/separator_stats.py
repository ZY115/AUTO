"""How hard are the task states to tell apart, and does exploration ever see it?

P3 failed because signature compatibility asks the wrong quantifier: "no evidence
contradicts" rather than "every consistent model agrees". Before building
anything else, two facts are worth having.

First, the **shortest separating suffix** between each pair of genuinely distinct
task states: the fewest events after which their feedback differs. If most pairs
separate in one or two events, the states are easy to tell apart and P3's failure
is about looking at the wrong events, not about the states being subtle.

Second, whether the agent's own evidence ever contains a separator for the pairs
it has to judge.

Nothing here trains anything; it reads the task machines the runs used.
"""
import json, sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile


def separating_depths(t):
    """Moore refinement: depth at which each pair of states first differs.

    Depth 1 means one event already gives different feedback. Depth d means the
    difference only shows after d events. Pairs that never separate are the same
    Myhill-Nerode class and are excluded.
    """
    n, A = t.nstates, t.alphabet
    out = lambda q, e: t.verdict(q, e)
    depth = {}
    frontier = []
    for i in range(n):
        for j in range(i + 1, n):
            if any(out(i, e) != out(j, e) for e in range(A)):
                depth[(i, j)] = 1
                frontier.append((i, j))
    d = 1
    while frontier:
        d += 1
        nxt = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in depth:
                    continue
                for e in range(A):
                    a, b = t.trans[i][e], t.trans[j][e]
                    k = (min(a, b), max(a, b))
                    if a != b and k in depth and depth[k] == d - 1:
                        depth[(i, j)] = d
                        nxt.append((i, j))
                        break
        frontier = nxt
    return depth


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    rows, alld = [], []
    for f in fams:
        t = TaskFile(ROOT / f"results/ceiling/{f['tag']}.task")
        dep = separating_depths(t)
        # Only pairs of states that are genuinely different, i.e. separable.
        vals = sorted(dep.values())
        alld += vals
        classes = len(set(t.minimal))
        rows.append(dict(tag=f['tag'], states=t.nstates, minimal=classes,
                         alphabet=t.alphabet, pairs=len(vals),
                         d1=sum(1 for v in vals if v == 1),
                         d2=sum(1 for v in vals if v == 2),
                         d3plus=sum(1 for v in vals if v >= 3),
                         maxd=max(vals) if vals else 0))
    from collections import Counter
    c = Counter(alld)
    tot = sum(c.values())
    print('24 个任务族里，每一对可区分的任务状态，最短区分后缀的长度')
    print(f"{'长度':>5} {'对数':>8} {'占比':>8}")
    for k in sorted(c):
        print(f"{k:>5} {c[k]:>8} {c[k]/tot:>7.1%}")
    print(f"\n合计 {tot} 对；长度 1 或 2 的占 {(c.get(1,0)+c.get(2,0))/tot:.1%}")
    print(f"最长 {max(c)}；平均 {sum(k*v for k,v in c.items())/tot:.2f}")
    (ROOT / 'results/separator_stats.json').write_text(
        json.dumps(dict(per_family=rows, distribution={str(k): v for k, v in sorted(c.items())}),
                   indent=2) + '\n')
    print('\n逐族（可区分对数 / 长度 1 / 长度 2 / 长度>=3 / 最长）')
    for r in rows[:8]:
        print(f"  {r['tag']:<22} 状态 {r['states']:>2} 最小类 {r['minimal']:>2}  "
              f"{r['pairs']:>4} / {r['d1']:>4} / {r['d2']:>3} / {r['d3plus']:>3} / {r['maxd']}")
