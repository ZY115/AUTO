"""Step 34: the minimal counterexample behind the Step 33 wall.

Step 33 found that observing every single-step verdict at two histories never
forced the version space to merge them — 0 of 144 sampled pairs, under budgets
up to the physically impossible one. That is not a quirk of our task family; it
is a property of the interface. This file exhibits the smallest machine on which
it happens, checks every clause of the claim with the solver, and verifies by
exhaustive search that no smaller machine can.

The machine: three live states, alphabet {a, b}.

        state | a              | b
        ------+----------------+----------------
          0   | advance -> 1   | advance -> 2
          1   | ignore  -> 1   | advance -> 0
          2   | ignore  -> 2   | advance -> 1

States 1 and 2 carry *identical* one-step rows, (a: ignore, b: advance), and are
reached by the two distinct histories (a) and (b). They are not equivalent: the
suffix `b a` separates them, because b returns 1 to the start where a advances,
and moves 2 to state 1 where a is ignored.

So an agent that has observed every single-step verdict at both histories, and
cannot go back to run a continuation, has no way to tell whether they are the
same state: both answers stay consistent with everything it has seen.

Note the machine has no accepting state, because the claim is about progress
verdicts alone. Adding a goal state reachable from 0 changes nothing in the
argument; it only makes the object a task one could actually run.
"""
import itertools, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import version_space as vs

ALPHA = 2                                    # a = 0, b = 1
V = {'advance': 0, 'ignore': 1}
DELTA = {0: {0: 1, 1: 2}, 1: {0: 1, 1: 0}, 2: {0: 2, 1: 1}}


def verdict(d, q, e):
    return 'ignore' if d[q][e] == q else 'advance'


def run(d, h, q=0):
    for e in h:
        q = d[q][e]
    return q


def row(d, q, alpha):
    return tuple(verdict(d, q, e) for e in range(alpha))


def equivalent(d, x, y, alpha):
    """Moore equivalence: same verdict on every suffix, by fixpoint refinement."""
    n = len(d)
    same = [[row(d, i, alpha) == row(d, j, alpha) for j in range(n)] for i in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if same[i][j] and any(not same[d[i][e]][d[j][e]] for e in range(alpha)):
                    same[i][j] = False; changed = True
    return same[x][y]


def onestep(d, h):
    """Every single-step verdict at history h — the whole of what the agent gets."""
    q = run(d, h)
    return [(list(h), e, V[verdict(d, q, e)]) for e in range(ALPHA)]


def minimality(max_states=2, max_alpha=3):
    """Can a machine with fewer states do this? Enumerate them all and see.

    A counterexample needs two reachable states with identical one-step rows that
    are nevertheless inequivalent. Below three states no machine has one, for any
    alphabet up to max_alpha.
    """
    found = []
    for n in range(1, max_states + 1):
        for alpha in range(1, max_alpha + 1):
            for flat in itertools.product(range(n), repeat=n * alpha):
                d = {q: {e: flat[q * alpha + e] for e in range(alpha)} for q in range(n)}
                reach, stack = {0}, [0]
                while stack:
                    q = stack.pop()
                    for e in range(alpha):
                        if d[q][e] not in reach:
                            reach.add(d[q][e]); stack.append(d[q][e])
                for x, y in itertools.combinations(sorted(reach), 2):
                    if row(d, x, alpha) == row(d, y, alpha) \
                            and not equivalent(d, x, y, alpha):
                        found.append((n, alpha, d))
    return found


def show(label, ok):
    print(f'  [{"通过" if ok else "失败"}] {label}')
    return ok


def main():
    d = DELTA
    h1, h2 = (0,), (1,)                      # the histories (a) and (b)
    q1, q2 = run(d, h1), run(d, h2)
    print(f'三状态反例：h1=(a) 到状态 {q1}，h2=(b) 到状态 {q2}\n')
    print('命题的各子句，逐条用求解器核对：')
    good = True

    good &= show(f'(1) 单步输出行完全相同：row({q1})={row(d,q1,ALPHA)} '
                 f'= row({q2})={row(d,q2,ALPHA)}',
                 row(d, q1, ALPHA) == row(d, q2, ALPHA))

    a1 = verdict(d, run(d, list(h1) + [1]), 0)
    a2 = verdict(d, run(d, list(h2) + [1]), 0)
    good &= show(f'(2) 后缀 "b a" 区分它们：h1 之后 a 是 {a1}，h2 之后 a 是 {a2}；'
                 f'Moore 等价={equivalent(d,q1,q2,ALPHA)}',
                 a1 != a2 and not equivalent(d, q1, q2, ALPHA))

    ev = onestep(d, ()) + onestep(d, h1) + onestep(d, h2)
    for k in (3, 4, 5, 6):
        ids, base = vs._base(ev, [h1, h2], ALPHA)
        sat = vs.consistent(ev, [h1, h2], ALPHA, k)
        sep = vs.separable(ev, h1, h2, ALPHA, k)
        mrg = vs._ctl(base, f':- state({ids[h1]},K1), state({ids[h2]},K2), K1 != K2.',
                      k).solve().satisfiable
        good &= show(f'(3) k={k}：全部单步证据下 可满足={sat}，存在分开它们的机器={sep}，'
                     f'存在合并它们的机器={mrg} → 不可判定', sat and sep and mrg)

    ev2 = ev + onestep(d, tuple(h1) + (1,)) + onestep(d, tuple(h2) + (1,))
    for k in (3, 4, 5, 6):
        ids, base = vs._base(ev2, [h1, h2], ALPHA)
        sat = vs.consistent(ev2, [h1, h2], ALPHA, k)
        mrg = vs._ctl(base, f':- state({ids[h1]},K1), state({ids[h2]},K2), K1 != K2.',
                      k).solve().satisfiable
        good &= show(f'(4) k={k}：补上 "b" 之后那一步的观察后 可满足={sat}，'
                     f'仍存在合并它们的机器={mrg} → 已判定为不等价', sat and not mrg)

    f = minimality()
    good &= show(f'(5) 穷举 ≤2 状态、字母表 ≤3 的全部确定性机器（共 '
                 f'{sum(n**(n*a) for n in (1,2) for a in (1,2,3))} 台）：'
                 f'反例 {len(f)} 个 → 三状态是最小的', not f)

    print(f'\n{"全部通过" if good else "有子句未通过"}')
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
