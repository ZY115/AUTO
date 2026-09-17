"""Does the task family force structure learning, or can a fixed feature stand in?

A representation chosen before the hidden task is known must be sufficient for
EVERY instance the family can produce. So the honest denominator for a
compression claim is the family-sufficient generic feature with the fewest
states, and the numerator is the minimal automaton of the single instance being
run. Picking the denominator by any other rule inflates the claim.

Two ways earlier families failed this test:

  pure unordered      the bag of completed events is family-sufficient and
                      equals the minimal automaton exactly, so no structure
                      learning is needed at all.
  one fixed pair      'bag plus which of those two came first' is again
                      writable in advance.

Randomising the order-sensitive pair per instance is necessary but not
sufficient. If the branch changes the very next required label, a window of
width n-1 still recovers the bit at the decision point, because the one missing
unordered label is inferable from the count; and once a branch-specific label
is observed, any window of width 1 gives the branch away.

So the branch consequence is pushed behind a shared segment of length d, the
Stage 7 bridge device: both branches require the same d labels before they
diverge. The deciding evidence then sits at an unpredictable position in the
unordered phase and at least d steps behind the decision, so a sufficient
window must span the whole unordered phase plus the bridge.

Nothing here trains anything. It is a precondition check: if the gap is not
present combinatorially, no learning run can produce it.
"""
from itertools import permutations
import json, os, sys

POOL = 3


def task_suffix(n, d, m, bit):
    """Shared bridge of length d, then a branch-specific tail of length m.

    The two tails must differ at their first label, or the instance has no
    branch at all and the family silently collapses back to pure unordered.
    Both tails also have to avoid repeating the last bridge label.
    """
    bridge = [n + (k % POOL) for k in range(d)]
    last = bridge[-1] if d else None
    choices = [n + c for c in range(POOL) if n + c != last]
    start = choices[bit % len(choices)]
    tail = [n + ((start - n + k) % POOL) for k in range(m)]
    return bridge + tail


def complete_histories(n, d, m, pair):
    """Every accepted successful-event sequence for one instance."""
    out = []
    for perm in permutations(range(n)):
        bit = 0 if pair is None or perm.index(pair[0]) < perm.index(pair[1]) else 1
        out.append(tuple(perm) + tuple(task_suffix(n, d, m, bit)))
    return out


def prefixes_and_futures(complete):
    fut = {}
    for h in complete:
        for k in range(len(h) + 1):
            fut.setdefault(h[:k], set()).add(h[k:])
    return {p: frozenset(f) for p, f in fut.items()}


def feature_set(n, d, m):
    feats = {'count': lambda p: (len(p),), 'bag': lambda p: (tuple(sorted(p)), len(p))}
    for w in range(1, n + d + m + 1):
        feats[f'window{w}'] = lambda p, w=w: (p[-w:], len(p))
        feats[f'bag+window{w}'] = lambda p, w=w: (tuple(sorted(p)), p[-w:], len(p))
    feats['full_history'] = lambda p: p
    return feats


def sufficient(fut, feat):
    seen = {}
    for p, f in fut.items():
        if seen.setdefault(feat(p), f) != f:
            return False
    return True


def analyse(n, d, m, randomised):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if randomised else [None]
    insts = [prefixes_and_futures(complete_histories(n, d, m, q)) for q in pairs]
    feats = feature_set(n, d, m)
    ok = [name for name, f in feats.items() if all(sufficient(x, f) for x in insts)]
    # denominator: the family-sufficient feature with the fewest states
    def size(name):
        return sum(len({feats[name](p) for p in x}) for x in insts) / len(insts)
    denom = min(ok, key=size)
    autos = [len(set(x.values())) for x in insts]
    dens = [len({feats[denom](p) for p in x}) for x in insts]
    return {
        'n': n, 'd': d, 'm': m, 'randomised': randomised, 'instances': len(insts),
        'family_sufficient': sorted(ok, key=size),
        'denominator': denom,
        'mean_automaton_states': sum(autos) / len(autos),
        'mean_denominator_states': sum(dens) / len(dens),
        'mean_history_states': sum(len(x) for x in insts) / len(insts),
        'mean_honest_ratio': sum(a / b for a, b in zip(dens, autos)) / len(autos),
    }


def main(out='results/family_design.json'):
    grid = [(False, 'A 纯无序，固定族', [(3, 0, 2), (4, 0, 2), (5, 0, 2)]),
            (True, 'B 敏感对随机，无桥段', [(3, 0, 2), (4, 0, 2), (5, 0, 2)]),
            (True, 'C 敏感对随机，加共享桥段',
             [(3, 2, 2), (3, 4, 2), (4, 2, 2), (4, 4, 2), (4, 6, 2),
              (5, 2, 2), (5, 4, 2)])]
    rows = []
    for randomised, tag, cells in grid:
        for n, d, m in cells:
            r = analyse(n, d, m, randomised); r['tag'] = tag; rows.append(r)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, 'w'), indent=2)
    tag = None
    for r in rows:
        if r['tag'] != tag:
            tag = r['tag']
            print(f"\n{tag}")
            print(f"  {'n':>2} {'d':>2} {'m':>2} {'实例':>4} {'|Q|':>6} {'|H|':>7}"
                  f" {'最小充分分母':>14} {'|D|':>7} {'诚实比':>7}")
        print(f"  {r['n']:>2} {r['d']:>2} {r['m']:>2} {r['instances']:>4}"
              f" {r['mean_automaton_states']:>6.1f} {r['mean_history_states']:>7.1f}"
              f" {r['denominator']:>14} {r['mean_denominator_states']:>7.1f}"
              f" {r['mean_honest_ratio']:>7.2f}")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main(*sys.argv[1:])
