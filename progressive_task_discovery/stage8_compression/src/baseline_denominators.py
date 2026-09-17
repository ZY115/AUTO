"""Compute the honest denominator for the compression ladder.

For every searched layout, enumerate the prefixes of the optimal histories and
count task-feature states under each candidate representation:

  full history            every distinct prefix
  minimal automaton       (completed unordered set, suffix index [, order bit])
  bag of events           (completed unordered set, suffix index)
  sliding window w        (last w successful events, progress count)

A representation is *sufficient* when it never maps two prefixes with different
minimal-automaton states onto the same feature. The compression ratio must be
reported against the smallest sufficient representation, not against full
history: a representation that is both sufficient and generic is a baseline a
reader will propose, so it belongs in the denominator.

Variant A is the pure unordered task. Variant B additionally makes the required
suffix depend on which of labels 0 and 1 was completed first, which places the
deciding evidence earlier than any bounded window can retain.
"""
import json, glob, os, sys
from itertools import permutations

def prefixes(layout, scope):
    """Task-feature histories to score.

    scope='optimal'    prefixes of the optimal trajectories only; what the
                       converged greedy policy must represent.
    scope='reachable'  every prefix the task language admits; what an
                       exploring learner can actually enter, and therefore the
                       set a representation must be sufficient over.
    """
    out = set()
    if scope == 'optimal':
        hs = layout['optimal_histories']
    else:
        n, suffix = layout['n'], tuple(layout['suffix'])
        hs = [list(perm) + list(suffix) for perm in permutations(range(n))]
    for h in hs:
        for i in range(len(h) + 1):
            out.add(tuple(h[:i]))
    return out

def bag_state(p, n):
    return (tuple(sorted(set(p[:min(len(p), n)]))), max(0, len(p) - n))

def true_state(p, n, order_sensitive):
    s = bag_state(p, n)
    if not order_sensitive:
        return s
    head = p[:min(len(p), n)]
    bit = None
    if 0 in head and 1 in head:
        bit = 0 if head.index(0) < head.index(1) else 1
    return s + (bit,)

def window_state(p, w):
    return (tuple(p[-w:]) if w > 0 else (), len(p))

def sufficient(pre, feature, truth):
    seen = {}
    for p in pre:
        f, t = feature(p), truth(p)
        if seen.setdefault(f, t) != t:
            return False
    return True

def analyse(layout, order_sensitive, scope):
    n = layout['n']
    pre = prefixes(layout, scope)
    truth = lambda p: true_state(p, n, order_sensitive)
    rec = {
        'target_bin': layout['target_bin'], 'n': n, 'm': layout['m'],
        'orders': layout['orders'], 'scope': scope,
        'history_states': len(pre),
        'automaton_states': len({truth(p) for p in pre}),
        'bag_states': len({bag_state(p, n) for p in pre}),
        'bag_sufficient': sufficient(pre, lambda p: bag_state(p, n), truth),
        'windows': {},
    }
    for w in range(0, n + 1):
        rec['windows'][w] = {
            'states': len({window_state(p, w) for p in pre}),
            'sufficient': sufficient(pre, lambda p, w=w: window_state(p, w), truth),
        }
    ok = [w for w in sorted(rec['windows']) if rec['windows'][w]['sufficient']]
    rec['min_sufficient_window'] = ok[0] if ok else None
    if ok:
        rec['denominator_states'] = min(rec['windows'][ok[0]]['states'], rec['history_states'])
        rec['denominator'] = f'window_{ok[0]}'
    else:
        rec['denominator_states'] = rec['history_states']
        rec['denominator'] = 'full_history'
    rec['ratio_vs_full_history'] = rec['history_states'] / rec['automaton_states']
    rec['ratio_vs_denominator'] = rec['denominator_states'] / rec['automaton_states']
    return rec

def main(outdir='results'):
    paths = sorted(glob.glob('docs/layouts/layout_*.json'),
                   key=lambda p: float(os.path.basename(p)[7:-5]))
    report = {}
    for scope in ('optimal', 'reachable'):
        for name, osens in (('A_pure_unordered', False), ('B_order_sensitive', True)):
            key = f'{scope}__variant_{name}'
            report[key] = [analyse(json.load(open(path)), osens, scope) for path in paths]
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, 'baseline_denominators.json')
    json.dump(report, open(dest, 'w'), indent=2)
    for variant, rows in report.items():
        print(variant)
        print(f"  {'bin':>5} {'|H|':>5} {'|Q|':>4} {'bag ok':>7} {'min w':>6} {'denom':>13} {'H/Q':>6} {'D/Q':>6}")
        for r in rows:
            print(f"  {r['target_bin']:>5} {r['history_states']:>5} {r['automaton_states']:>4}"
                  f" {str(r['bag_sufficient']):>7} {str(r['min_sufficient_window']):>6}"
                  f" {r['denominator']:>13} {r['ratio_vs_full_history']:>6.2f}"
                  f" {r['ratio_vs_denominator']:>6.2f}")
    print(f"\nwrote {dest}")

if __name__ == '__main__':
    main(*sys.argv[1:])
