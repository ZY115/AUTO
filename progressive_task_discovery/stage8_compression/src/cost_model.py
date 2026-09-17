"""Which cost model actually fits, and what the answer says about compression.

Step 3 fitted cost against state count with a straight line and read the
intercept as a floor that compression could never touch. Fitting a straight line
to a concave curve manufactures exactly such an intercept. Here the affine model
is compared against a power law with no intercept, and against one that lets the
optimal route length in multiplicatively.

Only representations that are sufficient for the task are used. An insufficient
one is solving a harder problem and does not lie on the same curve; including
the bag and narrow-window arms was what made the earlier per-task fits erratic.
"""
import collections, json, math, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ols(cols, y):
    n, k = len(y), len(cols)
    A = [[sum(cols[i][t] * cols[j][t] for t in range(n)) for j in range(k)] for i in range(k)]
    b = [sum(cols[i][t] * y[t] for t in range(n)) for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, k):
            f = A[r][i] / A[i][i]
            for c in range(i, k):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, k))) / A[i][i]
    return x


def r2(pred, y):
    my = statistics.mean(y)
    return 1 - sum((a - b) ** 2 for a, b in zip(pred, y)) / sum((v - my) ** 2 for v in y)


def collect(raw, rungs, arms, gamma=None):
    rows = [json.loads(l) for l in (ROOT / raw).open()]
    if gamma is not None:
        rows = [r for r in rows if r['gamma'] == gamma]
    metas = {m['tag']: m for m in json.loads((ROOT / rungs).read_text())}
    g = collections.defaultdict(list)
    for r in rows:
        g[(r['rung'], r['arm'])].append(r)
    cap = lambda rs: statistics.mean((x['first90'] if x['first90'] > 0 else x['budget']) for x in rs)
    pts = []
    for tag, m in metas.items():
        length = statistics.mean(m['lengths'])
        for arm in arms(m):
            rs = g[(tag, arm)]
            if not rs or not all(x['first90'] > 0 for x in rs):
                continue
            pts.append((statistics.mean(x['features'] for x in rs), length, cap(rs)))
    return pts


def report(name, pts):
    X = [p[0] for p in pts]
    L = [p[1] for p in pts]
    Y = [p[2] for p in pts]
    one = [1.0] * len(Y)
    a, b = ols([one, X], Y)
    lx, ly, ll = [math.log(v) for v in X], [math.log(v) for v in Y], [math.log(v) for v in L]
    c0, q = ols([one, lx], ly)
    d0, p1, q1 = ols([one, ll, lx], ly)
    print(f"\n{name}: {len(pts)} points, route length {min(L):.0f} to {max(L):.0f}")
    print(f"  affine     {a:8.0f} + {b:5.0f} x states          R2 = {r2([a + b * x for x in X], Y):.3f}")
    print(f"  power      {math.exp(c0):8.0f} x states^{q:.2f}              R2 = "
          f"{r2([math.exp(c0 + q * x) for x in lx], Y):.3f}")
    print(f"  with route {math.exp(d0):8.2f} x length^{p1:.2f} x states^{q1:.2f}  R2 = "
          f"{r2([math.exp(d0 + p1 * l + q1 * x) for l, x in zip(ll, lx)], Y):.3f}")
    print(f"  halving the state count buys {2 ** -q1:.2f}, not 0.50")
    return dict(name=name, points=len(pts), affine_intercept=a, affine_slope=b,
                power_exponent=q, joint_length=p1, joint_states=q1)


def main(out='results/cost_model.json'):
    rows = []
    rows.append(report('step 3 ladder',
                       collect('results/ladder32/raw.jsonl', 'results/family_maps/rungs.json',
                               lambda m: ('automaton', 'history', f"window{m['min_sufficient_window']}"))))
    rows.append(report('depth-versus-graph tasks',
                       collect('results/floor_gamma.jsonl', 'results/depth_vs_graph/rungs12.json',
                               lambda m: ('automaton', 'history'), gamma=.99)))
    pooled = (collect('results/ladder32/raw.jsonl', 'results/family_maps/rungs.json',
                      lambda m: ('automaton', 'history')) +
              collect('results/floor_gamma.jsonl', 'results/depth_vs_graph/rungs12.json',
                      lambda m: ('automaton', 'history'), gamma=.99))
    rows.append(report('both pooled', pooled))
    (ROOT / out).write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
