"""Build and verify maps for the randomised order-sensitive family.

Task. Complete n labels in any order. Whichever member of the instance's
sensitive pair finishes first sets a bit. Then walk a bridge of d labels shared
by both branches, then a tail of m labels selected by that bit. Wrong events
are ignored.

Forcing. A deterministic policy from one start follows one fixed trajectory, so
position and count alone would already suffice for it. Diversity has to come
from the environment: training alternates over a set of starts, evaluation
weights them equally, and starts are chosen so each has a different unique
optimal ordering with both bit values represented. This is the Stage 7 device
carried over to this family.

Everything here is verified before any learning: the junction is a cut vertex,
count aliases starts that need different next events, the minimum sufficient
window is measured rather than assumed, and optimal route lengths are recorded
so that no ordering is cheap by accident.
"""
from collections import deque
from functools import lru_cache
from pathlib import Path
import hashlib, itertools, json, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_maps import build, funnel_check

ROOT = Path(__file__).resolve().parents[1]
POOL = 3


class Task:
    """State machine for one instance, plus its Myhill-Nerode minimisation.

    `m2` lets the two branch tails have different lengths. That is what makes
    progress depth and task-graph distance come apart: at the end of the shared
    bridge both branches stand on the same cell with the same progress count,
    but the number of transitions still owed differs. With m2 equal to m the
    family is the earlier one, where depth is graph distance and the two cannot
    be told apart.
    """

    # When `fatal` is set, choosing the other branch's first tail label at the
    # commit point ends the episode in failure instead of being ignored. That is
    # the one place where position and progress count are identical across the
    # two histories, so the unrecoverable mistake is a memory mistake and never a
    # navigation mistake. Everything else stays reversible.
    def __init__(self, n, pair, d, m, m2=None, fatal=False):
        self.n, self.pair, self.d, self.m, self.fatal = n, pair, d, m, fatal
        self.m2 = m if m2 is None else m2
        self.alphabet = n + POOL
        self.bridge = [n + (k % POOL) for k in range(d)]
        last = self.bridge[-1] if d else None
        heads = [n + c for c in range(POOL) if n + c != last]
        lengths = [m, self.m2]
        self.tails = [[n + ((heads[b] - n + k) % POOL) for k in range(lengths[b])]
                      for b in range(2)]
        self.full = (1 << n) - 1
        self.dead = ('dead',)
        self.states = [(0, -1, 0)]
        self.index = {self.states[0]: 0}
        self.trans = []
        frontier = [self.states[0]]
        while frontier:
            nxt = []
            for st in frontier:
                row = [self.index[st]] * self.alphabet
                for e in range(self.alphabet):
                    dest = self.step(st, e)
                    if dest not in self.index:
                        self.index[dest] = len(self.states)
                        self.states.append(dest)
                        nxt.append(dest)
                    row[e] = self.index[dest]
                while len(self.trans) <= self.index[st]:
                    self.trans.append(None)
                self.trans[self.index[st]] = row
            frontier = nxt
        self.accepting = {i for i, st in enumerate(self.states) if self.done(st)}
        self.failing = {i for i, st in enumerate(self.states) if self.failed(st)}
        self.minimal, self.n_minimal = self.minimise()

    def suffix(self, bit):
        return self.bridge + self.tails[bit]

    def done(self, st):
        if st == self.dead:
            return False
        mask, bit, k = st
        return mask == self.full and k == len(self.suffix(bit))

    def failed(self, st):
        return st == self.dead

    def commit_point(self, st):
        """The one state where both branches look alike and must choose."""
        if st == self.dead:
            return False
        mask, bit, k = st
        return self.fatal and mask == self.full and k == self.d and bit in (0, 1)

    def step(self, st, e):
        if st == self.dead:
            return st
        mask, bit, k = st
        if self.done(st):
            return st
        if self.commit_point(st) and e == self.tails[1 - bit][0] and e != self.tails[bit][0]:
            return self.dead
        if mask != self.full:
            if 0 <= e < self.n and not mask >> e & 1:
                if bit < 0 and e in self.pair:
                    bit = 0 if e == self.pair[0] else 1
                return (mask | 1 << e, bit, 0)
            return st
        seq = self.suffix(bit)
        return (mask, bit, k + 1) if e == seq[k] else st

    def minimise(self):
        part = [2 if i in self.failing else 1 if i in self.accepting else 0
                for i in range(len(self.states))]
        while True:
            sig = {}
            new = []
            for i in range(len(self.states)):
                key = (part[i], tuple(part[self.trans[i][e]] for e in range(self.alphabet)))
                new.append(sig.setdefault(key, len(sig)))
            if new == part:
                return part, len(sig)
            part = new

    def run(self, events):
        st = self.states[0]
        for e in events:
            st = self.step(st, e)
        return st

    def required(self, st):
        """Labels that advance the task from this state; empty when finished."""
        return [e for e in range(self.alphabet) if self.step(st, e) != st]


def product_shortest(layout, task):
    """Backward BFS over (task state, cell); no learning input."""
    cells, moves = layout['cells'], layout['moves']
    N, T = len(cells), len(task.states)
    rev = [[] for _ in range(T * N)]
    for q in range(T):
        if q in task.accepting or q in task.failing:
            continue
        for s, row in enumerate(moves):
            for ns, event in row:
                nq = task.trans[q][event] if event >= 0 else q
                rev[nq * N + ns].append(q * N + s)
    dist = [-1] * (T * N)
    queue = deque()
    for q in task.accepting:
        for s in range(N):
            dist[q * N + s] = 0
            queue.append(q * N + s)
    while queue:
        z = queue.popleft()
        for prev in rev[z]:
            if dist[prev] < 0:
                dist[prev] = dist[z] + 1
                queue.append(prev)
    return dist, N


def optimal_orderings(layout, task, dist, N, start):
    """Every optimal successful-event sequence from one start, as a set."""
    @lru_cache(None)
    def walk(z):
        q, s = divmod(z, N)
        if q in task.accepting:
            return frozenset([()])
        out = set()
        for ns, event in layout['moves'][s]:
            nq = task.trans[q][event] if event >= 0 else q
            dest = nq * N + ns
            if dist[dest] == dist[z] - 1:
                head = (event,) if nq != q else ()
                for rest in walk(dest):
                    out.add(head + rest)
        return frozenset(out)
    result = walk(start)
    walk.cache_clear()
    return result


def trajectory(layout, task, dist, N, start):
    """The optimal path from a start, as (cell, events-so-far) checkpoints."""
    z = start
    q, s = divmod(z, N)
    seen, path = [], []
    while q not in task.accepting:
        for ns, event in layout['moves'][s]:
            nq = task.trans[q][event] if event >= 0 else q
            if dist[nq * N + ns] == dist[q * N + s] - 1:
                path.append((s, tuple(seen), q))
                if nq != q:
                    seen.append(event)
                q, s = nq, ns
                break
        else:
            raise RuntimeError('no descending move')
    path.append((s, tuple(seen), q))
    return path


def feature_fns(n, alphabet, max_w):
    fns = {'count': lambda h: (len(h),), 'bag': lambda h: tuple(sorted(h))}
    for w in range(1, max_w + 1):
        fns[f'window{w}'] = lambda h, w=w: (h[-w:], len(h))
    fns['history'] = lambda h: h
    return fns


def evaluate_layout(layout, task, starts_wanted):
    dist, N = product_shortest(layout, task)
    picked = {}
    for start in layout['candidate_starts']:
        if dist[start] < 0:
            continue
        orders = optimal_orderings(layout, task, dist, N, start)
        if len(orders) != 1:
            continue
        order = next(iter(orders))
        key = order[:task.n]
        if key not in picked or dist[start] < picked[key]['length']:
            picked[key] = dict(start=start, length=dist[start], history=order)
    chosen = sorted(picked.values(), key=lambda r: (r['history'], r['start']))[:starts_wanted]
    if len(chosen) < 2:
        return None
    bits = {task.run(r['history'][:task.n])[1] for r in chosen}
    if bits != {0, 1}:
        return None

    prefixes = set()
    for r in chosen:
        for i in range(len(r['history']) + 1):
            prefixes.add(tuple(r['history'][:i]))
    minimal_of = lambda h: task.minimal[task.index[task.run(h)]]
    fns = feature_fns(task.n, task.alphabet, task.n + task.d + task.m)

    # A representation must fix the next required label wherever the agent
    # stands. Alias two optimal checkpoints with the same cell and feature but
    # different requirements and a stationary policy cannot serve both.
    checkpoints = []
    for r in chosen:
        for cell, hist, _ in trajectory(layout, task, dist, N, r['start']):
            checkpoints.append((cell, hist))
    def aliased(name):
        seen = {}
        for cell, hist in checkpoints:
            key = (cell, fns[name](hist))
            req = tuple(task.required(task.run(hist)))
            if seen.setdefault(key, req) != req:
                return True
        return False

    windows = [w for w in range(0, task.n + task.d + task.m + 1)]
    ok = {}
    for name in ['count', 'bag'] + [f'window{w}' for w in range(1, max(windows) + 1)] + ['history']:
        ok[name] = not aliased(name)
    sufficient_windows = [w for w in range(1, max(windows) + 1) if ok[f'window{w}']]
    sizes = {name: len({fns[name](h) for h in prefixes}) for name in ok}
    candidates = [name for name, good in ok.items() if good]
    denom = min(candidates, key=lambda name: sizes[name]) if candidates else 'history'
    q_states = len({minimal_of(h) for h in prefixes})
    return dict(
        starts=[r['start'] for r in chosen],
        optimal_histories=[list(r['history']) for r in chosen],
        optimal_lengths=[r['length'] for r in chosen],
        orders=len(chosen),
        history_states=len(prefixes),
        minimal_states=q_states,
        feature_sizes=sizes,
        feature_sufficient=ok,
        min_sufficient_window=sufficient_windows[0] if sufficient_windows else None,
        count_aliased=not ok['count'],
        bag_aliased=not ok['bag'],
        denominator=denom,
        denominator_states=sizes[denom],
        honest_ratio=sizes[denom] / q_states,
        max_length=max(r['length'] for r in chosen),
    )


def search(n, d, m, instances, seeds, starts_wanted, side=9):
    best = None
    for seed in seeds:
        layout = build(n, instances, seed, side=side)
        if not funnel_check(layout):
            continue
        for pair in itertools.combinations(range(n), 2):
            task = Task(n, pair, d, m)
            info = evaluate_layout(layout, task, starts_wanted)
            if info is None or not info['count_aliased']:
                continue
            score = (-info['orders'], info['max_length'])
            if best is None or score < best[0]:
                best = (score, layout, task, info, seed, pair)
    return best


def main(out='results/family_maps'):
    started = time.monotonic()
    cells = [(3, 0, 2), (3, 4, 2), (4, 0, 2), (4, 4, 2)]
    outdir = ROOT / out
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, d, m in cells:
        seeds = range(90000 + 1000 * n + 10 * d, 90000 + 1000 * n + 10 * d + 24)
        found = search(n, d, m, instances=max(2, n), seeds=seeds, starts_wanted=12)
        if found is None:
            rows.append(dict(n=n, d=d, m=m, status='not found'))
            print(f'n={n} d={d} m={m}: no layout', flush=True)
            continue
        _, layout, task, info, seed, pair = found
        record = dict(n=n, d=d, m=m, seed=seed, pair=list(pair), status='ok',
                      bridge=task.bridge, tails=task.tails,
                      task_states=len(task.states), minimal_total=task.n_minimal, **info)
        (outdir / f'family_n{n}_d{d}_m{m}.json').write_text(
            json.dumps({**layout, **record}, indent=2) + '\n')
        rows.append(record)
        print(f"n={n} d={d} m={m} seed={seed} pair={pair} orders={info['orders']} "
              f"|H|={info['history_states']} |Q|={info['minimal_states']} "
              f"denom={info['denominator']}({info['denominator_states']}) "
              f"ratio={info['honest_ratio']:.2f} minw={info['min_sufficient_window']} "
              f"len<={info['max_length']}", flush=True)
    (outdir / 'summary.json').write_text(json.dumps(
        dict(rows=rows, seconds=time.monotonic() - started,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
