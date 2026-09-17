"""Export HRM CraftWorld task structure into this project's .task format.

Step 37's audit found that HRM CraftWorld does *not* repeat its subgoals more
than our own family does — our n5 tasks demand each event from 13.1 live states
against CraftWorld's best of 6. What CraftWorld really adds is that reaching a
subgoal is physically expensive: 13x13 four rooms with walls instead of our open
28-40 cells. So the thing worth testing is not "does CraftWorld behave
differently" but the factor underneath it:

    is skill reuse worth anything because subgoals repeat, or because each
    repetition costs a lot to learn?

That needs the task structure held fixed while navigation cost varies, which is
what this file builds. The three public task structures (book, book-and-quill,
cake) are taken verbatim; only the map changes.

**Soundness of the conversion.** HRM edges are DNF formulas, and some carry
negative literals — `y&~s` at a state that also has an `s` edge. Those negations
only break ties when two observables are seen at once. Our format fires exactly
one event per (cell, action), and this exporter places at most one object per
cell, so no two observables can ever co-occur and every negative literal is
vacuously true. The exporter asserts the shape it relies on rather than assuming
it.

Deliberately *not* varied in this round, so the cause stays identifiable: no
lava (no failing states), one object per class, four-way movement with no
orientation. Only the map's size and topology move.
"""
import json, sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[1] / 'external'
sys.path.insert(0, str(EXT))
from hrm_craftworld_audit import hierarchy          # stubs gym_minigrid on import

OUT = ROOT / 'results/craftworld'
TASKS = ['book', 'book-and-quill', 'cake']
INF = 10 ** 6


# --------------------------------------------------------------------------
# task structure
# --------------------------------------------------------------------------
def automaton(task):
    """Flatten one HRM hierarchy into (alphabet, trans, accepting, failing)."""
    a = hierarchy(task, flat=True).get_root_automaton()
    states = sorted(a.get_states())
    sid = {s: i for i, s in enumerate(states)}
    edges = []                                   # (from, positive literal, to)
    alpha = set()
    for st in states:
        for c, to in a.get_outgoing_conditions(st):
            assert not c.is_call(), f'{task}: flat hierarchy still calls out'
            lits = str(c.get_formula()).strip('()').split('&')
            pos = [l for l in lits if not l.startswith('~')]
            assert len(pos) == 1, f'{task}: {lits} is not one positive literal'
            edges.append((st, pos[0], to))
            alpha.add(pos[0])
    alphabet = sorted(alpha)
    aid = {l: i for i, l in enumerate(alphabet)}
    trans = [[i for _ in alphabet] for i in range(len(states))]
    for f, l, t in edges:
        trans[sid[f]][aid[l]] = sid[t]
    accepting = [1 if a.is_accept_state(s) else 0 for s in states]
    failing = [1 if a.has_reject_state() and a.is_reject_state(s) else 0
               for s in states]
    return dict(task=task, alphabet=alphabet, states=states, trans=trans,
                accepting=accepting, failing=failing)


def minimise(trans, accepting, failing):
    n, A = len(trans), len(trans[0])
    cls = [2 if failing[i] else 1 if accepting[i] else 0 for i in range(n)]
    while True:
        sig, new = {}, []
        for i in range(n):
            k = (cls[i],) + tuple(cls[trans[i][e]] for e in range(A))
            new.append(sig.setdefault(k, len(sig)))
        if new == cls:
            return cls
        cls = new


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------
def grid(kind, size):
    """Free cells of a minigrid-style layout. Border walls always; four rooms
    adds a cross wall with one door per segment, which is what makes a detour
    expensive rather than merely long."""
    wall = set()
    for i in range(size):
        wall |= {(i, 0), (i, size - 1), (0, i), (size - 1, i)}
    if kind == 'four_rooms':
        mid = size // 2
        doors = {(mid, mid // 2), (mid, mid + mid // 2),
                 (mid // 2, mid), (mid + mid // 2, mid)}
        for i in range(size):
            for c in ((mid, i), (i, mid)):
                if c not in doors:
                    wall.add(c)
    return [(x, y) for x in range(size) for y in range(size) if (x, y) not in wall]


def build_map(kind, size, alphabet, seed):
    """Place one object per observable, then tabulate (cell, action) movement."""
    cells = grid(kind, size)
    index = {c: i for i, c in enumerate(cells)}
    import random
    rng = random.Random(seed * 1000 + len(alphabet))
    spots = rng.sample(cells, len(alphabet))
    label = {index[c]: i for i, c in enumerate(spots)}
    moves, events = [], []
    for c in cells:
        m, e = [], []
        for a in range(4):
            x, y = c
            t = (x, y + 1) if a == 0 else (x, y - 1) if a == 1 else \
                (x - 1, y) if a == 2 else (x + 1, y)
            nxt = t if t in index else c
            m.append(index[nxt])
            e.append(label.get(index[nxt], -1))
        moves.append(m); events.append(e)
    return dict(cells=cells, index=index, moves=moves, events=events, label=label)


def product_distance(d, g):
    """Steps from every (task state, cell) to acceptance."""
    S, N = len(d['trans']), len(g['cells'])
    dist = [[INF] * N for _ in range(S)]
    q = deque()
    for s in range(S):
        if d['accepting'][s]:
            for c in range(N):
                dist[s][c] = 0; q.append((s, c))
    back = [[] for _ in range(N)]
    for pc in range(N):
        for a in range(4):
            back[g['moves'][pc][a]].append((pc, g['events'][pc][a]))
    while q:
        s, c = q.popleft()
        for pc, ev in back[c]:
            for ps in range(S):
                if d['failing'][ps]:
                    continue
                ns = ps if ev < 0 else d['trans'][ps][ev]
                if ns != s or dist[ps][pc] <= dist[s][c] + 1:
                    continue
                dist[ps][pc] = dist[s][c] + 1
                q.append((ps, pc))
    return dist


def event_cost(d, g, starts):
    """C_g: mean shortest path from the start set to firing event g, and N_g:
    how many live task states demand it. Their product is the redundant
    navigation a non-sharing learner has to pay for."""
    N = len(g['cells'])
    back = [[] for _ in range(N)]
    for pc in range(N):
        for a in range(4):
            back[g['moves'][pc][a]].append((pc, g['events'][pc][a]))
    out = {}
    for e in range(len(d['alphabet'])):
        dd = [INF] * N
        q = deque()
        for c in range(N):
            # One, not zero: standing next to the event still costs the step that
            # fires it. Zero here understated every C_g by exactly one step.
            if any(g['events'][c][a] == e for a in range(4)):
                dd[c] = 1; q.append(c)
        while q:
            c = q.popleft()
            for pc, _ in back[c]:
                if dd[pc] > dd[c] + 1:
                    dd[pc] = dd[c] + 1; q.append(pc)
        live = [s for s in range(len(d['trans']))
                if not d['failing'][s] and not d['accepting'][s]]
        ng = sum(1 for s in live
                 if d['trans'][s][e] != s and not d['failing'][d['trans'][s][e]])
        reach = [dd[c] for c in starts if dd[c] < INF]
        out[d['alphabet'][e]] = dict(C=sum(reach) / len(reach) if reach else None,
                                     N=ng)
    return out


def emit(d, g, path):
    S, N = len(d['trans']), len(g['cells'])
    minimal = minimise(d['trans'], d['accepting'], d['failing'])
    dist = product_distance(d, g)
    reach = sorted((c for c in range(N) if dist[0][c] < INF), key=lambda c: dist[0][c])
    starts = reach[len(reach) // 6::max(1, len(reach) // 8)][:8] or reach[:4]
    lengths = [dist[0][c] for c in starts]
    horizon = 2 * max(lengths)
    A = len(d['alphabet'])
    L = [f'{A} 0 0 0 0 {A}', str(S), ' '.join(map(str, minimal)),
         ' '.join(map(str, d['accepting'])), ' '.join(map(str, d['failing']))]
    L += [' '.join(map(str, r)) for r in d['trans']]
    L.append(str(N))
    for c in range(N):
        L.append(' '.join(f"{g['moves'][c][a]} {g['events'][c][a]}" for a in range(4)))
    L += [str(len(starts)), ' '.join(map(str, starts)), ' '.join(map(str, lengths)),
          f'{horizon} {max(lengths)} 1']
    cap = lambda v: min(v, 999)
    for _ in range(3):                       # the format expects three blocks
        for s in range(S):
            L.append(' '.join(str(cap(dist[s][c])) for c in range(N)))
    path.write_text('\n'.join(L) + '\n')
    cost = event_cost(d, g, starts)
    c_reuse = sum((v['N'] - 1) * v['C'] for v in cost.values()
                  if v['C'] is not None and v['N'] > 1)
    return dict(states=S, minimal=len(set(minimal)), cells=N, alphabet=d['alphabet'],
                starts=len(starts), horizon=horizon, lengths=lengths,
                c_reuse=round(c_reuse, 2),
                cost={k: (round(v['C'], 2) if v['C'] is not None else None, v['N'])
                      for k, v in cost.items()})


CONDS = [('cheap', 'open_plan', 7), ('expensive', 'four_rooms', 13)]

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    print(f'{"任务":<16} {"条件":<10} {"格":>5} {"状态":>5} {"起点最短路":>11} '
          f'{"时域":>6} {"C_reuse":>9}')
    for task in TASKS:
        d = automaton(task)
        for cond, kind, size in CONDS:
            for seed in range(6):
                g = build_map(kind, size, d['alphabet'], seed)
                tag = f'cw_{task.replace("-","")}_{cond}_{seed}'
                m = emit(d, g, OUT / f'{tag}.task')
                m.update(tag=tag, task=task, cond=cond, kind=kind, size=size, seed=seed)
                meta.append(m)
                if seed == 0:
                    print(f'{task:<16} {cond:<10} {m["cells"]:>5} {m["states"]:>5} '
                          f'{min(m["lengths"])}-{max(m["lengths"]):<7} '
                          f'{m["horizon"]:>6} {m["c_reuse"]:>9}')
    (OUT / 'tasks.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'\n写出 {len(meta)} 个任务到 {OUT}')
