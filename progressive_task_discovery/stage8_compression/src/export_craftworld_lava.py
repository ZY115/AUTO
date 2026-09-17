"""Step 38.5: the missing cell of the 2x2 — irreversibility.

Step 38 varied navigation cost and found reuse is worth 3.79x more when skills
are expensive to learn. It deliberately left out lava, so the one variable this
whole project started from — irreversible failure — has never met skill reuse.

The design is paired to the cell. For every task, map size and map seed, the
**same** grid is emitted twice:

    safe   the hazard label is present and inert: a self-loop, ignored
    lava   the same label at the same cells ends the episode in failure

So the alphabet, the object positions, the hazard positions, the walls and every
shortest path are identical between the two. Hazard cells stay walkable — they
are not turned into walls — so the geometry a learner faces is untouched and the
only thing that changes is what a wrong step costs. Start cells and the horizon
are computed once, from the lava variant, and reused for the safe one, so even
the evaluation protocol is shared.

Hazard density is 8% of free cells, which is 2 on the cheap map and 8 on the
expensive one: the same dose per unit of space, not the same count.
"""
import json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from export_craftworld import (hierarchy, grid, minimise, product_distance,
                               event_cost, TASKS, CONDS, INF)

OUT = ROOT / os.environ.get('CW_LAVA_OUT', 'results/craftworld_lava')
# Hazard density as a fraction of free cells. It is a free parameter, and it has
# to be chosen so the comparison is measurable at all: at 8% every arm including
# the best one fails to reach 90% on the hardest cell within a million steps, and
# a ratio between two censored arms says nothing. The dose picked below is the
# one where the strongest arm succeeds and the weaker arms visibly struggle,
# which is the only regime where the contrast can be read.
DENSITY = float(os.environ.get('CW_LAVA_DENSITY', 0.03))


def automaton(task, lava):
    """Flat HRM automaton plus a hazard symbol, inert or fatal."""
    a = hierarchy(task, flat=True).get_root_automaton()
    states = sorted(a.get_states())
    sid = {s: i for i, s in enumerate(states)}
    edges, alpha = [], set()
    for st in states:
        for c, to in a.get_outgoing_conditions(st):
            assert not c.is_call()
            pos = [l for l in str(c.get_formula()).strip('()').split('&')
                   if not l.startswith('~')]
            assert len(pos) == 1
            edges.append((st, pos[0], to)); alpha.add(pos[0])
    alphabet = sorted(alpha) + ['LAVA']
    aid = {l: i for i, l in enumerate(alphabet)}
    n = len(states) + (1 if lava else 0)
    trans = [[i for _ in alphabet] for i in range(n)]
    for f, l, t in edges:
        trans[sid[f]][aid[l]] = sid[t]
    accepting = [1 if a.is_accept_state(s) else 0 for s in states]
    failing = [0] * len(states)
    if lava:
        dead = len(states)
        accepting.append(0); failing.append(1)
        for q in range(len(states)):
            if not accepting[q]:
                trans[q][aid['LAVA']] = dead
    return dict(task=task, alphabet=alphabet, trans=trans,
                accepting=accepting, failing=failing)


def build_map(kind, size, n_obj, seed, attempt=0):
    """Objects first, then hazards on cells no object holds.

    `attempt` reshuffles only the hazards: a hazard can sit on the one square
    that made a required object safely reachable, and then the task cannot be
    completed at all. Object positions stay put across attempts so the pairing
    with the safe cell is unaffected.
    """
    cells = grid(kind, size)
    index = {c: i for i, c in enumerate(cells)}
    rng = random.Random(seed * 1000 + n_obj)
    spots = rng.sample(cells, n_obj)
    label = {index[c]: i for i, c in enumerate(spots)}
    free = [c for c in cells if index[c] not in label]
    # Zero is a legitimate dose: it is the bottom rung of the robustness sweep,
    # and it must mean no hazard at all rather than one.
    n_lava = round(DENSITY * len(cells))
    if DENSITY > 0:
        n_lava = max(1, n_lava)
    hazards = random.Random(seed * 1000 + n_obj + 7919 * attempt).sample(free, n_lava)
    for c in hazards:
        label[index[c]] = n_obj                 # the hazard symbol is last
    moves, events = [], []
    for c in cells:
        m, e = [], []
        for a in range(4):
            x, y = c
            t = (x, y + 1) if a == 0 else (x, y - 1) if a == 1 else \
                (x - 1, y) if a == 2 else (x + 1, y)
            nxt = t if t in index else c
            m.append(index[nxt]); e.append(label.get(index[nxt], -1))
        moves.append(m); events.append(e)
    return dict(cells=cells, index=index, moves=moves, events=events,
                label=label, hazards=[index[c] for c in hazards])


def emit_pair(task, kind, size, seed, tagbase):
    dl = automaton(task, lava=True)
    ds = automaton(task, lava=False)
    for attempt in range(200):
        g = build_map(kind, size, len(dl['alphabet']) - 1, seed, attempt)
        # One protocol for both cells: the lava variant decides it, because the
        # safe one would otherwise take shorter routes straight through a hazard.
        dist = product_distance(dl, g)
        N = len(g['cells'])
        haz = set(g['hazards'])
        reach = sorted((c for c in range(N) if dist[0][c] < INF and c not in haz),
                       key=lambda c: dist[0][c])
        if len(reach) >= 8:
            break
    else:
        raise RuntimeError(f'{tagbase}: no hazard layout leaves the task solvable')
    starts = reach[len(reach) // 6::max(1, len(reach) // 8)][:8] or reach[:4]
    lengths = [dist[0][c] for c in starts]
    horizon = 2 * max(lengths)
    out = {}
    for lab, d in (('lava', dl), ('safe', ds)):
        S = len(d['trans'])
        dd = product_distance(d, g)
        minimal = minimise(d['trans'], d['accepting'], d['failing'])
        A = len(d['alphabet'])
        L = [f'{A} 0 0 0 0 {A}', str(S), ' '.join(map(str, minimal)),
             ' '.join(map(str, d['accepting'])), ' '.join(map(str, d['failing']))]
        L += [' '.join(map(str, r)) for r in d['trans']]
        L.append(str(N))
        for c in range(N):
            L.append(' '.join(f"{g['moves'][c][a]} {g['events'][c][a]}"
                              for a in range(4)))
        L += [str(len(starts)), ' '.join(map(str, starts)),
              ' '.join(map(str, lengths)), f'{horizon} {max(lengths)} 1']
        cap = lambda v: min(v, 999)
        for _ in range(3):
            for q in range(S):
                L.append(' '.join(str(cap(dd[q][c])) for c in range(N)))
        tag = f'{tagbase}_{lab}'
        (OUT / f'{tag}.task').write_text('\n'.join(L) + '\n')
        cost = event_cost(d, g, starts)
        out[lab] = dict(tag=tag, states=S, cells=N, hazards=len(haz),
                        horizon=horizon, lengths=lengths,
                        c_g=round(sum(v['C'] for v in cost.values()
                                      if v['C'] is not None) / A, 2))
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    print(f'{"任务":<16} {"导航":<10} {"格":>5} {"危险格":>7} {"状态(safe/lava)":>16} '
          f'{"时域":>6} {"平均 C_g":>9}')
    only = os.environ.get('CW_LAVA_CONDS')
    conds = [c for c in CONDS if not only or c[0] == only]
    for task in TASKS:
        for cond, kind, size in conds:
            for seed in range(6):
                base = f'cwl_{task.replace("-","")}_{cond}_{seed}'
                o = emit_pair(task, kind, size, seed, base)
                for lab in ('safe', 'lava'):
                    m = dict(o[lab], task=task, cond=cond, risk=lab, seed=seed,
                             pair=base)
                    meta.append(m)
                if seed == 0:
                    print(f'{task:<16} {cond:<10} {o["safe"]["cells"]:>5} '
                          f'{o["safe"]["hazards"]:>7} '
                          f'{o["safe"]["states"]}/{o["lava"]["states"]:<14} '
                          f'{o["safe"]["horizon"]:>6} {o["safe"]["c_g"]:>9}')
    (OUT / 'tasks.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'\n写出 {len(meta)} 个任务到 {OUT}')
