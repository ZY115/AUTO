"""Export a public OfficeWorld task into this project's .task format.

P3 needs the core comparison rerun somewhere a reader can place it. OfficeWorld
is the domain ISA itself uses, it ships its own ground-truth automaton, and its
dynamics are exportable exactly: `get_observations` depends only on the agent's
cell, and movement is a deterministic function of cell and walls. So the map is a
pure (cell, action) -> (cell, label) table and nothing is approximated.

What is taken from the environment: the grid, the walls, the label of every cell,
and the task automaton. What is computed here: the Myhill-Nerode classes of that
automaton, the product distances used by the privileged shaping arms, and a
balanced set of start cells.

Run with the legacy interpreter, which is the one that has gym:
    .venv-rm/bin/python src/export_officeworld.py
"""
import itertools, json, sys, warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings('ignore')
import gym, gym_subgoal_automata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/officeworld'

TASKS = [
    ('coffee',   'OfficeWorldDeliverCoffee-v0'),
    ('mail',     'OfficeWorldDeliverMail-v0'),
    ('coffeemail','OfficeWorldDeliverCoffeeAndMail-v0'),
    ('patrolab', 'OfficeWorldPatrolAB-v0'),
    ('patrolabc','OfficeWorldPatrolABC-v0'),
]


def build(env_id, seed):
    e = gym.make(env_id, params={"generation": "random", "environment_seed": seed},
                 disable_env_checker=True)
    u = e.unwrapped
    u.reset()
    cells = [(x, y) for x in range(u.width) for y in range(u.height)]
    index = {c: i for i, c in enumerate(cells)}

    # Labels are a pure function of the cell.
    labels = {}
    for c in cells:
        u.agent = c
        obs = u.get_observations()
        labels[c] = sorted(obs)
    multi = [c for c in cells if len(labels[c]) > 1]

    alphabet = sorted({l for v in labels.values() for l in v})
    aid = {l: i for i, l in enumerate(alphabet)}

    # Movement: deterministic, walls respected, blocked moves stay put.
    moves, events = [], []
    for c in cells:
        row_m, row_e = [], []
        for a in range(4):
            x, y = c
            tx, ty = (x, y + 1) if a == 0 else (x, y - 1) if a == 1 else \
                     (x - 1, y) if a == 2 else (x + 1, y)
            u.agent = c
            nxt = (tx, ty) if u._is_valid_movement(c, (tx, ty)) else c
            row_m.append(index[nxt])
            lab = labels[nxt]
            row_e.append(aid[lab[0]] if len(lab) == 1 else -1)
        moves.append(row_m); events.append(row_e)

    aut = u.get_automaton()
    states = sorted(aut.get_states())
    sid = {s: i for i, s in enumerate(states)}
    trans = []
    for s in states:
        row = []
        for l in alphabet:
            nxt = aut.get_next_state(s, {l})
            row.append(sid[nxt] if nxt in sid else sid[s])
        trans.append(row)
    accepting = [1 if aut.is_accept_state(s) else 0 for s in states]
    failing = [1 if aut.is_reject_state(s) else 0 for s in states]
    return dict(u=u, cells=cells, index=index, labels=labels, multi=multi,
                alphabet=alphabet, moves=moves, events=events, states=states,
                trans=trans, accepting=accepting, failing=failing, aut=aut)


def minimise(trans, accepting, failing):
    """Myhill-Nerode classes by partition refinement."""
    n, A = len(trans), len(trans[0])
    cls = [2 if failing[i] else 1 if accepting[i] else 0 for i in range(n)]
    while True:
        sig = {}
        new = []
        for i in range(n):
            k = (cls[i],) + tuple(cls[trans[i][e]] for e in range(A))
            new.append(sig.setdefault(k, len(sig)))
        if new == cls:
            return cls
        cls = new


def product_distance(d):
    """Steps from every (task state, cell) to acceptance, by backward BFS."""
    S, N = len(d['states']), len(d['cells'])
    INF = 10 ** 6
    dist = [[INF] * N for _ in range(S)]
    q = deque()
    for s in range(S):
        if d['accepting'][s]:
            for c in range(N):
                dist[s][c] = 0; q.append((s, c))
    while q:
        s, c = q.popleft()
        for pc in range(N):
            for a in range(4):
                if d['moves'][pc][a] != c:
                    continue
                ev = d['events'][pc][a]
                for ps in range(S):
                    if d['failing'][ps]:
                        continue
                    ns = ps if ev < 0 else d['trans'][ps][ev]
                    if ns != s:
                        continue
                    if dist[ps][pc] > dist[s][c] + 1:
                        dist[ps][pc] = dist[s][c] + 1
                        q.append((ps, pc))
    return dist, INF


def emit(name, d, path, native=False, native_horizon=250):
    """`native` matches the published protocol: the environment's own single
    starting cell and the episode length from ISA's own configuration, instead of
    a balanced start set and a horizon derived from shortest paths."""
    S, N = len(d['states']), len(d['cells'])
    minimal = minimise(d['trans'], d['accepting'], d['failing'])
    dist, INF = product_distance(d)
    reach = [c for c in range(N) if dist[0][c] < INF]
    # A balanced start set: the reachable cells whose optimal routes are most
    # spread out, so no single opening dominates evaluation.
    reach.sort(key=lambda c: dist[0][c])
    if native:
        u = d['u']; u.reset()
        starts = [d['index'][u.agent]]
        lengths = [dist[0][starts[0]] if dist[0][starts[0]] < INF else native_horizon]
        horizon = native_horizon
    else:
        starts = reach[len(reach) // 6::max(1, len(reach) // 8)][:8] or reach[:4]
        lengths = [dist[0][c] for c in starts]
        horizon = 2 * max(lengths)
    L = [f"{len(d['alphabet'])} 0 0 0 0 {len(d['alphabet'])}", str(S),
         ' '.join(map(str, minimal)),
         ' '.join(map(str, d['accepting'])),
         ' '.join(map(str, d['failing']))]
    L += [' '.join(map(str, row)) for row in d['trans']]
    L.append(str(N))
    for c in range(N):
        L.append(' '.join(f"{d['moves'][c][a]} {d['events'][c][a]}" for a in range(4)))
    L += [str(len(starts)), ' '.join(map(str, starts)), ' '.join(map(str, lengths)),
          f"{horizon} {max(lengths)} 1"]
    cap = lambda v: min(v, 999)
    for s in range(S):
        L.append(' '.join(str(cap(dist[s][c])) for c in range(N)))
    for _ in range(2):                       # wrong-task and shuffled tables:
        for s in range(S):                   # unused by the arms P3 runs, but
            L.append(' '.join(str(cap(dist[s][c])) for c in range(N)))  # the format expects them
    path.write_text('\n'.join(L) + '\n')
    return dict(name=name, cells=N, states=S, minimal=len(set(minimal)),
                alphabet=d['alphabet'], starts=len(starts), horizon=horizon,
                lengths=lengths, multi_label_cells=len(d['multi']),
                accepting=sum(d['accepting']), failing=sum(d['failing']))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, env_id in TASKS:
        for seed in range(3):
            d = build(env_id, seed)
            for native in (False, True):
                tag = ('ow_native_' if native else 'ow_') + f'{name}_{seed}'
                m = emit(tag, d, OUT / f'{tag}.task', native=native)
                m['tag'] = tag; m['env'] = env_id; m['seed'] = seed; m['native'] = native
                meta.append(m)
                print(f"  {tag:<24} 任务状态 {m['states']}->{m['minimal']}"
                      f"  起点 {m['starts']}  时域 {m['horizon']}  最短路 {m['lengths'][:3]}")
    (OUT / 'tasks.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'\n写出 {len(meta)} 个任务到 {OUT}')
