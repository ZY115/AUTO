"""Step 33 layer 1: is the distinguishing evidence worth going to get?

Step 32 showed the redundant queries are not inferable — they are ambiguous for
want of evidence, not for want of inference. What remains is whether that
evidence is worth acquiring by *acting*, since the agent cannot ask about an
arbitrary string: it has to physically reach the event and then pay to learn the
verdict there.

The unit is the **merge**, not the query. Redundancy inside one true task state q
exists because the agent's abstraction has split the histories at q across
several nodes; each node beyond the first re-buys answers the class already
owns. Eliminating that redundancy means merging those nodes, and one merge is
one acquisition, so:

    cost(merge)    = C_nav  +  lambda      steps to reach a separating event,
                                           plus the one query that reads its
                                           verdict there
    benefit(merge) = S * lambda            the redundant queries that node would
                                           no longer have to buy

which breaks even at exactly  S > C_nav / lambda + 1.

This layer is the **oracle physical lower bound**: the true machine is used for
analysis only, to pick the cheapest event that separates anything at all and to
grant the merge for free once that one observation is made. A real agent can do
no better. If even this is not worth buying, the active line closes.
"""
import json, subprocess, sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
EXE = str(ROOT / 'compress')
INF = 10 ** 6


def event_distances(t):
    """For each event, the shortest number of steps from every cell to fire it.

    Zero means the agent is standing on a cell one of whose actions fires it, so
    the evidence is already underfoot.
    """
    back = defaultdict(list)
    for c in range(t.N):
        for a in range(4):
            m = t.moves[c][a]
            if m >= 0:
                back[m].append(c)
    out = []
    for e in range(t.alphabet):
        d = [INF] * t.N
        q = deque()
        for c in range(t.N):
            if any(t.events[c][a] == e for a in range(4)):
                d[c] = 0; q.append(c)
        while q:
            c = q.popleft()
            for pc in back[c]:
                if d[pc] > d[c] + 1:
                    d[pc] = d[c] + 1; q.append(pc)
        out.append(d)
    return out


def separators_for(t, q):
    """Events whose verdict at q differs from at least one other live state.

    An event every state answers the same way carries no information about which
    state we are in, so it is not a separator however cheap it is.
    """
    live = [s for s in range(t.nstates) if not t.failing[s] and not t.accepting[s]]
    out = []
    for e in range(t.alphabet):
        v = t.verdict(q, e)
        if any(t.verdict(s, e) != v for s in live):
            out.append(e)
    return out


def audit(task, seed=0):
    t = TaskFile(task)
    r = json.loads(subprocess.check_output([EXE, '--task', str(task),
        '--method', 'goal', '--goal-select', 'learned', '--cache-key', 'node',
        '--verify-budget', '2000', '--verify-policy', 'arrival', '--seed', str(seed),
        '--budget', '200000', '--every', '200000', '--theta', '20', '--quota', '1',
        '--strict-children', '0', '--dump-queries', '1'], text=True))
    log = r['queries']
    dist = event_distances(t)
    seen = set()
    clusters = defaultdict(list)          # true state -> its redundant queries
    owner = {}                            # true state -> node that bought first
    for x in log:
        q = t.run(x['h'])
        owner.setdefault(q, x['n'])
        key = (q, x['e'])
        if key in seen:
            clusters[q].append(x)
        else:
            seen.add(key)
    merges = []
    for q, qs in clusters.items():
        seps = separators_for(t, q)
        by_node = defaultdict(list)
        for x in qs:
            by_node[x['n']].append(x)
        for n, xs in by_node.items():
            # Where the agent stood when this node first cost the class a
            # duplicate: the moment the merge would have had to be in place.
            cell = xs[0]['c']
            here = xs[0]['e']
            if seps:
                nav = min(dist[e][cell] for e in seps)
                # The same, but forbidding the event the agent is in the middle
                # of firing — i.e. what it costs to go somewhere else on purpose.
                other = [e for e in seps if e != here]
                nav_o = min((dist[e][c2] for e in other for c2 in [cell]), default=INF)
            else:
                nav = nav_o = INF
            merges.append(dict(state=q, node=n, saved=len(xs),
                               nav=nav, nav_elsewhere=nav_o,
                               separators=len(seps), own_node=(n == owner.get(q))))
    return dict(task=Path(task).stem, queries=len(log),
                redundant=sum(len(v) for v in clusters.values()),
                clusters=len(clusters), merges=merges)


def main():
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    tasks = [str(ROOT / f"results/ceiling/{f['tag']}.task") for f in fams]
    out = [audit(tk) for tk in tasks]
    (ROOT / 'results/evidence_value.json').write_text(json.dumps(out, indent=2) + '\n')
    import statistics
    ms = [m for o in out for m in o['merges'] if m['nav'] < INF]
    red = sum(o['redundant'] for o in out)
    print(f'24 个族：冗余查询 {red} 次，{sum(o["clusters"] for o in out)} 个歧义簇，'
          f'{len(ms)} 次待办合并')
    print(f'每次合并平均可省 {statistics.mean(m["saved"] for m in ms):.1f} 次查询，'
          f'中位 {statistics.median(m["saved"] for m in ms):.0f}')
    print(f'\n{"取证步数 C_nav":>14} {"合并数":>7} {"可省查询":>9} {"平均每次省":>11}')
    for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, INF)]:
        g = [m for m in ms if lo <= m['nav'] <= hi]
        if not g: continue
        lab = f'{lo}' if lo == hi else (f'{lo}-{hi}' if hi < INF else f'>{lo-1}')
        print(f'{lab:>14} {len(g):>7} {sum(m["saved"] for m in g):>9} '
              f'{statistics.mean(m["saved"] for m in g):>11.1f}')
    print(f'\nC_nav 中位 {statistics.median(m["nav"] for m in ms):.1f}  '
          f'平均 {statistics.mean(m["nav"] for m in ms):.2f}')
    oth = [m['nav_elsewhere'] for m in ms if m['nav_elsewhere'] < INF]
    print(f'若禁止用"脚下正在触发的那个事件"，C_nav 中位 {statistics.median(oth):.1f}  '
          f'平均 {statistics.mean(oth):.2f}（{len(oth)}/{len(ms)} 次合并仍有别的分隔事件）')
    print('\n盈亏平衡 S > C_nav/lambda + 1：一次查询折合 lambda 个环境步')
    tot = sum(m['saved'] for m in ms)
    print(f'{"lambda":>7} {"划算的合并":>11} {"占比":>7} {"可省查询":>9} {"占冗余":>8} {"净省(查询当量)":>15}')
    for lam in (1, 3, 5, 10, 20, 50, 100):
        good = [m for m in ms if m['saved'] > m['nav'] / lam + 1]
        net = sum(m['saved'] - m['nav'] / lam - 1 for m in good)
        print(f'{lam:>7} {len(good):>11} {len(good)/len(ms):>6.1%} '
              f'{sum(m["saved"] for m in good):>9} {sum(m["saved"] for m in good)/tot:>7.1%} '
              f'{net:>15.0f}')
    return out, ms


if __name__ == '__main__':
    main()
