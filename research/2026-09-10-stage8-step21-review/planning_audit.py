"""Read-only diagnostic: perfect local event navigation versus product optimum.

Both use privileged deterministic map information; these are NOT trained arms.
Greedy recomputes the closest currently allowed event after each primitive step.
This tests planning headroom, not learning speed or learned-policy performance.
"""
import collections
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2] / 'progressive_task_discovery/stage8_compression'
sys.path.insert(0, str(PROJECT / 'src'))
from taskfile import TaskFile


def distances(t):
    rev = [[] for _ in range(t.N)]
    for s in range(t.N):
        for a in range(4):
            rev[t.moves[s][a]].append(s)
    ds = []
    for e in range(t.alphabet):
        d = [10**9] * t.N
        queue = collections.deque()
        for s in range(t.N):
            if e in t.events[s]:
                d[s] = 1
                queue.append(s)
        while queue:
            ns = queue.popleft()
            for s in rev[ns]:
                if d[s] > d[ns] + 1:
                    d[s] = d[ns] + 1
                    queue.append(s)
        ds.append(d)
    return ds


def optimum(t):
    rev = [[] for _ in range(t.N*t.nstates)]
    for q in range(t.nstates):
        if t.accepting[q] or t.failing[q]:
            continue
        for s in range(t.N):
            for a in range(4):
                nq = t.step(q, t.events[s][a])
                if not t.failing[nq]:
                    rev[nq*t.N+t.moves[s][a]].append(q*t.N+s)
    d = [10**9]*len(rev)
    queue = collections.deque()
    for q in range(t.nstates):
        if t.accepting[q]:
            for s in range(t.N):
                d[q*t.N+s] = 0
                queue.append(q*t.N+s)
    while queue:
        ns = queue.popleft()
        for s in rev[ns]:
            if d[s] > d[ns]+1:
                d[s] = d[ns]+1
                queue.append(s)
    return d


def run(path):
    t = TaskFile(path)
    ds, opt = distances(t), optimum(t)
    rows = []
    for start in t.starts:
        s, q, steps = start, 0, 0
        while not t.accepting[q] and not t.failing[q] and steps < t.horizon:
            allowed = [e for e in range(t.alphabet)
                       if t.trans[q][e] != q and not t.failing[t.trans[q][e]]]
            if not allowed:
                break
            g = min(allowed, key=lambda e: (ds[e][s], e))
            a = min(range(4), key=lambda a:
                    (1 if t.events[s][a] == g else 1+ds[g][t.moves[s][a]], a))
            q = t.step(q, t.events[s][a])
            s = t.moves[s][a]
            steps += 1
        rows.append(dict(start=start, optimal=opt[start], greedy=steps,
                         success=bool(t.accepting[q]), ratio=steps/opt[start]))
    return dict(tag=path.stem, starts=rows)


if __name__ == '__main__':
    rows = [run(p) for p in sorted((PROJECT/'results/ceiling').glob('*.task'))]
    flat = [r for m in rows for r in m['starts']]
    summary = dict(maps=len(rows), starts=len(flat),
                   solved=sum(r['success'] for r in flat),
                   suboptimal=sum(r['success'] and r['greedy'] > r['optimal'] for r in flat),
                   aggregate_ratio=sum(r['greedy'] for r in flat)/sum(r['optimal'] for r in flat),
                   max_ratio=max(r['ratio'] for r in flat))
    Path(__file__).with_name('planning_audit.json').write_text(
        json.dumps(dict(summary=summary, maps=rows), indent=2)+'\n')
    print(json.dumps(summary))
