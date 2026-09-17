"""Step 33 layer 3: amortisation on the trajectory the agent actually walked.

No counterfactual navigation and no oracle. Replay the real query log and ask
what the simplest rule that uses only the agent's own observations would have
bought: merge two nodes when they agree on every event both have been seen on,
with at least m events in common.

This is deliberately *not* safe. Step 32 established that no safe rule exists on
this data, so the only remaining question is the exchange rate: how many queries
does a risk-accepting rule save, and how many wrong answers does it hand back.
The error that matters is not a wrong merge in the abstract, it is a **wrong
reuse** — a cached verdict served to a node whose true verdict differs — because
that is the only way the mistake reaches a decision.

m sweeps the confidence knob: m=0 merges on no shared evidence at all, which is
the aggressive end; large m merges almost nothing.
"""
import json, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile

EXE = str(ROOT / 'compress')


def run_log(task, seed=0):
    return json.loads(subprocess.check_output([EXE, '--task', str(task),
        '--method', 'goal', '--goal-select', 'learned', '--cache-key', 'node',
        '--verify-budget', '2000', '--verify-policy', 'arrival', '--seed', str(seed),
        '--budget', '200000', '--every', '200000', '--theta', '20', '--quota', '1',
        '--strict-children', '0', '--dump-queries', '1'], text=True))['queries']


def replay(t, log, m):
    parent, obs, truth = {}, defaultdict(dict), {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    paid = saved = wrong = unsafe = 0
    for x in log:
        n, e, v = x['n'], x['e'], x['v']
        if n not in parent:
            parent[n] = n; truth[n] = t.run(x['h'])
        r = find(n)
        if e in obs[r]:
            saved += 1
            if obs[r][e] != v:
                wrong += 1                    # a cached verdict that was untrue here
            continue
        paid += 1
        obs[r][e] = v
        # Merge on agreement over the shared evidence.
        for o in list({find(p) for p in parent}):
            if o == find(n):
                continue
            a, b = obs[find(n)], obs[o]
            common = a.keys() & b.keys()
            if len(common) >= m and all(a[c] == b[c] for c in common):
                ra, rb = find(n), o
                if truth[ra] != truth[rb]:
                    unsafe += 1
                parent[rb] = ra
                for c, val in b.items():
                    a.setdefault(c, val)
                truth[ra] = truth[ra]
    return dict(m=m, paid=paid, saved=saved, wrong=wrong, unsafe=unsafe)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    tasks = [str(ROOT / f"results/ceiling/{f['tag']}.task") for f in fams]
    logs = [(TaskFile(tk), run_log(tk)) for tk in tasks]
    base = sum(len(l) for _, l in logs)
    # The oracle: one entry per (true state, event). This is the 86-style floor.
    floor = sum(len({(t.run(x['h']), x['e']) for x in l}) for t, l in logs)
    print(f'24 个族：基线实付 {base} 次查询，oracle（按真状态缓存）下界 {floor} 次，'
          f'冗余 {base-floor} 次')
    print(f'\n{"m":>3} {"实付":>7} {"省下":>7} {"占冗余":>8} {"错误复用":>9} {"错误率":>8} {"不安全合并":>11}')
    rows = []
    for m in (0, 1, 2, 3, 4, 6):
        agg = defaultdict(int)
        for t, l in logs:
            for k, v in replay(t, l, m).items():
                if k != 'm':
                    agg[k] += v
        rows.append((m, dict(agg)))
        print(f'{m:>3} {agg["paid"]:>7} {agg["saved"]:>7} '
              f'{agg["saved"]/(base-floor):>7.1%} {agg["wrong"]:>9} '
              f'{agg["wrong"]/max(1,agg["saved"]):>7.1%} {agg["unsafe"]:>11}')
    (ROOT / 'results/hindsight_merge.json').write_text(
        json.dumps([dict(m=m, **v) for m, v in rows], indent=2) + '\n')
