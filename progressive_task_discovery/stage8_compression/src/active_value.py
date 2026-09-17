"""Step 33 layer 2: can the agent *make* a merge safe by going and looking?

Layer 1 priced the walk and found it free: separating events are dense, and the
agent is usually standing on one. So navigation is not what blocks the active
line. This layer asks the harder question, without peeking at the true state.

The agent is at history h2 and wants to reuse an answer it bought at h1. Under
the universal quantifier that is safe only when *every* machine consistent with
its data puts h1 and h2 in the same state. Step 32 found that never holds. So:
which observations would make it hold, and can the agent take them?

Two limits make this much narrower than it first looks.

  position   The agent is at h2. h1 is in the past, and in a task with failing
             states may be gone for good. So it may observe only at h2.
  one visit  Firing an event that *advances* the task moves the agent off h2;
             firing a fatal one ends the episode. So a single visit yields every
             ignore-verdict event, plus at most one non-ignore event.

Both budgets are run: the honest one-visit budget, and an unrealistically
generous "observe everything at h2" upper bound. If the generous budget cannot
close the version space either, no active policy can, and the line closes for a
reason that has nothing to do with cost.
"""
import json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from evidence_value import event_distances, INF

EXE = str(ROOT / 'compress')
# TaskFile.verdict speaks in words; the ASP encoding speaks in 0/1/2. Mixing the
# two injects a fact no model can satisfy, which comes back looking exactly like
# "every consistent machine agrees". It did, for a whole run.
VCODE = {'advance': 0, 'ignore': 1, 'fatal': 2}


def run_log(task, seed=0):
    return json.loads(subprocess.check_output([EXE, '--task', str(task),
        '--method', 'goal', '--goal-select', 'learned', '--cache-key', 'node',
        '--verify-budget', '2000', '--verify-policy', 'arrival', '--seed', str(seed),
        '--budget', '200000', '--every', '200000', '--theta', '20', '--quota', '1',
        '--strict-children', '0', '--dump-queries', '1'], text=True))['queries']


def close_attempt(t, ev, h1, h2, k, generous):
    """Add true observations at h2 until every consistent machine merges h1,h2.

    Returns (closed, acquisitions, exhausted). `generous` lifts the one-visit
    rule, which is the upper bound on what any active policy could buy.
    """
    q2 = t.run(h2)
    ev = list(ev)
    need = [h1, h2]
    taken, spent = set(), 0

    def closed():
        # Unsatisfiable-to-separate only counts if the evidence is satisfiable in
        # the first place; otherwise the "agreement" is vacuous.
        if not vs.consistent(ev, need, t.alphabet, k):
            return None
        return not vs.separable(ev, h1, h2, t.alphabet, k)

    for _ in range(t.alphabet + 1):
        c = closed()
        if c is None:
            return None, spent, True
        if c:
            return True, spent, False
        cands = [w for w in vs.distinguishing_events(ev, h1, h2, t.alphabet, k)
                 if w not in taken]
        if not generous:
            # One visit: ignore-verdict events leave the agent where it is;
            # anything else ends the visit, so it may take only one of those.
            cands = [w for w in cands if t.verdict(q2, w) == 'ignore'] or cands[:1]
        if not cands:
            return False, spent, True
        w = cands[0]
        v = t.verdict(q2, w)
        ev.append((list(h2), w, VCODE[v]))
        if v == 'advance':
            need = need + [tuple(h2) + (w,)]   # the agent really does move there
        taken.add(w); spent += 1
        if not generous and v != 'ignore':
            # The visit is over: the agent has advanced or died.
            c = closed()
            return c, spent, c is not True
    c = closed()
    return c, spent, c is not True


def audit(task, cap=40, seed=0):
    t = TaskFile(task)
    log = run_log(task, seed)
    dist = event_distances(t)
    k = t.nstates                      # the smallest bound that contains the truth
    seen, rows = {}, []
    for i, x in enumerate(log):
        q = t.run(x['h']); key = (q, x['e'])
        if key not in seen:
            seen[key] = i; continue
        j = seen[key]
        h1, h2 = tuple(log[j]['h']), tuple(x['h'])
        if h1 == h2:
            continue                   # same history: not an abstraction failure
        ev = [(y['h'], y['e'], y['v']) for y in log[:i]]
        seps = [w for w in range(t.alphabet)
                if any(t.verdict(s, w) != t.verdict(q, w)
                       for s in range(t.nstates)
                       if not t.failing[s] and not t.accepting[s])]
        nav = min((dist[w][x['c']] for w in seps), default=INF)
        if not vs.consistent(ev, [h1, h2], t.alphabet, k):
            continue                   # the bound cannot hold the agent's own data
        one = close_attempt(t, ev, h1, h2, k, generous=False)
        gen = close_attempt(t, ev, h1, h2, k, generous=True)
        rows.append(dict(i=i, nav=nav,
                         one_closed=one[0], one_spent=one[1],
                         gen_closed=gen[0], gen_spent=gen[1]))
        if len(rows) >= cap:
            break
    return dict(task=Path(task).stem, k=k, alphabet=t.alphabet, rows=rows)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    tasks = [str(ROOT / f"results/ceiling/{f['tag']}.task") for f in fams]
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    out = []
    t0 = time.time()
    for tk in tasks:
        out.append(audit(tk, cap=cap))
        print(f"  {out[-1]['task']:<28} {len(out[-1]['rows']):>3} 对  "
              f"{time.time()-t0:6.1f}s", flush=True)
    (ROOT / 'results/active_value.json').write_text(json.dumps(out, indent=2) + '\n')
    rows = [r for o in out for r in o['rows']]
    n = len(rows)
    print(f'\n抽样 {n} 对"应当合并但版本空间不允许"的历史（{len(out)} 个族）')
    for lab, ck, sk in (('一次访问预算', 'one_closed', 'one_spent'),
                        ('放宽到"在 h2 上想看什么看什么"', 'gen_closed', 'gen_spent')):
        ok = [r for r in rows if r[ck] is True]
        bad = [r for r in rows if r[ck] is None]
        print(f'  {lab:<34} 能把版本空间闭合的: {len(ok)}/{n} = {len(ok)/n:.1%}'
              + (f'，平均花 {sum(r[sk] for r in ok)/len(ok):.1f} 次取证' if ok else '')
              + (f'（另有 {len(bad)} 次因证据本身在 k 下不可满足而作废）' if bad else ''))
