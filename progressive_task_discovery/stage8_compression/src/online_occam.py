"""Step 35: would the Occam prior have been safe to use *at the time*?

Step 34's 66.1% is a hindsight number: it used the whole log as evidence and
asked about pairs in the abstract. What decides whether the line is alive is a
narrower question — at the exact moment the baseline paid a redundant query,
with only the evidence it had then, could a minimal-machine learner have shown
the answer was already owned?

So the trajectory is held fixed. The baseline pays for everything, exactly as it
did; this replays its log and asks, at each query, whether a merge with some
earlier history was *forced* at k = k_min(D_t) + delta. A forced merge means the
query need not have been bought.

Two things are counted, and they are not the same:

  verdict correct   the answer that would have been reused matches the true
                    verdict here. This is the error that reaches a decision.
  identity correct  the partner really is the same task state. A reuse can be
                    right by luck while the attribution is wrong, and that luck
                    does not survive the next event.

delta is swept because Step 34 showed the prior's tightness is the whole lever:
one extra state dropped offline coverage from 66.1% to 6.5%.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from active_value import run_log

LIMIT = 6            # seconds per solve; a timeout is never read as an answer
DELTAS = (0, 1, 2)
MAXCAND = 8


def forced_same(ev, h1, h2, alphabet, k, limit=LIMIT):
    """Does every consistent machine put h1 and h2 in one state? None = not known."""
    ids, base = vs._base(ev, [h1, h2], alphabet)
    if vs.solve_bounded(vs._ctl(base, '', k, limit)) is not True:
        return None
    r = vs.solve_bounded(vs._ctl(
        base, f':- state({ids[h1]},K), state({ids[h2]},K).', k, limit))
    return None if r is None else (not r)


def audit(task, seed=0, cap=None):
    t = TaskFile(task)
    log = run_log(task, seed)
    if cap:
        log = log[:cap]
    record = defaultdict(dict)          # history -> event -> verdict the baseline bought
    bought = set()                      # (true state, event) already paid for
    k = 1
    stats = {d: defaultdict(int) for d in DELTAS}
    n_red = 0
    for i, x in enumerate(log):
        h2 = tuple(x['h']); e = x['e']; q2 = t.run(x['h'])
        redundant = (q2, e) in bought
        n_red += redundant
        ev = [(y['h'], y['e'], y['v']) for y in log[:i]]
        need = [h2] + [h for h in record]
        while k <= 40 and vs.consistent(ev, need, t.alphabet, k, LIMIT) is not True:
            k += 1                      # k_min only grows as evidence accumulates
        if k > 40:
            k = 1
        else:
            cands = [h for h in record if e in record[h] and h != h2][-MAXCAND:]
            for d in DELTAS:
                s = stats[d]
                s['queries'] += 1
                s['redundant'] += redundant
                hit = None
                for h1 in cands:
                    f = forced_same(ev, h1, h2, t.alphabet, k + d)
                    if f is None:
                        s['unknown'] += 1
                        continue
                    if f:
                        hit = h1; break
                if hit is None:
                    continue
                s['reuse'] += 1
                s['reuse_redundant'] += redundant
                if record[hit][e] == x['v']:
                    s['verdict_ok'] += 1
                if t.run(list(hit)) == q2:
                    s['identity_ok'] += 1
        record[h2][e] = x['v']
        bought.add((q2, e))
    return dict(task=Path(task).stem, ktrue=t.nstates, kmin=k, n=len(log),
                redundant=n_red, stats={d: dict(stats[d]) for d in DELTAS})


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    which = sys.argv[1] if len(sys.argv) > 1 else 'n3'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sel = [f for f in fams if which in f['tag']] or fams
    out, t0 = [], time.time()
    for f in sel:
        r = audit(str(ROOT / f"results/ceiling/{f['tag']}.task"), cap=cap or None)
        out.append(r)
        s0 = r['stats'][0]
        print(f"  {r['task']:<28} 查询 {r['n']:>3} 冗余 {r['redundant']:>3} "
              f"k_min={r['kmin']:<3} | k_min 复用 {s0.get('reuse',0):>3} "
              f"裁决对 {s0.get('verdict_ok',0):>3} 同态 {s0.get('identity_ok',0):>3} "
              f" {time.time()-t0:6.1f}s", flush=True)
    (ROOT / f'results/online_occam_{which}.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'\n{"prior":>9} {"复用次数":>9} {"裁决正确率":>11} {"归属正确率":>11} '
          f'{"覆盖冗余":>9} {"占冗余":>8} {"求解超时":>9}')
    tot_red = sum(r['redundant'] for r in out)
    for d in DELTAS:
        a = defaultdict(int)
        for r in out:
            for kk, v in r['stats'][d].items():
                a[kk] += v
        ru = a['reuse']
        lab = 'k_min' + ('' if d == 0 else f'+{d}')
        print(f'{lab:>9} {ru:>9} '
              f'{(a["verdict_ok"]/ru if ru else 0):>10.1%} '
              f'{(a["identity_ok"]/ru if ru else 0):>10.1%} '
              f'{a["reuse_redundant"]:>9} '
              f'{(a["reuse_redundant"]/tot_red if tot_red else 0):>7.1%} '
              f'{a["unknown"]:>9}')
