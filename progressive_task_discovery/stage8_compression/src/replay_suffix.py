"""Step 34: the one active strategy that needs no reset — replay a shared suffix.

Step 33 allowed acquisition only *at* h2, because h1 is in the past. That was too
narrow. The agent cannot go back to h1, but it can steer its own continuation
from h2 so as to run a suffix it already happened to run from h1. Nothing is
reset and nothing is revisited; it is ordinary forward play, chosen on purpose.

If that closes the version space, Step 33's conclusion is wrong and the active
line is alive. So it has to be tested before the proposition is written.

Two budgets:

  available  only suffixes the agent actually ran from h1 somewhere in the log,
             which is all a real agent could match
  oracle     every suffix up to length L, added until closure, which no policy
             can beat and which therefore bounds the whole idea
"""
import json, sys, time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from active_value import run_log, VCODE


def obs_along(t, h, w):
    """True single-step verdicts at every history along running suffix w from h.

    The agent walks forward: at each point it sees what every event does there,
    which is the same free-label plus paid-verdict it gets anywhere else.
    """
    out, cur = [], list(h)
    for step in range(len(w) + 1):
        q = t.run(cur)
        for e in range(t.alphabet):
            out.append((list(cur), e, VCODE[t.verdict(q, e)]))
        if step == len(w):
            break
        e = w[step]
        if t.verdict(q, e) != 'advance':
            break                     # the suffix does not actually go anywhere
        cur = cur + [e]
    return out


def forced(t, ev, h1, h2, k):
    """Does every consistent machine now put h1 and h2 in the same state?"""
    ids, base = vs._base(ev, [h1, h2], t.alphabet)
    if not vs._ctl(base, '', k).solve().satisfiable:
        return None
    return not vs._ctl(base, f':- state({ids[h1]},K1), state({ids[h2]},K2), K1 != K2.',
                       k).solve().satisfiable


def available_suffixes(log, h1, cap=6):
    """Suffixes the agent really did run from h1 at some point in the log."""
    out, seen = [], set()
    for x in log:
        h = tuple(x['h'])
        if len(h) > len(h1) and h[:len(h1)] == tuple(h1):
            w = h[len(h1):]
            if w not in seen:
                seen.add(w); out.append(w)
    out.sort(key=len)
    return out[:cap]


def audit(task, cap=6, seed=0, L=2, budget=8):
    t = TaskFile(task)
    log = run_log(task, seed)
    k = t.nstates
    seen, rows = {}, []
    for i, x in enumerate(log):
        q = t.run(x['h']); key = (q, x['e'])
        if key not in seen:
            seen[key] = i; continue
        h1, h2 = tuple(log[seen[key]]['h']), tuple(x['h'])
        if h1 == h2:
            continue
        ev0 = [(y['h'], y['e'], y['v']) for y in log[:i]]
        if not vs.consistent(ev0, [h1, h2], t.alphabet, k):
            continue

        # (a) replay what the agent itself already ran from h1
        ev = list(ev0)
        for w in available_suffixes(log, h1):
            ev += obs_along(t, h2, w)
        avail = forced(t, ev, h1, h2, k)

        # (b) oracle: keep adding suffixes until it closes or the budget runs out
        ev = list(ev0)
        spent, orc = 0, False
        for w in sorted(product(range(t.alphabet), repeat=L), key=lambda z: z):
            ev += obs_along(t, h1, w) + obs_along(t, h2, w)
            spent += 1
            f = forced(t, ev, h1, h2, k)
            if f is True:
                orc = True; break
            if spent >= budget:
                break
        rows.append(dict(available=avail, oracle=orc, spent=spent))
        if len(rows) >= cap:
            break
    return dict(task=Path(task).stem, rows=rows)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    which = sys.argv[1] if len(sys.argv) > 1 else 'n3'
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    sel = [f for f in fams if which in f['tag']] or fams
    out, t0 = [], time.time()
    for f in sel:
        out.append(audit(str(ROOT / f"results/ceiling/{f['tag']}.task"), cap))
        r = out[-1]['rows']
        print(f"  {out[-1]['task']:<28} {len(r):>2} 对  "
              f"可得重放闭合 {sum(1 for z in r if z['available'] is True)}  "
              f"oracle 闭合 {sum(1 for z in r if z['oracle'])}  {time.time()-t0:6.1f}s",
              flush=True)
    (ROOT / f'results/replay_suffix_{which}.json').write_text(json.dumps(out, indent=2) + '\n')
    rows = [r for o in out for r in o['rows']]
    n = len(rows)
    print(f'\n{n} 对：')
    print(f'  重放智能体自己跑过的后缀     闭合 {sum(1 for r in rows if r["available"] is True)}/{n}')
    print(f'  oracle 任选长度 2 的后缀      闭合 {sum(1 for r in rows if r["oracle"])}/{n}'
          f'，平均试了 {sum(r["spent"] for r in rows)/max(1,n):.1f} 个')
