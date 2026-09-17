"""Step 33 layer 2b: the ceiling on what one-step evidence can ever settle.

Layer 2 allowed acquisition only at h2, because h1 is in the past. This lifts
that — it lets the agent observe every event at *both* histories, which no real
policy can do — purely to find out where the wall is.

If the version space still keeps h1 and h2 apart after that, then one-step
evidence is simply not enough to certify the merge, and the remedy is not a
better query policy but a longer distinguishing *sequence*, which needs the
agent to come back. That is a different and much more expensive proposal.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from active_value import run_log, VCODE


def saturate(t, ev, h1, h2, k):
    """Observe every event at both histories, then ask if the merge is forced."""
    ev = list(ev)
    need = [h1, h2]
    for h in (h1, h2):
        q = t.run(list(h))
        for w in range(t.alphabet):
            v = t.verdict(q, w)
            ev.append((list(h), w, VCODE[v]))
            if v == 'advance':
                need.append(tuple(h) + (w,))
    if not vs.consistent(ev, need, t.alphabet, k):
        return None
    return not vs.separable(ev, h1, h2, t.alphabet, k)


def audit(task, cap=6, seed=0):
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
        ev = [(y['h'], y['e'], y['v']) for y in log[:i]]
        if not vs.consistent(ev, [h1, h2], t.alphabet, k):
            continue
        rows.append(saturate(t, ev, h1, h2, k))
        if len(rows) >= cap:
            break
    return dict(task=Path(task).stem, rows=rows)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out, t0 = [], time.time()
    for f in fams:
        out.append(audit(str(ROOT / f"results/ceiling/{f['tag']}.task"), cap))
        print(f"  {out[-1]['task']:<28} {len(out[-1]['rows']):>3} 对  "
              f"{time.time()-t0:6.1f}s", flush=True)
    (ROOT / 'results/active_both.json').write_text(json.dumps(out, indent=2) + '\n')
    rows = [r for o in out for r in o['rows']]
    n = len(rows)
    ok = sum(1 for r in rows if r is True)
    bad = sum(1 for r in rows if r is None)
    print(f'\n把两条历史上的每个事件都看遍（物理上做不到，只作上界）：'
          f'版本空间被迫合并的 {ok}/{n} = {ok/n:.1%}'
          + (f'，另有 {bad} 次证据在 k 下不可满足' if bad else ''))
