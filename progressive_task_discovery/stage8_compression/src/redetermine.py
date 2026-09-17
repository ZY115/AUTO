"""Step 34: redo Step 32's determinacy audit at the Occam bound.

Step 32 replayed the query log and asked, at each query, whether the evidence so
far already fixed the answer. It got zero, and concluded the attribution gap was
not identifiable. That audit ran at k = 18 and k = 28 — near the true state
count, and far above the smallest bound the evidence admits (8 to 12 here).

That matters more than it sounds. Once k reaches the number of observed nodes,
"give every node its own state" is itself a consistent machine, so nothing is
ever determined and nothing is ever forced. The bound, not the data, produced the
zero. Step 32 also saw determinations at k = 3, 4, 6 and dismissed them as
artefacts of too small a bound — correctly, because at those k the evidence is
not satisfiable at all, which the consistency guard now catches.

The right bound is the smallest k the evidence admits, recomputed as evidence
grows. That is the standard Occam prior of automaton learning, and it is the
strongest assumption a learner may make without inventing structure.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from active_value import run_log

LIMIT = 20


def determined_at(ev, h, e, alphabet, k, limit=LIMIT):
    """Which verdicts admit a machine consistent with the evidence, at bound k."""
    ids, base = vs._base(ev, [tuple(h)], alphabet)
    ok = []
    for v in (0, 1, 2):
        r = vs.solve_bounded(vs._ctl(
            base, f':- state({ids[tuple(h)]},K), not out(K,{e},{v}).', k, limit))
        if r is None:
            return None
        if r:
            ok.append(v)
    return ok


def audit_loo(task, limit_queries=40, seed=0):
    """The same question with *all* the evidence: leave one query out and ask.

    The online audit gives the learner almost nothing to work with at query 3,
    where k_min is tiny and the minimum machine over-merges wildly. This removes
    that confound: every other query in the run is available, so what is left is
    the prior itself.
    """
    t = TaskFile(task)
    log = run_log(task, seed)
    full = [(y['h'], y['e'], y['v']) for y in log]
    need = [tuple(x['h']) for x in log]
    k = 1
    while k <= 40 and vs.consistent(full, need, t.alphabet, k, LIMIT) is not True:
        k += 1
    det = amb = wrong = unknown = 0
    for i in range(min(limit_queries, len(log))):
        ev = full[:i] + full[i + 1:]
        ok = determined_at(ev, log[i]['h'], log[i]['e'], t.alphabet, k)
        if ok is None or not ok:
            unknown += 1
        elif len(ok) == 1:
            det += 1
            if ok[0] != log[i]['v']:
                wrong += 1
        else:
            amb += 1
    return dict(task=Path(task).stem, ktrue=t.nstates, kmin_last=k,
                checked=det + amb, determined=det, ambiguous=amb,
                contradicted=wrong, unknown=unknown)


def audit(task, limit_queries=40, seed=0):
    t = TaskFile(task)
    log = run_log(task, seed)
    k = 1
    det = amb = wrong = unknown = 0
    ks = []
    for i in range(min(limit_queries, len(log))):
        ev = [(y['h'], y['e'], y['v']) for y in log[:i]]
        need = [tuple(log[i]['h'])]
        while k <= 40 and vs.consistent(ev, need, t.alphabet, k, LIMIT) is not True:
            k += 1                       # k_min only grows as evidence accumulates
        if k > 40:
            unknown += 1; continue
        ks.append(k)
        ok = determined_at(ev, log[i]['h'], log[i]['e'], t.alphabet, k)
        if ok is None or not ok:
            unknown += 1
        elif len(ok) == 1:
            det += 1
            if ok[0] != log[i]['v']:
                wrong += 1
        else:
            amb += 1
    return dict(task=Path(task).stem, ktrue=t.nstates, kmin_last=(ks[-1] if ks else None),
                checked=det + amb, determined=det, ambiguous=amb,
                contradicted=wrong, unknown=unknown)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    which = sys.argv[1] if len(sys.argv) > 1 else 'n3d2m1'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    mode = sys.argv[3] if len(sys.argv) > 3 else 'online'
    fn = audit_loo if mode == 'loo' else audit
    sel = [f for f in fams if which in f['tag']] or fams[:2]
    out, t0 = [], time.time()
    for f in sel:
        r = fn(str(ROOT / f"results/ceiling/{f['tag']}.task"), n)
        out.append(r)
        print(f"  {r['task']:<28} k_true={r['ktrue']:<3} k_min={r['kmin_last']:<3} "
              f"已确定 {r['determined']:>3} / 歧义 {r['ambiguous']:>3} "
              f"(与真值矛盾 {r['contradicted']}, 未知 {r['unknown']})  "
              f"{time.time()-t0:6.1f}s", flush=True)
    (ROOT / f'results/redetermine_{which}.json').write_text(json.dumps(out, indent=2) + '\n')
    d = sum(r['determined'] for r in out); a = sum(r['ambiguous'] for r in out)
    w = sum(r['contradicted'] for r in out); u = sum(r['unknown'] for r in out)
    print(f'\n合计：已确定 {d} / 可判定 {d+a} = {d/max(1,d+a):.1%}，'
          f'其中与真值矛盾 {w}，未知 {u}')
