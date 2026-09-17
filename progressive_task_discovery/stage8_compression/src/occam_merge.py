"""Step 34: safe merging under the tightest state bound the data allows.

Step 33 asked whether the version space ever forces two histories together and
got 0 out of 144. That number was an artefact of the bound. The audit used
k = the true state count, which on these tasks (19-57) is far above the number of
observed nodes (5-40). Whenever k is at least the node count, "give every node
its own state" is itself a consistent model, so no merge can *ever* be forced, no
matter how much evidence is added. The question was answered by the encoding
before any data was looked at.

The legitimate bound is the Occam one, and it is what every practical automaton
learner uses: assume the fewest states consistent with what has been seen. Under
that prior a merge can be forced, and the question becomes empirical again:

  coverage   of the pairs that really are the same task state, how many does the
             prior force together — this is what would move 177 toward 86
  safety     of the pairs it forces together, how many are actually different
             states — a forced merge that is wrong is worse than no merge at all

k_min + 1 and k_min + 2 are reported alongside, because the prior's strength is
the whole lever and it decays fast.
"""
import json, random, sys, time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
import version_space as vs
from active_value import run_log


def verdict_pair(ev, h1, h2, alphabet, k, limit=20):
    """What the version space *proves* about two histories, if anything.

    Two different questions, and they are not each other's negation — most pairs
    answer neither.

      same  every consistent machine gives them one state, so sharing a paid
            answer between them is safe. Found by forbidding "different" and
            getting unsatisfiable.
      diff  no consistent machine gives them one state, so they are provably
            distinct. Found by forbidding "same" and getting unsatisfiable.

    Returns 'same', 'diff', 'open', or None when the bound cannot hold the
    evidence or the solver ran out of time. None is never counted as an answer.
    """
    ids, base = vs._base(ev, [h1, h2], alphabet)
    if vs.solve_bounded(vs._ctl(base, '', k, limit)) is not True:
        return None
    # a model in which they differ?
    d = vs.solve_bounded(vs._ctl(
        base, f':- state({ids[h1]},K), state({ids[h2]},K).', k, limit))
    # a model in which they coincide?
    m = vs.solve_bounded(vs._ctl(
        base, f':- state({ids[h1]},K1), state({ids[h2]},K2), K1 != K2.', k, limit))
    if d is None or m is None:
        return None
    if not d:
        return 'same'
    if not m:
        return 'diff'
    return 'open'


def kmin(ev, need, alphabet, hi=40, limit=20):
    for k in range(1, hi + 1):
        if vs.consistent(ev, need, alphabet, k, limit) is True:
            return k
    return None


def audit(task, seed=0, pairs=16):
    t = TaskFile(task)
    log = run_log(task, seed)
    ev = [(y['h'], y['e'], y['v']) for y in log]
    hs = sorted({tuple(x['h']) for x in log}, key=lambda z: (len(z), z))
    k0 = kmin(ev, hs, t.alphabet)
    if k0 is None:
        return None
    allp = [(a, b) for a, b in combinations(hs, 2)]
    same = [p for p in allp if t.run(list(p[0])) == t.run(list(p[1]))]
    diff = [p for p in allp if t.run(list(p[0])) != t.run(list(p[1]))]
    # Sample at random, not in sorted order: the shortest histories are the
    # best-determined ones, so taking the first few flatters the result.
    rng = random.Random(12345)
    sel = (rng.sample(same, min(pairs // 2, len(same)))
           + rng.sample(diff, min(pairs // 2, len(diff))))
    rows = []
    for h1, h2 in sel:
        truth = t.run(list(h1)) == t.run(list(h2))
        r = dict(truth=truth)
        for off in (0, 1, 2):
            r[f'k{off}'] = verdict_pair(ev, h1, h2, t.alphabet, k0 + off)
        rows.append(r)
    return dict(task=Path(task).stem, kmin=k0, nodes=len(hs), ktrue=t.nstates,
                n_same=len(same), n_diff=len(diff), rows=rows)


if __name__ == '__main__':
    fams = json.loads((ROOT / 'results/ceiling/families.json').read_text())
    which = sys.argv[1] if len(sys.argv) > 1 else 'n3'
    sel = [f for f in fams if which in f['tag']] or fams
    out, t0 = [], time.time()
    for f in sel:
        r = audit(str(ROOT / f"results/ceiling/{f['tag']}.task"))
        if r is None:
            continue
        out.append(r)
        print(f"  {r['task']:<28} k_min={r['kmin']:<3} 节点={r['nodes']:<3} "
              f"k_true={r['ktrue']:<3} {len(r['rows']):>3} 对  {time.time()-t0:6.1f}s",
              flush=True)
    (ROOT / f'results/occam_merge_{which}.json').write_text(json.dumps(out, indent=2) + '\n')
    rows = [r for o in out for r in o['rows']]
    same = [r for r in rows if r['truth']]
    diff = [r for r in rows if not r['truth']]
    print(f'\n{len(rows)} 对历史（同真状态 {len(same)}，不同真状态 {len(diff)}）')
    print(f'{"状态上界":>9} | {"判定为同态":>10} {"其中正确":>9} {"精确率":>8} {"覆盖同态对":>11} '
          f'| {"判定为异态":>10} {"其中正确":>9} {"精确率":>8} | {"未定":>6} {"未知":>6}')
    for off in (0, 1, 2):
        key = f'k{off}'
        ps = [r for r in rows if r[key] == 'same']
        pd = [r for r in rows if r[key] == 'diff']
        lab = 'k_min' + ('' if off == 0 else f'+{off}')
        print(f'{lab:>9} | {len(ps):>10} {sum(1 for r in ps if r["truth"]):>9} '
              f'{(sum(1 for r in ps if r["truth"])/len(ps) if ps else 0):>7.1%} '
              f'{(sum(1 for r in ps if r["truth"])/len(same) if same else 0):>10.1%} '
              f'| {len(pd):>10} {sum(1 for r in pd if not r["truth"]):>9} '
              f'{(sum(1 for r in pd if not r["truth"])/len(pd) if pd else 0):>7.1%} '
              f'| {sum(1 for r in rows if r[key] == "open"):>6} '
              f'{sum(1 for r in rows if r[key] is None):>6}')
