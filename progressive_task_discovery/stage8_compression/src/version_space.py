"""How much of the redundant-query gap is identifiable from the data the agent had?

P3 asked the wrong quantifier. Sharing a paid answer is safe only when **every**
task machine still consistent with the observed data gives the same answer, not
when merely *some* machine allows the two histories to coincide.

This measures the difference. It replays a real run's query log: at query i, the
evidence is queries 0..i-1, and the question is whether that evidence already
determined the answer. Determined means exactly one verdict admits a consistent
machine; ambiguous means two or more do, and then no algorithm can share safely
without either more observation, a structural assumption, or accepting risk.

Encoding, in ASP: assign every observed history to one of K latent states, with
the transition function and the output function both functional, and every paid
observation respected. A progress verdict moves to the extended history's state;
an ignore verdict is a self-loop; a fatal verdict ends the episode and constrains
nothing further.

K is an assumption, not data, so it is swept: the number of states the agent is
willing to posit is exactly what turns "possible" into "determined".
"""
import json, sys
from pathlib import Path

import clingo

PROG = """
1 { state(N,1..k) } 1 :- node(N).
:- node(0), not state(0,1).                                  % symmetry: root is state 1
1 { out(K,E,0..2) } 1 :- K = 1..k, event(E).
:- state(N,K), obs(N,E,V), not out(K,E,V).                   % observations must hold
trans(K1,E,K2) :- state(N,K1), succ(N,E,M), state(M,K2).     % progress moves
:- trans(K,E,A), trans(K,E,B), A < B.                        % ...deterministically
:- state(N,K), obs(N,E,1), trans(K,E,K2), K2 != K.           % ignore is a self-loop
:- state(N,K), obs(N,E,0), trans(K,E,K).                     % progress changes the state
"""


def build(evidence, nodes_needed):
    """Facts for a set of observations. `evidence` is [(history, event, verdict)]."""
    hs = set()
    for h, e, v in evidence:
        for k in range(len(h) + 1):
            hs.add(tuple(h[:k]))
    for h in nodes_needed:
        for k in range(len(h) + 1):
            hs.add(tuple(h[:k]))
    ids = {h: i for i, h in enumerate(sorted(hs, key=lambda x: (len(x), x)))}
    f = [f'node({i}).' for i in ids.values()]
    for h, i in ids.items():
        if h:
            f.append(f'succ({ids[h[:-1]]},{h[-1]},{i}).')
    for h, e, v in evidence:
        f.append(f'obs({ids[tuple(h)]},{e},{v}).')
    return ids, f


def determined(evidence, history, event, alphabet, k):
    """Which verdicts admit a machine consistent with the evidence."""
    ids, facts = build(evidence, [tuple(history)])
    base = '\n'.join(facts) + '\n' + ''.join(f'event({e}).' for e in range(alphabet)) + '\n'
    ok = []
    for v in (0, 1, 2):
        ctl = clingo.Control(['--warn=none'])
        ctl.configuration.solve.models = 1
        ctl.add('base', [], base + PROG +
                f'\n:- state({ids[tuple(history)]},K), not out(K,{event},{v}).\n'
                f'#const k={k}.\n')
        ctl.ground([('base', [])])
        if ctl.solve().satisfiable:
            ok.append(v)
        if len(ok) > 1:
            break              # already ambiguous, no need to test the third
    return ok


def audit_full(log, alphabet, k):
    """Every query in the log, reporting where in the log determinacy appears."""
    det = amb = wrong = unsat = 0
    first_det = None
    for i in range(len(log)):
        h, e, v = log[i]['h'], log[i]['e'], log[i]['v']
        ev = [(x['h'], x['e'], x['v']) for x in log[:i]]
        ok = determined(ev, h, e, alphabet, k)
        if not ok:
            unsat += 1                     # the state bound excludes the data itself
        elif len(ok) == 1:
            det += 1
            if first_det is None:
                first_det = i
            if ok[0] != v:
                wrong += 1
        else:
            amb += 1
    return dict(n=len(log), determined=det, ambiguous=amb, contradicted=wrong,
                unsat=unsat, first_determined=first_det)


def audit(log, alphabet, k, limit=None):
    n = len(log) if limit is None else min(limit, len(log))
    det = amb = wrong = 0
    for i in range(n):
        h, e, v = log[i]['h'], log[i]['e'], log[i]['v']
        ev = [(x['h'], x['e'], x['v']) for x in log[:i]]
        ok = determined(ev, h, e, alphabet, k)
        if len(ok) == 1:
            det += 1
            if ok[0] != v:
                wrong += 1
        else:
            amb += 1
    return dict(checked=n, determined=det, ambiguous=amb, contradicted=wrong)


if __name__ == '__main__':
    import subprocess
    task = sys.argv[1] if len(sys.argv) > 1 else 'results/ceiling/c_n3d2m1_4L2a7.task'
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    ROOT = Path(__file__).resolve().parents[1]
    r = json.loads(subprocess.check_output([str(ROOT / 'compress'), '--task', task,
        '--method', 'goal', '--goal-select', 'learned', '--cache-key', 'node',
        '--verify-budget', '2000', '--verify-policy', 'arrival', '--seed', '0',
        '--budget', '200000', '--every', '200000', '--theta', '20', '--quota', '1',
        '--strict-children', '0', '--dump-queries', '1'], text=True))
    A = max(x['e'] for x in r['queries']) + 1
    print(f"{task.split('/')[-1]}  查询 {len(r['queries'])} 条，事件表 {A}，假设状态数 k={k}，核查前 {limit} 条")
    print(' ', audit(r['queries'], A, k, limit))


# ---------------------------------------------------------------------------
# Step 33 layer 2: what it takes to *make* a merge safe, not just to notice one.
#
# Sharing an answer between two histories is safe only when every consistent
# machine puts them in the same state. Step 32 found that essentially never
# holds on the agent's own data. These ask the follow-up: which observations
# would make it hold, and can the agent even take them?
#
# The binding constraint is position. The agent is standing at the *later*
# history; the earlier one is in the past and, in a task with failing states,
# may be unreachable for good. So acquisition is allowed only at the node the
# agent currently occupies.
# ---------------------------------------------------------------------------

def _ctl(base, extra, k, limit=None):
    """A grounded control. `limit` caps the solve in seconds.

    Proving *unsatisfiability* at a tight state bound is the minimum-consistent-DFA
    problem, which is NP-hard, so an unbounded solve can hang. With a limit the
    result may come back neither satisfiable nor unsatisfiable, and callers must
    treat that as "not known", never as either answer.
    """
    c = clingo.Control(['--warn=none'])
    c.configuration.solve.models = 1
    c.add('base', [], base + PROG + extra + f'\n#const k={k}.\n')
    c.ground([('base', [])])
    c._deadline = limit          # read by solve_bounded
    return c


def _base(evidence, need, alphabet):
    ids, facts = build(evidence, need)
    return ids, ('\n'.join(facts) + '\n'
                 + ''.join(f'event({e}).' for e in range(alphabet)) + '\n')


def separable(evidence, h1, h2, alphabet, k):
    """True if some consistent machine still keeps h1 and h2 apart.

    False is the goal state: it means *every* consistent machine merges them, so
    reusing an answer across them is safe under the universal quantifier.
    """
    h1, h2 = tuple(h1), tuple(h2)
    ids, base = _base(evidence, [h1, h2], alphabet)
    c = _ctl(base, f':- state({ids[h1]},K), state({ids[h2]},K).', k)
    return c.solve().satisfiable


def distinguishing_events(evidence, h1, h2, alphabet, k):
    """Events some consistent machine answers differently at h1 and at h2.

    These are the only observations worth taking: an event every consistent
    machine already answers identically cannot shrink the version space.
    """
    h1, h2 = tuple(h1), tuple(h2)
    ids, base = _base(evidence, [h1, h2], alphabet)
    out = []
    for w in range(alphabet):
        c = _ctl(base, f':- state({ids[h1]},K1), state({ids[h2]},K2), '
                       f'out(K1,{w},V), out(K2,{w},V).', k)
        if c.solve().satisfiable:
            out.append(w)
    return out


def solve_bounded(ctl):
    """Solve, giving up after ctl._deadline seconds.

    Returns True (satisfiable), False (unsatisfiable) or None (out of time).
    libclingo takes no --time-limit, so the cap has to come from the async handle.
    """
    limit = getattr(ctl, '_deadline', None)
    if limit is None:
        r = ctl.solve()
        return True if r.satisfiable else (False if r.unsatisfiable else None)
    with ctl.solve(async_=True) as h:
        done = h.wait(limit)
        if not done:
            h.cancel()
            return None
        r = h.get()
        return True if r.satisfiable else (False if r.unsatisfiable else None)


def consistent(evidence, need, alphabet, k, limit=None):
    """Is the evidence satisfiable at all under a k-state bound?

    Every question in this module is asked as *unsatisfiability*, so a bound too
    small — or a malformed fact — makes the answer come back as a confident
    "every machine agrees". This guard is what tells the two apart, and it is not
    optional: Step 32 and Step 33 both produced wrong results without it.
    """
    ids, base = _base(evidence, need, alphabet)
    return solve_bounded(_ctl(base, '', k, limit))
