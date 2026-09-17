"""How much of what a frozen batch supports does the inference rule actually deliver?

Step 18 measured that the merge rule converts 0.28 to 0.50 of its own ceiling, and
that lifting the ceiling by collecting better made conversion *worse*. Conversion
is therefore worth attacking directly, and it can be attacked offline: collect one
batch per condition with merging switched off, then apply every inference rule to
that identical batch. Nothing about the data depends on which rule is being
scored, which the online experiments could never claim.

Two things are varied, because they are different kinds of claim:

  the partition        which histories the rule believes are the same state
  the verdict policy   what it is willing to say about an event it never saw there

Open-world answers only where some member of the class observed that event.
Closed-world completes the hypothesis: an event never seen to advance anywhere in
the class is asserted not to advance. That is exactly what a task automaton
claims, and it is a strictly stronger commitment than the evidence, so it is
scored separately rather than folded in.

"Unknown" is split into its two causes, because they call for different fixes:
the history cannot be folded into any class at all (unreachable), or it can but
the class has nothing to say about that event (no evidence).
"""
import json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
from inference import Node, rebuild
from passive_baselines import rpni, edsm

BUDGETS = {'中等': 10000, '较充分': 40000}


def collect(tag, seed, budget, directed, folder='overlap'):
    """One frozen batch. Merging is blocked during collection with an
    unreachable quota, so the behaviour that produced the data is the same for
    every inference rule scored against it."""
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/{folder}/{tag}.task'),
           '--method', 'merged', '--seed', str(seed), '--budget', str(budget),
           '--every', str(budget), '--dump-tree', '1',
           '--theta', '20', '--quota', '10000000', '--strict-children', '0']
    if directed:
        cmd += ['--beta', '.15', '--target', 'scarcest']
    return json.loads(subprocess.check_output(cmd, text=True))['tree']


def to_nodes(tree):
    A = tree['alphabet']
    out = []
    for raw in tree['nodes']:
        nd = Node(A, raw['visits'], raw['truth'])
        nd.advanced = list(raw['advanced'])
        nd.ignored = list(raw['ignored'])
        nd.fatal = list(raw.get('fatal', [0] * A))
        nd.succ = list(raw['succ'])
        out.append(nd)
    return out


def histories_of(nodes):
    hist, order = {0: ()}, [0]
    while order:
        i = order.pop()
        for e, j in enumerate(nodes[i].succ):
            if j >= 0 and j not in hist:
                hist[j] = hist[i] + (e,)
                order.append(j)
    return hist


class Folded:
    """A partition of the tree, folded forward over a history.

    A class knows a successor when any member knows it, and knows a verdict when
    any member observed it. Those two are what generalisation consists of here.
    """

    def __init__(self, nodes, blocks, alphabet):
        self.A = alphabet
        k = max(blocks) + 1
        self.succ = [[-1] * alphabet for _ in range(k)]
        self.adv = [set() for _ in range(k)]
        self.ign = [set() for _ in range(k)]
        self.fat = [set() for _ in range(k)]
        for i, nd in enumerate(nodes):
            b = blocks[i]
            for e in range(alphabet):
                if nd.succ[e] >= 0 and self.succ[b][e] < 0:
                    self.succ[b][e] = blocks[nd.succ[e]]
                if nd.advanced[e]:
                    self.adv[b].add(e)
                if nd.ignored[e]:
                    self.ign[b].add(e)
                if nd.fatal[e]:
                    self.fat[b].add(e)
        self.root = blocks[0]

    def fold(self, history):
        b = self.root
        for e in history:
            b = self.succ[b][e]
            if b < 0:
                return None
        return b

    def ask(self, history, e, closed):
        b = self.fold(history)
        if b is None:
            return 'unreachable', None
        if e in self.adv[b]:
            return 'answer', 'advance'
        if e in self.fat[b]:
            return 'answer', 'fatal'
        if e in self.ign[b]:
            return 'answer', 'ignore'
        if closed:
            # The hypothesis asserts what it has not seen: an event never seen to
            # advance anywhere in this class does not advance here.
            return 'answer', 'ignore'
        return 'no_evidence', None


def partitions(nodes, A):
    ident = list(range(len(nodes)))
    return {
        'none':      ident,
        'sweep_t20': rebuild(nodes, A, theta=20, quota=1, strict_children=False),
        'sweep_t5':  rebuild(nodes, A, theta=5, quota=1, strict_children=False),
        'sweep_t1':  rebuild(nodes, A, theta=1, quota=1, strict_children=False),
        'sweep_t1q0': rebuild(nodes, A, theta=1, quota=0, strict_children=False),
        'rpni':      rpni(nodes, A),
        'edsm':      edsm(nodes, A),
        'oracle':    [nd.truth for nd in nodes],
    }


def one(job):
    pair, size, seed, policy = job
    src, tgt = pair['mapA']['tag'], pair['mapB']['tag']
    tree = collect(src, seed, BUDGETS[size], policy == 'directed')
    nodes = to_nodes(tree)
    A = tree['alphabet']
    seen = set(histories_of(nodes).values())
    truth = TaskFile(ROOT / f'results/overlap/{tgt}.task')
    folded = {k: Folded(nodes, b, A) for k, b in partitions(nodes, A).items()}
    rows = []
    for h in sorted(truth.reachable_histories()):
        if h in seen:
            continue                      # the novel set is the whole question
        q = truth.run(h)
        if truth.accepting[q] or truth.failing[q]:
            continue
        for e in range(A):
            gold = truth.verdict(q, e)
            for name, f in folded.items():
                for closed in (False, True):
                    kind, pred = f.ask(h, e, closed)
                    out = kind if pred is None else ('correct' if pred == gold else 'wrong')
                    rows.append({'pair': src, 'overlap': pair['overlap'], 'size': size,
                                 'seed': seed, 'collect': policy, 'rule': name,
                                 'world': 'closed' if closed else 'open',
                                 'outcome': out, 'gold': gold, 'pred': pred})
    return rows


if __name__ == '__main__':
    pairs = json.loads((ROOT / 'results/overlap/pairs.json').read_text())
    jobs = [(p, s, seed, pol) for p in pairs for s in BUDGETS
            for seed in range(9100, 9104) for pol in ('plain', 'directed')]
    out = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, rows in enumerate(ex.map(one, jobs, chunksize=2)):
            out.extend(rows)
            if i % 50 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    with open(ROOT / 'results/conversion.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(out))
