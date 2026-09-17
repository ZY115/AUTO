"""Does structure learned on one map predict anything on histories that map never
produced?

This is the offline half of the transfer question, and it is worth answering
before any new algorithm is trained. Freeze a batch of source-map data at three
sizes. From that one batch build two artefacts: the history dictionary, which is
the prefix tree keyed by the exact event history, and the candidate task machine,
which is the same tree quotiented by the merge rule. Ask both to predict what the
task does with each event after each history the *target* map can produce. The
true task machine only marks the answers; it is never shown to either artefact.

The only thing that separates the two artefacts is generalisation: a class knows
a successor when any of its members knows it, so the machine can answer for
histories the dictionary has never seen. If it cannot, or if what it says there is
wrong, compressing the tree bought nothing outside the data.

Three outcomes per question, kept apart on purpose: unknown, correct, wrong. An
artefact that says "unknown" is honest and can be fallen back on; one that is
confidently wrong is what costs a decision.
"""
import json, subprocess, sys, os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile

BUDGETS = {'少量': 2000, '中等': 10000, '较充分': 40000}
MERGE = ['--theta', '20', '--quota', '1', '--strict-children', '0']


def dump_source(tag, seed, budget, directed=False):
    """Collect one batch of source-map data and freeze it.

    `directed` spends part of the same budget deliberately walking to events the
    current history has not tried, which is the project's structural-exploration
    arm. It is the obvious lever on coverage, and coverage turns out to be what
    binds here, so it is measured as a second collection policy rather than
    assumed.
    """
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/overlap/{tag}.task'),
           '--method', 'merged', '--seed', str(seed), '--budget', str(budget),
           '--every', str(budget), '--dump-tree', '1'] + MERGE
    if directed:
        cmd += ['--beta', '.15', '--target', 'scarcest', '--quota', '1']
    return json.loads(subprocess.check_output(cmd, text=True))['tree']


def node_histories(tree):
    """Recover each node's event history by walking the successor edges."""
    nodes, A = tree['nodes'], tree['alphabet']
    hist, order = {0: ()}, [0]
    while order:
        i = order.pop()
        for e in range(A):
            j = nodes[i]['succ'][e]
            if j >= 0 and j not in hist:
                hist[j] = hist[i] + (e,)
                order.append(j)
    return hist


def verdict_of(node, e):
    if node['advanced'][e]:
        return 'advance'
    if node.get('fatal', [0] * len(node['advanced']))[e]:
        return 'fatal'
    if node['ignored'][e]:
        return 'ignore'
    return None


class Dictionary:
    """Exact-history lookup. Answers only where the source data went."""

    def __init__(self, tree):
        self.nodes = tree['nodes']
        self.by_hist = {h: i for i, h in node_histories(tree).items()}

    def predict(self, history, e):
        i = self.by_hist.get(history)
        return None if i is None else verdict_of(self.nodes[i], e)


class Machine:
    """The tree quotiented by a partition, folded forward over the history.

    A class knows what any of its members knows, which is the whole of the
    generalisation on offer: an unseen history is answerable exactly when every
    one of its steps lands in a class whose successor under that event is known.
    """

    def __init__(self, tree, blocks):
        nodes, A = tree['nodes'], tree['alphabet']
        n = len(nodes)
        blocks = list(blocks)[:n] + [max(blocks[:n], default=0) + 1] * max(0, n - len(blocks))
        self.k = max(blocks) + 1
        self.succ = [[-1] * A for _ in range(self.k)]
        self.evid = [[None] * A for _ in range(self.k)]
        for i, nd in enumerate(nodes):
            b = blocks[i]
            for e in range(A):
                j = nd['succ'][e]
                if j >= 0 and self.succ[b][e] < 0:
                    self.succ[b][e] = blocks[j]
                v = verdict_of(nd, e)
                if v is not None and self.evid[b][e] is None:
                    self.evid[b][e] = v
        self.root = blocks[0]

    def fold(self, history):
        b = self.root
        for e in history:
            b = self.succ[b][e]
            if b < 0:
                return None
        return b

    def predict(self, history, e):
        b = self.fold(history)
        return None if b is None else self.evid[b][e]


def one(job):
    pair, size, seed, policy = job
    src, tgt = pair['mapA']['tag'], pair['mapB']['tag']
    tree = dump_source(src, seed, BUDGETS[size], directed=(policy == 'directed'))
    truth = TaskFile(ROOT / f'results/overlap/{tgt}.task')
    blocks = tree['blocks']
    # The oracle partition: the same tree, merged by the true task state. It
    # bounds what any merge rule could have bought from this batch of data.
    oracle_blocks = [nd['truth'] for nd in tree['nodes']]
    arts = {'dictionary': Dictionary(tree),
            'machine': Machine(tree, blocks),
            'oracle_merge': Machine(tree, oracle_blocks)}
    seen = set(Dictionary(tree).by_hist)
    rows = []
    for h in sorted(truth.reachable_histories()):
        q = truth.run(h)
        if truth.accepting[q] or truth.failing[q]:
            continue
        novel = h not in seen
        for e in range(truth.alphabet):
            gold = truth.verdict(q, e)
            for name, art in arts.items():
                p = art.predict(h, e)
                rows.append({'pair': pair['mapA']['tag'], 'band': pair['band'],
                             'overlap': pair['overlap'], 'size': size, 'seed': seed,
                             'policy': policy, 'artefact': name, 'novel': novel,
                             'outcome': 'unknown' if p is None
                                        else ('correct' if p == gold else 'wrong')})
    return rows


if __name__ == '__main__':
    pairs = json.loads((ROOT / 'results/overlap/pairs.json').read_text())
    jobs = [(p, s, seed, pol) for p in pairs for s in BUDGETS
            for seed in range(9000, 9008) for pol in ('plain', 'directed')]
    out = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, rows in enumerate(ex.map(one, jobs, chunksize=2)):
            out.extend(rows)
            if i % 100 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    with open(ROOT / 'results/generalisation.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(out), 'questions')
