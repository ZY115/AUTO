"""The same conversion audit where some events end the task.

The overlap family has no failing states, so it cannot price the one error that
matters most for a commitment: asserting that an event is harmless when it in
fact ends the task. Closed-world completion is exactly the rule that would make
that assertion, since it says "never seen to advance here, so it does nothing".
The irreversible family has three tasks with five maps each, which pairs into
sixty source/target pairs sharing a task machine.
"""
import json, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
from conversion_audit import collect, to_nodes, histories_of, Folded, partitions

BUDGETS = {'中等': 10000, '较充分': 40000}


def pairs_of():
    maps = json.loads((ROOT / 'results/irreversible/maps.json').read_text())
    by = defaultdict(list)
    for e in maps:
        by[e['task']].append(e['fatal']['tag'])
    out = []
    for task, tags in by.items():
        for a, b in permutations(sorted(tags), 2):
            out.append({'task': task, 'src': a, 'tgt': b})
    return out


def one(job):
    pair, size, seed = job
    tree = collect(pair['src'], seed, BUDGETS[size], False, folder='irreversible')
    nodes = to_nodes(tree)
    A = tree['alphabet']
    seen = set(histories_of(nodes).values())
    truth = TaskFile(ROOT / f"results/irreversible/{pair['tgt']}.task")
    folded = {k: Folded(nodes, b, A) for k, b in partitions(nodes, A).items()}
    rows = []
    for h in sorted(truth.reachable_histories()):
        if h in seen:
            continue
        q = truth.run(h)
        if truth.accepting[q] or truth.failing[q]:
            continue
        for e in range(A):
            gold = truth.verdict(q, e)
            for name, f in folded.items():
                for closed in (False, True):
                    kind, pred = f.ask(h, e, closed)
                    out = kind if pred is None else ('correct' if pred == gold else 'wrong')
                    rows.append({'pair': pair['src'] + '>' + pair['tgt'], 'task': pair['task'],
                                 'size': size, 'seed': seed, 'rule': name,
                                 'world': 'closed' if closed else 'open',
                                 'outcome': out, 'gold': gold, 'pred': pred})
    return rows


if __name__ == '__main__':
    ps = pairs_of()
    jobs = [(p, s, seed) for p in ps for s in BUDGETS for seed in range(9200, 9204)]
    out = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, rows in enumerate(ex.map(one, jobs, chunksize=2)):
            out.extend(rows)
            if i % 50 == 0:
                print(i, len(jobs), file=sys.stderr, flush=True)
    with open(ROOT / 'results/conversion_fatal.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print('wrote', len(out), 'from', len(ps), 'pairs')
