"""The project's merge rule against RPNI and EDSM, on identical evidence.

Gate 1 evidence is the whole reachable task language with every event tried
once, which is the setting where inference is not evidence-limited. Gate 2
grades the evidence down. Both reuse the audit's own node builder and scorer so
nothing about the comparison depends on the learner being compared.
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from family_maps import Task
from inference import rebuild, score
from inference_audit import reachable, build_nodes
from passive_baselines import rpni, edsm

RULES = [
    ('本项目 宽松', lambda n, a: rebuild(n, a, quota=0, strict_children=False)),
    ('本项目 严格', lambda n, a: rebuild(n, a, quota=1, strict_children=True)),
    ('RPNI',       lambda n, a: rpni(n, a)),
    ('EDSM',       lambda n, a: edsm(n, a)),
]


def main(out='results/passive_audit.json'):
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    rows = []
    print('Gate 1  完整可达任务语言，完整证据（精度 / 召回 / 类数，真实类数在最右）')
    head = f"{'台阶':>9} " + ' '.join(f'{n:>20}' for n, _ in RULES) + f" {'真实':>5}"
    print(head)
    for meta in metas:
        task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
        nodes = build_nodes(task, reachable(task))
        cells, rec = [], {}
        for name, rule in RULES:
            p, r, k, t = score(nodes, rule(nodes, task.alphabet))
            cells.append(f'{p:.2f} / {r:.2f} / {k}')
            rec[name] = dict(precision=p, recall=r, classes=k, true_classes=t)
        rows.append(dict(tag=meta['tag'], full=rec))
        print(f"{meta['tag']:>9} " + ' '.join(f'{c:>20}' for c in cells) +
              f" {rec[RULES[0][0]]['true_classes']:>5}")

    print('\nGate 2  证据按比例保留，8 次重复取均值（召回 / 精度）')
    keeps = (1.0, .9, .8, .65, .5)
    print(f"{'规则':>12} " + ' '.join(f'{k:>16}' for k in keeps))
    graded = {}
    for name, rule in RULES:
        line = []
        for keep in keeps:
            ps, rs = [], []
            for meta in metas:
                task = Task(meta['n'], tuple(meta['pair']), meta['d'], meta['m'])
                for rep in range(8):
                    rng = random.Random(hash((meta['tag'], keep, rep)) & 0xffff)
                    nodes = build_nodes(task, reachable(task), keep=keep, rng=rng)
                    p, r, _, _ = score(nodes, rule(nodes, task.alphabet))
                    ps.append(p); rs.append(r)
            line.append(f'{sum(rs)/len(rs):.2f} / {sum(ps)/len(ps):.2f}')
            graded.setdefault(name, {})[keep] = dict(
                recall=sum(rs) / len(rs), precision=sum(ps) / len(ps))
        print(f'{name:>12} ' + ' '.join(f'{c:>16}' for c in line))
    (ROOT / out).write_text(json.dumps(dict(gate1=rows, gate2=graded), indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
