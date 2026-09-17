"""RPNI and EDSM on the evidence a real agent actually collected.

The graded audit thins evidence at random, which is not how it goes missing: an
agent doing the task does not revisit a label it has already collected, so whole
columns are absent in a structured way. This runs the two standard passive
learners and the project's own rule over evidence dumped from real training, at
the same operating points the project reports.
"""
import collections, json, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import from_dump, rebuild, score
from passive_baselines import rpni, edsm

RULES = [('本项目 宽松', lambda nd, a: rebuild(nd, a, quota=1, strict_children=False)),
         ('本项目 严格', lambda nd, a: rebuild(nd, a, quota=1, strict_children=True)),
         ('RPNI',       lambda nd, a: rpni(nd, a)),
         ('EDSM',       lambda nd, a: edsm(nd, a))]


def dump(tag, seed, budget, beta, theta):
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/family_maps/{tag}.task'),
           '--method', 'merged', '--seed', str(seed), '--budget', str(budget),
           '--every', str(budget), '--theta', str(theta), '--quota', '10000000',
           '--beta', str(beta), '--target', 'scarcest', '--dump-tree', '1']
    return json.loads(subprocess.check_output(cmd, text=True))['tree']


def main(out='results/passive_real_audit.json', budget=200000, theta=50, seeds=8):
    budget, theta, seeds = int(budget), int(theta), int(seeds)
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    result = {}
    for beta in (0.0, 0.15):
        agg = collections.defaultdict(list)
        for meta in metas:
            for seed in range(seeds):
                nodes, alphabet = from_dump(dump(meta['tag'], seed, budget, beta, theta))
                for name, rule in RULES:
                    p, r, k, t = score(nodes, rule(nodes, alphabet))
                    agg[name].append((p, r, k, t))
        print(f'\nbeta={beta}  真实训练证据，{len(metas)} 个台阶 × {seeds} 个种子')
        print(f"{'规则':>14} {'精度':>7} {'召回':>7} {'类数':>7} {'真实类数':>8} {'最低精度':>8}")
        for name, _ in RULES:
            v = agg[name]
            print(f'{name:>14} {statistics.mean(x[0] for x in v):>7.3f} '
                  f'{statistics.mean(x[1] for x in v):>7.3f} '
                  f'{statistics.mean(x[2] for x in v):>7.1f} '
                  f'{statistics.mean(x[3] for x in v):>8.1f} '
                  f'{min(x[0] for x in v):>8.3f}')
            result.setdefault(str(beta), {})[name] = dict(
                precision=statistics.mean(x[0] for x in v),
                recall=statistics.mean(x[1] for x in v),
                classes=statistics.mean(x[2] for x in v),
                true_classes=statistics.mean(x[3] for x in v),
                worst_precision=min(x[0] for x in v))
    (ROOT / out).write_text(json.dumps(result, indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
