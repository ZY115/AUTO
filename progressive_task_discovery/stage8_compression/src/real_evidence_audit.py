"""Same rules, same evidence, no policy: this time the evidence is real.

The graded audit thins observations uniformly at random. Real missing evidence
is not random: an agent doing the task never revisits a label it has already
collected, so whole columns are absent in a structured way. This runs the four
rules over evidence dumped from actual training, which is the only version of
the question that decides anything about the online setting.
"""
import collections, json, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from inference import from_dump, rebuild, score
from test_inference import legacy_rebuild

RULES = [('旧规则', lambda nd, a: legacy_rebuild(nd, a)),
         ('新，无配额', lambda nd, a: rebuild(nd, a, quota=0, strict_children=False)),
         ('新，配额1', lambda nd, a: rebuild(nd, a, quota=1, strict_children=False)),
         ('新，配额2', lambda nd, a: rebuild(nd, a, quota=2, strict_children=False)),
         ('新，配额1+严格子', lambda nd, a: rebuild(nd, a, quota=1, strict_children=True))]


def dump(tag, seed, budget, beta, theta):
    cmd = [str(ROOT / 'compress'), '--task', str(ROOT / f'results/family_maps/{tag}.task'),
           '--method', 'merged', '--seed', str(seed), '--budget', str(budget),
           '--every', str(budget), '--theta', str(theta), '--quota', '10000000',
           '--beta', str(beta), '--target', 'scarcest', '--dump-tree', '1']
    return json.loads(subprocess.check_output(cmd, text=True))['tree']


def main(out='results/real_evidence_audit.json', budget=200000, theta=50):
    budget, theta = int(budget), int(theta)
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    rows = []
    for beta in (0.0, 0.15):
        agg = collections.defaultdict(list)
        cover = []
        for meta in metas:
            for seed in range(8):
                d = dump(meta['tag'], seed, budget, beta, theta)
                nodes, alphabet = from_dump(d)
                tried = sum(1 for nd in nodes for e in range(alphabet)
                            if nd.advanced[e] or nd.ignored[e])
                cover.append(tried / max(1, len(nodes) * alphabet))
                for label, rule in RULES:
                    p, r, k, t = score(nodes, rule(nodes, alphabet))
                    agg[label].append((p, r, k / t))
        print(f'\n绕路 beta={beta}，真实训练证据，覆盖率 {statistics.mean(cover):.2f}，'
              f'{len(cover)} 次运行，每次 {budget} 步')
        print(f"{'规则':>16} {'精度':>6} {'召回':>6} {'类数/真实':>9}")
        for label, _ in RULES:
            v = agg[label]
            print(f'{label:>16} {statistics.mean(x[0] for x in v):>6.2f} '
                  f'{statistics.mean(x[1] for x in v):>6.2f} {statistics.mean(x[2] for x in v):>9.2f}')
            rows.append(dict(beta=beta, rule=label, coverage=statistics.mean(cover),
                             precision=statistics.mean(x[0] for x in v),
                             recall=statistics.mean(x[1] for x in v),
                             ratio=statistics.mean(x[2] for x in v)))
    (ROOT / out).write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
