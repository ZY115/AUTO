"""Step 39 result: Y11 against Y10 in the authors' own code.

Outcome is the greedy evaluation the authors' own code already logs: every 100
episodes it runs the frozen policy once on each of the ten task instances and
records whether it reached the goal. Two summaries of that curve:

    final   mean success over the last quarter of the run
    ever    fraction of task instances that ever succeeded

Pairing is by (task, task instance, seed): Y11 and Y10 see the same maps and the
same hierarchy, and differ only in whether one goal policy serves every automaton
state. The bootstrap resamples task instances, never seeds.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / 'results/native_grid.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['protocol'] == 'author' and r['ok']]


def per_instance(r):
    out = {}
    for inst, c in r['curves'].items():
        if not c:
            continue
        tail = c[max(0, len(c) * 3 // 4):]
        out[inst] = (mean(x[1] for x in tail), any(x[1] > 0 for x in c))
    return out


D = defaultdict(dict)
for r in rows:
    for inst, v in per_instance(r).items():
        D[(r['task'], r['risk'], r['arm'])][(inst, r['seed'])] = v

TASKS = ['book', 'book-and-quill', 'cake']
print(f'{len(rows)} 次运行（作者协议），每次 10,000 幕、10 个任务实例、3 个种子\n')
print(f'{"任务":<16} {"风险":<6} {"Y11 末段":>9} {"Y10 末段":>9} {"差":>8} '
      f'{"Y11 曾成功":>10} {"Y10 曾成功":>10}')
pairs = defaultdict(list)
for t in TASKS:
    for risk in ('safe', 'lava'):
        a, b = D[(t, risk, 'Y11 shared')], D[(t, risk, 'Y10 no-reuse')]
        keys = sorted(set(a) & set(b))
        if not keys:
            continue
        d = [a[k][0] - b[k][0] for k in keys]
        pairs[(t, risk)] = d
        print(f'{t:<16} {risk:<6} {mean(a[k][0] for k in keys):>9.3f} '
              f'{mean(b[k][0] for k in keys):>9.3f} {mean(d):>+8.3f} '
              f'{mean(a[k][1] for k in keys):>10.3f} {mean(b[k][1] for k in keys):>10.3f}')


def boot(v, n=10000, seed=31):
    if not v:
        return 0, 0, 0
    rng = random.Random(seed)
    b = sorted(mean(v[rng.randrange(len(v))] for _ in v) for _ in range(n))
    return mean(v), b[250], b[9750]


print('\nH1：Y11 − Y10 的末段贪心成功率，按 (任务实例, 种子) 配对自助')
for t in TASKS:
    for risk in ('safe', 'lava'):
        if (t, risk) not in pairs:
            continue
        m, lo, hi = boot(pairs[(t, risk)])
        star = '  *' if lo > 0 else ('  （区间含 0）' if hi > 0 else '  * 反向')
        print(f'  {t:<16} {risk:<6} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}')
allv = [x for v in pairs.values() for x in v]
m, lo, hi = boot(allv)
print(f'  {"合并全部":<23} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]' + ('  *' if lo > 0 else ''))

print('\nH2：岩浆是否放大差距（lava 的差 − safe 的差）')
for t in TASKS:
    if (t, 'lava') in pairs and (t, 'safe') in pairs:
        dl, ds = mean(pairs[(t, 'lava')]), mean(pairs[(t, 'safe')])
        print(f'  {t:<16} lava {dl:+.3f}  safe {ds:+.3f}  差 {dl-ds:+.3f}')
