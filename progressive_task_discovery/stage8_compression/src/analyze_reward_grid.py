"""Step 40 analysis: the three-arm ordering under both reward protocols.

Two summaries, because they answer different questions and the audit was right
that conflating them caused trouble:

    AUC   area under the success-rate curve. Bigger is better. It is a perfectly
          good outcome, it is just not RMST, and earlier reports called it that.
    RMST  restricted mean survival time, E[min(T, tau)] with T the first step at
          which success reaches 90% and tau the budget. **Smaller is better**,
          and the unit is environment steps. Runs that never attain contribute
          tau, which is what makes the mean well defined under censoring.

Attainment is reported next to both, because a mean over attainers alone selects
a different population in each arm.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TAU = 1_000_000
rows = [json.loads(l) for l in (ROOT / 'results/reward_grid.jsonl').read_text().splitlines()]
meta = {m['tag']: m for m in
        json.loads((ROOT / 'results/craftworld_lava/tasks.json').read_text())}
ARMS = ['QRM', 'Y10 no-reuse', 'Y11 goal readout']


def key(r):
    m = meta[r['tag']]
    return m['cond'], m['risk'], r['protocol'], r['arm']


G = defaultdict(list)
for r in rows:
    G[key(r)].append(r)


def rmst(v):
    return mean(min(x['first90'], TAU) if x['first90'] > 0 else TAU for x in v)


def attain(v):
    return sum(1 for x in v if x['first90'] > 0) / len(v)


print(f'{len(rows)} 次运行。RMST 单位是步，**越小越好**；AUC 越大越好。\n')
print(f'{"导航":<10} {"风险":<6} {"协议":<8} ' +
      ' '.join(f'{a:>22}' for a in ARMS))
for cond in ('cheap', 'expensive'):
    for risk in ('safe', 'lava'):
        for proto in ('orig', 'nostep'):
            cells = []
            for a in ARMS:
                v = G[(cond, risk, proto, a)]
                cells.append(f'{rmst(v)/1000:7.0f}k {attain(v):5.0%} {mean(x["auc"] for x in v):5.2f}')
            print(f'{cond:<10} {risk:<6} {proto:<8} ' + ' '.join(f'{c:>22}' for c in cells))
print('（每格：RMST / 达标率 / AUC）')

print('\n三臂排序（按 RMST，越小越好）在每个格子里是什么')
for cond in ('cheap', 'expensive'):
    for risk in ('safe', 'lava'):
        for proto in ('orig', 'nostep'):
            r = sorted(ARMS, key=lambda a: rmst(G[(cond, risk, proto, a)]))
            print(f'  {cond:<10} {risk:<6} {proto:<8} ' + ' < '.join(x.split()[0] for x in r))


def boot(v, n=10000, seed=17):
    rng = random.Random(seed)
    b = sorted(mean(v[rng.randrange(len(v))] for _ in v) for _ in range(n))
    return mean(v), b[250], b[9750]


print('\nY11 相对 Y10 的 RMST 节省（步，正值表示 Y11 更快），按地图配对自助')
for cond in ('cheap', 'expensive'):
    for risk in ('safe', 'lava'):
        for proto in ('orig', 'nostep'):
            per = defaultdict(lambda: defaultdict(list))
            for r in rows:
                m = meta[r['tag']]
                if (m['cond'], m['risk'], r['protocol']) == (cond, risk, proto):
                    per[r['tag']][r['arm']].append(r)
            d = [rmst(per[t]['Y10 no-reuse']) - rmst(per[t]['Y11 goal readout'])
                 for t in per]
            m_, lo, hi = boot(d)
            star = '  *' if lo > 0 else ''
            print(f'  {cond:<10} {risk:<6} {proto:<8} {m_/1000:+8.1f}k '
                  f'[{lo/1000:+.1f}k, {hi/1000:+.1f}k]{star}')

print('\nQRM 相对 Y10：协议是否改变了谁在下面')
for cond in ('cheap', 'expensive'):
    for risk in ('safe', 'lava'):
        row = []
        for proto in ('orig', 'nostep'):
            q = rmst(G[(cond, risk, proto, 'QRM')])
            y = rmst(G[(cond, risk, proto, 'Y10 no-reuse')])
            row.append(f'{("QRM 更快" if q < y else "Y10 更快"):>8}（差 {abs(q-y)/1000:.0f}k）')
        print(f'  {cond:<10} {risk:<6} 原协议 {row[0]}   零步费 {row[1]}')
