import json, random, statistics
from collections import defaultdict
rows=[json.loads(l) for l in open('results/crossmap.jsonl')]
ARMS=['scratch','dict','structure','aggressive','selective','oracle']
pairs=sorted({r['pair'] for r in rows})
per=defaultdict(lambda: defaultdict(list))
for r in rows:
    per[r['arm']][r['pair']].append(r)
print('25 对地图 × 4 源种子 × 4 目标种子，目标预算 100,000')
print(f"{'臂':>11} {'目标达标步数':>11} {'AUC':>6} {'解出':>10} {'首次成功':>8} {'失败次数':>8} {'总成本':>9}")
for a in ARMS:
    g=[r for r in rows if r['arm']==a]
    print(f"{a:>11} {statistics.mean(r['first90'] for r in g):>11.0f} "
          f"{statistics.mean(r['auc'] for r in g):>6.3f} "
          f"{sum(1 for r in g if r['final']>=.9):>5}/{len(g):<4} "
          f"{statistics.mean(r['first_success'] for r in g if r['first_success']>0):>8.0f} "
          f"{statistics.mean(r['failures'] for r in g):>8.1f} "
          f"{statistics.mean(r['total_cost'] for r in g):>9.0f}")
rnd=random.Random(211)
def mean(a,sample,key='first90'):
    return statistics.mean(r[key] for p in sample for r in per[a][p])
def boot(a,b,key='first90',n=4000):
    pt=mean(b,pairs,key)/mean(a,pairs,key)
    d=sorted(mean(b,s,key)/mean(a,s,key) for s in
             ([rnd.choice(pairs) for _ in pairs] for _ in range(n)))
    return pt,d[int(.025*n)],d[int(.975*n)]
print('\n配对自助法（块=地图对，25 块），左比右，>1 表示左更快')
for a,b in [('dict','scratch'),('structure','dict'),('selective','dict'),
            ('selective','structure'),('selective','aggressive'),
            ('oracle','dict'),('oracle','scratch'),('aggressive','dict')]:
    p,lo,hi=boot(a,b)
    print(f"  {a:>11} ÷ {b:<11} {p:>6.3f} [{lo:.3f}, {hi:.3f}]")
print('\n把源图采样计入成本之后（总成本 = 目标达标步数 + 40,000 源步）')
for a,b in [('dict','scratch'),('selective','scratch'),('oracle','scratch')]:
    p,lo,hi=boot(a,b,'total_cost')
    print(f"  {a:>11} ÷ {b:<11} {p:>6.3f} [{lo:.3f}, {hi:.3f}]")
print('\n按顺序重叠度分档（目标达标步数）')
print(f"{'重叠':>6} " + ' '.join(f'{a:>11}' for a in ARMS))
for band in (1.0,0.5,0.33,0.0):
    ps=[p for p in pairs if any(abs(r['overlap']-band)<.01 for r in per['dict'][p])]
    print(f"{band:>6} " + ' '.join(f"{mean(a,ps):>11.0f}" for a in ARMS))
