import json, random, statistics, sys
from collections import defaultdict
stage=sys.argv[1] if len(sys.argv)>1 else 'pilot'
rows=[json.loads(l) for l in open(f'results/purity_{stage}.jsonl')]
ARMS=['scratch','dict_freq','dict_pure','struct_freq','struct_pure','oracle']
pairs=sorted({r['pair'] for r in rows})
per=defaultdict(lambda: defaultdict(list))
for r in rows: per[r['arm']][r['pair']].append(r)
print(f"=== {stage}，{len(pairs)} 对地图 × 4 源种子 × 4 目标种子 ===")
print(f"{'臂':>12} {'达标步数':>9} {'AUC':>6} {'解出':>9} {'收敛路线':>8} {'源频率介入':>9} {'其中改变决策':>11}")
for a in ARMS:
    g=[r for r in rows if r['arm']==a]
    rt=[r['route'] for r in g if r['route']]
    print(f"{a:>12} {statistics.mean(r['first90'] for r in g):>9.0f} "
          f"{statistics.mean(r['auc'] for r in g):>6.3f} "
          f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4} "
          f"{(statistics.mean(rt) if rt else 0):>8.1f} "
          f"{statistics.mean(r['rank_decisions'] for r in g):>9.0f} "
          f"{statistics.mean(r['rank_changed'] for r in g):>11.0f}")
rnd=random.Random(401)
def mean(a,s): return statistics.mean(r['first90'] for p in s for r in per[a][p])
def boot(a,b,n=4000):
    pt=mean(b,pairs)/mean(a,pairs)
    d=sorted(mean(b,s)/mean(a,s) for s in ([rnd.choice(pairs) for _ in pairs] for _ in range(n)))
    return pt,d[int(.025*n)],d[int(.975*n)]
print(f"\n配对自助法（块=地图对，{len(pairs)} 块），左比右，>1 表示左更快")
for a,b in [('dict_pure','dict_freq'),('struct_pure','struct_freq'),
            ('dict_freq','scratch'),('dict_pure','scratch'),
            ('struct_pure','dict_pure'),('struct_freq','dict_freq'),
            ('oracle','dict_pure'),('oracle','scratch')]:
    p,lo,hi=boot(a,b)
    print(f"  {a:>12} ÷ {b:<12} {p:>6.3f} [{lo:.3f}, {hi:.3f}]")
print('\n按重叠度分档（达标步数）')
bands=sorted({r['overlap'] for r in rows},reverse=True)
print(f"{'重叠':>6} " + ' '.join(f'{a:>12}' for a in ARMS))
for b in bands:
    ps=[p for p in pairs if any(abs(r['overlap']-b)<.01 for r in per['scratch'][p])]
    if not ps: continue
    print(f"{b:>6} " + ' '.join(f"{mean(a,ps):>12.0f}" for a in ARMS))
