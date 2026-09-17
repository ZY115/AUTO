import json, random
from collections import defaultdict
rows=[json.loads(l) for l in open('results/conversion.jsonl')]
RULES=['none','sweep_t20','sweep_t5','sweep_t1','sweep_t1q0','rpni','edsm','oracle']
pairs=sorted({r['pair'] for r in rows})
cell=defaultdict(lambda: defaultdict(list))
for r in rows:
    cell[(r['size'],r['collect'],r['rule'],r['world'])][r['pair']].append(r['outcome'])
def frac(key,sample,kinds):
    tot=hit=0
    for p in sample:
        v=cell[key][p];tot+=len(v);hit+=sum(1 for x in v if x in kinds)
    return hit/tot if tot else 0.0
rnd=random.Random(97)
def ci(fn,n=2000):
    pt=fn(pairs)
    d=sorted(fn([rnd.choice(pairs) for _ in pairs]) for _ in range(n))
    return pt,d[int(.025*n)],d[int(.975*n)]

for size in ('中等','较充分'):
    for coll in ('plain','directed'):
        print(f"\n=== 源预算 {size}，采集 {coll}，新历史（目标可达、源未见）===")
        print(f"{'推断规则':>11} {'世界':>5} {'答对':>7} {'答错':>7} {'折不进类':>8} {'类里无证据':>9} {'作答里精确率':>10}")
        for rule in RULES:
            for world in ('open','closed'):
                k=(size,coll,rule,world)
                c=frac(k,pairs,{'correct'}); w=frac(k,pairs,{'wrong'})
                u=frac(k,pairs,{'unreachable'}); ne=frac(k,pairs,{'no_evidence'})
                prec=c/(c+w) if c+w else float('nan')
                print(f"{rule:>11} {world:>5} {c:>7.3f} {w:>7.3f} {u:>8.3f} {ne:>9.3f} {prec:>10.3f}")

print("\n\n关键对照，块=地图对（25 块），源预算=较充分、普通采集")
for a,b,label in [(('较充分','plain','sweep_t20','open'),('较充分','plain','oracle','open'),'当前规则 ÷ 开放世界上限'),
                  (('较充分','plain','sweep_t20','closed'),('较充分','plain','sweep_t20','open'),'闭世界 ÷ 开放世界（同一划分）'),
                  (('较充分','plain','rpni','open'),('较充分','plain','sweep_t20','open'),'RPNI ÷ 当前规则（开放）'),
                  (('较充分','plain','sweep_t1','open'),('较充分','plain','sweep_t20','open'),'theta=1 ÷ theta=20（开放）')]:
    f=lambda s,a=a,b=b: frac(a,s,{'correct'})/max(frac(b,s,{'correct'}),1e-9)
    p,lo,hi=ci(f)
    print(f"  {label:<28} {p:>6.2f} [{lo:.2f}, {hi:.2f}]")

print("\n闭世界的代价：作答率与精确率，较充分、普通采集")
for rule in ('sweep_t20','rpni','oracle'):
    for world in ('open','closed'):
        k=('较充分','plain',rule,world)
        fa=lambda s,k=k: frac(k,s,{'correct','wrong'})
        fp=lambda s,k=k: frac(k,s,{'correct'})/max(frac(k,s,{'correct','wrong'}),1e-9)
        a=ci(fa); p=ci(fp)
        print(f"  {rule:>10} {world:>6}  作答 {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]   精确 {p[0]:.3f} [{p[1]:.3f},{p[2]:.3f}]")

print("\n折不进任何类的比例（这是任何推断规则都动不了的部分）")
for size in ('中等','较充分'):
    for coll in ('plain','directed'):
        k=(size,coll,'oracle','open')
        u=ci(lambda s,k=k: frac(k,s,{'unreachable'}))
        print(f"  {size:>5} {coll:>9}  {u[0]:.3f} [{u[1]:.3f}, {u[2]:.3f}]")
