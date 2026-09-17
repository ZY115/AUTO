import json,statistics,random
from collections import defaultdict
rows=[json.loads(l) for l in open("results/priorart.jsonl")]
by=defaultdict(list)
for r in rows: by[r["arm"]].append(r)
order=["index_plain","index_replay23","index_crm","learned_plain","learned_crm",
       "goal_norelabel","goal_relabel","goal_learned"]
print(f"{'臂':>16} {'更新/步':>8} {'达标步数':>9} {'AUC':>6} {'解出':>9} {'首次成功':>8} {'收敛路线':>8}")
for a in order:
    g=by[a];rt=[r["route"] for r in g if r["route"]]
    fs=[r["first_success"] for r in g if r["first_success"]>0]
    print(f"{a:>16} {statistics.mean(r['updates'] for r in g)/100000:>8.1f} "
          f"{statistics.mean(r['first90'] for r in g):>9.0f} "
          f"{statistics.mean(r['auc'] for r in g):>6.3f} "
          f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4} "
          f"{(statistics.mean(fs) if fs else 0):>8.0f} "
          f"{(statistics.mean(rt) if rt else 0):>8.1f}")
tasks=sorted({r["task"] for r in rows})
idx=defaultdict(list)
for r in rows: idx[(r["arm"],r["task"])].append(r["first90"])
def ratio(a,b,sample):
    na=statistics.mean(v for t in sample for v in idx[(a,t)])
    nb=statistics.mean(v for t in sample for v in idx[(b,t)])
    return nb/na
def boot(a,b,n=4000):
    rnd=random.Random(31)
    d=sorted(ratio(a,b,[rnd.choice(tasks) for _ in tasks]) for _ in range(n))
    return ratio(a,b,tasks),d[int(.025*n)],d[int(.975*n)]
print("\n配对自助法（块=任务，12块），左比右：")
pairs=[("index_crm","index_plain"),("index_crm","index_replay23"),
       ("index_replay23","index_plain"),("learned_crm","learned_plain"),
       ("goal_relabel","index_crm"),("goal_relabel","index_replay23"),
       ("goal_norelabel","index_crm"),("goal_relabel","goal_norelabel"),
       ("goal_learned","learned_crm")]
for a,b in pairs:
    p,lo,hi=boot(a,b)
    print(f"  {a:>15} ÷ {b:<15} {p:>6.2f} [{lo:.2f}, {hi:.2f}]")
print("\n逐任务：CRM 是否胜过等更新数的普通回放")
win=0
for t in tasks:
    c=statistics.mean(idx[("index_crm",t)]);r=statistics.mean(idx[("index_replay23",t)])
    win+= (c<r)
    print(f"  {t[-10:-5]:>6} crm {c:>7.0f}  replay23 {r:>7.0f}  {'crm 胜' if c<r else '回放胜'}")
print(f"  CRM 胜 {win}/12")
