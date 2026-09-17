import json,statistics,random
from collections import defaultdict
rows=[json.loads(l) for l in open("results/goal_vs_index.jsonl")]
BUDGET=100000
by=defaultdict(list)
for r in rows: by[r["arm"]].append(r)
def cens(r): return r["first90"] if r["first90"]>0 else BUDGET
order=["count","history","automaton","automaton_depth","history_product","goal_untried","goal_learned","goal_first","goal_value"]
print(f"{'arm':>18} {'任务维':>6} {'树节点':>6} {'达标步数':>9} {'AUC':>6} {'解出':>8} {'首次成功':>8} {'收敛路线':>8} {'更新数':>10}")
for a in order:
    g=by[a]
    solved=sum(1 for r in g if r["final"]>=0.9)
    rt=[r["route"] for r in g if r["route"]]
    print(f"{a:>18} {statistics.mean(r['features'] for r in g):>6.1f} "
          f"{statistics.mean(r['nodes'] for r in g):>6.1f} "
          f"{statistics.mean(cens(r) for r in g):>9.0f} "
          f"{statistics.mean(r['auc'] for r in g):>6.3f} "
          f"{solved:>4}/{len(g):<3} "
          f"{(statistics.mean([r['first_success'] for r in g if r['first_success']>0]) if any(r['first_success']>0 for r in g) else float('nan')):>8.0f} "
          f"{statistics.mean(rt) if rt else 0:>8.1f} "
          f"{statistics.mean(r['updates'] for r in g):>10.0f}")

# Paired block bootstrap over tasks, which are the blocks; seeds are never blocks.
tasks=sorted({r["task"] for r in rows})
idx={(r["arm"],r["task"]):[] for r in rows}
for r in rows: idx[(r["arm"],r["task"])].append(cens(r))
def ratio(a,b,sample):
    na=statistics.mean(v for t in sample for v in idx[(a,t)])
    nb=statistics.mean(v for t in sample for v in idx[(b,t)])
    return nb/na
def boot(a,b,n=4000):
    pt=ratio(a,b,tasks)
    rnd=random.Random(11)
    d=sorted(ratio(a,b,[rnd.choice(tasks) for _ in tasks]) for _ in range(n))
    return pt,d[int(.025*n)],d[int(.975*n)]
print("\n配对自助法（块=任务，12块）：")
for a,b in [("goal_value","automaton"),("goal_value","history"),
            ("goal_value","goal_untried"),("goal_first","goal_value"),
            ("automaton","history"),("goal_value","history_product"),
            ("goal_value","automaton_depth"),("goal_learned","goal_value"),
            ("goal_learned","automaton"),("goal_learned","history")]:
    p,lo,hi=boot(a,b)
    print(f"  {a} 相对 {b}: {p:.2f} [{lo:.2f}, {hi:.2f}]")
# Per-task, so a single averaged win cannot hide a split.
print("\n逐任务达标步数：")
print(f"{'task':>8} {'history':>9} {'automaton':>10} {'goal':>8} {'goal/auto':>10} {'路线 auto':>10} {'路线 goal':>10}")
for t in tasks:
    h=statistics.mean(idx[("history",t)]);a=statistics.mean(idx[("automaton",t)])
    g=statistics.mean(idx[("goal_value",t)])
    ra=[r["route"] for r in by["automaton"] if r["task"]==t and r["route"]]
    rg=[r["route"] for r in by["goal_value"] if r["task"]==t and r["route"]]
    print(f"{t[-10:-5]:>8} {h:>9.0f} {a:>10.0f} {g:>8.0f} {a/g:>10.2f} "
          f"{statistics.mean(ra) if ra else 0:>10.1f} {statistics.mean(rg) if rg else 0:>10.1f}")
