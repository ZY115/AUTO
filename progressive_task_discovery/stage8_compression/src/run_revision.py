"""Value transfer across a structure revision: this project's rule against JIRP's.

The reviewer noted that `--revision transfer`, which copies a new class's values
from whichever old class one of its members sat in, is not JIRP's rule. JIRP
transfers between hypothesis states that are equivalent with respect to future
behaviour. `--revision jirp` implements that as three rounds of signature
refinement over the class's own observable behaviour and its successors, and
transfers only where the match is unambiguous.
"""
import json,subprocess,os,sys,statistics,random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

MERGE=["--theta","20","--quota","1","--strict-children","0"]
BUDGET=100000
TASKS=[f"results/depth_vs_graph/gap{i:02d}.task" for i in range(12)]
REVS=["reset","replay","jirp","transfer"]

def one(job):
    rev,task,seed=job
    r=json.loads(subprocess.check_output(
        ["./compress","--task",task,"--seed",str(seed),"--budget",str(BUDGET),
         "--every","250","--method","merged","--revision",rev]+MERGE,text=True))
    return {"rev":rev,"task":task,"seed":seed,
            "first90":r["first90"] if r["first90"]>0 else BUDGET,
            "auc":r["auc"],"final":r["final_success"],
            "precision":r["merge_precision"],"recall":r["merge_recall"]}

if __name__=="__main__":
    jobs=[(v,t,s) for v in REVS for t in TASKS for s in range(12)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        out=list(ex.map(one,jobs,chunksize=4))
    with open("results/revision.jsonl","w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    idx=defaultdict(list)
    for r in out: idx[(r["rev"],r["task"])].append(r["first90"])
    print(f"{'修订策略':>10} {'达标步数':>9} {'AUC':>6} {'解出':>9}")
    for v in REVS:
        g=[r for r in out if r["rev"]==v]
        print(f"{v:>10} {statistics.mean(r['first90'] for r in g):>9.0f} "
              f"{statistics.mean(r['auc'] for r in g):>6.3f} "
              f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4}")
    rnd=random.Random(53)
    def mean(a,ms):return statistics.mean(v for t in ms for v in idx[(a,t)])
    def boot(a,b,n=4000):
        d=sorted((lambda s:mean(b,s)/mean(a,s))([rnd.choice(TASKS) for _ in TASKS])
                 for _ in range(n))
        return mean(b,TASKS)/mean(a,TASKS),d[int(.025*n)],d[int(.975*n)]
    print("\n配对自助法（块=任务，12块），左比右：")
    for a,b in [("transfer","jirp"),("jirp","reset"),("transfer","reset"),("jirp","replay")]:
        p,lo,hi=boot(a,b);print(f"  {a:>9} ÷ {b:<9} {p:>6.2f} [{lo:.2f}, {hi:.2f}]")
