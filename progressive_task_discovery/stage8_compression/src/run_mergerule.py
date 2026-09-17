"""The project's merge rule against RPNI and EDSM, inside the learning loop.

On real training evidence the three sit at very different operating points: the
project's sweep is cautious, precision .88 and recall .39, while the blue-fringe
learners merge hard, recall .83 and precision .47. Precision and recall do not
say which is better for the agent, so they are run in the loop, on the reversible
family and on the irreversible one, where a wrong merge stops being free.
"""
import json,subprocess,os,sys,statistics,random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

MERGE=["--theta","20","--quota","1","--strict-children","0"]
BUDGET=100000
GAP=[f"results/depth_vs_graph/gap{i:02d}.task" for i in range(12)]
FATAL=[f"results/irreversible/{e['fatal']['tag']}.task"
       for e in json.loads(open("results/irreversible/maps.json").read())]
ARMS={
 "history":  ["--method","history"],
 "sweep":    ["--method","merged"]+MERGE,
 "rpni":     ["--method","merged","--merge-rule","rpni"]+MERGE,
 "edsm":     ["--method","merged","--merge-rule","edsm"]+MERGE,
 "automaton":["--method","automaton"],
}

def one(job):
    arm,task,seed,doom=job
    cmd=["./compress","--task",task,"--seed",str(seed),"--budget",str(BUDGET),
         "--every","250"]+(["--doom-continues","1"] if doom else [])+ARMS[arm]
    r=json.loads(subprocess.check_output(cmd,text=True))
    return {"arm":arm,"task":task,"seed":seed,"family":"fatal" if doom else "gap",
            "first90":r["first90"] if r["first90"]>0 else BUDGET,"auc":r["auc"],
            "final":r["final_success"],"precision":r["merge_precision"],
            "recall":r["merge_recall"],"classes":r["classes"],"nodes":r["nodes"],
            "failures":r["failures"]}

if __name__=="__main__":
    jobs=[(a,t,s,False) for a in ARMS for t in GAP for s in range(12)]
    jobs+=[(a,t,s,True) for a in ARMS for t in FATAL for s in range(8)]
    out=[]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i,r in enumerate(ex.map(one,jobs,chunksize=4)):
            out.append(r)
            if i%200==0:print(i,len(jobs),file=sys.stderr,flush=True)
    with open("results/mergerule.jsonl","w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    for fam in ("gap","fatal"):
        rows=[r for r in out if r["family"]==fam]
        blocks=sorted({r["task"] for r in rows})
        idx=defaultdict(list)
        for r in rows: idx[(r["arm"],r["task"])].append(r["first90"])
        print(f"\n=== {fam} 族，{len(blocks)} 张地图 ===")
        print(f"{'臂':>10} {'达标步数':>9} {'AUC':>6} {'解出':>9} {'类数':>6} {'精度':>6} {'召回':>6}")
        for a in ARMS:
            g=[r for r in rows if r["arm"]==a]
            print(f"{a:>10} {statistics.mean(r['first90'] for r in g):>9.0f} "
                  f"{statistics.mean(r['auc'] for r in g):>6.3f} "
                  f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4} "
                  f"{statistics.mean(r['classes'] for r in g):>6.1f} "
                  f"{statistics.mean(r['precision'] for r in g):>6.3f} "
                  f"{statistics.mean(r['recall'] for r in g):>6.3f}")
        rnd=random.Random(41)
        def mean(a,ms):return statistics.mean(v for t in ms for v in idx[(a,t)])
        def boot(a,b,n=4000):
            d=sorted((lambda s:mean(b,s)/mean(a,s))([rnd.choice(blocks) for _ in blocks])
                     for _ in range(n))
            return mean(b,blocks)/mean(a,blocks),d[int(.025*n)],d[int(.975*n)]
        print("配对自助法（块=地图），左比右：")
        for a,b in [("rpni","sweep"),("edsm","sweep"),("sweep","history"),
                    ("rpni","history"),("automaton","sweep")]:
            p,lo,hi=boot(a,b)
            print(f"  {a:>6} ÷ {b:<10} {p:>6.2f} [{lo:.2f}, {hi:.2f}]")
