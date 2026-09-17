"""Does the goal reading of the automaton survive an irreversible mistake?

Reading the automaton as a goal provider commits the agent to walking to one
event. Where a wrong commit is unrecoverable that is a bet, and the agent making
it holds no task-indexed value function that could learn to hedge. This is the
condition that this project has repeatedly found to be the one that matters, so
the goal reading has to be measured under it before the headline can stand.

Heaviest dose only: the wrong commit is fatal and the episode is allowed to run
on, so the remaining steps are wasted rather than handing back a free early stop.
"""
import concurrent.futures, hashlib, json, random, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGE = ['--theta','20','--quota','1','--strict-children','0','--revision','transfer']
ARMS = {
    "history":      ["--method","history"],
    "merged":       ["--method","merged"]+MERGE,
    "automaton":    ["--method","automaton"],
    "goal_value":   ["--method","goal","--goal-select","value"],
    "goal_learned": ["--method","goal","--goal-select","learned"],
}
BUDGET=200000

def job(a):
    entry,arm,seed=a
    task=ROOT/f"results/irreversible/{entry['fatal']['tag']}.task"
    cmd=[str(ROOT/'compress'),'--task',str(task),'--seed',str(seed),'--budget',str(BUDGET),
         '--every','500','--doom-continues','1']+ARMS[arm]
    r=json.loads(subprocess.check_output(cmd,text=True))
    return {"arm":arm,"map":entry['fatal']['tag'],"seed":seed,
            "first90":r["first90"] if r["first90"]>0 else BUDGET,
            "auc":r["auc"],"final":r["final_success"],"failures":r["failures"],
            "doomed":r["doomed_steps"],"features":r["features"],"nodes":r["nodes"]}

if __name__=="__main__":
    maps=json.loads((ROOT/'results/irreversible/maps.json').read_text())
    jobs=[(e,a,s) for e in maps for a in ARMS for s in range(5000,5012)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(job,jobs))
    (ROOT/'results/irreversible/goal.jsonl').write_text(
        ''.join(json.dumps(r)+'\n' for r in rows))
    blocks=sorted({r["map"] for r in rows})
    def mean(arm,ms,key="first90"):
        v=[r[key] for r in rows if r["arm"]==arm and r["map"] in ms]
        return statistics.mean(v)
    print(f"{'arm':>14} {'达标步数':>9} {'AUC':>6} {'解出':>9} {'致命次数':>9} {'浪费步数':>10}")
    for a in ARMS:
        g=[r for r in rows if r["arm"]==a]
        print(f"{a:>14} {mean(a,blocks):>9.0f} {statistics.mean(r['auc'] for r in g):>6.3f} "
              f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4} "
              f"{statistics.mean(r['failures'] for r in g):>9.0f} "
              f"{statistics.mean(r['doomed'] for r in g):>10.0f}")
    # Blocks are maps, never seeds, and one block sample serves both sides.
    rnd=random.Random(23)
    def boot(x,y,n=4000):
        pt=mean(y,blocks)/mean(x,blocks)
        d=sorted((lambda s:mean(y,s)/mean(x,s))([rnd.choice(blocks) for _ in blocks])
                 for _ in range(n))
        return pt,d[int(.025*n)],d[int(.975*n)]
    print("\n配对自助法（块=地图，15块）：")
    for x,y in [("goal_value","automaton"),("goal_learned","goal_value"),
                ("goal_learned","merged"),("goal_learned","history"),
                ("automaton","history")]:
        p,lo,hi=boot(x,y)
        print(f"  {x} 相对 {y}: {p:.2f} [{lo:.2f}, {hi:.2f}]")
