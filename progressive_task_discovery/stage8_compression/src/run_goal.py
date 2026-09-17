"""The automaton read as a goal provider against the automaton read as a table index.

The question this answers is whether the task automaton is worth anything to RL
beyond naming the next sub-goal. If naming the goal is all of it, then the agent
needs no task-indexed value function at all: it needs the goal-conditioned skills
it already trains, plus a constant-memory fold over the event stream to say which
skill to run. The arms below hold that fold fixed and the true automaton
constant, and vary only how its output is consumed.
"""
import json,subprocess,itertools,os,sys
from concurrent.futures import ProcessPoolExecutor

TASKS=[f"results/depth_vs_graph/gap{i:02d}.task" for i in range(12)]
SEEDS=range(16)
BUDGET=100000
ARMS={
    # No task memory at all: the count of progress events.
    "count":       ["--method","count"],
    # The progress-event history, the project's standing baseline.
    "history":     ["--method","history"],
    # The true automaton as a table index: Q(cell, task state, action).
    "automaton":   ["--method","automaton"],
    # The true automaton as a goal provider: Q(cell, goal, action), goal chosen
    # among the events the automaton says advance the task from here.
    "goal_value":  ["--method","goal","--goal-select","value"],
    "goal_first":  ["--method","goal","--goal-select","first"],
    # The automaton the agent builds online, read as a goal provider: the goal is
    # whatever the discovered prefix tree says advanced the task at this history.
    "goal_learned":["--method","goal","--goal-select","learned"],
    # The goal named without consulting the automaton: an event this episode has
    # not collected yet. If this matches the arms above, the automaton is not
    # what supplies the goal.
    "goal_untried":["--method","goal","--goal-select","untried"],
    # The two strongest arms measured earlier in this project, at the scales
    # those experiments selected on development seeds, for scale.
    "history_product":["--method","history","--shape","product","--shape-scale","0.02"],
    "automaton_depth":["--method","automaton","--shape","frontier","--shape-scale","0.07"],
}

def one(job):
    arm,task,seed=job
    cmd=["./compress","--task",task,"--seed",str(seed),"--budget",str(BUDGET),
         "--every","100"]+ARMS[arm]
    r=json.loads(subprocess.check_output(cmd,text=True))
    ck=r["checkpoints"]
    tail=[c for c in ck if c[0]>=BUDGET*0.8 and c[1]>0]
    return {"arm":arm,"task":task,"seed":seed,
            "first90":r["first90"],"auc":r["auc"],"final":r["final_success"],
            "first_success":r["first_success"],"features":r["features"],"nodes":r["nodes"],
            "updates":r["updates"]+r["skill_updates"],
            # Route length once converged: the goal reading optimises each leg
            # separately, so this is where it can be worse than a table indexed
            # by the whole task state.
            "route":sum(c[2] for c in tail)/len(tail) if tail else None}

if __name__=="__main__":
    jobs=[(a,t,s) for a in ARMS for t in TASKS for s in SEEDS]
    out=[]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i,r in enumerate(ex.map(one,jobs,chunksize=4)):
            out.append(r)
            if i%100==0:print(i,len(jobs),file=sys.stderr,flush=True)
    with open("results/goal_vs_index.jsonl","w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    print("wrote",len(out))
