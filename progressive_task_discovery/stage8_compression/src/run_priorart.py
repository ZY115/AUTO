"""Head to head against the prior art this project's own plan made mandatory.

docs/PLAN.md line 326: before writing, the contribution has to be compared face
to face with JIRP's equivalent-state transfer and with reward machines'
counterfactual experience, and until then novelty may not be claimed.
REPORT_STEP9.md records that neither comparison was ever run.

It matters most for Step 15. That step attributed a 17x result to data sharing:
one physical step being evidence for every goal at once. But that is exactly what
counterfactual experience for reward machines does across task states, so the
claim is only about the reading if the same update rule, applied to the reading
that indexes by task state, fails to close the gap.

Two factors, crossed, plus the control that separates a smarter update from
simply more updates.
"""
import json,subprocess,os,sys
from concurrent.futures import ProcessPoolExecutor

TASKS=[f"results/depth_vs_graph/gap{i:02d}.task" for i in range(12)]
SEEDS=range(16)
BUDGET=100000
MERGE=["--theta","20","--quota","1","--strict-children","0"]
ARMS={
  # Index reading, the one every step from 7 to 15 measured. Six updates a step.
  "index_plain":    ["--method","automaton"],
  # Same reading and same update rule, but as many updates a step as CRM spends,
  # drawn from the replay buffer. This is what separates "counterfactual" from
  # "more Bellman updates".
  "index_replay23": ["--method","automaton","--replay","23"],
  # Counterfactual experience for reward machines, on the true machine.
  "index_crm":      ["--method","automaton","--crm","1"],
  # The same rule on the machine the agent infers online, which is what JIRP
  # plus QRM amounts to here.
  "learned_plain":  ["--method","merged"]+MERGE,
  "learned_crm":    ["--method","merged","--crm","1"]+MERGE,
  # Goal reading with the relabelling switched off: two updates a step, and the
  # goal it trains is the one it was actually pursuing.
  "goal_norelabel": ["--method","goal","--goal-select","value","--relabel","0"],
  "goal_relabel":   ["--method","goal","--goal-select","value"],
  "goal_learned":   ["--method","goal","--goal-select","learned"],
}

def one(job):
    arm,task,seed=job
    cmd=["./compress","--task",task,"--seed",str(seed),"--budget",str(BUDGET),
         "--every","100"]+ARMS[arm]
    r=json.loads(subprocess.check_output(cmd,text=True))
    ck=r["checkpoints"];tail=[c for c in ck if c[0]>=BUDGET*0.8 and c[1]>0]
    return {"arm":arm,"task":task,"seed":seed,
            "first90":r["first90"] if r["first90"]>0 else BUDGET,
            "auc":r["auc"],"final":r["final_success"],"first_success":r["first_success"],
            "features":r["features"],"updates":r["updates"]+r["skill_updates"],
            "route":sum(c[2] for c in tail)/len(tail) if tail else None}

if __name__=="__main__":
    jobs=[(a,t,s) for a in ARMS for t in TASKS for s in SEEDS]
    out=[]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i,r in enumerate(ex.map(one,jobs,chunksize=4)):
            out.append(r)
            if i%200==0:print(i,len(jobs),file=sys.stderr,flush=True)
    with open("results/priorart.jsonl","w") as f:
        for r in out:f.write(json.dumps(r)+"\n")
    print("wrote",len(out))
