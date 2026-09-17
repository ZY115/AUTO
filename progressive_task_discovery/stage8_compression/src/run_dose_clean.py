"""The dose-response for irreversibility, re-measured with clean failure semantics.

The original run of this experiment attributed every step taken after the task
had already failed to the healthy history that preceded the failure: those steps
were written as "this event was ignored here", counted as further failures, and
charged the whole remaining horizon again. The heaviest dose is exactly the
condition that produces such steps, so it was the most affected. This re-runs the
three doses on the corrected program, and adds the goal arms, which did not exist
when the original was run.
"""
import concurrent.futures, hashlib, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGE = ['--theta','20','--quota','1','--strict-children','0','--revision','transfer']
ARMS = {
    'history':     ['--method','history'],
    'merged':      ['--method','merged']+MERGE,
    'automaton':   ['--method','automaton'],
    'goal_learned':['--method','goal','--goal-select','learned'],
    'goal_true':   ['--method','goal','--goal-select','value'],
}
# ignored: a wrong commit costs nothing. terminates: the episode ends, which
# hands back a free early stop. wastes: the episode runs on with success already
# impossible, which is the waste the project's oldest finding was about.
DOSES = {'ignored': ('safe', 0), 'terminates': ('fatal', 0), 'wastes': ('fatal', 1)}
BUDGET = 200000

def job(a):
    entry, dose, arm, seed = a
    kind, doom = DOSES[dose]
    task = ROOT / f"results/irreversible/{entry[kind]['tag']}.task"
    cmd = [str(ROOT/'compress'),'--task',str(task),'--seed',str(seed),
           '--budget',str(BUDGET),'--every','500','--doom-continues',str(doom)]+ARMS[arm]
    r = json.loads(subprocess.check_output(cmd, text=True))
    return {'arm':arm,'dose':dose,'map':entry[kind]['tag'],'seed':seed,
            'first90': r['first90'] if r['first90']>0 else BUDGET,
            'auc':r['auc'],'final':r['final_success'],
            'failures':r['failures'],'doomed':r['doomed_steps']}

import subprocess
if __name__ == '__main__':
    maps = json.loads((ROOT/'results/irreversible/maps.json').read_text())
    jobs = [(e,d,a,s) for e in maps for d in DOSES for a in ARMS for s in range(6000,6016)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(job, jobs))
    out = ROOT/'results/irreversible/dose_clean.jsonl'
    out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (ROOT/'results/irreversible/dose_clean.manifest.json').write_text(json.dumps(dict(
        runner='src/run_dose_clean.py', maps=len(maps), seeds=16, budget=BUDGET,
        doses=list(DOSES), arms=list(ARMS), runs=len(rows),
        source_sha256=hashlib.sha256((ROOT/'src/compress.cpp').read_bytes()).hexdigest()),
        indent=2)+'\n')
    per = defaultdict(list)
    for r in rows: per[(r['arm'],r['dose'],r['map'])].append(r)
    blocks = sorted({r['map'] for r in rows if r['dose']=='wastes'})
    for dose in DOSES:
        bl = sorted({r['map'] for r in rows if r['dose']==dose})
        print(f"\n=== 剂量 {dose} ===")
        print(f"{'臂':>13} {'达标步数':>9} {'AUC':>6} {'解出':>9} {'致命次数':>8} {'浪费步数':>9}")
        for a in ARMS:
            g=[r for r in rows if r['arm']==a and r['dose']==dose]
            print(f"{a:>13} {statistics.mean(r['first90'] for r in g):>9.0f} "
                  f"{statistics.mean(r['auc'] for r in g):>6.3f} "
                  f"{sum(1 for r in g if r['final']>=.9):>4}/{len(g):<4} "
                  f"{statistics.mean(r['failures'] for r in g):>8.1f} "
                  f"{statistics.mean(r['doomed'] for r in g):>9.0f}")
        rnd=random.Random(67)
        def mean(a,ms):
            return statistics.mean(r['first90'] for m in ms for r in per[(a,dose,m)])
        def boot(a,b,n=4000):
            d=sorted((lambda s:mean(b,s)/mean(a,s))([rnd.choice(bl) for _ in bl]) for _ in range(n))
            return mean(b,bl)/mean(a,bl),d[int(.025*n)],d[int(.975*n)]
        for a,b in [('merged','history'),('automaton','history'),
                    ('goal_learned','merged'),('goal_true','automaton'),
                    ('goal_learned','goal_true')]:
            p,lo,hi=boot(a,b)
            print(f"  {a:>12} ÷ {b:<12} {p:>6.2f} [{lo:.2f}, {hi:.2f}]")
