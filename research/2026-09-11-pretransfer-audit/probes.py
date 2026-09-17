"""Bounded replay and reward-protocol sensitivity checks; not a confirmatory grid."""
import json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]/'progressive_task_discovery/stage8_compression'
ARMS={'QRM (CRM)':['--method','automaton','--crm','1'],
      'Y10 no-reuse readout':['--method','goal','--skill-key','perstate'],
      'Y11 goal readout':['--method','goal']}
SAVED=json.loads((BASE/'results/craftworld_lava_factorial.json').read_text())
def run(job):
    tag,arm,seed,protocol=job
    cmd=[str(OUT/'compress-rebuilt'),'--task',str(BASE/f'results/craftworld_lava/{tag}.task'),
         '--seed',str(seed),'--budget','1000000','--every','2000']+ARMS[arm]
    if protocol=='sparse': cmd+=['--cost','0','--reward-success','1']
    r=json.loads(subprocess.check_output(cmd,text=True))
    checks={}
    if protocol=='original':
        old=SAVED[tag][arm][seed]
        now={'auc':r['auc'],'final':r['final_success'],'f90':r['first90'],
             'fails':r['failures'],'fail90':r['fail_at_90'],'gd':sum(r['goal_deaths'])}
        checks['saved_summary_exact_match']=old==now
        existing=json.loads(subprocess.check_output([str(BASE/'compress')]+cmd[1:],text=True))
        checks['current_binary_full_output_match']=existing==r
    return dict(tag=tag,arm=arm,seed=seed,protocol=protocol,command=cmd,checks=checks,result=r)
if __name__=='__main__':
    tags=['cwl_book_expensive_0_lava','cwl_bookandquill_expensive_0_lava']
    jobs=[(t,a,s,p) for t in tags for a in ARMS for s in range(3)
          for p in ('original','sparse')]
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(run,jobs))
    (OUT/'probe_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows: print(r['tag'],r['arm'],r['seed'],r['protocol'],
                         r['result']['auc'],r['result']['first90'],r['checks'])
