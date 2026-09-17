"""Reproduce the two follow-ups to probes.py without changing source/results."""
import json, subprocess
from concurrent.futures import ThreadPoolExecutor
from probes import OUT, BASE

def execute(job):
    row,cmd=job
    return dict(**row,command=cmd,result=json.loads(subprocess.check_output(cmd,text=True)))

if __name__=='__main__':
    previous=json.loads((OUT/'probe_results.json').read_text())
    jobs=[]
    for row in previous:
        if row['protocol']!='original' or row['arm']!='QRM (CRM)':continue
        for name,cost,reward in [('zero_cost_original_reward','0',str(row['result']['reward_success'])),
                                ('original_cost_unit_reward','.01','1')]:
            cmd=row['command']+['--cost',cost,'--reward-success',reward]
            jobs.append((dict(tag=row['tag'],seed=row['seed'],protocol=name),cmd))
    with ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(execute,jobs))
    (OUT/'qrm_reward_isolation.json').write_text(json.dumps(rows,indent=2)+'\n')
    jobs=[]
    for row in previous:
        if row['protocol']!='sparse':continue
        tag=row['tag'][:-4]+'safe'
        cmd=row['command'].copy()
        cmd[cmd.index('--task')+1]=str(BASE/f'results/craftworld_lava/{tag}.task')
        jobs.append((dict(tag=tag,arm=row['arm'],seed=row['seed'],protocol='sparse'),cmd))
    with ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(execute,jobs))
    (OUT/'safe_sparse_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
