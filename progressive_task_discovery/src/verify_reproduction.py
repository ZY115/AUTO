"""Replay representative archived runs against the current executable."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
checks=[]
for stage in ['stage1_initial','stage3_skill_reuse','stage5_heldout','stage6_early_use']:
    rows=[json.loads(x) for x in (ROOT/'results'/stage/'raw.jsonl').read_text().splitlines()]
    first_seed=min(x['seed'] for x in rows)
    methods=['count','progressive','learned_qrm','count_replay','goal_reuse','delayed_goal','known_goal']
    for method in methods:
        candidates=[x for x in rows if x['method']==method and x['seed']==first_seed and x['k']==8 and x['condition'] in ['noisy','noisy_tight','benign']]
        if not candidates:continue
        old=candidates[-1]
        command=[str(ROOT/'pilot')]+old['command'][1:]
        new=json.loads(subprocess.check_output(command,text=True))
        for key in ['q_hash','trace_hash','checkpoints']:
            assert old[key]==new[key],(stage,method,key)
        checks.append({'stage':stage,'method':method,'seed':old['seed'],'condition':old['condition'],'passed':True})
(ROOT/'results/reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'{len(checks)} archived runs reproduced exactly (trajectory, Q table, evaluation checkpoints).')
