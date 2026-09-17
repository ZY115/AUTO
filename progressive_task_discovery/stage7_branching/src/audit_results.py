import collections
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_branch import independent_rollouts,shortest_route,stochastic_evaluation


def main():
    counts={};identity_pairs=0;checked=[];stochastic=[];alias_checks=0;edge_checks=0
    for folder in sorted((ROOT/'results').iterdir()):
        path=folder/'raw.jsonl'
        if not path.exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        counts[folder.name]={'runs':len(rows),'training_steps':sum(r['budget'] for r in rows)}
        index={(r['condition'],r['k'],r['seed'],r['epsilon'],r['method']):r for r in rows}
        for r in rows:
            assert r['updates']==6*r['budget']
            assert r['prefix_steps']+r['frontier_steps']==r['budget']
            for step,p,pa,pb,*rest in r['checkpoints']:assert abs(p-(pa+pb)/2)<1e-12
            expected=set()
            for sequence in r['task']:
                key=0
                for e in sequence:expected.add((key,e));key=key*6+e+1
            observed=set()
            for node in r['nodes']:
                for g,dest in enumerate(node['edges']):
                    if dest<0:continue
                    assert (node['key'],g) in expected
                    assert r['nodes'][dest]['key']==node['key']*6+g+1
                    assert 0<node['discovery_steps'][g]<=r['budget']
                    observed.add((node['key'],g));edge_checks+=1
            assert len(observed)<=2*r['k']
            branching=r.get('branching',1);bridge=r.get('bridge',0)
            if branching and (r['method']=='count_replay' or (bridge>0 and r['method']=='last_event_replay')):
                assert all(x[1]<=.5 for x in r['checkpoints'])
                assert r['junction_state_accuracy']==.5
                alias_checks+=1
            equivalent={'history_replay':'learned_replay','history_skill':'immediate_skill'}
            if not branching:equivalent['count_replay']='hand_replay'
            if r['method'] in equivalent:
                key=(r['condition'],r['k'],r['seed'],r['epsilon'],equivalent[r['method']])
                if key in index:
                    other=index[key]
                    assert r['q_hash']==other['q_hash'] and r['trace_hash']==other['trace_hash']
                    assert r['checkpoints']==other['checkpoints'];identity_pairs+=1
            if folder.name in ['heldout','long_bridge'] and r['seed'] in [400,431,500,531] and r['k'] in [4,12]:
                outcomes=independent_rollouts(r)
                assert [x[0] for x in outcomes]==r['final_branch_success']
                assert [x[1] for x in outcomes]==r['final_eval_steps']
                for b in range(2):assert shortest_route(r,b)==12*r['k']-11<=r['horizon']
                checked.append([folder.name,r['condition'],r['method'],r['k'],r['seed']])
                if folder.name=='heldout' and r['seed']==400 and r['method'] in ['count_replay','last_event_replay']:
                    for epsilon,key in [(.05,'stochastic_final_success_e005'),(.2,'stochastic_final_success_e02')]:
                        p=stochastic_evaluation(r,epsilon);assert abs(p-r[key])<1e-9
                        stochastic.append([r['condition'],r['method'],r['k'],r['seed'],epsilon])
    result={'stages':counts,'main_runs':sum(x['runs'] for x in counts.values()),'training_steps':sum(x['training_steps'] for x in counts.values()),
            'identity_pairs':identity_pairs,'aliased_run_ceiling_checks':alias_checks,'validated_positive_edges':edge_checks,
            'independent_frozen_policies':checked,'independent_stochastic_evaluations':stochastic,
            'all_accounting_checks_passed':True}
    (ROOT/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
    print('Independent policy checks:',len(checked),'stochastic checks:',len(stochastic))


if __name__=='__main__':main()
