"""Paired Gate statistics and predeclared sparse calibration, from raw data."""
from pathlib import Path
import csv,json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'stage7_branching'

def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def write_csv(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio_ci(a,b):
    a,b=np.array(a),np.array(b);rng=np.random.default_rng(812)
    ix=rng.integers(0,len(a),(10000,len(a)))
    return list(np.quantile(a[ix].mean(1)/b[ix].mean(1),[.025,.975]))
def checkpoint_metrics(r,budget):
    c=[x for x in r['checkpoints'] if x[0]<=budget]
    assert c[-1][0]==budget
    auc=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(c,c[1:]))/budget
    hits=[x[0] for x in c if x[1]>=.9]
    return auc,c[-1][1],min(hits) if hits else -1

def gate():
    out=[]
    for name in ['heldout','same_map_control']:
        new=read(ROOT/'results'/('gate_'+name)/'raw.jsonl')
        old={(r['condition'],r['k'],r['seed']):r for r in read(OLD/'results'/name/'raw.jsonl') if r['variant']=='immediate_skill'}
        for condition in ['h16','h24']:
            for k in [4,8,12]:
                rs=[r for r in new if r['condition']==condition and r['k']==k]
                a=[r['first90'] for r in rs];b=[old[condition,k,r['seed']]['first90'] for r in rs]
                assert min(a+b)>0
                for r in rs:
                    o=old[condition,k,r['seed']]
                    assert r['task']==o['task'] and r['epsilon']==o['epsilon']
                    assert r['updates']==o['updates']==6*r['budget']
                    assert r['wrong_binding_steps']==0
                lo,hi=ratio_ci(a,b)
                out.append(dict(dataset=name,condition=condition,k=k,n=len(rs),hand_skill=np.mean(a),immediate_skill=np.mean(b),ratio=np.mean(a)/np.mean(b),ratio_lo=lo,ratio_hi=hi,hand_auc=np.mean([r['auc'] for r in rs]),hand_final=np.mean([r['final_success'] for r in rs])))
    write_csv(ROOT/'results/gate_summary.csv',out)
    return out

def calibration():
    rows=read(ROOT/'results/sparse_development/raw.jsonl');groups=collections.defaultdict(list)
    for r in rows:groups[r['gamma'],r['reward_ratio'],r['method'],r['k']].append(r)
    out=[]
    for (g,r,m,k),rs in groups.items():
        for budget in [50000,100000,200000]:
            xs=[checkpoint_metrics(x,budget) for x in rs]
            L=12*k-11;R=r*.01*L
            optimal_return=-(.01*(1-g**(L-1))/(1-g))+g**(L-1)*R
            out.append(dict(gamma=g,reward_ratio=r,method=m,k=k,budget=budget,n=len(rs),auc=np.mean([x[0] for x in xs]),final_success=np.mean([x[1] for x in xs]),solved=sum(x[2]>=0 for x in xs),mean_capped_first90=np.mean([x[2] if x[2]>=0 else budget for x in xs]),shortest_length=L,terminal_reward=R,optimal_discounted_return=optimal_return,zero_above_optimal_return=optimal_return<0))
    write_csv(ROOT/'results/sparse_calibration.csv',out)
    scores=[]
    for g in [.98,.99,.995]:
        for r in [.5,2,8]:
            xs=[x for x in out if x['gamma']==g and x['reward_ratio']==r and x['budget']==100000]
            scores.append((np.mean([x['auc'] for x in xs]),-r,-g,g,r))
    _,_,_,g,r=max(scores)
    budgets=[b for b in [50000,100000,200000] if all(x['solved']/x['n']>=.9 for x in out if x['gamma']==g and x['reward_ratio']==r and x['budget']==b)]
    selected=dict(gamma=g,reward_ratio=r,budget=min(budgets) if budgets else None,epsilon=.02,alpha=.3,selection='pooled development AUC at 100k; budget>=90% solved per method/length',development_seeds=list(range(12)))
    target=ROOT/'docs/frozen_sparse.json'
    if target.exists():assert json.loads(target.read_text())==selected
    else:target.write_text(json.dumps(selected,indent=2)+'\n')
    return selected

if __name__=='__main__':
    print('Gate:',[r for r in gate() if r['condition']=='h24' and r['k']==12])
    print('Frozen sparse calibration:',calibration())
