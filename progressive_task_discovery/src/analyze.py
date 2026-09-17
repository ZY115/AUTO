"""Budget-aware summaries, paired bootstrap, and independent feasibility bounds."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = [(0,0),(4,0),(8,0),(0,4),(8,4),(0,8),(4,8),(8,8)]


def read(stage):
    return [json.loads(line) for line in (ROOT/'results'/stage/'raw.jsonl').read_text().splitlines()]


def capped(row, field='first90'):
    return row[field] if row[field] >= 0 else row['budget']


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(data):
    groups=collections.defaultdict(list)
    for r in data:
        groups[r['condition'],r['variant'],r['k']].append(r)
    output=[]
    for (condition,variant,k), rows in groups.items():
        times=np.array([capped(x) for x in rows],dtype=float)
        rng=np.random.default_rng(8721)
        bs=times[rng.integers(len(rows),size=(5000,len(rows)))].mean(axis=1)
        output.append(dict(condition=condition,variant=variant,k=k,n=len(rows),
            mean_capped_first90=times.mean(),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),
            median_capped_first90=np.median(times),solved=sum(x['first90']>=0 for x in rows),
            stable_solved=sum(x['stable90']>=0 for x in rows),mean_capped_stable90=np.mean([capped(x,'stable90') for x in rows]),
            auc=np.mean([x['auc'] for x in rows]),final_success=np.mean([x['final_success'] for x in rows]),
            final90=sum(x['final_success']>=.9 for x in rows),updates_per_step=np.mean([x['updates']/x['budget'] for x in rows])))
    return output


def paired(data, baseline):
    index={(x['condition'],x['k'],x['seed'],x['variant']):x for x in data}
    groups=collections.defaultdict(list)
    for x in data:
        key=(x['condition'],x['k'],x['seed'],baseline)
        if x['variant']!=baseline and key in index:
            groups[x['condition'],x['k'],x['variant']].append((index[key],x))
    output=[]
    for (condition,k,variant), pairs in groups.items():
        b=np.array([capped(x) for x,y in pairs]); v=np.array([capped(y) for x,y in pairs])
        ad=np.array([y['auc']-x['auc'] for x,y in pairs])
        rng=np.random.default_rng(12541)
        ix=rng.integers(len(b),size=(10000,len(b)))
        delta=(v[ix]-b[ix]).mean(axis=1)
        ratio=v[ix].mean(axis=1)/b[ix].mean(axis=1)
        ad_boot=ad[ix].mean(axis=1)
        output.append(dict(condition=condition,k=k,baseline=baseline,variant=variant,n=len(b),
            cost_delta=float((v-b).mean()),delta_low=np.quantile(delta,.025),delta_high=np.quantile(delta,.975),
            cost_ratio=float(v.mean()/b.mean()),ratio_low=np.quantile(ratio,.025),ratio_high=np.quantile(ratio,.975),
            auc_delta=ad.mean(),auc_delta_low=np.quantile(ad_boot,.025),auc_delta_high=np.quantile(ad_boot,.975),
            wins=int((v<b).sum()),ties=int((v==b).sum()),losses=int((v>b).sum())))
    return output


def feasibility(row):
    """Independent optimal finite-horizon DP (time-dependent privileged policy)."""
    path=[(4,4)]+[CELLS[e] for e in row['task']]
    shortest=sum(abs(x-u)+abs(y-v) for (x,y),(u,v) in zip(path,path[1:]))
    k,H,slip=row['k'],row['horizon'],row['slip']
    if slip==0:
        return shortest,float(shortest<=H)
    label={y*9+x:e for e,(x,y) in enumerate(CELLS)}
    dest=np.zeros((81,4),dtype=int); events=np.full((81,4),-1,dtype=int)
    for s in range(81):
        y,x=divmod(s,9)
        for b,(dx,dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
            z=max(0,min(8,y+dy))*9+max(0,min(8,x+dx))
            dest[s,b]=z
            if z!=s:events[s,b]=label.get(z,-1)
    value=np.zeros((k+1,81));value[k]=1
    nextq=np.array([q+(events==e).astype(int) for q,e in enumerate(row['task'])])
    for _ in range(H):
        action_value=value[nextq,dest[None,:,:]]
        new=np.zeros_like(value);new[k]=1
        new[:k]=(1-slip)*action_value.max(axis=2)+slip*action_value.mean(axis=2)
        value=new
    return shortest,float(value[0,40])


def audit(data):
    distinct={}
    for x in data:
        key=(x['condition'],x['k'],x['seed'])
        if key in distinct:
            assert distinct[key]['task']==x['task']
        distinct[key]=x
        assert sum(x['stage_steps'])==x['budget']
    output=[]
    for (condition,k,seed),x in distinct.items():
        shortest,bound=feasibility(x)
        output.append(dict(condition=condition,k=k,seed=seed,shortest_steps=shortest,horizon=x['horizon'],
                           optimal_success_bound=bound,threshold90_feasible=bound>=.9))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--baseline',default='count');p.add_argument('--audit',action='store_true')
    args=p.parse_args(); data=read(args.stage); out=ROOT/'results'/args.stage
    summary=summarize(data); write_csv(out/'summary.csv',summary)
    write_csv(out/f'paired_vs_{args.baseline}.csv',paired(data,args.baseline))
    if args.audit:
        bounds=audit(data);write_csv(out/'feasibility.csv',bounds)
        print('Unattainable 90% thresholds:',sum(not x['threshold90_feasible'] for x in bounds),'/',len(bounds))
    for x in summary:
        if x['k']==max(r['k'] for r in data):
            print(f"{x['condition']:12s} {x['variant']:30s} cost={x['mean_capped_first90']:8.0f} solved={x['solved']:2}/{x['n']} AUC={x['auc']:.3f} final={x['final_success']:.3f}")


if __name__=='__main__':
    main()
