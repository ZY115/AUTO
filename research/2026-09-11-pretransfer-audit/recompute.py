"""Read-only independent checks of saved Step 38/38.5 data and task files.
Run from any directory with Python 3; writes only beside this script.
"""
import hashlib
import json
import random
from collections import deque
from pathlib import Path
from statistics import mean, median

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'progressive_task_discovery/stage8_compression'
RES = BASE / 'results'
TAU = 1_000_000


def task(path):
    lines = iter(path.read_text().splitlines())
    ints = lambda: list(map(int, next(lines).split()))
    A = ints()[0]; Q = ints()[0]
    cls, acc, fail = ints(), ints(), ints()
    trans = [ints() for _ in range(Q)]
    N = ints()[0]
    rows = [ints() for _ in range(N)]
    moves = [r[::2] for r in rows]; events = [r[1::2] for r in rows]
    S = ints()[0]; starts, lengths, protocol = ints(), ints(), ints()
    assert S == len(starts)
    return dict(A=A,Q=Q,N=N,acc=acc,fail=fail,trans=trans,moves=moves,
                events=events,starts=starts,lengths=lengths,protocol=protocol)


def distance(t):
    """Independent reverse BFS on the explicit product graph."""
    N,Q = t['N'],t['Q']; rev = [[] for _ in range(N*Q)]
    for q in range(Q):
        if t['fail'][q] or t['acc'][q]: continue
        for s in range(N):
            for a in range(4):
                e=t['events'][s][a]; nq=q if e<0 else t['trans'][q][e]
                rev[nq*N+t['moves'][s][a]].append(q*N+s)
    d = [None]*(N*Q); queue=deque()
    for q in range(Q):
        if t['acc'][q]:
            for s in range(N): d[q*N+s]=0;queue.append(q*N+s)
    while queue:
        n=queue.popleft()
        for p in rev[n]:
            if d[p] is None: d[p]=d[n]+1;queue.append(p)
    return d


def boot(v, seed=5, n=10000):
    rng=random.Random(seed)
    vals=sorted(mean(rng.choices(v,k=len(v))) for _ in range(n))
    return dict(mean=mean(v),lo=vals[int(n*.025)],hi=vals[int(n*.975)])


def rmst(v):
    # All censoring is administrative at tau. A censored contribution of tau
    # estimates restricted mean time; it is NOT an imputed uncensored T90.
    return mean(x['f90'] if x['f90']>0 else TAU for x in v)


def main():
    R=json.loads((RES/'craftworld_lava_factorial.json').read_text())
    meta={x['tag']:x for x in json.loads((RES/'craftworld_lava/tasks.json').read_text())}
    arms=list(next(iter(R.values())))
    cells={}
    for cond in ('cheap','expensive'):
        for risk in ('safe','lava'):
            for a in arms:
                v=[r for t in R if meta[t]['cond']==cond and meta[t]['risk']==risk for r in R[t][a]]
                deaths=[x['fail90'] for x in v if x['fail90']>=0]
                cells[f'{cond}|{risk}|{a}']=dict(n=len(v),auc=mean(x['auc'] for x in v),
                    attained=sum(x['f90']>0 for x in v), final90=sum(x['final']>=.9 for x in v),
                    fell_below90=sum(x['f90']>0 and x['final']<.9 for x in v),
                    rmst_steps=rmst(v),attainment_area=1-rmst(v)/TAU,
                    median_failure_pre90=median(deaths) if deaths else None,
                    mean_failure_pre90=mean(deaths) if deaths else None,
                    mean_failure_fixed_budget=mean(x['fails'] for x in v),
                    median_failure_fixed_budget=median(x['fails'] for x in v))
    y10,y11='Y10 no-reuse readout','Y11 goal readout'
    pairs=[];geo=[]
    for tag,m in meta.items():
        if m['risk']!='lava':continue
        safe=tag[:-4]+'safe'
        auc_effect=lambda t:mean(x['auc'] for x in R[t][y11])-mean(x['auc'] for x in R[t][y10])
        time_effect=lambda t:(rmst(R[t][y10])-rmst(R[t][y11]))/TAU
        pairs.append(dict(task=m['task'],cond=m['cond'],seed=m['seed'],tag=tag,
                          auc=auc_effect(tag)-auc_effect(safe),
                          attainment_area=time_effect(tag)-time_effect(safe)))
        l=task(RES/'craftworld_lava'/f'{tag}.task');s=task(RES/'craftworld_lava'/f'{safe}.task')
        dl,ds=distance(l),distance(s)
        assert all(l[k]==s[k] for k in ('moves','events','starts','protocol'))
        ll=[dl[c] for c in l['starts']];sl=[ds[c] for c in s['starts']]
        assert ll==l['lengths']
        geo.append(dict(tag=tag,lava=ll,safe=sl,stored_safe=s['lengths'],
                        increased=sum(a>b for a,b in zip(ll,sl)),
                        max_increase=max(a-b for a,b in zip(ll,sl))))
    interactions={}
    for cond in ('all','cheap','expensive'):
        p=[v for v in pairs if cond=='all' or v['cond']==cond]
        interactions[cond]={metric:boot([x[metric] for x in p]) for metric in ('auc','attainment_area')}
    # Retain both navigation conditions together under each task/map-seed block.
    groups={}
    for p in pairs:groups.setdefault((p['task'],p['seed']),[]).append(p)
    interactions['18_task_seed_blocks']={k:boot([mean(p[k] for p in v) for v in groups.values()])
                                         for k in ('auc','attainment_area')}
    dose=[]
    for seedpath in sorted((RES/'craftworld_dose/d0.00').glob('*lava.task')):
        prev=None
        for density in ('0.00','0.01','0.02','0.03','0.04'):
            t=task(RES/f'craftworld_dose/d{density}'/seedpath.name)
            hazards={t['moves'][s][a] for s in range(t['N']) for a in range(4)
                     if t['events'][s][a]==t['A']-1}
            objects={(t['moves'][s][a],t['events'][s][a]) for s in range(t['N']) for a in range(4)
                     if 0<=t['events'][s][a]<t['A']-1}
            if prev:
                p,ph,po=prev
                dose.append(dict(tag=seedpath.stem,dose=density,nested=ph<=hazards,
                    same_moves=p['moves']==t['moves'],same_objects=po==objects,
                    same_starts=p['starts']==t['starts'],same_horizon=p['protocol'][0]==t['protocol'][0]))
            prev=t,hazards,objects
    (OUT/'recomputed.json').write_text(json.dumps(dict(cells=cells,interactions=interactions,
                        paired_effects=pairs,geometry=geo,dose_pairing=dose),indent=2)+'\n')
    print(json.dumps(dict(interactions=interactions,geometry=dict(maps=len(geo),
        maps_increased=sum(x['increased']>0 for x in geo),starts=sum(len(x['lava']) for x in geo),
        starts_increased=sum(x['increased'] for x in geo),max_increase=max(x['max_increase'] for x in geo)),
        dose_pairing={k:sum(not x[k] for x in dose) for k in ('nested','same_moves','same_objects','same_starts','same_horizon')}),indent=2))
    # Hash the directly relevant evidence, rather than environment caches.
    paths=[BASE/'src/compress.cpp',BASE/'compress',ROOT/'STEP39_PORTING.md']
    paths+=list((BASE/'src').glob('*.py'))+list((BASE/'tests').glob('*.py'))
    paths+=list(BASE.glob('REPORT*.md'))+list((BASE/'docs').glob('*.md'))
    paths+=list(RES.glob('*.json'))
    for folder in ('craftworld','craftworld_lava','craftworld_dose'):
        paths += [p for p in (RES/folder).rglob('*') if p.is_file()]
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))]
    (OUT/'current_snapshot.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
