"""Search static multi-instance layouts using exact product-graph shortest paths.
No training data or learned policy enters map selection.
"""
from collections import deque
from functools import lru_cache
from pathlib import Path
import hashlib,itertools,json,random,time
ROOT=Path(__file__).resolve().parents[1]
DIRS=[(0,-1),(1,0),(0,1),(-1,0)]

def build(n,L,seed,side=9):
    mid=side//2;room={(x,y) for x in range(side) for y in range(side)}
    J=(side,mid);hub=(side+4,mid)
    cells=room|{(x,mid) for x in range(side,side+9)}|{(hub[0],y) for y in range(mid-4,mid+5)}
    cells=sorted(cells);index={p:i for i,p in enumerate(cells)};rng=random.Random(seed)
    locations=rng.sample(sorted(room-{(side-1,mid)}),n*L)
    labels={p:g for g in range(n) for p in locations[g*L:(g+1)*L]}
    labels.update({(hub[0],mid-4):n,(hub[0],mid+4):n+1,(side+8,mid):n+2})
    moves=[]
    for p in cells:
        row=[]
        for dx,dy in DIRS:
            z=(p[0]+dx,p[1]+dy)
            if z not in index:z=p
            row.append([index[z],labels.get(z,-1) if z!=p else -1])
        moves.append(row)
    return dict(n=n,instances=L,seed=seed,side=side,cells=cells,labels=[[index[p],g] for p,g in sorted(labels.items())],moves=moves,junction=index[J],room=[index[p] for p in sorted(room)],candidate_starts=[index[p] for p in sorted(room) if p not in labels])

def nextq(q,event,n,suffix):
    full=(1<<n)-1
    if q<full:return q|(1<<event) if 0<=event<n else q
    j=q-full
    return q+1 if j<len(suffix) and event==suffix[j] else q

def product(layout,suffix):
    n=layout['n'];N=len(layout['cells']);terminal=(1<<n)-1+len(suffix)
    rev=[[] for _ in range((terminal+1)*N)]
    for q in range(terminal):
        for s,row in enumerate(layout['moves']):
            for ns,event in row:rev[nextq(q,event,n,suffix)*N+ns].append(q*N+s)
    dist=[-1]*len(rev);queue=deque()
    for s in range(N):dist[terminal*N+s]=0;queue.append(terminal*N+s)
    while queue:
        z=queue.popleft()
        for prev in rev[z]:
            if dist[prev]<0:dist[prev]=dist[z]+1;queue.append(prev)
    @lru_cache(None)
    def histories(z):
        q,s=divmod(z,N)
        if q==terminal:return frozenset([()])
        out=set()
        for ns,event in layout['moves'][s]:
            nq=nextq(q,event,n,suffix);dest=nq*N+ns
            if dist[dest]==dist[z]-1:
                for rest in histories(dest):out.add(((event,) if nq!=q else ())+rest)
        return frozenset(out)
    return dist,histories

def summarize(layout,m):
    n=layout['n'];suffix=[n+i%3 for i in range(m)]
    dist,histories=product(layout,suffix);N=len(layout['cells']);by_order={}
    for start in layout['candidate_starts']:
        hs=histories(start)
        if len(hs)==1:
            history=next(iter(hs));order=history[:n]
            # Prefer a short unique-order start; selection is physical, not based on RL.
            if order not in by_order or (dist[start],start)<(by_order[order]['length'],by_order[order]['start']):
                by_order[order]=dict(start=start,length=dist[start],history=history)
    prefixes={()};states={0}
    for r in by_order.values():
        q=0
        for t,e in enumerate(r['history']):q=nextq(q,e,n,suffix);prefixes.add(r['history'][:t+1]);states.add(q)
    return dict(m=m,suffix=suffix,starts=[r['start'] for _,r in sorted(by_order.items())],optimal_histories=[r['history'] for _,r in sorted(by_order.items())],optimal_lengths=[r['length'] for _,r in sorted(by_order.items())],orders=len(by_order),history_states=len(prefixes),task_states=len(states),compression=len(prefixes)/len(states),history_nonterminal=len(prefixes)-len(by_order),task_nonterminal=len(states)-1)

def funnel_check(layout):
    # J is a cut vertex separating every event instance in the room from all suffix cells.
    J=layout['junction'];suffix={s for s,g in layout['labels'] if g>=layout['n']}
    root=layout['room'][0];seen={root};queue=deque([root])
    while queue:
        s=queue.popleft()
        for ns,_ in layout['moves'][s]:
            if ns!=J and ns not in seen:seen.add(ns);queue.append(ns)
    return not bool(seen&suffix)

def main():
    start=time.monotonic();records=[];best={};goals=[1.2,2,3,4,6]
    configs=[(2,L) for L in [1,2,3]]+[(3,L) for L in [1,2,3,4]]+[(4,L) for L in [2,3,4,6,8]]
    # Finite, reproducible search. A missed bin is not a global impossibility proof.
    for ci,(n,L) in enumerate(configs):
        for rep in range(30):
            seed=81000+ci*100+rep;layout=build(n,L,seed)
            assert funnel_check(layout)
            # All suffix choices add a constant after the first suffix label. Search with m8;
            # chosen layouts are independently re-evaluated at their actual m before use.
            base=summarize(layout,8)
            for m in [1,4,8]:
                histories=[tuple(h[:n])+tuple(n+i%3 for i in range(m)) for h in base['optimal_histories']]
                prefixes={()};states={0};suffix=[n+i%3 for i in range(m)]
                for h in histories:
                    q=0
                    for t,e in enumerate(h):q=nextq(q,e,n,suffix);prefixes.add(h[:t+1]);states.add(q)
                ratio=len(prefixes)/len(states)
                rec=dict(n=n,L=L,seed=seed,m=m,orders=base['orders'],history_states=len(prefixes),task_states=len(states),compression=ratio,max_length=max(base['optimal_lengths'],default=0)-8*(8-m),funnel=True)
                records.append(rec)
                if base['orders']<2:continue
                for target in goals:
                    # Relative +/-10% target bin. Prefer lower physical length, then n and seed.
                    if abs(ratio-target)<=.1*target:
                        score=(rec['max_length'],n,abs(ratio-target),seed)
                        if target not in best or score<best[target][0]:best[target]=(score,layout,rec)
        print('searched',n,L,'candidates',len(records),'bins',sorted(best),'seconds',round(time.monotonic()-start,1),flush=True)
    out=ROOT/'results/map_search';out.mkdir(exist_ok=False)
    selected=[]
    for target,(_,layout,rec) in sorted(best.items()):
        actual=summarize(layout,rec['m']);assert abs(actual['compression']-rec['compression'])<1e-12
        assert actual['orders']==rec['orders']
        item={**layout,**actual,'target_bin':target,'funnel_verified':True}
        path=out/f'layout_{target:g}.json';path.write_text(json.dumps(item,indent=2)+'\n');selected.append(dict(target=target,path=str(path),n=layout['n'],instances=layout['instances'],**actual))
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'summary.json').write_text(json.dumps(dict(candidates=len(records),layouts=len(configs)*30,selected=selected,unreached_bins=[x for x in goals if x not in best],max_compression=max(r['compression'] for r in records),seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':main()
