"""Exhaustive unique-optimal-order coverage for static single-cell events."""
from pathlib import Path
from collections import deque
from functools import lru_cache
from itertools import permutations
import json, math
ROOT=Path(__file__).resolve().parents[1]

def coverage(cells,goals,terminal):
    cells=frozenset(cells);goals=tuple(goals)
    @lru_cache(None)
    def distance(start,target,blocked):
        q=deque([(start,0)]);seen={start}
        while q:
            s,d=q.popleft()
            if s==target:return d
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                z=s[0]+dx,s[1]+dy
                if z in cells and z not in blocked and z not in seen:seen.add(z);q.append((z,d+1))
        return math.inf
    orders=list(permutations(range(len(goals))))
    evidence=[];unique=set()
    for start in sorted(cells-set(goals)-{terminal}):
        costs=[]
        for order in orders:
            at=start;cost=0;remaining=set(goals)
            for i in order:
                target=goals[i];remaining.remove(target)
                cost+=distance(at,target,frozenset(remaining));at=target
            cost+=distance(at,terminal,frozenset())
            costs.append(cost)
        best=min(costs);winners=[orders[i] for i,c in enumerate(costs) if c==best and c<math.inf]
        if len(winners)==1:unique.add(winners[0])
        evidence.append(dict(start=start,length=best,optimal_orders=winners))
    assert len(unique)<=len(goals), 'Would contradict the first-event continuation theorem'
    return dict(n=len(goals),walkable=len(cells),goals=goals,terminal=terminal,required=math.factorial(len(goals)),unique_orders=sorted(unique),covered=len(unique),first_event_upper_bound=len(goals),all_starts=evidence)

def main():
    old={(x,1) for x in range(4,9)}|{(6,y) for y in range(1,13)}|{(x,6) for x in range(13)}
    examples=[coverage(old,[(5,1),(7,1)],(6,12)),coverage(old,[(5,1),(7,1),(0,6)],(6,12)),coverage({(x,y) for x in range(13) for y in range(13)},[(1,1),(10,2),(4,10)],(11,11))]
    out=dict(assumptions=['static deterministic physical state','one fixed cell per event','unordered events advance when first encountered','fixed suffix after all unordered events','only initial position varies','unique optimal order at each counted start'],examples=examples,n3_requirement_possible=False,n4_requirement_possible=False)
    (ROOT/'results/map_feasibility.json').write_text(json.dumps(out,indent=2)+'\n')
    for e in examples:print('n',e['n'],'cells',e['walkable'],'unique orders',e['covered'],'required',e['required'],'upper bound',e['first_event_upper_bound'])

if __name__=='__main__':main()
