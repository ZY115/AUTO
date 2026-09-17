"""Use the author's automaton interpreter (not simulator) to check export."""
import json, sys
from collections import deque
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'progressive_task_discovery/stage8_compression'
sys.path[:0]=[str(BASE/'src'),str(OUT)]
from recompute import task
from export_craftworld import hierarchy
from gym_hierarchical_subgoal_automata.automata.common import HierarchyState
from gym_hierarchical_subgoal_automata.automata.logic import TRUE

checks=[]; costs=[]
for m in json.loads((BASE/'results/craftworld/tasks.json').read_text()):
    t=task(BASE/f'results/craftworld/{m["tag"]}.task')
    h=hierarchy(m['task'],flat=True); a=h.get_root_automaton()
    states=sorted(a.get_states())
    assert a.get_initial_state()==states[0]
    count=0
    for q,s in enumerate(states):
        if a.is_terminal_state(s):continue
        for e in range(-1,t['A']):
            obs=set() if e<0 else {m['alphabet'][e]}
            actual=h.get_next_hierarchy_state(HierarchyState(s,a.get_name(),TRUE,[],[]),obs)
            expect=q if e<0 else t['trans'][q][e]
            assert actual.state_name==states[expect],(m['tag'],s,obs,actual,expect)
            count+=1
    checks.append(dict(tag=m['tag'],checked=count))
    # Reverse BFS seeded with the one-action costs of firing an event.
    rev=[[] for _ in range(t['N'])]
    for s in range(t['N']):
        for n in t['moves'][s]:rev[n].append(s)
    for e,label in enumerate(m['alphabet']):
        d=[None]*t['N'];queue=deque()
        for s in range(t['N']):
            if e in t['events'][s]:d[s]=1;queue.append(s)
        while queue:
            n=queue.popleft()
            for s in rev[n]:
                if d[s] is None:d[s]=d[n]+1;queue.append(s)
        C=sum(d[s] for s in t['starts'])/len(t['starts'])
        old=m['cost'][label][0]
        assert abs(C-old-1)<.0051
        costs.append(dict(tag=m['tag'],label=label,stored_C=old,correct_C=C))
(OUT/'structure_checks.json').write_text(json.dumps(dict(automaton_checks=checks,cost_checks=costs),indent=2)+'\n')
print('author-interpreter checks',sum(x['checked'] for x in checks),'all match')
print('event-cost checks',len(costs),'all stored values underestimate by exactly one action before rounding')
