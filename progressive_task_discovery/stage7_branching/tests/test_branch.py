import json
from collections import deque
from pathlib import Path
import subprocess
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CELLS=[(5,1),(7,1),(0,6),(12,6),(6,12)]
START=[(4,1),(8,1)]
VALID={(x,1) for x in range(4,9)}|{(6,y) for y in range(1,13)}|{(x,6) for x in range(13)}


def transition(position,action):
    dx,dy=[(0,-1),(1,0),(0,1),(-1,0)][action]
    dest=(position[0]+dx,position[1]+dy)
    if dest not in VALID:dest=position
    event=CELLS.index(dest) if dest!=position and dest in CELLS else -1
    return dest,event


def run(method,**kwargs):
    flags=dict(method=method,k=4,seed=19,budget=30000,every=1000)
    flags.update(kwargs)
    cmd=[str(ROOT/'branch')]
    for k,v in flags.items():cmd+=['--'+k.replace('_','-'),str(v)]
    return json.loads(subprocess.check_output(cmd,text=True))


def independent_rollouts(row):
    pol=np.array(row['final_policy']).reshape(2,row['k'],13,13)
    outcomes=[]
    for branch,start in enumerate(START):
        position=start;events=[];d=0;done=False
        for t in range(row['horizon']):
            action=pol[branch,d,position[1],position[0]]
            position,event=transition(position,action)
            events.append(event)
            # Independent ordered-subsequence parser, rescanned from raw event trace.
            j=0
            for e in events:
                if e==row['task'][branch][j]:
                    j+=1
                    if j==row['k']:break
            d=j
            if d==row['k']:done=True;break
        outcomes.append((int(done),t+1))
    return outcomes


def stochastic_evaluation(row,epsilon):
    """Backward DP, independent of C++ forward occupancy implementation."""
    k=row['k'];policy=np.array(row['final_policy']).reshape(2,k,169)
    states=sorted(VALID);sids=np.array([y*13+x for x,y in states])
    dest=np.zeros((len(states),4),dtype=int);events=np.full_like(dest,-1)
    for i,p in enumerate(states):
        for a in range(4):
            z,e=transition(p,a);dest[i,a]=z[1]*13+z[0];events[i,a]=e
    success=[]
    for b in range(2):
        val=np.zeros((k+1,169));val[k]=1
        for t in range(row['horizon']):
            nxt=np.zeros_like(val);nxt[k]=1
            for d in range(k):
                for a in range(4):
                    nd=d+(events[:,a]==row['task'][b][d])
                    p=epsilon/4+(1-epsilon)*(policy[b,d,sids]==a)
                    nxt[d,sids]+=p*val[nd,dest[:,a]]
            val=nxt
        x,y=START[b];success.append(val[0,y*13+x])
    return sum(success)/2


def shortest_route(row,branch):
    start=START[branch];queue=deque([(start,0,0)]);seen={(start,0)}
    while queue:
        pos,d,cost=queue.popleft()
        for a in range(4):
            z,event=transition(pos,a);nd=d+int(event==row['task'][branch][d])
            if nd==row['k']:return cost+1
            if (z,nd) not in seen:seen.add((z,nd));queue.append((z,nd,cost+1))
    raise AssertionError('unreachable task')


class BranchTests(unittest.TestCase):
    def test_cpp_checks_and_geometry(self):
        subprocess.check_call([str(ROOT/'branch'),'--self-test'],stdout=subprocess.DEVNULL)
        data=json.loads(subprocess.check_output([str(ROOT/'branch'),'--map'],text=True))
        self.assertEqual(len(data),len(VALID))
        for row in data:
            s=row[0];pos=(s%13,s//13)
            for a in range(4):
                dest,event=transition(pos,a)
                self.assertEqual(row[1+2*a:3+2*a],[dest[1]*13+dest[0],event])

    def test_mandatory_ambiguity_witness(self):
        results=[]
        for b,start in enumerate(START):
            pos=start;events=[]
            for a in ([1,1] if b==0 else [3,3])+[2]*5:
                pos,e=transition(pos,a)
                if e>=0:events.append(e)
            results.append((pos,events))
        self.assertEqual(results,[((6,6),[0]),((6,6),[1])])
        # Removing the central junction disconnects the three goal arms.
        reachable={START[0]};q=deque(reachable)
        while q:
            p=q.popleft()
            for a in range(4):
                z,_=transition(p,a)
                if z!=(6,6) and z not in reachable:reachable.add(z);q.append(z)
        self.assertTrue(all(g not in reachable for g in CELLS[2:]))

    def test_representation_equivalence(self):
        for a,b in [('history_replay','learned_replay'),('history_skill','immediate_skill')]:
            x,y=run(a),run(b)
            self.assertEqual(x['trace_hash'],y['trace_hash'])
            self.assertEqual(x['q_hash'],y['q_hash'])
            self.assertEqual(x['checkpoints'],y['checkpoints'])

    def test_independent_policy_and_shortest_paths(self):
        for method in ['count_replay','history_replay','hand_replay','delayed_skill','immediate_skill','oracle_skill']:
            row=run(method,k=8)
            outcome=independent_rollouts(row)
            self.assertEqual([x[0] for x in outcome],row['final_branch_success'])
            self.assertEqual([x[1] for x in outcome],row['final_eval_steps'])
            for b in range(2):self.assertEqual(shortest_route(row,b),12*row['k']-11)
            if method=='count_replay':self.assertLessEqual(row['final_success'],.5)

    def test_stochastic_evaluation(self):
        row=run('count_replay')
        for eps,name in [(.05,'stochastic_final_success_e005'),(.2,'stochastic_final_success_e02')]:
            self.assertAlmostEqual(stochastic_evaluation(row,eps),row[name],places=9)

    def test_budgets_and_exact_structure(self):
        for method in ['count_replay','history_replay','immediate_skill','delayed_skill']:
            r=run(method,budget=20357)
            self.assertEqual(r['updates'],6*r['budget'])
            self.assertEqual(r['prefix_steps']+r['frontier_steps'],r['budget'])
            self.assertEqual(r['wrong_binding_steps'],0)
            if method=='count_replay':self.assertEqual(r['junction_state_accuracy'],.5)
            else:self.assertEqual(r['junction_state_accuracy'],1)
            r2=run(method,budget=20357)
            self.assertEqual(r,r2)

    def test_shared_bridge_and_last_event_alias(self):
        for bridge in [1,3]:
            row=run('last_event_replay',k=8,bridge=bridge)
            decision=bridge+1
            self.assertEqual(row['task'][0][1:decision],row['task'][1][1:decision])
            self.assertNotEqual(row['task'][0][decision],row['task'][1][decision])
            self.assertEqual(row['junction_state_accuracy'],.5)
            self.assertLessEqual(row['final_success'],.5)
            other=run('immediate_skill',k=8,bridge=bridge)
            self.assertEqual(other['junction_state_accuracy'],1)
            self.assertEqual([x[0] for x in independent_rollouts(other)],other['final_branch_success'])
            for b in range(2):self.assertEqual(shortest_route(other,b),12*other['k']-11)

    def test_same_map_no_ambiguity_control(self):
        a=run('count_replay',branching=0,bridge=1,k=8,epsilon=.005,budget=60000)
        b=run('hand_replay',branching=0,bridge=1,k=8,epsilon=.005,budget=60000)
        self.assertEqual(a['task'][0][1:],a['task'][1][1:])
        self.assertEqual(a['q_hash'],b['q_hash'])
        self.assertEqual(a['trace_hash'],b['trace_hash'])
        self.assertEqual(a['final_success'],1)
        self.assertEqual(a['junction_state_accuracy'],1)


if __name__=='__main__':unittest.main(verbosity=2)
