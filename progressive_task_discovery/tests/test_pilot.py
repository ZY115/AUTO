import json
from pathlib import Path
import subprocess
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(method='count', **kwargs):
    flags = dict(method=method, k=4, seed=17, budget=12000, every=2000)
    flags.update(kwargs)
    cmd = [str(ROOT / 'pilot')]
    for key, value in flags.items():
        cmd += ['--'+key.replace('_', '-'), str(value)]
    return json.loads(subprocess.check_output(cmd, text=True))


def independent_evaluation(result):
    """Backward DP, independently constructed geometry and task-transition lookup.

    Unlike the C++ forward occupancy evaluator, V[h,q,y,x] is completion
    probability with h remaining steps. No simulator functions are imported.
    """
    targets = result['task']
    k, H, noise = len(targets), result['horizon'], result['slip']
    coordinates = [(0,0),(4,0),(8,0),(0,4),(8,4),(0,8),(4,8),(8,8)]
    label = {y*9+x:i for i,(x,y) in enumerate(coordinates)}
    policy = np.array(result['final_policy']).reshape(k,81)
    dest = np.zeros((81,4), dtype=int)
    emit = np.full((81,4), -1, dtype=int)
    for s in range(81):
        y,x = divmod(s,9)
        for a,(dx,dy) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):
            z = int(np.clip(y+dy,0,8))*9 + int(np.clip(x+dx,0,8))
            dest[s,a] = z
            if z != s:
                emit[s,a] = label.get(z,-1)
    value = np.zeros((k+1,81)); value[k] = 1
    time_value = np.zeros((k+1,81))
    for _ in range(H):
        new = np.zeros_like(value); new[k] = 1
        new_time = np.zeros_like(time_value)
        for q in range(k):
            for a in range(4):
                nxt = q+(emit[:,a] == targets[q]).astype(int)
                p = noise/4 + (1-noise)*(policy[q] == a)
                new[q] += p*value[nxt,dest[:,a]]
                new_time[q] += p*(1+time_value[nxt,dest[:,a]])
        value,time_value = new,new_time
    return float(value[0,40]),float(time_value[0,40])


class PilotTests(unittest.TestCase):
    def test_cpp_semantics(self):
        data = json.loads(subprocess.check_output([str(ROOT/'pilot'),'--self-test'],text=True))
        self.assertEqual(data['cpp_self_tests'],'passed')

    def test_equivalent_state_interfaces(self):
        for noise in [0,.2]:
            r = [run(m, slip=noise) for m in ['count','discovered','oracle_state']]
            self.assertEqual(len({x['trace_hash'] for x in r}),1)
            self.assertEqual(len({x['q_hash'] for x in r}),1)
            self.assertEqual(r[0]['checkpoints'],r[1]['checkpoints'])

    def test_count_scheduler_needs_no_edge_labels(self):
        for noise in [0,.2]:
            a,b = [run(m,slip=noise) for m in ['progressive','count_frontier']]
            self.assertEqual(a['trace_hash'],b['trace_hash'])
            self.assertEqual(a['q_hash'],b['q_hash'])
            self.assertTrue(all(x == -1 for x in b['learned_edges']))

    def test_independent_exact_evaluator(self):
        for method in ['flat','count','progressive_qrm','oracle_qrm','goal_reuse','progressive_goal','count_replay','option_qrm']:
            for noise in [0,.15]:
                r = run(method, k=2, slip=noise, hfactor=12, budget=20000)
                success,steps = independent_evaluation(r)
                self.assertAlmostEqual(success,r['final_success'],places=9)
                self.assertAlmostEqual(steps,r['final_eval_steps'],places=8)

    def test_budget_discovery_and_reproducibility(self):
        a,b = [run('learned_qrm',budget=12345) for _ in range(2)]
        self.assertEqual(a,b)
        self.assertEqual(sum(a['stage_steps']),12345)
        self.assertEqual(a['checkpoints'][-1][0],12345)
        self.assertGreaterEqual(a['updates'],12345)
        for event,true,step in zip(a['learned_edges'],a['task'],a['discovery_steps']):
            self.assertTrue(event == -1 or event == true)
            self.assertEqual(event == -1,step == 0)

    def test_equal_compute_control(self):
        for method in ['goal_reuse','count_replay','option_replay']:
            r=run(method)
            self.assertEqual(r['updates'],9*r['budget'])

    def test_behavior_regression_against_stage1_snapshot(self):
        source=ROOT/'results/stage1_initial/raw.jsonl'
        if not source.exists():
            self.skipTest('No historical fixture')
        records=[json.loads(x) for x in source.read_text().splitlines()]
        for old in records:
            if old['seed']!=0 or old['k']!=4:
                continue
            new=run(old['method'],k=4,seed=0,budget=old['budget'],every=5000)
            self.assertEqual(new['q_hash'],old['q_hash'])
            self.assertEqual(new['trace_hash'],old['trace_hash'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
