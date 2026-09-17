"""Checks 2 and 3 for the neural arm, plus two facts specific to it: that the
per-state banks really do share one replay buffer, and that gradient updates
actually happened (with er_start_size at its published 100000 a short run makes
no updates at all, so the check would pass vacuously)."""
import json, os, sys
from collections import Counter
sys.path.insert(0, 'src')
from ilasp.ilasp_common import set_ilasp_env_variables
set_ilasp_env_variables('src')
from reinforcement_learning.ihsa_hrl_dqn_algorithm import FormulaBankDQN, IHSAAlgorithmHRLDQN
from reinforcement_learning.ihsa_hrl_dqn_perstate_algorithm import IHSAAlgorithmHRLDQNPerState

T = {}


def run(tag, cfg, cls):
    T[tag] = {'sel': Counter(), 'rew': Counter(), 'grad': 0, 'banks': set(), 'bufs': set()}
    o_sel = FormulaBankDQN._get_subgoals_to_update
    o_pr = FormulaBankDQN._get_subgoal_pseudoreward
    o_up = FormulaBankDQN._update_q_functions_helper

    def sel(self):
        out = o_sel(self)
        T[tag]['sel'][len(out)] += 1
        T[tag]['banks'].add(id(self)); T[tag]['bufs'].add(id(self._er_buffer))
        return out

    def pr(self, f, o, t, g):
        r, term = o_pr(self, f, o, t, g)
        T[tag]['rew'][(round(r, 6), term)] += 1
        return r, term

    def up(self, batch):
        T[tag]['grad'] += 1
        return o_up(self, batch)

    FormulaBankDQN._get_subgoals_to_update = sel
    FormulaBankDQN._get_subgoal_pseudoreward = pr
    FormulaBankDQN._update_q_functions_helper = up
    alg = cls(json.load(open(cfg)))
    alg.run()
    FormulaBankDQN._get_subgoals_to_update = o_sel
    FormulaBankDQN._get_subgoal_pseudoreward = o_pr
    FormulaBankDQN._update_q_functions_helper = o_up
    return alg


SP = os.path.dirname(os.path.abspath(__file__))
run('Y11', f'{SP}/y11dqn/config.json', IHSAAlgorithmHRLDQN)
b = run('Y10', f'{SP}/y10dqn/config.json', IHSAAlgorithmHRLDQNPerState)

for tag in ('Y11', 'Y10'):
    t = T[tag]
    print(f'{tag}: 每次采样个数 {dict(t["sel"])}  梯度更新 {t["grad"]} 次  '
          f'bank 实例 {len(t["banks"])}  缓冲实例 {len(t["bufs"])}')
print(f'\n检查二 每步采样的子目标个数相同: '
      f'{"通过" if set(T["Y11"]["sel"]) == set(T["Y10"]["sel"]) else "不同"}'
      f'  Y11 {set(T["Y11"]["sel"])}  Y10 {set(T["Y10"]["sel"])}')
print(f'检查三 伪奖励取值相同: '
      f'{"通过" if set(T["Y11"]["rew"]) == set(T["Y10"]["rew"]) else "不同"}  {sorted(T["Y10"]["rew"])}')
print(f'\n梯度更新确实发生: Y11 {T["Y11"]["grad"]} 次，Y10 {T["Y10"]["grad"]} 次')
print(f'按状态分开的 bank {b.num_perstate_banks()} 份，共享缓冲 {b.shares_buffer()}，'
      f'实际缓冲对象数 {len(T["Y10"]["bufs"])}（应为 1）')
