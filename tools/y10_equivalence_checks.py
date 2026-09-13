"""Checks 2 and 3: the per-step update budget and the pseudoreward stream must be
identical between Y11 and Y10; only where the update lands may differ."""
import json, os, sys
from collections import Counter
sys.path.insert(0, 'src')
os.environ.setdefault("PYTHONHASHSEED", "0")
from ilasp.ilasp_common import set_ilasp_env_variables
set_ilasp_env_variables('src')
from reinforcement_learning.ihsa_hrl_tabular_algorithm import FormulaBankTabular, IHSAAlgorithmHRLTabular
from reinforcement_learning.ihsa_hrl_tabular_perstate_algorithm import IHSAAlgorithmHRLTabularPerState

TRACE = {}


def instrument(tag):
    TRACE[tag] = {'updates': [], 'rewards': Counter(), 'banks': set()}
    orig_sel = FormulaBankTabular._get_subgoals_to_update
    orig_pr = FormulaBankTabular._get_subgoal_pseudoreward

    def sel(self):
        out = orig_sel(self)
        TRACE[tag]['updates'].append(len(out))
        TRACE[tag]['banks'].add(id(self))
        return out

    def pr(self, formula, observation, is_terminal, is_goal_achieved):
        r, t = orig_pr(self, formula, observation, is_terminal, is_goal_achieved)
        TRACE[tag]['rewards'][(round(r, 6), t)] += 1
        return r, t

    FormulaBankTabular._get_subgoals_to_update = sel
    FormulaBankTabular._get_subgoal_pseudoreward = pr
    return orig_sel, orig_pr


def run(tag, cfg, cls):
    o1, o2 = instrument(tag)
    alg = cls(json.load(open(cfg)))
    alg.run()
    FormulaBankTabular._get_subgoals_to_update = o1
    FormulaBankTabular._get_subgoal_pseudoreward = o2
    return alg


SP = os.path.dirname(os.path.abspath(__file__))
a = run('Y11', f'{SP}/y11tab/config.json', IHSAAlgorithmHRLTabular)
b = run('Y10', f'{SP}/y10tab/config.json', IHSAAlgorithmHRLTabularPerState)

for tag in ('Y11', 'Y10'):
    t = TRACE[tag]
    print(f'{tag}: 更新调用 {len(t["updates"])} 次，每次采样个数分布 '
          f'{dict(Counter(t["updates"]))}，涉及 bank 实例 {len(t["banks"])} 个')
# 比的是「每次调用采样几个子目标」，不是「总共调用了几次」——后者等于各自走了多少步，
# 两臂策略不同自然不同，那是结果不是预算。
s11, s10 = set(TRACE["Y11"]["updates"]), set(TRACE["Y10"]["updates"])
print(f'\n检查二 每步采样的子目标个数相同: '
      f'{"通过" if s11 == s10 and len(s11) == 1 else "不同"}  Y11 {s11}  Y10 {s10}')
print(f'  （调用总次数 {len(TRACE["Y11"]["updates"])} vs {len(TRACE["Y10"]["updates"])} '
      f'只反映两臂各走了多少步）')
print(f'检查三 伪奖励取值分布相同: '
      f'{"通过" if set(TRACE["Y11"]["rewards"]) == set(TRACE["Y10"]["rewards"]) else "不同"}')
print(f'  Y11 伪奖励取值 {sorted(TRACE["Y11"]["rewards"])}')
print(f'  Y10 伪奖励取值 {sorted(TRACE["Y10"]["rewards"])}')
print(f'\nY10 实际建了 {b.num_perstate_banks()} 份按状态分开的 bank')
