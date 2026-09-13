"""首访诊断：C-full 在第一次进入一个后续任务状态时，会不会对一个已经通过广播学会的
目标重新大量随机探索，而 Y11 因为累计了该目标的执行经验、探索率已经降下来了？

这条通道是真实存在的，来源明确：`_get_formula_exploration_rate` 按
`_get_formula_q_function_step_count` 退火，而那个计数走 `_get_policy_bank`，
即**当前状态的库**。Y11 把一个子目标在所有状态下的执行次数累加成一个计数，
C-full 分状态计。

**但「18 个库执行计数为零」本身不是问题。** 如果短跑只真正走到过两个任务状态，
另外那些库没有执行计数就是预期行为，不能解释成「探索坏掉了」。真正要量的是上面
那个首访问题——这个脚本量它，不需要再加新臂。

每次首访记录四项：

    目标        这一步在追的子目标
    ε           当时用的探索率
    已学量      该目标在这个库里累计的 Q 值写入次数（广播带来的「已经学会」的量）
    随后探索数  首访后 N 步里实际抽到随机动作的次数

关键对照是同一个 (状态, 目标) 首访时 Y11 与 C-full 的 ε：如果 C-full 明显更高、
而「已学量」又不低，那就是「学会了却还在乱走」，是共享真正买到的东西之一。

    $HRM_PYTHON tools/first_visit_exploration.py [跟踪步数]
"""
import json, os, sys, tempfile
from collections import defaultdict
from pathlib import Path

HRM = Path(os.environ.get('HRM_LEARNING', '')).expanduser()
BASE = 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'
FOLLOW = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def instrument(alg_cls, cfg, tag, out):
    sys.path.insert(0, str(HRM / 'src'))
    from reinforcement_learning.ihsa_hrl_algorithm import IHSAAlgorithmHRL
    from reinforcement_learning.learning_algorithm import LearningAlgorithm

    seen, pending = {}, []
    o_rate = IHSAAlgorithmHRL._get_formula_exploration_rate
    o_act = LearningAlgorithm._choose_egreedy_action
    o_choose = IHSAAlgorithmHRL._choose_action

    def choose(self, domain_id, task_id, state, hierarchy, hierarchy_state):
        self._fv_key = (hierarchy_state.automaton_name, hierarchy_state.state_name)
        return o_choose(self, domain_id, task_id, state, hierarchy, hierarchy_state)

    def rate(self, task_id, fc):
        r = o_rate(self, task_id, fc)
        key = (getattr(self, '_fv_key', None), str(fc))
        if key[0] is not None and key not in seen:
            bank = self._get_policy_bank(task_id)
            try:
                root = bank.get_root(fc).get_formula_condition()
                learned = bank._q_function_update_counter.get(root, 0)
            except Exception:
                learned = -1
            seen[key] = dict(state=str(key[0]), goal=str(fc)[:40], eps=round(r, 6),
                             learned=learned, explored=0, steps=0)
            pending.append(seen[key])
        return r

    def act(self, task, state, q_function, epsilon):
        before = None
        if self.training_enable:
            import numpy as _np
            before = _np.random.get_state()
        a = o_act(self, task, state, q_function, epsilon)
        if self.training_enable:
            import numpy as _np
            _np.random.set_state(before)
            explored = _np.random.uniform(0, 1) <= epsilon
            _np.random.set_state(before)
            o_act(self, task, state, q_function, epsilon)   # 重放，保持流不变
            for rec in list(pending):
                rec['steps'] += 1
                rec['explored'] += int(explored)
                if rec['steps'] >= FOLLOW:
                    pending.remove(rec)
        return a

    IHSAAlgorithmHRL._choose_action = choose
    IHSAAlgorithmHRL._get_formula_exploration_rate = rate
    LearningAlgorithm._choose_egreedy_action = act
    try:
        alg_cls(dict(cfg, folder_name=str(out), checkpoint_folder=str(out))).run()
    finally:
        IHSAAlgorithmHRL._choose_action = o_choose
        IHSAAlgorithmHRL._get_formula_exploration_rate = o_rate
        LearningAlgorithm._choose_egreedy_action = o_act
    return seen


def main():
    if not HRM.is_dir():
        sys.exit('请先设置 HRM_LEARNING')
    os.chdir(HRM); sys.path.insert(0, 'src')
    from ilasp.ilasp_common import set_ilasp_env_variables
    set_ilasp_env_variables('src')
    from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular
    from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
        IHSAAlgorithmHRLTabularCrossState)

    cfg = json.loads((HRM / BASE).read_text())
    cfg.update(debug=False, num_episodes=600, state_format='tabular', use_gpu=False)
    w = Path(tempfile.mkdtemp(prefix='firstvisit_'))

    y11 = instrument(IHSAAlgorithmHRLTabular, cfg, 'Y11', w / 'y11')
    cfull = instrument(IHSAAlgorithmHRLTabularCrossState,
                       dict(cfg, algorithm='ihsa-hrl-crossstate'), 'Cfull', w / 'cf')

    print(f'跟踪窗口 {FOLLOW} 步。首访事件：Y11 {len(y11)} 个，C-full {len(cfull)} 个\n')
    common = sorted(set(y11) & set(cfull))
    if not common:
        print('两臂没有共同的 (状态, 目标) 首访——把 num_episodes 调大再跑。'); return
    print(f'{"任务状态":<22}{"目标":<16}'
          f'{"Y11 ε":>9}{"Cf ε":>9}{"Cf/Y11":>8}'
          f'{"Y11 已学":>9}{"Cf 已学":>9}{"Y11 探索":>9}{"Cf 探索":>8}')
    worse = 0
    for k in common:
        a, b = y11[k], cfull[k]
        ratio = b['eps'] / a['eps'] if a['eps'] else float('nan')
        worse += ratio > 1.01
        print(f'{a["state"][:21]:<22}{a["goal"][:15]:<16}'
              f'{a["eps"]:>9.4f}{b["eps"]:>9.4f}{ratio:>7.2f}x'
              f'{a["learned"]:>9,}{b["learned"]:>9,}'
              f'{a["explored"]:>9}{b["explored"]:>8}')
    print(f'\n{worse}/{len(common)} 个首访上 C-full 的探索率高于 Y11。')
    print('读法：如果 C-full 的「已学」不低（广播确实把这个目标学过了）而 ε 明显更高，')
    print('     那就是「学会了却还在乱走」——共享真正买到的东西之一，正是这个。')
    print('     反过来，如果 ε 相近，这条通道在当前规模下就不是主要解释。')


if __name__ == '__main__':
    main()
