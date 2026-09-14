"""第 2 步：让 crossstate_shared_sampling 具备做单因素干预的资格。

在这之前它不具备。两个原因：

**一、更新采样与行为探索共用 NumPy 全局流。** 改了抽哪些子目标，就会平移之后
每一次探索用的随机数，于是"差距变了"分不清是采样造成的还是随机数错位造成的。
`separate_update_rng` 给每个库自己的生成器，把全局流留给行为策略。

**二、探索日程本身两臂就不同。** ε 按**当前库**的执行计数退火，Y11 合并计、
C-full 分状态计，所以即使 Q 表一样，ε 也不一样。`subgoal_exploration_rate_const`
把 ε 钉成常数，这条通道就关掉了。

两件都做到之后，才能问：**在只剩"各库独立采样"这一个差异时，共用采样是否让
两臂逐步一致？** 注意比的是**每一步的动作**，不是回合摘要——摘要一致是弱得多的
证据，之前已经吃过一次亏。

    S1  分离更新流后，默认配置的结果不变（不是静默改了行为）
    S2  分离更新流确实把更新采样移出了全局流
    S3  固定 ε + 分离更新流 + 共用采样 → 两臂逐步动作完全一致
    S4  同上但关掉共用采样 → 两臂逐步动作不一致（说明 S3 不是平凡通过）

S3 通过，开关才算单因素。S3 不通过，说明除了采样还有别的东西在动，
**那时不能去跑机制干预**，得先找出那个东西。
"""
import hashlib, json, os, sys, tempfile
from pathlib import Path

HRM = Path(os.environ.get('HRM_LEARNING', '')).expanduser()
BASE = 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'
OK, BAD = '  [通过] ', '  [失败] '
fails = []


def check(label, cond, detail=''):
    print((OK if cond else BAD) + label + (('  ' + detail) if detail else ''))
    if not cond:
        fails.append(label)
    return cond


ACTIONS = []


def build(cls, over, out, episodes=120):
    d = json.loads((HRM / BASE).read_text())
    d.update(debug=False, num_episodes=episodes, state_format='tabular',
             use_gpu=False, seed=25101993)
    d['environments'] = [dict(d['environments'][0], name='book')]
    d['grid_params'] = dict(d['grid_params'], use_lava=True)
    d['neutralize_deadends'] = False
    d.update(over)
    out.mkdir(parents=True, exist_ok=True)
    d['folder_name'] = str(out); d['checkpoint_folder'] = str(out)
    return cls(d)


def run_recording(cls, over, out, episodes=120):
    """跑一次，记录每一步真正选中的原始动作。"""
    from reinforcement_learning.learning_algorithm import LearningAlgorithm
    seq = []
    orig = LearningAlgorithm._choose_egreedy_action

    def rec(self, task, state, q_function, epsilon):
        a = orig(self, task, state, q_function, epsilon)
        if self.training_enable:
            seq.append(int(a))
        return a

    LearningAlgorithm._choose_egreedy_action = rec
    try:
        build(cls, over, out, episodes).run()
    finally:
        LearningAlgorithm._choose_egreedy_action = orig
    return seq


def digest(seq):
    return hashlib.md5(bytes(bytearray(x % 256 for x in seq))).hexdigest()[:12]


def main():
    if not HRM.is_dir():
        sys.exit('请先设置 HRM_LEARNING')
    os.chdir(HRM); sys.path.insert(0, 'src')
    from ilasp.ilasp_common import set_ilasp_env_variables
    set_ilasp_env_variables('src')
    from reinforcement_learning.ihsa_hrl_tabular_algorithm import IHSAAlgorithmHRLTabular
    from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
        IHSAAlgorithmHRLTabularCrossState)
    import numpy as np

    w = Path(tempfile.mkdtemp(prefix='single_factor_'))
    CF = IHSAAlgorithmHRLTabularCrossState
    cf_on = {'algorithm': 'ihsa-hrl-crossstate'}

    # S1 默认配置不受影响
    a = run_recording(IHSAAlgorithmHRLTabular, {}, w / 'y11_default')
    b = run_recording(IHSAAlgorithmHRLTabular, {'separate_update_rng': False},
                      w / 'y11_explicit_off')
    check('S1. 分离更新流的开关默认关闭，结果与从前一致',
          a == b and len(a) > 0, f'{digest(a)} vs {digest(b)}  {len(a):,} 步')

    # S2 打开后，更新采样不再消耗全局流
    c = run_recording(IHSAAlgorithmHRLTabular, {'separate_update_rng': True},
                      w / 'y11_sep')
    check('S2. 打开后行为序列改变（更新采样确实移出了全局流）',
          a != c, f'{digest(a)} vs {digest(c)}')

    # S3 单因素条件下，共用采样应给出逐步一致
    ctrl = {'separate_update_rng': True, 'subgoal_exploration_rate_const': 0.3}
    y = run_recording(IHSAAlgorithmHRLTabular, ctrl, w / 'y11_ctrl')
    cs = run_recording(CF, {**cf_on, **ctrl, 'crossstate_shared_sampling': True},
                       w / 'cf_shared')
    n = min(len(y), len(cs))
    same = y[:n] == cs[:n] and len(y) == len(cs)
    first = next((i for i in range(n) if y[i] != cs[i]), None)
    check('S3. 固定 ε + 分离更新流 + 共用采样 → 两臂逐步动作完全一致', same,
          f'{digest(y)} vs {digest(cs)}  {len(y):,} / {len(cs):,} 步'
          + (f'，首个分歧在第 {first:,} 步' if first is not None else ''))

    # S4 关掉共用采样应当不一致，否则 S3 是平凡通过
    ci = run_recording(CF, {**cf_on, **ctrl, 'crossstate_shared_sampling': False},
                       w / 'cf_indep')
    check('S4. 同条件下关掉共用采样 → 两臂不一致（S3 非平凡）',
          y != ci, f'{digest(y)} vs {digest(ci)}')

    print('\n' + ('全部通过：开关现在是单因素的，可以用来做机制干预。'
                  if not fails else
                  f'{len(fails)} 项失败：' + '；'.join(fails) +
                  '\n开关还不是单因素的，先别跑机制干预。'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
