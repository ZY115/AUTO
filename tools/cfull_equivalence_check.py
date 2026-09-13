"""C-full 的验收检查。分两层，因为这两层能证明的东西不一样。

**第一层：固定经验流 + 共用采样 → Q 表逐位相同。** 这是可证的。表格 Q-learning 下，
若干张初值相同的表收到相同顺序、相同目标集合、相同伪奖励的更新，会始终保持相同。
这一层不依赖轨迹，直接把录下来的经验流重放进两组表再逐位比。

**第二层：完整在线运行。** 这一层**不能**从第一层推出来，因为除了 Q 表还有三条
通道会让两臂分叉：

  1. **各库独立采样。** 开了 ``formula_update_sel_num`` 之后每个库自己抽子集。
  2. **全局随机数流。** 那些抽样走的是 NumPy 全局流，额外抽样会平移之后每一次
     探索用的随机数。
  3. **探索日程。** ``_get_formula_exploration_rate`` 按
     ``_get_formula_q_function_step_count`` 退火，而那个计数走 ``_get_policy_bank``，
     即**当前状态的库**。Y11 把一个子目标在所有状态下的执行次数累加到一起，
     C-full 分状态计。实测 60 幕：两臂的计数已经在 46% 的探索率查询上不同，
     首次分叉在第 36 次查询（Y11 计 36，C-full 计 0）。

保留第 3 条是**故意的**——广播不该抬高执行计数。但这意味着在线的逐位一致只能在
额外对齐探索日程与两条随机数流之后才能要求。**短跑摘要一致是弱得多的证据**：
150 幕时探索率还在 0.9997 上下，两臂差值在第五位小数，很少改变抽到的动作。
那是跑得短，不是等价。

所以第二层这里只**报告**分叉的规模，不断言等价。正式 C-full 继续保留既定的独立
计数与独立采样，**不能据此宣布正常实验「只剩独立采样一个差异」**。

  A1  固定经验流 + 共用采样：各库 Q 表逐位等于单张共享表     ← 可证，核心
  A2  同上，各库彼此逐位相同
  B1  在线（全目标 + 共用采样）：回合摘要是否一致 + 分叉报告  ← 报告，不是断言
  C1  广播确实到达每一个合法状态库，且各库更新了哪些目标     ← 直接观测
  C2  执行计数的增量只落在当前状态的库上                     ← 直接观测
  C3  探索计数分叉的规模（这是一条活的通道，必须显式报出）
  D0  伪奖励不依赖任务状态（广播的前提）

用法（需要 HRM_LEARNING 与 HRM_PYTHON）：

    $HRM_PYTHON tools/cfull_equivalence_check.py
"""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

HRM = Path(os.environ.get('HRM_LEARNING', '')).expanduser()
PY = os.environ.get('HRM_PYTHON', sys.executable)
BASE = 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'
OK, BAD = '  [通过] ', '  [失败] '
fails = []


def check(label, cond, detail=''):
    print((OK if cond else BAD) + label + (('  ' + detail) if detail else ''))
    if not cond:
        fails.append(label)
    return cond


def run_cfg(tag, work, **over):
    d = json.loads((HRM / BASE).read_text())
    d.update(debug=True, num_episodes=150, state_format='tabular',
             algorithm='ihsa-hrl', use_gpu=False)
    d.update(over)
    out = Path(work) / tag
    out.mkdir(parents=True, exist_ok=True)
    d['folder_name'] = str(out)
    d['checkpoint_folder'] = str(out)
    cfg = out / 'config.json'
    cfg.write_text(json.dumps(d, indent=2))
    env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1')
    r = subprocess.run([PY, 'src/run_algorithm.py', str(cfg)], cwd=str(HRM),
                       env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith('Domain:')]
    return r.returncode, lines, r.stderr


h = lambda ls: hashlib.md5('\n'.join(ls).encode()).hexdigest()[:12] if ls else '空'


def replay_layer_a(work):
    """第一层：录下 Y11 的经验流与它当步用的目标集合，再把同一条流、同一个目标集合
    重放进（a）一张新的共享表、（b）一整组新的按状态分开的表。比 Q 表，不比轨迹。"""
    import numpy as np
    from reinforcement_learning.ihsa_hrl_tabular_algorithm import (
        FormulaBankTabular, IHSAAlgorithmHRLTabular)
    from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
        IHSAAlgorithmHRLTabularCrossState)

    cfg = json.loads((Path(work) / 'y11s' / 'config.json').read_text())
    cfg = dict(cfg, num_episodes=40, debug=False)

    stream = []
    orig_upd = FormulaBankTabular.update_q_functions
    orig_sel = FormulaBankTabular._get_subgoals_to_update

    def rec_upd(self, task, state, action, next_state, is_terminal, is_goal_achieved, obs):
        self._recording = []
        try:
            return orig_upd(self, task, state, action, next_state, is_terminal,
                            is_goal_achieved, obs)
        finally:
            stream.append(((task, state, action, next_state, is_terminal,
                            is_goal_achieved, obs), list(self._recording)))

    def rec_sel(self):
        out = list(orig_sel(self))
        if hasattr(self, '_recording'):
            self._recording.extend(out)
        return out

    FormulaBankTabular.update_q_functions = rec_upd
    FormulaBankTabular._get_subgoals_to_update = rec_sel
    try:
        IHSAAlgorithmHRLTabular(dict(cfg, folder_name=str(Path(work) / 'recA'),
                                     checkpoint_folder=str(Path(work) / 'recA'))).run()
    finally:
        FormulaBankTabular.update_q_functions = orig_upd
        FormulaBankTabular._get_subgoals_to_update = orig_sel

    # 重放目标：一张新的共享表，和一整组新的按状态分开的表。
    ref_alg = IHSAAlgorithmHRLTabular(dict(cfg, folder_name=str(Path(work) / 'refA'),
                                           checkpoint_folder=str(Path(work) / 'refA')))
    cf_alg = IHSAAlgorithmHRLTabularCrossState(
        dict(cfg, algorithm='ihsa-hrl-crossstate', crossstate_shared_sampling=True,
             folder_name=str(Path(work) / 'cfA'), checkpoint_folder=str(Path(work) / 'cfA')))
    cf_alg._perstate_key = None
    obs0 = stream[0][0][6]
    ref_alg._on_initial_observation(obs0)
    cf_alg._on_initial_observation(obs0)
    cf_alg._ensure_banks(0, 0)

    ref_bank = ref_alg._get_policy_bank(0)
    cf_banks = list(cf_alg._perstate_banks[0].values())

    def apply(bank, args, goals):
        bank._get_subgoals_to_update = lambda _g=goals: _g
        try:
            bank.update_q_functions(*args)
        finally:
            del bank._get_subgoals_to_update

    n = 0
    for args, goals in stream:
        if not goals:
            continue
        try:
            apply(ref_bank, args, goals)
            for b in cf_banks:
                apply(b, args, goals)
        except KeyError:
            break          # 目标尚未在某张表里建好，停在这里，下面按已重放的量判定
        n += 1

    ok_ref = all(set(b._q_functions) == set(ref_bank._q_functions) and
                 all(np.array_equal(b._q_functions[k], ref_bank._q_functions[k])
                     for k in ref_bank._q_functions)
                 for b in cf_banks)
    r0 = cf_banks[0]._q_functions
    ok_each = all(all(np.array_equal(b._q_functions[k], r0[k]) for k in r0)
                  for b in cf_banks)
    return ok_ref, ok_each, n, len(cf_banks)


def main():
    if not HRM.is_dir():
        sys.exit('请先设置 HRM_LEARNING')
    print('C-full 验收检查\n')
    sys.path.insert(0, str(HRM / 'src'))
    work = tempfile.mkdtemp(prefix='cfull_check_')

    try:
        from ilasp.ilasp_common import set_ilasp_env_variables
        set_ilasp_env_variables(str(HRM / 'src'))
        from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
            check_pseudoreward_is_state_independent,
            IHSAAlgorithmHRLTabularCrossState)
        check('D0. 伪奖励不依赖任务状态（广播的前提）',
              check_pseudoreward_is_state_independent())
    except Exception as ex:
        check('D0. 伪奖励不依赖任务状态（广播的前提）', False, f'{type(ex).__name__}: {ex}')
        print('\n无法继续。'); return 1

    # 先跑出配置文件供后面复用
    rc0, l11s, e0 = run_cfg('y11s', work)
    rc0b, lcfs, _ = run_cfg('cfulls', work, algorithm='ihsa-hrl-crossstate')

    # ---- 第一层：固定经验流 + 共用采样 ----
    os.chdir(HRM)
    try:
        ok_ref, ok_each, n, nb = replay_layer_a(work)
        check('A1. 固定经验流+共用采样：各库 Q 表逐位等于单张共享表', ok_ref and n > 0,
              f'重放 {n:,} 步，{nb} 个库')
        check('A2. 同上，各库彼此逐位相同', ok_each and nb > 1, f'{nb} 个库')
    except Exception as ex:
        import traceback; traceback.print_exc()
        check('A1. 固定经验流+共用采样：各库 Q 表逐位等于单张共享表', False,
              f'{type(ex).__name__}: {ex}')
        check('A2. 同上，各库彼此逐位相同', False, '同上')

    # ---- 第二层：在线，只报告不断言 ----
    ALL = dict(formula_update_sel_num=None)
    rc1, l11, _ = run_cfg('y11all', work, **ALL)
    rc2, lcf, _ = run_cfg('cfullall', work, algorithm='ihsa-hrl-crossstate',
                          crossstate_shared_sampling=True, **ALL)
    print(f'  [报告] B1. 在线（全目标+共用采样）回合摘要'
          f'{"一致" if l11 and l11 == lcf else "不一致"}  {h(l11)} vs {h(lcf)}')
    print('         这不是等价证明：短跑时探索率还在 0.9997 附近，两臂 ε 的差在第五位'
          '小数，很少改变抽到的动作。')

    # ---- 直接观测 ----
    try:
        import numpy as np
        cfg = json.loads((Path(work) / 'cfulls' / 'config.json').read_text())
        cfg.update(num_episodes=60, debug=False)
        alg = IHSAAlgorithmHRLTabularCrossState(cfg)

        seen = {'banks': [], 'goals': [], 'step_delta': []}
        orig = IHSAAlgorithmHRLTabularCrossState._update_q_functions

        def watch(self, domain_id, task_id, *a, **k):
            banks = self._perstate_banks.get(task_id, {})
            before = {key: sum(b._q_function_step_counter.values())
                      for key, b in banks.items()}
            r = orig(self, domain_id, task_id, *a, **k)
            banks = self._perstate_banks.get(task_id, {})
            after = {key: sum(b._q_function_step_counter.values()) for key, b in banks.items()}
            moved = [key for key in after if after[key] != before.get(key, 0)]
            seen['banks'].append(list(self._last_broadcast_banks))
            seen['goals'].append(list(self._last_broadcast_goals))
            seen['step_delta'].append(moved)
            return r

        IHSAAlgorithmHRLTabularCrossState._update_q_functions = watch
        try:
            alg.run()
        finally:
            IHSAAlgorithmHRLTabularCrossState._update_q_functions = orig

        legal = len(alg._legal_keys)
        full = [len(b) for b in seen['banks']]
        allgoals = [g for row in seen['goals'] for g in row]
        check('C1. 广播到达每一个合法状态库，且每个库都真的写了 Q 值',
              full and min(full) == legal and allgoals and min(allgoals) > 0,
              f'每步覆盖 {min(full)}–{max(full)}/{legal} 个库；'
              f'每库每步写入目标数 {min(allgoals)}–{max(allgoals)}')
        bad = [m for m in seen['step_delta'] if len(m) > 1]
        check('C2. 执行计数的增量只落在一个库上（广播没有放大它）',
              not bad, f'{len(seen["step_delta"]):,} 步中有 {len(bad)} 步动了多个库')

        ec = alg.execution_counts_by_bank(0)
        tot = sum(sum(v.values()) for v in ec.values())
        nz = sum(1 for v in ec.values() if sum(v.values()) > 0)
        c = alg.update_counts()
        print(f'  [报告] C3. 执行计数分布在 {nz}/{len(ec)} 个库上，合计 {tot:,}；'
              f'Y11 会把这 {tot:,} 次累加成每个子目标一个计数。')
        print(f'         探索率按它退火，所以这是一条活的通道，不是实现瑕疵。')
        print(f'  [报告] 成本：环境步 {c["broadcast_steps"]:,}，库级更新调用 '
              f'{c["bank_update_calls"]:,}，标量 TD 写入 {c["scalar_td_updates"]:,} '
              f'（{c["scalar_td_updates"]/max(1,c["broadcast_steps"]):.0f} 次/环境步）')
    except Exception as ex:
        import traceback; traceback.print_exc()
        check('C1. 广播到达每一个合法状态库，且每个库都真的写了 Q 值', False, str(ex))
        check('C2. 执行计数的增量只落在一个库上（广播没有放大它）', False, str(ex))

    print('\n' + ('断言项全部通过。报告项见上，它们不是通过/失败，是需要写进论文的事实。'
                  if not fails else f'{len(fails)} 项失败：' + '；'.join(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
