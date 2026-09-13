"""C-full 的验收条件：在全目标更新下，它必须与 Y11 逐位相同。

**为什么应该相同。** 表格 Q-learning 下，若干张初值相同的表，收到相同顺序、相同
目标集合、相同伪奖励的更新，会始终保持相同。共享一张表与同步更新 N 张相同的表，
在这些条件下是同一件事。所以：

  * 每一步，C-full 把同一条经验广播到每个合法状态的库；
  * 全目标更新（``formula_update_sel_num: null``）下没有采样，也就不消耗随机数；
  * 于是所有库始终彼此相同，也始终等于 Y11 那一张共享表；
  * 行为策略读的是当前状态的库，而那张库与共享表相同，所以轨迹也相同。

归纳下来，整条运行日志应当逐位相同。

**这只是验收条件，不是实验结论。** 它说的是：在这个受控设置下，参数共享可以视为
「避免重复计算和存储」的一种实现方式。在线实验里的差异因此必须从经验分配、采样、
探索这些环节去解释，而不能直接叫做「参数共享的收益」。正常实验仍保留各自独立的
计数器与采样，不要求在线行为相同——恰恰相反，检查三要求它们必须不同。

  检查一  全目标更新下 C-full 与 Y11 逐位相同        ← 核心
  检查二  C-full 的各个状态库彼此逐位相同            ← 广播确实到达了每一个
  检查三  开启采样后 C-full 与 Y11 不同              ← 说明广播不是空操作
  检查四  未访问状态的库也建立了并收到更新            ← 不是「访问后共享」
  检查五  广播没有抬高技能执行计数                    ← 更新数与执行数分离

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


def main():
    if not HRM.is_dir():
        sys.exit('请先设置 HRM_LEARNING')
    print('C-full 验收检查\n')

    # 先确认广播所依赖的那条性质，再谈等价。
    sys.path.insert(0, str(HRM / 'src'))
    try:
        from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
            check_pseudoreward_is_state_independent)
        check('0. 伪奖励不依赖任务状态（广播的前提）',
              check_pseudoreward_is_state_independent())
    except Exception as ex:
        check('0. 伪奖励不依赖任务状态（广播的前提）', False,
              f'{type(ex).__name__}: {ex}')

    work = tempfile.mkdtemp(prefix='cfull_check_')
    ALL = dict(formula_update_sel_num=None)      # 全目标更新，无采样

    rc1, l11, e1 = run_cfg('y11all', work, **ALL)
    rc2, lcf, e2 = run_cfg('cfullall', work, algorithm='ihsa-hrl-crossstate', **ALL)
    if rc1 or rc2:
        print((e1 or e2)[-1500:])
    check('1. 全目标更新下 C-full 与 Y11 逐位相同',
          rc1 == 0 and rc2 == 0 and l11 and l11 == lcf,
          f'{h(l11)} vs {h(lcf)}')

    # 采样开着时两者必须不同，否则广播根本没生效。
    rc3, l11s, _ = run_cfg('y11s', work)
    rc4, lcfs, _ = run_cfg('cfulls', work, algorithm='ihsa-hrl-crossstate')
    check('3. 开启采样后 C-full 与 Y11 不同（广播确实生效）',
          rc3 == 0 and rc4 == 0 and l11s and l11s != lcfs,
          f'{h(l11s)} vs {h(lcfs)}')

    # 检查二、四、五要看进程内部的状态，所以在本进程里再跑一次小的。
    try:
        import numpy as np
        from ilasp.ilasp_common import set_ilasp_env_variables
        set_ilasp_env_variables(str(HRM / 'src'))
        from reinforcement_learning.ihsa_hrl_tabular_crossstate_algorithm import (
            IHSAAlgorithmHRLTabularCrossState)
        cfg = json.loads((Path(work) / 'cfullall' / 'config.json').read_text())
        cfg['num_episodes'] = 60
        os.chdir(HRM)
        alg = IHSAAlgorithmHRLTabularCrossState(cfg)
        alg.run()

        # 库是「每个任务 × 每个合法状态」一份。跨任务的库本就不同（任务不同、
        # 目标不同），所以逐位比较只在同一个任务内部进行。
        ok_all, detail = True, []
        for task_id, tb in alg._perstate_banks.items():
            banks = list(tb.values())
            if len(banks) < 2:
                continue
            ref = banks[0]._q_functions
            same = all(set(b._q_functions) == set(ref) and
                       all(np.array_equal(b._q_functions[k], ref[k]) for k in ref)
                       for b in banks)
            ok_all &= same
            detail.append(f'任务{task_id}:{"同" if same else "异"}')
        check('2. 同一任务内各状态库彼此逐位相同', ok_all and detail,
              f'{len(alg._perstate_banks)} 个任务，各 '
              f'{len(next(iter(alg._perstate_banks.values())))} 个库')

        c = alg.update_counts()
        per_task = c['banks'] // max(1, len(alg._perstate_banks))
        check('4. 每个合法状态都建了库并参与更新',
              per_task == c['legal_state_keys'] and c['legal_state_keys'] > 1,
              f'每个任务 {per_task} 个库 / 合法状态 {c["legal_state_keys"]} 个，'
              f'共 {c["banks"]} 个库；TD 更新调用 {c["td_update_calls"]:,} 次 '
              f'/ 环境步 {c["broadcast_steps"]:,} 步 = {c["td_update_calls"]/max(1,c["broadcast_steps"]):.0f}x')

        # 执行计数存在库里，广播不应抬高它：所有库的执行计数之和应远小于
        # 更新计数，且只有当前状态那一个库会增长。
        steps = sum(sum(b._q_function_step_counter.values()) for b in banks)
        updates = sum(sum(b._q_function_update_counter.values()) for b in banks)
        check('5. 广播没有抬高技能执行计数',
              steps < updates,
              f'执行计数合计 {steps:,}，更新计数合计 {updates:,}')
    except Exception as ex:
        import traceback; traceback.print_exc()
        for n in ('2. C-full 的各个状态库彼此逐位相同',
                  '4. 每个合法状态都建了库并参与更新',
                  '5. 广播没有抬高技能执行计数'):
            check(n, False, f'{type(ex).__name__}: {ex}')

    print('\n' + ('全部通过。' if not fails else f'{len(fails)} 项失败：' + '；'.join(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
