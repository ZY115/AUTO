#!/usr/bin/env python3
"""Pre-flight checks. Run this before spending any GPU time.

Seven checks, each of which caught a real problem during development. A failure
here means the grid would produce numbers that look fine and mean nothing.

    export HRM_LEARNING=/path/to/hrm-learning
    python3 verify.py

Check 4 is the one that matters most: if the per-state arm with its key pinned to
a constant does not reproduce the shared arm bit for bit, the modification has
changed more than sharing and every comparison downstream is confounded.
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


def main():
    print('HRM 交接预检\n')
    if not check('1. HRM_LEARNING 指向一个 hrm-learning 检出',
                 HRM.exists() and (HRM / BASE).exists(), str(HRM)):
        sys.exit(1)

    # ---- 2. 环境与层级 ----
    try:
        import gym, gym_hierarchical_subgoal_automata  # noqa: F401
        expect = {'CraftWorldBook-v0': 3, 'CraftWorldBookAndQuill-v0': 5,
                  'CraftWorldCake-v0': 6}
        got = {}
        for task, n in expect.items():
            e = gym.make(task, params={
                'environment_seed': 0, 'max_episode_length': 1000,
                'grid_params': {'grid_type': 'four_rooms', 'width': 13, 'height': 13,
                                'size': 13, 'right_rooms_even': False,
                                'lava_locations': 'door_intersections',
                                'use_lava': True, 'num_lava': 1,
                                'use_lava_walls': False, 'max_objs_per_class': 2}})
            u = e.unwrapped; u.reset()
            got[task] = len(u.get_hierarchy().get_automata_names())
        check('2. 三个任务都能创建，子自动机数为 3 / 5 / 6',
              got == expect, str(got))
    except Exception as ex:
        check('2. 三个任务都能创建', False, f'{type(ex).__name__}: {ex}')

    # ---- 3. 配对实例 ----
    try:
        def snap(neutral):
            e = gym.make('CraftWorldBookAndQuill-v0', params={
                'environment_seed': 0, 'max_episode_length': 1000,
                'neutralize_deadends': neutral,
                'grid_params': {'grid_type': 'four_rooms', 'width': 13, 'height': 13,
                                'size': 13, 'right_rooms_even': False,
                                'lava_locations': 'door_intersections',
                                'use_lava': True, 'num_lava': 1,
                                'use_lava_walls': False, 'max_objs_per_class': 2}})
            u = e.unwrapped; u.reset()
            g = u.env.env.unwrapped
            objs = {}
            for x in range(g.width):
                for y in range(g.height):
                    c = g.grid.get(x, y)
                    if c is not None:
                        objs.setdefault(c.type, []).append((x, y))
            return (tuple(g.agent_pos), g.agent_dir,
                    {k: sorted(v) for k, v in sorted(objs.items())},
                    sorted(u.get_observables()))
        check('3. safe 与 lava 两格的网格/起点/可观测集逐项相同',
              snap(False) == snap(True),
              '若失败，说明补丁 01 未生效')
    except Exception as ex:
        check('3. 配对实例一致', False, f'{type(ex).__name__}: {ex}')

    work = tempfile.mkdtemp(prefix='hrm_verify_')

    # ---- 4. 折叠等价（表格） ----
    rc1, l1, e1 = run_cfg('y11', work)
    rc2, l2, e2 = run_cfg('y10c', work, algorithm='ihsa-hrl-perstate',
                          perstate_collapse=True)
    rc3, l3, e3 = run_cfg('y10', work, algorithm='ihsa-hrl-perstate')
    h = lambda ls: hashlib.md5('\n'.join(ls).encode()).hexdigest()[:12]
    check('4. 表格：键钉成常数后与 Y11 逐位相同',
          rc1 == 0 and rc2 == 0 and l1 and l1 == l2, f'{h(l1)} vs {h(l2)}')
    check('5. 表格：不钉死时与 Y11 不同（改动确实生效）',
          rc3 == 0 and l3 and l1 != l3, f'{h(l1)} vs {h(l3)}')

    # ---- 6. 折叠等价（神经） ----
    rc4, l4, _ = run_cfg('d11', work, state_format='full_obs', num_episodes=40,
                         er_start_size=200, er_buffer_size=20000, tgt_update_freq=100)
    rc5, l5, _ = run_cfg('d10c', work, state_format='full_obs', num_episodes=40,
                         algorithm='ihsa-hrl-perstate', perstate_collapse=True,
                         er_start_size=200, er_buffer_size=20000, tgt_update_freq=100)
    check('6. 神经：键钉成常数后与 Y11 逐位相同',
          rc4 == 0 and rc5 == 0 and l4 and l4 == l5, f'{h(l4)} vs {h(l5)}')

    # ---- 7. GPU ----
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else '无'
        check('7. CUDA 可用', cuda, f'torch {torch.__version__}, {name}')
    except Exception as ex:
        check('7. CUDA 可用', False, str(ex))

    print()
    if fails:
        print(f'{len(fails)} 项未通过，先修好再跑网格：')
        for f in fails:
            print('   -', f)
        sys.exit(1)
    print('全部通过。可以开始跑网格。')
    print('提醒：er_start_size 的发表值是 100000，短跑里一次梯度更新都不会发生，')
    print('      任何低于约 1000 幕的计时都只测了走路，不能用来推算总时长。')


if __name__ == '__main__':
    main()
