#!/usr/bin/env python3
"""Run the Y11-vs-Y10 grid in the authors' HRM code. Standalone: no dependency on
the originating project tree.

Two arms, and they differ in exactly one thing:

    Y11  ihsa-hrl            one goal policy per subgoal, shared by every
                             automaton state that asks for it (the authors' own
                             implementation, unmodified)
    Cfull ihsa-hrl-crossstate  parameters per state as in Y10, but every experience
                            is broadcast to every legal state's bank. Tabular only.
                            It costs one TD update per legal state per step — 20x
                            on the frl-bq hierarchy — so budget for it separately.

    Y10  ihsa-hrl-perstate   one goal policy per (automaton state, subgoal); the
                             same reading of the same hierarchy with cross-state
                             sharing removed

Two risk conditions on ONE grid. Flipping `use_lava` would move the objects — the
hazard takes a cell, displaces what was there and consumes random draws — so both
conditions use the same hazard-bearing grid and the safe one neutralises the
rejecting state in the hierarchy instead (`neutralize_deadends`, added by patch
01). Verified identical: grid, agent start, observable set, possible observations.

Every run appends one JSON line with its full command, the code hashes and the
per-instance greedy-evaluation curve. Analysis reads that file, never a summary.
Re-running skips whatever is already in it, so the grid resumes after an
interruption and can be rescoped without losing work.

    export HRM_LEARNING=/path/to/hrm-learning
    export HRM_PYTHON=/path/to/venv/bin/python
    python3 run_grid.py --state-format full_obs --episodes 200000 --workers 8
"""
import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HRM = Path(os.environ.get('HRM_LEARNING', '')).expanduser()
PY = os.environ.get('HRM_PYTHON', sys.executable)
BASE_CFG = 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'

ARMS = {'Y11': 'ihsa-hrl', 'Y10': 'ihsa-hrl-perstate',
        'Cfull': 'ihsa-hrl-crossstate'}
PROTOCOLS = {'author': 0.0, 'stepcost': -0.01}
TASKS = ['book', 'book-and-quill', 'cake']


def sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except OSError:
        return 'missing'


def hashes():
    r = HRM / 'src/reinforcement_learning'
    return {
        'perstate_tabular': sha(r / 'ihsa_hrl_tabular_perstate_algorithm.py'),
        'perstate_dqn': sha(r / 'ihsa_hrl_dqn_perstate_algorithm.py'),
        'crossstate_tabular': sha(r / 'ihsa_hrl_tabular_crossstate_algorithm.py'),
        'hrl_algorithm': sha(r / 'ihsa_hrl_algorithm.py'),
    }


def budget_id(a):
    """Fingerprint of everything that changes what a run *is*, beyond its tag.

    Resume works by skipping tags already recorded as ok. Without this, changing
    the step budget, the episode cap, the sampling rule or the dense-evaluation
    schedule and re-running the same command would silently skip runs done under
    the *old* settings and quietly mix two protocols in one file."""
    parts = (f"ep{a.episodes}", f"env{a.env_steps or 0}",
             f"sel{'d' if a.update_sel_num is None else a.update_sel_num}",
             f"dense{a.dense_eval or 0}x{a.dense_eval_until if a.dense_eval else 0}")
    return hashlib.md5('|'.join(parts).encode()).hexdigest()[:8]


def tag_of(task, arm, risk, proto, seed, fmt):
    return f"{task.replace('-', '')}_{fmt}_{risk}_{proto}_{arm}_s{seed}"


def make_config(a, tag, task, arm, risk, proto, seed):
    d = json.loads((HRM / BASE_CFG).read_text())
    d['algorithm'] = ARMS[arm]
    d['state_format'] = a.state_format
    d['num_episodes'] = a.episodes
    d['debug'] = False
    d['use_gpu'] = a.state_format != 'tabular'      # tabular has no network
    d['seed'] = 25101993 + seed
    d['environments'] = [dict(d['environments'][0], name=task)]
    d['grid_params'] = dict(d['grid_params'], use_lava=True)
    d['neutralize_deadends'] = (risk == 'safe')
    if a.env_steps:
        d['env_step_budget'] = a.env_steps
    if a.dense_eval:
        d['dense_eval_frequency'] = a.dense_eval
        d['dense_eval_until'] = a.dense_eval_until
    if a.update_sel_num is not None:
        # 0 表示不采样、全目标更新。这是等价性检查用的设置，正式实验不要用，
        # 因为它同时改变了所有臂的每步更新预算。
        d['formula_update_sel_num'] = None if a.update_sel_num == 0 else a.update_sel_num
    for k in ('pseudoreward_after_step', 'meta_pseudoreward_after_step'):
        d[k] = PROTOCOLS[proto]
    if a.er_start_size:
        d['er_start_size'] = a.er_start_size
    out = Path(a.work).expanduser() / tag
    out.mkdir(parents=True, exist_ok=True)
    d['folder_name'] = str(out)
    d['checkpoint_folder'] = str(out)
    (out / 'config.json').write_text(json.dumps(d, indent=2))
    return out / 'config.json', d


def harvest(folder, task):
    """Per task instance: the greedy-evaluation curve, `episode;reward;steps`."""
    curves = {}
    d = Path(folder) / task / 'reward_steps_greedy_logs'
    if not d.exists():
        return curves
    for f in sorted(d.glob('reward_steps-*.txt')):
        rows = []
        for line in f.read_text().splitlines():
            p = line.split(';')
            if len(p) == 3:
                rows.append((int(p[0]), float(p[1]), float(p[2])))
        curves[f.stem.split('-')[-1]] = rows
    return curves


def one(job):
    a, (task, arm, risk, proto, seed) = job
    tag = tag_of(task, arm, risk, proto, seed, a.state_format)
    cfg, conf = make_config(a, tag, task, arm, risk, proto, seed)
    env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    cmd = [PY, 'src/run_algorithm.py', str(cfg)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(HRM), env=env, capture_output=True, text=True)
    return dict(tag=tag, budget_id=budget_id(a),
                env_step_budget=a.env_steps, dense_eval=a.dense_eval,
                update_sel_num=a.update_sel_num,
                task=task, arm=arm, risk=risk, protocol=proto, seed=seed,
                state_format=a.state_format, episodes=a.episodes,
                seconds=round(time.time() - t0, 1), ok=(r.returncode == 0),
                cmd=cmd, stderr='' if r.returncode == 0 else r.stderr[-800:],
                hashes=hashes(), curves=harvest(conf['folder_name'], task))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-format', default='full_obs', choices=['tabular', 'full_obs'])
    p.add_argument('--episodes', type=int, default=200000)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tasks', default=','.join(TASKS))
    p.add_argument('--risks', default='safe,lava')
    p.add_argument('--protocols', default='author')
    p.add_argument('--arms', default='Y11,Y10')
    p.add_argument('--env-steps', type=int, default=None,
                   help='训练环境步数预算。到预算立即停止并保留未完成回合；'
                        '评估步与广播更新不计入。同幕数不等于同环境经验量。')
    p.add_argument('--dense-eval', type=int, default=None,
                   help='只记录、不反馈训练的密集评估间隔（幕）。原作者每 100 幕的'
                        '评估保持不变；写入独立的 reward_steps_dense_logs。')
    p.add_argument('--dense-eval-until', type=int, default=4000)
    p.add_argument('--update-sel-num', type=int, default=None,
                   help='覆盖 formula_update_sel_num。传 0 表示全目标更新（不采样），'
                        '那是 C-full 与 Y11 的等价性成立的设置，不是实验设置。')
    p.add_argument('--er-start-size', type=int, default=0,
                   help='override; leave 0 to keep the published 100000')
    p.add_argument('--work', default='/tmp/hrm_grid')
    p.add_argument('--out', default='grid.jsonl')
    a = p.parse_args()

    if not HRM.exists():
        sys.exit('set HRM_LEARNING to the hrm-learning checkout')

    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                rec = json.loads(line)
                # 同名但配置不同的旧记录不算已完成：预算、幕数上限、采样设置与
                # 密集评估配置都参与标识，否则改了预算重跑会静默跳过旧协议的运行。
                if rec.get('ok') and rec.get('budget_id', budget_id(a)) == budget_id(a):
                    done.add(rec['tag'])
            except Exception:
                pass

    jobs = [(a, (t, arm, rk, pr, s))
            for t in a.tasks.split(',')
            for arm in a.arms.split(',')
            for rk in a.risks.split(',')
            for pr in a.protocols.split(',')
            for s in range(a.seeds)
            if tag_of(t, arm, rk, pr, s, a.state_format) not in done]

    print(f'{len(done)} 已完成并跳过；本轮 {len(jobs)} 次，每次 {a.episodes} 幕，'
          f'{a.workers} 个进程，格式 {a.state_format}', flush=True)
    n = 0
    with out.open('a') as f, ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(one, jobs):
            f.write(json.dumps(rec) + '\n'); f.flush()
            n += 1
            status = 'ok' if rec['ok'] else 'FAILED ' + rec['stderr'][-200:]
            print(f'  [{n}/{len(jobs)}] {rec["tag"]:<46} {rec["seconds"]:>8.0f}s {status}',
                  flush=True)
    print(f'\n{n} 行写入 {out}')


if __name__ == '__main__':
    main()
