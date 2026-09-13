#!/usr/bin/env python3
"""Run the Y11-vs-Y10 grid in the authors' HRM code. Standalone: no dependency on
the originating project tree.

Two arms, and they differ in exactly one thing:

    Y11  ihsa-hrl            one goal policy per subgoal, shared by every
                             automaton state that asks for it (the authors' own
                             implementation, unmodified)
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

ARMS = {'Y11': 'ihsa-hrl', 'Y10': 'ihsa-hrl-perstate'}
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
        'hrl_algorithm': sha(r / 'ihsa_hrl_algorithm.py'),
    }


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
    return dict(tag=tag, task=task, arm=arm, risk=risk, protocol=proto, seed=seed,
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
                if rec.get('ok'):
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
