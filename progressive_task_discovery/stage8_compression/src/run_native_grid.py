"""Step 39 grid: Y11 against Y10 in the authors' own code.

Two hypotheses, both carried over from the local work:

  H1  RMST(Y11) < RMST(Y10) — the one ordering that held under both reward
      protocols locally, and the only one worth treating as a prediction.
  H2  the gap is larger with hazards than without.

**Tabular, not the published `full_obs`.** Measured on this machine: the tabular
arms run at ~55 episodes/second and the neural ones at 0.5, so the authors'
200,000-episode setting is about an hour per run tabular and about 111 hours per
run neural. The neural grid is not a single-machine job; the tabular one is. The
deviation is declared rather than hidden, and the neural arms are left for a
pilot. QRM has no tabular implementation at all, so it is absent here for the
same reason.

**Paired instances, not a flag flip.** Flipping `use_lava` moves the objects: the
hazard occupies a cell, displaces what was there and consumes random draws, so
the layout after it shifts. Both cells of the risk factor therefore use the *same*
lava-bearing grid, and the safe one neutralises the rejecting state in the
hierarchy instead (`neutralize_deadends`). Verified: grid, agent start, observable
set and possible observations are identical, and a random policy dies 92 times in
the lava cell and never in the safe one while stepping on the hazard 226 times.

Every run is archived as one JSONL line with its full config, the code hashes and
the greedy-evaluation curve per task.
"""
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HRM = ROOT.parents[1] / 'external/hrm-learning'
PY = str(ROOT.parents[1] / '.venv-hrm/bin/python')
BASE = HRM / 'src/config/examples/ihsa/07-cw-frl-bq-exploit-flat/config.json'
WORK = Path(os.environ.get('NATIVE_WORK', '/tmp/native_grid'))
OUT = ROOT / 'results/native_grid.jsonl'

TASKS = ['book', 'book-and-quill', 'cake']
ARMS = {'Y11 shared': 'ihsa-hrl', 'Y10 no-reuse': 'ihsa-hrl-perstate'}
# The step-cost half was dropped after the first task: at 10,000 episodes every
# one of its four cells sat at 0.000 for both arms, so continuing it would only
# have produced more zeros. Protocol sensitivity was already answered locally by
# the 6,480-run grid; what the native runs can add is H1 and H2, and those live
# entirely in the authors' own protocol. Set NATIVE_PROTOCOLS to bring it back.
ALL_PROTOCOLS = {'author': 0.0, 'stepcost': -0.01}
PROTOCOLS = {k: v for k, v in ALL_PROTOCOLS.items()
             if k in os.environ.get('NATIVE_PROTOCOLS', 'author').split(',')}
EPISODES = int(os.environ.get('NATIVE_EPISODES', 60000))
SEEDS = int(os.environ.get('NATIVE_SEEDS', 4))
WORKERS = int(os.environ.get('NATIVE_WORKERS', 8))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


HASHES = {
    'perstate_tabular': sha(HRM / 'src/reinforcement_learning/ihsa_hrl_tabular_perstate_algorithm.py'),
    'craftworld_env': sha(ROOT.parents[1] / 'external/hrm-formalism-envs/gym_hierarchical_subgoal_automata/envs/craftworld/craftworld_env.py'),
}


def make_config(tag, task, arm, risk, proto, seed):
    d = json.loads(BASE.read_text())
    d['algorithm'] = ARMS[arm]
    d['state_format'] = 'tabular'
    d['num_episodes'] = EPISODES
    d['debug'] = False
    d['seed'] = 25101993 + seed
    d['environments'] = [dict(d['environments'][0], name=task)]
    d['grid_params'] = dict(d['grid_params'], use_lava=True)
    d['neutralize_deadends'] = (risk == 'safe')
    for key in ('pseudoreward_after_step', 'meta_pseudoreward_after_step'):
        d[key] = PROTOCOLS[proto]
    out = WORK / tag
    out.mkdir(parents=True, exist_ok=True)
    d['folder_name'] = str(out)
    d['checkpoint_folder'] = str(out)
    (out / 'config.json').write_text(json.dumps(d, indent=2))
    return out / 'config.json', d


def harvest(folder, task):
    """The greedy-evaluation curve per task instance: episode;reward;steps."""
    curves = {}
    d = Path(folder) / task / 'reward_steps_greedy_logs'
    if not d.exists():
        return curves
    for f in sorted(d.glob('reward_steps-*.txt')):
        rows = []
        for line in f.read_text().splitlines():
            parts = line.split(';')
            if len(parts) == 3:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
        curves[f.stem.split('-')[-1]] = rows
    return curves


def tag_of(task, arm, risk, proto, seed):
    return f"{task.replace('-','')}_{risk}_{proto}_{arm.split()[0]}_s{seed}"


def one(job):
    task, arm, risk, proto, seed = job
    tag = tag_of(task, arm, risk, proto, seed)
    cfg, conf = make_config(tag, task, arm, risk, proto, seed)
    env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    t0 = time.time()
    r = subprocess.run([PY, 'src/run_algorithm.py', str(cfg)], cwd=str(HRM),
                       env=env, capture_output=True, text=True)
    return dict(tag=tag, task=task, arm=arm, risk=risk, protocol=proto, seed=seed,
                episodes=EPISODES, seconds=round(time.time() - t0, 1),
                ok=(r.returncode == 0), stderr=r.stderr[-400:] if r.returncode else '',
                hashes=HASHES, curves=harvest(conf['folder_name'], task))


def main():
    # Resume: anything already archived is kept and not rerun, so a grid can be
    # rescoped mid-flight without throwing away what it has already paid for.
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                done.add(json.loads(line)['tag'])
            except Exception:
                pass
    jobs = [(t, a, rk, p, s) for t in TASKS for a in ARMS for rk in ('safe', 'lava')
            for p in PROTOCOLS for s in range(SEEDS)
            if tag_of(t, a, rk, p, s) not in done]
    if done:
        print(f'已归档 {len(done)} 次，跳过；本轮只跑 {len(jobs)} 次')
    print(f'{len(jobs)} 次运行，每次 {EPISODES} 幕，{WORKERS} 个工作进程')
    # Measured on this machine, per 2,000 episodes: 408 s with hazards neutralised
    # and 42 s with them live, because a dying episode is short. The cost is driven
    # by episode length, not episode count, so the two halves are priced apart.
    per_ep = {'safe': 408 / 2000, 'lava': 42 / 2000}
    est = sum(per_ep[j[2]] * EPISODES for j in jobs) / WORKERS / 3600
    print(f'预计 {est:.1f} 小时\n', flush=True)
    n = 0
    with OUT.open('a') as f, ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(one, jobs):
            f.write(json.dumps(rec) + '\n'); f.flush()
            n += 1
            print(f'  [{n}/{len(jobs)}] {rec["tag"]:<44} {rec["seconds"]:>7.0f}s '
                  f'{"ok" if rec["ok"] else "失败 " + rec["stderr"][-120:]}', flush=True)
    print(f'\n写出 {n} 行到 {OUT}')


if __name__ == '__main__':
    main()
