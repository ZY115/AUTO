"""Step 40: how much of the three-arm ordering is the reward protocol?

The pre-transfer audit found that QRM's collapse under hazards is caused by the
-0.01 step cost, not by any inability to carry safety across task states: remove
the step cost while keeping the same terminal reward and QRM goes from 0/6 to
6/6 on the probe maps. On one map it then beats the per-state arm outright, so
`QRM < Y10 < Y11` is not a protocol-invariant ladder and must not be assumed.

This runs the whole public grid under both protocols and changes nothing else.

    protocol `orig`  step cost 0.01, terminal reward 8 * 0.01 * maxlen
    protocol `nostep` step cost 0,   terminal reward held at the same value

Holding the terminal reward fixed is the point: `reward_success` is derived from
the step cost, so setting the cost to zero without an override silently zeroes
the goal reward too and nothing learns at all. That mistake is easy to make and
was made once here before the override was added.

Every run is appended to a JSONL archive as it finishes: full command, source and
binary hashes, every checkpoint, and the summary. Analysis reads the archive, not
a pre-aggregated summary, which is what the audit asked for after the dose sweep
turned out to have kept only twenty aggregate numbers.
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'compress'
SRC = ROOT / 'src/compress.cpp'
OUT = ROOT / 'results/reward_grid.jsonl'
BUDGET, EVERY, SEEDS = 1_000_000, 2_000, 15

ARMS = {
    'QRM': ['--method', 'automaton', '--crm', '1'],
    'Y10 no-reuse': ['--method', 'goal', '--skill-key', 'perstate'],
    'Y11 goal readout': ['--method', 'goal'],
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


HASHES = {'src': sha(SRC), 'exe': sha(EXE)}


def one(job):
    tag, arm, proto, seed, reward = job
    cmd = [str(EXE), '--task', str(ROOT / f'results/craftworld_lava/{tag}.task'),
           '--seed', str(seed), '--budget', str(BUDGET), '--every', str(EVERY)]
    cmd += ARMS[arm]
    if proto == 'nostep':
        cmd += ['--cost', '0.0', '--reward-success', f'{reward:.6f}']
    r = json.loads(subprocess.check_output(cmd, text=True))
    return dict(tag=tag, arm=arm, protocol=proto, seed=seed, cmd=cmd,
                hashes=HASHES, first90=r['first90'], auc=r['auc'],
                final=r['final_success'], failures=r['failures'],
                fail_at_90=r['fail_at_90'], goal_deaths=r['goal_deaths'],
                checkpoints=r['checkpoints'])


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from taskfile import TaskFile
    meta = json.loads((ROOT / 'results/craftworld_lava/tasks.json').read_text())
    jobs = []
    for m in meta:
        t = TaskFile(str(ROOT / f"results/craftworld_lava/{m['tag']}.task"))
        reward = 8 * 0.01 * t.maxlen           # what `orig` derives it to be
        for arm in ARMS:
            for proto in ('orig', 'nostep'):
                for s in range(SEEDS):
                    jobs.append((m['tag'], arm, proto, s, reward))
    print(f'{len(jobs)} 次运行，预算 {BUDGET}，源码 {HASHES["src"]} 二进制 {HASHES["exe"]}')
    n = 0
    with OUT.open('w') as f, ProcessPoolExecutor(max_workers=10) as ex:
        for rec in ex.map(one, jobs, chunksize=8):
            f.write(json.dumps(rec) + '\n'); f.flush()
            n += 1
            if n % 500 == 0:
                print(f'  {n}/{len(jobs)}', flush=True)
    print(f'写出 {n} 行到 {OUT}')


if __name__ == '__main__':
    main()
