"""Run the compression ladder: same map, same task, different task representation."""
import concurrent.futures, hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arms(meta):
    """Every arm sees identical observations and does identical update counts."""
    w = meta['min_sufficient_window']
    out = [('count', {'method': 'count'}), ('bag', {'method': 'bag'}),
           (f'window{w - 2}', {'method': 'window', 'window': w - 2}),
           (f'window{w}', {'method': 'window', 'window': w}),
           ('history', {'method': 'history'}),
           ('merged', {'method': 'merged', 'theta': 400, 'revision': 'transfer'}),
           ('automaton', {'method': 'automaton'})]
    return out


def run(job):
    cmd = [str(ROOT / 'compress'), '--task', job['task'], '--seed', str(job['seed']),
           '--budget', str(job['budget']), '--every', str(job['every'])]
    for k, v in job['args'].items():
        cmd += ['--' + k, str(v)]
    started = time.monotonic()
    rec = json.loads(subprocess.check_output(cmd, text=True))
    rec.update(arm=job['arm'], rung=job['rung'], ratio=job['ratio'],
               wall_seconds=time.monotonic() - started)
    return rec


def main(name='ladder', budget=200000, seeds=12, workers=8, seed0=0):
    budget, seeds, workers, seed0 = int(budget), int(seeds), int(workers), int(seed0)
    metas = json.loads((ROOT / 'results/family_maps/rungs.json').read_text())
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for meta in metas:
        for arm, args in arms(meta):
            for seed in range(seed0, seed0 + seeds):
                jobs.append(dict(task=str(ROOT / f"results/family_maps/{meta['tag']}.task"),
                                 arm=arm, args=args, seed=seed, budget=budget,
                                 every=1000, rung=meta['tag'], ratio=meta['honest_ratio']))
    (out / 'spec.json').write_text(json.dumps(
        dict(name=name, budget=budget, seeds=seeds, seed0=seed0, jobs=len(jobs),
             source_sha256=hashlib.sha256((ROOT / 'src/compress.cpp').read_bytes()).hexdigest(),
             binary_sha256=hashlib.sha256((ROOT / 'compress').read_bytes()).hexdigest()), indent=2) + '\n')
    started = time.monotonic()
    done = 0
    with (out / 'raw.jsonl').open('w') as fh, \
         concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(run, jobs):
            fh.write(json.dumps(rec) + '\n')
            done += 1
            if done % 24 == 0:
                print(f'{done}/{len(jobs)} {round(time.monotonic() - started, 1)}s', flush=True)
    print(f'finished {len(jobs)} runs in {round(time.monotonic() - started, 1)}s')


if __name__ == '__main__':
    main(*sys.argv[1:])
