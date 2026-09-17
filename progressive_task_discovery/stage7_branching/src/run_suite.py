"""Run a frozen JSON experiment specification, retaining each raw result."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def execute(job):
    command = [str(ROOT / 'branch')]
    for key, value in job['args'].items():
        command += ['--' + key.replace('_', '-'), str(value)]
    started = time.monotonic()
    result = json.loads(subprocess.check_output(command, text=True))
    result.update(variant=job['variant'], condition=job['condition'], command=command,
                  wall_seconds=time.monotonic() - started)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('--workers', type=int, default=2)
    opt = parser.parse_args()
    spec = json.loads(Path(opt.spec).read_text())
    out = ROOT / 'results' / spec['name']
    out.mkdir(exist_ok=False)
    (out / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    shutil.copy(ROOT / 'src/branch.cpp', out / 'branch_snapshot.cpp')
    manifest = {'source_sha256': hashlib.sha256((ROOT / 'src/branch.cpp').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256((ROOT / 'branch').read_bytes()).hexdigest(),
                'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'workers': opt.workers}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    jobs = []
    for condition, variant, k, seed in itertools.product(spec['conditions'], spec['variants'], spec['lengths'], spec['seeds']):
        args = dict(spec.get('defaults', {}))
        args.update(condition['args'])
        args.update(variant['args'])
        args.update(k=k, seed=seed)
        jobs.append(dict(args=args, condition=condition['name'], variant=variant['name']))
    start = time.monotonic()
    with (out / 'raw.jsonl').open('w') as log, concurrent.futures.ProcessPoolExecutor(max_workers=opt.workers) as pool:
        for i, result in enumerate(pool.map(execute, jobs), 1):
            log.write(json.dumps(result, separators=(',', ':')) + '\n')
            log.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"{spec['name']}: {i}/{len(jobs)}, elapsed {time.monotonic()-start:.1f}s", flush=True)
    manifest.update(completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'), elapsed_seconds=time.monotonic()-start, runs=len(jobs))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
