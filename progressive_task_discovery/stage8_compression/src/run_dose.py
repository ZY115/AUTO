"""Step 38.5 robustness: is 3% a cherry-picked hazard density?

The main run fixed one dose and said so. That is the first thing to attack, so
this sweeps it and changes nothing else: same tasks, same map seeds, same arms,
same budget, expensive navigation only. The claim being defended is not the size
of the effect at any one dose but that the separation lives in a regime rather
than at a point.

Only three arms are needed for that, and they are the ladder that matters after
Step 38.5:

    QRM   task structure enters the value table, Q(s, q, a)
    Y10   task structure routes to a goal, one policy per (task state, goal)
    Y11   task structure routes to a goal, one policy per goal

Y01 is run with --meta-safe, which hands it oracle knowledge of which goals are
fatal. Without that its collapse is mostly about proposing the hazard as a goal,
which is a different phenomenon from the one under test.
"""
import json, subprocess, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
EXE = str(ROOT / 'compress')
BUDGET, EVERY, SEEDS = 1_000_000, 2_000, 15
DOSES = ['0.00', '0.01', '0.02', '0.03', '0.04']
ARMS = {
    'QRM':               ['--method', 'automaton', '--crm', '1'],
    'Y10 no-reuse':      ['--method', 'goal', '--skill-key', 'perstate'],
    'Y11 goal readout':  ['--method', 'goal'],
    'Y01 meta (safe)':   ['--method', 'goal', '--route', 'meta', '--meta-safe', '1'],
}


def one(job):
    dose, tag, arm, seed = job
    r = json.loads(subprocess.check_output(
        [EXE, '--task', str(ROOT / f'results/craftworld_dose/d{dose}/{tag}.task'),
         '--seed', str(seed), '--budget', str(BUDGET), '--every', str(EVERY)]
        + ARMS[arm], text=True))
    return dose, arm, r['auc'], r['first90'], r['failures'], r['fail_at_90']


def main():
    jobs = []
    for dose in DOSES:
        m = json.loads((ROOT / f'results/craftworld_dose/d{dose}/tasks.json').read_text())
        for x in m:
            if x['risk'] != 'lava':
                continue
            for a in ARMS:
                for s in range(SEEDS):
                    jobs.append((dose, x['tag'], a, s))
    R = defaultdict(list)
    with ProcessPoolExecutor(max_workers=10) as ex:
        for dose, arm, auc, f90, fails, f90c in ex.map(one, jobs, chunksize=16):
            R[(dose, arm)].append((auc, f90, fails, f90c))
    out = {f'{d}|{a}': dict(
        rmst=mean(x[0] for x in R[(d, a)]),
        attain=sum(1 for x in R[(d, a)] if x[1] > 0) / len(R[(d, a)]),
        fails=median(x[2] for x in R[(d, a)]),
        fail90=(median([x[3] for x in R[(d, a)] if x[3] >= 0])
                if any(x[3] >= 0 for x in R[(d, a)]) else None))
        for d in DOSES for a in ARMS}
    (ROOT / 'results/dose_sweep.json').write_text(json.dumps(out, indent=2) + '\n')

    haz = {d: json.loads((ROOT / f'results/craftworld_dose/d{d}/tasks.json').read_text())[0]['hazards']
           for d in DOSES}
    for lab, key, fmt in (('RMST', 'rmst', '{:.3f}'), ('达标率', 'attain', '{:.0%}'),
                          ('达标前不可逆失败(中位)', 'fail90', '{}')):
        print(f'\n{lab}（昂贵导航，104 自由格）')
        print(f'{"危险格":>6} {"密度":>6} ' + ' '.join(f'{a:>18}' for a in ARMS))
        for d in DOSES:
            row = []
            for a in ARMS:
                v = out[f'{d}|{a}'][key]
                row.append('—' if v is None else fmt.format(v))
            print(f'{haz[d]:>6} {d:>6} ' + ' '.join(f'{r:>18}' for r in row))

    print('\n分叉区间：Y11 相对 Y10 的 RMST 差')
    for d in DOSES:
        g = out[f'{d}|Y11 goal readout']['rmst'] - out[f'{d}|Y10 no-reuse']['rmst']
        a10 = out[f'{d}|Y10 no-reuse']['attain']
        a11 = out[f'{d}|Y11 goal readout']['attain']
        print(f'  危险格 {haz[d]}（{d}）  +{g:.3f}   达标率 {a10:.0%} -> {a11:.0%}')


if __name__ == '__main__':
    main()
