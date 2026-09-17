"""Step 36: where does the automaton's benefit come from — routing or reuse?

The query-attribution line is closed. This asks the control question instead:
the same task structure can enter RL as a value-table index or as a router that
says which already-learned skill to run. Which interface carries the benefit, and
how much of it is really just having reusable skills at all?

A 2x2 on two switches, plus the standard structural baseline beside it:

                       | no skill reuse          | skill reuse
    -------------------+-------------------------+---------------------------
    no automaton route | Y00  flat history RL     | Y01  learned meta-controller
    automaton route    | Y10  per-state skills    | Y11  goal readout

  Y01 picks its goal by a learned value over the physical cell and the same
       history feature the flat arm indexes by. It is never told the task state
       and never given a correct goal order, which would smuggle the whole
       decomposition back in as an option schedule.
  Y10 keeps the identical goal reading but learns a separate policy per (task
       state, goal), so nothing transfers between stages.
  QRM sits outside the square: it is the standard way to use a reward machine,
       Q(s, q, a), and is reported alongside rather than inside the factorial.

Structural arms all get the same true task machine. The non-structural ablations
deliberately do *not* receive goal information derived from it — that is the
channel being ablated. Everything else is shared: event labels, dynamics,
reward, and step budget.
"""
import json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXE = str(ROOT / 'compress')
# `every` sets the resolution of the learning curve. At 2,000 the goal arm is
# already saturated by the first checkpoint and AUC cannot separate anything;
# the differences all live in the first few thousand steps.
BUDGET, EVERY, SEEDS = 200_000, 250, 30

ARMS = {
    'Y00 flat history':      ['--method', 'history'],
    'Y01 skill-only meta':   ['--method', 'goal', '--route', 'meta'],
    'Y10 no-reuse readout':  ['--method', 'goal', '--skill-key', 'perstate'],
    'Y11 goal readout':      ['--method', 'goal'],
    'QRM (CRM)':             ['--method', 'automaton', '--crm', '1'],
}


def one(job):
    tag, arm, seed = job
    r = json.loads(subprocess.check_output(
        [EXE, '--task', str(ROOT / f'results/ceiling/{tag}.task'), '--seed', str(seed),
         '--budget', str(BUDGET), '--every', str(EVERY)] + ARMS[arm], text=True))
    return tag, arm, seed, dict(auc=r['auc'], final=r['final_success'],
                                first90=r['first90'])


def bootstrap(per_family, a, b, n=10000, seed=7):
    """Paired block bootstrap over families — never over seeds."""
    import random
    fams = sorted(per_family)
    d = [per_family[f][a] - per_family[f][b] for f in fams]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in d]
        out.append(mean(s))
    out.sort()
    return mean(d), out[int(.025 * n)], out[int(.975 * n)]


def main():
    fams = [f['tag'] for f in json.loads((ROOT / 'results/ceiling/families.json').read_text())]
    jobs = [(t, a, s) for t in fams for a in ARMS for s in range(SEEDS)]
    res = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for tag, arm, seed, v in ex.map(one, jobs, chunksize=16):
            res.setdefault(tag, {}).setdefault(arm, []).append(v)
    (ROOT / 'results/factorial.json').write_text(json.dumps(res, indent=2) + '\n')

    fam_auc = {t: {a: mean(x['auc'] for x in res[t][a]) for a in ARMS} for t in fams}
    fam_fin = {t: {a: mean(x['final'] for x in res[t][a]) for a in ARMS} for t in fams}
    print(f'{len(fams)} 个族 x {SEEDS} 个种子 x {len(ARMS)} 臂，预算 {BUDGET}\n')
    print(f'{"臂":<24} {"AUC":>8} {"终局成功率":>11} {"first90 中位":>13}')
    for a in ARMS:
        f9 = [x['first90'] for t in fams for x in res[t][a] if x['first90'] > 0]
        f9.sort()
        med = f9[len(f9) // 2] if f9 else None
        solved = sum(1 for t in fams for x in res[t][a] if x['first90'] > 0)
        tot = len(fams) * SEEDS
        print(f'{a:<24} {mean(fam_auc[t][a] for t in fams):>8.3f} '
              f'{mean(fam_fin[t][a] for t in fams):>11.3f} '
              f'{(str(med) if med else "-"):>9} ({solved}/{tot})')

    print('\n配对自助（按族重抽，不按种子），AUC 差值与 95% 区间')
    pairs = [
        ('Y01 skill-only meta', 'Y00 flat history', '只加技能复用'),
        ('Y10 no-reuse readout', 'Y00 flat history', '只加自动机路由'),
        ('Y11 goal readout', 'Y01 skill-only meta', '在技能之上再加路由'),
        ('Y11 goal readout', 'Y10 no-reuse readout', '在路由之上再加复用'),
        ('Y11 goal readout', 'Y00 flat history', '两者都加'),
        ('Y11 goal readout', 'QRM (CRM)', '目标读法 vs QRM'),
        ('QRM (CRM)', 'Y00 flat history', 'QRM vs 平坦'),
    ]
    for a, b, lab in pairs:
        m, lo, hi = bootstrap(fam_auc, a, b)
        star = '' if lo <= 0 <= hi else '  *'
        print(f'  {lab:<20} {a.split()[0]} - {b.split()[0]}: '
              f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}')

    y = {k: mean(fam_auc[t][k] for t in fams) for k in ARMS}
    d_route = ((y['Y10 no-reuse readout'] - y['Y00 flat history'])
               + (y['Y11 goal readout'] - y['Y01 skill-only meta'])) / 2
    d_reuse = ((y['Y01 skill-only meta'] - y['Y00 flat history'])
               + (y['Y11 goal readout'] - y['Y10 no-reuse readout'])) / 2
    d_int = ((y['Y11 goal readout'] - y['Y10 no-reuse readout'])
             - (y['Y01 skill-only meta'] - y['Y00 flat history']))
    print(f'\n主效应（AUC）：路由 {d_route:+.3f}   复用 {d_reuse:+.3f}   交互 {d_int:+.3f}')


if __name__ == '__main__':
    main()
