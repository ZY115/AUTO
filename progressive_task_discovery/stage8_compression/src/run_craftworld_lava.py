"""Step 38.5: does irreversible failure amplify skill reuse?

A paired 2x2 on the same maps: navigation cost (cheap / expensive) crossed with
irreversibility (safe / lava). Hazard cells are walkable in both cells of the
risk factor — in `safe` the label is an inert self-loop, in `lava` it ends the
episode — so geometry, objects, hazards, starts and horizon are identical and
the only thing that changes is what a wrong step costs.

**Why the headline measure is not a ratio.** Under lava the per-state arm does
not reach 90% at all: at a hazard density of only 2% it is still censored after
a million steps while the sharing arm gets there in four thousand. A ratio
between an attained and a censored arm is not a number, and imputing the budget
would invent one. So three things are reported side by side, as agreed:

    attainment   fraction of runs that ever reach 90%
    conditional  median steps to 90% among the runs that do
    RMST         area under the success curve to a fixed horizon, which is
                 well defined whether or not a run attains. It is the same
                 quantity as AUC, and the ceiling problem that made AUC useless
                 in Step 38 does not arise here because nothing saturates.

Diagnostics are recorded but are not optimisation targets: catastrophes before
reaching 90%, catastrophes in total, and deaths per goal.
"""
import json, math, random, subprocess, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
EXE = str(ROOT / 'compress')
BUDGET, EVERY, SEEDS = 1_000_000, 2_000, 15

ARMS = {
    'Y00 flat history':     ['--method', 'history'],
    'Y01 skill-only meta':  ['--method', 'goal', '--route', 'meta'],
    'Y10 no-reuse readout': ['--method', 'goal', '--skill-key', 'perstate'],
    'Y11 goal readout':     ['--method', 'goal'],
    'QRM (CRM)':            ['--method', 'automaton', '--crm', '1'],
}


def one(job):
    tag, arm, seed = job
    r = json.loads(subprocess.check_output(
        [EXE, '--task', str(ROOT / f'results/craftworld_lava/{tag}.task'),
         '--seed', str(seed), '--budget', str(BUDGET), '--every', str(EVERY)]
        + ARMS[arm], text=True))
    return tag, arm, dict(auc=r['auc'], final=r['final_success'],
                          f90=r['first90'], fails=r['failures'],
                          fail90=r['fail_at_90'],
                          gd=sum(r['goal_deaths']))


def boot(v, n=10000, seed=5):
    rng = random.Random(seed)
    b = sorted(mean(v[rng.randrange(len(v))] for _ in v) for _ in range(n))
    return mean(v), b[250], b[9750]


def main():
    meta = {m['tag']: m for m in
            json.loads((ROOT / 'results/craftworld_lava/tasks.json').read_text())}
    tags = sorted(meta)
    jobs = [(t, a, s) for t in tags for a in ARMS for s in range(SEEDS)]
    R = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=10) as ex:
        for tag, arm, v in ex.map(one, jobs, chunksize=8):
            R[tag][arm].append(v)
    (ROOT / 'results/craftworld_lava_factorial.json').write_text(
        json.dumps({t: {a: R[t][a] for a in ARMS} for t in tags}, indent=2) + '\n')

    def cell(cond, risk, arm, key):
        ts = [t for t in tags if meta[t]['cond'] == cond and meta[t]['risk'] == risk]
        return [v[key] for t in ts for v in R[t][arm]]

    print(f'3 任务 x 2 导航 x 2 风险 x 6 地图 x {SEEDS} 种子 x {len(ARMS)} 臂，'
          f'预算 {BUDGET}，危险格密度 3%\n')
    print(f'{"导航":<10} {"风险":<6} {"臂":<22} {"达标率":>7} {"条件中位 T90":>13} '
          f'{"RMST":>7} {"达标前灾难":>11} {"累计灾难":>10}')
    for cond in ('cheap', 'expensive'):
        for risk in ('safe', 'lava'):
            for a in ARMS:
                f9 = cell(cond, risk, a, 'f90')
                ok = [x for x in f9 if x > 0]
                f90c = cell(cond, risk, a, 'fail90')
                print(f'{cond:<10} {risk:<6} {a:<22} {len(ok)/len(f9):>6.0%} '
                      f'{(f"{median(ok):.0f}" if ok else "—"):>13} '
                      f'{mean(cell(cond, risk, a, "auc")):>7.3f} '
                      f'{(f"{median([x for x in f90c if x >= 0]):.0f}" if any(x >= 0 for x in f90c) else "—"):>11} '
                      f'{median(cell(cond, risk, a, "fails")):>10.0f}')
            print()

    print('预注册问题一：岩浆是否放大 skill reuse（Y10 -> Y11）')
    for cond in ('cheap', 'expensive'):
        for risk in ('safe', 'lava'):
            a = cell(cond, risk, 'Y10 no-reuse readout', 'f90')
            b = cell(cond, risk, 'Y11 goal readout', 'f90')
            ra, rb = sum(1 for x in a if x > 0) / len(a), sum(1 for x in b if x > 0) / len(b)
            ua = mean(cell(cond, risk, 'Y10 no-reuse readout', 'auc'))
            ub = mean(cell(cond, risk, 'Y11 goal readout', 'auc'))
            print(f'  {cond:<10} {risk:<6} 达标率 Y10 {ra:>4.0%} -> Y11 {rb:>4.0%}   '
                  f'RMST {ua:.3f} -> {ub:.3f}  (+{ub-ua:.3f})')
    pairs = []
    for t in tags:
        if meta[t]['risk'] != 'lava':
            continue
        s = t[:-4] + 'safe'
        du = (mean(v['auc'] for v in R[t]['Y11 goal readout'])
              - mean(v['auc'] for v in R[t]['Y10 no-reuse readout']))
        ds = (mean(v['auc'] for v in R[s]['Y11 goal readout'])
              - mean(v['auc'] for v in R[s]['Y10 no-reuse readout']))
        pairs.append(du - ds)
    m, lo, hi = boot(pairs)
    print(f'  差的差（RMST 口径，按 36 对地图配对）= {m:+.3f} [{lo:+.3f}, {hi:+.3f}]'
          + ('  *' if lo > 0 else '  （区间含 0）'))

    print('\n预注册问题二：岩浆是否放大 routing')
    for lab, x, y in (('Y00 -> Y10（无技能时加路由）', 'Y00 flat history', 'Y10 no-reuse readout'),
                      ('Y01 -> Y11（有技能时加路由）', 'Y01 skill-only meta', 'Y11 goal readout')):
        for cond in ('cheap', 'expensive'):
            row = []
            for risk in ('safe', 'lava'):
                row.append(mean(cell(cond, risk, y, 'auc')) - mean(cell(cond, risk, x, 'auc')))
            print(f'  {lab:<26} {cond:<10} safe {row[0]:+.3f}  lava {row[1]:+.3f}  '
                  f'差 {row[1]-row[0]:+.3f}')

    print('\n诊断：达到 90% 之前累计了多少次不可逆失败（仅 lava 格）')
    for cond in ('cheap', 'expensive'):
        for a in ARMS:
            v = [x for x in cell(cond, 'lava', a, 'fail90') if x >= 0]
            print(f'  {cond:<10} {a:<22} '
                  + (f'{median(v):>8.0f}（{len(v)} 次运行达标）' if v else '        — 无运行达标'))


if __name__ == '__main__':
    main()
