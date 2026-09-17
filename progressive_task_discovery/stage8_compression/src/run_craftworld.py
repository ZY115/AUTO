"""Step 38: is skill reuse worth anything because subgoals repeat, or because
each repetition is expensive to learn?

Three public HRM task structures are held fixed and only the map changes:

    cheap      open plan 7x7, 25 free cells
    expensive  four rooms 13x13, 104 free cells, walls and four doors

Nothing else moves. No lava (so no failing states), one object per class,
four-way movement with no orientation — the point is to leave exactly one cause
standing.

The pre-registered test is the difference of differences,

    (Y11 - Y10) on expensive  -  (Y11 - Y10) on cheap  >  0

because Y11 and Y10 read the same automaton the same way and differ only in
whether one skill serves every task state. The continuous version uses the
navigation a non-sharing learner has to repeat,

    C_reuse = sum_g (N_g - 1) * C_g

with C_g the mean shortest path from the start set to firing g, and N_g how many
live task states demand it. Our own family sat near the bottom of that axis,
which is the Step 36 result this is meant to explain.
"""
import json, subprocess, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXE = str(ROOT / 'compress')
BUDGET, EVERY, SEEDS = 400_000, 1_000, 20

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
        [EXE, '--task', str(ROOT / f'results/craftworld/{tag}.task'), '--seed', str(seed),
         '--budget', str(BUDGET), '--every', str(EVERY)] + ARMS[arm], text=True))
    return tag, arm, r['auc'], r['final_success'], r['first90']


def boot(vals, n=10000, seed=11):
    import random
    rng = random.Random(seed)
    out = sorted(mean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(n))
    return mean(vals), out[int(.025 * n)], out[int(.975 * n)]


def main():
    meta = {m['tag']: m for m in
            json.loads((ROOT / 'results/craftworld/tasks.json').read_text())}
    tags = sorted(meta)
    jobs = [(t, a, s) for t in tags for a in ARMS for s in range(SEEDS)]
    auc = defaultdict(lambda: defaultdict(list))
    fin = defaultdict(lambda: defaultdict(list))
    f90 = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=10) as ex:
        for tag, arm, a, f, n in ex.map(one, jobs, chunksize=8):
            auc[tag][arm].append(a); fin[tag][arm].append(f)
            # Censored at the budget when 90% was never reached, which is honest
            # for a ratio as long as the censoring rate is reported alongside.
            f90[tag][arm].append(n if n > 0 else BUDGET)
    A = {t: {a: mean(auc[t][a]) for a in ARMS} for t in tags}
    F = {t: {a: mean(fin[t][a]) for a in ARMS} for t in tags}
    def med(v):
        v = sorted(v); return v[len(v) // 2]
    N90 = {t: {a: med(f90[t][a]) for a in ARMS} for t in tags}
    CENS = {t: {a: sum(1 for x in f90[t][a] if x == BUDGET) / SEEDS for a in ARMS}
            for t in tags}
    (ROOT / 'results/craftworld_factorial.json').write_text(
        json.dumps({t: {'auc': A[t], 'final': F[t], 'first90': N90[t],
                   'censored': CENS[t], **{k: meta[t][k] for k in
                   ('task', 'cond', 'c_reuse', 'cells')}} for t in tags}, indent=2) + '\n')

    print(f'3 个任务结构 x 2 个导航条件 x 6 张地图 x {SEEDS} 个种子 x {len(ARMS)} 臂，'
          f'预算 {BUDGET}\n')
    print(f'{"任务":<15} {"条件":<10} {"C_reuse":>8} ' +
          ' '.join(f'{a.split()[0]:>6}' for a in ARMS) + f' {"Y11-Y10":>9}')
    for task in ('book', 'book-and-quill', 'cake'):
        for cond in ('cheap', 'expensive'):
            ts = [t for t in tags if meta[t]['task'] == task and meta[t]['cond'] == cond]
            cr = mean(meta[t]['c_reuse'] for t in ts)
            row = {a: mean(A[t][a] for t in ts) for a in ARMS}
            print(f'{task:<15} {cond:<10} {cr:>8.1f} ' +
                  ' '.join(f'{row[a]:>6.3f}' for a in ARMS) +
                  f' {row["Y11 goal readout"]-row["Y10 no-reuse readout"]:>+9.3f}')

    print('\n预注册检验 B：(Y11-Y10)|expensive - (Y11-Y10)|cheap > 0')
    d = []
    for task in ('book', 'book-and-quill', 'cake'):
        for s in range(6):
            c = f'cw_{task.replace("-","")}_cheap_{s}'
            e = f'cw_{task.replace("-","")}_expensive_{s}'
            dc = A[c]['Y11 goal readout'] - A[c]['Y10 no-reuse readout']
            de = A[e]['Y11 goal readout'] - A[e]['Y10 no-reuse readout']
            d.append(de - dc)
    m, lo, hi = boot(d)
    print(f'  差的差 = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]'
          + ('  *' if lo > 0 else '  （区间含 0）'))

    print('\n预注册检验 A：routing 在公开任务结构上是否仍然成立')
    for lab, x, y in (('Y11 - Y01（技能之上加路由）', 'Y11 goal readout', 'Y01 skill-only meta'),
                      ('Y10 - Y00（只加路由）', 'Y10 no-reuse readout', 'Y00 flat history'),
                      ('Y11 - QRM', 'Y11 goal readout', 'QRM (CRM)')):
        m, lo, hi = boot([A[t][x] - A[t][y] for t in tags])
        print(f'  {lab:<28} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]'
              + ('  *' if not (lo <= 0 <= hi) else ''))

    print('\nAUC 差值受天花板限制（Y11 几乎总在 1.0），所以效应量改用样本复杂度')
    print(f'{"任务":<15} {"条件":<10} {"Y10 的 first90":>14} {"Y11 的 first90":>14} '
          f'{"倍数":>7} {"删失":>7}')
    for task in ('book', 'book-and-quill', 'cake'):
        for cond in ('cheap', 'expensive'):
            ts = [t for t in tags if meta[t]['task'] == task and meta[t]['cond'] == cond]
            a = mean(N90[t]['Y10 no-reuse readout'] for t in ts)
            b = mean(N90[t]['Y11 goal readout'] for t in ts)
            c = mean(CENS[t]['Y10 no-reuse readout'] + CENS[t]['Y11 goal readout']
                     for t in ts) / 2
            print(f'{task:<15} {cond:<10} {a:>14.0f} {b:>14.0f} {a/b:>7.2f}x {c:>6.0%}')

    print('\n连续横轴：C_reuse 与 Y11-Y10 的关系（36 张地图）')
    pts = [(meta[t]['c_reuse'], A[t]['Y11 goal readout'] - A[t]['Y10 no-reuse readout'])
           for t in tags]
    pts.sort()
    n = len(pts)
    for lo_i, hi_i, lab in ((0, n // 3, '低三分位'), (n // 3, 2 * n // 3, '中三分位'),
                            (2 * n // 3, n, '高三分位')):
        g = pts[lo_i:hi_i]
        print(f'  {lab:<8} C_reuse {g[0][0]:>6.1f}-{g[-1][0]:>6.1f}   '
              f'Y11-Y10 平均 {mean(v for _, v in g):+.3f}')
    mx, my = mean(p[0] for p in pts), mean(p[1] for p in pts)
    sx = sum((p[0] - mx) ** 2 for p in pts) ** .5
    sy = sum((p[1] - my) ** 2 for p in pts) ** .5
    r = sum((p[0] - mx) * (p[1] - my) for p in pts) / (sx * sy) if sx and sy else 0
    print(f'  Pearson r = {r:+.3f}')


if __name__ == '__main__':
    main()
