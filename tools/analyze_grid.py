#!/usr/bin/env python3
"""Read a grid.jsonl and report H1 and H2. Standalone; no project dependencies.

    python3 analyze_grid.py results/runs/*.jsonl

Outcome is the greedy evaluation the authors' code already logs: every 100
episodes the frozen policy runs once on each of the ten task instances and the
reward is recorded. Two summaries of that curve, and **both are needed**:

    final   mean success over the last quarter of the run
    speed   episodes until ten consecutive evaluations average 90%

`final` is bounded. Once both arms reach 1.000 it reports +0.000 and looks like
"no difference" — which at 200,000 episodes on the safe condition is exactly what
happens, while `speed` shows the same runs differing by 20x to 132x. A bounded
measure at saturation is not an effect size; that mistake has been made twice in
this project already, so the tool now reports both every time.

Pairing is by (task, task instance, seed): Y11 and Y10 see the same maps, the
same hierarchy and the same seed, and differ only in whether one goal policy
serves every automaton state.

**The resampling unit is the map, not the (map, seed) pair.** A task has only ten
distinct maps; running three seeds gives thirty paired differences but not thirty
independent ones. Resampling all thirty treats the same map under three seeds as
three independent draws — pseudo-replication, and it reports a CI that is too
narrow. Measured on the 200k tabular grid, book/lava went from [+0.315, +0.613]
to [+0.220, +0.715] once the seeds were averaged within each map first. The tool
now prints the clustered interval and marks the flat one as diagnostic only.

**The risk x sharing interaction has a sign only once you name the measure and the
scale.** On book, all three of these are true of the same data at once, and they do
not contradict each other — additive and multiplicative interactions are simply
different quantities:

    final success   sharing gains +0.464 under lava, +0.000 under safe
    RMST saving     sharing saves 48,650 episodes under lava, 4,097 under safe
    RMST ratio      sharing is 1.47x faster under lava, 5.04x under safe

So report each one with its scale attached, and never promote any of them to an
unqualified "irreversibility amplifies reuse". Each also carries its own limit:
`final` is capped in the safe cell (both arms at 1.000, so the difference there
can only be +0.000); the additive quantity is dominated by scale (everything under
lava costs 10^5 episodes to begin with); the multiplicative one is squeezed by the
tau truncation (Y11 already spends most of the budget under lava, so the ratio
cannot exceed 200,000/Y11). A fixed budget is part of the research question, not
automatically an artifact — what is illegitimate is generalising past it.

**Read the interaction only when neither arm is on the floor.** If both arms are
at 0.000 under hazards, `lava minus safe` is negative because the safe condition
separated and the hazard condition has not started, which is a budget problem and
not evidence against H2. The script says so rather than printing a number that
invites the wrong reading.
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

FLOOR = 0.02       # below this an arm has not started learning


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get('ok'):
            continue
        # 参考数据来自更早的 runner，没有这些字段；补默认值以便对照。
        r.setdefault('state_format', 'tabular')
        r.setdefault('protocol', 'author')
        r.setdefault('episodes', 0)
        r['arm'] = r['arm'].split()[0]          # 'Y11 shared' -> 'Y11'
        rows.append(r)
    return rows


def per_instance(r):
    out = {}
    for inst, c in r['curves'].items():
        if not c:
            continue
        tail = c[max(0, len(c) * 3 // 4):]
        out[inst] = mean(x[1] for x in tail)
    return out


def per_instance_speed(r, thr=0.9, win=10, anchor='end'):
    """每个实例首次连续 win 个评估点平均达到 thr 的幕数；没到过就是 None。

    `anchor` 决定把这个时刻记在窗口的哪一端，**这不是细节**：

        'end'   窗口末端。首次**确认**达标的时刻，在线可得。
        'start' 窗口起点。回溯标记：用了随后 (win-1)*100 幕的信息，却把时间
                记回窗口开始，所以它不是任何在线判据能给出的时刻。

    评估间隔 100 幕、win=10，两者相差 900 幕。对慢的臂无所谓，对快的臂是数量级
    差别：book/safe 的 Y11 在 'start' 下 RMST 是 113 幕、在 'end' 下是 1,013 幕，
    于是共享的加速倍数从 37.15x 变成 5.04x。**两种定义都可以用，但必须写明用的是
    哪一种，且不得混用。** 默认取 'end'，因为它是在线可确认的那个。
    """
    out = {}
    for inst, c in r['curves'].items():
        if not c:
            continue
        hit = None
        for i in range(len(c) - win + 1):
            if mean(x[1] for x in c[i:i + win]) >= thr:
                hit = c[i][0] if anchor == 'start' else c[i + win - 1][0]
                break
        out[inst] = hit
    return out


def trend(r, segs=5):
    """把曲线切成若干段的平均成功率，用来区分「还在涨」和「已经平了」。"""
    out = []
    for inst, c in r['curves'].items():
        if not c:
            continue
        out.append([mean(x[1] for x in c[len(c) * k // segs:len(c) * (k + 1) // segs])
                    for k in range(segs)])
    if not out:
        return None
    return [mean(col) for col in zip(*out)]


def boot(v, n=10000, seed=31):
    if not v:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    b = sorted(mean(v[rng.randrange(len(v))] for _ in v) for _ in range(n))
    return mean(v), b[int(.025 * n)], b[int(.975 * n)]


def boot_by_map(a, b, n=10000, seed=31):
    """Paired bootstrap whose unit is the map: average the seeds within each map
    first, then resample maps. `a` and `b` are keyed by (instance, seed)."""
    per_map = defaultdict(list)
    for k in sorted(set(a) & set(b)):
        per_map[k[0]].append(a[k] - b[k])
    if not per_map:
        return 0.0, 0.0, 0.0, 0
    v = [mean(per_map[m]) for m in sorted(per_map)]
    rng = random.Random(seed)
    bs = sorted(mean(v[rng.randrange(len(v))] for _ in v) for _ in range(n))
    return mean(v), bs[int(.025 * n)], bs[int(.975 * n)], len(v)


def rmst_pair(a, b, tau):
    """两臂在同一批实例上的 RMST、比值、节省量与删失数。

    删失是这里的要害：速度口径对「从未达标」的实例无定义，直接丢掉就是在对成功者
    取条件，会偏袒失败更多的那一臂。RMST 把未达标者按整个预算计入，所有实例都参与。
    代价是它被 tau 截断，所以跨条件比**比值**时要留意各自的可达上限。"""
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    va = [tau if a[k] is None else a[k] for k in ks]
    vb = [tau if b[k] is None else b[k] for k in ks]
    return (mean(va), mean(vb), sum(a[k] is None for k in ks),
            sum(b[k] is None for k in ks), len(ks))


def rmst(speeds, tau):
    """E[min(T, tau)] over a dict of instance -> episodes-to-90% (None = never).

    The point of this is censoring: `speed` is undefined for a run that never
    reaches the threshold, and dropping those runs silently conditions on success
    — which flatters whichever arm fails more often. Here a non-attainer counts
    as the full budget, so every run contributes. It is still bounded by tau, so
    read the note in the module docstring before comparing RMST across cells."""
    if not speeds:
        return None, 0, 0
    v = [tau if x is None else x for x in speeds.values()]
    got = sum(x is not None for x in speeds.values())
    return mean(v), got, len(v)


ANCHOR = 'end'          # 达标时刻记在窗口末端；'start' 是回溯标记，见 per_instance_speed
ALT = 'start'
TAU = 200000            # RMST 的截断点，等于本网格的训练预算


def main():
    global ANCHOR, ALT
    argv = [a for a in sys.argv[1:] if not a.startswith('--anchor')]
    if any(a.startswith('--anchor') for a in sys.argv[1:]):
        ANCHOR = [a for a in sys.argv[1:] if a.startswith('--anchor')][0].split('=')[1]
        ALT = 'start' if ANCHOR == 'end' else 'end'
    paths = argv or sorted(Path('results/runs').glob('*.jsonl'))
    if not paths:
        sys.exit('用法: analyze_grid.py <一个或多个 .jsonl>；'
                 '不带参数时读 results/runs/*.jsonl')
    rows = []
    for p in paths:
        rows += load(p)
    if not rows:
        sys.exit('这些文件里没有成功的运行')
    print(f'读入 {len(paths)} 个文件: ' + ', '.join(Path(p).name for p in paths))
    protos = sorted({r['protocol'] for r in rows})
    fmts = sorted({r['state_format'] for r in rows})
    tasks = [t for t in ('book', 'book-and-quill', 'cake')
             if any(r['task'] == t for r in rows)]
    print(f'{len(rows)} 次成功运行  格式 {fmts}  协议 {protos}  '
          f'幕数 {sorted({r["episodes"] for r in rows})}\n')

    D, S, S2, T = (defaultdict(dict), defaultdict(dict),
                   defaultdict(dict), defaultdict(list))
    for r in rows:
        key = (r['state_format'], r['protocol'], r['task'], r['risk'], r['arm'])
        for inst, v in per_instance(r).items():
            D[key][(inst, r['seed'])] = v
        for inst, v in per_instance_speed(r, anchor=ANCHOR).items():
            S[key][(inst, r['seed'])] = v
        for inst, v in per_instance_speed(r, anchor=ALT).items():
            S2[key][(inst, r['seed'])] = v
        tr = trend(r)
        if tr:
            T[key].append(tr)

    for fmt in fmts:
        for proto in protos:
            print(f'--- {fmt} / {proto} ---')
            print(f'{"任务":<16} {"风险":<6} {"Y11":>8} {"Y10":>8} {"差":>9} '
                  f'{"95% 区间（按地图成组）":>22} {"（按地图×种子，仅诊断）":>24}')
            gaps = {}
            for t in tasks:
                for risk in ('safe', 'lava'):
                    a = D[(fmt, proto, t, risk, 'Y11')]
                    b = D[(fmt, proto, t, risk, 'Y10')]
                    keys = sorted(set(a) & set(b))
                    if not keys:
                        continue
                    m, lo, hi, nmap = boot_by_map(a, b)
                    _, flo, fhi = boot([a[k] - b[k] for k in keys])
                    gaps[(t, risk)] = (m, mean(a[k] for k in keys), mean(b[k] for k in keys))
                    print(f'{t:<16} {risk:<6} {mean(a[k] for k in keys):>8.3f} '
                          f'{mean(b[k] for k in keys):>8.3f} {m:>+9.3f} '
                          f'{f"[{lo:+.3f}, {hi:+.3f}] 地图={nmap}":>22} '
                          f'{f"[{flo:+.3f}, {fhi:+.3f}] n={len(keys)}":>24}'
                          + ('  *' if lo > 0 else ''))

            print(f'\n速度口径  达到 90% 所需幕数（RMST，删失按 {TAU:,} 幕计入）')
            print(f'  达标时刻记在窗口{"末端（首次确认）" if ANCHOR == "end" else "起点（回溯标记）"}'
                  f'；另一种定义的数字附在括号里，供判断敏感度')
            print(f'{"任务":<16} {"风险":<6} {"Y11":>11} {"Y10":>11} {"倍数":>8} '
                  f'{"节省幕数":>11}  {"删失":>9}')
            for t_ in tasks:
                for risk in ('safe', 'lava'):
                    pr = rmst_pair(S[(fmt, proto, t_, risk, 'Y11')],
                                   S[(fmt, proto, t_, risk, 'Y10')], TAU)
                    pr2 = rmst_pair(S2[(fmt, proto, t_, risk, 'Y11')],
                                    S2[(fmt, proto, t_, risk, 'Y10')], TAU)
                    if not pr:
                        continue
                    ra, rb, ca, cb, n = pr
                    alt = f'({pr2[1]/pr2[0]:.2f}x)' if pr2 and pr2[0] else ''
                    print(f'{t_:<16} {risk:<6} {ra:>11,.0f} {rb:>11,.0f} {rb/ra:>7.2f}x '
                          f'{rb-ra:>11,.0f}  Y11 {ca}/{n} Y10 {cb}/{n}   另一定义 {alt}')

            print('\nH2  岩浆是否放大共享的收益')
            print('  三个统计量，三种尺度。它们可以给出不同方向而彼此不矛盾，')
            print('  所以每一条都要连尺度一起报，任何一条都不能单独当成 H2 的答案。')
            print(f'{"任务":<16} {"末段成功率之差":>22} {"RMST 节省量（加性）":>24} {"RMST 倍数（乘性）":>22}')
            for t_ in tasks:
                if (t_, 'lava') not in gaps or (t_, 'safe') not in gaps:
                    continue
                dl, ya_l, yb_l = gaps[(t_, 'lava')]
                ds, _, _ = gaps[(t_, 'safe')]
                cells = []
                for risk in ('safe', 'lava'):
                    pr = rmst_pair(S[(fmt, proto, t_, risk, 'Y11')],
                                   S[(fmt, proto, t_, risk, 'Y10')], TAU)
                    cells.append(pr)
                add = mul = '—'
                if all(cells):
                    add = f'{(cells[1][1]-cells[1][0])-(cells[0][1]-cells[0][0]):+,.0f} 幕'
                    mul = f'{(cells[1][1]/cells[1][0])/(cells[0][1]/cells[0][0]):.3f}x'
                floor = '  ← 两臂都在地板上，下列数字照报，但不要当效应量' \
                        if max(ya_l, yb_l) < FLOOR else ''
                print(f'{t_:<16} {f"lava {dl:+.3f} − safe {ds:+.3f} = {dl-ds:+.3f}":>22} '
                      f'{add:>24} {mul:>22}{floor}')
            print('\n  读法：加性看「共享省下多少幕」，乘性看「共享让它快几倍」。')
            print('  book 上前者在岩浆下更大、后者在岩浆下更小，两句都对。')
            print('  但注意两条各自的限制：末段成功率在 safe 格封顶（两臂都 1.000，差只能是 0）；')
            print('  加性量受量纲支配（岩浆下两臂本来就要十万幕级）；乘性量受 tau 截断')
            print(f'  （岩浆下 Y11 已用掉大半预算，倍数上限只有 {TAU:,}/Y11 那么大）。')
            print('  能成立的结论是「交互方向依赖所选指标与尺度」，')
            print('  不能成立的是不加限定的「不可逆性普遍放大复用收益」。')
            print()


if __name__ == '__main__':
    main()
