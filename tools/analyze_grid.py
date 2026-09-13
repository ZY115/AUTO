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
serves every automaton state. The bootstrap resamples task instances; it never
resamples seeds.

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


def per_instance_speed(r, thr=0.9, win=10):
    """每个实例首次连续 win 个评估点平均达到 thr 的幕数；没到过就是 None。"""
    out = {}
    for inst, c in r['curves'].items():
        if not c:
            continue
        hit = None
        for i in range(len(c) - win + 1):
            if mean(x[1] for x in c[i:i + win]) >= thr:
                hit = c[i][0]
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


def main():
    paths = sys.argv[1:] or sorted(Path('results/runs').glob('*.jsonl'))
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

    D, S, T = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for r in rows:
        key = (r['state_format'], r['protocol'], r['task'], r['risk'], r['arm'])
        for inst, v in per_instance(r).items():
            D[key][(inst, r['seed'])] = v
        for inst, v in per_instance_speed(r).items():
            S[key][(inst, r['seed'])] = v
        tr = trend(r)
        if tr:
            T[key].append(tr)

    for fmt in fmts:
        for proto in protos:
            print(f'--- {fmt} / {proto} ---')
            print(f'{"任务":<16} {"风险":<6} {"Y11":>8} {"Y10":>8} {"差":>9} {"95% 区间":>20}')
            gaps = {}
            for t in tasks:
                for risk in ('safe', 'lava'):
                    a = D[(fmt, proto, t, risk, 'Y11')]
                    b = D[(fmt, proto, t, risk, 'Y10')]
                    keys = sorted(set(a) & set(b))
                    if not keys:
                        continue
                    d = [a[k] - b[k] for k in keys]
                    m, lo, hi = boot(d)
                    gaps[(t, risk)] = (m, mean(a[k] for k in keys), mean(b[k] for k in keys))
                    print(f'{t:<16} {risk:<6} {mean(a[k] for k in keys):>8.3f} '
                          f'{mean(b[k] for k in keys):>8.3f} {m:>+9.3f} '
                          f'{f"[{lo:+.3f}, {hi:+.3f}]":>20}'
                          + ('  *' if lo > 0 else ''))

            print(f'\n速度口径  达到 90% 所需幕数（中位），以及 Y10/Y11 的倍数')
            print(f'{"任务":<16} {"风险":<6} {"Y11":>10} {"Y10":>10} {"倍数":>9} {"95% 区间":>20}')
            for t in tasks:
                for risk in ('safe', 'lava'):
                    a = S[(fmt, proto, t, risk, 'Y11')]
                    b = S[(fmt, proto, t, risk, 'Y10')]
                    keys = [k for k in set(a) & set(b) if a.get(k) and b.get(k)]
                    na = sum(1 for k in set(a) & set(b) if not a.get(k))
                    nb = sum(1 for k in set(a) & set(b) if not b.get(k))
                    if not keys:
                        tot = len(set(a) & set(b))
                        print(f'{t:<16} {risk:<6} 无法判读：Y11 有 {na}/{tot}、'
                              f'Y10 有 {nb}/{tot} 个实例从未达到 90%')
                        continue
                    lr = [math.log(b[k] / a[k]) for k in keys]
                    m, lo, hi = boot(lr)
                    note = f'  （另有 Y11 {na}、Y10 {nb} 个实例从未达标，未计入）' if (na or nb) else ''
                    print(f'{t:<16} {risk:<6} {median(a[k] for k in keys):>10,.0f} '
                          f'{median(b[k] for k in keys):>10,.0f} {math.exp(m):>8.2f}x '
                          f'{f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}]":>20}'
                          + ('  *' if lo > 0 else '') + note)

            print('\nH2  岩浆是否放大差距')
            for t in tasks:
                if (t, 'lava') not in gaps or (t, 'safe') not in gaps:
                    continue
                dl, ya_l, yb_l = gaps[(t, 'lava')]
                ds, _, _ = gaps[(t, 'safe')]
                if max(ya_l, yb_l) < FLOOR:
                    # 地板还分两种：还在涨（加预算有用）和已经平了（加预算没用）
                    tr = T.get((fmt, proto, t, 'lava', 'Y11'), [])
                    verdict = ''
                    if tr:
                        avg = [mean(col) for col in zip(*tr)]
                        rise = avg[-1] - avg[len(avg) // 2]
                        verdict = ('，且末段仍在上升，加预算可能有用'
                                   if rise > 0.02 else
                                   f'，且后半程已经平了（{avg[len(avg)//2]:.3f} → {avg[-1]:.3f}），'
                                   '加预算大概率没用')
                    print(f'  {t:<16} 无法判读：岩浆条件下两臂都在地板上 '
                          f'({ya_l:.3f} / {yb_l:.3f}){verdict}')
                else:
                    print(f'  {t:<16} lava {dl:+.3f}  safe {ds:+.3f}  交互 {dl-ds:+.3f}')
            print()


if __name__ == '__main__':
    main()
