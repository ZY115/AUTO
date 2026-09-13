#!/usr/bin/env python3
"""中途查看：在 run 还没结束时读工作目录里的贪心评估日志。

`run_grid.py` 只在一次 run 完整结束后才写一行 jsonl，所以跑到一半时输出文件是空的。
但作者的代码是持续往 `reward_steps_greedy_logs/` 里追加的，所以进度和趋势随时可读。

    python3 tools/peek.py /scratch/hrm           # 默认少于 2 万幕不下结论
    python3 tools/peek.py /scratch/hrm 50000     # 把门槛提到 5 万幕

用途是**提前止损**，不是提前下结论：如果某个条件跑到一半仍然平在零附近而且不再上升，
那再等几天也不会变，可以当场改规模；反过来如果两臂正在分开，就安心等它跑完。

不要拿中途数字当结果报出去——曲线还没跑完，末段成功率的定义（最后四分之一）
在中途是不成立的。这里报的是「到目前为止」的分段趋势。
"""
import re, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

TAG = re.compile(r'^(?P<task>.+?)_(?P<fmt>tabular|full_obs)_(?P<risk>safe|lava)_'
                 r'(?P<proto>author|stepcost)_(?P<arm>Y11|Y10)_s(?P<seed>\d+)$')


def read_curves(run_dir):
    """每个任务实例一条 (幕数, 奖励, 步数) 曲线。"""
    out = []
    for d in Path(run_dir).rglob('reward_steps_greedy_logs'):
        for f in sorted(d.glob('reward_steps-*.txt')):
            rows = []
            for line in f.read_text().splitlines():
                p = line.split(';')
                if len(p) == 3:
                    try:
                        rows.append((int(p[0]), float(p[1]), float(p[2])))
                    except ValueError:
                        pass
            if rows:
                out.append(rows)
    return out


def segs(curves, n=4):
    """把所有实例的评估点按幕数排序后等分成 n 段，返回每段的平均成功率。

    按评估点个数切而不是按幕数区间切：评估是从第 100 幕才开始的，按区间切会让
    第一段空掉，而右开区间又会把最后一个点漏掉，两头都是假的「无数据」。
    """
    pts = sorted((x for c in curves for x in c), key=lambda x: x[0])
    if not pts:
        return []
    out = []
    for k in range(n):
        chunk = pts[len(pts) * k // n:len(pts) * (k + 1) // n]
        out.append(mean(x[1] for x in chunk) if chunk else None)
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/scratch/hrm')
    if not root.exists():
        sys.exit(f'找不到工作目录 {root}')
    runs = sorted(d for d in root.iterdir() if d.is_dir() and TAG.match(d.name))
    if not runs:
        sys.exit(f'{root} 下没有形如 <任务>_<格式>_<风险>_<协议>_<臂>_s<种子> 的子目录')

    data = defaultdict(list)
    prog = {}
    for d in runs:
        m = TAG.match(d.name).groupdict()
        cs = read_curves(d)
        prog[d.name] = max((c[-1][0] for c in cs), default=0)
        if cs:
            data[(m['task'], m['risk'], m['arm'])] += cs

    print(f'{len(runs)} 个 run 在 {root}\n')
    print(f'{"run":<48} {"已评估到第几幕":>14}')
    for name in sorted(prog, key=lambda k: -prog[k]):
        print(f'  {name:<46} {prog[name]:>14,}')

    # 幕长是重新估算总时长的关键：随机策略每幕都撞满 max_episode_length，
    # 学会之后一幕可能只要几十步。拿开头测速外推会高估数倍到十倍。
    print('\n每幕步数的变化（决定剩余时间，越小越快）')
    print(f'{"任务":<16} {"风险":<6} {"臂":<5} {"最早 1/4":>10} {"最近 1/4":>10} {"已缩短":>9}')
    for (task, risk, arm) in sorted(data):
        pts = sorted((x for c in data[(task, risk, arm)] for x in c), key=lambda x: x[0])
        if len(pts) < 8:
            continue
        early = mean(x[2] for x in pts[:len(pts) // 4])
        late = mean(x[2] for x in pts[-len(pts) // 4:])
        print(f'{task:<16} {risk:<6} {arm:<5} {early:>10.0f} {late:>10.0f} '
              f'{(early / late if late else 1):>8.1f}x')

    print('\n按四分段看趋势（所有种子与实例合并，到目前为止）')
    print(f'{"任务":<16} {"风险":<6} {"臂":<5} ' + ' '.join(f'{f"第{i+1}/4段":>9}' for i in range(4)))
    for (task, risk, arm) in sorted(data):
        s = segs(data[(task, risk, arm)])
        cells = ' '.join(f'{v:>9.3f}' if v is not None else f'{"—":>9}' for v in s)
        print(f'{task:<16} {risk:<6} {arm:<5} {cells}')

    # 太早就宣布「平了」是误报：神经臂要跨过 er_start_size 才开始训练，
    # safe 大约第 100 幕、lava 大约第 3,300 幕，之后还要若干万幕才看得出趋势。
    MIN_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    print(f'\n判读（少于 {MIN_EP:,} 幕的条件一律不下结论）')
    for task in sorted({k[0] for k in data}):
        for risk in ('safe', 'lava'):
            a = segs(data.get((task, risk, 'Y11'), []))
            b = segs(data.get((task, risk, 'Y10'), []))
            if not a or a[-1] is None:
                continue
            reached = max((c[-1][0] for c in data.get((task, risk, 'Y11'), [])), default=0)
            ya, yb = a[-1], (b[-1] if b and b[-1] is not None else 0.0)
            if max(ya, yb) >= 0.02:
                print(f'  {task:<16} {risk:<6} 已分开：Y11 {ya:.3f} / Y10 {yb:.3f}')
            elif reached < MIN_EP:
                print(f'  {task:<16} {risk:<6} 还太早，只跑到第 {reached:,} 幕，不下结论')
            else:
                half = a[len(a) // 2] or 0.0
                note = ('两臂都近零但仍在上升，继续等' if ya - half > 0.02
                        else '两臂都近零且已经平了——再等也不会变，建议改规模')
                print(f'  {task:<16} {risk:<6} {note}（Y11 {ya:.3f} / Y10 {yb:.3f}）')


if __name__ == '__main__':
    main()
