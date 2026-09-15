#!/usr/bin/env python3
"""从本目录的归档数据重算 docs/ORIGINAL_QUESTION.md 里引用的数字。

    python3 discovery/src/reproduce.py

每一条都把重算值和文档里写的值并排打出来，对不上就标出来。**对不上不一定是
数据错了**——也可能是文档引用时口径写得不够细（比如中位数与 RMST、达标计数与
求解计数）。不一致一律照报，不要改数去凑。
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent.parent
CENSOR = 200_000
ok = bad = 0


def load(rel):
    p = HERE / 'results' / rel
    op = gzip.open if p.suffix == '.gz' else open
    with op(p, 'rt') as f:
        return [json.loads(l) for l in f if l.strip()]


def cmp(label, got, want, tol=0.02):
    """want=None 表示文档没给这个数，只打印。"""
    global ok, bad
    if want is None:
        print(f'    {label:<38} {got}')
        return
    good = abs(got - want) <= tol * max(abs(want), 1)
    ok, bad = ok + good, bad + (not good)
    mark = '✓' if good else '✗ 文档写的是'
    print(f'    {label:<38} {got:>12,.0f}   {mark} {want:,}' if good
          else f'    {label:<38} {got:>12,.0f}   {mark} {want:,}')


print('一、二　中途反馈的预算曲线  (verify.jsonl)')
v = load('verify.jsonl.gz')
by = defaultdict(list)
for r in v:
    by[(r['arm'], r['level'])].append(r)
for lv, want_n, want_rmst in ((0, 0, None), (80, 61, 115_757),
                              (160, None, 20_710), (320, 144, 1_264)):
    g = by[('learned', lv)]
    solved = sum(r['first90'] < CENSOR for r in g)
    rmst = mean(min(r['first90'], CENSOR) for r in g)
    tag = '只有终点奖励' if lv == 0 else f'可问 {lv} 次'
    print(f'  {tag}')
    if want_n is not None:
        cmp('达标数', solved, want_n, tol=0)
    else:
        print(f'    {"达标数":<38} {solved}/144   （文档写 127，见下方说明）')
    cmp('RMST（删失按 20 万计）', rmst, want_rmst)
    if lv == 0:
        cmp('结构树节点数（中位）', median(r['nodes'] for r in g), 1, tol=0)
g = by[('given', -1)]
cmp('  直接给规则：RMST', mean(min(r['first90'], CENSOR) for r in g), 1_142)

print('\n三、四　信号强度与「学出来 vs 真值」  (depth_vs_graph)')
d = load('depth_vs_graph/final12.jsonl.gz') + load('depth_vs_graph/final_product.jsonl.gz')
shapes = defaultdict(list)
for r in d:
    shapes[r['shape']].append(r)
print(f'    可用的 shape: {sorted(k for k in shapes if k)}')
# 文档里的「只给一个标量进度计数器」在数据里叫 frontier，不叫 count。
for s, want in (('none', 28_990), ('frontier', 4_336), ('learned', 6_250),
                ('oracle', 5_904), ('product', 1_128)):
    g = shapes.get(s)
    if not g:
        print(f'    {s:<38} 本次归档里没有这个 shape')
        continue
    sol = [r for r in g if r['first90'] > 0]      # first90 <= 0 表示从未达到
    cmp(s, mean(r['first90'] for r in sol), want, tol=0.03)

print('\n五　原始记忆不可行  (rawhistory.jsonl)')
# 三个要比的臂在 arm 字段上，不在 method；method 会把 +shape 的两臂混进来。
# 状态数是 features。first90 <= 0 表示从未达到，不是一个小数值。
rh = load('rawhistory.jsonl.gz')
m = defaultdict(list)
for r in rh:
    m[r['arm']].append(r)
for k, want_states, want_steps in (('rawhistory', 14_545, None),
                                   ('history', 75, 28_714),
                                   ('automaton', 24, 12_901)):
    g = m[k]
    sol = [r for r in g if r['first90'] > 0]
    print(f'  {k}：解出 {len(sol)}/{len(g)}')
    cmp('状态数 features', mean(r['features'] for r in g), want_states)
    if want_steps:
        cmp('步数', mean(r['first90'] for r in sol), want_steps)
    elif not sol:
        print(f'    {"步数":<38} 从未达到   ✓ 文档写 never')

print('\n七　Occam 界：离线与实时  (occam_*.json)')
for f, label in (('occam_merge_n3.json', '离线'), ('online_occam_n3.json', '实时')):
    p = HERE / 'results' / f
    if not p.exists():
        continue
    j = json.loads(p.read_text())
    print(f'  {label}（{f}）顶层键: {sorted(j)[:8] if isinstance(j, dict) else type(j).__name__}')

print(f'\n对上 {ok} 项，对不上 {bad} 项。')
print('对不上的请先查口径（中位数/RMST、达标/求解），不要改数去凑。')
sys.exit(0)
