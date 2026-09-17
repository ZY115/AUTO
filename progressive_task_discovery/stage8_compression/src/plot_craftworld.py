"""Step 38: what makes skill reuse worth anything.

Left: sample complexity by arm on the two navigation conditions. AUC cannot show
this — Y11 sits at the ceiling in every cell, so AUC differences only measure how
far the other arm has fallen (r = -0.998 against Y10's own AUC). Steps to 90%
has no ceiling and censors nowhere here.

Right: the reuse benefit against the two candidate explanations. Each point is
one task instance. Repetition count is what our own family has most of and it
tracks nothing; the cost of learning one skill tracks it.
"""
import json, math, sys
from collections import deque
from pathlib import Path
from statistics import mean, median
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC',
                                          'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from taskfile import TaskFile
INF = 10 ** 6
ARMS = ['Y00 flat history', 'Y01 skill-only meta', 'Y10 no-reuse readout',
        'Y11 goal readout', 'QRM (CRM)']
SHORT = {'Y00 flat history': '平坦历史', 'Y01 skill-only meta': '仅技能',
         'Y10 no-reuse readout': '仅路由', 'Y11 goal readout': '目标读法',
         'QRM (CRM)': 'QRM'}
COL = {'Y00 flat history': '#718096', 'Y01 skill-only meta': '#dd6b20',
       'Y10 no-reuse readout': '#3182ce', 'Y11 goal readout': '#2f855a',
       'QRM (CRM)': '#805ad5'}


def comp(t):
    back = [[] for _ in range(t.N)]
    for pc in range(t.N):
        for a in range(4):
            back[t.moves[pc][a]].append(pc)
    Cs, Ns = [], []
    for e in range(t.alphabet):
        dd = [INF] * t.N
        q = deque()
        for c in range(t.N):
            # One, not zero: firing the event costs the step that fires it.
            if any(t.events[c][a] == e for a in range(4)):
                dd[c] = 1; q.append(c)
        while q:
            c = q.popleft()
            for pc in back[c]:
                if dd[pc] > dd[c] + 1:
                    dd[pc] = dd[c] + 1; q.append(pc)
        live = [s for s in range(t.nstates)
                if not t.failing[s] and not t.accepting[s]]
        ng = sum(1 for s in live
                 if t.trans[s][e] != s and not t.failing[t.trans[s][e]])
        r = [dd[c] for c in t.starts if dd[c] < INF]
        if r:
            Cs.append(sum(r) / len(r)); Ns.append(ng)
    return mean(Cs), mean(Ns)


cw = json.loads((ROOT / 'results/craftworld_factorial.json').read_text())
own = json.loads((ROOT / 'results/factorial.json').read_text())

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.8))

conds = ['cheap', 'expensive']
w = 0.16
for i, a in enumerate(ARMS):
    v = [mean(cw[t]['first90'][a] for t in cw if cw[t]['cond'] == c) for c in conds]
    ax.bar(np.arange(2) + (i - 2) * w, v, width=w, color=COL[a], label=SHORT[a])
ax.set_yscale('log')
ax.set_xticks(range(2))
ax.set_xticklabels(['cheap（开放 7x7，25 格）', 'expensive（四房间 13x13，104 格）'],
                   fontsize=9)
ax.set_ylabel('达到 90% 成功率所需步数（对数轴）')
ax.set_title('任务结构完全不变，只换地图', fontsize=10)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=.2, which='both')

pts = []
for t, v in cw.items():
    C, N = comp(TaskFile(str(ROOT / f'results/craftworld/{t}.task')))
    pts.append(('CW-' + v['cond'], C, N,
                v['first90']['Y10 no-reuse readout'] / v['first90']['Y11 goal readout']))
for t, v in own.items():
    C, N = comp(TaskFile(str(ROOT / f'results/ceiling/{t}.task')))
    f = lambda a: median([x['first90'] if x['first90'] > 0 else 200000 for x in v[a]])
    pts.append(('自建族', C, N, f('Y10 no-reuse readout') / f('Y11 goal readout')))

GC = {'CW-cheap': '#90cdf4', 'CW-expensive': '#2b6cb0', '自建族': '#dd6b20'}
for g in GC:
    s = [p for p in pts if p[0] == g]
    bx.scatter([p[1] for p in s], [p[3] for p in s], s=34, alpha=.75,
               color=GC[g], edgecolor='none', label=g)
    bx.scatter([mean(p[1] for p in s)], [math.exp(mean(math.log(p[3]) for p in s))],
               s=180, marker='D', color=GC[g], edgecolor='k', linewidth=.8, zorder=4)
bx.set_yscale('log')
bx.set_xlabel('学会一个技能的平均代价 $C_g$（从起点到触发该事件的最短路）')
bx.set_ylabel('复用带来的样本复杂度倍数\n(仅路由 / 目标读法)')
bx.set_title('菱形是组均值；重复次数 $N_g$ 在自建族最高（8.2）而收益最低', fontsize=10)
bx.legend(fontsize=9, loc='upper left')
bx.grid(alpha=.2, which='both')

fig.suptitle('第三十八步：技能复用值钱，是因为每个技能贵，不是因为它被重复调用得多',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
out = ROOT / 'results/step38_craftworld.png'
fig.savefig(out, dpi=160)
print(out)
