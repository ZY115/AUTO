"""Step 38.5: what irreversibility does to the two factors.

Left: the 2x2. Under lava on the expensive map the per-state arm reaches 90% in
11% of runs and the sharing arm in 95%, on identical maps with identical hazards.

Right: the mechanism. Catastrophes suffered before reaching 90% — if sharing a
skill carries safety across task states, the sharing arm should not have to
relearn the same hazard once per task state, and it does not.
"""
import json
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
R = json.loads((ROOT / 'results/craftworld_lava_factorial.json').read_text())
meta = {m['tag']: m for m in
        json.loads((ROOT / 'results/craftworld_lava/tasks.json').read_text())}
ARMS = ['Y00 flat history', 'Y01 skill-only meta', 'Y10 no-reuse readout',
        'Y11 goal readout', 'QRM (CRM)']
SHORT = {'Y00 flat history': '平坦历史', 'Y01 skill-only meta': '仅技能',
         'Y10 no-reuse readout': '仅路由', 'Y11 goal readout': '目标读法',
         'QRM (CRM)': 'QRM'}
COL = {'Y00 flat history': '#718096', 'Y01 skill-only meta': '#dd6b20',
       'Y10 no-reuse readout': '#3182ce', 'Y11 goal readout': '#2f855a',
       'QRM (CRM)': '#805ad5'}
CELLS = [('cheap', 'safe'), ('cheap', 'lava'), ('expensive', 'safe'),
         ('expensive', 'lava')]
NAME = {('cheap', 'safe'): '便宜导航\n无岩浆', ('cheap', 'lava'): '便宜导航\n有岩浆',
        ('expensive', 'safe'): '昂贵导航\n无岩浆',
        ('expensive', 'lava'): '昂贵导航\n有岩浆'}


def pick(cond, risk, arm, key):
    return [v[key] for t in R if meta[t]['cond'] == cond and meta[t]['risk'] == risk
            for v in R[t][arm]]


fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.8))
w = .16
for i, a in enumerate(ARMS):
    v = [mean(pick(c, r, a, 'auc')) for c, r in CELLS]
    ax.bar(np.arange(4) + (i - 2) * w, v, width=w, color=COL[a], label=SHORT[a])
ax.set_xticks(range(4))
ax.set_xticklabels([NAME[c] for c in CELLS], fontsize=9)
ax.set_ylabel('RMST（固定预算内成功率曲线下面积）')
ax.set_title('同一批地图，危险格位置也相同；只有踩上去是否不可逆', fontsize=10)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=.2)
ax.annotate('11% → 95%\n达标率', xy=(3 + .5 * w, .52), fontsize=8.5, ha='center',
            color='#2f855a')

idx = np.arange(2)
for i, a in enumerate(ARMS):
    v = []
    for c in ('cheap', 'expensive'):
        f = [x for x in pick(c, 'lava', a, 'fail90') if x >= 0]
        v.append(median(f) if f else np.nan)
    bx.bar(idx + (i - 2) * w, v, width=w, color=COL[a], label=SHORT[a])
bx.set_yscale('log')
bx.set_xticks(range(2))
bx.set_xticklabels(['便宜导航 + 岩浆', '昂贵导航 + 岩浆'], fontsize=9)
bx.set_ylabel('达到 90% 之前累计的不可逆失败次数（对数轴）')
bx.set_title('平坦历史臂在昂贵地图上一次都没达标，故无柱', fontsize=10)
bx.legend(fontsize=8, ncol=2)
bx.grid(axis='y', alpha=.2, which='both')

fig.suptitle('第三十八步半：不可逆失败把技能复用的价值放大得最厉害', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
out = ROOT / 'results/step38_5_lava.png'
fig.savefig(out, dpi=160)
print(out)
