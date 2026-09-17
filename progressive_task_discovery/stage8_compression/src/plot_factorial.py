"""Step 36: where the automaton's benefit comes from.

Left: the 2x2. Right: the same arms against task size, which is where the two
factors separate — routing holds flat as tasks grow, everything without it does
not.
"""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC',
                                          'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / 'results/factorial.json').read_text())
ARMS = ['Y00 flat history', 'Y01 skill-only meta', 'Y10 no-reuse readout',
        'Y11 goal readout', 'QRM (CRM)']
LAB = {'Y00 flat history': '平坦历史 RL\n(无路由, 无复用)',
       'Y01 skill-only meta': '学习型元控制器\n(无路由, 有复用)',
       'Y10 no-reuse readout': '每状态独立技能\n(有路由, 无复用)',
       'Y11 goal readout': '目标读法\n(有路由, 有复用)',
       'QRM (CRM)': 'QRM\n(标准结构基线)'}
COL = {'Y00 flat history': '#718096', 'Y01 skill-only meta': '#dd6b20',
       'Y10 no-reuse readout': '#3182ce', 'Y11 goal readout': '#2f855a',
       'QRM (CRM)': '#805ad5'}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8))

y = {a: mean(mean(x['auc'] for x in res[t][a]) for t in res) for a in ARMS}
per = {a: [mean(x['auc'] for x in res[t][a]) for t in res] for a in ARMS}
xs = np.arange(len(ARMS))
ax.bar(xs, [y[a] for a in ARMS], color=[COL[a] for a in ARMS], width=.62)
for i, a in enumerate(ARMS):
    ax.scatter(np.full(len(per[a]), i) + np.random.default_rng(i).normal(0, .07, len(per[a])),
               per[a], s=9, color='k', alpha=.28, zorder=3)
    ax.text(i, y[a] + .012, f'{y[a]:.3f}', ha='center', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([LAB[a] for a in ARMS], fontsize=8)
ax.set_ylabel('AUC（学习曲线下面积）')
ax.set_ylim(.6, 1.045)
ax.set_title('24 族 x 30 种子；每个黑点是一个族', fontsize=10)
ax.grid(axis='y', alpha=.2)

groups = ['n3', 'n4', 'n5']
for a in ARMS:
    v = [mean(mean(x['auc'] for x in res[t][a]) for t in res if g in t) for g in groups]
    bx.plot(range(3), v, '-o', color=COL[a], lw=1.8, ms=6,
            label=LAB[a].replace('\n', ' '))
bx.set_xticks(range(3))
bx.set_xticklabels(['n3（19–27 状态）', 'n4（29–37）', 'n5（49–57）'], fontsize=9)
bx.set_ylabel('AUC')
bx.set_title('任务越长，路由的价值越大；没有路由的两臂掉得最快', fontsize=10)
bx.legend(fontsize=7.5, loc='lower left')
bx.grid(alpha=.2)

fig.suptitle('第三十六步：自动机的收益来自「告诉我下一步做什么」，不是来自「技能可以复用」',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
out = ROOT / 'results/step36_factorial.png'
fig.savefig(out, dpi=160)
print(out)
