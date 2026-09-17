"""Step 33: the break-even picture for buying distinguishing evidence.

One point is one merge the agent failed to make. x is what walking to the
evidence would cost in environment steps; y is how many redundant queries that
merge would have removed. A merge is worth making when

    S  >  C_nav / lambda  +  1

the +1 being the query that reads the verdict once you get there. Each line is
one exchange rate lambda; a point above a line pays for itself at that rate.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC',
                                          'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
INF = 10 ** 6
d = json.loads((ROOT / 'results/evidence_value.json').read_text())
ms = [m for o in d for m in o['merges'] if m['nav'] < INF]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
rng = np.random.default_rng(0)
for ax, key, title in (
        (axes[0], 'nav', 'oracle 下界：任何分隔事件都可用'),
        (axes[1], 'nav_elsewhere', '禁止用脚下正在触发的那个事件')):
    x = np.array([m[key] for m in ms], float)
    y = np.array([m['saved'] for m in ms], float)
    keep = x < INF
    x, y = x[keep], y[keep]
    ax.scatter(x + rng.normal(0, .06, x.size), y + rng.normal(0, .12, y.size),
               s=14, alpha=.30, edgecolor='none', color='#2b6cb0')
    xs = np.linspace(0, max(6, x.max()) * 1.05, 200)
    for lam, c in ((1, '#c53030'), (3, '#dd6b20'), (10, '#38a169')):
        ax.plot(xs, xs / lam + 1, color=c, lw=1.3, label=f'$\\lambda$={lam}')
    ax.set_xlabel('取证导航步数 $C_{nav}$')
    ax.set_ylabel('该次合并可省的查询数 $S$')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-.4, max(6, x.max()) * 1.05)
    ax.set_ylim(0, min(y.max() * 1.05, np.percentile(y, 99.5) * 1.6))
    ax.legend(title='盈亏线 $S=C_{nav}/\\lambda+1$', fontsize=8, title_fontsize=8,
              loc='upper right')
    ax.grid(alpha=.18)
fig.suptitle('Step 33 第一层：主动取证的物理成本 vs 它能省下的反馈', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
out = ROOT / 'results/step33_breakeven.png'
fig.savefig(out, dpi=160)
print(out)
print(f'{len(ms)} 次合并；C_nav=0 的占 {sum(1 for m in ms if m["nav"]==0)/len(ms):.1%}')
