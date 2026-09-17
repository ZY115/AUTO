"""Step 33 layer 3: what a risk-accepting merge rule can actually buy.

x is the number of wrong reuses the rule commits — a cached verdict served to a
node whose true verdict differs, which is the only way the mistake reaches a
decision. y is the redundant queries it removes. The oracle sits at (0, 2190);
everything the agent can reach on its own data is far to the right of it.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC',
                                          'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'results/hindsight_merge.json').read_text())
RED = 2190

fig, ax = plt.subplots(figsize=(6.4, 4.8))
x = [r['wrong'] for r in rows]
y = [r['saved'] for r in rows]
ax.plot(x, y, '-o', color='#2b6cb0', lw=1.4, ms=6, zorder=3)
for r in rows:
    ax.annotate(f"m={r['m']}", (r['wrong'], r['saved']),
                textcoords='offset points', xytext=(7, -3), fontsize=9)
ax.scatter([0], [RED], marker='*', s=220, color='#c53030', zorder=4,
           label='oracle（按真状态缓存）')
ax.scatter([0], [0], marker='s', s=50, color='#4a5568', zorder=4, label='不合并')
ax.axhline(RED, color='#c53030', ls=':', lw=1)
ax.set_xlabel('错误复用次数（把错的裁决发给了一个决策）')
ax.set_ylabel('省下的冗余查询数')
ax.set_title('第三层：只用智能体自己的证据，能省多少 / 会错多少', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=.2)
fig.tight_layout()
out = ROOT / 'results/step33_hindsight.png'
fig.savefig(out, dpi=160)
print(out)
for r in rows:
    if r['wrong']:
        print(f"  m={r['m']}: 每犯一次错换来 {r['saved']/r['wrong']:.1f} 次省下的查询，"
              f"覆盖冗余 {r['saved']/RED:.1%}")
