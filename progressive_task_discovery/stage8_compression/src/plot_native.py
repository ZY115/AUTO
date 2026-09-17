"""Step 39: the reuse effect in the authors' own code, against task depth."""
import json
from collections import defaultdict
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
rows = [json.loads(l) for l in (ROOT / 'results/native_grid.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['protocol'] == 'author' and r['ok']]
TASKS = [('book', 5), ('book-and-quill', 8), ('cake', 9)]
D = defaultdict(list)
for r in rows:
    for inst, c in r['curves'].items():
        if not c:
            continue
        tail = c[max(0, len(c) * 3 // 4):]
        D[(r['task'], r['risk'], r['arm'])].append(mean(x[1] for x in tail))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.2, 4.6))
x = np.arange(3)
for arm, col, lab in (('Y11 shared', '#2f855a', 'Y11 共享目标技能'),
                      ('Y10 no-reuse', '#3182ce', 'Y10 每状态独立技能')):
    ax.plot(x, [mean(D[(t, 'safe', arm)]) for t, _ in TASKS], '-o',
            color=col, lw=2, ms=8, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f'{t}\n接受深度 {d}' for t, d in TASKS], fontsize=9)
ax.set_ylabel('末段贪心评估成功率')
ax.set_ylim(-.05, 1.08)
ax.set_title('作者代码、作者协议、10,000 幕、无岩浆', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=.2)
for i, (t, _) in enumerate(TASKS):
    d = mean(D[(t, 'safe', 'Y11 shared')]) - mean(D[(t, 'safe', 'Y10 no-reuse')])
    ax.annotate(f'+{d:.3f}', (i, mean(D[(t, 'safe', 'Y10 no-reuse')]) + d / 2),
                fontsize=9, ha='left', xytext=(8, -4), textcoords='offset points')

w = .35
for j, (risk, lab) in enumerate((('safe', '无岩浆'), ('lava', '有岩浆'))):
    for k, (arm, col) in enumerate((('Y11 shared', '#2f855a'), ('Y10 no-reuse', '#3182ce'))):
        v = [mean(D[(t, risk, arm)]) for t, _ in TASKS]
        bx.bar(x + (j - .5) * .44 + (k - .5) * .2, v, width=.19, color=col,
               alpha=1.0 if risk == 'safe' else .45,
               label=f'{"Y11" if k == 0 else "Y10"} {lab}')
bx.set_xticks(x)
bx.set_xticklabels([t for t, _ in TASKS], fontsize=9)
bx.set_ylabel('末段贪心评估成功率')
bx.set_title('有岩浆时两臂都贴地板——一万幕测不了 H2', fontsize=10)
bx.legend(fontsize=8, ncol=2)
bx.grid(axis='y', alpha=.2)

fig.suptitle('第三十九步：共享目标技能的优势在作者代码里复现，且随任务深度增大', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93])
out = ROOT / 'results/step39_native.png'
fig.savefig(out, dpi=160)
print(out)
